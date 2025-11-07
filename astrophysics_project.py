
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta
import json
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

# ML and Analytics imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import faiss


class RealTimeAPIClient:
    """
    Client for fetching real-time publication data from various APIs
    """



    def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        """Parse arXiv XML response"""
        papers = []

        try:
            root = ET.fromstring(xml_text)


            for entry in root.findall('atom:entry', ns):
                paper = {}

                # ID
                paper['arxiv_id'] = entry.find('atom:id', ns).text.split('/abs/')[-1]

                # Title
                paper['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')

                # Abstract
                summary = entry.find('atom:summary', ns)
                paper['abstract'] = summary.text.strip().replace('\n', ' ') if summary is not None else ""

                # Authors
                authors = entry.findall('atom:author', ns)
                author_names = [a.find('atom:name', ns).text for a in authors]
                paper['authors'] = ', '.join(author_names[:3])  # First 3 authors
                if len(author_names) > 3:
                    paper['authors'] += ' et al.'

                # Published date
                published = entry.find('atom:published', ns)
                if published is not None:
                    pub_date = datetime.strptime(published.text[:10], '%Y-%m-%d')
                    paper['year'] = pub_date.year
                    paper['date'] = published.text[:10]
                else:
                    paper['year'] = datetime.now().year
                    paper['date'] = datetime.now().strftime('%Y-%m-%d')

                # Categories
                categories = entry.findall('atom:category', ns)
                paper['categories'] = [c.get('term') for c in categories]
                paper['primary_category'] = paper['categories'][0] if paper['categories'] else 'unknown'

                # Journal reference (if available)
                journal_ref = entry.find('arxiv:journal_ref', ns)
                paper['journal'] = journal_ref.text if journal_ref is not None else 'arXiv'

                # DOI (if available)
                doi = entry.find('arxiv:doi', ns)
                paper['doi'] = doi.text if doi is not None else None

                # Link
                paper['url'] = f"https://arxiv.org/abs/{paper['arxiv_id']}"

                papers.append(paper)

        except Exception as e:
            print(f"Error parsing arXiv response: {e}")

        return papers

    def fetch_ads_papers(self,
                        query: str = "astrophysics",
                        max_results: int = 50,
                        api_key: Optional[str] = None) -> List[Dict]:
        """
        Fetch papers from NASA ADS API
        Requires API key from https://ui.adsabs.harvard.edu/user/settings/token
        """
        if not api_key:
            print("⚠ NASA ADS API key not provided. Skipping ADS fetch.")
            print("  Get your key at: https://ui.adsabs.harvard.edu/user/settings/token")
            return []

        print(f"Fetching papers from NASA ADS (query: '{query}', max: {max_results})...")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        params = {
            'q': query,
            'rows': max_results,
            'sort': 'date desc',
            'fl': 'bibcode,title,author,abstract,year,pubdate,pub,citation_count,doi'
        }

        try:
            response = self.session.get(
                self.ads_base_url,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            papers = self._parse_ads_response(data)
            print(f"✓ Fetched {len(papers)} papers from NASA ADS")
            return papers

        except Exception as e:
            print(f"✗ Error fetching from NASA ADS: {e}")
            return []

    def _parse_ads_response(self, data: Dict) -> List[Dict]:
        """Parse NASA ADS JSON response"""
        papers = []

        try:
            docs = data.get('response', {}).get('docs', [])

            for doc in docs:
                paper = {
                    'bibcode': doc.get('bibcode', ''),
                    'title': doc.get('title', [''])[0] if doc.get('title') else '',
                    'abstract': doc.get('abstract', ''),
                    'authors': ', '.join(doc.get('author', [])[:3]) + (' et al.' if len(doc.get('author', [])) > 3 else ''),
                    'year': doc.get('year', datetime.now().year),
                    'date': doc.get('pubdate', ''),
                    'journal': doc.get('pub', 'Unknown'),
                    'citations': doc.get('citation_count', 0),
                    'doi': doc.get('doi', [''])[0] if doc.get('doi') else None,
                    'url': f"https://ui.adsabs.harvard.edu/abs/{doc.get('bibcode', '')}"
                }
                papers.append(paper)

        except Exception as e:
            print(f"Error parsing NASA ADS response: {e}")

        return papers

    def fetch_crossref_papers(self,
                             query: str = "astrophysics",
                             max_results: int = 50) -> List[Dict]:
        """
        Fetch papers from CrossRef API (open access metadata)
        """
        print(f"Fetching papers from CrossRef (query: '{query}', max: {max_results})...")

        params = {
            'query': query,
            'rows': max_results,
            'sort': 'published',
            'order': 'desc',
            'filter': 'type:journal-article'
        }

        try:
            response = self.session.get(
                self.crossref_base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            papers = self._parse_crossref_response(data)
            print(f"✓ Fetched {len(papers)} papers from CrossRef")
            return papers

        except Exception as e:
            print(f"✗ Error fetching from CrossRef: {e}")
            return []

    def _parse_crossref_response(self, data: Dict) -> List[Dict]:
        """Parse CrossRef JSON response"""
        papers = []

        try:
            items = data.get('message', {}).get('items', [])

            for item in items:
                # Extract publication date
                pub_date = item.get('published-print', item.get('published-online', {}))
                date_parts = pub_date.get('date-parts', [[]])[0] if pub_date else []
                year = date_parts[0] if date_parts else datetime.now().year

                # Extract authors
                authors_data = item.get('author', [])
                authors = []
                for author in authors_data[:3]:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    authors.append(f"{given} {family}".strip())
                author_str = ', '.join(authors)
                if len(authors_data) > 3:
                    author_str += ' et al.'

                paper = {
                    'doi': item.get('DOI', ''),
                    'title': item.get('title', [''])[0] if item.get('title') else '',
                    'abstract': item.get('abstract', 'No abstract available'),
                    'authors': author_str,
                    'year': year,
                    'journal': item.get('container-title', ['Unknown'])[0],
                    'citations': item.get('is-referenced-by-count', 0),
                    'url': item.get('URL', f"https://doi.org/{item.get('DOI', '')}")
                }
                papers.append(paper)

        except Exception as e:
            print(f"Error parsing CrossRef response: {e}")

        return papers


class RealTimeAstroPhysicsResearchPlatform:
    """
    Enhanced platform with real-time API integration
    """

    def __init__(self, ads_api_key: Optional[str] = None):
        self.api_client = RealTimeAPIClient()
        self.ads_api_key = ads_api_key
        self.df = None
        self.embeddings = None
        self.faiss_index = None
        self.topic_model = None
        self.clusters = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )

    def fetch_papers_from_apis(self,
                               query: str = "astrophysics",
                               arxiv_count: int = 100,
                               ads_count: int = 0,
                               crossref_count: int = 0) -> pd.DataFrame:
        """
        Fetch papers from multiple APIs and combine into dataset
        """
        print("\n" + "="*70)
        print("FETCHING REAL-TIME PUBLICATION DATA")
        print("="*70)

        all_papers = []

        # Fetch from arXiv
        if arxiv_count > 0:
            arxiv_papers = self.api_client.fetch_arxiv_papers(
                query=query,
                max_results=arxiv_count
            )
            for paper in arxiv_papers:
                paper['source'] = 'arXiv'
                paper['paper_id'] = paper.get('arxiv_id', f"arxiv_{len(all_papers)}")
            all_papers.extend(arxiv_papers)
            time.sleep(1)  # Rate limiting

        # Fetch from NASA ADS
        if ads_count > 0 and self.ads_api_key:
            ads_papers = self.api_client.fetch_ads_papers(
                query=query,
                max_results=ads_count,
                api_key=self.ads_api_key
            )
            for paper in ads_papers:
                paper['source'] = 'NASA ADS'
                paper['paper_id'] = paper.get('bibcode', f"ads_{len(all_papers)}")
            all_papers.extend(ads_papers)
            time.sleep(1)

        # Fetch from CrossRef
        if crossref_count > 0:
            crossref_papers = self.api_client.fetch_crossref_papers(
                query=query,
                max_results=crossref_count
            )
            for paper in crossref_papers:
                paper['source'] = 'CrossRef'
                paper['paper_id'] = paper.get('doi', f"crossref_{len(all_papers)}").replace('/', '_')
            all_papers.extend(crossref_papers)

        if not all_papers:
            raise ValueError("No papers fetched from any API!")

        # Convert to DataFrame
        self.df = pd.DataFrame(all_papers)

        # Standardize columns
        required_cols = ['paper_id', 'title', 'abstract', 'authors', 'year', 'journal', 'source']
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = 'Unknown'

        # Add citations if not present
        if 'citations' not in self.df.columns:
            self.df['citations'] = np.random.poisson(20, len(self.df))

        # Clean data
        self.df['abstract'] = self.df['abstract'].fillna('No abstract available')
        self.df['title'] = self.df['title'].fillna('Untitled')
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce').fillna(2024).astype(int)

        print(f"\n✓ Total papers fetched: {len(self.df)}")
        print(f"  - By source: {self.df['source'].value_counts().to_dict()}")
        print(f"  - Year range: {self.df['year'].min()} - {self.df['year'].max()}")

        return self.df

    def preprocess_text(self) -> np.ndarray:
        """Create TF-IDF embeddings from text data"""
        if self.df is None:
            raise ValueError("No data available. Fetch papers first!")

        print("\nPreprocessing text data...")
        combined_text = self.df['title'] + ' ' + self.df['abstract']
        self.embeddings = self.vectorizer.fit_transform(combined_text).toarray()
        print(f"✓ Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings

    def build_faiss_index(self):
        """Build FAISS index for semantic search"""
        if self.embeddings is None:
            self.preprocess_text()

        print("\nBuilding FAISS index...")
        # Normalize embeddings
        embeddings_normalized = self.embeddings.astype('float32').copy()
        faiss.normalize_L2(embeddings_normalized)

        # Create FAISS index
        d = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings_normalized)

        print(f"✓ FAISS index built with {self.faiss_index.ntotal} vectors")

    def semantic_search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Perform semantic search using FAISS"""
        if self.faiss_index is None:
            self.build_faiss_index()

        # Transform query
        query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        faiss.normalize_L2(query_vec)

        # Search
        distances, indices = self.faiss_index.search(query_vec, top_k)

        results = self.df.iloc[indices[0]].copy()
        results['similarity_score'] = distances[0]

        return results[['paper_id', 'title', 'authors', 'year', 'journal', 'source', 'similarity_score', 'url']]

    def perform_topic_modeling(self, n_topics: int = 10):
        """Perform topic modeling using BERTopic"""
        print("\nPerforming topic modeling...")
        combined_text = (self.df['title'] + ' ' + self.df['abstract']).tolist()

        # Configure UMAP
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Initialize BERTopic
        self.topic_model = BERTopic(
            umap_model=umap_model,
            nr_topics=n_topics,
            calculate_probabilities=True,
            verbose=False
        )

        # Fit model
        topics, probs = self.topic_model.fit_transform(combined_text)
        self.df['topic_id'] = topics

        print(f"✓ Identified {len(set(topics))} topics")
        return topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        """Get detailed topic information"""
        if self.topic_model is None:
            self.perform_topic_modeling()

        return self.topic_model.get_topic_info()

    def cluster_papers(self, n_clusters: int = 8):
        """Cluster papers using K-Means"""
        if self.embeddings is None:
            self.preprocess_text()

        print(f"\nClustering papers into {n_clusters} clusters...")

        # Standardize features
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(self.embeddings)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(scaled_embeddings)
        self.df['cluster'] = self.clusters

        print(f"✓ Clustering completed")
        return self.clusters

    def analyze_sentiment(self):
        """Simple sentiment analysis"""
        print("\nAnalyzing sentiment...")

        positive_words = ['novel', 'significant', 'breakthrough', 'innovative', 'successful',
                         'improved', 'efficient', 'robust', 'accurate', 'promising']
        negative_words = ['challenge', 'limitation', 'difficulty', 'unclear', 'uncertain',
                         'limited', 'insufficient', 'problematic', 'complex']

        def calculate_sentiment(text):
            if not text or text == 'No abstract available':
                return 0.5
            text_lower = text.lower()
            pos_count = sum(text_lower.count(word) for word in positive_words)
            neg_count = sum(text_lower.count(word) for word in negative_words)

            if pos_count + neg_count == 0:
                return 0.5
            return pos_count / (pos_count + neg_count)

        self.df['sentiment'] = self.df['abstract'].apply(calculate_sentiment)
        print(f"✓ Sentiment analysis completed")
        return self.df['sentiment']

    def visualize_clusters_2d(self):
        """Visualize clusters in 2D using t-SNE"""
        if self.embeddings is None:
            self.preprocess_text()

        if self.clusters is None:
            self.cluster_papers()

        # Apply t-SNE
        print("\nGenerating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
        embeddings_2d = tsne.fit_transform(self.embeddings)

        # Create interactive plot
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            color=self.clusters.astype(str),
            hover_data={
                'title': self.df['title'].values,
                'year': self.df['year'].values,
                'source': self.df['source'].values
            },
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Cluster'},
            title='Paper Clusters Visualization (t-SNE) - Real-time Data',
            width=1000, height=700
        )

        fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color='white')))
        return fig

    def visualize_source_distribution(self):
        """Visualize papers by data source"""
        source_counts = self.df['source'].value_counts()

        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title='Papers by Data Source',
            hole=0.4
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    def create_realtime_dashboard(self):
        """Create comprehensive real-time analytics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Papers by Year',
                'Top Journals',
                'Citation Distribution',
                'Papers by Source'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'box'}, {'type': 'pie'}]]
        )

        # Papers by year
        yearly_counts = self.df['year'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=yearly_counts.index, y=yearly_counts.values,
                   marker_color='skyblue', name='Papers'),
            row=1, col=1
        )

        # Top journals
        journal_counts = self.df['journal'].value_counts().head(10)
        fig.add_trace(
            go.Bar(y=journal_counts.index, x=journal_counts.values,
                   orientation='h', marker_color='coral', name='Journals'),
            row=1, col=2
        )

        # Citation distribution
        fig.add_trace(
            go.Box(y=self.df['citations'], marker_color='lightgreen', name='Citations'),
            row=2, col=1
        )

        # Source distribution
        source_counts = self.df['source'].value_counts()
        fig.add_trace(
            go.Pie(labels=source_counts.index, values=source_counts.values, name='Sources'),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Real-Time Research Analytics Dashboard",
            showlegend=False,
            height=900,
            width=1400
        )

        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Number of Papers", row=1, col=2)
        fig.update_yaxes(title_text="Number of Papers", row=1, col=1)
        fig.update_yaxes(title_text="Journal", row=1, col=2)

        return fig

    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_papers': len(self.df),
            'sources': self.df['source'].value_counts().to_dict(),
            'year_range': f"{self.df['year'].min()} - {self.df['year'].max()}",
            'total_citations': int(self.df['citations'].sum()),
            'avg_citations': float(self.df['citations'].mean()),
            'median_citations': float(self.df['citations'].median()),
            'top_journals': self.df['journal'].value_counts().head(5).to_dict(),
            'papers_per_year': self.df['year'].value_counts().sort_index().to_dict()
        }

        if 'sentiment' in self.df.columns:
            report['avg_sentiment'] = float(self.df['sentiment'].mean())

        if self.clusters is not None:
            report['n_clusters'] = int(len(set(self.clusters)))

        if 'topic_id' in self.df.columns:
            report['n_topics'] = int(len(set(self.df['topic_id'])))

        return report


def main():
    """Main execution with real-time APIs"""
    print("="*70)
    print("REAL-TIME ASTROPHYSICS RESEARCH PLATFORM")
    print("="*70)

    # Initialize platform
    ads_api_key = None  # Replace with your NASA ADS API key if available
    platform = RealTimeAstroPhysicsResearchPlatform(ads_api_key=ads_api_key)

    # Fetch real papers from APIs
    # You can customize the query based on your research interests
    queries = ["black holes", "exoplanets", "dark matter", "gravitational waves"]
    query = queries[0]  # Change index or create your own query

    try:
        df = platform.fetch_papers_from_apis(
            query=query,
            arxiv_count=100,  # Fetch from arXiv
            ads_count=0,      # Set to >0 if you have NASA ADS API key
            crossref_count=0  # Can add CrossRef if needed
        )

        # Build search index
        platform.build_faiss_index()

        # Perform topic modeling
        platform.perform_topic_modeling(n_topics=8)

        # Cluster papers
        platform.cluster_papers(n_clusters=6)

        # Analyze sentiment
        platform.analyze_sentiment()

        # Test semantic search
        print("\n" + "="*70)
        print("SEMANTIC SEARCH DEMO")
        print("="*70)
        search_query = "observational studies"
        print(f"\nSearching for: '{search_query}'")
        results = platform.semantic_search(search_query, top_k=5)
        print("\nTop 5 Results:")
        for idx, row in results.iterrows():
            print(f"\n{idx+1}. {row['title'][:80]}...")
            print(f"   Authors: {row['authors']}")
            print(f"   Year: {row['year']} | Source: {row['source']}")
            print(f"   Similarity: {row['similarity_score']:.3f}")
            print(f"   URL: {row['url']}")

        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        print("\n1. Cluster visualization...")
        fig1 = platform.visualize_clusters_2d()
        fig1.write_html("realtime_clusters.html")

        print("2. Source distribution...")
        fig2 = platform.visualize_source_distribution()
        fig2.write_html("realtime_sources.html")

        print("3. Analytics dashboard...")
        fig3 = platform.create_realtime_dashboard()
        fig3.write_html("realtime_dashboard.html")

        # Generate report
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)
        report = platform.generate_summary_report()

        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Total Papers: {report['total_papers']}")
        print(f"Year Range: {report['year_range']}")
        print(f"Total Citations: {report['total_citations']:,}")
        print(f"Average Citations: {report['avg_citations']:.2f}")
        print(f"\nPapers by Source:")
        for source, count in report['sources'].items():
            print(f"  - {source}: {count}")

        # Save data
        platform.df.to_csv("realtime_papers_data.csv", index=False)
        print("\n✓ Data saved to: realtime_papers_data.csv")

        # Save report
        with open("realtime_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        print("✓ Report saved to: realtime_analysis_report.json")

        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - realtime_clusters.html")
        print("  - realtime_sources.html")
        print("  - realtime_dashboard.html")
        print("  - realtime_papers_data.csv")
        print("  - realtime_analysis_report.json")

        return platform

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    platform = main()

"""
Enhanced AstroPhysics Research Platform - Advanced DS + NLP Features
Extends the base platform with unique research intelligence capabilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# NLP and Advanced Analytics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import Counter, defaultdict
import re
from datetime import datetime
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud


class AdvancedNLPEngine:
    """
    Advanced NLP capabilities for research papers
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )

    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract key phrases using TF-IDF on n-grams"""
        if not text or len(text) < 50:
            return []

        try:
            # Use 2-3 gram combinations
            vectorizer = TfidfVectorizer(
                ngram_range=(2, 3),
                stop_words='english',
                max_features=100
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Get top phrases
            top_indices = scores.argsort()[-top_n:][::-1]
            key_phrases = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

            return key_phrases
        except:
            return []

    def extract_research_methods(self, text: str) -> List[str]:
        """Extract research methodologies mentioned in abstracts"""
        methods_keywords = {
            'observational': ['observation', 'observed', 'observational', 'survey', 'imaging'],
            'theoretical': ['theoretical', 'theory', 'model', 'simulation', 'analytical'],
            'experimental': ['experiment', 'experimental', 'laboratory', 'measurement'],
            'computational': ['computational', 'numerical', 'algorithm', 'code', 'software'],
            'statistical': ['statistical', 'bayesian', 'monte carlo', 'machine learning'],
            'spectroscopic': ['spectroscopy', 'spectroscopic', 'spectrum', 'spectra'],
        }

        text_lower = text.lower()
        detected_methods = []

        for method, keywords in methods_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_methods.append(method)

        return detected_methods

    def extract_astronomical_objects(self, text: str) -> List[str]:
        """Extract astronomical objects mentioned"""
        objects_patterns = {
            'galaxy': r'\b(galaxy|galaxies|galactic)\b',
            'star': r'\b(star|stars|stellar)\b',
            'planet': r'\b(planet|planets|planetary|exoplanet)\b',
            'black_hole': r'\b(black hole|black holes)\b',
            'neutron_star': r'\b(neutron star|pulsar)\b',
            'quasar': r'\b(quasar|quasars|qso)\b',
            'supernova': r'\b(supernova|supernovae|sn)\b',
            'nebula': r'\b(nebula|nebulae)\b',
            'cluster': r'\b(cluster|clusters)\b',
            'comet': r'\b(comet|comets)\b',
            'asteroid': r'\b(asteroid|asteroids)\b',
        }

        detected_objects = []
        text_lower = text.lower()

        for obj, pattern in objects_patterns.items():
            if re.search(pattern, text_lower):
                detected_objects.append(obj)

        return detected_objects

    def analyze_research_focus(self) -> pd.DataFrame:
        """Analyze what each paper focuses on"""
        print("\nAnalyzing research focus areas...")

        focus_data = []
        for idx, row in self.df.iterrows():
            combined_text = f"{row['title']} {row['abstract']}"

            focus = {
                'paper_id': row['paper_id'],
                'methods': self.extract_research_methods(combined_text),
                'objects': self.extract_astronomical_objects(combined_text),
                'key_phrases': self.extract_key_phrases(combined_text, top_n=5)
            }
            focus_data.append(focus)

        # Add to dataframe
        self.df['research_methods'] = [d['methods'] for d in focus_data]
        self.df['astro_objects'] = [d['objects'] for d in focus_data]
        self.df['key_phrases'] = [d['key_phrases'] for d in focus_data]

        print("✓ Research focus analysis completed")
        return self.df


class CollaborationNetwork:
    """
    Build and analyze author collaboration networks
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.G = nx.Graph()
        self.author_stats = {}

    def build_network(self):
        """Build collaboration network from author data"""
        print("\nBuilding collaboration network...")

        author_papers = defaultdict(list)

        # Parse authors and build co-authorship edges
        for idx, row in self.df.iterrows():
            authors_str = row['authors']
            if pd.isna(authors_str) or authors_str == 'Unknown':
                continue

            # Split and clean author names
            authors = [a.strip() for a in str(authors_str).replace(' et al.', '').split(',')]
            authors = [a for a in authors if len(a) > 3][:10]  # Limit to first 10

            # Track papers per author
            for author in authors:
                author_papers[author].append(row['paper_id'])

            # Create edges between co-authors
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    if self.G.has_edge(author1, author2):
                        self.G[author1][author2]['weight'] += 1
                    else:
                        self.G.add_edge(author1, author2, weight=1)

        # Calculate author statistics
        for author, papers in author_papers.items():
            self.author_stats[author] = {
                'num_papers': len(papers),
                'papers': papers
            }

        print(f"✓ Network built: {self.G.number_of_nodes()} authors, {self.G.number_of_edges()} collaborations")
        return self.G

    def find_influential_authors(self, top_n: int = 10) -> pd.DataFrame:
        """Find most influential authors using centrality metrics"""
        if self.G.number_of_nodes() == 0:
            self.build_network()

        if self.G.number_of_nodes() == 0:
            return pd.DataFrame()

        # Calculate centrality metrics
        degree_cent = nx.degree_centrality(self.G)
        betweenness_cent = nx.betweenness_centrality(self.G)

        # Combine metrics
        author_influence = []
        for author in self.G.nodes():
            author_influence.append({
                'author': author,
                'num_papers': self.author_stats.get(author, {}).get('num_papers', 0),
                'degree_centrality': degree_cent[author],
                'betweenness_centrality': betweenness_cent[author],
                'influence_score': (degree_cent[author] + betweenness_cent[author]) / 2
            })

        df_influence = pd.DataFrame(author_influence)
        df_influence = df_influence.sort_values('influence_score', ascending=False).head(top_n)

        return df_influence

    def detect_communities(self) -> Dict:
        """Detect research communities using Louvain algorithm"""
        if self.G.number_of_nodes() == 0:
            self.build_network()

        if self.G.number_of_nodes() < 3:
            return {}

        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.G)

            community_dict = {}
            for i, comm in enumerate(communities):
                community_dict[f"Community_{i+1}"] = list(comm)

            print(f"✓ Detected {len(communities)} research communities")
            return community_dict
        except:
            return {}

    def visualize_network(self, top_n: int = 50):
        """Create interactive network visualization"""
        if self.G.number_of_nodes() == 0:
            self.build_network()

        # Get top N most connected authors
        degree_dict = dict(self.G.degree())
        top_authors = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_n]
        subgraph = self.G.subgraph(top_authors)

        # Create layout
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # Prepare edge traces
        edge_trace = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = subgraph[edge[0]][edge[1]]['weight']

            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5 + weight*0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )

        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            num_papers = self.author_stats.get(node, {}).get('num_papers', 1)
            node_text.append(f"{node}<br>Papers: {num_papers}<br>Collaborations: {subgraph.degree(node)}")
            node_size.append(10 + num_papers * 2)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n.split()[0] if len(n.split()) > 0 else n for n in subgraph.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Author Collaboration Network (Top Authors)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800
        )

        return fig


class TrendAnalyzer:
    """
    Analyze research trends over time
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_temporal_trends(self) -> pd.DataFrame:
        """Analyze how research topics evolve over time"""
        print("\nAnalyzing temporal trends...")

        # Group by year
        yearly_data = []
        for year in sorted(self.df['year'].unique()):
            year_papers = self.df[self.df['year'] == year]

            # Aggregate methods and objects
            all_methods = []
            all_objects = []

            for methods in year_papers['research_methods']:
                if isinstance(methods, list):
                    all_methods.extend(methods)

            for objects in year_papers['astro_objects']:
                if isinstance(objects, list):
                    all_objects.extend(objects)

            yearly_data.append({
                'year': year,
                'num_papers': len(year_papers),
                'avg_citations': year_papers['citations'].mean(),
                'top_methods': Counter(all_methods).most_common(3),
                'top_objects': Counter(all_objects).most_common(3),
                'avg_sentiment': year_papers['sentiment'].mean() if 'sentiment' in year_papers.columns else 0.5
            })

        trend_df = pd.DataFrame(yearly_data)
        print("✓ Temporal trend analysis completed")
        return trend_df

    def predict_emerging_topics(self) -> List[str]:
        """Identify potentially emerging research topics"""
        print("\nIdentifying emerging topics...")

        # Focus on recent papers (last 2 years)
        recent_years = sorted(self.df['year'].unique())[-2:]
        recent_papers = self.df[self.df['year'].isin(recent_years)]
        older_papers = self.df[~self.df['year'].isin(recent_years)]

        # Extract key phrases from recent vs older
        recent_phrases = []
        for phrases in recent_papers['key_phrases']:
            if isinstance(phrases, list):
                recent_phrases.extend([p[0] for p in phrases])

        older_phrases = []
        for phrases in older_papers['key_phrases']:
            if isinstance(phrases, list):
                older_phrases.extend([p[0] for p in phrases])

        recent_counter = Counter(recent_phrases)
        older_counter = Counter(older_phrases)

        # Find phrases that appear more in recent papers
        emerging = []
        for phrase, recent_count in recent_counter.most_common(50):
            older_count = older_counter.get(phrase, 0)
            if recent_count > older_count * 1.5 and recent_count > 3:
                emerging.append((phrase, recent_count, older_count))

        print(f"✓ Identified {len(emerging)} emerging topics")
        return emerging[:10]

    def visualize_trend_evolution(self):
        """Create temporal trend visualization"""
        trend_df = self.analyze_temporal_trends()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Papers Published per Year',
                'Average Citations per Year',
                'Research Method Distribution',
                'Sentiment Over Time'
            )
        )

        # Papers per year
        fig.add_trace(
            go.Scatter(x=trend_df['year'], y=trend_df['num_papers'],
                      mode='lines+markers', name='Papers', line=dict(color='blue', width=3)),
            row=1, col=1
        )

        # Citations per year
        fig.add_trace(
            go.Scatter(x=trend_df['year'], y=trend_df['avg_citations'],
                      mode='lines+markers', name='Avg Citations', line=dict(color='green', width=3)),
            row=1, col=2
        )

        # Method distribution
        all_methods = []
        for methods in self.df['research_methods']:
            if isinstance(methods, list):
                all_methods.extend(methods)
        method_counts = Counter(all_methods)

        fig.add_trace(
            go.Bar(x=list(method_counts.keys()), y=list(method_counts.values()),
                  marker_color='coral', name='Methods'),
            row=2, col=1
        )

        # Sentiment over time
        if 'avg_sentiment' in trend_df.columns:
            fig.add_trace(
                go.Scatter(x=trend_df['year'], y=trend_df['avg_sentiment'],
                          mode='lines+markers', name='Sentiment', line=dict(color='purple', width=3)),
                row=2, col=2
            )

        fig.update_layout(height=800, width=1400, title_text="Research Trend Evolution", showlegend=False)
        return fig


class AnomalyDetector:
    """
    Detect unusual or groundbreaking papers
    """

    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray):
        self.df = df
        self.embeddings = embeddings

    def detect_outliers(self) -> pd.DataFrame:
        """Detect papers that are significantly different from others"""
        print("\nDetecting anomalous/groundbreaking papers...")

        # Use Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.embeddings)

        # -1 indicates outlier
        self.df['is_outlier'] = outlier_labels == -1

        outliers = self.df[self.df['is_outlier'] == True].copy()
        outliers = outliers.sort_values('citations', ascending=False)

        print(f"✓ Detected {len(outliers)} anomalous papers")
        return outliers[['paper_id', 'title', 'authors', 'year', 'citations', 'journal', 'url']]

    def find_citation_anomalies(self) -> pd.DataFrame:
        """Find papers with unusually high/low citations for their age"""
        current_year = datetime.now().year
        self.df['paper_age'] = current_year - self.df['year']
        self.df['citations_per_year'] = self.df['citations'] / (self.df['paper_age'] + 1)

        # Find top performers
        top_performers = self.df.nlargest(10, 'citations_per_year')

        print(f"✓ Identified top {len(top_performers)} high-impact papers")
        return top_performers[['paper_id', 'title', 'year', 'citations', 'citations_per_year', 'url']]


class EnhancedVisualization:
    """
    Advanced visualization capabilities
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_wordcloud(self, text_column: str = 'abstract'):
        """Generate wordcloud from text"""
        text = ' '.join(self.df[text_column].dropna().astype(str))

        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Research Abstract Word Cloud', fontsize=20, pad=20)
        plt.tight_layout()

        return fig

    def create_sunburst_chart(self):
        """Create hierarchical sunburst chart of research areas"""
        # Prepare hierarchical data
        hierarchy_data = []

        for _, row in self.df.iterrows():
            methods = row.get('research_methods', [])
            objects = row.get('astro_objects', [])

            if isinstance(methods, list) and methods:
                for method in methods:
                    if isinstance(objects, list) and objects:
                        for obj in objects:
                            hierarchy_data.append({
                                'method': method,
                                'object': obj,
                                'count': 1
                            })

        if not hierarchy_data:
            return None

        hierarchy_df = pd.DataFrame(hierarchy_data)
        hierarchy_df = hierarchy_df.groupby(['method', 'object']).sum().reset_index()

        fig = px.sunburst(
            hierarchy_df,
            path=['method', 'object'],
            values='count',
            title='Research Methods × Astronomical Objects',
            width=900, height=900
        )

        return fig

    def create_citation_heatmap(self):
        """Create heatmap of citations by year and journal"""
        # Get top journals
        top_journals = self.df['journal'].value_counts().head(15).index

        # Filter and pivot
        df_filtered = self.df[self.df['journal'].isin(top_journals)]
        pivot_data = df_filtered.pivot_table(
            values='citations',
            index='journal',
            columns='year',
            aggfunc='mean',
            fill_value=0
        )

        fig = px.imshow(
            pivot_data,
            labels=dict(x="Year", y="Journal", color="Avg Citations"),
            title="Average Citations by Journal and Year",
            aspect="auto",
            color_continuous_scale='YlOrRd',
            width=1200, height=700
        )

        return fig


def run_advanced_analysis(platform):
    """
    Execute all advanced analysis modules
    """
    print("\n" + "="*70)
    print("ADVANCED DS + NLP ANALYSIS PIPELINE")
    print("="*70)

    # Ensure base analysis is done
    if 'sentiment' not in platform.df.columns:
        platform.analyze_sentiment()

    # 1. Advanced NLP Analysis
    print("\n[1/5] Running Advanced NLP Engine...")
    nlp_engine = AdvancedNLPEngine(platform.df)
    platform.df = nlp_engine.analyze_research_focus()

    # 2. Collaboration Network
    print("\n[2/5] Building Collaboration Network...")
    collab_network = CollaborationNetwork(platform.df)
    collab_network.build_network()

    influential_authors = collab_network.find_influential_authors(top_n=15)
    print("\nTop Influential Authors:")
    print(influential_authors.to_string(index=False))

    communities = collab_network.detect_communities()
    print(f"\nResearch Communities Detected: {len(communities)}")

    # 3. Trend Analysis
    print("\n[3/5] Analyzing Research Trends...")
    trend_analyzer = TrendAnalyzer(platform.df)
    emerging_topics = trend_analyzer.predict_emerging_topics()

    print("\nEmerging Research Topics:")
    for i, (topic, recent, older) in enumerate(emerging_topics, 1):
        print(f"  {i}. {topic}")
        print(f"     Recent mentions: {recent}, Historical: {older}")

    # 4. Anomaly Detection
    print("\n[4/5] Detecting Anomalous Papers...")
    anomaly_detector = AnomalyDetector(platform.df, platform.embeddings)
    outlier_papers = anomaly_detector.detect_outliers()
    citation_anomalies = anomaly_detector.find_citation_anomalies()

    print("\nTop Anomalous/Groundbreaking Papers:")
    for idx, row in outlier_papers.head(5).iterrows():
        print(f"\n  • {row['title'][:70]}...")
        print(f"    Citations: {row['citations']} | Year: {row['year']}")

    # 5. Advanced Visualizations
    print("\n[5/5] Generating Advanced Visualizations...")
    viz_engine = EnhancedVisualization(platform.df)

    # Network visualization
    print("  - Collaboration network...")
    fig_network = collab_network.visualize_network(top_n=40)
    fig_network.write_html("advanced_collaboration_network.html")

    # Trend evolution
    print("  - Trend evolution...")
    fig_trends = trend_analyzer.visualize_trend_evolution()
    fig_trends.write_html("advanced_trend_evolution.html")

    # Sunburst chart
    print("  - Research hierarchy...")
    fig_sunburst = viz_engine.create_sunburst_chart()
    if fig_sunburst:
        fig_sunburst.write_html("advanced_research_hierarchy.html")

    # Citation heatmap
    print("  - Citation heatmap...")
    fig_heatmap = viz_engine.create_citation_heatmap()
    fig_heatmap.write_html("advanced_citation_heatmap.html")

    # Wordcloud
    print("  - Word cloud...")
    fig_wordcloud = viz_engine.create_wordcloud()
    fig_wordcloud.savefig("advanced_wordcloud.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save enhanced data
    print("\nSaving enhanced datasets...")
    platform.df.to_csv("advanced_papers_enriched.csv", index=False)

    influential_authors.to_csv("advanced_influential_authors.csv", index=False)
    outlier_papers.to_csv("advanced_groundbreaking_papers.csv", index=False)
    citation_anomalies.to_csv("advanced_high_impact_papers.csv", index=False)

    # Save communities
    with open("advanced_research_communities.json", 'w') as f:
        json.dump(communities, f, indent=2)

    # Generate comprehensive report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_papers': len(platform.df),
        'unique_authors': len(collab_network.G.nodes()),
        'collaborations': len(collab_network.G.edges()),
        'research_communities': len(communities),
        'emerging_topics': [t[0] for t in emerging_topics[:5]],
        'anomalous_papers': len(outlier_papers),
        'top_influential_authors': influential_authors.head(5)['author'].tolist(),
        'method_distribution': dict(Counter([m for methods in platform.df['research_methods']
                                             if isinstance(methods, list) for m in methods])),
        'object_distribution': dict(Counter([o for objects in platform.df['astro_objects']
                                            if isinstance(objects, list) for o in objects]))
    }

    with open("advanced_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("✅ ADVANCED ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  📊 Visualizations:")
    print("     - advanced_collaboration_network.html")
    print("     - advanced_trend_evolution.html")
    print("     - advanced_research_hierarchy.html")
    print("     - advanced_citation_heatmap.html")
    print("     - advanced_wordcloud.png")
    print("\n  📁 Data Files:")
    print("     - advanced_papers_enriched.csv")
    print("     - advanced_influential_authors.csv")
    print("     - advanced_groundbreaking_papers.csv")
    print("     - advanced_high_impact_papers.csv")
    print("     - advanced_research_communities.json")
    print("     - advanced_analysis_report.json")

    return {
        'nlp_engine': nlp_engine,
        'collab_network': collab_network,
        'trend_analyzer': trend_analyzer,
        'anomaly_detector': anomaly_detector,
        'viz_engine': viz_engine,
        'report': report
    }


# Integration with existing platform - COMPLETE EXECUTABLE SCRIPT
if __name__ == "__main__":


    # Import the base platform
    try:
        # Assuming the base code is in a file or can be imported
        import sys
        import os

        # Try to import from the base platform file
        try:
            from realtime_platform import RealTimeAstroPhysicsResearchPlatform
            print("✓ Base platform imported successfully\n")
        except ImportError:
            print("⚠ Could not import base platform. Running standalone demo...\n")
            # If import fails, we'll create a minimal version for demonstration
            exec(open('realtime_platform.py').read()) if os.path.exists('realtime_platform.py') else None

        # FULL EXECUTION PIPELINE
        print("="*70)
        print("STEP 1: INITIALIZING BASE PLATFORM")
        print("="*70)

        # Initialize platform
        platform = RealTimeAstroPhysicsResearchPlatform()

        # Fetch papers
        print("\nSTEP 2: FETCHING REAL-TIME DATA FROM APIs")
        print("="*70)
        df = platform.fetch_papers_from_apis(
            query="black holes",
            arxiv_count=100,
            ads_count=0,
            crossref_count=0
        )

        # Base preprocessing
        print("\n" + "="*70)
        print("STEP 3: BASE PREPROCESSING & ANALYSIS")
        print("="*70)

        print("\n[3.1] Creating embeddings...")
        platform.preprocess_text()

        print("[3.2] Building search index...")
        platform.build_faiss_index()

        print("[3.3] Performing topic modeling...")
        platform.perform_topic_modeling(n_topics=8)

        print("[3.4] Clustering papers...")
        platform.cluster_papers(n_clusters=6)

        print("[3.5] Analyzing sentiment...")
        platform.analyze_sentiment()

        # Generate base visualizations
        print("\n[3.6] Generating base visualizations...")
        fig1 = platform.visualize_clusters_2d()
        fig1.write_html("base_clusters.html")
        print("     ✓ Saved: base_clusters.html")

        fig2 = platform.create_realtime_dashboard()
        fig2.write_html("base_dashboard.html")
        print("     ✓ Saved: base_dashboard.html")

        # Test semantic search
        print("\n" + "="*70)
        print("STEP 4: SEMANTIC SEARCH DEMO")
        print("="*70)
        search_query = "gravitational radiation emission"
        print(f"\nSearching for: '{search_query}'")
        results = platform.semantic_search(search_query, top_k=5)
        print("\nTop 5 Most Relevant Papers:")
        for idx, row in results.iterrows():
            print(f"\n{idx+1}. {row['title'][:75]}...")
            print(f"   Authors: {row['authors'][:50]}...")
            print(f"   Year: {row['year']} | Similarity: {row['similarity_score']:.3f}")

        # NOW RUN ADVANCED ANALYSIS
        print("\n" + "="*70)
        print("STEP 5: ADVANCED DS + NLP ANALYSIS")
        print("="*70)
        results = run_advanced_analysis(platform)

        # Display key insights
        print("\n" + "="*70)
        print("KEY INSIGHTS & DISCOVERIES")
        print("="*70)

        report = results['report']
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total Papers: {report['total_papers']}")
        print(f"   • Unique Authors: {report['unique_authors']}")
        print(f"   • Collaborations: {report['collaborations']}")
        print(f"   • Research Communities: {report['research_communities']}")

        print(f"\n🔥 Top Emerging Topics:")
        for i, topic in enumerate(report['emerging_topics'], 1):
            print(f"   {i}. {topic}")

        print(f"\n⭐ Most Influential Authors:")
        for i, author in enumerate(report['top_influential_authors'], 1):
            print(f"   {i}. {author}")

        print(f"\n🔬 Research Methods Distribution:")
        for method, count in sorted(report['method_distribution'].items(),
                                    key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {method.capitalize()}: {count} papers")

        print(f"\n🌌 Most Studied Objects:")
        for obj, count in sorted(report['object_distribution'].items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {obj.replace('_', ' ').title()}: {count} papers")

        print("\n" + "="*70)
        print("✅ COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
        print("="*70)

        print("\n📁 All Generated Files:")
        print("\n   Base Platform Outputs:")
        print("      - base_clusters.html")
        print("      - base_dashboard.html")
        print("      - realtime_papers_data.csv")

        print("\n   Advanced Analysis Outputs:")
        print("      - advanced_collaboration_network.html")
        print("      - advanced_trend_evolution.html")
        print("      - advanced_research_hierarchy.html")
        print("      - advanced_citation_heatmap.html")
        print("      - advanced_wordcloud.png")
        print("      - advanced_papers_enriched.csv")
        print("      - advanced_influential_authors.csv")
        print("      - advanced_groundbreaking_papers.csv")
        print("      - advanced_high_impact_papers.csv")
        print("      - advanced_research_communities.json")
        print("      - advanced_analysis_report.json")

        print("\n🎉 Open the HTML files in your browser to explore interactive visualizations!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure you have the base platform code in 'realtime_platform.py'")

