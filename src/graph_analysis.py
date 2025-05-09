import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import random
from typing import Dict, List, Tuple, Set
import logging
from collections import defaultdict, Counter
import simpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import community  # python-louvain package
import ast
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import pickle
import os

# Set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/graph_analysis.log'),
        logging.StreamHandler()
    ]
)

def safe_eval(x):
    """Safely evaluate string representations of lists"""
    try:
        if pd.isna(x) or x == '':
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            # Try using ast.literal_eval
            try:
                result = ast.literal_eval(x)
                if isinstance(result, list):
                    return result
                return []
            except:
                # If failed, try simple string splitting
                return [item.strip() for item in x.split(',') if item.strip()]
        return []
    except:
        return []

class ResearchGraphBuilder:
    def __init__(self, data_path: str):
        """Initialize research graph builder"""
        self.data = pd.read_csv(data_path, encoding='utf-8')
        self.G = nx.Graph()  # Main graph
        self.author_graph = nx.Graph()  # Author collaboration graph
        self.keyword_graph = nx.Graph()  # Keyword graph
        self.time_graph = nx.Graph()  # Time graph
        self.institution_graph = nx.Graph()  # Institution graph
        self.year_graph = nx.Graph()  # Year graph
        
        self.analysis_results = {}  # Dictionary to store all computed metrics and results
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data"""
        # Process author lists
        self.data['authors'] = self.data['authors'].apply(safe_eval)
        
        # Process keywords
        self.data['keywords'] = self.data['keywords'].apply(safe_eval)
        
        # Process institution information
        self.data['institution_display_names'] = self.data['institution_display_names'].apply(safe_eval)
        
        # Process detailed institution information
        self.data['detailed_affiliations'] = self.data['detailed_affiliations'].apply(safe_eval)
        
        # Ensure year is numeric
        self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
        
        # Generate unique ID
        self.data['paper_id'] = self.data.index
        
        logging.info(f"Data preprocessing completed, processed {len(self.data)} records")

    def build_main_graph(self):
        """Build the main graph containing all nodes and multiple edge types"""
        # Add all nodes
        for _, row in self.data.iterrows():
            paper_id = str(row['paper_id'])  # Use index as ID
            self.G.add_node(paper_id, 
                           doi=row.get('doi'),
                           title=row.get('title'),
                           year=row.get('year'),
                           institution=row.get('institution_display_names'),
                           agency=row.get('Agency'),
                           keywords=row.get('keywords'),
                           abstract=row.get('abstract'))
        
        # Add author shared edges
        self._add_author_edges()
        
        # Add keyword shared edges
        self._add_keyword_edges()
        
        # Add institution shared edges
        self._add_institution_edges()
        
        self.analysis_results['main_graph_summary'] = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G) if self.G else 0
        }
        
        logging.info(f"Main graph construction completed, containing {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")

    def _add_author_edges(self):
        """Add author shared edges"""
        for i, row in self.data.iterrows():
            authors = row['authors']
            paper_id = str(row['paper_id'])
            
            for j, other_row in self.data.iterrows():
                if i < j:  # Avoid duplication
                    other_authors = other_row['authors']
                    common_authors = set(authors) & set(other_authors)
                    if common_authors:
                        self.G.add_edge(paper_id, str(other_row['paper_id']),
                                      weight=len(common_authors),
                                      edge_type='author_shared')

    def _add_keyword_edges(self):
        """Add keyword shared edges"""
        # Use TF-IDF to calculate keyword similarity
        all_keywords = []
        for keywords in self.data['keywords']:
            all_keywords.append(' '.join(map(str, keywords)))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_keywords)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        for i in range(len(self.data)):
            for j in range(i+1, len(self.data)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.1:  # Set threshold
                    self.G.add_edge(str(self.data.iloc[i]['paper_id']),
                                  str(self.data.iloc[j]['paper_id']),
                                  weight=similarity,
                                  edge_type='keyword_similarity')

    def _add_institution_edges(self):
        """Add institution shared edges"""
        for i, row in self.data.iterrows():
            institutions_i = set(row['institution_display_names'])
            for j, other_row in self.data.iterrows():
                if i < j:  # Avoid duplication
                    institutions_j = set(other_row['institution_display_names'])
                    common_institutions = institutions_i & institutions_j
                    if common_institutions:
                        self.G.add_edge(str(row['paper_id']),
                                      str(other_row['paper_id']),
                                      weight=len(common_institutions),
                                      edge_type='institution_shared')

    def _normalize_institution_name(self, name: str) -> str:
        """Normalize institution name"""
        if pd.isna(name):
            return ""
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove extra spaces and punctuation
        name = ' '.join(name.split())
        name = name.strip('.,;:')
        
        # Remove common suffixes
        suffixes = ['inc', 'llc', 'ltd', 'limited', 'corporation', 'corp']
        for suffix in suffixes:
            if name.endswith(f' {suffix}'):
                name = name[:-len(suffix)].strip()
        
        # Standardize common abbreviations
        abbreviations = {
            'univ': 'university',
            'inst': 'institute',
            'tech': 'technology',
            'nat': 'national',
            'int': 'international',
            'res': 'research',
            'lab': 'laboratory',
            'hosp': 'hospital',
            'med': 'medical',
            'sch': 'school',
            'col': 'college'
        }
        
        for abbr, full in abbreviations.items():
            name = name.replace(f' {abbr} ', f' {full} ')
        
        return name

    def build_institution_graph(self):
        """Build institution subgraph"""
        # Preprocess institution names
        institution_map = {}
        normalized_institutions = {}
        
        for _, row in self.data.iterrows():
            institutions = row['institution_display_names']
            paper_id = str(row['paper_id'])
            
            # Standardize institution names
            for institution in institutions:
                if pd.isna(institution):
                    continue
                    
                normalized_name = self._normalize_institution_name(institution)
                if not normalized_name:
                    continue
                    
                if normalized_name not in institution_map:
                    institution_map[normalized_name] = set()
                institution_map[normalized_name].add(institution)
                
                if normalized_name not in normalized_institutions:
                    normalized_institutions[normalized_name] = set()
                normalized_institutions[normalized_name].add(paper_id)
        
        # Add nodes and edges
        for norm_name, papers in normalized_institutions.items():
            if not self.institution_graph.has_node(norm_name):
                self.institution_graph.add_node(norm_name, 
                                              papers=papers,
                                              original_names=list(institution_map[norm_name]))
        
        # Add edges between institutions
        for i, (norm_name1, papers1) in enumerate(normalized_institutions.items()):
            for norm_name2, papers2 in list(normalized_institutions.items())[i+1:]:
                common_papers = len(papers1 & papers2)
                if common_papers > 0:
                    self.institution_graph.add_edge(norm_name1, norm_name2, 
                                                 weight=common_papers,
                                                 common_papers=list(papers1 & papers2))
        
        # Remove isolated nodes
        isolated_nodes = [node for node in self.institution_graph.nodes() 
                         if self.institution_graph.degree(node) == 0]
        self.institution_graph.remove_nodes_from(isolated_nodes)
        
        logging.info(f"Institution graph construction completed, containing {self.institution_graph.number_of_nodes()} nodes and {self.institution_graph.number_of_edges()} edges")

    def build_year_graph(self) -> None:
        """Build year graph based on shared authors between papers in adjacent years."""
        logging.info("Building year graph...")
        self.year_graph = nx.Graph()

        # Filter out rows with invalid years
        valid_year_data = self.data.dropna(subset=['year'])
        valid_year_data['year'] = valid_year_data['year'].astype(int) # Ensure integer years

        # Get all unique years and sort them
        years = sorted(valid_year_data['year'].unique())

        if not years:
            logging.warning("No valid years found to build year graph.")
            return

        # Add year nodes and store associated paper IDs
        year_papers = defaultdict(list)
        for year in years:
            papers_in_year = valid_year_data[valid_year_data['year'] == year]['paper_id'].tolist()
            if papers_in_year:
                 self.year_graph.add_node(year, papers=papers_in_year)
                 year_papers[year] = set(papers_in_year) # Use set for faster lookups

        # Add edges based on shared authors between papers in adjacent years (sliding window)
        window_size = 3 # Connect years up to 3 apart
        edge_count = 0
        authors_by_paper = pd.Series(self.data['authors'].values, index=self.data['paper_id']).to_dict()

        for i in range(len(years)):
            for j in range(i + 1, min(i + window_size + 1, len(years))):
                year1, year2 = years[i], years[j]

                # Ensure both years are actually nodes in the graph
                if year1 not in year_papers or year2 not in year_papers:
                    continue

                papers1 = year_papers[year1]
                papers2 = year_papers[year2]

                # Find common authors efficiently
                authors1 = set(author for paper_id in papers1 for author in authors_by_paper.get(paper_id, []))
                authors2 = set(author for paper_id in papers2 for author in authors_by_paper.get(paper_id, []))
                common_authors_count = len(authors1.intersection(authors2))

                if common_authors_count > 0:
                    self.year_graph.add_edge(year1, year2, weight=common_authors_count, type='author_continuity')
                    edge_count += 1

        self.analysis_results['year_graph_summary'] = {
             'num_nodes': self.year_graph.number_of_nodes(),
             'num_edges': self.year_graph.number_of_edges(),
             'density': nx.density(self.year_graph) if self.year_graph else 0
        }
        logging.info(f"Year graph construction completed: {self.year_graph.number_of_nodes()} nodes, {self.year_graph.number_of_edges()} edges.")

    def compute_year_metrics(self) -> Dict:
        """Calculate year graph metrics"""
        if not self.year_graph:
            return {}
        
        metrics = {
            'density': nx.density(self.year_graph),
            'year_stats': {},
            'temporal_communities': self._detect_temporal_communities()
        }
        
        # Calculate statistics for each year
        for year in self.year_graph.nodes():
            papers = self.year_graph.nodes[year]['papers']
            metrics['year_stats'][year] = {
                'paper_count': len(papers),
                'degree': self.year_graph.degree(year),
                'clustering': nx.clustering(self.year_graph, year)
            }
        
        return metrics

    def _detect_temporal_communities(self) -> Dict:
        """Detect time communities"""
        communities = {}
        
        try:
            # Use Louvain method to detect time communities
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(self.year_graph)
            
            # Organize results into time periods
            time_periods = defaultdict(list)
            for year, community_id in partition.items():
                time_periods[community_id].append(year)
            
            # Sort each time period
            for community_id, years in time_periods.items():
                years.sort()
                communities[f'period_{community_id}'] = {
                    'years': years,
                    'start_year': min(years),
                    'end_year': max(years),
                    'paper_count': sum(len(self.year_graph.nodes[year]['papers']) for year in years)
                }
            
            logging.info(f"Detected {len(communities)} time communities")
        except Exception as e:
            logging.warning(f"Time community detection failed: {str(e)}")
        
        return communities

    def compute_advanced_metrics(self):
        """Calculate advanced network metrics"""
        self.metrics = {}
        
        try:
            # Community detection - using multiple methods
            self.metrics['communities'] = {}
            
            # 1. Community detection for institution graph
            if self.institution_graph.number_of_nodes() > 0 and self.institution_graph.number_of_edges() > 0:
                try:
                    # Use Louvain method
                    import community.community_louvain as community_louvain
                    louvain_partition = community_louvain.best_partition(self.institution_graph)
                    self.metrics['communities']['institution_louvain'] = louvain_partition
                    logging.info(f"Louvain method detected {len(set(louvain_partition.values()))} communities")
                except Exception as e:
                    logging.warning(f"Louvain community detection failed: {str(e)}")
                
                try:
                    # Use Girvan-Newman method
                    communities_iterator = nx.community.girvan_newman(self.institution_graph)
                    first_communities = next(communities_iterator)
                    self.metrics['communities']['institution_gn'] = first_communities
                    logging.info(f"Girvan-Newman method detected {len(first_communities)} communities")
                except Exception as e:
                    logging.warning(f"Girvan-Newman community detection failed: {str(e)}")
                
                try:
                    # Use Label Propagation method
                    lp_communities = list(nx.community.label_propagation_communities(self.institution_graph))
                    self.metrics['communities']['institution_lp'] = lp_communities
                    logging.info(f"Label Propagation method detected {len(lp_communities)} communities")
                except Exception as e:
                    logging.warning(f"Label Propagation community detection failed: {str(e)}")
            
            # 2. Community detection for year graph
            if self.year_graph.number_of_nodes() > 0 and self.year_graph.number_of_edges() > 0:
                try:
                    # Use Louvain method
                    year_louvain = community.best_partition(self.year_graph)
                    self.metrics['communities']['year_louvain'] = year_louvain
                    logging.info(f"Year graph Louvain method detected {len(set(year_louvain.values()))} communities")
                except Exception as e:
                    logging.warning(f"Year graph community detection failed: {str(e)}")
            
        except Exception as e:
            logging.warning(f"Community detection process failed: {str(e)}")
            self.metrics['communities'] = {}
        
        # Influence analysis
        self.metrics['influence'] = self._compute_influence_metrics()
        
        # Temporal evolution analysis
        self.metrics['temporal_evolution'] = {
            'year_activity': self._compute_year_activity(),
            'topic_evolution': self._compute_topic_evolution(),
            'institution_collaboration': self._compute_institution_collaboration()
        }
        
        return self.metrics

    def _compute_influence_metrics(self) -> Dict:
        """Calculate influence metrics"""
        influence = {}
        
        try:
            # Calculate PageRank
            if self.institution_graph.number_of_nodes() > 0:
                pagerank = nx.pagerank(self.institution_graph)
                influence['pagerank'] = pagerank
            
            # Calculate centrality metrics
            if self.institution_graph.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(self.institution_graph)
                betweenness_centrality = nx.betweenness_centrality(self.institution_graph)
                influence['degree_centrality'] = degree_centrality
                influence['betweenness_centrality'] = betweenness_centrality
        except Exception as e:
            logging.warning(f"Influence metrics calculation failed: {str(e)}")
            influence = {}
        
        return influence

    def _compute_year_activity(self):
        """Calculate year-wise research activity"""
        year_activity = defaultdict(int)
        for _, row in self.data.iterrows():
            year = row.get('year')
            if pd.notnull(year) and year != -1:
                year_activity[int(year)] += 1
        return dict(year_activity)

    def _compute_topic_evolution(self):
        """Calculate research topic evolution"""
        topic_evolution = defaultdict(lambda: defaultdict(int))
        for _, row in self.data.iterrows():
            year = row.get('year')
            keywords = row.get('keywords', [])
            if pd.notnull(year) and year != -1 and keywords:
                for keyword in keywords:
                    topic_evolution[int(year)][str(keyword)] += 1
        return dict(topic_evolution)

    def _compute_institution_collaboration(self):
        """Calculate institution collaboration temporal evolution"""
        collaboration_evolution = defaultdict(lambda: defaultdict(int))
        for _, row in self.data.iterrows():
            year = row.get('year')
            institutions = row.get('institution_display_names', [])
            if pd.notnull(year) and year != -1 and institutions:
                for i in range(len(institutions)):
                    for j in range(i+1, len(institutions)):
                        collaboration_evolution[int(year)][(str(institutions[i]), str(institutions[j]))] += 1
        return dict(collaboration_evolution)

    def analyze_temporal_trends(self):
        """Analyze temporal trends"""
        trends = {}
        
        # 1. Publication counts trend
        year_counts = self._compute_year_activity()
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        
        # Linear regression prediction
        X = np.array(years).reshape(-1, 1)
        y = np.array(counts)
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for next 3 years
        future_years = np.array(range(max(years) + 1, max(years) + 4)).reshape(-1, 1)
        future_predictions = model.predict(future_years)
        
        trends['publication_counts'] = {
            'historical': dict(zip(years, counts)),
            'prediction': dict(zip(future_years.flatten(), future_predictions)),
            'growth_rate': model.coef_[0],
            'r2_score': model.score(X, y)
        }
        
        # 2. Topic evolution trend
        topic_evolution = self._compute_topic_evolution()
        top_topics = self._get_top_topics(topic_evolution)
        
        trends['topic_evolution'] = {
            'top_topics': top_topics,
            'topic_trends': self._analyze_topic_trends(topic_evolution, top_topics)
        }
        
        # 3. Institution collaboration trend
        collaboration_evolution = self._compute_institution_collaboration()
        trends['collaboration_evolution'] = {
            'historical': collaboration_evolution,
            'top_collaborations': self._get_top_collaborations(collaboration_evolution)
        }
        
        return trends
        
    def _get_top_topics(self, topic_evolution: Dict) -> List[str]:
        """Get most popular research topics"""
        topic_counts = defaultdict(int)
        for year_topics in topic_evolution.values():
            for topic, count in year_topics.items():
                topic_counts[topic] += count
                
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
    def _analyze_topic_trends(self, topic_evolution: Dict, top_topics: List[Tuple[str, int]]) -> Dict:
        """Analyze topic trends"""
        topic_trends = {}
        years = sorted(topic_evolution.keys())
        
        for topic, _ in top_topics:
            topic_counts = [topic_evolution[year].get(topic, 0) for year in years]
            
            # Calculate trend
            X = np.array(years).reshape(-1, 1)
            y = np.array(topic_counts)
            model = LinearRegression()
            model.fit(X, y)
            
            topic_trends[topic] = {
                'historical': dict(zip(years, topic_counts)),
                'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                'growth_rate': model.coef_[0],
                'r2_score': model.score(X, y)
            }
            
        return topic_trends
        
    def _get_top_collaborations(self, collaboration_evolution: Dict) -> List[Tuple[Tuple[str, str], int]]:
        """Get most frequent institution collaborations"""
        collaboration_counts = defaultdict(int)
        for year_collaborations in collaboration_evolution.values():
            for collaboration, count in year_collaborations.items():
                collaboration_counts[collaboration] += count
                
        return sorted(collaboration_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
    def predict_future_trends(self, num_years: int = 3) -> Dict:
        """Predict future trends"""
        predictions = {}
        
        # 1. Predict publication counts
        year_counts = self._compute_year_activity()
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        
        # Use polynomial regression for prediction
        X = np.array(years).reshape(-1, 1)
        y = np.array(counts)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        
        future_years = np.array(range(max(years) + 1, max(years) + num_years + 1)).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        future_predictions = model.predict(future_poly)
        
        predictions['publication_counts'] = dict(zip(future_years.flatten(), future_predictions))
        
        # 2. Predict topic evolution
        topic_evolution = self._compute_topic_evolution()
        top_topics = self._get_top_topics(topic_evolution)
        
        topic_predictions = {}
        for topic, _ in top_topics:
            topic_counts = [topic_evolution[year].get(topic, 0) for year in years]
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, topic_counts)
            future_topic_predictions = model.predict(future_poly)
            topic_predictions[topic] = dict(zip(future_years.flatten(), future_topic_predictions))
            
        predictions['topic_evolution'] = topic_predictions
        
        return predictions

    def analyze_institution_network(self) -> Dict:
        """Analyze institution collaboration network"""
        network_analysis = {}
        
        # 1. Calculate institution centrality metrics
        centrality_metrics = self._compute_institution_centrality()
        network_analysis['centrality'] = centrality_metrics
        
        # 2. Identify key collaboration networks
        key_collaborations = self._identify_key_collaborations()
        network_analysis['key_collaborations'] = key_collaborations
        
        # 3. Analyze institution community structure
        communities = self._analyze_institution_communities()
        network_analysis['communities'] = communities
        
        # 4. Calculate institution influence
        influence_metrics = self._compute_institution_influence()
        network_analysis['influence'] = influence_metrics
        
        return network_analysis

    def _compute_institution_centrality(self) -> Dict:
        """Calculate institution centrality metrics"""
        centrality = {}
        
        # Degree centrality
        centrality['degree'] = nx.degree_centrality(self.institution_graph)
        
        # Betweenness centrality
        centrality['betweenness'] = nx.betweenness_centrality(self.institution_graph)
        
        # Closeness centrality
        centrality['closeness'] = nx.closeness_centrality(self.institution_graph)
        
        # Eigenvector centrality
        centrality['eigenvector'] = nx.eigenvector_centrality(self.institution_graph, max_iter=1000)
        
        # Calculate combined centrality score
        combined_centrality = {}
        for node in self.institution_graph.nodes():
            combined_centrality[node] = (
                centrality['degree'][node] * 0.3 +
                centrality['betweenness'][node] * 0.3 +
                centrality['closeness'][node] * 0.2 +
                centrality['eigenvector'][node] * 0.2
            )
        centrality['combined'] = combined_centrality
        
        return centrality

    def _identify_key_collaborations(self) -> Dict:
        """Identify key collaboration networks"""
        key_collaborations = {}
        
        # 1. Key collaborations based on collaboration strength
        edge_weights = [(u, v, d['weight']) for u, v, d in self.institution_graph.edges(data=True)]
        edge_weights.sort(key=lambda x: x[2], reverse=True)
        key_collaborations['by_strength'] = edge_weights[:10]
        
        # 2. Key collaborations based on collaboration frequency
        collaboration_frequency = defaultdict(int)
        for _, row in self.data.iterrows():
            institutions = row['institution_display_names']
            for i in range(len(institutions)):
                for j in range(i+1, len(institutions)):
                    collaboration_frequency[(institutions[i], institutions[j])] += 1
        
        key_collaborations['by_frequency'] = sorted(
            collaboration_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 3. Key collaborations based on influence
        influence_based = []
        for u, v, weight in edge_weights:
            u_influence = self.institution_graph.nodes[u].get('influence', 0)
            v_influence = self.institution_graph.nodes[v].get('influence', 0)
            influence_score = (u_influence + v_influence) * weight
            influence_based.append((u, v, influence_score))
        
        key_collaborations['by_influence'] = sorted(influence_based, key=lambda x: x[2], reverse=True)[:10]
        
        return key_collaborations

    def _analyze_institution_communities(self) -> Dict:
        """Analyze institution community structure"""
        communities_result = {}
        
        # 1. Use Louvain method to detect communities
        try:
            if self.institution_graph.number_of_edges() > 0:
                import community.community_louvain as community_louvain
                louvain_partition = community_louvain.best_partition(self.institution_graph)
                communities_result['louvain'] = self._process_communities(louvain_partition)
                logging.info(f"Louvain method detected {len(set(louvain_partition.values()))} communities")
        except Exception as e:
            logging.warning(f"Louvain community detection failed: {str(e)}")
        
        # 2. Use Girvan-Newman method to detect communities
        try:
            if self.institution_graph.number_of_edges() > 0:
                gn_communities = list(nx.community.girvan_newman(self.institution_graph))
                if gn_communities:
                    first_partition = tuple(sorted(c) for c in next(gn_communities))
                    communities_result['girvan_newman'] = self._process_communities(first_partition)
                    logging.info(f"Girvan-Newman method detected {len(first_partition)} communities")
        except Exception as e:
            logging.warning(f"Girvan-Newman community detection failed: {str(e)}")
        
        # 3. Use Label Propagation method to detect communities
        try:
            if self.institution_graph.number_of_edges() > 0:
                lp_communities = list(nx.community.label_propagation_communities(self.institution_graph))
                communities_result['label_propagation'] = self._process_communities(lp_communities)
                logging.info(f"Label Propagation method detected {len(list(lp_communities))} communities")
        except Exception as e:
            logging.warning(f"Label Propagation community detection failed: {str(e)}")
        
        return communities_result

    def _process_communities(self, communities) -> Dict:
        """Process community detection results"""
        processed = {}
        community_features = {}
        
        if isinstance(communities, dict):
            # Process Louvain method results
            community_groups = defaultdict(list)
            for node, community_id in communities.items():
                community_groups[community_id].append(node)
            processed = dict(community_groups)
        elif isinstance(communities, (list, tuple)):
            # Process results from other methods
            for i, community in enumerate(communities):
                if isinstance(community, (set, list, tuple)):
                    processed[i] = list(community)
                else:
                    processed[i] = [community]
        
        # Calculate features for each community
        for community_id, members in processed.items():
            if not members:
                continue
            
            subgraph = self.institution_graph.subgraph(members)
            community_features[community_id] = {
                'size': len(members),
                'density': nx.density(subgraph) if subgraph.number_of_edges() > 0 else 0,
                'top_institutions': sorted(
                    [(inst, self.institution_graph.degree(inst)) for inst in members],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'internal_connections': subgraph.number_of_edges()
            }
        
        return {
            'groups': processed,
            'features': community_features
        }

    def _compute_institution_influence(self) -> Dict:
        """Calculate institution influence"""
        influence = {}
        
        # 1. PageRank influence
        influence['pagerank'] = nx.pagerank(self.institution_graph)
        
        # 2. HITS algorithm influence
        try:
            hubs, authorities = nx.hits(self.institution_graph)
            influence['hubs'] = hubs
            influence['authorities'] = authorities
        except Exception as e:
            logging.warning(f"HITS algorithm calculation failed: {str(e)}")
        
        # 3. Collaboration network-based influence
        collaboration_influence = {}
        for node in self.institution_graph.nodes():
            # Calculate weighted degree centrality
            weighted_degree = sum(d['weight'] for _, _, d in self.institution_graph.edges(node, data=True))
            # Calculate average influence of neighbors
            neighbor_influence = sum(
                self.institution_graph.nodes[neighbor].get('pagerank', 0)
                for neighbor in self.institution_graph.neighbors(node)
            )
            collaboration_influence[node] = (weighted_degree + neighbor_influence) / 2
        
        influence['collaboration'] = collaboration_influence
        
        return influence

    def save_to_file(self, filepath: str = "output/analysis_results.pkl"):
        """Save analysis results to file
        
        Args:
            filepath: Save path
        """
        # Create output directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save analyzer instance
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logging.info(f"Analysis results have been saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str = "output/analysis_results.pkl"):
        """Load analysis results from file
        
        Args:
            filepath: File path
            
        Returns:
            ResearchGraphBuilder: Loaded analyzer instance
        """
        # Load analyzer instance
        with open(filepath, 'rb') as f:
            analyzer = pickle.load(f)
        
        logging.info(f"Analysis results loaded from {filepath}")
        return analyzer

    def build_author_graph(self):
        """Build author collaboration network graph"""
        self.author_graph = nx.Graph()
        
        # Create author-paper mapping
        author_papers = defaultdict(list)
        for _, row in self.data.iterrows():
            paper_id = str(row['paper_id'])
            authors = row['authors']
            
            for author in authors:
                if pd.notna(author) and author:
                    author_papers[author].append(paper_id)
        
        # Add author nodes
        for author, papers in author_papers.items():
            self.author_graph.add_node(author, papers=papers)
        
        # Add collaboration edges
        for i, (author1, papers1) in enumerate(author_papers.items()):
            for author2, papers2 in list(author_papers.items())[i+1:]:
                common_papers = set(papers1) & set(papers2)
                if common_papers:
                    self.author_graph.add_edge(author1, author2, 
                                             weight=len(common_papers),
                                             common_papers=list(common_papers))
        
        # Remove isolated nodes
        isolated_nodes = [node for node in self.author_graph.nodes() 
                         if self.author_graph.degree(node) == 0]
        self.author_graph.remove_nodes_from(isolated_nodes)
        
        logging.info(f"Author collaboration graph construction completed, containing {self.author_graph.number_of_nodes()} nodes and {self.author_graph.number_of_edges()} edges")
    
    def build_keyword_graph(self):
        """Build keyword co-occurrence network graph"""
        self.keyword_graph = nx.Graph()
        
        # Create keyword-paper mapping
        keyword_papers = defaultdict(list)
        for _, row in self.data.iterrows():
            paper_id = str(row['paper_id'])
            keywords = row['keywords']
            
            for keyword in keywords:
                if pd.notna(keyword) and keyword:
                    keyword_papers[keyword].append(paper_id)
        
        # Add keyword nodes
        for keyword, papers in keyword_papers.items():
            self.keyword_graph.add_node(keyword, papers=papers)
        
        # Add co-occurrence edges
        for i, (keyword1, papers1) in enumerate(keyword_papers.items()):
            for keyword2, papers2 in list(keyword_papers.items())[i+1:]:
                common_papers = set(papers1) & set(papers2)
                if common_papers:
                    self.keyword_graph.add_edge(keyword1, keyword2, 
                                              weight=len(common_papers),
                                              common_papers=list(common_papers))
        
        # Remove isolated nodes
        isolated_nodes = [node for node in self.keyword_graph.nodes() 
                          if self.keyword_graph.degree(node) == 0]
        self.keyword_graph.remove_nodes_from(isolated_nodes)
        
        logging.info(f"Keyword graph construction completed, containing {self.keyword_graph.number_of_nodes()} nodes and {self.keyword_graph.number_of_edges()} edges")

    def compute_graph_metrics(self, graph: nx.Graph, graph_name: str, top_n: int = 10) -> Dict:
        """Computes standard network metrics for a given graph and identifies top nodes."""
        logging.info(f"Computing metrics for {graph_name}...")
        metrics = {}
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes == 0:
            logging.warning(f"{graph_name} is empty. Skipping metric calculation.")
            return {
                "num_nodes": 0, "num_edges": 0, "density": 0,
                "degree_centrality": {"top_n": [], "all": {}},
                "betweenness_centrality": {"top_n": [], "all": {}},
                "clustering_coefficient": {"average": 0, "all": {}}
            }

        metrics['num_nodes'] = num_nodes
        metrics['num_edges'] = num_edges
        metrics['density'] = nx.density(graph)

        # Degree Centrality
        try:
            degree_centrality = nx.degree_centrality(graph)
            # Sort nodes by degree centrality and get top N
            top_degree_nodes = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:top_n]
            metrics['degree_centrality'] = {"top_n": top_degree_nodes, "all": degree_centrality}
        except Exception as e:
            logging.warning(f"Could not compute degree centrality for {graph_name}: {e}")
            metrics['degree_centrality'] = {"top_n": [], "all": {}}

        # Betweenness Centrality (can be slow for large graphs)
        try:
            # Consider using k for approximation if graph is too large: k=min(1000, num_nodes//10)
            betweenness_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)
            top_betweenness_nodes = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)[:top_n]
            metrics['betweenness_centrality'] = {"top_n": top_betweenness_nodes, "all": betweenness_centrality}
        except Exception as e:
            logging.warning(f"Could not compute betweenness centrality for {graph_name}: {e}")
            metrics['betweenness_centrality'] = {"top_n": [], "all": {}}

        # Clustering Coefficient
        try:
            clustering_coefficient = nx.clustering(graph)
            average_clustering = nx.average_clustering(graph) # Avg clustering coeff
            metrics['clustering_coefficient'] = {"average": average_clustering, "all": clustering_coefficient}
        except Exception as e:
            logging.warning(f"Could not compute clustering coefficient for {graph_name}: {e}")
            metrics['clustering_coefficient'] = {"average": 0, "all": {}}

        logging.info(f"Finished computing metrics for {graph_name}.")
        return metrics

    def compute_all_graph_metrics(self):
         """Compute metrics for all constructed graphs."""
         if self.institution_graph:
             self.analysis_results['institution_graph_metrics'] = self.compute_graph_metrics(self.institution_graph, "Institution Graph")
         if self.author_graph:
             self.analysis_results['author_graph_metrics'] = self.compute_graph_metrics(self.author_graph, "Author Graph")
         if self.keyword_graph:
             self.analysis_results['keyword_graph_metrics'] = self.compute_graph_metrics(self.keyword_graph, "Keyword Graph")
         # Add other graphs if needed (e.g., main graph G, year graph)
         # Metrics for year_graph might be less standard - density already captured

    def run_community_detection(self, graph: nx.Graph, graph_name: str) -> Dict:
        """Runs Louvain community detection and returns community features."""
        logging.info(f"Running Louvain community detection for {graph_name}...")
        communities_result = {}
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes == 0 or num_edges == 0:
             logging.warning(f"{graph_name} has no nodes or edges, skipping community detection.")
             return {"num_communities": 0, "top_communities_by_size": [], "modularity": None, "features": {}}

        try:
            # Ensure the graph is treated as undirected for Louvain if necessary
            # Louvain implementation might handle this internally, but explicit check is safe.
            partition = community.best_partition(graph)
            modularity = community.modularity(partition, graph)

            community_groups = defaultdict(list)
            for node, community_id in partition.items():
                community_groups[community_id].append(node)

            num_communities = len(community_groups)
            logging.info(f"Louvain detected {num_communities} communities in {graph_name} with modularity {modularity:.4f}")

            # Calculate features for each community
            community_features = {}
            community_sizes = []
            for community_id, members in community_groups.items():
                 if not members: continue
                 subgraph = graph.subgraph(members)
                 size = len(members)
                 density = nx.density(subgraph) if subgraph.number_of_edges() > 0 else 0
                 # Get top 5 central nodes within the community (using degree as proxy)
                 subgraph_degrees = dict(subgraph.degree())
                 top_nodes_in_community = sorted(members, key=lambda node: subgraph_degrees.get(node, 0), reverse=True)[:5]

                 community_features[str(community_id)] = { # Ensure key is string for JSON
                     'size': size,
                     'density': density,
                     'top_central_nodes': top_nodes_in_community,
                     'internal_edges': subgraph.number_of_edges()
                     # Could add avg clustering coeff within community
                 }
                 community_sizes.append({'id': str(community_id), 'size': size})

            # Get top communities by size
            top_communities = sorted(community_sizes, key=lambda x: x['size'], reverse=True)[:10]

            communities_result = {
                "algorithm": "Louvain",
                "num_communities": num_communities,
                "modularity": modularity,
                "top_communities_by_size": top_communities,
                "features": community_features # Contains details for all communities
            }

        except Exception as e:
            logging.warning(f"Louvain community detection failed for {graph_name}: {e}")
            communities_result = {"num_communities": 0, "top_communities_by_size": [], "modularity": None, "features": {}}

        return communities_result

    def run_all_community_detection(self):
        """Run community detection for relevant graphs."""
        if self.institution_graph:
            self.analysis_results['institution_communities'] = self.run_community_detection(self.institution_graph, "Institution Graph")
        if self.author_graph:
             self.analysis_results['author_communities'] = self.run_community_detection(self.author_graph, "Author Graph")
        if self.keyword_graph:
             self.analysis_results['keyword_communities'] = self.run_community_detection(self.keyword_graph, "Keyword Graph")

    def compute_temporal_analysis(self, top_n_keywords=20, top_n_collaborations=10):
        """Compute temporal analysis statistics"""
        logging.info("Computing temporal analysis...")
        
        # Initialize analysis results if not present
        if 'temporal_analysis' not in self.analysis_results:
            self.analysis_results['temporal_analysis'] = {}
        
        # Process keywords over time
        keywords_by_year = {}
        
        
        all_keywords = []
        for kws in self.data['Keywords']:
            if isinstance(kws, list):
                all_keywords.extend([kw for kw in kws if kw and pd.notna(kw)])
            elif isinstance(kws, str):
                
                all_keywords.extend([kw.strip() for kw in kws.split(';') if kw.strip()])
            
        
        # Get publication years
        years = self.data['year'].dropna().astype(int).unique()
        years = sorted(years)
        
        # 确保years不为空
        if len(years) == 0:
            logging.warning("No valid years found in the data.")
            return
        
        # Calculate keyword trends over time
        for year in years:
            year_data = self.data[self.data['year'] == year]
            keywords_this_year = []
            
            
            for kws in year_data['Keywords']:
                if isinstance(kws, list):
                    keywords_this_year.extend([kw for kw in kws if kw and pd.notna(kw)])
                elif isinstance(kws, str):
                    keywords_this_year.extend([kw.strip() for kw in kws.split(';') if kw.strip()])
                
            
            keyword_counts = Counter(keywords_this_year)
            keywords_by_year[year] = keyword_counts
        
        # Get top keywords overall
        all_keyword_counts = Counter(all_keywords)
        top_keywords = [item[0] for item in all_keyword_counts.most_common(top_n_keywords)]
        
        # Track these top keywords over time
        keyword_trends = {keyword: [] for keyword in top_keywords}
        for year in years:
            year_counts = keywords_by_year.get(year, Counter())
            for keyword in top_keywords:
                keyword_trends[keyword].append((year, year_counts.get(keyword, 0)))
        
        # Store results
        self.analysis_results['temporal_analysis']['keyword_trends'] = keyword_trends
        self.analysis_results['temporal_analysis']['top_keywords'] = top_keywords
        
        # Process author collaborations over time
        if 'author_collaborations_by_year' not in self.analysis_results['temporal_analysis']:
            collaborations_by_year = {}
            
            for year in years:
                year_data = self.data[self.data['year'] == year]
                collaborations_this_year = []
                
                
                for authors in year_data['authors']:
                    if isinstance(authors, list):
                        
                        if len(authors) > 1:
                            for i in range(len(authors)):
                                for j in range(i+1, len(authors)):
                                    collaborations_this_year.append(tuple(sorted([authors[i], authors[j]])))
                    elif isinstance(authors, str):
                        
                        author_list = [a.strip() for a in authors.split(';') if a.strip()]
                        if len(author_list) > 1:
                            for i in range(len(author_list)):
                                for j in range(i+1, len(author_list)):
                                    collaborations_this_year.append(tuple(sorted([author_list[i], author_list[j]])))
                
                collab_counts = Counter(collaborations_this_year)
                collaborations_by_year[year] = collab_counts
            
            # Store collaboration results
            self.analysis_results['temporal_analysis']['collaborations_by_year'] = collaborations_by_year
            
            # Get top collaborations overall
            all_collabs = []
            for year_collabs in collaborations_by_year.values():
                all_collabs.extend(year_collabs.elements())
            
            top_collaborations = [item[0] for item in Counter(all_collabs).most_common(top_n_collaborations)]
            
            # Track top collaborations over time
            collab_trends = {collab: [] for collab in top_collaborations}
            for year in years:
                year_counts = collaborations_by_year.get(year, Counter())
                for collab in top_collaborations:
                    collab_trends[collab].append((year, year_counts.get(collab, 0)))
            
            self.analysis_results['temporal_analysis']['top_collaborations'] = top_collaborations
            self.analysis_results['temporal_analysis']['collaboration_trends'] = collab_trends
        
        # Process publication trends
        publication_counts = self.data['year'].value_counts().sort_index()
        publication_trend = [(int(year), count) for year, count in publication_counts.items()]
        self.analysis_results['temporal_analysis']['publication_trend'] = publication_trend
        
        return self.analysis_results['temporal_analysis']

    def save_analysis_to_json(self, filepath: str = "results/graph_metrics.json"):
        """Saves the computed analysis results to a JSON file."""
        logging.info(f"Saving analysis results to JSON: {filepath}")
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert complex objects (like sets or numpy types) to JSON serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                # Handle NaN and Inf
                if np.isnan(obj): return None # Or 'NaN' as string
                if np.isinf(obj): return None # Or 'Infinity'/' -Infinity'
                return float(obj)
            elif isinstance(obj, (set, tuple)):
                return list(obj)
            elif isinstance(obj, (datetime, pd.Timestamp)):
                 return obj.isoformat()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()} # Ensure keys are strings
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            # Add other type conversions if needed
            return obj

        try:
            # Apply conversion recursively to the results dictionary
            serializable_results = convert_to_serializable(self.analysis_results)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            logging.info(f"Successfully saved analysis results to {filepath}")
        except TypeError as e:
            logging.error(f"Error serializing results to JSON: {e}. Results might be incomplete.")
            # Optionally try saving partially or logging the problematic part
            # print("Problematic data snippet:", self.analysis_results) # Be careful printing large dicts
        except Exception as e:
            logging.error(f"An unexpected error occurred while saving results to JSON: {e}")

class ResearchDES:
    def __init__(self, env: simpy.Environment, graph_builder: ResearchGraphBuilder):
        """Initialize DES simulator"""
        self.env = env
        self.graph_builder = graph_builder
        self.publication_queue = simpy.Resource(env, capacity=10)  # Assume there are 10 reviewers
        self.papers = {}  # Store paper status
        self.stats = defaultdict(list)
        self.institution_stats = defaultdict(lambda: defaultdict(list))  # Institution-wise statistics
        self.reviewer_pool = []  # Reviewer pool
        self.review_assignments = defaultdict(list)  # Reviewer assignment status
        self.paper_status = {}  # New paper status dictionary
        
    def add_reviewer(self, reviewer_id: str, institution: str, expertise: List[str]) -> None:
        """Add reviewer to reviewer pool"""
        self.reviewer_pool.append({
            'id': reviewer_id,
            'institution': institution,
            'expertise': expertise,
            'current_assignments': 0,
            'max_assignments': 5  # Each reviewer can review up to 5 papers at a time
        })
        
    def assign_reviewers(self, paper_id: str, authors: List[str]) -> bool:
        """Assign reviewers to paper"""
        if paper_id in self.review_assignments:
            return True  # Already assigned reviewers
        
        # Filter out reviewers from the author's institution
        available_reviewers = [
            r for r in self.reviewer_pool
            if r['institution'] not in authors and r['current_assignments'] < r['max_assignments']
        ]
        
        if len(available_reviewers) < 3:
            return False  # Not enough reviewers
        
        # Randomly select 3 reviewers
        selected_reviewers = random.sample(available_reviewers, 3)
        
        # Update reviewer assignment
        self.review_assignments[paper_id] = [r['id'] for r in selected_reviewers]
        self.paper_status[paper_id]['reviewers'] = [r['id'] for r in selected_reviewers]
        
        # Update reviewer's current task count
        for reviewer in selected_reviewers:
            reviewer['current_assignments'] += 1
        
        return True
        
    def submit_paper(self, paper_id: str, institution: str):
        """Submit paper"""
        self.papers[paper_id] = {
            'status': 'submitted',
            'submission_time': self.env.now,
            'review_time': None,
            'publication_time': None,
            'institution': institution,
            'reviewers': [],
            'review_status': 'pending'
        }
        self.stats['submissions'].append((paper_id, self.env.now))
        self.institution_stats[institution]['submissions'].append((paper_id, self.env.now))
        
        # Assign reviewers
        if self.assign_reviewers(paper_id, [institution]):
            self.papers[paper_id]['review_status'] = 'assigned'
            self.papers[paper_id]['reviewers'] = self.review_assignments[paper_id]
            
    def review_paper(self, paper_id: str) -> None:
        """Review paper"""
        if paper_id not in self.review_assignments:
            logging.warning(f"Paper {paper_id} not assigned reviewers")
            return
        
        for reviewer_id in self.review_assignments[paper_id]:
            # Simulate review process
            review_time = random.uniform(1, 5)  # 1-5 day review time
            review_score = random.uniform(1, 5)  # 1-5 point rating
            
            # Record review results
            if 'review_results' not in self.paper_status[paper_id]:
                self.paper_status[paper_id]['review_results'] = []
            
            self.paper_status[paper_id]['review_results'].append({
                'reviewer': reviewer_id,
                'time': review_time,
                'score': review_score
            })
            self.paper_status[paper_id]['reviews_completed'] += 1
            
            # Update reviewer status
            for reviewer in self.reviewer_pool:
                if reviewer['id'] == reviewer_id:
                    reviewer['current_assignments'] -= 1
                    break
        
        # Use longest review time as paper review completion time
        self.papers[paper_id]['review_time'] = self.env.now
        self.papers[paper_id]['review_status'] = 'completed'
        self.stats['reviews'].append((paper_id, self.env.now))
        self.institution_stats[self.papers[paper_id]['institution']]['reviews'].append((paper_id, self.env.now))
        
    def publish_paper(self, paper_id: str):
        """Publish paper"""
        if paper_id not in self.papers or self.papers[paper_id]['review_status'] != 'completed':
            logging.warning(f"Paper {paper_id} not completed review, cannot publish")
            return
            
        publication_time = random.uniform(15, 45)  # 15-45 day publication time
        yield self.env.timeout(publication_time)
        self.papers[paper_id]['publication_time'] = self.env.now
        self.papers[paper_id]['status'] = 'published'
        self.stats['publications'].append((paper_id, self.env.now))
        self.institution_stats[self.papers[paper_id]['institution']]['publications'].append((paper_id, self.env.now))
        
    def simulate_review_process(self) -> None:
        """Simulate paper review process"""
        # Initialize reviewer pool
        self.reviewer_pool = []
        self.review_assignments = {}
        self.paper_status = {}
        
        # Select experienced reviewers from institution graph
        experienced_institutions = sorted(
            self.institution_graph.nodes(),
            key=lambda x: self.institution_graph.degree(x),
            reverse=True
        )[:20]  # Select top 20 institutions with highest degree centrality as primary reviewer source
        
        # Add 3-5 reviewers for each institution
        for institution in experienced_institutions:
            reviewer_count = random.randint(3, 5)
            for i in range(reviewer_count):
                reviewer_id = f"{institution}_reviewer_{i}"
                expertise = random.sample(list(self.institution_graph.nodes()), 3)  # Randomly select 3 areas as expertise
                self.add_reviewer(reviewer_id, institution, expertise)
        
        # Simulate paper submission and review process
        for paper in self.papers:
            paper_id = paper['id']
            self.paper_status[paper_id] = {
                'status': 'submitted',
                'submitted_time': time.time(),
                'reviews_completed': 0,
                'reviewers': [],
                'review_results': []
            }
            
            # Assign reviewers
            success = self.assign_reviewers(paper_id, paper.get('authors', []))
            if not success:
                logging.warning(f"Paper {paper_id} not assigned enough reviewers")
                continue
            
            # Simulate review process
            self.review_paper(paper_id)
            
            # Decide whether to publish based on review results
            if self.paper_status[paper_id]['reviews_completed'] >= 2:  # At least 2 completed reviews needed
                avg_score = sum(review['score'] for review in self.paper_status[paper_id]['review_results']) / \
                           len(self.paper_status[paper_id]['review_results'])
                if avg_score >= 3.0:  # Average score 3.0 or above can be published
                    self.paper_status[paper_id]['status'] = 'published'
                else:
                    self.paper_status[paper_id]['status'] = 'rejected'
            else:
                self.paper_status[paper_id]['status'] = 'pending'

    def get_review_statistics(self) -> Dict:
        """Get review statistics"""
        stats = {
            'total_papers': len(self.papers),
            'submitted': 0,
            'under_review': 0,
            'reviewed': 0,
            'published': 0,
            'rejected': 0,
            'pending': 0
        }
        
        for paper_id, status in self.paper_status.items():
            stats['submitted'] += 1
            if status['status'] == 'submitted':
                stats['under_review'] += 1
            elif status['status'] == 'published':
                stats['published'] += 1
                stats['reviewed'] += 1
            elif status['status'] == 'rejected':
                stats['rejected'] += 1
                stats['reviewed'] += 1
            elif status['status'] == 'pending':
                stats['pending'] += 1
        
        return stats

    def run_simulation(self, num_papers: int = 100):
        """Run simulation"""
        # Initialize reviewer pool
        for i in range(20):  # Add 20 reviewers
            expertise = random.sample(['economics', 'health', 'education', 'technology', 'social'], 
                                   k=random.randint(1, 3))
            institution = random.choice(list(self.graph_builder.institution_graph.nodes()))
            self.add_reviewer(f"reviewer_{i}", institution, expertise)
            
        institutions = list(self.graph_builder.institution_graph.nodes())
        if not institutions:
            logging.warning("No available institutions, cannot run simulation")
            return
            
        for i in range(num_papers):
            paper_id = f"paper_{i}"
            institution = random.choice(institutions)
            
            # Initialize paper status
            self.paper_status[paper_id] = {
                'status': 'submitted',
                'submitted_time': self.env.now,
                'reviews_completed': 0,
                'reviewers': [],
                'review_results': []
            }
            
            # Submit paper
            self.submit_paper(paper_id, institution)
            
            # Assign reviewers
            if self.assign_reviewers(paper_id, [institution]):
                # Simulate review process
                self.review_paper(paper_id)
                
                # Decide whether to publish based on review results
                if self.paper_status[paper_id]['reviews_completed'] >= 2:  # At least 2 completed reviews needed
                    avg_score = sum(review['score'] for review in self.paper_status[paper_id]['review_results']) / \
                               len(self.paper_status[paper_id]['review_results'])
                    if avg_score >= 3.0:  # Average score 3.0 or above can be published
                        self.paper_status[paper_id]['status'] = 'published'
                        self.stats['publications'].append((paper_id, self.env.now))
                        self.institution_stats[institution]['publications'].append((paper_id, self.env.now))
                    else:
                        self.paper_status[paper_id]['status'] = 'rejected'
                else:
                    self.paper_status[paper_id]['status'] = 'pending'
            
            yield self.env.timeout(random.uniform(1, 7))  # 1-7 day submission interval

def main():
    try:
        # Create log directory
        os.makedirs('logs', exist_ok=True)
        
        # Build graph
        print("Building research graph...")
        graph_builder = ResearchGraphBuilder('../New_And_Original_ResearchOutputs.csv')
        
        print("Building main graph...")
        graph_builder.build_main_graph()
        
        print("Building institution graph...")
        graph_builder.build_institution_graph()
        
        print("Building year graph...")
        graph_builder.build_year_graph()
        
        # Build author and keyword graphs
        print("Building author graph...")
        graph_builder.build_author_graph()
        
        print("Building keyword graph...")
        graph_builder.build_keyword_graph()
        
        # Calculate network metrics
        print("Calculating network metrics...")
        metrics = graph_builder.compute_advanced_metrics()
        
        # Run DES simulation
        print("Running simulation...")
        env = simpy.Environment()
        des = ResearchDES(env, graph_builder)
        env.process(des.run_simulation())
        env.run(until=365)  # Simulate one year
        
        # Output results
        print("\nAnalysis results:")
        print(f"Main graph density: {nx.density(graph_builder.G):.4f}")
        print(f"Institution graph density: {nx.density(graph_builder.institution_graph):.4f}")
        print(f"Year graph density: {nx.density(graph_builder.year_graph):.4f}")
        
        # Output DES statistics
        print("\nSimulation statistics:")
        print(f"Total papers submitted during simulation: {len(des.stats['submissions'])}")
        print(f"Total papers reviewed during simulation: {len(des.stats['reviews'])}")
        print(f"Total papers published during simulation: {len(des.stats['publications'])}")
        
        # Output institution statistics
        print("\nInstitution statistics:")
        for institution, stats in des.institution_stats.items():
            print(f"\nInstitution {institution}:")
            print(f"Total papers submitted: {len(stats['submissions'])}")
            print(f"Total papers reviewed: {len(stats['reviews'])}")
            print(f"Total papers published: {len(stats['publications'])}")
            
        # Output temporal evolution analysis
        print("\nTemporal evolution analysis:")
        year_activity = metrics['temporal_evolution']['year_activity']
        for year in sorted(year_activity.keys()):
            print(f"{year} year: {year_activity[year]} papers")
            
        # Output community detection results
        if metrics['communities'].get('institution_louvain'):
            num_communities = len(set(metrics['communities']['institution_louvain'].values()))
            print(f"\nDetected {num_communities} institution communities")
            
        # Save analysis results
        graph_builder.save_to_file()
        
        # Save statistics to JSON
        os.makedirs('../results', exist_ok=True)
        
        # Prepare statistics dictionary
        statistics = {
            "graph_metrics": {
                "main_graph_density": float(f"{nx.density(graph_builder.G):.4f}"),
                "institution_graph_density": float(f"{nx.density(graph_builder.institution_graph):.4f}"),
                "year_graph_density": float(f"{nx.density(graph_builder.year_graph):.4f}")
            },
            "simulation_statistics": {
                "papers_submitted": len(des.stats['submissions']),
                "papers_reviewed": len(des.stats['reviews']),
                "papers_published": len(des.stats['publications'])
            },
            "institution_statistics": {},
            "temporal_evolution": {str(year): count for year, count in year_activity.items()}
        }
        
        # Add institution statistics
        for institution, stats in des.institution_stats.items():
            statistics["institution_statistics"][institution] = {
                "papers_submitted": len(stats['submissions']),
                "papers_reviewed": len(stats['reviews']),
                "papers_published": len(stats['publications'])
            }
        
        # Add community detection results
        if metrics['communities'].get('institution_louvain'):
            statistics["community_detection"] = {
                "num_communities": len(set(metrics['communities']['institution_louvain'].values()))
            }
        
        # Add detailed review statistics from DES simulation
        review_stats = des.get_review_statistics()
        statistics["simulation_statistics"].update({
            "total_papers": review_stats.get('total_papers', 0),
            "under_review": review_stats.get('under_review', 0),
            "reviewed": review_stats.get('reviewed', 0),
            "published": review_stats.get('published', 0),
            "rejected": review_stats.get('rejected', 0),
            "pending": review_stats.get('pending', 0)
        })
        
        # Add review time statistics if available
        review_times = []
        publication_times = []
        total_process_times = []
        
        for paper_id, paper_info in des.papers.items():
            if paper_info.get('review_time') is not None and paper_info.get('submission_time') is not None:
                review_duration = paper_info['review_time'] - paper_info['submission_time']
                review_times.append(review_duration)
                
            if paper_info.get('publication_time') is not None and paper_info.get('review_time') is not None:
                publication_duration = paper_info['publication_time'] - paper_info['review_time']
                publication_times.append(publication_duration)
                
            if paper_info.get('publication_time') is not None and paper_info.get('submission_time') is not None:
                total_duration = paper_info['publication_time'] - paper_info['submission_time']
                total_process_times.append(total_duration)
        
        if review_times:
            statistics["simulation_statistics"]["review_time_analysis"] = {
                "avg_review_time": sum(review_times) / len(review_times),
                "min_review_time": min(review_times),
                "max_review_time": max(review_times)
            }
            
        if publication_times:
            statistics["simulation_statistics"]["publication_time_analysis"] = {
                "avg_publication_time": sum(publication_times) / len(publication_times),
                "min_publication_time": min(publication_times),
                "max_publication_time": max(publication_times)
            }
            
        if total_process_times:
            statistics["simulation_statistics"]["total_process_time_analysis"] = {
                "avg_total_process_time": sum(total_process_times) / len(total_process_times),
                "min_total_process_time": min(total_process_times),
                "max_total_process_time": max(total_process_times)
            }
        
        # Add reviewer statistics
        if des.reviewer_pool:
            reviewer_stats = {
                "total_reviewers": len(des.reviewer_pool),
                "avg_assignments_per_reviewer": sum(r['current_assignments'] for r in des.reviewer_pool) / len(des.reviewer_pool),
                "max_assignments": max(r['current_assignments'] for r in des.reviewer_pool),
                "expertise_distribution": {}
            }
            
            # Count expertise areas
            expertise_counts = defaultdict(int)
            for reviewer in des.reviewer_pool:
                for area in reviewer['expertise']:
                    expertise_counts[area] += 1
            
            # Get top 5 expertise areas
            top_expertise = sorted(expertise_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            reviewer_stats["top_expertise_areas"] = {area: count for area, count in top_expertise}
            
            statistics["simulation_statistics"]["reviewer_statistics"] = reviewer_stats
        
        # Add advanced metrics based on DES simulation
        if hasattr(des, 'paper_status') and des.paper_status:
            # Calculate acceptance rate
            published_count = sum(1 for status in des.paper_status.values() if status.get('status') == 'published')
            reviewed_count = sum(1 for status in des.paper_status.values() 
                              if status.get('status') in ['published', 'rejected'])
            
            if reviewed_count > 0:
                acceptance_rate = published_count / reviewed_count
            else:
                acceptance_rate = 0
                
            # Calculate average review score
            all_scores = []
            for paper_status in des.paper_status.values():
                if 'review_results' in paper_status:
                    for review in paper_status['review_results']:
                        if 'score' in review:
                            all_scores.append(review['score'])
            
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
            else:
                avg_score = 0
                
            statistics["simulation_statistics"]["advanced_metrics"] = {
                "acceptance_rate": acceptance_rate,
                "average_review_score": avg_score,
                "total_reviews_completed": len(all_scores)
            }
        
        # Save to JSON file
        with open('../results/statistics.json', 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=4, ensure_ascii=False)
        
        print(f"\nStatistics saved to ../results/statistics.json")
        
    except Exception as e:
        print(f"Error occurred during execution: {str(e)}")
        logging.error(f"Error occurred during execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 