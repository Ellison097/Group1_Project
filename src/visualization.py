import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict, Counter
import os

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchGraphVisualizer:
    def __init__(self, graph_analyzer):
        """Initialize visualizer
        
        Args:
            graph_analyzer: GraphAnalyzer instance containing all analysis results
        """
        self.analyzer = graph_analyzer
        self.data = graph_analyzer.data
        self.main_graph = graph_analyzer.G
        self.institution_graph = graph_analyzer.institution_graph
        self.year_graph = graph_analyzer.year_graph
        
    def plot_publication_trend(self) -> go.Figure:
        """Draw publication trend
        
        Returns:
            plotly.graph_objects.Figure: Publication trend chart
        """
        yearly_counts = self.data['year'].value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_counts.index,
            y=yearly_counts.values,
            mode='lines+markers',
            name='Publications Count'
        ))
        
        fig.update_layout(
            title='Publication Trend (1983-2026)',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            template='plotly_white'
        )
        
        return fig
    
    def plot_institution_network(self) -> go.Figure:
        """Draw institution collaboration network
        
        Returns:
            plotly.graph_objects.Figure: Institution network graph
        """
        pos = nx.spring_layout(self.institution_graph)
        
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='rgb(128,128,128)'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            marker=dict(
                showscale=False,
                color='rgb(0,0,255)',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Degree Centrality',
                    xanchor='left',
                    titleside='right'
                )
            ))

        for edge in self.institution_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_degrees = dict(self.institution_graph.degree())
        node_colors = list(node_degrees.values())
        
        for node in self.institution_graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([f'{node}<br>Degree Centrality: {node_degrees[node]:.2f}'])

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Institution Collaboration Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
        
        return fig
    
    def plot_community_distribution(self) -> go.Figure:
        """Draw community distribution
        
        Returns:
            plotly.graph_objects.Figure: Community distribution chart
        """
        # Get communities detected by Louvain method
        communities = self.analyzer.metrics['communities'].get('institution_louvain', {})
        if not communities:
            # If no Louvain result, try other methods
            communities = self.analyzer.metrics['communities'].get('institution_gn', [])
            if communities:
                community_sizes = [len(comm) for comm in communities]
            else:
                return self._create_empty_plot("No community data available")
        else:
            # For Louvain method, need to count nodes for each community ID
            community_sizes = []
            community_groups = defaultdict(list)
            for node, comm_id in communities.items():
                community_groups[comm_id].append(node)
            community_sizes = [len(nodes) for nodes in community_groups.values()]
        
        fig = go.Figure(data=[go.Histogram(x=community_sizes, nbinsx=30)])
        fig.update_layout(
            title='Research Community Size Distribution',
            xaxis_title='Community Size (Nodes)',
            yaxis_title='Number of Communities',
            template='plotly_white'
        )
        
        return fig
    
    def plot_centrality_distribution(self) -> go.Figure:
        """Draw centrality distribution
        
        Returns:
            plotly.graph_objects.Figure: Centrality distribution chart
        """
        centrality = nx.degree_centrality(self.institution_graph)
        values = list(centrality.values())
        
        fig = go.Figure(data=[go.Histogram(x=values)])
        fig.update_layout(
            title='Institution Degree Centrality Distribution',
            xaxis_title='Degree Centrality',
            yaxis_title='Number of Institutions',
            template='plotly_white'
        )
        
        return fig
    
    def plot_temporal_network(self) -> go.Figure:
        """Draw temporal network
        
        Returns:
            plotly.graph_objects.Figure: Temporal network graph
        """
        pos = nx.spring_layout(self.year_graph)
        
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='rgb(128,128,128)'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            marker=dict(
                showscale=False,
                color='rgb(0,0,255)',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Publications Count',
                    xanchor='left',
                    titleside='right'
                )
            ))

        for edge in self.year_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_sizes = [len(self.year_graph.nodes[node]['papers']) 
                     for node in self.year_graph.nodes()]
        
        for node in self.year_graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([f'{node} Year<br>Papers: {len(self.year_graph.nodes[node]["papers"])}'])

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Temporal Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
        
        return fig
    
    def plot_review_statistics(self) -> go.Figure:
        """Draw review statistics
        
        Returns:
            plotly.graph_objects.Figure: Review statistics chart
        """
        stats = self.analyzer.get_review_statistics()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Submitted', 'Under Review', 'Published', 'Rejected', 'Pending'],
                y=[stats['submitted'], stats['under_review'], 
                   stats['published'], stats['rejected'], stats['pending']],
                marker_color=['blue', 'red', 
                             'green', 'yellow', 'purple']
            )
        ])
        
        fig.update_layout(
            title='Paper Review Status Statistics',
            xaxis_title='Status',
            yaxis_title='Number of Papers',
            template='plotly_white'
        )
        
        return fig
    
    def plot_institution_metrics(self) -> go.Figure:
        """Draw institution metrics
        
        Returns:
            plotly.graph_objects.Figure: Institution metrics chart
        """
        # Get institution metrics
        institution_metrics = self.analyzer.metrics.get('influence', {})
        if not institution_metrics:
            return self._create_empty_plot("No institution metrics available")
        
        # Sort institutions by overall score
        sorted_institutions = sorted(
            institution_metrics.items(), 
            key=lambda x: x[1]['overall_score'], 
            reverse=True
        )[:20]  # Top 20
        
        institutions = [inst for inst, _ in sorted_institutions]
        overall_scores = [metrics['overall_score'] for _, metrics in sorted_institutions]
        influence_scores = [metrics['influence_score'] for _, metrics in sorted_institutions]
        collaboration_scores = [metrics['collaboration_score'] for _, metrics in sorted_institutions]
        
        # Create subplots
        fig = make_subplots(rows=1, cols=1)
        
        # Add traces
        fig.add_trace(
            go.Bar(
                y=institutions,
                x=overall_scores,
                name='Overall Score',
                orientation='h',
                marker_color='blue'
            )
        )
        
        fig.update_layout(
            title='Top 20 Institutions by Overall Impact Score',
            xaxis_title='Score',
            template='plotly_white',
            height=800,
            barmode='group'
        )
        
        return fig
    
    def plot_degree_distribution(self) -> go.Figure:
        """Draw degree distribution
        
        Returns:
            plotly.graph_objects.Figure: Degree distribution chart
        """
        degrees = [d for n, d in self.institution_graph.degree()]
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=degree_counts.index,
            y=degree_counts.values,
            mode='lines+markers',
            name='Actual degree distribution'
        ))
        
        # Add power-law fit
        x = degree_counts.index
        y = degree_counts.values
        try:
            popt, _ = curve_fit(lambda x, a, b: a * np.power(x, b), x, y)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=popt[0] * np.power(x, popt[1]),
                mode='lines',
                name=f'Power-law fit (γ = {popt[1]:.2f})',
                line=dict(dash='dash')
            ))
        except Exception as e:
            logger.warning(f"Could not fit power law: {e}")
        
        fig.update_layout(
            title='Institution Network Degree Distribution',
            xaxis_title='Degree',
            yaxis_title='Count',
            xaxis_type='log',
            yaxis_type='log',
            template='plotly_white'
        )
        
        return fig
    
    def plot_community_heatmap(self) -> go.Figure:
        """Draw community heatmap
        
        Returns:
            plotly.graph_objects.Figure: Community heatmap
        """
        # Get communities detected by Louvain method
        communities = self.analyzer.metrics['communities'].get('institution_louvain', {})
        if not communities:
            return self._create_empty_plot("No community data available")
        
        # Organize communities into groups
        community_groups = defaultdict(list)
        for node, comm_id in communities.items():
            community_groups[comm_id].append(node)
        
        n_communities = len(community_groups)
        if n_communities < 2:
            return self._create_empty_plot("Not enough communities to generate heatmap")
        
        # Create inter-community connection matrix
        matrix = np.zeros((n_communities, n_communities))
        for u, v, data in self.institution_graph.edges(data=True):
            if u in communities and v in communities:
                comm_u = communities[u]
                comm_v = communities[v]
                if comm_u != comm_v:
                    matrix[comm_u, comm_v] += data.get('weight', 1)
                    matrix[comm_v, comm_u] += data.get('weight', 1)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=[[0, 'white'], [1, 'blue']],
            showscale=True
        ))
        
        # Add labels
        fig.update_layout(
            title='Community Interaction Heatmap',
            xaxis_title='Community ID',
            yaxis_title='Community ID',
            height=700,
            width=700,
            template='plotly_white'
        )
        
        return fig
    
    def plot_temporal_metrics(self) -> go.Figure:
        """Draw temporal metrics
        
        Returns:
            plotly.graph_objects.Figure: Temporal metrics chart
        """
        # Get temporal metrics
        temporal_metrics = self.analyzer.metrics.get('temporal_evolution', {})
        if not temporal_metrics:
            return self._create_empty_plot("No temporal metrics available")
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=('Clustering Coefficient Over Time', 
                                         'Average Path Length Over Time'))
        
        # Add traces
        years = sorted(temporal_metrics.keys())
        clustering = [temporal_metrics[year]['clustering'] for year in years]
        path_length = [temporal_metrics[year]['path_length'] for year in years]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=clustering,
                mode='lines+markers',
                name='Clustering Coefficient'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=path_length,
                mode='lines+markers',
                name='Average Path Length'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text='Network Evolution Metrics',
            template='plotly_white'
        )
        
        return fig
    
    def plot_year_activity(self) -> go.Figure:
        """Draw yearly research activity
        
        Returns:
            plotly.graph_objects.Figure: Yearly activity chart
        """
        # Get year activity data
        year_activity = self.analyzer._compute_year_activity()
        if not year_activity:
            return self._create_empty_plot("No year activity data available")
        
        years = sorted(year_activity.keys())
        paper_counts = [year_activity[year] for year in years]
        
        fig = go.Figure(data=[
            go.Bar(
                x=years,
                y=paper_counts,
                marker_color='blue'
            )
        ])
        
        fig.update_layout(
            title='Research Activity by Year',
            xaxis_title='Year',
            yaxis_title='Number of Papers',
            template='plotly_white'
        )
        
        return fig
    
    def plot_topic_evolution(self) -> go.Figure:
        """Draw topic evolution
        
        Returns:
            plotly.graph_objects.Figure: Topic evolution chart
        """
        # Get topic evolution data
        topic_evolution = self.analyzer._compute_topic_evolution()
        if not topic_evolution:
            return self._create_empty_plot("No topic evolution data available")
        
        years = sorted(topic_evolution.keys())
        
        # Get top topics
        all_topics = set()
        for year in years:
            all_topics.update(topic_evolution[year].keys())
        
        # Sort topics by total count
        topic_total_counts = {}
        for topic in all_topics:
            topic_total_counts[topic] = sum(
                topic_evolution[year].get(topic, 0) for year in years
            )
        
        top_topic_names = [
            topic for topic, count in sorted(
                topic_total_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5 topics
        ]
        
        # Create traces for each topic
        traces = []
        for topic in top_topic_names:
            y_values = [topic_evolution[year].get(topic, 0) for year in years]
            traces.append(go.Scatter(
                x=years,
                y=y_values,
                name=topic,
                mode='lines+markers'
            ))
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            title='Evolution of Major Research Topics',
            xaxis_title='Year',
            yaxis_title='Number of Papers',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def plot_institution_collaboration(self) -> go.Figure:
        """Draw institution collaboration network
        
        Returns:
            plotly.graph_objects.Figure: Institution collaboration network
        """
        if not self.analyzer.institution_graph:
            return self._create_empty_plot("No institution collaboration data available")
            
        # Get node positions
        pos = nx.spring_layout(self.analyzer.institution_graph)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.analyzer.institution_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            # Use node degree as node size
            degree = self.analyzer.institution_graph.degree(node)
            node_size.append(degree * 10)
            # Set color based on node degree
            if degree > 5:
                node_color.append('darkred')  # High collaboration institutions as dark red
            elif degree > 2:
                node_color.append('red')  # Medium collaboration institutions as red
            else:
                node_color.append('lightsalmon')  # Low collaboration institutions as light salmon
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_width = []
        
        for u, v, data in self.analyzer.institution_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_width.append(data.get('weight', 1))
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Collaboration Count',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=1, color='darkgray')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Institution Collaboration Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[dict(
                             text="",
                             showarrow=False,
                             xref="paper", yref="paper"
                         )],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        return fig
    
    def plot_main_graph(self) -> go.Figure:
        """Draw main research graph
        
        Returns:
            plotly.graph_objects.Figure: Main graph
        """
        if not hasattr(self.analyzer, 'G') or self.analyzer.G.number_of_nodes() == 0:
            return self._create_empty_plot("Main graph data not available or empty")
        
        # Get node positions
        pos = nx.spring_layout(self.analyzer.G)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.analyzer.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            attrs = self.analyzer.G.nodes[node]
            title = attrs.get('title', 'Unknown')
            year = attrs.get('year', 'Unknown')
            doi = attrs.get('doi', 'Unknown')
            
            node_text.append(f"Title: {title}<br>Year: {year}<br>DOI: {doi}")
            
            # Use node degree as node size
            degree = self.analyzer.G.degree(node)
            node_size.append(10 + degree * 2)
            
            # Set color based on year if available
            if year != 'Unknown' and isinstance(year, (int, float)):
                if year < 2000:
                    node_color.append('blue')  # Older papers (before 2000)
                elif year < 2010:
                    node_color.append('green')  # Papers from 2000-2009
                elif year < 2020:
                    node_color.append('orange')  # Papers from 2010-2019
                else:
                    node_color.append('red')  # Recent papers (2020+)
            else:
                node_color.append('gray')  # Unknown year
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_text = []
        
        for u, v, data in self.analyzer.G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge type
            edge_type = data.get('edge_type', 'Unknown')
            
            edge_text.append(f"Type: {edge_type}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='lightgray'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                size=node_size,
                color=node_color
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Main Research Graph',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[dict(
                             text="",
                             showarrow=False,
                             xref="paper", yref="paper"
                         )],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        return fig
        
    def plot_author_graph(self) -> go.Figure:
        """Draw author collaboration graph
        
        Returns:
            plotly.graph_objects.Figure: Author graph
        """
        if not hasattr(self.analyzer, 'G') or self.analyzer.G.number_of_nodes() == 0:
            return self._create_empty_plot("Main graph data not available or empty")
        
        # Create temporary author graph
        author_graph = nx.Graph()
        author_papers = defaultdict(list)
        
        # Extract author information from node attributes of the main graph
        for node in self.analyzer.G.nodes():
            paper_id = node
            authors = self.analyzer.G.nodes[node].get('authors', [])
            
            if not authors:
                # Try to get authors from original data
                paper_idx = int(node) if node.isdigit() else -1
                if 0 <= paper_idx < len(self.analyzer.data):
                    authors = self.analyzer.data.iloc[paper_idx].get('authors', [])
            
            for author in authors:
                if pd.notna(author) and author:
                    author_papers[author].append(paper_id)
        
        # Add author nodes
        for author, papers in author_papers.items():
            author_graph.add_node(author, papers=papers)
        
        # Add collaboration edges
        for i, (author1, papers1) in enumerate(author_papers.items()):
            for author2, papers2 in list(author_papers.items())[i+1:]:
                common_papers = set(papers1) & set(papers2)
                if common_papers:
                    author_graph.add_edge(author1, author2, 
                                        weight=len(common_papers),
                                        common_papers=list(common_papers))
        
        # Remove isolated nodes
        isolated_nodes = [node for node in author_graph.nodes() 
                         if author_graph.degree(node) == 0]
        author_graph.remove_nodes_from(isolated_nodes)
        
        # If graph is empty, return empty graph
        if author_graph.number_of_nodes() == 0:
            return self._create_empty_plot("No author collaboration data available")
        
        # Get node positions
        pos = nx.spring_layout(author_graph, k=0.5, iterations=50)  # Increase node spacing
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_labels = []
        
        # Select top 100 most collaborative authors to show more authors
        top_authors = sorted(author_graph.nodes(), 
                            key=lambda x: author_graph.degree(x), 
                            reverse=True)[:100]
        top_author_graph = author_graph.subgraph(top_authors)
        
        for node in top_author_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            papers = author_graph.nodes[node].get('papers', [])
            paper_count = len(papers)
            
            # Show full author name
            node_text.append(f"Author: {node}<br>Papers: {paper_count}<br>Collaborations: {top_author_graph.degree(node)}")
            
            # Show short author name label
            name_parts = node.split()
            if len(name_parts) > 0:
                last_name = name_parts[-1]
                if len(name_parts) > 1:
                    initials = ''.join([n[0] for n in name_parts[:-1]])
                    short_name = f"{initials}. {last_name}"
                else:
                    short_name = last_name
                node_labels.append(short_name)
            else:
                node_labels.append(node)
            
            # Use paper count as node size
            node_size.append(10 + paper_count * 3)
            
            # Use degree (collaboration count) for color
            degree = top_author_graph.degree(node)
            if degree > 15:
                node_color.append('darkred')  # Highly collaborative authors
            elif degree > 10:
                node_color.append('red')  # Very collaborative authors
            elif degree > 5:
                node_color.append('orange')  # Moderately collaborative authors
            elif degree > 2:
                node_color.append('green')  # Somewhat collaborative authors
            else:
                node_color.append('blue')  # Less collaborative authors
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_width = []
        
        for u, v, data in top_author_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Adjust edge width based on collaboration count
            weight = data.get('weight', 1)
            edge_width.append(0.5 + weight * 0.5)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace with labels
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=8, color='black'),  # 小字体显示名称
            customdata=node_labels,  # 存储简短的作者名
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Collaboration Level',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=1, color='darkgray')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Top 100 Author Collaboration Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        return fig
        
    def plot_keyword_graph(self) -> go.Figure:
        """Draw keyword co-occurrence graph
        
        Returns:
            plotly.graph_objects.Figure: Keyword graph
        """
        if not hasattr(self.analyzer, 'keyword_graph') or not self.analyzer.keyword_graph:
            return self._create_empty_plot("No keyword graph data available")
            
        # Get node positions
        pos = nx.spring_layout(self.analyzer.keyword_graph)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.analyzer.keyword_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            papers = self.analyzer.keyword_graph.nodes[node].get('papers', [])
            paper_count = len(papers)
            
            node_text.append(f"Keyword: {node}<br>Papers: {paper_count}")
            
            # Use paper count as node size
            node_size.append(5 + paper_count)
            
            # Use degree (co-occurrence count) for color
            degree = self.analyzer.keyword_graph.degree(node)
            if degree > 20:
                node_color.append('darkred')  # Highly connected keywords
            elif degree > 10:
                node_color.append('red')  # Moderately connected keywords
            elif degree > 5:
                node_color.append('orange')  # Somewhat connected keywords
            else:
                node_color.append('blue')  # Less connected keywords
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_width = []
        
        for u, v, data in self.analyzer.keyword_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Use weight for edge width
            weight = data.get('weight', 1)
            if weight > 5:
                width = 3  # Strong co-occurrence
            elif weight > 2:
                width = 2  # Moderate co-occurrence
            else:
                width = 1  # Weak co-occurrence
                
            edge_width.append(width)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=10, color='black'),
            marker=dict(
                showscale=False,
                size=node_size,
                color=node_color,
                line=dict(width=1, color='darkgray')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Keyword Co-occurrence Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        return fig
    
    def plot_top_institution_collaborations(self) -> go.Figure:
        """Draw top institution collaborations
        
        Returns:
            plotly.graph_objects.Figure: Top institution collaborations
        """
        # Create institution collaboration network from original data
        institution_collab = defaultdict(int)
        
        for _, row in self.analyzer.data.iterrows():
            institutions = row.get('institution_display_names', [])
            
            # Process institution list - split by semicolon
            if isinstance(institutions, str):
                institutions = [inst.strip() for inst in institutions.split(';')]
            
            if len(institutions) > 1:
                # Count institution pairs
                for i in range(len(institutions)):
                    for j in range(i+1, len(institutions)):
                        if pd.notna(institutions[i]) and pd.notna(institutions[j]):
                            pair = tuple(sorted([institutions[i], institutions[j]]))
                            institution_collab[pair] += 1
        
        # Get top 15 most common collaboration relationships
        top_collaborations = sorted(
            institution_collab.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:15]
        
        if not top_collaborations:
            return self._create_empty_plot("No institution collaboration data available")
        
        # Prepare plotting data - use more concise institution name display
        institutions = []
        for pair in top_collaborations:
            inst1 = pair[0][0][:20] + "..." if len(pair[0][0]) > 20 else pair[0][0]
            inst2 = pair[0][1][:20] + "..." if len(pair[0][1]) > 20 else pair[0][1]
            institutions.append(f"{inst1} & {inst2}")
            
        counts = [pair[1] for pair in top_collaborations]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=counts,
                y=institutions,
                orientation='h'
            )
        ])
        
        fig.update_layout(
            title='Top 15 Institution Collaborations',
            xaxis_title='Collaboration Count',
            yaxis_title='Institution Pairs',
            template='plotly_white',
            height=800,  # Increase height to give text more space
            margin=dict(l=250, r=20, t=50, b=50),  # Increase left margin to accommodate institution names
            yaxis=dict(
                tickfont=dict(size=10)  # Reduce y-axis label font size
            )
        )
        
        return fig
    
    def plot_top_keyword_cooccurrences(self) -> go.Figure:
        """Draw top keyword co-occurrences
        
        Returns:
            plotly.graph_objects.Figure: Top keyword co-occurrences
        """
        # Create keyword co-occurrence network from original data
        keyword_cooccur = defaultdict(int)
        
        for _, row in self.analyzer.data.iterrows():
            keywords = row.get('keywords', [])
            
            # Process keyword list - split by comma
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(',')]
            
            if len(keywords) > 1:
                # Count keyword pairs
                for i in range(len(keywords)):
                    for j in range(i+1, len(keywords)):
                        if pd.notna(keywords[i]) and pd.notna(keywords[j]):
                            pair = tuple(sorted([keywords[i], keywords[j]]))
                            keyword_cooccur[pair] += 1
        
        # Get top 20 most common co-occurrence relationships
        top_cooccurrences = sorted(
            keyword_cooccur.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        if not top_cooccurrences:
            return self._create_empty_plot("No keyword co-occurrence data available")
        
        # Prepare plotting data
        keywords = [f"{pair[0][0]} + {pair[0][1]}" for pair in top_cooccurrences]
        counts = [pair[1] for pair in top_cooccurrences]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=counts,
                y=keywords,
                orientation='h'
            )
        ])
        
        fig.update_layout(
            title='Top 20 Keyword Co-occurrences',
            xaxis_title='Co-occurrence Count',
            yaxis_title='Keyword Pairs',
            template='plotly_white',
            height=700
        )
        
        return fig
    
    def plot_top_keywords(self) -> go.Figure:
        """Draw top keywords
        
        Returns:
            plotly.graph_objects.Figure: Top keywords
        """
        # Extract keywords from original data
        keyword_counts = defaultdict(int)
        
        for _, row in self.analyzer.data.iterrows():
            keywords = row.get('keywords', [])
            
            # Process keyword list - split by comma
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(',')]
            
            for keyword in keywords:
                if pd.notna(keyword) and keyword:
                    keyword_counts[keyword] += 1
        
        # Get top 30 most common keywords
        top_keywords = sorted(
            keyword_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:30]
        
        if not top_keywords:
            return self._create_empty_plot("No keyword data available")
        
        # Prepare plotting data
        keywords = [pair[0] for pair in top_keywords]
        counts = [pair[1] for pair in top_keywords]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=counts,
                y=keywords,
                orientation='h'
            )
        ])
        
        fig.update_layout(
            title='Top 30 Keywords',
            xaxis_title='Occurrence Count',
            yaxis_title='Keywords',
            template='plotly_white',
            height=800
        )
        
        return fig
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for different edge types
        
        Args:
            edge_type: Type of edge
            
        Returns:
            str: Color for the edge
        """
        color_map = {
            'author': 'blue',
            'citation': 'red',
            'institution': 'green',
            'keyword': 'magenta',
            'topic': 'purple',
            'author_shared': 'blue',
            'keyword_similarity': 'magenta',
            'institution_shared': 'green'
        }
        
        return color_map.get(edge_type, 'gray')
    
    def save_all_plots(self):
        """Save all plots to HTML files in the output directory"""
        # Create output directory
        output_dir = "output/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Main graph
        fig_main = self.plot_main_graph()
        fig_main.write_html(f"{output_dir}/main_graph.html")
        
        # Institution graphs
        fig_institution = self.plot_institution_collaboration()
        fig_institution.write_html(f"{output_dir}/institution_collaboration.html")
        
        # Top institution collaborations
        fig_top_institution = self.plot_top_institution_collaborations()
        fig_top_institution.write_html(f"{output_dir}/top_institution_collaborations.html")
        
        fig_degree = self.plot_degree_distribution()
        fig_degree.write_html(f"{output_dir}/centrality_distribution.html")
        
        fig_community = self.plot_community_distribution()
        fig_community.write_html(f"{output_dir}/community_distribution.html")
        
        fig_heatmap = self.plot_community_heatmap()
        fig_heatmap.write_html(f"{output_dir}/community_heatmap.html")
        
        # Author graph
        fig_author = self.plot_author_graph()
        fig_author.write_html(f"{output_dir}/author_graph.html")
        
        # Keyword graphs
        fig_keyword = self.plot_keyword_graph()
        fig_keyword.write_html(f"{output_dir}/keyword_graph.html")
        
        # Top keyword co-occurrences
        fig_top_keyword_cooccur = self.plot_top_keyword_cooccurrences()
        fig_top_keyword_cooccur.write_html(f"{output_dir}/top_keyword_cooccurrences.html")
        
        # Top keywords
        fig_top_keywords = self.plot_top_keywords()
        fig_top_keywords.write_html(f"{output_dir}/top_keywords.html")
        
        # Temporal analysis
        fig_year = self.plot_year_activity()
        fig_year.write_html(f"{output_dir}/year_activity.html")
        
        fig_topic = self.plot_topic_evolution()
        fig_topic.write_html(f"{output_dir}/topic_evolution.html")
        
        # Year subplots analysis
        fig_year_subplots = self.plot_year_subplots()
        fig_year_subplots.write_html(f"{output_dir}/year_subplots.html")
        
        # Research output graph
        fig_output_graph = self.plot_research_output_graph()
        fig_output_graph.write_html(f"{output_dir}/research_output_graph.html")
        
        logger.info(f"All plots saved to {output_dir}")
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message
        
        Args:
            message: Message to display
            
        Returns:
            plotly.graph_objects.Figure: Empty plot with message
        """
        fig = go.Figure()
        
        fig.update_layout(
            title=message,
            annotations=[
                dict(
                    text=message,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ]
        )
        
        return fig
        
    def plot_year_subplots(self) -> go.Figure:
        """Create year-based research output subplots visualization
        
        Create a figure with multiple subplots showing research trends by year:
        1. Number of papers published per year
        2. Average citations per year
        3. Most active institutions by year
        4. Popular keywords trends by year
        
        Returns:
            plotly.graph_objects.Figure: Annual research output subplots
        """
        # Check if data is available
        if not hasattr(self.analyzer, 'data') or len(self.analyzer.data) == 0:
            return self._create_empty_plot("No data available for annual analysis")
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Papers Published per Year", 
                "Average Citations per Year", 
                "Most Active Institutions by Year",
                "Popular Keywords by Year"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Prepare year range
        years = []
        for _, row in self.analyzer.data.iterrows():
            year = row.get('year')
            if pd.notna(year) and isinstance(year, (int, float)):
                years.append(int(year))
        
        if not years:
            return self._create_empty_plot("No year data available for analysis")
        
        min_year, max_year = min(years), max(years)
        year_range = list(range(min_year, max_year + 1))
        
        # 1. Papers per year (bar chart)
        year_counts = Counter(years)
        paper_counts = [year_counts.get(year, 0) for year in year_range]
        
        fig.add_trace(
            go.Bar(
                x=year_range, 
                y=paper_counts,
                marker_color='royalblue'
            ),
            row=1, col=1
        )
        
        # 2. Average citations per year (line chart)
        year_citations = defaultdict(list)
        for _, row in self.analyzer.data.iterrows():
            year = row.get('year')
            citations = row.get('citation_count', 0)
            if pd.notna(year) and isinstance(year, (int, float)) and pd.notna(citations):
                year_citations[int(year)].append(float(citations))
        
        avg_citations = []
        for year in year_range:
            citations = year_citations.get(year, [])
            avg = sum(citations) / len(citations) if citations else 0
            avg_citations.append(avg)
        
        fig.add_trace(
            go.Scatter(
                x=year_range, 
                y=avg_citations,
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # 3. Most active institutions by year (heatmap)
        year_institutions = defaultdict(Counter)
        for _, row in self.analyzer.data.iterrows():
            year = row.get('year')
            institutions = row.get('institution_display_names', [])
            
            if pd.notna(year) and isinstance(year, (int, float)):
                if isinstance(institutions, str):
                    institutions = [inst.strip() for inst in institutions.split(';')]
                
                for inst in institutions:
                    if pd.notna(inst) and inst:
                        year_institutions[int(year)][inst] += 1
        
        # Get top 5 institutions per year
        top_institutions_by_year = {}
        all_top_institutions = set()
        for year in year_range:
            counter = year_institutions.get(year, Counter())
            top_5 = [inst for inst, _ in counter.most_common(5)]
            top_institutions_by_year[year] = top_5
            all_top_institutions.update(top_5)
        
        if all_top_institutions:
            # Show maximum 10 top institutions
            top_overall = list(all_top_institutions)[:10]
            
            # Create heatmap data
            institution_data = []
            for inst in top_overall:
                inst_data = []
                for year in year_range:
                    counter = year_institutions.get(year, Counter())
                    inst_data.append(counter.get(inst, 0))
                institution_data.append(inst_data)
            
            # Shorten institution names for display
            short_names = []
            for inst in top_overall:
                if len(inst) > 20:
                    short_name = inst[:17] + "..."
                else:
                    short_name = inst
                short_names.append(short_name)
            
            fig.add_trace(
                go.Heatmap(
                    z=institution_data,
                    x=year_range,
                    y=short_names,
                    colorscale='Viridis'
                ),
                row=2, col=1
            )
        
        # 4. Popular keywords by year (line chart)
        year_keywords = defaultdict(Counter)
        for _, row in self.analyzer.data.iterrows():
            year = row.get('year')
            keywords = row.get('keywords', [])
            
            if pd.notna(year) and isinstance(year, (int, float)):
                if isinstance(keywords, str):
                    keywords = [kw.strip() for kw in keywords.split(',')]
                
                for kw in keywords:
                    if pd.notna(kw) and kw:
                        year_keywords[int(year)][kw] += 1
        
        # Get top 5 keywords across all years
        all_keywords = Counter()
        for year_counter in year_keywords.values():
            all_keywords.update(year_counter)
        
        top_keywords = [kw for kw, _ in all_keywords.most_common(5)]
        
        if top_keywords:
            for kw in top_keywords:
                kw_data = []
                for year in year_range:
                    counter = year_keywords.get(year, Counter())
                    kw_data.append(counter.get(kw, 0))
                
                fig.add_trace(
                    go.Scatter(
                        x=year_range,
                        y=kw_data,
                        mode='lines+markers',
                        name=kw
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Annual Research Output Analysis",
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update X-axis range for consistency
        fig.update_xaxes(title_text="Year", range=[min_year-0.5, max_year+0.5])
        
        # Update Y-axis titles
        fig.update_yaxes(title_text="Number of Papers", row=1, col=1)
        fig.update_yaxes(title_text="Average Citations", row=1, col=2)
        fig.update_yaxes(title_text="Institution", row=2, col=1)
        fig.update_yaxes(title_text="Keyword Occurrences", row=2, col=2)
        
        return fig
        
    def build_research_output_graph(self) -> nx.Graph:
        """Build research output graph with papers as nodes and shared attributes as edges
        
        Nodes: Research papers
        Edges: Shared authors, keywords, institutions, etc.
        
        Returns:
            nx.Graph: Research output graph
        """
        if not hasattr(self.analyzer, 'data') or len(self.analyzer.data) == 0:
            logger.warning("No data available to build research output graph")
            return nx.Graph()
        
        # Create new graph
        output_graph = nx.Graph()
        
        # Create nodes for each paper
        for i, row in self.analyzer.data.iterrows():
            paper_id = str(i)
            title = row.get('title', f"Paper {i}")
            year = row.get('year', None)
            doi = row.get('doi', None)
            
            # Add node
            output_graph.add_node(
                paper_id,
                title=title,
                year=year,
                doi=doi,
                type='paper'
            )
        
        # Create edges - based on shared authors
        author_papers = defaultdict(list)
        for i, row in self.analyzer.data.iterrows():
            paper_id = str(i)
            authors = row.get('authors', [])
            
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(',')]
            
            for author in authors:
                if pd.notna(author) and author:
                    author_papers[author].append(paper_id)
        
        # Add shared author edges
        for author, papers in author_papers.items():
            if len(papers) > 1:  # Only when author has written at least 2 papers
                for i in range(len(papers)):
                    for j in range(i+1, len(papers)):
                        # If edge exists, increase weight
                        if output_graph.has_edge(papers[i], papers[j]):
                            output_graph[papers[i]][papers[j]]['weight'] += 1
                            output_graph[papers[i]][papers[j]]['authors'].append(author)
                        else:
                            output_graph.add_edge(
                                papers[i], papers[j],
                                edge_type='author_shared',
                                weight=1,
                                authors=[author]
                            )
        
        # Create edges - based on shared keywords
        keyword_papers = defaultdict(list)
        for i, row in self.analyzer.data.iterrows():
            paper_id = str(i)
            keywords = row.get('keywords', [])
            
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(',')]
            
            for keyword in keywords:
                if pd.notna(keyword) and keyword:
                    keyword_papers[keyword].append(paper_id)
        
        # Add shared keyword edges
        for keyword, papers in keyword_papers.items():
            if len(papers) > 1:  # Only when keyword appears in at least 2 papers
                for i in range(len(papers)):
                    for j in range(i+1, len(papers)):
                        # If edge exists, increase weight
                        if output_graph.has_edge(papers[i], papers[j]):
                            output_graph[papers[i]][papers[j]]['weight'] += 0.5
                            if 'keywords' in output_graph[papers[i]][papers[j]]:
                                output_graph[papers[i]][papers[j]]['keywords'].append(keyword)
                            else:
                                output_graph[papers[i]][papers[j]]['keywords'] = [keyword]
                        else:
                            output_graph.add_edge(
                                papers[i], papers[j],
                                edge_type='keyword_similarity',
                                weight=0.5,
                                keywords=[keyword]
                            )
        
        logger.info(f"Research output graph built: {output_graph.number_of_nodes()} nodes, {output_graph.number_of_edges()} edges")
        return output_graph
    
    def plot_research_output_graph(self) -> go.Figure:
        """Draw research output graph showing relationships between papers
        
        Returns:
            plotly.graph_objects.Figure: Research output graph
        """
        # Build research output graph
        output_graph = self.build_research_output_graph()
        
        if output_graph.number_of_nodes() == 0:
            return self._create_empty_plot("No data available to build research output graph")
        
        # If graph is too large, limit to most important nodes
        if output_graph.number_of_nodes() > 100:
            # Select most important nodes based on degree and edge weights
            node_importance = {}
            for node in output_graph.nodes():
                node_importance[node] = sum(data.get('weight', 1) for _, _, data in output_graph.edges(node, data=True))
            
            top_nodes = sorted(node_importance.keys(), key=lambda x: node_importance[x], reverse=True)[:100]
            output_graph = output_graph.subgraph(top_nodes)
        
        # Get node positions
        pos = nx.spring_layout(output_graph, k=0.6, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in output_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            title = output_graph.nodes[node].get('title', f"Paper {node}")
            year = output_graph.nodes[node].get('year', 'Unknown')
            doi = output_graph.nodes[node].get('doi', 'Unknown')
            
            node_text.append(f"Title: {title}<br>Year: {year}<br>DOI: {doi}")
            
            # Node size based on degree
            degree = output_graph.degree(node, weight='weight')
            node_size.append(10 + degree * 2)
            
            # Node color based on year
            if year != 'Unknown' and isinstance(year, (int, float)):
                if year < 2000:
                    node_color.append('blue')  # Before 2000
                elif year < 2010:
                    node_color.append('green')  # 2000-2009
                elif year < 2020:
                    node_color.append('orange')  # 2010-2019
                else:
                    node_color.append('red')  # After 2020
            else:
                node_color.append('gray')  # Unknown year
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_width = []
        edge_text = []
        
        for u, v, data in output_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_type = data.get('edge_type', 'Unknown')
            weight = data.get('weight', 1)
            
            if edge_type == 'author_shared':
                authors = data.get('authors', [])
                text = f"Shared Authors: {', '.join(authors[:3])}"
                if len(authors) > 3:
                    text += f" and {len(authors)} others"
                edge_text.append(text)
            elif edge_type == 'keyword_similarity':
                keywords = data.get('keywords', [])
                text = f"Shared Keywords: {', '.join(keywords[:3])}"
                if len(keywords) > 3:
                    text += f" and {len(keywords)} others"
                edge_text.append(text)
            else:
                edge_text.append(f"Relationship Type: {edge_type}")
            
            edge_width.append(0.5 + weight)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='lightgray'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Year Distribution',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=1, color='darkgray')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Research Output Relationship Graph',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        return fig


def main():
    """Main function for visualization"""
    import argparse
    from graph_analysis import ResearchGraphBuilder
    
    parser = argparse.ArgumentParser(description="Visualize research graphs")
    parser.add_argument("--data", required=True, help="Path to data CSV file")
    parser.add_argument("--output", default="output/visualizations", 
                      help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Build graph
    builder = ResearchGraphBuilder(args.data)
    builder.analyze()
    
    # Create visualizer
    visualizer = ResearchGraphVisualizer(builder)
    
    # Set output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save all plots
    visualizer.save_all_plots()
    

if __name__ == "__main__":
    main() 