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
from collections import defaultdict
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
        self.main_graph = graph_analyzer.doi_graph
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
            line=dict(width=0.5, color='gray'),
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
                showscale=False,
                size=node_size,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Collaboration Count',
                    xanchor='left',
                    titleside='right'
                )
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
        if not self.main_graph:
            return self._create_empty_plot("No main graph data available")
            
        # Get node positions
        pos = nx.spring_layout(self.main_graph)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.main_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            attrs = self.main_graph.nodes[node]
            title = attrs.get('title', 'Unknown')
            year = attrs.get('year', 'Unknown')
            doi = attrs.get('doi', 'Unknown')
            
            node_text.append(f"Title: {title}<br>Year: {year}<br>DOI: {doi}")
            
            # Use node degree as node size
            degree = self.main_graph.degree(node)
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
        edge_color = []
        
        for u, v, data in self.main_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge type
            edge_type = data.get('type', 'Unknown')
            edge_color.append(self._get_edge_color(edge_type))
            
            edge_text.append(f"Type: {edge_type}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=edge_color),
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
        if not hasattr(self.analyzer, 'author_graph') or not self.analyzer.author_graph:
            return self._create_empty_plot("No author graph data available")
            
        # Get node positions
        pos = nx.spring_layout(self.analyzer.author_graph)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.analyzer.author_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            papers = self.analyzer.author_graph.nodes[node].get('papers', [])
            paper_count = len(papers)
            
            node_text.append(f"Author: {node}<br>Papers: {paper_count}")
            
            # Use paper count as node size
            node_size.append(5 + paper_count * 2)
            
            # Use degree (collaboration count) for color
            degree = self.analyzer.author_graph.degree(node)
            if degree > 10:
                node_color.append('darkred')  # Highly collaborative authors
            elif degree > 5:
                node_color.append('red')  # Moderately collaborative authors
            elif degree > 2:
                node_color.append('orange')  # Somewhat collaborative authors
            else:
                node_color.append('blue')  # Less collaborative authors
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_width = []
        
        for u, v, data in self.analyzer.author_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Use weight for edge width
            weight = data.get('weight', 1)
            if weight > 3:
                width = 4  # Strong collaboration
            elif weight > 1:
                width = 2  # Multiple collaborations
            else:
                width = 1  # Single collaboration
                
            edge_width.append(width)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
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
                showscale=False,
                size=node_size,
                color=node_color,
                line=dict(width=1, color='darkgray')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Author Collaboration Network',
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
        
    def plot_citation_graph(self) -> go.Figure:
        """Draw citation network graph
        
        Returns:
            plotly.graph_objects.Figure: Citation graph
        """
        if not hasattr(self.analyzer, 'citation_graph') or not self.analyzer.citation_graph:
            return self._create_empty_plot("No citation graph data available")
            
        # Get node positions - use hierarchical layout for citations
        pos = nx.spring_layout(self.analyzer.citation_graph)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.analyzer.citation_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            attrs = self.analyzer.citation_graph.nodes[node]
            title = attrs.get('title', 'Unknown')
            year = attrs.get('year', 'Unknown')
            
            node_text.append(f"Title: {title}<br>Year: {year}")
            
            # Use citation count for node size
            in_degree = self.analyzer.citation_graph.in_degree(node)
            out_degree = self.analyzer.citation_graph.out_degree(node)
            node_size.append(5 + in_degree * 2)
            
            # Color by year if available
            if year != 'Unknown' and isinstance(year, (int, float)):
                if year < 2000:
                    node_color.append('blue')  # Older papers
                elif year < 2010:
                    node_color.append('green')  # Medium-aged papers
                elif year < 2020:
                    node_color.append('orange')  # Newer papers
                else:
                    node_color.append('red')  # Recent papers
            else:
                node_color.append('gray')  # Unknown year
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_text = []
        
        for u, v in self.analyzer.citation_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge text - explains citation direction
            u_title = self.analyzer.citation_graph.nodes[u].get('title', 'Unknown')
            v_title = self.analyzer.citation_graph.nodes[v].get('title', 'Unknown')
            edge_text.append(f"{u_title} cites {v_title}")
        
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
                color=node_color,
                line=dict(width=1, color='darkgray')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Citation Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
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
            'topic': 'purple'
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
        
        fig_degree = self.plot_degree_distribution()
        fig_degree.write_html(f"{output_dir}/centrality_distribution.html")
        
        fig_community = self.plot_community_distribution()
        fig_community.write_html(f"{output_dir}/community_distribution.html")
        
        fig_heatmap = self.plot_community_heatmap()
        fig_heatmap.write_html(f"{output_dir}/community_heatmap.html")
        
        # Author graph
        fig_author = self.plot_author_graph()
        fig_author.write_html(f"{output_dir}/author_graph.html")
        
        # Keyword graph
        fig_keyword = self.plot_keyword_graph()
        fig_keyword.write_html(f"{output_dir}/keyword_graph.html")
        
        # Citation graph
        fig_citation = self.plot_citation_graph()
        fig_citation.write_html(f"{output_dir}/citation_graph.html")
        
        # Temporal analysis
        fig_year = self.plot_year_activity()
        fig_year.write_html(f"{output_dir}/year_activity.html")
        
        fig_topic = self.plot_topic_evolution()
        fig_topic.write_html(f"{output_dir}/topic_evolution.html")
        
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