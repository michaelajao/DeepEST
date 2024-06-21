import plotly.graph_objects as go
import networkx as nx
import torch
import pandas as pd
import sys
# sys.path.append("../")
from models import *
from trainer import *
from plotly.offline import iplot



def visualize_graph_from_edges(edges_index, node_names=None):
    """
    Visualizes a graph from edge indices using Plotly.

    Parameters:
    ----------
    edges_index : torch.Tensor
        A 2D tensor containing the indices of the edges in the graph.
        The tensor should have shape [2, N] where N is the number of edges.
    node_names : list of str, optional
        A list of names for the nodes in the graph. If None, default names will be generated.

    Returns:
    -------
    None
        Visualizes the graph using Plotly.
    """
    edge_indices = edges_index.to_sparse().coalesce().indices()
    G = nx.Graph()
    for i in range(edge_indices.size(1)):
        x = edge_indices[0, i].item()
        y = edge_indices[1, i].item()
        G.add_edge(x, y)

    nodes_positions = nx.random_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodes_positions[edge[0]]
        x1, y1 = nodes_positions[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node, pos in nodes_positions.items():
        x, y = pos
        node_x.append(x)
        node_y.append(y)
        if node_names is not None and node < len(node_names):
            node_text.append(node_names[node])
        else:
            node_text.append(f'Node {node}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text', 
        text=node_text,
        textposition='top center',  
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,  # 节点大小保持不变
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Graph Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        font=dict(size=16),  
                        legend=dict(font=dict(size=16)),  
                        xaxis_title_font=dict(size=16),
                        yaxis_title_font=dict(size=16)
                    ))
    fig.show()

if __name__ == "__main__":
    rows = torch.tensor([0, 1, 1, 2, 3])
    cols = torch.tensor([1, 0, 2, 3, 1])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) 
    edges_index = torch.sparse_coo_tensor(indices=torch.stack((rows, cols)), values=values, size=(4, 4))
    nodes_count = edges_index.shape[0]
    node_names = [f'Node {i}' for i in range(nodes_count)] 
    visualize_graph_from_edges(edges_index, node_names)
