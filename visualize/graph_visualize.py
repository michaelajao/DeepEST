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
    edge_indices = edges_index.to_sparse().coalesce().indices()
    # 创建 NetworkX 图
    G = nx.Graph()
    # 添加边到图中
    for i in range(edge_indices.size(1)):
        x = edge_indices[0, i].item()
        y = edge_indices[1, i].item()
        G.add_edge(x, y)

    # 估计节点位置（随机布局）
    nodes_positions = nx.random_layout(G)

    # 绘制边
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

    # 绘制节点
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
        mode='markers+text',  # 添加 text 模式
        text=node_text,
        textposition='top center',  # 文本显示在节点上方
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
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.show()

if __name__ == "__main__":
    # claim_data = torch.Tensor(pd.read_pickle('../data/claim_tensor.pkl'))
    # county_data = torch.Tensor(pd.read_pickle('../data/county_tensor.pkl'))
    # hospitalizations_data = torch.Tensor(pd.read_pickle('../data/hospitalizations.pkl'))
    # distance_matrix = torch.Tensor(pd.read_pickle('../data/distance_mat.pkl'))
    # data_time = pd.read_pickle('../data/date_range.pkl') # 这个是list
    # dynamic_data = torch.cat((claim_data, torch.unsqueeze(hospitalizations_data, -1)), -1)
    # static_data = county_data
    # label = torch.unsqueeze(hospitalizations_data, -1)
    # dynamic_data = dynamic_data[:50]
    # static_data = static_data[:50]
    # label = label[:50]
    # threshold = 5000
    # nodes_count = 50
    # edges_index = construct_adjacency_matrix(static_data, threshold)
    rows = torch.tensor([0, 1, 1, 2, 3])
    cols = torch.tensor([1, 0, 2, 3, 1])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 假设对应的值
    # 构建稀疏张量
    edges_index = torch.sparse_coo_tensor(indices=torch.stack((rows, cols)), values=values, size=(4, 4))
    nodes_count = edges_index.shape[0]
    node_names = [f'Node {i}' for i in range(nodes_count)]  # 示例节点名称
    visualize_graph_from_edges(edges_index, node_names)
