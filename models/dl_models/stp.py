import torch.nn as nn
import torch
from torch_geometric.nn import GATConv
from models.dl_models.dl_base import spatial_temporal_model


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads)

    def forward(self, x, edge_index):
        # x: [B, N, in_channels] 节点特征张量
        # edge_index: [2, E] 边索引矩阵
        # 输出: [B, N, heads * out_channels] 节点特征张量
        B, N, _ = x.size()  # 获取batch_size和节点数
        x = x.reshape(B * N, -1)  # 将x展平成[B * N, in_channels]
        x = self.gat_conv(x, edge_index)  # [B * N, heads * out_channels]
        x = x.reshape(B, N, -1)  # 将x恢复成[B, N, heads * out_channels]
        return x


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: [B, N, T, input_size] 输入特征张量
        # 输出: [B, N, T, hidden_size] 隐藏状态张量
        B, N, T, _ = x.size()  # 获取batch_size、时间步数和节点数
        x = x.reshape(B * N, T, -1)  # 将x展平成[B * N ,T ,input_size]
        output, _ = self.gru(x)  # 通过GRU层得到[1 ,B * N ,hidden_size]
        # h = h.reshape(1 ,B ,N ,-1) # 将h恢复成[1 ,B ,N ,hidden_size]
        # h = h.permute(1 ,0 ,2 ,3) # 将h变换成[B ,1 ,N ,hidden_size]
        output = output.reshape(B, N, T, -1)  # [B, N, T, H]
        return output


class STPModel(spatial_temporal_model):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        edge_index: list,
        gat_heads: int = 1,
    ):
        super(STPModel, self).__init__()
        self.input_size = input_size  # 节点特征维度
        self.gat_hidden_size = hidden_size  # GAT隐藏层维度
        self.gat_heads = gat_heads  # GAT注意力头数
        self.gru_hidden_size = hidden_size  # GRU隐藏层维度
        self.output_size = output_size  # 输出维度
        self.edge_index = edge_index
        self.gat_layer = GATLayer(input_size, self.gat_hidden_size, gat_heads)
        self.gru_layer = GRULayer(gat_heads * self.gat_hidden_size, self.gru_hidden_size)
        self.linear_layer = nn.Linear(self.gru_hidden_size, self.output_size)

    def forward(self, node_features):
        B, N, T, F = node_features.size()  # 获取 batch_size、节点数、时间步数和特征维度
        self.edge_index = self.edge_index.repeat(1, B)
        gat_out = torch.zeros(B, N, T, self.gat_heads * self.gat_hidden_size).to(node_features.device)
        for t in range(T):  # 遍历每个时间步
            x = node_features[:, :, t, :]  # 取出当前时间步的节点特征张量
            x = self.gat_layer(x, self.edge_index)  # 通过 GAT 层得到空间特征向量
            gat_out[:, :, t, :] = x

        gru_output = self.gru_layer(gat_out)  # 通过 GRU 层得到时间特征向量
        gru_output = gru_output[:, :, -1, :]  # 取最后一个时间步的隐藏状态
        outputs = self.linear_layer(gru_output)
        # print("11")
        return outputs
