import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
import math
import torch.nn.functional as F
import scipy.sparse as sp
from models.dl_models.dl_base import spatial_temporal_model
from torch.nn import Parameter


def normalize_adj2(adj):
    """
    Symmetrically normalize the adjacency matrix of a graph.

    Parameters:
    ----------
    adj : scipy.sparse matrix
        The adjacency matrix of the graph.

    Returns:
    -------
    scipy.sparse matrix
        The symmetrically normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a PyTorch sparse tensor.

    Parameters:
    ----------
    sparse_mx : scipy.sparse matrix
        A sparse matrix.

    Returns:
    -------
    torch.sparse.FloatTensor
        A PyTorch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvLayer(nn.Module):
    """
    A graph convolutional layer module.

    Parameters:
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        If set to True, the layer will have a bias term, by default True.

    Attributes:
    ----------
    weight : torch.Tensor
        The weight tensor for the linear transformation.
    bias : torch.Tensor, optional
        The bias tensor, if bias is True.

    Methods:
    -------
    forward(feature, adj)
        Forward pass of the graph convolutional layer.
    __repr__
        String representation of the layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class cola_gnn(spatial_temporal_model):
    """
    A Graph Neural Network model for spatio-temporal data, named 'cola_gnn'.

    Parameters:
    ----------
    input_size : int
        Input feature size.
    num_positions : int
        Number of positions or nodes in the graph.
    input_window : int
        Size of the input time window.
    output_window : int
        Size of the output time window.
    origin_adj : torch.Tensor
        The original adjacency matrix of the graph.
    hidden_size : int, optional
        Size of the hidden layers, by default 64.
    kernels : int, optional
        Number of kernels in the convolutional layers, by default 10.
    n_layer : int, optional
        Number of layers in the RNN, by default 1.
    dropout : float, optional
        Dropout rate, by default 0.5.
    
    Methods:
    -------
    init_weights()
        Initialize the weights of the model.
    forward(x, feat=None)
        Forward pass of the 'cola_gnn' model.
    """
    def __init__(
        self,
        input_size: int,
        num_positions: int,
        input_window: int,
        output_window: int,
        origin_adj: int,
        hidden_size: int = 64,
        kernels: int = 10,
        n_layer: int = 1,
        dropout: int = 0.5
    ):
        super().__init__()
        self.x_h = input_size
        self.f_h = num_positions
        self.m = num_positions
        self.w = input_window
        self.h = output_window
        # self.adj = adj
        self.o_adj = origin_adj
        self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(origin_adj.cpu().numpy())).to_dense()
        self.dropout = dropout
        self.n_hidden = hidden_size
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu
        self.Wb = Parameter(torch.Tensor(self.m, self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = kernels
        self.conv = nn.Conv1d(self.x_h, self.k, self.w)
        long_kernal = self.w//2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal, dilation=2)
        long_out = self.w-2*(long_kernal-1)
        self.long_out = long_out
        self.n_spatial = 10
        self.conv1 = GraphConvLayer((1+long_out)*self.k, self.n_hidden)  # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
        self.rnn = nn.RNN(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=n_layer, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size + self.n_spatial, output_window)

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, input_window)
            self.residual = nn.Linear(self.residual_window, 1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)  # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):
        '''
        Args:  x: (batch, time_step, m)
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        '''
        b, m, w, f = x.size()
        orig_x = x
        # print('-' * 100)
        # print('x_origin.shape is ', orig_x.shape)
        x = x.contiguous().view(-1, x.size(2), 1, f)
        x = torch.squeeze(x)
        # print('x_2.shape is ', x.shape)
        r_out, hc = self.rnn(x, None)
        # print('r_out.shape is ', r_out.shape)
        last_hid = r_out[:, -1, :]
        # print('last_hid_1.shape is ', last_hid.shape)
        last_hid = last_hid.view(-1, self.m, self.n_hidden)
        # print('last_hid_2.shape is', last_hid.shape)
        # print('-' * 100)
        out_temporal = last_hid  # [b, m, 64]
        hid_rpt_m = last_hid.repeat(1, self.m, 1).view(b, self.m, self.m, self.n_hidden)  # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1, 1, self.m).view(b, self.m, self.m, self.n_hidden)  # b,m,m,w continuous w one window data
        # print('hid_rpt_m.shape is ', hid_rpt_m.shape, 'hid_rpt_w.shape is ', hid_rpt_w.shape)
        # print('self.W1.T().shape is ', self.W1.t().shape, 'self.W2.T().shape is ', self.W2.t().shape, 'self.V.shape() is ', self.V.shape,'self.bv.shape is ', self.bv.shape)
        a_mx = self.act(hid_rpt_m @ self.W1.t() + hid_rpt_w @ self.W2.t() + self.b1) @ self.V + self.bv  # row, all states influence one state
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        # print('a_mx.shape is ', a_mx.shape)
        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            # print('h_tmp1.shape is ',h_tmp.shape)
            h_tmp = h_mids[:, i:i+1, :, :].permute(0, 1, 3, 2).contiguous()
            # print(h_tmp.shape) # [b, 1, feature, input_window]
            h_tmp = torch.squeeze(h_tmp, 1)
            # print('h_tmp2.shape is ',h_tmp.shape)
            r = self.conv(h_tmp)  # [32, 10/k, 1]
            # print('r.shape is ', r.shape)
            r_long = self.conv_long(h_tmp)
            # print('r_long.shape is ', r_long.shape)/
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l, dim=1)
        # print('r_l.shape is ',r_l.shape)
        r_long_l = torch.stack(r_long_l, dim=1)
        # print('r_long_l.shape is ',r_long_l.shape)
        r_l = torch.cat((r_l, r_long_l), -1)
        r_l = r_l.view(r_l.size(0), r_l.size(1), -1)
        r_l = torch.relu(r_l)
        # print('r_l2.shape is ', r_l.shape)
        adjs = self.adj.repeat(b, 1)
        # print('adjs1.shape is ', adjs.shape)
        adjs = adjs.view(b, self.m, self.m)
        # print('adjs2.shape is ', adjs.shape)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        # print('c.shape is ', c.shape)
        a_mx = adjs * c + a_mx * (1-c)
        adj = a_mx
        x = r_l
        # print('x.shape is ', x.shape, 'adj.shape is ', adj.shape)
        # print('long_out is ', self.long_out, 'self.k is ', self.k, 'self.n_hidden is ', self.n_hidden)
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        out = torch.cat((out_spatial, out_temporal), dim=-1)
        # print('out_spitial.shape is ', out_spatial.shape, 'out_temporal.shape is ', out_temporal.shape, 'out.shape is ', out.shape)
        out = self.out(out)
        out = torch.squeeze(out)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]  # Step backward # [batch, res_window, m]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)  # [batch*m, res_window]
            z = self.residual(z)  # [batch * m, 1]
            z = z.view(-1, self.m)  # [batch, m]
            out = out * self.ratio + z  # [batch, m]

        return out
