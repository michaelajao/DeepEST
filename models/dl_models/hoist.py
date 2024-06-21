
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.nn.utils import weight_norm

    
class HOIST(nn.Module):
    def __init__(self, dynamic_dims, input_window, output_window, static_dims = None, distance_dims = None, rnn_dim=64, signs=None, device='cpu'):
        """
        HOIST (Harnessing spatio-temporal dynamics and interactions for forecasting) Model.

        This model is designed for forecasting tasks that involve spatio-temporal data.
        It integrates dynamic features, static features, and distance features to predict future outcomes.

        Parameters:
        ----------
        dynamic_dims : list of int
            List of integers representing the number of features in each dynamic feature category.
        input_window : int
            The size of the input time window.
        output_window : int
            The size of the output time window.
        static_dims : list of int, optional
            List of integers representing the number of features in each static feature category.
        distance_dims : int, optional
            Integer representing the number of distance types.
        rnn_dim : int, optional
            Number of hidden units in the RNN layer (default is 64).
        signs : list of int, optional
            List of 1 or -1 indicating the field direction of each dynamic feature category.
        device : str, optional
            The device to run the model on (default is 'cpu').

        Inputs:
        ------
        dynamic : list of torch.FloatTensor
            List of dynamic feature tensors with shape (N, T, D_k).
        static : list of torch.FloatTensor, optional
            List of static feature tensors with shape (N, D_k).
        distance : torch.FloatTensor, optional
            Distance feature tensor with shape (N, N, D_k).
        h0 : torch.FloatTensor, optional
            Initial hidden state of the RNN layer.

        Attributes:
        ----------
        dynamic_weights : torch.nn.ModuleList
            List of modules for calculating weights of dynamic features.
        rnn : torch.nn.LSTM
            The LSTM layer for processing time series data.
        linear : torch.nn.Linear
            Linear layer for transforming RNN output.
        linear_1 : torch.nn.Linear
            Linear layer for predicting intermediate outcomes.
        linear_2 : torch.nn.Linear
            Linear layer for predicting final outcomes based on output window.

        Note:
        ----
        If both static and distance features are None, spatial relationships won't be used.
        """
        
        super(HOIST, self).__init__()
        self.dynamic_dims = dynamic_dims
        self.dynamic_feats = len(dynamic_dims)
        self.static_dims = static_dims
        self.distance_dims = distance_dims
        self.device = device
        self.rnn_dim = rnn_dim
        self.signs = signs
        self.output_window = output_window
        self.input_window = input_window
        if self.signs != None:
            try:
                assert len(self.signs) == self.dynamic_feats
                assert all([s == 1 or s == -1 for s in self.signs])
            except:
                raise ValueError('The signs should be a list of 1 or -1 with the same length as dynamic_dims.')
        
        self.dynamic_weights = nn.ModuleList([nn.Sequential(nn.Linear(self.dynamic_dims[i], rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, self.dynamic_dims[i]), nn.Sigmoid()) for i in range(self.dynamic_feats)])
        
        self.total_feats = np.sum(self.dynamic_dims)       
        self.rnn = nn.LSTM(self.total_feats, rnn_dim, batch_first=True)
        
        self.linear = nn.Linear(rnn_dim, rnn_dim)
        self.linear_1 = nn.Linear(rnn_dim, 1)
        self.linear_2 = nn.Linear(input_window, output_window)
        
        self.static_dims = static_dims
        if self.static_dims != None:
            self.static_feats = len(static_dims)
    
            self.w_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.static_dims[i], self.static_dims[i]).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device) for i in range(self.static_feats)])
            self.a_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*self.static_dims[i], 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device) for i in range(self.static_feats)])

        if self.distance_dims != None:
            self.W_dis = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(distance_dims, distance_dims).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
            self.a_dis = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(distance_dims, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
    
    def forward(self, x, distance = None, h0 = None):
        b, n, t, f = x.shape
        if self.static_dims == None:
            static = None
            dynamic_temp = x
            dynamic = [dynamic_temp.clone().view(-1, dynamic_temp.shape[2], dynamic_temp.shape[3])]
        else:
            static_temp = x[:, :, 0,  0: self.static_dims[0]]
            static = [static_temp.clone().view(-1, static_temp.shape[2])]
            dynamic_temp = x[:, :, :, self.static_dims[0]: ]
            dynamic = [dynamic_temp.clone().view(-1, dynamic_temp.shape[2], dynamic_temp.shape[3])]
            # print(f'static_temp.shape is {static_temp.shape}, dynamic_temp.shape is {dynamic_temp.shape}')
        try:
            assert len(dynamic) == self.dynamic_feats
        except:
            print('The number of dynamic features is not correct.')
            return None
        if self.static_dims != None:
            try:
                assert len(static) == self.static_feats
            except:
                print('The number of static features is not correct.')
                return None
        if self.distance_dims != None:
            try:
                assert distance.shape[2] == self.distance_dims
            except:
                print('The number of distance features is not correct.')
                return None
        
        static_dis = []
        N = dynamic[0].shape[0]
        T = dynamic[0].shape[1]
        if self.static_dims != None:
            for i in range(self.static_feats):
                h_i = torch.mm(static[i], self.w_list[i])
                h_i = torch.cat([h_i.unsqueeze(1).repeat(1, N, 1), h_i.unsqueeze(0).repeat(N, 1, 1)], dim=2)
                d_i = torch.sigmoid(h_i @ self.a_list[i]).reshape(N, N)
                static_dis.append(d_i)

        if self.distance_dims != None:
            h_i = distance @ self.W_dis
            h_i = torch.sigmoid(h_i @ self.a_dis).reshape(N, N)
            static_dis.append(h_i)
            
        if self.static_dims != None or self.distance_dims != None:
            static_dis = torch.stack(static_dis, dim=0)
            static_dis = static_dis.sum(0)
            static_dis = torch.softmax(static_dis, dim=-1)
        
        dynamic_weights = []
        for i in range(self.dynamic_feats):
            cur_weight = self.dynamic_weights[i](dynamic[i].reshape(N*T, -1)).reshape(N, T, -1)
            if self.signs != None:
                cur_weight = cur_weight * self.signs[i]
            dynamic_weights.append(cur_weight)
        dynamic_weights = torch.cat(dynamic_weights, dim=-1)

        if h0 is None:
            h0 = torch.randn(1, N, self.rnn_dim).to(self.device)
        dynamic = torch.cat(dynamic, dim=-1)
        h, (hn, cn) = self.rnn(dynamic_weights*dynamic)
        if self.static_dims != None or self.distance_dims != None:
            h_att = h.reshape(N,1,T,self.rnn_dim).repeat(1,N,1,1)
            h = h + (h_att * static_dis.reshape(N,N,1,1)).sum(1)
        y = self.linear(h)
        y  = self.linear_1(y).squeeze(-1)
        y = self.linear_2(F.leaky_relu(y)).reshape(b, n, -1)
        return y
        # return y, [static_dis, dynamic_weights, hn]