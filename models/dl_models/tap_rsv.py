import torch
import torch.nn as nn
from DeepEST.models.dl_models.dl_base import spatial_temporal_model


class TAPRSV(spatial_temporal_model):
    """
    TAPRSV (Temporal and Attribute-aware Pointer and Spatial-Variant) Model.

    This model integrates static and dynamic features using a combination of
    linear layers, GRU (Gated Recurrent Unit), and attention mechanisms to
    predict outcomes based on temporal and attribute-aware inputs.

    Parameters:
    ----------
    num_positions : int
        Number of positions or regions for which predictions are made.
    static_dims : int
        Number of features for the static inputs.
    dynamic_dims : int
        Number of features for the dynamic inputs.
    output_window : int
        Size of the output prediction window.
    hidden : int, optional
        Number of hidden units (default is 64).
    disease_emb_num : int, optional
        Dimension of the disease embedding space. If None, disease similarity attention is not used.

    Attributes:
    ----------
    static : torch.nn.Linear
        Linear layer for static feature transformation.
    Kw, Qw : torch.nn.Linear
        Linear layers for disease similarity attention.
    rnn, rnn_disease : torch.nn.GRU
        GRU layers for dynamic feature processing with and without disease attention.
    fc : torch.nn.Sequential
        Fully connected layers for final prediction.
    """
    def __init__(self, num_positions, static_dims, dynamic_dims, output_window, hidden=64, disease_emb_num= None):
        super(TAPRSV, self).__init__()
        # static features
        self.static_dims = static_dims
        self.dynamic_dims = dynamic_dims
        self.static = nn.Linear(static_dims, hidden)
        
        if disease_emb_num is not None:
            # for disease similarity attention
            self.Kw = nn.Linear(disease_emb_num, hidden)
            self.Qw = nn.Linear(disease_emb_num, hidden)
        
        self.rnn = nn.GRU(dynamic_dims, hidden, 2, batch_first=True)
        self.rnn_disease = nn.GRU(2*dynamic_dims, hidden, 2, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_window),
        )
        self.init()
        
    @staticmethod
    def get_last_visit(hidden_states, mask):
        last_visit = torch.sum(mask,1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state

    def forward(self, x, disease_emb= None, mask= None):
        b, n, t, f = x.shape
        """ get static features """
        static_temp = x[:, :, 0,  0: self.static_dims]
        static_features = static_temp.view(-1, static_temp.shape[2])
        dynamic_temp = x[:, :, :, self.static_dims: ]
        dynamic_features = dynamic_temp.clone().view(-1, dynamic_temp.shape[2], dynamic_temp.shape[3])
        
        static_emb = self.static(static_features)

        if disease_emb is not None:
            """ dynamic features """
            K = self.Kw(disease_emb)
            Q = self.Qw(disease_emb)
            adj = torch.softmax(K @ Q.T, dim=-1)
            gru_output, _ = self.rnn_disease(torch.concat([torch.einsum("ikj,jr->ikr", dynamic_features, adj), dynamic_features], -1))
            
        else:
            gru_output, _ = self.rnn(dynamic_features)
            
        if mask is None:
            mask = torch.ones(dynamic_features.shape[0], dynamic_features.shape[1], dtype=torch.int64)
        dynamic_emb = self.get_last_visit(gru_output, mask)
        """ final """
        # print(f'dynamic_emb.shape is {dynamic_emb.shape}, static_emb.shape is {static_emb.shape}')
        # print(f'torch.concat([dynamic_emb, static_emb]).shape is {torch.concat([dynamic_emb, static_emb]).shape}')
        final = self.fc(torch.concat([dynamic_emb, static_emb], 1))
        # final = torch.maximum(final, torch.ones_like(final) * 0)
        # final = torch.minimum(final, torch.ones_like(final) * 200)
        return torch.exp(final).reshape(b, n, -1)
    
    def init(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                m.bias.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)