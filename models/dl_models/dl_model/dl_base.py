import torch.nn as nn


class dl_model(nn.Module):
    def __init__(self):
        super(dl_model, self).__init__()


class temporal_model(dl_model):
    def __init__(self):
        super(temporal_model, self).__init__()


class spatio_temporal_model(dl_model):
    def __init__(self):
        super(spatio_temporal_model, self).__init__()


class PredictionHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(PredictionHead, self).__init__()
        self.hidden_dim = (hidden_dim,)
        self.output_dim = (output_dim,)
        self.act = act_layer()
        self.prediction_head_outcome = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(4 * hidden_dim, output_dim),
            nn.Dropout(drop),
            act_layer(),
        )

    def forward(self, x):
        x = self.act(x)
        outcome = self.prediction_head_outcome(x)
        return outcome
