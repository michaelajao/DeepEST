import torch.nn as nn
from models.dl_models.dl_base import temporal_model, PredictionHead


class GRU(temporal_model):
    def __init__(self, input_size, hidden_size, output_size,  num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.GRU = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.pred = PredictionHead(self.hidden_size, self.output_size)

    def forward(self, x):
        r_out, _ = self.GRU(x,None)
        output = self.pred(r_out[:, -1, :])
        return output
