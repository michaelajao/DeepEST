import torch.nn as nn
from models.dl_models.dl_base import temporal_model, PredictionHead


class LSTM(temporal_model):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.pred = PredictionHead(self.hidden_size, self.output_size)

    def forward(self, x, h_n=None, h_c=None, his_info=False):
        if his_info:
            r_out, (h_n, h_c) = self.lstm(x, (h_n, h_c))
            output = self.pred(r_out)
            return output, h_n, h_c
        else:
            r_out, (h_n, h_c) = self.lstm(x)
            output = self.pred(r_out[:, -1, :])
            return output
