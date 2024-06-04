import torch.nn as nn
from models.dl_models.dl_base import temporal_model, PredictionHead


class RNN(temporal_model):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn = nn.RNN(
            input_size=self.input_size,  # 每个时间步的每个个数据的大小
            hidden_size=self.hidden_size,  # 每个细胞中神经元个数
            num_layers=self.num_layers,  # 每个细胞中FC的层数
            batch_first=True
        )
        self.pred = PredictionHead(self.hidden_size, self.output_size)

    def forward(self, x):
        # x (batch, time_step, input_size)  输入格式
        # h_state (n_layers, batch, hidden_size)  最后一步的状态
        # r_out (time_step, batch, hidden_size)  保存了每一步的隐藏状态
        r_out, _ = self.rnn(x, None)
        output = self.pred(r_out[:, -1, :])
        return output
