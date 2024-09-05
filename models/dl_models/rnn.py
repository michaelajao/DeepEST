import torch.nn as nn
from DeepEST.models.dl_models.dl_base import temporal_model, PredictionHead


class RNN(temporal_model):
    """
    A Recurrent Neural Network (RNN) model for processing temporal data.

    This class extends the temporal_model class and implements a basic RNN network,
    suitable for sequences of data where the temporal order is important. It includes
    a prediction head for generating the final output.

    Parameters:
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    output_size : int
        The number of features in the output.
    num_layers : int, optional
        The number of recurrent layers, default is 1.

    Attributes:
    -----------
    rnn : torch.nn.RNN
        The RNN module.
    pred : PredictionHead
        The prediction head for the output layer.

    Methods:
    -------
    forward(x)
        Forward pass of the RNN model.
    """
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
        """
        Performs a forward pass through the RNN model and the prediction head.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor of shape (batch, time_step, input_size).

        Returns:
        -------
        torch.Tensor
            The output tensor from the prediction head.
        """
        r_out, _ = self.rnn(x, None)
        output = self.pred(r_out[:, -1, :])
        return output
