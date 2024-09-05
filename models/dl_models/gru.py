import torch.nn as nn
from DeepEST.models.dl_models.dl_base import temporal_model, PredictionHead


class GRU(temporal_model):
    """
    A Gated Recurrent Unit (GRU) model for processing temporal data.

    This class extends the temporal_model class and implements a GRU network,
    suitable for sequences of data where the temporal order is important.
    It includes a prediction head for generating the final output.

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
    GRU : torch.nn.GRU
        The GRU module.
    pred : PredictionHead
        The prediction head for the output layer.

    Methods:
    -------
    forward(x)
        Forward pass of the GRU model.
    """

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
        """
        Performs a forward pass through the GRU model and the prediction head.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor of shape (batch, seq_length, input_size).

        Returns:
        -------
        torch.Tensor
            The output tensor from the prediction head.
        """
        r_out, _ = self.GRU(x,None)
        output = self.pred(r_out[:, -1, :])
        return output
