import torch.nn as nn
from DeepEST.models.dl_models.dl_base import temporal_model, PredictionHead


class LSTM(temporal_model):
    """
    An Long Short-Term Memory (LSTM) model for processing temporal data.

    This class extends the temporal_model class and implements an LSTM network,
    suitable for sequences of data where the temporal order is important and
    long-term dependencies need to be captured. It includes a prediction head for
    generating the final output.

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
    lstm : torch.nn.LSTM
        The LSTM module.
    pred : PredictionHead
        The prediction head for the output layer.

    Methods:
    -------
    forward(x, h_n=None, h_c=None, his_info=False)
        Forward pass of the LSTM model with an option to retain or reset the hidden and cell states.
    """
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
        """
        Performs a forward pass through the LSTM model and the prediction head.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor of shape (batch, seq_length, input_size).
        h_n : torch.Tensor, optional
            The initial hidden state tensor, if None uses the default initialization.
        h_c : torch.Tensor, optional
            The initial cell state tensor, if None uses the default initialization.
        his_info : bool, optional
            Whether to return the hidden and cell states for each time step.

        Returns:
        -------
        torch.Tensor
            The output tensor from the prediction head.

        If `his_info` is True, it also returns:
        -------
        tuple
            A tuple of (h_n, h_c) containing the hidden and cell states.
        """
        if his_info:
            r_out, (h_n, h_c) = self.lstm(x, (h_n, h_c))
            output = self.pred(r_out)
            return output, h_n, h_c
        else:
            r_out, (h_n, h_c) = self.lstm(x)
            output = self.pred(r_out[:, -1, :])
            return output
