import torch.nn as nn

class dl_model(nn.Module):
    """
    Base class for deep learning models.

    This class serves as a foundation for more specific deep learning models.
    It inherits from PyTorch's nn.Module, ensuring all necessary deep learning
    functionalities are available.

    Parameters:
    ----------
    None
    """
    def __init__(self):
        super(dl_model, self).__init__()


class temporal_model(dl_model):
    """
    Class for models that focus on temporal data.

    This class extends the dl_model class and can be used to create models
    that are specifically designed to handle temporal data sequences.

    Parameters:
    ----------
    None
    """
    def __init__(self):
        super(temporal_model, self).__init__()


class spatial_temporal_model(dl_model):
    """
    Class for models that handle both spatial and temporal data.

    This class extends the dl_model class and is suitable for models that
    need to process data with spatial and temporal dimensions.

    Parameters:
    ----------
    None
    """
    def __init__(self):
        super(spatial_temporal_model, self).__init__()


class PredictionHead(nn.Module):
    """
    A prediction head module for neural networks.

    This module is designed to be used as the output layer of a neural network,
    typically for regression or classification tasks. It consists of a sequence
    of fully connected layers with activation functions and dropout for regularization.

    Parameters:
    ----------
    hidden_dim : int
        The size of the hidden dimension of the input feature vector.
    output_dim : int
        The size of the output dimension, typically the number of target classes or regression targets.
    act_layer : torch.nn.Module, optional
        The activation layer to use, defaults to nn.GELU.
    drop : float, optional
        The dropout rate, between 0 and 1, defaults to 0.0.

    Methods:
    -------
    forward(x)
        Forward pass of the prediction head module.
    """
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
        """
        Performs a forward pass through the prediction head module.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor to the prediction head.

        Returns:
        -------
        torch.Tensor
            The output tensor after passing through the prediction head.
        """
        x = self.act(x)
        outcome = self.prediction_head_outcome(x)
        return outcome
