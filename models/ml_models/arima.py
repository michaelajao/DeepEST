import pmdarima as pm
import numpy as np
import matplotlib.pyplot as plt
from DeepEST.models.ml_models.ml_base import ml_model


class ARIMA(ml_model):
    """
    AutoRegressive Integrated Moving Average (ARIMA) model for time series forecasting.

    This class extends the ml_model class and implements the ARIMA model,
    which is a popular statistical method for time series forecasting that includes
    autoregression, differencing, and moving average.

    Parameters:
    ----------
    None

    Attributes:
    -----------
    model : pmdarima.arima.model.ARIMA
        The ARIMA model instance.

    Methods:
    -------
    train(x, y)
        Train the ARIMA model using the provided data.
    validate(x, y)
        Validate the ARIMA model and plot the actual versus predicted data.
    """
    def __init__(self):
        """
        Initializes the ARIMA instance with no pre-defined model.
        """
        self.model = None

    def train(self, x, y):
        """
        Train the ARIMA model using the time series data.

        Parameters:
        ----------
        x : np.ndarray
            Time steps, should be of shape (N,).
        y : np.ndarray
            The time series data, should be a 1-dimensional array.

        Notes:
        -----
        The model attribute is set to the trained ARIMA model.
        """
        self.model = pm.auto_arima(y)

    def validate(self, x, y):
        """
        Validate the ARIMA model by predicting and comparing against actual data.

        Parameters:
        ----------
        x : np.ndarray
            Time steps, should be of shape (N,).
        y : np.ndarray
            The actual time series data.

        Returns:
        -------
        np.ndarray
            The predicted values from the ARIMA model.

        Notes:
        -----
        This method also plots the actual versus predicted data and displays it.
        """
        t = np.arange(len(y))
        y_pred = self.model.predict(len(y))
        plt.plot(t, y, label='Actual data')
        plt.plot(t, y_pred, label='Predicted data')
        plt.xlabel('Time')
        plt.ylabel('Number of infected people')
        plt.legend()
        plt.show()
        return y_pred
