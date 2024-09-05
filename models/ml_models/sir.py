import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from DeepEST.models.ml_models.ml_base import ml_model


class SIR(ml_model):
    """
    Simple epidemiological SIR (Susceptible, Infected, Recovered) model.

    This class implements the SIR model, which is a system of ordinary differential
    equations used to describe the time evolution of the susceptible, infected, and
    recovered populations during an epidemic.

    Parameters:
    ----------
    S0 : float
        Initial number of susceptible individuals.
    I0 : float
        Initial number of infected individuals.
    R0 : float
        Initial number of recovered individuals.

    Attributes:
    -----------
    beta : float
        The infection rate (transmission rate).
    gamma : float
        The recovery rate (inverse of the infectious period).

    Methods:
    -------
    __init__(S0, I0, R0)
        Initializes the SIR instance with initial conditions.
    sir_model(y, t, beta, gamma)
        Defines the SIR model equations.
    fit_odeint(t, beta, gamma)
        Fits the SIR model to the data using the ODE integrator.
    train(x, y)
        Trains the SIR model by fitting the model parameters to the training data.
    validate(x, y)
        Validates the SIR model on the validation data and plots the results.
    """
    def __init__(self, S0, I0, R0) -> None:
        """
        Initializes the SIR model with initial conditions for the susceptible,
        infected, and recovered populations.

        Parameters:
        ----------
        S0 : float
            Initial susceptible population.
        I0 : float
            Initial infected population.
        R0 : float
            Initial recovered population.
        """
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.beta = 0
        self.gamma = 0

    def sir_model(self, y, t, beta, gamma):
        """
        Computes the derivatives of the SIR model.

        Parameters:
        ----------
        y : tuple
            The current values of S, I, and R.
        t : float
            Current time.
        beta : float
            The infection rate.
        gamma : float
            The recovery rate.

        Returns:
        -------
        list
            The derivatives [dS_dt, dI_dt, dR_dt] at time t.
        """
        S, I, R = y
        N = S + I + R
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return [dS_dt, dI_dt, dR_dt]

    def fit_odeint(self, t, beta, gamma):
        """
        Fits the SIR model to the given time points.

        Parameters:
        ----------
        t : np.ndarray
            Timesteps.
        beta : float
            The infection rate.
        gamma : float
            The recovery rate.

        Returns:
        -------
        np.ndarray
            The infected population I at each time point.
        """
        return odeint(self.sir_model, (self.S0, self.I0, self.R0), t, args=(beta, gamma))[:, 1]

    def train(self, x, y):
        """
        Trains the SIR model by estimating the beta and gamma parameters.

        Parameters:
        ----------
        x : np.ndarray
            Time steps or indices (unused in this method).
        y : np.ndarray
            The observed number of infected individuals.

        Returns:
        -------
        tuple
            The estimated beta and gamma values.

        Notes:
        -----
        The training process involves fitting the SIR model to the data using
        non-linear least squares to estimate the parameters beta and gamma.
        """
        t = np.arange(len(y))
        self.train_len = len(y)
        popt, pcov = curve_fit(self.fit_odeint, t, y, maxfev=5000)
        beta, gamma = popt
        print(f'Estimated beta = {beta}, Estimated gamma = {gamma}')
        self.beta = beta
        self.gamma = gamma
        return beta, gamma

    def validate(self, x, y):
        """
        Validates the SIR model on a separate dataset and plots the results.

        Parameters:
        ----------
        x : np.ndarray
            Time steps or indices (unused in this method).
        y : np.ndarray
            The observed number of infected individuals for validation.

        Returns:
        -------
        np.ndarray
            The predicted number of infected individuals.

        Notes:
        -----
        This method plots the validation results, showing the actual and predicted
        number of infected individuals for comparison.
        """
        self.val_len = len(y)
        t = np.arange(self.train_len + self.val_len)
        # print(f'self.train_len is {self.train_len}, self.val_len is {self.val_len}')
        y_pred = self.fit_odeint(t, self.beta, self.gamma)
        plt.plot(t[self.train_len: self.train_len + self.val_len], y, label='Actual data')
        plt.plot(t[self.train_len: self.train_len + self.val_len], y_pred[self.train_len: self.train_len + self.val_len], label='Predicted data')
        plt.xlabel('Time')
        plt.ylabel('Number of infected people')
        plt.legend()
        plt.show()
        return y_pred

