import numpy as np
import matplotlib.pyplot as plt
from models.ml_models.ml_base import ml_model
from scipy.integrate import odeint
from scipy.optimize import curve_fit


class SEIR(ml_model):
    """
    SEIR epidemiological model for simulating the spread of infectious diseases.

    The SEIR model is an extension of the SIR model, accounting for the latent
    period during which individuals are infected but not yet infectious.

    Parameters:
    ----------
    S0 : float
        Initial number of susceptible individuals.
    E0 : float
        Initial number of exposed (latent) individuals.
    I0 : float
        Initial number of infectious individuals.
    R0 : float
        Initial number of recovered individuals.

    Attributes:
    -----------
    beta : float
        The infection rate of the disease (new infections per contact).
    gamma : float
        The recovery rate of the disease (1/duration of infection).
    alpha : float
        The rate of transition from the exposed to the infectious class.
    train_len : int
        The length of the training data.
    val_len : int
        The length of the validation data.

    Methods:
    -------
    __init__(S0, E0, I0, R0)
        Initializes the SEIR model with initial conditions for each compartment.
    seir_model(y, t, beta, gamma, alpha)
        Defines the differential equations for the SEIR model.
    fit_odeint(t, beta, gamma, alpha)
        Fits the SEIR model to the provided time series data.
    train(x, y)
        Estimates the model parameters by fitting to the training data.
    validate(x, y)
        Validates the model on the validation data and visualizes the results.
    """
    def __init__(self, S0, E0, I0, R0) -> None:
        """
        Initialize the SEIR model with the initial conditions for susceptible,
        exposed, infectious, and recovered individuals.
        """
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.beta = 0 # Infection rate, to be estimated
        self.gamma = 0 # Recovery rate, to be estimated
        self.alpha = 0 # Latent rate, to be estimated
        self.train_len = 0
        self.val_len = 0

    def seir_model(self, y, t, beta, gamma, alpha):
        """
        Define the SEIR model differential equations.

        Parameters:
        ----------
        y : array_like
            Current state vector [S, E, I, R].
        t : float
            Current time.
        beta : float
            Infection rate.
        gamma : float
            Recovery rate.
        alpha : float
            Latent rate.

        Returns:
        -------
        array_like
            The rate of change of the state vector [dS/dt, dE/dt, dI/dt, dR/dt].
        """
        S, E, I, R = y
        N = S + E + I + R  # Total number of people 
        dS_dt = -beta * S * I / N  # Rate of change in susceptible individuals
        dE_dt = beta * S * I / N - alpha * E  # Rate of change of the exposed
        dI_dt = alpha * E - gamma * I  # Rate of change of infected persons
        dR_dt = gamma * I  # Rate of change of the remover
        return [dS_dt, dE_dt, dI_dt, dR_dt]

    def fit_odeint(self, t, beta, gamma, alpha):
        """
        Use the odeint function to fit the SEIR model to the time series data.

        Parameters:
        ----------
        t : array_like
            Time points.
        beta : float
            Infection rate.
        gamma : float
            Recovery rate.
        alpha : float
            Latent rate.

        Returns:
        -------
        array_like
            The number of infectious individuals I over time.
        """
        return odeint(self.seir_model, (self.S0, self.E0, self.I0, self.R0), t, args=(beta, gamma, alpha))[:, 2]

    def train(self, x, y):
        """
        Fit the SEIR model to the training data to estimate the parameters beta, gamma, and alpha.

        Parameters:
        ----------
        x : array_like
            Time steps (currently unused).
        y : array_like
            Observed number of infectious individuals over time.

        Notes:
        -----
        The training process uses curve fitting to find the best parameters that
        minimize the difference between the model predictions and the observed data.
        """
        self.train_len = len(y)
        t = np.arange(len(y))
        popt, _ = curve_fit(self.fit_odeint, t, y)
        self.beta = popt[0]
        self.gamma = popt[1]
        self.alpha = popt[2]
        print(f'beta = {self.beta}, gamma = {self.gamma}, alpha = {self.alpha}')

    def validate(self, x, y):
        """
        Validate the SEIR model on the validation data and plot the results.

        Parameters:
        ----------
        x : array_like
            Time steps (currently unused).
        y : array_like
            Observed number of infectious individuals over the validation period.

        Returns:
        -------
        array_like
            Predicted number of infectious individuals over the validation period.

        Notes:
        -----
        The method extends the training time points to include the validation period,
        fits the model to the extended time series, and plots the actual versus predicted
        number of infectious individuals.
        """
        self.val_len = len(y)
        t = np.arange(self.train_len + self.val_len)
        # print(f'self.train_len is {self.train_len}, self.val_len is {self.val_len}')
        y_pred = self.fit_odeint(t, self.beta, self.gamma, self.alpha)
        plt.plot(t[self.train_len: self.train_len + self.val_len], y, label='Actual data')
        plt.plot(t[self.train_len: self.train_len + self.val_len], y_pred[self.train_len: self.train_len + self.val_len], label='Predicted data')
        plt.xlabel('Time')
        plt.ylabel('Number of infected people')
        plt.legend()
        plt.show()
        return y_pred
