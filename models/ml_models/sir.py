import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

class SIR:
    def __init__(self, S0, I0, R0) -> None:
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.beta = 0
        self.gamma = 0

    def sir_model(self, y, t, beta, gamma):
        S, I, R = y
        N = S + I + R
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return [dS_dt, dI_dt, dR_dt]

    def fit_odeint(self, t, beta, gamma):
        return odeint(self.sir_model, (self.S0, self.I0, self.R0), t, args=(beta, gamma))[:, 1]

    def train(self, x, y):
        t = np.arange(len(y))
        self.train_len = len(y)
        popt, pcov = curve_fit(self.fit_odeint, t, y, maxfev=5000)
        beta, gamma = popt
        print(f'Estimated beta = {beta}, Estimated gamma = {gamma}')
        self.beta = beta
        self.gamma = gamma
        return beta, gamma

    def validate(self, x, y):
        # t = np.arange(len(y))
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

