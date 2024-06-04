import numpy as np
import matplotlib.pyplot as plt
from models.ml_models.ml_base import ml_model
from scipy.integrate import odeint
from scipy.optimize import curve_fit


class SEIR(ml_model):
    def __init__(self, S0, E0, I0, R0) -> None:
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.beta = 0
        self.gamma = 0
        self.alpha = 0
        self.train_len = 0
        self.val_len = 0

    def seir_model(self, y, t, beta, gamma, alpha):
        # y: [S, E, I, R] 四个状态的人数
        # t: 时间
        # beta: 感染率
        # gamma: 移出率
        # alpha: 潜伏率
        # 输出: [dS/dt, dE/dt, dI/dt, dR/dt] 四个状态的变化率
        S, E, I, R = y
        N = S + E + I + R  # 总人数
        dS_dt = -beta * S * I / N  # 易感者的变化率
        dE_dt = beta * S * I / N - alpha * E  # 暴露者的变化率
        dI_dt = alpha * E - gamma * I  # 感染者的变化率
        dR_dt = gamma * I  # 移出者的变化率
        return [dS_dt, dE_dt, dI_dt, dR_dt]

    def fit_odeint(self, t, beta, gamma, alpha):
        # t: 时间序列
        # beta: 感染率
        # gamma: 移出率
        # alpha: 潜伏率
        # 输出: 模型预测的感染者人数序列
        return odeint(self.seir_model, (self.S0, self.E0, self.I0, self.R0), t, args=(beta, gamma, alpha))[:, 2]

    def train(self, x, y):
        # N = self.I0 + self.E0 + self.R0 + self.S0
        self.train_len = len(y)
        t = np.arange(len(y))
        popt, _ = curve_fit(self.fit_odeint, t, y)
        self.beta = popt[0]
        self.gamma = popt[1]
        self.alpha = popt[2]
        print(f'beta = {self.beta}, gamma = {self.gamma}, alpha = {self.alpha}')

    def validate(self, x, y):
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
