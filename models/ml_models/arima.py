import pmdarima as pm
import numpy as np
import matplotlib.pyplot as plt
from models.ml_models.ml_base import ml_model


class ARIMA(ml_model):
    # 'the shape of train_data should be (N,), which means the data has N timesteps'
    def __init__(self):
        # self.train_data = train_data
        # self.train_label = train_label
        # print(f"self.train_data.shape is {self.train_data.shape}, self.train_label.shape is {self.train_label.shape}")
        self.model = None

    def train(self, x, y):
        self.model = pm.auto_arima(y)

    def validate(self, x, y):
        t = np.arange(len(y))
        y_pred = self.model.predict(len(y))
        plt.plot(t, y, label='Actual data')
        plt.plot(t, y_pred, label='Predicted data')
        plt.xlabel('Time')
        plt.ylabel('Number of infected people')
        plt.legend()
        plt.show()
        return y_pred
