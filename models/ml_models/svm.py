from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import numpy as np
from models.ml_models.ml_base import ml_model
from sklearn.multioutput import MultiOutputRegressor
class SVM(ml_model):
    """
    SVM model for regression tasks.

    This class extends the ml_model class and implements a Support Vector Machine
    regressor, which is a powerful method for both classification and regression
    (known as SVR when used for regression).

    Parameters:
    ----------
    C : float, optional
        Regularization parameter, by default 1.0.
    kernel : str, optional
         Specifies the kernel type to be used in the algorithm, by default 'rbf'.
    gamma : str, optional
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid', by default 'scale'.
    epsilon : float, optional
        Epsilon in the epsilon-SVR model, by default 0.1.
    """

    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma='scale',
        epsilon=0.1
    ):
        super(SVM, self).__init__()
        self.svm = SVR(
            C=C,
            kernel=kernel,
            gamma=gamma,
            epsilon=epsilon
        )
        
        self.multioutput_model = MultiOutputRegressor(self.svm)

    def train(self, x, y):
        y = torch.squeeze(y).numpy()
        self.multioutput_model.fit(x, y)

    def validate(self, x, y):
        y = torch.squeeze(y).numpy()
        y_pred = self.multioutput_model.predict(x)
        print(f'x.shape is {x.shape}, y.shape is {y.shape}, y_pred.shape is {y_pred.shape}')
        print(f'mse is {mean_squared_error(y,y_pred)}')