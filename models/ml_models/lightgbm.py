import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from DeepEST.models.ml_models.ml_base import ml_model
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import torch
import numpy as np

# Assuming ml_model is already defined

class LightGBM(ml_model):
    """
    LightGBM model for regression tasks.
    [Include detailed description and parameters as per XGBoost example]
    """
    def __init__(
        self,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        num_leaves=31,
        random_state=42
    ):
        self.lgbm = lgb.LGBMRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            random_state=random_state
        )
        self.multioutput_model = MultiOutputRegressor(self.lgbm)

    def train(self, x, y):
        """
        Train the LightGBM model.

        Parameters:
        ----------
        x : np.ndarray or pd.DataFrame
            Features for training (shape [n_samples, n_features]).
        y : torch.Tensor
            Target values for training (shape [n_samples, n_outputs]).
        """
        y = torch.squeeze(y).numpy()
        print(f'x.shape is {x.shape}, y.shape is {y.shape}')
        self.multioutput_model.fit(x, y)

    def validate(self, x, y):
        """
        Validate the LightGBM model and plot the actual vs predicted values.

        Parameters:
        ----------
        x : np.ndarray or pd.DataFrame
            Features for validation (shape [n_samples, n_features]).
        y : torch.Tensor
            Target values for validation (shape [n_samples, n_outputs]).
        """
        y = torch.squeeze(y).numpy()
        y_pred = self.multioutput_model.predict(x)
        print(f'x.shape is {x.shape}, y.shape is {y.shape}, y_pred.shape is {y_pred.shape}')
        print(f'mse is {mean_squared_error(y,y_pred)}')
        