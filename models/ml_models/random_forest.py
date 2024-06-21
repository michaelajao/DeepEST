import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from models.ml_models.ml_base import ml_model
import matplotlib.pyplot as plt
import torch
import numpy as np
class RandomForest(ml_model):
    """
    Random Forest model for regression tasks.

    This class extends the ml_model class and implements a Random Forest regressor,
    which is an ensemble learning method for regression that operates by constructing
    a multitude of decision trees.

    Parameters:
    ----------
    n_estimators : int, optional
        The number of trees in the forest, by default 100.
    max_depth : int or None, optional
        The maximum depth of the tree, by default None (unlimited).
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node, by default 2.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node, by default 1.
    random_state : int or None, optional
        Controls the randomness of the estimator, by default None.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None
    ):
        super(RandomForest, self).__init__()
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def train(self, x, y):
        y = torch.squeeze(y).numpy()
        self.rf.fit(x, y)

    def validate(self, x, y):
        y = torch.squeeze(y).numpy()
        y_pred = self.rf.predict(x)
        print(f'x.shape is {x.shape}, y.shape is {y.shape}, y_pred.shape is {y_pred.shape}')
        print(f'mse is {mean_squared_error(y, y_pred)}')
        # Further validation logic and plotting...