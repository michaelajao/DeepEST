import numpy as np
import matplotlib.pyplot as plt
import torch
from DeepEST.models.ml_models.ml_base import ml_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


class XGBoost(ml_model):
    """
    XGBoost model for regression tasks.

    This class extends the ml_model class and implements an XGBoost regressor,
    which is an efficient and versatile machine learning algorithm for a wide range
    of regression tasks.

    Parameters:
    ----------
    max_depth : int, optional
        The maximum depth of a tree, by default 6.
    learning_rate : float, optional
        The step size at each iteration while moving toward a minimum, by default 0.1.
    n_estimators : int, optional
        The number of gradient boosted trees, by default 100.
    objective : str, optional
        The objective function to be minimized, by default 'reg:squarederror'.
    booster : str, optional
        The type of booster, by default 'gbtree'.
    gamma : int, optional
        The minimum loss reduction required to make a further partition on a leaf node of the tree, by default 0.
    min_child_weight : int, optional
        The minimum sum of instance weight (hessian) needed in a child, by default 1.
    subsample : float, optional
        The fraction of observations to be sampled for each tree, by default 1.
    colsample_bytree : float, optional
        The fraction of features to be used for each tree, by default 1.
    reg_alpha : float, optional
        The L1 regularization term on weights, by default 0.
    reg_lambda : float, optional
        The L2 regularization term on weights, by default 1.
    random_state : int, optional
        Controls the randomness of the estimator, by default 0.

    Attributes:
    -----------
    XGB : xgboost.XGBRegressor
        The XGBoost regressor model.
    multioutput_model : MultiOutputRegressor
        The multi-output regressor wrapping the XGBoost model.

    Methods:
    -------
    train(x, y)
        Train the XGBoost model with the provided data.
    validate(x, y)
        Validate the XGBoost model and plot the results.
    """
    def __init__(
        self,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective='reg:squarederror',
        booster='gbtree',
        gamma=0,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=0,
        reg_lambda=1,
        random_state=0
    ):
        """
        Initialize the XGBoost model with the specified hyperparameters.
        """
        self.XGB = xgboost.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            booster=booster,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state
        )
        self.multioutput_model = MultiOutputRegressor(self.XGB)

    def train(self, x, y):
        """
        Train the XGBoost model.

        Parameters:
        ----------
        x : np.ndarray or pd.DataFrame
            Features for training.
        y : torch.Tensor
            Target values for training.
        """
        y = torch.squeeze(y)
        self.multioutput_model.fit(x, y)

    def validate(self, x, y):
        """
        Validate the XGBoost model and plot the actual vs predicted values.

        Parameters:
        ----------
        x : np.ndarray or pd.DataFrame
            Features for validation.
        y : torch.Tensor
            Target values for validation.

        Returns:
        -------
        np.ndarray
            Predicted values from the XGBoost model.

        Notes:
        -----
        This method calculates and prints the mean squared error, and plots
        the actual vs predicted values for visual comparison.
        """
        y = torch.squeeze(y)
        y_pred = self.multioutput_model.predict(x)

        print(f'x.shape is {x.shape}, y.shape is {y.shape}, y_pred.shape is {y_pred.shape}')

        print(f'mse is {mean_squared_error(y,y_pred)}')
        true_y = np.mean(np.array(y), 0)
        pred_y = np.mean(y_pred, 0)
        # true_y = y[0]
        # pred_y = y_pred[0]
        t = np.arange(len(true_y))
        plt.plot(t, true_y, label='Actual data')
        plt.plot(t, pred_y, label='Predicted data')
        plt.xlabel('Time')
        plt.ylabel('Number of infected people')
        plt.legend()
        plt.show()
        return y_pred
