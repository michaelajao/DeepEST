import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from models.ml_models.ml_base import ml_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


class XGBoost(ml_model):
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
        self.XGB = xgb.XGBRegressor(
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
        y = torch.squeeze(y)
        self.multioutput_model.fit(x, y)

    def validate(self, x, y):
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
