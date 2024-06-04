from datetime import datetime
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange
from datasets import *
from models import *
from evaluator import *
import matplotlib.pyplot as plt

class ml_trainer:
    def __init__(
        self,
        trainset: TemporalDataSet,
        valset:TemporalDataSet,
        testset:TemporalDataSet,
        model_type: str = None,
    ):
        self.model_type = model_type
        self.train_data = trainset.origin_data
        self.train_label = trainset.origin_label
        self.val_data = valset.origin_data
        self.val_label = valset.origin_label
        self.test_data = testset.origin_data
        self.test_label = testset.origin_label
    def train(self):
        if self.model_type == "arima":
            print(f'torch.squeeze(self.train_label).shape is {self.train_label.shape}')
            model = pm.auto_arima(torch.squeeze(self.train_label))
            print(f"best model is {model}") #拟合出来最佳模型
            n_preds = torch.squeeze(self.val_label).shape[0]
            print(f'n_preds is {n_preds}')
            y_pred = model.predict(n_preds)
            y_target = torch.squeeze(self.val_label)
            print(getMetrics(y_pred, y_target))
            plt.plot(y_target, label="y_target")
            plt.plot(y_pred, label = "y_forecast")
            plt.legend(loc = 2)
            plt.show()
        if self.model.type == "xgboost":
            pass
        print(f'self.train_data.shape is {self.train_data.shape }, self.train_label.shape is {self.train_label.shape }')


if __name__ == "__main__":
    import pandas as pd
    claim_data = torch.Tensor(pd.read_pickle('data/claim_tensor.pkl'))
    county_data = torch.Tensor(pd.read_pickle('data/county_tensor.pkl'))
    hospitalizations_data = torch.Tensor(pd.read_pickle('data/hospitalizations.pkl'))
    distance_matrix = torch.Tensor(pd.read_pickle('data/distance_mat.pkl'))
    data_time = pd.read_pickle('data/date_range.pkl') #这个是list
    claim_data.shape, county_data.shape, hospitalizations_data.shape, distance_matrix.shape,
    dynamic_data = torch.cat((claim_data,torch.unsqueeze(hospitalizations_data, -1)), -1)
    static_data = county_data
    label = torch.unsqueeze(hospitalizations_data, -1)
    # dynamic_data = dynamic_data[:500]
    # static_data = static_data[:500]
    # label = label[:500]
    num_positions = dynamic_data.shape[0]
    spatio_indexes = []
    for i in range(3):
        spatio_indexes.append(list(range(0,num_positions)))
    train_dynamic_data, train_static_data, train_label,val_dynamic_data, val_static_data, val_label, test_dynamic_data, test_static_data, test_label = split_by_spatio_temporal(dynamic_data= dynamic_data, static_data=static_data,label=label,temporal_rate=[0.6,0.2,0.2], spatio_indexes=spatio_indexes)
    trainTemporalSet = TemporalDataSet(static_data=train_static_data, dynamic_data=train_dynamic_data, label=train_label, mode = 0, method = "mean")
    valTemporalSet = TemporalDataSet(static_data=val_static_data, dynamic_data= val_dynamic_data, label= val_label, mode = 0, method = "mean")
    testTemporalSet = TemporalDataSet(static_data=test_static_data, dynamic_data= test_dynamic_data, label = test_label, mode = 0, method = "mean")
    trainer = ml_trainer(trainTemporalSet,valTemporalSet,testTemporalSet,model_type="arima")
    trainer.train()
