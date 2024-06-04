from typing import Optional, Type
import numpy as np
import torch
import os
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
from datasets import *
from models import *
from evaluator import *
import logging
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datasets import preprocess
from visualize import plot_line_chart


def get_logger(name, file_name='log_file', mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 以下两行是为了在jupyter notebook 中不重复输出日志
    if logger.root.handlers:
        logger.root.handlers[0].setLevel(logging.WARNING)

    # handler_stdout = logging.StreamHandler() #need_recover
    # handler_stdout.setLevel(logging.INFO)
    # handler_stdout.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(handler_stdout)#need_recover

    handler_file = logging.FileHandler(filename=file_name, mode= mode, encoding='utf-8')
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler_file)

    return logger




class DlTrainer:
    """Trainer for PyTorch models.

    Args:
        model: PyTorch model.
        # todo checkpoint_path: Path to the checkpoint. Default is None, which means
            the model will be randomly initialized.
        #todo metrics: List of metric names to be calculated. Default is None, which
            means the default metrics in each metrics_fn will be used.
        device: Device to be used for training. Default is None, which means
            the device will be GPU if available, otherwise CPU.
        enable_logging: Whether to enable logging. Default is True.
        output_path: Path to save the output. Default is "./output".
        exp_name: Name of the experiment. Default is current datetime.
    """
    def __init__(
        self,
        model: nn.Module,
        checkpoint_base:  str = "./checkpoints/",
        logfile_base: str = "./logfiles",
        # metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        # enable_logging: bool = True,
        # output_path: Optional[str] = None,
        # exp_name: Optional[str] = None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.device = device
        self.model.to(self.device)
        if not os.path.exists(logfile_base):
            try:
                # 创建文件夹
                os.makedirs(logfile_base)
                print(f"文件夹 '{logfile_base}' 创建成功！")
            except OSError as e:
                print(f"创建文件夹 '{logfile_base}' 失败: {e}")
        logfile_name = logfile_base + type(self.model).__name__
        self.logger = get_logger(__name__, logfile_name)
        self.checkpoint_base = checkpoint_base
        if not os.path.exists(checkpoint_base):
            try:
                # 创建文件夹
                os.makedirs(checkpoint_base)
                print(f"文件夹 '{checkpoint_base}' 创建成功！")
            except OSError as e:
                print(f"创建文件夹 '{checkpoint_base}' 失败: {e}")

    def train(
        self,
        epochs: int = 100,
        LR: int = 0.001,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        weight_decay: int = 0.0,

    ):
        self.checkpoint_path = self.checkpoint_base + type(self.model).__name__ + 'params.pth'
        optimizer = optimizer_class(self.model.parameters(), lr=LR, weight_decay=weight_decay)
        data_iterator = iter(train_dataloader)
        steps_per_epoch = len(train_dataloader)
        criterion = torch.nn.MSELoss()
        writer = SummaryWriter()
        min_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            total_batch = 0
            total_loss = 0
            for i in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                x, y_true = data[0], data[1]
                x.to(self.device)
                cur_batch = x.shape[0]
                total_batch = total_batch + cur_batch
                # print(f'x.shape is {x.shape}')
                y_true.to(self.device)
                y_pred = self.model(x)
                # print(f'y_true.shape is {torch.squeeze(y_true).shape}, y_pred.shape is {y_pred.shape}')
                # print(f'y_true is {torch.squeeze(y_true)}, y_pred.shape is {y_pred}')
                loss = criterion(y_pred, torch.squeeze(y_true))
                total_loss = total_loss + loss * cur_batch 
                # logger.info()
                writer.add_scalar('loss', loss , global_step= steps_per_epoch * epoch + i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss = total_loss / total_batch
            val_loss = self.validate(val_dataloader= val_dataloader)
            if val_loss < min_loss:
                min_loss = val_loss
                print('---------------Save Best Checkpoint---------------')
                self.logger.info('---------------Save Best Checkpoint---------------')
                print(f'update min_loss, loss is {val_loss}')
                self.logger.info(f'update min_loss, loss is {val_loss}')
                torch.save(self.model.state_dict(), self.checkpoint_path)

            print(f'mean_loss is {mean_loss}')

    def validate(
        self,
        val_dataloader: DataLoader,
        LR: int = 0.0002,
        fig = True
    ):
        data_iterator = iter(val_dataloader)
        steps_per_epoch = len(val_dataloader)
        criterion = torch.nn.MSELoss()
        y_target = []
        y_forecast = []
        self.model.eval()
        total_batch = 0
        total_loss = 0
        for _ in trange(
            steps_per_epoch,
            smoothing=0.05,
        ):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(val_dataloader)
                data = next(data_iterator)
            x, y_true = data[0], data[1]
            x.to(self.device)
            cur_batch = x.shape[0]
            total_batch = total_batch + cur_batch
            # print(f'x.shape is {x.shape},y_true.shape is {y_true.shape}')
            y_true.to(self.device)
            y_pred = self.model(x)
            # print(f'y_true.shape is {torch.squeeze(y_true).shape}, y_pred.shape is {y_pred.shape}')
            y_target += np.array(torch.squeeze(y_true).detach().flatten()).tolist()
            y_forecast += np.array(y_pred.detach()).flatten().tolist()
            # print(f'y_true.shape is {torch.squeeze(y_true).shape}, y_pred.shape is {y_pred.shape}')
            # print(f'y_true is {torch.squeeze(y_true)}, y_pred.shape is {y_pred}')
            loss = criterion(y_pred,torch.squeeze(y_true))
            total_loss = total_loss + loss * cur_batch 
            print(f'cur_loss is {loss}')
        mean_loss = total_loss / total_batch
        # self.logger.info(f'validate_loss is {mean_loss}')
        # print(getMetrics(y_target, y_forecast))
        return mean_loss


class DlTester:
    """Tester for PyTorch models.

    Args:
        model: PyTorch model.
        device: Device to be used for training. Default is None, which means
            the device will be GPU if available, otherwise CPU.
        enable_logging: Whether to enable logging. Default is True.
        output_path: Path to save the output. Default is "./output".
        exp_name: Name of the experiment. Default is current datetime.
    """
    def __init__(
        self,
        model: nn.Module,
        logfile_base: str = './logfiles/',
        checkpoint_base: str = './checkpoints/',
        # metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        # enable_logging: bool = True,
        # output_path: Optional[str] = None,
        # exp_name: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.checkpoint_path = checkpoint_base + type(self.model).__name__ + 'params.pth'
        if not os.path.exists(logfile_base):
            try:
                # 创建文件夹
                os.makedirs(logfile_base)
                print(f"文件夹 '{logfile_base}' 创建成功！")
            except OSError as e:
                print(f"创建文件夹 '{logfile_base}' 失败: {e}")
        logfile_name = logfile_base + type(self.model).__name__
        self.logger = get_logger(__name__, logfile_name, mode='a')

    def test(
        self,
        test_dataloader: DataLoader,
        LR: int = 0.0002,
        fig: bool = True,
        visualize: bool = True,
        preprocess: PreprocessData = None
    ):
        state_dict = torch.load(self.checkpoint_path)
        self.model.load_state_dict(state_dict=state_dict)
        data_iterator = iter(test_dataloader)
        steps_per_epoch = len(test_dataloader)
        criterion = torch.nn.MSELoss()
        y_target = []
        y_forecast = []
        self.model.eval()
        total_batch = 0
        total_loss = 0
        predict_list = []
        label_list = []
        for _ in trange(
            steps_per_epoch,
            smoothing=0.05,
        ):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(test_dataloader)
                data = next(data_iterator)
            x, y_true = data[0], data[1]
            x.to(self.device)
            cur_batch = x.shape[0]
            total_batch = total_batch + cur_batch
            # print(f'x.shape is {x.shape},y_true.shape is {y_true.shape}')
            y_true.to(self.device)
            y_pred = self.model(x)
            print(f'y_true.shape is {torch.squeeze(y_true).shape}, y_pred.shape is {y_pred.shape}')
            predict_list.append(torch.squeeze(y_pred, -1))
            label_list.append(torch.squeeze(y_true, -1))
            y_target += np.array(torch.squeeze(y_true).detach().flatten()).tolist()
            y_forecast += np.array(y_pred.detach()).flatten().tolist()
            # print(f'y_true.shape is {torch.squeeze(y_true).shape}, y_pred.shape is {y_pred.shape}')
            # print(f'y_true is {torch.squeeze(y_true)}, y_pred.shape is {y_pred}')
            loss = criterion(y_pred,torch.squeeze(y_true))
            total_loss = total_loss + loss * cur_batch 
            # print(f'cur_loss is {loss}')

        mean_loss = total_loss / total_batch
        print(f'test_loss is {mean_loss}')
        # self.logger.info(f'test_loss is {mean_loss}')
        print(getMetrics(y_target, y_forecast))
        self.logger.info(getMetrics(y_target, y_forecast))

        if visualize:
            predict_tensor = torch.cat(predict_list).detach().cpu()
            label_tensor = torch.cat(label_list).detach().cpu()
            predict_tensor = preprocess.reverse_test_norm(predict_tensor) if preprocess is not None else predict_tensor
            label_tensor = preprocess.reverse_test_norm(label_tensor) if preprocess is not None else label_tensor
            print(f'predict_tensor.shape is {predict_tensor.shape}, label_tensor.shape is {label_tensor.shape}')
            plot_line_chart(predict_tensor, label_tensor)




class MlTrainer:
    def __init__(
        self,
        model,
    ):
        self.model = model

    def train(self, train_dataloader: DataLoader,):
        data_iterator = iter(train_dataloader)
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_dataloader)
            data = next(data_iterator)
        x, y = data[0], data[1]
        y = torch.squeeze(y, -1)
        if isinstance(self.model, SIR) or isinstance(self.model, SEIR):
            y = torch.squeeze(y).to(torch.int)
        print(f'x.shape is {x.shape}, y.shape is {y.shape}')
        self.model.train(x, y)

    def evaluate(self, val_dataloader: DataLoader):
        data_iterator = iter(val_dataloader)
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(val_dataloader)
            data = next(data_iterator)
        x, y = data[0], data[1]
        if isinstance(self.model, SIR) or isinstance(self.model, SEIR):
            y = torch.squeeze(y).to(torch.int)
        self.model.validate(x, y)
