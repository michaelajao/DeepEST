from typing import Optional, Type
import numpy as np
import torch
import os
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cwd, ".."))
sys.path.insert(0, os.path.join(cwd, "."))
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
    """
    Configures and returns a logger instance.

    Parameters:
    ----------
    name : str
        Name of the logger.
    file_name : str, optional
        Filename for the log file, default is 'log_file'.
    mode : str, optional
        Mode in which the file is opened, default is 'w' (overwrite).

    Returns:
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.root.handlers:
        logger.root.handlers[0].setLevel(logging.WARNING)


    handler_file = logging.FileHandler(filename=file_name, mode= mode, encoding='utf-8')
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler_file)

    return logger




class dl_trainer:
    """
    Trainer class for deep learning models that handles training and validation processes.

    Attributes:
    ----------
    model : torch.nn.Module
        The PyTorch model instance to be trained and validated.
    device : str
        The device on which the model and data are placed ('cuda' or 'cpu').
    logger : logging.Logger
        The logger instance for logging training and validation information.
    checkpoint_path : str
        The file path where the best model checkpoint is saved.
    """
    def __init__(
        self,
        model: nn.Module,
        checkpoint_base:  str = "./checkpoints/",
        logfile_base: str = "./logfiles/",
        device: Optional[str] = None,
    ):
        """
        Initializes the deep learning trainer with the given model and settings.

        Parameters:
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        checkpoint_base : str, default is "./checkpoints/"
            Base directory for saving model checkpoints.
        logfile_base : str, default is "./logfiles"
            Base directory for saving log files.
        device : str, optional
            Device to run the training on ('cuda' or 'cpu'). If None, it chooses automatically based on availability.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.device = device
        self.model.to(self.device)
        if not os.path.exists(logfile_base):
            try:
                # 创建文件夹
                os.makedirs(logfile_base)
            except OSError as e:
                raise RuntimeError(f"Fail to create dir '{logfile_base}': {e}")
        logfile_name = logfile_base + type(self.model).__name__
        self.logger = get_logger(__name__, logfile_name)
        self.checkpoint_base = checkpoint_base
        if not os.path.exists(checkpoint_base):
            try:
                # 创建文件夹
                os.makedirs(checkpoint_base)
            except OSError as e:
                raise RuntimeError(f"Fail to create dir '{checkpoint_base}': {e}")

    def train(
        self,
        epochs: int = 100,
        LR: int = 0.001,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        weight_decay: int = 0.0,

    ):
        """
        Trains the model for a fixed number of epochs.

        Parameters:
        ----------
        epochs : int
            Number of epochs to train the model.
        LR : float
            Learning rate for the optimizer.
        train_dataloader : torch.utils.data.DataLoader
            Dataloader for the training dataset.
        val_dataloader : torch.utils.data.DataLoader
            Dataloader for the validation dataset.
        optimizer_class : torch.optim.Optimizer subclass
            Optimizer class to use for training.
        weight_decay : float
            Weight decay coefficient for regularization.
        """
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
                y_true.to(self.device)
                y_pred = self.model(x)
                loss = criterion(y_pred, torch.squeeze(y_true))
                total_loss = total_loss + loss * cur_batch 
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
        """
        Validates the model on the validation dataset.

        Parameters:
        ----------
        val_dataloader : torch.utils.data.DataLoader
            Dataloader for the validation dataset.
        LR : float
            Learning rate (not used in validation).
        fig : bool
            Whether to plot figures (not implemented in this stub).

        Returns:
        -------
        float
            The mean validation loss.
        """
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
            y_true.to(self.device)
            y_pred = self.model(x)
            y_target += np.array(torch.squeeze(y_true).detach().flatten()).tolist()
            y_forecast += np.array(y_pred.detach()).flatten().tolist()
            loss = criterion(y_pred,torch.squeeze(y_true))
            total_loss = total_loss + loss * cur_batch 
        mean_loss = total_loss / total_batch
        return mean_loss


class dl_tester:
    """
    Tester class for PyTorch models to evaluate their performance on test data.

    Attributes:
    ----------
    model : torch.nn.Module
        The PyTorch model to be tested.
    device : str
        The device on which the model and data are placed ('cuda' or 'cpu').
    checkpoint_path : str
        The file path to load the trained model's state dictionary.
    logger : logging.Logger
        The logger instance for logging test results.
    """
    def __init__(
        self,
        model: nn.Module,
        logfile_base: str = './logfiles/',
        checkpoint_base: str = './checkpoints/',
        device: Optional[str] = None,
    ):
        """
        Initializes the tester with the given model and settings.

        Parameters:
        ----------
        model : torch.nn.Module
            The PyTorch model to be tested.
        logfile_base : str, default is './logfiles/'
            Base directory for saving log files.
        checkpoint_base : str, default is './checkpoints/'
            Base directory for loading model checkpoints.
        device : str, optional
            Device to run the testing on ('cuda' or 'cpu'). If None, it chooses automatically based on availability.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.checkpoint_path = checkpoint_base + type(self.model).__name__ + 'params.pth'
        if not os.path.exists(logfile_base):
            try:
                os.makedirs(logfile_base)
            except OSError as e:
                raise RuntimeError(f"Fail to create dir '{logfile_base}': {e}")
        logfile_name = logfile_base + type(self.model).__name__
        self.logger = get_logger(__name__, logfile_name, mode='a')

    def test(
        self,
        test_dataloader: DataLoader,
        LR: int = 0.0002,
        fig: bool = True,
        visualize: bool = True,
        preprocess: preprocess_data = None
    ):
        """
        Tests the model on the test dataset and optionally visualizes the results.

        Parameters:
        ----------
        test_dataloader : torch.utils.data.DataLoader
            Dataloader for the test dataset.
        LR : int
            Learning rate (unused in testing).
        fig : bool
            Whether to plot figures (unused in this method).
        visualize : bool
            Whether to visualize the predictions and labels.
        preprocess : preprocess_data
            An instance of preprocess_data for any preprocessing needed before visualization.

        Returns:
        -------
        None
        """
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
            y_true.to(self.device)
            y_pred = self.model(x)
            print(f'y_true.shape is {torch.squeeze(y_true).shape}, y_pred.shape is {y_pred.shape}')
            predict_list.append(torch.squeeze(y_pred, -1))
            label_list.append(torch.squeeze(y_true, -1))
            y_target += np.array(torch.squeeze(y_true).detach().flatten()).tolist()
            y_forecast += np.array(y_pred.detach()).flatten().tolist()
            loss = criterion(y_pred,torch.squeeze(y_true))
            total_loss = total_loss + loss * cur_batch 


        mean_loss = total_loss / total_batch
        print(f'test_loss is {mean_loss}')
        print(getMetrics(y_target, y_forecast))
        self.logger.info(getMetrics(y_target, y_forecast))

        if visualize:
            predict_tensor = torch.cat(predict_list).detach().cpu()
            label_tensor = torch.cat(label_list).detach().cpu()
            predict_tensor = preprocess.reverse_test_norm(predict_tensor) if preprocess is not None else predict_tensor
            label_tensor = preprocess.reverse_test_norm(label_tensor) if preprocess is not None else label_tensor
            plot_line_chart(predict_tensor, label_tensor)




class ml_trainer:
    """
    A trainer class for machine learning models that are not based on neural networks.

    Attributes:
    ----------
    model : object
        An instance of a machine learning model.
    """
    def __init__(
        self,
        model,
    ):
        """
        Initializes the ml_trainer with the given model.

        Parameters:
        ----------
        model : object
            An instance of a machine learning model, which must have train and validate methods.
        """
        self.model = model

    def train(self, train_dataloader: DataLoader,):
        """
        Trains the machine learning model using the provided training data.

        Parameters:
        ----------
        train_dataloader : torch.utils.data.DataLoader
            A DataLoader for the training dataset.
        """
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
        self.model.train(x, y)

    def evaluate(self, val_dataloader: DataLoader):
        """
        Evaluates the machine learning model using the provided validation data.

        Parameters:
        ----------
        val_dataloader : torch.utils.data.DataLoader
            A DataLoader for the validation dataset.
        """
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
