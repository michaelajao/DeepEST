from torch.utils.data import DataLoader
from datasets import dl_dataset, ml_dataset
import torch

def get_dataloader(
    dataset: torch.utils.data.Dataset, 
    batch_size: int = 8, 
    shuffle=False):
    """
    Creates a PyTorch DataLoader from a given dataset.

    The function is designed to handle both custom datasets (`dl_dataset`) and
    standard PyTorch datasets. It uses the `batch_size` and `shuffle` parameters to
    configure the DataLoader accordingly.

    Parameters:
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to create a DataLoader for, which can be a custom dataset
        like `dl_dataset` or `ml_dataset` or any PyTorch compatible dataset.
    batch_size : int, optional
        The size of the batches to create from the dataset, by default 8.
    shuffle : bool, optional
        Whether to shuffle the dataset at the beginning of each epoch, by default False.

    Returns:
    -------
    torch.utils.data.DataLoader
        A DataLoader object that enables efficient batching, shuffling, and sampling of the dataset.

    Raises:
    ------
    TypeError
        If the `dataset` is not an instance of torch.utils.data.Dataset or a custom dataset class.

    Examples:
    --------
    >>> # Example usage with custom dataset
    >>> custom_dataset = dl_dataset(torch.randn(100, 10), torch.randint(0, 2, (100, 1)))
    >>> dataloader = get_dataloader(custom_dataset, batch_size=16, shuffle=True)
    >>> for data, labels in dataloader:
    >>>     # Perform training with the batch of data and labels
    >>>     pass
    """
    if isinstance(dataset, dl_dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        batch_size = dataset.get_data_shape()[0]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    return dataloader
