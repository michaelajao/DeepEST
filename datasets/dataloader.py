from torch.utils.data import DataLoader
from datasets import DlDataset


def get_dataloader(dataset, batch_size: int = 8, shuffle=False):
    if isinstance(dataset, DlDataset):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        batch_size = dataset.getDataShape()[0]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    return dataloader
