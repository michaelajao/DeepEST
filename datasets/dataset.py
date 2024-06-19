from torch.utils.data import Dataset


class dl_dataset(Dataset):
    """
    A custom PyTorch Dataset class for deep learning models.

    This class serves as a wrapper around the data and label tensors to be used in
    deep learning frameworks. It implements the necessary methods for a PyTorch
    Dataset object, allowing for easy integration with PyTorch's DataLoader for
    batching and shuffling.

    Parameters:
    ----------
    data : torch.Tensor
        The data tensor to be used for the dataset.
    label : torch.Tensor
        The label tensor corresponding to the data.

    Methods:
    -------
    __init__(self, data, label)
        Initializes the dataset with data and label tensors.
    __getitem__(self, index)
        Retrieves an item from the dataset based on its index.
    __len__(self)
        Returns the length of the dataset.
    get_data_shape(self)
        Returns the shape of the data tensor.
    get_label_shape(self)
        Returns the shape of the label tensor.

    Examples:
    --------
    >>> data = torch.randn(100, 10)  # Example data tensor (100 samples, 10 features)
    >>> labels = torch.randint(0, 2, (100, 1))  # Example label tensor (100 binary labels)
    >>> dl_dataset = dl_dataset(data, labels)
    >>> print(len(dl_dataset))
    100
    >>> print(dl_dataset.get_data_shape())
    torch.Size([100, 10])
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def get_data_shape(self):
        return self.data.shape

    def get_label_shape(self):
        return self.label.shape


class ml_dataset(Dataset):
    """
    A custom PyTorch Dataset class for machine learning models.

    Similar to the dl_dataset class, this class is designed to work with machine
    learning frameworks. It provides an interface for accessing data and labels
    that can be used with machine learning libraries.

    Parameters:
    ----------
    data : torch.Tensor
        The data tensor to be used for the dataset.
    label : torch.Tensor
        The label tensor corresponding to the data.

    Methods:
    -------
    __init__(self, data, label)
        Initializes the dataset with data and label tensors.
    __getitem__(self, index)
        Retrieves an item from the dataset based on its index.
    __len__(self)
        Returns the length of the dataset.
    get_data_shape(self)
        Returns the shape of the data tensor.
    get_label_shape(self)
        Returns the shape of the label tensor.

    Examples:
    --------
    >>> data = torch.randn(100, 5)  # Example data tensor (100 samples, 5 features)
    >>> labels = torch.randint(0, 2, (100,))  # Example label tensor (100 binary labels)
    >>> ml_dataset = ml_dataset(data, labels)
    >>> print(len(ml_dataset))
    100
    >>> print(ml_dataset.get_data_shape())
    torch.Size([100, 5])
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def get_data_shape(self):
        return self.data.shape

    def get_label_shape(self):
        return self.label.shape
