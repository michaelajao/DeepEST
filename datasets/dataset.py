from torch.utils.data import Dataset


class DlDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def getDataShape(self):
        return self.data.shape

    def getLabelShape(self):
        return self.label.shape


class MlDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def getDataShape(self):
        return self.data.shape

    def getLabelShape(self):
        return self.label.shape
