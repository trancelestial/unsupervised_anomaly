import torch
from torch.utils.data import Dataset, Subset
from src.base.dataset import Dataset as DatasetAbs
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class OldKaggle(DatasetAbs):
    def __init__(self, normal_class=0):
        super().__init__()

        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(2))
        self.outlier_classes.remove(normal_class)

        scaler = StandardScaler()

        train_dataset = _KaggleDataset(path='../data/creditcard_train.csv', scaler=scaler, scaler_fit=True, only_normal=True)
        train_x = train_dataset

        # normal_idx = np.argwhere(train_dataset.labels == normal_class)
        # self.train_set = Subset(train_x, normal_idx)
        self.train_set = train_x

        test_dataset = _KaggleDataset(path='../data/creditcard_test.csv', scaler=scaler)
        self.test_set = test_dataset

class _KaggleDataset(Dataset):
    def __init__(self, path, scaler=None, scaler_fit=False, only_normal=False):
        data = pd.read_csv(path)

        data_x = data.drop(['Time', 'Amount', 'Class'], axis=1)
        # print('->',data_x.shape)
        labels = data['Class']

        if scaler != None:
            if scaler_fit:
                scaler.fit(data_x)
            data_x = scaler.transform(data_x)

        if only_normal:
            normal_idx = np.argwhere(labels == 0)
            data_x = data_x[normal_idx]
            labels = np.zeros((len(normal_idx),))
            data_x = np.array(data_x).squeeze(axis=1).tolist()

        # print(np.array(data_x).shape)


        self.data = torch.tensor(data_x).double()
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], index
