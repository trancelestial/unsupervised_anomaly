from .base_dataset import BaseDataset
from torch.utils.data import DataLoader

class Dataset(BaseDataset):

    def __init__(self):
        super().__init__()

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
        DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers)

        return train_loader, test_loader