import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Data(Dataset):

    def __init__(self, data_path=None):
        '''
        Arguments:
            Path to train data folder (string): Folder with cell images
        '''

        self.data_path = data_path
        self.mean, self.std = [0.5] * 3, [0.5] * 3
        if self.data_path:
            self.data = self.download(self.data_path)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.n = len(value)

    def download(self, path_to_data):
        """Load data from image folder"""

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.data = ImageFolder(root=path_to_data,
                                transform=transforms.Compose([
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    normalize
                                    ]))

        return self.data

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx][0]

class Splitter():

    def __init__(self, obj):

        self.obj = obj
        self.train_size = 1
        self.test_size = 0
        self.valid_size = 0

    def train_test_split(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.train_size = 1 - test_size

        self.train_max_index = int(self.train_size * self.obj.n)

        metadata = self.obj.metadata.copy()
        metadata = metadata.sample(frac=1, random_state=random_state)

        train_metadata = metadata.iloc[:self.train_max_index]
        test_metadata = metadata.iloc[self.train_max_index:]

        train_obj = LabeledData()
        test_obj = LabeledData()

        train_obj.metadata = train_metadata
        test_obj.metadata = test_metadata
        return train_obj, test_obj


    def train_validation_test_split(self, valid_size=0.2, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.valid_size = valid_size
        self.train_size = 1 - test_size - valid_size

        self.train_max_index = int(self.train_size * self.obj.n)
        self.test_max_index = int((self.train_size + self.test_size) * self.obj.n)

        metadata = self.obj.metadata.copy()
        metadata = metadata.sample(frac=1, random_state=random_state)

        train_metadata = metadata.iloc[:self.train_max_index]
        test_metadata = metadata.iloc[self.train_max_index:self.test_max_index]
        valid_metadata = metadata.iloc[self.test_max_index:]

        train_obj = LabeledData()
        test_obj = LabeledData()
        valid_obj = LabeledData()

        train_obj.metadata = train_metadata
        test_obj.metadata = test_metadata
        valid_obj.metadata = valid_metadata

        return train_obj, test_obj, valid_obj


if __name__ == '__main__':

    import time
    start = time.time()
    obj = Data()
    obj.download('/Users/vaden4d/Documents/ds/roi-gan/cars_train')
    train_data = obj.data
    data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    print(train_data[0][0].size())
    print('Time: ', time.time() - start)
    for i, (imgs, _) in enumerate(data_loader):
        print(imgs)
        print(imgs.size())
        if i > 2:
            break
