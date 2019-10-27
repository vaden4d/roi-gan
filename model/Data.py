import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.functions import pil_loader

class Data(Dataset):

    def __init__(self, data_path, mean, std):
        '''
        Arguments:
            Path to train data folder (string): Folder with cell images
        '''

        self.data_path = data_path
        self.transform = transforms.Compose([
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])
        
        _, _, self.files = next(os.walk(self.data_path))
        self.files = list(filter(lambda x: x.endswith('.jpg'), self.files))

        self.n = len(self.files)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        path = os.path.join(self.data_path, self.files[idx])
        image = pil_loader(path)
        image = self.transform(image)

        return image
'''
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
'''

if __name__ == '__main__':

    import time
    start = time.time()
    obj = Data('celeba/', [0.5] * 3, [0.5] * 3)
    data_loader = DataLoader(obj, batch_size=64, shuffle=True)
    print('Time: ', time.time() - start)
    for i, batch in enumerate(data_loader):
        start = time.time()
        print(i)
        print(batch.size())
        print('Time: ', time.time() - start)
        if i > 5:
            break
