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
        if mean == None or std == None:
            self.transform = transforms.Compose([
                                        transforms.Resize((64, 64)),
                                        transforms.ToTensor()
                                        ])
        else:
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

if __name__ == '__main__':

    import time
    start = time.time()
    obj = Data('celeba/', None, None)

    from utils.compute_statistics import compute_mean_std
    mean, std = compute_mean_std(obj)
    print(mean, std)
    '''
    data_loader = DataLoader(obj, batch_size=64, shuffle=True)
    print('Time: ', time.time() - start)
    for i, batch in enumerate(data_loader):
        start = time.time()
        print(i)
        print(batch.size())
        print('Time: ', time.time() - start)
        if i > 5:
            break
    '''
