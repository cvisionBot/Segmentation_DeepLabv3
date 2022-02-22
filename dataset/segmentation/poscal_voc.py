# Lib Load
import os
import cv2
import glob
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

'''
Dataset 참고 사이트
https://bo-10000.tistory.com/38
'''

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return 


class PascalVocDataset(Dataset):
    cmap = voc_cmap()
    def __init__(self, path, transforms=None):
        super(PascalVocDataset, self).__init__()
        self.transforms = transforms
        self.image = glob.glob(path + '/*.jpg')
        self.train_list = dict()
        
        for image in self.image:
            self.mask = image.replace('jpg', 'png')
            self.train_list[image] = self.mask 

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_file = self.image[index]
        img = cv2.imread(img_file)
        mask = self.train_list[img_file]
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        
        return img, mask


class PoscalVoc(pl.LightningDataModule):
    def __init__(self, path, workers, transforms, batch_size=None):
        super(PoscalVoc, self).__init__()
        self.path = path
        self.transforms = transforms
        self.batch_size = batch_size
        self.workers = workers

    def make_dataloader(self):
        return DataLoader(PascalVocDataset(self.path, transforms=self.transforms),
                    batch_size = self.batch_size,
                    num_workers = self.workers,
                    persistent_workers = self.workers > 0,
                    pin_memory = self.workers > 0,)
    