# Lib Load
import os
import cv2
import glob
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from dataset.segmentation.utils import collater, visualize_input
import numpy as np
from PIL import Image

'''
Dataset 참고 사이트
https://bo-10000.tistory.com/38
'''
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class PascalVocDataset(Dataset):
    def __init__(self, path, transforms=None):
        super(PascalVocDataset, self).__init__()
        self.transforms = transforms
        self.image = glob.glob(path + '/*.jpg')
        self.train_list = dict()
        
        for image in self.image:
            self.mask = image.replace('jpg', 'png')
            self.mask = Image.open(self.mask)
            self.mask = np.array(self.mask)
            # self.mask = np.where(self.mask==255, 0, self.mask)
            self.train_list[image] = self.mask 

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_file = self.image[index]
        img = cv2.imread(img_file)
        mask = self.train_list[img_file]
        transform = self.transforms(image=img, mask=mask)
        return transform


class PascalVoc(pl.LightningDataModule):
    def __init__(self, train_path, val_path, workers, train_transforms, valid_transforms, batch_size=None):
        super(PascalVoc, self).__init__()
        self.train_path = train_path
        self.valid_path = val_path
        self.train_transforms = train_transforms
        self.val_transforms = valid_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(PascalVocDataset(self.train_path, transforms=self.train_transforms),
                    batch_size= self.batch_size,
                    shuffle=True,
                    num_workers=self.workers,
                    persistent_workers=self.workers > 0,
                    pin_memory= self.workers > 0,
                    collate_fn=collater)
    
    def val_dataloader(self):
        return DataLoader(PascalVocDataset(self.valid_path, transforms=self.val_transforms),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.workers,
                    persistent_workers=self.workers > 0,
                    pin_memory= self.workers > 0,
                    collate_fn=collater)
    

if __name__ == '__main__':
    import albumentations
    import albumentations.pytorch

    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(320, 320, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])
    print('make data')
    loader = DataLoader(PascalVocDataset(
        transforms=train_transforms, path='/mnt/test'),
        batch_size=1, shuffle=True, collate_fn=collater
    )

    for batch, sample in enumerate(loader):
        '''
        Test Augmentation - output tensor
        imgs = sample['image'] : [1, 3, 320, 320]
        labels = sample['mask'] : [1, 320, 320]
        '''
        img = sample['img']
        label = sample['label']
        visualize_input(label)
        
        
    