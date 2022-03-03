# Lib Load
import os
import cv2
import glob
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from dataset.segmentation.utils import collater
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
            self.mask = np.where(self.mask==255, 0, self.mask)
            self.train_list[image] = self.mask 

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_file = self.image[index]
        img = cv2.imread(img_file)
        mask = self.train_list[img_file]
        transform = self.transforms(image=img, mask=mask)
        return transform


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
                    pin_memory = self.workers > 0,
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
        transforms=train_transforms, path='/mnt/Segmentation'),
        batch_size=2, shuffle=True, collate_fn=collater
    )

    for batch, sample in enumerate(loader):
        '''
        Test Augmentation - output tensor
        imgs = sample['image'] : [1, 3, 320, 320]
        labels = sample['mask'] : [1, 320, 320]
        '''
        img = sample['img']
        print('# # # # # img # # # # #')
        print(img.shape)
        print(img)
        label = sample['label']
        print('# # # # # label # # # # #')
        print(label.shape)
        print(label)
        
        
    