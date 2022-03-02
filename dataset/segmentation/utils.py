import torch
import numpy as np
from matplotlib import pyplot as plt

def make_one_hot(mask, class_count):
        expand_dim = torch.unsqueeze(mask, 0)
        print('shape : ', expand_dim.shape)
        print('type : ', expand_dim.dtype)
        one_hot = torch.zeros((expand_dim.shape[0], class_count) + expand_dim.shape[1:], dtype=torch.int64)
        print('shape : ', one_hot.shape)
        print('type : ', one_hot.dtype)
        one_hot = one_hot.scatter_(1, expand_dim.unsqueeze(1), 1.0) + 1e-6
        return one_hot

def collater(data):
    total_class = 20
    imgs = [s['image'] for s in data]
    masks = [torch.tensor(s['mask'], dtype=torch.int64) for s in data]
    batch_size = len(imgs)

    segment_mask = torch.zeros((batch_size, total_class, masks[0].shape[0], masks[0].shape[1]))
    for idx, annot in enumerate(masks):
        annot = make_one_hot(annot, total_class)
        segment_mask[idx, :, :, :] = annot

    return {'img' : torch.stack(imgs), 'label' : segment_mask}
