import torch
import numpy as np

def make_one_hot(mask, class_count):
        expand_dim = torch.unsqueeze(mask, 0)
        expand_dim = expand_dim.type(torch.int64)
        one_hot = torch.zeros((expand_dim.shape[0], class_count) + expand_dim.shape[1:], dtype=torch.int64)
        one_hot = one_hot.scatter_(1, expand_dim.unsqueeze(1), 1.0) + 1e-6
        one_hot = one_hot[:, 1:, :, :]
        return one_hot

def collater(data):
    total_class = 21 # for one_hot coding
    total_channel = 20 # for output
    imgs = [s['image'] for s in data]
    masks = [s['mask'].clone().detach() for s in data]
    batch_size = len(imgs)

    segment_mask = torch.zeros((batch_size, total_channel, masks[0].shape[0], masks[0].shape[1]))
    for idx, annot in enumerate(masks):
        annot = make_one_hot(annot, total_class)
        segment_mask[idx, :, :, :] = annot

    return {'img' : torch.stack(imgs), 'label' : segment_mask}
