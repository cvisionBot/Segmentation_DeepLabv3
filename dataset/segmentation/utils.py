import os
import cv2
import torch
import random
import numpy as np

def make_one_hot(mask, class_count, ignore_index=255): 
        expand_dim = torch.unsqueeze(mask, 0)
        expand_dim = expand_dim.type(torch.int64)
        # one_hot = torch.zeros((expand_dim.shape[0], class_count) + expand_dim.shape[1:], dtype=torch.int64)
        # one_hot = one_hot.scatter_(1, expand_dim.unsqueeze(1), 1.0)
        # one_hot = one_hot[:, 1:, :, :]
        # return one_hot
        one_hot = torch.zeros((expand_dim.shape[0], ignore_index+1) + expand_dim.shape[1:], dtype=torch.int64)
        one_hot = one_hot.scatter_(1, expand_dim.unsqueeze(1), 1.0) + 1e-6
        one_hot = torch.split(one_hot, [class_count, ignore_index+1 - class_count], dim=1)[0]
        return one_hot

def collater(data):
    total_class = 20 # for one_hot coding
    imgs = [s['image'] for s in data]
    masks = [s['mask'].clone().detach() for s in data]
    batch_size = len(imgs)

    segment_mask = torch.zeros((batch_size, total_class, masks[0].shape[0], masks[0].shape[1]))
    for idx, annot in enumerate(masks):
        annot = make_one_hot(annot, total_class)
        segment_mask[idx, :, :, :] = annot

    return {'img' : torch.stack(imgs), 'label' : segment_mask}


def parse_names(names_file):
    names_file = os.getcwd()+names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()


def gen_random_colors(names):
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for i in range(len(names))]
    return colors


def visualize_all_input(output, colors):
    # output shape : 1, 320, 320
    # class image : 3, 320, 320
    image = torch.zeros((3, 320, 320))
    output = output.expand(3, 320, 320)
    
    for c in range(1, 21):
        color = colors[c-1]
    
        idx = output[:, :, :] == c
        image[0, idx[0]] = color[0]
        image[1, idx[0]] = color[1]
        image[2, idx[0]] = color[2]

    image = torch.permute(image, (1, 2, 0))
    return image


def visualize_input(output):
    names = parse_names('/dataset/segmentation/names/pascal_voc.txt')
    colors = gen_random_colors(names)
    output = torch.argmax(output, dim=1)
    result = visualize_all_input(output, colors)
    result = result.detach().cpu().numpy()
    cv2.imwrite('./inference/result/input.png', result)
