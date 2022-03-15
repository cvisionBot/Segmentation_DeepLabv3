import os
import cv2
import copy
import random
from sklearn import preprocessing
import torch
import argparse
import numpy as np

from models.segmentor.deeplabv3 import DeepLabv3, DeepLabv3Plus
from module.segmentator import Segmentor
from utils.module_select import get_model
from utils.yaml_helper import get_train_configs
from utils.utility import preprocess_input

from torchsummary import summary


def parse_names(names_file):
    names_file = os.getcwd()+names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()


def gen_random_colors(names):
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for i in range(len(names))]
    return colors


def decode_output(output, thresh_hold=0.0):
    # output shape : [1, 20, 320, 320]
    num_class = output.size()[1]
    for i in range(num_class):
        output[:, i, :, :] = torch.where(output[:, i, :, :] > thresh_hold, 1, 0)
    return output

def visualize_segmentation(output, idx, color, cfg):
    # output shape : 1, 320, 320
    # class image : 3, 320, 320
    _, h, w = output.size()
    image_buff = torch.zeros((cfg['in_channels'], h, w)).to('cuda')
    output = output.expand(3, 320, 320)
    class_image = output + image_buff
    class_image[0, :, :] = class_image[0, :, :] * color[0]
    class_image[1, :, :] = class_image[1, :, :] * color[1]
    class_image[2, :, :] = class_image[2, :, :] * color[2]
    class_image = torch.permute(class_image, (1, 2, 0))
    return class_image

def visualize_all_segmentation(output, names, colors, cfg):
    # output shape : 1, 320, 320
    image = torch.zeros((cfg['in_channels'], cfg['input_size'], cfg['input_size']))
    output = output.expand(3, 320, 320)
    
    for c in range(1, 21):
        color = colors[c-1]
    
        idx = output[:, :, :] == c
        image[0, idx[0]] = color[0]
        image[1, idx[0]] = color[1]
        image[2, idx[0]] = color[2]

    image = torch.permute(image, (1, 2, 0))
    return image


def main(cfg, image_name, save):
    names = parse_names(cfg['names'])
    colors = gen_random_colors(names)

    # Preprocess Image
    image = cv2.imread(image_name)
    image = cv2.resize(image, (cfg['input_size'], cfg['input_size']))
    image_inp = preprocess_input(image)
    image_inp = image_inp.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp = image_inp.cuda()
    
    # Load trained model
    backbone = get_model(cfg['backbone'])
    model = DeepLabv3(Backbone=backbone, num_classes=cfg['classes'], in_channels=cfg['in_channels'])
    if torch.cuda.is_available:
        model = model.to('cuda')
    
    model_module = Segmentor.load_from_checkpoint(
        '/home/insig/Segmentation/saved/ResNet_DeepLabv3_Pascal/version_2/checkpoints/last.ckpt',
        model=model
    )
    model_module.eval()
    # summary(model_module, input_size=(cfg['in_channels'], cfg['input_size'], cfg['input_size']))

    output = model_module(image_inp)
    output = decode_output(output)
    # output = torch.argmax(output, dim=1)

    # result = visualize_all_segmentation(output, names, colors, cfg)
    # result = result.detach().cpu().numpy()
    # cv2.imwrite('./inference/result/inferece.png', result)

    
    #Decode Class Split - except argmax

    for i in range(output.size()[1]):
        color = colors[i]
        class_image = visualize_segmentation(output[:, i, :, :], i, color, cfg)
        class_image = class_image.detach().cpu().numpy()
        cv2.imwrite('./inference/result/'+str(i)+'_class.png', class_image)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Train config file')
    parser.add_argument('--save', action='store_true', help='Train config file')
    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    
    main(cfg, './inference/sample/test.jpg', args.save)