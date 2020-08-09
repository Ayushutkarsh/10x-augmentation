import os
from os.path import normpath, basename, dirname, join
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian

IMAGE_DIR = 'dataset/images'
GT_DIR = 'dataset/annotations'

def save_images(image,gt,image_name,gt_name,aug_type):

    aug_dir_image = join(dirname(IMAGE_DIR),basename(normpath(IMAGE_DIR))+'_'+aug_type)
    aug_dir_gt = join(dirname(GT_DIR),basename(normpath(GT_DIR))+'_'+aug_type)

    if not os.path.exists(aug_dir_image):
        os.makedirs(aug_dir_image)
    if not os.path.exists(aug_dir_gt):
        os.makedirs(aug_dir_gt)
    
    new_image_name = image_name.split('.')[0] +"_"+aug_type+".png"
    new_gt_name = gt_name.split('.')[0] + "_" + aug_type + ".png"
    
    new_image_path = join(aug_dir_image,new_image_name)
    new_gt_path = join(aug_dir_gt,new_gt_name)

    image.save(new_image_path)
    gt.save(new_gt_path)

def vertical_flip(image, gt):
    image = F.vflip(image)
    gt = F.vflip(gt)
    return image,gt

def horizontal_flip(image, gt):
    image = F.hflip(image)
    gt = F.hflip(gt)
    return image,gt
    

class rotation(torch.nn.Module):
    def __init__(self,degree):
        super(rotation,self).__init__()
        self.degree = degree
    def forward(self,image,gt):
        print("Using degree amount --> ",self.degree)
        image = F.rotate(image, angle=self.degree,fill=(255,))
        gt = F.rotate(gt, angle=self.degree,fill=(255,))
        return image,gt

def gaussian_noise(image,gt):
    image = image.filter(ImageFilter.BLUR)
    #Noise will be added to only the image, not gt
    return image,gt

AUGMENTATIONS = {
'vertical_flip':vertical_flip,'horizontal_flip':horizontal_flip,
'rot+5':rotation(5),
'rot+10':rotation(10),
'rot+15':rotation(15),
'rot-5':rotation(355),
'rot-10':rotation(350),
'rot-15':rotation(345),
'gaussian':gaussian_noise
}


def augmentations (image_dir,gt_dir):
    image_paths = list(map(lambda x: os.path.join(image_dir, x), os.listdir(image_dir)))
    

    for aug_type in AUGMENTATIONS:
        print(" Augmentation type ---> ", aug_type)
        for image_path in image_paths:
            image_name = image_path.split('/')[-1]
            gt_name = image_name.split('.')[0] + 'seg.png'
            gt_path = os.path.join(gt_dir,gt_name)

            image = Image.open(image_path)
            gt = Image.open(gt_path)

            image,gt = AUGMENTATIONS[aug_type](image,gt)
            save_images(image,gt,image_name,gt_name,aug_type)

def main():
    augmentations(IMAGE_DIR,GT_DIR)



if __name__ == "__main__":
    main()