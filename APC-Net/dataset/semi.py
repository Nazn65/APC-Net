import h5py

from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
import h5py

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('splits/{}/val.txt'.format(name), 'r') as f:
                self.ids = f.read().splitlines()


    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]
        if img.ndim == 2:
            image = Image.fromarray(img, mode='L')
        elif img.ndim == 3 and img.shape[2] == 3:
            image = Image.fromarray(img, mode='RGB')
        elif img.ndim == 3 and img.shape[2] == 4:
            image =Image.fromarray(img,mode='RGBA')
        img = image


        if mask.ndim == 2:
            image = Image.fromarray(mask, mode='L')
        elif mask.ndim == 3 and img.shape[2] == 3:
            image = Image.fromarray(mask, mode='RGB')
        elif mask.ndim == 3 and mask.shape[2] == 4:
            image =Image.fromarray(mask,mode='RGBA')
        mask = image

        if self.mode == 'val':
            img_ori = np.array(img)
            img, mask = normalize(img, mask)
            return img, mask, id, img_ori

        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
             return normalize(img, mask)

        img_w, img_s1, img_s2, img_ori = deepcopy(img), deepcopy(img), deepcopy(img), deepcopy(img)
        img_ori = np.array(img_ori)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2, _ = normalize(img_s2, ignore_mask)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return img_w, img_s1, img_s2, img_ori, ignore_mask, cutmix_box1, cutmix_box2, id

    def __len__(self):
        return len(self.ids)
