import logging, glob, os
from os import listdir
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from multiprocessing import Pool


class BasicDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform

        if fold == 'train':
            self.train = True
            self.task_dir = os.path.join(data_dir, 'train')
        elif fold == 'val':
            self.val = True
            self.task_dir = os.path.join(data_dir, 'val')

        self.files = glob.glob(self.task_dir+'/*')
        self.files.sort()
        for file in self.files:
            data = np.load(file, allow_pickle=True).item()
            im_starry = data['image_stars']
            im_clean = data['image_clean']
            im_starry = np.where(np.isnan(im_starry), im_clean, im_starry)
            self.data.append((im_starry[np.newaxis], im_clean[np.newaxis]))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im_starry, im_clean = self.data[idx]

        if self.transform is not None:
            im_starry = self.transform(im_starry)

        if self.target_transform is not None:
            im_clean = self.target_transform(im_clean)

        return im_starry, im_clean

def data_rewrite(file):
    name = file.split('/')[-1]
    out_name = 'data/train2/{}'.format(name)
    data = np.load(file, allow_pickle=True).item()
    im_starry = data['image_stars']
    im_clean = data['image_ori']
    im_starry = np.where(np.isnan(im_starry), im_clean, im_starry)
    array = np.stack((im_starry,im_clean))
    np.save(out_name, array)

def data_rewriter_par():
    files = glob.glob('data/train/*')
    pool = Pool()
    pool.map(data_rewrite, files)
