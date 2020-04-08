# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

patch_size, stride = 64, 32
aug_times = 1
batch_size = 8


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): mask patches
        ys (Tensor): image patches
    """
    def __init__(self, xs, ys):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.ys = ys

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_y = self.ys[index]
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(mask, img):
    # get multiscale patches from a single image
    h, w = img.shape
    patches_x = []
    patches_y = []
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = mask[i:i+patch_size, j:j+patch_size]
            y = img[i:i+patch_size, j:j+patch_size]
            for k in range(0, aug_times):
                mode = np.random.randint(0, 8)
                x_aug = data_aug(x, mode=mode)
                y_aug = data_aug(y, mode=mode)
                patches_x.append(x_aug)
                patches_y.append(y_aug)
    return patches_x, patches_y

# FIXME data_dir directory 
def testgenerator(data_dir='../dataset/test'):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/im*')  # get name list of all .png files
    # initialize
    data_x = []
    data_y = []
    # generate patches
    for i in range(len(file_list)):
        ypath = file_list[i]
        num = ypath.split('.npy')[0].split('/im')[-1]
        xpath = data_dir + '/mask' + num + '.npy'
        xi = np.load(xpath)
        yi = np.load(ypath)
        # pad to the nearest multiple of 4 so that unet doesnt fail
        padr = (4 - xi.shape[0] % 4) % 4
        padc = (4 - xi.shape[1] % 4) % 4
        yi = np.pad(yi, ((0,padr),(0,padc)))
        xi = np.pad(xi, ((0,padr),(0,padc)))
        data_x.append(xi)
        data_y.append(yi)
    return data_x, data_y


def datagenerator(data_dir='../dataset/train', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/im*')  # get name list of all .png files
    # initialize
    data_x = []
    data_y = []
    # generate patches
    for i in range(len(file_list)):
        ypath = file_list[i]
        num = ypath.split('.npy')[0].split('/im')[-1]
        xpath = data_dir + '/mask' + num + '.npy'
        x, y = np.load(xpath), np.load(ypath)
        patches_x, patches_y = gen_patches(x, y)
        for patch_x, patch_y in zip(patches_x, patches_y):
            data_x.append(patch_x)
            data_y.append(patch_y)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data_x, data_y = np.array(data_x), np.array(data_y)
    data_x, data_y = np.expand_dims(data_x, axis=3), np.expand_dims(data_y, axis=3)
    discard_n = len(data_x)-len(data_x)//batch_size*batch_size  # because of batch namalization
    data_x = np.delete(data_x, range(discard_n), axis=0)
    data_y = np.delete(data_y, range(discard_n), axis=0)
    # print('^_^-training data finished-^_^')
    return data_x, data_y


if __name__ == '__main__':

    data_x, data_y = datagenerator(data_dir='../dataset/train')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')
