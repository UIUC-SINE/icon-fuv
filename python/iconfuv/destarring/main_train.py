# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

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

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='unet', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--depth', default=5, type=int, help='depth')
parser.add_argument('--channels', default=16, type=int, help='depth')
parser.add_argument('--train_data', default='../dataset/train', type=str, help='path of train data')
parser.add_argument('--epoch', default=1000, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
depth = args.depth
reg_param = 0.01

save_dir = os.path.join('models/{}_ch8_l1'.format(args.model))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
    #Encoder
        C = 8
        self.conv1 = nn.Conv2d(1, C, (3, 3), (1, 1), (1, 1))
        self.batch1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, 2*C, (3, 3), (1, 1), (1, 1))
        self.batch2 = nn.BatchNorm2d(2*C)
        self.conv3 = nn.Conv2d(2*C, 4*C, (3, 3), (1, 1), (1, 1))
        self.batch3 = nn.BatchNorm2d(4*C)
        self.conv4 = nn.Conv2d(4*C, 2*C, (3, 3), (1, 1), (1, 1))
        self.batch4 = nn.BatchNorm2d(2*C)
        self.conv5 = nn.Conv2d(2*C, C, (3, 3), (1, 1), (1, 1))
        self.batch5 = nn.BatchNorm2d(C)
        self.conv6 = nn.Conv2d(C, 1, (3, 3), (1, 1), (1, 1))
        self.batch6 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.maxpoll = nn.MaxPool2d(2,ceil_mode=False)
        self.dropout = nn.Dropout(0.5)
        self.up1 = nn.ConvTranspose2d(4*C, 2*C, (4,4),stride=(2,2),padding=(1,1))
        self.up2 = nn.ConvTranspose2d(2*C, C, (4,4),stride=(2,2),padding=(1,1))
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        x1 = self.batch1(self.relu(self.conv1(x)))
        x2 = self.maxpoll(x1)
        x3 = self.batch2(self.relu(self.conv2(x2)))
        x4 = self.maxpoll(x3)
        x5 = self.batch3(self.relu(self.conv3(x4)))

        x6 = self.up1(x5)
        x7 = torch.cat([x6, x3],1)
        x8 = self.batch4(self.relu(self.conv4(x7)))
        x9 = self.up2(x8)
        x10 = torch.cat([x9, x1],1)
        x11 = self.batch5(self.relu(self.conv5(x10)))
        x12 = self.sigmoid(self.batch6(self.conv6(x11)))

        return x12

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class DnCNN(nn.Module):
    def __init__(self, depth=depth, n_channels=args.channels, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.Sigmoid())
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

class binary_cross_entropy(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(binary_cross_entropy, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.BCELoss(input, target, reduction='sum')


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    # model = DnCNN()
    model = unet()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
    model.train()
    criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = binary_cross_entropy()
    # criterion = nn.BCELoss(reduction='sum')
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    if cuda:
        model = model.cuda()
         # device_ids = [0]
         # model = nn.DataParallel(model, device_ids=device_ids).cuda()
         # criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    xs_val, ys_val = dg.testgenerator()

    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs, ys = dg.datagenerator(data_dir=args.train_data)
        # Normalize patches
        ys -= np.mean(ys, axis=(1,2))[:,np.newaxis, np.newaxis]
        ys /= np.std(ys, axis=(1,2))[:,np.newaxis, np.newaxis]

        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        ys = torch.from_numpy(ys.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, ys)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            pre = model(batch_yx[0])
            loss = criterion(pre, batch_yx[1]) + reg_param * torch.norm(pre, p=1)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        loss_val = 0
        for i in range(len(xs_val)):
            yi = torch.from_numpy(ys_val[i][np.newaxis, np.newaxis])
            xi = torch.from_numpy(xs_val[i][np.newaxis, np.newaxis])
            loss_val += criterion(model(yi), xi) + reg_param * torch.norm(pre, p=1)
        loss_val /= len(xs_val)

        log('epoch = %4d , loss = %4.4f , loss_val = %4.4f, time = %4.2f s' % (epoch+1, epoch_loss/n_count,loss_val, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
