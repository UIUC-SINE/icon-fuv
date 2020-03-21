import torch
import torch.nn as nn

class unet(nn.Module):
    def __init__(self, C=4):
        super(unet, self).__init__()
        # C = 8
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
                nn.init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DnCNN(nn.Module):
    def __init__(self, depth=3, n_channels=16, image_channels=1, use_bnorm=True, kernel_size=3):
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
                nn.init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
