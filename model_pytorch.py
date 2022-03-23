"""Models of PyTorch version."""


import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

import numpy as np


def _conv3x3(in_channels, out_channels, stride=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with 'same' padding of zeros."""
    padding = dilation  # (kernel_size // 2) + dilation - 1
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding,
                     stride=stride, dilation=dilation, bias=True)

class DlhsdNetAfterDCT(nn.Module):
    def __init__(self, block_dim, ft_length, aug=False):
        super().__init__()
        self.aug = aug
        self.random_horizontal_flip = RandomHorizontalFlip()
        self.random_vertical_flip = RandomVerticalFlip()
        self.conv1_1 = nn.Sequential(
            _conv3x3(ft_length, 16),
            nn.ReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            _conv3x3(16, 16),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Sequential(
            _conv3x3(16, 32),
            nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
            _conv3x3(32, 32),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        if block_dim == 16:
            fc1_length = 256
        else:
            raise NotImplementedError
        self.fc1 = nn.Sequential(
        nn.Linear(block_dim * block_dim * 2, fc1_length, bias=True),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc1_length, 2, bias=True)
        self.initialize_weights()

    def forward(self, x):
        if self.aug:
            x = self.random_horizontal_flip(x)
            x = self.random_vertical_flip(x)
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.pool1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.pool2(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

    def _initialize_layer(self, layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self._initialize_layer(m)


class DCT128x128(nn.Module):
    def __init__(self, filter_path) -> None:
        super().__init__()
        w = np.expand_dims(np.load(filter_path), 1)
        state = {'weight': torch.from_numpy(w).float()}
        self.kernel = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=128, stride=128,
            padding='valid', bias=False
        )
        self.kernel.load_state_dict(state)

    def forward(self, x):
        return self.kernel(x)

if __name__ == '__main__':
    import torch
    net = DlhsdNetAfterDCT(block_dim=16, ft_length=32, aug=True)
    x = torch.randn((1, 32, 16, 16))
    out = net(x)
    print(out)
    dct = DCT128x128('dct_filter.npy')
    print(dct.kernel.state_dict())
