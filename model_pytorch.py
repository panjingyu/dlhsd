"""Models of PyTorch version."""


import torch
import torch.fft
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

import numpy as np
from scipy.fftpack import dct


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
        # w = np.swapaxes(w, -1, -2)
        state = {'weight': torch.from_numpy(w).float()}
        self.kernel = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=128, stride=128,
            padding=0, bias=False
        )
        self.kernel.load_state_dict(state)

    def forward(self, x):
        return self.kernel(x)

def cutblock(img, block_size, block_dim):
    blockarray=[]
    for i in range(0, block_dim):
        blockarray.append([])
        for j in range(0, block_dim):
            blockarray[i].append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])
    return np.asarray(blockarray)

def dct_torch(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    if torch.__version__ < '1.8' and True:
        Vc = torch.rfft(v, 1, onesided=False)
    else:
        # FIXME: inconsistent with pytorch version < 1.8
        Vc = torch.fft.rfft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V

#calculate 2D DCT of a matrix
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def dct2_torch(img):
    return dct_torch(dct_torch(img.T, norm='ortho').T, norm='ortho')

def zigzag(arr, fealen):
    idx = 0
    feature = np.zeros(fealen, dtype=np.float)
    for i in range(fealen):
        if idx >= fealen:
            break
        elif i == 0:
            feature[0]=arr[0,0]
            idx=idx+1
        elif i%2==1:
            for j in range(0, i+1):
                if idx<fealen:
                    feature[idx]=arr[j, i-j]
                    idx=idx+1
                else:
                    break
        elif i%2==0:
            for j in range(0, i+1):
                if idx<fealen:
                    feature[idx]=arr[i-j, j]
                    idx=idx+1
                else:
                    break
    return feature

def zigzag_torch(t, fealen):
    if fealen != 32:
        t_idx = torch.empty(fealen, device=t.device, dtype=torch.long)
        idx = 0
        for i in range(fealen):
            if idx >= fealen:
                break
            elif i == 0:
                t_idx[0] = 0
                idx=idx+1
            elif i%2==1:
                for j in range(0, i+1):
                    if idx<fealen:
                        t_idx[idx] = j * t.size(0) + i - j
                        idx=idx+1
                    else:
                        break
            elif i%2==0:
                for j in range(0, i+1):
                    if idx<fealen:
                        t_idx[idx] = (i - j) * t.size(0) + j
                        idx=idx+1
                    else:
                        break
    else:
        # for fealen == 32, just use:
        t_idx = torch.tensor([0, 1, 128, 256, 129, 2, 3, 130, 257, 384, 512, 385, 258, 131, 4, 5, 132, 259, 386, 513, 640, 768, 641, 514, 387, 260, 133, 6, 7, 134, 261, 388],
                             dtype=torch.long, device=t.device)
    tt = torch.index_select(t.flatten(), dim=0, index=t_idx)
    return tt

def subfeature(imgraw, fealen):
    if fealen > len(imgraw) ** 2:
        print ('ERROR: Feature vector length exceeds block size.')
        print ('Abort.')
        quit()
    scaled =dct2(imgraw)
    feature = zigzag(scaled, fealen)
    return feature

def subfeature_torch(imgraw, fealen):
    if fealen > len(imgraw) ** 2:
        print ('ERROR: Feature vector length exceeds block size.')
        print ('Abort.')
        quit()
    imgraw = torch.from_numpy(imgraw).cuda()
    scaled = dct2_torch(imgraw)
    feature = zigzag_torch(scaled, fealen)
    return feature

# Generate DCT from image
def feature(img, block_size, block_dim, fealen):
    img = img / 255
    feaarray = np.empty(fealen*block_dim*block_dim).reshape(fealen, block_dim, block_dim)
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            if (blocked[i, j].max() == 0):
                feaarray[:,i,j] = 0
                continue
            # featemp=subfeature(blocked[i,j], fealen)
            featemp=subfeature_torch(blocked[i,j], fealen)
            feaarray[:,i,j]=featemp
    return feaarray

# Generate DCT from image
def feature_torch(img, block_size, block_dim, fealen):
    img = img / 255
    feaarray = torch.empty((fealen, block_dim, block_dim))
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            if blocked[i, j].max() == 0:
                feaarray[:,i,j] = 0
                continue
            featemp=subfeature_torch(blocked[i,j], fealen)
            feaarray[:,i,j]=featemp
    return feaarray


if __name__ == '__main__':
    def imwrite(filename, a):
        import cv2
        a = (a - a.min()) / (a.max() - a.min()) * 255
        cv2.imwrite(filename, np.array(a))
    imgraw = np.load('imgraw.npy')
    arr = torch.from_numpy(imgraw)
    a = dct2_torch(arr)
    # mydct_conv = np.zeros((32, 128, 128), dtype=float)
    # m, n = 0, 0
    # c = 0
    # for i in range(128):
    #     for j in range(128):
    #         mydct_conv[c][i][j] = np.cos(np.pi / 128 * (i + .5) * m) * np.cos(np.pi / 128 * (j + .5) * n) / (4 * 128)
    # print('mydct_conv:')
    # print(mydct_conv[c])
    # np.save('mydct_conv.npy', mydct_conv)
    dct_module = DCT128x128('mydct_conv.npy')

    # from tqdm import trange
    # for i in trange(128):
    #     for j in range(128):
    #         v = np.zeros((128,) * 2, dtype=np.uint8)
    #         v[i, j] = 255
    #         vv = feature_torch(v, 128, 1, 32)
    #         dct_module.kernel.weight[:, 0, i, j] = vv.flatten()
    # mydct_conv = dct_module.kernel.weight.detach().squeeze().numpy()
    # np.save('mydct_conv.npy', mydct_conv)
    b = dct_module(arr.unsqueeze(0).unsqueeze(0).float())
    a = feature_torch(imgraw*255, 128, 1, 32)
    print(b.flatten())
    print(a.flatten())
