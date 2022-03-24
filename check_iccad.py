from PIL import Image
import numpy as np
import torch
import cv2

from model_pytorch import DCT128x128

train_path_file = open('train.txt', 'r')
train_example_path, label = train_path_file.readline().split()
print(train_example_path, label)
example_img = Image.open(train_example_path)
print(example_img)
example = np.asarray(example_img)
print(example.shape)
print(example.dtype)

example_resized = cv2.resize(example, (2048,) * 2, interpolation=cv2.INTER_NEAREST)
example_img = example_img.resize((2048,) * 2, resample=Image.NEAREST)
example = np.asarray(example_img)
print((example - example_resized).max())
print((example == example_resized).min())

cv2.imwrite('tmp1.png', example_resized)
cv2.imwrite('tmp2.png', example)
cv2.imwrite('tmp3.png', example_resized - example)

dct = DCT128x128('dct_filter.npy')
example_resized = torch.from_numpy(np.expand_dims(example_resized, (0, 1))).float()
dct_1 = dct(example_resized)
example = torch.from_numpy(np.expand_dims(example, (0, 1))).float()
dct_2 = dct(example)
print(dct_1)
print(dct_2)
print(dct_1 - dct_2)
