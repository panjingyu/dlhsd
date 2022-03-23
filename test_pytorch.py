import configparser as cp
import sys
import time
import os
from tqdm import trange
import cv2

import numpy as np
import torch
from model_pytorch import DlhsdNetAfterDCT, DCT128x128

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/via_config.ini')
parser.add_argument('--cure-l', type=str, default=None)
parser.add_argument('--cure-h', type=str, default=None)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--id', type=int, required=True, choices=[1, 2, 3, 4])
args = parser.parse_args()

log_file = 'test-id{}'.format(args.id)
pre_len = len(log_file)
if args.aug:
    log_file += '.aug'
if args.cure_l is not None and args.cure_h is not None:
    log_file += '.cureL{}H{}'.format(args.cure_l, args.cure_h)
    cure_h = float(args.cure_h)
    cure_l = float(args.cure_l)
else:
    cure_h, cure_l = 0, 0

if args.save_path is not None:
    save_path = args.save_path
else:
    save_path = 'models/vias/' + log_file[pre_len+1:] + '/'
log_file += '.log'
log_file = os.path.join('log', log_file)

import logging
from log_helper import StreamToLogger

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filename=log_file,
    filemode='a'
    )
log = logging.getLogger('')
sys.stdout = StreamToLogger(log,logging.INFO, sys.stdout)

print(args)
print('AUG={}, CURE_L={}, CURE_H={}'.format(args.aug, cure_l, cure_h))
print('model dir = {}'.format(save_path))

'''
Initialize Path and Global Params
'''
infile = cp.ConfigParser()
infile.read(args.config)

test_path   = infile.get('dir','test{}_path'.format(args.id))
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
imgdim   = int(infile.get('feature','img_dim'))
aug  = 0
'''
Prepare the Input
'''
with open(test_path,"r") as testfile:
    test_list = testfile.readlines()


bs = 512

net = DlhsdNetAfterDCT(blockdim, fealen, aug=False).cuda()
net.load_state_dict(torch.load(os.path.join(save_path, 'model-9999.pt')))
dct = DCT128x128('dct_filter.npy').cuda()
dct.eval()


chs = 0   #correctly predicted hs
cnhs= 0   #correctly predicted nhs
ahs = 0   #actual hs
anhs= 0   #actual hs
start   = time.time()
for titr in trange(len(test_list), desc='Detecting ID {}'.format(args.id)):
    tdata = cv2.imread(test_list[titr].split()[0],0)/255
    tdata = np.reshape(tdata, [1, 1, 2048, 2048])
    tdata = torch.from_numpy(tdata).float().cuda()
    x_data = dct(tdata)
    out = net(x_data)
    predict = out.argmax(dim=1, keepdim=True)
    chs += predict.item()
    ahs += 1

if not ahs ==0:
    hs_accu = 1.0*chs/ahs

end       = time.time()

print('Hotspot Detection Accuracy is %f'%hs_accu)

print('Test Runtime is %f seconds'%(end-start))
