from data import Data, processlabel
import numpy as np
import configparser as cp
import sys
import os
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_pytorch import DlhsdNetAfterDCT
from cure import regularizer


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/via_config.ini')
parser.add_argument('--cure-l', type=str, default=None)
parser.add_argument('--cure-h', type=str, default=None)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--log', type=None, default=None)
args = parser.parse_args()

aug = args.aug

log_file = 'train_pytorch'
log_prefix_len = len(log_file)
if aug:
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
    save_path = 'models/vias/' + log_file[log_prefix_len+1:] + '/'
if args.log is not None:
    log_file = args.log
log_file += '.log'
log_file = os.path.join('log', log_file)

tb_writer = SummaryWriter(log_dir=save_path)

'''
Initialize Path and Global Params
'''
infile = cp.ConfigParser()
infile.read(args.config)
train_path = infile.get('dir','train_path')

fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
imgdim = int(infile.get('feature','img_dim'))
val_num = int(infile.get('train','val_num'))
delta = float(infile.get('train','delta'))
validation  = int(infile.get('train','validation'))

import logging
from log_helper import StreamToLogger

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filename=log_file,
    filemode='w'
    )
log = logging.getLogger('')

'''
Prepare the Optimizer
'''

train_data = Data(train_path, train_path+'/label.csv', preload=True)
if validation == 1:
    valid_data = Data(train_path, train_path+'/label.csv', preload=True)
    hs_idx = np.where(valid_data.label_buffer==1)[0]
    valid_idx = hs_idx[:val_num]#rd.sample(hs_idx, val_num)#hs_idx[(np.random.rand(val_num)*hs_idx.size).astype(int)]
    mask = np.ones(len(valid_data.label_buffer), dtype=bool)
    mask[valid_idx]=False
    valid_data.ft_buffer = valid_data.ft_buffer[valid_idx]
    valid_data.label_buffer = valid_data.label_buffer[valid_idx]
    valid_data.reset()
    valid_data.stat()

    train_data.ft_buffer = train_data.ft_buffer[mask]
    train_data.label_buffer = train_data.label_buffer[mask]
    train_data.reset()
    train_data.stat()

sys.stdout = StreamToLogger(log,logging.INFO, sys.stdout)
sys.stderr = StreamToLogger(log,logging.ERROR, sys.stderr)

print(args)
print('AUG={}, CURE_L={}, CURE_H={}'.format(aug, cure_l, cure_h))
print('model dir = {}'.format(save_path))
os.makedirs(save_path, exist_ok=True)

lr = 1e-3
net = DlhsdNetAfterDCT(blockdim, fealen, aug=aug).to('cuda')
loss = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr, betas=[.9, .999],
                 amsgrad=True)

maxitr = 10000
bs     = 16  #training batch size

l_step = 20   #display step
c_step = 2000 #check point step
d_step = 3000 #lr decay step
ckpt   = True

def loss_to_bias(loss,  alpha, threshold=0.3):
    ''' calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
    '''
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1+torch.exp(alpha*loss))
    return bias

def to_tensor(batch):
    batch_size = batch.shape[0]
    x_data = torch.from_numpy(batch.reshape(batch_size, blockdim, blockdim, fealen))
    x_data = x_data.permute([0, 3, 1, 2])   # NHWC -> NCHW
    return x_data.float()

'''
Start the training
'''

acc_val = []
for step in range(maxitr):
    #batch = get_batch(data_list, label_list, bs)
    batch = train_data.nextbatch_beta(bs, fealen)
    batch_data = batch[0]
    batch_label= batch[1]
    batch_nhs  = batch[2]
    batch_label_all_without_bias = processlabel(batch_label)
    batch_label_nhs_without_bias = processlabel(batch[3])
    # print(batch_nhs.shape, type(batch_nhs))
    with torch.no_grad():
        x_data = to_tensor(batch_nhs).cuda()
        y_gt = torch.from_numpy(batch_label_nhs_without_bias).cuda()
        net.eval()
        net_out1 = net(x_data)
        nhs_loss = loss(net_out1, y_gt)
        # nhs_loss = loss.eval(feed_dict={x_data: batch_nhs, y_gt: batch_label_nhs_without_bias})
        delta1 = loss_to_bias(nhs_loss.detach(), alpha=6)
        batch_label_all_with_bias = processlabel(batch_label, delta1=delta1)
        # print(batch_data.shape)
        x_data = to_tensor(batch_data).cuda()
        y_gt = torch.from_numpy(batch_label_all_without_bias).cuda()
        net_out2 = net(x_data)
        training_loss = loss(net_out2, y_gt)
        learning_rate = lr
        net_predict2 = net_out2.argmax(dim=1, keepdim=True)
        correct = net_predict2.eq(y_gt.argmax(dim=1, keepdim=True)).cpu()
        training_acc = correct.sum() / correct.numel()
    y_gt = torch.from_numpy(batch_label_all_with_bias).cuda()
    net.train()
    opt.zero_grad()
    reg, norm_grad = regularizer(net, x_data.detach(), y_gt, loss, h=cure_h, lambda_=cure_l)
    net_out = net(x_data)
    loss_ = loss(net_out, y_gt)
    loss_ += reg
    loss_.backward()
    opt.step()
    tb_writer.add_scalar('loss/all', training_loss, step)
    tb_writer.add_scalar('training_acc', training_acc, step)
    tb_writer.add_scalar('loss/nhs', nhs_loss, step)
    tb_writer.add_scalar('bias', delta1, step)
    tb_writer.add_scalar('norm_grad', norm_grad, step)
    if step % l_step == 0:
        format_str = ('%s: step %d, loss = %.2f, learning_rate = %f, training_accu = %f, nhs_loss = %.2f, bias = %.3f, norm_grad = %.3f')
        print (format_str % (datetime.now(), step, training_loss, learning_rate, training_acc, nhs_loss, delta1, norm_grad))
    if step % c_step == 0 or step == maxitr-1:
        path = save_path + 'model-'+str(step)+'.pt'
        torch.save(net.state_dict(), path)
        if validation==1:
            with torch.no_grad():
                x_data = to_tensor(valid_data.ft_buffer).cuda()
                y_gt = torch.from_numpy(processlabel(valid_data.label_buffer)).cuda()
                net.eval()
                val_out = net(x_data)
                val_predict = val_out.argmax(dim=1, keepdim=True)
                correct = val_predict.eq(y_gt.argmax(dim=1, keepdim=True)).cpu()
                acc_v = correct.sum() / correct.numel()
                acc_val.append([step, acc_v])
                tb_writer.add_scalar('validation_acc', acc_v, step)
                print("Validation Accuracy is %g" % acc_v)

if validation==1:
    head = ['step', 'acc']
    df = pd.DataFrame(acc_val, columns = head)
    df.to_csv(os.path.join(save_path,"cv.csv"))

tb_writer.close()