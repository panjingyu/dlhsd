import argparse
import copy
import numpy as np
import os
import pandas as pd
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import get_timestamp, to_tensor
from utils.data import DataGds, processlabel
from utils.model import DlhsdNetAfterDCT
from utils.cure import regularizer
from utils.log_helper import make_logger, get_log_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cure-l', type=float, default=0.)
    parser.add_argument('--cure-h', type=float, default=0.)
    parser.add_argument('--saved', type=str, default='./saved/')
    args = parser.parse_args()
    return args

def loss_to_bias(loss, alpha, threshold=0.3):
    ''' calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
    '''
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1 + torch.exp(alpha * loss))
    return bias

def main(args):
    log_id = get_log_id(args)
    log_id += f'@{get_timestamp()}'
    saved_path = os.path.join(args.saved, log_id)
    os.makedirs(saved_path)
    log_path = os.path.join(saved_path, 'train.log')
    logger = make_logger(log_path, log_stdin=True)
    tb_writer = SummaryWriter(log_dir=saved_path)

    print(args)

    '''
    Prepare the Optimizer
    '''

    lr = args.lr
    val_num = 100
    blockdim = 16
    fealen = 32
    aug = True
    cure_h, cure_l = args.cure_h, args.cure_l
    train_data = DataGds('./data/train/')
    valid_data = copy.deepcopy(train_data)

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

    net = DlhsdNetAfterDCT(blockdim, fealen, aug=aug).to('cuda')
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr, betas=[.9, .999],
                     amsgrad=True)

    maxitr = 40000
    bs     = 16  #training batch size

    l_step = 20   #display step
    c_step = 2000 #check point step


    '''
    Start the training
    '''

    acc_val = []
    for step in range(maxitr):
        batch = train_data.nextbatch_beta(bs, fealen)
        batch_data = batch[0]
        batch_label= batch[1]
        batch_nhs  = batch[2]
        batch_label_all_without_bias = processlabel(batch_label)
        batch_label_nhs_without_bias = processlabel(batch[3])
        net.aug = aug
        with torch.no_grad():
            x_data = to_tensor(batch_nhs, blockdim, fealen).cuda()
            y_gt = torch.from_numpy(batch_label_nhs_without_bias).cuda()
            net.eval()
            net_out1 = net(x_data)
            nhs_loss = loss(net_out1, y_gt)
            delta1 = loss_to_bias(nhs_loss.detach(), alpha=6)
            batch_label_all_with_bias = processlabel(batch_label, delta1=delta1)
            x_data = to_tensor(batch_data, blockdim, fealen).cuda()
            y_gt = torch.from_numpy(batch_label_all_without_bias).cuda()
            net_out2 = net(x_data)
            training_loss = loss(net_out2, y_gt)
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
            format_str = ('%s: step %d, loss = %.2f, training_accu = %f, nhs_loss = %.2f, bias = %.3f, norm_grad = %.3f')
            print (format_str % (datetime.now(), step, training_loss, training_acc, nhs_loss, delta1, norm_grad))
            tb_writer.flush()
        if step % c_step == 0 or step == maxitr-1:
            path = os.path.join(saved_path, f'model-{step}.pt')
            torch.save(net.state_dict(), path)
            # Validation
            with torch.no_grad():
                x_data = to_tensor(valid_data.ft_buffer, blockdim, fealen).cuda()
                y_gt = torch.from_numpy(processlabel(valid_data.label_buffer)).cuda()
                net.eval()
                net.aug = False
                val_out = net(x_data)
                val_predict = val_out.argmax(dim=1, keepdim=True)
                correct = val_predict.eq(y_gt.argmax(dim=1, keepdim=True)).cpu()
                acc_v = correct.sum() / correct.numel()
                acc_val.append([step, acc_v])
                tb_writer.add_scalar('validation_acc', acc_v, step)
                print("Validation Accuracy is %g" % acc_v)

    # Validation
    head = ['step', 'acc']
    df = pd.DataFrame(acc_val, columns = head)
    df.to_csv(os.path.join(saved_path,"cv.csv"))

    tb_writer.close()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    main(args)