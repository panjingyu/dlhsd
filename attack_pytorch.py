import cv2
import numpy as np
import configparser as cp
import sys
import os

import torch

from model_pytorch import DCT128x128, DlhsdNetAfterDCT
debug = False


def get_image_from_input_id(test_file_list, id):
    '''
    return a image and its label
    '''
    img = cv2.imread(test_file_list[id].split()[0], 0)
    label = int(test_file_list[id].split()[1])
    return img, label

def _find_shapes(img_):
    shapes = [] #[upper_left_corner_location_y, upper_left_corner_location_x, y_length, x_length]
    img = np.copy(img_)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i][j] == 255 and img[i-1][j] == 0 and img[i][j-1] == 0:
                j_ = j
                while j_ < img.shape[1]-1 and img[i][j_] == 255:
                    j_ += 1
                x_length = j_ - j
                i_ = i
                while i_ < img.shape[0]-1 and img[i_][j] == 255:
                    i_ += 1
                y_length = i_ - i
                shapes.append([i,j,y_length,x_length])
                img[i:i+y_length][j:j+x_length] = 0
    return np.array(shapes)

def _find_vias(shapes_):
    shapes = np.copy(shapes_)
    squares = shapes[np.where(shapes[:,2]==shapes[:,3])]
    squares_shape = squares[:,2]
    vias_shape = np.amax(squares_shape, axis=0)
    vias_idx = np.where(squares_shape == vias_shape)
    vias = squares[vias_idx]
    srafs = np.delete(shapes, vias_idx, 0)
    return vias, srafs

def _generate_sraf_sub(srafs, save_img=False, save_dir="generate_sraf_sub/"):
    sub = []
    # black_img_ = cv2.imread("black.png", 0)
    black_img_ = np.zeros((2048, 2048), dtype=np.uint8)
    for item in srafs:
        black_img = np.copy(black_img_)
        black_img[item[0]:item[0]+item[2], item[1]:item[1]+item[3]] = -255
        sub.append(black_img)
    if save_img:
        count = 1
        for item in srafs:
            black_img = np.copy(black_img_)
            black_img[item[0]:item[0]+item[2], item[1]:item[1]+item[3]] = 255
            cv2.imwrite(save_dir+str(count)+".png", black_img)
            count += 1
    return np.array(sub, dtype=np.float32)

def _generate_sraf_add(img, vias, srafs, insert_shape=[40,90], save_img=False, save_dir="generate_sraf_add/"):
    add = []
    min_dis_to_vias = 100
    max_dis_to_vias = 500
    min_dis_to_sraf = 60
    # black_img_ = cv2.imread("black.png", 0)
    img_size = 2048
    black_img_ = np.zeros(shape=(img_size, img_size), dtype=np.uint8)
    black_img = np.copy(black_img_)
    # cv2.imwrite('black.png', black_img)
    for item in vias:
        center = [item[0]+int(item[2]/2), item[1]+int(item[3]/2)]
        black_img[max(0, center[0]-max_dis_to_vias):min(black_img.shape[0], center[0]+max_dis_to_vias), max(0, center[1]-max_dis_to_vias):min(black_img.shape[1], center[1]+max_dis_to_vias)] = 255
    for item in vias:
        center = [item[0]+int(item[2]/2), item[1]+int(item[3]/2)]
        black_img[max(0, center[0]-min_dis_to_vias):min(black_img.shape[0], center[0]+min_dis_to_vias), max(0, center[1]-min_dis_to_vias):min(black_img.shape[1], center[1]+min_dis_to_vias)] = 0
    for item in srafs:
        black_img[max(0, item[0]-min_dis_to_sraf):min(black_img.shape[0], item[0]+item[2]+min_dis_to_sraf), max(0, item[1]-min_dis_to_sraf):min(black_img.shape[1], item[1]+item[3]+min_dis_to_sraf)] = 0
    # iterate the space and add sraf one by one. srafs are generated randomly with width = 40 and length in range insert_shape
    for i in range(1, black_img.shape[0]-1):
        for j in range(1, black_img.shape[1]-1):
            if black_img[i][j] == 0:
                continue
            shape = np.random.randint(insert_shape[0], high=insert_shape[1]+1, size=2)
            shape[np.random.randint(0,high=2)] = 40
            if i+shape[0] <= black_img.shape[0] and j+shape[1] <= black_img.shape[1] and np.all(black_img[i:i+shape[0],j:j+shape[1]] == 255):
                img = np.copy(black_img_)
                img[i:i+shape[0],j:j+shape[1]] = 255
                add.append(img)
                black_img[max(0, i-min_dis_to_sraf):min(black_img.shape[0], i+shape[0]+min_dis_to_sraf), max(0, j-min_dis_to_sraf):min(black_img.shape[1], j+shape[1]+min_dis_to_sraf)] = 0
    if save_img:
        count = 1
        for item in add:
            cv2.imwrite(save_dir+str(count)+".png", item)
            count += 1
    return np.array(add, dtype=np.float32)

def generate_candidates(test_file_list, id):
    '''
    gengerate all candidates and save them
    '''
    print("Generating candidates...", end='')
    img, _ = get_image_from_input_id(test_file_list, id)
    img_path = test_file_list[id].split()[0]
    img_dir = os.path.dirname(img_path)
    img_shape_dir = os.path.join(img_dir, '_shapes')
    img_name = os.path.basename(img_path)
    img_shapes_path = os.path.join(img_shape_dir, img_name + '.npy')
    if os.path.isfile(img_shapes_path):
        shapes = np.load(img_shapes_path)
    else:
        shapes = _find_shapes(img)
        os.makedirs(img_shape_dir, exist_ok=True)
        np.save(img_shapes_path, shapes)
    vias, srafs = _find_vias(shapes)
    add = _generate_sraf_add(img, vias, srafs, save_img=False) # FIXME: slow!
    sub = _generate_sraf_sub(srafs, save_img=False)
    print(f'Done. Total candidates: {len(add)+len(sub)}')
    return np.concatenate((add, sub))

def load_candidates(sub_dir="generate_sraf_sub/", add_dir="generate_sraf_add/"):
    '''
    load candidates. call this function if candidates have been saved
    by calling gengerate_candidates() in previous run.
    '''
    print("Loading candidates...")
    X = []
    for root, dirs, files in os.walk(add_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32)
                X.append(img)
    for root, dirs, files in os.walk(sub_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32)
                X.append(img)
    print("Loading candidates done. Total candidates: "+str(len(X)))
    return np.array(X)

def generate_adversarial_image(img, X, alpha):
    img = img.astype(np.int32)
    #X = np.absolute(X).astype(np.int32)
    X = X.astype(np.int32)
    alpha = alpha.astype(np.int32)
    return (img+np.sum(X*np.expand_dims(alpha,-1),axis=0)).astype(np.uint8)

def generate_adversarial_image_torch(img, X, idx):
    tmp = X[idx].sum(dim=0).view_as(img)
    img_t = img + tmp
    if img_t.max() > 256:
        print('Warning: SRAF overlapping detected!')
        img_t.clip_(0, 255)
    return img_t

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/dct_config.ini')
parser.add_argument('--cure-l', type=str, default=None)
parser.add_argument('--cure-h', type=str, default=None)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--id', type=int, required=True, choices=[1, 2, 3, 4])
args = parser.parse_args()

log_file = 'attack_{}'.format(args.id)
pre_len = len(log_file)
if args.aug:
    log_file += '.aug'
if args.cure_l is not None and args.cure_h is not None:
    log_file += '.cureL{}H{}'.format(args.cure_l, args.cure_h)
if args.save_path is not None:
    model_path = args.save_path
else:
    model_path = 'models/vias/' + log_file[pre_len+1:] + '/'
log_file += '.log'

if args.log is not None:
    log_file = args.log + '.log'


from log_helper import StreamToLogger
import logging

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filename=log_file,
        filemode='a'
        )
log = logging.getLogger('')
sys.stdout = StreamToLogger(log,logging.INFO, sys.stdout)


'''
Initialize Path and Global Params
'''
infile = cp.ConfigParser()
infile.read(args.config)

test_path   = infile.get('dir','test_path_txt_{}'.format(args.id))
test_list = open(test_path).readlines()
# model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
blocksize   = int(infile.get('feature','block_size'))
imgdim   = int(infile.get('feature','img_dim'))
lr = float(infile.get('attack', 'attack_learning_rate'))
max_iter = int(infile.get('attack', 'max_iter'))
_max_candidates = int(infile.get('attack', 'max_candidates'))
max_perturbation = int(infile.get('attack', 'max_perturbation'))
alpha_threshold = float(infile.get('attack', 'alpha_threshold'))
attack_path = infile.get('attack', 'attack_path_txt')
img_save_dir = 'dct/attack_'+log_file[pre_len:-4]+str(_max_candidates)+'_'+str(max_iter)+'/'
os.makedirs(img_save_dir, exist_ok=True)

'''
Prepare the Input
'''
test_list_hs = [int(item.split()[1]) for item in test_list]
test_list_hs = np.array(test_list_hs)
idx = np.where(test_list_hs == 1) #total = 80152, hs = 6107


def attack_trial(alpha, X, img_t, net, target_idx):
    a = alpha.detach()
    idx = torch.zeros_like(a, dtype=torch.bool)
    for i in range(max_perturbation):
        max_idx = a.argmax()
        idx[max_idx] = True
        a[max_idx] = -float('inf')
        perturbation = X[idx].sum(dim=0).view_as(img_t)
        in_all = img_t + perturbation
        out = net(in_all)
        diff = out[:,1] - out[:,0]
        if diff <= -0.01:
            aimg = generate_adversarial_image_torch(img_t, X, idx)
            print(aimg.size())
            out = net(aimg)
            pred = out.argmax(1).item()
            if pred == 1:
                print('False attack')
                continue
            cv2.imwrite(img_save_dir+str(target_idx)+'.png', aimg.cpu().squeeze().numpy())
            print("ATTACK SUCCEED: sarfs add: "+str(len(idx)))
            print("****************")
            return 1
    return 0

'''
Start attack
'''

def attack(target_idx, net):
    # test misclassification
    img, _ = get_image_from_input_id(test_list, target_idx)
    net.eval()
    with torch.no_grad():
        img_t = torch.from_numpy(img).float().view(1, 1, *img.shape).cuda()
        out = net(img_t)
        pred = out.argmax(dim=1)
        if pred.item() == 0:
            print(f'Misclassification: ID={target_idx}')
            return -1

    print(f'start attacking on id: {target_idx}')
    max_candidates = _max_candidates
    # generate candidates
    X = generate_candidates(test_list, target_idx)
    np.random.shuffle(X)
    if max_candidates < X.shape[0]:
        X = X[:max_candidates]
    else:
        max_candidates = X.shape[0]
    X = torch.from_numpy(X).cuda()
    # TODO: fix follwoing

    alpha = torch.full((max_candidates,),
                       fill_value=1 / (1 + np.exp(-10)),
                       requires_grad=True,
                       device='cuda')
    la = torch.tensor(1e5, requires_grad=True, device='cuda')

    opt = torch.optim.RMSprop([alpha, la], lr=lr)

    '''
    first attack method by minimizing L(alpha, lambda)
    '''
    interval = 10

    net.eval()
    for iter in range(max_iter):
        # opt.run(feed_dict={input_placeholder: input_images, t_X: X})
        opt.zero_grad()
        perturbation = alpha.matmul(X.flatten(1))
        perturbation = perturbation.view_as(img_t)
        loss_1 = perturbation.norm()
        in_all = img_t + perturbation
        out = net(in_all)
        diff = out[:,1] - out[:,0]
        loss = loss_1 + la * diff
        loss.backward()
        opt.step()

        if iter % interval == 0:

            if diff < -0.0:
                interval = 5
                ret = attack_trial(alpha, X, img_t, net, target_idx)
                if ret == 1:
                    return 1

    print("max iteration reached")
    ret = attack_trial(alpha, X, img_t, net, target_idx)
    if ret == 1:
        return 1

    print("ATTACK FAIL: sraf not enough")
    print("****************")
    return 0


def main():
    dct = DCT128x128('mydct_conv.npy', div_255=True).cuda()
    net = DlhsdNetAfterDCT(16, 32, aug=False).cuda()
    ckpt_path = os.path.join(model_path, 'model-9999.pt')
    net.load_state_dict(torch.load(ckpt_path))
    dct_net = torch.nn.Sequential(dct, net)
    success = 0
    total = 0
    for id in idx[0]:
        assert id >= 0 and id < 21514, f'Invalid ID: {id}'
        ret = attack(id, dct_net)
        if ret != -1:
            total += 1
            success += ret
        print(f'success attack: [{success:3d} / {total:3d}]')


if __name__ == '__main__':
    main()