from model import *
import configparser as cp
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
from tqdm import trange
import cv2

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
infile = cp.SafeConfigParser()
infile.read(args.config)

test_path   = infile.get('dir','test{}_path'.format(args.id))
# model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
imgdim   = int(infile.get('feature','img_dim'))
# aug  = int(infile.get('train','aug'))
aug  = 0
'''
Prepare the Input
'''
with open(test_path,"r") as testfile:
    test_list = testfile.readlines()

#test_data = data(test_path, test_path+'/label.csv', preload=False)
#maxlen = test_data.maxlen
x_data = tf.placeholder(tf.float32, shape=[None, 2048,2048, 1])              #input FT
#y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
#x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                               #ground truth label
                                     #ground truth label without bias
                            #reshap to NHWC
# predict= forward(x_data, is_training=False)
predict= forward_dct(x_data, is_training=False)



#x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                             #reshap to NHWC
#x_ud   = tf.map_fn(lambda img: tf.image.flip_up_down(img), x)                   #up down flipped
#x_lr   = tf.map_fn(lambda img: tf.image.flip_left_right(img), x)                #left right flipped
#x_lu   = tf.map_fn(lambda img: tf.image.flip_up_down(img), x_lr)                #both flipped
# predict_or = forward_dct(x_data, is_training=False)                                      #do forward
#predict_ud = forward(x_ud, is_training=False, reuse=True)
#predict_lr = forward(x_lr, is_training=False, reuse=True)
#predict_lu = forward(x_lu, is_training=False, reuse=True)
# if aug==1:
#     predict = (predict_or + predict_lr + predict_lu + predict_ud)/4.0
# else:
#     predict = predict_or

y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label

y      = tf.cast(tf.argmax(predict, 1), tf.int32)
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                               #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
'''
Start testing
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

bs = 512

#df = pd.DataFrame()
#cvs = df.read_csv(os.path.join(model_path,"cv.csv"), usecols=['step','acc'])
#step = int(cvs[np.argmax(cvs.acc.values])

#ckpt = tf.train.get_checkpoint_state(model_path)
#if ckpt and ckpt.model_checkpoint_path:
#    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#    print(ckpt_name)


with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver()
    # saver.restore(sess, os.path.join(model_path, "model-9999.ckpt"))
    saver.restore(sess, os.path.join(save_path, "model-9999.ckpt"))
    chs = 0   #correctly predicted hs
    cnhs= 0   #correctly predicted nhs
    ahs = 0   #actual hs
    anhs= 0   #actual hs
    start   = time.time()
    # bar = Bar('Detecting', max=400)
    for titr in trange(len(test_list), desc='Detecting ID {}'.format(args.id)):
        # print(test_list[titr].split()[0])
        tdata = cv2.imread(test_list[titr].split()[0],0)/255
        tdata = np.reshape(tdata, [1, 2048, 2048, 1])
        tmp_y = predict.eval(feed_dict={x_data: tdata})
        # print(tmp_y)
        if tmp_y[0,0]-tmp_y[0,1]<0:
            chs+=1
        ahs += 1

        # bar.next()
    # bar.finish()
    """
    bar = Bar('Detecting', max=test_data.maxlen//bs+1)
    for titr in range(0, test_data.maxlen//bs+1):
        if not titr == test_data.maxlen//bs:
            tbatch = test_data.nextbatch_without_balance_alpha(bs)
        else:
            tbatch = test_data.nextbatch_without_balance_alpha(test_data.maxlen-titr*bs)
        tdata = tbatch[0]
        tlabel= processlabel(tbatch[1])
        tmp_y    = y.eval(feed_dict={x_data: tdata, y_gt: tlabel})
        tmp_label= np.argmax(tlabel, axis=1)
        tmp      = tmp_label+tmp_y
        chs += sum(tmp==2)
        cnhs+= sum(tmp==0)
        ahs += sum(tmp_label)
        anhs+= sum(tmp_label==0)
        bar.next()
    bar.finish()
    """

    if not ahs ==0:
        hs_accu = 1.0*chs/ahs

    end       = time.time()

print('Hotspot Detection Accuracy is %f'%hs_accu)

print('Test Runtime is %f seconds'%(end-start))




