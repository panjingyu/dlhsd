from model import *
import configparser as cp
import sys
import os
from datetime import datetime
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='via_config.ini')
parser.add_argument('--cure-l', type=str, default=None)
parser.add_argument('--cure-h', type=str, default=None)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--aug', action='store_true')
args = parser.parse_args()

aug = args.aug

log_file = 'train'
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
    save_path = 'models/' + 'vias' + log_file[5:] + '/'
log_file += '.log'

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(args.config)
train_path = infile.get('dir','train_path')

fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
imgdim = int(infile.get('feature','img_dim'))
val_num = int(infile.get('train','val_num'))
delta = float(infile.get('train','delta'))
validation  = int(infile.get('train','validation'))


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

import logging
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filename=log_file,
        filemode='a'
        )
log = logging.getLogger('')
sys.stdout = StreamToLogger(log,logging.INFO)
sys.stderr = StreamToLogger(log,logging.ERROR)

print(args)
print('AUG={}, CURE_L={}, CURE_H={}'.format(aug, cure_l, cure_h))
print('model dir = {}'.format(save_path))
'''
Prepare the Optimizer
'''


train_data = data(train_path, train_path+'/label.csv', preload=True)
if validation == 1:
    valid_data = data(train_path, train_path+'/label.csv', preload=True)
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


x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])              #input FT
x  = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])
if aug:
    x = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x)
    x = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), x)
y_gt   = tf.placeholder(tf.float32, shape=[None, 2]) #ground truth label
#ground truth label without bias
#reshap to NHWC
predict = forward(x)
loss   = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict)
loss   = tf.reduce_mean(loss) #calc batch loss
weights = tf.trainable_variables()
x_grad = tf.gradients(loss, x)
x_grad_r = tf.reshape(x_grad, shape=(-1, blockdim * blockdim * fealen))
n = tf.norm(x_grad_r, axis=1)
n = tf.expand_dims(n, axis=1)
z = tf.reshape(tf.sign(x_grad_r) / (n + 1e-7), shape=(-1, blockdim, blockdim, fealen))
predict_z = forward(x + cure_h * z)
loss_z = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict_z)
loss_z = tf.reduce_mean(loss_z)
loss = loss + cure_l * loss_z
#calc batch loss without bias
y      = tf.cast(tf.argmax(predict, 1), tf.int32)
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32)) #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
gs     = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)       #define global step
#lr     = tf.train.exponential_decay(0.001, gs, decay_steps=20000, decay_rate = 0.65, staircase = True) #initial learning rate and lr decay
lr_holder = tf.placeholder(tf.float32, shape=[])
lr     = 0.001 #initial learning rate and lr decay
opt    = tf.train.AdamOptimizer(lr_holder, beta1=0.9)
dr     = 0.65 #learning rate decay rate

opt    = opt.minimize(loss, gs)
maxitr = 10000
bs     = 16  #training batch size

l_step = 5    #display step
c_step = 2000 #check point step
d_step = 3000 #lr decay step
ckpt   = True

'''
Start the training
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
acc_val = []
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver(max_to_keep=400)

    for step in range(maxitr):
        #batch = get_batch(data_list, label_list, bs)
        batch = train_data.nextbatch_beta(bs, fealen)
        batch_data = batch[0]
        batch_label= batch[1]
        batch_nhs  = batch[2]
        batch_label_all_without_bias = processlabel(batch_label)
        batch_label_nhs_without_bias = processlabel(batch[3])
        nhs_loss = loss.eval(feed_dict={x_data: batch_nhs, y_gt: batch_label_nhs_without_bias})
        #delta1 = delta*step/maxitr
        delta1 = loss_to_bias(nhs_loss, 6)
        batch_label_all_with_bias = processlabel(batch_label, delta1 = delta1)
        training_loss, learning_rate, training_acc = \
            loss.eval(feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias}), \
            lr, accu.eval(feed_dict={x_data:batch_data, y_gt:batch_label_all_without_bias})
        opt.run(feed_dict={x_data: batch_data, y_gt: batch_label_all_with_bias, lr_holder: lr})
        if step % l_step == 0:
            format_str = ('%s: step %d, loss = %.2f, learning_rate = %f, training_accu = %f, nhs_loss = %.2f, bias = %.3f')
            print (format_str % (datetime.now(), step, training_loss, learning_rate, training_acc, nhs_loss, delta1))
        if step % c_step == 0 or step == maxitr-1:
            path = save_path + 'model-'+str(step)+'.ckpt'
            saver.save(sess, path)
            if validation==1:
                acc_val.append([step,accu.eval(feed_dict={x_data:valid_data.ft_buffer, y_gt:processlabel(valid_data.label_buffer)})])
                print("Validation Accuracy is %g"%acc_val[-1][1])

        #if step % d_step == 0 and step >0:
        #    lr = lr * dr
if validation==1:
    head = ['step', 'acc']
    df = pd.DataFrame(acc_val, columns = head)
    df.to_csv(os.path.join(save_path,"cv.csv"))