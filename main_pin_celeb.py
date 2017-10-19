import numpy as np
import tensorflow as tf
from common import *
from cnn_began import Encoder, Decoder, DiscriminatorCNN
from data_loader import get_loader

#################################################################
# space reserved to set parameters for the run
encoded_dimension = 10
generator_layers = 3
node_growth_per_layer = 2
num_channels = 3
data_format = 'NHWC'
adam_learning_rate = 1e-5
#################################################################
print 'Creating CelebA loader...'
scale_size = 64
batch_size = 16
loader_train = get_loader(root='../data/CelebA', scale_size=scale_size, data_format='NHWC', batch_size=batch_size, split='train', is_grayscale=False, seed=None)
loader_valid = get_loader(root='../data/CelebA', scale_size=scale_size, data_format='NHWC', batch_size=batch_size, split='validate', is_grayscale=False, seed=None)
loader_test = get_loader(root='../data/CelebA', scale_size=scale_size, data_format='NHWC', batch_size=batch_size, split='test', is_grayscale=False, seed=None)

#################################################################

# with tf.variable_scope('main'):

x = loader_train

# x_out, z, disc_vars = DiscriminatorCNN(x=x, input_channel=num_channels, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False)

z, enc_vars = Encoder(x, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False)
x_out, dec_vars = Decoder(z, input_channel=num_channels, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, final_size=scale_size)

ae_loss = tf.losses.mean_squared_error(x, x_out)

train_ae = tf.train.AdamOptimizer(adam_learning_rate).minimize(ae_loss, var_list=enc_vars + dec_vars)
# train_ae = tf.train.AdamOptimizer(adam_learning_rate).minimize(ae_loss, var_list=disc_vars)
init_op = tf.global_variables_initializer()

def prod(x):
    out = 0
    for xi in x:
        out *= int(xi)
    return out

for v in enc_vars + dec_vars:
    print '{:10d}'.format(prod(v.shape)), v.name.ljust(50), int(np.exp(sum([np.log(int(it)) for it in v.shape])))

################################################################
sess = tf.Session()
sess.run(init_op)
print 'Initialized'
z_temp = sess.run(z)
print np.array(z_temp).sum()
print np.array(sess.run(x_out)).sum()
ael = sess.run(ae_loss)
print '{:5d} {:-12.2f}'.format(-1, ael)
for q in range(10):
    _, ael = sess.run(train_ae, ae_loss)
    print '{:5d} {:-12.2f}'.format(q, ael)