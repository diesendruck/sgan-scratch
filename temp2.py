# This version implements GAN

from cnn_began import Encoder, Decoder
import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image
from time import *

#######################################################################
# Run setup
#######################################################################
# Data loading params
image_size = 64  # Edge-size of the square images for input and output, scale_size = 64 default
image_channels = 3  # How many channels in the image (don't change this, 3 is hardcoded in places)
data_dir = '../data/CelebA/splits/train/'  # Where to find the training images
oos_dir = '../data/CelebA/splits/validate/'  # Where to find the validation images
data_format = 'NHWC'  # How the dimensions of the data are ordered (only NHWC is supported right now)
scale_size = image_size  # See above
label_file = '../data/CelebA/list_attr_celeba.txt'
label_choices = [4, 15, 20, 22, 24]  # Which labels to use (will print your choices once labels are loaded)
n_labels = len(label_choices)  # How many label choices you made
# CNN params
dimension_g = 16  # Dimension of the generators' inputs
encoded_dimension = 64 # Dimension of the encoded layer, znum = 256 by default
cnn_layers = 6  # How many layers in each convolutional layer
node_growth_per_layer = 4 # Linear rate of growth between CNN layers, hidden_num = 128 default

# Training params
batch_size_x = 64  # Nubmer of samples in each training cycle, default 16
batch_size_g = 64  # Number of generated samples, default 16
adam_beta_1 = 0.5   # Anti-decay rate of first moment in ADAM optimizer
adam_beta_2 = 0.999 # Anti-decay rate of second moment in ADAM optimizer
learning_rate_initial = 0.00008 # 1e-4 # Base learning rate for the ADAM optimizers; may be decreased over time, default 0.00008
learning_rate_decay = 1000.  # How many steps to reduce the learning rate by a factor of e
learning_rate_minimum = 0.00008 # 1e-4  # Floor for the learning rate
training_steps = 125000  # Steps of the ADAM optimizers
print_interval = 10  # How often to print a line of output
graph_interval = 100  # How often to output the graphics set

# Penalty params
PIN_penalty_mode = ['MSE', 'CE'][1]
gamma_target = 1.0  # Target ration of L(G(y))/L(x_trn), default 1
lambda_pin_value = 0.  # Scaling factor of penalty for label mismatch
kappa = 0.  # Initial value of kappa for BEGAN
kappa_learning_rate = 0.0005  # Learning rate for kappa
lambda_k_initial = 0.001 # Initial value for lambda_k

# Supervisor params
logdir = 'log/began_celeb/v0_5_giant/'
imgdir = 'img/began_celeb/v0_5/'

# Make sure imgdir exists
for idx in range(len(imgdir.split('/')[:-1])):
    if not os.path.exists('/'.join(imgdir.split('/')[:idx+1])):
        os.mkdir('/'.join(imgdir.split('/')[:idx+1]))

#######################################################################
# Functions
#######################################################################
def now():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


def load_labels(label_file, label_choices=None):
    lines = open(label_file, 'r').readlines()
    label_names = lines[1].strip().split(' ')
    if label_choices is None:
        label_choices = range(len(label_names))
    n_labels = len(label_choices)
    label_names = [label_names[choice] for choice in label_choices]
    'Labels:'
    for ln in xrange(n_labels):
        print '  {:2d} {:2d} {}'.format(ln, label_choices[ln], label_names[ln])
    
    file_to_idx = {lines[i][:10]: i for i in xrange(2, len(lines))}
    labels = np.array([[int(it) for it in line.strip().split(' ')[1:] if it != ''] for line in lines[2:]])[:,
             label_choices]
    return labels, label_names, file_to_idx


def img_and_lbl_queue_setup(filenames, labels):
    labels_tensor = tf.constant(labels, dtype=tf.float32)
    
    filenames_tensor = tf.constant(filenames)
    fnq = tf.RandomShuffleQueue(capacity=200, min_after_dequeue=100, dtypes=tf.string)
    fnq_enq_op = fnq.enqueue_many(filenames_tensor)
    filename = fnq.dequeue()
    
    reader = tf.WholeFileReader()
    flnm, data = reader.read(fnq)
    image = tf_decode(data, channels=3)
    image.set_shape([178, 218, image_channels])
    image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
    image = tf.to_float(image)
    
    image_index = [tf.cast(tf.string_to_number(tf.substr(flnm, len(data_dir), 6)) - 1, tf.int32)]
    image_labels = tf.reshape(tf.gather(labels_tensor, indices=image_index, axis=0), [n_labels])
    imq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=[tf.float32, tf.float32],
                                shapes=[[128, 128, image_channels], [n_labels]])
    imq_enq_op = imq.enqueue([image, image_labels])
    imgs, img_lbls = imq.dequeue_many(batch_size_x)
    imgs = tf.image.resize_nearest_neighbor(imgs, size=[image_size, image_size])
    imgs = tf.subtract(1., tf.divide(imgs, 255.5))
    qr_f = tf.train.QueueRunner(fnq, [fnq_enq_op] * 3)
    qr_i = tf.train.QueueRunner(imq, [imq_enq_op] * 3)
    return imgs, img_lbls, qr_f, qr_i


def load_practice_images(data_dir, n_images, labels):
    filenames = os.listdir(data_dir)[:n_images]
    img_shape = Image.open(data_dir + filenames[0]).size
    imgs = np.zeros([len(filenames), img_shape[1], img_shape[0], 3], np.float32)
    for i in xrange(n_images):
        tmp = Image.open(data_dir + filenames[i])
        imgs[i, :, :, :] = np.array(tmp.getdata()).reshape(tmp.size[1], tmp.size[0], 3)
    lbls = labels[[int(filename[:6]) - 1 for filename in filenames], :]
    return imgs, lbls, filenames


def preprocess(input_imgs, image_size):
    imgs = tf.to_float(input_imgs)
    imgs = 1. - imgs / 255.5
    imgs = tf.image.crop_to_bounding_box(imgs, 50, 25, 128, 128)
    imgs = tf.to_float(imgs)
    imgs = tf.image.resize_nearest_neighbor(imgs, size=[image_size, image_size])
    return imgs


def pin_cnn(input, lbls=None, n_labels=None, reuse=False, encoded_dimension=16, cnn_layers=4,
            node_growth_per_layer=4, data_format='NHWC', image_channels=3):
    z, enc_vars = Encoder(input, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                          data_format=data_format, reuse=reuse)
    # if lbls is None:
    #     z_pin = tf.concat([tf.tanh(z[:, :n_labels]), z[:, n_labels:]], 1)
    # else:
    #     z_pin = tf.concat([tf.tanh(z[:, :n_labels]) * (1. - tf.abs(lbls)) + lbls * tf.abs(lbls), z[:, n_labels:]], 1)
    output, dec_vars = Decoder(z, input_channel=image_channels, repeat_num=cnn_layers,
                               hidden_num=node_growth_per_layer, data_format=data_format, reuse=reuse,
                               final_size=scale_size)
    output = tf.maximum(tf.minimum(output, 1.), 0.)
    return z, output, enc_vars, dec_vars


#######################################################################
# Load filenames and labels
#######################################################################
filenames = [data_dir + it for it in os.listdir(data_dir)]
ext = filenames[0][-3:]
if ext == "jpg":
    tf_decode = tf.image.decode_jpeg
elif ext == "png":
    tf_decode = tf.image.decode_png

labels, label_names, file_to_idx = load_labels(label_file, label_choices)

#######################################################################
# starting the managed session routine
#######################################################################
with tf.Graph().as_default():
    #######################################################################
    # model definition
    #######################################################################
    
    # The model for the real-data training samples
    imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(filenames, labels)
    x_trn = imgs
    z_trn, x_out_trn, enc_vars, dec_vars = pin_cnn(input=x_trn, lbls=img_lbls if lambda_pin_value > 0. else None,
                                                   n_labels=n_labels, reuse=False,
                                                   encoded_dimension=encoded_dimension, cnn_layers=cnn_layers,
                                                   node_growth_per_layer=node_growth_per_layer, data_format=data_format,
                                                   image_channels=image_channels)
    
    # The model for the generated_data training samples
    y = tf.random_normal([batch_size_g, dimension_g], dtype=tf.float32)
    x_gen, gen_vars = Decoder(y, input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                              data_format=data_format, reuse=False, final_size=scale_size, var_scope='Gen')
    x_gen = tf.maximum(tf.minimum(x_gen, 1.), 0.)
    z_gen, x_out_gen, _, _ = pin_cnn(input=x_gen, n_labels=n_labels, reuse=True, encoded_dimension=encoded_dimension,
                                     cnn_layers=cnn_layers, node_growth_per_layer=node_growth_per_layer,
                                     data_format=data_format, image_channels=image_channels)
    
    # Define the losses
    loss_x = tf.losses.mean_squared_error(x_trn, x_out_trn)
    loss_g = tf.losses.mean_squared_error(x_gen, x_out_gen)
    zp = z_trn[:, :n_labels]
    if PIN_penalty_mode == 'CE':
        loss_z = -tf.reduce_sum((zp * (img_lbls + 1.) / 2. - tf.log(1. + tf.exp(zp))) * tf.abs(img_lbls))
    elif PIN_penalty_mode == 'MSE':
        loss_z = tf.losses.mean_squared_error(img_lbls, tf.tanh(zp), weights=tf.abs(img_lbls))
    else:
        raise 'Need a valid penalty mode the PIN'
    lambda_pin = tf.placeholder(tf.float32, [])
    lambda_ae = tf.placeholder(tf.float32, [])
    loss_gan = loss_x - lambda_ae * loss_g
    loss_gen = loss_g
    loss_pin = loss_gan + lambda_pin * loss_z
    
    # Set up the optimizers
    adam_learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])
    train_gan = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph, beta1=adam_beta_1, beta2=adam_beta_2).minimize(loss_pin, var_list=enc_vars + dec_vars)
    train_gen = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph, beta1=adam_beta_1, beta2=adam_beta_2).minimize(loss_gen, var_list=gen_vars)
    train_cla = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph, beta1=adam_beta_1, beta2=adam_beta_2).minimize(loss_z, var_list=enc_vars)
    
    # Set up the initializer (NOTE: Keep this after the optimizers, which have parameters to be initialized.)
    init_op = tf.global_variables_initializer()
    
    #######################################################################
    # Extra output nodes for graphics
    #######################################################################
    # Run the model on a consistent selection of in-sample pictures
    img_ins, lbls_ins, fs_ins = load_practice_images(data_dir, n_images=8, labels=labels)
    x_ins = preprocess(img_ins, image_size=image_size)
    z_ins, x_out_ins, _, _ = pin_cnn(input=x_ins, n_labels=n_labels, reuse=True,
                                     encoded_dimension=encoded_dimension, cnn_layers=cnn_layers,
                                     node_growth_per_layer=node_growth_per_layer, data_format=data_format,
                                     image_channels=image_channels)
    
    # Run the model on a consistent selection of out-of-sample pictures
    img_oos, lbls_oos, fs_oos = load_practice_images(oos_dir, n_images=8, labels=labels)
    x_oos = preprocess(img_oos, image_size=image_size)
    z_oos, x_out_oos, _, _ = pin_cnn(input=x_oos, n_labels=n_labels, reuse=True,
                                     encoded_dimension=encoded_dimension, cnn_layers=cnn_layers,
                                     node_growth_per_layer=node_growth_per_layer, data_format=data_format,
                                     image_channels=image_channels)
    
    # Run the model on a consistent selection of out-of-sample pictures
    x_demo = preprocess(img_ins[:1, :, :, :], image_size=image_size)
    x_demo = tf.tile(x_demo, [n_labels + 1, 1, 1, 1])
    modifier = np.ones([n_labels + 1, n_labels], np.float32)
    for n in range(n_labels):
        modifier[n + 1, n] *= -1.
    lbls_demo = tf.tile(tf.cast(lbls_ins[:1, :], tf.float32), [n_labels + 1, 1]) * modifier
    z_demo, x_out_demo, _, _ = pin_cnn(input=x_demo, lbls=lbls_demo, n_labels=n_labels, reuse=True,
                                       encoded_dimension=encoded_dimension, cnn_layers=cnn_layers,
                                       node_growth_per_layer=node_growth_per_layer, data_format=data_format,
                                       image_channels=image_channels)
    
    x_trn_short = x_trn[:8, :, :, :]
    x_gen_short = x_gen[:8, :, :, :]
    x_out_trn_short = x_out_trn[:8, :, :, :]
    x_out_gen_short = x_out_gen[:8, :, :, :]
    #######################################################################
    # Graph running
    #######################################################################
    sv = tf.train.Supervisor(logdir=logdir)
    with sv.managed_session() as sess:
        # sess = tf.Session()
        # coord = tf.train.Coordinator()
        coord = sv.coord
        enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
        enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)
        sess.run(init_op)
        
        # Print some individuals just to test label alignment
        i, l = sess.run([imgs, img_lbls])
        
        results = np.zeros([training_steps + 1, 5])
        lx, lg, lz, lp = sess.run([loss_x, loss_g, loss_z, loss_pin],
                                  feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa})
        results[0, :] = [lx, lg, lz, lp, kappa]
        
        print 'Date                  Step    Loss_X    Loss_G    Loss_Z  Loss_PIN     kappa learning_rate'
        for step in xrange(1, training_steps + 1):
            learning_rate_current = max(learning_rate_minimum,
                                        np.exp(np.log(learning_rate_initial) - step / learning_rate_decay))
            sess.run([train_gan, train_gen],
                     feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa, adam_learning_rate_ph: learning_rate_current})
            # sess.run([train_cla], feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa, adam_learning_rate_ph: learning_rate_current})
            lx, lg, lz, lp = sess.run([loss_x, loss_g, loss_z, loss_pin],
                                      feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa})
            kappa = max(0.1, min(0.9, kappa + kappa_learning_rate * (gamma_target * lx - lg)))
            results[step, :] = [lx, lg, lz, lp, kappa]
            print_cycle = (step % print_interval == 0) or (step == 1)
            if print_cycle:
                image_print_cycle = (step % graph_interval == 0) or (step == 1)
                print '{} {:6d} {:-9.3f} {:-9.3f} {:-9.3f} {:-9.3f} {:-9.3f} {:-10.8f} {}'.format(now(), step, lx, lg, lz, lp,
                                                                                                  kappa, learning_rate_current,
                                                                                                  ' Graphing' if image_print_cycle else '')

# #######################################################################
# # Clean up the Tensorflow graph
# #######################################################################
# coord.request_stop()
# coord.join(enq_f_threads + enq_i_threads)
# 
# sess.close()
# tf.reset_default_graph()


#######################################################################
# Post processing
#######################################################################
