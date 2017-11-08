from cnn_began import Encoder, Decoder
import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image
from time import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ffnn import ffnn
import json

#######################################################################
# Run setup
#######################################################################
# Geome param
shape_type = ['rectangles', 'polygons', 'stars'][0]

# Supervisor params
tag = 'cae_019'
logdir = 'log/cae_geome_{}/{}/'.format(shape_type, tag)
imgdir = 'img/cae_geome_{}/{}/'.format(shape_type, tag)

# Data loading params
image_size = 64  # Edge-size of the square images for input and output, scale_size = 64 default
image_channels = 3  # How many channels in the image (don't change this, 3 is hardcoded in places)
data_dir = '../data/geome/{}/splits/train/'.format(shape_type)   # Where to find the training images
oos_dir = '../data/geome/{}/splits/validate/'.format(shape_type) # Where to find the validation images
data_format = 'NHWC'  # How the dimensions of the data are ordered (only NHWC is supported right now)
scale_size = image_size  # See above
label_file = '../data/geome/{}/list_attr_{}.txt'.format(shape_type, shape_type)
label_choices = range(6) #range(40) # [4, 15, 20, 22, 24]  # Which labels to use (will print your choices once labels are loaded)
n_labels = len(label_choices)  # How many label choices you made
# CNN params
dimension_g = 4  # Dimension of the generators' inputs
encoded_dimension = 32 # 64 # Dimension of the encoded layer, znum = 256 by default
cnn_layers = 4  # How many layers in each convolutional layer
node_growth_per_layer = 8 # 4 # Linear rate of growth between CNN layers, hidden_num = 128 default

# Training params
first_iteration = 1
batch_size_x = 128 # 64  # Nubmer of samples in each training cycle, default 16
adam_beta_1 = 0.5   # Anti-decay rate of first moment in ADAM optimizer
adam_beta_2 = 0.999 # Anti-decay rate of second moment in ADAM optimizer
learning_rate_initial = 0.0010 # 1e-4 # Base learning rate for the ADAM optimizers; may be decreased over time, default 0.00008
learning_rate_decay = 2000.  # How many steps to reduce the learning rate by a factor of e
learning_rate_minimum = 0.0000001 # 1e-4  # Floor for the learning rate
training_steps = 25000  # Steps of the ADAM optimizers
print_interval = 10  # How often to print a line of output
graph_interval = 100  # How often to output the graphics set

# Penalty params
PIN_penalty_mode = ['MSE', 'CE'][1]
gamma_target = 1.0  # Target ration of L(G(y))/L(x_trn), default 1
lambda_pin_value = 0.100  # Scaling factor of penalty for label mismatch
kappa = 1.000  # Initial value of kappa for BEGAN
kappa_range = [0., 1.]
kappa_learning_rate = 0.0005  # Learning rate for kappa
lambda_k_initial = 0.001 # Initial value for lambda_k

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
    if lbls is None:
        z_pin = tf.concat([tf.tanh(z[:, :n_labels]), z[:, n_labels:]], 1)
    else:
        z_pin = tf.concat([tf.tanh(z[:, :n_labels]) * (1. - tf.abs(lbls)) + lbls, z[:, n_labels:]], 1)
    output, dec_vars = Decoder(z_pin, input_channel=image_channels, repeat_num=cnn_layers,
                               hidden_num=node_growth_per_layer, data_format=data_format, reuse=reuse,
                               final_size=scale_size)
    output = tf.maximum(tf.minimum(output, 1.), 0.)
    return z, z_pin, output, enc_vars, dec_vars


def CrossEntropy(scores, labels):
    return -tf.reduce_sum((scores * (labels + 1.) / 2. - tf.log(1. + tf.exp(scores))) * tf.abs(labels)) / tf.cast(scores.shape[0], tf.float32)

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
tf.Graph().as_default()

#######################################################################
# model definition
#######################################################################

# The model for the real-data training samples
imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(filenames, labels)
cae_embeddings, cae_enc_vars = Encoder(imgs, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, var_scope='CAE_Encoder')
c_embeddings, c_enc_vars     = Encoder(imgs, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, var_scope='C_Encoder')
ae_embeddings, ae_enc_vars   = Encoder(imgs, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, var_scope='AE_Encoder')
cae_label_predictions, cae_ffnn_vars = ffnn(cae_embeddings, num_layers=5, width=[[2 * n_labels]] * 4 + [[n_labels]], output_dim=n_labels, activations=[tf.tanh], activate_last_layer=False, reuse=False, var_scope='CAE_FFNN')
c_label_predictions, c_ffnn_vars     = ffnn(  c_embeddings, num_layers=5, width=[[2 * n_labels]] * 4 + [[n_labels]], output_dim=n_labels, activations=[tf.tanh], activate_last_layer=False, reuse=False, var_scope='C_FFNN')
cae_autoencoded_images, cae_dec_vars = Decoder(cae_embeddings, input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, final_size=scale_size, var_scope='CAE_Decoder')
ae_autoencoded_images, ae_dec_vars   = Decoder( ae_embeddings, input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, final_size=scale_size, var_scope='AE_Decoder')

# Define the losses
lambda_ae = tf.placeholder(tf.float32, [])
cae_loss_ae = tf.losses.mean_squared_error(cae_autoencoded_images, imgs)
ae_loss_ae  = tf.losses.mean_squared_error( ae_autoencoded_images, imgs)
cae_loss_lbls_ce = CrossEntropy(cae_label_predictions, img_lbls)
c_loss_lbls_ce   = CrossEntropy(  c_label_predictions, img_lbls)
cae_loss_lbls_mse = tf.losses.mean_squared_error(img_lbls, tf.tanh(cae_label_predictions), weights=tf.abs(img_lbls))
c_loss_lbls_mse   = tf.losses.mean_squared_error(img_lbls, tf.tanh(  c_label_predictions), weights=tf.abs(img_lbls))
cae_loss_combined = cae_loss_lbls_ce + lambda_ae * cae_loss_ae
c_loss_combined   =   c_loss_lbls_ce
ae_loss_combined  =                    lambda_ae *  ae_loss_ae

# Set up the optimizers
adam_learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])
train_cae = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize(cae_loss_combined, var_list=cae_enc_vars + cae_ffnn_vars + cae_dec_vars)
train_c   = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize(  c_loss_combined, var_list=  c_enc_vars +   c_ffnn_vars               )
train_ae  = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize( ae_loss_combined, var_list= ae_enc_vars +                  ae_dec_vars)

# Set up the initializer (NOTE: Keep this after the optimizers, which have parameters to be initialized.)
init_op = tf.global_variables_initializer()

#######################################################################
# Extra output nodes for graphics
#######################################################################
# Run the model on a consistent selection of in-sample pictures
img_oos, lbls_oos, fs_oos = load_practice_images(oos_dir, n_images=batch_size_x, labels=labels)
x_oos = preprocess(img_oos, image_size=image_size)
lbls_oos_tf = tf.constant(lbls_oos, dtype=tf.float32)

cae_e_oos, _ = Encoder(x_oos, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, var_scope='CAE_Encoder')
c_e_oos, _   = Encoder(x_oos, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, var_scope=  'C_Encoder')
ae_e_oos, _  = Encoder(x_oos, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, var_scope= 'AE_Encoder')
cae_lp_oos, _ = ffnn(cae_e_oos, num_layers=5, width=[[2 * n_labels]] * 4 + [[n_labels]], output_dim=n_labels, activations=[tf.tanh], activate_last_layer=True, var_scope='CAE_FFNN', reuse=True)
c_lp_oos, _   = ffnn(  c_e_oos, num_layers=5, width=[[2 * n_labels]] * 4 + [[n_labels]], output_dim=n_labels, activations=[tf.tanh], activate_last_layer=True, var_scope=  'C_FFNN', reuse=True)
cae_aei_oos, _ = Decoder(cae_e_oos, input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size, var_scope='CAE_Decoder')
ae_aei_oos, _  = Decoder( ae_e_oos, input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size, var_scope= 'AE_Decoder')

images_short = imgs[:8, :, :, :]
cae_autoencoded_images_short = cae_autoencoded_images[:8, :, :, :]
ae_autoencoded_images_short  =  ae_autoencoded_images[:8, :, :, :]

cae_loss_oos_ae = tf.losses.mean_squared_error(x_oos, cae_aei_oos)
ae_loss_oos_ae  = tf.losses.mean_squared_error(x_oos,  ae_aei_oos)
cae_loss_oos_ce = CrossEntropy(cae_lp_oos, lbls_oos_tf)
c_loss_oos_ce   = CrossEntropy(  c_lp_oos, lbls_oos_tf)
cae_loss_oos_mse = tf.losses.mean_squared_error(lbls_oos, tf.tanh(cae_lp_oos))
c_loss_oos_mse   = tf.losses.mean_squared_error(lbls_oos, tf.tanh(c_lp_oos))

#######################################################################
# Graph running
#######################################################################
sv = tf.train.Supervisor(logdir=logdir)
with sv.managed_session() as sess:
    sv.start_standard_services(sess)
    coord = sv.coord
    enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
    enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)
    sess.run(init_op)
    
    # Print some individuals just to test label alignment
    i, l = sess.run([imgs, img_lbls])
    plt.figure(figsize=[8, 8])
    plt.subplot(1, 2, 1)
    plt.imshow(1. - i[:8, :, :, :].reshape([-1, image_size, 3]), interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(l[:8, :], interpolation='nearest')
    plt.xticks(range(n_labels), label_names, rotation=90)
    plt.savefig(imgdir + 'label_alignment.png')
    plt.close()
    
    losses = [cae_loss_ae, ae_loss_ae, cae_loss_lbls_ce, c_loss_lbls_ce, cae_loss_oos_ae, ae_loss_oos_ae, cae_loss_oos_ce, c_loss_oos_ce]
    # losses = [cae_loss_ae, cae_loss_lbls_ce, cae_loss_combined, cae_loss_oos_mse, cae_loss_oos_ce]
    loss_values = sess.run(losses, feed_dict={lambda_ae: kappa})
    results = np.zeros([training_steps + 1, len(losses) + 1])
    results[0, :] = loss_values + [kappa]

    print '                              ______________In Sample_____________ |  ____________Out of Sample___________     Kappa Learning Rate'
    print '                      Step    CAE_AE     AE_AE     CAE_C       C_C    CAE_AE     AE_AE     CAE_C       C_C     Kappa Learning Rate'
    for step in xrange(first_iteration, training_steps + 1):
        learning_rate_current = max(learning_rate_minimum, np.exp(np.log(learning_rate_initial) - step / learning_rate_decay))
        #sess.run([train_combined], feed_dict={lambda_ae: kappa, adam_learning_rate_ph: learning_rate_current})
        _, _, _, loss_values = sess.run([train_cae, train_ae, train_c, losses], feed_dict={lambda_ae: kappa, adam_learning_rate_ph: learning_rate_current})
        # kappa = max(kappa_range[0], min(kappa_range[1], kappa + kappa_learning_rate * (gamma_target * lx - lg)))
        # kappa = kappa + kappa_learning_rate * (gamma_target * lx - lg)
        results[step, :] = loss_values + [kappa]
        print_cycle = (step % print_interval == 0) or (step == 1)
        if print_cycle:
            image_print_cycle = (step % graph_interval == 0) or (step == 1)
            losses_string = ' '.join(['{:-9.5f}'.format(val) for val in loss_values])
            print '{} {:6d} {} {:-9.3f} {:-10.8f} {}'.format(now(), step, losses_string, kappa, learning_rate_current, ' Graphing' if image_print_cycle else '')
            if image_print_cycle:
                # Print some individuals just to test label alignment
                i, iae, l, zed = sess.run([images_short, cae_autoencoded_images_short, img_lbls, cae_label_predictions])
                i = np.clip(i, 0., 1.)
                iae = np.clip(iae, 0., 1.)
                plt.figure(figsize=[8, 8])
                plt.subplot(1, 4, 1)
                plt.imshow(1. - i.reshape([-1, image_size, 3]), interpolation='nearest')
                plt.subplot(1, 4, 2)
                plt.imshow(1. - iae.reshape([-1, image_size, 3]), interpolation='nearest')
                plt.subplot(1, 4, 3)
                plt.imshow(l[:8, :], interpolation='nearest', cmap=plt.get_cmap('Greys'))
                plt.xticks(range(n_labels), label_names, rotation=90)
                plt.subplot(1, 4, 4)
                plt.imshow(np.tanh(zed[:8, :n_labels]), interpolation='nearest', cmap=plt.get_cmap('Greys'))
                plt.xticks(range(n_labels), label_names, rotation=90)
                plt.savefig(imgdir + 'label_alignment_{:06d}.png'.format(step))
                plt.close()
                print '                              ______________In Sample_____________ |  ____________Out of Sample___________     Kappa Learning Rate'
                print '                      Step    CAE_AE     AE_AE     CAE_C       C_C    CAE_AE     AE_AE     CAE_C       C_C     Kappa Learning Rate'

                plt.figure(figsize=[8,8])
                # results_names = ['Autoencoder Loss', 'Classifier Loss', 'Combined Loss', 'OOS Autoencoder Loss', 'OOS Classifier Loss', 'Kappa']
                results_names = ['C-AE AE Loss', 'AE Loss', 'C-AE C Loss', 'C Loss', 'C-AE AE OOS Loss', 'AE OOS Loss', 'C-AE C OOS Loss', 'C OOS Loss', 'Kappa']
                for idx in range(results.shape[1]):
                    plt.subplot(5, 2, idx+1)
                    plt.plot(results[:step, idx])
                    plt.title(results_names[idx])
                plt.savefig(imgdir + 'numeric_results_{:06d}.png'.format(step))
                plt.close()
                
                json.dump(results.tolist(), open(imgdir + 'numeric_results.json', 'w'))
