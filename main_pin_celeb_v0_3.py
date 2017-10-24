# This version implements GAN

from cnn_began import Encoder, Decoder
import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image
from time import *
from matplotlib import pyplot as plt

#######################################################################
# Run setup
#######################################################################
# Data loading params
image_size = 64     # Edge-size of the square images for input and output
image_channels = 3  # How many channels in the image (don't change this, 3 is hardcoded in places)
data_dir = '../data/CelebA/splits/train/'   # Where to find the training images
oos_dir = '../data/CelebA/splits/validate/' # Where to find the validation images
data_format = 'NHWC'    # How the dimensions of the data are ordered (only NHWC is supported right now)
scale_size = image_size # See above
label_file = '../data/CelebA/list_attr_celeba.txt'
label_choices = [4, 15, 20, 22, 24] # Which labels to use (will print your choices once labels are loaded)
n_labels = len(label_choices)   # How many label choices you made

# CNN params
dimension_g = 16    # Dimension of the generators' inputs
encoded_dimension = 64  # Dimension of the encoded layer
cnn_layers = 6    # How many layers in each convolutional layer
node_growth_per_layer = 4   # Linear rate of growth between CNN layers

# Training params
batch_size_x = 64    # Nubmer of samples in each training cycle
batch_size_g = 64  # Number of generated samples
learning_rate_initial = 1e-4    # Base learning rate for the ADAM optimizers; may be decreased over time
learning_rate_decay = 10000.      # How many steps to reduce the learning rate by a factor of e
training_steps = 125000 # Steps of the ADAM optimizers
print_interval = 10     # How often to print a line of output
graph_interval = 100   # How often to output the graphics set

# Penalty params
gamma_target = 0.5 # Target ration of L(G(y))/L(x_trn)
lambda_pin_value = 0.   # Scaling factor of penalty for label mismatch
lambda_ae_value = 1.    # Scaling factor of penalty for auto-encoded generated image quality (ignored in BEGAN)
kappa = 0.  # Initial value of kappa for BEGAN
kappa_learning_rate = 0.001 # Learning rate for kappa

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
    
    image_index = [tf.cast(tf.string_to_number(tf.substr(flnm, len(data_dir), 6))-1, tf.int32)]
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
    lbls = labels[[int(filename[:6])-1 for filename in filenames], :]
    return imgs, lbls, filenames


def preprocess(input_imgs, image_size):
    imgs = tf.to_float(input_imgs)
    imgs = 1. - imgs / 255.5
    imgs = tf.image.crop_to_bounding_box(imgs, 50, 25, 128, 128)
    imgs = tf.to_float(imgs)
    imgs = tf.image.resize_nearest_neighbor(imgs, size=[image_size, image_size])
    return imgs


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
# model definition
#######################################################################

# The model for the real-data training samples
imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(filenames, labels)
x = imgs
z, enc_vars = Encoder(x, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                      data_format=data_format, reuse=False)
if lambda_pin_value > 0.:
    z_pinned = tf.concat([img_lbls, z[:, n_labels:]], 1)
else:
    z_pinned = tf.concat([tf.tanh(z[:, :n_labels]), z[:, n_labels:]], 1)
x_out, dec_vars = Decoder(z_pinned, input_channel=image_channels, repeat_num=cnn_layers,
                          hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, final_size=scale_size)

# The model for the generated_data training samples
y = tf.random_normal([batch_size_g, dimension_g], dtype=tf.float32)
x_g, gen_vars = Decoder(y, input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                        data_format=data_format, reuse=False, final_size=scale_size, var_scope='Gen')
z_g, _ = Encoder(x_g, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                 data_format=data_format, reuse=True)
z_pinned_g = tf.concat([tf.tanh(z_g[:, :n_labels]), z_g[:, n_labels:]], 1)
x_out_g, _ = Decoder(z_pinned_g, input_channel=image_channels, repeat_num=cnn_layers,
                     hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)

# Define the losses
loss_x = tf.losses.mean_squared_error(x, x_out)
loss_g = tf.losses.mean_squared_error(x_g, x_out_g)
zp = z[:, :n_labels]
loss_zi = zp * (img_lbls + 1.) / 2. - tf.log(1. + tf.exp(zp))
loss_z = tf.reduce_sum(loss_zi)
# loss_z = z[:, :n_labels] * (img_lbls == 1.) - tf.log(1 + tf.exp(z[:, :n_labels]))
loss_z = tf.losses.mean_squared_error(img_lbls, tf.tanh(z[:, :n_labels]), weights=tf.abs(img_lbls))
lambda_pin = tf.placeholder(tf.float32, [])
lambda_ae = tf.placeholder(tf.float32, [])
loss_gan = loss_x - lambda_ae * loss_g
loss_gen = loss_g
loss_pin = loss_gan + lambda_pin * loss_z

# Set up the optimizers
adam_learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])
train_gan = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize(loss_pin, var_list=enc_vars + dec_vars)
train_gen = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize(loss_gan, var_list= gen_vars)
train_cla = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize(loss_z, var_list=enc_vars)

# Set up the initializer (NOTE: Keep this after the optimizers, which have parameters to be initialized.)
init_op = tf.global_variables_initializer()

#######################################################################
# Extra output nodes for graphics
#######################################################################
# Run the model on a consistent selection of in-sample pictures
img_ins, lbls_ins, fs_ins = load_practice_images(data_dir, n_images=8, labels=labels)
x_ins = preprocess(img_ins, image_size=image_size)
z_ins, _ = Encoder(x_ins, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                   data_format=data_format, reuse=True)
if lambda_pin_value > 0.:
    z_pinned_ins = tf.concat([tf.tanh(z_ins[:, :n_labels]), z_ins[:, n_labels:]], 1)
else:
    z_pinned_ins = z_ins
x_out_ins, _ = Decoder(z_pinned_ins, input_channel=image_channels, repeat_num=cnn_layers,
                       hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)

# Run the model on a consistent selection of out-of-sample pictures
img_oos, lbls_oos, fs_oos = load_practice_images(oos_dir, n_images=8, labels=labels)
x_oos = preprocess(img_oos, image_size=image_size)
z_oos, _ = Encoder(x_oos, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                   data_format=data_format, reuse=True)
if lambda_pin_value > 0.:
    z_pinned_oos = tf.concat([tf.tanh(z_ins[:, :n_labels]), z_ins[:, n_labels:]], 1)
else:
    z_pinned_oos = z_ins
x_out_oos, _ = Decoder(z_pinned_oos, input_channel=image_channels, repeat_num=cnn_layers,
                       hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)

# Run the model on a consistent selection of out-of-sample pictures
x_demo = preprocess(img_ins[:1, :, :, :], image_size=image_size)
z_demo, _ = Encoder(x_demo, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                    data_format=data_format, reuse=True)
if lambda_pin_value > 0.:
    z_pinned_demo = tf.concat([tf.tanh(z_demo[:, :n_labels]), z_demo[:, n_labels:]], 1)
else:
    z_pinned_demo = z_demo
x_demo_mod = tf.tile(x_demo, [n_labels + 1, 1, 1, 1])
modifier = np.ones([n_labels + 1, encoded_dimension], np.float32)
for d in range(n_labels): modifier[d + 1, d] *= -1.
z_pinned_demo_mod = tf.tile(z_pinned_demo, [n_labels + 1, 1]) * modifier
x_out_demo, _ = Decoder(z_pinned_demo_mod, input_channel=image_channels, repeat_num=cnn_layers,
                        hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)


#######################################################################
# Graph running
#######################################################################
sess = tf.Session()
coord = tf.train.Coordinator()
enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)
sess.run(init_op)

# Print some individuals just to test label alignment
i, l = sess.run([imgs, img_lbls])
plt.figure(figsize=[8, 8])
plt.subplot(1,2,1)
plt.imshow(1.-i[:8,:,:,:].reshape([-1, image_size, 3]), interpolation='nearest')
plt.subplot(1,2,2)
plt.imshow(l[:8,:], interpolation='nearest')
plt.xticks(range(n_labels), label_names, rotation=90)
plt.savefig('img/label_alignment.png')
plt.close()

results = np.zeros([training_steps + 1, 5])
lx, lg, lz, lp = sess.run([loss_x, loss_g, loss_z, loss_pin], 
                          feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa})
results[0, :] = [lx, lg, lz, lp, kappa]

for step in xrange(1, training_steps + 1):
    learning_rate_current = np.exp(np.log(learning_rate_initial) - step / learning_rate_decay)
    sess.run([train_gan, train_gen], feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa, adam_learning_rate_ph: learning_rate_current})
    # sess.run([train_cla], feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa, adam_learning_rate_ph: learning_rate_current})
    lx, lg, lz, lp = sess.run([loss_x, loss_g, loss_z, loss_pin], feed_dict={lambda_pin: lambda_pin_value, lambda_ae: kappa})
    kappa = max(0., min(1., kappa + kappa_learning_rate * (gamma_target * lx - lg)))
    results[step, :] = [lx, lg, lz, lp, kappa]
    print_cycle = (step % print_interval == 0) or (step == 1)
    if print_cycle:
        image_print_cycle = (step % graph_interval == 0) or (step == 1)
        print '{} {:6d} {:-9.3f} {:-9.3f} {:-9.3f} {:-9.3f} {:-9.3f} {:-10.8f} {}'.format(now(), step, lx, lg, lz, lp,
                                                                                          kappa, learning_rate_current, ' Graphing' if image_print_cycle else '')
        if image_print_cycle:
            output = sess.run([x[:8, :, :, :], x_g[:8, :, :, :], x_ins, x_oos, x_demo_mod, 
                               x_out[:8, :, :, :], x_out_g[:8, :, :, :], x_out_ins, x_out_oos, x_out_demo])
            print '  ', ', '.join(['{:6.2f}'.format(item.mean()) for item in output])
            for idx in range(len(output)):
                output[idx] = output[idx].reshape([-1, image_size, 3])
                # print idx, output[idx].shape
            plt.figure(figsize=[16, 8])
            plt.subplot(1, 5, 1)
            plt.imshow(1. - np.append(output[0], output[5], 1), interpolation='nearest')
            plt.title('In-Sample Production')
            plt.subplot(1, 5, 2)
            plt.imshow(1. - np.append(output[1], output[6], 1), interpolation='nearest')
            plt.title('Generated')
            plt.subplot(1, 5, 3)
            plt.imshow(1. - np.append(output[2], output[7], 1), interpolation='nearest')
            plt.title('In-Sample')
            plt.subplot(1, 5, 4)
            plt.imshow(1. - np.append(output[3], output[8], 1), interpolation='nearest')
            plt.title('Out-of-Sample')
            plt.subplot(1, 5, 5)
            plt.imshow(1. - np.append(output[4], output[9], 1), interpolation='nearest')
            plt.title('Changing Vars')
            plt.savefig('img/sample_images_{}.png'.format(step))
            plt.close()
            
            # Print some individuals just to test label alignment
            i, l, zed = sess.run([imgs, img_lbls, z])
            plt.figure(figsize=[8, 8])
            plt.subplot(1, 3, 1)
            plt.imshow(1. - i[:8, :, :, :].reshape([-1, image_size, 3]), interpolation='nearest')
            plt.subplot(1, 3, 2)
            plt.imshow(l[:8, :], interpolation='nearest')
            plt.xticks(range(n_labels), label_names, rotation=90)
            plt.subplot(1, 3, 3)
            plt.imshow(np.tanh(zed[:8, :n_labels]), interpolation='nearest')
            plt.xticks(range(n_labels), label_names, rotation=90)
            plt.savefig('img/label_alignment_{}.png'.format(step))
            plt.close()
            print l.mean(0), np.tanh(zed[:, :n_labels]).mean(0)

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
