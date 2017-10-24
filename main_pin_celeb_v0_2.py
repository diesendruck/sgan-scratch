# This version adds graphical outputs

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
batch_size = 128
image_size = 64
image_channels = 3
data_dir = '../data/CelebA/splits/train/'
oos_dir = '../data/CelebA/splits/validate/'
encoded_dimension = 64
generator_layers = 6
node_growth_per_layer = 4
data_format = 'NHWC'
num_channels = 3
scale_size = image_size
label_choices = [4, 15, 20, 22, 24]
n_labels = len(label_choices)
learning_rate = 1e-5
training_steps = 125000
lambda_pin_value = 1.
print_interval = 10
graph_interval = 1000



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
    return labels, file_to_idx


def img_and_lbl_queue_setup(filenames, labels):
    labels_tensor = tf.constant(labels, dtype=tf.float32);
    del labels
    
    filenames_tensor = tf.constant(filenames);
    del filenames
    fnq = tf.RandomShuffleQueue(capacity=200, min_after_dequeue=100, dtypes=tf.string)
    fnq_enq_op = fnq.enqueue_many(filenames_tensor)
    filename = fnq.dequeue()
    
    reader = tf.WholeFileReader()
    flnm, data = reader.read(fnq)
    image = tf_decode(data, channels=3)
    image.set_shape([178, 218, image_channels])
    image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
    image = tf.to_float(image)
    
    image_index = [tf.cast(tf.string_to_number(tf.substr(flnm, len(data_dir), 6)), tf.int32)]
    image_labels = tf.reshape(tf.gather(labels_tensor, indices=image_index, axis=0), [n_labels])
    imq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=[tf.float32, tf.float32],
                                shapes=[[128, 128, image_channels], [n_labels]])
    imq_enq_op = imq.enqueue([image, image_labels])
    imgs, img_lbls = imq.dequeue_many(batch_size)
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
    lbls = labels[[int(filename[:6]) for filename in filenames], :]
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

label_file = '../data/CelebA/list_attr_celeba.txt'
labels, file_to_idx = load_labels(label_file, label_choices)


#######################################################################
# model definition
#######################################################################

imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(filenames, labels)
x = imgs
z, enc_vars = Encoder(x, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False)
z_tanh = tf.tanh(z)
z2 = tf.pad(img_lbls, [[0,0], [0, encoded_dimension - n_labels]])
z_pinned = z_tanh * (1. - tf.abs(z2)) + z2 * tf.abs(z2)
pin_loss = tf.losses.mean_squared_error(z_tanh, z_pinned, weights=tf.abs(z2))
if lambda_pin_value == 0.:
    z_pinned = z
x_out, dec_vars = Decoder(z_pinned, input_channel=num_channels, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, final_size=scale_size)
ae_loss = tf.losses.mean_squared_error(x, x_out)
lambda_pin = tf.placeholder(tf.float32, [])
overall_loss = ae_loss + lambda_pin * pin_loss

img_ins, lbls_ins, fs_ins = load_practice_images(data_dir, n_images = 8, labels=labels)
x_ins = preprocess(img_ins, image_size=image_size)
z_ins, _ = Encoder(x_ins, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True)
z_tanh_ins = tf.tanh(z_ins)
x_out_ins, _ = Decoder(z_tanh_ins, input_channel=num_channels, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)

img_oos, lbls_oos, fs_oos = load_practice_images(oos_dir, n_images = 8, labels=labels)
x_oos = preprocess(img_oos, image_size=image_size)
z_oos, _ = Encoder(x_oos, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True)
z_tanh_oos = tf.tanh(z_oos)
x_out_oos, _ = Decoder(z_tanh_oos, input_channel=num_channels, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)

x_demo = preprocess(img_ins[:1, :, :, :], image_size=image_size)
z_demo, _ = Encoder(x_demo, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True)
modifier = np.ones([n_labels + 1, encoded_dimension], np.float32)
for d in range(n_labels): modifier[d+1, d] *= -1.
z_demo_mod = tf.tile(z_demo, [n_labels+1, 1]) * modifier
x_demo_mod = tf.tile(x_demo, [n_labels+1, 1, 1, 1])
z_demo_mod_tanh = tf.tanh(z_demo_mod)
x_out_demo, _ = Decoder(z_demo_mod_tanh, input_channel=num_channels, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)

alr_ph = tf.placeholder(dtype=tf.float32, shape=[])
train_op = tf.train.AdamOptimizer(learning_rate=alr_ph).minimize(overall_loss)
init_op = tf.global_variables_initializer()



#######################################################################
# Graph running
#######################################################################
sess = tf.Session()
coord = tf.train.Coordinator()
enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)

sess.run(init_op)
for step in xrange(1, training_steps+1):
    current_lr = np.exp(-7 - step/20000)
    _, pl, ael, ol = sess.run([train_op, pin_loss, ae_loss, overall_loss], feed_dict={lambda_pin:lambda_pin_value, alr_ph:learning_rate})
    print_cycle = step % print_interval == 0
    if print_cycle:
        image_print_cycle = step % graph_interval == 0
        print '{} {:6d} {:-9.3f} {:-9.3f} {:-9.3f} {}'.format(now(), step, pl, ael, ol, 'Graphing' if image_print_cycle else '')
        if image_print_cycle:
            output = sess.run([x[:8, :, :, :], x_ins, x_oos, x_demo_mod, x_out[:8, :, :, :], x_out_ins, x_out_oos, x_out_demo])
            print '  ', ', '.join(['{:6.2f}'.format(item.mean()) for item in output])
            for idx in range(8):
                output[idx] = output[idx].reshape([-1, image_size, 3])
            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(1. - np.append(output[0], output[4], 1), interpolation='nearest')
            plt.title('In-Sample Production')
            plt.subplot(1, 4, 2)
            plt.imshow(1. - np.append(output[1], output[5], 1), interpolation='nearest')
            plt.title('In-Sample')
            plt.subplot(1, 4, 3)
            plt.imshow(1 - np.append(output[2], output[6], 1), interpolation='nearest')
            plt.title('Out-of-Sample')
            plt.subplot(1, 4, 4)
            plt.imshow(1. - np.append(output[3], output[7], 1), interpolation='nearest')
            plt.title('Changing Vars')
            plt.savefig('img/temp_{}.png'.format(step))
            plt.close()


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
