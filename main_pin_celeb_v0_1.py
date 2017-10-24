# Base version that runs

import tensorflow as tf
import numpy as np
import os, sys

#######################################################################
# Run setup
#######################################################################
batch_size = 100
image_size = 64
image_channels = 3
data_dir = '../data/CelebA/splits/train/'
encoded_dimension = 64
generator_layers = 4
node_growth_per_layer = 4
data_format = 'NHWC'
num_channels = 3
scale_size = image_size
label_choices = [4, 15, 20, 22, 24]
n_labels = len(label_choices)
learning_rate = 1e-3
training_steps = 1000
lambda_pin_value = 1.


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
lines = open(label_file, 'r').readlines()
label_names = lines[1].strip().split(' ')
label_names = [label_names[choice] for choice in label_choices]
for ln in xrange(n_labels):
    print '{:2d} {:2d} {}'.format(ln, label_choices[ln], label_names[ln])

# file_to_idx = {lines[i][:10]:i for i in xrange(2, len(lines))}
labels = np.array([[int(it) for it in line.strip().split(' ')[1:] if it != ''] for line in lines[2:]])[:, label_choices]
n_labels = len(label_names)
del lines


#######################################################################
# Queue setup
#######################################################################
labels_tensor = tf.constant(labels, dtype=tf.float32); del labels

filenames_tensor = tf.constant(filenames); del filenames
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
imq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=[tf.float32, tf.float32], shapes=[[128, 128, image_channels], [n_labels]])
imq_enq_op = imq.enqueue([image, image_labels])
imgs, img_lbls = imq.dequeue_many(batch_size)
imgs = tf.image.resize_nearest_neighbor(imgs, size=[image_size, image_size])


qr_f = tf.train.QueueRunner(fnq, [fnq_enq_op] * 3)
qr_i = tf.train.QueueRunner(imq, [imq_enq_op] * 3)
sess = tf.Session()
coord = tf.train.Coordinator()
enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)

#######################################################################
# model definition
#######################################################################
from cnn_began import Encoder, Decoder, DiscriminatorCNN
x = imgs
z, enc_vars = Encoder(x, z_num=encoded_dimension, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False)
z2 = tf.pad(img_lbls, [[0,0], [0, encoded_dimension - n_labels]])
z_pinned = z * (1 - tf.abs(z2)) + z2 * (tf.abs(z2))
pin_loss = tf.losses.mean_squared_error(z, z_pinned, weights=tf.abs(z2))
x_out, dec_vars = Decoder(z_pinned, input_channel=num_channels, repeat_num=generator_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=False, final_size=scale_size)
ae_loss = tf.losses.mean_squared_error(x, x_out)
lambda_pin = tf.placeholder(tf.float32, [])
overall_loss = ae_loss + lambda_pin * pin_loss

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(overall_loss)
init_op = tf.global_variables_initializer()

#######################################################################
# Graph running
#######################################################################
sess.run(init_op)
for step in xrange(training_steps):
    _, pl, ael, ol = sess.run([train_op, pin_loss, ae_loss, overall_loss], feed_dict={lambda_pin:lambda_pin_value})
    print '{:6d} {:-9.3f} {:-9.3f} {:-9.3f}'.format(step, pl, ael, ol)



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
