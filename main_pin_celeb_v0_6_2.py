# From v0_6_1, we use re-encoding in order to validate that:
#   Auto-encoded X is similar to X
#   Auto-encoded X with changed labels is dissimilar to X
#   Re-encoding of the auto-encoded X results in the same embeddings and labels fed to the decoder
#   Un-altered true model:
#       X --> \hat{H} + \hat{Y} --> \tilde{X} --> \tilde{H} + \tilde{Y}
#   Altered true model:
#             \hat{H} + Y' --> \tilde{X}' --> \tilde{H}' + \tilde{Y}'
#   Losses to consider:
#       Expression              Idea                                                        short_name
#       X        , \tilde{X}    Autoencoding error for real image                           ae
#       Y        , \hat{Y}      Labeling error for (supervised) real images                 label
#       \hat{H}  , \tilde{H}    Re-encoding error for latent features of unaltered image    re_embed
#       \hat{Y}  , \tilde{Y}    Re-encoding error for unaltered labels                      re_label
#       \tilde{X}, \tilde{X}'   Similarity of counter- and real- autoencodings              counter_similarity
#       \hat{H}' , \tilde{H}    Re-encoding error for latent features of altered image      counter_re_embed
#       \hat{Y}' , \tilde{Y}    Re-encoding error for altered labels                        counter_re_label

# From v0_6_2, changing to L1 loss on autoencoding

from cnn_began import Encoder, Decoder
import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image
from time import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#######################################################################
# Run setup
#######################################################################

# Supervisor params
tag = 'pin_celeb_v062a'
logdir = 'log/pin_celeb/{}/'.format(tag)
imgdir = 'img/pin_celeb/{}/'.format(tag)

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
encoded_dimension = 128 # 64 # Dimension of the encoded layer, znum = 256 by default
cnn_layers = 4  # How many layers in each convolutional layer
node_growth_per_layer = 32 # 4 # Linear rate of growth between CNN layers, hidden_num = 128 default

# Training params
first_iteration = 1
batch_size_x = 64 # 64  # Nubmer of samples in each training cycle, default 16
batch_size_g = 64 # 64  # Number of generated samples, default 16
adam_beta_1 = 0.5   # Anti-decay rate of first moment in ADAM optimizer
adam_beta_2 = 0.999 # Anti-decay rate of second moment in ADAM optimizer
learning_rate_initial = 2e-5 # 0.00010 # 1e-4 # Base learning rate for the ADAM optimizers; may be decreased over time, default 0.00008
learning_rate_decay = 2500.  # How many steps to reduce the learning rate by a factor of e
learning_rate_minimum = 1e-8 # 0.0000001 # 1e-4  # Floor for the learning rate
training_steps = 50000  # Steps of the ADAM optimizers
print_interval = 10  # How often to print a line of output
graph_interval = 100  # How often to output the graphics set

# Penalty params
PIN_penalty_mode = ['MSE', 'CE'][1]
gamma_target = 1.0  # Target ration of L(G(y))/L(x_trn), default 1
lambda_pin_value = 0.100  # Scaling factor of penalty for label mismatch
kappa = 0.072  # Initial value of kappa for BEGAN
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


def CrossEntropy(scores, labels):
    return -tf.reduce_sum((scores * (labels + 1.) / 2. - tf.log(1. + tf.exp(scores))) * tf.abs(labels)) / tf.cast(scores.shape[0], tf.float32)


def pin_cnn(images, true_labels=None, counter_labels = None, n_labels=None, reuse=False, encoded_dimension=16, cnn_layers=4,
            node_growth_per_layer=4, data_format='NHWC', image_channels=3):
    embeddings, enc_vars = Encoder(images, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                          data_format=data_format, reuse=reuse)
    latent_embeddings = embeddings[:, n_labels:]
    scores = embeddings[:, :n_labels]
    estimated_labels = tf.tanh(embeddings[:, :n_labels])
    
    if true_labels is None:
        fixed_labels = estimated_labels
    else:
        fixed_labels = estimated_labels * (1 - tf.abs(true_labels)) + true_labels
    
    autoencoded, dec_vars = Decoder(tf.concat([latent_embeddings, fixed_labels], 1), input_channel=image_channels, repeat_num=cnn_layers,
                               hidden_num=node_growth_per_layer, data_format=data_format, reuse=reuse,
                               final_size=scale_size)
    autoencoded = tf.maximum(tf.minimum(autoencoded, 1.), 0.)
    reencoded_embeddings, _ = Encoder(autoencoded, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer,
                          data_format=data_format, reuse=True)
    reencoded_latent_embeddings = reencoded_embeddings[:, n_labels:]
    reencoded_scores = reencoded_embeddings[:, :n_labels]
    reencoded_estimated_labels = tf.tanh(reencoded_embeddings[:, :n_labels])
    output = {'images':images,
              'true_labels':true_labels,
              'embeddings':embeddings,
              'scores':scores, 
              'latent_embeddings':latent_embeddings, 
              'estimated_labels':estimated_labels, 
              'fixed_labels':fixed_labels, 
              'autoencoded':autoencoded,
              'reencoded_embeddings':reencoded_embeddings,
              'reencoded_scores':reencoded_scores, 
              'reencoded_latent_embeddings':reencoded_latent_embeddings, 
              'reencoded_estimated_labels':reencoded_estimated_labels,
              'enc_vars':enc_vars,
              'dec_vars':dec_vars,
              'losses': {'ae': tf.reduce_sum(tf.abs(images - autoencoded)) / tf.cast(images.shape[0], tf.float32), # tf.losses.mean_squared_error(images, autoencoded),
                        'label': tf.zeros_like(scores) if true_labels is None else CrossEntropy(scores, true_labels),
                        're_embed': tf.losses.mean_squared_error(latent_embeddings, reencoded_latent_embeddings),
                        're_label': CrossEntropy(reencoded_scores, fixed_labels)
                        }
              }
    if true_labels is not None:
        output['true_labels'] = true_labels
    
    if counter_labels is None:
        counter_output = 'Nothing here'
    else:
        counter_fixed_labels = estimated_labels * (1 - tf.abs(counter_labels)) + counter_labels
        counter_autoencoded, _ = Decoder(tf.concat([latent_embeddings, counter_fixed_labels], 1), input_channel=image_channels, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True, final_size=scale_size)
        counter_autoencoded = tf.maximum(tf.minimum(counter_autoencoded, 1.), 0.)
        counter_reencoded_embeddings, _ = Encoder(counter_autoencoded, z_num=encoded_dimension, repeat_num=cnn_layers, hidden_num=node_growth_per_layer, data_format=data_format, reuse=True)
        counter_reencoded_latent_embeddings = counter_reencoded_embeddings[:, n_labels:]
        counter_reencoded_scores = counter_reencoded_embeddings[:, :n_labels]
        counter_reencoded_estimated_labels = tf.tanh(counter_reencoded_embeddings[:, :n_labels])
        output.update({'counter_labels': counter_labels, 
                       'counter_fixed_labels':counter_fixed_labels, 
                       'counter_autoencoded':counter_autoencoded, 
                       'counter_reencoded_scores':counter_reencoded_scores, 
                       'counter_reencoded_latent_embeddings':counter_reencoded_latent_embeddings, 
                       'counter_reencoded_estimated_labels':counter_reencoded_estimated_labels
                       })
        output['losses'].update({'counter_similarity': tf.losses.mean_squared_error(autoencoded, counter_autoencoded),
                                 'counter_re_embed': tf.losses.mean_squared_error(latent_embeddings, counter_reencoded_latent_embeddings),
                                 'counter_re_label': CrossEntropy(counter_reencoded_scores, counter_fixed_labels)
                                })
    return output




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
x_trn = imgs
# # .375 = flipp 1/5 of the bits
# counter_labels = img_lbls * (tf.round(tf.random_uniform(img_lbls.shape, minval=.375, maxval=1.)) * 2. - 1.)
temp = np.ones([batch_size_x, n_labels])
for n in range(n_labels):
    temp[n, n % n_labels] = -1.
counter_label_modifier = tf.constant(temp, dtype=img_lbls.dtype)
counter_labels = img_lbls * counter_label_modifier
trn = pin_cnn(images=x_trn, true_labels=img_lbls if lambda_pin_value > 0. else None, counter_labels=counter_labels, n_labels=n_labels, reuse=False, encoded_dimension=encoded_dimension, cnn_layers=cnn_layers, node_growth_per_layer=node_growth_per_layer, data_format=data_format, image_channels=image_channels)

# Define the objective from the training losses
loss_names = ['ae', 'label', 're_embed', 're_label', 'counter_similarity', 'counter_re_embed', 'counter_re_label']
l = trn['losses']
# objective = l['ae'] + l['label'] + l['re_embed'] + l['re_label'] - l['counter_similarity'] + l['counter_re_embed'] + l['counter_re_label']
objective = 1000. * l['ae'] + 100 * l['label'] + l['re_embed'] + l['re_label'] + l['counter_re_embed'] + l['counter_re_label']
# need to think hard about how to weight these...

# Set up the optimizers
adam_learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])
trainer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate_ph).minimize(objective, var_list=trn['enc_vars'] + trn['dec_vars'])

# Set up the initializer (NOTE: Keep this after the optimizers, which have parameters to be initialized.)
init_op = tf.global_variables_initializer()

#######################################################################
# Extra output nodes for graphics
#######################################################################
# Run the model on a consistent selection of in-sample pictures
img_ins, lbls_ins, fs_ins = load_practice_images(data_dir, n_images=8, labels=labels)
x_ins = preprocess(img_ins, image_size=image_size)
ins = pin_cnn(images=x_ins, n_labels=n_labels, reuse=True, encoded_dimension=encoded_dimension, cnn_layers=cnn_layers, node_growth_per_layer=node_growth_per_layer, data_format=data_format, image_channels=image_channels)

# Run the model on a consistent selection of in-sample pictures
n_big = 1024
img_big, lbls_big, fs_big = load_practice_images(data_dir, n_images=n_big, labels=labels)
x_big = preprocess(img_big, image_size=image_size)
big = pin_cnn(images=x_big, n_labels=n_labels, reuse=True, encoded_dimension=encoded_dimension, cnn_layers=cnn_layers, node_growth_per_layer=node_growth_per_layer, data_format=data_format, image_channels=image_channels)

# Run the model on a consistent selection of out-of-sample pictures
img_oos, lbls_oos, fs_oos = load_practice_images(oos_dir, n_images=8, labels=labels)
x_oos = preprocess(img_oos, image_size=image_size)
oos = pin_cnn(images=x_oos, n_labels=n_labels, reuse=True, encoded_dimension=encoded_dimension, cnn_layers=cnn_layers, node_growth_per_layer=node_growth_per_layer, data_format=data_format, image_channels=image_channels)

# Run the model on a consistent selection of out-of-sample pictures
x_demo = preprocess(img_ins[:1, :, :, :], image_size=image_size)
x_demo = tf.tile(x_demo, [n_labels + 1, 1, 1, 1])
modifier = np.ones([n_labels + 1, n_labels], np.float32)
for n in range(n_labels):
    modifier[n + 1, n] *= -1.

lbls_demo = tf.tile(tf.cast(lbls_ins[:1, :], tf.float32), [n_labels + 1, 1]) * modifier
demo = pin_cnn(images=x_demo, true_labels=lbls_demo, n_labels=n_labels, reuse=True, encoded_dimension=encoded_dimension, cnn_layers=cnn_layers, node_growth_per_layer=node_growth_per_layer, data_format=data_format, image_channels=image_channels)

x_trn_short = trn['images'][:8, :, :, :]
x_out_trn_short = trn['autoencoded'][:8, :, :, :]

#######################################################################
# Graph running
#######################################################################

sv = tf.train.Supervisor(logdir=logdir)

with sv.managed_session() as sess:
    # sess = tf.Session()
    # coord = tf.train.Coordinator()
    sv.start_standard_services(sess)
    coord = sv.coord
    
    enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
    enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)
    sess.run(init_op)
    
    # Print some individuals just to test label alignment
    i, l = sess.run([trn['images'], trn['true_labels']])
    plt.figure(figsize=[8, 8])
    plt.subplot(1, 2, 1)
    plt.imshow(1. - i[:8, :, :, :].reshape([-1, image_size, 3]), interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(l[:8, :], interpolation='nearest')
    plt.xticks(range(n_labels), label_names, rotation=90)
    plt.savefig(imgdir + 'label_alignment.png')
    plt.close()
    
    results = np.zeros([training_steps + 1, len(loss_names) + 1])
    eval_losses = sess.run(trn['losses'], feed_dict={})
    results[0, :] = [eval_losses[key] for key in loss_names] + [kappa]
    
    for step in xrange(first_iteration, training_steps + 1):
        learning_rate_current = max(learning_rate_minimum,
                                    np.exp(np.log(learning_rate_initial) - step / learning_rate_decay))
        
        sess.run([trainer], feed_dict={adam_learning_rate_ph: learning_rate_current})
        
        eval_losses = sess.run(trn['losses'], feed_dict={})
        # kappa = max(kappa_range[0], min(kappa_range[1], kappa + kappa_learning_rate * (gamma_target * lx - lg)))
        # kappa = kappa + kappa_learning_rate * (gamma_target * lx - lg)
        results[step, :] = [eval_losses[key] for key in loss_names] + [kappa]
        
        print_cycle = (step % print_interval == 0) or (step == 1)
        if print_cycle:
            image_print_cycle = (step % graph_interval == 0) or (step == 1)
            
            print '{} {:6d} {} {:-9.3f} {:-10.8f} {}'.format(now(), step, ' '.join(['{:-9.3f}'.format(eval_losses[key]) for key in loss_names]), kappa, learning_rate_current, ' Graphing' if image_print_cycle else '')
            if image_print_cycle:
                output = sess.run([trn['images'], trn['images'], ins['images'], oos['images'], demo['images'],
                                   trn['autoencoded'], trn['counter_autoencoded'], ins['autoencoded'], oos['autoencoded'], demo['autoencoded']])
                print '  ', ', '.join(['{:6.2f}'.format(item.mean()) for item in output])
                tmp = output[-1] + 0.
                for idx in reversed(range(6)):
                    tmp[idx, :, :, :] -= tmp[0, :, :, :]
                tmp = np.abs(tmp.reshape([-1, image_size, 3]))
                for idx in range(len(output)):
                    output[idx] = output[idx][:8, :, :, :].reshape([-1, image_size, 3])
                    # print idx, output[idx].shape
                plot_names = ['In-Sample Production', 'In-Sample CounterFactual', 'In-Sample (fixed)', 'Out-of-Sample (fixed)',
                              'Manipulated']
                plt.figure(figsize=[16, 8])
                for image_idx in range(5):
                    plt.subplot(1, 6, image_idx + 1)
                    plt.imshow(1. - np.append(output[image_idx], output[image_idx + 5], 1), interpolation='nearest')
                    plt.title(plot_names[image_idx])
                    if image_idx == 4:
                        plt.yticks([image_size * (n + .5) for n in range(n_labels + 1)], ['None'] + label_names,
                                   rotation=90)
                plt.subplot(1, 6, 6)
                plt.imshow(1. - tmp / tmp.max(), interpolation='nearest')
                plt.title('Diff from base')
                plt.savefig(imgdir + 'sample_images_{:06d}.png'.format(step))
                plt.close()
                
                # Print some individuals just to test label alignment
                i, l, h = sess.run([x_trn_short, img_lbls, trn['estimated_labels']])
                plt.figure(figsize=[8, 8])
                plt.subplot(1, 3, 1)
                plt.imshow(1. - i.reshape([-1, image_size, 3]), interpolation='nearest')
                plt.subplot(1, 3, 2)
                plt.imshow(l[:8, :], interpolation='nearest', cmap=plt.get_cmap('Greys'))
                plt.xticks(range(n_labels), label_names, rotation=90)
                plt.subplot(1, 3, 3)
                plt.imshow(np.tanh(h[:8, :n_labels]), interpolation='nearest', cmap=plt.get_cmap('Greys'))
                plt.xticks(range(n_labels), label_names, rotation=90)
                plt.savefig(imgdir + 'label_alignment_{:06d}.png'.format(step))
                plt.close()
                print l.mean(0), np.tanh(h[:, :n_labels]).mean(0)
                print 'Date                  Step    Loss_X    Loss_G    Loss_Z  Loss_PIN     kappa learning_rate'
                
                e = sess.run(big['embeddings'])
                m = e.mean(0)
                v = e.var(0)
                # M = np.linalg.inv(e.transpose().dot(e)).dot(e.transpose()).dot(l)
                normalized_embeddings = (e - m) / np.sqrt(v)
                
                plt.imshow(np.abs(normalized_embeddings.transpose().dot(normalized_embeddings)), interpolation='nearest', cmap=plt.get_cmap('Greys'))
                plt.colorbar()
                plt.title('Correlation of cae_embeddings')
                plt.savefig(imgdir + 'emedding_correlation_{:06d}.png'.format(step))
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
