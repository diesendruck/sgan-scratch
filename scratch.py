import tensorflow as tf
import numpy as np
import os, sys

data_dir = '../data/CelebA/splits/train/'
filenames = os.listdir(data_dir)[:7]
filenames_tensor = tf.constant(filenames)
fnq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=tf.string)
filename = fnq.dequeue()
enq_op = fnq.enqueue_many(filenames_tensor)
qr_f = tf.train.QueueRunner(fnq, [enq_op] * 4)
sess = tf.Session()
coord = tf.train.Coordinator()
enqueue_threads = qr_f.create_threads(sess, coord=coord, start=True)

output = ['xxxxxx.jpg'] * 1000
for step in xrange(1000):
    if coord.should_stop():
        break
    output[step] = sess.run(filename)

print set(output)

coord.request_stop()
coord.join(enqueue_threads)
sess.close()
tf.reset_default_graph()

########################################
import tensorflow as tf
import numpy as np
import os, sys

batch_size = 100
image_size = 64
image_channels = 3
data_dir = '../data/CelebA/splits/train/'
filenames = [data_dir + it for it in os.listdir(data_dir)]
ext = filenames[0][-3:]
if ext == "jpg":
    tf_decode = tf.image.decode_jpeg
elif ext == "png":
    tf_decode = tf.image.decode_png


filenames_tensor = tf.constant(filenames)
fnq = tf.RandomShuffleQueue(capacity=200, min_after_dequeue=100, dtypes=tf.string)
fnq_enq_op = fnq.enqueue_many(filenames_tensor)
filename = fnq.dequeue()

reader = tf.WholeFileReader()
filename_null, data = reader.read(fnq)
image = tf_decode(data, channels=3)
image.set_shape([178, 218, image_channels])
image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
image = tf.to_float(image)

imq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=tf.float32, shapes=[128, 128, image_channels])
imq_enq_op = imq.enqueue(image)
imgs = imq.dequeue_many(batch_size)
imgs = tf.image.resize_nearest_neighbor(imgs, size=[image_size, image_size])

qr_f = tf.train.QueueRunner(fnq, [fnq_enq_op] * 3)
qr_i = tf.train.QueueRunner(imq, [imq_enq_op] * 3)
sess = tf.Session()
coord = tf.train.Coordinator()
enq_f_threads = qr_f.create_threads(sess, coord=coord, start=True)
enq_i_threads = qr_i.create_threads(sess, coord=coord, start=True)

output = [0.] * 1000
for step in xrange(1000):
    print step
    if coord.should_stop():
        break
    # print sess.run(filename)
    imgs_out = sess.run(imgs)
    # print imgs_out[0, :3, :3, 0]
    output[step] = imgs_out.sum()

print output

coord.request_stop()
coord.join(enq_f_threads + enq_i_threads)

sess.close()
tf.reset_default_graph()
