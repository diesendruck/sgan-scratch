import numpy as np
import tensorflow as tf
from common import *

# Data parameters
data_n = 1000
latent_features = 4
signal_to_noise = 10.


# Network parameters
activations = [tf.tanh]
width = [[5], [5], [2], [2]]
encoder_depth = 3
activate_last_layer = False

# Training parameters
iterations = 10001
ADAM_rate = 1e-3
batch_size = 250
pin_rate = 0.5
pinneable = [1.]*2

#################################################################

# Data simulation
latent = np.identity(latent_features)[np.random.choice(latent_features, data_n),:]
latent_tanh = latent * 2. - 1.
# means = np.random.multivariate_normal([0.,0.], np.identity(2), latent_features)
means = np.array([[1,1], [-1,1], [-1,-1], [1, -1]])
points = latent.dot(means) + np.random.multivariate_normal([0., 0.], np.identity(2) / signal_to_noise, data_n)
latent_tanh = latent.dot(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]))

# Graph construction
data_batch = tf.placeholder(tf.float64, [batch_size, 2])
encoded_values = tf.placeholder(tf.float64,[batch_size, 2])
pins = tf.placeholder(tf.float64, [batch_size, 2])
autoencoded_values, embedding = ff_pin(data_batch, encoded_values, pins, width=width, activations=activations, 
                                       activate_last_layer=activate_last_layer, output_dim=2, encoder_depth=encoder_depth)

loss = tf.nn.l2_loss(autoencoded_values - data_batch)
feature_loss = tf.nn.l2_loss(encoded_values - embedding)

train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
init_op = tf.global_variables_initializer()

from matplotlib import pyplot as plt
def make_grey_chart(data):
    fig = plt.imshow(data, interpolation = 'nearest', cmap=plt.get_cmap('Greys'))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

sess = tf.Session()
sess.run(init_op)
for it in range(iterations):
    data_selection = np.random.choice(data_n, batch_size)
    generated_pins = (np.random.random([batch_size, 2]) < pin_rate).dot(np.diag(pinneable))
    feed = {data_batch: points[data_selection, :], encoded_values: latent_tanh[data_selection,:], pins: generated_pins}
    # for val in feed.values():
    #         print val.shape
    _, transformed, embedded, loss_est, fl_est = sess.run([train_op, autoencoded_values, embedding, loss, feature_loss], feed_dict=feed)
    if it % 1000 == 0:
        print '{:6d} {:8.1f} {:8.1f}'.format(it, loss_est, fl_est)
        plt.subplot(1,2,1); make_grey_chart(embedded[:10,:])
        plt.subplot(1,2,2); make_grey_chart(latent_tanh[data_selection,:][:10,:])
        plt.colorbar()
        plt.savefig('temp/pin/iter_' + str(it) + '.png')
        plt.close()