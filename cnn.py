import numpy as np
import tensorflow as tf
import warnings

def cnn(images, channels = 1, width = [25, 50, 100], num_layers = 3, activations = [tf.tanh], patch_size = 5, stride = 1, activate_last_layer=False, output_dim=None, reuse=False):
    # Reading some implicit parameters
    channels = images.shape[3]
    if num_layers == None:
        if isinstance(width, list):
            num_layers = len(width)
        else:
            raise 'Need to provide num_layers or width as list'
    
    if reuse == None:
        local_scope = tf.get_variable_scope().name + '/' + scope
        scope_in_use = max([obj.name[:len(local_scope)] == local_scope for obj in tf.global_variables()] + [False])
        reuse = scope_in_use
        if scope_in_use == True:
            warnings.warn('Re-using variables for ' + local_scope + ' scope')
    
    # Process the width and activation inputs into useable numbers
    if isinstance(width, list):
        if isinstance(width[0], list):
            layer_widths = width
        else:
            layer_widths = [width] * num_layers
    else:
        layer_widths = [[width] * len(activations)] * num_layers
    
    if output_dim == None:
        output_dim == sum(layer_widths[-1])
    
    # Check for coherency of final layer if needed
    if activate_last_layer and sum(layer_widths[num_layers - 1]) != output_dim:
        print layer_widths
        raise BaseException(
            'activate_last_layer == True but implied final layer width doesn\'t match output_dim \n (implied depth: ' + str(
                sum(layer_widths[num_layers - 1])) + ', explicit depth: ' + str(output_dim) + ')')
    
    # Set up the layers, kernels, convolutions, and biases
    H = [None] * num_layers
    for l in range(num_layers):
        with tf.variable_scope('Layer_' + str(l), reuse=reuse):
            kernel = tf.get_variable('kernel', shape=[patch_size, patch_size, channels, sum(layer_widths[l])], dtype=tf.float64)
            bias = tf.get_variable('bias', shape=sum(layer_widths[l]), dtype=tf.float64)
            
            conv = tf.nn.conv2d(images if l == 0 else H[l-1], kernel, [1, stride, stride, 1], padding='SAME')
            Hs = []
            H[l] = conv + bias
    
    return H[l]

# def decnn(input, output_shape, widths = [])

import tensorflow as tf
import numpy as np

# A = tf.constant(np.random.random([100, 30, 30, 3]).astype(np.float32))
# B = tf.constant(np.random.random([5, 5, 3, 10]).astype(np.float32))
A = tf.constant(np.ones([100, 28, 28, 3]).astype(np.float32))
B = tf.constant(np.ones([5, 5, 3, 10]).astype(np.float32))
C = tf.nn.conv2d(A, B, [1,1,1,1], padding='SAME', data_format='NHWC')
print C.shape

with tf.Session() as sess:
    out = sess.run(C)

from matplotlib import pyplot as plt
plt.imshow(out[0,:,:,0], interpolation='nearest', cmap=plt.get_cmap('Greys')); plt.colorbar(); plt.savefig('temp/conv.png'); plt.close()
print out.shape