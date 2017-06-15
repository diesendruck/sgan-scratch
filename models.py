import pdb
import tensorflow as tf
layers = tf.layers


def generator(z, g_layers_width, g_layers_depth, g_activations, g_out_dim,
        reuse=False):
    with tf.variable_scope('generator', reuse=reuse) as vs:
        x = layers.dense(z, g_layers_width, activation=g_activations)

        for idx in range(g_layers_depth - 1):
            x = layers.dense(x, g_layers_width, activation=g_activations)

        out = layers.dense(x, g_out_dim, activation=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out


def autoencoder(x, d_layers_width, d_layers_depth, d_activations, d_encoded_dim,
        d_out_dim, reuse=False):
    with tf.variable_scope('autoencoder', reuse=reuse) as vs:
        x = layers.dense(x, d_layers_width, activation=d_activations)

        for idx in range(d_layers_depth - 1):
            x = layers.dense(x, d_layers_width, activation=d_activations)

        x = layers.dense(x, d_encoded_dim, activation=None)

        for idx in range(d_layers_depth):
            x = layers.dense(x, d_layers_width, activation=d_activations)

        out = layers.dense(x, d_out_dim, activation=None)
    return out


def encoder(x, d_layers_width, d_layers_depth, d_activations, d_encoded_dim,
        reuse=False):
    with tf.variable_scope('autoencoder_enc', reuse=reuse) as vs:
        x = layers.dense(x, d_layers_width, activation=d_activations)

        for idx in range(d_layers_depth - 1):
            x = layers.dense(x, d_layers_width, activation=d_activations)

        out = layers.dense(x, d_encoded_dim, activation=None)
    return out


def decoder(z, d_layers_width, d_layers_depth, d_activations, d_out_dim,
        reuse=False):
    with tf.variable_scope('autoencoder_dec', reuse=reuse) as vs:
        x = layers.dense(z, d_layers_width, activation=d_activations)

        for idx in range(d_layers_depth - 1):
            x = layers.dense(x, d_layers_width, activation=d_activations)

        out = layers.dense(x, d_out_dim, activation=None)
    return out

