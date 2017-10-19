import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import re
import warnings
import sys
import tensorflow as tf
from glob import glob
from utils import *
import common

class SGAN(object):
    def __init__(self, sess, max_iter=50001, optim='adam', learning_rate=1e-2,
            d_per_iter=1, g_per_iter=2,
            real_n=1000, real_dim=2, fake_n=1000, z_dim=3, g_out_dim=2,
            g_layers_depth=3, g_layers_width=None, g_activations=None,
            d_out_dim=2, d_layers_depth=5, d_layers_width=5,
            d_activations=None, d_batch_size=25, x_lims=None, y_lims=None,
            grid_gran=21, grid_n=None, dataset='gaussian', expt='test_began'):
        """Basic attributes of the simple GAN.
        
        Args:
          sess: TensorFlow Session
          ...

        """
        self.gamma = 0.5
        self.lambda_k = 0.001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.d_lr = 8e-4
        self.g_lr = 8e-4


        self.sess = sess
        self.max_iter = max_iter
        self.optim = optim
        self.learning_rate = learning_rate

        self.d_per_iter = d_per_iter 
        self.g_per_iter = g_per_iter

        self.real_n = real_n  
        self.real_dim = real_dim  
        self.fake_n = fake_n

        self.z_dim = z_dim
        self.g_out_dim = real_dim
        self.g_layers_depth = g_layers_depth
        self.g_layers_width = [[5]] * (g_layers_depth - 1) + [[g_out_dim]]
        self.g_activations = [tf.nn.elu]

        self.d_out_dim = d_out_dim
        self.d_layers_depth = d_layers_depth
        self.d_layers_width = [[5], [5], [2], [5], [5]] 
        self.d_activations = [tf.nn.elu]
        self.d_batch_size = d_batch_size
        
        self.x_lims = [-6., 2.]
        self.y_lims = [-2., 6.]
        self.grid_gran = grid_gran
        self.grid_n = grid_gran ** 2
        self.grid, self.x_grid, self.y_grid = self.make_grid()

        self.dataset = dataset  
        self.real_points = load_2d_data(dataset, real_n, real_dim)

        self.expt = expt

        self.build_model()


    def build_model(self):
        self.data_sample = tf.placeholder(
                tf.float64, [self.d_batch_size, self.real_dim])
        self.k_t = tf.Variable(0., dtype=tf.float64, trainable=False, name='k_t')
        self.z = tf.placeholder(tf.float64, [self.fake_n, self.z_dim])

        self.decoder_params = {
            'num_layers': self.g_layers_depth,
            'width': self.g_layers_width,
            'output_dim': self.g_out_dim,
            'activations': self.g_activations,
            'activate_last_layer': False,
            'scope': 'decoder'}
        self.ae_params = {
            'num_layers': self.d_layers_depth,
            'width': self.d_layers_width,
            'output_dim': self.d_out_dim,
            'activations': self.d_activations,
            'activate_last_layer': False,
            'scope': 'autoencoder'}
        self.gen = common.ffnn(self.z, reuse=False, **self.decoder_params) 
        self.ae_real = common.ffnn(self.data_sample, reuse=False,
                **self.ae_params)
        self.ae_gen = common.ffnn(self.gen, reuse=True, **self.ae_params)
        self.ae_grid = common.ffnn(tf.convert_to_tensor(self.grid), reuse=True,
                    **self.ae_params)

        # Define autoencoder losses.
        self.ae_loss_real = tf.reduce_mean(tf.abs(self.ae_real - self.data_sample))
        self.ae_loss_gen = tf.reduce_mean(tf.abs(self.ae_gen - self.gen))

        self.ae_loss_real_vals = tf.reduce_sum(
                tf.abs(self.ae_real - self.data_sample), 1)
        self.ae_loss_gen_vals = tf.reduce_sum(
                tf.abs(self.ae_gen - self.gen), 1)
        self.ae_loss_grid_vals = tf.reduce_sum(
                tf.abs(self.ae_grid - self.grid), 1)

        self.d_loss = self.ae_loss_real - self.k_t * self.ae_loss_gen
        self.g_loss = tf.reduce_mean(tf.abs(self.ae_gen - self.gen)) 

        # Build optimization ops.
        self.g_vars = [
            var for var in tf.global_variables() if 'decoder' in var.name]
        self.d_vars = [
            var for var in tf.global_variables() if 'autoencoder' in var.name]
        def prod(x):
            out = 0
            for xi in x:
                out *= int(xi)
            return out
        for v in tf.global_variables():
            print '{:10d}'.format(prod(v.shape)), v.name.ljust(50), v.shape


    def train(self):
        if self.optim == "adagrad":
            optimizer = tf.train.AdagradOptimizer
        else:
            optimizer = tf.train.AdamOptimizer

        d_optim = optimizer(self.d_lr).minimize(self.d_loss,
                var_list=self.d_vars)
        g_optim = optimizer(self.g_lr).minimize(self.g_loss,
                var_list=self.g_vars)

        self.balance = self.gamma * self.ae_loss_real - self.ae_loss_gen
        self.measure = self.ae_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, 
                tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        tf.global_variables_initializer().run()

        for it in range(self.max_iter):
            # Run training.
            self.sess.run(
                [self.k_update, self.measure, self.d_loss, self.g_loss,
                    self.k_t],
                feed_dict={
                    self.z: self.gen_z(),
                    self.data_sample: self.gen_data_sample()})

            # For iterations, plot results.
            if ((it <= 10 and it % 1 == 0) or
                    (it < 500 and it % 25 == 0) or
                    (it % 1000 == 0)):
                # Get data to plot and summarize.
                generated = self.sess.run(
                    self.gen,
                    feed_dict={
                        self.z: self.gen_z(),
                        self.data_sample: self.gen_data_sample()})

                # Run scores and unpack contents for grid, data, and gen.
                grid_scores, real_scores, gen_scores = self.sess.run(
                    [self.ae_loss_grid_vals,
                     self.ae_loss_real_vals, 
                     self.ae_loss_gen_vals],
                    feed_dict={
                        self.z: self.gen_z(),
                        self.data_sample: self.gen_data_sample()})

                # Collect summary items for printing.
                summary_items = [
                        it,
                        round_list(self.sess.run( [self.d_loss, self.g_loss], feed_dict = {self.z: self.gen_z(), self.data_sample: self.gen_data_sample()})),
                        round_list(self.real_points.mean(0)),
                        round_list(generated.mean(0)),
                        round_list(self.real_points.var(0)),
                        round_list(generated.var(0))]
                si = summary_items
                print ("iteration: {}, [d_loss, g_loss]: {}, data_mean: {},"
                       " gen_mean: {}, data_var: {}, gen_var: {}").format(
                    si[0], si[1], si[2], si[3], si[4], si[5])

                # Plot results.
                dest_graphs_dir = "./temp/{}/graphs".format(self.expt)
                dest_scores_dir = "./temp/{}/scores".format(self.expt)
                target_dirs = [dest_graphs_dir, dest_scores_dir]
                for d in target_dirs:
                    if not os.path.exists(d):
                        os.makedirs(d)
                        print "Made dir: {}".format(d)

                fig, ax = plt.subplots()
                d_grid = np.reshape(
                        grid_scores, [self.grid_gran, self.grid_gran])
                # im = ax.pcolormesh(x_grid, y_grid, d_grid, vmin=-1, vmax=1)
                dx = round(self.x_grid[1] - self.x_grid[0], 1)
                xx, yy = np.mgrid[
                        slice(self.x_lims[0], self.x_lims[1] + dx, dx),
                        slice(self.y_lims[0], self.y_lims[1] + dx, dx)]
                im = ax.pcolor(xx, yy, d_grid, vmin=-1, vmax=1)
                fig.colorbar(im)

                ax.scatter(self.real_points[:, 0], self.real_points[:, 1],
                        c='white', alpha=.05, marker="+")
                ax.scatter(generated[:, 0], generated[:, 1], color='r', alpha=0.05)
                ax.set_xlim(self.x_lims)
                ax.set_ylim(self.y_lims)
                ax.set_title("iter {}".format(it))
                fig.savefig('{}/graph_{}.png'.format(dest_graphs_dir, it))
                plt.close(fig)

                fig, ax = plt.subplots()
                scores = np.reshape(np.concatenate(
                    (grid_scores, real_scores, gen_scores), axis=0), (-1,))
                ax.plot(scores)
                fig.savefig('{}/score_{}.png'.format(dest_scores_dir, it))
                plt.close(fig)

                # if it % 100 == 0:
                if 0:
                    outputs = natural_sort(glob("{}/graph*.png".format(
                        dest_graphs_dir)))
                    attachments = " "
                    for o in outputs:
                        attachments += " -a {}".format(o)

                    os.system('echo $PWD | mutt -s "scratch epoch {}" {} {}'.format(
                        it, "momod@utexas.edu", attachments))



    def make_grid(self):
        x_grid = np.linspace(self.x_lims[0], self.x_lims[1], self.grid_gran)
        y_grid = np.linspace(self.y_lims[0], self.y_lims[1], self.grid_gran)
        grid = np.asarray([[i, j] for i in x_grid for j in y_grid])
        return grid, x_grid, y_grid
        

    def gen_data_sample(self):
        return self.real_points[np.random.choice(self.real_n,
            self.d_batch_size), :]


    def gen_z(self):
        return np.random.uniform(size=[self.fake_n, self.z_dim],
                low=-1.0, high=1.0)
        #return np.random.normal(size=[self.fake_n, self.z_dim])


def main():
    with tf.Session() as sess:
        sgan = SGAN(sess)
        sgan.train()

if __name__ == "__main__":
    main()
