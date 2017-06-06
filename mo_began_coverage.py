import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pprint 
import re
import warnings
import sys
import tensorflow as tf
layers = tf.layers
from config import get_config
from glob import glob
from utils import *
import common


class SGAN(object):
    def __init__(self, sess, config):
        """Basic attributes of the simple GAN.
        
        Args:
          sess: TensorFlow Session
          config: Configuration from config.py 

        """
        self.sess = sess

        self.d_lr = config.d_lr 
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.gamma_d = config.gamma_d 
        self.gamma_g = config.gamma_g
        self.lambda_k_d = config.lambda_k_d
        self.lambda_k_g = config.lambda_k_g 

        self.max_iter = config.max_iter
        self.optimizer = config.optimizer

        self.real_n = config.real_n  
        self.real_dim = config.real_dim  
        self.gen_n = config.gen_n


        self.d_out_dim = config.d_out_dim
        self.d_layers_depth = config.d_layers_depth
        self.d_layers_width = config.d_layers_width 
        self.d_activations = tf.nn.elu
        self.d_batch_size = config.d_batch_size

        self.z_dim = config.z_dim
        self.g_out_dim = config.real_dim
        self.g_layers_depth = config.g_layers_depth
        self.g_layers_width = config.g_layers_depth 
        self.g_activations = tf.nn.elu
        
        self.x_lims = [-6., 2.]
        self.y_lims = [-2., 6.]
        self.grid_gran = 21 
        self.grid_n = self.grid_gran ** 2
        self.grid, self.x_grid, self.y_grid = self.make_grid()

        self.dataset = config.dataset  
        self.expt = config.expt

        self.build_model()

        self.log_dir = "./logs/{}".format(config.expt)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_dir)
        self.step = tf.Variable(0, name='step', trainable=False)
        sv = tf.train.Supervisor(logdir=self.log_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)


    def generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse) as vs:
            x = layers.dense(z, self.g_layers_width,
                    activation=self.g_activations)
            
            for idx in range(self.g_layers_depth - 1):
                x = layers.dense(x, self.g_layers_width,
                    activation=self.g_activations)

            out = layers.dense(x, self.g_out_dim,
                activation=None)

        variables = tf.contrib.framework.get_variables(vs)
        return out, variables    


    def autoencoder(self, x, reuse=False):
        with tf.variable_scope("autoencoder", reuse=reuse) as vs:
            x = layers.dense(x, self.d_layers_width,
                activation=self.d_activations)
            
            for idx in range(self.d_layers_depth - 1):
                x = layers.dense(x, self.d_layers_width,
                    activation=self.d_activations)

            x = layers.dense(x, 2, activation=self.d_activations)

            for idx in range(self.d_layers_depth):
                x = layers.dense(x, self.d_layers_width,
                    activation=self.d_activations)

            out = layers.dense(x, self.d_out_dim,
                activation=None)

        variables = tf.contrib.framework.get_variables(vs)
        return out, variables    


    def build_model(self):
        self.real_points = common.load_2d_data(self.dataset, self.real_n, 
                self.real_dim)
        self.real_sample = tf.placeholder(
                tf.float64, [self.d_batch_size, self.real_dim],
                name='real_sample')
        self.z_coverage = tf.Variable(
                tf.random_normal([self.real_n, self.real_dim], stddev=0.1,
                    dtype=tf.float64),
                name='z_coverage')
        self.z = tf.placeholder(tf.float64, [self.real_n, self.z_dim])
        self.k_d = tf.Variable(0., dtype=tf.float64, trainable=False, name='k_d')
        self.k_g = tf.Variable(0., dtype=tf.float64, trainable=False, name='k_g')

        # Set params for network architectures.
        self.gen, _ = self.generator(self.z, reuse=False)
        self.ae_real, _ = self.autoencoder(self.real_sample, reuse=False)
        self.ae_gen, _ = self.autoencoder(self.gen, reuse=True)
        self.ae_grid, _ = self.autoencoder(
                tf.convert_to_tensor(self.grid), reuse=True)

        # Define autoencoder losses.
        self.ae_loss_real = tf.reduce_mean(tf.abs(self.ae_real - self.real_sample))
        self.ae_loss_gen = tf.reduce_mean(tf.abs(self.ae_gen - self.gen))

        self.ae_loss_real_vals = tf.reduce_sum(
                tf.abs(self.ae_real - self.real_sample), 1)
        self.ae_loss_gen_vals = tf.reduce_sum(
                tf.abs(self.ae_gen - self.gen), 1)
        self.ae_loss_grid_vals = tf.reduce_sum(
                tf.abs(self.ae_grid - self.grid), 1)

        # Define coverage losses.
        self.coverage_loss = tf.reduce_mean(tf.abs(self.real_points - self.gen))

        # Collect and define main losses.
        self.d_loss = self.ae_loss_real - self.k_d * self.ae_loss_gen
        self.g_loss = self.ae_loss_gen #+ self.k_g * self.coverage_loss

        # Build optimization ops.
        self.g_vars = [
            var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [
            var for var in tf.global_variables() if 'autoencoder' in var.name]

        if self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer
        else:
            optimizer = tf.train.AdamOptimizer

        d_optim = optimizer(self.d_lr).minimize(self.d_loss,
                var_list=self.d_vars)
        g_optim = optimizer(self.g_lr).minimize(self.g_loss,
                var_list=self.g_vars)

        self.balance_d = self.gamma_d * self.ae_loss_real - self.ae_loss_gen
        self.balance_g = self.gamma_g * self.coverage_loss - self.ae_loss_gen
        self.measure = self.ae_loss_real + tf.abs(self.balance_d)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_d_update = tf.assign(
                self.k_d, 
                tf.clip_by_value(
                    self.k_d + self.lambda_k_d * self.balance_d, 0, 1))
            #self.k_g_update = tf.assign(
            #    self.k_g,
            #    tf.clip_by_value(
            #        self.k_g + self.lambda_k_g * self.balance_g, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/ae_loss_gen", self.ae_loss_gen),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/coverage_loss", self.coverage_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_d", self.k_d),
            tf.summary.scalar("misc/k_g", self.k_g),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

        tf.global_variables_initializer().run()


    def train(self, config):

        for it in range(self.max_iter):
            # Run training.
            self.sess.run(self.k_d_update,
                feed_dict={
                    self.z: self.gen_z(),
                    self.real_sample: self.gen_real_sample()})
            #self.sess.run(
            #    [self.k_d_update, self.k_g_update],
            #    feed_dict={
            #        self.z: self.gen_z(),
            #        self.real_sample: self.gen_real_sample()})
            #pdb.set_trace()

            # For iterations, plot results.
            if ((it <= 10 and it % 1 == 0) or
                    (it < 500 and it % 25 == 0) or
                    (it % 1000 == 0)):
                # Get data to plot and summarize.
                generated = self.sess.run(
                    self.gen,
                    feed_dict={
                        self.z: self.gen_z(),
                        self.real_sample: self.gen_real_sample()})

                # Run scores and unpack contents for grid, data, and gen.
                grid_scores, real_scores, gen_scores = self.sess.run(
                    [self.ae_loss_grid_vals,
                     self.ae_loss_real_vals, 
                     self.ae_loss_gen_vals],
                    feed_dict = {
                        self.z: self.gen_z(),
                        self.real_sample: self.gen_real_sample()})

                # Collect summary items for printing.
                summary_items = [
                        it,
                        round_list(
                            self.sess.run(
                                [self.d_loss, self.g_loss, self.k_d, self.k_g,
                                 self.ae_loss_real, self.ae_loss_gen,
                                 self.coverage_loss],
                                feed_dict = {
                                    self.z: self.gen_z(),
                                    self.real_sample: self.gen_real_sample()}),
                                2),
                        round_list(self.real_points.mean(0), 2),
                        round_list(generated.mean(0), 2),
                        round_list(self.real_points.var(0), 2),
                        round_list(generated.var(0), 2)]
                si = summary_items
                print ("it: {}, losses(d,g,c): ({}, {}, {}) "
                       "means(r,g): ({}, {}),  vars(r,g): ({}, {}) "
                       "k(d,g): ({}, {}) "
                       "ae_loss(r,g): ({}, {})").format(
                    si[0], si[1][0], si[1][1], si[1][6],
                    si[2], si[3], si[4], si[5],
                    si[1][2], si[1][3], si[1][4], si[1][5])
                       
                # Summary writer.
                #summary_results = self.sess.run(self.summary_op,
                #    feed_dict = {
                #        self.z: self.gen_z(),
                #        self.real_sample: self.real_points})
                #self.summary_writer.add_summary(summary_results, it)
                #self.summary_writer.flush()

                # Plot results.
                dest_graphs_dir = "./results/{}/graphs".format(self.expt)
                dest_scores_dir = "./results/{}/scores".format(self.expt)
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
        

    def gen_real_sample(self):
        return self.real_points[np.random.choice(self.real_n,
            self.d_batch_size), :]


    def gen_z(self):
        return np.random.uniform(size=[self.gen_n, self.z_dim],
                low=-1.0, high=1.0)
        #return np.random.normal(size=[self.gen_n, self.z_dim])


def main(config):
    pp = pprint.PrettyPrinter()
    pp.pprint(config)

    with tf.Session() as sess:
        sgan = SGAN(sess, config)
        sgan.train(config)

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
