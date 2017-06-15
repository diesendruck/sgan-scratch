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

        # Training vars.
        self.expt = config.expt
        self.is_train = config.is_train
        self.d_per_iter = config.d_per_iter
        self.g_per_iter = config.g_per_iter
        self.c_per_iter = config.c_per_iter
        self.d_lr = config.d_lr 
        self.g_lr = config.g_lr
        self.c_lr = config.c_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.gamma_d = config.gamma_d 
        self.gamma_g = config.gamma_g
        self.lambda_k_d = config.lambda_k_d
        self.lambda_k_g = config.lambda_k_g 
        self.max_iter = config.max_iter
        self.optimizer = config.optimizer

        # Output vars.
        self.summary_step = config.summary_step
        self.checkpoint_step = config.checkpoint_step
        self.gen_step = config.gen_step
        self.gen_coverage_step = config.gen_coverage_step
        self.coverage_stop = config.coverage_stop
        self.email = config.email 
        self.email_step = config.email_step
        self.x_lims = [-7., 3.]
        self.y_lims = [-3., 7.]
        self.grid_gran = 21 
        self.grid_n = self.grid_gran ** 2
        self.grid, self.x_grid, self.y_grid = self.make_grid()

        # Network vars.
        self.d_out_dim = config.d_out_dim
        self.d_encoded_dim = config.d_encoded_dim
        self.d_layers_depth = config.d_layers_depth
        self.d_layers_width = config.d_layers_width 
        self.z_dim = config.z_dim
        self.g_out_dim = config.real_dim
        self.g_layers_depth = config.g_layers_depth
        self.g_layers_width = config.g_layers_depth 
        self.d_activations = tf.nn.elu
        self.g_activations = tf.nn.elu

        # Data vars.
        self.dataset = config.dataset  
        self.real_n = config.real_n  
        self.real_dim = config.real_dim  
        self.gen_n = config.gen_n
        self.d_batch_size = config.d_batch_size

        self.build_model()
        self.prepare_dirs_and_logging()


    def prepare_dirs_and_logging(self):
        # Set up directories based on experiment name.
        self.logs_dir = './results/{}/logs'.format(self.expt)
        self.graphs_dir = './results/{}/graphs'.format(self.expt)
        self.checkpoints_dir = './results/{}/checkpoints'.format(self.expt)
        dirs_to_make = [self.logs_dir, self.graphs_dir, self.checkpoints_dir]
        for d in dirs_to_make:
            if not os.path.exists(d):
                os.makedirs(d)
                print('Made dir: {}'.format(d))

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir + '/train',
                self.sess.graph)
        self.step = tf.Variable(0, name='step', trainable=False)
        sv = tf.train.Supervisor(logdir=self.logs_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)


    def generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse) as vs:
            x = layers.dense(z, self.g_layers_width,
                    activation=self.g_activations)
            
            for idx in range(self.g_layers_depth - 1):
                x = layers.dense(x, self.g_layers_width,
                    activation=self.g_activations)

            out = layers.dense(x, self.g_out_dim,
                activation=None)

        variables = tf.contrib.framework.get_variables(vs)
        return out 


    def autoencoder(self, x, reuse=False):
        with tf.variable_scope('autoencoder', reuse=reuse) as vs:
            x = layers.dense(x, self.d_layers_width,
                activation=self.d_activations)
            
            for idx in range(self.d_layers_depth - 1):
                x = layers.dense(x, self.d_layers_width,
                    activation=self.d_activations)

            x = layers.dense(x, self.d_encoded_dim, activation=None)

            for idx in range(self.d_layers_depth):
                x = layers.dense(x, self.d_layers_width,
                    activation=self.d_activations)

            out = layers.dense(x, self.d_out_dim,
                activation=None)
        return out    


    def encoder(self, x, reuse=False):
        with tf.variable_scope('autoencoder_enc', reuse=reuse) as vs:
            x = layers.dense(x, self.d_layers_width,
                activation=self.d_activations)
            
            for idx in range(self.d_layers_depth - 1):
                x = layers.dense(x, self.d_layers_width,
                    activation=self.d_activations)

            out = layers.dense(x, self.d_encoded_dim, activation=None)
        return out    


    def decoder(self, z, reuse=False):
        with tf.variable_scope('autoencoder_dec', reuse=reuse) as vs:
            x = layers.dense(z, self.d_layers_width,
                activation=self.d_activations)

            for idx in range(self.d_layers_depth - 1):
                x = layers.dense(x, self.d_layers_width,
                    activation=self.d_activations)

            out = layers.dense(x, self.d_out_dim,
                activation=None)
        return out    


    def build_model(self):
        self.real_points = common.load_2d_data(self.dataset, self.real_n, 
                self.real_dim)
        self.real_sample = tf.placeholder(
                tf.float64, [None, self.real_dim],
                name='real_sample')
        self.z_coverage = tf.Variable(
                tf.random_normal([self.real_n, self.z_dim], stddev=0.1,
                    dtype=tf.float64),
                name='z_coverage')
        self.z = tf.placeholder(tf.float64, [None, self.z_dim])
        self.k_d = tf.Variable(0., dtype=tf.float64, trainable=False,
                name='k_d')
        self.k_g = tf.Variable(0., dtype=tf.float64, trainable=False,
                name='k_g')

        # Compute generator and autoencoder outputs.
        self.gen = self.generator(self.z, reuse=False)
        self.ae_real = self.decoder(self.encoder(self.real_sample, reuse=False),
                reuse=False)
        self.ae_gen = self.decoder(self.encoder(self.gen, reuse=True),
                reuse=True)
        self.ae_grid = self.decoder(self.encoder(tf.convert_to_tensor(
                self.grid), reuse=True), reuse=True)

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
        self.gen_z_coverage = self.generator(self.z_coverage, reuse=True)
        self.coverage_loss = tf.reduce_mean(
                tf.abs(self.real_points - self.gen_z_coverage),
                name='coverage_mean')

        # Collect and define main losses.
        self.d_loss = self.ae_loss_real - self.k_d * self.ae_loss_gen
        #self.g_loss = self.ae_loss_gen 
        # TODO: Decide whether coverage loss belongs in g_loss.
        self.g_loss = self.ae_loss_gen + self.k_g * self.coverage_loss

        # Build optimization ops.
        self.g_vars = [
            var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [
            var for var in tf.global_variables() if 'autoencoder' in var.name]
        self.coverage_vars = [
            var for var in tf.global_variables() if 'coverage' in var.name]

        if self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer
        else:
            optimizer = tf.train.AdamOptimizer

        # Define optimization nodes.
        d_optim = optimizer(self.d_lr).minimize(self.d_loss,
                var_list=self.d_vars)
        g_optim = optimizer(self.g_lr).minimize(self.g_loss,
                var_list=self.g_vars)
        self.coverage_optim = optimizer(self.c_lr).minimize(self.coverage_loss,
                var_list=self.coverage_vars)

        self.emp_gamma_d = self.ae_loss_gen / self.ae_loss_real
        self.emp_gamma_g = self.ae_loss_gen / self.coverage_loss
        self.balance_d = self.gamma_d * self.ae_loss_real - self.ae_loss_gen
        self.balance_g = self.gamma_g * self.coverage_loss - self.ae_loss_gen
        self.measure = self.ae_loss_real + tf.abs(self.balance_d)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_d_update = tf.assign(
                self.k_d, 
                tf.clip_by_value(
                    self.k_d + self.lambda_k_d * self.balance_d, 0, 1))
            self.k_g_update = tf.assign(
                self.k_g,
                tf.clip_by_value(
                    self.k_g + self.lambda_k_g * self.balance_g, 0, 1))

        # Set up summary items.
        self.summary_op = tf.summary.merge([
            tf.summary.scalar('loss/d_loss', self.d_loss),
            tf.summary.scalar('loss/ae_loss_real', self.ae_loss_real),
            tf.summary.scalar('loss/ae_loss_gen', self.ae_loss_gen),
            tf.summary.scalar('loss/g_loss', self.g_loss),
            tf.summary.scalar('loss/coverage_loss', self.coverage_loss),
            tf.summary.scalar('balance/emp_gamma_d', self.emp_gamma_d),
            tf.summary.scalar('balance/emp_gamma_g', self.emp_gamma_g),
            tf.summary.scalar('balance/measure', self.measure),
            tf.summary.scalar('balance/k_d', self.k_d),
            tf.summary.scalar('balance/k_g', self.k_g),
            tf.summary.scalar('training/d_lr', self.d_lr),
            tf.summary.scalar('training/g_lr', self.g_lr),
        ])

        tf.global_variables_initializer().run()


    def train(self):
        # Load checkpoints, if they exist.
        self.counter = 1
        self.load_checkpoints()  

        # Run training.
        for it in range(1, self.max_iter):
            self.it = it
            for _ in range(self.d_per_iter):
                self.sess.run([self.k_d_update],
                    feed_dict={
                        self.z: self.gen_z(self.gen_n),
                        self.real_sample: self.gen_real_sample()})
            for _ in range(self.g_per_iter):
                self.sess.run([self.k_g_update],
                    feed_dict={
                        self.z: self.gen_z(self.gen_n),
                        self.real_sample: self.gen_real_sample()})
            if it < self.coverage_stop:
                for _ in range(self.c_per_iter):
                    self.sess.run(self.coverage_optim)

            # Save summary (viewable in TensorBoard).
            if it % self.summary_step == 1:
                summary_results = self.sess.run(
                    self.summary_op,
                    feed_dict = {
                        self.z: self.gen_z(self.gen_n),
                        self.real_sample: self.real_points})
                self.summary_writer.add_summary(summary_results, self.counter)
                self.summary_writer.flush()

            # Save checkpoint.
            if it % self.checkpoint_step == 1:
                self.saver.save(self.sess, 
                        os.path.join(self.checkpoints_dir, 'sgan'),
                        global_step=self.counter) 
                print(' [*] Saved checkpoint {} to {}'.format(self.counter,
                    self.checkpoints_dir))

            # Email results.
            if self.email and it % self.email_step == 1:
                self.email_results()

            # Plot generator results.
            if it % self.gen_step == 1:
                generated = self.sess.run(
                    self.gen,
                    feed_dict={
                        self.z: self.gen_z(self.gen_n)})

                self.console_summary(generated)
                self.plot_and_save_fig(generated)

            # Plot generator coverage results.
            if it % self.gen_coverage_step == 1:
                gen_z_coverage = self.sess.run(self.gen_z_coverage)
                self.plot_and_save_fig(gen_z_coverage, tag='coverage')

            self.counter += 1


    def test(self):
        could_load = self.load_checkpoints()
        if not could_load:
            raise ValueError(
                ' [!] Need to train model before testing.')

        z1, z2 = self.gen_z(2)

        interpolated_zs = np.empty((0, self.z_dim)) 
        for ratio in np.linspace(0, 1, 11):
            mix = np.reshape(ratio * z1 + (1 - ratio) * z2, (1, 2)) 
            interpolated_zs = np.append(interpolated_zs, mix, axis=0)

        gens = self.sess.run(self.gen, {self.z: interpolated_zs}) 
        self.plot_and_save_fig(gens, tag='interpolated', alpha=0.5)
        print(' [*] Plotted interpolation at {}'.format(
            self.graphs_dir)) 


    def load_checkpoints(self):
        print(' [*] Reading checkpoints') 
        ckpt = tf.train.get_checkpoint_state(self.checkpoints_dir)

        if ckpt and ckpt.model_checkpoint_path:

            # Check if user wants to continue.
            user_input = raw_input(
                'Found checkpoint {}. Proceed? (y/n) '.format(
                    ckpt.model_checkpoint_path))
            if user_input != 'y':
                raise ValueError(
                    ' [!] Cancelled. To start fresh, rm checkpoint files.')

            # Rewrite any necessary variables, based on loaded ckpt.
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoints_dir,
                ckpt_name))
            self.counter = int(next(
                re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            print(' [*] Successfully loaded {}'.format(ckpt_name))
            could_load = True
            return could_load
        else:
            print(' [!] Failed to find a checkpoint')
            could_load = False 
            return could_load


    def console_summary(self, generated):
        dl, gl, kd, kg, ae_r, ae_g, cl = round_list(
            self.sess.run(
                [self.d_loss, self.g_loss, self.k_d, self.k_g,
                 self.ae_loss_real, self.ae_loss_gen,
                 self.coverage_loss],
                feed_dict = {
                    self.z: self.gen_z(self.gen_n),
                    self.real_sample: self.gen_real_sample()}),
                2)
        rm = round_list(self.real_points.mean(0), 2)
        gm = round_list(generated.mean(0), 2)
        cm = round_list(self.z_coverage.eval().mean(0), 2)
        rv = round_list(self.real_points.var(0), 2)
        gv = round_list(generated.var(0), 2)
        cv = round_list(self.z_coverage.eval().var(0), 2)
        summ = ('it: {}, losses(d,g,c): ({}, {}, {}) '
                'means(r,g,c): ({}, {}, {}),  vars(r,g,c): ({}, {}, {}) '
                'k(d,g): ({}, {}) '
                'ae_loss(r,g): ({}, {})').format(
            self.it, dl, gl, cl, rm, gm, cm, rv, gv, cv, kd, kg, ae_r, ae_g)
        print(summ)


    def plot_and_save_fig(self, generated, tag=None, alpha=0.05):
        if tag: 
            tag_dir = os.path.join(self.graphs_dir, tag)
            if not os.path.exists(tag_dir):
                os.makedirs(tag_dir)
                print('Made dir: {}'.format(tag_dir))

        grid_scores = self.sess.run(self.ae_loss_grid_vals)

        fig, ax = plt.subplots()
        d_grid = np.reshape(grid_scores,
                [self.grid_gran, self.grid_gran])
        dx = round(self.x_grid[1] - self.x_grid[0], 1)
        xx, yy = np.mgrid[
                slice(self.x_lims[0], self.x_lims[1] + dx, dx),
                slice(self.y_lims[0], self.y_lims[1] + dx, dx)]
        im = ax.pcolor(xx, yy, d_grid, cmap='viridis_r', vmin=-1, vmax=1)
        fig.colorbar(im)

        ax.scatter(self.real_points[:, 0], self.real_points[:, 1],
                c='white', alpha=.05, marker='+')
        ax.scatter(generated[:, 0], generated[:, 1], color='r',
                alpha=alpha)
        ax.set_xlim(self.x_lims)
        ax.set_ylim(self.y_lims)

        if tag:
            title = 'counter {} {}'.format(self.counter, tag)
            filename = '{}/{}/graph_{}_{}.png'.format(self.graphs_dir, tag,
                    self.counter, tag)
        else:
            title = 'counter {}'.format(self.counter)
            filename = '{}/graph_{}.png'.format(self.graphs_dir, self.counter)
        ax.set_title(title)
        fig.savefig(filename)

        plt.close(fig)


    def email_results(self):
        outputs = natural_sort(glob('{}/graph_*.png'.format(
            self.graphs_dir)))
        attachments = ' '
        for o in outputs:
            attachments += ' -a {}'.format(o)

        os.system(('echo $PWD | mutt -s "sgan-began-coverage {}, '
                   'epoch {}" {} {}').format(
            self.expt, it, 'momod@utexas.edu', attachments))


    def make_grid(self):
        x_grid = np.linspace(self.x_lims[0], self.x_lims[1], self.grid_gran)
        y_grid = np.linspace(self.y_lims[0], self.y_lims[1], self.grid_gran)
        grid = np.asarray([[i, j] for i in x_grid for j in y_grid])
        return grid, x_grid, y_grid
        

    def gen_real_sample(self):
        return self.real_points[np.random.choice(self.real_n,
            self.d_batch_size), :]


    def gen_z(self, gen_n):
        return np.random.uniform(size=[gen_n, self.z_dim],
                low=-1.0, high=1.0)
        #return np.random.normal(size=[gen_n, self.z_dim])


def main(config):
    print '\n', config, '\n'
    with tf.Session() as sess:
        sgan = SGAN(sess, config)
        if config.is_train:
            sgan.train()
        else:
            sgan.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
