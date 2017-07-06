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
from models import *
from munkres import Munkres
from scipy.spatial.distance import cdist
from utils import *


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
        self.lambda_normality_loss = config.lambda_normality_loss
        self.max_iter = config.max_iter
        self.optimizer = config.optimizer
        self.training_z = config.training_z
        self.normality_dist_fn = config.normality_dist_fn

        # Output vars.
        self.tensorboard_step = config.tensorboard_step
        self.checkpoint_step = config.checkpoint_step
        self.plot_and_print_step = config.plot_and_print_step
        self.plot_z_preimage_step = config.plot_z_preimage_step
        self.email = config.email 
        self.email_step = config.email_step
        self.x_lims = [-7., 3.]
        self.y_lims = [-3., 7.]
        self.grid_gran = 21 
        self.grid_n = self.grid_gran ** 2
        self.grid, self.x_grid, self.y_grid = make_grid(
                self.x_lims, self.y_lims, self.grid_gran)

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
        self.batch_size = config.batch_size

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


    def build_model(self):
        # Define data, placeholders, and tracking variables.
        self.real_points = load_2d_data(
                self.dataset, self.real_n, self.real_dim)
        self.real_sample = tf.placeholder(
                tf.float64, [None, self.real_dim], name='real_sample')
        self.real_sample_opt_trans = tf.placeholder(
                tf.float64, [None, self.real_dim], name='real_sample_opt_trans')
        self.z = tf.placeholder(tf.float64, [None, self.z_dim], name='z')
        self.z_opt_trans = tf.placeholder(tf.float64, [None, self.z_dim],
                name='z_opt_trans')
        self.z_preimage = tf.Variable(tf.random_normal(
                [self.real_n, self.z_dim], stddev=0.1, dtype=tf.float64),
                name='z_preimage')
        self.k_d = tf.Variable(0., dtype=tf.float64, trainable=False,
                name='k_d')
        self.k_g = tf.Variable(0., dtype=tf.float64, trainable=False,
                name='k_g')

        # Compute generator and autoencoder outputs.
        self.gen_z = generator(
                self.z, self.g_layers_width, self.g_layers_depth,
                self.g_activations, self.g_out_dim, reuse=False)
        self.gen_z_opt_trans = generator(
                self.z_opt_trans, self.g_layers_width, self.g_layers_depth,
                self.g_activations, self.g_out_dim, reuse=True)
        self.gen_z_preimage = generator(self.z_preimage,
                self.g_layers_width, self.g_layers_depth,
                self.g_activations, self.g_out_dim, reuse=True)
        self.ae_real_sample = decoder(
                encoder(
                    self.real_sample, self.d_layers_width, self.d_layers_depth,
                    self.d_activations, self.d_encoded_dim, reuse=False),
                self.d_layers_width, self.d_layers_depth, self.d_activations,
                self.d_out_dim, reuse=False)
        self.ae_gen_z = decoder(
                encoder(
                    self.gen_z, self.d_layers_width, self.d_layers_depth,
                    self.d_activations, self.d_encoded_dim, reuse=True),
                self.d_layers_width, self.d_layers_depth, self.d_activations,
                self.d_out_dim, reuse=True)
        self.ae_grid = decoder(
                encoder(
                    tf.convert_to_tensor(self.grid), self.d_layers_width,
                    self.d_layers_depth, self.d_activations, self.d_encoded_dim,
                    reuse=True),
                self.d_layers_width, self.d_layers_depth, self.d_activations, 
                self.d_out_dim, reuse=True)

        # Define autoencoder losses.
        self.ae_loss_real = tf.reduce_mean(
                tf.abs(self.ae_real_sample - self.real_sample))
        self.ae_loss_gen = tf.reduce_mean(tf.abs(self.ae_gen_z - self.gen_z))

        self.ae_loss_real_vals = tf.reduce_sum(
                tf.abs(self.ae_real_sample - self.real_sample), 1)
        self.ae_loss_gen_vals = tf.reduce_sum(
                tf.abs(self.ae_gen_z - self.gen_z), 1)
        self.ae_loss_grid_vals = tf.reduce_sum(
                tf.abs(self.ae_grid - self.grid), 1)

        # Define losses.
        self.d_loss = self.ae_loss_real - self.k_d * self.ae_loss_gen
        self.normality_loss = tf.py_func(self.normality_dist, [self.z_preimage],
                tf.float64)
        self.gen_z_preimage_loss = tf.reduce_mean(
                tf.abs(self.real_points - self.gen_z_preimage)
                ) + self.lambda_normality_loss * self.normality_loss
        # Define coverage loss formulas: munkres, moments.
        self.coverage_loss = tf.reduce_mean(tf.abs( 
                self.gen_z_opt_trans - self.real_sample_opt_trans))
        self.gen_z_m1 = tf.reduce_mean(tf.pow(self.gen_z, 1), axis=0)
        self.gen_z_m2 = tf.reduce_mean(tf.pow(self.gen_z, 2), axis=0)
        self.gen_z_var = self.gen_z_m2 - tf.square(self.gen_z_m1) 
        self.real_m1 = tf.reduce_mean(tf.pow(self.real_sample, 1), axis=0)
        self.real_m2 = tf.reduce_mean(tf.pow(self.real_sample, 2), axis=0)
        self.real_var = self.real_m2 - tf.square(self.real_m1) 
        self.cvg_loss_m1 = tf.norm(self.gen_z_m1 - self.real_m1)
        self.cvg_loss_m2 = tf.norm(self.gen_z_m2 - self.real_m2)
        self.cvg_loss_var = tf.norm(self.gen_z_var - self.real_var)
        self.coverage_loss_moments = (
                self.cvg_loss_m1 + self.cvg_loss_m2 + self.cvg_loss_var)

        if self.training_z in ['preimage', 'mix']:
            self.g_loss = self.ae_loss_gen + self.k_g * (self.coverage_loss_moments +
                    self.gen_z_preimage_loss)
        else:
            #self.g_loss = self.coverage_loss
            self.g_loss = self.ae_loss_gen + self.k_g * self.coverage_loss
            #self.g_loss = self.ae_loss_gen + self.coverage_loss_moments

        # Build optimization ops.
        self.g_vars = [
            var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [
            var for var in tf.global_variables() if 'autoencoder' in var.name]
        self.preimage_vars = [
            var for var in tf.global_variables() if 'preimage' in var.name]

        if self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer
        else:
            optimizer = tf.train.AdamOptimizer

        # Define optimization nodes.
        self.d_optim = optimizer(self.d_lr).minimize(
                self.d_loss, var_list=self.d_vars)
        self.g_optim = optimizer(self.g_lr).minimize(
                self.g_loss, var_list=self.g_vars)
        self.z_optim = optimizer(self.c_lr).minimize(
                self.gen_z_preimage_loss, var_list=self.preimage_vars)

        self.emp_gamma_d = self.ae_loss_gen / self.ae_loss_real
        self.emp_gamma_g = self.ae_loss_gen / self.coverage_loss
        self.balance_d = self.gamma_d * self.ae_loss_real - self.ae_loss_gen
        self.balance_g = self.gamma_g * self.coverage_loss - self.ae_loss_gen
        self.measure = self.ae_loss_real + tf.abs(self.balance_d)

        #with tf.control_dependencies([self.d_optim, self.g_optim]):
        with tf.control_dependencies([self.d_optim]):
            self.k_d_update = tf.assign(
                self.k_d, 
                tf.clip_by_value(
                    self.k_d + self.lambda_k_d * self.balance_d, 0, 1))
        with tf.control_dependencies([self.g_optim]):
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
            tf.summary.scalar('loss/normality_loss', self.normality_loss),
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
        print("Starting training. Output depends on definitions in config.py.")

        # Run training.
        for _ in range(1, self.max_iter):
            # Choose training mode.
            if self.training_z == 'mix':
                training_z = self.mix_random_z_and_z_preimage()
            elif self.training_z == 'preimage':
                training_z = self.z_preimage.eval()
            else:
                training_z = self.get_random_z(self.real_n)

            # Run each optimization in sequence.
            for _ in range(self.d_per_iter):
                self.sess.run([self.k_d_update],
                    feed_dict={
                        self.z: training_z, 
                        self.real_sample: self.get_real_sample(self.batch_size)})

            for _ in range(self.g_per_iter):
                # Compute munkres permutation before sending to TF graph.
                z_opt_trans = self.get_random_z(self.batch_size)
                gen_z_opt_trans = self.sess.run(
                    self.gen_z_opt_trans,
                    feed_dict={
                        self.z_opt_trans: z_opt_trans})
                real_sample = self.get_real_sample(self.batch_size)
                distances = cdist(gen_z_opt_trans, real_sample).tolist()
                indices = Munkres().compute(distances)
                real_sample_opt_trans = np.array(
                        [real_sample[j] for (i, j) in indices])
                # Perform G update.
                self.sess.run([self.k_g_update],
                    feed_dict={
                        self.z: training_z, 
                        self.z_opt_trans: z_opt_trans,
                        self.real_sample_opt_trans: real_sample_opt_trans})

            if self.training_z in ['preimage', 'mix']:
                for _ in range(self.c_per_iter):
                    self.sess.run(self.z_optim)

            self.output(z_opt_trans, real_sample_opt_trans)
            self.counter += 1


    def output(self, z_opt_trans, real_sample_opt_trans):
        # Save summary (viewable in TensorBoard).
        if self.counter % self.tensorboard_step == 0:
            summary_results = self.sess.run(
                self.summary_op,
                feed_dict = {
                    self.z: self.get_random_z(self.gen_n),
                    self.z_opt_trans: z_opt_trans,
                    self.real_sample: self.real_points,
                    self.real_sample_opt_trans: real_sample_opt_trans})
            self.summary_writer.add_summary(summary_results, self.counter)
            self.summary_writer.flush()

        # Save checkpoint.
        if self.counter % self.checkpoint_step == 0:
            self.saver.save(self.sess, 
                os.path.join(self.checkpoints_dir, 'sgan'),
                global_step=self.counter) 
            print(' [*] Saved checkpoint {} to {}'.format(self.counter,
                self.checkpoints_dir))

        # Plot generator results, and print console summary.
        if self.counter % self.plot_and_print_step == 0:
            generated = self.sess.run(
                self.gen_z,
                feed_dict={
                    self.z: self.get_random_z(self.gen_n)})
            self.console_summary(generated, z_opt_trans, real_sample_opt_trans)
            self.plot_and_save_fig(generated)

        # Plot z_preimage and generator of z_preimage.
        if self.training_z in ['preimage', 'mix']:
            if self.counter % self.plot_z_preimage_step == 0:
                plot_z_preimage(self.z_preimage, self.graphs_dir,
                        self.counter, self.normality_loss.eval())
                gen_z_preimage = self.sess.run(self.gen_z_preimage)
                self.plot_and_save_fig(gen_z_preimage, tag='gen_z_preimage')

        # Email results.
        if self.email and self.counter % self.email_step == 0:
            email_results(self.graphs_dir, self.expt, self.counter)


    def test(self):
        could_load = self.load_checkpoints()
        if not could_load:
            raise ValueError(
                ' [!] Need to train model before testing.')

        z1, z2 = self.get_random_z(2)

        interpolated_zs = np.empty((0, self.z_dim)) 
        for ratio in np.linspace(0, 1, 11):
            interp = np.reshape(ratio * z1 + (1 - ratio) * z2, (1, 2)) 
            interpolated_zs = np.append(interpolated_zs, interp, axis=0)

        gens = self.sess.run(self.gen_z, {self.z: interpolated_zs}) 
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


    def console_summary(self, generated, z_opt_trans, real_sample_opt_trans):
        dl, gl, kd, kg, ae_r, ae_g, cl, nl, pil = round_list(
            self.sess.run(
                [self.d_loss, self.g_loss, self.k_d, self.k_g,
                 self.ae_loss_real, self.ae_loss_gen,
                 self.coverage_loss, self.normality_loss,
                 self.gen_z_preimage_loss],
                feed_dict = {
                    self.z: self.get_random_z(self.gen_n),
                    self.z_opt_trans: z_opt_trans,
                    self.real_sample: self.get_real_sample(self.batch_size),
                    self.real_sample_opt_trans: real_sample_opt_trans}))
        rm = round_list(self.real_points.mean(0))
        gm = round_list(generated.mean(0))
        pim = round_list(self.z_preimage.eval().mean(0))
        rv = round_list(self.real_points.var(0))
        gv = round_list(generated.var(0))
        piv = round_list(self.z_preimage.eval().var(0))
        summ = ('counter: {}, losses(d,g,cvg,norm,pre): ({}, {}, {}, {}, {}) '
                'means(r,g,cvg): ({}, {}, {}),  vars(r,g,c): ({}, {}, {}) '
                'k(d,g): ({}, {}) '
                'ae_loss(r,g): ({}, {})').format(
            self.counter, dl, gl, cl, nl, pil, rm, gm, pim, rv, gv, piv, kd, kg,
            ae_r, ae_g)
        print(summ)


    def plot_and_save_fig(self, generated, tag=None, alpha=0.1):
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
                c='white', alpha=.1, marker='+')
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


    def get_real_sample(self, batch_size):
        return self.real_points[np.random.choice(self.real_n,
            batch_size), :]


    def get_random_z(self, gen_n):
        return np.random.uniform(size=[gen_n, self.z_dim],
                low=-1.0, high=1.0)
        #return np.random.normal(size=[gen_n, self.z_dim])


    def mix_random_z_and_z_preimage(self):
        # With diminishing prob, mix, otherwise just generate using
        # get_random_z().
        OPTION = 2

        if OPTION == 1:
            is_mix = np.random.binomial(1,
                p=np.min([1, 1 - float(self.counter)/self.max_iter]))
            #if is_mix:
            if 1:
                ratio = np.random.uniform()
                z_preimage = self.sess.run(self.z_preimage)
                z_random = self.get_random_z(self.real_n)
                z = (1 - ratio) * z_random + ratio * z_preimage
            else:
                z = self.get_random_z(self.real_n) 
            return z

        elif OPTION == 2:
            z_preimage = self.sess.run(self.z_preimage)
            z_random = self.get_random_z(self.real_n)
            ratio = float(self.counter)/self.max_iter
            z = (1 - ratio) * z_preimage + ratio * z_random
            return z


    def munkres_dist(self, array1, array2):
        # See https://github.com/bmc/munkres/blob/master/munkres.py.
        distances = cdist(array1, array2).tolist()
        m = Munkres()
        indices = m.compute(distances)
        max_cost = 0
        cost = 0
        for row, col in indices:
            lowest_cost = distances[row][col]
            cost += lowest_cost
            if cost > max_cost:
                max_cost = cost
        return cost


    def normality_dist(self, test_sample):
        if self.normality_dist_fn == 'munkres':
            # See https://github.com/bmc/munkres/blob/master/munkres.py.
            target_sample = self.get_random_z(self.real_n)
            cost = munkres_dist(test_sample, target_sample)
            return cost 

        elif self.normality_dist_fn == 'stein':
            h = 1.
            n = test_sample.shape[0]
            d = test_sample.shape[1]
            max_combos = 5000


            all_combos = [(val_i, val_j) 
                    for i, val_i in enumerate(test_sample) 
                    for j, val_j in enumerate(test_sample) if i != j]
            num_combos = min(max_combos, len(all_combos))
            subset_indices = np.random.permutation(range(len(all_combos)))
            combos = [all_combos[i] for i in 
                    np.random.choice(subset_indices, num_combos, replace=False)]

            # Kernelized Stein Discrepancy, U-Statistic.
            def u_fn(x1, x2, test_sample):
                kernel = np.exp(-1. / h * sum((x1 - x2)**2))
                part1 = np.inner(-x1, -x2) * kernel 
                part2 = np.inner(-x2, kernel * (-2. / h * (x1 - x2)))
                part3 = np.inner(-x1, kernel * (2. / h * (x1 - x2)))
                part4 = [(
                    kernel * (2. / h) + 
                    (kernel * ((-2. / h) *
                        (x1 - x2)[i]) * (2. / h * (x1 - x2)[i])))
                    for i in range(d)]
                part4 = sum(part4)
                return part1 + part2 + part3 + part4

            u_sum = 0
            for c in combos:
                x1, x2 = c
                u_sum += u_fn(x1, x2, test_sample)

            dist = 1. / (n * (n-1)) * u_sum
            return dist 


def main(config):
    print '\n', config, '\n'

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf_run_config = tf.ConfigProto()
    tf_run_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_run_config) as sess:
        sgan = SGAN(sess, config)
        if config.is_train:
            sgan.train()
        else:
            sgan.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
