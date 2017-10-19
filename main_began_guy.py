import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
from common import *   # This imports from common.py, which should be in the same directory

##############################################################################################
# BEGAN parameter setting
gamma_target = 2.
lambda_rate = 1e-3
z_n = 1000
data_batch_size = 100
encoded_dim = 4
max_iter = 50001

# Set real data generation parameters.
data_n = 1000  # Number of points in true dataset.
data_dim = 2  # Dimension of each point in true dataset.
dataset = "smile"  # One of {"gaussian", "concentric", "swissroll"}
export_directory = "smile_1"

# Set graphing parameters
x_lims = [-4., 4.]
y_lims = [-4., 4.]

# Define encoder size and architecture.
# Note: it will take as input both true and generated data.
enc_layers_depth = 4
enc_layers_width = [[10, 10], [5,5], [3,3], []]
enc_activations = [tf.nn.tanh, tf.nn.elu]
enc_output = encoded_dim
enc_activate_last_layer = False

# Define decoder size and architecture.
dec_layers_depth = 5
dec_activations = [tf.nn.tanh, tf.nn.elu]
dec_output = data_dim
dec_layers_width = [[5,5]] * (dec_layers_depth - 1) + [[dec_output]]
dec_activate_last_layer = False

# Define generator size and architecture
z_dim = 10
gen_layers_depth = 5
gen_activations = [tf.nn.tanh, tf.nn.elu]
gen_output = data_dim
gen_layers_width = 5
gen_activate_last_layer = False

# Define grid, for evaluation of discriminator.
grid_gran = 21
x_grid = np.linspace(x_lims[0], x_lims[1], grid_gran)
y_grid = np.linspace(y_lims[0], y_lims[1], grid_gran)
grid = np.asarray([[i, j] for i in x_grid for j in y_grid])
grid_n = len(grid)

##############################################################################################

# Make true data.
points = load_2d_data(dataset, data_n, data_dim)

# Reset graph and create placeholders.
tf.reset_default_graph()
data_sample = tf.placeholder(tf.float64, [data_batch_size, data_dim])
Z = tf.placeholder(tf.float64, [z_n, encoded_dim])
grid_tf = tf.constant(grid, dtype=tf.float64)
def gen_data_sample():
    return points[np.random.choice(data_n, data_batch_size), :]
def gen_Z():
    return np.random.normal(size = [z_n, encoded_dim])

# Build generator out of several hidden layers.
generator_pars = {'num_layers': gen_layers_depth, 'width': gen_layers_width, 'output_dim': gen_output, 
                  'activations': gen_activations, 'activate_last_layer': gen_activate_last_layer, 'scope': "Generator"}
encoder_pars   = {'num_layers':enc_layers_depth, 'width':enc_layers_width, 'output_dim':enc_output, 
                  'activations':enc_activations, 'activate_last_layer':enc_activate_last_layer, 'scope': "Encoder"}
decoder_pars   = {'num_layers':dec_layers_depth, 'width':dec_layers_width, 'output_dim':dec_output, 
                  'activations':dec_activations, 'activate_last_layer':dec_activate_last_layer, 'scope': "Decoder"}

generated    = ffnn(           Z, reuse = False, **generator_pars)

encoded_real = ffnn( data_sample, reuse = False, **encoder_pars)
encoded_gen  = ffnn(   generated, reuse =  True, **encoder_pars)
encoded_grid = ffnn(     grid_tf, reuse =  True, **encoder_pars)

decoded_real = ffnn(encoded_real, reuse = False, **decoder_pars)
decoded_gen  = ffnn(encoded_gen , reuse =  True, **decoder_pars)
decoded_grid = ffnn(encoded_grid, reuse =  True, **decoder_pars)

# Define discriminator and generator losses.
Err_real      = tf.losses.absolute_difference(decoded_real, data_sample) / data_batch_size
Err_generated = tf.losses.absolute_difference(decoded_gen, generated) / z_n
Err_grid      = tf.reduce_sum(tf.abs(decoded_grid - grid), 1)
Dloss_real      = Err_real
Dloss_generated = Err_generated

k = tf.placeholder(tf.float32, [1])
autoencoder_objective = Err_real - k * Err_generated
generator_objective = Err_generated

# Build optimization ops.
d_vars = [var for var in tf.global_variables() if 'coder' in var.name]
g_vars = [var for var in tf.global_variables() if 'Generator' in var.name]
d_train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(autoencoder_objective, var_list=d_vars)
g_train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(  generator_objective, var_list=g_vars)

# Begin running the model.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

k_hist = [0.] * (max_iter + 1)
for it in range(max_iter):
    # Run training.
    _, _, eg, er = sess.run([d_train_op, g_train_op, Err_generated, Err_real], 
                            feed_dict={Z: gen_Z(), data_sample: gen_data_sample(), k: np.array([k_hist[it]])})
    k_hist[it + 1] = k_hist[it] + lambda_rate * (gamma_target * er - eg)
    
    # For some iterations, plot results.
    if ((it <= 10 and it % 1 == 0) or (it < 500 and it % 25 == 0) or (it % 1000 == 0)):
        # if (it % 10000 == 0):
        # Get the generated data and scores for grid, data, and gen.
        fake_data, grid_scores, ae_obj, g_obj = sess.run(
            [generated, Err_grid, autoencoder_objective, generator_objective], 
            feed_dict={Z:gen_Z(), data_sample:gen_data_sample(), k: np.array([k_hist[it]])})
        
        # Collect summary items for printing.
        summary_items = [it, round_list([ae_obj, g_obj]),
                         round_list(points.mean(0)), round_list(fake_data.mean(0)),
                         round_list(points.var(0)), round_list(fake_data.var(0))]
        si = summary_items
        print "iteration: {}, [d_loss, g_loss]: {}, data_mean: {}, gen_mean: {}, data_var: {}, gen_var: {}".format(*si[:6])
        
        # Plot results.
        dest_graphs_dir = "./temp/{}/graphs".format(export_directory)
        dest_scores_dir = "./temp/{}/scores".format(export_directory)
        target_dirs = [dest_graphs_dir, dest_scores_dir]
        for d in target_dirs:
            if not os.path.exists(d):
                os.makedirs(d)
                print "Made dir: {}".format(d)
        
        fig, ax = plt.subplots()
        d_grid = np.reshape(grid_scores, [grid_gran, grid_gran])
        # im = ax.pcolormesh(x_grid, y_grid, d_grid, vmin=-1, vmax=1)
        dx = round(x_grid[1] - x_grid[0], 1)
        xx, yy = np.mgrid[slice(x_lims[0], x_lims[1] + dx, dx),
                          slice(y_lims[0], y_lims[1] + dx, dx)]
        im = ax.pcolor(xx, yy, d_grid, vmin=d_grid.min(), vmax=d_grid.max())
        fig.colorbar(im)
        
        ax.scatter(points[:, 0], points[:, 1], c='white', alpha=.3, marker="+")
        ax.scatter(fake_data[:, 0], fake_data[:, 1], color='r', alpha=0.3)
        ax.scatter([-2, -2], [2, 2], c='white', alpha=1, marker="+")
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_title("iter {}".format(it))
        fig.savefig('{}/graph_{}.png'.format(dest_graphs_dir, it))
        plt.close(fig)
        
        # fig, ax = plt.subplots()
        # ax.plot(scores);
        # fig.savefig('{}/score_{}.png'.format(dest_scores_dir, it));
        # plt.close(fig)
        
        # if it % 100 == 0:
        if 0:
            outputs = natural_sort(glob("{}/graph*.png".format(
                dest_graphs_dir)))
            attachments = " "
            for o in outputs:
                attachments += " -a {}".format(o)
            
            # os.system('echo $PWD | mutt -s "scratch epoch {}" {} {}'.format(
            #     it, "guywcole@utexas.edu", attachments)) 

