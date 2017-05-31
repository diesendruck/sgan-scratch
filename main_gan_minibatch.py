import os, pdb, re, warnings, sys, matplotlib, copy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
from common import *   # This imports from common.py, which should be in the same directory

##############################################################################################

# Set configuration parameters.
max_iter = 50001
d_per_iter = 1
g_per_iter = 2
x_lims = [-6., 0.]
y_lims = [0., 6.]
d_update = True
g_update = True if d_update == False else False
data_n = 1000  # Number of points in true dataset.
data_dim = 2  # Dimension of each point in true dataset.
dataset = "gaussian"  # One of {"gaussian", "concentric", "swissroll"}
expt = "test_gaussian"

# Define generator size and architecture.
g_n = 1000
z_dim = 3
g_layers_depth = 5
g_activations = [tf.nn.tanh, tf.nn.elu]
g_output = data_dim
g_layers_width = [[5]] * (g_layers_depth-1) + [[g_output]]

# Define discriminator size and architecture.
# Note: it will take as input both true and generated data.
d_n = data_n + g_n
d_layers_depth = 5
d_layers_width = 5
d_activations = [tf.nn.tanh, tf.nn.elu]
d_output = 1
d_batch_size = 25

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
data_sample = tf.placeholder(tf.float64, [d_batch_size, data_dim])
Z = tf.placeholder(tf.float64, [g_n, z_dim])
def gen_data_sample():
    return points[np.random.choice(data_n, d_batch_size),:]
def gen_Z():
    return np.random.normal(size = [g_n, z_dim])

# Build generator out of several hidden layers.
gen_pars = {'num_layers':g_layers_depth, 'width':g_layers_width, 'output_dim':g_output, 'activations':g_activations, 'activate_last_layer':False, 'scope': "Generator"}
dis_pars = {'num_layers':d_layers_depth, 'width':d_layers_width, 'output_dim':d_output, 'activations':d_activations, 'activate_last_layer':False, 'scope': "Discriminator"}
G = ffnn(Z, **gen_pars)
D_output = ffnn(tf.concat([grid, data_sample, G], 0), **dis_pars)
D_scores = tf.nn.tanh(D_output)

# Define discriminator and generator losses.
D_target = tf.constant([[1]] * d_batch_size + [[-1.]] * g_n)
D_target_real      = tf.constant([[1]] * d_batch_size)
D_target_generated = tf.constant([[1]] * g_n)
Dloss_real      = tf.losses.mean_squared_error(D_scores[grid_n:grid_n+d_batch_size], D_target_real)
Dloss_generated = tf.losses.mean_squared_error(D_scores[-g_n:], D_target_generated)
Dloss = Dloss_real - Dloss_generated
Gloss = Dloss_generated

# Build optimization ops.
d_vars = [var for var in tf.global_variables() if 'Discriminator' in var.name]
g_vars = [var for var in tf.global_variables() if 'Generator' in var.name]
d_train_op = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(Dloss, var_list=d_vars)
g_train_op = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(Gloss, var_list=g_vars)

# Begin running the model.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)


# Helper function for summary output.
def round_list(l):
    return [round(i, 3) for i in l]


for it in range(max_iter):
    # Run training.
    if d_update == True:
        for _ in range(d_per_iter):
            sess.run(d_train_op, feed_dict={Z:gen_Z(), data_sample:gen_data_sample()})
    if g_update == True:
        for _ in range(g_per_iter):
            sess.run(g_train_op, feed_dict={Z:gen_Z(), data_sample:gen_data_sample()})
    
    if d_update == True:
        g_update = True
        d_update = False
    elif g_update == True:
        g_update = False
        d_update = True
    
    # For iterations, plot results.
    if ((it <= 10 and it % 1 == 0) or
            (it < 500 and it % 25 == 0) or
            (it % 1000 == 0)):
        # if (it % 10000 == 0):
        # Get data to plot and summarize.
        generated = sess.run(G, feed_dict={Z:gen_Z(), data_sample:gen_data_sample()})
        
        # Run scores and unpack contents for grid, data, and gen.
        scores = sess.run(D_scores, feed_dict={Z:gen_Z(), data_sample:gen_data_sample()})
        grid_scores = scores[:grid_n]
        data_scores = scores[grid_n:grid_n + data_n]
        gen_scores = scores[-g_n:]
        
        # Collect summary items for printing.
        summary_items = [it, round_list(sess.run([Dloss, Gloss], feed_dict={Z:gen_Z(), data_sample:gen_data_sample()})),
                         round_list(points.mean(0)), round_list(generated.mean(0)),
                         round_list(points.var(0)), round_list(generated.var(0))]
        si = summary_items
        print "iteration: {}, [d_loss, g_loss]: {}, data_mean: {}, gen_mean: {}, data_var: {}, gen_var: {}".format(
            si[0], si[1], si[2], si[3], si[4], si[5])
        
        # Plot results.
        dest_graphs_dir = "./temp/{}/graphs".format(expt)
        dest_scores_dir = "./temp/{}/scores".format(expt)
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
        im = ax.pcolor(xx, yy, d_grid, vmin=-1, vmax=1)
        fig.colorbar(im)
        
        ax.scatter(points[:, 0], points[:, 1], c='white', alpha=.3, marker="+")
        ax.scatter(generated[:, 0], generated[:, 1], color='r', alpha=0.3)
        ax.scatter([-2, -2], [2, 2], c='white', alpha=1, marker="+")
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_title("iter {}".format(it))
        fig.savefig('{}/graph_{}.png'.format(dest_graphs_dir, it))
        plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.plot(scores);
        fig.savefig('{}/score_{}.png'.format(dest_scores_dir, it));
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
