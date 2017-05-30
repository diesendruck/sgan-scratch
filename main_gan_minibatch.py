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


def ffnn(input, num_layers = 3, width = 3, output_dim = 10, activations = [tf.tanh], activate_last_layer = True, scope = "FFNN", reuse = None):
    # This function will create or reuse a sub-variable_scope to implement a feedforward neural network, with arguments:
    #   input: a tensorflow array (constant, variable, or placeholder) of shape [batch_size, input_dim]
    #   num_layers: how many layers deep the network should be
    #   width: can be:
    #       - a single integer, in which case all layers will be width * len(activations) long
    #       - a len(activations)-length list of integers, in which case all layers will have sum(width) nodes where width[a] 
    #           nodes use activations[a]
    #       - a num_layers-length list of len(activations)-length lists of integers, in case each layer l will have 
    #           sum(width[l]) nodes where width[l][a] nodes use activation[a]
    #       NOTE: if activate_last_layer is True, then the implied number of nodes for the final layer must match the 
    #           specified output_dim!
    #   output_dim: the desired dimension of each row of the output (provide a single integer)
    #   activations: a list of tensorflow functions that will transform the data at each layer.
    #   activate_last_layer: a boolean to denote whether to provide transformed or untransformed output of the final 
    #       layer.  Note that numerical stability can sometimes be improved by post-processing untransformed data.
    #   scope: character string to use as the name of the sub-variable_scope
    #   reuse: whether to re-use an existing scope or create a new one.  If left blank, will only create if necessary 
    #       and re-use otherse.
    
    # Reading some implicit figures
    batch_size, input_dim = input._shape_as_list()
    
    # If variable re-use hasn't been specified, figure out if the scope is in use and should be re-used
    if reuse == None:
        local_scope = tf.get_variable_scope().name + '/' + scope
        scope_in_use = max([obj.name[:len(local_scope)]==local_scope for obj in tf.global_variables()] + [False])
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
    # Check for coherency of final layer if needed
    if activate_last_layer and sum(layer_widths[num_layers-1]) != output_dim:
        print layer_widths
        raise BaseException('activate_last_layer == True but implied final layer width doesn\'t match output_dim \n (implied depth: ' + str(sum(layer_widths[num_layers-1])) + ', explicit depth: ' + str(output_dim) + ')')
    
    # Set-up/retrieve the appropriate nodes within scope
    with tf.variable_scope(scope):
        Ws = [tf.get_variable("W_" + str(l), shape=[input_dim if l == 0 else sum(layer_widths[l-1]), output_dim if l == num_layers - 1 else sum(layer_widths[l])], dtype=tf.float64) for l in range(num_layers)]
        Bs = [tf.get_variable("B_" + str(l), shape=[output_dim if l == num_layers - 1 else sum(layer_widths[l])], dtype=tf.float64) for l in range(num_layers)]
        Hs = [None] * num_layers
        HLs = [None] * num_layers
        for l in range(num_layers):
            HLs[l] = tf.add(tf.matmul(input if l == 0 else Hs[l-1], Ws[l]), Bs[l])
            if l < num_layers - 1 or activate_last_layer == True:
                Hs[l] = tf.concat([activations[a](HLs[l][:, sum(layer_widths[l][0:a]):sum(layer_widths[l][0:a+1])]) for a in range(len(activations))], 1)
            else:
                Hs[l] = HLs[l]
    return Hs[l]

def natural_sort(l):
    """Helper function to sort globbed files naturally."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_2d_data(dataset, data_n):
    """Generates nparray of 2d data.

    Args:
        dataset: String name of dataset.
        data_n: Int number of points to generate.

    Returns:
        points: Numpy array of points, of size (data_n, 2).
    """
    if dataset == "concentric":
        center = [-3, 3]
        radius_1 = 0.5
        radius_2 = 1.5
        variance = 0.1
        theta = np.concatenate(
            (np.random.uniform(0, 2 * np.pi, data_n / 2),
             np.random.uniform(0, 2 * np.pi, data_n / 2)), axis=0)
        radii = np.concatenate(
            (np.random.normal(radius_1, variance, data_n / 2),
             np.random.normal(radius_2, variance, data_n / 2)), axis=0)
        points = np.array([center[0] + radii * np.cos(theta),
                           center[1] + radii * np.sin(theta)]).transpose()
    elif dataset == "gaussian":
        center = [-3, 3]
        variance = 0.1
        points = np.random.multivariate_normal(
            center, np.identity(data_dim) * variance, data_n)
        points = np.asarray(points)
    elif dataset == "swissroll":
        center = [-3, 3]
        variance = 0.01
        num_rolls = 2
        max_radius = 4
        theta = np.linspace(0, 2 * np.pi * num_rolls, data_n)
        radii = (np.linspace(0, max_radius, data_n) +
                 np.random.normal(0, variance))
        points = np.array([center[0] + radii * np.cos(theta),
                           center[1] + radii * np.sin(theta)]).transpose()
    return points

##############################################################################################

# Make true data.
points = load_2d_data(dataset, data_n)

# Define generator size and architecture.
g_n = 1000
z_dim = 3
g_layers_depth = 5
g_activations = [tf.nn.elu]
g_output = data_dim
g_layers_width = [[5]] * (g_layers_depth-1) + [[g_output]]

# Define grid, for evaluation of discriminator.
grid_gran = 21
x_grid = np.linspace(x_lims[0], x_lims[1], grid_gran)
y_grid = np.linspace(y_lims[0], y_lims[1], grid_gran)
grid = np.asarray([[i, j] for i in x_grid for j in y_grid])
grid_n = len(grid)

# Define discriminator size and architecture.
# Note: it will take as input both true and generated data.
d_n = data_n + g_n
d_layers_depth = 5
d_layers_width = 10
d_activations = [tf.nn.tanh, tf.nn.relu]
d_output = 1
d_batch_size = 25

# Reset graph and create placeholders.
tf.reset_default_graph()
data_sample = tf.placeholder(tf.float64, [d_batch_size, data_dim])
Z = tf.placeholder(tf.float64, [g_n, z_dim])
def gen_data_sample():
    return points[np.random.choice(data_n, d_batch_size),:]
def gen_Z():
    return np.random.normal(size = [g_n, z_dim])

# Build generator out of several hidden layers.
gen_pars = {'num_layers':g_layers_depth, 'width':g_layers_width, 'output_dim':g_output, 'activations':g_activations, 'activate_last_layer':True, 'scope': "Generator"}
dis_pars = {'num_layers':d_layers_depth, 'width':d_layers_width, 'output_dim':d_output, 'activations':d_activations, 'activate_last_layer':False, 'scope': "Discriminator"}
G = ffnn(Z, **gen_pars)
D_scores = ffnn(tf.concat([grid, data_sample, G], 0), **dis_pars)

# Define discriminator and generator losses.
D_target = tf.constant([[1]] * d_batch_size + [[-1.]] * g_n)
D_target_real      = tf.constant([[1]] * d_batch_size)
D_target_generated = tf.constant([[1]] * g_n)
Dloss_real      = tf.losses.mean_squared_error(D_scores[grid_n:grid_n+data_n], D_target_real)
Dloss_generated = tf.losses.mean_squared_error(D_scores[-g_n:], D_target_generated)
Dloss = tf.sub(Dloss_real, Dloss_generated)
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
