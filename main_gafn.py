import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import re
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


def natural_sort(l):
  """Helper function to sort globbed files naturally."""
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
  return sorted(l, key = alphanum_key)


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
        center = [0, 0]
        variance = 0.1
        points = np.random.multivariate_normal(
                center, np.identity(data_dim) * variance, data_n)
        points = np.asarray(points)
    elif dataset == "swissroll":
        center = [0, 0]
        variance = 0.01
        num_rolls = 2
        max_radius = 4
        theta = np.linspace(0, 2 * np.pi * num_rolls, data_n)
        radii = (np.linspace(0, max_radius, data_n) +
                 np.random.normal(0, variance))
        points = np.array([center[0] + radii * np.cos(theta),
                           center[1] + radii * np.sin(theta)]).transpose()
    return points

# Make true data.
points = load_2d_data(dataset, data_n)

# Define fuzzy dumb generator size and architecture.
f_n = data_n
f_mean = np.mean(points, 0)
f_var = np.cov(points.transpose())
f_blur = 4.   
assert f_blur > 1., "f_blur not greater than 1."
fuzzy = np.random.multivariate_normal(f_mean, f_var * f_blur, f_n)

# Define generator size and architecture.
g_n = 1000
z_dim = 3
g_layers_depth = 5
g_layers_width = 5
g_activation = tf.nn.relu
g_output = data_dim

# Define grid, for evaluation of discriminator.
grid_gran = 21
x_grid = np.linspace(x_lims[0], x_lims[1], grid_gran)
y_grid = np.linspace(y_lims[0], y_lims[1], grid_gran)
grid = np.asarray([[i, j] for i in x_grid for j in y_grid])
grid_n = len(grid)

# Define discriminator size and architecture.
# Note: it will take as input both true and generated data.
d_n = data_n + f_n + g_n
d_layers_depth = 5
d_layers_width = 10
d_activation = tf.nn.tanh
d_output = 1

# Reset graph and fix random Z input to generator.
tf.reset_default_graph()
data = tf.constant(points)
fuzz = tf.constant(fuzzy)
Znp = np.random.normal(size=[g_n, z_dim])
Z = tf.constant(Znp)

# Build generator out of several hidden layers.
HG = [None] * g_layers_depth
for l in range(g_layers_depth):
    W = tf.get_variable("G_Weights" + str(l), shape = [z_dim if l == 0 else g_layers_width, g_output if l == g_layers_depth - 1 else g_layers_width], dtype=tf.float64)
    B = tf.get_variable("G_Bias" + str(l), shape = [g_output if l == g_layers_depth - 1 else g_layers_width], dtype=tf.float64)
    if l == g_layers_depth - 1:
        H = tf.add(tf.matmul(HG[l - 1][2], W), B)

    else:
        # NOTE: Defines contents of input layer.
        H = g_activation(tf.add(tf.matmul(Z if l == 0 else HG[l - 1][2], W), B))

    HG[l] = [W, B, H]
G = HG[-1][2]

# Build discriminator out of several hidden layers.
HD = [None] * d_layers_depth
for l in range(d_layers_depth):
    W = tf.get_variable("D_Weights" + str(l), shape = [data_dim if l == 0 else d_layers_width, d_output if l == d_layers_depth - 1 else d_layers_width], dtype=tf.float64)
    B = tf.get_variable("D_Bias" + str(l), shape = [d_output if l == d_layers_depth - 1 else d_layers_width], dtype=tf.float64)

    # NOTE: Defines contents of input layer.
    H = d_activation(tf.add(tf.matmul(tf.concat([grid, data, fuzz, G], 0) if l == 0 else HD[l - 1][2], W), B))

    HD[l] = [W, B, H]
D_scores = HD[-1][2]

# Define discriminator and generator losses.
f_penalty = (f_blur - 1.)/(f_blur)  # f_blur >> 1
D_target = tf.constant([[1]]*data_n + [[-1.]]*f_n + [[-1.]]*g_n)
Dloss = tf.losses.mean_squared_error(D_scores[grid_n:], D_target, weights=[[1.]]*data_n + [[f_penalty]]*f_n + [[1.]]*g_n)
# D_target = tf.constant([[1]]*data_n + [[-1.]]*f_n)
# Dloss = tf.losses.mean_squared_error(D_scores[grid_n:-g_n], D_target)

G_target = tf.constant([[1.]]*g_n)
Gloss = tf.losses.mean_squared_error(D_scores[grid_n + data_n + f_n:], G_target)

# Build optimization ops.
d_vars = [HD[l][j] for j in range(2) for l in range(d_layers_depth)]
g_vars = [HG[l][j] for j in range(2) for l in range(g_layers_depth)]
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
            sess.run(d_train_op)
    if g_update == True:
        for _ in range(g_per_iter):
            sess.run(g_train_op)
    
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
    #if (it % 10000 == 0):
        # Get data to plot and summarize.
        generated = sess.run(G)
        
        # Run scores and unpack contents for grid, data, and gen.
        scores = sess.run(D_scores)
        grid_scores = scores[:grid_n]
        data_scores = scores[grid_n:grid_n+data_n]
        fuzz_scores = scores[grid_n+data_n:-g_n]
        gen_scores = scores[-g_n:]
        
        # Collect summary items for printing.
        summary_items = [it, round_list(sess.run([Dloss, Gloss])),
            round_list(points.mean(0)), round_list(generated.mean(0)),
            round_list(points.var(0)), round_list(generated.var(0))]
        si = summary_items
        print "iteration: {}, [d_loss, g_loss]: {}, data_mean: {}, gen_mean: {}, data_var: {}, gen_var: {}".format(si[0], si[1], si[2], si[3], si[4], si[5])
        
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
        #im = ax.pcolormesh(x_grid, y_grid, d_grid, vmin=-1, vmax=1)
        dx = round(x_grid[1] - x_grid[0], 1)
        xx, yy = np.mgrid[slice(x_lims[0], x_lims[1]+dx, dx),
                          slice(y_lims[0], y_lims[1]+dx, dx)]
        im = ax.pcolor(xx, yy, d_grid, vmin=-1, vmax=1)
        fig.colorbar(im)
        
        ax.scatter(fuzzy[:, 0],  fuzzy[:, 1], c='grey', alpha=0.3, marker='.')
        ax.scatter(points[:, 0], points[:, 1], c='white', alpha=.3, marker="+")
        ax.scatter(generated[:, 0], generated[:, 1], color='r', alpha=0.3)
        ax.scatter([-2,-2], [2,2], c='white', alpha=1, marker="+")
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_title("iter {}".format(it))
        fig.savefig('{}/graph_{}.png'.format(dest_graphs_dir, it))
        plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.plot(scores);
        fig.savefig('{}/score_{}.png'.format(dest_scores_dir, it));
        plt.close(fig)


        #if it % 100 == 0:
        if 0:
            outputs = natural_sort(glob("{}/graph*.png".format(
                dest_graphs_dir)))
            attachments = " "
            for o in outputs:
              attachments += " -a {}".format(o)

            os.system('echo $PWD | mutt -s "scratch epoch {}" {} {}'.format(
              it, "momod@utexas.edu", attachments)) 
