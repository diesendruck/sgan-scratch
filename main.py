import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
import tensorflow as tf

# Constants for controlling runs.
max_iter = 1500
d_per_iter = 10
g_per_iter = 10
x_lims = [-4, 4]
y_lims = [-4, 4]

# Define real data points.
data_n = 200
data_dim = 2
center = [1.5, -1.5]
variance = 0.1
points = np.random.multivariate_normal(center, np.identity(data_dim) * variance, data_n)
points = np.asarray(points)

# Define generator size and architecture.
g_n = 200
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
d_n = data_n + g_n
d_layers_depth = 5
d_layers_width = 5
d_activation = tf.nn.tanh
d_output = 1

# Reset graph and fix random Z input to generator.
tf.reset_default_graph()
data = tf.constant(points)
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
            H = g_activation(tf.add(tf.matmul(Z if l == 0 else HG[l - 1][2], W), B))
        HG[l] = [W, B, H]
G = HG[-1][2]

# Build discriminator out of several hidden layers.
HD = [None] * d_layers_depth
for l in range(d_layers_depth):
        W = tf.get_variable("D_Weights" + str(l), shape = [data_dim if l == 0 else d_layers_width, d_output if l == d_layers_depth - 1 else d_layers_width], dtype=tf.float64)
        B = tf.get_variable("D_Bias" + str(l), shape = [d_output if l == d_layers_depth - 1 else d_layers_width], dtype=tf.float64)
        # NOTE: Defines contents of input layer.
        H = d_activation(tf.add(tf.matmul(tf.concat([grid, data, G], 0) if l == 0 else HD[l - 1][2], W), B))
        HD[l] = [W, B, H]
D_scores = HD[-1][2]

# Define discriminator and generator losses.
D_target = tf.constant([[1.]]*data_n + [[-1.]]*g_n)
G_target = tf.constant([[1.]]*g_n)
Dloss = tf.losses.mean_squared_error(D_scores[grid_n:], D_target)
Gloss = tf.losses.mean_squared_error(D_scores[grid_n + data_n:], G_target)

# Build optimization ops.
g_vars = [HG[l][j] for j in range(2) for l in range(g_layers_depth)]
d_vars = [HD[l][j] for j in range(2) for l in range(d_layers_depth)]
g_train_op = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(Gloss, var_list=g_vars)
d_train_op = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(Dloss, var_list=d_vars)

# Begin running the model.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Helper function for summary output.
def round_list(l):
    return [round(i, 3) for i in l]

for it in range(max_iter):
    # Run training.
    for _ in range(d_per_iter):
        sess.run(d_train_op)
    for _ in range(g_per_iter):
        sess.run(g_train_op)

    # For iterations, plot results.
    if it % 5 == 0 or iter == max_iter - 1:
        # Get data to plot and summarize.
        generated = sess.run(G)

        # Run scores and unpack contents for grid, data, and gen.
        scores = sess.run(D_scores)
        grid_scores = scores[:grid_n]
        data_scores = scores[grid_n:-g_n]
        gen_scores = scores[-g_n:]

        # Collect summary items for printing.
        summary_items = [l, round_list(sess.run([Dloss, Gloss])),
            round_list(points.mean(0)), round_list(generated.mean(0)),
            round_list(points.var(0)), round_list(generated.var(0))]
        si = summary_items
        print "layer: {}, [d_loss, g_loss]: {}, data_mean: {}, gen_mean: {}, data_var: {}, gen_var: {}".format(si[0], si[1], si[2], si[3], si[4], si[5])

        # Plot results.
        fig, ax = plt.subplots()
        d_grid = np.reshape(grid_scores, [grid_gran, grid_gran])
        #im = ax.pcolormesh(x_grid, y_grid, d_grid, vmin=-1, vmax=1)
        dx = round(x_grid[1] - x_grid[0], 1)
        xx, yy = np.mgrid[slice(x_lims[0], x_lims[1]+dx, dx),
                          slice(y_lims[0], y_lims[1]+dx, dx)]
        im = ax.pcolor(xx, yy, d_grid, vmin=-1, vmax=1)
        fig.colorbar(im)

        ax.scatter(points[:, 0], points[:, 1], c='cornflowerblue', alpha=0.3, marker="+")
        ax.scatter(generated[:, 0], generated[:, 1], color='r', alpha=0.3)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        fig.savefig('temp/graphs/graph_{}.png'.format(it))
        plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.plot(scores);
        fig.savefig('temp/scores/score_{}.png'.format(it));
        plt.close(fig)


