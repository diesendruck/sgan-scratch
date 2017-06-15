import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from glob import glob


def natural_sort(l):
    """Helper function to sort globbed files naturally."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_2d_data(dataset, data_n, data_dim):
    """Generates nparray of 2d data.

    Args:
        dataset: String name of dataset.
        data_n: Int number of points to generate.

    Returns:
        points: Numpy array of points, of size (data_n, 2).
    """
    group1_n = data_n / 4
    group2_n = data_n - group1_n

    if dataset == "concentric":
        center = [-3, 3]
        radius_1 = 0.5
        radius_2 = 1.5
        variance = 0.1
        theta = np.concatenate(
            (np.random.uniform(0, 2 * np.pi, group1_n),
             np.random.uniform(0, 2 * np.pi, group2_n)), axis=0)
        radii = np.concatenate(
            (np.random.normal(radius_1, variance, group1_n),
             np.random.normal(radius_2, variance, group2_n)), axis=0)
        points = np.array([center[0] + radii * np.cos(theta),
                           center[1] + radii * np.sin(theta)]).transpose()

    elif dataset == "gaussian":
        center = [-3, 3]
        variance = 0.1
        points = np.random.multivariate_normal(center, np.identity(data_dim) * variance, data_n)
        points = np.asarray(points)

    elif dataset == "swissroll":
        center = [-2, 2]
        variance = 0.1
        num_rolls = 2
        max_radius = 3
        theta = np.linspace(0, 2 * np.pi * num_rolls, data_n)
        radii = (np.linspace(0, max_radius, data_n) +
                 np.random.normal(0, variance, data_n))
        points = np.array([center[0] + radii * np.cos(theta),
                           center[1] + radii * np.sin(theta)]).transpose()

    elif dataset == "smile":
        piece = np.random.choice(4, data_n)
        var_scale = .1
        points = np.zeros([data_n,2])
        for n in range(data_n):
            if   piece[n] == 0:
                points[n,:] = np.random.multivariate_normal([-4, 3], np.array([[1., 0], [0, 1.]]) * var_scale, [1])
            elif piece[n] == 1:
                points[n,:] = np.random.multivariate_normal([0, 3], np.array([[2., 0], [0, .5]]) * var_scale, [1])
            elif piece[n] == 2:
                points[n,:] = np.random.multivariate_normal([-2, 1], np.array([[.5, 0], [0, 2.]]) * var_scale, [1])
            elif piece[n] == 3:
                x = np.random.uniform(-4., 0., 1)[0]
                y = .25 * (x+2)**2 - 1
                points[n,:] = np.random.multivariate_normal([x,y], np.array([[.25,0],[0,.25]])*var_scale, [1])
        # plt.scatter(points[:,0], points[:,1]); plt.savefig('temp.png'); plt.close()

    else:
        raise ValueError('Dataset must be in ["gaussian", "concentric", "swissroll", "smile"')

    return points


# Helper function for summary output.
def round_list(l):
    return [round(i, 3) for i in l]


def email_results(graphs_dir, expt, counter):
    outputs = natural_sort(glob(
        graphs_dir + '/graph_*001.png'))
    attachments = ' '
    for o in outputs:
        attachments += ' -a {}'.format(o)

    os.system(('echo $PWD | mutt -s "sgan-began-coverage {}, '
               'epoch {}" {} {}').format(
        expt, counter, 'momod@utexas.edu', attachments))

def make_grid(x_lims, y_lims, grid_gran):
    x_grid = np.linspace(x_lims[0], x_lims[1], grid_gran)
    y_grid = np.linspace(y_lims[0], y_lims[1], grid_gran)
    grid = np.asarray([[i, j] for i in x_grid for j in y_grid])
    return grid, x_grid, y_grid


def plot_z_coverage(z_coverage, graphs_dir, counter):
    fig, ax = plt.subplots()
    points = z_coverage.eval()
    ax.scatter(points[:, 0], points[:, 1],
            c='r', alpha=.5, marker='+')
    filename = '{}/z_coverage_{}.png'.format(graphs_dir, counter)
    fig.savefig(filename)
    plt.close(fig)
