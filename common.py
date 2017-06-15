import tensorflow as tf
import numpy as np
from ffnn import ffnn
from ff_pin import ff_pin

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
        points = np.random.multivariate_normal(center, np.identity(data_dim) * variance, data_n)
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
    elif dataset == "smile":
        piece = np.random.choice(4, data_n)
        var_scale = .1
        points = np.zeros([data_n,2])
        for n in range(data_n):
            if   piece[n] == 0:
                points[n,:] = np.random.multivariate_normal([-2,2], np.array([[1.,0],[0,1.]])*var_scale, [1])
            elif piece[n] == 1:
                points[n,:] = np.random.multivariate_normal([ 2,2], np.array([[2.,0],[0,.5]])*var_scale, [1])
            elif piece[n] == 2:
                points[n,:] = np.random.multivariate_normal([ 0,0], np.array([[.5,0],[0,2.]])*var_scale, [1])
            elif piece[n] == 3:
                x = np.random.uniform(-3., 3., 1)[0]
                y = .25 * x**2 - 2
                points[n,:] = np.random.multivariate_normal([x,y], np.array([[.25,0],[0,.25]])*var_scale, [1])
        # plt.scatter(points[:,0], points[:,1]); plt.savefig('temp.png'); plt.close()
    return points

# Helper function for summary output.
def round_list(l):
    return [round(i, 3) for i in l]

