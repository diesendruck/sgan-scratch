import tensorflow as tf
import numpy as np

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
    with tf.variable_scope(scope, reuse=reuse):
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

