import tensorflow as tf
import warnings
from ffnn import ffnn

# This function creates a feed-forward partial-information network

# This function will create or reuse a sub-variable_scope, with arguments:
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
#   Returns: a tf node of the output layer

# Need to now parameters for:
#   - Input -> Latent-Unfixed
#   - Input -> Latent-Fixed
#   - [Latend-Fixed, Latent-Unfixed] -> Output

ident = lambda x:x

def ff_pin(input, encoded_values, pins, width=[[5, 5, 0], [3, 3, 0], [4, 1, 0], [3, 3, 0], [5, 5, 0], [0, 0, 2]],
           encoder_depth=3, num_layers = None, output_dim=None, activations=[tf.tanh, tf.nn.relu, ident], 
           activate_last_layer=True, scope="FF_PIN", reuse=None):
    
    # Reading some implicit parameters
    if num_layers == None:
        if isinstance(width, list):
            num_layers = len(width)
        else:
            raise 'Need to provide num_layers or width as list'
    
    if reuse == None:
        local_scope = tf.get_variable_scope().name + '/' + scope
        scope_in_use = max([obj.name[:len(local_scope)] == local_scope for obj in tf.global_variables()] + [False])
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
     
    if output_dim == None:
        output_dim == sum(layer_widths[-1])
    
    # Check for coherency of final layer if needed
    if activate_last_layer and sum(layer_widths[num_layers - 1]) != output_dim:
        print layer_widths
        raise BaseException(
            'activate_last_layer == True but implied final layer width doesn\'t match output_dim \n (implied depth: ' + str(
                sum(layer_widths[num_layers - 1])) + ', explicit depth: ' + str(output_dim) + ')')
    
    # The actual work
    with tf.variable_scope(scope, reuse=reuse):
        encoded_prediction = ffnn(input, num_layers=encoder_depth, width=layer_widths[:encoder_depth], 
                                  output_dim=sum(layer_widths[encoder_depth-1]), activations=activations, 
                                  activate_last_layer=True, scope="Encoder", reuse=reuse)
        encoded_pinned = pins * encoded_values + (1 - pins) * encoded_prediction
        output = ffnn(encoded_pinned, num_layers=num_layers-encoder_depth, width=layer_widths[encoder_depth:], 
                      output_dim=output_dim, activations=activations, activate_last_layer=activate_last_layer, 
                      scope="Decoder", reuse=reuse)
    return output, encoded_prediction
