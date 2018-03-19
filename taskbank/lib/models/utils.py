from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

# ------layers-------
@slim.add_arg_scope
def add_conv_layer( *args, **kwargs ):
    # net, out_channels, kernel, stride, scope ):
    net = slim.conv2d( *args, **kwargs )
    tf.add_to_collection( tf.GraphKeys.ACTIVATIONS, net )
    if 'scope' in kwargs:
        print( '\t\t{scope}'.format( scope=kwargs['scope'] ), net.get_shape() )
    return net

@slim.add_arg_scope
def add_conv_transpose_layer( *args, **kwargs ):
    net = slim.conv2d_transpose( *args, **kwargs )
    tf.add_to_collection( tf.GraphKeys.ACTIVATIONS, net )
    if 'scope' in kwargs:
        print( '\t\t{scope}'.format( scope=kwargs['scope'] ), net.get_shape() )
    return net

@slim.add_arg_scope
def add_flatten_layer( net, batch_size, scope ):
    net = tf.reshape(net, shape=[batch_size, -1],
                        name=scope)
    print('\t\t{scope}'.format( scope=scope ), net.get_shape() )
    return net

@slim.add_arg_scope
def add_gaussian_noise_layer( input_layer, std, scope ):
    with tf.variable_scope( scope ) as sc:
        noise = tf.random_normal( shape=input_layer.get_shape(), mean=0.0, stddev=std, 
                    dtype=tf.float32 )
        print('\t\t{scope}'.format( scope=scope ), noise.get_shape() ) 
        return input_layer + noise

@slim.add_arg_scope
def add_reshape_layer( net, shape, scope ):
    net = tf.reshape(net, shape=shape, name=scope)
    print('\t\t{scope}'.format( scope=scope ), net.get_shape() )
    return net

@slim.add_arg_scope
def add_squeeze_layer( net, scope ):
    net = tf.squeeze(net, squeeze_dims=[1,2] , name=scope) # tf 0.12.0rc: squeeze_dims -> axis
    print('\t\t{scope}'.format( scope=scope ), net.get_shape() )
    return net

@slim.add_arg_scope
def add_conv_fc_layer( *args, **kwargs ):
    '''
        Sets up a FC-Conv layer using the args passed in
    '''
    net = args[ 0 ]
    pre_fc_shape = [ int( x ) for x in net.get_shape() ]
    kwargs[ 'kernel_size' ] = pre_fc_shape[1:3]
    kwargs[ 'stride' ] = 1
    kwargs[ 'padding' ] = 'VALID'
    net = add_conv_layer( *args, **kwargs )
    net = add_squeeze_layer( net, scope='squeeze' )
    tf.add_to_collection( tf.GraphKeys.ACTIVATIONS, net )
    return net

@slim.add_arg_scope
def add_fc_with_dropout_layer( net, is_training, num_outputs, dropout=0.8, activation_fn=None, reuse=None, scope=None ):
    '''
        Sets up a FC layer with dropout using the args passed in
    '''
    #print(activation_fn)
    net = slim.fully_connected(net, num_outputs,
            activation_fn=activation_fn,
            reuse=reuse,
            scope=scope)
    net = slim.dropout(net, keep_prob=dropout, is_training=is_training)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)
    if 'scope' is not None:
        print( '\t\t{scope}'.format( scope=scope ), net.get_shape() )
    return net


@slim.add_arg_scope
def add_fc_layer( net, is_training, num_outputs, activation_fn=None, reuse=None, scope=None ):
    '''
        Sets up a FC layer using the args passed in
    '''
    net = slim.fully_connected(net, num_outputs,
            activation_fn=activation_fn,
            reuse=reuse,
            scope=scope)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)
    if 'scope' is not None:
        print( '\t\t{scope}'.format( scope=scope ), net.get_shape() )
    return net


# ------activation fns-------
# These are not activation functions because they have additional parameters. 
#  However, they return an activation function with the specified parameters 
#  'baked in'. 
def leaky_relu( leak=0.2, name='leaky_relu' ):
    return lambda x: tf.maximum( x, leak * x, name='leaky_relu' )


# ------normalization fns-------

# ------utils from tf 0.12.0rc-------
if tf.__version__ == '0.10.0':
    print( "Building for Tensorflow version {0}".format( tf.__version__ ) )
    def convert_collection_to_dict(collection):
        """Returns a dict of Tensors using get_tensor_alias as key.
        Args:
            collection: A collection.
        Returns:
            A dictionary of {get_tensor_alias(tensor): tensor}
        """
        return {get_tensor_alias(t): t for t in tf.get_collection(collection)}

    def get_tensor_alias(tensor):
        """Given a tensor gather its alias, its op.name or its name.
        If the tensor does not have an alias it would default to its name.
        Args:
            tensor: A `Tensor`.
        Returns:
            A string with the alias of the tensor.
        """
        if hasattr(tensor, 'alias'):
            alias = tensor.alias
        else:
            if tensor.name[-2:] == ':0':
                # Use op.name for tensor ending in :0
                alias = tensor.op.name
            else:
                alias = tensor.name
        return alias
else:
    print( "Building for Tensorflow version {0}".format( tf.__version__ ) )
    convert_collection_to_dict = slim.utils.convert_collection_to_dict
