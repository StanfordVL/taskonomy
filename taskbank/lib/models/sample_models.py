''' sample_models.py

    Contains models that are used for for learning
'''
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.utils import *
import models.resnet_v1 as resnet_v1

def resnet_encoder_with_avgpool( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=True,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        #net = add_conv_fc_layer( net, hidden_size, normalizer_fn=None, activation_fn=None, scope='encoder/output' )
        net = add_squeeze_layer( net, scope='encoder/squeeze' )
        if final_fc:
            net = add_fc_layer(net, is_training, hidden_size, activation_fn=None, scope='encoder/fc') 
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def resnet_encoder( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=False,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        net = add_conv_fc_layer( net, hidden_size, normalizer_fn=None, activation_fn=None, scope='encoder/output' )  
        if final_fc:
            net = add_fc_layer(net, is_training, hidden_size, activation_fn=None, scope='encoder/fc') 
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def resnet_encoder_fully_convolutional_16x16x4( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, flatten=False, batch_size=32, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=False,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        net = add_conv_layer( net, 4, [3, 3], stride=1, scope='compress1' )
        if flatten:
            net = add_flatten_layer( net, batch_size, scope='encoder/squeeze' )
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def resnet_encoder_fully_convolutional_16x16x8( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, flatten=False, batch_size=32, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=False,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        net = add_conv_layer( net, 8, [3, 3], stride=1, scope='compress1' )
        if flatten:
            net = add_flatten_layer( net, batch_size, scope='encoder/squeeze' )
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def resnet_encoder_fully_convolutional_8x8x16( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, flatten=False, batch_size=32, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=False,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        net = add_conv_layer( net, 16, [3, 3], stride=1, scope='compress1' )
        if flatten:
            net = add_flatten_layer( net, batch_size, scope='encoder/squeeze' )
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def resnet_encoder_fully_convolutional_8x8x16_1( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=False,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        net = add_conv_layer( net, 16, [1, 1], stride=1, scope='compress1' )
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def resnet_encoder_fully_convolutional_1024( input_placeholder, is_training, resnet_build_fn, hidden_size=4096,reuse=None, 
    output_stride=None, weight_decay=0.0001, batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, final_fc=False, **kwargs):
    '''
        Makes a resnet encoder using the resnet_build_fn. Sets up tf-slim scope and 
        passes in relevant arguments. 
    '''
    # end_points_collection = 'encoder_end_points'
    with slim.arg_scope( resnet_v1.resnet_arg_scope( is_training,
                weight_decay=weight_decay, batch_norm_decay=batch_norm_decay, 
                batch_norm_epsilon=batch_norm_epsilon, batch_norm_scale=batch_norm_scale) ): 
    #         outputs_collections=end_points_collection):
        net, end_points = resnet_build_fn(inputs=input_placeholder,
                    num_classes=None,
                    global_pool=False,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope='encoder')
        net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='compress1' )
        end_points[ 'encoder_output' ] = net      
        return net, end_points

def encoder_tiny( input_placeholder, is_training, hidden_size=4096,
                            weight_decay=0.0001, scope='encoder', reuse=None ):
    ''' An encoder with only one FC layer to hidden_size '''
    print('\tbuilding encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_placeholder.get_shape())
            net = add_conv_fc_layer( input_placeholder, hidden_size, activation_fn=None, scope='fc1' )    
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def encoder_multilayers_fc(input_placeholder, is_training, 
                        layer_num, hidden_size, output_size,
                        weight_decay=0.0001, scope="three_layer_fc_network", dropout=0.5, reuse=None):
    ''' An encoder with three FC layers with every but last FC layer
        output to hidden_size, the final FC layer will have no
        acitvation instead of relu for other layers'''

    print('\t building multilayers FC encoder', scope)

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                #weights_regularizer=slim.l2_regularizer(weight_decay) ):
                weights_regularizer=slim.l2_regularizer(weight_decay)):

            print('\t\tinput with size:', input_placeholder.get_shape())
            net = input_placeholder

            # FC layer 1~(i-1)
            for i in range(layer_num - 1):
                net = add_fc_with_dropout_layer(net, is_training, hidden_size, activation_fn=tf.nn.relu, dropout=dropout, scope='fc'+str(i))

            # Last FC layer
            net = add_fc_layer(net, is_training, output_size, activation_fn=None, scope='fc'+str(layer_num)) 
        
            # Softmax Activation
            #net = slim.softmax(net, scope='predictions') 

            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def encoder_multilayers_fc_bn(input_placeholder, is_training, 
                        layer_num, hidden_size, output_size,
                        weight_decay=0.0001, scope="three_layer_fc_network", dropout=0.5, reuse=None, batch_norm_decay=0.9,
                        batch_norm_epsilon=1e-5, batch_norm_scale=True,batch_norm_center=True, initial_dropout=False):
    ''' An encoder with three FC layers with every but last FC layer
        output to hidden_size, the final FC layer will have no
        acitvation instead of relu for other layers'''

    print('\t building multilayers FC encoder', scope)
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                #weights_regularizer=slim.l2_regularizer(weight_decay) ):
                weights_regularizer=slim.l2_regularizer(weight_decay),
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):

            print('\t\tinput with size:', input_placeholder.get_shape())
            net = input_placeholder
            if initial_dropout:
                net = tf.layers.dropout(
                        net,
                        rate=1.-dropout,
                        training=is_training)
            # FC layer 1~(i-1)
            for i in range(layer_num - 1):
                net = add_fc_with_dropout_layer(net, is_training, hidden_size, activation_fn=tf.nn.relu, dropout=dropout, scope='fc'+str(i))

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                 activation_fn=None,
                 normalizer_fn=None,
                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                 weights_regularizer=slim.l2_regularizer(weight_decay) ):
            # Last FC layer
            net = add_fc_layer(net, is_training, output_size, activation_fn=None, scope='fc'+str(layer_num)) 
        
            # Softmax Activation
            #net = slim.softmax(net, scope='predictions') 

            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def encoder_multilayers_fc_bn_res(input_placeholder, is_training, 
                        layer_num, hidden_size, output_size,
                        batch_norm_decay=0.95, batch_norm_epsilon=1e-5, 
                        weight_decay=0.0001, scope="three_layer_fc_network", dropout=0.8, reuse=None):
    ''' An encoder with three FC layers with every but last FC layer
        output to hidden_size, the final FC layer will have no
        acitvation instead of relu for other layers'''
    batch_norm_params = {'center': True,
                         'scale': True,
                         'decay': batch_norm_decay,
                         'epsilon': batch_norm_epsilon,
                         'is_training': is_training}

    print('\t building multilayers FC encoder', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):

            print('\t\tinput with size:', input_placeholder.get_shape())
            net = input_placeholder

            # FC layer 1~(i-1)
            for i in range(layer_num - 1):
                if dropout < 1.0:
                    is_training_dropout = False
                    net = add_fc_with_dropout_layer(net, is_training_dropout, hidden_size,
                        activation_fn=tf.nn.relu, dropout=dropout, scope='fc'+str(i))
                else:
                    net = add_fc_layer(net, is_training, hidden_size, activation_fn=tf.nn.relu, dropout=dropout, scope='fc'+str(i))

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            # Last FC layer
            net = add_fc_layer(net, is_training, output_size, activation_fn=None, scope='fc'+str(layer_num)) 
            
            # Make residual connection
            net = net + input_placeholder

            # Softmax Activation
            #net = slim.softmax(net, scope='predictions') 

            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points


def encoder_multilayers_fc_bn_res_no_dropout(input_placeholder, is_training, 
                        layer_num, hidden_size, output_size,
                        weight_decay=0.0001, scope="three_layer_fc_network", dropout=0.8,
                         batch_norm_decay=0.9, batch_norm_epsilon=1e-5, 
                         reuse=None):
    ''' An encoder with three FC layers with every but last FC layer
        output to hidden_size, the final FC layer will have no
        acitvation instead of relu for other layers'''
    batch_norm_params = {'center': True,
                         'scale': True,
                        'decay': batch_norm_decay,
                        'epsilon': batch_norm_epsilon,
                         'is_training': is_training}

    print('\t building multilayers FC encoder', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):

            print('\t\tinput with size:', input_placeholder.get_shape())
            net = input_placeholder

            # FC layer 1~(i-1)
            for i in range(layer_num - 1):
                    net = add_fc_layer(net, is_training, hidden_size, activation_fn=tf.nn.relu, scope='fc'+str(i))

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            # Last FC layer
            net = add_fc_layer(net, is_training, output_size, activation_fn=None, scope='fc'+str(layer_num-1)) 
            
            # Make residual connection
            net = net + input_placeholder

            # Softmax Activation
            #net = slim.softmax(net, scope='predictions') 

            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points


def encoder_multilayers_fc_bn_res_no_dropout_normalize_input(input_placeholder, is_training, 
                        layer_num, hidden_size, output_size,
                        weight_decay=0.0001, scope="three_layer_fc_network", dropout=0.8,
                         batch_norm_decay=0.9, batch_norm_epsilon=1e-5, 
                         reuse=None):
    ''' An encoder with three FC layers with every but last FC layer
        output to hidden_size, the final FC layer will have no
        acitvation instead of relu for other layers'''
    batch_norm_params = {'center': True,
                         'scale': True,
                        'decay': batch_norm_decay,
                        'epsilon': batch_norm_epsilon,
                         'is_training': is_training}

    print('\t building multilayers FC encoder', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):

            inputs  = tf.layers.batch_normalization(
                input_placeholder,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)

            print('\t\tinput with size:', input_placeholder.get_shape())
            net = inputs

            # FC layer 1~(i-1)
            for i in range(layer_num - 1):
                    net = add_fc_layer(net, is_training, hidden_size, activation_fn=tf.nn.relu, scope='fc'+str(i))

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            # Last FC layer
            net = add_fc_layer(net, is_training, output_size, activation_fn=None, scope='fc'+str(layer_num-1)) 
            
            # Make residual connection
            net = net + inputs

            # Softmax Activation
            #net = slim.softmax(net, scope='predictions') 

            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

 
def encoder_fully_convolutional_11_layer(input_placeholder, is_training, hidden_size=4096,
                            weight_decay=None, scope='encoder', reuse=None):
    '''Sample encoder that takes in 128x128 images
    128x128 input size
    ReLU activation except for encoding/latent representation
    Currently uses xavier initialization
    Architecture:
        conv 3x3 stride 2 (output will be 64x64x64)
        conv 3x3 stride 1 (output will be 64x64x64)
        conv 3x3 stride 1 (output will be 64x64x64)
        conv 3x3 stride 2 (output will be 32x32x128)
        conv 3x3 stride 1 (output will be 32x32x128)
        conv 3x3 stride 2 (output will be 16x16x256)
        conv 3x3 stride 1 (output will be 16x16x256)
        conv 3x3 stride 2 (output will be 8x8x512)
        conv 3x3 stride 1 (output will be 8x8x512)
        conv 3x3 stride 2 (output will be 4x4x1024)
        conv 3x3 stride 1 (output will be 4x4x1024)
        reshape to 16384
    '''
    print('\tbuilding encoder')
    batch_norm_params = {'center': True,
                         'scale': True,
                         'is_training': is_training}
                         # 'outputs_collections': tf.GraphKeys.ACTIVATIONS}
    with tf.variable_scope(scope, reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            net = input_placeholder
            print('\t\tinput', net.get_shape())

            # conv1-3
            net = add_conv_layer( net, 64, [3, 3], stride=2, scope='conv1' )
            net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv2' )
            net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv3' )
            # conv 4-11
            for i in range(4, 12, 2):
                out_channels = 64*(2**(i//3))
                net = add_conv_layer( net, out_channels, [3, 3], stride=2, scope='conv{0}'.format(i) )
                net = add_conv_layer( net, out_channels, [3, 3], stride=1, scope='conv{0}'.format(i+1) )
            # fc12
            pre_fc_shape = [ int( x ) for x in net.get_shape() ]
            net = add_conv_layer( net, hidden_size, pre_fc_shape[1:3], stride=1, padding='VALID', activation_fn=None, scope='fc12' )
            net = add_squeeze_layer( net, scope='squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points


def decoder_tiny( encoder_output, is_training, num_output_channels=3,
            weight_decay=0.0001, scope='decoder', reuse=None ):
    ''' An encoder with only one FC layer to hidden_size 

        Args:
            encoder_output: A Tensor
            is_training: Used for batchnorm
            num_output_channels: The number of filters
            scope: Naming for this part of the graph
            reuse: Whether to share weights with previous calls of this fn
        
        Returns:
            A (B, H, W, num_output_channels) output Tensor
    '''
    print('\tbuilding decoder')
    batch_norm_params = {'center': True,
                         'scale': True,
                         'is_training': is_training}
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope( [slim.conv2d, slim.conv2d_transpose,
                    slim.fully_connected],
                    outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                        slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):

                with slim.arg_scope([slim.conv2d_transpose], stride=2):
                    encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                    print('\t\tinput', encoder_shape )
                    net = tf.reshape( encoder_output, 
                            shape=( encoder_shape[0], 1, 1, encoder_shape[1] ),
                            name='decoder_reshape' )
                    net = add_conv_layer( net, 8192, [1,1], stride=1, scope='fc1' )
                    net = add_reshape_layer( net, [-1, 8, 8, 128], scope='rehape_to_img')
                    net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv2' )
                    net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv3' )
                    net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv4' )
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points


def decoder_fully_convolutional_11_layer(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
    '''10000 dimension input vector
    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512]
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256]
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 5x5 kernel, stride 2, output shape [32, 32, 128]
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 5x5 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 5x5 kernel, stride 2, output shape [128, 128, 32]
        conv2d 5x5 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {'center': True,
                         'scale': True,
                         'is_training': is_training}
                         # 'outputs_collections': tf.GraphKeys.ACTIVATIONS}

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=utils.leaky_relu( leak=0.2 ),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer( weight_decay ),
                            weights_initializer=slim.variance_scaling_initializer() ):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                
                net = tf.reshape( encoder_output, 
                        shape=( encoder_shape[0], 1, 1, encoder_shape[1] ),
                        name='decoder_reshape' )
                net = add_conv_layer( net, 8192, [1,1], stride=1, scope='fc1' )
                net = add_reshape_layer( net, [-1, 4, 4, 512], scope='rehape_to_img')
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv3' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv4' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_transpose_layer( net, 128, [3, 3], scope='deconv6' )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv12' )
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                        slim.fully_connected],
                    outputs_collections=end_points_collection,
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    weights_initializer=slim.variance_scaling_initializer()):
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_256_resolution_fully_convolutional_16x16_segmentation(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv1' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv2' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv4' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv6' )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv13' )
                #net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv14' )        
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points


def decoder_fc_15_layer_256_resolution_fully_convolutional_8x8_segmentation(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv1' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv2' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv4' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv6' )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv13' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv14' )        
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_256_resolution_fully_convolutional_8x8x16(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv1' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv2' )
                net = add_conv_transpose_layer( net, 512, [3, 3], scope='deconv4' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv6' )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv13' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv14' )        
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_256_resolution_fully_convolutional_1024(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv3' )
                net = add_conv_transpose_layer( net, 512, [3, 3], scope='deconv4' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv6' )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv13' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv14' )        
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_256_resolution_fully_convolutional_16x16x4(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv3' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv4' )
#                 net = add_conv_transpose_layer( net, 512, [3, 3], scope='deconv4' )
                # if dropout_keep_prob:
                    # net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv6' )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv13' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv14' )        
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points


def decoder_fc_15_layer_256_resolution_extra_convolutional_16x16x4(
            encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv3' )
#                 net = add_conv_transpose_layer( net, 512, [3, 3], scope='deconv4' )
                # if dropout_keep_prob:
                    # net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv4' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv5' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_6/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv7' )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv8' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv9' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv10' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 16, [3, 3], stride=1, scope='conv13' )
                net = add_conv_layer( net, 16, [3, 3], stride=1, scope='conv14' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv15' )        
                net = add_conv_layer( net, 16, [3, 3], stride=1, scope='conv16' )
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points


def decoder_fc_15_layer_256_resolution(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                
                net = tf.reshape( encoder_output, 
                        shape=( encoder_shape[0], 1, 1, encoder_shape[1] ),
                        name='decoder_reshape' )
                net = add_conv_layer( net, 8192, [1,1], stride=1, scope='fc1' )
                net = add_reshape_layer( net, [-1, 4, 4, 512], scope='rehape_to_img')
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv3' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv4' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_transpose_layer( net, 128, [3, 3], scope='deconv6' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_6/dropout'  )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv8' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 64, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 32, [3, 3], scope='deconv10' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv12' )
                net = add_conv_layer( net, 32, [3, 3], stride=1, scope='conv13' )
                net = add_conv_transpose_layer( net, 16, [3, 3], scope='deconv14' )        
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_64_resolution_8x8(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv3' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv4' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv6' )
                net = add_conv_transpose_layer( net, 512, [3, 3], scope='deconv6' )
                
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_6/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv7' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv8' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv10' )               
                
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_10/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv12' )               

            net = add_conv_layer( net, num_output_channels, [1, 1], stride=1, 
                    normalizer_fn=None, activation_fn=None, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_64_resolution_16x16(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 1024, [3, 3], stride=1, scope='conv3' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv4' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv5' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv6' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv_extra' )
#                 net = add_conv_transpose_layer( net, 512, [3, 3], scope='deconv6' )
                # if dropout_keep_prob:
                    # net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_6/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv7' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv8' )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv10' )               
                
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_10/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv11' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv12' )               

            net = add_conv_layer( net, num_output_channels, [1, 1], stride=1, 
                    normalizer_fn=None, activation_fn=None, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_fc_15_layer_64_resolution(encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder', reuse=None, **kwargs):

    '''8000 dimension input vector
    dropout: A float between 0 and 1.

    128x128 output size
    Architecture:
        fc to 8192
        reshape to batch_size x 4 x 4 x 512
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        conv2d 3x3 kernel, stride 1, output shape [4, 4, 512]
        deconv2d 3x3 kernel, stride 2, output shape [8, 8, 512] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [8, 8, 512]
        deconv2d 3x3 kernel, stride 2, output shape [16, 16, 256] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [16, 16, 256]
        deconv2d 3x3 kernel, stride 2, output shape [32, 32, 128] - dropout?
        conv2d 3x3 kernel, stride 1, output shape [32, 32, 128]
        deconv2d 3x3 kernel, stride 2, output shape [64, 64, 64]
        conv2d 3x3 kernel, stride 1, output shape [64, 64, 64]
        deconv2d 3x3 kernel, stride 2, output shape [128, 128, 32]
        conv2d 3x3 kernel, stride 1, output shape [128, 128, 3]
        tanh
    '''
    print('\tbuilding decoder')
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,
                             slim.fully_connected],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            outputs_collections=end_points_collection,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                net = encoder_output
                
                encoder_shape = [ int( x ) for x in encoder_output.get_shape() ]
                print('\t\tinput', net.get_shape())
                
                net = tf.reshape( encoder_output, 
                        shape=( encoder_shape[0], 1, 1, encoder_shape[1] ),
                        name='decoder_reshape' )
                net = add_conv_layer( net, 8192, [1,1], stride=1, scope='fc1' )
                net = add_reshape_layer( net, [-1, 4, 4, 512], scope='rehape_to_img')
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv2' )
                net = add_conv_layer( net, 512, [3, 3], stride=1, scope='conv3' )
                net = add_conv_transpose_layer( net, 256, [3, 3], scope='deconv4' )
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_4/dropout'  )
                net = add_conv_layer( net, 256, [3, 3], stride=1, scope='conv5' )
                net = add_conv_transpose_layer( net, 128, [3, 3], scope='deconv6' )
                
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_6/dropout'  )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv7' )
                net = add_conv_transpose_layer( net, 128, [3, 3], scope='deconv8' )               
                
                if dropout_keep_prob:
                    net = tf.nn.dropout( net , keep_prob=0.5, name='deconv_8/dropout'  )
                net = add_conv_layer( net, 128, [3, 3], stride=1, scope='conv9' )
                net = add_conv_transpose_layer( net, 128, [3, 3], scope='deconv10' )               

            net = add_conv_layer( net, num_output_channels, [1, 1], stride=1, 
                    normalizer_fn=None, activation_fn=None, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points
