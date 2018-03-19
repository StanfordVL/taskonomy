''' gan_discriminators.py

    Contains discriminators that can be used for a GAN or cGAN loss
'''
from __future__ import absolute_import, division, print_function
from   models.utils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def pix2pix_discriminator( decoder_output, is_training, n_layers=3, n_channel_multiplier=64, stride=2,
            weight_decay=0.0001, activation_fn=leaky_relu( leak=0.2 ), 
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='discriminator', reuse=None, **kwargs):
    '''
        Builds the discriminator from the pix2pix paper. This doesn't not contain any dropout layers.

        Structure:
            input                       -> [ B x H x W x C ]
            conv - relu                 -> [ B x (H/2) x (W/2) x n_channel_multiplier ] 
            conv - batchnorm - relu     -> [ B x (H/4) x (W/4) x n_channel_multiplier*2^k ]
                ... Depends on n_layers 
            conv - batchnorm - relu     -> [ B x H' x W' x n_channel_multiplier*8 ]
            conv - sigmoid              -> [ B x H' x W' x n_channel_multiplier*8 ] 
            avg_pool                    -> [ B ] 

        Args:
            decoder_output: input to the discriminator. Before given to the fn, this may be concatenated
                channel-wise with the input_imgs so that the discriminator can also see the input imgs. 
            is_training: A bool
            n_layers: The number of conv_layers to use. At least 2. 
            n_channel_multiplier: Proportional to how many channels to use in the discriminator.
            weight_decay: Value to use for L2 weight regularization.
            batch_norm_decay: passed through  
            batch_norm_epsilon: passed through  
            batch_norm_scale: passed through  
            batch_norm_center: passed through  
            scope: Scope that all variables will be under
            reuse: Whether to reuse variables in this scope
            **kwargs: Allows other args to be passed in. Unused. 

        Returns: 
            The output of avg_pool.
    '''
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    print('\tbuilding discriminator')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        if reuse: 
            sc.reuse_variables()
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                outputs_collections=end_points_collection,
                weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    stride=stride):
                net = decoder_output
                print('\t\tinput', net.get_shape())
                
                # First layer: conv2d-ReLu
                net = add_conv_layer( net, n_channel_multiplier, [5, 5], scope='conv1', normalizer_fn=None )
                
                # Central layers: conv2d-batchNorm-ReLu
                for k_layer in range(1, n_layers-2 ):
                    nf_mult = min( 2**k_layer, 8 )
                    net = add_conv_layer( net, n_channel_multiplier * nf_mult, [5, 5], 
                            scope='conv{0}'.format( k_layer + 1 ) )
                
                # Last two layers: conv2d-batchnorm-relu, but with stride=1
                nf_mult = min( 2**(n_layers-1), 8 )
                net = add_conv_layer( net, n_channel_multiplier * nf_mult, [4, 4], 
                        stride=1, scope='conv{0}'.format( n_layers - 1 ) )
                net = add_conv_layer( net, 1, [4, 4], 
                        stride=1, normalizer_fn=None, 
                        activation_fn=None, scope='conv{0}'.format( n_layers ) )
                probs = tf.reduce_mean( tf.nn.sigmoid( net ), reduction_indices=[1,2,3], name='avg_pool' )
                print('\t\tlogits', net.get_shape())
                end_points = convert_collection_to_dict(end_points_collection)
                end_points[ 'probabilities' ] = probs
                return net, end_points
                

@slim.add_arg_scope
def pixelGAN_dsicriminator( decoder_output, is_training, n_channel_multiplier=64,
            weight_decay=0.0001, 
            batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='discriminator', reuse=None, **kwargs):
    '''
        Builds the a discriminator that only looks at pixels - no spatial information is used!
        Probably crappy, but appeared in the code for https://arxiv.org/pdf/1611.07004v1.pdf,
        and it's super easy to implement. 

        
        Structure:
            conv2d - ReLU               -> [H, W, n_channel_multiplier]
            conv2d - BatchNorm - ReLU   -> [H, W, 2 * n_channel_multiplier]
            conv2d                      -> [H, W, 1]

    '''
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'center': batch_norm_center,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    print('\tbuilding discriminator')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                outputs_collections=end_points_collection,
                weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d],
                    padding='VALID',
                    stride=1):
                net = decoder_output
                print('\t\tinput', net.get_shape())
                net = add_conv_layer( net, n_channel_multiplier, [1, 1], scope='conv1', normalizer_fn=None )
                net = add_conv_layer( net, n_channel_multiplier * 2, [1, 1], scope='conv2' )
                net = add_conv_layer( net, n_channel_multiplier, [1, 1], scope='conv3', 
                        normalizer_fn=None, activation_fn=tf.sigmoid )
                net = tf.reduce_mean( net, reduction_indices=[1,2,3], name='avg_pool' )
                print('\t\tprobabilities', net.get_shape())
                end_points = convert_collection_to_dict(end_points_collection)
                return net, end_points
                
