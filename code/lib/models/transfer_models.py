''' transfer_models.py

    Contains models that are used for for learning
'''
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

import pdb
from   models.utils import *
import models.resnet_v1 as resnet_v1

DO_NOT_REPLACE_TARGET_DECODER = False


def passthrough(
    input_placeholder,
    is_training,
    hidden_size,
    flatten_output=False,
    reuse=None,
    scope=None):
    ''' Return the representations '''
    images_placeholder, net = input_placeholder
    if flatten_output:
        print("\t\tFlattening output")
        print(net.get_shape().as_list()[0])
        net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
    return net, {}


def transfer_multilayer_conv(
    input_placeholder,
    is_training, 
    num_layers,
    hidden_size,
    output_channels,
    kernel_size=[3, 3], stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9, batch_norm_epsilon=1e-5, 
    scope="multilayer_conv", reuse=None):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    raise NotImplementedError("Use 'transfer_multilayer_conv_with_bn_ends' instead")
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }
    images_placeholder, representation_placholder = input_placeholder

    print('\t building multilayers transfer conv. Scope:', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placholder.get_shape())
            net = representation_placholder

            # Make the intermediate conv layers (output [w, h, hidden_size])
            for i in range(1, num_layers):
                net = add_conv_layer( net, hidden_size, kernel_size, 
                    stride=stride, scope='conv{0}'.format(i) )

            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv{0}'.format(num_layers) )
 
             
            # Apply residual connection
            net = net + representation_placholder

            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_multilayer_conv_with_bn_ends(
    input_placeholder,
    is_training, 
    num_layers,
    hidden_size,
    output_channels,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()
    # max_decay = 0.9
    # batch_norm_decay = max_decay - tf.train.inverse_time_decay(
    #             max_decay, 
    #             inc, 
    #             10000, 
    #             1., 
    #             staircase=False, name=None)
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }
    images_placeholder, representation_placholder = input_placeholder

    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', input_placeholder.get_shape())
        
        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs  = tf.layers.batch_normalization(
            input_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            name="tranfer_bn_in")
            # inputs = input_placeholder

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            # Identity connection            
            net = inputs

            # Make the intermediate conv layers (output [w, h, hidden_size])
            for i in range(1, num_layers):
                net = add_conv_layer( net, hidden_size, kernel_size,
                    stride=stride, scope='conv{0}'.format(i) )

            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer(
                    net, output_channels, kernel_size,
                    stride=stride,
                    activation_fn=None,
                    scope='conv{0}'.format(num_layers) )

            # Apply residual connection
            # net = net + inputs

            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points



def transfer_multilayer_conv_with_bn_input(
    input_placeholder,
    is_training, 
    num_layers,
    hidden_size,
    output_channels,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        net = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")
            # inputs = input_placeholder

        # if not is_training:


        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            # Make the intermediate conv layers (output [w, h, hidden_size])
            for i in range(1, num_layers):
                net = add_conv_layer( net, hidden_size, kernel_size,
                    stride=stride, scope='conv{0}'.format(i) )

            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer(
                    net, output_channels, kernel_size,
                    stride=stride,
                    activation_fn=None,
                    scope='conv{0}'.format(num_layers) )

            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def side_encoder( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, scope='conv_64_1' )

            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, scope='conv_64_2' )

            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=2, scope='conv_32_1' )

            # 32x32 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, scope='conv_32_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=2, scope='conv_16_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=1, scope='conv_16_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, scope='conv_16_3' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_with_dilated_conv( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, rate=2, scope='conv_64_1' )

            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, rate=2,  scope='conv_64_2' )

            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=2, scope='conv_32_1' )

            # 32x32 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, rate=2, scope='conv_32_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=2, scope='conv_16_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=1, rate=2, scope='conv_16_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, rate=2, scope='conv_16_3' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l8( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, scope='conv_64_1' )

            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, scope='conv_64_2' )

            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=2, scope='conv_32_1' )

            # 32x32 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, scope='conv_32_2' )

            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, scope='conv_32_3' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=2, scope='conv_16_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=1, scope='conv_16_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, scope='conv_16_3' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points


def side_encoder_with_dilated_conv_l8( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, rate=3, scope='conv_64_1' )

            net = add_conv_layer( net, max(3, output_hidden_size//4), kernel_size, 
                stride=1, rate=3,  scope='conv_64_2' )

            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=2, scope='conv_32_1' )

            # 32x32 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, rate=2, scope='conv_32_2' )

            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, rate=2, scope='conv_32_3' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=2, scope='conv_16_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=1, rate=2, scope='conv_16_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, rate=2, scope='conv_16_3' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l4( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=2, scope='conv_32_1' )

            # 32x32 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, scope='conv_32_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=2, scope='conv_16_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, scope='conv_16_2' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points


def side_encoder_with_dilated_conv_l4( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=2, scope='conv_32_1' )

            # 32x32 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, rate=2, scope='conv_32_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=2, scope='conv_16_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, rate=2, scope='conv_16_2' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l2( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, scope='conv_64_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=4, scope='conv_16_1' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l2_fat( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                stride=1, scope='conv_64_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size*2, kernel_size, 
                activation_fn=None,
                stride=4, scope='conv_16_1' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l2_slim( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size//4, kernel_size, 
                stride=1, scope='conv_64_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                activation_fn=None,
                stride=4, scope='conv_16_1' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l2_deep( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, scope='conv_64_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=4, scope='conv_16_1' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, scope='conv_16_2' )

            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=1, scope='conv_16_3' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_l2_shallow( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 --> 16x16
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=4, scope='conv_16_1' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def side_encoder_with_dilated_conv_l2( 
    input_image,
    downsample_shape,
    output_hidden_size,
    is_training,
    n_layers=3,
    kernel_size=[3, 3],
    interpolation_method=tf.image.ResizeMethod.BILINEAR,
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    scope='side_encoder',
    reuse=None
):
    ''' An side encoder to perform some operations on pixels '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }

    print('\tbuilding side encoder')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        input_image = tf.image.resize_images(
            input_image,
            downsample_shape,
            method=interpolation_method,
            align_corners=False
        )

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput', input_image.get_shape())
            net = input_image
            # 64x64 block
            net = add_conv_layer( net, output_hidden_size//2, kernel_size, 
                stride=1, rate=2, scope='conv_64_1' )

            # 16x16 block
            net = add_conv_layer( net, output_hidden_size, kernel_size, 
                activation_fn=None,
                stride=4, scope='conv_16_1' )
            end_points = convert_collection_to_dict(end_points_collection)    
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            net = inputs
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_1')
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_2')

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0_pix_only(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [pixel_encoding, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0_no_image(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.,
    first_fc=False,
    resize_zoom=False):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        
        end_points = {}
        if is_training:
            # Clip extreme values
            if first_fc and not resize_zoom:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0],
                    name='input_moments',
                    keep_dims=True
                )
            else:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0,1,2],
                    name='input_moments',
                    keep_dims=True
                )

            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            
        end_points['clipped_reps'] = representation_placeholder

        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")
        
        end_points['first_bn'] = inputs

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())


            if first_fc and not resize_zoom:
                inputs = add_fc_layer(inputs, is_training, 2048, scope='first_fc')
                inputs = tf.reshape(inputs, [-1, 16, 16, 8])
            elif resize_zoom:
                inputs = tf.transpose(inputs, [0,3,2,1])
                inputs = tf.image.resize_images(inputs, [16,16])

            net = inputs
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_1')
            end_points['net1_1_output'] = net

            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_2')
            end_points['net1_2_output'] = net
            net2 = inputs
            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_1')
            end_points['net2_1_output'] = net2
            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_2')
            end_points['net2_2_output'] = net2

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, net2],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )
            end_points['net_output'] = net

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            end_points['after_last_bn'] = net
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            # end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0_no_image_fat(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.,
    first_fc=False,
    resize_zoom=False):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        
        end_points = {}
        if is_training:
            # Clip extreme values
            if first_fc and not resize_zoom:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0],
                    name='input_moments',
                    keep_dims=True
                )
            else:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0,1,2],
                    name='input_moments',
                    keep_dims=True
                )

            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            
        end_points['clipped_reps'] = representation_placeholder

        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")
        
        end_points['first_bn'] = inputs

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())


            if first_fc and not resize_zoom:
                inputs = add_fc_layer(inputs, is_training, 2048, scope='first_fc')
                inputs = tf.reshape(inputs, [-1, 16, 16, 8])
            elif resize_zoom:
                inputs = tf.transpose(inputs, [0,3,2,1])
                inputs = tf.image.resize_images(inputs, [16,16])

            net = inputs
            net = add_conv_layer(net, output_channels*2, kernel_size,
                    stride=stride, scope='rep_conv_1')
            end_points['net1_1_output'] = net

            net = add_conv_layer(net, output_channels*2, kernel_size,
                    stride=stride, scope='rep_conv_2')
            end_points['net1_2_output'] = net
            net2 = inputs
            net2 = add_conv_layer(net2, output_channels*2, kernel_size,
                    stride=stride, scope='second_rep_conv_1')
            end_points['net2_1_output'] = net2
            net2 = add_conv_layer(net2, output_channels*2, kernel_size,
                    stride=stride, scope='second_rep_conv_2')
            end_points['net2_2_output'] = net2

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, net2],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )
            end_points['net_output'] = net

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            end_points['after_last_bn'] = net
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            # end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0_no_image_slim(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.,
    first_fc=False,
    resize_zoom=False):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        
        end_points = {}
        if is_training:
            # Clip extreme values
            if first_fc and not resize_zoom:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0],
                    name='input_moments',
                    keep_dims=True
                )
            else:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0,1,2],
                    name='input_moments',
                    keep_dims=True
                )

            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            
        end_points['clipped_reps'] = representation_placeholder

        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")
        
        end_points['first_bn'] = inputs

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())


            if first_fc and not resize_zoom:
                inputs = add_fc_layer(inputs, is_training, 2048, scope='first_fc')
                inputs = tf.reshape(inputs, [-1, 16, 16, 8])
            elif resize_zoom:
                inputs = tf.transpose(inputs, [0,3,2,1])
                inputs = tf.image.resize_images(inputs, [16,16])

            net = inputs
            net = add_conv_layer(net, output_channels/2, kernel_size,
                    stride=stride, scope='rep_conv_1')
            end_points['net1_1_output'] = net

            net = add_conv_layer(net, output_channels/2, kernel_size,
                    stride=stride, scope='rep_conv_2')
            end_points['net1_2_output'] = net
            net2 = inputs
            net2 = add_conv_layer(net2, output_channels/2, kernel_size,
                    stride=stride, scope='second_rep_conv_1')
            end_points['net2_1_output'] = net2
            net2 = add_conv_layer(net2, output_channels/2, kernel_size,
                    stride=stride, scope='second_rep_conv_2')
            end_points['net2_2_output'] = net2

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, net2],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )
            end_points['net_output'] = net

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            end_points['after_last_bn'] = net
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            # end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0_no_image_deep(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.,
    first_fc=False,
    resize_zoom=False):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        
        end_points = {}
        if is_training:
            # Clip extreme values
            if first_fc and not resize_zoom:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0],
                    name='input_moments',
                    keep_dims=True
                )
            else:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0,1,2],
                    name='input_moments',
                    keep_dims=True
                )

            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            
        end_points['clipped_reps'] = representation_placeholder

        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")
        
        end_points['first_bn'] = inputs

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())


            if first_fc and not resize_zoom:
                inputs = add_fc_layer(inputs, is_training, 2048, scope='first_fc')
                inputs = tf.reshape(inputs, [-1, 16, 16, 8])
            elif resize_zoom:
                inputs = tf.transpose(inputs, [0,3,2,1])
                inputs = tf.image.resize_images(inputs, [16,16])

            net = inputs
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_1')
            end_points['net1_1_output'] = net

            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_2')
            end_points['net1_2_output'] = net

            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_3')
            end_points['net1_3_output'] = net

            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_4')
            end_points['net1_4_output'] = net

            net2 = inputs
            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_1')
            end_points['net2_1_output'] = net2

            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_2')
            end_points['net2_2_output'] = net2
            
            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_3')
            end_points['net2_3_output'] = net2

            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_4')
            end_points['net2_4_output'] = net2
            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, net2],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )
            end_points['net_output'] = net

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            end_points['after_last_bn'] = net
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            # end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l0_no_image_shallow(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.,
    first_fc=False,
    resize_zoom=False):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        
        end_points = {}
        if is_training:
            # Clip extreme values
            if first_fc and not resize_zoom:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0],
                    name='input_moments',
                    keep_dims=True
                )
            else:
                mean, variance = tf.nn.moments(
                    representation_placeholder,
                    [0,1,2],
                    name='input_moments',
                    keep_dims=True
                )

            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            
        end_points['clipped_reps'] = representation_placeholder

        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")
        
        end_points['first_bn'] = inputs

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())


            if first_fc and not resize_zoom:
                inputs = add_fc_layer(inputs, is_training, 2048, scope='first_fc')
                inputs = tf.reshape(inputs, [-1, 16, 16, 8])
            elif resize_zoom:
                inputs = tf.transpose(inputs, [0,3,2,1])
                inputs = tf.image.resize_images(inputs, [16,16])

            net = inputs
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_1')
            end_points['net1_1_output'] = net

            net2 = inputs
            net2 = add_conv_layer(net2, output_channels, kernel_size,
                    stride=stride, scope='second_rep_conv_1')
            end_points['net2_1_output'] = net2

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, net2],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )
            end_points['net_output'] = net

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            end_points['after_last_bn'] = net
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            # end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points




def transfer_two_stream_with_bn_ends_l2l0_no_image_trans(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())


            net = inputs
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_1')
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_2')

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points



def transfer_two_stream_with_bn_ends_l2l2(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            net = inputs
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_1')
            net = add_conv_layer(net, output_channels, kernel_size,
                    stride=stride, scope='rep_conv_2')

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [net, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv2' )
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l2l2_pix_only(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [pixel_encoding, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv2' )
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l1_pix_only(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [pixel_encoding, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l3_pix_only(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [pixel_encoding, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )

            net = add_conv_layer( net, hidden_size, kernel_size, 
                    stride=stride, scope='conv2' )
 
            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv3' )
            
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l6_pix_only(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )
            
            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [pixel_encoding, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            for i in range(1, 6):
                net = add_conv_layer( net, hidden_size, kernel_size, 
                    stride=stride, scope='conv{0}'.format(i) )
 
            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv6' )
            
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l1(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [inputs, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )

            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l3(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [inputs, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )

            net = add_conv_layer( net, hidden_size, kernel_size, 
                    stride=stride, scope='conv2' )
 
            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv3' )
            
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends_l6(
    input_placeholder,
    is_training, 
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    renorm=False,
    renorm_momentum=0.8,
    renorm_clipping=None,
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None,
    clip_extreme_inputs_sigma=5.):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    images_placeholder, representation_placeholder = input_placeholder
    if not (type(batch_norm_decay) == float):
        batch_norm_decay = batch_norm_decay()

    if renorm \
        and renorm_clipping \
        and not(type(renorm_clipping) == float):
        renorm_clipping = renorm_clipping()

    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training,
        'renorm': renorm,
        'renorm_decay': renorm_momentum,
        'renorm_clipping': renorm_clipping
    }
    # pdb.set_trace()
    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print('\t\tinput with size:', representation_placeholder.get_shape())
        

        if is_training:
            # Clip extreme values
            mean, variance = tf.nn.moments(
                representation_placeholder,
                [0,1,2],
                name='input_moments',
                keep_dims=True
            )
            std = tf.sqrt(variance)
            representation_placeholder = tf.clip_by_value(
                representation_placeholder,
                mean - clip_extreme_inputs_sigma * std,
                mean + clip_extreme_inputs_sigma * std,
                name='clip_extreme_inputs'
            )
            


        # Apply BN on input in order to do label whitening
        print("\t\tApplying bn to input")
        inputs = tf.layers.batch_normalization(
            representation_placeholder,
            axis=-1,
            momentum=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            training=is_training,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            name="tranfer_bn_in")

        print('\t building multilayers transfer conv', scope)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )
            
            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [inputs, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            for i in range(1, 6):
                net = add_conv_layer( net, hidden_size, kernel_size, 
                    stride=stride, scope='conv{0}'.format(i) )
 
            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv6' )
            
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points

def transfer_two_stream_with_bn_ends(
    input_placeholder,
    is_training, 
    num_layers,
    hidden_size,
    output_channels,
    side_encoder_func=side_encoder_l2,
    kernel_size=[3, 3],
    stride=1,
    weight_decay=0.0001, 
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5, 
    flatten_output=False,
    scope="multilayer_conv",
    reuse=None):
    ''' Make a network of shape:

        Input: [w, h, c]
        for i in range(num_layers):
            conv_i: [w, h, hidden_size]
        Output conv: [w, h, output_channels]
    '''
    batch_norm_params = {
        'center': True, 'scale': True, 
        'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
        'is_training': is_training }
    images_placeholder, representation_placeholder = input_placeholder

    print('\t building multilayers transfer conv', scope)
    with tf.variable_scope(scope, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(weight_decay) ):
            print('\t\tinput with size:', representation_placeholder.get_shape())

            # Apply BN on input in order to do label whitening
            print("\t\tApplying bn to input")
            inputs  = tf.layers.batch_normalization(
                representation_placeholder,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)

            pixel_encoding, side_encoder_end_points = side_encoder_func(
                images_placeholder,
                (64, 64),
                output_channels,
                is_training,
                weight_decay=weight_decay,
                batch_norm_decay=batch_norm_decay,
                batch_norm_epsilon=batch_norm_epsilon
            )

            # Make the intermediate conv layers (output [w, h, hidden_size])
            net = tf.concat(
                [inputs, pixel_encoding],
                axis=-1,
                name='concat_reps_and_pix_encoding'
            )

            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv1' )

            for i in range(2, num_layers):
                net = add_conv_layer( net, hidden_size, kernel_size, 
                    stride=stride, scope='conv{0}'.format(i) )
 
            # Make the output conv layers (output [w, h, output_channels])
            net = add_conv_layer( net, output_channels, kernel_size, 
                    stride=stride, scope='conv{0}'.format(num_layers) )
            
            # Apply residual connection?
            # net = net + inputs

            # Apply BN on the outputs as a form of label whitening
            print("\t\tApplying bn to output")
            net = tf.layers.batch_normalization(
                net,
                axis=-1,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                training=is_training)
            if flatten_output:
                print("\t\tFlattening output")
                print(net.get_shape().as_list()[0])
                net = add_flatten_layer( net, net.get_shape().as_list()[0], scope='transfer/squeeze' )
            end_points = convert_collection_to_dict(end_points_collection)
            return net, end_points


################################################
# DECODERS FOR TRANSFER
################################################

def decoder_tiny_transfer_8_slim( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    num_layers = 8
                    channel_size = 32 
                    num_conv_deconv = num_layers - 4
                    num_deconv_only = 4 - num_conv_deconv
                    for i in range(num_conv_deconv):
                        net = add_conv_layer( net, channel_size, [3,3], stride=1, scope='conv_{i}'.format(i=i) )
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i) )
                    for i in range(num_deconv_only):
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i+num_conv_deconv) )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_8_slimmer( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    num_layers = 8
                    channel_size = 16 
                    num_conv_deconv = num_layers - 4
                    num_deconv_only = 4 - num_conv_deconv
                    for i in range(num_conv_deconv):
                        net = add_conv_layer( net, channel_size, [3,3], stride=1, scope='conv_{i}'.format(i=i) )
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i) )
                    for i in range(num_deconv_only):
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i+num_conv_deconv) )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_8_slimmest( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    num_layers = 8
                    channel_size = 8 
                    num_conv_deconv = num_layers - 4
                    num_deconv_only = 4 - num_conv_deconv
                    for i in range(num_conv_deconv):
                        net = add_conv_layer( net, channel_size, [3,3], stride=1, scope='conv_{i}'.format(i=i) )
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i) )
                    for i in range(num_deconv_only):
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i+num_conv_deconv) )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points


def decoder_tiny_transfer_1( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    net = add_conv_transpose_layer( net, 64, [16, 16], stride=16, scope='deconv_0' )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_1_upsample( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = tf.image.resize_images(
                        encoder_output,
                        (128,128),
                        align_corners=False
                    )
                    net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_0' )
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_2_upsample( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = tf.image.resize_images(
                        encoder_output,
                        (64, 64),
                        align_corners=False
                    )
                    net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_0' )
                    net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_1' )
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_2( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    net = add_conv_transpose_layer( net, 64, [4, 4], stride=4, scope='deconv_0' )
                    net = add_conv_transpose_layer( net, 64, [4, 4], stride=4, scope='deconv_1' )
            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_4( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    num_layers = 4
                    num_conv_deconv = num_layers - 4
                    num_deconv_only = 4 - num_conv_deconv
                    for i in range(num_conv_deconv):
                        net = add_conv_layer( net, 64, [3,3], stride=1, scope='conv_{i}'.format(i=i) )
                        net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_{i}'.format(i=i) )
                    for i in range(num_deconv_only):
                        net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_{i}'.format(i=i+num_conv_deconv) )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_6( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    num_layers = 6
                    num_conv_deconv = num_layers - 4
                    num_deconv_only = 4 - num_conv_deconv
                    for i in range(num_conv_deconv):
                        net = add_conv_layer( net, 64, [3,3], stride=1, scope='conv_{i}'.format(i=i) )
                        net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_{i}'.format(i=i) )
                    for i in range(num_deconv_only):
                        net = add_conv_transpose_layer( net, 64, [3, 3], scope='deconv_{i}'.format(i=i+num_conv_deconv) )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_tiny_transfer_8( encoder_output, is_training, num_output_channels=3, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            weight_decay=0.0001, scope='decoder', reuse=None ):
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
                    net = encoder_output
                    num_layers = 8
                    channel_size = 64
                    num_conv_deconv = num_layers - 4
                    num_deconv_only = 4 - num_conv_deconv
                    for i in range(num_conv_deconv):
                        net = add_conv_layer( net, channel_size, [3,3], stride=1, scope='conv_{i}'.format(i=i) )
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i) )
                    for i in range(num_deconv_only):
                        net = add_conv_transpose_layer( net, channel_size, [3, 3], scope='deconv_{i}'.format(i=i+num_conv_deconv) )

            net = add_conv_layer( net, num_output_channels, [3, 3], stride=1, 
                    normalizer_fn=None, activation_fn=tf.tanh, scope='decoder_output' )
        end_points = convert_collection_to_dict(end_points_collection)
        return net, end_points

def decoder_full(
            encoder_output, is_training, num_output_channels=3, 
            weight_decay=0.0001, dropout_keep_prob=None, activation_fn=tf.nn.relu,
            batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_center=True, 
            scope='decoder_full', reuse=None):

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
