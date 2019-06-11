'''Losses used in encoder/decoder and d2 encoder.

    from: KChen @ https://github.com/kchen92/joint-representation/blob/master/lib/losses.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_gan_loss( discriminator_predictions_real, discriminator_predictions_fake, 
    real_label=1.0, fake_label=0.0, epsilon=1e-7, scope=None, self=None ):
    '''
        Returns the loss from the output of a discriminator
        Warnings: When building the train_op, make sure to have them update
            only the discriminator/generator variables as appropriate!

        Args: 
            discriminator_predictions_real: A Tensor of [batch_size,] of discriminator 
                results on real data
            discriminator_predictions_fake: A Tensor of [batch_size,] of discriminator 
                results on fake data
            real_label: The label to use for real images
            fake_label: Label to use for fake images
            scope: The scope tht al variables will be declared under

        Returns:
            generator_loss, discriminator_loss_real, discriminator_loss_fake
    '''
    if scope is not None:
        scope = scope + '_gan_loss'
    else: 
        scope = ''

    # create labels
    labels_real = tf.constant( real_label, dtype=tf.float32, 
                shape=discriminator_predictions_real.get_shape(), name='real_labels' )
    labels_fake = tf.constant( fake_label, dtype=tf.float32, 
                shape=discriminator_predictions_fake.get_shape(), name='fake_labels' )
    self.labels_real = labels_real
    self.labels_fake = labels_fake
    with tf.variable_scope( scope ) as sc:
        # raise NotImplementedError('make log loss')
        loss_d_real = slim.losses.sigmoid_cross_entropy( discriminator_predictions_real, labels_real, 
                    scope='discriminator/real' )
        loss_d_fake = slim.losses.sigmoid_cross_entropy( discriminator_predictions_fake, labels_fake, 
                    scope='discriminator/fake' )
        loss_g = slim.losses.sigmoid_cross_entropy( discriminator_predictions_fake, 1. - labels_fake, 
                    scope='generator' ) # Generator should make images look real

        # loss_d_real = slim.losses.log_loss( discriminator_predictions_real, labels_real, 
        #             epsilon=epsilon, scope='discriminator/real' )
        # loss_d_fake = slim.losses.log_loss( discriminator_predictions_fake, labels_fake, 
        #             epsilon=epsilon, scope='discriminator/fake' )
        # loss_g = slim.losses.log_loss( discriminator_predictions_fake, 1. - labels_fake, # Generator should make images look real
        #             epsilon=epsilon, scope='generator' )
    return loss_g, loss_d_real, loss_d_fake
    
# Common L-norm losses
def get_l1_loss(predictions, targets, scope=None):
    '''Return sum of squares loss after masking.
    '''
    if scope is not None:
        scope = scope + '_abs_diff_loss'
    return slim.losses.absolute_difference( predictions, targets, scope=scope )

def get_l1_loss_with_mask(predictions, targets, mask, scope=None):
    '''Return sum of squares loss after masking.
    '''
    if scope is not None:
        scope = scope + '_abs_diff_loss'
    if tf.__version__ == '0.10.0':
        return slim.losses.absolute_difference(predictions, targets, weight=mask, scope=scope)
    else:
        return slim.losses.absolute_difference(predictions, targets, weights=mask, scope=scope)

def get_l2_loss(predictions, targets, scope=None):
    '''Return sum of squares loss after masking.
    '''
    if scope is not None:
        scope = scope + '_mse_loss'
    #  will be renamed to mean_square_error in next tensorflow version. weight->weights
    if tf.__version__ == '0.10.0':
        return slim.losses.sum_of_squares(predictions, targets, scope=scope)
    else:
        return slim.losses.mean_squared_error(predictions, targets, scope=scope)


def get_l2_loss_with_mask(output_img, desired_img, mask, scope=None):
    '''Return sum of squares loss after masking.
    '''
    if scope is not None:
        scope = scope + '_mse_loss'
    #  will be renamed to mean_squared_error in next tensorflow version. weight->weights
    if tf.__version__ == '0.10.0':
        return slim.losses.sum_of_squares(output_img, desired_img,
                weight=mask, scope=scope)
    else:
        return slim.losses.mean_squared_error(output_img, desired_img,
                weights=mask, scope=scope) 

def get_cosine_distance_loss(predictions, targets, dim=1, scope=None):
    '''Assume predictions and targets are vectors
    '''
    if scope is not None:
        scope = scope + '_cos_dist_loss'
    # unit-normalize
    normalized_predictions = tf.nn.l2_normalize(predictions, dim=1,
                                                name='normalize_predictions')
    normalized_targets = tf.nn.l2_normalize(targets, dim=1,
                                            name='normalize_targets')
    return slim.losses.cosine_distance(normalized_predictions,
                                       normalized_targets, dim,
                                       scope=scope)

def get_sparse_softmax_loss(predictions, targets, mask, scope=None):
    ''' Compute Softmax Cross Entropy losses between predictions and targets
        Can leverage mask as either pure masking or weight
    '''
    if scope is not None:
        scope = scope + '_softmax_loss'
    #return tf.nn.softmax_cross_entropy_with_logits(logits=predictions, 
    #        labels=targets,  name='softmax_loss')
    return tf.losses.sparse_softmax_cross_entropy( targets, predictions, weights=mask, scope=scope)

def get_softmax_loss(predictions, targets, mask, scope=None):
    ''' Compute Softmax Cross Entropy losses between predictions and targets
        Can leverage mask as either pure masking or weight
    '''
    if scope is not None:
        scope = scope + '_softmax_loss'
    #return tf.nn.softmax_cross_entropy_with_logits(logits=predictions, 
    #        labels=targets,  name='softmax_loss')
    return tf.losses.softmax_cross_entropy( targets, predictions, weights=mask,
                                scope=scope)

