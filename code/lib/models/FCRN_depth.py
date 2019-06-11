
from __future__ import absolute_import, division, print_function

from   models.base_net import BaseNet
import losses.all as losses_lib
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pdb
import optimizers.train_steps as train_steps
import optimizers.ops as optimize
from functools import partial
import models.fcrn
from models.fcrn import ResNet50UpProj

class FCRN_depth(BaseNet):
    '''Standard encoder decoder model
    Encodes an input into a low-dimensional representation and reconstructs
    the input from the low-dimensional representation. Uses l2 loss.
    Assumes inputs are scaled to [0, 1] (which will be rescaled to [-1, 1].
    '''

    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(FCRN_depth, self).__init__(global_step, cfg)
        if 'hidden_size' not in cfg:
            raise ValueError( "config.py for encoder-decoder must specify 'hidden_size'" )
        #self.ones_mask = self.build_ones_mask()


    def build_ones_mask(self):
        '''Build a mask of ones which has the same size as the input.
        '''
        cfg = self.cfg
        H, W = cfg['target_dim']
        C = cfg['target_num_channels']
        batch_size = cfg['batch_size']
        mask = tf.constant(1.0, dtype=cfg['target_dtype'], shape=[batch_size, H, W, C],
                           name='identity_mask')
        return mask


    def _compute_nnz_mask(self, mask):
        '''Compute the number of nonzero elements in a tensor which only
        contains elements of 0 or 1 (such as a mask).
        '''
        return tf.reduce_sum(mask)

    def build_model(self, input_imgs, is_training, targets=None, masks=None, privileged_input=None):
        '''Builds the model. Assumes that the input is from range [0, 1].
            Args:
            input_imgs: list of input images (scaled between -1 and 1) with the
                       dimensions specified in the cfg
            is_training: flag for whether the model is in training mode or not
            mask: mask used for computing sum of squares loss. If None, we assume
                  it is np.ones.
        '''
        print('building model')
        cfg = self.cfg
        self.is_training = is_training

        if masks is None:
            masks = tf.constant( 1, dtype=tf.float32, shape=[], name='constant_mask' )
        
        net = ResNet50UpProj({'data': input_imgs}, cfg['batch_size'], 1, False)
        decoder_output = net.get_output()
        decoder_output = decoder_output * 128.
        decoder_output = tf.log(decoder_output + 1.) / 11.090354888959125
#         if self.decoder_only:
            # encoder_output = input_imgs Assume that the input is the representation
        # else:
            # encoder_output = self.build_encoder(input_imgs, is_training)
        # print("enc:", encoder_output.shape)
        # decoder_output = self.build_decoder(encoder_output, is_training)
        # print("tar:", targets.shape)


        # set up losses
        if targets is None:
            losses = self.get_losses( decoder_output, input_imgs, masks )
        else:
            losses = self.get_losses( decoder_output, targets, masks )

        # use weight regularization
        if 'omit_weight_reg' in cfg and cfg['omit_weight_reg']:
            add_reg = False
        else:
            add_reg = True
        
        # get losses
        #regularization_loss = tf.add_n( slim.losses.get_regularization_losses(), name='losses/regularization_loss' )
        #total_loss = slim.losses.get_total_loss( add_regularization_losses=add_reg,
        #                                         name='losses/total_loss')

        self.input_images = input_imgs
        self.target_images = targets
        self.targets = targets
        self.masks = masks
        self.decoder_output = decoder_output
        self.losses = losses
        self.total_loss = losses[0] 
        # self.init_op = tf.global_variables_initializer()

        # add summaries
        if  self.extended_summaries:
            slim.summarize_variables()
            slim.summarize_weights()
            slim.summarize_biases()
            slim.summarize_activations()
        slim.summarize_collection(tf.GraphKeys.LOSSES)
        #slim.summarize_tensor( regularization_loss )
        #slim.summarize_tensor( total_loss )
        self.model_built = True


    def get_losses( self, output_imgs, desired_imgs, masks ):
        '''Returns the loss. May be overridden.
        Args:
            output_imgs: Tensor of images output by the decoder.
            desired_imgs: Tensor of target images to be output by the decoder.
            masks: Tensor of masks to be applied when computing sum of squares
                    loss.
            
        Returns:
            losses: list of tensors representing each loss component
        '''
        print('setting up losses...')
        self.output_images = output_imgs
        self.target_images = desired_imgs
        self.masks = masks
        with tf.variable_scope('losses'):
            l1_loss = losses_lib.get_l1_loss_with_mask(
                    self.output_images,
                    self.target_images,
                    self.masks,
                    scope='d1')
        losses = [l1_loss]
        return losses

    def get_classification_loss(self, logits, labels):
        with tf.variable_scope('losses'):
            classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(# slim.losses.sparse_softmax_cross_entropy(
                logits, labels, name='softmax_loss'))
            slim.losses.add_loss(classification_loss)
        losses = [classification_loss]
        return losses

    def get_train_step_fn( self ):
        '''
            Returns: 
                A train_step funciton which takes args:
                    (sess, train_ops, global_stepf)
        '''
        return partial( train_steps.discriminative_train_step_fn,
                return_accuracy=False )

    def build_train_op( self, global_step ):
        '''
            Builds train ops for discriminative task
            
            Args:
                global_step: A Tensor to be incremented
            Returns:
                [ loss_op, accuracy ]
        '''
        if not self.model_built or self.total_loss is None :
            raise RuntimeError( "Cannot build optimizers until 'build_model' ({0}) and 'get_losses' {1} are run".format(
                    self.model_built, self.losses_built ) )
        self.global_step = global_step

        t_vars = tf.trainable_variables()

        # Create the optimizer train_op for the generator

        self.optimizer = optimize.build_optimizer( global_step=self.global_step, cfg=self.cfg )
        if 'clip_norm' in self.cfg:
            self.loss_op = optimize.create_train_op( self.total_loss, self.optimizer, update_global_step=True, clip_gradient_norm=self.cfg['clip_norm'])
        else:
            if self.is_training:
                self.loss_op = optimize.create_train_op( self.total_loss, self.optimizer, update_global_step=True )
            else:
                self.loss_op = optimize.create_train_op( self.total_loss, self.optimizer, is_training=False, update_global_step=True )

        # Create a train_op for the discriminator

        self.train_op = [ self.loss_op, 0 ]
        self.train_op_built = True
        return self.train_op


