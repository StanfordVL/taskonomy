'''Standard Simple feedforward model

    feedforward takes in a single image

    Model-specific config.py options: (inherits from models.base_net):
        'batch_size': An int. The number of input bundle to use in a batch
        'hidden_size': An int. The size of representation size before FC layer
        In metric network: 
            'output_size': For discriminative task, the size of output.

        Encoder:
            'encoder': A function that will build take 'input_placeholder', 'is_training', 'hidden_size', and returns a representation.
            -'encoder_kwargs': A Dict of all args to pass to 'encoder'.

'''

from __future__ import absolute_import, division, print_function
from functools import partial
from models.base_net import BaseNet
import losses.all as losses_lib
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.sample_models import * 
from models.resnet_v1 import * 
import optimizers.train_steps as train_steps
import optimizers.ops as optimize
import pdb

class StandardFeedforward(BaseNet):
    '''
    '''
    
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(StandardFeedforward, self).__init__(global_step, cfg)
        self.cfg = cfg

        if 'hidden_size' not in cfg:
            raise ValueError("config.py for Feedforward Network must specify 'hidden_size'")
        if 'encoder' not in cfg:
            raise ValueError("config.py for Feedforward Network must specify 'encoder'")
        if 'metric_net' not in cfg:
            raise ValueError("config.py for Feedforward Network must specify 'metric_net'")
        if 'loss_threshold' in cfg:
            self.threshold = tf.constant(cfg['loss_threshold'])
        else:
            self.threshold = None   

        self.is_l1 = 'is_l1' in cfg and cfg['is_l1']

    def build_encoder(self, input_imgs, is_training):
        '''Builds encoder.
        Args:
            input_img: input image to encode after scaling to [-1, 1]
            is_training: flag for whether the model is in training mode.
        Returns:
            encoder_output: tensor representing the ouptut of the encoder
        '''
        encoder_kwargs = {}
        if 'encoder_kwargs' in self.cfg:
            encoder_kwargs = self.cfg['encoder_kwargs']
        else: 
            print("Not using 'kwargs' arguments for encoder.")
        
        with tf.variable_scope("feedforward") as scope:
            encoder_output, end_points = self.cfg['encoder'](
                    input_imgs,
                    is_training,
                    reuse=None,
                    hidden_size=self.cfg['hidden_size'],
                    scope=scope,
                    **encoder_kwargs)
        encoder_output = tf.reshape(encoder_output, [-1,16,16,8])
        self.encoder_endpoints = end_points
        return encoder_output

    def build_postprocess(self, encoder_output, is_training):
        '''Build the post-process on feedforward network structure output.
        The default approach will be a three layer fully connected networks
        Args:
            encoder_output: a tensor output representations of input image
            is_training: flag for wheter the model is in training mode.
        Returns:
            final_output: final output for the whole model 
        '''
        metric_kwargs = {}
        if 'metric_kwargs' in self.cfg:
            metric_kwargs = self.cfg['metric_kwargs']
        else: 
            raise ValueError("config.py for Feedforward Network must specify 'metric_kwargs'")
        encoder_output = tf.contrib.layers.flatten(encoder_output)
        final_output, end_points = self.cfg['metric_net'](
                encoder_output,
                is_training,
                **metric_kwargs)
        self.metric_endpoints = end_points
        return final_output

    def build_model(self, input_imgs, is_training, targets, masks=None, privileged_input=None):
        '''Builds the model. Assumes that the input is from range [0, 1].
        Args:
            input_imgs: batch of input images (scaled between -1 and 1) with the
                       dimensions specified in the cfg
            is_training: flag for whether the model is in training mode or not
            mask: mask used for computing sum of squares loss. If None, we assume
                  it is np.ones.
        '''
        print('building model')
        cfg = self.cfg
        self.is_training= is_training
        self.masks = masks
        if self.decoder_only:
            encoder_output = input_imgs
        else:
            encoder_output = self.build_encoder(input_imgs, is_training)

        final_output = self.build_postprocess(encoder_output, is_training)

        losses = self.get_losses(final_output, targets, is_softmax='l2_loss' not in cfg)
        # use weight regularization
        if 'omit_weight_reg' in cfg and cfg['omit_weight_reg']:
            add_reg = False
        else:
            add_reg = True
        
        # get losses
        regularization_loss = tf.add_n( slim.losses.get_regularization_losses(), name='losses/regularization_loss' )
        total_loss = slim.losses.get_total_loss( add_regularization_losses=add_reg,
                                                 name='losses/total_loss')

        self.input_images = input_imgs
        self.targets = targets
        self.masks = masks
        self.encoder_output = encoder_output
        self.decoder_output = final_output
        self.losses = losses
        self.total_loss = total_loss

        # add summaries
        if  self.extended_summaries:
            slim.summarize_variables()
            slim.summarize_weights()
            slim.summarize_biases()
            slim.summarize_activations()
        slim.summarize_collection(tf.GraphKeys.LOSSES)
        slim.summarize_tensor( regularization_loss )
        slim.summarize_tensor( total_loss )
        self.model_built = True

    def get_losses(self, final_output, target, is_softmax=True):
        '''Returns the loss for a Siamese Network.
        Args:
            final_output: tensor that represent the final output of the image bundle.
            target: Tensor of target to be output by the siamese network.
            
        Returns:
            losses: list of tensors representing each loss component
        '''
        print('setting up losses...')
        self.target = target
        self.final_output = final_output
        self.predicted = slim.softmax(final_output)
        with tf.variable_scope('losses'):
            if is_softmax:
                if len(target.shape) == len(final_output.shape):
                    correct_prediction = tf.equal(tf.argmax(final_output,1), tf.argmax(target, 1))
                    if len(self.masks.shape) == 2:
                        self.masks = tf.squeeze(self.masks)
                    siamese_loss = tf.reduce_mean(
                        losses_lib.get_softmax_loss(
                                final_output, 
                                target,
                                self.masks, 
                                scope='softmax_loss'))
                else:
                    correct_prediction = tf.equal(tf.argmax(final_output,1), target)

                    siamese_loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=final_output, 
                                labels=target,
                                name='softmax_loss'))


                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.siamese_loss = siamese_loss
            else:
                # If it's not softmax, it's l2 norm loss.
                self.accuracy = 0
                # self.l2_loss = tf.losses.mean_squared_error(
                    # final_output,
                    # target,
                    # scope='d1',
                    # loss_collection=tf.GraphKeys,
                    # reduction="none") 
                target = tf.to_float(target)
                final_output = tf.to_float(final_output)
                # self.l2_loss = tf.norm(target - final_output, axis=1)
                #self.l2_loss_sum = tf.reduce_sum(self.l2_loss, 1)  
                # print(self.l2_loss)
                if self.is_l1:
                    self.l_loss = losses_lib.get_l1_loss(
                        final_output,
                        target,
                        scope='d1')            
                    print('Using L1 loss.....')
                else:
                    self.l_loss = losses_lib.get_l2_loss(
                        final_output,
                        target,
                        scope='d1') 
                                
                self.siamese_loss = self.l_loss
                self.robust_l_loss = self.l_loss
                # siamese_loss = self.l2_loss
                # if self.threshold is not None:
                #     ind = tf.unstack(siamese_loss)
                #     siamese_loss = [ tf.cond(tf.greater(x, self.threshold), 
                #                             lambda: self.threshold + self.threshold * tf.log(x / self.threshold),
                #                             lambda: x) for x in ind ]
                #     self.robust_l2_loss = siamese_loss
                #     siamese_loss = tf.stack(siamese_loss)
                          
                # self.siamese_loss = tf.reduce_sum(siamese_loss) / self.cfg['batch_size'] 
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.siamese_loss)

        losses = [self.siamese_loss]
        return losses

    def get_train_step_fn( self ):
        '''
            Returns: 
                A train_step funciton which takes args:
                    (sess, train_ops, global_stepf)
        '''
        return partial( train_steps.discriminative_train_step_fn,
                return_accuracy=self.cfg['return_accuracy'] )

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

        self.train_op = [ self.loss_op, self.accuracy ]
        self.train_op_built = True
        return self.train_op


