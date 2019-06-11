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
from   functools import partial
import numpy as np
import pdb

from   models.base_net import BaseNet
from   models.basic_feedforward import StandardFeedforward
import losses.all as losses_lib
import tensorflow as tf
import tensorflow.contrib.slim as slim
from   models.sample_models import * 
from   models.resnet_v1 import * 
import optimizers.train_steps as train_steps
import optimizers.ops as optimize

class ConstantPredictorSegmentation(StandardFeedforward):
    '''
    '''
    
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(StandardFeedforward, self).__init__(global_step, cfg)
        self.cfg = cfg
        self.batch_size = cfg['batch_size'] 
        self.num_pixels = cfg['num_pixels']

        idxes = np.asarray([range(self.batch_size)] * self.num_pixels).T
        self.batch_index_slice = tf.cast(tf.stack(idxes), cfg['target_dtype'])
        
        self.input_type = cfg['input_dtype']

        self.cfg = cfg

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

        
        # decoder_output (32, 128, 128, 64)
        constant_input = tf.zeros(
            [self.batch_size] + list(input_imgs.shape[1:3]) + [64],
            name='const_input',
            dtype=tf.float32 )

        predictions = tf.get_variable(
            "constant_prediction",
            list(input_imgs.shape[1:3]) + [64],
            # initializer=tf.zeros_initializer(),
            dtype=tf.float32)

        final_output = constant_input + predictions

        print("Outputs: ", final_output.shape)
        print("Targets: ", targets.shape)

        # add_fc_layer
        # if self.decoder_only:
        #     encoder_output = input_imgs
        # else:
        #     encoder_output = self.build_encoder(input_imgs, is_training)

        # final_output = self.build_postprocess(encoder_output, is_training)

        losses = self.get_losses(final_output, targets, masks)
        total_loss = slim.losses.get_total_loss( 
            add_regularization_losses=False,
            name='losses/total_loss')

        self.input_images = input_imgs
        self.targets = targets
        # self.encoder_output = encoder_output
        self.decoder_output = final_output
        self.losses = losses
        self.total_loss = total_loss

        # # add summaries
        slim.summarize_variables()
        slim.summarize_tensor( total_loss )
        self.model_built = True

    def get_losses( self, output_vectors, idx_segments, masks ):
        '''Returns the metric loss for 'num_pixels' embedding vectors. 
        
        Args:
            output_imgs: Tensor of images output by the decoder.
            desired_imgs: Tensor of target images to be output by the decoder.
            masks: Tensor of masks to be applied when computing sum of squares
                    loss.
            
        Returns:
            losses: list of tensors representing each loss component
        '''
        print('setting up losses...')
        self.output_images = output_vectors
        self.target_images = idx_segments
        self.masks = masks

        with tf.variable_scope('losses'):
            last_axis = 2
            fir, sec, seg_id = tf.unstack(idx_segments, axis=last_axis)

            idxes = tf.stack([self.batch_index_slice, fir, sec], axis=last_axis)
            self.embed = tf.gather_nd( output_vectors, idxes )
            embed = self.embed
            square = tf.reduce_sum( embed*embed, axis=-1 )
            square_t = tf.expand_dims(square, axis=-1)
            square = tf.expand_dims(square, axis=1)

            pairwise_dist = square - 2 * tf.matmul(embed, tf.transpose(embed, perm=[0,2,1])) + square_t 
            pairwise_dist = tf.clip_by_value( pairwise_dist, 0, 80)
            #pairwise_dist = 0 - pairwise_dist
            self.pairwise_dist = pairwise_dist
            pairwise_exp = tf.exp(pairwise_dist) + 1
            sigma = tf.divide(2 ,  pairwise_exp)
            sigma = tf.clip_by_value(sigma,1e-7,1.0 - 1e-7)
            self.sigma = sigma
            same = tf.log(sigma)
            diff = tf.log(1 - sigma)

            self.same = same
            self.diff = diff
            
            seg_id_i = tf.tile(tf.expand_dims(seg_id, -1), [1, 1, self.num_pixels])
            seg_id_j = tf.transpose(seg_id_i, perm=[0,2,1])

            seg_comp = tf.equal(seg_id_i, seg_id_j)
            seg_same = tf.cast(seg_comp, self.input_type) 
            seg_diff = 1 - seg_same

            loss_matrix = seg_same * same + seg_diff * diff
            reduced_loss = 0 - tf.reduce_mean(loss_matrix) # / self.num_pixels
         
        tf.add_to_collection(tf.GraphKeys.LOSSES, reduced_loss)
        self.metric_loss = reduced_loss
        losses = [reduced_loss]
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

        self.train_op = [ self.loss_op, 0 ]
        self.train_op_built = True
        return self.train_op


class ConstantPredictorPose(StandardFeedforward):
    '''
    '''
    
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(StandardFeedforward, self).__init__(global_step, cfg)
        self.cfg = cfg
                
        # Normalize for fixated relative camera pose
        import pickle
        import os
        with open(os.path.join(cfg['root_dir'], 'lib/data/camera_mean_and_std.pkl'), 'rb') as fp:
            data = pickle.load(fp)
    
        self.std  = np.asarray([ 10.12015407, 8.1103528, 1.09171896, 1.21579016, 0.26040945, 10.05966329])
        self.mean = np.asarray([ -2.67375523e-01, -1.19147040e-02, 1.14497274e-02, 1.10903410e-03, 2.10509948e-02, -4.02013549e+00])
        self.mean = tf.cast(tf.stack(self.mean), cfg['target_dtype'])

        self.std = tf.cast(tf.stack(self.std), cfg['target_dtype'])

        if 'loss_threshold' in cfg:
            self.threshold = tf.constant(cfg['loss_threshold'])
        else:
            self.threshold = None

        self.cycle_rate = 0.5

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

        
        # decoder_output (32, 128, 128, 64)
        constant_input = tf.zeros(
            # [self.batch_size] + list(input_imgs.shape[1:3]) + [64],
            list(targets.shape),
            name='const_input',
            dtype=tf.float32 )

        predictions = tf.get_variable(
            "constant_prediction",
            [1] + list(targets.shape)[1:],
            # initializer=tf.zeros_initializer(),
            dtype=tf.float32)

        final_output = constant_input + predictions

        print("Predictions: ", predictions.shape)
        print("Outputs: ", final_output.shape)
        print("Targets: ", targets.shape)



        final_output_12 = tf.slice(final_output, [0,0], [self.cfg['batch_size'], 6])
        final_output_13 = tf.slice(final_output, [0,6], [self.cfg['batch_size'], 6])
        final_output_23 = tf.slice(final_output, [0,12], [self.cfg['batch_size'], 6])

        target12 = tf.slice(targets, [0,0], [self.cfg['batch_size'], 6])
        target13 = tf.slice(targets, [0,6], [self.cfg['batch_size'], 6])
        target23 = tf.slice(targets, [0,12], [self.cfg['batch_size'], 6])

        target_total = [target12, target13, target23]
        final_output = [final_output_12, final_output_13, final_output_23]
        # add_fc_layer
        # if self.decoder_only:
        #     encoder_output = input_imgs
        # else:
        #     encoder_output = self.build_encoder(input_imgs, is_training)

        # final_output = self.build_postprocess(encoder_output, is_training)

        losses = self.get_losses(final_output, target_total, 'l2_loss' not in cfg)
        total_loss = slim.losses.get_total_loss( 
            add_regularization_losses=False,
            name='losses/total_loss')

        self.input_images = input_imgs
        self.targets = targets
        # self.encoder_output = encoder_output
        self.decoder_output = final_output
        self.losses = losses
        self.total_loss = total_loss

        # # add summaries
        slim.summarize_variables()
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
        with tf.variable_scope('losses'):
            if is_softmax:
                correct_prediction = tf.equal(tf.argmax(final_output,1), target)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                siamese_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=final_output, 
                                labels=target,
                                name='softmax_loss'))
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
                self.l2_loss = []
                for i in range(len(final_output)):
                    target[i] = tf.to_float(target[i])
                    final_output[i] = tf.to_float(final_output[i])
                    self.l2_loss.append( tf.norm(target[i] - final_output[i], axis=1) )
                #self.l2_loss_sum = tf.reduce_sum(self.l2_loss, 1)  

                
                siamese_loss = self.l2_loss
                self.robust_l2_loss = []
                if self.threshold is not None:
                    for i in range(len(siamese_loss)):

                        ind = tf.unstack(siamese_loss[i])
                        siamese_loss[i] = [ tf.cond(tf.greater(x, self.threshold), 
                                            lambda: self.threshold + self.threshold * tf.log(x / self.threshold),
                                            lambda: x) for x in ind ]
                        self.robust_l2_loss.append(siamese_loss[i])
                        siamese_loss[i] = tf.stack(siamese_loss[i])
                     
                self.siamese_losses = []
                for i in range(len(siamese_loss)):
                    self.siamese_losses.append( tf.reduce_sum(siamese_loss[i]) / self.cfg['batch_size'] )
                self.siamese_losses[1] = self.siamese_losses[1] * self.cycle_rate

        
        self.cycle_loss_total = tf.add_n(self.siamese_losses)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.cycle_loss_total)
        losses = [self.cycle_loss_total]
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



class ConstantPredictorL2(StandardFeedforward):
    '''
    '''
    
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(StandardFeedforward, self).__init__(global_step, cfg)
        self.cfg = cfg
        self.is_l1 = 'is_l1' in cfg and cfg['is_l1']

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

        
        # decoder_output (32, 128, 128, 64)
        constant_input = tf.zeros(
            # [self.batch_size] + list(input_imgs.shape[1:3]) + [64],
            list(targets.shape),
            name='const_input',
            dtype=tf.float32 )

        predictions = tf.get_variable(
            "constant_prediction",
            [1] + list(targets.shape)[1:],
            # initializer=tf.zeros_initializer(),
            dtype=tf.float32)

        final_output = constant_input + predictions

        print("Predictions: ", predictions.shape)
        print("Outputs: ", final_output.shape)
        print("Targets: ", targets.shape)

        losses = self.get_losses(final_output, targets, 'l2_loss' not in cfg)
        total_loss = slim.losses.get_total_loss( 
            add_regularization_losses=False,
            name='losses/total_loss')

        self.input_images = input_imgs
        self.targets = targets
        # self.encoder_output = encoder_output
        self.decoder_output = final_output
        self.losses = losses
        self.total_loss = total_loss

        # # add summaries
        slim.summarize_variables()
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
        with tf.variable_scope('losses'):
            if is_softmax:
                correct_prediction = tf.equal(tf.argmax(final_output,1), target)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                siamese_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=final_output, 
                                labels=target,
                                name='softmax_loss'))
                self.siamese_loss = siamese_loss
            else:
                # If it's not softmax, it's l2 norm loss.
                self.accuracy = 0
#                 self.l2_loss = tf.losses.mean_squared_error(
                    # final_output,
                    # target,
                    # scope='d1',
                    # loss_collection=tf.GraphKeys,
                    # reduction="none") 
                target = tf.to_float(target)
                final_output = tf.to_float(final_output)
                self.l2_loss = tf.norm(target - final_output, axis=1)
                #self.l2_loss_sum = tf.reduce_sum(self.l2_loss, 1)  

                
                siamese_loss = self.l2_loss
                #if self.threshold is not None:
                if False:
                    ind = tf.unstack(siamese_loss)
                    siamese_loss = [ tf.cond(tf.greater(x, self.threshold), 
                                            lambda: self.threshold + self.threshold * tf.log(x / self.threshold),
                                            lambda: x) for x in ind ]
                    self.robust_l2_loss = siamese_loss
                    siamese_loss = tf.stack(siamese_loss)
                          
                self.siamese_loss = tf.reduce_sum(siamese_loss) / self.cfg['batch_size'] 
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


