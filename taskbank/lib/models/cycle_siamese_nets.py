'''Standard Siamese model

    The Siamese Network takes input as a list of image (a list of ndarray)

    Model-specific config.py options: (inherits from models.base_net):
        'batch_size': An int. The number of input bundle to use in a batch
        'num_input': An int. The number of images within an input bundle
        'hidden_size': An int. The size of representation size before FC layer
        'output_size': For discriminative task, the size of output.

        Encoder:
            'encoder': A function that will build take 'input_placeholder', 'is_training', 'hidden_size', and returns a representation.
            -'encoder_kwargs': A Dict of all args to pass to 'encoder'.

'''

from __future__ import absolute_import, division, print_function

from functools import partial
from models.siamese_nets import StandardSiamese
import losses.all as losses_lib
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.sample_models import * 
from models.resnet_v1 import * 
import optimizers.train_steps as train_steps
import optimizers.ops as optimize
import pdb
import numpy as np

class CycleSiamese(StandardSiamese):
    '''
    '''
    
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(CycleSiamese, self).__init__(global_step, cfg)
        self.cfg = cfg

        if 'hidden_size' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'hidden_size'")
        if 'num_input' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'num_input'")
        if 'encoder' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'encoder'")
        if 'metric_net' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'metric_net'")
        
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

    def build_siamese_output_postprocess(self, encoder_output, is_training, scope=None):
        '''build the post-process on siamese network structure output.
        the default approach will be a three layer fully connected networks
        args:
            encoder_output: a list of tensors of output representations of each input image
            is_training: flag for wheter the model is in training mode.
        returns:
            final_output: final output for the whole model 
        '''
        metric_kwargs = {}
        if 'metric_kwargs' in self.cfg:
            metric_kwargs = self.cfg['metric_kwargs']
        else: 
            raise valueerror("config.py for siamese network must specify 'metric_kwargs'")
        if scope is not None:
            metric_kwargs['scope'] = scope

        concat_output = tf.concat(values=encoder_output,axis=1)
        final_output, end_points = self.cfg['metric_net'](
                concat_output,
                is_training,
                **metric_kwargs)
        self.metric_endpoints = end_points
        return final_output

    def denormalize_fixated_camera_pose(self, fixated_camera_pose):
        return fixated_camera_pose * self.std + self.mean
        
    def normalize_fixated_camera_pose(self, fixated_camera_pose):
        return (fixated_camera_pose - self.mean) / self.std

    def euler2mat(self, rotation):
        angle1 = tf.slice(rotation, [0,0], [self.cfg['batch_size'], 1]) 
        angle2 = tf.slice(rotation, [0,1], [self.cfg['batch_size'], 1]) 
        angle3 = tf.slice(rotation, [0,2], [self.cfg['batch_size'], 1]) 

        c1 = tf.cos(angle1)
        c2 = tf.cos(angle2)
        c3 = tf.cos(angle3)

        s1 = tf.sin(angle1)
        s2 = tf.sin(angle2)
        s3 = tf.sin(angle3)

        cc = c1*c3
        cs = c1*s3
        sc = s1*c3
        ss = s1*s3

        m11 = c2*c3
        m12 = s2*sc-cs
        m13 = s2*cc+ss
        m21 = c2*s3
        m22 = s2*ss+cc
        m23 = s2*cs-sc
        m31 = -s2
        m32 = c2*s1
        m33 = c2*c1

        m1 = tf.concat([m11,m12,m13], 1)
        m2 = tf.concat([m21,m22,m23], 1)
        m3 = tf.concat([m31,m32,m33], 1)

        m = tf.stack([m1,m2,m3], axis=1)

        return m


    def atan2(self, y, x, epsilon=1.0e-12):
        # Add a small number to all zeros, to avoid division by zero:
        x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
        y = tf.where(tf.equal(y, 0.0), y+epsilon, y)
            
        angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
        angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
        angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
        angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
        angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
        angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
        return angle
        
    def mat2euler(self, rotation_matrix):
        m11 = tf.slice(rotation_matrix, [0,0,0], [self.cfg['batch_size'], 1, 1])
        m12 = tf.slice(rotation_matrix, [0,0,1], [self.cfg['batch_size'], 1, 1])
        m13 = tf.slice(rotation_matrix, [0,0,2], [self.cfg['batch_size'], 1, 1])
        m21 = tf.slice(rotation_matrix, [0,1,0], [self.cfg['batch_size'], 1, 1])
        m22 = tf.slice(rotation_matrix, [0,1,1], [self.cfg['batch_size'], 1, 1])
        m23 = tf.slice(rotation_matrix, [0,1,2], [self.cfg['batch_size'], 1, 1])
        m31 = tf.slice(rotation_matrix, [0,2,0], [self.cfg['batch_size'], 1, 1])
        m32 = tf.slice(rotation_matrix, [0,2,1], [self.cfg['batch_size'], 1, 1])
        m33 = tf.slice(rotation_matrix, [0,2,2], [self.cfg['batch_size'], 1, 1])

        cy = tf.sqrt(m11*m11+m21*m21)

        eps4 = 4.0e-16
        ax = tf.where(tf.greater(cy,eps4), self.atan2(m32,m33), self.atan2(-m23,m22))
        ay = self.atan2(-m31,cy)
        az = tf.where(tf.greater(cy,eps4), self.atan2(m21,m11), tf.zeros_like(ay))

        rotation = tf.concat([ax,ay,az], 1)

        return tf.squeeze(rotation, axis=-1)

    def calculate_combined_relative_camera_pose(self, pose1, pose2):
        rotation1 = tf.slice(pose1, [0,0], [self.cfg['batch_size'], 3])
        translation1 = tf.slice(pose1, [0,3], [self.cfg['batch_size'], 3])

        rotation2 = tf.slice(pose2, [0,0], [self.cfg['batch_size'], 3])
        translation2 = tf.slice(pose2, [0,3], [self.cfg['batch_size'], 3])

        rotation_matrix_1 = self.euler2mat(rotation1)
        rotation_matrix_2 = self.euler2mat(rotation2)

        translation = tf.squeeze(tf.matmul(rotation_matrix_2,
                                 tf.expand_dims(translation1, -1))) + translation2
        
        rotation = self.mat2euler(tf.matmul(rotation_matrix_2, rotation_matrix_1))

        pose = tf.concat([rotation, translation], 1)
        return pose


    def build_model(self, input_imgs, is_training, targets, masks=None, privileged_input=None):
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

        if self.decoder_only:
            encoder_output = input_imgs # Assume that the input is the representation
        else:
            encoder_output = self.build_encoder(input_imgs, is_training)

        final_output_12 = self.build_siamese_output_postprocess(encoder_output, is_training, scope="three_layer_fc_network12")

        final_output_23 = self.build_siamese_output_postprocess(encoder_output, is_training, scope="three_layer_fc_network23")

        final_output_13 = self.calculate_combined_relative_camera_pose(
                self.denormalize_fixated_camera_pose(final_output_12),
                self.denormalize_fixated_camera_pose(final_output_23))

        final_output_13 = self.normalize_fixated_camera_pose(final_output_13)

        #final_output = tf.concat(1, [final_output_12, final_output_13, final_output_23])

        target12 = tf.slice(targets, [0,0], [self.cfg['batch_size'], 6])
        target13 = tf.slice(targets, [0,6], [self.cfg['batch_size'], 6])
        target23 = tf.slice(targets, [0,12], [self.cfg['batch_size'], 6])

        final_output = [final_output_12, final_output_13, final_output_23]
        target_total = [target12, target13, target23]

        losses = self.get_losses(final_output, target_total, is_softmax='l2_loss' not in cfg)
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
        self.encoder_output = encoder_output
        self.losses = losses
        self.total_loss = total_loss
        self.decoder_output = final_output
        # add summaries
        if  self.extended_summaries:
            slim.summarize_variables()
            slim.summarize_weights()
            slim.summarize_biases()
            slim.summarize_activations()
        slim.summarize_collection(tf.GraphKeys.LOSSES)
        tf.summary.scalar('accuracy', self.accuracy)
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


