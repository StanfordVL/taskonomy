'''Provides encoder-decoder model with a cGAN and softmax loss. 

    Model-specific config.py options: (others are inherited from models.encoder_decoder.StandardED)
        Regularization:
            -instance_noise_sigma: A float representing the standard deviation of the noise to use.
                This noise will be added to the output of the generator before it is fed to the 
                discriminator. (0.0)

        Training schedule:
            -n_g_steps_before_d: The number of generator steps to run before running the discriminator
                train_op. (1)
            -n_d_steps_after_g: The number of discriminator steps to run after running generator 
                train_op. (1)
            -discriminator_learning_args: A Dict that may contain the same keys as the one for
                generator learning args. If this Dict is specified for the discriminator, it will be 
                used instead of the generator one and all values must be explicitly respecified. ({})
            
        Losses
            -l_norm_weight_prop: Generator loss will weight l-norm by  'l_norm_weight_prop', and 
                gan_loss will be weighted by 1. - 'l_norm_weight_prop'. (0.5)
            -gan_loss_kwargs: Arguments to be passed into losses.all.get_gan_loss -- note that
                this is where label smoothing should be specified. ({})

    Assumes there is one input and one output.
'''
from __future__ import absolute_import, division, print_function

from   functools import partial
import losses.all as losses_lib
from   models.encoder_decoder import StandardED
from   models.utils import add_gaussian_noise_layer
import optimizers.ops as optimize 
import optimizers.train_steps as train_steps 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from data.load_ops import rescale_image

class EDWithSoftmaxRegenerationCGAN(StandardED):
    '''Standard encoder decoder model
    Encodes an input into a low-dimensional representation and reconstructs
    the input from the low-dimensional representation. Uses l2 loss.
    Assumes inputs are scaled to [0, 1] (which will be rescaled to [-1, 1].
    '''
    def _set_default_properties( self ):
        # regularization
        self.add_weight_reg = True
        self.instance_noise_sigma = None
        
        # loss
        self.l_norm_weight_prop = 0.5
        
        # training schedule
        self.n_g_steps_before_d = 1
        self.n_d_steps_after_g = 1

        self.losses_built = False
        self.metrics_built = False
        self.model_built = False
        self.train_op_built = False
        self.summaries_built = False
        self.decoder_only = False

 
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(EDWithSoftmaxRegenerationCGAN, self).__init__(global_step, cfg)
        if 'hidden_size' not in cfg:
            raise ValueError( "config.py for encoder-decoder must specify 'hidden_size'" )
        self._set_default_properties()

        # regularization params
        if 'omit_weight_reg' in cfg and cfg['omit_weight_reg']:
            self.add_weight_reg = False
        if 'instance_noise_sigma' in cfg:
            with tf.variable_scope( 'instance_noise' ):
                self.instance_noise_sigma = self._anneal_tensor( 
                    initial_rate=cfg[ 'instance_noise_sigma' ], 
                    anneal_schedule_fn=self._try_get( 'instance_noise_anneal_fn', cfg, default=None ),
                    schedule_fn_kwargs=self._try_get( 'instance_noise_anneal_fn_kwargs', cfg, default={} ),
                    global_step=global_step, 
                    title='instance noise sigma' ) 
        
        # loss params
        if 'l_norm_weight_prop' in cfg:
            self.l_norm_weight_prop = cfg[ 'l_norm_weight_prop' ]
        self.gan_loss_weight_prop = 1. - self.l_norm_weight_prop
        
        # training schedule
        if 'n_g_steps_before_d' in cfg:
            self.n_g_steps_before_d = cfg[ 'n_g_steps_before_d' ]
        if 'n_d_steps_after_g' in cfg:
            self.n_d_steps_after_g = cfg[ 'n_d_steps_after_g' ]
        if 'init_g_steps' in cfg:
            self.init_g_steps = cfg['init_g_steps']
        self.is_l2 = 'l2_loss' in cfg
        self.secret_scope = False

        a = np.load(os.path.join(cfg['root_dir'], 'lib/data', 'pts_in_hull.npy'))
        a = np.expand_dims(a, axis=0)
        a = np.expand_dims(a, axis=0)
        self.trans_kernel = tf.cast(tf.stack(a), cfg['input_dtype'])
        self.input_size = cfg['input_dim']


    def build_discriminator( self, input_imgs, decoder_output, is_training, reuse=False ):
        ''' Build the descriminator for GAN loss. 
        Args: 
            input_imgs: 
            decoder_output: The output of the decoder
            is_training: A bool that is true if the model should be build in training mode
        Returns:
            discriminator_output
        '''
        discriminator_kwargs = {}
        if 'discriminator_kwargs' in self.cfg:
            discriminator_kwargs = self.cfg['discriminator_kwargs']
        else:
            print( "Not using 'kwargs' arguments for discriminator_kwargs." )
        
        # possibly add instance noise
        if 'instance_noise_sigma' in self.cfg:
            decoder_output = add_gaussian_noise_layer( decoder_output, std=self.instance_noise_sigma,
                        scope='gaussian_noise' )
        
        # condition discriminator on input
        augmented_images = tf.concat( 
                values=[input_imgs, decoder_output], 
                axis=len( decoder_output.get_shape() ) - 1 )
                # , name='condition_discriminator' )
        self.augmented_images.append( augmented_images )

        # build discriminator
        discriminator_output, end_points = self.cfg['discriminator'](
                augmented_images, 
                is_training, 
                reuse=reuse,
                **discriminator_kwargs )
        self.discriminator_endpoints.append( end_points )
        return discriminator_output

    def colorized_image_from_softmax(self, targets, decoder_output):
        ''' Regenerate colorized image from softmax distribution for all colors

        Notes:
            This is a constant mapping from distribution to actual image

        Args:
            decoder_output: list of input images (scaled between -1 and 1) with the
                       dimensions specified in the cfg
        '''
        resize_shape = tf.stack([self.input_size[0],self.input_size[1]])
        softmax_to_ab = tf.nn.convolution(decoder_output, self.trans_kernel, 'SAME' )
        resized_output = tf.image.resize_images(softmax_to_ab, 
                resize_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        softmax_to_ab = tf.nn.convolution(targets, self.trans_kernel, 'SAME' )
        resized_target = tf.image.resize_images(softmax_to_ab, 
                resize_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
        return resized_target, resized_output 
           



    def build_model(self, input_imgs, is_training, targets, masks=None, privileged_input=None):
        '''Builds the model. Assumes that the input is from range [-1, 1].
        Notes:
            Stocasticity is not supplied in this function. If desired, it must
            be defined in the encoder/decoder model method. 
        Args:
            input_imgs: list of input images (scaled between -1 and 1) with the
                       dimensions specified in the cfg
            is_training: flag for whether the model is in training mode or not
            mask: mask used for computing sum of squares loss. If None, we assume
                  it is np.ones.
        '''
        print('building model')
        self.input_images = input_imgs
        self.privileged_input = privileged_input
        if self.privileged_input is None:
            self.privileged_input = input_imgs
        self.target_images = targets
        self.targets = targets
        self.masks = masks
        
        # build generator
        if masks is None:
            masks = tf.constant( 1, dtype=tf.float32, shape=[], name='constant_mask' )
        if self.decoder_only:
            self.encoder_output = input_imgs # Assume that the input is the representation
        else:
            self.encoder_output = self.build_encoder(input_imgs, is_training)
        self.decoder_output = self.build_decoder( self.encoder_output, is_training )
        temp = slim.softmax(self.decoder_output * 2.606)
        self.generated_target, self.generated_output = self.colorized_image_from_softmax(self.target_images, temp)
        # build discriminator
        self.augmented_images = []
        self.discriminator_endpoints = []
        self.discriminator_output_real = self.build_discriminator( # run once on real targets
                    self.privileged_input, self.generated_target, is_training ) 
        self.discriminator_output_fake = self.build_discriminator( # run once on the output
                    self.privileged_input, self.generated_output, is_training, reuse=True )
        # self.discriminator_output_real = self.build_discriminator( # run once on real targets
        #             self.privileged_input, self.target_images, is_training ) 
        # self.discriminator_output_fake = self.build_discriminator( # run once on the output
        #             self.privileged_input, self.decoder_output, is_training, reuse=True )

        resized_output = tf.reshape(self.decoder_output, [-1, 313])
        resized_target = tf.reshape(targets, [-1, 313])
        masks = tf.reshape(masks, [-1])
        # set up losses
        _ = self.get_losses( resized_output, resized_target, masks, 
             discriminator_predictions_real=self.discriminator_output_real,
             discriminator_predictions_fake=self.discriminator_output_fake )
        
        # record accuracies
        self._build_metrics( scope='metrics')

        # add summaries
        self._build_summaries()

        # discriminator accuracies
        self.model_built = True

    def build_train_op( self, global_step ):
        '''
            Builds two train ops, one for the Generator and one for the Discriminator. These can 
            be run independently any number of times, and each time will increase the global_step.

            Args:
                global_step: A Tensor to be incremented
            Returns:
                [ g_train_op, d_train_op ]
        '''
        if not self.model_built or not self.losses_built :
            raise RuntimeError( "Cannot build optimizers until 'build_model' ({0}) and 'get_losses' {1} are run".format(
                    self.model_built, self.losses_built ) )
        self.global_step = global_step
        self.global_step_copy = tf.identity( global_step, name='global_step_copy' )

        t_vars = tf.trainable_variables()
    
        # Create the optimizer train_op for the generator
        self.g_optimizer = optimize.build_optimizer( global_step=self.global_step, cfg=self.cfg )
        self.g_vars = slim.get_variables( scope='encoder', collection=tf.GraphKeys.TRAINABLE_VARIABLES )
        self.g_vars += slim.get_variables( scope='decoder', collection=tf.GraphKeys.TRAINABLE_VARIABLES )
        self.g_train_op = optimize.create_train_op( self.loss_g_total, self.g_optimizer, 
                    variables_to_train=self.g_vars, update_global_step=True )
        self.g_lnorm_op = optimize.create_train_op( self.softmax_loss, self.g_optimizer, 
                    variables_to_train=self.g_vars, update_global_step=True )


        # Create a train_op for the discriminator
        if 'discriminator_learning_args' in self.cfg: # use these
            discriminator_learning_args = self.cfg[ 'discriminator_learning_args' ]
        else:
            discriminator_learning_args = self.cfg 
        self.d_optimizer = optimize.build_optimizer( global_step=self.global_step, cfg=discriminator_learning_args )
        self.d_vars = slim.get_variables( scope='discriminator', collection=tf.GraphKeys.TRAINABLE_VARIABLES )
        self.d_vars += slim.get_variables( scope='discriminator_1', collection=tf.GraphKeys.TRAINABLE_VARIABLES )
        self.d_train_op = slim.learning.create_train_op( self.loss_d_total, self.d_optimizer, 
                    variables_to_train=self.d_vars )

        self.train_op = [ self.g_train_op, self.d_train_op, self.g_lnorm_op, self.accuracy]
        self.train_op_built = True
        return self.train_op

    def get_losses( self, output_imgs, desired_imgs, masks, 
            discriminator_predictions_real, discriminator_predictions_fake ):
        '''Returns the loss. May be overridden.
        Args:
            output_imgs: Tensor of images output by the decoder.
            desired_imgs: Tensor of target images to be output by the decoder.
            masks: Tensor of masks to be applied when computing sum of squares
                    loss.
            discriminator_predictions_real: A Tensor of labels output from the 
                discriminator when the input was real
            discriminator_predictions_fake: A Tensor of labels output from the 
                discriminator when the input was generated
            
        Returns:
            losses: list of tensors representing each loss component. Of type
                [ l1_loss, loss_g, loss_d_real, loss_d_fake ]
        '''
        print('setting up losses...')
        with tf.variable_scope('losses'):
            # L-norm loss
            correct_prediction = tf.equal(tf.argmax(output_imgs,-1), tf.argmax(desired_imgs,-1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.softmax_loss = tf.reduce_mean(losses_lib.get_softmax_loss( output_imgs, desired_imgs, masks ) )
            tf.add_to_collection(tf.GraphKeys.LOSSES, self.softmax_loss)

            # GAN loss
            gan_loss_kwargs = {}
            if 'gan_loss_kwargs' in self.cfg:
                gan_loss_kwargs = self.cfg['gan_loss_kwargs']
            if 'real_label' not in gan_loss_kwargs:
                gan_loss_kwargs[ 'real_label' ] = 1.0
            if 'fake_label' not in gan_loss_kwargs:
                gan_loss_kwargs[ 'fake_label' ] = 0.0
            self.real_label = gan_loss_kwargs[ 'real_label' ]
            self.fake_label = gan_loss_kwargs[ 'fake_label' ]
            self.d_threshhold_value = ( self.real_label + self.fake_label ) / 2.
            self.loss_g, self.loss_d_real, self.loss_d_fake = losses_lib.get_gan_loss( 
                        discriminator_predictions_real, discriminator_predictions_fake,
                        self=self,
                        **gan_loss_kwargs )
            
            # Make the regularization loss accessible
            # Make the regularization loss accessible
            if self.decoder_only:
                self.encoder_regularization_loss = tf.constant(0.0)
            else:    
                self.encoder_regularization_loss = tf.add_n( 
                            tf.losses.get_regularization_losses( scope=self._add_secret_scope('encoder') ), 
                            name='reg_loss_encoder' )
            self.decoder_regularization_loss = tf.add_n( 
                        tf.losses.get_regularization_losses( scope=self._add_secret_scope('decoder' )), 
                        name='reg_loss_decoder'  )
            self.discriminator_regularization_loss = tf.add_n( 
                        tf.losses.get_regularization_losses( scope=self._add_secret_scope('discriminator') ), 
                        name='reg_loss_discriminator'  )

            # all losses
            self.loss_g_total = tf.add_n( [
                        self.l_norm_weight_prop * self.softmax_loss,
                        self.gan_loss_weight_prop * self.loss_g,
                        self.encoder_regularization_loss,
                        self.decoder_regularization_loss ], 
                        name='generator_loss' )
            self.loss_d_total = tf.add_n( [
                        self.loss_d_real / 2.,
                        self.loss_d_fake / 2., 
                        self.discriminator_regularization_loss ],
                        name='discriminator_loss' )
            self.total_loss = tf.add_n( [ 
                        self.loss_g_total, 
                        self.loss_d_total ], 
                        name='total_loss' )
        
        if self.decoder_only:
            self.total_loss = self.l_norm_weight_prop * self.softmax_loss + self.gan_loss_weight_prop * self.loss_g

        self.losses = [self.softmax_loss, self.loss_g, self.loss_d_real, self.loss_d_fake]
        self.losses_built = True
        return self.losses

    def get_train_step_fn( self ):
        ''' 
            Returns:
                A train_step function which takes args:
                    ( sess, g_and_d_train_ops, global_step, train_step_kwargs )
        '''
        # return train_steps.gan_train_step_fn
        return partial( train_steps.gan_train_step_fn,
                return_accuracy=True,
                n_g_steps_before_d=self.n_g_steps_before_d,
                n_d_steps_after_g=self.n_d_steps_after_g,
                init_g_steps=self.init_g_steps)

    def _anneal_tensor( self, initial_rate, anneal_schedule_fn=None, schedule_fn_kwargs={},
                global_step=None, title='learning rate' ):
        ''' Anneals an input tensor. '''
        print( "setting up {0} annealing schedule:".format( title ) )
        print( "\tinitial_rate:", initial_rate )
        safe_title = title.replace( ' ', '_' )
        # set up the learning rate
        if not anneal_schedule_fn:  # use constant
            print( '\tNo annealing schedule given. Using constant {0}.'.format( title ) )
            annealed_tensor = tf.constant( initial_rate, name=safe_title )
        else:  # use user-supplied value
            if not schedule_fn_kwargs:
                print( "\tNo kwargs found for {0} schedule.".format( title ) )
            if 'name' not in schedule_fn_kwargs:
                schedule_fn_kwargs[ 'name' ] = safe_title
            if global_step is None:
                raise ValueError( 'If using an annealing schedule, global_step must be given.' )
            schedule_fn_kwargs[ 'global_step' ] = global_step
            # call the user's fn
            annealed_tensor = anneal_schedule_fn( 
                        initial_rate, 
                        **schedule_fn_kwargs )
            print( "\t", schedule_fn_kwargs )
        summary_name = annealed_tensor.name.replace( ":", "_" )
        slim.summarize_tensor( annealed_tensor, tag='{0}/{1}'.format( safe_title, summary_name ) )
        return annealed_tensor

    def _build_metrics( self, scope='metrics' ):
        ''' Run after self.get_losses. '''
        if not self.losses_built:
            raise RuntimeError( "Cannot _build_metrics until 'get_losses' is run" )

        with tf.variable_scope( scope ) as sc:
            is_real_label_greater = self.real_label > self.fake_label
            print( "\tFake/real threshhold:", self.d_threshhold_value )
            self.real_accuracy = self._get_accuracy( self.discriminator_output_real, self.d_threshhold_value, 
                        greater_than=is_real_label_greater, scope='accuracy_real' )
            self.fake_accuracy =  self._get_accuracy( self.discriminator_output_fake, self.d_threshhold_value, 
                        greater_than=not is_real_label_greater, scope='accuracy_real' )
        self.metrics_built = True

    def _build_summaries( self ):
        if not self.losses_built:
            raise RuntimeError( "Cannot _build_summaries until 'get_losses' ({0}) and _build_metrics ({1}) is run".format(
                self.losses_built, self.metrics_built ) )

        # add check for losses, metrics built
        if  self.extended_summaries:
            slim.summarize_variables()
            slim.summarize_weights()
            slim.summarize_biases()
            slim.summarize_activations()
        tf.summary.scalar( 'metrics/d_accuracy_on_real', self.real_accuracy )
        tf.summary.scalar( 'metrics/d_accuracy_on_fake', self.fake_accuracy )

        # losses
        slim.summarize_collection(tf.GraphKeys.LOSSES)
        slim.summarize_tensor( self.encoder_regularization_loss )
        slim.summarize_tensor( self.decoder_regularization_loss )
        slim.summarize_tensor( self.discriminator_regularization_loss )
        slim.summarize_tensor( self.loss_g_total ) #, tag='losses/generator_total_loss' )
        slim.summarize_tensor( self.loss_d_total ) #, tag='losses/discriminator_total_loss' )
        self.summaries_built = True

    def _get_accuracy( self, observed_vals, threshhold_val, greater_than, scope='accuracy' ):
        comparison_op = tf.greater if greater_than else tf.less
        with tf.variable_scope( scope ):
            return tf.reduce_mean( tf.cast(
                        comparison_op( observed_vals, threshhold_val ), tf.float32 ) )

    def _try_get( self, key, dictionary, default=None ):
        return dictionary[ key ] if key in dictionary else default

    def _add_secret_scope( self, scope ):
        if self.secret_scope:
            return self.secret_scope + "/" + scope
        return scope
