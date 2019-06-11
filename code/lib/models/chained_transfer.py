'''Standard encoder-decoder model
Assumes there is one input and one output.

    Model-specific config.py options: (inherits from models.base_net)
        'batch_size': An int. The number of images to use in a batch
        'hidden_size': An int. The number of hidden neurons to use. 
        'target_num_channels': The number of channels to output from the decoder

        Encoder:
            'encoder': A function that will build take 'input_placeholder', 'is_training', 
                'hidden_size', and returns a representation. 
            -'encoder_kwargs': A Dict of all the args to pass to 'encoder'. The Dict should
                not include the mandatory arguments given above. ({}) 
        Decoder:
            'decoder': A function that will build take 'encoder_output', 'is_training', 
                'num_output_channels' (value from 'target_num_channels'), and returns a 
                batch of representation vectors. 
            -'decoder_kwargs': A Dict of all the args to pass to 'decoder'. The Dict should
                not include the mandatory arguments given above. ({}) 
        
'''
from __future__ import absolute_import, division, print_function

from   models.base_net import BaseNet
from   models.encoder_decoder import StandardED
import losses.all as losses_lib
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pdb
import optimizers.train_steps as train_steps
import optimizers.ops as optimize
import sys
from functools import partial




class ChainedTransferNet(BaseNet):
    '''Transfer using perceptial loss

    The encoder does the transfer from input-target representation, 
    and the decoder is frozen. Losses are backpropagated throught the
    target decoder, and only the encoder weights are changed. 

    TODO(sasha): Specify the transfer encoder model
    '''

    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(ChainedTransferNet, self).__init__(global_step, cfg)
        if 'hidden_size' not in cfg:
            raise ValueError( "config.py for encoder-decoder must specify 'hidden_size'" )
        self.cfg = cfg

        # Load the target model
        # sys.path.insert(0, cfg['config_dir_target'])
        # import config as target_config
        # sys.path.pop(0)
        # self.target_cfg = cfg['target_cfgs'][-1] #target_config.get_cfg(nopause=True)
        self.decoders = []
        for decoder_cfg in cfg['target_cfgs']:
            self.decoders.append(
                decoder_cfg['model_type'](global_step, decoder_cfg)
            )
            self.decoders[-1].decoder_only = True

        # self.decoder1 = self.target_cfg['model_type'](global_step,  cfg['target_cfgs'][0])
        # self.decoder1.decoder_only = True

        # self.decoder2 = self.target_cfg['model_type'](global_step, self.target_cfg)
        # self.decoder2.decoder_only = True

        self.encoder_scope = 'transfer'
        # if 'multiple_input_tasks' in self.cfg:
        #     self.encoder_scope = 'funnel'
        # self.decoder.decoder_only = False

    def build_encoder(self, input_imgs, is_training, scope_name='transfer'):
        '''Builds the encoder.
        Args:
            input_img: input image to encoder after scaling to [-1, 1]
            is_training: flag for whether the model is in training mode.
        Returns:
            encoder_output: tensor representing the output of the encoder
        '''
        encoder_kwargs = {}
        if 'encoder_kwargs' in self.cfg:
            encoder_kwargs = self.cfg['encoder_kwargs']
        else:
            print( "Not using 'kwargs' arguments for encoder." )
        # Assume that if there are more than three dimensions [batch_size, rep_size]
        #   that the last dimension is the number of inputs: 
        #   [batch_size, rep_size, num_input]
        if 'num_input' not in self.cfg:
            self.cfg['num_input'] = 1 
            print('Setting num_input <- 1')
        encoder_output = []
        end_points = []

        # with tf.variable_scope("transfer") as scope: 
        for index in range(self.cfg['num_input']):
            if self.cfg['num_input'] == 1:
                input_i = input_imgs
            else:
                with tf.variable_scope(scope_name) as scope: 
                    input_i = tf.squeeze(input_imgs[:, index, :])
            reuse = (index > 0)
            ith_output, ith_end_points = self.cfg['encoder'](
                    input_i, 
                    is_training, 
                    reuse=reuse,
                    hidden_size=self.cfg[ 'hidden_size' ], 
                    scope=scope_name,
                    **encoder_kwargs )
            # scope.reuse_variables()
            encoder_output.append(ith_output)
            end_points.append(ith_end_points)

        if self.cfg['num_input'] == 1:
            end_points = end_points[0]
            encoder_output = encoder_output[0]

        self.encoder_endpoints = end_points
        return encoder_output


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
        img_in, representations = input_imgs 
        print("Encoder input shape")
        print(representations.shape)
        encoder_output = representations
        # encoder_output = self.build_encoder(representations, is_training)
        # current_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # self.encoder_output = encoder_output

        # if self.encoder_scope == 'funnel':
            # encoder_output = (img_in, encoder_output)

        self.decoder_savers = []
        self.decoder_ckpt_paths = self.cfg['model_paths']
        print("Building decoder")
        for i in [0, 1]:
            
            scope_name = 'transfer_{}_{}'.format(i, i+1)
            # with tf.variable_scope('transfer_{}_{}'.format(i, i+1)) as scope:
            encoder_output = self.build_encoder(encoder_output, is_training, scope_name)
            current_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self.encoder_output = encoder_output

            scope_name = 'decoder_{}'.format(i)
            with tf.variable_scope(scope_name) as scope: 
                self.decoders[i].secret_scope = scope_name
                self.decoders[i].build_model(encoder_output, is_training=False, targets=targets[i], masks=masks[i], privileged_input=img_in)
                new_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - current_vars
                for v in tuple(new_vars):
                    if 'global_step' in v.name:
                        new_vars.remove(v)
                
                def name_in_checkpoint(var):
                    return var.op.name.replace("decoder_{}/".format(i), "")

                variables_to_restore = {name_in_checkpoint(var):var for var in new_vars}
                self.decoder_savers.append(tf.train.Saver(variables_to_restore))
                current_vars |= new_vars
        print("Finished building decoder")

        self.decoder_saver = self.build_saver()    
        # use weight regularization
        if 'omit_weight_reg' in cfg and cfg['omit_weight_reg']:
            add_reg = False
        else:
            add_reg = True

        # get losses
        self.regularization_loss = tf.add_n( 
            tf.losses.get_regularization_losses(scope=self.encoder_scope), 
            name='losses/{}_regularization_loss'.format(self.encoder_scope) )
        self.input_images = img_in
        self.input_representations = representations
        self.target_images = targets
        self.losses = [d.total_loss for d in self.decoders] #self.decoder.losses
        self.total_loss = sum(self.losses) + self.regularization_loss #total_loss
        # self.init_op = tf.global_variables_initializer()

        # add summaries
        if self.extended_summaries:
            slim.summarize_variables()
            slim.summarize_weights()
            slim.summarize_biases()
            slim.summarize_activations()
        slim.summarize_tensor( self.regularization_loss, tag='losses/{}_regularizaton_loss'.format(self.encoder_scope) )
        slim.summarize_tensor( self.total_loss, tag='losses/{}_total_loss'.format(self.encoder_scope) )
        self.model_built = True

    def build_saver(self):
        decoder_savers = self.decoder_savers
        decoder_ckpt_paths = self.decoder_ckpt_paths
        class MultiSaver(object):
            def restore(self, sess, path):
                for i, restorer in enumerate(decoder_savers):
                    print("Restoring from {}".format(decoder_ckpt_paths[i]))
                    restorer.restore(sess, decoder_ckpt_paths[i])
        return MultiSaver()

    def get_train_step_fn( self ):
        '''
            Returns: 
                A train_step funciton which takes args:
                    (sess, train_ops, global_stepf)
        '''
        return partial( train_steps.discriminative_train_step_fn,
                return_accuracy=False )

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
        print('setting up losses...++')
        self.output_images = output_imgs
        self.target_images = desired_imgs
        self.masks = masks

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

        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_scope)
        
        # Create the optimizer train_op for the generator

        self.optimizer = optimize.build_optimizer( global_step=self.global_step, cfg=self.cfg )
        if 'clip_norm' in self.cfg:
            self.loss_op = optimize.create_train_op( 
                self.total_loss, self.optimizer, 
                update_global_step=True, clip_gradient_norm=self.cfg['clip_norm'],
                variables_to_train=vars_to_train )
        else:
            if self.is_training:
                self.loss_op = optimize.create_train_op( 
                    self.total_loss, self.optimizer, 
                    update_global_step=True, variables_to_train=vars_to_train )
            else:
                self.loss_op = optimize.create_train_op( 
                    self.total_loss, self.optimizer, 
                    is_training=False, update_global_step=True,
                    variables_to_train=vars_to_train )

        self.train_op = [ self.loss_op, 0 ]
        self.train_op_built = True
        return self.train_op

    def init_fn(self, sess):
        for i, restorer in enumerate(self.decoder_savers):
            model_path = self.decoder_ckpt_paths[i]
            print('******* USING SAVED TARGET MODEL FOR DECODER {}*******'.format(i))
            print('restoring target model ...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                print("Trying to load checkpoint from {}...".format(model_path))
                restorer.restore(sess, model_path)
                print("Success!")
            else:
                print("Trying to find checkpoint in directory {}...".format(model_path))
                restorer.restore(sess, ckpt.model_checkpoint_path)
                print("Success!")
        print('target model restored')