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




class TransferNet(BaseNet):
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
        super(TransferNet, self).__init__(global_step, cfg)
        if 'hidden_size' not in cfg:
            raise ValueError( "config.py for encoder-decoder must specify 'hidden_size'" )
        self.cfg = cfg

        # Load the target model
        # sys.path.insert(0, cfg['config_dir_target'])
        # import config as target_config
        # sys.path.pop(0)
        self.target_cfg = self.cfg['target_cfg'] #target_config.get_cfg(nopause=True)
         
        if 'replace_target_decoder' in self.cfg \
            and self.cfg['replace_target_decoder']:
            if 'decoder' in self.target_cfg and 'metric_net' in self.target_cfg:
                raise ValueError("Trying to replace target decoder with {} but both 'decoder' and 'metric_net' are in target cfg")

            if 'decoder' in self.target_cfg:
                self.target_cfg['decoder'] = self.cfg['replace_target_decoder']
            elif 'metric_net' not in self.target_cfg:
                #self.target_cfg['metric_net'] = self.cfg['replace_target_decoder']
            #else:
                raise ValueError("Trying to replace target decoder with {} but neither 'decoder' and 'metric_net' are in target cfg")

        self.decoder = self.target_cfg['model_type'](global_step, self.target_cfg)
        self.decoder.decoder_only = True
        self.encoder_scope = 'transfer'
        self.global_step = global_step
        self.finetune_decoder = 'finetune_decoder' in cfg and cfg['finetune_decoder']
        self.retrain_decoder = 'retrain_decoder' in cfg and cfg['retrain_decoder']
        self.unlock_decoder = 'unlock_decoder' in cfg and cfg['unlock_decoder']
        # if 'multiple_input_tasks' in self.cfg:
        #     self.encoder_scope = 'funnel'
        # self.decoder.decoder_only = False

    def build_encoder(self, input_imgs, is_training):
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
         
        scope_name = self.encoder_scope
        # with tf.variable_scope("transfer") as scope: 
        for index in range(self.cfg['num_input']):
            if self.cfg['num_input'] == 1:
                input_i = input_imgs
            else:
                with tf.variable_scope(scope_name) as scope: 
                    if type(input_imgs) is tuple:
                        # input_i = (tf.squeeze(input_imgs[0][:,index,:]),
                        #         tf.squeeze(input_imgs[1][:,index,:]) )
                        input_i = (input_imgs[0][:,index,:],
                                input_imgs[1][:,index,:] )
                    else:
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

            There are multiple types of transfers that we might care about:
                'funnel': Train the transfer net on several types of representations
                'finetune_decoder': Train the transfer net along with the decoder. The decoder
                    should already be pretrained for the target task
                'retrain decoder': Take a trained transfer net model, clear the decoder, 
                    and train it from scratch after using the transferred representations. 
        '''
        print('building model')
        cfg = self.cfg

        # Get image and representation parts
        input_placeholder = input_imgs
        img_in, representations = input_placeholder 
        self.finetune_src_encoder_imagenet = 'finetune_encoder_imagenet' in cfg
        if self.finetune_src_encoder_imagenet:
           
            self.src_encoder = cfg['input_cfg']['model_type'](self.global_step, cfg['input_cfg'])
            representations = self.src_encoder.build_encoder(img_in, is_training=is_training) 
            encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)      
            for v in tuple(encoder_vars):
                if 'global_step' in v.name:
                    encoder_vars.remove(v)
            self.encoder_vars = encoder_vars
            self.encoder_saver_imagenet = tf.train.Saver(encoder_vars)
            #self.finetune_encoder_imagenet_saver = tf.
        input_placeholder = (img_in, representations)
        input_imgs = input_placeholder
        print("Represntation input shape")
        print(representations.shape)
    
        # Determine what part of the model we are training/freezing
        self.is_training = is_training
        self.training_encoder = is_training
        self.training_decoder = is_training
        self.restoring_encoder = True
        self.restoring_decoder = True
        if self.retrain_decoder: 
            self.training_encoder = False # Retraining decoder means that we have trained a transfer
            self.restoring_encoder = True # Retraining decoder means that we have trained a transfer
            self.restoring_decoder = self.finetune_decoder
        else:
            self.restoring_encoder = False
            self.restoring_decoder = True
            if not self.finetune_decoder:
                self.training_decoder = False

        if self.unlock_decoder:
            self.restoring_encoder = False
            self.restoring_decoder = False
            self.training_decoder = is_training
            self.retrain_decoder = True

        # Build encoder
        if not 'metric_net_only' in cfg:
            encoder_output = self.build_encoder(input_placeholder, self.training_encoder)
        else:
            encoder_output = representations
        current_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self.encoder_output = encoder_output

        # funnel networks are going to other versions of perceptual_transfer nets
        if self.encoder_scope == 'funnel':
            encoder_output = (img_in, encoder_output)

        # What to do with the decoder
        print("Building decoder")
        self.decoder.build_model(
            encoder_output,
            is_training=self.training_decoder,
            targets=targets,
            masks=masks,
            privileged_input=img_in)
        
        # Make the saver which we will restore from
        if self.finetune_decoder: # self.retrain_decoder:
            decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)      
        else:
            decoder_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - current_vars
        for v in tuple(decoder_vars):
            if 'global_step' in v.name:
                decoder_vars.remove(v)

        self.decoder_saver = tf.train.Saver(decoder_vars)
        print("Finished building decoder")

       
        # use weight regularization
        if 'omit_weight_reg' in cfg and cfg['omit_weight_reg']:
            add_reg = False
        else:
            add_reg = True


        regularization_scope = self.encoder_scope
        if self.retrain_decoder or self.finetune_decoder:
            regularization_scope = None
        # get losses
        regularization_loss = tf.add_n( 
            slim.losses.get_regularization_losses(scope=regularization_scope), 
            name='losses/{}_regularization_loss'.format(regularization_scope) )

        total_loss = self.decoder.total_loss + regularization_loss
        self.input_images = img_in
        self.input_representations = representations
        self.target_images = targets
        self.losses = self.decoder.losses
        self.total_loss = total_loss
        # self.init_op = tf.global_variables_initializer()

        # add summaries
        if self.extended_summaries:
            slim.summarize_variables()
            slim.summarize_weights()
            slim.summarize_biases()
            slim.summarize_activations()
        # slim.summarize_collection(tf.GraphKeys.LOSSES)
        slim.summarize_tensor( regularization_loss, tag='losses/{}_regularizaton_loss'.format(self.encoder_scope) )
        slim.summarize_tensor( total_loss, tag='losses/{}_total_loss'.format(self.encoder_scope) )
        self.model_built = True

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
        print('setting up losses...')
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


        if self.training_decoder and not self.training_encoder:
            vars_to_train = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            vars_to_train = vars_to_train - set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_scope))
            vars_to_train = list(vars_to_train)
        elif self.training_decoder and self.training_encoder:
            vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_scope)
        print("---------------------------Trainable Variables---------------------------")
        for name, var in sorted([(v.name, v) for v in vars_to_train]):
            print(var)
        print("---------------------------END Trainable Variables---------------------------")

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
        if self.finetune_src_encoder_imagenet and self.is_training:
            print(self.cfg['src_encoder_ckpt'])
            s3_dir = "/".join(self.cfg['src_encoder_ckpt'].split('/')[4:])
            import subprocess
            subprocess.call("aws s3 cp s3://task-preprocessing-512-oregon/{}.index /home/ubuntu/temp".format(s3_dir), shell=True)
            subprocess.call("aws s3 cp s3://task-preprocessing-512-oregon/{}.meta /home/ubuntu/temp".format(s3_dir), shell=True)
            subprocess.call("aws s3 cp s3://task-preprocessing-512-oregon/{}.data-00000-of-00001 /home/ubuntu/temp".format(s3_dir), shell=True)
            self.cfg['src_encoder_ckpt'] = '/home/ubuntu/temp/{}'.format(self.cfg['src_encoder_ckpt'].split('/')[-1]) 
            self.encoder_saver_imagenet.restore(sess, self.cfg['src_encoder_ckpt'])
            print('Finished Loading...')
        if self.retrain_decoder and not self.finetune_decoder:
            return
        print('******* USING SAVED TARGET MODEL *******')
        print('restoring target model ...')
        ckpt = tf.train.get_checkpoint_state(self.cfg['model_path'])
        if ckpt is None:
            print("Trying to load checkpoint from {}....".format(self.cfg['model_path']))
            self.decoder_saver.restore(sess, self.cfg['model_path'])
            print("Success!")
        else:
            print("Trying to find checkpoint in directory {}...".format(self.cfg['model_path']))
            self.decoder_saver.restore(sess, ckpt.model_checkpoint_path)
            print("Success!")
        
        # var = [v for v in tf.global_variables() if 'decoder/deconv10/BatchNorm/moving_mean' in v.name][0]
        # print("CURRENT VAR:", var.name)
        # print(sess.run(var))


        # correct_var = tf.contrib.framework.load_variable(
        #     '/home/ubuntu/s3/model_log/autoencoder/model.permanent-ckpt',
        #     'decoder/deconv10/BatchNorm/moving_mean')
        # print("CORRECT VAR:")
        # print(correct_var)
        # exit(0)

        # print(tf.contrib.framework.list_variables(ckpt.model_checkpoint_path))
        # test_var = 'three_layer_fc_network/fc3/weights'
        # print(tf.contrib.framework.load_variable(ckpt.model_checkpoint_path, test_var ))
        # print(sess.run(
        #     [v for v in 
        #         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        #         if test_var in v.name][0]))
        print('target model restored')
