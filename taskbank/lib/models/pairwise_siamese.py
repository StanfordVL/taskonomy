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
from models.util import *
import optimizers.train_steps as train_steps
import optimizers.ops as optimize
import pdb

class PairWiseSiamese(StandardSiamese):
    '''
    '''
    
    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(PairWiseSiamese, self).__init__(global_step, cfg)
        self.cfg = cfg

        if 'hidden_size' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'hidden_size'")
        if 'num_input' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'num_input'")
        if 'encoder' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'encoder'")
        if 'metric_net' not in cfg:
            raise ValueError("config.py for Siamese Network must specify 'metric_net'")
        if 'output_size' not in cfg:
            raise ValueError("config.py for pairwise Siamese Network must have effective output size")
    
        if 'loss_threshold' in cfg:
            self.threshold = tf.constant(cfg['loss_threshold'])
        else:
            self.threshold = None

    def build_encoder(self, input_imgs, is_training):
        '''Builds a single encoder.
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
        encoder_output = []
        end_points = []
        
        with tf.variable_scope("siamese") as scope:
            inputs = []
            for index in range(self.cfg['num_input']):
                inputs.append(tf.squeeze(input_imgs[:,index,:,:,:]))
            for i in range(self.cfg['num_input']):
                for j in range(self.cfg['num_input']):
                    if i == j:
                        continue
                    else:
                        if i == 0:
                            ith_output, ith_end_points = self.cfg['encoder'](
                                    inputs[i],
                                    is_training,
                                    reuse=None,
                                    hidden_size=self.cfg['hidden_size'],
                                    scope=scope,
                                    **encoder_kwargs)
                            jth_output, jth_end_points = self.cfg['encoder'](
                                    inputs[j],
                                    is_training,
                                    reuse=True,
                                    hidden_size=self.cfg['hidden_size'],
                                    scope=scope,
                                    **encoder_kwargs)
                            
                        else:
                            ith_output, ith_end_points = self.cfg['encoder'](
                                    input_i,
                                    is_training,
                                    reuse=True,
                                    hidden_size=self.cfg['hidden_size'],
                                    scope=scope,
                                    **encoder_kwargs)
                            jth_output, jth_end_points = self.cfg['encoder'](
                                    inputs[j],
                                    is_training,
                                    reuse=True,
                                    hidden_size=self.cfg['hidden_size'],
                                    scope=scope,
                                    **encoder_kwargs)
                        
                        scope.reuse_variables()
                        ith_encoder_output = tf.concat(values=[ith_output, jth_output], concat_dim=1)
                        encoder_output.append(ith_encoder_output)
                        end_points.append(ith_end_points)
                        end_points.append(jth_end_points)

        self.encoder_endpoints = end_points
        return encoder_output

    def build_siamese_output_postprocess(self, encoder_output, is_training):
        '''Build the post-process on siamese network structure output.
        The default approach will be a three layer fully connected networks
        Args:
            encoder_output: a list of tensors of output representations of each input image
            is_training: flag for wheter the model is in training mode.
        Returns:
            final_output: final output for the whole model 
        '''
        metric_kwargs = {}
        if 'metric_kwargs' in self.cfg:
            metric_kwargs = self.cfg['metric_kwargs']
        else: 
            raise ValueError("config.py for Siamese Network must specify 'metric_kwargs'")

        final_output = []
        end_points = []

        for i in range(len(encoder_output)):
            final_output_ith, end_points_ith = self.cfg['metric_net'](
                    encoder_output[i],
                    is_training,
                    **metric_kwargs)
            metric_kwargs['reuse'] = True
            final_output.append(final_output_ith)
            end_points.append(end_points_ith)

        final_output = tf.concat(values=final_output, concat_dim=1)
        print('\t building multilayers FC encoder')
        with tf.variable_scope("final_linear") as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(weight_decay) ):

                final_output = add_fc_layer(final_output, self.cfg['output_size'], activation_fn=None, scope='fc') 

        self.metric_endpoints = end_points
        return final_output

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

        encoder_output = self.build_encoder(input_imgs, is_training)

        final_output = self.build_siamese_output_postprocess(encoder_output, is_training)
        
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
        self.encoder_output = encoder_output
        self.decoder_output = final_output
        self.losses = losses
        self.total_loss = total_loss
        self.masks = masks

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
                                final_output, 
                                target,
                                name='softmax_loss'))
            else:
                # If it's not softmax, it's l2 norm loss.
                self.accuracy = 0
                siamese_loss = losses_lib.get_l2_loss(
                    final_output,
                    target,
                    scope='d1') 

                if self.threshold is not None:
                
                    siamese_loss = tf.cond(tf.greater(siamese_loss, self.threshold), 
                                            lambda: self.threshold + self.threshold * tf.log(siamese_loss / self.threshold),
                                            lambda: siamese_loss)
                          
        tf.add_to_collection(tf.GraphKeys.LOSSES, siamese_loss)

        losses = [siamese_loss]
        return losses

