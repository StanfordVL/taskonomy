'''Segmentation encoder-decoder model

Assumes there is one input and one output.

The output is an embedding vector for each pixel, the size of the embedding vector is
the number of channels for target specified by the config file, aka 'target_num_channel'.

The target is a vector of pixel location & segmentation ID. The number of pixels is specified 
by 'num_pixels' in config file.

    Model-specific config.py options: (inherits from models.base_net)
        'batch_size': An int. The number of images to use in a batch
        'hidden_size': An int. The number of hidden neurons to use. 
        'target_num_channels': The number of channels to output from the decoder
        'num_pixels': The number of pixels sampled for metric learning.

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

from   models.encoder_decoder_segmentation import SegmentationED
import losses.all as losses_lib
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pdb
import optimizers.train_steps as train_steps
import optimizers.ops as optimize
from models.utils import add_fc_layer
from functools import partial

class SemSegED(SegmentationED):
    ''' Segmentation encoder decoder model
    Encodes an input into a low-dimensional representation and reconstructs
    the input from the low-dimensional representation. Uses metric loss.

    Metric loss follows the function of paper: Semantic Instance Segmentation via Deep Metric Learning
    (Equation 1)

    Assumes inputs are scaled to [0, 1] (which will be rescaled to [-1, 1].
    '''

    def __init__(self, global_step, cfg):
        '''
        Args:
            cfg: Configuration.
        '''
        super(SemSegED, self).__init__(global_step, cfg)
        if 'hidden_size' not in cfg:
            raise ValueError( "config.py for encoder-decoder must specify 'hidden_size'" )
        if 'num_pixels' not in cfg:
            raise ValueError( "config.py for segmentation must specify 'num_pixels'(how many pixels to sample)")

        self.batch_size = cfg['batch_size'] 
        self.num_pixels = cfg['num_pixels']

        idxes = np.asarray([range(self.batch_size)] * self.num_pixels).T
        self.batch_index_slice = tf.cast(tf.stack(idxes), cfg['target_dtype'])
        
        self.input_type = cfg['input_dtype']

        self.cfg = cfg


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
            fir, sec, seg_id, weights = tf.unstack(idx_segments, axis=last_axis)

            idxes = tf.stack([self.batch_index_slice, fir, sec], axis=last_axis)
            self.embed = tf.gather_nd( output_vectors, idxes )
            embed = self.embed
            self.class_logits = add_fc_layer(output_vectors, self.is_training, self.cfg['num_classes']) 
            self.selected_logits = tf.gather_nd( self.class_logits, idxes )
            class_logits_flat = tf.reshape(self.selected_logits, [-1, self.cfg['num_classes']])
            seg_id_flat = tf.reshape(seg_id, [-1])
            weights_flat = tf.reshape(weights, [-1])
            temp = losses_lib.get_sparse_softmax_loss(
                    class_logits_flat, seg_id_flat, weights_flat)
            self.softmax_loss = tf.reduce_mean(temp) 
            tf.add_to_collection(tf.GraphKeys.LOSSES, self.softmax_loss)

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


