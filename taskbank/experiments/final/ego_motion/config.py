from __future__ import absolute_import, division, print_function

import functools
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.insert( 1, os.path.realpath( '../../models' ) )
sys.path.insert( 1, os.path.realpath( '../../lib' ) )

import data.load_ops as load_ops
from   data.load_ops import mask_if_channel_le
from   data.task_data_loading import load_and_specify_preprocessors, load_and_specify_preprocessors_for_single_filename_to_imgs

from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
from   models.gan_discriminators import pix2pix_discriminator
from   models.resnet_v1 import resnet_v1_50_16x16
from   models.sample_models import *
from   models.utils import leaky_relu

def get_cfg( nopause=False ):
    cfg = {}
    cfg['is_discriminative'] = True
    # params
    cfg['num_epochs'] = 30
    cfg['model_path'] = None 

    # logging
    config_dir = os.path.dirname(os.path.realpath( __file__ ))
    task_name = os.path.basename( config_dir )

    # model
    cfg['model_type'] = architectures.CycleSiamese
    cfg['weight_decay'] = 2e-6
    cfg['instance_noise_sigma'] = 0.1
    cfg['instance_noise_anneal_fn'] = tf.train.inverse_time_decay
    cfg['instance_noise_anneal_fn_kwargs'] = {
        'decay_rate': 0.2,
        'decay_steps': 1000
    }

    batch_size = 8 
    cfg['encoder'] = resnet_encoder_fully_convolutional_16x16x8
    cfg['hidden_size'] = 1024
    cfg['encoder_kwargs'] = {
        'resnet_build_fn' : resnet_v1_50_16x16,
        'weight_decay': cfg['weight_decay'],
        'flatten': True,
        'batch_size': batch_size

    }

    cfg['return_accuracy']=False
   
    # learning
    cfg['initial_learning_rate'] = 1e-4
    cfg[ 'optimizer' ] = tf.train.AdamOptimizer
    cfg[ 'optimizer_kwargs' ] = {}
    cfg[ 'discriminator_learning_args' ] = {
        'initial_learning_rate':1e-5,
        'optimizer': tf.train.GradientDescentOptimizer,
        'optimizer_kwargs': {}
    }
    def pwc(initial_lr, **kwargs):
        global_step = kwargs['global_step']
        del kwargs['global_step']
        return tf.train.piecewise_constant(global_step, **kwargs)
    cfg['learning_rate_schedule'] = pwc
    cfg['learning_rate_schedule_kwargs' ] = {
            'boundaries': [np.int64(0), np.int64(80000)], # need to be int64 since global step is...
            'values': [cfg['initial_learning_rate'], cfg['initial_learning_rate']/10]
    }    
    #cfg['clip_norm'] = 1
    
    # inputs
    cfg['input_dim'] = (256, 256)  # (1024, 1024)
    cfg['input_num_channels'] = 3
    cfg['input_dtype'] = tf.float32
    cfg['num_input'] = 3
    cfg['input_domain_name'] = 'rgb'
    cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image   
    cfg['input_preprocessing_fn_kwargs'] = {
        'new_dims': cfg['input_dim'],
        'new_scale': [-1, 1]
    }
    cfg['single_filename_to_multiple']=True

    # outputs
    cfg['target_dim'] = 18  # (1024, 1024)
    cfg['target_dtype'] = tf.float32
    cfg['target_from_filenames'] = load_ops.triplet_fixated_egomotion

    # Post processing
    cfg['metric_net'] = encoder_multilayers_fc
    cfg['metric_kwargs'] = {
        'hidden_size': 1024,
        'layer_num': 3,
        'output_size': 6
    } 
    cfg['l2_loss']=True

    # input pipeline
    cfg['preprocess_fn'] = load_and_specify_preprocessors_for_single_filename_to_imgs
    cfg['randomize'] = False 
    cfg['num_read_threads'] = 30 
    cfg['batch_size'] = 8 
    cfg['inputs_queue_capacity'] = 4096 

    # Checkpoints and summaries
    cfg['summary_save_every_secs'] = 0
    cfg['checkpoint_save_every_secs'] = 600 
    RuntimeDeterminedEnviromentVars.register_dict( cfg )  # These will be loaded at runtime
    print_cfg( cfg, nopause=nopause )
    return cfg

def print_cfg( cfg, nopause=False ):
    print('-------------------------------------------------')
    print('config:')
    template = '\t{0:30}{1}'
    for key in sorted( cfg.keys() ):
        print(template.format(key, cfg[key]))
    print('-------------------------------------------------')
    
    if nopause:
        return
    raw_input('Press Enter to continue...')
    print('-------------------------------------------------')
