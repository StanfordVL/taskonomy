"""
    utils.py

    Contains some useful functions for creating models
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import pickle
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import concurrent.futures


import init_paths
import data.load_ops as load_ops
from   data.load_ops import create_input_placeholders_and_ops, get_filepaths_list
import general_utils
import optimizers.train_steps as train_steps
import models.architectures as architectures

def get_available_devices():
    from tensorflow.python.client import device_lib
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return device_lib.list_local_devices()

def get_max_steps( num_samples_epoch, cfg , is_training=True):
    if cfg['num_epochs']:
        max_steps = num_samples_epoch * cfg['num_epochs'] // cfg['batch_size']
    else: 
        max_steps = None
    if not is_training:
        max_steps = num_samples_epoch // cfg['batch_size']
    print( 'number of steps per epoch:',
           num_samples_epoch // cfg['batch_size'] )
    print( 'max steps:', max_steps )
    return max_steps

def load_config( cfg_dir, nopause=False ):
    ''' 
        Raises: 
            FileNotFoundError if 'config.py' doesn't exist in cfg_dir
    '''
    if not os.path.isfile( os.path.join( cfg_dir, 'config.py' ) ):
        raise ImportError( 'config.py not found in {0}'.format( cfg_dir ) )
    import sys
    sys.path.insert( 0, cfg_dir )
    from config import get_cfg
    cfg = get_cfg( nopause )
    # cleanup
    try:
        del sys.modules[ 'config' ]
    except:
        pass
    sys.path.remove(cfg_dir)

    return cfg

def print_start_info( cfg, max_steps, is_training=False ):
    model_type = 'training' if is_training else 'testing'
    print("--------------- begin {0} ---------------".format( model_type ))
    print('number of epochs', cfg['num_epochs'])
    print('batch size', cfg['batch_size'])
    print('total number of training steps:', max_steps)


##################
# Model building
##################
def create_init_fn( cfg, model ):
    # restore model
    if cfg['model_path'] is not None:
        print('******* USING SAVED MODEL *******')
        checkpoint_path = cfg['model_path']
        model['model'].decoder
        # Create an initial assignment function.
        def InitAssignFn(sess):
            print('restoring model...')
            sess.run(init_assign_op, init_feed_dict)
            print('model restored')

        init_fn = InitAssignFn
    else:
        print('******* TRAINING FROM SCRATCH *******')
        init_fn = None
    return init_fn 

def setup_and_restore_model( sess, inputs, cfg, is_training=False ):
    model = setup_model( inputs, cfg, is_training=False )
    model[ 'saver_op' ].restore( sess, cfg[ 'model_path' ] )
    return model

def setup_input( cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds input tensors from the config.
    '''
    inputs = {}
    # Generate placeholder input tensors
    placeholders, batches, load_and_enqueue, enqueue_op = create_input_placeholders_and_ops( cfg )
    
    input_batches = list( batches ) # [ inputs, targets, mask, data_idx ]
    
    inputs[ 'enqueue_op' ] = enqueue_op
    inputs[ 'load_and_enqueue' ] = load_and_enqueue
    inputs[ 'max_steps' ] = 6666 
    inputs[ 'num_samples_epoch' ] = 6666 

    inputs[ 'input_batches' ] = input_batches
    inputs[ 'input_batch' ] = input_batches[0]
    inputs[ 'target_batch' ] = input_batches[1]
    inputs[ 'mask_batch' ] = input_batches[2]
    inputs[ 'data_idxs' ] = input_batches[3]
    inputs[ 'placeholders' ] = placeholders
    inputs[ 'input_placeholder' ] = placeholders[0]
    inputs[ 'target_placeholder' ] = placeholders[1]
    inputs[ 'mask_placeholder' ] = placeholders[2]
    inputs[ 'data_idx_placeholder' ] = placeholders[3]
    return inputs


def setup_model( inputs, cfg, is_training=False ):
    '''
        Sets up the `model` dict, and instantiates a model in 'model',
        and then calls model['model'].build

        Args:
            inputs: A dict, the result of setup_inputs
            cfg: A dict from config.py
            is_training: Bool, used for batch norm and the like

        Returns:
            model: A dict with 'model': cfg['model_type']( cfg ), and other
                useful attributes like 'global_step'
    '''
    validate_model( inputs, cfg )
    model = {}
    model[ 'global_step' ] = slim.get_or_create_global_step()

    model[ 'input_batch' ] = tf.identity( inputs[ 'input_batch' ] )
    if 'representation_batch' in inputs:
        model[ 'representation_batch' ] = tf.identity( inputs[ 'representation_batch' ] )
    model[ 'target_batch' ] = tf.identity( inputs[ 'target_batch' ] )
    model[ 'mask_batch' ] = tf.identity( inputs[ 'mask_batch' ] )
    model[ 'data_idxs' ] = tf.identity( inputs[ 'data_idxs' ] )

    # instantiate the model
    if cfg[ 'model_type' ] == 'empty':
        return model
    else:
        model[ 'model' ] = cfg[ 'model_type' ]( global_step=model[ 'global_step' ], cfg=cfg )

    # build the model
    if 'representation_batch' in inputs:
        input_imgs = (inputs[ 'input_batch' ], inputs[ 'representation_batch' ])
    else:
        input_imgs = inputs[ 'input_batch' ]
    model[ 'model' ].build_model( 
            input_imgs=input_imgs,
            targets=inputs[ 'target_batch' ],
            masks=inputs[ 'mask_batch' ],
            is_training=is_training )
    
    if is_training:
        model[ 'model' ].build_train_op( global_step=model[ 'global_step' ] )
        model[ 'train_op' ] = model[ 'model' ].train_op
        model[ 'train_step_fn' ] = model[ 'model' ].get_train_step_fn()
        model[ 'train_step_kwargs' ] = train_steps.get_default_train_step_kwargs( 
            global_step=model[ 'global_step' ],
            max_steps=inputs[ 'max_steps' ],
            log_every_n_steps=10 )

    #model[ 'init_op' ] = model[ 'model' ].init_op
    if hasattr( model['model'], 'init_fn' ):
        model[ 'init_fn' ] = model['model'].init_fn
    else:
        model[ 'init_fn' ] = None

    max_to_keep = cfg['num_epochs'] * 2
    if 'max_ckpts_to_keep' in cfg:
        max_to_keep = cfg['max_ckpts_to_keep']
    model[ 'saver_op' ] = tf.train.Saver(max_to_keep=max_to_keep)
    return model

def validate_model( inputs, cfg ):
    general_utils.validate_config( cfg )



