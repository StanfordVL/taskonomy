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
from   data.task_data_loading import load_and_preprocess_img, load_and_preprocess_img_fast, load_and_specify_preprocessors, create_input_placeholders_and_ops_transfer, load_target
from   general_utils import RuntimeDeterminedEnviromentVars
import general_utils
import models.architectures as architectures

from   models.gan_discriminators import pix2pix_discriminator
from   models.resnet_v1 import resnet_v1_50
from   models.sample_models import *
from   models.utils import leaky_relu

PRE_INPUT_TASK = "<PRE_INPUT_TASK>" # e.g. autoencoder
INPUT_TASK = "<INPUT_TASK>" # e.g. autoencoder
TARGET_TASK = "<TARGET_TASK>" # e.g. vanishing_point
SAVE_TO_S3 = True

def get_cfg( nopause=False ):
    root_dir = '/home/ubuntu/task-taxonomy-331b'
    cfg = {}

    representation_task = INPUT_TASK
    representation_dir = 'representations'
    transitive = False
    if PRE_INPUT_TASK and PRE_INPUT_TASK != "PRE_INPUT_TASK":
        representation_task = PRE_INPUT_TASK + '__' + INPUT_TASK + '__' +  '1024'
        representation_dir = 'representations_transfer_1024'
        transitive = True

    ### -----CHANGE HERE--------------------------
    
    # cfg['representations_file'] = '/home/ubuntu/s3/model_log/{task}/{task}_representations.pkl'.format(task=INPUT_TASK)
    cfg['train_representations_file'] = '/home/ubuntu/s3/model_log/{subdir}/{task}_{train_split}_representations.pkl'.format(
        subdir=representation_dir, task=representation_task, train_split='train' if transitive else 'val')
    cfg['val_representations_file'] = '/home/ubuntu/s3/model_log/{subdir}/{task}_{train_split}_representations.pkl'.format(
        subdir=representation_dir, task=representation_task, train_split='val' if transitive else 'test')
    cfg['input_config_dir'] = '/home/ubuntu/task-taxonomy-331b/experiments/aws_second/{}'.format(INPUT_TASK)
    cfg['config_dir_target'] = '/home/ubuntu/task-taxonomy-331b/experiments/aws_second/{}'.format(TARGET_TASK)

    # Where the target model is saved
    cfg['model_path'] = '/home/ubuntu/s3/model_log/{}/model.permanent-ckpt'.format(TARGET_TASK)
    # cfg['model_path'] = '/home/ubuntu/s3/model_log/{}/logs/slim-train'.format(TARGET_TASK)
    ### -------------------------------------------    
    # Automatically populate data loading variables
    # input_cfg = general_utils.load_config(cfg['input_config_dir'], nopause=True)

    target_cfg = general_utils.load_config(cfg['config_dir_target'], nopause=True)
    general_utils.update_keys(cfg, "input", target_cfg)
    general_utils.update_keys(cfg, "target", target_cfg)
    general_utils.update_keys(cfg, "is_discriminative", target_cfg)
    general_utils.update_keys(cfg, "num_input", target_cfg)
    general_utils.update_keys(cfg, "single_filename_to_multiple", target_cfg)
    general_utils.update_keys(cfg, "preprocess_fn", target_cfg)
    general_utils.update_keys(cfg, "mask_by_target_func", target_cfg)
    general_utils.update_keys(cfg, "depth_mask", target_cfg)
    general_utils.update_keys(cfg, "mask_fn", target_cfg)
    general_utils.update_keys(cfg, "find_target_in_config", target_cfg)
    
    cfg['target_model_type'] = target_cfg['model_type']

    # kludge
    # cfg['list_of_fileinfos'] = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/train_image_split_0.npy') )    
    cfg['train_list_of_fileinfos'] = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/val_image_split_0.npy') )    
    cfg['val_list_of_fileinfos'] = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/test_image_split_0.npy') )    
    
    # Now use 'val' for training and 'test' for validation... :(
    tmp = target_cfg['train_filenames']
    target_cfg['train_filenames'] = str(target_cfg['val_filenames'])
    target_cfg['val_filenames'] = str(target_cfg['test_filenames'])
    target_cfg['test_filenames'] = None # str(tmp)

    general_utils.update_keys(cfg, "train_filenames", target_cfg)
    general_utils.update_keys(cfg, "val_filenames", target_cfg)
    general_utils.update_keys(cfg, "test_filenames", target_cfg)
    general_utils.update_keys(cfg, "dataset_dir", target_cfg)
    
    cfg['target_cfg'] = target_cfg

    # params
    cfg['root_dir'] = root_dir
    # cfg['num_epochs'] = 30 if INPUT_TASK == 'random' else 4
    cfg['num_epochs'] = 4

    cfg['weight_decay'] = 1e-6  # 1e-7, 1
    cfg['model_type'] = architectures.TransferNet

    ## DATA LOADING
    # representations
    cfg['representation_dim'] = 1024
    cfg['representation_dtype'] = tf.float32

    ## SETUP MODEL
    # Transfer
    cfg['encoder'] = encoder_multilayers_fc_bn_res_no_dropout # encoder_multilayers_fc_bn
    if INPUT_TASK == 'random':
        cfg['encoder'] = encoder_multilayers_fc_bn_res_no_dropout_normalize_input
    # cfg['encoder'] = encoder_multilayers_fc
    cfg['hidden_size'] = int("<HIDDEN_SIZE>")
    cfg['encoder_kwargs'] = {
        'output_size': cfg['representation_dim'],
        'batch_norm_epsilon': 1e-5,
        'batch_norm_decay': 0.9, #0.95
        'weight_decay': cfg['weight_decay'],
        'layer_num': 3
    }

    # learning
    general_utils.update_keys(cfg, "initial_learning_rate", target_cfg)
    general_utils.update_keys(cfg, "optimizer", target_cfg)
    general_utils.update_keys(cfg, "clip_norm", target_cfg)
    cfg['learning_rate_schedule'] = tf.train.piecewise_constant
    cfg['learning_rate_schedule_kwargs' ] = {
        'boundaries': [0, 2500],
        'values': [cfg['initial_learning_rate'], cfg['initial_learning_rate']/10]
    }

    # cfg['initial_learning_rate'] = 1e-4  # 1e-6, 1e-1
    # cfg[ 'optimizer' ] = tf.train.AdamOptimizer
    # cfg[ 'optimizer_kwargs' ] = {}

    ## LOCATIONS
    # logging
    config_dir = os.path.dirname(os.path.realpath( __file__ ))
    task_name = os.path.basename( config_dir )
    if transitive:
        if SAVE_TO_S3:
            log_root = '/home/ubuntu/s3/experiment_models/transfer_transitive_{}/{}/{}'.format(cfg['hidden_size'], INPUT_TASK, TARGET_TASK)
        else:
            log_root = '/home/ubuntu/experiment_models/transfer_transitive_{}/{}/{}'.format(cfg['hidden_size'], INPUT_TASK, TARGET_TASK)
    else:
        if SAVE_TO_S3:
            log_root = '/home/ubuntu/s3/experiment_models/transfer_{}/{}/{}'.format(cfg['hidden_size'], INPUT_TASK, TARGET_TASK)
        else:
            log_root = '/home/ubuntu/experiment_models/transfer_{}/{}/{}'.format(cfg['hidden_size'], INPUT_TASK, TARGET_TASK)
    cfg['log_root'] = log_root
    cfg['log_dir'] = os.path.join(log_root, 'logs')

    # input pipeline
    data_dir = '/home/ubuntu/s3'
    cfg['meta_file_dir'] = 'assets/aws_data'
    cfg['create_input_placeholders_and_ops_fn'] = create_input_placeholders_and_ops_transfer
    cfg['randomize'] = True 
    cfg['num_read_threads'] = 300
    cfg['batch_size'] = 32
    cfg['inputs_queue_capacity'] = 4096  
    

    # Checkpoints and summaries
    cfg['summary_save_every_secs'] = 300
    cfg['checkpoint_save_every_secs'] = 3000 

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
