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
import models.transfer_models as transfer_models
from   models.utils import leaky_relu

PRE_INPUT_TASK = "<PRE_INPUT_TASK>" # e.g. autoencoder
INPUT_TASK = "<INPUT_TASK>" # e.g. autoencoder
TARGET_TASK = "<TARGET_TASK>" # e.g. vanishing_point
NUM_LAYERS = int("<NUM_LAYERS>") # e.g. vanishing_point
KERNEL_SIZE = int("<KERNEL_SIZE>")

REPLACE_TARGET_DECODER_FUNC_NAME = "<REPLACE_TARGET_DECODER_FUNC_NAME>"
REPLACE_TARGET_DECODER_LAYERS = "<REPLACE_TARGET_DECODER_LAYERS>"
VAL_DATA_USED="<REPLACE_VAL_DATA_USED>"
CFG_DIR = "final"
SAVE_TO_S3 = True


FLATTEN_TASKS = [
    'fix_pose', 
    'non_fixated_pose',
    'point_match',
    'ego_motion',
    'jigsaw',
    'vanishing_point',
    'vanishing_point_well_defined'
]

def get_cfg( nopause=False ):
    root_dir = '/home/ubuntu/task-taxonomy-331b'
    cfg = {}
    import pdb

    representation_task = INPUT_TASK
    representation_dir = 'representations'
    transitive = False
    if PRE_INPUT_TASK and PRE_INPUT_TASK != "PRE_INPUT_TASK":
        representation_task = PRE_INPUT_TASK + '__' + INPUT_TASK + '__' +  '1024'
        representation_dir = 'representations_transfer_1024'
        transitive = True
    cfg['finetune_decoder'] = ("<FINETUNE_DECODER>" == "True")
    cfg['retrain_decoder'] = ("<RETRAIN_DECODER>" == "True")
    cfg['unlock_decoder'] = ("<UNLOCK_DECODER>" == "True")

    ### -----CHANGE HERE--------------------------
    cfg['config_dir_input'] = '/home/ubuntu/task-taxonomy-331b/experiments/{}/{}'.format(
        CFG_DIR,
        INPUT_TASK)
    cfg['config_dir_target'] = '/home/ubuntu/task-taxonomy-331b/experiments/{}/{}'.format(
        CFG_DIR,
        TARGET_TASK)
    ### -------------------------------------------    
    # Automatically populate data loading variables
    input_cfg = general_utils.load_config(cfg['config_dir_input'], nopause=True)
    cfg['input_cfg'] = input_cfg

    # Replace loading info with the version from the target config
    target_cfg = general_utils.load_config(cfg['config_dir_target'], nopause=True)
    cfg['target_cfg'] = target_cfg
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

    # For segmentation
    general_utils.update_keys(cfg, "num_pixels", target_cfg)
    general_utils.update_keys(cfg, "only_target_discriminative", target_cfg)
    general_utils.update_keys(cfg, "num_pixels", target_cfg)
    

    # kludge where we list all of the files
    # cfg['list_of_fileinfos'] = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/train_image_split_0.npy') )    
    cfg['train_list_of_fileinfos'] = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/val_image_split_{}.npy'.format(VAL_DATA_USED)) )    
    cfg['val_list_of_fileinfos'] = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/test_image_split_0.npy'))     
    
    # Define where the extracted representations are stored
    cfg['train_representations_file'] = os.path.join(
        input_cfg['log_root'], representation_task,
        '{task}_{train_split}_representations.pkl'.format(
            task=representation_task,
            train_split='train' if transitive else 'val'))
    cfg['val_representations_file'] = os.path.join(
        input_cfg['log_root'], representation_task,
        '{task}_{train_split}_representations.pkl'.format(
            task=representation_task,
            train_split='val' if transitive else 'test'))
    
    # Now use 'val' for training and 'test' for validation... :(
    tmp = target_cfg['train_filenames']
    target_cfg['train_filenames'] = os.path.abspath( os.path.join( root_dir, 
        'assets/aws_data/val_split_image_info_{}.pkl'.format(VAL_DATA_USED)) )  
    # Multi-input tasks need the 'closure' of the small test set so that they, too, have 
    #  the required number of inputs
    if TARGET_TASK in ['ego_motion', 'fix_pose', 'point_match', 'non_fixated_pose'] \
        and VAL_DATA_USED not in ['16k','30k']:
        target_cfg['train_filenames'] = target_cfg['train_filenames'].replace(
            'val_split_image_info_', 'val_split_image_info_multi_in_'
        )
        cfg['train_list_of_fileinfos'] = cfg['train_list_of_fileinfos'].replace(
            'val_image_split_', 'val_image_split_multi_in_'
        )

    if TARGET_TASK == 'ego_motion':
        target_cfg['train_filenames'] = os.path.abspath( os.path.join( root_dir, 
            'assets/aws_data/val_camera_fixated_trips_info_{}.pkl'.format(VAL_DATA_USED)) )  
    if TARGET_TASK == 'fix_pose':
        target_cfg['train_filenames'] = os.path.abspath( os.path.join( root_dir, 
            'assets/aws_data/val_camera_fixated_pairs_info_{}.pkl'.format(VAL_DATA_USED)) )  
    if TARGET_TASK == 'point_match':
        target_cfg['train_filenames'] = os.path.abspath( os.path.join( root_dir, 
            'assets/aws_data/val_point_match_info_{}.pkl'.format(VAL_DATA_USED)) )  
    if TARGET_TASK == 'non_fixated_pose':
        target_cfg['train_filenames'] = os.path.abspath( os.path.join( root_dir, 
            'assets/aws_data/val_camera_nonfixated_pairs_info_{}.pkl'.format(VAL_DATA_USED)) ) 

    target_cfg['val_filenames'] = str(target_cfg['test_filenames'])
    target_cfg['test_filenames'] = None # str(tmp)

    general_utils.update_keys(cfg, "train_filenames", target_cfg)
    general_utils.update_keys(cfg, "val_filenames", target_cfg)
    general_utils.update_keys(cfg, "test_filenames", target_cfg)
    general_utils.update_keys(cfg, "dataset_dir", target_cfg)

    # Params for training
    cfg['root_dir'] = root_dir
    # cfg['num_epochs'] = 30 if INPUT_TASK == 'random' else 4
    if VAL_DATA_USED == '1k':
        cfg['num_epochs'] = 200
        if TARGET_TASK in ['ego_motion', 'fix_pose', 'point_match', 'non_fixated_pose']:
            cfg['num_epochs'] = 200
    elif VAL_DATA_USED == '2k':
        cfg['num_epochs'] = 100
    elif VAL_DATA_USED == '4k':
        cfg['num_epochs'] = 50
    elif VAL_DATA_USED == '8k':
        cfg['num_epochs'] = 50 
    elif VAL_DATA_USED == '5h':
        cfg['num_epochs'] = 400 
    elif VAL_DATA_USED == '1h':
        cfg['num_epochs'] = 1000
    elif VAL_DATA_USED == '16k':
        cfg['num_epochs'] = 30
    else:
        cfg['num_epochs'] = 15     

    #cfg['num_epochs'] = 15 if not cfg['finetune_decoder'] else 7 if not cfg['finetune_decoder'] else 7
    cfg['max_ckpts_to_keep'] = cfg['num_epochs']
    cfg['target_model_type'] = target_cfg['model_type']

    cfg['weight_decay'] = 1e-5  # 1e-7, 1
    cfg['model_type'] = architectures.TransferNet

    ## DATA LOADING
    # representations
    cfg['representation_dim'] = [16, 16, 8]
    cfg['representation_dtype'] = tf.float32
    def create_bn_decay_schedule():
        # We stick it in a callable function so that the variables will be created on the main graph
        with tf.variable_scope("batchnorm_decay", reuse=False) as sc:
            global_step = tf.Variable(0, name='counter')
            inc = tf.assign_add(global_step, 1, name='bn_increment')
            return tf.train.piecewise_constant(
                global_step, 
                boundaries=[np.int32(0), np.int32(1500), np.int32(6682), np.int32(6700)],
                values=[0.2, 0.5, 0.9, 0.999]
            )

    def create_clipping_schedule():
        # We stick it in a callable function so that the variables will be created on the main graph
        with tf.variable_scope("batchrenorm_decay", reuse=False) as sc:
            global_step = tf.Variable(0, name='counter')
            inc = tf.assign_add(global_step, 1, name='bn_increment')
            rmax = tf.train.piecewise_constant(
                global_step, 
                boundaries=[
                    np.int32(0),
                    np.int32(1500),
                    np.int32(2000),
                    np.int32(2500),
                    np.int32(3000),
                    np.int32(3500),
                    np.int32(4000)],
                values=[
                    1.,
                    1.5,
                    2.,
                    2.5,
                    3.]
            )
            rmin = 1./rmax
            dmax = tf.train.piecewise_constant(
                global_step, 
                boundaries=[
                    np.int32(0),
                    np.int32(1500),
                    np.int32(2000),
                    np.int32(2500),
                    np.int32(3000),
                    np.int32(3500),
                    np.int32(4000),
                    np.int32(4500),
                    np.int32(5000),
                    ],
                values=[
                    0.,
                    0.5, 
                    1.0, 
                    1.5, 
                    2.0, 
                    2.5, 
                    3.0,
                    3.5,
                    5.0]
            )
            return {'rmax': rmax, 'rmin': rmin, 'dmax': dmax}


    ## SETUP MODEL
    # Transfer
    cfg['encoder'] = transfer_models.transfer_two_stream_with_bn_ends_l2l0_no_image_deep
    if INPUT_TASK == 'pixels':
        raise NotImplementedError("Cannot transfer from 'pixels' when using no_image!")
    # if INPUT_TASK == 'random':
        # cfg['encoder'] = transfer_multilayer_conv_with_bn_ends
    cfg['hidden_size'] = int("<HIDDEN_SIZE>") # This will be the number of interior channels
    cfg['encoder_kwargs'] = {
        'side_encoder_func' : transfer_models.side_encoder_l2,
        'output_channels': cfg['representation_dim'][-1],
        'kernel_size': [KERNEL_SIZE, KERNEL_SIZE],
        'stride': 1,
        'batch_norm_epsilon': 1e-5,
        'batch_norm_decay': 0.8, #0.95
        'weight_decay': cfg['weight_decay'],
        'flatten_output': 'flatten' in target_cfg['encoder_kwargs']
        # 'flatten_output': (TARGET_TASK in FLATTEN_TASKS)
    }
    cfg['encoder_kwargs']['renorm'] = True
    cfg['encoder_kwargs']['renorm_momentum'] = 0.9
    cfg['encoder_kwargs']['renorm_clipping'] = create_clipping_schedule

    # Type of decoder to use
    cfg['replace_target_decoder'] = transfer_models.__dict__[REPLACE_TARGET_DECODER_FUNC_NAME]

    # learning
    general_utils.update_keys(cfg, "initial_learning_rate", target_cfg)
    general_utils.update_keys(cfg, "optimizer", target_cfg)
    general_utils.update_keys(cfg, "clip_norm", target_cfg)
    def pwc(initial_lr, **kwargs):
        global_step = kwargs['global_step']
        del kwargs['global_step']
        return tf.train.piecewise_constant(global_step, **kwargs)
    cfg['learning_rate_schedule'] = pwc
    cfg['learning_rate_schedule_kwargs' ] = {
        'boundaries': [np.int64(0), np.int64(5000)], # need to be int64 since global step is...
        'values': [cfg['initial_learning_rate'], cfg['initial_learning_rate']/10]
    }

    # cfg['initial_learning_rate'] = 1e-4  # 1e-6, 1e-1
    # cfg[ 'optimizer' ] = tf.train.AdamOptimizer
    # cfg[ 'optimizer_kwargs' ] = {}

    ## LOCATIONS
    # logging
    config_dir = os.path.dirname(os.path.realpath( __file__ ))
    task_name = os.path.basename( config_dir )
    cfg['log_root'] = config_dir.replace('task-taxonomy-331b/', 's3/')
    if cfg['finetune_decoder'] and cfg['retrain_decoder']:
        cfg['log_root'] = os.path.join(cfg['log_root'], 'rt_ft')
    elif cfg['retrain_decoder']:
        cfg['log_root'] = os.path.join(cfg['log_root'], 'rt_no_ft')
    elif cfg['finetune_decoder']:
        cfg['log_root'] = os.path.join(cfg['log_root'], 'ft')
    elif cfg['unlock_decoder']:
        cfg['log_root'] = os.path.join(cfg['log_root'], 'scratch')

    cfg['log_dir'] = os.path.join(cfg['log_root'], 'logs')
    
    # Set the model path that we will restore from
    # Where the target decoder is stored
    cfg['model_path'] = '{}/{}/model.permanent-ckpt'.format(target_cfg['log_root'], TARGET_TASK)
    if cfg['retrain_decoder']:
        cfg['model_path'] = tf.train.latest_checkpoint(
            os.path.join(
                cfg['log_root'],
                '..',
                'logs',
                'slim-train'
            )
        )

    # input pipeline
    data_dir = '/home/ubuntu/s3'
    cfg['meta_file_dir'] = 'assets/aws_data'
    cfg['create_input_placeholders_and_ops_fn'] = create_input_placeholders_and_ops_transfer
    cfg['randomize'] = True 
    cfg['num_read_threads'] = 300
    cfg['batch_size'] = 32
    cfg['inputs_queue_capacity'] = 4096  
    

    # Checkpoints and summaries
    cfg['summary_save_every_secs'] = 600
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
