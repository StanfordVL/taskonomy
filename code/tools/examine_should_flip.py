'''
  Name: extract.py
  Desc: Extract losses.
  Usage:
    python encode_inputs.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import pdb
import pickle
from   runstats import Statistics
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import time

import init_paths
from   data.load_ops import resize_rescale_image
from   data.task_data_loading import load_and_specify_preprocessors_for_representation_extraction
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
from   models.sample_models import *
import utils

parser = argparse.ArgumentParser(description='Extract losses of encoder decoder model.')
parser.add_argument( '--cfg_dir', dest='cfg_dir', help='directory containing config.py file, should include a checkpoint directory' )
parser.set_defaults(cfg_dir="/home/ubuntu/task-taxonomy-331b/experiments/second_order/DO_NOT_REPLACE_TARGET_DECODER/16k")

parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)


parser.add_argument('--task', dest='task')
parser.add_argument('--data-split', dest='data_split')
parser.set_defaults(data_split="val")

parser.add_argument('--out-dir', dest='out_dir')
parser.set_defaults(out_dir="")

parser.add_argument('--out-name', dest='out_name')
parser.set_defaults(out_name="")

parser.add_argument('--representation-task', dest='representation_task')
parser.set_defaults(representation_task="")

parser.add_argument('--print-every', dest='print_every')
parser.set_defaults(print_every="100")


parser.add_argument('--train-mode', dest='train_mode', action='store_true')
# parser.set_defaults(print_every="100")

# TRAIN_MODE =True 
def main( _ ):
    args = parser.parse_args()
    global TRAIN_MODE
    TRAIN_MODE = args.train_mode
    #task_list = ["autoencoder", "colorization","curvature", "denoise", "edge2d", "edge3d", "ego_motion", "fix_pose", "impainting", "jigsaw", "keypoint2d", "keypoint3d", "non_fixated_pose", "point_match", "reshade", "rgb2depth", "rgb2mist", "rgb2sfnorm", "room_layout", "segment25d", "segment2d", "vanishing_point"]
    #single channel for colorization !!!!!!!!!!!!!!!!!!!!!!!!! COME BACK TO THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!
    #task_list = [ args.task ]
    with open('/home/ubuntu/task-taxonomy-331b/tools/config_list.txt', 'r') as fp:
        task_list = [x.split(' ')[0] for x in fp.readlines()]
        #config_prefix = 'second_order/DO_NOT_REPLACE_TARGET_DECODER/16k'
        #task_list = ['{}/{}'.format(config_prefix, x) for x in task_list]

    # Get available GPUs
    local_device_protos = utils.get_available_devices()
    print( 'Found devices:', [ x.name for x in local_device_protos ] )  
    # set GPU id
    if args.gpu_id:
        print( 'using gpu %d' % args.gpu_id )
        os.environ[ 'CUDA_VISIBLE_DEVICES' ] = str( args.gpu_id )
    else:
        print( 'no gpu specified' )
    
    should_flip_dict = {}
    for task in task_list:
        flip_loss = []
        for flip_num in range(2):
            task_dir = os.path.join(args.cfg_dir, task)
            cfg = utils.load_config( task_dir, nopause=args.nopause )
            root_dir = cfg['root_dir']
            if flip_num == 0:
                cfg['val_representations_file'] = cfg['val_representations_file'][::-1]

            if args.data_split == 'train':
                split_file = cfg['train_filenames']
            elif args.data_split == 'val':
                split_file = cfg['val_filenames']
            elif args.data_split == 'test':
                split_file = cfg['test_filenames']
            else: 
                raise NotImplementedError("Unknown data split section: {}".format(args.data_split))
            cfg['train_filenames'] = split_file
            cfg['val_filenames'] = split_file
            cfg['test_filenames'] = split_file 

            if 'train_list_of_fileinfos' in cfg:
                split_file_ =  cfg['{}_representations_file'.format(args.data_split)]
                cfg['train_representations_file'] = split_file_
                cfg['val_representations_file'] = split_file_
                cfg['test_representations_file'] = split_file_

                split_file_ =  cfg['{}_list_of_fileinfos'.format(args.data_split)]
                cfg['train_list_of_fileinfos'] = split_file_
                cfg['val_list_of_fileinfos'] = split_file_
                cfg['test_list_of_fileinfos'] = split_file_

            if args.representation_task:
                split_file_ = args.representation_task
                if 'multiple_input_tasks' in cfg:
                    split_file_ = [split_file_]
                cfg['train_representations_file'] = split_file_
                cfg['val_representations_file'] = split_file_
                cfg['test_representations_file'] = split_file_
            
            # Try latest checkpoint by epoch
            cfg['model_path'] = tf.train.latest_checkpoint(
                    os.path.join(
                        cfg['log_root'],
                        'logs',
                        'slim-train'
                    ))
            # Try latest checkpoint by time
            if cfg['model_path'] is None:
                cfg['model_path'] = tf.train.latest_checkpoint(
                    os.path.join(
                        cfg['log_root'],
                        'logs',
                        'slim-train',
                        'time'
                    ))      

            # Try to get one saved manually
            if cfg['model_path'] is None:  
                continue
                #cfg['model_path'] = os.path.join(cfg['log_root'], task, "model.permanent-ckpt") 

            cfg['randomize'] = False
            cfg['num_epochs'] = 3
            cfg['batch_size'] = 32 
            cfg['num_read_threads'] = 3
            if 'batch_size' in cfg['encoder_kwargs']:
                cfg['encoder_kwargs']['batch_size'] = cfg['batch_size']
            try:
                cfg['target_cfg']['batch_size'] = cfg['batch_size']
            except:
                pass
            try:
                cfg['target_cfg']['encoder_kwargs']['batch_size'] = cfg['batch_size']
            except:
                pass

            loss_dir = args.cfg_dir
            curr_loss = run_extract_losses_5_steps( args, cfg, loss_dir, task )
            flip_loss.append(curr_loss)
        should_flip_dict[task] = (flip_loss[0] > flip_loss[1])
        with open('/home/ubuntu/task-taxonomy-331b/tools/second_order_should_flip.pkl', 'wb') as fp:
            pickle.dump(should_flip_dict, fp)


def get_extractable_losses(cfg, model):
    ''' returns: loss_names, loss_ops '''
    model = model['model']
    transfer = (cfg['model_type'] == architectures.TransferNet)
    losses = [('total_loss', model.total_loss)]
    target_model_type = cfg['model_type']
    if transfer:
        target_model_type = cfg['target_model_type']
        model = model.decoder

    if target_model_type == architectures.EncoderDecoderWithCGAN:
        losses.append(('l1_loss', model.l1_loss))
        losses.append(('gan_loss', model.loss_g))
    elif target_model_type == architectures.EDSoftmaxRegenCGAN:
        losses.append(('xentropy', model.softmax_loss))
        losses.append(('gan_loss', model.loss_g))
    elif target_model_type == architectures.Siamese:
        if 'l2_loss' in cfg and cfg['l2_loss']:
            losses.append(('l2_loss', model.task_loss))
        else:
            losses.append(('xentropy', model.task_loss))
    elif target_model_type == architectures.BasicFF:
        if 'l2_loss' in cfg and cfg['l2_loss']:
            losses.append(('l2_loss', model.siamese_loss))
        elif 'l2_loss' in cfg['target_cfg'] and cfg['target_cfg']['l2_loss']:
            losses.append(('l2_loss', model.siamese_loss))
        else:
            losses.append(('xentropy', model.siamese_loss))
    elif target_model_type == architectures.SegmentationEncoderDecoder:
            losses.append(('metric_loss', model.metric_loss))
    elif target_model_type == architectures.SemSegED:    
            losses.append(('metric_loss', model.metric_loss))
    elif target_model_type == architectures.CycleSiamese:    
            losses.append(('cycle_loss', model.cycle_loss_total))
    elif target_model_type == architectures.EDSoftmax:    
            losses.append(('xentropy', model.softmax_loss))
    else:
        raise NotImplementedError("Extracting losses not implemented for {}".format(
            target_model_type
        ))
    return zip(*losses)

def run_extract_losses_5_steps( args, cfg, save_dir, given_task ):
    transfer = (cfg['model_type'] == architectures.TransferNet)
    if transfer:
        get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn_transfer
        setup_input_fn = utils.setup_input_transfer
    else:
        setup_input_fn = utils.setup_input
        get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn

    # set up logging
    tf.logging.set_verbosity( tf.logging.ERROR )
    stats = Statistics()

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
        #RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        #RuntimeDeterminedEnviromentVars.populate_registered_variables()
        max_steps = get_max_steps(inputs[ 'max_steps' ], args.data_split)

        # build model (and losses and train_op)
        model = utils.setup_model( inputs, cfg, is_training=False )
        loss_names, loss_ops = get_extractable_losses(cfg, model)
        if 'l1_loss' in loss_names:
            display_loss = 'l1_loss'
        elif 'l2_loss' in loss_names:
            display_loss = 'l2_loss'
        elif 'xentropy' in loss_names:
            display_loss = 'xentropy'
        elif 'metric_loss' in loss_names:
            display_loss = 'metric_loss'
        elif 'cycle_loss' in loss_names:
            display_loss = 'cycle_loss'
        else:
            display_loss = 'total_loss'

        # set up metrics to evaluate
        names_to_values, names_to_updates = setup_metrics( inputs, model, cfg )

        # execute training 
        start_time = time.time()
        utils.print_start_info( cfg, max_steps, is_training=False )

        # start session and restore model
        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        if cfg['model_path'] is None:
            print('Please specify a checkpoint directory')
            return	
        print('Attention, model_path is ', cfg['model_path']) 
        model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )

        utils.print_start_info( cfg, max_steps, is_training=False )
        data_prefetch_init_fn = get_data_prefetch_threads_init_fn( inputs, cfg, 
            is_training=False, use_filename_queue=False )
        prefetch_threads = threading.Thread(
            target=data_prefetch_init_fn,
            args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
        prefetch_threads.start()
        
        # run one example so that we can calculate some statistics about the representations
        # results = training_runners['sess'].run( [ *loss_ops ] )       
        # losses = results[0]
        x = 0
        for step in range( 3 ):
            results = training_runners['sess'].run( [ *loss_ops ] )    
            x = x + results[0]
            if training_runners['coord'].should_stop():
                break

    tf.reset_default_graph()
    utils.request_data_loading_end( training_runners )
    utils.end_data_loading_and_sess( training_runners )
    return x / 2.

def setup_metrics( inputs, model, cfg ):
    # predictions = model[ 'model' ].
    # Choose the metrics to compute:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map( {} )
    return  {}, {}

def get_xentropy_loss(predict, label, mask):
    epsilon = np.finfo(float).eps
    # idxs = label.argmax(axis=-1).flatten()
    idxs = label.flatten().astype(np.int64)
    predict = predict.reshape(len(idxs), predict.shape[-1])
    predict[predict < epsilon] = epsilon
    predict[predict > 1 - epsilon] = 1 - epsilon
    losses = -np.log(predict)[range(len(idxs)), idxs]
    return (losses * mask.flatten()).mean()



def get_max_steps(original_max_steps, data_split):
    n_images = None
    if data_split == 'train':
        n_images = 129380
    elif data_split == 'val':
        n_images = 29933
    elif data_split == 'test':
        n_images = 17853
    else: 
        raise NotImplementedError('Unknown data split {}'.format(data_split))
    if original_max_steps != n_images:
        print("Adjusting number of steps from {} -> {}".format(
            max(original_max_steps, n_images),
            min(original_max_steps, n_images)
        ))
    return min(original_max_steps, n_images)


if __name__=='__main__':
    main( '' )

