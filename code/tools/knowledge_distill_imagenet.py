'''
  Name: extract.py
  Desc: Extract representations.
  Usage:
    python encode_inputs.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import pickle
import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import threading

import init_paths
from   data.load_ops import resize_rescale_imagenet
from   data.task_data_loading import load_and_specify_preprocessors_for_representation_extraction
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import data.load_ops as load_ops
import models.architectures as architectures
from   models.sample_models import *
import utils
import skimage.io

parser = argparse.ArgumentParser(description='Extract representations of encoder decoder model.')
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)

parser.add_argument('--task', dest='task')
parser.add_argument('--data-split', dest='data_split')
parser.set_defaults(data_split="val")

parser.add_argument('--idx', dest='idx', type=int)
parser.set_defaults(idx=0)

parser.add_argument('--out-dir', dest='out_dir')
parser.set_defaults(out_dir="")

# VERY BAD KLUDGE. SEE 

def main( _ ):
    args = parser.parse_args()
    # Get available GPUs
    local_device_protos = utils.get_available_devices()
    print( 'Found devices:', [ x.name for x in local_device_protos ] )  
    # set GPU id
   
    task_dir = os.path.join('/home/ubuntu/task-taxonomy-331b/experiments', 'class_gt')
    cfg = utils.load_config( task_dir, nopause=args.nopause )
    
    dataset_dir = '/home/ubuntu/s3/meta'
    train = np.load(os.path.join(dataset_dir, 'train_image_split_0.npy'))
    val = np.load(os.path.join(dataset_dir, 'val_image_split_0.npy'))
    total = np.concatenate((train, val))
    import math
    num_split = 200.
    unit_size = math.ceil(len(total) / num_split)
    total = total[args.idx * unit_size: (args.idx + 1) * unit_size]
    # split_file = '/home/ubuntu/this_split.npy'
    # with open(split_file, 'wb') as fp:
        # np.save(fp, total)
    # cfg['preprocess_fn'] = load_and_specify_preprocessors_for_representation_extraction

    # cfg['train_filenames'] = split_file
    # cfg['val_filenames'] = split_file
    # cfg['test_filenames'] = split_file 
    

    cfg['randomize'] = False
    cfg['num_epochs'] = 2
    cfg['num_read_threads'] = 1
    run_extract_representations( args, cfg, total)


def run_extract_representations( args, cfg, file_to_process):
    setup_input_fn = utils.setup_input
    # set up logging
    tf.logging.set_verbosity( tf.logging.INFO )

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        tf.logging.set_verbosity( tf.logging.INFO )
        inputs = {}
        inputs['input_batch'] = tf.placeholder( tf.float32, shape=[1,224,224,3], name='input_placeholder')
        inputs['target_batch'] = tf.placeholder( tf.float32, shape=[1,1000], name='target_placeholder' )
        inputs['mask_batch'] = tf.placeholder( tf.float32, shape=[1], name='mask_placeholder' )
        inputs['data_idxs'] = tf.placeholder( tf.int32, shape=[1], name='data_idx_placeholder')
        inputs['num_samples_epoch'] = len(file_to_process) 
        inputs['max_steps'] = len(file_to_process) 
        
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        
        # build model (and losses and train_op)
        model = utils.setup_model( inputs, cfg, is_training=False )
        m = model['model']

        # execute training 
        utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

        # start session and restore model
        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        try:
            if cfg['model_path'] is None:
                print('Please specify a checkpoint directory')
                return	
            
            to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  
            for v in tuple(to_restore):     
                if 'global_step' in v.name:
                    to_restore.remove(v)
                            
            saver_for_kd = tf.train.Saver(to_restore)
            saver_for_kd.restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
            #model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )

            for step, filename in enumerate(file_to_process):
                start_time = time.time()
                if step % 100 == 0: 
                    print( 'Step {0} of {1}'.format( step, inputs[ 'max_steps' ] - 1 ))
                m,p,v = filename.decode('UTF-8').split('/')    
                print(filename)
                img_name = '/home/ubuntu/s3/{}/rgb/point_{}_view_{}_domain_rgb.png'.format(m, p, v)
                sfm_dir = 's3://taskonomy-unpacked-oregon/{}/softmax_1000'.format(m)
                os.system('sudo mkdir -p /home/ubuntu/s3/{}/softmax_1000/'.format(m))
                os.system('mkdir -p /home/ubuntu/temp/{}/'.format(m))
                npy_name = 'point_{}_view_{}.npy'.format(p, v)
                if os.path.isfile('/home/ubuntu/s3/{}/softmax_1000/{}'.format(m, npy_name)):
                    continue
                if not os.path.isfile(img_name):
                    continue
                img = skimage.io.imread(img_name, as_grey=False)
                img = resize_rescale_imagenet(img, new_dims=(224,224))
                img = np.reshape(img, (1,224,224,3))
                feed_dict = {inputs['input_batch'] : img}
                predicted = training_runners['sess'].run( model['model'].encoder_output, feed_dict=feed_dict )
                # maxs = np.amax(predicted, axis=-1)
                # softmax = np.exp(predicted - np.expand_dims(maxs, axis=-1))
                # sums = np.sum(softmax, axis=-1)
                # softmax = softmax / np.expand_dims(sums, -1)
                # print(softmax)
                # pdb.set_trace()
                local_npy = os.path.join('/home/ubuntu/temp/{}'.format(m), npy_name) 
                with open(local_npy, 'wb') as fp:
                    np.save(fp, predicted)
                os.system('aws s3 mv {} {}/'.format(local_npy, sfm_dir))
                if training_runners['coord'].should_stop():
                    break
                end_train_time = time.time() - start_time
                print('time to extract  %.3f ' % (end_train_time))

        finally:
            utils.request_data_loading_end( training_runners )
            utils.end_data_loading_and_sess( training_runners )

if __name__=='__main__':
    main( '' )

