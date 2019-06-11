'''
  Name: extract.py
  Desc: Extract representations.
  Usage:
    python encode_inputs.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import pickle

import init_paths
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
from   models.sample_models import *
from   data.load_ops import resize_rescale_image
import utils
import pdb
import random
import threading

parser = argparse.ArgumentParser(description='Extract representations of encoder decoder model.')
parser.add_argument( '--cfg_dir', dest='cfg_dir', help='directory containing config.py file, should include a checkpoint directory' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)
parser.set_defaults(cfg_dir="/home/ubuntu/task-taxonomy-331b/experiments/extract_rep")

def main( _ ):
    args = parser.parse_args()
    #task_list = ["autoencoder", "colorization","curvature", "denoise", "edge2d", "edge3d", "ego_motion", "fix_pose", "impainting", "jigsaw", "keypoint2d", "keypoint3d", "non_fixated_pose", "point_match", "reshade", "rgb2depth", "rgb2mist", "rgb2sfnorm", "room_layout", "segment25d", "segment2d", "vanishing_point"]
    #single channel for colorization !!!!!!!!!!!!!!!!!!!!!!!!! COME BACK TO THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!
    #task_list = [ "point_match"]
    task_list = [ "vanishing_point"]

    # Get available GPUs
    local_device_protos = utils.get_available_devices()
    print( 'Found devices:', [ x.name for x in local_device_protos ] )  
    # set GPU id
    if args.gpu_id:
        print( 'using gpu %d' % args.gpu_id )
        os.environ[ 'CUDA_VISIBLE_DEVICES' ] = str( args.gpu_id )
    else:
        print( 'no gpu specified' )
    
    for task in task_list:
        task_dir = os.path.join(args.cfg_dir, task)
        cfg = utils.load_config( task_dir, nopause=args.nopause )
        root_dir = cfg['root_dir']
        cfg['randomize'] = False
        cfg['num_epochs'] = 1
        run_rand_baseline( args, cfg, task )


def run_rand_baseline( args, cfg, given_task ):
    # set up logging
    tf.logging.set_verbosity( tf.logging.INFO )

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        tf.logging.set_verbosity( tf.logging.INFO )
        inputs = utils.setup_input( cfg, is_training=False, use_filename_queue=False )
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        
        # build model (and losses and train_op)
        model = utils.setup_model( inputs, cfg, is_training=False )

        # set up metrics to evaluate
        names_to_values, names_to_updates = setup_metrics( inputs, model, cfg )

        # execute training 
        start_time = time.time()
        utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

        # start session and restore model
        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        try:
            
            utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

            data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=False )
            #training_runners[ 'threads' ] = data_prefetch_init_fn( training_runners[ 'sess' ], training_runners[ 'coord' ] )
            prefetch_threads = threading.Thread(
                target=data_prefetch_init_fn,
                args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
            prefetch_threads.start()
            
            # run one example so that we can calculate some statistics about the representations
            targets = training_runners['sess'].run( inputs[ 'target_batch' ] )         
       
            # run the remaining examples
            for step in range( inputs[ 'max_steps' ] - 1 ):
            #for step in range( 10 ):
                if step % 100 == 0: 
                    print( 'Step {0} of {1}'.format( step, inputs[ 'max_steps' ] - 1 ))
               
                target = training_runners['sess'].run( inputs[ 'target_batch' ] )  
                targets = np.append( targets, target, axis=0)

                if training_runners['coord'].should_stop():
                    break

            rand_idx = [random.randint(0, targets.shape[0] - 1) for i in range(targets.shape[0])] 
            rand_target = [targets[i] for i in rand_idx]
            rand_target = np.vstack(rand_target)

            counter = 0
            sum = 0
            for step in range( inputs[ 'max_steps' ] - 1 ):
            #for step in range( 10 ):
                if step % 100 == 0: 
                    print( 'Step {0} of {1}'.format( step, inputs[ 'max_steps' ] - 1 ))
               
                tar = targets[step*cfg['batch_size']:(step+1)*cfg['batch_size']]
                rand = rand_target[step*cfg['batch_size']:(step+1)*cfg['batch_size']]

                losses = training_runners['sess'].run( model['model'].losses, feed_dict={
                    inputs['target_batch']: tar, model['model'].final_output:rand})
                sum += losses[0]
                counter += 1
                
                if training_runners['coord'].should_stop():
                    break

            print(sum)
            print(counter)
            print('random_baseline has loss: {loss}'.format(loss=sum/counter))
            end_train_time = time.time() - start_time
            
        finally:
            utils.request_data_loading_end( training_runners )
            utils.end_data_loading_and_sess( training_runners )

def setup_metrics( inputs, model, cfg ):
    # predictions = model[ 'model' ].
    # Choose the metrics to compute:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map( {} )
    return  {}, {}

if __name__=='__main__':
    main( '' )
