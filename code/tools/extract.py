'''
  Name: extract.py
  Desc: Extract representations.
  Usage:
    python extract.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np

import init_paths
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
from   models.sample_models import *
import utils

parser = argparse.ArgumentParser(description='Extract representations of encoder decoder model.')
parser.add_argument( 'cfg_dir', help='directory containing config.py file, should include a checkpoint directory' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)

def main( _ ):
    args = parser.parse_args()

    # Get available GPUs
    local_device_protos = utils.get_available_devices()
    print( 'Found devices:', [ x.name for x in local_device_protos ] )  
    # set GPU id
    if args.gpu_id:
        print( 'using gpu %d' % args.gpu_id )
        os.environ[ 'CUDA_VISIBLE_DEVICES' ] = str( args.gpu_id )
    else:
        print( 'no gpu specified' )

    cfg = load_config( args.cfg_dir )
    run_training( cfg )


def run_training( cfg ):
    # set up logging
    tf.logging.set_verbosity( tf.logging.INFO )

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = utils.setup_input( cfg, is_training=False, use_filename_queue=True )
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()

        # build model (and losses and train_op)
        model = setup_model( inputs, cfg, is_training=False )

        # set up metrics to evaluate
        names_to_values, names_to_updates = setup_metrics( inputs, model, cfg )

        # execute training 
        start_time = time.time()
        utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

        # start session and restore model
        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        if cfg['model_path'] is None:
            print('Please specify a checkpoint directory')
	    return	
	cfg['randomize'] = False
        model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
        
        utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=is_training )

        data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=True )
        training_runners[ 'threads' ] = data_prefetch_init_fn( training_runners[ 'sess' ], training_runners[ 'coord' ] )

        representations, input_batch, target_batch, data_idx = training_runners['sess'].run( [
                model['model'].encoder_output,
                inputs['input_batch'],  inputs['target_batch'], inputs[ 'data_idxs' ],  inputs[ 'mask_batch' ] ] )        

        print('Got first batch representation with size:%s' % (representations.shape))
        for step in xrange( inputs[ 'max_steps' ] - 1):
            encoder_output, input_batch, target_batch, data_idx = training_runners['sess'].run( [
                    model['model'].encoder_output,
                    inputs['input_batch'],  inputs['target_batch'], inputs[ 'data_idxs' ],  inputs[ 'mask_batch' ] ] )
            representations = np.append(representations, encoder_output, axis=0)

            if training_runners['coord'].should_stop():
                break

        print('The size of representations is %s while we expect it to run for %d steps with batchsize %d' % (representations.shape, inputs['max_steps'], cfg['batch_size']))

        utils.request_data_loading_end( training_runners )
        utils.end_data_loading_and_sess( training_runners )

        end_train_time = time.time() - start_time
        print('time to train %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
        print('avg time per epoch: %.3f hrs' % ( (end_train_time/(60*60)) / cfg['num_epochs']) )


def setup_metrics( inputs, model, cfg ):
    # predictions = model[ 'model' ].
    # Choose the metrics to compute:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map( {} )
    return  {}, {}

if __name__=='__main__':
    main( '' )
