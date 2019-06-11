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
import utils

parser = argparse.ArgumentParser(description='Extract representations of encoder decoder model.')
parser.add_argument( 'cfg_dir', help='directory containing config.py file, should include a checkpoint directory' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=False)

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

    cfg = utils.load_config( args.cfg_dir, nopause=args.nopause )
    run_extract_representations( args, cfg )


def run_extract_representations( args, cfg ):
    # set up logging
    tf.logging.set_verbosity( tf.logging.INFO )

    with tf.Graph().as_default() as g:
        cfg['randomize'] = False
        cfg['num_epochs'] = 1
        # cfg['num_read_threads'] = 5
        # cfg['batch_size']=2
        #if cfg['model_path'] is None:
        #    cfg['model_path'] = tf.train.latest_checkpoint( os.path.join( args.cfg_dir, "logs/slim-train/" ) )
        cfg['model_path'] = os.path.join( args.cfg_dir, "logs/slim-train/model.ckpt-59690")
        # create ops and placeholders
        tf.logging.set_verbosity( tf.logging.INFO )
        inputs = utils.setup_input( cfg, is_training=False, use_filename_queue=True )
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
            if cfg['model_path'] is None:
                print('Please specify a checkpoint directory')
                return	
            
            model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
            
            utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

            data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=True )
            training_runners[ 'threads' ] = data_prefetch_init_fn( training_runners[ 'sess' ], training_runners[ 'coord' ] )
            
            # run one example so that we can calculate some statistics about the representations
            filenames = []
            representations, data_idx = training_runners['sess'].run( [ 
                    model['model'].encoder_output, inputs[ 'data_idxs' ] ] )        
            filenames += [ inputs[ 'filepaths_list'][ i ] for i in data_idx ]
            print( 'Got first batch representation with size: {0}'.format( representations.shape ) )

            # run the remaining examples
            for step in xrange( inputs[ 'max_steps' ] - 1 ):
                if step % 100 == 0: 
                    print( 'Step {0} of {1}'.format( step, inputs[ 'max_steps' ] - 1 ))
                encoder_output, data_idx = training_runners['sess'].run( [
                        model['model'].encoder_output, inputs[ 'data_idxs' ] ] )        
                representations = np.append(representations, encoder_output, axis=0)
                filenames += [ inputs[ 'filepaths_list'][ i ] for i in data_idx ]

                if training_runners['coord'].should_stop():
                    break

            print('The size of representations is %s while we expect it to run for %d steps with batchsize %d' % (representations.shape, inputs['max_steps'], cfg['batch_size']))

            end_train_time = time.time() - start_time
            save_path = os.path.join( args.cfg_dir, '../representations.pkl' )
            with open( save_path, 'wb' ) as f:
                pickle.dump( { 'filenames': filenames, 'representations': representations }, f )
            print( 'saved representations to {0}'.format( save_path ))
            print('time to train %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
            print('avg time per epoch: %.3f hrs' % ( (end_train_time/(60*60)) / cfg['num_epochs']) )
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
