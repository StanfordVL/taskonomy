'''
  Name: test.py
  Desc: Executes testing of a network.

  Usage:
    python test.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

import init_paths
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
from   models.sample_models import *
import utils

parser = argparse.ArgumentParser(description='Train encoder decoder model.')
parser.add_argument( 'cfg_dir', help='directory containing config.py file' )
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

        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=True )
        training_runners[ 'threads' ] = data_prefetch_init_fn( training_runners[ 'sess' ], training_runners[ 'coord' ] )
        try:
            # This just returns the imput as output. It is for testing data
            #  input only. 
            for step in xrange( inputs[ 'max_steps' ] ):
                input_batch, target_batch, data_idx = training_runners['sess'].run( [ 
                        model['input_batch'],  model['target_batch'], model[ 'data_idxs' ] ] )

                if training_runners['coord'].should_stop():
                    break
        finally:
            utils.request_data_loading_end( training_runners )
            utils.end_data_loading_and_sess( training_runners )
        # else: # Use tf.slim
        #     train_log_dir = os.path.join( cfg['log_dir'], 'slim-train' )

        #     # When ready to use a model, use the code below
        #     train(  model[ 'train_op' ],
        #             train_log_dir,
        #             get_data_prefetch_threads_init_fn( inputs, cfg ), 
        #             global_step=model[ 'global_step' ],
        #             number_of_steps=inputs[ 'max_steps' ],
        #             init_fn=model[ 'init_fn' ],
        #             save_summaries_secs=300,
        #             save_interval_secs=600,
        #             saver=model[ 'saver_op' ] ) 

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
