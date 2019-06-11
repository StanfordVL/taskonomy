'''
  Name: val_test.py
  Desc: Executes validation or test set of a network. The actual code executed is specified from a 
    configuration file located in some directory and the configuration file must be
    named config.py. 
            
  Usage:
  ------
   For Validation Set:
    python val_test.py /path/to/cfgdir/ --is_test=0

   For Test Set:
    python val_test.py /path/to/cfgdir/ --is_test=1
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import pdb


import init_paths
from   general_utils import RuntimeDeterminedEnviromentVars
from   models.sample_models import *
import utils
import threading

parser = argparse.ArgumentParser(description='Train encoder decoder model.')
parser.add_argument( 'cfg_dir', help='directory containing config.py file' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--is_test', dest='is_test',
                    help='0 for validation set, 1 for testing set',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)


def main( _ ):
    args = parser.parse_args()

    print(args)
    # Get available GPUs
    local_device_protos = utils.get_available_devices()
    print( 'Found devices:', [ x.name for x in local_device_protos ] )  
    # set gpu
    if args.gpu_id is not None:
        print( 'using gpu %d' % args.gpu_id )
        os.environ[ 'CUDA_VISIBLE_DEVICES' ] = str( args.gpu_id )
    else:
        print( 'no gpu specified' )
    # load config and run training
    cfg = utils.load_config( args.cfg_dir, nopause=args.nopause )
    cfg['task_name'] = args.cfg_dir.split('/')[-1]
    cfg['task_name'] = 'class_selected'
    cfg['num_epochs'] = 1
    run_val_test( cfg )

def run_val_test( cfg ):
    # set up logging
    tf.logging.set_verbosity( tf.logging.INFO )

    tf.reset_default_graph()
    training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }
    # create ops and placeholders
    inputs = utils.setup_input( cfg, is_training=False )
    RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
    RuntimeDeterminedEnviromentVars.populate_registered_variables()

    # build model (and losses and train_op)
    model = utils.setup_model( inputs, cfg, is_training=False )
#        full_path = tf.train.latest_checkpoint(checkpoint_dir)
#        step = full_path.split('-')[-1]

#    model_path = os.path.join('/home/ubuntu/s3/model_log', cfg['task_name'], 'model.permanent-ckpt')

    model_path = os.path.join('/home/ubuntu/s3/model_log_final', cfg['task_name'], 'model.permanent-ckpt')
    model[ 'saver_op' ].restore( training_runners[ 'sess' ], model_path )
    m = model[ 'model' ]
    # execute training 
    start_time = time.time()
    utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

    data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=False )

    prefetch_threads = threading.Thread(
        target=data_prefetch_init_fn,
        args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
    prefetch_threads.start()

    print("Dataloading workers dispatched....")

    return_accuracy= 'return_accuracy' in cfg and cfg['return_accuracy'],

    losses_mean = AverageMeter()
    accuracy_mean = AverageMeter()
    for step in range( inputs[ 'max_steps' ] ):
        #print(step)
        if return_accuracy:
            ( 
                data_idx, loss, accuracy
            ) = training_runners['sess'].run( [ 
                model[ 'data_idxs' ], 
                m.losses[0], m.accuracy] )
            losses_mean.update(loss)
            accuracy_mean.update(accuracy)
            if step % 100 == 0:
                print('Step: {step} with Current Losses mean: {loss}; with accuracy: {accur}'.format(
                    step=step, loss=losses_mean.avg,accur=accuracy_mean.avg))
        else:
            ( 
                data_idx, loss
            ) = training_runners['sess'].run( [ 
                model[ 'data_idxs' ], 
                m.losses[0]] )
            losses_mean.update(loss)
            if step % 100 == 0:
                print('Step: {step} with Current Losses mean: {loss}'.format(
                    step=step, loss=losses_mean.avg))
    if return_accuracy:
        print('Final Losses mean: {loss}; with accuracy: {accur}'.format(
                    loss=losses_mean.avg,accur=accuracy_mean.avg))
    else:
        print('Final Losses mean: {loss}'.format(loss=losses_mean.avg))


    end_train_time = time.time() - start_time
    print('time to train %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
    print('avg time per epoch: %.3f hrs' % ( (end_train_time/(60*60)) / cfg['num_epochs']) )

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=='__main__':
  main( '' )
  # tf.app.run()
