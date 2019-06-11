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
from   data.load_ops import class_1000_imagenet
from   data.load_ops import class_selected_imagenet
from   data.load_ops import class_places 
from   data.task_data_loading import load_and_specify_preprocessors_for_representation_extraction
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
from   models.sample_models import *
import utils

parser = argparse.ArgumentParser(description='Extract accuracy for a transfer to class 1000 on ImageNet')
parser.add_argument( '--cfg_dir', dest='cfg_dir', help='directory containing config.py file, should include a checkpoint directory' )
parser.set_defaults(cfg_dir="/home/ubuntu/task-taxonomy-331b/experiments")

parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)

parser.add_argument('--selected', dest='is_selected', action='store_true')
parser.set_defaults(is_selected=False)

parser.add_argument('--transfer', dest='pretrain_transfer', action='store_true')
parser.set_defaults(pretrain_transfer=False)

parser.add_argument('--transfer-type', dest='pretrain_transfer_type')
parser.set_defaults(pretrain_transfer_type='rep_only_taskonomy')

parser.add_argument('--task', dest='task')

parser.add_argument('--data-split', dest='data_split')
parser.set_defaults(data_split="val" )

parser.add_argument('--out-dir', dest='out_dir')
parser.set_defaults(out_dir="")

parser.add_argument('--out-name', dest='out_name')
parser.set_defaults(out_name="")

parser.add_argument('--representation-task', dest='representation_task')
parser.set_defaults(representation_task="")

parser.add_argument('--print-every', dest='print_every')
parser.set_defaults(print_every="10")

parser.add_argument('--hidden', dest='hidden', type=int)
parser.set_defaults(hidden=128)

parser.add_argument('--layers', dest='layers', type=int)
parser.set_defaults(layers=2)

parser.add_argument('--num-epochs', dest='num_epochs', type=int)
parser.set_defaults(num_epochs=100)

parser.add_argument('--dropout', dest='dropout', type=float)
parser.set_defaults(dropout=0.5)

parser.add_argument('--from-scratch', dest='from_scratch', action='store_true')
parser.set_defaults(from_scratch=False)

parser.add_argument('--data-use', dest='data_used', type=int)
parser.set_defaults(data_used=50000)

parser.add_argument('--train-encoder', dest='train_encoder', action='store_true')
parser.set_defaults(train_encoder=False)

parser.add_argument('--metric-only', dest='metric_only', action='store_true')
parser.set_defaults(metric_only=False)

parser.add_argument('--train-mode', dest='train_mode', action='store_true')

parser.add_argument('--places-knowledge', dest='add_places_knowledge', action='store_true')
parser.set_defaults(add_places_knowledge=False)

parser.add_argument('--alex', dest='add_alexnet', action='store_true')
parser.set_defaults(add_alexnet=False)


# parser.set_defaults(print_every="100")

# TRAIN_MODE =True 
def main( _ ):
    args = parser.parse_args()
    global TRAIN_MODE
    TRAIN_MODE = args.train_mode
    #task_list = ["autoencoder", "colorization","curvature", "denoise", "edge2d", "edge3d", "ego_motion", "fix_pose", "impainting", "jigsaw", "keypoint2d", "keypoint3d", "non_fixated_pose", "point_match", "reshade", "rgb2depth", "rgb2mist", "rgb2sfnorm", "room_layout", "segment25d", "segment2d", "vanishing_point"]
    #single channel for colorization !!!!!!!!!!!!!!!!!!!!!!!!! COME BACK TO THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!
    task_list = [ args.task ]

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
        to_task = 'class_places'
        if args.pretrain_transfer:
            task_dir = os.path.join(args.cfg_dir, args.pretrain_transfer_type, 
                    'DO_NOT_REPLACE_TARGET_DECODER/16k',  "{}__{}__8__unlocked".format(task, to_task))
        else:
            task_dir = os.path.join(args.cfg_dir, 'final', task)
        cfg = utils.load_config( task_dir, nopause=args.nopause )
        
        root_dir = cfg['root_dir']
        split_file = os.path.abspath( os.path.join( root_dir, 'assets/aws_data/val_places_info.pkl') )
        cfg['dataset_dir'] = '/home/ubuntu/place'

        cfg['train_filenames'] = split_file
        cfg['val_filenames'] = split_file
        cfg['test_filenames'] = split_file 

        if 'train_list_of_fileinfos' in cfg:
            if type(cfg['train_representations_file']) is not list:
                split_file_ =  os.path.join(
                                cfg['input_cfg']['log_root'], task,
                                '{task}_val_places_representations.pkl'.format( task=task ))
            else:
                split_file_ = []
                for fname in cfg['train_representations_file']:
                    split_file_.append(fname.replace('val', 'val_places'))
                if args.add_places_knowledge:
                    split_file_.append(os.path.join(
                                    cfg['input_cfg'][0]['log_root'], 'class_places',
                                    'class_places_val_places_representations.pkl'))
                    cfg['representation_dim'] = [16, 16, 8*len(split_file_)]
                if args.add_alexnet:
                    split_file_.append(os.path.join(
                                    cfg['input_cfg'][0]['log_root'], 'alex',
                                    'alex_val_places_representations.pkl'))
                    cfg['representation_dim'] = [16, 16, 8*(len(split_file_) + 1 )]
                               

            cfg['train_representations_file'] = split_file_
            cfg['val_representations_file'] = split_file_
            cfg['test_representations_file'] = split_file_

            split_file_ =  os.path.join(root_dir, 'assets/aws_data/val_places.npy')
            cfg['train_list_of_fileinfos'] = split_file_
            cfg['val_list_of_fileinfos'] = split_file_
            cfg['test_list_of_fileinfos'] = split_file_

        # cfg['resize_interpolation_order'] = 0
        # if cfg['model_path'] is None:
            # cfg['model_path'] = os.path.join(cfg['dataset_dir'], "model_log", task, "model.permanent-ckpt") 
        cfg['target_from_filenames'] = class_places 
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
            cfg['model_path'] = os.path.join(cfg['log_root'], task, "model.permanent-ckpt") 
            # cfg['model_path'] = os.path.join(cfg['log_root'], 'logs', 'slim-train', 'time', "model.ckpt-1350") 
        
        finetuned=True
        if args.from_scratch:
            cfg['model_path'] = os.path.join(cfg['log_root'], task, 'places_scratch_{}_{}'.format(args.layers, args.data_used)) 
        else:
            cfg['model_path'] = os.path.join(cfg['log_root'], task, "places")
        if args.train_encoder:
            cfg['finetune_encoder_imagenet'] = True 
            cfg['model_path'] = os.path.join(cfg['log_root'], task, "places_encoder")

        if args.metric_only:
            cfg['metric_net_only'] = True
            cfg['target_cfg']['metric_kwargs'] = {
                'hidden_size': args.hidden,
                'layer_num': args.layers,
                'output_size': 63,
                'initial_dropout': True,
                'dropout':args.dropout
            }  

        cfg['randomize'] = False
        cfg['num_epochs'] = 3
        cfg['batch_size'] = 32 if TRAIN_MODE else 1
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
        
        #cfg['finetune_encoder_imagenet'] = True
        cfg['data_used'] = 100000000
        #cfg['target_dim'] = 31
        #cfg['target_cfg']['target_dim'] = 31
        #cfg['target_cfg']['metric_kwargs']['output_size'] =  cfg['target_dim']

        run_extract_losses( args, cfg, loss_dir, task )


def run_extract_losses( args, cfg, save_dir, given_task ):
    transfer = (cfg['model_type'] == architectures.TransferNet)
    if transfer:
        get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn_transfer
        setup_input_fn = utils.setup_input_transfer
        if given_task == 'pixels':
            get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn_transfer_imagenet
            setup_input_fn = utils.setup_input_transfer_imagenet

    else:
        setup_input_fn = utils.setup_input
        get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn

    # set up logging
    tf.logging.set_verbosity( tf.logging.ERROR )
    stats = Statistics()
    top5_stats = Statistics()
    print_every = int(args.print_every)

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
        #RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        #RuntimeDeterminedEnviromentVars.populate_registered_variables()
        max_steps = get_max_steps(inputs[ 'max_steps' ], args.data_split)

        # build model (and losses and train_op)
        model = utils.setup_model( inputs, cfg, is_training=False )

        # set up metrics to evaluate
        names_to_values, names_to_updates = setup_metrics( inputs, model, cfg )

        # execute training 
        start_time = time.time()
        utils.print_start_info( cfg, max_steps, is_training=False )

        # start session and restore model
        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        try:
            if cfg['model_path'] is None:
                print('Please specify a checkpoint directory')
                return	
            print('Attention, model_path is ', cfg['model_path']) 
            model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )

            # var = [v for v in tf.global_variables() if 'decoder' in v.name][0]
            # print(training_runners[ 'sess' ].run(var))

            utils.print_start_info( cfg, max_steps, is_training=False )
            data_prefetch_init_fn = get_data_prefetch_threads_init_fn( inputs, cfg, 
                is_training=False, use_filename_queue=False )
            prefetch_threads = threading.Thread(
                target=data_prefetch_init_fn,
                args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
            prefetch_threads.start()
            
            # run one example so that we can calculate some statistics about the representations
            filenames = []
            accuracies = []
            if transfer:
                accuracy_op = model['model'].decoder.accuracy
                final_output = model['model'].decoder.final_output 
            else:
                accuracy_op = model['model'].accuracy
                final_output = model['model'].final_output 
            results = training_runners['sess'].run( [ 
                    inputs[ 'data_idxs' ], model['model'].global_step,
                    accuracy_op, final_output ] )
            #pdb.set_trace()
            gs = results[1] 
            data_idx = results[0]
            accuracy = results[2]
            filenames.extend(data_idx)
            accuracies.append(accuracy)
            print("Step number: {}".format(gs))
            # print(loss_names_to_vals, data_idx)
            # return

            # run the remaining examples
            start = time.perf_counter()
            for step in range( max_steps - 1 ):
                results = training_runners['sess'].run( [
                        inputs[ 'data_idxs' ], 
                        final_output,
                        inputs['target_batch'],
                        accuracy_op ] )    
                data_idx = results[0]
                accuracy = results[-1]
                logits = results[1]
                gt = results[2]
                sorted_top5 = np.argsort(logits[0])[::-1][:5]
                sorted_gt = np.argsort(gt[0])[::-1][0]
                top5 = 0.
                if sorted_gt in sorted_top5:
                    top5 = 1.
                filenames.extend(data_idx)
                accuracies.append(accuracy)
                stats.push(accuracy)
                top5_stats.push(top5)
                if step % print_every == 0 and step > 0: 
                    print( 'Step {0} of {1}: ({5}: {2:.3f} || Top 5: {3:.3f} :: ({4:.2f} secs/step)'.format( 
                        step, max_steps - 1,
                        stats.mean(), 
                        top5_stats.mean(),
                        # stats.variance(),
                        (time.perf_counter() - start) / print_every,
                        'accuracy'
                        ))
                    start = time.perf_counter()

                if training_runners['coord'].should_stop():
                    break

            os.system("sudo touch /home/ubuntu/s3/places_accuracy/{}/{}_{}_{}_{}_{}.txt".format(
                args.data_used, given_task, int(stats.mean() * 1000) / 10., int(top5_stats.mean() * 1000) / 10., args.layers, args.dropout))
            print('The size of losses is %s while we expect it to run for %d steps with batchsize %d' % (len(filenames), inputs['max_steps'], cfg['batch_size']))

            end_train_time = time.time() - start_time
            if args.out_name:
                out_name = args.out_name
            else:
                out_name = '{task}_{split}_places_accuracy.pkl'.format(task=given_task, split=args.data_split)
            save_path = os.path.join( save_dir, out_name )
            
            val_accuracy = {}
            with open( save_path, 'wb' ) as f:
                val_accuracy['file_indexes'] = filenames
                val_accuracy['global_step'] = gs
                val_accuracy['accuracy'] = accuracies
                pickle.dump( val_accuracy, f )
            
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                os.system("sudo cp {fp} {out}/".format(fp=save_path, out=args.out_dir))
            else:
                if transfer:
                    copy_to = cfg['log_root']
                else:
                    copy_to = os.path.join(cfg['log_root'], given_task)
                os.system("sudo mv {fp} {dst}/".format(fp=save_path, dst=copy_to))
                print("sudo mv {fp} {dst}/".format(fp=save_path, dst=copy_to))
                # if transfer:
                #     os.makedirs('/home/ubuntu/s3/model_log/losses_transfer/', exist_ok=True)
                #     os.system("sudo cp {fp} /home/ubuntu/s3/model_log/losses_transfer/".format(fp=save_path))
                # else:
                #     os.makedirs('/home/ubuntu/s3/model_log/losses/', exist_ok=True)
                #     os.system("sudo cp {fp} /home/ubuntu/s3/model_log/losses/".format(fp=save_path))

            print( 'saved losses to {0}'.format( save_path ))
            print('time to extract %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
        finally:
            utils.request_data_loading_end( training_runners )
            utils.end_data_loading_and_sess( training_runners )

def setup_metrics( inputs, model, cfg ):
    # predictions = model[ 'model' ].
    # Choose the metrics to compute:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map( {} )
    return  {}, {}


def get_max_steps(original_max_steps, data_split):
    n_images = None
    if data_split == 'train':
        n_images = 129380
    elif data_split == 'val':
        n_images = 50000 
    elif data_split == 'test':
        n_images = 6300 
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

