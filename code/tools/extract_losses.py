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
parser.set_defaults(cfg_dir="/home/ubuntu/task-taxonomy-331b/experiments/final")

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

parser.add_argument('--generalize', dest='gen', action='store_true')
parser.set_defaults(gen=False)

parser.add_argument('--train-mode', dest='train_mode', action='store_true')
# parser.set_defaults(print_every="100")
SPLIT_FILE = None
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
        task_dir = os.path.join(args.cfg_dir, task)
        cfg = utils.load_config( task_dir, nopause=args.nopause )
        root_dir = cfg['root_dir']

        if args.gen:
            cfg['train_filenames'] = os.path.abspath( os.path.join( '/home/ubuntu/s3', 'meta/val_split_image_info.pkl') )
            cfg['val_filenames'] = os.path.abspath( os.path.join( '/home/ubuntu/s3', 'meta/test_split_image_info.pkl') )
            cfg['test_filenames'] = os.path.abspath( os.path.join( '/home/ubuntu/s3', 'meta/test_split_image_info.pkl') )

        if args.data_split == 'train':
            split_file = cfg['train_filenames']
        elif args.data_split == 'val':
            split_file = cfg['val_filenames']
        elif args.data_split == 'test':
            split_file = cfg['test_filenames']
        else: 
            raise NotImplementedError("Unknown data split section: {}".format(args.data_split))
        global SPLIT_FILE
        SPLIT_FILE = split_file
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
        
        # Flip order 
        # with open('/home/ubuntu/task-taxonomy-331b/tools/second_order_should_flip.pkl', 'rb') as fp:
        #     to_flip_dict = pickle.load(fp)
        # if not to_flip_dict[task.split('/')[-1]]:
        #     cfg['val_representations_file'] = cfg['val_representations_file'][::-1]

        # Try latest checkpoint by epoch
        #if cfg['model_path'] is None:
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
            if task.split('/')[0] == 'final':
                cfg['model_path'] = os.path.join(cfg['log_root'], task.split('/')[-1], "model.permanent-ckpt") 
                task = task.split('/')[-1]
            else:
                cfg['model_path'] = os.path.join(cfg['log_root'], task, "model.permanent-ckpt") 
            # cfg['model_path'] = os.path.join(cfg['log_root'], 'logs', 'slim-train', 'time', "model.ckpt-1350") 

        cfg['randomize'] = False
        cfg['num_epochs'] = 1
        cfg['batch_size'] = 32 if TRAIN_MODE else 1
        cfg['num_read_threads'] = 1
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
        run_extract_losses( args, cfg, loss_dir, task )


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
        elif 'target_cfg' in cfg and 'l2_loss' in cfg['target_cfg'] and cfg['target_cfg']['l2_loss']:
            losses.append(('l2_loss', model.siamese_loss))
        else:
            losses.append(('xentropy', model.siamese_loss))
    elif target_model_type == architectures.SegmentationEncoderDecoder \
            or target_model_type == architectures.ConstantPredictorSegmentation:    
            losses.append(('metric_loss', model.metric_loss))
    elif target_model_type == architectures.SemSegED:    
            losses.append(('metric_loss', model.metric_loss))
    elif target_model_type == architectures.CycleSiamese \
        or target_model_type == architectures.ConstantPredictorPose:    
            losses.append(('cycle_loss', model.cycle_loss_total))
    elif target_model_type == architectures.EDSoftmax:    
            losses.append(('xentropy', model.softmax_loss))
    else:
        raise NotImplementedError("Extracting losses not implemented for {}".format(
            target_model_type
        ))
    return zip(*losses)

def get_xentropy_loss(predict, label, mask):
    epsilon = 1e-9
    # idxs = label.argmax(axis=-1).flatten()
    pdb.set_trace()
    idxs = label.flatten().astype(np.int64)
    predict = predict.reshape(len(idxs), predict.shape[-1])
    predict[predict < epsilon] = epsilon
    predict[predict > 1 - epsilon] = 1 - epsilon
    losses = -np.log(predict)[range(len(idxs)), idxs]
    return (losses * mask.flatten())[mask.flatten() > 0.0].mean() #It's what tensorflow uses...

def get_dense_xentropy_loss(predict, label, mask):
    epsilon = 1e-8
    # idxs = label.argmax(axis=-1).flatten()
    # predict = predict.reshape(len(idxs), predict.shape[-1])
    predict[predict < epsilon] = epsilon
    predict[predict > 1 - epsilon] = 1 - epsilon
    losses = np.sum(-np.multiply(label, np.log(predict)) * mask[..., np.newaxis], axis=-1)
    return losses[losses > 0.0].mean()

def run_extract_losses( args, cfg, save_dir, given_task ):
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
    print_every = int(args.print_every)

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
        #RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        #RuntimeDeterminedEnviromentVars.populate_registered_variables()
        max_steps = get_max_steps(inputs[ 'max_steps' ], args.data_split)
        # pdb.set_trace()
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
            loss_names_to_vals = {name: [] for name in loss_names}
            results = training_runners['sess'].run( [ 
                    inputs[ 'data_idxs' ], 
                    inputs['target_batch'],  inputs['mask_batch'], 
                    *loss_ops ] )       
            #gs = results[1] 
            data_idx = results[0]
            losses = results[3:]
            target_input = results[1]
            mask_input = results[2]
            for i, name in enumerate(loss_names):
                loss_names_to_vals[name].append(losses[i])
            filenames.extend(data_idx)
            print("Step number: {}".format(1), (data_idx))
            # print(target_input, target_input.sum())
            # return
            # training_runners['sess'].run([v for v in tf.global_variables() if "transfer/rep_conv_1/weights" in v.name][0])
            # run the remaining examples
            start = time.perf_counter()
            for step in range( max_steps - 1 ):
                results = training_runners['sess'].run( [
                        inputs[ 'data_idxs' ], 
                        # [v for v in tf.global_variables() if "transfer/rep_conv_1/weights/(weights)" in v.name][0], 
                        # model['model'].encoder_endpoints['net1_1_output'], 
                        # model['model'].encoder_endpoints['net1_2_output'], 
                        *loss_ops ] )    
                data_idx = results[0]
                # print(data_idx)
                losses = results[1:]
                # p, t, m = results[1], results[2], results[3]
                # losses = results[4:]

                # print(p.mean(), t)
                for i, name in enumerate(loss_names):
                    loss_names_to_vals[name].append(losses[i])
                filenames.extend(data_idx)
                stats.push(loss_names_to_vals[display_loss][-1])
                
                # baseline_loss = get_xentropy_loss(p, t, m)
                # tf_loss = loss_names_to_vals[display_loss][-1]
                # print('tf {} | ours {}'.format(tf_loss, baseline_loss))
                # pdb.set_trace()

                
                if step % print_every == 0 and step > 0: 
                    print( 'Step {0} of {1}: ({5} loss: {2:.3f} || stddev: {3:.3f} :: ({4:.2f} secs/step)'.format( 
                        step, max_steps - 1,
                        stats.mean(), 
                        np.sqrt(stats.variance()),
                        # stats.variance(),
                        (time.perf_counter() - start) / print_every,
                        display_loss
                        ))
                    start = time.perf_counter()

                if training_runners['coord'].should_stop():
                    break

            print('The size of losses is %s while we expect it to run for %d steps with batchsize %d' % (len(filenames), inputs['max_steps'], cfg['batch_size']))

            end_train_time = time.time() - start_time
            if args.out_name:
                out_name = args.out_name
            else:
                out_name = '{task}_{split}_losses.pkl'.format(task=given_task, split=args.data_split)
            save_path = os.path.join( save_dir, out_name )

            with open( save_path, 'wb' ) as f:
                loss_names_to_vals['file_indexes'] = filenames
                loss_names_to_vals['global_step'] = 0 
                pickle.dump( loss_names_to_vals, f )
            
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
    # if data_split == 'train':
    #     n_images = 129380
    # elif data_split == 'val':
    #     n_images = 29933
    # elif data_split == 'test':
    #     n_images = 17853
    # else: 
    #     raise NotImplementedError('Unknown data split {}'.format(data_split))
    print(SPLIT_FILE)
    if 'train' in SPLIT_FILE:
        n_images = 129380
    elif 'val' in SPLIT_FILE:
        n_images = 29933
    elif 'test' in SPLIT_FILE:
        n_images = 17853
    else: 
        raise NotImplementedError('Unknown data split {}'.format(data_split))
    if original_max_steps != n_images:
        print("Adjusting number of steps from {} -> {}".format(
            max(original_max_steps, n_images),
            min(original_max_steps, n_images)
        ))
    #pdb.set_trace()
    return min(original_max_steps, n_images)


if __name__=='__main__':
    main( '' )

