'''
  Name: extract_losses_for_avg_img.py
  Desc: Extract losses for a model which always predicts the average image
  Usage:
    python extract_losses_for_avg_img.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import threading
import numpy as np
import pdb
import pickle
from   runstats import Statistics

import init_paths
from   data.load_ops import resize_rescale_image
from   data.task_data_loading import load_and_specify_preprocessors_for_representation_extraction
import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars
import models.architectures as architectures
import utils

parser = argparse.ArgumentParser(description='Extract losses of encoder decoder model.')
parser.add_argument( '--cfg_dir', dest='cfg_dir', help='directory containing config.py file, should include a checkpoint directory' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)

# parser.add_argument('--is_val', dest='is_val', action='store_true')
# parser.set_defaults(is_val=True)

parser.set_defaults(cfg_dir="/home/ubuntu/task-taxonomy-331b/experiments/aws_second")

parser.add_argument('--task', dest='task')
parser.add_argument('--data-split', dest='data_split')
parser.set_defaults(data_split="val")

parser.add_argument('--out-dir', dest='out_dir')
parser.set_defaults(out_dir="")

parser.add_argument('--out-name', dest='out_name')
parser.set_defaults(out_name="")

parser.add_argument('--avg-type', dest='avg_type')
parser.set_defaults(avg_type="median")

parser.add_argument('--loss-type', dest='loss_type')
parser.set_defaults(loss_type="pose")

parser.add_argument('--print-every', dest='print_every')
parser.set_defaults(print_every="100")

def main( _ ):
    global args
    args = parser.parse_args()

    #task_list = ["autoencoder", "colorization","curvature", "denoise", "edge2d", "edge3d", "ego_motion", "fix_pose", "impainting", "jigsaw", "keypoint2d", "keypoint3d", "non_fixated_pose", "point_match", "reshade", "rgb2depth", "rgb2mist", "rgb2sfnorm", "room_layout", "segment25d", "segment2d", "vanishing_point"]
    #single channel for colorization !!!!!!!!!!!!!!!!!!!!!!!!! COME BACK TO THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!
    task_list = [ args.task ]
    #task_list = [ "vanishing_point"]

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
            
        if cfg['model_path'] is None:
            cfg['model_path'] = os.path.join(cfg['dataset_dir'], "model_log", task, "model.permanent-ckpt") 
        if cfg['model_type'] == architectures.TransferNet:
            cfg['model_path'] = os.path.join(cfg['log_root'], task, "model.permanent-ckpt") 
        cfg['randomize'] = False
        cfg['num_epochs'] = 1
        cfg['num_read_threads'] = 20
        cfg['batch_size'] = 1

        loss_dir = args.cfg_dir
        avg_img = load_avg_img(cfg, args)
        # print(np.sum(avg_img, axis=2))
        print(avg_img.max())
        print("Average image size: {}".format(avg_img.shape))
        avg_img = map_img_to_target_range(avg_img, cfg)
        run_extract_losses( avg_img, args, cfg, loss_dir, task )
        # pdb.set_trace()

def run_extract_losses( avg_img, args, cfg, save_dir, given_task ):
    transfer = (cfg['model_type'] == architectures.TransferNet)
    if transfer:
        get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn_transfer
        setup_input_fn = utils.setup_input_transfer
    else:
        setup_input_fn = utils.setup_input
        get_data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn
    
    stats = Statistics()
    
    # set up logging
    tf.logging.set_verbosity( tf.logging.ERROR )

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        
        # build model (and losses and train_op)
        # model = utils.setup_model( inputs, cfg, is_training=False )
        loss_names = [avg_img_to_loss_type(args.avg_type, given_task)] # Keep format the same as extract_losses.py
        loss_fn = get_loss_op(loss_names[0])

        # execute training 
        start_time = time.time()
        max_steps = get_max_steps(inputs[ 'max_steps' ], args.data_split)
        utils.print_start_info( cfg, max_steps, is_training=False )

        # start session and restore model
        training_runners = { 'sess': tf.Session(), 'coord': tf.train.Coordinator() }
        try:
            utils.print_start_info( cfg, max_steps, is_training=False )

            data_prefetch_init_fn = get_data_prefetch_threads_init_fn( inputs, cfg, 
                is_training=False, use_filename_queue=False )
            #training_runners[ 'threads' ] = data_prefetch_init_fn( training_runners[ 'sess' ], training_runners[ 'coord' ] )
            prefetch_threads = threading.Thread(
                target=data_prefetch_init_fn,
                args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
            prefetch_threads.start()
            
            # run one example so that we can calculate some statistics about the representations
            filenames = []
            loss_names_to_vals = {name: [] for name in loss_names}
            start = time.perf_counter()

            print_every = int(args.print_every)
            # run the remaining examples
            for step in range( max_steps ):
                data_idx, target, mask = training_runners['sess'].run( [ 
                            inputs[ 'data_idxs' ], 
                            inputs['target_batch'], 
                            inputs['mask_batch'] ] )       
                loss = loss_fn(avg_img, target, mask)
                # print(loss)
                assert np.isfinite(loss) and loss >= 0.0
                loss_names_to_vals[loss_names[0]].append(loss)
                filenames.extend(data_idx)
                stats.push(loss)

                if step % print_every == 0 and step > 0: 
                    print( 'Step {0} of {1}: (Mean {5}: {2:.3f} || stddev: {3:.3f} :: ({4:.2f} secs/step)'.format( 
                        step, max_steps - 1,
                        stats.mean(), 
                        np.sqrt(stats.variance()),
                        (time.perf_counter() - start) / print_every, 
                        loss_names[0]))
                    start = time.perf_counter()
                if training_runners['coord'].should_stop():
                    break

            print('The size of losses is %s while we expect it to run for %d steps with batchsize %d' % (len(filenames), inputs['max_steps'], cfg['batch_size']))

            end_train_time = time.time() - start_time
            if args.out_name:
                out_name = args.out_name
            else:
                if args.data_split == "val":
                    split_name = "train" 
                if args.data_split == "test":
                    split_name = "val" 
                else:
                    raise ValueError("Cannot adequately name output for data split {}".format(args.data_split))
                out_name = '{avg_type}__{task}_{split}_losses.pkl'.format(
                    task=given_task, split=split_name,
                    avg_type="marginal" if args.avg_type == 'dense_marginal' else args.avg_type)
            save_path = os.path.join( save_dir, out_name )

            with open( save_path, 'wb' ) as f:
                loss_names_to_vals['file_indexes'] = filenames
                loss_names_to_vals['global_step'] = 0
                if 'dense_xentropy_loss' in loss_names_to_vals:
                    loss_names_to_vals['xentropy_loss'] = loss_names_to_vals['dense_xentropy_loss']
                    del loss_names_to_vals['dense_xentropy_loss']
                pickle.dump( loss_names_to_vals, f )
            
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                os.system("sudo mv {fp} {out}/".format(fp=save_path, out=args.out_dir))
            else:
                if transfer:
                    os.makedirs('/home/ubuntu/s3/model_log/losses_transfer/', exist_ok=True)
                    os.system("sudo mv {fp} /home/ubuntu/s3/model_log/losses_transfer/".format(fp=save_path))
                else:
                    os.makedirs('/home/ubuntu/s3/model_log/losses/', exist_ok=True)
                    os.system("sudo mv {fp} /home/ubuntu/s3/model_log/losses/".format(fp=save_path))

            print( 'saved losses to {0}'.format( save_path ))
            print('time to extract %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
        finally:
            utils.request_data_loading_end( training_runners )
            utils.end_data_loading_and_sess( training_runners )

def get_bit_depth(cfg):
    if 'target_num_channels' in cfg:
        if cfg['target_num_channels'] == 3:
            # Assume 8-bit depth RGB image
            return 8
        elif cfg['target_num_channels'] == 2:
            # Assume 8-bit depth RGB image
            return 8
        elif cfg['target_num_channels'] == 1:
            # Assume 16-bit depth single-channel image
            return 16

def get_dtype(bit_depth):
    if bit_depth == 8:
        return np.uint8
    if bit_depth == 16:
        return np.uint16

def get_n_channels(cfg):
    if 'target_num_channels' in cfg:
        return cfg['target_num_channels']

def get_loss_op(name):
    if name == 'xentropy_loss':
        return get_xentropy_loss
    elif name == 'l1_loss':
        return get_l1_loss
    elif name == 'l2_loss':
        return get_l2_loss
    elif name == 'dense_xentropy_loss':
        return get_dense_xentropy_loss
    elif name == 'pose_loss':
        return get_pose_loss
    elif name == 'mse_loss':
        return get_mse_loss
    raise NotImplementedError(name)

def get_l1_loss(predict, label, mask):
    if label.shape[-1] == 1:
        label = np.squeeze(label, axis=3)
        mask = np.squeeze(mask, axis=3)
    return (np.abs(predict - label) * mask)[mask > 0.0].mean()

def get_l2_loss(predict, label, mask):
    return np.sqrt(np.sum((predict - label)**2 * mask))[mask > 0.0].mean()

def get_mse_loss(predict, label, mask):
    return np.sum((predict - label)**2 * mask)[mask > 0.0].mean()

def get_xentropy_loss(predict, label, mask):
    epsilon = 1e-9
    # idxs = label.argmax(axis=-1).flatten()
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

def get_pose_loss(predict, label, mask):
    res = np.sqrt(np.sum((predict - label)**2 * mask))
    if res > 1.:
        res = 1. + np.log(res)
    return res


def load_avg_img(cfg, args):
    with open(os.path.join(cfg['log_dir'], "{}_label.pkl").format(args.avg_type), 'rb') as f:
        return pickle.load(f)

def map_img_to_target_range(img, cfg):
    if 'target_num_channels' in cfg and cfg['target_num_channels'] in [1,2,3]:
        return img.astype(np.float32) / ((2**(get_bit_depth(cfg)-1)-1)) - 1
    return img

def map_to_img(val, cfg):
    if 'target_num_channels' in cfg and cfg['target_num_channels'] in [1,3]:
        newval = np.round(
            (val + 1.) * (2**(get_bit_depth(cfg)-1)-1)).astype(
            get_dtype(get_bit_depth(cfg)))
    return newval

def avg_img_to_loss_type(name, task):
    global args
    if name == 'mean':
        if args.loss_type == 'pose':
            return 'pose_loss'
        elif args.loss_type == 'mse':
            return 'mse_loss'
        else:
            return 'l2_loss'
    elif name == 'median':
        return 'l1_loss'
    elif name == 'marginal':
        pdb.set_trace()
        if args.loss_type == 'dense_xentropy':
            return 'dense_xentropy_loss'
        else:
            return 'xentropy_loss'
    elif name == 'dense_marginal':
        return 'dense_xentropy_loss'
    raise NotImplementedError("Unknown average type: {}".format(name))

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

