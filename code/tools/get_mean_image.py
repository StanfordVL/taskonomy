'''
  Name: get_mean_image.py
  Desc: Return the average image over a set of data. Supports mean and median. 
  Usage:
    python get_mean_image.py /path/to/cfgdir/ --stat-type median
'''
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import os
import pdb
import pickle as pkl
from   runstats import Statistics
import scipy
from   skimage import io as io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import time
import traceback as tb
import torch


# Import from local modules
import init_paths
from   general_utils import RuntimeDeterminedEnviromentVars
from   models.sample_models import *
import utils


parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument( 'cfg_dir', help='directory containing config.py file' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)

parser.add_argument('--is-training', dest='is_training', action='store_true')
parser.set_defaults(is_training=False)


parser.add_argument('--stat-type', dest='stat_type')
parser.set_defaults(stat_type='mean')

parser.add_argument('--data-split', dest='data_split')
parser.set_defaults(data_split='val')

parser.add_argument('--print-every', dest='print_every')
parser.set_defaults(print_every=10)

parser.add_argument('--not-one-hot', dest='not_one_hot', action='store_true')
parser.set_defaults(not_one_hot=False)

# PATCH_SIZE = (128,256)
PATCH_SIZE = None
# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False

def main( _ ):
    global PATCH_SIZE
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
    adjust_cfg_for_extract(args, cfg)

    print( os.path.join(cfg['log_dir'], "{}_label.png").format(args.stat_type))
    if 'metric_kwargs' in cfg:
        cfg['target_dim_tuple'] = (cfg['metric_kwargs']['output_size'],)
    if type(cfg['target_dim']) == int:
        cfg['target_dim_tuple'] = (cfg['target_dim'],)
    else:
        cfg['target_dim_tuple'] = cfg['target_dim']

    run_training( cfg, args.cfg_dir, args )

def run_training( cfg, cfg_dir, args ):
    if args.stat_type == "mean":
        statistic = MeanMeter(cfg)
    elif args.stat_type == 'median':
        statistic = MedianMeter(cfg)
    elif args.stat_type == 'marginal':
        statistic = DiscreteDistributionMeter(cfg, args.not_one_hot)
    elif args.stat_type == 'dense_marginal':
        statistic = DenseDiscreteDistributionMeter(cfg)
    elif args.stat_type == 'moments':
        statistic = MomentsMeter(cfg)
    else:
        raise NotImplementedError("No average defined for type: {}".format(args.stat_type))
    
    # set up logging
    tf.logging.set_verbosity( tf.logging.ERROR )

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = utils.setup_input( cfg, is_training=False )
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()

        # execute training 
        start_time = time.time()
        max_steps = get_max_steps(inputs[ 'max_steps' ], args.data_split)
        utils.print_start_info( cfg, max_steps, is_training=False )
        data_prefetch_threads_init_fn = utils.get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False )
        training_runners = { 'sess': tf.Session(), 
                    'coord': tf.train.Coordinator() }

        prefetch_threads = threading.Thread(
            target=data_prefetch_threads_init_fn,
            args=(
                training_runners[ 'sess' ],
                training_runners[ 'coord' ]))
        prefetch_threads.start()
        
        
        target_batch = training_runners['sess'].run( inputs['target_batch'] )
        # training_runners[ 'threads' ] = data_prefetch_init_fn( training_runners[ 'sess' ], training_runners[ 'coord' ] )
        try:
            # This just returns the imput as output. It is for testing data
            #  input only. 
            start_time = time.time()
            batch_time = time.time()
            k = int(args.print_every)
            for step in range( max_steps ):
                target_batch, mask_batch = training_runners['sess'].run( 
                    [ inputs['target_batch'], inputs['mask_batch'] ] )
                target_batch = map_to_img(target_batch.mean(axis=0), cfg)
                if len(mask_batch.shape) > 1:
                    mask_batch = mask_batch.mean(axis=0)
                else:
                    mask_batch = 1

                statistic.update(
                    target_batch,
                    mask_batch)
                if (step+1) % k == 0:
                    print('Step %d/%d: %.2f s/step ' % (
                        step+1, max_steps, (time.time() - batch_time)/k ))
                    batch_time = time.time()
                    # print(statistic.get())
                    # break 
                if training_runners['coord'].should_stop():
                    break

            end_train_time = time.time() - start_time
            print('time to train %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
            print('avg time per epoch: %.3f hrs' % ( (end_train_time/(60*60)) / cfg['num_epochs']) )
            if args.stat_type == 'moments':
                save_moments(statistic, cfg, args)
            else:
                save_data(statistic, cfg, args)
        finally:
            utils.request_data_loading_end( training_runners )
            utils.end_data_loading_and_sess( training_runners )


class MeanMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dtype = get_dtype(get_bit_depth(cfg))
        self.is_float = self.dtype == np.float64 or self.dtype == np.float32
        self.reset()

    def reset(self):
        self._avg = 0
        self.sum = self.make_sums_array()
        self.count = np.zeros_like(self.sum)

    def update(self, val, n=1):
        self.saved_val = val
        if type(n) != int:
            n = n.astype(np.uint8)
        self.sum += np.multiply(val, n)
        self.count += n

    def get(self):
        self._avg = self.sum.astype(np.float64) / self.count
        if not self.is_float:
            self._avg = np.round(self._avg)
        return self._avg

    def make_sums_array(self):
        # freqs has shape W x H x N_channels 
        cfg = self.cfg
        shape = tuple(
            list(cfg['target_dim_tuple']) + 
            get_n_channels(self.cfg))
        if self.is_float:
            sums = np.zeros(shape, dtype=np.float64)
        else:
            sums = np.zeros(shape, dtype=np.int64)
        return sums



class MedianMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dtype = get_dtype(get_bit_depth(cfg))
        self.is_float = self.dtype == np.float64 or self.dtype == np.float32
        if self.is_float:
            raise NotImplementedError(
                "MedianMeter cannot find the median of a stream of floats.")
        self.reset()
    
    def reset(self):
        self.freqs = self.make_freqs_array()

    def update(self, val, mask=1):
        start = time.time()
        self.val = torch.LongTensor( val.astype(np.int64).flatten()[np.newaxis,:] )

        if type(mask) == int:
            mask = torch.IntTensor(self.val.size()).fill_(mask)
        else:
            mask = torch.IntTensor(mask.astype(np.int32).flatten()[np.newaxis,:])

        if USE_CUDA:
            self.val = self.val.cuda()
            mask = mask.cuda()

        self.freqs.scatter_add_(0, self.val, mask) 
        # print("Update finished in {}".format(time.time() - start))
        self.saved_val = val

    def get(self):
        self.freqs = self.freqs.cpu().numpy()
        self.freqs = np.cumsum(
            self.freqs, 
            axis=0, out=self.freqs)
        self._avg = np.apply_along_axis(
            lambda a: a.searchsorted(a[-1] / 2.), 
            axis=0, 
            arr=self.freqs)\
        .reshape(
            tuple( [-1] + list(self.cfg['target_dim_tuple']) + get_n_channels(self.cfg)))
        return self._avg

    def make_freqs_array(self):
        # freqs has shape N_categories x W x H x N_channels 
        cfg = self.cfg
        shape = tuple([2**get_bit_depth(self.cfg)] + 
                        # list(cfg['target_dim']) + 
                        list(cfg['target_dim_tuple']) + 
                        get_n_channels(self.cfg))
        freqs = torch.IntTensor(shape[0], int(np.prod(shape[1:]))).zero_()
        if USE_CUDA:
            freqs = freqs.cuda()

        return freqs

class SparseDiscreteDistributionMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_float = True
        self.support_size = 1
        if get_n_channels(self.cfg):
            self.support_size = get_n_channels(self.cfg)[0]
        self.reset()

    def reset(self):
        self.freqs = self.make_freqs_array()

    def update(self, val, mask=1):
        start = time.time()
        if len(val.shape) == 0: # Consider this an image
            val = val.reshape((1,1))
        else:
            val = np.argmax(val, axis=-1)
        self.val = torch.LongTensor( val.astype(np.int64).flatten()[np.newaxis, :] )

        if type(mask) == int:
            mask = torch.DoubleTensor(self.val.size()).fill_(mask)
        else:
            mask = torch.DoubleTensor(mask.astype(np.float64).flatten()[np.newaxis,:])
        if USE_CUDA:
            self.val = self.val.cuda()
            mask = mask.cuda()
        self.freqs.scatter_add_(0, self.val, mask) 
        # print("Update finished in {}".format(time.time() - start))
        self.saved_val = val

    def get(self):
        self.freqs = self.freqs.cpu().numpy()
        self._avg = self.freqs.astype(np.float64) / np.sum(self.freqs, axis=0)[np.newaxis, ...]
        self._avg = self._avg.reshape(
            tuple( [-1] + list(self.cfg['target_dim_tuple'])))
        self._avg = np.moveaxis(self._avg, 0, -1)
        return self._avg

    def make_freqs_array(self):
        # freqs has shape N_categories x W x H x N_channels 
        cfg = self.cfg
        shape = tuple(get_n_channels(self.cfg) +
                        list(cfg['target_dim_tuple']))
        freqs = torch.DoubleTensor(shape[0], int(np.prod(shape[1:]))).zero_()
        if USE_CUDA:
            freqs = freqs.cuda()

        return freqs


class DiscreteDistributionMeter(object):
    def __init__(self, cfg, not_one_hot):
        self.cfg = cfg
        self.is_float = True
        self.support_size = 1
        if get_n_channels(self.cfg):
            self.support_size = get_n_channels(self.cfg)[0]
        self.reset()
        self.not_one_hot = not_one_hot

    def reset(self):
        self.freqs = self.make_freqs_array()

    def update(self, val, mask=1):
        start = time.time()
        if len(val.shape) == 0: # Consider this an image
            val = val.reshape((1,1))
        elif self.not_one_hot:
            val = val
        else:
            val = np.argmax(val, axis=-1)
        self.val = torch.LongTensor( val.astype(np.int64).flatten()[np.newaxis, :] )

        if type(mask) == int:
            mask = torch.DoubleTensor(self.val.size()).fill_(mask)
        else:
            mask = torch.DoubleTensor(mask.astype(np.float64).flatten()[np.newaxis,:])
        if USE_CUDA:
            self.val = self.val.cuda()
            mask = mask.cuda()
        # pdb.set_trace()
        self.freqs.scatter_add_(0, self.val, mask) 
        # print("Update finished in {}".format(time.time() - start))
        self.saved_val = val

    def get(self):
        self.freqs = self.freqs.cpu().numpy()
        self._avg = self.freqs.astype(np.float64) / np.sum(self.freqs, axis=0)[np.newaxis, ...]
        self._avg = self._avg.reshape(
            tuple( [-1] + list(self.cfg['target_dim_tuple'])))
        self._avg = np.moveaxis(self._avg, 0, -1)
        return self._avg

    def make_freqs_array(self):
        # freqs has shape N_categories x W x H x N_channels 
        cfg = self.cfg
        shape = tuple(get_n_channels(self.cfg) +
                        list(cfg['target_dim_tuple']))
        freqs = torch.DoubleTensor(shape[0], int(np.prod(shape[1:]))).zero_()
        if USE_CUDA:
            freqs = freqs.cuda()

        return freqs

class DenseDiscreteDistributionMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_float = True
        self.support_size = 1
        if get_n_channels(self.cfg):
            self.support_size = get_n_channels(self.cfg)[0]
        elif len(cfg['target_dim_tuple']) == 1:
            self.support_size = cfg['target_dim_tuple'][0]
        self.reset()

    def reset(self):
        self.freqs = self.make_freqs_array()

    def update(self, val, mask=1):
        start = time.time()
        # if len(val.shape) == 0: # Consider this an image
            # val = val.reshape((1,1))
        # else:
        val = np.moveaxis(val, -1, 0)
        self.val = torch.DoubleTensor(
            val.reshape(
                (self.support_size, int(np.prod(list(val.shape)[1:]))))\
                .astype(np.float64) )

        if type(mask) == int:
            mask = torch.DoubleTensor(self.val.size()).fill_(mask)
        else:
            mask = torch.DoubleTensor(mask.astype(np.float64).flatten()[np.newaxis,:])

        if USE_CUDA:
            self.val = self.val.cuda()
            mask = mask.cuda()
        self.freqs += self.val * mask
        # print("Update finished in {}".format(time.time() - start))
        self.saved_val = val

    def get(self):
        self.freqs = self.freqs.cpu().numpy()
        self._avg = self.freqs.astype(np.float64) / np.sum(self.freqs, axis=0)[np.newaxis, ...]
        self._avg = self._avg.reshape(
            tuple( [-1] + list(self.cfg['target_dim_tuple'])))
        self._avg = np.moveaxis(self._avg, 0, -1)
        return self._avg

    def make_freqs_array(self):
        # freqs has shape N_categories x W x H x N_channels 
        cfg = self.cfg
        shape = tuple(get_n_channels(self.cfg) +
                        list(cfg['target_dim_tuple']))
        freqs = torch.DoubleTensor(shape[0], int(np.prod(shape[1:]))).zero_()
        if USE_CUDA:
            freqs = freqs.cuda()

        return freqs

class MomentsMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dtype = get_dtype(get_bit_depth(cfg))
        self.is_float = self.dtype == np.float64 or self.dtype == np.float32
        self.reset()
        print("Label", self.stats_arr.shape)

    def reset(self):
        self._avg = 0
        self.stats_arr = self.make_statistics_array()

    def update(self, val, mask=1):
        # print(val)
        self.saved_val = val
        for index, value in np.ndenumerate( val ):
            if not np.isfinite(value):
                raise("A value is not finite")
            if type(mask) == int or len(mask.shape()) == 0:
                maskval = mask
            else:
                maskval = mask[index]
            if maskval == 0:
                continue
            else:
                value = value * mask

            self.stats_arr[index].push(value)
            # print(len(self.stats_arr[index]), value)
    def get(self):
        means = np.zeros_like(self.stats_arr)
        std = np.zeros_like(self.stats_arr)
        for index, value in np.ndenumerate( self.stats_arr ):
            means[index] = value.mean()
            std[index] = value.stddev()
        return means, std

    def make_statistics_array(self):
        # freqs has shape W x H x N_channels 
        cfg = self.cfg
        shape = tuple(
            list(cfg['target_dim_tuple']) + 
            get_n_channels(self.cfg))
        self.shape = shape
        def make_stats_array_recursive(shape):
            if len(shape) == 0:
                return Statistics()
            else:
                return [make_stats_array_recursive(shape[1:])
                    for _ in range(shape[0])]
        return np.array(make_stats_array_recursive(list(shape)))

def map_to_img(val, cfg):
    if 'target_num_channels' in cfg and cfg['target_num_channels'] in [1,3]:
        return np.round(
            (val + 1.) * (2**(get_bit_depth(cfg)-1)-1)).astype(
            get_dtype(get_bit_depth(cfg)))
    else:
        return val

def save_data(statistic, cfg, args):
    # Save the statistic data
    print("Computing statistic")
    start = time.time()
    statistic_image = np.squeeze(statistic.get()).astype(
        get_dtype(get_bit_depth(cfg)))
    # print(statistic_image)
    print("dtype: {}".format(statistic_image.dtype))
    print("shape: {}".format(statistic_image.shape))
    print("min/max: {}/{}".format(statistic_image.min(), statistic_image.max()))
    print("Computed statistic ({:.2f} secs)".format(time.time() - start))

    print("Writing pkl")
    with open(os.path.join(cfg['log_dir'], "{}_label.pkl").format(args.stat_type), 'wb') as f:
        pkl.dump(statistic_image, f)
   
    print("Writing png to {}".format(
        os.path.join(cfg['log_dir'], "{}_label.png").format(args.stat_type)))
    if len(statistic_image.shape) == 0:
        statistic_image = statistic_image[np.newaxis, np.newaxis]
        saved_val = np.squeeze(statistic.saved_val)[np.newaxis, np.newaxis]
    elif len(statistic_image.shape) == 1:
        print(statistic_image.shape)
        statistic_image = statistic_image[:, np.newaxis]
        try:
            saved_val = np.squeeze(statistic.saved_val)[:, np.newaxis]
        except: # it's an index for a 1-hot encoding
            saved_val = np.zeros_like(statistic_image)
            saved_val[int(statistic.saved_val)] = 1.
    else:
        saved_val = np.squeeze(statistic.saved_val)

    if len(statistic_image.shape) == 2 or statistic_image.shape[-1] in [1,3,4]:
        io.imsave( os.path.join(cfg['log_dir'], "{}_label.png").format(args.stat_type), statistic_image )
        try:
            io.imsave( os.path.join(cfg['log_dir'], "single_label.png"), saved_val.astype(
                get_dtype(get_bit_depth(cfg))) )
        except:
            tb.print_exc()

    print("Done :)")

def save_moments(statistic, cfg, args):
    print("Computing statistic")
    start = time.time()
    print("Writing pkl to {}".format(
        os.path.join(cfg['log_dir'], "{}_label.pkl").format(args.stat_type)))
    
    means, std = statistic.get()
    print(means)
    print(std)
    print("Writing pkl")
    with open(os.path.join(cfg['log_dir'], "{}_label.pkl").format(args.stat_type), 'wb') as f:
        pkl.dump({'mean': means, 'std': std}, f)


def get_bit_depth(cfg):
    # pdb.set_trace()
    if 'target_num_channels' in cfg:
        if cfg['target_num_channels'] == 3:
            # Assume 8-bit depth RGB image
            return 8
        elif cfg['target_num_channels'] == 1:
            # Assume 16-bit depth single-channel image
            return 16
        elif cfg['target_num_channels'] == 2:
            # Assume 16-bit depth single-channel image
            return 8

def get_dtype(bit_depth):
    if bit_depth == 8:
        return np.uint8
    if bit_depth == 16:
        return np.uint16
    if bit_depth is None:
        return np.float64

def get_n_channels(cfg):
    if 'target_num_channels' in cfg:
        return [cfg['target_num_channels']]
    else:
        return []

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

def adjust_cfg_for_extract(args, cfg):
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

    cfg['num_epochs'] = 1
    cfg['randomize'] = False
    cfg['num_read_threads'] = 50
    # if args.avg_type == 'median':
    cfg['batch_size'] = 1
    # cfg['num_read_threads'] = 1
    # if args.representation_task:
    #     split_file_ = args.representation_task
    #     if 'multiple_input_tasks' in cfg:
    #         split_file_ = [split_file_]
    #     cfg['train_representations_file'] = split_file_
    #     cfg['val_representations_file'] = split_file_
    #     cfg['test_representations_file'] = split_file_
    
    return cfg
if __name__=='__main__':
  main( '' )
  # tf.app.run()
