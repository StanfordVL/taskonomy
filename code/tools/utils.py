"""
    utils.py

    Contains some useful functions for creating models
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import pickle
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import concurrent.futures


import init_paths
import data.load_ops as load_ops
from   data.load_ops import create_input_placeholders_and_ops, get_filepaths_list
import general_utils
# from   lib.savers.aws_saver import AwsSaver
AwsSaver = tf.train.Saver
import optimizers.train_steps as train_steps
import models.architectures as architectures

def get_available_devices():
    from tensorflow.python.client import device_lib
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return device_lib.list_local_devices()

def get_max_steps( num_samples_epoch, cfg , is_training=True):
    if cfg['num_epochs']:
        max_steps = num_samples_epoch * cfg['num_epochs'] // cfg['batch_size']
    else: 
        max_steps = None
    if not is_training:
        max_steps = num_samples_epoch // cfg['batch_size']
    print( 'number of steps per epoch:',
           num_samples_epoch // cfg['batch_size'] )
    print( 'max steps:', max_steps )
    return max_steps

def load_config( cfg_dir, nopause=False ):
    ''' 
        Raises: 
            FileNotFoundError if 'config.py' doesn't exist in cfg_dir
    '''
    if not os.path.isfile( os.path.join( cfg_dir, 'config.py' ) ):
        raise ImportError( 'config.py not found in {0}'.format( cfg_dir ) )
    import sys
    sys.path.insert( 0, cfg_dir )
    from config import get_cfg
    cfg = get_cfg( nopause )
    # cleanup
    try:
        del sys.modules[ 'config' ]
    except:
        pass
    sys.path.remove(cfg_dir)

    return cfg

def print_start_info( cfg, max_steps, is_training=False ):
    model_type = 'training' if is_training else 'testing'
    print("--------------- begin {0} ---------------".format( model_type ))
    print('number of epochs', cfg['num_epochs'])
    print('batch size', cfg['batch_size'])
    print('total number of training steps:', max_steps)


##################
# Model building
##################
def create_init_fn( cfg, model ):
    # restore model
    if cfg['model_path'] is not None:
        print('******* USING SAVED MODEL *******')
        checkpoint_path = cfg['model_path']
        model['model'].decoder
        # Create an initial assignment function.
        def InitAssignFn(sess):
            print('restoring model...')
            sess.run(init_assign_op, init_feed_dict)
            print('model restored')

        init_fn = InitAssignFn
    else:
        print('******* TRAINING FROM SCRATCH *******')
        init_fn = None
    return init_fn 

def setup_and_restore_model( sess, inputs, cfg, is_training=False ):
    model = setup_model( inputs, cfg, is_training=False )
    model[ 'saver_op' ].restore( sess, cfg[ 'model_path' ] )
    return model

def setup_input_3m( cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds input tensors from the config.
    '''
    inputs = {}
    if is_training:
        filepaths_list = get_filepaths_list( cfg[ 'train_filenames' ] )
    else: 
        filepaths_list = get_filepaths_list( cfg[ 'val_filenames' ] )


    # Generate placeholder input tensors
    num_samples_epoch = filepaths_list['total_size']
    inputs[ 'unit_size' ] = filepaths_list['unit_size']
    filepaths_list = [ os.path.join( '/home/ubuntu/s3/meta', i) for i in filepaths_list['filename_list']]
    if use_filename_queue:
        filename_queue_dict = load_ops.create_filename_enqueue_op( cfg )
        inputs[ 'next_idx_op' ] = filename_queue_dict[ 'dequeue_op' ]
        inputs[ 'filename_enqueue_op' ] = filename_queue_dict[ 'enqueue_op' ]
        inputs[ 'next_idx_placeholder'] = filename_queue_dict[ 'data_idx_placeholder' ] 
        inputs[ 'fname_q' ] = filename_queue_dict
    placeholders, batches, load_and_enqueue, enqueue_op = create_input_placeholders_and_ops( cfg )
    
    input_batches = list( batches ) # [ inputs, targets, mask, data_idx ]
    max_steps = get_max_steps( num_samples_epoch, cfg , is_training=is_training)
    
    inputs[ 'enqueue_op' ] = enqueue_op
    inputs[ 'filepaths_list' ] = filepaths_list
    inputs[ 'load_and_enqueue' ] = load_and_enqueue
    inputs[ 'max_steps' ] = max_steps
    inputs[ 'num_samples_epoch' ] = num_samples_epoch

    inputs[ 'input_batches' ] = input_batches
    inputs[ 'input_batch' ] = input_batches[0]
    inputs[ 'target_batch' ] = input_batches[1]
    inputs[ 'mask_batch' ] = input_batches[2]
    inputs[ 'data_idxs' ] = input_batches[3]
    inputs[ 'placeholders' ] = placeholders
    inputs[ 'input_placeholder' ] = placeholders[0]
    inputs[ 'target_placeholder' ] = placeholders[1]
    inputs[ 'mask_placeholder' ] = placeholders[2]
    inputs[ 'data_idx_placeholder' ] = placeholders[3]
    return inputs

def setup_input( cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds input tensors from the config.
    '''
    inputs = {}
    if is_training:
        filepaths_list = get_filepaths_list( cfg[ 'train_filenames' ] )
    else: 
        filepaths_list = get_filepaths_list( cfg[ 'val_filenames' ] )


    # Generate placeholder input tensors
    num_samples_epoch = filepaths_list['total_size']
    inputs[ 'unit_size' ] = filepaths_list['unit_size']
    filepaths_list = [ os.path.join(cfg['root_dir'], cfg['meta_file_dir'], i) for i in filepaths_list['filename_list']]
    if use_filename_queue:
        filename_queue_dict = load_ops.create_filename_enqueue_op( cfg )
        inputs[ 'next_idx_op' ] = filename_queue_dict[ 'dequeue_op' ]
        inputs[ 'filename_enqueue_op' ] = filename_queue_dict[ 'enqueue_op' ]
        inputs[ 'next_idx_placeholder'] = filename_queue_dict[ 'data_idx_placeholder' ] 
        inputs[ 'fname_q' ] = filename_queue_dict
    placeholders, batches, load_and_enqueue, enqueue_op = create_input_placeholders_and_ops( cfg )
    
    input_batches = list( batches ) # [ inputs, targets, mask, data_idx ]
    max_steps = get_max_steps( num_samples_epoch, cfg , is_training=is_training)
    
    inputs[ 'enqueue_op' ] = enqueue_op
    inputs[ 'filepaths_list' ] = filepaths_list
    inputs[ 'load_and_enqueue' ] = load_and_enqueue
    inputs[ 'max_steps' ] = max_steps
    inputs[ 'num_samples_epoch' ] = num_samples_epoch

    inputs[ 'input_batches' ] = input_batches
    inputs[ 'input_batch' ] = input_batches[0]
    inputs[ 'target_batch' ] = input_batches[1]
    inputs[ 'mask_batch' ] = input_batches[2]
    inputs[ 'data_idxs' ] = input_batches[3]
    inputs[ 'placeholders' ] = placeholders
    inputs[ 'input_placeholder' ] = placeholders[0]
    inputs[ 'target_placeholder' ] = placeholders[1]
    inputs[ 'mask_placeholder' ] = placeholders[2]
    inputs[ 'data_idx_placeholder' ] = placeholders[3]
    return inputs

def setup_input_transfer(cfg, is_training=False, use_filename_queue=False):
    '''
        Builds input tensors from the config.
    '''
    inputs = {}
    if is_training:
        filepaths_list = get_filepaths_list( cfg[ 'train_filenames' ] )
        fileinfos_list_path = cfg[ 'train_list_of_fileinfos' ]
        representation_file = cfg['train_representations_file']
    else: 
        filepaths_list = get_filepaths_list( cfg[ 'val_filenames' ] )
        fileinfos_list_path = cfg[ 'val_list_of_fileinfos' ]
        representation_file = cfg['val_representations_file']

    if 'multiple_input_tasks' in cfg:
        num_samples_epoch = filepaths_list['total_size']
    else:
        with open(representation_file, 'rb')  as f:
            representations_list = pickle.load(f)
        data_used = len(np.load(fileinfos_list_path))
        num_samples_epoch = min([data_used, filepaths_list['total_size'], len(representations_list['file_indexes'] ) ])

    if 'data_used' in cfg:
        num_samples_epoch = min(num_samples_epoch, cfg['data_used'])

    inputs[ 'unit_size' ] = filepaths_list['unit_size']    
    # Generate placeholder input tensors
    filepaths_list = [ os.path.join(cfg['root_dir'], cfg['meta_file_dir'], i) for i in filepaths_list['filename_list']]
    if use_filename_queue:
        filename_queue_dict = load_ops.create_filename_enqueue_op( cfg )
        inputs[ 'next_idx_op' ] = filename_queue_dict[ 'dequeue_op' ]
        inputs[ 'filename_enqueue_op' ] = filename_queue_dict[ 'enqueue_op' ]
        inputs[ 'next_idx_placeholder'] = filename_queue_dict[ 'data_idx_placeholder' ] 
        inputs[ 'fname_q' ] = filename_queue_dict
    placeholders, batches, load_and_enqueue, enqueue_op = create_input_placeholders_and_ops( cfg )
    
    input_batches = list( batches ) # [ inputs, targets, mask, data_idx ]
    max_steps = get_max_steps( num_samples_epoch, cfg , is_training=is_training)
    
    inputs[ 'enqueue_op' ] = enqueue_op
    inputs[ 'filepaths_list' ] = filepaths_list
    inputs[ 'list_of_fileinfos' ] = fileinfos_list_path
    inputs[ 'representations_file' ] = representation_file
    inputs[ 'load_and_enqueue' ] = load_and_enqueue
    inputs[ 'max_steps' ] = max_steps
    inputs[ 'num_samples_epoch' ] = num_samples_epoch

    inputs[ 'input_batches' ] = input_batches
    inputs[ 'input_batch' ] = input_batches[0]
    inputs[ 'representation_batch' ] = input_batches[1]
    inputs[ 'target_batch' ] = input_batches[2]
    inputs[ 'mask_batch' ] = input_batches[3]
    inputs[ 'data_idxs' ] = input_batches[4]
    inputs[ 'placeholders' ] = placeholders
    inputs[ 'input_placeholder' ] = placeholders[0]
    inputs[ 'representation_placeholder' ] = placeholders[1]
    inputs[ 'target_placeholder' ] = placeholders[2]
    inputs[ 'mask_placeholder' ] = placeholders[3]
    inputs[ 'data_idx_placeholder' ] = placeholders[4]
    return inputs

def setup_input_transfer_imagenet(cfg, is_training=False, use_filename_queue=False):
    '''
        Builds input tensors from the config.
    '''
    inputs = {}
    if is_training:
        filepaths_list = get_filepaths_list( cfg[ 'train_filenames' ] )
        fileinfos_list_path = cfg[ 'train_list_of_fileinfos' ]
        representation_file = cfg['train_representations_file']
    else: 
        filepaths_list = get_filepaths_list( cfg[ 'val_filenames' ] )
        fileinfos_list_path = cfg[ 'val_list_of_fileinfos' ]
        representation_file = cfg['val_representations_file']


    num_samples_epoch = min(filepaths_list['total_size'], cfg['data_used'])

    inputs[ 'unit_size' ] = filepaths_list['unit_size']    
    # Generate placeholder input tensors
    filepaths_list = [ os.path.join(cfg['root_dir'], cfg['meta_file_dir'], i) for i in filepaths_list['filename_list']]
    if use_filename_queue:
        filename_queue_dict = load_ops.create_filename_enqueue_op( cfg )
        inputs[ 'next_idx_op' ] = filename_queue_dict[ 'dequeue_op' ]
        inputs[ 'filename_enqueue_op' ] = filename_queue_dict[ 'enqueue_op' ]
        inputs[ 'next_idx_placeholder'] = filename_queue_dict[ 'data_idx_placeholder' ] 
        inputs[ 'fname_q' ] = filename_queue_dict
    placeholders, batches, load_and_enqueue, enqueue_op = create_input_placeholders_and_ops( cfg )
    
    input_batches = list( batches ) # [ inputs, targets, mask, data_idx ]
    max_steps = get_max_steps( num_samples_epoch, cfg , is_training=is_training)
    
    inputs[ 'enqueue_op' ] = enqueue_op
    inputs[ 'filepaths_list' ] = filepaths_list
    inputs[ 'list_of_fileinfos' ] = fileinfos_list_path
    inputs[ 'representations_file' ] = representation_file
    inputs[ 'load_and_enqueue' ] = load_and_enqueue
    inputs[ 'max_steps' ] = max_steps
    inputs[ 'num_samples_epoch' ] = num_samples_epoch

    inputs[ 'input_batches' ] = input_batches
    inputs[ 'input_batch' ] = input_batches[0]
    inputs[ 'representation_batch' ] = input_batches[1]
    inputs[ 'target_batch' ] = input_batches[2]
    inputs[ 'mask_batch' ] = input_batches[3]
    inputs[ 'data_idxs' ] = input_batches[4]
    inputs[ 'placeholders' ] = placeholders
    inputs[ 'input_placeholder' ] = placeholders[0]
    inputs[ 'representation_placeholder' ] = placeholders[1]
    inputs[ 'target_placeholder' ] = placeholders[2]
    inputs[ 'mask_placeholder' ] = placeholders[3]
    inputs[ 'data_idx_placeholder' ] = placeholders[4]
    return inputs



def setup_model( inputs, cfg, is_training=False ):
    '''
        Sets up the `model` dict, and instantiates a model in 'model',
        and then calls model['model'].build

        Args:
            inputs: A dict, the result of setup_inputs
            cfg: A dict from config.py
            is_training: Bool, used for batch norm and the like

        Returns:
            model: A dict with 'model': cfg['model_type']( cfg ), and other
                useful attributes like 'global_step'
    '''
    validate_model( inputs, cfg )
    model = {}
    model[ 'global_step' ] = slim.get_or_create_global_step()

    model[ 'input_batch' ] = tf.identity( inputs[ 'input_batch' ] )
    if 'representation_batch' in inputs:
        model[ 'representation_batch' ] = tf.identity( inputs[ 'representation_batch' ] )
    model[ 'target_batch' ] = tf.identity( inputs[ 'target_batch' ] )
    model[ 'mask_batch' ] = tf.identity( inputs[ 'mask_batch' ] )
    model[ 'data_idxs' ] = tf.identity( inputs[ 'data_idxs' ] )

    # instantiate the model
    if cfg[ 'model_type' ] == 'empty':
        return model
    else:
        model[ 'model' ] = cfg[ 'model_type' ]( global_step=model[ 'global_step' ], cfg=cfg )

    # build the model
    if 'representation_batch' in inputs:
        input_imgs = (inputs[ 'input_batch' ], inputs[ 'representation_batch' ])
    else:
        input_imgs = inputs[ 'input_batch' ]
    model[ 'model' ].build_model( 
            input_imgs=input_imgs,
            targets=inputs[ 'target_batch' ],
            masks=inputs[ 'mask_batch' ],
            is_training=is_training )
    
    if is_training:
        model[ 'model' ].build_train_op( global_step=model[ 'global_step' ] )
        model[ 'train_op' ] = model[ 'model' ].train_op
        model[ 'train_step_fn' ] = model[ 'model' ].get_train_step_fn()
        model[ 'train_step_kwargs' ] = train_steps.get_default_train_step_kwargs( 
            global_step=model[ 'global_step' ],
            max_steps=inputs[ 'max_steps' ],
            log_every_n_steps=10 )

    #model[ 'init_op' ] = model[ 'model' ].init_op
    if hasattr( model['model'], 'init_fn' ):
        model[ 'init_fn' ] = model['model'].init_fn
    else:
        model[ 'init_fn' ] = None

    max_to_keep = cfg['num_epochs'] * 2
    if 'max_ckpts_to_keep' in cfg:
        max_to_keep = cfg['max_ckpts_to_keep']
    model[ 'saver_op' ] = AwsSaver(max_to_keep=max_to_keep)
    return model


def setup_model_chained_transfer( inputs, cfg, is_training=False ):
    '''
        Sets up the `model` dict, and instantiates a model in 'model',
        and then calls model['model'].build

        Args:
            inputs: A dict, the result of setup_inputs
            cfg: A dict from config.py
            is_training: Bool, used for batch norm and the like

        Returns:
            model: A dict with 'model': cfg['model_type']( cfg ), and other
                useful attributes like 'global_step'
    '''
    validate_model( inputs, cfg )
    model = {}
    model[ 'global_step' ] = 0 

    model[ 'input_batch' ] = tf.identity( inputs[ 'input_batch' ] )
    if 'representation_batch' in inputs:
        model[ 'representation_batch' ] = tf.identity( inputs[ 'representation_batch' ] )
    model[ 'target_batch' ] = [tf.identity( op ) for op in inputs[ 'target_batch' ]]
    model[ 'mask_batch' ] = [tf.identity( op ) for op in inputs[ 'mask_batch' ]]
    model[ 'data_idxs' ] = tf.identity( inputs[ 'data_idxs' ] )

    # instantiate the model
    if cfg[ 'model_type' ] == 'empty':
        return model
    else:
        model[ 'model' ] = cfg[ 'model_type' ]( global_step=model[ 'global_step' ], cfg=cfg )

    # build the model
    if 'representation_batch' in inputs:
        input_imgs = (inputs[ 'input_batch' ], inputs[ 'representation_batch' ])
    else:
        input_imgs = inputs[ 'input_batch' ]
    model[ 'model' ].build_model( 
            input_imgs=input_imgs,
            targets=inputs[ 'target_batch' ],
            masks=inputs[ 'mask_batch' ],
            is_training=is_training )
    
    model[ 'model' ].build_train_op( global_step=model[ 'global_step' ] )
    model[ 'train_op' ] = model[ 'model' ].train_op
    model[ 'train_step_fn' ] = model[ 'model' ].get_train_step_fn()
    model[ 'train_step_kwargs' ] = train_steps.get_default_train_step_kwargs( 
        global_step=model[ 'global_step' ],
        max_steps=inputs[ 'max_steps' ],
        log_every_n_steps=10 )

    #model[ 'init_op' ] = model[ 'model' ].init_op
    if hasattr( model['model'], 'init_fn' ):
        model[ 'init_fn' ] = model['model'].init_fn
    else:
        model[ 'init_fn' ] = None
    model[ 'saver_op' ] = AwsSaver(max_to_keep=cfg['num_epochs'])
    return model


def validate_model( inputs, cfg ):
    general_utils.validate_config( cfg )




################
# Data loading
################
# def get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=False ):
    # '''
        # Builds a function which, when called with (sess, supervisor), will
        # spin up a bunch of threads (exact # specified in cfg) that preload data.
        # That function returns all the threading.Thread's in a list.

    # ''' 
    # if use_filename_queue:
        # def data_prefetch_threads_init_fn( sess, supervisor ):
            # if 'num_input' in cfg and cfg['num_input'] and 'single_filename_to_multiple' not in cfg:
                # filename_load_function = load_ops.load_from_filename_queue_multiple
                # queue_filename = load_ops.enqueue_filenames_multiple
            # else:
                # filename_load_function = load_ops.load_from_filename_queue
                # queue_filename = load_ops.enqueue_filenames
            # print(filename_load_function)
            # print(queue_filename)
            # threads = [ 
                    # threading.Thread(  # add the data enqueueing threads
                            # target=filename_load_function,
                            # args=(sess, supervisor),
                            # kwargs={ 'input_filepaths': inputs[ 'filepaths_list' ],
                                        # 'input_placeholder': inputs[ 'input_placeholder' ], 
                                        # 'target_placeholder': inputs[ 'target_placeholder' ], 
                                        # 'mask_placeholder': inputs[ 'mask_placeholder' ], 
                                        # 'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                                        # 'rs_dim': cfg['input_dim'], 
                                        # 'enqueue_op': inputs[ 'enqueue_op' ], 
                                        # 'data_idx_dequeue_op': inputs[ 'next_idx_op' ],
                                        # 'is_training': is_training,
                                        # 'cfg': cfg } )
                    # for i in xrange( cfg['num_read_threads'] ) ]
            # threads.append( 
                # threading.Thread(  # the master thread to enqueue filenames
                            # target=queue_filename,
                            # args=(sess, supervisor),
                            # kwargs={ 
                                # 'input_filepaths': inputs[ 'filepaths_list' ], 
                                # 'data_idx_placeholder': inputs[ 'next_idx_placeholder' ], 
                                # 'enqueue_op': inputs[ 'filename_enqueue_op' ], 
                                # 'is_training': is_training, 
                                # 'cfg': cfg } ) )
            # for t in threads: t.start()
            # return threads                    
    # else: 
        # def data_prefetch_threads_init_fn( sess, supervisor ):
            # threads = [ 
                    # threading.Thread(
                            # target=inputs[ 'load_and_enqueue' ],
                            # args=(sess, supervisor),
                            # kwargs={ 'input_filepaths': inputs[ 'filepaths_list' ],
                                        # 'input_placeholder': inputs[ 'input_placeholder' ], 
                                        # 'target_placeholder': inputs[ 'target_placeholder' ], 
                                        # 'mask_placeholder': inputs[ 'mask_placeholder' ], 
                                        # 'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                                        # 'rs_dim': cfg['input_dim'], 
                                        # 'enqueue_op': inputs[ 'enqueue_op' ], 
                                        # 'is_training': is_training,
                                        # 'cfg': cfg } )
                    # for i in xrange( cfg['num_read_threads'] ) ]
            # for t in threads: t.start()
            # return threads
    # return data_prefetch_threads_init_fn

################
def get_data_prefetch_threads_init_fn( inputs, cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds a function which, when called with (sess, supervisor), will
        spin up a bunch of threads (exact # specified in cfg) that preload data.
        That function returns all the threading.Thread's in a list.

    ''' 
    if use_filename_queue:
        def data_prefetch_threads_init_fn( sess, supervisor ):
            if 'num_input' in cfg and cfg['num_input'] and 'single_filename_to_multiple' not in cfg:
                filename_load_function = load_ops.load_from_filename_queue_multiple
                queue_filename = load_ops.enqueue_filenames_multiple
            else:
                filename_load_function = load_ops.load_from_filename_queue
                queue_filename = load_ops.enqueue_filenames
            print(filename_load_function)
            print(queue_filename)
            threads = [ 
                    threading.Thread(  # add the data enqueueing threads
                            target=filename_load_function,
                            args=(sess, supervisor),
                            kwargs={ 'input_filepaths': inputs[ 'filepaths_list' ],
                                        'input_placeholder': inputs[ 'input_placeholder' ], 
                                        'target_placeholder': inputs[ 'target_placeholder' ], 
                                        'mask_placeholder': inputs[ 'mask_placeholder' ], 
                                        'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                                        'rs_dim': cfg['input_dim'], 
                                        'enqueue_op': inputs[ 'enqueue_op' ], 
                                        'data_idx_dequeue_op': inputs[ 'next_idx_op' ],
                                        'is_training': is_training,
                                        'cfg': cfg } )
                    for i in range( cfg['num_read_threads'] ) ]
            threads.append( 
                threading.Thread(  # the master thread to enqueue filenames
                            target=queue_filename,
                            args=(sess, supervisor),
                            kwargs={ 
                                'input_filepaths': inputs[ 'filepaths_list' ], 
                                'data_idx_placeholder': inputs[ 'next_idx_placeholder' ], 
                                'enqueue_op': inputs[ 'filename_enqueue_op' ], 
                                'is_training': is_training, 
                                'cfg': cfg } ) )
            for t in threads: t.start()
            return threads                    
    else: 
        def data_prefetch_threads_init_fn( sess, supervisor ):
            from functools import partial
            kwargs={    'sess': sess,
                        'supervisor': supervisor,
                        'input_filepaths': inputs[ 'filepaths_list' ],
                        'step': cfg['num_read_threads'],
                        'unit_size' : inputs['unit_size'],
                        'num_samples_epoch': inputs[ 'num_samples_epoch' ], 
                        'input_placeholder': inputs[ 'input_placeholder' ], 
                        'target_placeholder': inputs[ 'target_placeholder' ], 
                        'mask_placeholder': inputs[ 'mask_placeholder' ], 
                        'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                        'rs_dim': cfg['input_dim'], 
                        'enqueue_op': inputs[ 'enqueue_op' ], 
                        'is_training': is_training,
                        'cfg': cfg } 

            mapfunc = partial(inputs['load_and_enqueue'], **kwargs )
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['num_read_threads'] + 2) as executor:
                result = executor.map(mapfunc, range(cfg['num_read_threads']))
            
#             pool = GreenPool(size=cfg['num_read_threads'])
            # threads = []
            # for i in range( cfg['num_read_threads'] ):
                # kwargs={ 'input_filepaths': inputs[ 'filepaths_list' ],
                                        # 'seed': i,
                                        # 'step': cfg['num_read_threads'],
                                        # 'unit_size' : inputs['unit_size'],
                                        # 'num_samples_epoch': inputs[ 'num_samples_epoch' ], 
                                        # 'input_placeholder': inputs[ 'input_placeholder' ], 
                                        # 'target_placeholder': inputs[ 'target_placeholder' ], 
                                        # 'mask_placeholder': inputs[ 'mask_placeholder' ], 
                                        # 'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                                        # 'rs_dim': cfg['input_dim'], 
                                        # 'enqueue_op': inputs[ 'enqueue_op' ], 
                                        # 'is_training': is_training,
                                        # 'cfg': cfg } 
                # args = (sess, supervisor)
                # threads.append(pool.spawn(inputs[ 'load_and_enqueue' ], *args, **kwargs))
            # pool.waitall()

#             threads = [ 
                    # threading.Thread(
                            # target=inputs[ 'load_and_enqueue' ],
                            # args=(sess, supervisor),
                            # kwargs={ 'input_filepaths': inputs[ 'filepaths_list' ],
                                        # 'seed': i,
                                        # 'step': cfg['num_read_threads'],
                                        # 'unit_size' : inputs['unit_size'],
                                        # 'num_samples_epoch': inputs[ 'num_samples_epoch' ], 
                                        # 'input_placeholder': inputs[ 'input_placeholder' ], 
                                        # 'target_placeholder': inputs[ 'target_placeholder' ], 
                                        # 'mask_placeholder': inputs[ 'mask_placeholder' ], 
                                        # 'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                                        # 'rs_dim': cfg['input_dim'], 
                                        # 'enqueue_op': inputs[ 'enqueue_op' ], 
                                        # 'is_training': is_training,
                                        # 'cfg': cfg } )
                    # for i in range( cfg['num_read_threads'] ) ]
            # for t in threads: t.start()
            return  result

    return data_prefetch_threads_init_fn

def end_data_loading_and_sess( training_runners ):
    """ Run after request_training_end """
    #training_runners[ 'coord' ].join( training_runners[ 'threads' ] )
    print('joined threads and done training! :)')
    training_runners[ 'sess' ].close()


def request_data_loading_end( training_runners ):
    """ Run after start_data_prefetch_threads """
    print('Requesting coordinator to stop.')
    training_runners[ 'coord' ].request_stop()




##################################
#   Transferring Input workers   #
def load_filepaths_list( filenames_filepath ):
    """
        Reads in the list of filepaths from the given fname

        Args:
            fname: A path to a file containing a list of filepaths.
                May be pickled or json.

        Returns:
            A List of filenames 
    """
    ext = os.path.splitext( filenames_filepath )[1]
    if ext == '.json':
        with open( filenames_filepath, 'r' ) as fp:
            data_sources = json.load( fp )
    elif ext == '.npy':
        with open( filenames_filepath, 'rb' ) as fp:
            data_sources = np.load( fp )
    else:
        with open( filenames_filepath, 'rb' ) as fp:
            data_sources = pickle.load( fp )
    return data_sources

def get_data_prefetch_threads_init_fn_transfer( inputs, cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds a function which, when called with (sess, supervisor), will
        spin up a bunch of threads (exact # specified in cfg) that preload data.
        That function returns all the threading.Thread's in a list.

    ''' 
    ##############################################################
    #   For kwargs, add representation file as additional input  #
    def data_prefetch_threads_init_fn( sess, supervisor ):
        from functools import partial
        import pickle

        if 'multiple_input_tasks' in cfg:
            with open(inputs['list_of_fileinfos'], 'rb') as f:
                print(inputs['list_of_fileinfos'])
                fileinfos = np.load(inputs['list_of_fileinfos'])
                fileinfo_to_fileinfo_idx = {str(fi.decode('utf-8')): i for i, fi in enumerate(fileinfos)}
            
            fileinfo_to_task_to_representation = {}
            for rep_file_path in inputs['representations_file']:
                # cfg['multiple_input_tasks']:
                print(rep_file_path)
        
                # with open(inputs['representations_file'].format(task=task), 'rb') as f:
                with open(rep_file_path, 'rb') as f:
                    representations = pickle.load(f)
                representation_idx_to_fileinfo_idx = representations['file_indexes']
                representations = representations['representations']
                fileinfo_to_task_to_representation[rep_file_path] = {}

                for representation_idx, fileinfo_idx in enumerate(representation_idx_to_fileinfo_idx):
                    if fileinfo_idx < len(fileinfos):
                        fileinfo_to_task_to_representation[rep_file_path][fileinfos[fileinfo_idx].decode('utf-8')] = representations[representation_idx]

            print("Prepared mapping from fileinfo to representation.")
            # Define mapping fn
            print("----------------------------------")
            print('filenames len:', len(fileinfos))
            print("----------------------------------")

            def fileinfo_to_representation_fn(fileinfo):
                assert type(fileinfo) is str
                #if task == 'RANDOM':
                #    task = random.choice(inputs['representations_file'])
                # assert fileinfo in fileinfo_to_task_to_representation[task]
                list_of_rep = []
                for t in inputs['representations_file']:
                    rep = fileinfo_to_task_to_representation[t][fileinfo] 
                    if 16 not in rep.shape:
                        rep = np.reshape(rep, (16,16,-1))

                    list_of_rep.append(rep)
                return np.concatenate(list_of_rep, axis=-1)
        else:
            print("Preparing representations...")
            # Load the representations and find the mapping of fileinfos reps
            with open(inputs['representations_file'], 'rb') as f:
                print(inputs['representations_file'])
                representations = pickle.load(f)
                representation_idx_to_fileinfo_idx = representations['file_indexes']
                representations = representations['representations']
            
            with open(inputs['list_of_fileinfos'], 'rb') as f:
                print(inputs['list_of_fileinfos'])
                fileinfos = np.load(inputs['list_of_fileinfos'])
                fileinfo_to_fileinfo_idx = {str(fi.decode('utf-8')): i for i, fi in enumerate(fileinfos)}
            
            
            fileinfo_to_representation = {}
            for representation_idx, fileinfo_idx in enumerate(representation_idx_to_fileinfo_idx):
                if fileinfo_idx < len(fileinfos):
                    fileinfo_to_representation[fileinfos[fileinfo_idx].decode('utf-8')] = representations[representation_idx]
            print("Prepared mapping from fileinfo to representation.")
            # Define mapping fn
            print("----------------------------------")
            print('filenames len:', len(fileinfos))
            print('reps len:', len(representations))
            print("n uniq reps:", len(fileinfo_to_representation))
            print("----------------------------------")

            def fileinfo_to_representation_fn(fileinfo):
                assert type(fileinfo) is str
                return fileinfo_to_representation[fileinfo] 
            
        if len(inputs[ 'filepaths_list' ]) > 1:
            tf.logging.error('Race condition when more than 1 data shard used. Using only first shard.')
            inputs[ 'filepaths_list' ] = [inputs[ 'filepaths_list' ][0]]
            inputs[ 'num_samples_epoch' ]  = inputs['unit_size']

        assert len(inputs[ 'filepaths_list' ]) == 1, "Race condition when more than 1 data shard used."
        # Rewrite thins and 
        curr_filename_list = load_filepaths_list( inputs[ 'filepaths_list' ][0] )
        if 'data_used' in cfg:
            curr_filename_list = curr_filename_list[:cfg['data_used']]
        print(inputs['load_and_enqueue'])
        kwargs={    'sess': sess,
                    'supervisor': supervisor,
                    'input_filepaths': inputs[ 'filepaths_list' ],
                    'step': cfg['num_read_threads'],
                    'unit_size' : inputs['unit_size'],
                    'num_samples_epoch': inputs[ 'num_samples_epoch' ], 
                    'input_placeholder': inputs[ 'input_placeholder' ], 
                    'representation_placeholder': inputs[ 'representation_placeholder' ], 
                    'target_placeholder': inputs[ 'target_placeholder' ], 
                    'mask_placeholder': inputs[ 'mask_placeholder' ], 
                    'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                    'rs_dim': cfg['input_dim'], 
                    'representations': representations, 
                    'enqueue_op': inputs[ 'enqueue_op' ], 
                    'is_training': is_training,
                    'cfg': cfg,
                    'fileinfo_to_representation_fn': fileinfo_to_representation_fn,
                    'curr_filename_list': curr_filename_list
                    } 

        mapfunc = partial(inputs['load_and_enqueue'], **kwargs )
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['num_read_threads'] + 2) as executor:
            result = executor.map(mapfunc, range(cfg['num_read_threads']))

        return  result
    return data_prefetch_threads_init_fn


def get_data_prefetch_threads_init_fn_transfer_imagenet( inputs, cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds a function which, when called with (sess, supervisor), will
        spin up a bunch of threads (exact # specified in cfg) that preload data.
        That function returns all the threading.Thread's in a list.

    ''' 
    ##############################################################
    #   For kwargs, add representation file as additional input  #
    def data_prefetch_threads_init_fn( sess, supervisor ):
        from functools import partial
        import pickle

        def fileinfo_to_representation_fn(fileinfo):
            return np.ones(cfg['representation_dim'], dtype=float)  

            
        if len(inputs[ 'filepaths_list' ]) > 1:
            tf.logging.error('Race condition when more than 1 data shard used. Using only first shard.')
            inputs[ 'filepaths_list' ] = [inputs[ 'filepaths_list' ][0]]
        assert len(inputs[ 'filepaths_list' ]) == 1, "Race condition when more than 1 data shard used."
        # Rewrite thins and 
        curr_filename_list = load_filepaths_list( inputs[ 'filepaths_list' ][0] )
        curr_filename_list = curr_filename_list[:cfg['data_used']]
        print(inputs['load_and_enqueue'])
        kwargs={    'sess': sess,
                    'supervisor': supervisor,
                    'input_filepaths': inputs[ 'filepaths_list' ],
                    'step': cfg['num_read_threads'],
                    'unit_size' : inputs['unit_size'],
                    'num_samples_epoch': inputs[ 'num_samples_epoch' ], 
                    'input_placeholder': inputs[ 'input_placeholder' ], 
                    'representation_placeholder': inputs[ 'representation_placeholder' ], 
                    'target_placeholder': inputs[ 'target_placeholder' ], 
                    'mask_placeholder': inputs[ 'mask_placeholder' ], 
                    'data_idx_placeholder': inputs[ 'data_idx_placeholder' ],
                    'rs_dim': cfg['input_dim'], 
                    'representations': None, 
                    'enqueue_op': inputs[ 'enqueue_op' ], 
                    'is_training': is_training,
                    'cfg': cfg,
                    'fileinfo_to_representation_fn': fileinfo_to_representation_fn,
                    'curr_filename_list': curr_filename_list
                    } 

        mapfunc = partial(inputs['load_and_enqueue'], **kwargs )
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['num_read_threads'] + 2) as executor:
            result = executor.map(mapfunc, range(cfg['num_read_threads']))

        return  result

    return data_prefetch_threads_init_fn



