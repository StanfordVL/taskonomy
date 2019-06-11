'''
  Name: task_data_loading.py

  Desc: Loads inputs and targets for different tasks
'''
from __future__ import absolute_import, division, print_function

from   data.load_ops import *
import numpy as np
from   PIL import Image
import skimage
import tensorflow as tf
import traceback as tb
import pdb

DEFAULT_INTERPOLATION_ORDER = 1
USE_PIL = False 

def load_and_preprocess_img( filename_template, cfg, is_training=False ):
    """
        Args:
            filename_template: the filename, with {domain} where the 
                domain should be interpolated
            cfg: A config.py dict
        Returns:
            input_img: scaled between [0, 1]
            target_img: scaled between [0, 1]
    """
    if 'resize_interpolation_order' not in cfg:
        cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER
    target_mask = 1.0 
    if 'num_input' in cfg and cfg['num_input'] > 1:
        input_img = np.empty((cfg['num_input'], 
                                cfg['input_dim'][0], 
                                cfg['input_dim'][1], 
                                cfg['input_num_channels']), dtype=np.float32)
        for i in range(len( filename_template )):
            img = load_raw_image( 
                    make_filename_for_domain( filename_template[i], cfg['input_domain_name'] ), 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
            # process input
            img = resize_image( img, cfg[ 'input_dim'], interp_order=cfg['resize_interpolation_order'] )
            if cfg['input_domain_name'] == 'depth':
                img = np.log( 1 + img ) / ( np.log( 2. ** 16.0 ) ) # FOR DEPTH ONLY     
            input_img[i,:,:,:] = img
    else:
        input_img = load_raw_image( 
                make_filename_for_domain( filename_template, cfg['input_domain_name'] ), 
                color=( cfg['input_num_channels'] >= 3 ) )
        # process input
        input_img = resize_image( input_img, cfg[ 'input_dim'], interp_order=cfg['resize_interpolation_order'] )
        if cfg['input_domain_name'] == 'depth': 
            input_img = np.log( 1 + input_img ) / ( np.log( 2. ** 16.0 ) ) # FOR DEPTH ONLY

    target_img, target_mask = load_target(filename_template, input_img, cfg)
  
    return input_img, target_img, target_mask    

def load_and_specify_preprocessors_for_input_depends_on_target( filename, cfg, is_training=False ):
    '''
        Applies config.py specified preprocessing functions to images that will be fed to network
        Note that this is designed for camera pose for which the filename corresponds to a pair of images
        
        Args:
            filename_template: the filename, with {domain} where the 
                domain will be replaced with domain_name from cfg
            cfg:  A config.py dict. Should contain
                'input_preprocessing_fn', 'input_preprocessing_fn_kwargs', 'input_dim', 'input_num_channels', 'input_domain_name'
                'target_preprocessing_fn', 'target_preprocessing_fn_kwargs', 'target_dim', 'target_num_channels', 'target_domain_name'
        Returns:
            input_img: cfg[ 'input_preprocessing_fn' ]( raw_input_img, cfg['input_preprocessing_fn_kwargs'] )
            target_img: cfg[ 'target_preprocessing_fn' ]( raw_target_img, cfg['target_preprocessing_fn_kwargs'] )
            target_mask: cfg[ 'mask_fn' ]( img=target_img, cfg[ 'mask_fn_kwargs' ] )
    '''
    if 'resize_interpolation_order' not in cfg:
        cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER

    target_img, target_mask = load_target(filename, None, cfg)

    # target generated, to generate input based on target
    if 'find_target_in_config' in cfg and cfg['find_target_in_config']:
        if 'target_dict' not in cfg:
            raise ValueError("Config for task that need to generate target from config, must provide a dictionary for potential targets")
        
        target_arg_for_input = cfg['target_dict'][target_img]
    else:
        target_arg_for_input = target_img
    
    if 'num_input' in cfg and cfg['num_input'] > 1:
        input_img = np.empty((cfg['num_input'], 
                                cfg['input_dim'][0], 
                                cfg['input_dim'][1], 
                                cfg['input_num_channels']), dtype=np.float32)

        filename_template = make_image_filenames(filename, cfg['num_input'])

        if type(filename_template) == list:
            filename_template = filename_template[0]

        if type(filename_template) == list:
            for i in range(len( filename_template )):
                img = load_raw_image( 
                        filename_template[i], 
                        color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
                # process input
                if 'find_target_in_config' in cfg and cfg['find_target_in_config']:
                    img = cfg[ 'input_preprocessing_fn' ]( img, target=target_arg_for_input, **cfg['input_preprocessing_fn_kwargs'] )
                else:
                    img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )

                input_img[i,:,:,:] = img

        else:
            # this means that there is only one filename, the task need to generate num_input inputs from single image
            img = load_raw_image( 
                    filename_template, 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
            if 'find_target_in_config' in cfg and cfg['find_target_in_config']:
                input_img = cfg[ 'input_preprocessing_fn' ]( img, target=target_arg_for_input, **cfg['input_preprocessing_fn_kwargs'] )
            else:
                input_img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )     
    else:
        #inputs
        input_img = load_raw_image( 
                make_filename_for_domain( filename, cfg['input_domain_name'] ), 
                color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb' ) )
        # inputs
        input_img = cfg[ 'input_preprocessing_fn' ]( input_img, **cfg['input_preprocessing_fn_kwargs'] )

    return input_img, target_img, target_mask 



def load_and_specify_preprocessors_for_single_filename_to_imgs( filename, cfg, is_training=False ):
    '''
        Applies config.py specified preprocessing functions to images that will be fed to network
        Note that this is designed for camera pose for which the filename corresponds to a pair of images
        
        Args:
            filename_template: the filename, with {domain} where the 
                domain will be replaced with domain_name from cfg
            cfg:  A config.py dict. Should contain
                'input_preprocessing_fn', 'input_preprocessing_fn_kwargs', 'input_dim', 'input_num_channels', 'input_domain_name'
                'target_preprocessing_fn', 'target_preprocessing_fn_kwargs', 'target_dim', 'target_num_channels', 'target_domain_name'
        Returns:
            input_img: cfg[ 'input_preprocessing_fn' ]( raw_input_img, cfg['input_preprocessing_fn_kwargs'] )
            target_img: cfg[ 'target_preprocessing_fn' ]( raw_target_img, cfg['target_preprocessing_fn_kwargs'] )
            target_mask: cfg[ 'mask_fn' ]( img=target_img, cfg[ 'mask_fn_kwargs' ] )
    '''
    if 'resize_interpolation_order' not in cfg:
        cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER

    if 'num_input' in cfg and cfg['num_input'] > 1:
        input_img = np.empty((cfg['num_input'], 
                                cfg['input_dim'][0], 
                                cfg['input_dim'][1], 
                                cfg['input_num_channels']), dtype=np.float32)

        filename_template = make_image_filenames(filename, cfg['num_input'])

        for i in range(len( filename_template )):
            img = load_raw_image( 
                    filename_template[i], 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
            # process input
            img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
            input_img[i,:,:,:] = img
    else:
        #inputs
        input_img = load_raw_image( 
                make_filename_for_domain( filename, cfg['input_domain_name'] ), 
                color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb' ) )
        # inputs
        input_img = cfg[ 'input_preprocessing_fn' ]( input_img, **cfg['input_preprocessing_fn_kwargs'] )
    # print(filename)
    target_img, target_mask = load_target(filename, input_img, cfg)

    return input_img, target_img, target_mask 

def load_and_specify_preprocessors( filename_template, cfg, is_training=False ):
    '''
        Applies config.py specified preprocessing functions to images that will be fed to network
        
        Args:
            filename_template: the filename, with {domain} where the 
                domain will be replaced with domain_name from cfg
            cfg:  A config.py dict. Should contain
                'input_preprocessing_fn', 'input_preprocessing_fn_kwargs', 'input_dim', 'input_num_channels', 'input_domain_name'
                'target_preprocessing_fn', 'target_preprocessing_fn_kwargs', 'target_dim', 'target_num_channels', 'target_domain_name'
        Returns:
            input_img: cfg[ 'input_preprocessing_fn' ]( raw_input_img, cfg['input_preprocessing_fn_kwargs'] )
            target_img: cfg[ 'target_preprocessing_fn' ]( raw_target_img, cfg['target_preprocessing_fn_kwargs'] )
            target_mask: cfg[ 'mask_fn' ]( img=target_img, cfg[ 'mask_fn_kwargs' ] )
    '''
    try:
        if 'resize_interpolation_order' not in cfg:
            cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER
        
        # def load_input():
        if 'num_input' in cfg and cfg['num_input'] > 1:
            input_img = np.empty((cfg['num_input'], 
                                    cfg['input_dim'][0], 
                                    cfg['input_dim'][1], 
                                    cfg['input_num_channels']), dtype=np.float32)
            for i in range(len( filename_template )):
                img = load_raw_image( 
                        make_filename_for_domain( filename_template[i], cfg['input_domain_name'] ), 
                        color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
                # process input
                img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
                input_img[i,...] = img
        else:
            #inputs
            input_img = load_raw_image( 
                    make_filename_for_domain( filename_template, cfg['input_domain_name'] ), 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb' ),
                    use_pil=USE_PIL)
            # inputs
            input_img = cfg[ 'input_preprocessing_fn' ]( input_img, **cfg['input_preprocessing_fn_kwargs'] )
        # return input_img
        
        # def load_target_and_mask()
        target_img, target_mask = load_target(filename_template, input_img, cfg, use_pil=USE_PIL)
        return input_img, target_img, target_mask    
    except:
        print(tb.print_exc())
        raise


def load_and_specify_preprocessors_multiple( filename_template, cfg, is_training=False ):
    '''
        Applies config.py specified preprocessing functions to images that will be fed to network
        
        Args:
            filename_template: the filename, with {domain} where the 
                domain will be replaced with domain_name from cfg
            cfg:  A config.py dict. Should contain
                'input_preprocessing_fn', 'input_preprocessing_fn_kwargs', 'input_dim', 'input_num_channels', 'input_domain_name'
                'target_preprocessing_fn', 'target_preprocessing_fn_kwargs', 'target_dim', 'target_num_channels', 'target_domain_name'
        Returns:
            input_img: cfg[ 'input_preprocessing_fn' ]( raw_input_img, cfg['input_preprocessing_fn_kwargs'] )
            target_img: cfg[ 'target_preprocessing_fn' ]( raw_target_img, cfg['target_preprocessing_fn_kwargs'] )
            target_mask: cfg[ 'mask_fn' ]( img=target_img, cfg[ 'mask_fn_kwargs' ] )
    '''
    try:
        if 'resize_interpolation_order' not in cfg:
            cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER

        if 'num_input' in cfg and cfg['num_input'] > 1:
            input_img = np.empty((cfg['num_input'], 
                                    cfg['input_dim'][0], 
                                    cfg['input_dim'][1], 
                                    cfg['input_num_channels']), dtype=np.float32)
            for i in range(len( filename_template )):
                img = load_raw_image( 
                        make_filename_for_domain( filename_template[i], cfg['input_domain_name'] ), 
                        color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
                # process input
                img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
                input_img[i,:,:,:] = img
        else:
            #inputs
            input_img = load_raw_image( 
                    make_filename_for_domain( filename_template, cfg['input_domain_name'] ), 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb' ) )
            # inputs
            input_img = cfg[ 'input_preprocessing_fn' ]( input_img, **cfg['input_preprocessing_fn_kwargs'] )

        target_masks = []
        target_imgs = []
        for c in cfg['target_cfgs']:
            target_img, target_mask = load_target(filename_template, input_img, c)
            target_imgs.append(target_img)
            target_masks.append(target_mask)

        return input_img, target_imgs, target_masks    
    except:
        print(tb.print_exc())
        raise

def load_and_specify_preprocessors_for_representation_extraction( filename, cfg, is_training=False ):
    '''
        Applies config.py specified preprocessing functions to images that will be fed to network
        Note that this is designed for camera pose for which the filename corresponds to a pair of images
        
        Args:
            filename_template: the filename, with {domain} where the 
                domain will be replaced with domain_name from cfg
            cfg:  A config.py dict. Should contain
                'input_preprocessing_fn', 'input_preprocessing_fn_kwargs', 'input_dim', 'input_num_channels', 'input_domain_name'
                'target_preprocessing_fn', 'target_preprocessing_fn_kwargs', 'target_dim', 'target_num_channels', 'target_domain_name'
        Returns:
            input_img: cfg[ 'input_preprocessing_fn' ]( raw_input_img, cfg['input_preprocessing_fn_kwargs'] )
            target_img: cfg[ 'target_preprocessing_fn' ]( raw_target_img, cfg['target_preprocessing_fn_kwargs'] )
            target_mask: cfg[ 'mask_fn' ]( img=target_img, cfg[ 'mask_fn_kwargs' ] )
    '''
    if 'resize_interpolation_order' not in cfg:
        cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER

    if 'num_input' in cfg and cfg['num_input'] > 1:
        input_img = np.empty((cfg['num_input'], 
                                cfg['input_dim'][0], 
                                cfg['input_dim'][1], 
                                cfg['input_num_channels']), dtype=np.float32)

        filename_template = make_image_filenames(filename, cfg['num_input'])
        for i in range(len( filename_template )):
            img = load_raw_image( 
                    filename_template[i], 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
            # process input
            img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
            input_img[i,:,:,:] = img
        # img = load_raw_image( 
        #         filename_template[0], 
        #         color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
        # # process input
        # img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
        # for i in range(len( filename_template )):
        #     input_img[i,:,:,:] = img
    else:
        #inputs
        input_img = load_raw_image( 
                make_filename_for_domain( filename, cfg['input_domain_name'] ), 
                color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb' ) )
        # inputs
        input_img = cfg[ 'input_preprocessing_fn' ]( input_img, **cfg['input_preprocessing_fn_kwargs'] )

    target_img, target_mask = load_random_target( cfg )
    return input_img, target_img, target_mask 

def load_random_target( cfg ):
    if cfg['target_dtype'] is tf.float32:
        dtype = np.float32
    elif cfg['target_dtype'] is tf.float64:
        dtype = np.float64
    elif cfg['target_dtype'] is tf.int32:
        dtype = np.int32
    else:
        dtype = np.int64

    if 'is_discriminative' in cfg or 'only_target_discriminative' in cfg and cfg['only_target_discriminative']:
        if type(cfg['target_dim']) is int:
            if cfg['target_dim'] > 1:
                target_img = np.ones(( cfg['target_dim'] ), dtype=dtype)
            else:
                target_img = 0
        else:
            target_img = np.ones(( cfg['target_dim'][0], cfg['target_dim'][1] ), dtype=dtype)
        mask_shape = []
    else:
        target_img = np.ones(( cfg['target_dim'][0], cfg['target_dim'][1], cfg['target_num_channels'] ), dtype=dtype)
        mask_shape = ( cfg['target_dim'][0], cfg['target_dim'][1], cfg['target_num_channels'] )

    if 'mask_fn' in cfg or 'depth_mask' in cfg and cfg['depth_mask']:
        if 'only_target_discriminative' in cfg:
            #mask_shape = [ cfg['output_dim'][0], cfg['output_dim'][1], cfg['target_num_channels'] ]
            mask_shape = []
    elif 'mask_by_target_func' in cfg and cfg['mask_by_target_func']:
        if type(cfg['target_dim']) == int:
            mask_shape = [1]
        else:
            mask_shape = [ cfg['target_dim'][0], cfg['target_dim'][1] ]
    else:
        mask_shape = []

    if mask_shape == []:
        mask = 1.0
    else:
        mask = np.ones(mask_shape, dtype=np.float32)
    return target_img, mask


def load_target( filename_template, input_img, cfg, use_pil=False ):
    '''
        Applies config.py specified preprocessing functions to target that will be fed to network
        
        Args:
            filename_template: the filename, with {domain} where the 
                domain will be replaced with domain_name from cfg
            cfg:  A config.py dict. Should contain
                'target_preprocessing_fn', 'target_preprocessing_fn_kwargs', 'target_dim', 'target_num_channels', 'target_domain_name'
                for discriminative target, cfg needs to contain 'target_from_filenames'
        Returns:
            target_img: cfg[ 'target_preprocessing_fn' ]( raw_target_img, cfg['target_preprocessing_fn_kwargs'] )
            target_mask: cfg[ 'mask_fn' ]( img=target_img, cfg[ 'mask_fn_kwargs' ] )
    '''
    if 'resize_interpolation_order' not in cfg:
        cfg['resize_interpolation_order'] = DEFAULT_INTERPOLATION_ORDER
    target_mask = 1.0
    if 'is_discriminative' in cfg or 'only_target_discriminative' in cfg and cfg['only_target_discriminative']:
        if 'target_from_filenames' not in cfg:
            raise ValueError("Config for discriminative task must provide a function that takes in two filenames and compute the target as output")
        if 'target_from_filenames_kwargs' not in cfg:
            target_func_kwargs = {}
        else:
            target_func_kwargs = cfg['target_from_filenames_kwargs']
        if 'depth_mask' in cfg and cfg['depth_mask']:
            depth_values = load_raw_image( 
                make_filename_for_domain( filename_template, 'depth' ), 
                color=False,
                use_pil=use_pil)
            cfg['mask_fn'] = mask_if_channel_ge # given target image as input
            cfg['mask_fn_kwargs'] = {
                    'img': '<TARGET_IMG>',
                    'channel_idx': 0,
                    'threshhold': 64500, 
                    'broadcast_to_dim': cfg['target_num_channels']
            }
            temp_mask = make_mask( input_img, depth_values, cfg, mask_dim= cfg['output_dim'] )
            if 'target_depend_on_mask' in cfg:
                target_func_kwargs['mask'] = temp_mask

        if 'mask_by_target_func' in cfg and cfg['mask_by_target_func']:
            target_img, target_mask = cfg['target_from_filenames'](filename_template, **target_func_kwargs)  
        else:
            target_img = cfg['target_from_filenames'](filename_template, **target_func_kwargs)  
    else:
        # apply mask to raw target img
        target_img = load_raw_image( 
                make_filename_for_domain( filename_template, cfg['target_domain_name'] ), 
                color=( cfg['target_num_channels'] >= 3 or cfg['target_domain_name'] == 'curvature' or cfg['target_domain_name'] == 'rgb'),
                use_pil=use_pil )
    
        #target_img = resize_image( target_img, cfg['target_dim'], interp_order=cfg['resize_interpolation_order'] ) # Resize along with target
        # apply mask to raw img
        if 'depth_mask' in cfg and cfg['depth_mask']:
            depth_values = load_raw_image( 
                make_filename_for_domain( filename_template, 'depth' ), 
                color=False,
                use_pil=use_pil)
            cfg['mask_fn'] = mask_if_channel_ge # given target image as input
            cfg['mask_fn_kwargs'] = {
                    'img': '<TARGET_IMG>',
                    'channel_idx': 0,
                    'threshhold': 64500, 
                    'broadcast_to_dim': cfg['target_num_channels']
            }
            target_mask = make_mask( input_img, depth_values, cfg )
        else:        
            target_mask = make_mask( input_img, target_img, cfg )

        # targets
        if 'mask_by_target_func' in cfg and cfg['mask_by_target_func']:
            target_img, target_mask = cfg[ 'target_preprocessing_fn' ]( target_img, **cfg['target_preprocessing_fn_kwargs'] )
        else:
            target_img = cfg[ 'target_preprocessing_fn' ]( target_img, **cfg['target_preprocessing_fn_kwargs'] )
    

    return target_img, target_mask


def load_and_preprocess_img_fast( filename_template, cfg, is_training=False ):
    """
        Slightly worse quality images than the one above, but roughly twice as fast
            due to the use of PIL instead of Skimage.
        Args:
            filename_template: the filename, with {domain} where the 
                domain should be interpolated
            cfg: A config.py dict
        Returns:
            input_img: scaled between [0, 1]
            target_img: scaled between [0, 1]
    """
    if 'resize_interpolation_order' not in cfg:
        cfg['resize_interpolation_order'] =  Image.NEAREST
    
    if 'is_discriminative' in cfg:
        raise ValueError("Using 'load_and_preprocess_img_fast' for discriminative task is not adviced, this function hasn't been updated for discriminative tasks")
    # inputs
    input_img = Image.open(make_filename_for_domain( 
        filename_template, cfg['input_domain_name'] ) )

    # targets
    target_img = Image.open(make_filename_for_domain( 
        filename_template, cfg['target_domain_name'] ) )

 
    input_img = input_img.resize( cfg[ 'input_dim'], Image.NEAREST ) # cfg['resize_interpolation_order']
    input_img = rescale_image( np.array( input_img ), [-1, 1] )
    if len( input_img.shape )== 2:
        input_img = input_img[:,:,np.newaxis] 
    
    # apply mask to raw target img
    target_mask = make_mask( input_img, target_img, cfg )

    # process target
    target_img_np = np.array(target_img)
    if len( targest_img_np.shape ) == 2:
        target_img_np = target_img_np[:,:,np.newaxis]

    target_img = target_img.resize( cfg[ 'target_dim' ], Image.NEAREST )
    target_img = rescale_image( np.array( target_img ), [-1, 1] )
    if len( target_img.shape ) == 2:
        target_img = target_img[:,:,np.newaxis] 
    # print( "Loaded", filename_template )
    return input_img, target_img, target_mask    


def load_only_raw_images( filename_template, cfg, is_training=False ):
    if 'num_input' in cfg and cfg['num_input'] > 1:
        input_img = np.empty((cfg['num_input'], 
                                cfg['input_dim'][0], 
                                cfg['input_dim'][1], 
                                cfg['input_num_channels']), dtype=np.float32)
        for i in range(len( filename_template )):
            img = load_raw_image( 
                    make_filename_for_domain( filename_template[i], cfg['input_domain_name'] ), 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
            # process input
            img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
            input_img[i,:,:,:] = img
    else:
        input_img = load_raw_image( 
                make_filename_for_domain( filename_template, cfg['input_domain_name'] ), 
                color=( cfg['input_num_channels'] >= 3 ) )
    if 'is_discriminative' not in cfg:
        target_img = load_raw_image( 
                make_filename_for_domain( filename_template, cfg['target_domain_name'] ), 
                color=( cfg['target_num_channels'] >= 3 ) )
    else:
        raise ValueError("Using 'load_only_raw_images' for discriminative task if not advised; if only getting input_images, please use 'load_only_raw_inputs'")
    return input_img, target_img, target_img

def load_only_raw_inputs( filename_template, cfg, is_training=False ):
    if 'num_input' in cfg and cfg['num_input'] > 1:
        input_img = np.empty((cfg['num_input'], 
                                cfg['input_dim'][0], 
                                cfg['input_dim'][1], 
                                cfg['input_num_channels']), dtype=np.float32)
        for i in range(len( filename_template )):
            img = load_raw_image( 
                    make_filename_for_domain( filename_template[i], cfg['input_domain_name'] ), 
                    color=( cfg['input_num_channels'] >= 3 or cfg['input_domain_name'] == 'rgb') )
            # process input
            img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
            input_img[i,:,:,:] = img
    else:
        input_img = load_raw_image( 
                make_filename_for_domain( filename_template, cfg['input_domain_name'] ), 
                color=( cfg['input_num_channels'] >= 3 ) )
    return input_img


def resize_rescale_mask_tf( input_batch, target_batch, mask_batch, cfg, is_training=False ):
    """
        Resizes and rescales inputs and targets according to the cfg file. 
        Also optionally generates a mask from mask_batch.

        Args:
            input_batch: A (H, W, K) Tensor of input images
            target_batch: A (H_t, W_t, K_t) Tensor of target images
            mask_batch: A (H_t, W_t, K_t) Tensor of target images
            cfg: The dict from config.py

        Returns:
            input_batch_processed: (H', W', K)
            target_batch_processed: (H'_t, W'_t, K)
            mask_batch_processed: (H'_t, W'_t, K)

    """
    input_batch = rescale_images_tf( input_batch, cfg['input_dtype'], scope='rescale_input' )
    input_batch = resize_images_tf( input_batch, new_dims=cfg['input_dims'], dtypt=cfg['input_dtype'], scope='resize_input' )

    if 'mask_fn' in cfg:
        kwargs = {}
        if 'mask_fn_kwargs' in cfg:
            kwargs = cfg[ 'mask_fn_kwargs' ]
        mask_batch = cfg[ 'mask_fn' ]( img=target_batch, **kwargs ) # apply mask
        mask_dim = cfg[ 'target_dim']
        mask_batch = resize_images_tf( input_imgs=mask_batch, new_dims=mask_dim, dtype=tf.float32, scope='resize_mask' ) # Resize along with target
    else:
        mask_batch = tf.constant( 1.0, dtype=tf.float32, name='const_mask' )
    
    target_batch = rescale_images_tf( target_batch, cfg['target_dtype'], scope='rescale_target' )
    target_batch = resize_images_tf( target_batch, new_dims=cfg['target_dims'], dtypt=cfg['target_dtype'], scope='resize_input' )

    return input_batch, target_batch, mask_batch

def make_filename_for_domain( template, domain ):
    if len(template.split('/')) == 6 or len(template.split('/')) == 8:
        return template
    if template.split('/')[-1].isdigit(): 
        template = template.split('/')
        if template[0] == '':
            template[0] = os.sep
        #model_id, point_id, view_id = template.split('/')
        template[-1] = "point_{point_id}_view_{view_id}_domain_{{domain}}.png".format(
                    point_id=template[-2], view_id=template[-1])
        template[-2] = '{domain}'
        if domain == "keypoint2d":
            template[-2] = 'keypoint_2d'
        template = os.path.join(*template)
    return template.format( domain=domain )


def make_mask( input_img, target_img, cfg, mask_dim=None ):
    '''
    Takes in input and target images, as well as a given config dict. Builds a 
        mask and returns it. 

    Args:
        input_img: A numpy array
        target_img: A numpy array
        cfg should be a dict, usually specified in config.py, and may contain:
            'mask_fn': A function which returns the mask. If not given, this function
                returns 1.0, a Python float. 
            'mask_fn_kwargs': A dict of kwargs to be passed to the function
                There are some keyword key-value pairs that can be replaced:
                'img': This can contain one of [ '<INPUT_IMG>', '<TARGET_IMG>' ]
                'input_img': One of [ '<INPUT_IMG>' ]
                'target_img': [ '<TARGET_IMG>' ]
    
    Returns:
        mask
    '''
    if 'mask_fn' not in cfg:
        return 1.0

    if 'is_discriminative' in cfg:
        print("Using mask for discriminative task, proceed with caution")
    
    instance_kwargs = {}
    if 'mask_fn_kwargs' in cfg:
        instance_kwargs = cfg[ 'mask_fn_kwargs' ].copy()
        master_kwargs = cfg[ 'mask_fn_kwargs' ]

    def replace_instance_kwargs_keyword_with_img( keyword ):
        ''' Replaces 'keyword' in kwargs with the proper image '''
        if keyword not in master_kwargs:
            return
        keyword_replacement_dict = { 
            '<INPUT_IMG>': input_img,
            '<TARGET_IMG>': target_img
        }
        if master_kwargs[ keyword ] not in keyword_replacement_dict:
            raise ValueError( 'Acceptable values for {0} in mask_fn_kwargs are: {1}. Currently: {2}'.format( 
                        keyword, keyword_replacement_dict.keys(), master_kwargs[ keyword ] ) )
        instance_kwargs[ keyword ] = keyword_replacement_dict[ master_kwargs[ keyword ] ]

    # Replace kwargs
    if 'img' not in master_kwargs:
        # raise DeprecationWarning( "Omitting 'img' from cfg['mask_fn_kwargs'] is deprecated and support will be removed.")
        master_kwargs[ 'img' ] = '<TARGET_IMG>'
    replace_instance_kwargs_keyword_with_img( 'img' )
    replace_instance_kwargs_keyword_with_img( 'input_img' )
    replace_instance_kwargs_keyword_with_img( 'target_img' )

    target_mask = cfg[ 'mask_fn' ]( **instance_kwargs ) # apply mask
    if mask_dim is None:
        mask_dim = cfg[ 'target_dim' ]
    target_mask = resize_image( target_mask, mask_dim, interp_order=cfg['resize_interpolation_order'] ) # Resize along with target
    target_mask[target_mask<0.99] = 0. #Binarize mask -- be conservative.
    return target_mask


