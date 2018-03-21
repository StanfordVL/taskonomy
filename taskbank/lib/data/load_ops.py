'''
    Name: load_ops.py

    Desc: Input pipeline using feed dict method to provide input data to model.
        Some of this code is taken from Richard Zhang's colorzation github
            and python caffe library.
        Other parts of this code have been taken from Kevin Chen's library


'''

from __future__ import absolute_import, division, print_function

import itertools
import json
import math
import numpy as np
from   numpy import linalg as LA
import os
from   PIL import Image
import PIL
import pdb
import pickle
import random
import scipy
from   scipy.ndimage.interpolation import zoom
from   scipy.ndimage.filters import gaussian_filter
import skimage
import skimage.io
from   skimage.transform import resize
import sklearn.neighbors as nn
import string
import subprocess
import sys
import tensorflow as tf
from   transforms3d import euler
import transforms3d
import traceback as tb


if tf.__version__ == '0.10.0':
    tf_summary_scalar = tf.scalar_summary
else:
    tf_summary_scalar = tf.summary.scalar

#######################
# Loading fns
#######################

def load_scaled_image( filename, color=True ):
    """
    Load an image converting from grayscale or alpha as needed.
    From KChen

    Args:
        filename : string
        color : boolean
            flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).
    Returns
        image : an image with type np.float32 in range [0, 1]
            of size (H x W x 3) in RGB or
            of size (H x W x 1) in grayscale.
    By kchen 
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def load_raw_image( filename, color=True, use_pil=False ):
    """
    Load an image converting from grayscale or alpha as needed.
    Adapted from KChen

    Args:
        filename : string
        color : boolean
            flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).
    Returns
        image : an image with image original dtype and image pixel range
            of size (H x W x 3) in RGB or
            of size (H x W x 1) in grayscale.
    """
    if use_pil:
        img = Image.open( filename )
    else:
        img = skimage.io.imread(filename, as_grey=not color)

    if use_pil:
        return img

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
 
    return img

#########################
# Image manipulation fns
#########################

def resize_rescale_imagenet(img, new_dims, interp_order=1, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = img[:,:,[2,1,0]] * 255.
    mean_bgr = [103.062623801, 115.902882574, 123.151630838]
    img = img - mean_bgr
    return img

def resize_rescale_image_low_sat(img, new_dims, new_scale, interp_order=1, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = np.clip(img, 0.1, 0.9)
    img = rescale_image( img, new_scale, current_scale=current_scale, no_clip=no_clip )
    return img

def resize_rescale_image_low_sat_2(img, new_dims, new_scale, interp_order=1, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = np.clip(img, 0.2, 0.8)
    img = rescale_image( img, new_scale, current_scale=current_scale, no_clip=no_clip )
    return img

def resize_rescale_image(img, new_dims, new_scale, interp_order=1, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = rescale_image( img, new_scale, current_scale=current_scale, no_clip=no_clip )

    return img

def resize_rescale_image_gaussian_blur(img, new_dims, new_scale, interp_order=1, blur_strength=4, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = rescale_image( img, new_scale, current_scale=current_scale, no_clip=True )
    blurred = gaussian_filter(img, sigma=blur_strength)
    if not no_clip:
        min_val, max_val = new_scale
        np.clip(blurred, min_val, max_val, out=blurred)
    return blurred

def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)

    By kchen @ https://github.com/kchen92/joint-representation/blob/24b30ca6963d2ec99618af379c1e05e1f7026710/lib/data/input_pipeline_feed_dict.py
    """
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        interps = [PIL.Image.NEAREST, PIL.Image.BILINEAR]
        return skimage.util.img_as_float(im.resize(new_dims, interps[interp_order]))
        
    if all( new_dims[i] == im.shape[i] for i in range( len( new_dims ) ) ):
        resized_im = im #return im.astype(np.float32)
    elif im.shape[-1] == 1 or im.shape[-1] == 3:
        resized_im = resize(im, new_dims, order=interp_order, preserve_range=True)
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    # resized_im = resized_im.astype(np.float32)
    return resized_im


def rescale_image(im, new_scale=[-1.,1.], current_scale=None, no_clip=False):
    """
    Rescales an image pixel values to target_scale
    
    Args:
        img: A np.float_32 array, assumed between [0,1]
        new_scale: [min,max] 
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image
    """
    im = skimage.img_as_float(im).astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val) 
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val

    return im 

def resize_and_rescale_image_log( img, new_dims, offset=1., normalizer=1.):
    """
        Resizes and rescales an img to log-linear
        
        Args:
            img: A np array
            offset: Shifts values by offset before taking log. Prevents 
                taking the log of a negative number
            normalizer: divide by the normalizing factor after taking log
        Returns:
            rescaled_image
    """
    img =  np.log( float( offset ) + img ) / normalizer 
    img = resize_image(img, new_dims)
    return img

def rescale_image_log( img, offset=1., normalizer=1. ):
    """
        Rescales an img to log-linear
        
        Args:
            img: A np array
            offset: Shifts values by offset before taking log. Prevents 
                taking the log of a negative number
            normalizer: divide by the normalizing factor after taking log
        Returns:
            rescaled_image
    """
    return np.log( float( offset ) + img ) / normalizer 

################
# Curvature     #
#################

def curvature_preprocess(img, new_dims, interp_order=1):
    img = resize_image(img, new_dims, interp_order)
    img = img[:,:,:2]
    img = img - [123.572, 120.1]
    img = img / [31.922, 21.658]
    return img

def curvature_preprocess_gaussian_with_blur(img, new_dims, interp_order=1, blur_strength=4):
    k1 = img[:,:,0].astype(np.float32) - 128.0 
    k2 = img[:,:,1].astype(np.float32) - 128.0
    curv = k1 * k2
    curv = curv * 8.0 / (127.0 ** 2)
    curv = curv[:,:,np.newaxis]
    curv = resize_image(curv, new_dims, interp_order)
    blurred = gaussian_filter(curv, sigma=blur_strength)
    return blurred

def curvature_preprocess_gaussian(img, new_dims, interp_order=1):
    k1 = img[:,:,0].astype(np.float32) - 128.0 
    k2 = img[:,:,1].astype(np.float32) - 128.0
    curv = k1 * k2
    curv = curv * 8.0 / (127.0 ** 2)
    curv = curv[:,:,np.newaxis]
    curv = resize_image(curv, new_dims, interp_order)
    return curv

#################
# Denoising     #
#################

def random_noise_image(img, new_dims, new_scale, interp_order=1 ):
    """
        Add noise to an image

        Args:
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
    Returns:
            a noisy version of the original clean image
    """
    img = skimage.util.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = skimage.util.random_noise(img, var=0.01)
    img = rescale_image( img, new_scale )
    return img

#################
# Colorization  #
#################

def to_light_low_sat(img, new_dims, new_scale, interp_order=1 ):
    """
    Turn an image into lightness 
        
        Args:
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
    Returns:
        a lightness version of the original image
    """
    img = skimage.img_as_float( img )
    img = np.clip(img, 0.2, 0.8)
    img = resize_image( img, new_dims, interp_order )
    img = skimage.color.rgb2lab(img)[:,:,0]
    img = rescale_image( img, new_scale, current_scale=[0,100])
    return np.expand_dims(img,2)

def to_light(img, new_dims, new_scale, interp_order=1 ):
    """
    Turn an image into lightness 
        
        Args:
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
    Returns:
        a lightness version of the original image
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = skimage.color.rgb2lab(img)[:,:,0]
    img = rescale_image( img, new_scale, current_scale=[0,100])
    return np.expand_dims(img,2)

def to_ab(img, new_dims, new_scale, interp_order=1 ):
    """
    Turn an image into ab 
        
        Args:
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
    Returns:
        a ab version of the original image
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = skimage.color.rgb2lab(img)[:,:,1:]
    img = rescale_image( img, new_scale, current_scale=[-100,100])
    return img

def ab_image_to_prob(img, new_dims, root, interp_order=1):
    """
    Turn an image into a probability distribution across color pair specified in pts_in_hull.npy
    It's referencing: https://github.com/richzhang/colorization
        
        Args:
        im : (H x W x K) ndarray
    Returns:
        Color label ground truth across 313 possible ab color combinations
    """
    img = resize_image( img, new_dims, interp_order ).astype('uint8')
    img = skimage.color.rgb2lab(img)[:,:,1:]
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cc = np.load(os.path.join(curr_dir, 'pts_in_hull.npy'))
    K = cc.shape[0]
    NN = 10
    sigma = 5.
    nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(cc)

    num_pixels = img.shape[0] * img.shape[1]
    img_flattened = img.reshape(num_pixels, img.shape[2])

    encoded_flattened = np.zeros((num_pixels, K))
    point_index = np.arange(0,num_pixels, dtype='int')[:, np.newaxis]

    (dists, inds) = nbrs.kneighbors(img_flattened)
    wts = np.exp(-dists**2/(2*sigma**2))
    wts = wts/np.sum(wts,axis=1)[:,np.newaxis]

    encoded_flattened[point_index, inds] = wts
    encoded = encoded_flattened.reshape([img.shape[0], img.shape[1], K])

    ############## Prior Boost Mask #################
    prior_factor = np.load(os.path.join(curr_dir, 'prior_factor_in_door.npy'))
    encoded_maxid = np.argmax(encoded, axis=-1)
    mask = prior_factor[encoded_maxid]

    return encoded, mask

###################
# Context Encoder #
###################

def context_encoder_input( img, new_dims, new_scale, interp_order=1 ):
    '''
    Context encoder input function, substitute the middle section with constant

    Returns:
    ----------
        img: with center 1/4 being constant average value
    '''
    img = resize_rescale_image(img, new_dims, new_scale, interp_order=interp_order) 
  
    H,W,K = img.shape
    img[ int(H/4):int(3*H/4), int(W/4):int(3*W/4), :] = 0 
    return img

def context_encoder_output(img, new_dims, new_scale, interp_order=1 ):
    '''
    Context encoder target function, take out the middle chunk
    '''
    whole_dims = (new_dims[0]*2, new_dims[1]*2)
    img = resize_rescale_image(img, whole_dims, new_scale, interp_order=interp_order) 

    H,W,_ = img.shape
    center_piece = img[ int(H/4):int(H/4)+new_dims[0]
                      , int(W/4):int(W/4)+new_dims[1], :] 
    return center_piece

#################################
# Discriminative Target Process #
#################################

def parse_filename( filename ):
    """
    Filename is in the format:
        '/{PATH_TO_DATA_ROOT}/{MODEL_ID}/{domain}
            /point_{POINT_ID}_view_{VIEW_ID}_domain_{DOMAIN_NAME}.png'
    
    Parameters:
    -----------
        filename: a string in the formate specified above.

    Returns:
    -----------
        path_to_root: path to data root directory
        domain: domain name
        model_id: model id
        point_id: point id
        view_id: view id
    """
    root_length = 3 #can be changed
    preappend_slash = True
    file_structure_length = 3

    components = filename.split('/')[preappend_slash:]
    assert len(components) == root_length + file_structure_length

    if preappend_slash:
        path_to_root = os.path.join("/" , *components[:root_length])
    else:
        path_to_root = os.path.join(*components[:root_length])


    model_id = components[root_length]
    name_components = components[-1].split('_')

    if len(name_components) == 6:
        domain = components[root_length+1]
        point_id = name_components[1]
        view_id = name_components[3]
    elif len(name_components) == 1:
        view_id = name_components[0]
        point_id = components[root_length+1]
        domain = 'rgb'
    return path_to_root, domain, model_id, point_id, view_id

def generate_rgb_image_filename_from_ID(root, model_id, point_id, view_id):
    '''
    Given the root, model_id, point_id, view_id of an image, return the rgb
    file path of that image. The file path is in the format:

    /{root}/{model_id}/rgb/
        point_{point_id}_view_{view_id}_domain_rgb.png

    Parameters:
    -----------
        root: path to root
        model_id: id of the model
        point_id: the id number of the point
        view_id: the id number of views

    Returns:
    -----------
        path: file path to the image file
    '''
    filename = "point_{point_id}_view_{view_id}_domain_rgb.png".format(
                    point_id=point_id, view_id=view_id)

    path = os.path.join(root, model_id, 'rgb', filename)

    return path

def make_image_filenames( filename, num_input):
    '''
    Turn one image filename that contains the information of a image pair into multiple
    image filenames.

    For camera pose matching.

    The filename should be in the same format, except the point_id and view_id field is
    multiple integers with length num_input separated by commas:
    /{PATH_TO_ROOT}/{MODEL_ID}/{domain}/{LIST_OF_POINT_IDS}_
        view_{LIST_OF_VIEW_IDS}_{SOMETHING ELSE}

    Parameters:
    -----------
        filename: A filename that in the format specified as above.
        num_input: length of the LIST_OF_POINT_IDS

    Returns:
    -----------
        filenames: A list of image filenames
    '''
    if len(filename.split('/')) == 6 or len(filename.split('/')) == 8 :
        return [filename] * num_input
    root, domain, model_id, point_ids, view_ids = parse_filename( filename )

    model_ids = model_id.split(',')
    point_ids = point_ids.split(',')
    view_ids = view_ids.split(',')

    if len(view_ids) != num_input:
        if len(view_ids) == 1 and len(point_ids) == 1:
            image_name = generate_rgb_image_filename_from_ID(root, model_id, point_ids[0], view_ids[0])
            image_name = [image_name] * num_input 
            return image_name
        else:
            raise ValueError("num_input doesn't match the length of view_ids")
        

    filenames = []
    if len(point_ids) == 1:
        point_id = point_ids[0]
        for index in range(num_input):
            view_id = view_ids[index]
            filenames.append(generate_rgb_image_filename_from_ID(root, model_id, point_id, view_id))
    else:
        for index in range(num_input):
            view_id = view_ids[index]
            point_id = point_ids[index]
            if len(model_ids) > 1:
                model_i = model_ids[index]
            else:
                model_i = model_id
            filenames.append(generate_rgb_image_filename_from_ID(root, model_i, point_id, view_id))

    return filenames

###################
# Point Matching  #
###################

def point_match_new( filename ):
    model_ids = filename.split('/')[0]
    if len(model_ids.split(',')) == 2:
        return 0
    point_ids = filename.split('/')[-2]
    if len(point_ids.split(',')) == 2:
        return 0
    return 1

################################
# Camera Pose Helper functions #
################################

def parse_fixated_filename( filename ):
    """
    Fixated filename is stored in similar format as single filename, but with multiple views

    Return a list of filenames that has root directory specifid by root_dir

    Parameters:
    -----------
        filename: filename in the specific format

    Returns:
    -----------
        full_paths: a list of full path to camera pose info for the point-view pair
    """
    root, domain, model_id, point_id, num_views = parse_filename( filename )
    view_ids = num_views.split(',')

    new_domain = "fixatedpose"
    domain = "points"
    full_paths = []

    for view_id in view_ids:
        filename = 'point_{point_id}_view_{view_id}_domain_{domain}.json'.format(
                    point_id=point_id,
                    view_id=view_id,
                    domain=new_domain)

        full_path = os.path.join(root, model_id, domain, filename)
        full_paths.append(full_path)

    return full_paths

def parse_nonfixated_filename( filename ):
    """
    Nonfixated filename is stored in the format:
    '/{ROOT}/{MODEL_ID}/{POINT_IDS}/{VIEW_IDS}'

    POINT_IDS and VIEW_IDS are lists that are separated by comma.

    Return a list of filenames that has root directory specifid by root_dir

    Parameters:
    -----------
        filename: filename in the specific format

    Returns:
    -----------
        full_paths: a list of full path to camera pose info for the point-view pair
    """
    root, domain,  model_id, num_points, num_views = parse_filename( filename )
    point_ids = num_points.split(',')
    view_ids = num_views.split(',')

    domain = "points"
    new_domain = "fixatedpose"
    full_path = []
    for i in range(len(point_ids)):
        filename = 'point_{point_id}_view_{view_id}_domain_{domain}.json'.format(
                    point_id=point_ids[i],
                    view_id=view_ids[i],
                    domain=new_domain)

        full_path_i = os.path.join(root, model_id, domain, filename)
        full_path.append(full_path_i)

    return full_path

    
def calculate_relative_camera_location(full_path1, full_path2):
    """
    Given two file path to two json files, extract the 'camera_location'
    and 'camera_rotation_final' field, and calcualte the relative camera pose

    Parameters:
    __________
        full_path1, full_path2: paths to json information

    Returns:
    __________
        camera_poses: vector that encode the camera pose info for two images
    """
    assert os.path.isfile(full_path1) and os.path.isfile(full_path2)
    with open(full_path1, 'r') as fp:
        data1 = json.load(fp)
    with open(full_path2, 'r') as fp:
        data2 = json.load(fp)
    key = ['camera_location', 'camera_rotation_final']
    location1 = data1[key[0]]
    location2 = data2[key[0]]
    translation = np.asarray(location1) - np.asarray(location2)
    return translation


def calculate_relative_camera_pose(full_path1, full_path2, fixated=True, raw=False):
    """
    Given two file path to two json files, extract the 'camera_location'
    and 'camera_rotation_final' field, and calcualte the relative camera pose

    Parameters:
    __________
        full_path1, full_path2: paths to json information

    Returns:
    __________
        camera_poses: vector that encode the camera pose info for two images
    """
    assert os.path.isfile(full_path1) and os.path.isfile(full_path2)
    with open(full_path1, 'r') as fp:
        data1 = json.load(fp)
    with open(full_path2, 'r') as fp:
        data2 = json.load(fp)
    key = ['camera_location', 'camera_rotation_final']
    location1 = np.asarray(data1[key[0]])
    rotation1 = data1[key[1]]
    matrix1 = euler.euler2mat(*rotation1, axes='sxyz')
    location2 = np.asarray(data2[key[0]])
    rotation2 = data2[key[1]]
    matrix2 = euler.euler2mat(*rotation2, axes='sxyz')
    relative_rotation_matrix = np.matmul(np.transpose( matrix2 ), matrix1)
    relative_rotation = euler.mat2euler(relative_rotation_matrix, axes='sxyz')
    translation = np.matmul(np.transpose(matrix2), location1 - location2)
    pose = np.hstack((relative_rotation, translation))
    if not raw:
        if fixated:
            std  = np.asarray([ 10.12015407, 8.1103528, 1.09171896, 1.21579016, 0.26040945, 10.05966329])
            mean = np.asarray([ -2.67375523e-01, -1.19147040e-02, 1.14497274e-02, 1.10903410e-03, 2.10509948e-02, -4.02013549e+00])
        else:
            mean = np.asarray([ -9.53197445e-03,  -1.05196691e-03,  -1.07545642e-02,
                        2.08785638e-02,  -9.27858049e-02,  -2.58052205e+00])
            std = np.asarray([ 1.02316223,  0.66477511,  1.03806996,  5.75692889,  1.37604962,
                        7.43157247])
        pose = (pose - mean)/std   
    return pose


########################################
# Fixated and Non-fixated Camera Pose  #
########################################

def nonfixated_camera_pose( filename ):
    """
    Return two 6DOF camera pose vectors for two images of nonfixated view.
        Filename is in the format:
        '/{PATH_TO_DATA_ROOT}/{MODEL_ID}/{domain}
            /point_{POINT_ID}_view_{VIEW_ID}_domain_{DOMAIN_NAME}.png'
    

    Parameters:
    ----------
        filename: a filename that embodies what point we are examining

    Returns:
    -----------
        camera_poses: vector that encode the camera pose info for two images
    """
    if isinstance(filename, list):
        raise ValueError("Having more than two inputs to a fixated camera pose problem")
    
    full_paths = parse_nonfixated_filename( filename ) 

    if len(full_paths) != 2:
        raise ValueError(
            "camera pose should have filename with 2 point-view, {filename}".format(filename=filename))

    pose = calculate_relative_camera_pose(full_paths[0], full_paths[1], fixated=False)

    return pose

def nonfixated_camera_rot( filename ):
    """
    Return two 6DOF camera pose vectors for two images of nonfixated view.
        Filename is in the format:
        '/{PATH_TO_DATA_ROOT}/{MODEL_ID}/{domain}
            /point_{POINT_ID}_view_{VIEW_ID}_domain_{DOMAIN_NAME}.png'
    

    Parameters:
    ----------
        filename: a filename that embodies what point we are examining

    Returns:
    -----------
        camera_poses: vector that encode the camera pose info for two images
    """
    if isinstance(filename, list):
        raise ValueError("Having more than two inputs to a fixated camera pose problem")
    
    full_paths = parse_nonfixated_filename( filename ) 

    if len(full_paths) != 2:
        raise ValueError(
            "camera pose should have filename with 2 point-view, {filename}".format(filename=filename))

    pose = calculate_relative_camera_pose(full_paths[0], full_paths[1], fixated=False)

    rot = pose[:3]
    return rot

def fixated_camera_pose( filename ):
    """
    Return two 6DOF camera pose vectors for two images of fixated view.
        Filename is in the format:
        '/{PATH_TO_DATA_ROOT}/{MODEL_ID}/{domain}
            /point_{POINT_ID}_view_{VIEW_ID}_domain_{DOMAIN_NAME}.png'
    

    Parameters:
    ----------
        filename: a filename that embodies what point we are examining

    Returns:
    -----------
        camera_poses: vector that encode the camera pose info for two images
    """
    if isinstance(filename, list):
        raise ValueError("Having more than two inputs to a fixated camera pose problem")
    
    full_paths = parse_fixated_filename(filename)

    if len(full_paths) != 2:
        raise ValueError(
            "camera pose should have filename with 2 point-view, {filename}".format(filename=filename))


    pose = calculate_relative_camera_pose(full_paths[0], full_paths[1])

    return pose

def fixated_camera_rot( filename ):
    """
    Return two 6DOF camera pose vectors for two images of fixated view.
        Filename is in the format:
        '/{PATH_TO_DATA_ROOT}/{MODEL_ID}/{domain}
            /point_{POINT_ID}_view_{VIEW_ID}_domain_{DOMAIN_NAME}.png'
    

    Parameters:
    ----------
        filename: a filename that embodies what point we are examining

    Returns:
    -----------
        camera_poses: vector that encode the camera pose info for two images
    """
    if isinstance(filename, list):
        raise ValueError("Having more than two inputs to a fixated camera pose problem")
    
    full_paths = parse_fixated_filename(filename)

    if len(full_paths) != 2:
        raise ValueError(
            "camera pose should have filename with 2 point-view, {filename}".format(filename=filename))


    pose = calculate_relative_camera_pose(full_paths[0], full_paths[1])
    rot = pose[:3]
    return rot

#################
# Ego-Motion    #
#################

def triplet_fixated_egomotion( filename ):
    """
    Given a filename that contains 3 different point-view combos, parse the filename
    and return the pair-wise camera pose.

    Parameters:
    -----------
        filename: a filename in the specific format.

    Returns:
    -----------
        egomotion: a numpy array of length 18 (3x6). 
                   (a concatanation of 3 6-DOF relative camera pose vector)
    """
    if isinstance(filename, list):
        raise ValueError("Having more than two inputs to a fixated camera pose problem")
    
    full_paths = parse_fixated_filename(filename)

    if len(full_paths) != 3 :
        raise ValueError("quadruplet first view prediction with list shorter than 3")

#     perm = range(3)
# random.shuffle(perm)

    #full_paths = [full_paths[i] for i in perm]
    poses = []

    for i in range(2):
        for j in range(i+1, 3):
            pose = calculate_relative_camera_pose(full_paths[i], full_paths[j])
            poses.append(pose)

    poses = np.hstack(poses)

    return poses

#################
# Jigsaw        #
#################

def jigsaw_rand_index( filename ):
    return random.randint(0,99)

def hamming_distance(p1, p2):
    '''
    Calculate the Hamming distance between two permutations
    '''
    if len(p1) != len(p2):
        raise ValueError('two permutations have different length...')

    total_diff = sum(e1 != e2 for e1, e2 in zip(p1, p2))
    return total_diff / len(p1)

def get_max_hamming_distance_index(p, current):
    '''
    This function take in two sets of permutation, calcuate which permutation should
    be added to the current set, which is the permutation that maximize the sum of
    Hamming distance from current permutations.

    Parameters:
    -----------
        p: the set of all candidate permutations
        current: current set of chosen permutations

    Returns:
    -----------
        next_index: the index in p that maximize Hamming distance
    '''
    max_index = -1
    max_distance = -1

    for i in range(len(p)):
        entry_i_dist = 0
        for j in range(len(current)):
            entry_i_dist += hamming_distance(p[i], current[j])

        if entry_i_dist > max_distance:
            max_index = i
            max_distance = entry_i_dist

    return max_index, max_distance

def generate_permutation_set(length):
    '''
    This function generate the set of maximum Hamming distance permutation.

    The set has size 100.

    Returns:
    ---------
        perm: set with 100 permutations that maximize Hamming distance.
    '''
    perm = []
    total = math.factorial(9)
    
    #p = list(itertools.permutations(range(9)))
    p = []
    for i in itertools.permutations(range(9)):
        p.append(i)
        print(i)
    print('Finished generating entire set with size {s}'.format(s=len(p)))
    p0 = random.randint(0,total-1)

    perm.append(p.pop(p0))

    for i in range(length-1):
        print('entry {x} added...'.format(x=i+1))
        next_index,_ = get_max_hamming_distance_index(p, perm)
        perm.append(p.pop(next_index))

    asset_dir = "../"
    store_location = os.path.join( asset_dir, 'jigsaw_max_hamming_set.npy')

    with open(store_location, 'wb') as store:
        np.save(store, perm)

    return perm

def generate_jigsaw_input_with_dropping( img, target, new_dims, new_scale, interp_order=1):
    '''
    Generate the 9 pieces input for Jigsaw task.

    Parameters:
    -----------
        img: input image
        target: length 9 permutation

    Return:
    -----------
        input_imgs: 9 image pieces
    '''
    if len(target) != 9:
        raise ValueError('Target permutation of Jigsaw is supposed to have lenght 9, getting {x} here'.format(len(target)))

    img = rescale_image( img, new_scale )
    
    H,W,K = img.shape
    to_drop = random.sample(list(range(K)), K-1)

    for channel in to_drop:
        img[:,:,channel] = np.random.normal(0.0, 0.01, (H,W))

    unitH = int(H / 3)
    unitW = int(W / 3)
    
    cropH = int(unitH * 0.9)
    cropW = int(unitW * 0.9)
    
    startH = unitH - cropH 
    startW = unitW - cropW
    
    input_imgs = np.empty((9, new_dims[0], new_dims[1], K), dtype=np.float32) 

    for i in range(9):
        pos = target[i]
        posH = int(pos / 3) * unitH + random.randint(0, startH)
        posW = int(pos % 3) * unitW + random.randint(0, startW)

        img_piece = img[posH:posH+cropH,posW:posW+cropW,:]

        input_imgs[i,:,:,:] = resize_image(img_piece, new_dims, interp_order)

    return input_imgs

def generate_jigsaw_input( img, target, new_dims, new_scale, interp_order=1):
    '''
    Generate the 9 pieces input for Jigsaw task.

    Parameters:
    -----------
        img: input image
        target: length 9 permutation

    Return:
    -----------
        input_imgs: 9 image pieces
    '''
    if len(target) != 9:
        raise ValueError('Target permutation of Jigsaw is supposed to have lenght 9, getting {x} here'.format(len(target)))

    img = rescale_image( img, new_scale )
    
    H,W,K = img.shape
    unitH = int(H / 3)
    unitW = int(W / 3)
    
    cropH = int(unitH * 0.9)
    cropW = int(unitW * 0.9)
    
    startH = unitH - cropH 
    startW = unitW - cropW
    
    input_imgs = np.empty((9, new_dims[0], new_dims[1], K), dtype=np.float32) 

    for i in range(9):
        pos = target[i]
        posH = int(pos / 3) * unitH + random.randint(0, startH)
        posW = int(pos % 3) * unitW + random.randint(0, startW)

        img_piece = img[posH:posH+cropH,posW:posW+cropW,:]

        input_imgs[i,:,:,:] = resize_image(img_piece, new_dims, interp_order)

    return input_imgs

def generate_jigsaw_input_for_representation_extraction( img, new_dims, new_scale, interp_order=1):
    '''
    Generate the 9 pieces input for Jigsaw task.

    Parameters:
    -----------
        img: input image
        target: length 9 permutation

    Return:
    -----------
        input_imgs: 9 copies of the input image
    '''
    img = rescale_image( img, new_scale )
    H,W,K = img.shape
    input_imgs = np.empty((9, new_dims[0], new_dims[1], K), dtype=np.float32) 
    return resize_image(img, new_dims, interp_order)#input_imgs

###################
# Vanishing Point #
###################

def get_camera_matrix( view_dict, flip_xy=False ):
    position = view_dict[ 'camera_location' ]
    rotation_euler = view_dict[ 'camera_rotation_final' ]
    R = transforms3d.euler.euler2mat( *rotation_euler, axes='sxyz' )
    camera_matrix = transforms3d.affines.compose(  position, R, np.ones(3) )
    
    if flip_xy:
        # For some reason the x and y are flipped in room layout
        temp = np.copy(camera_matrix[0,:])
        camera_matrix[0,:] = camera_matrix[1,:]
        camera_matrix[1,:] = -temp
    return camera_matrix

def get_camera_rot_matrix(view_dict, flip_xy=False):
    return get_camera_matrix(view_dict, flip_xy=True)[:3, :3]

def rotate_world_to_cam( points, view_dict ):
    cam_mat = get_camera_rot_matrix( view_dict, flip_xy=True )
    new_points = cam_mat.T.dot(points).T[:,:3]
    return new_points

def vanishing_point( filename ):
    '''
    Hemisphere projection of TOVP.

    Returns:
    --------
        vanishing_point: length 9 vector
    '''
    root, domain, model_id, point_id, view_id = parse_filename(filename)

    fname = 'point_{point_id}_view_{view_id}_domain_{domain}.json'.format(
                    point_id=point_id,
                    view_id=view_id,
                    domain='fixatedpose')
    json_file = os.path.join(root, model_id, 'points', fname)

    with open(json_file, 'r') as fp:
        data = json.load(fp)

    if 'vanishing_points_gaussian_sphere' not in data:
        return model_id
    vps = data['vanishing_points_gaussian_sphere']

    vanishing_point = np.hstack((vps['x'], vps['y'], vps['z']))
    return vanishing_point

def rotation_to_make_axes_well_defined(view_dict):
    ''' Rotates the world coords so that the -z direction of the camera 
        is within 45-degrees of the global +x axis '''
    axes_xyz = np.eye(3)
    apply_90_deg_rot_k_times = [
        transforms3d.axangles.axangle2mat(axes_xyz[-1], k * math.pi/2)
        for k in range(4) ]

    global_x = np.array([axes_xyz[0]]).T
    global_y = np.array([axes_xyz[1]]).T
    best = (180., "Nothing")
    for world_rot in apply_90_deg_rot_k_times:
        global_x_in_cam = rotate_world_to_cam( 
                world_rot.dot(global_x), view_dict )
        global_y_in_cam = rotate_world_to_cam( 
                world_rot.dot(global_y), view_dict )
        # Project onto camera's horizontal (xz) plane
        degrees_away_x = math.degrees(
                        math.acos(np.dot(global_x_in_cam, -axes_xyz[2]))
                        )
        degrees_away_y = math.degrees(
                        math.acos(np.dot(global_y_in_cam, -axes_xyz[2]))
                        )
        total_degrees_away = abs(degrees_away_x) + abs(degrees_away_y)
        best = min(best, (total_degrees_away, np.linalg.inv(world_rot))) # python is neat
    return best[-1]

def vanishing_point_well_defined( filename ):
    root, domain, model_id, point_id, view_id = parse_filename(filename)

    fname = 'point_{point_id}_view_{view_id}_domain_{domain}.json'.format(
                    point_id=point_id,
                    view_id=view_id,
                    domain='fixatedpose')
    json_file = os.path.join(root, model_id, 'points', fname)

    with open(json_file, 'r') as fp:
        data = json.load(fp)

    cam_mat = get_camera_matrix( data, flip_xy=True )
    world_transformation = rotation_to_make_axes_well_defined(data)
    cam_mat[:3,:3] = np.dot(world_transformation, cam_mat[:3, :3])
    R = cam_mat[:3,:3]
    dist = 1.0
    compass_points = [
                    (dist, 0, 0),
                    (0, dist, 0),
                    (0, 0, dist) ]
    vanishing_point = [np.dot( np.linalg.inv(R), p ) for p in compass_points]
    return np.array(vanishing_point).flatten()


###############
# Room Layout #
###############

def get_room_layout_cam_mat_and_ranges(view_dict, make_x_major=False):

    # Get BB information
    bbox_ranges = view_dict['bounding_box_ranges']
    # BB seem to be off w.r.t. the camera matrix
    ranges = [ bbox_ranges['x'], -np.array(bbox_ranges['y'])[::-1], bbox_ranges['z'] ]

    camera_matrix = get_camera_matrix(view_dict, flip_xy=True)
    if not make_x_major:
        return camera_matrix, ranges
    # print(world_points[:,-1])
    # print(view_dict['camera_location'])
    axes_xyz = np.eye(3)
    apply_90_deg_rot_k_times = [
            transforms3d.axangles.axangle2mat(axes_xyz[-1], k * math.pi/2)
            for k in range(4) ]

    def make_world_x_major(view_dict):
        ''' Rotates the world coords so that the -z direction of the camera 
            is within 45-degrees of the global +x axis '''
        global_x = np.array([axes_xyz[0]]).T
        best = (180., "Nothing")
        for world_rot in apply_90_deg_rot_k_times:
            global_x_in_cam = rotate_world_to_cam( 
                    world_rot.dot(global_x), view_dict )
            # Project onto camera's horizontal (xz) plane
            degrees_away = math.degrees(
                            math.acos(np.dot(global_x_in_cam, -axes_xyz[2]))
                            )
            best = min(best, (degrees_away, np.linalg.inv(world_rot))) # python is neat
            # if abs(degrees_away) < 45.:
            #     return np.linalg.inv(world_rot)
        return best[-1]
    def update_ranges(world_rot, ranges):
        new_ranges = np.dot(world_rot, ranges)
        for i, rng in enumerate(new_ranges): # make sure rng[0] < rng[1]
            if rng[0] > rng[1]:
                new_ranges[i] = [rng[1], rng[0]]
        return new_ranges

    world_rot = np.zeros((4,4))
    world_rot[3,3] = 1.
    world_rot[:3,:3] = make_world_x_major(view_dict)
    ranges = update_ranges(world_rot[:3,:3], ranges)
    camera_matrix = np.dot(world_rot, camera_matrix)
    return camera_matrix, ranges


def room_layout( filename ):
    '''
    Room Bounding Box.
    Returns:
    --------
        bb: length 6 vector
    '''
    root, domain, model_id, point_id, view_id = parse_filename(filename)

    fname = 'point_{point_id}_view_{view_id}_domain_{domain}.json'.format(
                    point_id=point_id,
                    view_id=view_id,
                    domain='fixatedpose')
    json_file = os.path.join(root, model_id, 'points', fname)
    with open(json_file) as fp:
        data = json.load(fp)

    def homogenize( M ):
        return np.concatenate( [M, np.ones( (M.shape[0],1) )], axis=1 )

    def convert_world_to_cam( points, cam_mat=None ):
        new_points = points.T
        homogenized_points = homogenize( new_points )
        new_points = np.dot( homogenized_points, np.linalg.inv(cam_mat).T )[:,:3]
        return new_points
    
    mean = np.array([0.006072743318127848, 0.010272365569691076, -3.135909774145468, 
            1.5603802322235532, 5.6228218371102496e-05, -1.5669352793761442,
            5.622875878174759, 4.082800262277375, 2.7713941642895956])
    std = np.array([0.8669452525283652, 0.687915294956501, 2.080513632043758, 
            0.19627420479282623, 0.014680602791251812, 0.4183827359302299,
            3.991778013006544, 2.703495278378409, 1.2269185938626304])
    camera_matrix, bb = get_room_layout_cam_mat_and_ranges(data, make_x_major=True)
    camera_matrix_euler = transforms3d.euler.mat2euler(camera_matrix[:3,:3], axes='sxyz')
    vertices = np.array(list(itertools.product( *bb )))
    vertices_cam = convert_world_to_cam(vertices.T, camera_matrix)
    cube_center = np.mean(vertices_cam, axis=0)

    x_scale, y_scale, z_scale = bb[:,1] - bb[:,0] # maxes - mins
    bbox_cam = np.hstack(
        (cube_center, 
        camera_matrix_euler,
        x_scale, y_scale, z_scale))
    bbox_cam = (bbox_cam - mean) / std 
    return bbox_cam

####################
# ImageNet Softmax #
####################

def np_softmax(logits):
    maxs = np.amax(logits, axis=-1)
    softmax = np.exp(logits - np.expand_dims(maxs, axis=-1))
    sums = np.sum(softmax, axis=-1)
    softmax = softmax / np.expand_dims(sums, -1)
    return softmax

def class_1000_softmax( template ):
    '''
    Class 1000 softmax prediction

    Returns:
    --------
        sfm: 1000 classes sfm
    '''
    num_classes = 1000
    if template.split('/')[-1].isdigit(): 
        template = template.split('/')
        if template[0] == '':
            template[0] = os.sep
        template[-1] = "point_{point_id}_view_{view_id}.npy".format(
                    point_id=template[-2], view_id=template[-1])
        template[-2] = 'softmax_1000'
        template = os.path.join(*template)
    if not os.path.isfile(template):
        return np.ones((1000)) / 1000. , np.zeros((1)) 
    if os.stat(template).st_size < 100:
        return np.ones((1000)) / 1000. , np.zeros((1)) 
    logits = np.load(template)
    logits = np.squeeze(logits)
    sfm = np_softmax(logits)
    return sfm, np.ones((1))

def class_1000_imagenet( template ):
    '''
    Class 1000 ImageNet Ground Truth

    Returns:
    --------
        sfm: 1000 classes sfm
    '''
    num_classes = 1000
    class_id = template.split('/')[-2]
    try:
        class_idx = int(class_id)
    except ValueError:
        import pickle
        class_id = template.split('/')[-1].split('_')[0]
        with open('/home/ubuntu/task-taxonomy-331b/lib/data/class_idx.pkl', 'rb') as fp:
            correspondence = pickle.load(fp)
        class_idx = int(correspondence[class_id])
    sfm = np.zeros((1000), dtype=(np.float32))
    sfm[class_idx] = 1.
    return sfm, np.ones((1))

def class_places( template ):
    list_of_classes = ["alcove", "assembly_line", "atrium-public", "attic", "auto_factory", "bank_vault", "basement", "bathroom", "bedchamber", "bedroom", "biology_laboratory", "booth-indoor", "bow_window-indoor", "chemistry_lab", "childs_room", "clean_room", "closet", "computer_room", "conference_room", "corridor", "dining_room", "dorm_room", "dressing_room", "elevator-door", "elevator_shaft", "engine_room", "escalator-indoor", "garage-indoor", "greenhouse-indoor", "home_office", "home_theater", "hospital_room", "hotel_room", "kitchen", "laundromat", "living_room", "lobby", "mezzanine", "nursery", "nursing_home", "office", "office_cubicles", "operating_room", "pantry", "parking_garage-indoor", "physics_laboratory", "playroom", "reception", "recreation_room", "repair_shop", "restaurant_kitchen", "server_room", "shower", "stable", "staircase", "storage_room", "television_room", "ticket_booth", "utility_room", "veterinarians_office", "waiting_room", "wet_bar", "youth_hostel"]
    class_idx = list_of_classes.index(template.split('/')[-2])
    sfm = np.zeros((63), dtype=(np.float32))
    sfm[class_idx] = 1.
    return sfm, np.ones((1))

def class_places_workspace_and_home( template ):
    class_to_keep = np.asarray([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
        1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
        0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,
        0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
        0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,
        0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
        1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,
        0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
        0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
        0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
        0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.])
    if template.split('/')[-1].isdigit(): 
        template = template.split('/')
        if template[0] == '':
            template[0] = os.sep
        template[-1] = "point_{point_id}_view_{view_id}.npy".format(
                    point_id=template[-2], view_id=template[-1])
        template[-2] = 'places'
        template = os.path.join(*template)
    sfm = np.load(template)
    sfm_selected = sfm[class_to_keep.astype(bool)]
    sfm_selected = sfm_selected / np.sum(sfm_selected)
    return sfm_selected, np.ones((1))


####################
# Segmentation     #
####################
def semantic_segment_rebalanced( template, new_dims, domain, root='/home/ubuntu/task-taxonomy-331b' ):
    '''
    Segmentation

    Returns:
    --------
        pixels: size num_pixels x 3 numpy array
    '''
    if template.split('/')[-1].isdigit(): 
        template = template.split('/')
        if template[0] == '':
            template[0] = os.sep
        template[-1] = "point_{point_id}_view_{view_id}_domain_{{domain}}.png".format(
                    point_id=template[-2], view_id=template[-1])
        template[-2] = '{domain}'
        template = os.path.join(*template)
    filename = template.format( domain=domain )
    if not os.path.isfile(filename):
        return np.zeros(tuple(new_dims)), np.zeros(tuple(new_dims))
    if os.stat(filename).st_size < 100:
        return np.zeros(tuple(new_dims)), np.zeros(tuple(new_dims))
    img = skimage.io.imread( filename )
    img = scipy.misc.imresize(img, tuple(new_dims), interp='nearest') 
    mask = img > 0.1
    mask = mask.astype(float)
    img[img == 0] = 1
    img = img - 1
    prior_factor = np.load(os.path.join(root,'lib', 'data', 'semseg_prior_factor.npy'))
    rebalance = prior_factor[img]
    mask = mask * rebalance
    return img, mask 

def segment_pixel_sample( template, new_dims, num_pixels, domain, mask=None ):
    '''
    Segmentation

    Returns:
    --------
        pixels: size num_pixels x 3 numpy array
    '''
    if template.split('/')[-1].isdigit(): 
        template = template.split('/')
        if template[0] == '':
            template[0] = os.sep
        template[-1] = "point_{point_id}_view_{view_id}_domain_{{domain}}.png".format(
                    point_id=template[-2], view_id=template[-1])
        template[-2] = '{domain}'
        template = os.path.join(*template)
    filename = template.format( domain=domain )
    
    img = skimage.io.imread( filename )
    img = scipy.misc.imresize(img, tuple(new_dims), interp='nearest') 

    if mask is None:
        all_pixels = list(itertools.product(range(img.shape[0]), range(img.shape[1])))
        pixs = random.sample(all_pixels, num_pixels)
    else:
        valid_pixels = list(zip(*np.where(np.squeeze(mask[:,:,0]) != 0))) 
        pixs = random.sample(valid_pixels, num_pixels)

    pix_segment = [list(i) + [int(img[i[0]][i[1]])] for i in pixs]
    pix_segment = np.array(pix_segment)
    return pix_segment


#########################
# Mask fns              #
#########################
def mask_out_value( img, masked_value ):
    '''
        Args:
            img: A (H, W, C) np array
            masked_value: An array where when the image has img[h,w] == masked_value, 
                the mask will be 0

        Returns:
            mask: A (H, W) np array
    '''
    h, w, c = img.shape
    mask = reduce( 
        np.logical_or, 
        [ img[:, :, i] != val for i, val in enumerate( masked_value ) ] )
    if len( mask.shape ) == 2:
        mask = mask[:, :, np.newaxis].astype( np.float32 )
    return np.broadcast_to(mask, img.shape )

def mask_if_channel_le( img, threshhold, channel_idx, broadcast_to_shape=None, broadcast_to_dim=None ):
    '''
        Returns a mask that masks an entire pixel iff the channel
            specified has values le a specified value
    '''
    h, w, c = img.shape
    mask = ( img[:, :, channel_idx] > threshhold ) # keep if gt
    if len( mask.shape ) == 2:
        mask = mask[:, :, np.newaxis].astype( np.float32 )
    if broadcast_to_shape is not None:
        return np.broadcast_to( mask, broadcast_to_shape )
    elif broadcast_to_dim is not None:
        return np.broadcast_to( mask, [h,w,broadcast_to_dim])
    else:
        return np.broadcast_to( mask, img.shape )

def mask_if_channel_ge( img, threshhold, channel_idx, broadcast_to_shape=None, broadcast_to_dim=None ):
    '''
        Returns a mask that masks an entire pixel iff the channel
            specified has values ge a specified value
    '''
    h, w, c = img.shape
    mask = ( img[:, :, channel_idx] < threshhold ) # keep if lt
    if len( mask.shape ) == 2:
        mask = mask[:, :, np.newaxis].astype( np.float32 )
    if broadcast_to_shape is not None:
        return np.broadcast_to( mask, broadcast_to_shape )
    elif broadcast_to_dim is not None:
        return np.broadcast_to( mask, [h,w,broadcast_to_dim])
    else:
        return np.broadcast_to( mask, img.shape )

########################
# Enqueuing operations
########################
def create_input_placeholders( cfg ):
    """
        Builds placeholder Tensors and Ops for loading and enqueuing
            data.

        Args:
            cfg: A Dict of configuration settings from some config.py

        Returns:
            placeholders: ( input_data, target_data )
    """          
    # placeholder shapes
    if 'is_discriminative' in cfg:
        # make placeholders
        if 'num_input' in cfg and cfg['num_input'] > 1:
            input_shape = [cfg['num_input'], cfg['input_dim'][0], cfg['input_dim'][1], cfg['input_num_channels'] ]
        else:
            input_shape = [ cfg['input_dim'][0], cfg['input_dim'][1], cfg['input_num_channels'] ]
    else:
        input_shape = [ cfg['input_dim'][0], cfg['input_dim'][1], cfg['input_num_channels'] ]

    if 'is_discriminative' in cfg or 'only_target_discriminative' in cfg and cfg['only_target_discriminative']:
        if type(cfg['target_dim']) is int:
            if cfg['target_dim'] > 1:
                target_shape = [ cfg['target_dim'] ]
            else:
                target_shape = []
        else:
            target_shape = [ cfg['target_dim'][0], cfg['target_dim'][1] ]
    else:
        target_shape = [ cfg['target_dim'][0], cfg['target_dim'][1], cfg['target_num_channels'] ]

    # make placeholders
    input_placeholder = tf.placeholder( cfg['input_dtype'], shape=input_shape,
        name='input_placeholder')
    target_placeholder = tf.placeholder( cfg['target_dtype'], shape=target_shape,
        name='target_placeholder' )
    return input_placeholder, target_placeholder


def create_input_placeholders_and_ops( cfg ):
    """
        Builds placeholder Tensors and Ops for loading and enqueuing
            data.

        Args:
            cfg: A Dict of configuration settings from some config.py

        Returns:
            placeholders: ( input_data, target_data )
            batches: ( input_batch (WxHxCxN), target_batch (W'xH'xC'xN) )
            load_and_enqueue: A reference to the function in this file.close
            enqueue_op: A TF op that takes { input_placeholder, target_placeholder}
    """   
    if 'create_input_placeholders_and_ops_fn' in cfg:
        return cfg['create_input_placeholders_and_ops_fn'](cfg)
    try:  
        idx_shape = []

        loading_fn = load_and_enqueue   
        if 'is_discriminative' in cfg:
            # make placeholders
            if 'num_input' in cfg and cfg['num_input'] > 1:
                input_shape = [cfg['num_input'], cfg['input_dim'][0], cfg['input_dim'][1], cfg['input_num_channels'] ]
                if 'single_filename_to_multiple' not in cfg:
                    loading_fn = load_and_enqueue_multiple
                    idx_shape = [ cfg['num_input'] ]
            else:
                input_shape = [ cfg['input_dim'][0], cfg['input_dim'][1], cfg['input_num_channels'] ]
        else:   
            # placeholder shapes
            input_shape = [ cfg['input_dim'][0], cfg['input_dim'][1], cfg['input_num_channels'] ]

        
        if 'is_discriminative' in cfg or 'only_target_discriminative' in cfg and cfg['only_target_discriminative']:
            if type(cfg['target_dim']) is int:
                if cfg['target_dim'] > 1:
                    target_shape = [ cfg['target_dim'] ]
                else:
                    target_shape = [] 
            else:
                target_shape = [ cfg['target_dim'][0], cfg['target_dim'][1] ]
        else:
            target_shape = [ cfg['target_dim'][0], cfg['target_dim'][1], cfg['target_num_channels'] ]

        # make placeholders
        input_placeholder = tf.placeholder( cfg['input_dtype'], shape=input_shape, name='input_placeholder')
        target_placeholder = tf.placeholder( cfg['target_dtype'], shape=target_shape, name='target_placeholder' )

        mask_dtype = tf.float32
        if 'mask_fn' in cfg or 'depth_mask' in cfg and cfg['depth_mask']:
            mask_shape = target_shape
            if 'only_target_discriminative' in cfg:
                mask_shape = []
        elif 'mask_by_target_func' in cfg and cfg['mask_by_target_func']:
            if type(cfg['target_dim']) == int:
                mask_shape = [1]
            else:
                mask_shape = [ cfg['target_dim'][0], cfg['target_dim'][1] ]
        else:
            mask_shape = []
        mask_placeholder = tf.placeholder( tf.float32, shape=mask_shape, name='mask_placeholder' )

        data_idx_dtype = tf.int32
        data_idx_placeholder = tf.placeholder( tf.int32, name='data_idx_placeholder', shape= idx_shape )

        # create queue, enqueue, and dequeue ops
        q = tf.FIFOQueue( 
                capacity=cfg['inputs_queue_capacity'], 
                dtypes=[ cfg['input_dtype'], cfg['target_dtype'], mask_dtype, data_idx_dtype ],
                shapes=[ input_shape, target_shape, mask_shape, idx_shape ]  ) # mask shape = target shape
        tf_summary_scalar( 
                'queue/{0}/fraction_of_{1}_full'.format( q.name, cfg['inputs_queue_capacity'] ),
                tf.to_float( q.size() ) * ( 1. / cfg['inputs_queue_capacity'] ) )

        enqueue_op = q.enqueue( [ input_placeholder, target_placeholder, mask_placeholder, data_idx_placeholder ] )
        input_batch, target_batch, mask_batch, data_idx_batch = q.dequeue_many( cfg['batch_size'] )

        if 'tf_preprocessing_fn' in cfg:
            input_batch, target_batch, mask_batch = cfg['tf_preprocessing_fn'](
                        input_batch=input_batch, target_batch=target_batch, mask_batch=mask_batch, **cfg )
        
        batches = ( input_batch, target_batch, mask_batch, data_idx_batch )
        placeholders = ( input_placeholder, target_placeholder, mask_placeholder, data_idx_placeholder )
        return placeholders, batches, loading_fn, enqueue_op
    except:
        tb.print_exc()
        raise


def create_filename_enqueue_op( cfg ):
    """
        Builds placeholder Tensors and Ops for loading and enqueuing
            filenames to a queue.

        Args:
            cfg: A Dict of configuration settings from some config.py

        Returns:
            A Dict containing
                data_idx_dtype: tf.int32
                data_idx_placeholder: A scalar of type data_idx_dtype
                enqueue_op: A TF op that takes { data_idx_placeholder }
                dequeue_op: A TF op that returns an instantiated data_idx_placeholder
    """          
    return_dict = {}
    return_dict[ 'data_idx_dtype' ] = tf.int32
    if 'num_input' in cfg and cfg['num_input'] > 1 and 'single_filename_to_multiple' not in cfg:
        idx_shape = [ cfg['num_input'] ]
    else:
        idx_shape = []
    return_dict[ 'data_idx_placeholder' ] = tf.placeholder( 
                return_dict[ 'data_idx_dtype' ], 
                name='data_idx_placeholder', 
                shape=idx_shape )

    # create queue, enqueue, and dequeue ops
    capacity = cfg['inputs_queue_capacity'] * 4
    q = tf.FIFOQueue( 
            capacity=capacity, 
            dtypes=[ return_dict[ 'data_idx_dtype' ] ],
            shapes=[ idx_shape ],
            name='filename_fifoqueue' ) # mask shape = target shape
    tf_summary_scalar( 
            'queue/{0}/fraction_of_{1}_full'.format( q.name, capacity ),
            tf.to_float( q.size() ) * ( 1. / capacity ) )
    return_dict[ 'queue' ] = q
    return_dict[ 'enqueue_op' ] = q.enqueue( [ return_dict[ 'data_idx_placeholder' ] ] )
    return_dict[ 'dequeue_op' ] = q.dequeue()
    return return_dict


def get_filepaths_list( filenames_filepath ):
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
    if not isinstance(data_sources, dict):
        raise ValueError('List of filenames to file-list should be a dictionary')
    if 'unit_size' not in data_sources:
        data_sources['unit_size'] = 10000000
    if not all(key in data_sources for key in ['total_size', 'unit_size', 'filename_list']):
        raise ValueError('data_sources file should specify total_size and filename_list')
    if not isinstance(data_sources['filename_list'] , (list, tuple)):
        raise ValueError('filename_list should be either a list or a tuple')
    return data_sources

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

def load_and_enqueue( seed, sess, supervisor, input_filepaths, step, unit_size, num_samples_epoch,
        input_placeholder, target_placeholder, mask_placeholder, data_idx_placeholder, rs_dim,
        enqueue_op, is_training, cfg ):
    """
        Continuously loads data and enqueues it into a data queue. This 
        will continue until coord.should_stop() is true. This is the updated version.

        Takes in a list of input filename files, it will go through each file.

        Args:
            sess: A TF session
            supervisor: A TF coordinator/supervisor (or something implementing should_stop())
            input_filepaths: A List of a list of all filepaths for the data
            seed: i-th thread (total $step$ threads) only loads file with index x % $step$ = i
            step: how many total threads are there
            num_samples_epoch: how many sampels are there per epoch
            input_placeholder: A placeholder Tensor for input data
            target_placeholder: A placholder Tensor for target data
            data_idx_placeholder: A placeholder Scalar
            enqueue_op: A TF op that takes { input_placeholder, target_placeholder, data_idx }
            cfg: A Dict of configuration settings from some config.py

        Returns:
            Nothing. Data is in the queue
    """
    try:
        count = seed
        if unit_size * len(input_filepaths) < num_samples_epoch or unit_size * (len(input_filepaths) - 1) >= num_samples_epoch: 
            # something went wrong here:
            raise ValueError('Epoch size is wrong, please recheck')

        # Load first filename list file
        curr_split = 0
        curr_filename_list = load_filepaths_list( input_filepaths[curr_split] )
        
        num_samples_epoch = num_samples_epoch - num_samples_epoch % cfg['batch_size']
        while True:
            # try: # need try-catch because some data files are corrupt
            epoch_idx = count % num_samples_epoch
            if 'randomize' in cfg and cfg['randomize']:
                # Using a random sample from filename list is not recommended
                next_idx = np.random.randint( len( curr_filename_list) )
                #next_idx = (next_idx // step) * step + seed
            else: 
                next_idx = epoch_idx % unit_size 

            cur_input_fname = curr_filename_list[ next_idx ].decode('UTF-8')

            if cfg['dataset_dir']:
                cur_input_fname = os.path.join( cfg['dataset_dir'], cur_input_fname )
            
            try:
                import pdb
                #pdb.set_trace()
                img_in, img_target, mask_target = cfg['preprocess_fn']( cur_input_fname, cfg, is_training )
            except FileNotFoundError:
                continue 
            except:
                print("Error loading {}".format(cur_input_fname))

            if supervisor.should_stop(): # Put the check at the end so that we don't use a closed sess
                break
            if count / num_samples_epoch > cfg['num_epochs']:
                break

            sess.run( enqueue_op, feed_dict={ input_placeholder: img_in,
                                                target_placeholder: img_target, 
                                                mask_placeholder: mask_target, 
                                                data_idx_placeholder: next_idx } )
            count += step
            next_split = (count % num_samples_epoch) // unit_size 
            if next_split is not curr_split:
                # Using Next Split
                curr_split = next_split
                curr_filename_list = load_filepaths_list( input_filepaths[curr_split] )

    except tf.errors.CancelledError:
        pass
    except:
        tb.print_exc()
        raise
    tf.logging.info('closing thread...')


def enqueue_filenames( sess, supervisor, input_filepaths, 
        data_idx_placeholder, enqueue_op, is_training, cfg ):
    """
        Continuously loads data and enqueues it into a data queue. This 
        will continue until coord.should_stop() is true.

        Args:
            sess: A TF session
            supervisor: A TF coordinator/supervisor (or something implementing should_stop())
            input_filepaths: A List of all filepaths for the data
            data_idx_placeholder: A placeholder Tensor which will contain an idx into input_filepaths
            enqueue_op: A TF op that takes { input_placeholder, target_placeholder, data_idx }
            is_training: A bool
            cfg: A Dict of configuration settings from some config.py

        Returns:
            Nothing. Data is in the queue
    """       
    try:
        count = 0
        while True:
            # try: # need try-catch because some data files are corrupt
            if 'randomize' in cfg and cfg['randomize']:
                next_idx = np.random.randint( len( input_filepaths ) )
            else: 
                next_idx = count % len( input_filepaths )

            if supervisor.should_stop(): # Put the check at the end so that we don't use a closed sess
                break
            
            sess.run( enqueue_op, feed_dict={ data_idx_placeholder: next_idx} )
            count += 1
    except tf.errors.CancelledError:
            pass
    print('closing filename thread...')

