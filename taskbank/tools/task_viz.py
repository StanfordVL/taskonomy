import matplotlib
matplotlib.use('Agg')
import time
from   multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
import subprocess
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import scipy.misc
from skimage import color
import init_paths
from models.sample_models import *
from lib.data.synset import *
import scipy
import skimage
import transforms3d
import math

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import itertools

def load_raw_image_center_crop( filename, color=True ):
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

    img = skimage.io.imread(filename, as_grey=not color)

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
 
    H,W,C = img.shape
    l = min(H,W) // 2
    img = img[H//2 - l:H//2 + l,W//2 - l:W//2 + l,:]
    
    return img

def inpainting_bbox(predicted, to_store_name):
    predicted = np.squeeze(predicted)
    predicted = (predicted + 1.) / 2.
    predicted = np.clip(predicted, 0., 1.)
    im = Image.fromarray(np.uint8(predicted* 255))
    draw = ImageDraw.Draw(im) 
    draw.line( [(64,64), (64,192) ], fill='red', width=5)
    draw.line( [(64,192), (192,192) ], fill='red', width=5)
    draw.line( [(192,64), (192,192) ], fill='red', width=5)
    draw.line( [(64,64), (192,64) ], fill='red', width=5)
    with_bb = np.array(im).astype(float) / 255.
    scipy.misc.toimage(np.squeeze(with_bb), cmin=0.0, cmax=1.0).save(to_store_name)

def classification(predicted, synset, to_store_name):
    predicted = predicted.squeeze()
    sorted_pred = np.argsort(predicted)[::-1]
    top_5_pred = [synset[sorted_pred[i]] for i in range(5)]
    to_print_pred = "Top 5 prediction: \n {}\n {}\n {}\n {} \n {}".format(*top_5_pred)
    img = Image.new('RGBA', (400, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('lib/data/DejaVuSerifCondensed.ttf', 25)
    d.text((20, 5), to_print_pred, fill=(255, 0, 0), font=fnt)
    img.save(to_store_name, 'PNG')

def semseg_single_image( predicted, img, to_store_name ):
    label = np.argmax(predicted, axis=-1)
    COLORS = ('white','red', 'blue', 'yellow', 'magenta', 
            'green', 'indigo', 'darkorange', 'cyan', 'pink', 
            'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
            'purple', 'darkviolet')
    rgb = (img + 1.) / 2.
    preds = [color.label2rgb(np.squeeze(x), np.squeeze(y), colors=COLORS, kind='overlay')[np.newaxis,:,:,:] for x,y in zip(label, rgb)]
    predicted = preds[0].squeeze()
    process_semseg_frame(predicted, to_store_name)

def depth_single_image( predicted, to_store_name ):
    predicted = np.exp(predicted * np.log( 2.0**16.0 )) - 1.0
    predicted = np.log(predicted) / 11.09
    predicted = ( predicted - 0.64 ) / 0.18
    predicted = ( predicted + 1. ) / 2
    predicted[:,0,0,:] = 0.
    predicted[:,1,0,:] = 1.
    scipy.misc.toimage(np.squeeze(predicted), cmin=0.0, cmax=1.0).save(to_store_name)

def curvature_single_image( predicted, to_store_name):
    std = [31.922, 21.658]
    mean = [123.572, 120.1]
    predicted = (predicted * std) + mean
    predicted[:,0,0,:] = 0.
    predicted[:,1,0,:] = 1.
    predicted = np.squeeze(np.clip(predicted.astype(int) / 255., 0., 1. )[:,:,:,0])
    scipy.misc.toimage(np.squeeze(predicted), cmin=0.0, cmax=1.0).save(to_store_name)

def simple_rescale_img( predicted, to_store_name ):
    predicted = (predicted + 1.) / 2.
    predicted = np.clip(predicted, 0., 1.)
    predicted[:,0,0,:] = 0.
    predicted[:,1,0,:] = 1.
    scipy.misc.toimage(np.squeeze(predicted), cmin=0.0, cmax=1.0).save(to_store_name)

def rescale_l_for_display( batch, rescale=True ):
    '''
    Prepares network output for display by optionally rescaling from [-1,1],
    and by setting some pixels to the min/max of 0/1. This prevents matplotlib
    from rescaling the images. 
    '''
    if rescale:
        display_batch = [ ( im.copy() + 1. ) * 50. for im in batch ]
    else:
        display_batch = batch.copy()
    for im in display_batch:
        im[0,0,0] = 1.0  # Adjust some values so that matplotlib doesn't rescale
        im[0,1,0] = 0.0  # Now adjust the min
    return display_batch

def single_img_colorize( predicted, input_batch, to_store_name ):
    maxs = np.amax(predicted, axis=-1)
    softmax = np.exp(predicted - np.expand_dims(maxs, axis=-1))
    sums = np.sum(softmax, axis=-1)
    softmax = softmax / np.expand_dims(sums, -1)
    kernel = np.load('lib/data/pts_in_hull.npy')
    gen_target_no_temp = np.dot(softmax, kernel)

    images_resized = np.zeros([0, 256, 256, 2], dtype=np.float32)
    for image in range(gen_target_no_temp.shape[0]):
        temp = scipy.ndimage.zoom(np.squeeze(gen_target_no_temp[image]), (4, 4, 1), mode='nearest')
        images_resized = np.append(images_resized, np.expand_dims(temp, axis=0), axis=0)
    inp_rescale = rescale_l_for_display(input_batch)
    output_lab_no_temp = np.concatenate((inp_rescale, images_resized), axis=3).astype(np.float64)
    for i in range(input_batch.shape[0]):
        output_lab_no_temp[i,:,:,:] = skimage.color.lab2rgb(output_lab_no_temp[i,:,:,:])
    predicted = output_lab_no_temp
    scipy.misc.toimage(np.squeeze(predicted), cmin=0.0, cmax=1.0).save(to_store_name)

def segmentation_pca( predicted, to_store_name ):
    predicted = np.squeeze(predicted)
    from sklearn.decomposition import PCA  
    x = np.zeros((256,256,3), dtype='float')
    k_embed = 8
    embedding_flattened = predicted.reshape((-1,64))
    pca = PCA(n_components=3)
    pca.fit(np.vstack(embedding_flattened))
    lower_dim = pca.transform(embedding_flattened).reshape((256,256,-1))
    x = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
    scipy.misc.toimage(np.squeeze(x), cmin=0.0, cmax=1.0).save(to_store_name)
    
def show_jigsaw(input_batch, perm, name):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(6, 6))
    outer = gridspec.GridSpec(3, 3)
    outer.update(wspace=0.05, hspace=0.05)
    for i in range(9):
        img = input_batch[i, :, :, :].copy()
        img[0,0,0] = 1.0 
        img[0,1,0] = 0.0 
        ax = plt.subplot(outer[int(perm[i]/3),perm[i]%3])
        ax.axis('off')
        ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
        ax.imshow( np.squeeze(img) )
    fig.savefig(name, dpi=128, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def process_semseg_frame(img, name):   
    labels = ['bottle', 'chair', 'couch', 'plant',
              'bed', 'd.table', 'toilet', 'tv', 'microw', 
              'oven', 'toaster', 'sink', 'fridge', 'book',
              'clock', 'vase']

    colors = ['red', 'blue', 'yellow', 'magenta', 
              'green', 'indigo', 'darkorange', 'cyan', 'pink', 
              'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
              'purple', 'darkviolet']

    N = len(labels)
    interval = 40
    loc = [ interval*16 - interval*x for x in list(range(16)) ]
    #loc = [[x%(interval*9), 1-x//(interval*9)] for x in loc]
    loc = [[350, x] for x in loc]
    data = np.asarray(loc)
    from matplotlib import gridspec

    fig = plt.figure(figsize=[ 6., 4.]) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[4.27,1]) 
    gs.update(left=0.05, right=0.72,wspace=0., hspace=0.)
    ax = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[0])

    #fig, [ax,ax2] = plt.subplots(1, 2, sharex=False)
    ax.margins(0,0)
    ax2.margins(0,0)
    ax.scatter(
        data[:, 0], data[:, 1], marker='s', c=colors, edgecolors=colors, s=100,
        cmap=plt.get_cmap('Spectral'))

    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
        ax.annotate(
            label,
            xy=(x, y), xytext=(-7, -5),
            textcoords='offset points', ha='right', va='bottom', fontsize=min(7,7*6/len(label)))
            #,bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    ax.set_xlim([300, 370])
    ax.set_aspect('equal')
    ax.set_ylim([20, 660])
    ax2.imshow(img)
    ax.set_axis_off()
    ax2.set_axis_off()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    # fig.canvas.draw()
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    fig.savefig(name, dpi=128, bbox_inches='tight', pad_inches=0.0)



def get_synset(task):
    global synset
    synset_1000 = [" ".join(i.split(" ")[1:]) for i in synset]
    select = np.asarray([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
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

    with open('lib/data/places_class_names.txt', 'r') as fp:
        synset_places = [x.rstrip()[4:-1] for x,y in zip(fp.readlines(), select) if y == 1.]
    if task == 'class_places':
        synset = synset_places
    elif task == 'class_1000':
        synset = synset_1000   
    return synset

import transforms3d
def get_pix( point, K, flip=False ):
    if flip:
        point = ( 0.0, 0.0, -1.0 ) - point
    pix = np.dot( K, point )
    pix /= pix[2]
    pix[1] = max(36, min(220, pix[1]))
    pix[0] = max(36, min(220, pix[0]))
    return pix[1], pix[0]

def point_to(p, zone):
    left = zone == 'l'
    x = p[0] - 128
    y = p[1] - 128
    ops = [(x,y), (-x,-y), (-y, x), (y, -x)]
    if left:
        if abs(x) < 15:
            result = [(abs(y), -abs(x))]
        else:
            result = [(a,b) for (a,b) in ops if a <=0 and b>=0] # x -, y +
    else:
        if abs(x) < 15:
            result = [(abs(y), abs(x))]
        else:
            result = [(a,b) for (a,b) in ops if a >=0 and b>=0] # x +, y +
    result = (result[0][0] + 128, result[0][1] + 128)
    return result
        

def get_K(resolution, fov):
    resolution, _ = resolution
    focal_length = 1. / ( 2 * math.tan( fov / 2. ) ) 
    focal_length *= resolution
    offset = resolution /2.
    K = np.array(
        ((   focal_length,    0, offset),
        (    0  ,  focal_length, offset),
        (    0  ,  0,      1       )), dtype=np.float64)
    K[:,1] = -K[:,1]
    K[:,2] = -K[:,2]
    return K

def plot_vanishing_point(predicted, input_batch_display, name, verbose=False):
    resolution = 256
    data = { 'resolution': ( resolution, resolution ), 
                'points': [], 
                'room_layout': -1, 
                'room_type':'test', 
                'name':'test' }
    fov = 1.5
    K = get_K((resolution, resolution), fov)
    center_point = ( 0.0, 0.0, -1.0 )
    from PIL import Image, ImageDraw
    y = [ get_pix( p[:3] + center_point, K )    for i,p in enumerate(predicted.reshape(3,3)) ]
    y = [ [ p[ 1 ] , p[ 0 ] ] for p in y ]
    y = np.asarray(y)    
    rescaled_input = input_batch_display * 255
#     im = Image.fromarray(np.uint8(rescaled_input))
#     draw = ImageDraw.Draw(im) 
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(input_batch_display)
    
    # y[0] = point_to(y[0], 'l')
    # y[1] = point_to(y[1], 'r')
    # y[0] = (256-y[0][0], 256-y[0][1])
    # y[1] = (256-y[1][0], 256-y[1][1])
    # y[2] = (128 - abs(y[2][0] - 128), y[2][1])
    color = ['r', 'g', 'b']
    for i in [0,1,2]:
        arr = ax.arrow(128, 128, y[i][0]-128, y[i][1]-128, width=8, head_width=15, head_length=30, fc=color[i], ec=color[i])
        ax.add_patch(arr)
    for i in [0,1,2]:
        arr = ax.arrow(128, 128, 128-y[i][0], 128-y[i][1], width=8, head_width=15, head_length=30, fc=color[i], ec=color[i])
        ax.add_patch(arr)
        
        #draw.line( [(128,128), y[i] ], fill='red', width=5)
        #norm = math.sqrt( (y[i][0] - 128)**2 + (y[i][1] - 128)**2 )
        #mid = (128 + (y[i][0] - 128) * 100 / norm, 128 + (y[i][1] - 128) * 50 / norm)

    # fig.canvas.draw()
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # return skimage.img_as_float(data)
    fig.savefig(name, dpi = 256) 
    
def plot_vanishing_point_smoothed(predicted, input_batch_display, name, prev, verbose=False):
    resolution = 256
    data = { 'resolution': ( resolution, resolution ), 
                'points': [], 
                'room_layout': -1, 
                'room_type':'test', 
                'name':'test' }
    fov = 1.5
    K = get_K((resolution, resolution), fov)
    center_point = ( 0.0, 0.0, -1.0 )
    from PIL import Image, ImageDraw
    y = [ get_pix( p[:3] + center_point, K )    for i,p in enumerate(predicted.reshape(3,3)) ]
    y = [ [ p[ 1 ] , p[ 0 ] ] for p in y ]
    y = np.asarray(y)
    y = y - 128.
    if len(prev) >= 5:
        y = y * 0.66 + prev[0] * 0.22 + prev[1] * 0.07 + + prev[2] * 0.02 + + prev[3] * 0.02 + + prev[4] * 0.01
    rescaled_input = input_batch_display * 255
#     im = Image.fromarray(np.uint8(rescaled_input))
#     draw = ImageDraw.Draw(im) 
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(input_batch_display)
    
    color = ['r', 'g', 'b']
    for i in [0,1,2]:
        arr = ax.arrow(128, 128, y[i][0], y[i][1], width=8, head_width=15, head_length=30, fc=color[i], ec=color[i])
        ax.add_patch(arr)
    for i in [0,1,2]:
        arr = ax.arrow(128, 128, -y[i][0], -y[i][1], width=8, head_width=15, head_length=30, fc=color[i], ec=color[i])
        ax.add_patch(arr)
        
        #draw.line( [(128,128), y[i] ], fill='red', width=5)
        #norm = math.sqrt( (y[i][0] - 128)**2 + (y[i][1] - 128)**2 )
        #mid = (128 + (y[i][0] - 128) * 100 / norm, 128 + (y[i][1] - 128) * 50 / norm)

    # fig.canvas.draw()
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # return skimage.img_as_float(data)
    fig.savefig(name, dpi = 256)
    return y
    
# Extracts the camera corners and edge lengths from room layout labels
import itertools
def get_cam_corners_and_edge( input_array ):
    center = input_array[:3]
    edge_lengths = input_array[-3:]
    axis = transforms3d.euler.euler2mat( *input_array[3:6], axes='sxyz' )
    y = axis[0] * edge_lengths[0] / 2
    x = axis[1] * edge_lengths[1] / 2
    z = axis[2] * edge_lengths[2] / 2
    corners_for_cam = np.empty((8,3))
    corners_for_cam[0] = center - x - y - z
    corners_for_cam[1] = center - x - y + z
    corners_for_cam[2] = center - x + y - z
    corners_for_cam[3] = center - x + y + z
    corners_for_cam[4] = center + x - y - z
    corners_for_cam[5] = center + x - y + z
    corners_for_cam[6] = center + x + y - z
    corners_for_cam[7] = center + x + y + z
    return corners_for_cam, edge_lengths
def get_cam_corners_and_edge_ordered( input_array ):
    center = input_array[:3]
    edge_lengths = input_array[-3:]
    axis = transforms3d.euler.euler2mat( *input_array[3:6], axes='sxyz' )
    y = axis[0] * edge_lengths[0] / 2
    x = axis[1] * edge_lengths[1] / 2
    z = axis[2] * edge_lengths[2] / 2
    corners_for_cam = np.empty((8,3))
    corners_for_cam[0] = center - x + y - z
    corners_for_cam[1] = center + x + y - z
    corners_for_cam[2] = center + x - y - z
    corners_for_cam[3] = center - x - y - z
    corners_for_cam[4] = center - x + y + z
    corners_for_cam[5] = center + x + y + z
    corners_for_cam[6] = center + x - y + z
    corners_for_cam[7] = center - x - y + z
    return corners_for_cam, edge_lengths
def permute_orig_cols_display( array ):
    return np.stack( [array[:,0], array[:,2], array[:,1]], axis=1 )
def check_if_point_in_fustrum( point, fov ):
    return all( [np.abs( math.atan( coord / point[2] ) ) < fov/2. for coord in point[:2]] )
def get_corner_idxs_in_view( corners, fov ):
    in_view = []
    for idx, point in enumerate( corners ):
        if check_if_point_in_fustrum( point, fov ):
            in_view.append( idx )
    return in_view

def plot_bb_c( pred_corners, pred_edge, corner_idx_in_view_pred, ax=None ):
    if ax is None:
        ax = plt
    dark_edge = [(0,1),(1,2),(2,3),(0,3)]
    mid_edge = [(0,4),(1,5),(2,6),(3,7)]
    light_edge = [(4,5),(5,6),(6,7),(0,7)]
    for (s_idx, s), (e_idx, e) in itertools.combinations( enumerate(pred_corners), 2 ):
        if any( [np.isclose( np.linalg.norm( s-e ), el, atol=1e-04 ) for el in pred_edge] ):
            if min(s_idx, e_idx) < 4 and max(s_idx, e_idx) < 4:
                c = (0.54,0,0)
            elif min(s_idx, e_idx) < 4 and max(s_idx, e_idx) > 3:
                c = (0.77, 0,0)
            else:
                c = 'r'

            ax.plot3D(*zip(s, e), color=c, linewidth=5)
    return ax
    
def plot_bb( pred_corners, pred_edge, corner_idx_in_view_pred, ax=None ):
    if ax is None:
        ax = plt
    for (s_idx, s), (e_idx, e) in itertools.combinations( enumerate(pred_corners), 2 ):
        if any( [np.isclose( np.linalg.norm( s-e ), el, atol=1e-04 ) for el in pred_edge] ):
            ax.plot3D(*zip(s, e), color='r', linewidth=5)
    return ax
def plot_points_with_bb( pred_corners, pred_edge, cube_only=False, fov=None, space='camera', 
                        fig=None, subplot=(1,1,1) ):
    is_camera_space = space.lower()=='camera'
    in_view_pred = get_corner_idxs_in_view( pred_corners, fov )
    pred_corners = permute_orig_cols_display( pred_corners )
    total_corners = pred_corners
    mins = np.min( total_corners, axis=0 )
    maxes = np.max( total_corners, axis=0)
    largest_range = (maxes - mins).max()   
    #axis_ranges = [[m, m + largest_range] for m in mins ]
    if cube_only:
        axis_ranges = [[-6, 6], [-6, 6], [-6, 6]]
    else:
        axis_ranges = [[-6, 6], [-8, 1.5], [-1.2, 7]]
    axes = ['x', 'z', 'y'] if space.lower() == 'camera' else ['x', 'y', 'z']
    axis_idx = {v:k for k,v in enumerate(axes)}    
    from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
    ax = fig.add_subplot(*subplot, projection='3d') 
    ax._axis3don = False
    ax.set_xlim(axis_ranges[ axis_idx['x'] ])
    ax.set_zlim(axis_ranges[ axis_idx['y'] ])
    ax.set_ylim(axis_ranges[ axis_idx['z'] ])
    ax.set_xlabel(axes[0], fontsize=12)
    ax.set_ylabel(axes[1], fontsize=12)
    ax.set_zlabel(axes[2], fontsize=12)
    plot_bb_c( pred_corners, pred_edge, in_view_pred, ax=ax)
    if not cube_only:
        ax.scatter(0,0,0, zdir='r', c='m', s=50)
    theta = np.arctan2(1, 0) * 180 / np.pi
    ax.view_init(30, theta)
    ax.invert_xaxis()
    return ax

# Visualization for room layout    
def plot_room_layout( predicted, img, name, prev, cube_only=False, overlay=False, keep_ratio=True, verbose=False, show_gt_from_json=False, no_pred=False ):
    # Load the input depth image and pose file
    #Make figure
    if len(prev) >= 5:
        predicted = predicted * 0.66 + prev[0] * 0.22 + prev[1] * 0.07 + + prev[2] * 0.02 + + prev[3] * 0.02 + + prev[4] * 0.01
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    predicted[-3:] = np.absolute(predicted[-3:])
    if cube_only:
        predicted[:3] = [0,0,-1]
        if keep_ratio:
            predicted[-3:] = 7 * predicted[-3:] / np.prod(predicted[-3:]) ** (1/3)
        else:
            predicted[-3:] = [8,8,8]
    corners_for_cam_prediction, edge_lengths_pred = get_cam_corners_and_edge_ordered(predicted)
    camera_space_plot = plot_points_with_bb( pred_corners=corners_for_cam_prediction[:,:3], 
                            pred_edge=edge_lengths_pred, cube_only=cube_only,
                            fov=1, space='camera',
                            subplot=(1,1,1), fig=fig)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    x = skimage.transform.resize(data, [256,256])
    if not overlay:
        x = skimage.img_as_float(x)
        scipy.misc.toimage(x, cmin=0.0, cmax=1.0).save(name)
    else:
        from PIL import Image, ImageDraw, ImageFont
        img0s = img*255
        img0s = img0s.astype('uint8')
        xs = x * 255
        xs = xs.astype('uint8')
        rgb = Image.fromarray(img0s).convert("RGBA")
        overlay = Image.fromarray(xs).convert("RGBA")
        datas = overlay.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            elif item[0] == 255 and item[1] > 20 and item[2] > 20 :
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        overlay.putdata(newData)
        lol = overlay.split()[3]
        text_img = Image.new('RGBA', (256,256), (0, 0, 0, 0))
        text_img.paste(rgb, (0,0))
        text_img.paste(overlay, (0,0), mask=overlay)
        fig = plt.figure()
        fig.set_size_inches(1, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(text_img)
        #return skimage.img_as_float(np.array(text_img))
        fig.savefig(name, dpi = 256) 
        plt.close()

        
def cam_pose( predicted, name, is_fixated=False):
    import json
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(5, 5))
    predicted = np.squeeze(predicted)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    from transforms3d import euler
    ax.set_title("Green:Ref. Camera(img2);Red: Pred")

    if is_fixated:
        mean = [ 10.12015407,   8.1103528 ,   1.09171896,   
                1.21579016, 0.26040945,  10.05966329]
        std = [ -2.67375523e-01,  -1.19147040e-02,   1.14497274e-02,
                1.10903410e-03,   2.10509948e-02,  -4.02013549e+00]
    else:
        mean = [ -9.53197445e-03,  -1.05196691e-03,  -1.07545642e-02, 
                2.08785638e-02,  -9.27858049e-02,  -2.58052205e+00]
        std = [ 1.02316223,  0.66477511,  1.03806996,  
                5.75692889,  1.37604962,  7.43157247]
    predicted = predicted * std
    predicted = predicted + mean
    
    cam_origin = np.asarray([0,0,0])
    cam_direction = np.asarray([0,0,-1])
    
    translation = predicted[3:]  
    rotation = euler.euler2mat(*predicted[:3], axes='sxyz')

    c12_location = np.matmul(rotation, cam_origin) + translation
    c12_direction = np.matmul(rotation, cam_direction) + translation
    
    points = np.vstack([cam_origin, cam_direction, c12_location, c12_direction, c12_direction-c12_location])
    axis_min = np.amin(points, axis=0) 
    axis_max = np.amax(points, axis=0)
    axis_size = (axis_max - axis_min) / 10.
    axis_min = axis_min - axis_size
    axis_max = axis_max + axis_size

    length = 3.
    ratio = 0.5

    ax.quiver(cam_origin[2],
          cam_origin[0],
          cam_origin[1],
          cam_direction[2]-cam_origin[2],
          cam_direction[0]-cam_origin[0],
          cam_direction[1]-cam_origin[1], pivot='tail', arrow_length_ratio=ratio, length=length, colors=[0,1,0])

    ax.quiver(c12_location[2],
          c12_location[0],
          c12_location[1],
          c12_direction[2]-c12_location[2],
          c12_direction[0]-c12_location[0],
          c12_direction[1]-c12_location[1], pivot='tail', arrow_length_ratio=ratio, length=length, colors=[1,0,0])
    
    axis_mid = (axis_min + axis_max) / 2.
    axis_length = (axis_max - axis_min)
    axis_length[axis_length < 5.] = 5.
    axis_min = axis_mid - axis_length/2.
    axis_max = axis_mid + axis_length/2.
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Y axis')

    ax.set_xlim([axis_min[2], axis_max[2]])
    ax.set_ylim([axis_min[0], axis_max[0]])
    ax.set_zlim([axis_min[1], axis_max[1]])

    
    theta = np.arctan2(1, 0) * 180 / np.pi
    ax.view_init(60, theta-90)
    fig.savefig(name, dpi = 256) 

def ego_motion(predicted, name):
    import json
    from mpl_toolkits.mplot3d import Axes3D

    pose12 = np.squeeze(predicted[0])
    pose23 = np.squeeze(predicted[-1])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("Green:Ref. Camera(img2);Red--img1;Blue--img3")

    from mpl_toolkits.mplot3d import Axes3D
    from transforms3d import euler
    fixated_std  = np.asarray([ 10.12015407,   8.1103528 ,   1.09171896,   1.21579016,
         0.26040945,  10.05966329])
    fixated_mean = np.asarray([ -2.67375523e-01,  -1.19147040e-02,   1.14497274e-02,
         1.10903410e-03,   2.10509948e-02,  -4.02013549e+00])
    
    pose12 = pose12 * fixated_std
    pose12 = pose12 + fixated_mean
    pose23 = pose23 * fixated_std
    pose23 = pose23 + fixated_mean
    
    cam_origin = np.asarray([0,0,0])
    cam_direction = np.asarray([0,0,-1])
    
    # Calculating Camera 1's world refernce position
    translation_12 = pose12[3:]  
    rotation_12 = euler.euler2mat(*pose12[:3], axes='sxyz')
        
    c12_location = np.matmul(rotation_12, cam_origin) + translation_12
    c12_direction = np.matmul(rotation_12, cam_direction) + translation_12
    

    # Calculating Camera 3's world refernce position
    translation_32 = -pose23[3:]  
    rotation_32 = np.linalg.inv(euler.euler2mat(*pose23[:3], axes='sxyz'))
        
    c32_location = np.matmul(rotation_32, cam_origin + translation_32 )
    c32_direction = np.matmul(rotation_32, cam_direction + translation_32 )
    
    points = np.vstack([c12_location , c12_direction , c32_location , c32_direction, cam_origin, 
                        cam_direction, c12_direction-c12_location , c32_direction-c32_location])
    axis_min = np.amin(points, axis=0) 
    axis_max = np.amax(points, axis=0)
    axis_size = (axis_max - axis_min) / 10.
    axis_min = axis_min - axis_size
    axis_max = axis_max + axis_size
    length = 3.
    ratio = 0.5

    ax.quiver(cam_origin[2],
          cam_origin[0],
          cam_origin[1],
          cam_direction[2]-cam_origin[2],
          cam_direction[0]-cam_origin[0],
          cam_direction[1]-cam_origin[1], pivot='tail', arrow_length_ratio=ratio, length=length, colors=[0,1,0])

    ax.quiver(c12_location[2],
          c12_location[0],
          c12_location[1],
          c12_direction[2]-c12_location[2],
          c12_direction[0]-c12_location[0],
          c12_direction[1]-c12_location[1], pivot='tail', arrow_length_ratio=ratio, length=length, colors=[1,0,0])
    
    ax.quiver(c32_location[2],
          c32_location[0],
          c32_location[1],
          c32_direction[2]-c32_location[2],
          c32_direction[0]-c32_location[0],
          c32_direction[1]-c32_location[1], pivot='tail', arrow_length_ratio=ratio, length=length, colors=[0,0,1])
    
    
    axis_mid = (axis_min + axis_max) / 2.
    axis_length = (axis_max - axis_min)
    axis_length[axis_length < 5.] = 5.
    axis_min = axis_mid - axis_length/2.
    axis_max = axis_mid + axis_length/2.
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Y axis')

    ax.set_xlim([axis_min[2], axis_max[2]])
    ax.set_ylim([axis_min[0], axis_max[0]])
    ax.set_zlim([axis_min[1], axis_max[1]])

    
    theta = np.arctan2(1, 0) * 180 / np.pi
    ax.view_init(60, theta-90)
    fig.savefig(name, dpi = 256) 
