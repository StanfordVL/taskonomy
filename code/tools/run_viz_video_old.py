from __future__ import absolute_import, division, print_function

import argparse
import importlib
import itertools

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

from PIL import Image, ImageDraw, ImageFont

target_tasks = "autoencoder curvature denoise edge2d edge3d keypoint2d keypoint3d reshade rgb2depth rgb2mist rgb2sfnorm segment25d segment2d vanishing_point_well_defined segmentsemantic_rb class_1000 class_places colorization jigsaw impainting_whole"
list_of_tasks = target_tasks.split(" ")
ON_TEST_SET = True 
IN_TRAIN_MODE = False

parser = argparse.ArgumentParser(description='Viz Single Task')
parser.add_argument('--idx', dest='idx',
                    help='Task to run', type=int)
parser.set_defaults(idx=-1)

parser.add_argument('--n-parallel', dest='n_parallel',
                    help='Number of models to run in parallel', type=int)
parser.set_defaults(n_parallel=1)

parser.add_argument('--task', dest='task')
parser.set_defaults(task='NONE')

parser.add_argument('--config', dest='config')
parser.set_defaults(config='NOT_SET')

tf.logging.set_verbosity(tf.logging.ERROR)

def remove_dups(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def run_to_task(task_to):

    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars
    import models.architectures as architectures
    from   data.load_ops import resize_rescale_image
    import utils
    from   data.task_data_loading import load_and_specify_preprocessors_for_representation_extraction
    import lib.data.load_ops as load_ops
    import pdb
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

    with open('/home/ubuntu/task-taxonomy-331b/lib/data/places_class_names.txt', 'r') as fp:
        synset_places = [x.rstrip()[4:-1] for x,y in zip(fp.readlines(), select) if y == 1.]

    
    tf.logging.set_verbosity(tf.logging.ERROR)
   
    args = parser.parse_args()
    if args.task is not 'NONE':
        args.idx = list_of_tasks.index(args.task)
    for idx, task in enumerate(list_of_tasks):
        if idx != args.idx and args.idx != -1:
            continue
        if task == 'class_places':
            synset = synset_places
        elif task == 'class_1000':
            synset = synset_1000
        print("Doing {task}".format(task=task))
        general_utils = importlib.reload(general_utils)
        tf.reset_default_graph()
        training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

        # task = '{f}__{t}__{hs}'.format(f=task_from, t=task_to, hs=args.hs)
        CONFIG_DIR = '/home/ubuntu/task-taxonomy-331b/experiments/final/{TASK}'.format(TASK=task)

        ############## Load Configs ##############
        cfg = utils.load_config( CONFIG_DIR, nopause=True )
        RuntimeDeterminedEnviromentVars.register_dict( cfg )
        split_file = os.path.join('/home/ubuntu/task-taxonomy-331b/assets/aws_data/', 'video2_info.pkl')
        cfg['train_filenames'] = split_file
        cfg['val_filenames'] = split_file
        cfg['test_filenames'] = split_file 

        cfg['num_epochs'] = 2
        cfg['randomize'] = False
        root_dir = cfg['root_dir']
        cfg['num_read_threads'] = 1
        print(cfg['log_root'])
        cfg['model_path'] = os.path.join(
                cfg['log_root'],
                task,
                'model.permanent-ckpt'
            )

        print( cfg['model_path'])
        if cfg['model_path'] is None:
            continue
        cfg['dataset_dir'] = '/home/ubuntu'
        cfg['preprocess_fn'] = load_and_specify_preprocessors_for_representation_extraction
        ############## Set Up Inputs ##############
        # tf.logging.set_verbosity( tf.logging.INFO )
        inputs = utils.setup_input( cfg, is_training=ON_TEST_SET, use_filename_queue=False ) # is_training determines whether to use train/validaiton
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        start_time = time.time()
        # utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

        ############## Set Up Model ##############
        model = utils.setup_model( inputs, cfg, is_training=IN_TRAIN_MODE )
        m = model[ 'model' ]
        model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )

        ############## Start dataloading workers ##############
        data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn( 
            inputs, cfg, is_training=ON_TEST_SET, use_filename_queue=False )

        prefetch_threads = threading.Thread(
            target=data_prefetch_init_fn,
            args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
        prefetch_threads.start()
       
        list_of_fname = np.load('/home/ubuntu/task-taxonomy-331b/assets/aws_data/video2_fname.npy')
        import errno

        try:
            os.mkdir('/home/ubuntu/{}'.format(task))
            os.mkdir('/home/ubuntu/{}/vid1'.format(task))
            os.mkdir('/home/ubuntu/{}/vid2'.format(task))
            os.mkdir('/home/ubuntu/{}/vid3'.format(task))
            os.mkdir('/home/ubuntu/{}/vid4'.format(task))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        curr_comp = np.zeros((3,64))
        curr_fit_img = np.zeros((256,256,3))
        embeddings = []
        ############## Run First Batch ##############

        for step_num in range(inputs['max_steps'] - 1):
        #for step_num in range(1):
            #if step_num > 0 and step_num % 20 == 0:
            print(step_num)
            if not hasattr(m, 'masks'):
                ( 
                    input_batch, target_batch, 
                    data_idx, 
                    predicted, loss,
                ) = training_runners['sess'].run( [ 
                    m.input_images, m.targets,
                    model[ 'data_idxs' ], 
                    m.decoder_output, m.total_loss] )
                mask_batch = 1.
            else:
                ( 
                    input_batch, target_batch, mask_batch,
                    data_idx, 
                    predicted, loss,
                ) = training_runners['sess'].run( [ 
                    m.input_images, m.targets, m.masks,
                    model[ 'data_idxs' ], 
                    m.decoder_output, m.total_loss] )

            if task == 'segment2d' or task == 'segment25d':
                from sklearn.decomposition import PCA  
                x = np.zeros((32,256,256,3), dtype='float')
                k_embed = 8
#                 for i in range(predicted.shape[0]):
                    # embedding_flattened = np.squeeze(predicted[i]).reshape((-1,64))
                    # pca = PCA(n_components=3)
                    # pca.fit(embedding_flattened)
                    # min_order = None
                    # min_dist = float('inf')
                    # for order in itertools.permutations([0,1,2]):
                        # reordered = pca.components_[list(order), :]
                        # dist = np.linalg.norm(curr_comp-reordered)
                        # if dist < min_dist:
                            # min_order = list(order)
                            # min_dist = dist
                    # print(min_order)
                    # pca.components_ = pca.components_[min_order, :]
                    # curr_comp = pca.components_
                    # lower_dim = pca.transform(embedding_flattened).reshape((256,256,-1))
                    # lower_dim = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
                    # x[i] = lower_dim
                for i in range(predicted.shape[0]):
                    embedding_flattened = np.squeeze(predicted[i]).reshape((-1,64))
                    embeddings.append(embedding_flattened)
                    if len(embeddings) > k_embed:
                        embeddings.pop(0)
                    pca = PCA(n_components=3)
                    pca.fit(np.vstack(embeddings))
                    min_order = None
                    min_dist = float('inf')
                    copy_of_comp = np.copy(pca.components_)
                    for order in itertools.permutations([0,1,2]):
                        #reordered = pca.components_[list(order), :]
                        #dist = np.linalg.norm(curr_comp-reordered)
                        pca.components_ = copy_of_comp[order, :]
                        lower_dim = pca.transform(embedding_flattened).reshape((256,256,-1))
                        lower_dim = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
                        dist = np.linalg.norm(lower_dim - curr_fit_img)
                        if dist < min_dist:
                            min_order = order 
                            min_dist = dist
                    pca.components_ = copy_of_comp[min_order, :]
                    lower_dim = pca.transform(embedding_flattened).reshape((256,256,-1))
                    lower_dim = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
                    curr_fit_img = np.copy(lower_dim)
                    x[i] = lower_dim
                predicted = x
            if task == 'curvature':
                std = [31.922, 21.658]
                mean = [123.572, 120.1]
                predicted = (predicted * std) + mean
                predicted[:,0,0,:] = 0.
                predicted[:,1,0,:] = 1.
                predicted = np.squeeze(np.clip(predicted.astype(int) / 255., 0., 1. )[:,:,:,0])

            just_rescale = ['autoencoder', 'denoise', 'edge2d', 
                            'edge3d', 'keypoint2d', 'keypoint3d',
                            'reshade', 'rgb2sfnorm']
            if task in just_rescale:
                predicted = (predicted + 1.) / 2.
                predicted = np.clip(predicted, 0., 1.)
                predicted[:,0,0,:] = 0.
                predicted[:,1,0,:] = 1.


            just_clip = ['rgb2depth', 'rgb2mist']
            if task in just_clip:
                predicted[:,0,0,:] = 0.
                predicted[:,1,0,:] = 1.

            if task == 'segmentsemantic_rb':
                label = np.argmax(predicted, axis=-1)
                COLORS = ('white','red', 'blue', 'yellow', 'magenta', 
                        'green', 'indigo', 'darkorange', 'cyan', 'pink', 
                        'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
                        'purple', 'darkviolet')
                rgb = (input_batch + 1.) / 2.
                preds = [color.label2rgb(np.squeeze(x), np.squeeze(y), colors=COLORS, kind='overlay')[np.newaxis,:,:,:] for x,y in zip(label, rgb)]
                predicted = np.vstack(preds) 

            if task in ['class_1000', 'class_places']:
                for file_idx, predict_output in zip(data_idx, predicted):
                    to_store_name = list_of_fname[file_idx].decode('utf-8').replace('video', task)
                    to_store_name = os.path.join('/home/ubuntu', to_store_name)
                    sorted_pred = np.argsort(predict_output)[::-1]
                    top_5_pred = [synset[sorted_pred[i]] for i in range(5)]
                    to_print_pred = "Top 5 prediction: \n {}\n {}\n {}\n {} \n {}".format(*top_5_pred)
                    img = Image.new('RGBA', (400, 200), (255, 255, 255))
                    d = ImageDraw.Draw(img)
                    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf', 25)
                    d.text((20, 5), to_print_pred, fill=(255, 0, 0), font=fnt)
                    img.save(to_store_name, 'PNG')
            else:
                for file_idx, predict_output in zip(data_idx, predicted):
                    to_store_name = list_of_fname[file_idx].decode('utf-8').replace('video', task)
                    to_store_name = os.path.join('/home/ubuntu', to_store_name)
                    scipy.misc.toimage(np.squeeze(predict_output), cmin=0.0, cmax=1.0).save(to_store_name)

        subprocess.call('tar -czvf /home/ubuntu/{t}.tar.gz /home/ubuntu/{t}'.format(t=task), shell=True)
        subprocess.call('aws s3 cp /home/ubuntu/{t}.tar.gz s3://task-preprocessing-512-oregon/video2/'.format(t=task), shell=True)
        subprocess.call('ffmpeg -r 29.97 -f image2 -s 256x256 -i /home/ubuntu/{t}/vid2/020%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p {t}_2.mp4'.format(t=task), shell=True)
        subprocess.call('aws s3 cp {t}_2.mp4 s3://task-preprocessing-512-oregon/video2/'.format(t=task), shell=True)

                

        ############## Clean Up ##############
        training_runners[ 'coord' ].request_stop()
        training_runners[ 'coord' ].join()
        
        # if os.path.isfile(pickle_dir): 
        #     with open(pickle_dir, 'rb') as fp:
        #         all_outputs = pickle.load(fp)
                
        ############## Store to dict ##############
        
        print("Done: {}".format(task))
        # os.system("sudo cp {d} /home/ubuntu/s3/model_log".format(d=pickle_dir))

        ############## Reset graph and paths ##############            
        tf.reset_default_graph()
        training_runners['sess'].close()

    return

if __name__ == '__main__':
    run_to_task(None)
    # with Pool(args.n_parallel) as p:
    #     p.map(run_to_task, list_of_tasks)


