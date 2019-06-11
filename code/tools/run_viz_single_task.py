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


import init_paths
from   models.sample_models import *
target_tasks = "autoencoder colorization curvature denoise edge2d edge3d ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm room_layout segment25d segment2d vanishing_point_well_defined segmentsemantic_rb class_selected class_1000"
list_of_tasks = target_tasks.split(" ")
ON_TEST_SET = True 
IN_TRAIN_MODE = False

parser = argparse.ArgumentParser(description='Viz Single Task')
parser.add_argument('--idx', dest='idx',
                    help='Task to run', type=int)

parser.add_argument('--hs', dest='hs',
                    help='Hidden size to use', type=int)

parser.add_argument('--n-parallel', dest='n_parallel',
                    help='Number of models to run in parallel', type=int)
parser.set_defaults(n_parallel=1)

tf.logging.set_verbosity(tf.logging.ERROR)


ipython_std_out = sys.stdout
# Disabe
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = ipython_std_out
    
# Force Print
def forcePrint(str):
    enablePrint()
    print(str)
    sys.stdout.flush()
    blockPrint()

def remove_dups(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

pairs = list(itertools.product(list_of_tasks, list_of_tasks))


args = parser.parse_args()
idx_to_run = args.idx
if idx_to_run == -1:
    pairs_to_run = pairs
else:
    pairs_to_run = pairs[idx_to_run:idx_to_run+1]


def run_to_task(task_to):

    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars
    import models.architectures as architectures
    from   data.load_ops import resize_rescale_image
    import utils
    from   data.task_data_loading import load_and_specify_preprocessors_for_representation_extraction
    import lib.data.load_ops as load_ops
    tf.logging.set_verbosity(tf.logging.ERROR)

    all_outputs = {}
    pickle_dir = 'viz_output_single_task.pkl'
    import os
    if os.path.isfile(pickle_dir):
        with open( pickle_dir, 'rb') as fp:
            all_outputs = pickle.load(fp)
        
    for task in list_of_tasks:
        if task in all_outputs:
            print("{} already exists....\n\n\n".format(task))
            continue
        print("Doing {task}".format(task=task))
        general_utils = importlib.reload(general_utils)
        tf.reset_default_graph()
        training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

        # task = '{f}__{t}__{hs}'.format(f=task_from, t=task_to, hs=args.hs)
        CONFIG_DIR = '/home/ubuntu/task-taxonomy-331b/experiments/final/{TASK}'.format(TASK=task)

        ############## Load Configs ##############
        cfg = utils.load_config( CONFIG_DIR, nopause=True )
        RuntimeDeterminedEnviromentVars.register_dict( cfg )
        split_file = cfg['test_filenames'] if ON_TEST_SET else cfg['val_filenames']
        cfg['train_filenames'] = split_file
        cfg['val_filenames'] = split_file
        cfg['test_filenames'] = split_file 

        cfg['num_epochs'] = 1
        cfg['randomize'] = False
        root_dir = cfg['root_dir']
        cfg['num_read_threads'] = 1
        print(cfg['log_root'])
        if task == 'jigsaw':
            continue
        cfg['model_path'] = os.path.join(
                cfg['log_root'],
                task,
                'model.permanent-ckpt'
            )

        print( cfg['model_path'])
        if cfg['model_path'] is None:
            continue

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
        
        ############## Run First Batch ##############
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
            for i in range(predicted.shape[0]):
                embedding_flattened = np.squeeze(predicted[i]).reshape((-1,64))
                pca = PCA(n_components=3)
                pca.fit(embedding_flattened)
                lower_dim = pca.transform(embedding_flattened).reshape((256,256,-1))
                lower_dim = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
                x[i] = lower_dim
            predicted = x


        ############## Clean Up ##############
        training_runners[ 'coord' ].request_stop()
        training_runners[ 'coord' ].join()
        
        # if os.path.isfile(pickle_dir): 
        #     with open(pickle_dir, 'rb') as fp:
        #         all_outputs = pickle.load(fp)
                
        ############## Store to dict ##############
        to_store = {
            'input': input_batch,
            'target': target_batch,
            'mask': mask_batch,
            'data_idx':data_idx,
            'output':predicted}
        all_outputs[task] = to_store
        
        print("Done: {}".format(task))
        # os.system("sudo cp {d} /home/ubuntu/s3/model_log".format(d=pickle_dir))

        ############## Reset graph and paths ##############            
        tf.reset_default_graph()
        training_runners['sess'].close()
        try:
            del sys.modules[ 'config' ]
        except:
            pass
        sys.path = remove_dups(sys.path)
        print("FINISHED: {}\n\n\n\n\n\n".format(task))
        pickle_dir = 'viz_output_single_task.pkl'
        with open( pickle_dir, 'wb') as fp:
            pickle.dump(all_outputs, fp)
        try:
            subprocess.call("aws s3 cp {} s3://task-preprocessing-512-oregon/visualizations/".format(pickle_dir), shell=True)
        except:
            subprocess.call("sudo cp {} /home/ubuntu/s3/visualizations/".format(pickle_dir), shell=True)

    return

if __name__ == '__main__':
    run_to_task(None)
    # with Pool(args.n_parallel) as p:
    #     p.map(run_to_task, list_of_tasks)


