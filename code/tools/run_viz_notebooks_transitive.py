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
#src_tasks = "autoencoder colorization curvature denoise edge2d edge3d ego_motion fix_pose impainting_whole keypoint2d keypoint3d non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm room_layout segment25d segment2d vanishing_point_well_defined segmentsemantic_rb class_selected class_1000 random"
src_tasks = "/home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1ktransitive/reshade__curvature__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1k/reshade__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1ktransitive/rgb2mist__curvature__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1k/rgb2mist__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1ktransitive/edge3d__curvature__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1k/edge3d__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1ktransitive/rgb2sfnorm__curvature__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1k/rgb2sfnorm__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1ktransitive/rgb2depth__curvature__segment25d__8__unlocked /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/1k/rgb2depth__segment25d__8__unlocked"
list_of_src_tasks = src_tasks.split(" ")
# list_of_tasks = "denoise rgb2depth segment2d".split(" ")
#list_of_tasks = "autoencoder segment2d room_layout".split(" ")

#list_of_tasks = "segment2d".split(" ")
#TRANSFER_TYPE = 'full_taskonomy_beta1'
TRANSFER_TYPE = 'transitive'

parser = argparse.ArgumentParser(description='Viz Transfer')
parser.add_argument('--idx', dest='idx',
                    help='Task to run', type=int)
parser.set_defaults(idx=-1)

parser.add_argument('--hs', dest='hs',
                    help='Hidden size to use', type=int)
parser.set_defaults(hs=8)

parser.add_argument('--arch', dest='arch',
                    help='Transfer architecture')
parser.set_defaults(arch="DO_NOT_REPLACE_TARGET_DECODER")

parser.add_argument('--data', dest='data',
                    help='Transfer architecture')
parser.set_defaults(data="16k")

parser.add_argument('--t', dest='list_of_tasks',
                    help='Transfer architecture')

parser.add_argument('--n-parallel', dest='n_parallel',
                    help='Number of models to run in parallel', type=int)
parser.set_defaults(n_parallel=1)

parser.add_argument('--no-regenerate-data', dest='no_regenerate_data', action='store_true')
parser.set_defaults(no_regenerate_data=False)

parser.add_argument('--second-order', dest='second_order', action='store_true')
parser.set_defaults(second_order=False)

tf.logging.set_verbosity(tf.logging.ERROR)


ON_TEST_SET = False 
IN_TRAIN_MODE = False

ipython_std_out = sys.stdout
# Disable
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



args = parser.parse_args()

list_of_tasks = args.list_of_tasks.split(" ")
pairs = list(itertools.product(list_of_tasks, list_of_tasks))
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
    from importlib import reload
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

        # for arch in ['regular', 'shallow', 'dilated_shallow', 'dilated_regular']:
    arch = args.arch
    data_amount = args.data
    if not args.no_regenerate_data:
    #if False:
        all_outputs = {}
        pickle_dir = 'viz_{task_to}_transfer_{hs}_{arch}.pkl'.format(arch=arch, hs=args.hs, task_to=task_to)
        subprocess.call("aws s3 cp s3://task-preprocessing-512-oregon/visualizations/transitive_viz/viz_{}.pkl {}".format(task_to, pickle_dir), shell=True)
        import os
        if os.path.isfile(pickle_dir):
            with open( pickle_dir, 'rb') as fp:
                all_outputs = pickle.load(fp)

        global list_of_src_tasks 
        for i, task_from in enumerate(list_of_src_tasks):
            task_key = task_from.split('/')[-1]
            if task_key in all_outputs:
                print("{} already exists....\n\n\n".format(task_key))
                continue
            print("Doing from {task_from} to {task_to}".format(task_from=task_key, task_to=task_to))
            general_utils = importlib.reload(general_utils)
            tf.reset_default_graph()
            training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

            
            CONFIG_DIR = task_from
            print(CONFIG_DIR)

            ############## Load Configs ##############
            cfg = utils.load_config( CONFIG_DIR, nopause=True )
            RuntimeDeterminedEnviromentVars.register_dict( cfg )
            if args.second_order and not to_flip_dict[task]:
                cfg['val_representations_file'] = cfg['val_representations_file'][::-1]
            cfg['num_epochs'] = 1
            cfg['randomize'] = False
            root_dir = cfg['root_dir']
            cfg['num_read_threads'] = 1
            cfg['model_path'] = tf.train.latest_checkpoint(
                    os.path.join(
                        cfg['log_root'],
                        'logs',
                        'slim-train'
                        #'time'
                    ))
            # print(cfg['model_path'])
            if cfg['model_path'] is None and task == 'random':
                cfg['model_path'] = tf.train.latest_checkpoint(
                    os.path.join(
                        cfg['log_root'],
                        'logs',
                        'slim-train',
                        'time'
                    ))
            if cfg['model_path'] is None:
                continue

            ############## Set Up Inputs ##############
            # tf.logging.set_verbosity( tf.logging.INFO )
            inputs = utils.setup_input_transfer( cfg, is_training=ON_TEST_SET, use_filename_queue=False ) # is_training determines whether to use train/validaiton
            RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
            RuntimeDeterminedEnviromentVars.populate_registered_variables()
            start_time = time.time()
            # utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

            ############## Set Up Model ##############
            model = utils.setup_model( inputs, cfg, is_training=True )
            m = model[ 'model' ]
            model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )

            ############## Start dataloading workers ##############
            data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn_transfer( 
                inputs, cfg, is_training=ON_TEST_SET, use_filename_queue=False )

            prefetch_threads = threading.Thread(
                target=data_prefetch_init_fn,
                args=( training_runners[ 'sess' ], training_runners[ 'coord' ] ))
            prefetch_threads.start()

            ############## Run First Batch ##############
            ( 
                input_batch, representation_batch, target_batch, 
                data_idx, 
                encoder_output, predicted, loss,
            ) = training_runners['sess'].run( [ 
                m.input_images, m.input_representations, m.decoder.targets,
                model[ 'data_idxs' ], 
                m.encoder_output, m.decoder.decoder_output, m.total_loss] )
            if task_to == 'segment2d' or task_to == 'segment25d':
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
            if task_to == 'segmentsemantic_rb':
                predicted = np.argmax(predicted, axis=-1)
            ############## Clean Up ##############
            training_runners[ 'coord' ].request_stop()
            training_runners[ 'coord' ].join()
            
            # if os.path.isfile(pickle_dir): 
            #     with open(pickle_dir, 'rb') as fp:
            #         all_outputs = pickle.load(fp)
                    
            ############## Store to dict ##############
            to_store = {'data_idx':data_idx, 'output':predicted, 'loss':loss}
            all_outputs[task_key] = to_store
            

            # os.system("sudo cp {d} /home/ubuntu/s3/model_log".format(d=pickle_dir))

            ############## Reset graph and paths ##############            
            tf.reset_default_graph()
            training_runners['sess'].close()
            #print(sys.modules.keys())
            #del sys.modules[ 'config' ]
            sys.path = remove_dups(sys.path)
            print('Current Directory: ', os.getcwd())
            pickle_dir = 'viz_{task_to}_transfer_{hs}_{arch}.pkl'.format(arch=arch, hs=args.hs, task_to=task_to)
            with open( pickle_dir, 'wb') as fp:
                pickle.dump(all_outputs, fp)
            subprocess.call("aws s3 cp {} s3://task-preprocessing-512-oregon/visualizations/transitive_viz/viz_{}.pkl".format(pickle_dir, task_to), shell=True)

    # Run jupyter nb
    print('Running Jupyter Notebooks...')
    #os.makedirs("/home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/transfer_{hs}_{arch}".format(hs=args.hs, arch=arch), exist_ok=True)
    notebook_path = '/home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/Visual_{task_to}'.format(task_to=task_to)
    subprocess.call(
        "jupyter nbconvert \
            --execute {notebook_path}.ipynb \
            --to html \
            --ExecutePreprocessor.kernel_name=python3 \
            --ExecutePreprocessor.timeout=1200 ".format(notebook_path=notebook_path, arch=arch, hs=args.hs, task_to=task_to),
        shell=True)
    subprocess.call("aws s3 cp {}.html s3://task-preprocessing-512-oregon/visualizations/{}/".format(
        notebook_path, TRANSFER_TYPE ), shell=True)


            #--output /home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/temp/Visual_{task_to}.ipynb \

if __name__ == '__main__':
    #with Pool(args.n_parallel) as p:
    #    p.map(run_to_task, list_of_tasks)
    list_of_tasks = args.list_of_tasks.split(" ")
    for task_to in list_of_tasks:
        run_to_task(task_to)
        print(task_to)
