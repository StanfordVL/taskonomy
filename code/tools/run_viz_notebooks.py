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
list_of_src_tasks = "autoencoder room_layout rgb2depth segment2d random".split(" ")
list_of_tasks = "autoencoder room_layout rgb2depth segment2d".split(" ")
# list_of_tasks = "autoencoder".split(" ")

ARCH_TYPE = 'dilated_l1l2'
TRANSFER_TYPE = os.path.join('pix_stream', ARCH_TYPE)

parser = argparse.ArgumentParser(description='Viz Transfer')
parser.add_argument('--idx', dest='idx',
                    help='Task to run', type=int)

parser.add_argument('--hs', dest='hs',
                    help='Hidden size to use', type=int)

parser.add_argument('--n-parallel', dest='n_parallel',
                    help='Number of models to run in parallel', type=int)
parser.set_defaults(n_parallel=1)

parser.add_argument('--no-regenerate-data', dest='no_regenerate_data', action='store_true')
parser.set_defaults(no_regenerate_data=False)

tf.logging.set_verbosity(tf.logging.ERROR)


ON_TEST_SET = True
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

    training_regimens = [''] #, '__rt_ft']
    # for arch in ['regular', 'shallow', 'dilated_shallow', 'dilated_regular']:
    for training_regimen in training_regimens:
        if not args.no_regenerate_data:
            all_outputs = {}
            dst_folder_name = ARCH_TYPE + '_' + arch
            for task_from in list_of_src_tasks:
        
                print("Doing from {task_from} to {task_to}".format(task_from=task_from, task_to=task_to))
                if task_from == task_to: 
                    continue
                blockPrint()

                general_utils = importlib.reload(general_utils)
                tf.reset_default_graph()
                training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

                task = '{f}__{t}__{hs}{training_regimen}'.format(
                    f=task_from,
                    t=task_to,
                    hs=args.hs,
                    training_regimen=training_regimen
                )
                CONFIG_DIR = '/home/ubuntu/task-taxonomy-331b/experiments/transfers/{transfer_type}/{TASK}'.format(
                    transfer_type=TRANSFER_TYPE,
                    training_regimen=training_regimen,
                    TASK=task
                )
                print(CONFIG_DIR)
                # exit(0)
                ############## Load Configs ##############
                cfg = utils.load_config( CONFIG_DIR, nopause=True )
                RuntimeDeterminedEnviromentVars.register_dict( cfg )
                split_file = cfg['val_filenames'] if ON_TEST_SET else cfg['train_filenames']
                cfg['train_filenames'] = split_file
                cfg['val_filenames'] = split_file
                cfg['test_filenames'] = split_file 

                cfg['num_epochs'] = 1
                cfg['randomize'] = False
                root_dir = cfg['root_dir']
                cfg['num_read_threads'] = 1
                cfg['model_path'] = tf.train.latest_checkpoint(
                        os.path.join(
                            cfg['log_root'],
                            'logs',
                            'slim-train'
                        ))

                if cfg['model_path'] is None:
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
                inputs = utils.setup_input_transfer( cfg, is_training=False, use_filename_queue=False ) # is_training determines whether to use train/validaiton
                RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
                RuntimeDeterminedEnviromentVars.populate_registered_variables()
                start_time = time.time()
                # utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=False )

                ############## Set Up Model ##############
                model = utils.setup_model( inputs, cfg, is_training=IN_TRAIN_MODE )
                m = model[ 'model' ]
                model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
                ############## Start dataloading workers ##############
                data_prefetch_init_fn = utils.get_data_prefetch_threads_init_fn_transfer( 
                    inputs, cfg, is_training=False, use_filename_queue=False )

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

                ############## Clean Up ##############
                training_runners[ 'coord' ].request_stop()
                training_runners[ 'coord' ].join()
                
                # if os.path.isfile(pickle_dir): 
                #     with open(pickle_dir, 'rb') as fp:
                #         all_outputs = pickle.load(fp)
                        
                ############## Store to dict ##############
                to_store = {'data_idx':data_idx, 'output':predicted}
                all_outputs[task_from] = to_store
                

                # os.system("sudo cp {d} /home/ubuntu/s3/model_log".format(d=pickle_dir))

                ############## Reset graph and paths ##############            
                tf.reset_default_graph()
                training_runners['sess'].close()
                try:
                    del sys.modules[ 'config' ]
                except:
                    pass
                sys.path = remove_dups(sys.path)
                enablePrint()
                print("exit")
            # exit(0)

            print('Current Directory: ', os.getcwd())
            pickle_dir = 'viz_{task_to}_transfer_{hs}{training_regimen}.pkl'.format(
                training_regimen=training_regimen,
                hs=args.hs,
                task_to=task_to
            )
            with open( pickle_dir, 'wb') as fp:
                pickle.dump(all_outputs, fp)

            subprocess.call(
                "mv {} /home/ubuntu/s3/visualizations/".format(pickle_dir),
                shell=True
            )
            print(all_outputs.keys())
        # Run jupyter nb
        print('Running Jupyter Notebooks...')
        os.makedirs(
            "/home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/transfer_{hs}_{arch}".format(
                hs=args.hs,
                arch=ARCH_TYPE),
            exist_ok=True
        )

        subprocess.call(
            "cp \
                /home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/Visual_{task_to}.ipynb \
                /home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/transfer_{hs}_{arch}/Visual_{task_to}{training_regimen}.ipynb".format(
                arch=ARCH_TYPE, 
                task_to=task_to,
                hs=args.hs,
                training_regimen=training_regimen),
            shell=True
        )

        subprocess.call(
            "jupyter nbconvert \
                --execute /home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/transfer_{hs}_{arch}/Visual_{task_to}{training_regimen}.ipynb \
                --output /home/ubuntu/task-taxonomy-331b/notebooks/transfer_viz/transfer_{hs}_{arch}/Visual_{task_to}{training_regimen}.ipynb \
                --to notebook \
                --ExecutePreprocessor.kernel_name=python3 \
                --ExecutePreprocessor.timeout=600 ".format(
                arch=ARCH_TYPE,
                hs=args.hs,
                task_to=task_to,
                training_regimen=training_regimen),
            shell=True
        )


if __name__ == '__main__':
    with Pool(args.n_parallel) as p:
       p.map(run_to_task, list_of_tasks)
    # run_to_task(list_of_tasks)
