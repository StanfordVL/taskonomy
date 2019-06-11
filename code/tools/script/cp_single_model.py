'''
  Name: cp_transfer_model.py
  Desc: Copy current model to dir.
  Usage:
    python cp_transfer_model.py
'''
from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import concurrent.futures

MAX_CONCURRENT=10
def main( _ ):
    # root_dir = '/home/ubuntu/s3/model_log_new/nonfix_new_bn'
    # root_dir = '/home/ubuntu/s3/model_log/transfer'
    # root_dir = '/home/ubuntu/s3/model_log/transfer_transitive'
    root_dir = '/home/ubuntu/s3/model_log_final/'
    log_dir = 'logs/slim-train/'

    list_of_task = ['vanishing_point_well_defined']
    #         '3d_to_denoise__1024', 
    #         '2d_to_denoise__1024']
    # list_of_task = os.listdir(root_dir)
    # list_of_task = 'reshade__denoise__1024 reshade__edge3d__1024 reshade__edge2d__1024'.split(" ")
    # print(list_of_task)
    task_step_list = []
    for task in list_of_task:
        # if 'pixels' not in task:
            # continue
        # if 'autoencoder' not in task:
            # continue
        ckpt_dir = os.path.join( root_dir, task, log_dir )
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        print(ckpt_dir)
        if full_path is None:
            print('Checkpoint not found for {}'.format(task))
            # continue
            task = '/home/ubuntu/s3/model_log_final/{}/'.format(task)
            step = 38400
        else:
            step = full_path.split('-')[-1]
        task_step_list.append( (task, step) )
    
    #print(task_step_list)
    start_model_copy_threads( task_step_list, root_dir, log_dir )


def start_model_copy_threads( task_step_list, root_dir, log_dir ):
    from functools import partial

    mapfunc = partial(cp_task, root_dir=root_dir, log_dir=log_dir )
    iter_task_step_list = iter(task_step_list)
    #with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_step_list)) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        result = executor.map(mapfunc, iter_task_step_list)
    #concurrent.futures.wait(result)

def cp_task( task_step_pair, root_dir='', log_dir=''):
    task, step = task_step_pair
    model_cp_from = 'model.permanent-ckpt-{step}.{{suffix}}'.format(step=step)
    complete_from = os.path.join( root_dir, task, log_dir, model_cp_from )
    model_cp_to = 'model.permanent-ckpt.{suffix}'
    complete_to = os.path.join( root_dir, task, model_cp_to )

    suffix_list = ['data-00000-of-00001', 'index', 'meta']

    for suffix in suffix_list:
        cp_from = complete_from.format(suffix=suffix)
        cp_to = complete_to.format(suffix=suffix)
        command = "sudo cp {cp_from} {cp_to}".format(cp_from=cp_from, cp_to=cp_to)
        #print(command)
        os.system(command) 
    print('{task} has finished'.format(task=task))




if __name__=='__main__':
    main( '' )
