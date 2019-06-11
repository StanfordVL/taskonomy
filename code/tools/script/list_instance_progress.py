'''
  Name: list_instance_progress.py
  Desc: Prints out the step count at the last saved checkpoint for each saved model
  Usage:
    python list_instance_progress.py
'''
from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import concurrent.futures

def main( _ ):
    root_dir = '/home/ubuntu/s3/model_log/transfer'
    log_dir = 'logs/slim-train/'

    # list_of_task = ['autoencoder__autoencoder__1024']
    list_of_task = os.listdir(root_dir)
    # list_of_task = "autoencoder__rgb2depth__1024 edge3d__autoencoder__1024 edge3d__edge3d__1024 edge3d__rgb2sfnorm__1024 impainting__keypoint3d__1024 keypoint3d__rgb2sfnorm__1024 reshade__keypoint2d__1024 reshade__keypoint3d__1024 rgb2depth__keypoint3d__1024 rgb2sfnorm__edge2d__1024 rgb2sfnorm__reshade__1024".split(" ")
    #print(list_of_task)
    task_step_list = []
    for task in list_of_task:
        ckpt_dir = os.path.join( root_dir, task, log_dir )
        try:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            step = full_path.split('-')[-1]
        except:
            print("Problem with {}".format(task))
            step = 0
        task_step_list.append( (task, step) )
    
    #print(task_step_list)
    print_steps(task_step_list)
    # start_model_copy_threads( task_step_list, root_dir, log_dir )

def print_steps(task_step_list):
    for task, step in task_step_list:
        print("{}: \t{}".format(task, step))

    print("Not running:")
    bad_count = 0
    bad_list = []
    for task, step in task_step_list:
        if int(step) < 7000:
            bad_count += 1
            bad_list.append(task)
            print("{}: \t{}".format(task, step))
    print("Bad count: {}".format(bad_count))
    print("bad list")
    print(" ".join(bad_list))


if __name__=='__main__':
    main( '' )
