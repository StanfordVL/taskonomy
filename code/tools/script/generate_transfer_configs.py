# generate_transfer_configs.py
# Desc:
#   Generates a config file that can be used for transferring task representations
# Usage:
#  python generate_transfer_configs.py -h

import argparse
import os
import shutil
import subprocess
import sys
import pickle


parser = argparse.ArgumentParser(description='Extract representations of encoder decoder model.')
parser.add_argument( '--prototype_path', 
        dest='prototype_path', 
        help='Path to the prototype config file' )
parser.set_defaults(prototype_path="/home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype.py")


# Source and target tasks
parser.add_argument('--input-task', 
        dest='input_task',
        help='Source task')
parser.set_defaults(input_task="autoencoder")

parser.add_argument('--target-task', 
        dest='target_task',
        help='Destination task')
parser.set_defaults(target_task="point_match")



# Functions for transfer net
parser.add_argument('--hidden-size', 
        dest='hidden_size',
        help='How wide to make the transfer MLP')
parser.set_defaults(hidden_size= "1024")

parser.add_argument('--num-layers', 
        dest='num_layers',
        help='How deep to make the transfer net')
parser.set_defaults(num_layers= "3")

parser.add_argument('--kernel-size', 
        dest='kernel_size',
        help='How wide to make the transfer MLP')
parser.set_defaults(kernel_size= "3")



# Training Regimen
parser.add_argument('--finetune-decoder', 
        dest='finetune_decoder',
        help='Whether to use pretrained weights decoder during training')
parser.set_defaults(finetune_decoder=False)

parser.add_argument('--retrain-decoder', 
        dest='retrain_decoder',
        help='Whether to lock the transfer fn and retrain decoder')
parser.set_defaults(retrain_decoder=False)


parser.add_argument('--unlock-decoder', 
        dest='unlock_decoder',
        help='Whether to unlock the decoder')
parser.set_defaults(unlock_decoder=False)


# Replace target decoder
parser.add_argument('--target-decoder-func', 
        dest='target_decoder_func',
        help='Function with which to replace the target decoder')
parser.set_defaults(target_decoder_func='DO_NOT_REPLACE_TARGET_DECODER')

parser.add_argument('--data-used', 
        dest='data_used',
        help='How much data to use for training')
parser.set_defaults(data_used='30k')

# Things for second order transfer
parser.add_argument('--second-task', 
        dest='second_task',
        help='another task. representations must be available. can pass in multiple second-tasks by separating by comma')
parser.set_defaults(second_task="NO_SECOND_TASK")

# Things for finetune chained transfer
parser.add_argument('--preinput-task', 
        dest='preinput_task',
        help='preinput task. Representations must be available.')
parser.set_defaults(preinput_task="PRE_INPUT_TASK")


# Things for funnel chained transfer
parser.add_argument('--input-cluster', 
        dest='input_cluster',
        help='Input cluster. Representations must be available.')
parser.set_defaults(input_cluster="INPUT_CLUSTER")

parser.add_argument('--intermediary-task', 
        dest='intermediary_task',
        help='Source task')
parser.set_defaults(input_task="autoencoder")


# Output settings
parser.add_argument('--dest-dir', 
        dest='dest_dir',
        help='Dir to save the new config file to')
parser.set_defaults(dest_dir= "/home/ubuntu/task-taxonomy-331b/experiments/transfers/")


args = parser.parse_args()

INPUT_TASK = "autoencoder"
TARGET_TASK = "vanishing_point"
PROTOTYPE_PATH = "/home/ubuntu/task-taxonomy-331b/experiments/transfers/config_prototype.py"
DEST_DIR = "/home/ubuntu/task-taxonomy-331b/experiments/transfers/"


CLUSTER_TO_TASKS = {
    '2d': "autoencoder denoise impainting",
    '3d': "edge3d keypoint3d reshade rgb2sfnorm rgb2depth",
    'conv': "edge2d keypoint2d"
}
CLUSTER_TO_TASKS['all'] = " ".join([v for _, v in CLUSTER_TO_TASKS.items()])

def replace_in_file(filepath, orig, replacement):
    command = "sed -i \"s/{orig}/{replacement}/g\" {filepath}".format(**locals())
    subprocess.call(command, shell=True)
    return command

import pdb
def make_config_for_transfer(src_task, target_task, hidden_size, num_layers, kernel_size, finetune_decoder=False):
    if args.preinput_task != "PRE_INPUT_TASK":
        config_dir = os.path.join(args.dest_dir, "{}__{}__{}__{}".format(args.preinput_task, src_task, target_task, hidden_size))
    else:
        config_dir = os.path.join(args.dest_dir, "{}__{}__{}".format(src_task, target_task, hidden_size))

    if args.second_task != "NO_SECOND_TASK":
        config_dir = os.path.join(args.dest_dir, "{}__{}__{}__{}".format(src_task, args.second_task, target_task, hidden_size))
    
    should_ft = finetune_decoder and finetune_decoder != "False"
    should_rt = args.retrain_decoder and args.retrain_decoder != "False"
    shold_unlock = args.unlock_decoder and args.unlock_decoder != "False"
    if should_ft and should_rt:
        config_dir += '__rt_ft'    
    elif should_ft:
        config_dir += '__ft'
    elif should_rt:
        config_dir += '__rt_no_ft'
    elif shold_unlock:
        config_dir += '__unlocked'

    config_path = os.path.join(config_dir, "config.py")

    os.makedirs(config_dir, exist_ok=True)
#     shutil.copy(args.prototype_path, config_path)

    with open(args.prototype_path, 'r') as f:
        data= f.read()
    data = data.replace("<PRE_INPUT_TASK>", args.preinput_task)
    if args.second_task != "NO_SECOND_TASK":
        data = data.replace("<INPUT_TASK>", ','.join([src_task, args.second_task]))
    else:
        try:
            order_num = int(src_task)
            is_higher_order = True
        except ValueError:
            is_higher_order = False
        if is_higher_order:
            print('Doing order higher than 4...')
            # Load pickle file
            with open('/home/ubuntu/task-taxonomy-331b/tools/final_first_order_rank.pkl', 'rb') as fp:
                task_order_pickle = pickle.load(fp)
            src_tasks_order = [ x[0] for x in task_order_pickle[target_task] ]
            src_tasks_order.remove('random')
            if len(src_tasks_order) < order_num:
                return
            all_tasks_string = ','.join(src_tasks_order[:order_num])
            data = data.replace("<INPUT_TASK>", all_tasks_string)
        else:
            if src_task == 'FULL':
                all_tasks_string = "autoencoder colorization curvature denoise edge2d edge3d ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm room_layout segment25d segment2d vanishing_point_well_defined segmentsemantic_rb class_selected class_1000"
                all_tasks_string = all_tasks_string.split(' ')
                all_tasks_string.remove(target_task)
                all_tasks_string = ','.join(all_tasks_string)
                data = data.replace("<INPUT_TASK>", all_tasks_string)
            elif src_task == 'FULL_select':
                all_tasks_string = "autoencoder colorization curvature denoise edge2d edge3d ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm room_layout segment25d segment2d vanishing_point_well_defined"
                all_tasks_string = all_tasks_string.split(' ')
                try:
                    all_tasks_string.remove(target_task)
                except ValueError:
                    print('not removing')
                all_tasks_string = ','.join(all_tasks_string)
                data = data.replace("<INPUT_TASK>", all_tasks_string)
            else:
                data = data.replace("<INPUT_TASK>", src_task)
    data = data.replace("<TARGET_TASK>", target_task)
    data = data.replace("<HIDDEN_SIZE>", hidden_size)
    data = data.replace("<NUM_LAYERS>", num_layers)
    data = data.replace("<KERNEL_SIZE>", kernel_size)
    data = data.replace("<FINETUNE_DECODER>", str(finetune_decoder))
    data = data.replace("<RETRAIN_DECODER>", str(args.retrain_decoder))
    data = data.replace("<UNLOCK_DECODER>", str(args.unlock_decoder))

    data = data.replace("<REPLACE_TARGET_DECODER_FUNC_NAME>", str(args.target_decoder_func))
    data = data.replace("<REPLACE_VAL_DATA_USED>", str(args.data_used))
    with open(config_path, 'w') as f:
        f.write(data)
#     replace_in_file(config_path, "<INPUT_TASK>", src_task)
#     replace_in_file(config_path, "<TARGET_TASK>", target_task)
#     replace_in_file(config_path, "<HIDDEN_SIZE>", int(hidden_size))
    
    print("COMMAND TO RUN CONFIG:")
    print("----------------------")
    print("python -u /home/ubuntu/task-taxonomy-331b/tools/transfer.py {}".format(config_dir))



def make_config_for_funnel(src_cluster, intermediary_task, target_task, hidden_size):
    config_dir = os.path.join(args.dest_dir, "{}_{}__{}__{}".format(src_cluster, intermediary_task, target_task, hidden_size))
    config_path = os.path.join(config_dir, "config.py")

    os.makedirs(config_dir, exist_ok=True)
#     shutil.copy(args.prototype_path, config_path)

    with open(args.prototype_path, 'r') as f:
        data= f.read()
    
    data = data.replace("<INPUT_TASKS>", CLUSTER_TO_TASKS[src_cluster])
    data = data.replace("<INTERMEDIARY_TASK>", intermediary_task)
    data = data.replace("<FINAL_TASK>", target_task)
    data = data.replace("<HIDDEN_SIZE>", hidden_size)
    with open(config_path, 'w') as f:
        f.write(data)
    
    print("COMMAND TO RUN CONFIG:")
    print("----------------------")
    print("python -u /home/ubuntu/task-taxonomy-331b/tools/transfer.py {}".format(config_dir))


def make_config_for_chained(src_task, intermediary_task, target_task, hidden_size):
    config_dir = os.path.join(args.dest_dir, "chained__{}__{}__{}__{}".format(src_task, intermediary_task, target_task, hidden_size))
    config_path = os.path.join(config_dir, "config.py")

    os.makedirs(config_dir, exist_ok=True)
#     shutil.copy(args.prototype_path, config_path)

    with open(args.prototype_path, 'r') as f:
        data= f.read()
    
    data = data.replace("<INPUT_TASK>", src_task)
    data = data.replace("<INTERMEDIATE_TASK>", intermediary_task)
    data = data.replace("<TARGET_TASK>", target_task)
    data = data.replace("<HIDDEN_SIZE>", hidden_size)
    with open(config_path, 'w') as f:
        f.write(data)
    
    print("COMMAND TO RUN CONFIG:")
    print("----------------------")
    print("python -u /home/ubuntu/task-taxonomy-331b/tools/transfer.py {}".format(config_dir))


    # INPUT_TASK=["autoencoder", "denoise", "impainting"]
# INTERMEDIARY_TASK = "denoise" # e.g. vanishing_point
# FINAL_TASK = "impainting" # e.g. vanishing_point
# HIDDEN_SIZE = 1024 # e.g. vanishing_point
if __name__=="__main__":
    if args.preinput_task != 'PRE_INPUT_TASK':
        make_config_for_transfer(
            src_task=args.intermediary_task,
            target_task=args.target_task,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            finetune_decoder=args.finetune_decoder)
    elif args.input_cluster == "INPUT_CLUSTER":
        make_config_for_transfer(
            src_task=args.input_task,
            target_task=args.target_task,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            finetune_decoder=args.finetune_decoder)
    else:
        make_config_for_chained(args.input_cluster, args.intermediary_task, args.target_task, args.hidden_size)
        # make_config_for_funnel(args.input_cluster, args.intermediary_task, args.target_task, args.hidden_size)
