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


parser = argparse.ArgumentParser(description='Extract representations of encoder decoder model.')
parser.add_argument( '--prototype_path', 
        dest='prototype_path', 
        help='Path to the prototype config file' )
parser.set_defaults(prototype_path="/home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype.py")


# Source and target tasks
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

parser.add_argument('--random', 
        dest='random_sample', action='store_true')
parser.set_defaults(random_sample=False)

args = parser.parse_args()


# INTERMEDIARY_TASK = "denoise" # e.g. vanishing_point
# FINAL_TASK = "impainting" # e.g. vanishing_point
# HIDDEN_SIZE = 1024 # e.g. vanishing_point
if __name__=="__main__":
    import pickle
    import itertools
    import random
    with open('/home/ubuntu/task-taxonomy-331b/tools/ranked_first_order_transfers.pkl', 'rb') as fp:
        data = pickle.load(fp)
    if args.random_sample:
        top_5 = data[args.target_task][:5]
        rest = data[args.target_task][5:]
        all_comb = list(itertools.product(top_5, rest))
        random.seed(9001)
        top_5_combinations = random.sample(all_comb, 16)
    else:
        top_5 = data[args.target_task][:5]
        top_5_combinations = list(itertools.combinations(top_5, 3))
    configs = [] 
    for first_task,second_task,third_task in top_5_combinations:
        command = "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \\\n--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \\\n--input-task {first_task} --target-task {target_task} \\\n--hidden-size {hidden_size} \\\n--kernel-size {kernel_size} \\\n--num-layers {num_layer} \\\n--finetune-decoder False \\\n--retrain-decoder False \\\n--unlock-decoder True \\\n--target-decoder-func {target_decoder_func} \\\n--data-used {data_used} \\\n--second-task {second_task},{third_task} \\\n--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/third_order/{target_decoder_func}/{data_used}/\n".format(
                first_task=first_task,
                target_task=args.target_task,
                hidden_size=args.hidden_size,
                kernel_size=args.kernel_size,
                num_layer=args.num_layers,
                target_decoder_func=args.target_decoder_func,
                data_used=args.data_used,
                second_task=second_task,
                third_task=third_task)
        os.system(command)
        config_name = "{}__{},{}__{}__8__unlocked \\".format(first_task, second_task, third_task, args.target_task)
        configs.append(config_name)

    if args.target_task in ['ego_motion', 'fix_pose', 'point_match', 'non_fixated_pose'] \
            and args.data_used == '1k':
        with open('/home/ubuntu/task-taxonomy-331b/tools/config_list_multi.txt', 'a') as fp:
            for config in configs:
                fp.write("third_order/DO_NOT_REPLACE_TARGET_DECODER/1k/%s\n" % config)

    with open('/home/ubuntu/task-taxonomy-331b/tools/config_list_third_order.txt', 'a') as fp:
        for config in configs:
            fp.write("%s\n" % config)

