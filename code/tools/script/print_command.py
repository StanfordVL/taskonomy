from __future__ import absolute_import, division, print_function

import argparse

parser = argparse.ArgumentParser(description='Extract losses of encoder decoder model.')
parser.add_argument('task', help='task to run')
parser.add_argument('--action', dest='action')
parser.add_argument('--data-split')
parser.set_defaults(data_split="NOTHING")

args = parser.parse_args()

data_split = ""
if args.data_split != "NOTHING":
    data_split = "--data-split {}".format(args.data_split)

if args.action == "EXTRACT_REPS":
    print("python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers --task {} {}".format(args.task, data_split))
elif args.action == "EXTRACT_LOSSES":
    print("python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers --task {} {}".format(args.task, data_split))
elif args.action == "TRANSFER":
    print("python -u /home/ubuntu/task-taxonomy-331b/tools/transfer.py /home/ubuntu/task-taxonomy-331b/experiments/transfers/{}".format(args.task))
else:
    exit(1)
