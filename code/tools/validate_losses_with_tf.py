#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Downloads the losses from tensorboard and saves into --dst-folder.
    Then it smooths the losses and saves the best ones into a CSV file.
'''
from __future__ import ( division, absolute_import, print_function, unicode_literals )

import argparse
import csv
import os
import numpy as np
import pickle as pkl
import urllib.request
import pandas as pd

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument( 'tensorboard_url', help='URL containing the address of a running tensorboard isntance' )

parser.add_argument( '--hidden-size', help='Suffix to use for mlps')
parser.add_argument( '--loss-name', help='the url-encoded name for the tensorflow loss')
parser.add_argument( '--dst-folder', help='folder to save the loss csvs')
parser.add_argument( '--smooth-k', type=int, help='size of the smoothing kernel')

parser.set_defaults(hidden_size="1024")
parser.set_defaults(loss_name="losses%2Fgenerator_l1_loss")
parser.set_defaults(dst_folder="losses")
parser.set_defaults(smooth_k=1)

TASKS = "autoencoder	denoise	edge2d	edge3d	impainting	keypoint2d	keypoint3d	reshade	rgb2depth	rgb2sfnorm"

def main():
    args = parser.parse_args()
    tasks = TASKS.split("\t")

    # losses_for_dst = {t: "losses%2Fabsolute_difference%2Fvalue_summary" for t in tasks}
    losses_for_dst = {t: args.loss_name for t in tasks}
    losses_for_dst = {t: args.loss_name for t in tasks}

    # Download losses
    print("Downloading losses from tensorboard...")
    os.makedirs(args.dst_folder, exist_ok=True)
    for src in tasks:
        for dst in tasks:
            try:
                path = make_url(args.tensorboard_url, 
                    src, dst, losses_for_dst[dst], args.hidden_size)
                urllib.request.urlretrieve(path, '{}/{}__{}__{}.csv'.format(
                    args.dst_folder, src, dst, args.hidden_size))
            except:
                path = make_url(args.tensorboard_url, 
                    src, dst, "losses%2Fabsolute_difference%2Fvalue_summary", args.hidden_size)
                urllib.request.urlretrieve(path, '{}/{}__{}__{}.csv'.format(
                    args.dst_folder, src, dst, args.hidden_size))
             


    # Smooth and save
    print("Smoothing losses and saving in loss.pkl...")
    smooth_k = args.smooth_k
    results = {}
    subdir = args.dst_folder
    transfers = os.listdir(subdir)
    for transfer in transfers:
        if '.pkl' in transfer:
            continue
        src, dst, _ = transfer.split("__")
        with open(os.path.join(subdir, transfer), 'r') as f:
            reader = csv.DictReader(f)
            res = [row for row in reader]
            vals = np.array([float(v['Value']) for v in res])
            kernel = np.ones((smooth_k,)) / float(smooth_k)
            smoothed = np.convolve(vals, kernel)[(smooth_k+1)//2:-smooth_k//2]
            results["{src}->{dst}".format(**locals())] = smoothed[-1] #.min()

    with open("{}/l1_loss_tensorboard_train.pkl".format(subdir), 'wb') as f:
        pkl.dump(results, f)

    # Now compute the difference between this and the TF loss
    print("Calculating diffs between TB and TF...")
    transfer_losses = pd.read_csv("/home/ubuntu/s3/model_log/results_{}/transfer_losses.csv".format(args.hidden_size))
    raw_losses = transfer_losses.mean(axis=0)
    del raw_losses['index']
    for dst in tasks:
        for src in tasks:
            tf = raw_losses[src + '_' + dst]
            tb = results[src + '->' + dst]
            print("{}->{}: TB: {} | TF: {}".format(
                src, dst, tb, tf))


def make_url(tensorboard_url, src, dst, loss, hidden_size):
    path = tensorboard_url +\
        "/data/scalars?run={src}__{dst}__{hs}%2Flogs%2Fslim-train%2Ftime&".format(
            src=src, dst=dst, hs=hidden_size
        ) + \
        "tag={loss}&format=csv".format(loss=loss)
    # print(path)
    return path





import csv
import os
import numpy as np
import pickle as pkl



if __name__ == '__main__':
    main()
# http://ec2-34-209-45-36.us-west-2.compute.amazonaws.com:6006/data/scalars?run=rgb2sfnorm__rgb2sfnorm__1024%2Flogs%2Fslim-train%2Ftime&tag=losses%2Fgenerator_l1_loss&format=csv
# http://ec2-34-209-45-36.us-west-2.compute.amazonaws.com:6006/data/scalars?run=autoencoder__autoencoder__1024%2Flogs%2Fslim-train%2Ftime&tag=losses%2Fgenerator_l1_loss&format=csv
