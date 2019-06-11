"""
    Sample triplets for extraction:
    e.g.
    python generate_samples_for_transitivity.py  \
        > /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples.txt

    cut -d ' ' -f -2 /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples.txt \
        | uniq > /home/ubuntu/task-taxonomy-331b/assets/transitivity/first_transfer.txt

    cut -d ' ' -f 2- /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples.txt \
        | uniq > /home/ubuntu/task-taxonomy-331b/assets/transitivity/second_transfer.txt
    
"""
from __future__ import print_function
import random
import copy

TASKS="autoencoder denoise edge2d edge3d impainting keypoint2d keypoint3d reshade rgb2depth rgb2sfnorm".split(" ")

selected = []

for intermediate in TASKS:
	valid = list(set(TASKS) - set([intermediate]))
	random.shuffle(valid)
	srcs = valid[:3]
	dsts = list(set(valid) - set(srcs))[:3]
	random.shuffle(dsts)
	for src in srcs:
		for dst in dsts:
			print(src, intermediate, dst)
			selected.append((src, intermediate, dst))
