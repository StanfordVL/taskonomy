'''
    Implements a saver which first saves locally, and then uses aws-cli to push it to S3.
'''

from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tempfile
import glob
import pdb
import subprocess

### VERY BAD HACKS
### TODO: REMOVE
S3_MOUNT_POINT = '/home/ubuntu/s3'
S3_MOUNTED_BUCKET = 'task-preprocessing-512-oregon'

class AwsSaver(tf.train.Saver):
    """Saves and restores variables."""

    def save(self, sess, save_path, **kwargs):
        dirs, fname = os.path.split(save_path)
        dst_dir = dirs.replace(
            S3_MOUNT_POINT,
            "s3://{}".format(S3_MOUNTED_BUCKET) )
        
        # Save locally
        tmp_path = os.path.join(tempfile.gettempdir(), fname)
        model_checkpoint_path = super(AwsSaver, self).save(sess, tmp_path, **kwargs)
        print(model_checkpoint_path)
        _, curr_ckpt_fname = os.path.split(model_checkpoint_path)
        # Move to AWS
        ckpt_files = glob.glob(tmp_path + '*')
        for ckpt_file in ckpt_files:
            _, fname = os.path.split(ckpt_file)
            tmp_path = os.path.join(tempfile.gettempdir(), fname)

            aws_mv_command = "aws s3 mv {} {}".format(
                tmp_path, 
                os.path.join(dst_dir, fname))

            subprocess.call(aws_mv_command, shell=True)

        # Get current s3 checkpoint path and delete old ones
        curr_ckpt_path = os.path.join(dirs, curr_ckpt_fname)
        self._MaybeDeleteOldCheckpoints(curr_ckpt_path)
        
        # Update the checkpoint file
        ckpt_state = tf.train.get_checkpoint_state(dirs)
        all_model_checkpoint_paths = None
        if ckpt_state is not None:
            all_model_checkpoint_paths = ckpt_state.all_model_checkpoint_paths
        tf.train.update_checkpoint_state(
            dirs,
            curr_ckpt_path,
            all_model_checkpoint_paths)
