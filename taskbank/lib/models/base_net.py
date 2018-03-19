'''
    General config.py options that can be used for all models. :
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

from optimizers import ops as optimizers

class BaseNet(object):

    def __init__(self, global_step, cfg):
        self.cfg = cfg
        self.decoder_only = False
        self.extended_summaries = False
        if 'extended_summaries' in cfg:
            self.extended_summaries = cfg['extended_summaries']

    def build_model(self):
        raise NotImplementedError( 'build_model not implemented')

    def get_losses(self):
        raise NotImplementedError( 'get_losses not implemented')

    def build_train_op( self, global_step ):
        if not self.model_built or self.total_loss is None:
            raise RuntimeError( "Cannot build optimizers until 'build_model' ({0}) and 'get_losses' {1} are run".format(
                    self.model_built, self.total_loss is not None ) )
        self.global_step = global_step
        self.optimizer = optimizers.build_optimizer( global_step=global_step, cfg=self.cfg )
        self.train_op = slim.learning.create_train_op( self.total_loss, self.optimizer )

    def train_step(self):
        raise NotImplementedError( 'train_step not implemented' )

    def get_train_step_fn(self):
        return slim.learning.train_step