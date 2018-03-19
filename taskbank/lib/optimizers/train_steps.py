'''
    Defines train_step_fns which allow fancy training regimens
'''
from __future__ import absolute_import, division, print_function
import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
########################
# Train step functions
########################
def gan_train_step_fn( sess, g_and_d_train_ops, global_step, return_accuracy=False,
    n_g_steps_before_d=1, n_d_steps_after_g=1, init_g_steps=0, train_step_kwargs={} ):
    '''
        Executes a training step for a GAN. This may have multiple 
            generator updates and may have multiple discriminator
            updates. The global_step may be updated several times, but
            this is all considered one step. 

        Args:
            sess
            g_and_d_train_ops: A Tuple of ( g_train_op, d_train_op )
            global_step: A Tensor that will be incremented 
                n_g_steps_before_d + n_d_steps_after_g times.
            n_g_steps_before_d: Run g_train_op this many times
            n_d_steps_after_g: Run d_train_op this many times after running 
                g_train_op
            train_step_kwargs: Currently only 'should_log' fn is used.
        
        Returns:
            mean_g_losses, should_stop
    '''
    start_time = time.time()
    if return_accuracy:
        g_train_op, d_train_op, g_lnorm_op, accuracy_op = g_and_d_train_ops
        accuracy = sess.run(accuracy_op)
    else:
        g_train_op, d_train_op, g_lnorm_op = g_and_d_train_ops
    curr_global_step = sess.run( global_step )
   
    if init_g_steps > 0 and curr_global_step < init_g_steps:
        g_losses = sess.run( g_lnorm_op )
        d_losses = 0
    else:
        if n_g_steps_before_d >= 1 and n_d_steps_after_g == 1:
            g_losses = [ sess.run( g_train_op ) for i in range( n_g_steps_before_d - 1 ) ]
            d_loss, last_g_loss = sess.run( [g_train_op, d_train_op] )
            g_losses.append( last_g_loss )
            d_losses = [ d_loss ]
        else:
            g_losses = [ sess.run( g_train_op ) for i in range( n_g_steps_before_d ) ]
            d_losses = [ sess.run( d_train_op ) for i in range( n_d_steps_after_g ) ]
    np_global_step = sess.run( global_step ) #train_step_kwargs[ 'increment_global_step_op' ] )
    time_elapsed = time.time() - start_time
    
    # Logging
    if 'should_log' in train_step_kwargs:
        if train_step_kwargs['should_log']( np_global_step ):
            if return_accuracy:
                tf.logging.info('global step %d: g_loss = %.4f, d_loss = %.4f (%.2f sec/step), accuracy = %.4f',
                        np_global_step, np.mean( g_losses ), np.mean( d_losses ), time_elapsed, accuracy)
            else:
                tf.logging.info('global step %d: g_loss = %.4f, d_loss = %.4f (%.2f sec/step)',
                        np_global_step, np.mean( g_losses ), np.mean( d_losses ), time_elapsed)

    should_stop = should_stop_fn( np_global_step, **train_step_kwargs )
    if return_accuracy:
        return np.mean( g_losses ), np.mean(accuracy), should_stop
    return np.mean( g_losses ), should_stop

def discriminative_train_step_fn( sess, train_ops, global_step, return_accuracy=False, train_step_kwargs={} ):
    '''
        Executes a training step a discriminative network for one step.
        Args:
            sess
            global_step: A Tensor that will be incremented 
            train_step_kwargs: Currently only 'should_log' fn is used.
        
        Returns:
            loss, should_stop
    '''
    start_time = time.time()
    loss_op, accuracy_op = train_ops
    if return_accuracy:
        accuracy = sess.run(accuracy_op)
    discriminative_loss = sess.run(loss_op)
    np_global_step = sess.run( global_step )
    time_elapsed = time.time() - start_time

    # Logging
    if 'should_log' in train_step_kwargs:
        if train_step_kwargs['should_log']( np_global_step ):
            if return_accuracy:
                tf.logging.info('global step %d: loss = %.4f, accuracy = %.4f (%.2f sec/step)',
                        np_global_step, np.mean(discriminative_loss), np.mean(accuracy), time_elapsed)
            else:
                tf.logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                        np_global_step, np.mean(discriminative_loss), time_elapsed)

    should_stop = should_stop_fn( np_global_step, **train_step_kwargs )
    if return_accuracy:
        return np.mean( discriminative_loss ), np.mean(accuracy), should_stop
    return np.mean( discriminative_loss ), should_stop


######################################
# Generating args for train_step_fns
######################################
def get_default_train_step_kwargs( global_step, max_steps, log_every_n_steps=1, trace_every_n_steps=None ):
    ''' Sets some default arguments for any train_step_fn '''
    with tf.name_scope('train_step'):
        train_step_kwargs = { 'max_steps': max_steps }

        if max_steps:
            should_stop_op = tf.greater_equal(global_step, max_steps)
        else:
            should_stop_op = tf.constant(False)
        train_step_kwargs['should_stop'] = should_stop_op
        train_step_kwargs['should_log'] = lambda x: ( x % log_every_n_steps == 0 )
        if trace_every_n_steps is not None:
            train_step_kwargs['should_trace'] = tf.equal(
                tf.mod(global_step, trace_every_n_steps), 0)
            train_step_kwargs['logdir'] = logdir

        train_step_kwargs[ 'global_step_copy' ] = tf.identity( global_step, name='global_step_copy' )
        train_step_kwargs[ 'increment_global_step_op' ] = tf.assign( global_step, global_step+1 )

        return train_step_kwargs


######################
#  should_stop_fn
###################### 
def should_stop_fn( np_global_step, max_steps=None, **kwargs ):
    '''
        Determines whether training/testing should stop. Currently
        works only based on max_steps, but this could also use forms of
        early stopping based on loss. 

        Args:
            np_global_step: An int or np.int
            max_steps: An int
            **kwargs: Currently unused
        Returns:
            should_stop: A bool
    '''
    if max_steps and np_global_step >= max_steps:
        return True
    else:
        return False
