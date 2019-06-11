'''
    Defines which optimizers are allowable from the config.py files.
    Additional optimizers can be defined here. 
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb

DEFAULT_ANNEALING_SCEDULE = 'NONE'

def build_optimizer( global_step, cfg ):
    '''
        Builds optimizer from cfg file. Expects
            cfg[ 'optimizer' ]: A TF Optimizer fn such as tf.train.GradientDescentOptimizer
            If cfg[ 'optimizer' ] requires arguments, then they must be supplied in cfg[ 'optimizer_kwargs' ]^
                ^ learning rate doesn't need to be specified as it is created from `build_step_size_tensor`. See
                  `build_step_size_tensor` for more. 
        
        Args:
            global_step: A Tensor that contains the global_step information. Used for determining the learning rate
            cfg: A dict from a config.py.
        
        Returns:
            optimizer: A TF Optimizer 
    ''' 
    if 'optimizer' not in cfg: 
        raise ValueError( "'optimizer' must be specified in config.py")
    if not cfg[ 'optimizer_kwargs' ] and cfg[ 'optimizer' ] not in [ tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer ] :
        raise ValueError( "The arguments for the optimizer {0} must be given, named, in 'optimizer_kwargs'".format( cfg[ 'optimizer' ] ))

    # Generate the Tensor containing learning rate + annealing schedule
    cfg[ 'optimizer_kwargs' ][ 'learning_rate' ] = build_step_size_tensor( global_step, cfg )
    optimizer =  cfg[ 'optimizer' ]( **cfg[ 'optimizer_kwargs' ] )
    print( "\t", optimizer )
    print( "\t", cfg[ 'optimizer_kwargs' ] )
    return optimizer



def build_step_size_tensor( global_step, cfg ):
    '''
        Creates an op to determine learning rate. It expects a value in 
        cfg[ 'initial_learning_rate' ] and it will default to a constant learning rate. 
        In addition, a 'learning_rate_schedule' may be specified, and it should accept
            *args = cfg[ 'initial_learning_rate' ]
            **kwargs = cfg[ 'learning_rate_schedule_kwargs' ] + { 'name', 'global_step' }
        It should return a scalar Tensor containing the learning rate.

        Args:
            cfg: A dict from a config.py

        Returns: 
            scalar Tensor containing the step size 
    '''
    print( "setting up learning rate annealing schedule:" )
    if 'initial_learning_rate' not in cfg:
        raise ValueError( "'initial_learning_rate' must be specified in config.py" )
    print( "\tinitial_learning_rate:", cfg[ 'initial_learning_rate' ] )

    # set up the learning rate
    if 'learning_rate_schedule' not in cfg:  # use constant
        print( '\tNo annealing schedule found in config.py. Using constant learning rate.' )
        step_size_tensor = tf.constant( cfg[ 'initial_learning_rate' ], name='step_size' )
    else:  # use user-supplied value
        if 'learning_rate_schedule_kwargs' not in cfg:
            print( "\tNo kwargs found for learning rate schedule.")
            cfg['learning_rate_schedule_kwargs' ] = {}
        if 'name' not in cfg['learning_rate_schedule_kwargs' ]:
            cfg['learning_rate_schedule_kwargs' ][ 'name' ] = 'step_size'
        cfg['learning_rate_schedule_kwargs' ][ 'global_step' ] = global_step
        # call the user's fn
        step_size_tensor = cfg[ 'learning_rate_schedule' ]( 
                    cfg[ 'initial_learning_rate' ], 
                    **cfg['learning_rate_schedule_kwargs' ] )
        print( "\t", cfg['learning_rate_schedule_kwargs' ] )
    summary_name = step_size_tensor.name.replace( ":", "_" )
    slim.summarize_tensor( step_size_tensor, tag='learning_rate/{0}'.format( summary_name ) )
    return step_size_tensor


def create_train_op(
    total_loss,
    optimizer,
    is_training=True,
    global_step=None,
    update_global_step=True,
    update_ops=None,
    variables_to_train=None,
    clip_gradient_norm=0,
    summarize_gradients=False,
    gate_gradients=tf.train.Optimizer.GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    gradient_multipliers=None):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `None`, then slim.variables.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    clip_gradient_norm: If greater than 0 then the gradients would be clipped
      by it.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    gradient_multipliers: A dictionary of either `Variables` or `Variable` op
      names to the coefficient by which the associated gradient should be
      scaled.
  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  if global_step is None:
    global_step = slim.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    logging.warning('update_ops in create_train_op does not contain all the '
                    ' update_ops in GraphKeys.UPDATE_OPS')

  if not is_training:
      #grad_updates = optimizer.apply_gradients([], global_step=global_step)
      return total_loss

  if variables_to_train is None:
    # Default to tf.trainable_variables()
    variables_to_train = tf.trainable_variables()
  else:
    # Make sure that variables_to_train are in tf.trainable_variables()
    for v in variables_to_train:
      assert v in tf.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss, variables_to_train, gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  # Scale gradients.
  if gradient_multipliers:
    with tf.name_scope('multiply_grads'):
      grads = multiply_gradients(grads, gradient_multipliers)

  # Clip gradients.
  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
        clip_grad = []
        for grad, var in grads:
            if grad is not None:
                clipped = tf.clip_by_norm(grad, clip_gradient_norm)
                clip_grad.append((clipped, var))
            else:
                clip_grad.append((grad, var))
        grads = clip_grad

  # Summarize gradients.
  if summarize_gradients:
    with tf.name_scope('summarize_grads'):
      add_gradients_summaries(grads)

  # Create gradient updates.
  if not update_global_step:
    global_step = None 
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with tf.name_scope('train_op'):
    # Make sure total_loss is valid.
    total_loss = tf.check_numerics(total_loss,
                                          'LossTensor is inf or nan')

    # Make sure update_ops are computed before total_loss.
    # Ensure the train_tensor computes grad_updates.
    with tf.control_dependencies( [grad_updates ] ):
        if update_ops:
            with tf.control_dependencies(update_ops):
                train_op = tf.identity(total_loss)
        else:
            train_op = tf.identity(total_loss)

  # Add the operation used for training to the 'train_op' collection
  train_ops = tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
    train_ops.append(train_op)

  return train_op
