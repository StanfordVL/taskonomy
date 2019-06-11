'''
  Name: train.py
  Desc: Executes training of a network. The actual code executed is specified from a 
    configuration file located in some directory and the configuration file must be
    named config.py. 

    The configuration file must contain a method called get_cfg( nopause=True ) which 
    should return a Dict of configuration options when called. What options can/must 
    be specified depends on what model is being used. These options can be found in 
    the corresponding module that implements the model in the lib/models folder. 

    There are many options that must be speified regardless of the model being used. 
    They typically pertain to the input pipeline being used. The entire input pipeline
    is currently geared towards encoder-decoder models, but this will change as the 
    codebase becomes more flexible. 

    Here are some options that may be specified for any model. Options which are not
    required are denoted with a '-' in front (e.g. -'my_option'). If they have a 
    default value, it is given at the end of the description in parens. 

        Data pipeline:
            Data locations:
                'train_filenames': A pkl file that contains an array of training data 
                    information, encoded as Strings. These are not the images themselves, 
                    but may contain filenames to the image on disk.  
                'val_filenames': As above, but for validation data.
                'test_filenames': As above, but for test data.
                'dataset_dir': The folder that all the data is stored in. This may just be 
                    something like '/', and then all filenames in 'train_filenames' will
                    give paths relative to 'dataset_dir'. For example, if 'dataset_dir'='/',
                    then train_filenames might have entries like 'path/to/data/img_01.png'
                'preprocess_fn': A function that is given one entry from one of the filenames
                    files above, and returns something that is fed to the input queue. For 
                    example:
                        input_l_img, target_ab_img, mask = preprocess_fn( 'path/to/data/img_01.png' )
                'randomize': Whether to randomize the input data. This should be true for 
                    training and false for testing or debugging. 
                'num_read_threads': The number of 'preprocess_fn' threads to run in parallel. 
                'inputs_queue_capacity': Stores a queue of inputs/targets/masks to use. This 
                    controls how many examples to store in the queue. A good rule of thumb is 
                    to have at least enough examples here to fill a few batches.  

            Inputs: (Also valid for Targets)
                'input_dim': A Tuple, the shape of the input/target images.
                'input_num_channels': The number of channels in input/target images. 
                'input_dtype': The Tensorflow dtype of the input/target
                'input_domain_name': A name given to the input/target domain. This may be
                    used in some functions.
                'input_preprocessing_fn': Returns the image that will be fed into the 
                    model. This funciton is given the input/target image. 
                -'input_preprocessing_fn_kwargs': These are passed into 'input_preprocessing_fn' 
                    along with the input/target image. ({}) 
            
            Masks: Masks can be applied to a loss in order to not apply gradients to 
                specific examples an parts of examples. Currently, a mask is created as
                part of the input pipeline, and is specified from the config file.
                -'mask_fn': The function to use to determine the mask. It is given the 
                    target image as input.   
                -'mask_fn_kwargs': These are passed into 'mask_fn' along with the target
                    image. ({})
                
        Logging:
            'log_dir': An absolute path to the logging directory. Checkpopints and 
                summaries will be saved to this directory. 
            'summary_save_every_secs': How many seconds between saving out summaries.
            'checkpoint_save_every_secs': How many seconds between saving checkpoints. 

        Training:
            'batch_size': The size of each batch. 
            'num_epochs': The maximum number of epochs to train for. It may not train 
                for this many epochs, depending on the early stopping criteria used. 
                    
        Optimization: 
            'initial_learning_rate': The initial learning rate to use for the model. 
                If there are additional training ops or there are fancier training
                functions (e.g. GAN loss), then they will be specified in another way
                that is documented in that model module. 
            
            See optimizers.ops.build_optimizer for details on how to specify an optimizer.
            See optimizers.ops.build_step_size_tensor for details on how to anneal learning rate

            
  Usage:
    python train.py /path/to/cfgdir/ --gpu gpu_id
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import pdb


import init_paths
from   general_utils import RuntimeDeterminedEnviromentVars
from   models.sample_models import *
import utils
import threading

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument( 'cfg_dir', help='directory containing config.py file' )
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    type=int)
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.set_defaults(nopause=True)


def main( _ ):
    args = parser.parse_args()

    print(args)
    # Get available GPUs
    local_device_protos = utils.get_available_devices()
    print( 'Found devices:', [ x.name for x in local_device_protos ] )  
    # set gpu
    if args.gpu_id is not None:
        print( 'using gpu %d' % args.gpu_id )
        os.environ[ 'CUDA_VISIBLE_DEVICES' ] = str( args.gpu_id )
    else:
        print( 'no gpu specified' )
    # load config and run training
    cfg = utils.load_config( args.cfg_dir, nopause=args.nopause )
    # cfg['num_read_threads'] = 1
    run_training( cfg, args.cfg_dir )

def run_training( cfg, cfg_dir ):
    # set up logging
    tf.logging.set_verbosity( tf.logging.INFO )

    with tf.Graph().as_default() as g:
        # create ops and placeholders
        inputs = utils.setup_input_transfer( cfg, is_training=True )
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        
        # build model (and losses and train_op)
        model = utils.setup_model_chained_transfer( inputs, cfg, is_training=True )

        # execute training 
        start_time = time.time()
        utils.print_start_info( cfg, inputs[ 'max_steps' ], is_training=True )
        train_log_dir = os.path.join( cfg['log_dir'], 'slim-train' )
        permanent_checkpoint_dir = os.path.join( cfg['log_dir'], 'checkpoints' )

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        # When ready to use a model, use the code below
        train(  model[ 'train_op' ],
                train_log_dir,
                utils.get_data_prefetch_threads_init_fn_transfer( inputs, cfg, is_training=True ),
                train_step_fn=model[ 'train_step_fn' ], 
                train_step_kwargs=model[ 'train_step_kwargs' ], 
                global_step=model[ 'global_step' ],
                number_of_steps=inputs[ 'max_steps' ],
                number_of_epochs=cfg['num_epochs'],
                init_fn=model[ 'init_fn' ],
                save_checkpoint_every=inputs['max_steps'] // (cfg['num_epochs'] * 2),
                cfg_dir=cfg_dir,
                #RuntimeDeterminedEnviromentVars.steps_per_epoch,
                permanent_checkpoint_dir=permanent_checkpoint_dir,
                save_summaries_secs=cfg['summary_save_every_secs'],
                save_interval_secs=cfg['checkpoint_save_every_secs'],
                saver=model[ 'saver_op' ], 
                return_accuracy= 'return_accuracy' in cfg and cfg['return_accuracy'],
                    session_config=session_config ) 

        end_train_time = time.time() - start_time
        print('time to train %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time/(60*60)))
        print('avg time per epoch: %.3f hrs' % ( (end_train_time/(60*60)) / cfg['num_epochs']) )


_USE_DEFAULT = slim.learning._USE_DEFAULT
def train(train_op,
          logdir,
          data_prefetch_threads_init_fn,
          train_step_fn=slim.learning.train_step,
          train_step_kwargs=_USE_DEFAULT,
          log_every_n_steps=10,
          graph=None,
          master='',
          is_chief=True,
          global_step=None,
          number_of_steps=None,
          number_of_epochs=1,
          init_op=_USE_DEFAULT,
          init_feed_dict=None,
          local_init_op=_USE_DEFAULT,
          init_fn=None,
          save_checkpoint_every=None,
          permanent_checkpoint_dir=None,
          ready_op=_USE_DEFAULT,
          summary_op=_USE_DEFAULT,
          save_summaries_secs=30,
          summary_writer=_USE_DEFAULT,
          startup_delay_steps=0,
          saver=None,
          save_interval_secs=600,
          sync_optimizer=None,
          session_config=None,
          trace_every_n_steps=None,
          cfg_dir=None,
          return_accuracy=False):
    """ Adapted from slim """
    """Runs a training loop using a TensorFlow supervisor.
    When the sync_optimizer is supplied, gradient updates are applied
    synchronously. Otherwise, gradient updates are applied asynchronous.
    Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
        return the loss value.
    logdir: The directory where training logs are written to. If None, model
        checkpoints and summaries will not be written.
    data_prefetch_threads_init_fn: A function to be called with (sv)
    train_step_fn: The function to call in order to execute a single gradient
        step. The function must have take exactly four arguments: the current
        session, the `train_op` `Tensor`, a global step `Tensor` and a dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
        default, two `Boolean`, scalar ops called "should_stop" and "should_log"
        are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
        and global step and logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
        default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
        replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
        then slim.variables.get_or_create_global_step() is used.
    number_of_steps: The max number of gradient steps to take during training.
        If the value is left as None, training proceeds indefinitely.
    number_of_epochs: Number of epochs to train, default to 1
    init_op: The initialization operation. If left to its default value, then
        the session is initialized by calling `tf.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
        value, then the session is initialized by calling
        `tf.local_variables_initializer()` and `tf.initialize_all_tables()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
        callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
        default value, then the session checks for readiness by calling
        `tf.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None`
        to indicate that no summaries should be written. If unset, we
        create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
        that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created
        and used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer. If the
        argument is supplied, gradient updates will be synchronous. If left as
        `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.ConfigProto` that will be used to
        configure the `Session`. If left as `None`, the default will be used.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
        and add it to the summaries every `trace_every_n_steps`. If None, no trace
        information will be produced or saved.
    Returns:
    the value of the loss function after training.
    Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
        non-zero when `sync_optimizer` is supplied, if `number_of_steps` is
        negative, or if `trace_every_n_steps` is not `None` and no `logdir` is
        provided.
    """
    from time import gmtime, strftime 
    curr_time = strftime("%m%d_%H:%M:%S", gmtime())
    logtxt_name = "train_log_{time}.txt".format(time=curr_time)
    aws_log_fname = os.path.join(os.path.dirname(logdir), logtxt_name)
    local_log_fname = os.path.join(cfg_dir, "train_log.txt")
    log_cp = "cp {l} {aws}".format(l=local_log_fname, aws=aws_log_fname)
    mkdir_op = "mkdir -p {dir}".format(dir=os.path.dirname(logdir))

    os.system(mkdir_op)
    os.system(log_cp)
    if train_op is None:
        raise ValueError('train_op cannot be None.')

    if logdir is None:
        if summary_op != _USE_DEFAULT:
            raise ValueError('Cannot provide summary_op because logdir=None')
        if saver is not None:
            raise ValueError('Cannot provide saver because logdir=None')
        if trace_every_n_steps is not None:
            raise ValueError('Cannot provide trace_every_n_steps because '
                        'logdir=None')

    if sync_optimizer is not None and startup_delay_steps > 0:
        raise ValueError(
            'startup_delay_steps must be zero when sync_optimizer is supplied.')

    if number_of_steps is not None and number_of_steps <= 0:
        raise ValueError(
            '`number_of_steps` must be either None or a positive number.')

    if number_of_epochs <= 0:
        raise ValueError(
            '`number_of_epochs` must be a positive number.')

   

    graph = graph or tf.get_default_graph()
    with graph.as_default():
        if global_step is None:
            global_step = slim.get_or_create_global_step()
        saver = saver or tf.train.Saver(max_to_keep=number_of_epochs*2)

        with tf.name_scope('init_ops'):
            if init_op == _USE_DEFAULT:
                if tf.__version__ == '0.10.0':
                    init_op = tf.initialize_all_variables()
                else:
                    init_op = tf.global_variables_initializer()

            if ready_op == _USE_DEFAULT:
                ready_op = tf.report_uninitialized_variables()

            if local_init_op == _USE_DEFAULT:
                if tf.__version__ == '0.10.0':
                    local_init_op = tf.initialize_local_variables()
                else:
                    local_init_op = tf.group(
                            tf.local_variables_initializer(),
                            tf.initialize_all_tables())

        if summary_op == _USE_DEFAULT:
            if tf.__version__ == '0.10.0':
                tf.merge_all_summaries()
            else:
                summary_op = tf.summary.merge_all()

        if summary_writer == _USE_DEFAULT:
                summary_writer = tf.train.Supervisor.USE_DEFAULT
        cleanup_op = None

        if train_step_kwargs == _USE_DEFAULT:
            with tf.name_scope('train_step'):
                train_step_kwargs = {}

                if number_of_steps:
                    should_stop_op = tf.greater_equal(global_step, number_of_steps)
                else:
                    should_stop_op = tf.constant(False)
                train_step_kwargs['should_stop'] = should_stop_op
                train_step_kwargs['should_log'] = tf.equal(
                    tf.mod(global_step, log_every_n_steps), 0)
                if is_chief and trace_every_n_steps is not None:
                    train_step_kwargs['should_trace'] = tf.equal(
                        tf.mod(global_step, trace_every_n_steps), 0)
                    train_step_kwargs['logdir'] = logdir

        saver_time = tf.train.Saver()
        sv = tf.train.Supervisor(
            graph=graph,
            is_chief=is_chief,
            logdir=os.path.join(logdir, 'time'),
            init_op=init_op,
            init_feed_dict=init_feed_dict,
            local_init_op=local_init_op,
            ready_op=ready_op,
            summary_op=summary_op,
            summary_writer=summary_writer,
            global_step=global_step,
            saver=saver_time,
            save_summaries_secs=save_summaries_secs,
            save_model_secs=save_interval_secs,
            init_fn=init_fn)

        
        if summary_writer is not None:
            train_step_kwargs['summary_writer'] = sv.summary_writer

        prefetch_threads = []
        should_retry = True
        while should_retry:
            try:
                should_retry = False
                with sv.managed_session(
                    master, start_standard_services=False, config=session_config) as sess:
                    tf.logging.info('Starting prefetch threads.')
                    prefetch_threads = threading.Thread(
                        target=data_prefetch_threads_init_fn,
                        args=(sess, sv))
                    prefetch_threads.start()
                    #prefetch_threads = data_prefetch_threads_init_fn( sess, sv )

                    tf.logging.info('Starting Session.')
                    sv.start_queue_runners( sess )
                    tf.logging.info('Queue Runner Started')
                    if is_chief and logdir is not None:
                        sv.start_standard_services(sess)
                    if is_chief and sync_optimizer is not None:
                        sv.start_queue_runners( sess, [chief_queue_runner] )
                    try:
                        step_count = 0
                        save_count = tf.train.global_step(sess, global_step) // save_checkpoint_every 

                        losses = AverageMeter()
                        accuracy = AverageMeter()
                        tf.logging.info('Starting Training Loop')

                        while not sv.should_stop():
                            if return_accuracy:
                                total_loss, accur, should_stop = train_step_fn(
                                    sess, train_op, global_step, train_step_kwargs=train_step_kwargs)
                                losses.update(total_loss)
                                accuracy.update(accur)
                                if accuracy.count % 100 == 0:
                                    tf.logging.info('==>Global Step %d: current mean train accuracy of last %d step is: %.4f', 
                                            tf.train.global_step(sess, global_step), accuracy.count, accuracy.avg)
                                    if accuracy.count % 1000 == 0:
                                        accuracy.reset()
                            else:
                                total_loss, should_stop = train_step_fn(
                                    sess, train_op, global_step, train_step_kwargs=train_step_kwargs)
                                losses.update(total_loss)
                                if losses.count % 100 == 0:
                                    tf.logging.info('==>Global Step %d: current mean loss of last %d step is: %.4f', 
                                            tf.train.global_step(sess, global_step), losses.count, losses.avg)
                                    os.system(log_cp)
                                    if losses.count % 1000 == 0:
                                        losses.reset()

                            if should_stop:
                                tf.logging.info('Stopping Training.')
                                break

                            if save_checkpoint_every and (tf.train.global_step(sess, global_step) // save_checkpoint_every > save_count ):
                                tf.logging.info('Checkpoint time! Saving model at step %d to disk (every %d step)' 
                                        % (tf.train.global_step(sess, global_step), save_checkpoint_every))
                                save_path = os.path.join( logdir, 'model.permanent-ckpt' )
                                saver.save(sess, save_path, global_step=sv.global_step)
                                save_count += 1
                        if logdir and sv.is_chief:
                            tf.logging.info('Finished training! Saving model to disk.')
                            saver.save(sess, sv.save_path, global_step=sv.global_step)
                    except:
                        if sv.is_chief and cleanup_op is not None:
                            tf.logging.info('About to execute sync_clean_up_op!')
                            sess.run(cleanup_op)
                        raise

            except tf.errors.AbortedError:
                # Always re-run on AbortedError as it indicates a restart of one of the
                # distributed tensorflow servers.
                tf.logging.info('Retrying training!')
                should_retry = True
#             finally:
                # if prefetch_threads:
                    # sv.coord.join( prefetch_threads )

        return total_loss

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=='__main__':
  main( '' )
  # tf.app.run()
