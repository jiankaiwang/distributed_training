#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:29:00 2019
@author: jiankaiwang
@date: 2019/03
@version: v1
@description:
  train a cifar10 recognition model via multiple GPUs in tensorflow
@version:
  keras: 2.1.6
  tensorflow: 1.11.0
@usage:
  python tf_cifar10_multiple_gpus.py --num_gpus=2 --max_steps=10000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
#import cifar10

import tf_cifar10_model as cifar10

# In[]

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# In[]

def tower_loss(scope, images, labels):
  """
  calculate the total loss on a single tower running the cifar model.
  
  args:
    scope: unique prefix string identifying the cifar tower, e.g. tower_0
    images: 4D tensor of shape [batch_size, height, width, 3]
    labels: 1D tensor of shape [batch_size]
    
  returns:
    Tensor of shape [] containing the total loss for a batch of data
  """
    
  # build a inference graph
  logits = cifar10.inference(images)
  
  # build the portion of the graph calculating the losses
  _ = cifar10.loss(logits, labels)
  
  # assemble all of the losses for the current tower only
  losses = tf.get_collection('losses', scope)
  
  # calculate the total loss for the current tower
  total_loss = tf.add_n(losses, name="total_loss")
  
  # attach a scalar summary to all individual losses and the total loss
  for loss in losses + [total_loss]:
    # remove the tag 'tower_[0-9]/' from the name
    # and help the clarity of presentation on tensorboard
    loss_name = re.sub("{}_[0-9]*".format(cifar10.TOWER_NAME), '', loss.op.name)
    tf.summary.scalar(loss_name, loss)
    
  return total_loss
  

# In[]
  
def average_gradients(tower_grads):
  """
  Calculate the average gradient for each shared variable across all towers.
  Provide a synchronization point across all towers.
  
  argvs:
    tower_grads: a list of (gradient, variable) tuples
    
  returns:
    average_grads: a list of (gradient, variable) tuples where the gradient has
                   been averaged across all towers
  """
  
  average_grads = []
  
  for grad_and_vars in zip(*tower_grads):
    # grad_and_vars is like ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
    grads = []
    
    for g, _ in grad_and_vars:
      
      # expand a dim to the gradients to represent the tower
      expanded_g = tf.expand_dims(g, 0)
      
      # append to a list like adding in a tower dimension
      grads.append(expanded_g)
    
    # average over the 'tower' dimension
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    
    # rearange the variables because variables are redundant and shared across towers as well
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    
  return average_grads

# In[]
  
def train():
  """
  Train a model for cifar-10 dataset in a number of steps.
  """
  
  with tf.Graph().as_default(), tf.device("/cpu:0"):
    # global_step counts the number of train() calls, is also equal to the 
    # total training steps (== batches processed * the number of gpus).
    global_step = tf.get_variable('global_step', [], 
                                  initializer=tf.constant_initializer(0), 
                                  trainable=False)
    
    # adjust the learning rate
    num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / \
                            FLAGS.batch_size / \
                            FLAGS.num_gpus
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
    
    # decay the learning rate exponentially based on the number of steps
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE, 
                                    global_step, 
                                    decay_steps, 
                                    cifar10.LEARNING_RATE_DECAY_FACTOR, 
                                    staircase=True)
    
    # create an optimizer that calculates gradient descent
    opt = tf.train.GradientDescentOptimizer(lr)
    
    # load the images and labels and prefetch a part of data
    images, labels = cifar10.distorted_inputs()
    
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity = 2 * FLAGS.num_gpus)
    
    # calculate the gradients for each model tower
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:{}'.format(i)):
          with tf.name_scope("{}_{}".format(cifar10.TOWER_NAME, i)) as scope:
            # degueues one batch for single GPU
            image_batch, label_batch = batch_queue.dequeue()
            
            # calculate total loss for one tower of the cifar model
            # constructs the cifar model but shares the variables across all towers
            loss = tower_loss(scope, image_batch, label_batch)
            
            # reuse the variables for the next tower
            tf.get_variable_scope().reuse_variables()
            
            # retain the summaries from the final tower
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            
            # calculate the gradients for the batch of data on the cifar data.
            grads = opt.compute_gradients(loss)
            
            # keep tracking of all towers' gradients
            tower_grads.append(grads)
    
    # here we calculate the average of each gradient
    # this is synchronization point across all towers
    grads = average_gradients(tower_grads)
    
    # add the summary to track the learning rate
    summaries.append(tf.summary.scalar('learning_rate', lr))
    
    # add histograms for gradients
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '_gradients', grad))
    
    # apply the gradients to adjust the shared variables
    apply_gradient_opt = opt.apply_gradients(grads, global_step=global_step)
    
    # add histograms for trainable variables
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))
      
    # track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variable_averages_opt = variable_averages.apply(tf.trainable_variables())
    
    # group all updates to a single train optimizer
    train_opts = tf.group(apply_gradient_opt, variable_averages_opt)
    
    # create a saver
    saver = tf.train.Saver(tf.global_variables())
    
    # build the summary operation from the latest tower
    summary_opt = tf.summary.merge(summaries)
    
    # start running operations on the graph
    # allow_soft_placement must be set to True once building towers on GPU
    # because some of opts don't have GPU implementations
    default_conf = tf.ConfigProto(allow_soft_placement=True, 
                                  log_device_placement=FLAGS.log_device_placement)
    
    with tf.Session(config=default_conf) as sess:
      # initializing operations
      sess.run(tf.global_variables_initializer())
      
      # start the queue runners
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
      
      for step in range(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_opts, loss])
        peroid = time.time() - start_time
        
        assert not np.isnan(loss_value), "Model diverged with loss (NaN)"
        
        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
          examples_per_sec = num_examples_per_step / peroid
          sec_per_batch = peroid / FLAGS.num_gpus
          
          print("{}: step {}, loss = {} ({} examples/sec; {} sec/batch)".format(
              datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
          
        if step % 100 == 0:
          summary_str = sess.run(summary_opt)
          summary_writer.add_summary(summary_str, step)
        
        # save the checkpoint
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          ckpt_path = os.path.join(FLAGS.train_dir, 'cifar10_model.ckpt')
          saver.save(sess, ckpt_path, global_step=step)
          
      # stop all queues
      coord.request_stop()
      coord.join(threads)

# In[]
  
def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  
  # start a training task
  train()

# In[]

if __name__ == "__main__":
  tf.app.run()
























