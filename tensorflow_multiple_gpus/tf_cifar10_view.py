#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@date: 2019/03
@version: v1
@description:
  A quick start to understand both the queue and coordinator.
"""

import tf_cifar10_dataset as cifar10_input
import tensorflow as tf

# In[]

with tf.Graph().as_default():  
  # load the images and labels and prefetch a part of data
  images, labels = cifar10_input.distorted_inputs(128)
  
  batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity = 2 * 2)
  
  image_batch, label_batch = batch_queue.dequeue()
  
  default_conf = tf.ConfigProto(allow_soft_placement=True)
    
  with tf.Session(config=default_conf) as sess:
    # initializing operations
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    imgs, lbls = sess.run([image_batch, label_batch])
    print("the shape of batch images: {}".format(imgs.shape))
    print("labels of batch images: {}".format(lbls))
    
    coord.request_stop()
    coord.join(threads)
    

    
    