#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: v1
@date: 2019/03
@description:
  train a cifar10 recognition model in tensorflow
@version:
  keras: 2.1.6
  tensorflow: 1.11.0
@usage:
  python tf_cifar10_training.py
"""

import tensorflow as tf
from keras.datasets import cifar10
import os

# In[]

# load the cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
NUM_CLASS = 10
VAL_SIZE = 100
TRAINING_DATA = 50000
VAL_DATA = 10000

# In[]

learning_rate = 1e-4
training_epochs = 21
batch_size = 128
display_step = 5
keep_prob = 0.5

step_per_epoch = TRAINING_DATA // batch_size
model_path = os.getcwd()

# In[]

# create a simple data generator
def _generate_data(img, lable):
  _img = tf.image.per_image_standardization(img)
  return tf.convert_to_tensor(_img, dtype="float"), tf.one_hot(lable, NUM_CLASS)

def _train_data_generator():
  # create a training data generator
  _training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  training_dataset = _training_dataset.map(_generate_data)
  batch_training_dataset = \
    training_dataset.shuffle(buffer_size=TRAINING_DATA).batch(batch_size).repeat(training_epochs)
  training_data_iter = batch_training_dataset.make_initializable_iterator()
  return training_data_iter

def _val_data_generator():
  # create a validation data generator
  _val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  val_dataset = _val_dataset.map(_generate_data)
  batch_val_dataset = val_dataset.shuffle(buffer_size=VAL_DATA).batch(VAL_SIZE).repeat(training_epochs)
  batch_val_iter = batch_val_dataset.make_initializable_iterator()
  return batch_val_iter

# In[]

def conv2d(input, weight_shape, bias_shape):
  in_count = weight_shape[0] * weight_shape[1] * weight_shape[2]
  weight_init = tf.random_normal_initializer(stddev=(2.0/in_count)**0.5)
  W = tf.get_variable("W", weight_shape, initializer=weight_init)
  bias_init = tf.constant_initializer(value=0)
  b = tf.get_variable("b", bias_shape, initializer=bias_init)
  conv_out = tf.nn.conv2d(input, W, strides=[1,1,1,1], padding="SAME")
  return tf.nn.relu(tf.nn.bias_add(conv_out, b))

def max_pool(input, k=2):
  return tf.nn.max_pool(value=input, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME")

def layer(input, weight_shape, bias_shape):
  in_count = weight_shape[0] * weight_shape[1]
  w_init = tf.random_normal_initializer(stddev=(2.0/in_count)**0.5)
  b_init = tf.constant_initializer(value=0)
  W = tf.get_variable("W", weight_shape, initializer=w_init)
  b = tf.get_variable("b", bias_shape, initializer=b_init)
  return tf.nn.relu(tf.matmul(input, W) + b)

def loss(output, y):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)
  return tf.reduce_mean(cross_entropy)

def inference(input, keep_prob=0.5):
  """build a model"""
  x = tf.reshape(input, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
  with tf.variable_scope("conv_1"):
    conv_1 = conv2d(x, [5, 5, 3, 32], [32])                 # conv_1: 32 x 32 x 32
    pool_1 = max_pool(conv_1)                               # pool_1: 16 x 16 x 32
  with tf.variable_scope("conv_2"):
    conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])           # conv_2: 16 x 16 x 64
    pool_2 = max_pool(conv_2)                               # pool_2: 8 x 8 x 64
  with tf.variable_scope("fc"):
    pool_2_flat = tf.reshape(pool_2, [-1, 8 * 8 * 64])      # from pool_2
    fc_1 = layer(pool_2_flat, [8 * 8 * 64, 1024], [1024])   # fc_1: 1024
    fc_1_dropout = tf.nn.dropout(fc_1, keep_prob=keep_prob) # dropout
  with tf.variable_scope("out"):
    out = layer(fc_1_dropout, [1024, NUM_CLASS], [NUM_CLASS])
  return out

# In[]

def training(cost, global_step):
  """define a training process"""
  tf.summary.scalar("cost", cost)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
                                     beta1=9e-1, beta2=0.999, 
                                     epsilon=1e-8, name="Adam")
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  train_opt = optimizer.minimize(cost, global_step=global_step)
  return train_opt
  
def evaluate(output, y):
  """define the evaluation"""
  compare = tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1))
  accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))
  tf.summary.scalar("eval", accuracy)
  return accuracy

# In[]

with tf.Graph().as_default():
  with tf.variable_scope("cifar10") as scope:
    
    training_data_iter = _train_data_generator()
    batch_val_iter = _val_data_generator()
    
    (x, _y) = training_data_iter.get_next()
    #x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), _x)
    y = tf.reshape(_y, [-1, NUM_CLASS])
    global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)
    kp = tf.placeholder("float")
    
    # training 
    (val_x, _val_y) = batch_val_iter.get_next()
    logits = inference(x, kp)
    cost = loss(logits, y)
    train_opt = training(cost, global_step)
    
    scope.reuse_variables()
    
    # val
    val_logits = inference(val_x, kp)
    val_y = tf.reshape(_val_y, [-1, NUM_CLASS])
    val_cost = loss(val_logits, val_y)    
    val_opt = evaluate(val_logits, val_y)
    
    init_var = tf.global_variables_initializer()
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      initializers = tf.group([training_data_iter.initializer, \
                               batch_val_iter.initializer, \
                               init_var])
      sess.run(initializers)
      summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)
      
      for e in range(training_epochs):
        for s in range(step_per_epoch):
          cost_val, _ = sess.run([cost, train_opt], feed_dict={kp: keep_prob})
          
        if e % display_step == 0:
          y, loss, acc, record, step = sess.run([val_y, val_cost, val_opt, summary, global_step], 
                                             feed_dict={kp: 1.0})
          print("Epoch {}: accuracy {} and loss {}".format(e, acc, loss))
          summary_writer.add_summary(record, step)
          saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step)
            
print("Training finished.")

