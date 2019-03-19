#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@date: 2019/03
@description:
  A simple example demostrates how to use the cifar10 model.
@version:
  keras: 2.1.6
  tensorflow: 1.11.0
"""

import tensorflow as tf
import os
from tensorflow.keras.datasets import cifar10 as keras_cifar10
import tf_cifar10_model as cifar10
import cv2
import numpy as np
import matplotlib.pyplot as plt

# In[]

model_dir = "/tmp/cifar10_train"
ckpt_dir = os.path.join(model_dir, "checkpoint")

# In[]

def _list_variables_with_values(sess, variables):
  """list variables and their values in the model
  
  Args:
    sess: an active tensorflow session
    variables: trainable varaibles
    
  Returns:
    None
  """
  for v in variables:
    print("{}:{}".format(v.name, sess.run(v)))
    
# In[]
    
def _list_variables(sess, variables):
  """list variables only in the model
  
  Args:
    sess: an active tensorflow session
    variables: trainable varaibles
    
  Returns:
    None
  """
  for v in variables:
    print("{}".format(v.name))
    
# In[]
    
def _plot_img(img):
  plt.imshow(img)
  plt.show()
    
# In[]

def main(argv=None):
  
  # load the cifar10 dataset
  (_, _), (x_test, y_test) = keras_cifar10.load_data()
  
  # the directory path of all checkpoints and meta data
  model_path = os.path.expanduser(model_dir)
  
  # get the latest checkpoint
  latest_ckpt = tf.train.latest_checkpoint(model_path)
  print("restore checkpoint: {}".format(latest_ckpt))
  
  # randomly select an image, resize it and expand a dimenion at axis=0
  # the shape of tgtImg is (1, 24, 24, 3)
  select_id = np.random.randint(0,len(y_test))
  tgtImg = cv2.resize(x_test[select_id], (24,24))
  tgtImg = np.expand_dims(tgtImg, 0)
  
  # plot image
  #_plot_img(x_test[select_id])

  with tf.Graph().as_default():
    # allow_soft_placement=True allows inference without GPUs,
    # even the model is trained on multiple GPUs
    default_conf = tf.ConfigProto(allow_soft_placement=True)
    
    # a placeholder for images
    image = tf.placeholder(tf.float32, shape=(1,24,24,3))
    logits = cifar10.inference(image)
    
    # restore the variables from the latest checkpoint
    saver = tf.train.Saver()
    
    with tf.Session(config=default_conf) as sess:
      saver.restore(sess, latest_ckpt)
      
      # list all variables and their values
      #_list_variables_with_values(sess, tf.trainable_variables())
      
      # only list variables
      #_list_variables(sess, tf.trainable_variables())

      # inference for a single image
      pred, = sess.run([logits], feed_dict={image: tgtImg})
      print(pred)
    
    # list the top-3 result
    top_pred_cls_idx = np.argsort(pred[0])[::-1]
    top_n = 3
    print("Prediction of Top {}: {}".format(top_n, top_pred_cls_idx[0:top_n]))
    print("Label: {}".format(y_test[select_id]))

# In[]

if __name__ == "__main__":
  main()
      














  