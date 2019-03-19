#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: v1
@date: 2019/03
@version:
  keras: 2.1.6
@usage:
  python keras_cifar10_multiple_gpus.py --num_gpus=2 --num_epochs=20
"""

import os
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import argparse

# In[]

num_gpus = 2
batch_size = 100
height = 32
width = 32
num_classes = 10
epochs = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# In[]

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument(\
      '--num_gpus',\
      type=int,\
      default=2,\
      help='the number of gpus'\
  )
  parser.add_argument(\
      '--num_epochs',\
      type=int,\
      default=20,\
      help='the number of epochs'\
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  num_gpus = FLAGS.num_gpus
  epochs = FLAGS.num_epochs

  # load the cifar-10 datasets
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  
  # In[]
  
  # Instantiate the base model (or "template" model).
  # We recommend doing this with under a CPU device scope,
  # so that the model's weights are hosted on CPU memory.
  # Otherwise they may end up hosted on a GPU, which would
  # complicate weight sharing.
  with tf.device('/cpu:0'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
  # Replicates the model on 2 GPUs.
  # This assumes that your machine has 8 available GPUs.
  parallel_model = multi_gpu_model(model, gpus=num_gpus, 
                                   cpu_merge=True, 
                                   cpu_relocation=False)
  
  parallel_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop', 
                         metrics=['accuracy'])
  
  # In[]
  
  # This `fit` call will be distributed on 8 GPUs.
  # Since the batch size is 256, each GPU will process 32 samples.
  parallel_model.fit(x_train, y_train, 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     validation_data=(x_test, y_test))
  
  # Save model via the template model (which shares the same weights):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  model.save(os.path.join(save_dir, model_name))























