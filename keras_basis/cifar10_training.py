#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: v1
@date: 2019/03
@version:
  keras: 2.1.6
@usage:
  python cifar10_training.py  
"""

import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling2D, Conv2D
from keras.models import save_model

# In[]

# We use a data generator
class DataGenerator(keras.utils.Sequence):
  def __init__(self, list_IDs, batch_size=32, dim=(32,32), n_channels=3,
               n_classes=10, shuffle=True, istraining=True):
    self.dim = dim
    self.batch_size = batch_size
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()
    self.istraining = istraining

  def __len__(self):
    """the number of batches per epoch"""
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
    """Generate data for one batch"""
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    X, y = self.__data_generation(list_IDs_temp)
    return X, y

  def on_epoch_end(self):
    """Updates indexes after each epoch"""
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
          
  def __preprocess_images(self, images_array):
    """Normalize each channel to each images."""
    for img in images_array:
      for channel in range(self.n_channels):
        img[:,:,channel] = (img[:,:,channel] - np.mean(img[:,:,channel])) / \
                          np.std(img[:,:,channel])
    return images_array

  def __data_generation(self, list_IDs_temp):
    """Generates data containing batch_size samples"""
    # load the dataset first, no partition data available
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    if self.istraining:
      for i, ID in enumerate(list_IDs_temp):
          X[i,] = x_train[ID,]
          y[i] = y_train[ID]
    else:
      for i, ID in enumerate(list_IDs_temp):
          X[i,] = x_test[ID,]
          y[i] = y_test[ID]

    return self.__preprocess_images(X), \
            keras.utils.to_categorical(y, num_classes=self.n_classes)

# In[]
            
# load the dataset and preprocess it
training_data_count = 50000
training_params = {'dim': (32,32), 'batch_size': 64, 
          'n_classes': 10, 'n_channels': 3, 
          'shuffle': True, 'istraining': True}
training_generator = DataGenerator(list(range(0, training_data_count)), **training_params)

validation_data_count = 10000
validation_params = {'dim': (32,32), 'batch_size': 1000, 
          'n_classes': 10, 'n_channels': 3, 
          'shuffle': True, 'istraining': False}
validation_generator = DataGenerator(list(range(0,validation_data_count)), **validation_params)

# In[]

# parameters
num_class = 10
nb_filters = 16
pool_size = (2,2)
kernel_size = (3,3)
input_shape = (32,32,3)
dropout = 0.25

# build a simple model
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('softmax'))

# model summary
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[]

# start training
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=2,
                    use_multiprocessing=True,
                    workers=1)

# save the model
save_model(model, "cifar10_keras.h5")

