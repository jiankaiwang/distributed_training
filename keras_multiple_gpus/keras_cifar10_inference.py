#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: v1
@date: 2019/03
@version:
  keras: 2.1.6
"""

import os
import keras
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10

# In[]

num_classes = 10

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# In[]

modelpath = os.path.join(save_dir, model_name)
assert os.path.isfile(modelpath), "Model file was not found."
model = load_model(modelpath)
    
# In[]
    
# load the cifar-10 datasets
(_, _), (x_test, y_test) = cifar10.load_data()

y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255

# In[]

pred = model.predict(x_test[0:2])
print("Prediction / Inference: ", np.argmax(pred, axis=1))
print("Truth Lables: ", np.argmax(y_test[0:2], axis=1))