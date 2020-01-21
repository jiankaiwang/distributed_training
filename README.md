# Model Training in Keras and Tensorflow

This repository is a tutorial targeting how to train a deep neural network model in a higher efficient way. In this repository, we focus on two main frameworks that are Keras and Tensorflow. Parts of the content (scripts, documents, etc.) were referred from Tensorflow/Model repository and Keras document. The relative framework version please refer to the top description of each script. 

Since 2019, Tensorflow has officially released version 2 and lots of new API or functionalities are introduced, e.g. `tf.function`, etc. In the new version of Tensorflow, the Keras APIs were merged into the Tensorflow Core and were updated to operate the Tensorflow 2 core.  However, Tensorflow version 1 is still updating and upgrading so the docs and scripts still remain.

The document to this repository :

* **Tensorflow 1** : **https://ppt.cc/fl8Qex**
* **Tensorflow 2**

The following are the list for the main reference.

* **Keras Documentation** : https://keras.io/
* **Tensorflow/Models** : https://github.com/tensorflow/models
* **Tensorflow Official Website**: http://tensorflow.org/

## Content

### Tensorflow 1: 2015 ~ Now

At the beginning of each framework, we introduced the basic model training process and its implementation script as well. After that, we introduced two different types of advanced model training that are (1) in multiple graphics cards (GPUs) and (2) in multiple machines/containers with multiple GPUs.

We used [CIFAR-10 image dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and tried to train a deep neural network model recognizing them (a typical classification problem) as an example.

#### Keras

* Basic Concept and Implementation : [keras_basis](keras_basis/)
* Multiple GPUs : [keras_multiple_gpus](keras_multiple_gpus/)

#### Tensorflow

* Basic Concept and Implementation : [tensorflow_basis](tensorflow_basis/)
* Multiple GPUs : [tensorflow_multiple_gpus](tensorflow_multiple_gpus/)
* Training on a Cluster : [tensorflow_cluster](tensorflow_cluster/)

### Tensorflow 2: 2019 ~ Now

In Tensorflow 2, the Keras APIs are specified to the `Tensorflow.Keras` APIs, not the original Keras from `https://keras.io/`.

#### TF.Keras APIs

* A Training Workflow using `TF2.keras` on Multiple Accelerators: [TF2Keras_Distributed_Training](tf2keras_multiple_gpus/)

#### Tensorflow Core