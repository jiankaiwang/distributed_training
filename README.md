# Model Training in Keras and Tensorflow

This repository is a tutorial targeting how to train a deep neural network model in a higher efficient way. In this repository, we focus on two main frameworks that are Keras and Tensorflow. Parts of the content (scripts, documents, etc.) were referred from Tensorflow/Model repository and Keras document. The relative framework version please refer to the top description of each script. **Notice the backend of Keras is Tensorflow in this repository.**

The document to this repository :

*   **Google Doc** : **https://ppt.cc/fl8Qex**

The following are the list for the main reference.

*   **Keras Documentation** : https://keras.io/

*   **Tensorflow/Models** : https://github.com/tensorflow/models

## Content

At the beginning of each framework, we introduced the basic model training process and its implementation script as well. After that, we introduced two different types of advanced model training that are (1) in multiple graphics cards (GPUs) and (2) in multiple machines/containers with multiple GPUs.

We used [CIFAR-10 image dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and tried to train a deep neural network model recognizing them (a typical classification problem) as an example.

### Keras

*   Basic Concept and Implementation : [keras_basis](keras_basis/)
*   Multiple GPUs : [keras_multiple_gpus](keras_multiple_gpus/)

### Tensorflow

-   Basic Concept and Implementation : [tensorflow_basis](tensorflow_basis/)
-   Multiple GPUs : [tensorflow_multiple_gpus](tensorflow_multiple_gpus/)
-   Training on a Cluster : [tensorflow_cluster](tensorflow_cluster/)



