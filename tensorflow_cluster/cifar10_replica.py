# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2018 JiankaiWang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed CIFAR10 training and validation, with model replicas.

There are three CNN models included, Resnet_v2, Inception_v3 and a simple MLP.
A simple CNN model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.

@version: v1
@version:
  keras: 2.1.6
  tensorflow: 1.11.0
@notice:
  add functions building network _inception_v3(), _resnet_v2(), _mlp()
  add data generator and prefetch_queue mechanism
  add cnn_model selection
  add a saver
  edit validation feed
@Usage:
for a parameter server,
python cifar10_replica.py --job_name="ps" --task_index=0 --num_gpus=0

for workers,
python cifar10_replica.py --job_name="worker" --task_index=0 --num_gpus=1 \
                          --train_steps=2001 --sync_replicas=True
python cifar10_replica.py --job_name="worker" --task_index=1 --num_gpus=1 \
                          --train_steps=2001 --sync_replicas=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time
import numpy as np

import math
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
import cifar10_dataset as cifar10_input

# In[]

flags = tf.app.flags
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update "
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 20, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean(
    "sync_replicas", False,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_string("cnn_model", "mlp", 
                    "resnet_v2, inception_v3 or mlp")
flags.DEFINE_string("train_dir", "/tmp/cifar10_model", "path for saving ckpts")

FLAGS = flags.FLAGS

RAW_IMAGE_PIXELS = 32
IMAGE_PIXELS = 24
NUMS_CLS = 10

# In[]

def _inception_v3(x):
  with tf.name_scope("inception_v3"):
    x_ = tf.image.resize_images(x, (299,299))
    y, _ = inception.inception_v3(x_, num_classes=10, is_training=True)
    return y
  
def _resnet_v2(x):
  with tf.name_scope("resnet_v2"):
    x_ = tf.image.resize_images(x, (224,224))
    y, _ = resnet_v2.resnet_v2_101(x_, num_classes=10, is_training=True)
    return y
  
def _mlp(x):
  with tf.name_scope("mlp"):
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS * 3, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
    
    _x = tf.image.resize_images(x, [IMAGE_PIXELS, IMAGE_PIXELS])
    x_ = tf.reshape(_x, [-1, IMAGE_PIXELS * IMAGE_PIXELS * 3])
    hid_lin = tf.nn.xw_plus_b(x_, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)
    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))    
    return y
  
# In[]

def main(unused_argv):  
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  
  # load the dataset
  images, labels = cifar10_input.distorted_inputs(FLAGS.batch_size)
  
  batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity = 2 * 2)  
  
  image_batch, label_batch = batch_queue.dequeue()
  label_batch = tf.one_hot(label_batch, NUMS_CLS)

  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU
  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Here we construct a simple training model. 
    # Ops: located on the worker specified with FLAGS.task_index
    x = image_batch
    y_ = label_batch
    
    if FLAGS.cnn_model == "resnet_v2":
      # we build a trainable model based on slim
      y = _resnet_v2(x)

      cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    elif FLAGS.cnn_model == "inception_v3":
      # we build a trainable model based on slim
      y = _inception_v3(x)
      
      cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    else:      
      # we build a simple mlp network
      y = _mlp(x)
      
      cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.sync_replicas:
      if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
      else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="cifar10_sync_replicas")

    train_step = opt.minimize(cross_entropy, global_step=global_step)

    if FLAGS.sync_replicas:
      local_init_op = opt.local_step_init_op
      if is_chief:
        local_init_op = opt.chief_init_op

      ready_for_local_init_op = opt.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()
      
    # create a saver
    saver = tf.train.Saver()   

    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()

    if FLAGS.sync_replicas:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    else:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          recovery_wait_secs=1,
          global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps",
                        "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    while True:
      # Training with no feed due to queue handling
      _, step = sess.run([train_step, global_step])
      local_step += 1

      now = time.time()
      print("%f: Worker %d: training step %d done (global step: %d)" %
            (now, FLAGS.task_index, local_step, step))

      if step >= FLAGS.train_steps:
        break
      
      elif step % 1000 == 0:
        # save the checkpoint
        ckpt_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, ckpt_path, global_step=step)

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
    
    # get val dataset
    _x_test, _y_test = cifar10_input._load_cifar10_val_dataset(FLAGS.batch_size)
    
    # Validation feed    
    _pred, = sess.run([y], feed_dict={x: _x_test})
    pred = np.argmax(_pred, axis=1)
    truth = np.argmax(_y_test, axis=1)
    accuracy = np.sum(np.equal(truth, pred).astype("int")) / truth.shape[0]
    print("After %d training step(s), accuracy = %g" %
          (FLAGS.train_steps, accuracy))    

if __name__ == "__main__":
  tf.app.run()
