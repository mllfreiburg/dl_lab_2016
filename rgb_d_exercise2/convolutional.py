# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple, end-to-end, LeNet-5-like convolutional rgbd_10 model example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import input_data


# TODO
# These are some useful constants that you can use in your code.
# Feel free to ignore them or change them.
# TODO 
IMAGE_SIZE = 32
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 30 
EVAL_BATCH_SIZE = 1024
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
# This is where the data gets stored
TRAIN_DIR = 'data'


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def fake_data(num_images, channels):
  """Generate a fake dataset that matches the dimensions of rgbd_10 dataset."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, channels),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    NUM_CHANNELS = 1
    train_data, train_labels = fake_data(256, NUM_CHANNELS)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    num_epochs = 1
  else:
    if (FLAGS.use_rgbd):
      NUM_CHANNELS = 4
      print('****** RGBD_10 dataset ******') 
      print('* Input: RGB-D              *')
      print('* Channels: 4               *') 
      print('*****************************')
    else:
      NUM_CHANNELS = 3
      print('****** RGBD_10 dataset ******') 
      print('* Input: RGB                *')
      print('* Channels: 3               *') 
      print('*****************************')
    # Load input data
    data_sets = input_data.read_data_sets(TRAIN_DIR, FLAGS.use_rgbd)
    num_epochs = NUM_EPOCHS

    train_data = data_sets.train.images
    train_labels= data_sets.train.labels
    test_data = data_sets.test.images
    test_labels = data_sets.test.labels 
    validation_data = data_sets.validation.images
    validation_labels = data_sets.validation.labels

  train_size = train_labels.shape[0]

  # TODO:
  # After this you should define your network and train it.
  # Below you find some starting hints. For more info have
  # a look at online tutorials for tensorflow:
  # https://www.tensorflow.org/versions/r0.11/tutorials/index.html
  # Your goal for the exercise will be to train the best network you can
  # please describe why you did chose the network architecture etc. in
  # the one page report, and include some graph / table showing the performance
  # of different network architectures you tried.
  #
  # Your end result should be for RGB-D classification, however, you can
  # also load the dataset with NUM_CHANNELS=3 to only get an RGB version.
  # A good additional experiment to run would be to cmompare how much
  # you can gain by adding the depth channel (how much better the classifier can get)
  # TODO:
  
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # TODO
  # define your model here
  # TODO

  # TODO
  # compute the loss of the model here
  # TODO
  loss = 0

  # TODO
  # then create an optimizer to train the model
  # HINT: you can use the various optimizers implemented in TensorFlow.
  #       For example, google for: tf.train.AdamOptimizer()
  # TODO

  # TODO
  # Make sure you also define a function for evaluating on the validation
  # set so that you can track performance over time
  # TODO

  # Create a local session to run the training.
  with tf.Session() as sess:
    # TODO
    # Make sure you initialize all variables before starting the tensorflow training
    # TODO
    
    # Loop through training steps here
    # HINT: always use small batches for training (as in SGD in your last exercise)
    # WARNING: The dataset does contain quite a few images if you want to test something quickly
    #          It might be useful to only train on a random subset!
    # For example use something like :
    # for step in max_steps:
    # Hint: make sure to evaluate your model every once in a while
    # For example like so:
    #print('Minibatch loss: {}'.format(loss))
    #print('Validation error: {}'.format(validation_error_you_computed)
    
    # Finally, after the training! calculate the test result!
    # WARNING: You should never use the test result to optimize
    # your hyperparameters/network architecture, only look at the validation error to avoid
    # overfitting. Only calculate the test error in the very end for your best model!
    # if test_this_model_after_training:
    #     print('Test error: {}'.format(test_error))
    #     print('Confusion matrix:') 
    #      print(confusion_matrix(test_labels, numpy.argmax(eval_in_batches(test_data, sess), 1)))
    pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_rgbd',
      default=False,
      help='Use rgb-d input data (4 channels).',
      action='store_true'
  )
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.'
  )
  FLAGS = parser.parse_args()

  tf.app.run()
