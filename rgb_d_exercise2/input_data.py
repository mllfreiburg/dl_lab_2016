# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functions for downloading and reading rgbd_10 objects data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os
import scipy.misc
import glob
import re

import numpy
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

SOURCE_URL = 'http://www2.informatik.uni-freiburg.de/~eitel/deep_learning_course/rgbd_10/data/'
IMAGE_SIZE = 32

if (int(scipy.__version__.split('.')[1]) < 17):
  print('Minimum required scipy version:', 1.7)  
  exit(1)
  
def maybe_download(filename, work_directory):
  """Download the data from the website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    print('Downloading data ...')
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', SOURCE_URL+filename, 'to', filepath, statinfo.st_size, 'bytes.')
    extract_tar(filename, work_directory)
  return filepath

def pad_image(img):
  imsz = img.shape
  mxdim  = numpy.max(imsz)

  offs_col = int((mxdim - imsz[1])/2)
  offs_row = int((mxdim - imsz[0])/2)  

  img_canvas = numpy.zeros(  (mxdim,mxdim,img.shape[2]), dtype='uint8' )
  img_canvas.fill(128)
  img_canvas[offs_row:offs_row+int(imsz[0]), offs_col:offs_col+int(imsz[1])] = img
  return (img_canvas)

def load_images(image_set, filepath, end):
  print('Loading rgb', len(image_set), 'images from', filepath)
  images = numpy.empty((len(image_set), IMAGE_SIZE , IMAGE_SIZE , 3), dtype=numpy.float32)
  for i in xrange(0,len(image_set)):
    img = scipy.misc.imread(os.path.join(filepath, image_set[i]+end), False, 'RGB')
    img = pad_image(img)
    img = scipy.misc.imresize(img, (IMAGE_SIZE,IMAGE_SIZE,3), 'bilinear')
    # Convert from [0, 255] -> [0.0, 1.0].
    img = img.astype(numpy.float32)
    img = numpy.multiply(img, 1.0 / 255.0)
    images[i] = img

  return images

def load_rgbd_images(image_set, filepath_rgb, filepath_depth, end):
  print('Loading rgbd', len(image_set), 'images from', filepath_rgb, 'and', filepath_depth)
  channels = 4 
  images = numpy.empty((len(image_set), IMAGE_SIZE , IMAGE_SIZE , channels), dtype=numpy.float32)
  for i in xrange(0,len(image_set)):
    # RGB image
    img = scipy.misc.imread(os.path.join(filepath_rgb, image_set[i]+end), False, 'RGB')
    img = pad_image(img)
    img = scipy.misc.imresize(img, (IMAGE_SIZE,IMAGE_SIZE,3), 'bilinear')
    # Convert from [0, 255] -> [0.0, 1.0].
    img = img.astype(numpy.float32)
    img = numpy.multiply(img, 1.0 / 255.0)

    # Depth image
    depth = scipy.misc.imread(os.path.join(filepath_depth, image_set[i].replace("crop", "depthcrop")+end), False, 'F')
    depth = depth.reshape(depth.shape[0], depth.shape[1],1)
    depth = pad_image(depth)
    depth = scipy.misc.toimage(depth[:,:,0])
    depth = scipy.misc.imresize(depth, (IMAGE_SIZE,IMAGE_SIZE), 'bilinear', 'F')
    depth = numpy.asarray(depth)
    # Convert from [0, max(depth)) -> [0.0, 1.0].
    depth = depth.astype(numpy.float32)
    depth = numpy.multiply(depth, 1.0/numpy.max(depth))

    rgbd = numpy.zeros(  (IMAGE_SIZE,IMAGE_SIZE,channels), dtype=numpy.float32 )
    rgbd[:,:,0] = img[:,:,0]
    rgbd[:,:,1] = img[:,:,1]
    rgbd[:,:,2] = img[:,:,2]
    rgbd[:,:,3] = depth
    images[i] = rgbd 
  return images  

def load_labels(image_set, label_file):
  label_db = {}
  with open(label_file) as f:
    for line in f.readlines():
      key = line.split(' ')[0]
      label = line.split(' ')[1]
      label_db[key] = int(label)
  labels = numpy.empty((len(image_set)), dtype=numpy.uint8)
  for i in xrange(0,len(image_set)):
    first_digit = re.search("\d", image_set[i])
    category = image_set[i][:(first_digit.start()-1)]
    labels[i] = label_db[category]
  return labels  

def extract_tar(filename, train_dir):
  print('Extracting', filename)
  tar =  tarfile.open(os.path.join(train_dir, filename))
  tar.extractall(path=train_dir)  


def load_image_set_index(image_set, train_dir):
  """Load the indexes listed in this dataset's image set file."""
  # Example path to image set file:
  # self._data_path + /ImageSets/val.txt
  image_set_file = os.path.join(train_dir, 'sets', 
                                image_set + '.txt')
  assert os.path.exists(image_set_file), \
          'Path does not exist: {}'.format(image_set_file)
  with open(image_set_file) as f:
      instance_index = [x.strip() for x in f.readlines()]
  print(image_set,'set:', instance_index)
  image_index = []
  counter = 0
  for index in instance_index:
    search_result = glob.glob(train_dir+'/images/'+index+'_*_*_crop.png')
    counter +=len(search_result)
    for result in search_result:
      ind = (result.split('/')[-1]).split('.')[:-1][0]
      image_index.append(ind)
  print('Number of datapoints', image_set, ':', len(image_index))
  return image_index

class DataSet(object):

  def __init__(self, images, labels, one_hot=False, rgbd=False):

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if rgbd:
      assert images.shape[3] == 4

    else:
      assert images.shape[3] == 3
    # images = images.reshape(images.shape[0],
    #                         images.shape[1] * images.shape[2]*images.shape[3])

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed


def read_data_sets(train_dir, rgbd, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  IMAGES = 'images.tar.gz'
  DEPTH_IMAGES='depth.tar.gz'
  LABELS = 'labels.tar.gz'
  IMAGE_SETS = 'sets.tar.gz'

  # Dowload and extract data
  local_file = maybe_download(IMAGES, train_dir)
  local_file = maybe_download(IMAGE_SETS, train_dir)
  local_file = maybe_download(LABELS, train_dir)
  if (rgbd):
    local_file = maybe_download(DEPTH_IMAGES, train_dir)

  # Load train/val/test sets 
  train_set = load_image_set_index('train', train_dir)
  validation_set = load_image_set_index('validation', train_dir)
  test_set = load_image_set_index('test', train_dir)
  random.shuffle(train_set)
  random.shuffle(validation_set)
  random.shuffle(test_set)

  # Load images
  filepath = os.path.join(train_dir, IMAGES.split('.')[-3])
  train_images = numpy.array([])  
  validation_images = numpy.array([])
  test_images = numpy.array([]) 
  if (rgbd):
    filepath_depth = os.path.join(train_dir, DEPTH_IMAGES.split('.')[-3])
    train_images = load_rgbd_images(train_set, filepath, filepath_depth, '.png')
    validation_images = load_rgbd_images(validation_set, filepath, filepath_depth, '.png')
    test_images = load_rgbd_images(test_set, filepath, filepath_depth, '.png')
  else:
    train_images = load_images(train_set, filepath, '.png')
    validation_images = load_images(validation_set, filepath, '.png')
    test_images = load_images(test_set, filepath, '.png')

  
  # Load labels
  filepath = os.path.join(train_dir, LABELS.split('.')[-3], LABELS.split('.')[-3]+'.txt')
  train_labels = load_labels(train_set, filepath)
  validation_labels = load_labels(validation_set, filepath)
  test_labels = load_labels(test_set, filepath)

  # Create dataset
  data_sets.train = DataSet(train_images, train_labels, False, rgbd)
  data_sets.validation = DataSet(validation_images, validation_labels, False, rgbd)
  data_sets.test = DataSet(test_images, test_labels, False, rgbd)
  print('Finished dataset creation')
  return data_sets
