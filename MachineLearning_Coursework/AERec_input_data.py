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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  
import csv
import tensorflow as tf
DATASET_DIR = 'ratings.csv'

def get_date_set():
    DATA_SET =np.zeros(shape=[672,163950], dtype=float)
    with open(DATASET_DIR) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            DATA_SET[int(row[0])][int(row[1])] = (row[2])
    DATA_SET = tf.convert_to_tensor(DATA_SET,dtype=tf.float32)
    partitions = np.random.randint(0,2,(1,672))
    partitions = partitions.repeat(163950)
    partitions = tf.convert_to_tensor(partitions,dtype = tf.int32)
    partitions = tf.reshape(partitions,[672,163950])
    
    TRAIN_SET,TEST_SET = tf.dynamic_partition(DATA_SET,partitions,2)
    return TRAIN_SET,TEST_SET

with tf.Session() as sess:
    np.set_printoptions(threshold='nan')
    print(sess.run(get_date_set()))