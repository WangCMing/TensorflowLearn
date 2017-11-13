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
import tensorflow as tf
import csv

DATASET_DIR = 'ratings.csv'

def get_date_set():
    TRAIN_SET = np.zeros(shape=[601,163950], dtype=float)
    TEST_SET = np.zeros(shape=[72,163950], dtype=float)
    with open(DATASET_DIR) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            if((int(row[0]))>=600):
                break
            TRAIN_SET[int(row[0])][int(row[1])] = (row[2])
        print("break_______________")
        for row in f_csv:
            TEST_SET[int(row[0])-600][int(row[1])] = (row[2])
    return DATASET_DIR,TEST_SET