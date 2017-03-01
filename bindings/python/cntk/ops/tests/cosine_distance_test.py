# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the cosine distance class.
"""

import numpy as np
import pytest
from .. import *
from ...axis import Axis
from ... import sequence

def test_cosine_distance():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1,5))
  
  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
  tgt = input_variable(shape=(5))
  tgt_br = sequence.broadcast_as(tgt, src)
  cos_seq = cosine_distance(src, tgt_br)
  assert len(cos_seq.dynamic_axes)==2
  assert cos_seq.dynamic_axes[1].name=="Seq"
  val = cos_seq.eval({src:[a], tgt:[b]})
  expected = [[ 1., 0.914659,  0.878459,  0.86155,   0.851852]] 
  assert np.allclose(val, expected)

def test_cosine_distance_with_negative_samples():
  a = np.array([[1., 1., 0., 0., 0.],
                [0., 1., 1., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 1., 1.],
                [1., 0., 0., 0., 1.]], dtype=np.float32)
  b = np.array([[1., 1., 0., 0., 0.],
                [0., 1., 1., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 1., 1.],
                [1., 0., 0., 0., 1.]], dtype=np.float32)

  qry = input_variable(shape=(5))
  doc = input_variable(shape=(5))
  num_neg_samples = 2
  model = cosine_distance_with_negative_samples(qry, doc, shift=1, num_negative_samples=num_neg_samples)
  result = model.eval({qry:[a], doc:[b]})

  # We expect 1 row per minibatch
  np.allclose(result.shape[1], a.shape[0])

  # We expect the number of columns to be number of negative samples + 1
  np.allclose(result.shape[2], num_neg_samples+1)

  # The first value is exact match, second ony 1 element match and last one is 0 match
  np.allclose(result[0], np.tile([1, 0.5, 0.], (a.shape[0],1)))

