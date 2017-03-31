# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the sequnce_softmax.
"""

import numpy as np
import pytest
from .. import *
from cntk.losses import *
from ...axis import Axis
from ... import sequence, input

def test_sequnce_max():
  np.random.seed(0)
  a = np.float32(np.random.rand(20,100,1))
  src = sequence.input(shape=(1), sequence_axis=Axis("Seq"))
  out = sequence.reduce_max(src)
  val = out.eval({src:a})
  expected = np.max(a, 1) 
  assert np.allclose(val, expected)

def np_softmax(a):
  m = np.reshape(np.repeat(np.max(a, 1), a.shape[1], 1), a.shape)
  e = np.exp((a-m)*10)
  s = np.reshape(np.repeat(np.sum(e, 1), a.shape[1], 1), a.shape)
  return e/s
  
def test_sequnce_softmax():
  np.random.seed(0)
  a = np.float32(np.random.rand(20,100,1))
  src = sequence.input(shape=(1), sequence_axis=Axis("Seq"))
  out = sequence.softmax(src)
  val = out.eval({src:a})
  expected = np_softmax(a)
  assert np.allclose(val, expected)
