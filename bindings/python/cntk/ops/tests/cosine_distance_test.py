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
  expected = [[ 1.,        0.914659,  0.878459,  0.86155,   0.851852]] 
  assert np.allclose(val, expected)


def test_cos_distane_backward():
  x = input_variable(shape=(2,), dynamic_axes=[Axis.default_batch_axis(), Axis("B")], needs_gradient=True)
  y = input_variable(shape=(2,), dynamic_axes=[Axis.default_batch_axis(), Axis("B")], needs_gradient=True)
  z = cosine_distance(x, y);
  a = np.reshape(np.float32([0.25,0.5,0.1,1]), (1,2,2))
  b = np.reshape(np.float32([-0.5,1.5,-0.3,-1]), (1,2,2))
  bwd, fwd = z.forward({x:a, y:b}, [z.output], set([z.output]))
  value = list(fwd.values())[0]
  expected = [[0.707107, -0.981665]]
  assert np.allclose(value, expected)
  grad = z.backward(bwd, {z.output:np.ones_like(value)}, set([x, y]))
  x_driv_expected = np.ndarray((1,2,2), dtype=np.float32, buffer=np.float32([-1.131371, 0.565686, -0.188727, 0.018873]))
  y_driv_expected = np.ndarray((1,2,2), dtype=np.float32, buffer = np.float32([0.424264, 0.141421,-0.174876, 0.052463]))
  assert (np.all(np.absolute(grad[x]-x_driv_expected) < 1e-6))
  assert (np.all(np.absolute(grad[y]-y_driv_expected) < 1e-6))
