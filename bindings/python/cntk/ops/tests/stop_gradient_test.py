# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the stop_gradient class.
"""

import numpy as np
import pytest
from .. import *
from ...axis import Axis

def test_stop_gradient():
  x = input_variable(shape=(2,), dynamic_axes=[Axis.default_batch_axis(), Axis("B")], needs_gradient=True)
  y = input_variable(shape=(2,), dynamic_axes=[Axis.default_batch_axis(), Axis("B")], needs_gradient=True)
  z = element_times(x, y);
  w = z + stop_gradient(z)
  a = np.reshape(np.float32([0.25,0.5,0.1,1]), (1,2,2))
  b = np.reshape(np.float32([-1.25,1.5,0.1,-1]), (1,2,2))
  bwd, fwd = w.forward({x:a, y:b}, [w.output], set([w.output]))
  value = list(fwd.values())[0]
  expected = np.multiply(a,b)*2
  assert np.allclose(value, expected)
  grad = w.backward(bwd, {w.output:np.ones_like(value)}, set([x, y]))
  assert np.allclose(grad[x], b)
  assert np.allclose(grad[y], a)
