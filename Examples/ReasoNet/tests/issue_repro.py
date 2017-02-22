import sys
import os
import numpy as np
from cntk.blocks import Placeholder, Constant, Parameter
import cntk.ops as ops
from cntk import Axis
from cntk.utils import  _as_tuple, sanitize_input 
import cntk.utils as utils
import cntk.learner as learner
from cntk.ops import sequence


def test_seq_reduce_sum():
  axies = [Axis.default_batch_axis(), Axis("Seq")]
  seq = ops.input_variable(shape=(2,), dtype=np.float32, dynamic_axes=axies, needs_gradient=True, name='seq')
  wei = ops.input_variable(shape=(1,), dtype=np.float32, dynamic_axes=axies, needs_gradient=True, name='wei')
  ssum = ops.seq_reduce_sum(seq, wei)
  x = np.reshape(np.float32([1,2, 1,2, 1,2, 3,4, -1,-3, -2,-5]), (3,2,2))
  w = np.reshape(np.float32([1,1,1,0.2,0.5, 0.1]), (3,2,1))
  arg_map = {seq:x, wei: w}
  bwd, out_map = ssum.forward(arg_map, [ssum.output], set([ssum.output]))
  value = list(out_map.values())[0]
  print(value)
  expected = np.sum(np.multiply(x, w), 1)
  print(expected)
  grad = ssum.backward(bwd, {ssum.output: np.ones_like(value)}, set([seq, wei]))
  print(grad)
  print()
  print(x)
  print(w)

test_seq_reduce_sum()
