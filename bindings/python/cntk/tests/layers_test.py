# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

from ..layers import *
from ..blocks import  init_default_or_glorot_uniform, Parameter, _INFERRED, Placeholder
from ..utils import _as_tuple
from ..ops import  sigmoid, times, tanh, element_times, plus, combine, input_variable
from ..axis import Axis

def test_layers_name(device_id): 
    from cntk import placeholder_variable, combine
    I = placeholder_variable(name='input')
    p = Dense(10, name='dense10')(I)
    assert(I.name == 'input')
    assert(p.root_function.name == 'dense10')
    
    q = Convolution((3,3), 3, name='conv33')(I)
    assert(q.root_function.name == 'conv33')

def gru_cell(shape, init=init_default_or_glorot_uniform, name=''): # (x, (h,c))
  shape = _as_tuple(shape)

  if len(shape) != 1 :
    raise ValueError("gru_cell: shape must be vectors (rank-1 tensors)")

  # determine stacking dimensions
  cell_shape_stacked = shape * 2  # patched dims with stack_axis duplicated 4 times

  # parameters
  Wz = Parameter(cell_shape_stacked, init = init, name='Wz')
  Wr = Parameter(cell_shape_stacked, init = init, name='Wr')
  Wh = Parameter(cell_shape_stacked, init = init, name='Wh')
  Uz = Parameter( _INFERRED + shape, init = init, name = 'Uz')
  Ur = Parameter( _INFERRED + shape, init = init, name = 'Ur')
  Uh = Parameter( _INFERRED + shape, init = init, name = 'Uh')

  def create_s_placeholder():
    # we pass the known dimensions here, which makes dimension inference easier
    return Placeholder(shape=shape, name='S') # (h, c)

  # parameters to model function
  x = Placeholder(name='gru_block_arg')
  prev_status = create_s_placeholder()

  # formula of model function
  Sn_1 = prev_status

  z = sigmoid(times(x, Uz, name='x*Uz') + times(Sn_1, Wz, name='Sprev*Wz'), name='z')
  r = sigmoid(times(x, Ur, name='x*Ur') + times(Sn_1, Wr, name='Sprev*Wr'), name='r')
  h = tanh(times(x, Uh, name='x*Uh') + times(element_times(Sn_1, r, name='Sprev*r'), Wh), name='h')
  s = plus(element_times((1-z), h, name='(1-z)*h'), element_times(z, Sn_1, name='z*SPrev'), name=name)
  apply_x_s = combine([s])
  apply_x_s.create_placeholder = create_s_placeholder
  return apply_x_s

def test_recurrence():
  r = Recurrence(gru_cell(5), go_backwards=False)
  a = input_variable(shape=(5,), dynamic_axes=[Axis.default_batch_axis(), Axis('Seq')])
  x = np.reshape(np.arange(0,25, dtype=np.float32), (1,5,5))
  rt = r(a).eval({a:x})
  print(rt)
