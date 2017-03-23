import sys
import os
from datetime import datetime
import numpy as np
from cntk import Trainer, Axis, device, combine
from cntk.layers.blocks import Stabilizer, _initializer_for,  _INFERRED, Parameter, Placeholder
from cntk.layers import Recurrence, Convolution
from cntk.ops import input, cross_entropy_with_softmax, classification_error, sequence, reduce_sum, \
    parameter, times, element_times, past_value, plus, placeholder, reshape, constant, sigmoid, convolution, tanh, times_transpose, greater, cosine_distance, element_divide, element_select, exp, future_value, past_value
from cntk.internal import _as_tuple, sanitize_input
from cntk.initializer import uniform, glorot_uniform

def gru_cell(shape, init=glorot_uniform(), name=''): # (x, (h,c))
  """ GRU cell function
  """
  shape = _as_tuple(shape)

  if len(shape) != 1 :
    raise ValueError("gru_cell: shape must be vectors (rank-1 tensors)")

  # determine stacking dimensions
  cell_shape_stacked = shape * 2  # patched dims with stack_axis duplicated 2 times

  # parameters
  Wz = Parameter(cell_shape_stacked, init = init, name='Wz')
  Wr = Parameter(cell_shape_stacked, init = init, name='Wr')
  Wh = Parameter(cell_shape_stacked, init = init, name='Wh')
  Uz = Parameter(_INFERRED + shape, init = init, name = 'Uz')
  Ur = Parameter(_INFERRED + shape, init = init, name = 'Ur')
  Uh = Parameter(_INFERRED + shape, init = init, name = 'Uh')

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

def seq_max(x, broadcast=True, name=''):
  """
  Get the max value in the sequence values

  Args:
    x: input sequence
    broadcast: if broadcast is True, the max value will be broadcast along with the input sequence,
    else only a single value will be returned
    name: the name of the operator
  """
  m = placeholder(shape=(1,), dynamic_axes = x.dynamic_axes, name='max')
  o = element_select(greater(x, future_value(m)), x, future_value(m))
  rlt = o.replace_placeholders({m:sanitize_input(o)})
  if broadcast:
    pv = placeholder(shape=(1,), dynamic_axes = x.dynamic_axes, name='max_seq')
    max_seq = element_select(sequence.is_first(x), sanitize_input(rlt), past_value(pv))
    max_out = max_seq.replace_placeholders({pv:sanitize_input(max_seq)})
  else:
    max_out = sequence.first(rlt) 
  return sanitize_input(max_out)

def seq_softmax(x, name = ''):
  """
  Compute softmax along with a squence values
  """
  x_exp = exp((x-seq_max(x))*10)
  x_softmax = element_divide(x_exp, sequence.broadcast_as(sequence.reduce_sum(x_exp), x), name = name)
  return x_softmax

def cosine_similarity(src, tgt, name=''):
  """
  Compute the cosine similarity of two squences.
  Src is a sequence of length 1
  Tag is a sequence of lenght >=1
  """
  src_br = sequence.broadcast_as(src, tgt, name='src_broadcast')
  sim = cosine_distance(src_br, tgt, name)
  return sim

def project_cosine_sim(att_dim, init = glorot_uniform(), name=''):
  """
  Compute the project cosine similarity of two input sequences, where each of the input will be projected to a new dimention space (att_dim) via Wi/Wm
  """
  Wi = Parameter(_INFERRED + tuple((att_dim,)), init = init, name='Wi')
  Wm = Parameter(_INFERRED + tuple((att_dim,)), init = init, name='Wm')
  status = placeholder(name='status')
  memory = placeholder(name='memory')
  projected_status = times(status, Wi, name = 'projected_status')
  projected_memory = times(memory, Wm, name = 'projected_memory')
  sim = cosine_similarity(projected_status, projected_memory, name= name+ '_sim')
  return seq_softmax(sim, name = name)

def termination_gate(init = glorot_uniform(), name=''):
  Wt = Parameter( _INFERRED + tuple((1,)), init = init, name='Wt')
  status = placeholder(name='status')
  return sigmoid(times(status, Wt), name=name)
