import sys
import os
import numpy as np
from ReasoNet.model import create_model, create_reader, gru_cell
import ReasoNet.asr as asr
from cntk.blocks import Placeholder, Constant,initial_state_default_or_None, _is_given, _get_current_default_options
from cntk.ops import input_variable, past_value, future_value
from cntk.io import MinibatchSource
from cntk import Trainer, Axis, device, combine
from cntk.layers import Recurrence, Convolution
from cntk.utils import _as_tuple

def testModel(data):
  reader = create_reader(data, 10, False)
  model = create_model(10, 5, 3)
  data_bind = {}
  for arg in model.arguments:
    if arg.name == 'query':
      data_bind[arg] = reader.streams.query
    if arg.name == 'context':
      data_bind[arg] = reader.streams.context
    if arg.name == 'entities':
      data_bind[arg] = reader.streams.entities
  batch = reader.next_minibatch(1, data_bind)
  var = model.eval(batch)
  for o in model.outputs:
    print('-----------------------')
    print(o.name)
    print(np.around(var[o], decimals=3))

def testASR(data):
  reader = asr.create_reader(data, 10, False)
  model = asr.create_model(10, 5, 1)
  data_bind = {}
  for arg in model.arguments:
    if arg.name == 'query':
      data_bind[arg] = reader.streams.query
    if arg.name == 'context':
      data_bind[arg] = reader.streams.context
    if arg.name == 'entities':
      data_bind[arg] = reader.streams.entities
  batch = reader.next_minibatch(1, data_bind)
  var = model.eval(batch)
  for o in model.outputs:
    print('-----------------------')
    print(o.name)
    print(np.around(var[o], decimals=3))


def testGRU():
  g = gru_cell(5)
  x = np.reshape(np.arange(0,25, dtype=np.float32), (1,5,5))
  a = input_variable(shape=(5,), dynamic_axes=[Axis.default_batch_axis(), Axis('Seq')])
  y = np.float32([1,2,0.1,0.2,1])
  s = Constant(y)
  q = g(a,s).eval({a:x})
  print(q)
  r = Recurrence(gru_cell(5))
  rt = r(a).eval({a:x})
  print(rt)

#testGRU()
#testASR("test.idx")
testModel("test.idx")
