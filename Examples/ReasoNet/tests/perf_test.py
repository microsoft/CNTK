import sys
import time
import os
import numpy as np
from cntk.blocks import Placeholder, Constant, Parameter
import cntk.ops as ops
from cntk import Axis
from cntk.utils import  _as_tuple, sanitize_input 
import cntk.utils as utils
import cntk.learner as learner
from cntk.ops import sequence

def gen_data(*shape):
  return np.float32(np.random.rand(*shape))

def test_cos_distance():
  x = ops.input_variable(shape=(768,), dynamic_axes=[Axis.default_batch_axis(), Axis("B")], needs_gradient=True)
  y = ops.input_variable(shape=(768,), dynamic_axes=[Axis.default_batch_axis(), Axis("B")], needs_gradient=True)
  #q = sequence.first(x)
  #w = sequence.broadcast_as(q, y)
  z = ops.cosine_distance(x, y);
  #z = ops.constant_ref(z0)
  batch_num = 40
  batch_size = 30
  a = gen_data(batch_num*batch_size,1500,768) 
  b = gen_data(batch_num*batch_size,1500,768) 
  #o = z.eval({y:b})
  #print(o)
  i = 0
  bwd, fwd = z.forward({x:a[i*batch_size:(i+1)*batch_size], y:b[i*batch_size:(i+1)*batch_size]}, [z.output], set([z.output]))
  forward_total = 0
  backward_total = 0
  grad_x = np.ndarray((batch_size, 1500, 768), np.float32)
  grad_y = np.ndarray((batch_size, 1500, 768), np.float32)
  value = np.ndarray((batch_size, 1500), np.float32)
  value = 0
  grad_x = 0
  grad_y = 0
  for i in range(batch_num):
    t0 = time.clock()
    bwd, fwd = z.forward({x:a[i*batch_size:(i+1)*batch_size], y:b[i*batch_size:(i+1)*batch_size]}, [z.output], set([z.output]))
    t1 = time.clock()
    #value += fwd[z.output]
    t2 = time.clock()
    grad = z.backward(bwd, {z.output:np.ones_like(fwd[z.output])}, set([x, y]))
    t3 = time.clock()
    grad_x += grad[x]
    grad_y += grad[y]
    forward_total += t1-t0
    backward_total += t3-t2
  #print(value[0,0:10])
  print(grad_x[0,0, 0:10])
  print(grad_y[0,0, 0:10])
  print("forward: {0}, backward: {1}".format(forward_total, backward_total))

test_cos_distance()
