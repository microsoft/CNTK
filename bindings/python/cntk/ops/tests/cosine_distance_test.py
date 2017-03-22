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
from cntk.losses import *
from ...axis import Axis
from ... import sequence, input

def test_cosine_distance():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1,5))
  
  src = sequence.input(shape=(5), sequence_axis=Axis("Seq"))
  tgt = input(shape=(5))
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

  qry = sequence.input(shape=(5))
  doc = sequence.input(shape=(5))
  num_neg_samples = 2
  model = cosine_distance_with_negative_samples(qry, doc, shift=1, num_negative_samples=num_neg_samples)
  result = model.eval({qry:[a], doc:[b]})

  # We expect 1 row per minibatch
  np.allclose(result.shape[1], a.shape[0])

  # We expect the number of columns to be number of negative samples + 1
  np.allclose(result.shape[2], num_neg_samples+1)

  # The first value is exact match, second ony 1 element match and last one is 0 match
  np.allclose(result[0], np.tile([1, 0.5, 0.], (a.shape[0],1)))

def test_rank0_output():
  x = sequence.input(shape=(768,), sequence_axis=Axis("B"), needs_gradient=True)
  y = sequence.input(shape=(768,), sequence_axis=Axis("B"), needs_gradient=True)
  z = cosine_distance(x, y)
  batch_num = 2
  batch_size = 30
  a = np.float32(np.random.rand(batch_num*batch_size,1500,768))
  b = np.float32(np.random.rand(batch_num*batch_size,1500,768))
  for i in range(batch_num):
    bwd, fwd = z.forward({x:a[i*batch_size:(i+1)*batch_size], y:b[i*batch_size:(i+1)*batch_size]}, [z.output], set([z.output]))
    grad = z.backward(bwd, {z.output:np.ones_like(fwd[z.output])}, set([x, y]))


class numpy_cos:
  def __init__(self, a, b):
    self.a = a
    self.b = b
  
  def forward(self):
    self.dot = np.sum(self.a*self.b, -1)
    self.a_sqrt = np.sqrt(np.sum(np.square(self.a), -1)+1e-9)
    self.b_sqrt = np.sqrt(np.sum(np.square(self.b), -1)+1e-9)
    self.sim = self.dot/(self.a_sqrt*self.b_sqrt)
    return self.sim

  def backward(self):
    self.a_sqrt = np.reshape(self.a_sqrt, self.a_sqrt.shape + (1,))
    self.b_sqrt = np.reshape(self.b_sqrt, self.b_sqrt.shape + (1,))
    self.sim = np.reshape(self.sim, self.sim.shape + (1,))
    ga = self.b/(self.a_sqrt*self.b_sqrt) - self.sim*self.a/np.square(self.a_sqrt)
    gb = self.a/(self.a_sqrt*self.b_sqrt) - self.sim*self.b/np.square(self.b_sqrt)
    return {'a':ga, 'b':gb}

def test_cos_distane_backward():
  x = sequence.input(shape=(2,), sequence_axis=Axis("B"), needs_gradient=True)
  y = sequence.input(shape=(2,), sequence_axis=Axis("B"), needs_gradient=True)
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

def test_cos_distane_backward2():
  x = sequence.input(shape=(100,), sequence_axis=Axis("B"), needs_gradient=True)
  y = sequence.input(shape=(100,), sequence_axis=Axis("B"), needs_gradient=True)
  z = cosine_distance(x, y);
  np.random.seed(0)
  a = np.float32(np.random.rand(10,50,100))
  b = np.float32(np.random.rand(10,50,100))
  bwd, fwd = z.forward({x:a, y:b}, [z.output], set([z.output]))
  value = list(fwd.values())[0]
  expected_cos = numpy_cos(a,b)
  expected = expected_cos.forward()
  assert np.allclose(value, expected)
  grad = z.backward(bwd, {z.output:np.ones_like(value)}, set([x, y]))
  bwd = expected_cos.backward()
  x_driv_expected = bwd['a']
  y_driv_expected = bwd['b']
  assert (np.all(np.absolute(grad[x]-x_driv_expected) < 1e-6))
  assert (np.all(np.absolute(grad[y]-y_driv_expected) < 1e-6))

def test_cos_distane_backward3():
  x = sequence.input(shape=(100,), sequence_axis=Axis("B"), needs_gradient=True)
  z = cosine_distance(x, x);
  np.random.seed(0)
  a = np.float32(np.random.rand(10,50,100))
  b = a
  bwd, fwd = z.forward({x:a}, [z.output], set([z.output]))
  value = list(fwd.values())[0]
  expected_cos = numpy_cos(a,b)
  expected = expected_cos.forward()
  assert np.allclose(value, expected)
  grad = z.backward(bwd, {z.output:np.ones_like(value)}, set([x]))
  bwd = expected_cos.backward()
  x_driv_expected = bwd['a']+bwd['b']
  assert (np.all(np.absolute(grad[x]-x_driv_expected) < 1e-6))

