import sys
import os
import numpy as np
from ReasoNet.model import text_convolution 
from ReasoNet.model import gru_cell 
from cntk.blocks import Placeholder, Constant
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, reduce_sum, \
    parameter, times, element_times, past_value, plus, placeholder_variable, splice, reshape, constant, sigmoid, convolution, cosine_distance, times_transpose
from cntk import Axis

def testTextConvolution():
  text = np.reshape(np.arange(25.0, dtype=np.float32), (1, 5,5))
  x = input_variable(shape=(1, 5, 5,)) 
  c = text_convolution(3, 5, 5)(x)
  v = c.eval([text])[0]
  print(v)

def testSplice():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, -25, -1, dtype=np.float32), (1,5,5))
  va = constant(value=a)
  vb = constant(value=b)
  vc = splice((va, vb), 2)
  print(vc.shape)
  print(vc.eval())

def cossim(a,b):
  src = constant(a)
  tgt = constant(b)
  val = cosine_distance(src, tgt).eval()
  return val

def testCosinDistance():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1,5))
  
  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
  tgt = input_variable(shape=(5))
  tgt_br = sequence.broadcast_as(tgt, src)
  cos_seq = cosine_distance(src, tgt_br)
  val = cos_seq.eval({src:[a], tgt:[b]})
  print("Cosine similarity\r\n{0}\r\n  #\r\n{1}".format(a,b))
  print("==>")
  print(val)
  print("==================")
  print("Expected: ")
  for i in range(0, 5): 
    print("{0}:{1}".format(i, cossim(a[i], b[0])))

def dotproduct(a,b):
  src = constant(a)
  tgt = constant(b)
  val = reduce_sum(element_times(src, tgt))
  return val

def testReduceSum():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1,5))
  
  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
  tgt = input_variable(shape=(5))
  tgt_br = sequence.broadcast_as(tgt, src)
  reduceSum = reduce_sum(element_times(src, tgt_br), axis=0)
  val = reduceSum.eval({src:[a], tgt:[b]})
  print("Reduce_sum\r\n{0}\r\n  #\r\n{1}".format(a,b))
  print("==>")
  print(val)
  print("==================")
  print("Expected: ")
  for i in range(0, 5): 
    print("{0}:{1}".format(i, dotproduct(a[i], b[0]).eval()))


def testElementTimes():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (5))
  
#  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
#  tgt = input_variable(shape=(5))
#  tgt_br = sequence.broadcast_as(tgt, src)
#  reduceSum = reduce_sum(element_times(src, tgt_br), axis=0)
  val = element_times(b.reshape(5,1),a).eval()
  print("ElementTimes\r\n{0}\r\n  #\r\n{1}".format(a,b))
  print("==>")
  print(val)
  print("==================")
  print("Expected: ")
  for i in range(0, 5): 
    print("{0}:{1}".format(i, element_times(a[i], b[i]).eval()))

def testElementTimes2():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1, 5))
  
#  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
#  tgt = input_variable(shape=(5))
#  tgt_br = sequence.broadcast_as(tgt, src)
#  reduceSum = reduce_sum(element_times(src, tgt_br), axis=0)
  val = reduce_sum(element_times(b,a).eval(), axis=1).eval()
  print("ElementTimes\r\n{0}\r\n  #\r\n{1}".format(a,b))
  print("==>")
  print(val)
  print("==================")
  print("Expected: ")
  for i in range(0, 5): 
    print("{0}:{1}".format(i, dotproduct(a[i], b[0]).eval()))

def testTimesTranspose():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1, 5))
  
#  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
#  tgt = input_variable(shape=(5))
#  tgt_br = sequence.broadcast_as(tgt, src)
#  reduceSum = reduce_sum(element_times(src, tgt_br), axis=0)
  val = times_transpose(a,b).eval()
  print("ElementTimes\r\n{0}\r\n  #\r\n{1}".format(a,b))
  print("==>")
  print(val)
  print("==================")
  print("Expected: ")
  for i in range(0, 5): 
    print("{0}:{1}".format(i, dotproduct(a[i], b[0]).eval()))

def testGRU():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1,5))
  src = input_variable(shape=(5, ))
  #src = input_variable(shape=(1, 5, ), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
  tgt = constant(b)
  gru = gru_cell(5)
  o_0 = gru(src, tgt)
  sgru = gru(src, o_0.output).output[0]
  print(sgru.eval(a))
  

def reduce_times_sum(first, second):
  # define a recursive expression for \sum_{i=1}^t (first_i * second_i)
  running_sum_ph = placeholder_variable(shape=first.shape)
  print("Second: {0}".format(second.shape))
  print("Fist:{0}".format(first.shape))
  t = times(second, first)
  print("Times: {0}".format(t.output.shape))
  return t
  #running_sum = plus(reshape(times(second, first), shape=(5)), past_value(running_sum_ph))

  #print("Plus: {0}".format(running_sum.output.shape))
  #running_sum.replace_placeholders({running_sum_ph : running_sum.output})
  #return sequence.last(running_sum)

def testReduceTimesSum():
  a = np.reshape(np.float32([[0,1,0,0,0],[1,0,0,0,0],[1,0,0,0,0], [0,0,0,1,0],[0,0,0,0,1]]), (5,5))
  b = np.reshape(np.float32([0.1, 0.2, 0.2, 0.1, 0.4]), (1,5))
  src = input_variable(shape=(5, 5,))
#  src = input_variable(shape=(1, 5, ), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
  tgt = constant(b)
  print(reduce_times_sum(src,tgt).eval([a]))


