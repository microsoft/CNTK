import sys
import os
modulePath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../..")
sys.path.insert(0, modulePath)
import cntk.device as device
import numpy as np
from ReasoNet.model import create_model, create_reader, gru_cell
import ReasoNet.model as rsn
import ReasoNet.asr as asr
from cntk.blocks import Placeholder, Constant,initial_state_default_or_None, _is_given, _get_current_default_options
from cntk.ops import input_variable, past_value, future_value
from cntk.io import MinibatchSource
from cntk import Trainer, Axis, device, combine
from cntk.layers import Recurrence, Convolution
from cntk.utils import _as_tuple
import cntk.ops as ops
import cntk
import ReasoNet.wordvocab as vocab
import ReasoNet.utils as utils
import math

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
  batch = reader.next_minibatch(2, data_bind)
  var = model.eval(batch)
  for o in model.outputs:
    print('-----------------------')
    print(o.name)
    print(np.around(var[o], decimals=3))

def testReasoNetLoss(data):
  reader = create_reader(data, 101000, False)
  model = create_model(101000, 300, 3)
  loss = rsn.loss(model)
  data_bind = {}
  for arg in loss.arguments:
    if arg.name == 'query':
      data_bind[arg] = reader.streams.query
    if arg.name == 'context':
      data_bind[arg] = reader.streams.context
    if arg.name == 'entities':
      data_bind[arg] = reader.streams.entities
    if arg.name == 'labels':
      data_bind[arg] = reader.streams.label
  batch = reader.next_minibatch(100, data_bind)
  var = loss.eval(batch)
  for o in loss.outputs:
    print('-----------------------')
    print(o.name)
    print(np.around(var[o], decimals=3))
  #pred = var[loss.outputs[-1]]
  #for i in pred:
  #  print("Prediction: {0}".format(np.argmin(np.reshape(i, 10))))

def testReasoNetPred(data):
  reader = create_reader(data, 10, False)
  model = create_model(10, 5, 3)
  pred = rsn.pred(model)
  data_bind = {}
  for arg in pred.arguments:
    if arg.name == 'query':
      data_bind[arg] = reader.streams.query
    if arg.name == 'context':
      data_bind[arg] = reader.streams.context
    if arg.name == 'entities':
      data_bind[arg] = reader.streams.entities
    if arg.name == 'labels':
      data_bind[arg] = reader.streams.label
  #data_bind = {model.context:reader.streams.context, model.query:reader.streams.query, model.entities:reader.streams.entities}
  batch = reader.next_minibatch(100, data_bind)
  var = pred.eval(batch)
  for o in pred.outputs:
    print('-----------------------')
    print(o.name)
    print(np.around(var[o], decimals=3))
  pred = var[pred.outputs[-1]]
  print('-----------------------')
  for i in pred:
    print('')
    print("Prediction: {0}\n\t=>{1}\n".format(i, np.argmax(np.reshape(i, 10))))

def testReasoNetTrain(data, epoch_size, max_epochs=1, vocab_dim=101000, hidden_dim=300, embedding_dim=100, max_rl_iter =5, embedding_path=None, vocab_path=None, eval_path=None, eval_size=None):
  train_data = create_reader(data, vocab_dim, True, rand_size=epoch_size)
  eval_data = create_reader(eval_path, vocab_dim, False, rand_size=eval_size) if eval_path is not None else None
  embedding_init = None
  if embedding_path:
    scale = math.sqrt(6/(vocab_dim+embedding_dim))*2
    init = utils.uniform_initializer(scale, -scale/2)
    embedding_init = vocab.load_embedding(embedding_path, vocab_path, embedding_dim, init)

  model = create_model(vocab_dim, hidden_dim, embedding_init=embedding_init, embedding_dim=embedding_dim, max_rl_iter=max_rl_iter)
  rsn.train(model, train_data, max_epochs=max_epochs, epoch_size=epoch_size, save_model_flag=True, model_name=os.path.basename(data), eval_data=eval_data, eval_size=eval_size)

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
  batch = reader.next_minibatch(2, data_bind)
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

def testSparse(data, vocab_dim=101000, hidden_dim=300, max_rl_iter =5):
  reader = create_reader(data, vocab_dim, False)
  data = reader.next_minibatch(1)
  context_data = data[reader.streams.context]
  context_dynamic_axes = [cntk.Axis.default_batch_axis(), cntk.Axis('Context')]
  context_var= input_variable(shape=(vocab_dim), is_sparse=True, dynamic_axes=context_dynamic_axes, name='context')
  q = ops.times(1, context_var)
  print(q.output.is_sparse)
  su = ops.sequence.reduce_sum(ops.reshape(context_var, vocab_dim))
  o = su.eval({context_var:context_data})
  print(o)

def test_load_embedding(embedding_path, vocab_path, dim):
  init = utils.uniform_initializer()
  emb = vocab.load_embedding(embedding_path, vocab_path, dim, init)
  print(emb.shape)
  print(emb[0])
  print(emb[-1])

def test_next_minibatch(data_path, data_size, eval_path, eval_size, vocab_dim):
  data=create_reader(data_path, vocab_dim, True, rand_size=data_size)
  eval_data=create_reader(eval_path, vocab_dim, False, rand_size=eval_size)
  i=0
  seqs = 0
  while i<data_size:
    mb = data.next_minibatch(12000)
    samples = mb[data.streams.label].num_samples
    seqs += mb[data.streams.label].num_sequences
    i+=samples
  print("Samples:{}, Seqs:{}".format(i, seqs))

  i=0
  seqs = 0
  while i<eval_size:
    mb = eval_data.next_minibatch(12000)
    samples = mb[eval_data.streams.label].num_samples
    seqs += mb[eval_data.streams.label].num_sequences
    i+=samples
  print("Samples:{}, Seqs:{}".format(i, seqs))

#testGRU()
#testASR("test.idx")
#testModel("test.idx")
#testReasoNetLoss("test.idx")
#testReasoNetLoss("test.idx")
#testReasoNetTrain("test.idx", 10, vocab_dim=10, hidden_dim=5, max_rl_iter=3)
#testReasoNetPred("test.idx")

#testReasoNetTrain("data/vocab.101000/test.10.idx", 12273, max_epochs=5, vocab_dim=101100, hidden_dim=384, max_rl_iter=5, embedding_path='data/Glove_Embedding/glove.6B.100d.txt', vocab_path='data/vocab.101000/vocab.101000.idx')
#testReasoNetTrain("data/vocab.101000/test.1000.idx", 1168157, max_epochs=5, vocab_dim=101100, hidden_dim=384, max_rl_iter=5, embedding_path='data/Glove_Embedding/glove.6B.100d.txt', vocab_path='data/vocab.101000/vocab.101000.idx',eval_path="data/vocab.101000/eval.1000.idx", eval_size=1159400)
testReasoNetTrain("data/vocab.101000/train.269000.idx", 314766717, max_epochs=5, vocab_dim=101100, hidden_dim=384, max_rl_iter=5, embedding_path='data/Glove_Embedding/glove.6B.100d.txt', vocab_path='data/vocab.101000/vocab.101000.idx',eval_path="data/vocab.101000/eval.1000.idx", eval_size=1159400)

#testSparse("test.idx", 101000)
#test_load_embedding('data/Glove_Embedding/glove.6B.100d.txt', 'vocab.101000.idx', 100)
#test_next_minibatch("data/vocab.101000/test.1000.idx", 1168157, eval_path="data/vocab.101000/eval.1000.idx", eval_size=1159400, vocab_dim=101100)
