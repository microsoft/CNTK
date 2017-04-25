import os
import numpy as np
from datetime import datetime
import math
from cntk import Trainer, Axis, device, combine
from cntk.layers.blocks import Stabilizer, _initializer_for,  _INFERRED, Parameter, Placeholder
from cntk.layers import Recurrence, Convolution, Dense
from cntk.ops import input, sequence, reduce_sum, \
    parameter, times, element_times, plus, placeholder, reshape, constant, sigmoid, convolution, tanh, times_transpose, greater, element_divide, element_select, exp
from cntk.losses import cosine_distance
from cntk.internal import _as_tuple, sanitize_input
from cntk.initializer import uniform, glorot_uniform

try:
  from wordvocab import *
except Exception:
  from .wordvocab import *

class logger:
  __name=''
  __logfile=''

  @staticmethod
  def init(name=''):
    if not os.path.exists("model"):
      os.mkdir("model")
    if not os.path.exists("log"):
      os.mkdir("log")
    if name=='' or name is None:
      logger.__name = 'train'
    else:
      logger.__name = name
    logger.__logfile = 'log/{}_{}.log'.format(logger.__name, datetime.now().strftime("%m-%d_%H.%M.%S"))
    if os.path.exists(logger.__logfile):
      os.remove(logger.__logfile)
    print('Log with log file: {0}'.format(logger.__logfile))

  @staticmethod
  def log(message, toconsole=True):
    if logger.__logfile == '' or logger.__logfile is None:
      logger.init()
    if toconsole:
      print(message)
    with open(logger.__logfile, 'a') as logf:
      logf.write("{}| {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))

class uniform_initializer:
  def __init__(self, scale=1, bias=0, seed=0):
    self.seed = seed
    self.scale = scale
    self.bias = bias
    np.random.seed(self.seed)

  def reset(self):
    np.random.seed(self.seed)

  def next(self, size=None):
    return np.random.uniform(0, 1, size)*self.scale + self.bias

def create_random_matrix(rows, columns):
  scale = math.sqrt(6/(rows+columns))*2
  rand = uniform_initializer(scale, -scale/2)
  embedding = [None]*rows
  for i in range(rows):
    embedding[i] = np.array(rand.next(columns), dtype=np.float32)
  return np.ndarray((rows, columns), dtype=np.float32, buffer=np.array(embedding))

def load_embedding(embedding_path, vocab_path, dim, init=None):
  entity_vocab, word_vocab = Vocabulary.load_vocab(vocab_path)
  vocab_dim = len(entity_vocab) + len(word_vocab) + 1
  entity_size = len(entity_vocab)
  item_embedding = [None]*vocab_dim
  with open(embedding_path, 'r') as embedding:
    for line in embedding.readlines():
      line = line.strip('\n')
      item = line.split(' ')
      if item[0] in word_vocab:
        item_embedding[word_vocab[item[0]].id + entity_size + 1] = np.array(item[1:], dtype="|S").astype(np.float32)
  if init != None:
    init.reset()

  for i in range(vocab_dim):
    if item_embedding[i] is None:
      if init:
        item_embedding[i] = np.array(init.next(dim), dtype=np.float32)
      else:
        item_embedding[i] = np.array([0]*dim, dtype=np.float32)
  return np.ndarray((vocab_dim, dim), dtype=np.float32, buffer=np.array(item_embedding))

def project_cosine(project_dim, init = glorot_uniform(), name=''):
  """
  Compute the project cosine similarity of two input sequences,
  where each of the input will be projected to a new dimention space (project_dim) via Wi/Wm
  """
  Wi = Parameter(_INFERRED + (project_dim,), init = init, name='Wi')
  Wm = Parameter(_INFERRED + (project_dim,), init = init, name='Wm')

  status = placeholder(name='status')
  memory = placeholder(name='memory')

  projected_status = times(status, Wi, name = 'projected_status')
  projected_memory = times(memory, Wm, name = 'projected_memory')
  status_br = sequence.broadcast_as(projected_status, projected_memory, name='status_broadcast')
  sim = cosine_distance(status_br, projected_memory, name= name)
  return sim

def attention_score(att_dim, init = glorot_uniform(), name=''):
  """
  Compute the attention score,
  where each of the input will be projected to a new dimention space (att_dim) via Wi/Wm
  """
  sim = project_cosine(att_dim, init, name= name+ '_sim')
  return sequence.softmax(10*sim, name = name)

def termination_gate(init = glorot_uniform(), name=''):
  return Dense(1, activation = sigmoid, init=init, name= name)

