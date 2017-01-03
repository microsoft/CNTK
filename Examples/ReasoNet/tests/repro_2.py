import sys
import os
py_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../..")
sys.path.insert(0, py_path)
module_path = os.path.join(py_path, 'ReasoNet')

import cntk.device as device
import numpy as np
import ReasoNet.model_repro2 as rsn
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

# TODO: this script is to repro the issue of commit
#commit 4f6bff84c9b309c53e06ab6ba93aed3a8d030534
#Author: Amit Agarwal <amitaga@microsoft.com>
#Date:   Thu Jan 5 03:18:55 2017 -0800
#
#    CNTK v2 library: Fix a bug in where node dynamic axis determination where multiple where applications on same condition were yielding different dynamic axes
#

def testReasoNetTrain(data, epoch_size, max_epochs=1, vocab_dim=101000, hidden_dim=300, embedding_dim=100, max_rl_iter =5, embedding_path=None, vocab_path=None, eval_path=None, eval_size=None):
  train_data = rsn.create_reader(data, vocab_dim, True, rand_size=epoch_size)
  eval_data = rsn.create_reader(eval_path, vocab_dim, False, rand_size=eval_size) if eval_path is not None else None
  embedding_init = None
  if embedding_path:
    scale = math.sqrt(6/(vocab_dim+embedding_dim))*2
    init = utils.uniform_initializer(scale, -scale/2)
    embedding_init = vocab.load_embedding(embedding_path, vocab_path, embedding_dim, init)

  model = rsn.create_model(vocab_dim, hidden_dim, embedding_init=embedding_init, embedding_dim=embedding_dim, max_rl_iter=max_rl_iter)
  rsn.train(model, train_data, max_epochs=max_epochs, epoch_size=epoch_size, save_model_flag=True, model_name=os.path.basename(data), eval_data=eval_data, eval_size=eval_size)

testReasoNetTrain(os.path.join(module_path, "data/vocab.101000/test.1000.idx"), 1168157, max_epochs=5, vocab_dim=101100,
    hidden_dim=384, max_rl_iter=5,
    vocab_path=os.path.join(module_path, 'data/vocab.101000/vocab.101000.idx'),
    eval_path=os.path.join(module_path, "data/vocab.101000/eval.1000.idx"),
    eval_size=1159400)
