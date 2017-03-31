import sys
import os
py_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
py_path = os.path.join(py_path, "..", "..", "..", "..", "Examples/LanguageUnderstanding/")
sys.path.insert(0, py_path)
print(py_path)
module_path = os.path.join(py_path, 'ReasoNet')

import cntk.device as device
import numpy as np
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.ops import input
from cntk.io import MinibatchSource
from cntk import Trainer, Axis, device, combine
from cntk.layers import Recurrence, Convolution
import cntk.ops as ops
import cntk
import math
import pytest

def test_reasonet(device_id, is_1bit_sgd):
  print("Device Id: {0}".format(device_id))
  if device_id < 0:
    pytest.skip('test only runs on GPU')
    
  if is_1bit_sgd != 0:
    pytest.skip('test doesn\'t support 1bit sgd')

  import ReasoNet.reasonet as rsn
  device.set_default_device(cntk_device(device_id))
  data_path = os.path.join(module_path, "Data/fast_test.txt")
  eval_path = os.path.join(module_path, "Data/fast_test.txt")
  vocab_dim = 101100
  entity_dim = 101
  epoch_size=1159400
  eval_size=1159400
  hidden_dim=256
  max_rl_iter=5
  max_epochs=1
  embedding_dim=300
  att_dim = 384
  params = rsn.model_params(vocab_dim = vocab_dim, entity_dim = entity_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim, embedding_init = None, attention_dim = att_dim, dropout_rate = 0.2)

  train_data = rsn.create_reader(data_path, vocab_dim, entity_dim, True)
  eval_data = rsn.create_reader(eval_path, vocab_dim, entity_dim, False) if eval_path is not None else None
  embedding_init = None

  model = rsn.create_model(params)
  learner = rsn.create_adam_learner(model.parameters)
  (train_loss, train_acc, eval_acc) = rsn.train(model, params, learner, train_data, max_epochs=max_epochs, epoch_size=epoch_size, save_model_flag=False, model_name=os.path.basename(data_path), eval_data=eval_data, eval_size=eval_size, check_point_freq=1, minibatch_size = 5000)
  assert abs(train_loss - 0.08067)<1e-2
  assert abs(train_acc - 0.21635)<1e-2
  if sys.version_info >= (3,):
    assert abs(eval_acc - 0.304)<1e-2
  else:
    assert abs(eval_acc - 0.312)<1e-2
