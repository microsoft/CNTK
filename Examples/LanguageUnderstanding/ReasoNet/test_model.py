import sys
import os
import cntk.device as device
import numpy as np
import math
try:
  from .utils import *
  from .reasonet import *
except Exception:
  from utils import *
  from reasonet import *
from cntk import load_model

module_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
def test_cnn_model(model_path):  
  logger.init("cnn_test")
  test_path = os.path.join(module_path, "Data/cnn/test.ctf")
  test_size=2291183
  vocab_path = os.path.join(module_path, "Data/cnn/cnn.vocab")
  vocab_dim = 101585
  entity_dim = 586
  hidden_dim=256
  max_rl_steps=5
  embedding_dim=300
  att_dim = 384
  minibatch_size=50000
  share_rnn = True
  glove_path = os.path.join(module_path, "Data/glove/glove.6B.{0}d.txt".format(embedding_dim))

  test_data = create_reader(test_path, vocab_dim, entity_dim, False) 
  embedding_init = None 

  params = model_params(vocab_dim = vocab_dim, entity_dim = entity_dim, hidden_dim = hidden_dim, 
                        embedding_dim = embedding_dim, attention_dim=att_dim, max_rl_steps = max_rl_steps,
                        embedding_init = embedding_init, dropout_rate = 0.2, share_rnn_param = share_rnn)

  model = load_model(model_path)
  loss_func = loss(model, params)
  bind = bind_data(loss_func, test_data)
  context_stream = get_context_bind_stream(bind)
  loss_sum = 0
  acc_sum = 0
  samples_sum = 0
  i = 0
  start = os.times()
  while i<test_size:
    mbs = min(test_size - i, minibatch_size)
    mb = test_data.next_minibatch(mbs, bind)
    outs = loss_func.eval(mb)
    loss_value = np.sum(outs[loss_func.outputs[0]])
    acc = np.sum(outs[loss_func.outputs[-1]])
    i += mb[context_stream].num_samples
    samples = mb[context_stream].num_sequences
    samples_sum += samples
    acc_sum += acc
    loss_sum += loss_value
    sys.stdout.write('.')
    sys.stdout.flush()
  end = os.times()
  total = end.elapsed - start.elapsed
  print("")
  print("Evaluation acc: {0}, loss: {1}, samples: {2} in {3} seconds".format(acc_sum/samples_sum, loss_sum/samples_sum, samples_sum, total))

test_cnn_model("model/model_training.ctf_001.dnn")
