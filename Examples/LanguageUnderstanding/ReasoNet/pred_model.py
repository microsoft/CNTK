import sys
import os
import cntk.device as device
import numpy as np
import math
from io import open
try:
  from .utils import *
  from .reasonet import *
  from .prepare_cnn_data import prepare_data
except Exception:
  from utils import *
  from reasonet import *
  from prepare_cnn_data import prepare_data
from cntk import load_model
import time

module_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
def pred_cnn_model(model_path, output):  
  prepare_data()
  logger.init("cnn_test")
  if os.path.exists(output):
    os.remove(output)
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

  entity_table, word_table = Vocabulary.load_vocab(vocab_path)
  model = load_model(model_path)
  predict_func = predict(model, params)
  bind = bind_data(predict_func, test_data)
  context_stream = get_context_bind_stream(bind)
  samples_sum = 0
  i = 0
  instance_id = 0
  start = time.time()
  while i<test_size:
    mbs = min(test_size - i, minibatch_size)
    mb = test_data.next_minibatch(mbs, bind)
    pred = predict_func.eval(mb)
    ans = np.nonzero(pred)
    with open(output, mode='a', encoding='utf-8') as outf:
      for id in ans[1]:
        outf.write(u"{0}\t{1}\n".format(instance_id, entity_table.lookup_by_id(id)))
        instance_id += 1
    i += mb[context_stream].num_samples
    samples = mb[context_stream].num_sequences
    samples_sum += samples
    sys.stdout.write('.')
    sys.stdout.flush()
  end = time.time()
  total = end - start
  print("")
  print("Evaluated samples: {0} in {1} seconds".format(samples_sum, total))

pred_cnn_model("model/model_training.ctf_001.dnn", "pred.txt")
