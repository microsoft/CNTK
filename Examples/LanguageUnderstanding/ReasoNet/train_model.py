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

module_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
def train_cnn_model():  
  logger.init("cnn_train")
  data_path = os.path.join(module_path, "Data/cnn/training.ctf")
  eval_path = os.path.join(module_path, "Data/cnn/validation.ctf")
  vocab_path = os.path.join(module_path, "Data/cnn/cnn.vocab")
  vocab_dim = 101585
  entity_dim = 586
  epoch_size=289716292
  eval_size=2993016
  hidden_dim=256
  max_rl_steps=5
  max_epochs=5
  embedding_dim=300
  att_dim = 384
  minibatch_size=50000
  share_rnn = True
  glove_path = os.path.join(module_path, "Data/glove/glove.6B.{0}d.txt".format(embedding_dim))

  train_data = create_reader(data_path, vocab_dim, entity_dim, True)
  eval_data = create_reader(eval_path, vocab_dim, entity_dim, False) \
		if eval_path is not None else None

  scale = math.sqrt(6/(vocab_dim+embedding_dim))*2
  init = uniform_initializer(scale, -scale/2)
  embedding_init = load_embedding(glove_path, vocab_path, embedding_dim, init) if os.path.exists(glove_path) \
    else None

  params = model_params(vocab_dim = vocab_dim, entity_dim = entity_dim, hidden_dim = hidden_dim, 
                        embedding_dim = embedding_dim, attention_dim=att_dim, max_rl_steps = max_rl_steps,
                        embedding_init = embedding_init, dropout_rate = 0.2, share_rnn_param = share_rnn)

  model = create_model(params)
  learner = create_adam_learner(model.parameters)
  (train_loss, train_acc, eval_acc) = train(model, params, learner, train_data, 
                                            max_epochs=max_epochs, epoch_size=epoch_size, 
                                            save_model_flag=True, model_name=os.path.basename(data_path),
                                            eval_data=eval_data, eval_size=eval_size, check_point_freq=0.1,
                                            minibatch_size = minibatch_size)

train_cnn_model()
