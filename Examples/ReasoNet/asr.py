"""
ReasoNet model in CNTK
@author penhe@microsoft.com
"""
import sys
import os
import numpy as np
from cntk import Trainer, Axis, device, combine
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, reduce_sum, \
    parameter, times, element_times, past_value, plus, placeholder_variable, splice, reshape, constant, sigmoid, convolution, tanh, times_transpose, greater, cosine_distance, element_divide, element_select
import cntk.ops as ops
from cntk.blocks import LSTM, Stabilizer, _get_current_default_options, _is_given, _initializer_for, _resolve_activation, _INFERRED, Parameter, Placeholder, Block, init_default_or_glorot_uniform
from cntk.layers import Recurrence, Convolution
from cntk.initializer import uniform, glorot_uniform
from cntk.utils import get_train_eval_criterion, get_train_loss, Record, _as_tuple, sanitize_input, value_to_seq
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
import cntk.utils as utils
import cntk.learner as learner
from datetime import datetime
import math
import cntk.io as io
import cntk.cntk_py as cntk_py
import ReasoNet.utils as rs_utils

########################
# variables and stuff  #
########################

# model dimensions
#vocab_dim = 40000
#embed_dim = 200
#hidden_dim = 256
#src_max_len = 120
#ctx_max_len = 1989

# stabilizer
#cntk_py.disable_forward_values_sharing() 
#cntk_py.set_computation_network_trace_level(1)
stabilize = Stabilizer()
if not os.path.exists("model"):
  os.mkdir("model")
if not os.path.exists("log"):
  os.mkdir("log")
log_name='train'
logfile=None
def log(message, toconsole=True):
  global logfile
  global log_name
  if logfile is None:
    logfile = 'log/{}_{}.log'.format(log_name, datetime.now().strftime("%m-%d_%H.%M.%S"))
    print('Log with log file: {0}'.format(logfile))
    if os.path.exists(logfile):
      os.remove(logfile)

  if toconsole:
     print(message)
  with open(logfile, 'a') as logf:
    logf.write("{}| {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))

def create_reader(path, vocab_dim, entity_dim, randomize, rand_size= io.DEFAULT_RANDOMIZATION_WINDOW, size=INFINITELY_REPEAT):
  return MinibatchSource(CTFDeserializer(path, StreamDefs(
    context  = StreamDef(field='C', shape=vocab_dim, is_sparse=True),
    query    = StreamDef(field='Q', shape=vocab_dim, is_sparse=True),
    entities  = StreamDef(field='E', shape=1, is_sparse=False),
    label   = StreamDef(field='L', shape=1, is_sparse=False),
    entity_ids   = StreamDef(field='EID', shape=entity_dim, is_sparse=True)
    )), randomize=randomize)
  #, randomization_window = rand_size)

def bidirectionalLSTM(hidden_dim, x, splice_outputs=True, init = init_default_or_glorot_uniform):
  bwd = Recurrence(LSTM(hidden_dim), go_backwards=True ) (stabilize(x))
  if splice_outputs:
    # splice the outputs together
    hc = splice((fwd, bwd))
    return hc
  else:
    # return both (in cases where we want the 'final' hidden status)
    return (fwd, bwd)

def bidirectional_gru(hidden_dim, x, splice_outputs=True, name='', init = init_default_or_glorot_uniform):
  W = Parameter(_INFERRED +  _as_tuple(hidden_dim), init=init, name='gru_params')
  dualstatus = ops.optimized_rnnstack(x, W, hidden_dim, 1, True, recurrent_op='gru', name=name)
  if splice_outputs:
    # splice the outputs together
    return dualstatus
  else:
    # return both (in cases where we want the 'final' hidden status)
    return (ops.slice(sequence.last(dualstatus), -1, 0, hidden_dim, name='fwd'), ops.slice(sequence.first(dualstatus), -1, hidden_dim, hidden_dim*2, name='bwd'), dualstatus) 

def seq_max(x):
  m = placeholder_variable(shape=(1,), dynamic_axes = x.dynamic_axes, name='max')
  o = element_select(greater(x, ops.future_value(m)), x, ops.future_value(m))
  rlt = o.replace_placeholders({m:sanitize_input(o)})
  max_v = sequence.is_first(x)
  pv = placeholder_variable(shape=(1,), dynamic_axes = x.dynamic_axes, name='max_seq')
  max_seq = element_select(sequence.is_first(x), utils.sanitize_input(rlt), ops.past_value(pv))
  max_br = max_seq.replace_placeholders({pv:utils.sanitize_input(max_seq)})
  return utils.sanitize_input(max_br)

def attention_module(context_memory, query_memory, entity_memory, init_status, hidden_dim, init = init_default_or_glorot_uniform):
  e_mem = ops.placeholder_variable(shape=entity_memory.shape, dynamic_axes=entity_memory.dynamic_axes, name='e_mem')
  c_mem = ops.placeholder_variable(shape=context_memory.shape, dynamic_axes=context_memory.dynamic_axes, name='c_mem')
  i_status = ops.placeholder_variable(shape=(hidden_dim), name='init_status')
  att = ops.times_transpose(c_mem, sequence.broadcast_as(i_status, c_mem))
  att_exp = ops.exp(10*(att-seq_max(att)))
  attention = att_exp/sequence.broadcast_as(sequence.reduce_sum(att_exp), att_exp)
  block_func = ops.as_block(attention, [(c_mem, ops.sanitize_input(context_memory)), (i_status, ops.sanitize_input(init_status))], 'Attention_module', 'Attention_module')
  return block_func

def create_constant_embedding(vocab_dim, embedding_dim):
  scale = math.sqrt(6/(vocab_dim+embedding_dim))*2
  rand = rs_utils.uniform_initializer(scale, -scale/2)
  embedding = [None]*vocab_dim
  for i in range(vocab_dim):
    embedding[i] = rand.next(embedding_dim)
  return np.ndarray((vocab_dim, embedding_dim), dtype=np.float32, buffer=np.array(embedding))

#
# TODO: CNTK current will convert sparse variable to dense after reshape function
def create_model(vocab_dim, entity_dim, hidden_dim,  embedding_init=None,  embedding_dim=100, dropout_rate=None, init = init_default_or_glorot_uniform, model_name='asr'):
  global log_name
  if model_name is not None:
    log_name = model_name+'_log'
  log("Create model: dropout_rate: {0}, init:{1}, embedding_init: {2}".format(dropout_rate, init, embedding_init))
  # Query and Doc/Context/Paragraph inputs to the model
  batch_axis = Axis.default_batch_axis()
  query_seq_axis = Axis('sourceAxis')
  context_seq_axis = Axis('contextAxis')
  query_dynamic_axes = [batch_axis, query_seq_axis]
  query_sequence = input_variable(shape=(vocab_dim), is_sparse=True, dynamic_axes=query_dynamic_axes, name='query')
  context_dynamic_axes = [batch_axis, context_seq_axis]
  context_sequence = input_variable(shape=(vocab_dim), is_sparse=True, dynamic_axes=context_dynamic_axes, name='context')
  candidate_dynamic_axes = [batch_axis, context_seq_axis]
  entity_ids_mask = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context_dynamic_axes, name='entities')
  # embedding
  if embedding_init is None:
    embedding = Parameter(shape=(vocab_dim, embedding_dim), init=uniform(0.2))
    embedding_matrix = constant(create_constant_embedding(vocab_dim, embedding_dim), shape=(vocab_dim, embedding_dim))
  else:
    embedding = parameter(shape=(vocab_dim, embedding_dim), init=None)
    embedding.value = embedding_init
    embedding_matrix = constant(embedding_init, shape=(vocab_dim, embedding_dim))

  # TODO: Add dropout to embedding
  if dropout_rate is not None:
    query_embedding  = ops.dropout(times(query_sequence , embedding), dropout_rate, name='query_embedding')
    context_embedding = ops.dropout(times(context_sequence, embedding), dropout_rate, name='context_embedding')
  else:
    query_embedding  = times(query_sequence , embedding, name='query_embedding')
    context_embedding = times(context_sequence, embedding, name='context_embedding')
  
  entity_embedding = ops.times(context_sequence, embedding_matrix, name='constant_entity_embedding')
  mask_embedding = ops.element_select(entity_ids_mask, entity_embedding, context_embedding)
  #mask_embedding = context_embedding
  # get source and context representations
  context_memory = bidirectional_gru(hidden_dim, mask_embedding, name='Context_Mem', init=init)            # shape=(hidden_dim*2, *), *=context_seq_axis
  #context_memory = bidirectional_gru(hidden_dim, context_embedding, name='Context_Mem', init=init)            # shape=(hidden_dim*2, *), *=context_seq_axis
  entity_condition = greater(entity_ids_mask, 0, name='condidion')
  entities_all = sequence.gather(entity_condition, entity_condition, name='entities_all')
  entity_memory = sequence.gather(context_memory, entity_condition, name='Candidate_Mem')
  entity_ids = input_variable(shape=(entity_dim), is_sparse=True, dynamic_axes=entity_memory.dynamic_axes, name='entity_ids')
  #entity_memory = sequence.scatter(sequence.gather(context_memory, entity_condition, name='Candidate_Mem'), entities_all)
  qfwd, qbwd, query_memory  = bidirectional_gru(hidden_dim, query_embedding, splice_outputs=False, name='Query_Mem', init=init) # shape=(hidden_dim*2, *), *=query_seq_axis
  init_status = splice((qfwd, qbwd), name='Init_Status') # get last fwd status and first bwd status
  result = attention_module(context_memory, query_memory, entity_memory, init_status, hidden_dim*2)
  ans_prob = sequence.gather(ops.sanitize_input(result), entity_condition, name='Final_Ans')
  return Block(ans_prob, 'ReasoNet', members = Record(vocab_dim=vocab_dim, hidden_dim=hidden_dim, context=context_sequence,
    query=query_sequence, entities=entity_ids_mask, entity_condition=entity_condition,
    entities_all=entities_all, entity_ids=entity_ids, entity_dim=entity_dim))

def pred(model):
  answers = model.outputs[-1]
  entity_ids = model.entity_ids
  entity_dim = model.entity_dim
  entity_id_matrix = ops.reshape(entity_ids, entity_dim)
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix), name='prediction_prob')
  return combine([expand_pred])

def pred_eval(model):
  model_args = {arg.name:arg for arg in model.arguments}
  answers = model.outputs[-1]
  entity_ids = model.entity_ids
  entity_dim = model.entity_dim
  entity_condition = model.entity_condition
  entities_all = model.entities_all
  context = model_args['context']
  labels_raw = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context.dynamic_axes, name='labels')
  answers = model.outputs[-1]
  labels = sequence.gather(labels_raw, entity_condition, name='EntityLabels')
  entity_id_matrix = ops.reshape(entity_ids, entity_dim)
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix), name='prediction_prob')
  expand_label = ops.greater_equal(sequence.reduce_sum(element_times(labels, entity_id_matrix)), 1)
  return combine([expand_pred, expand_label])

def accuracy_func(pred, label, name='accuracy'):
  pred_max = ops.hardmax(pred, name='pred_max')
  norm_label = ops.equal(label, [1], name='norm_label')
  acc = ops.times_transpose(pred_max, norm_label, name='accuracy')
  return acc

def softmax_cross_entropy(pred, label, mask, gama=10, name=''):
  pred_exp = ops.exp(gama*pred)
  #pred_exp = ops.exp(gama*(pred-ops.reduce_max(pred)))
  sum_exp = ops.reduce_sum(element_times(pred_exp, mask))
  soft_max = ops.negate(ops.element_times(label, ops.log(pred_exp/sum_exp)))
  return ops.reduce_sum(soft_max, name=name)

def loss(model):
  model_args = {arg.name:arg for arg in model.arguments}
  context = model_args['context']
  entities = model_args['entities']
  wordvocab_dim = model.vocab_dim
  labels_raw = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context.dynamic_axes, name='labels')
  entity_condition = model.entity_condition
  entities_all = model.entities_all
  answers = model.outputs[-1]
  labels = sequence.gather(labels_raw, entity_condition, name='EntityLabels')
  entity_ids = model.entity_ids
  entity_dim = model.entity_dim
  entities_all = model.entities_all
  entity_id_matrix = ops.reshape(entity_ids, entity_dim)
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix))
  expand_label = ops.greater_equal(sequence.reduce_sum(element_times(labels, entity_id_matrix)), 1)
  expand_candidate_mask = ops.greater_equal(sequence.reduce_sum(entity_id_matrix), 1)
  cross_entroy = softmax_cross_entropy(expand_pred, expand_label, expand_candidate_mask, gama=10, name='CrossEntropy')
  accuracy = accuracy_func(expand_pred, expand_label, name='accuracy')
  apply_loss = combine([cross_entroy, answers, labels, accuracy])
  return Block(apply_loss, 'AvgSoftMaxCrossEntropy', Record(labels=labels_raw))

def loss2(model):
  model_args = {arg.name:arg for arg in model.arguments}
  context = model_args['context']
  entities = model_args['entities']
  wordvocab_dim = model.vocab_dim
  labels_raw = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context.dynamic_axes, name='labels')
  entity_condition = model.entity_condition
  entities_all = model.entities_all
  answers = model.outputs[-1]
  labels = sequence.gather(labels_raw, entity_condition, name='EntityLabels')
  entity_ids = model.entity_ids
  entity_dim = model.entity_dim
  entity_id_matrix = ops.reshape(entity_ids, entity_dim)
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix)) + 0.0000001
  expand_label = ops.greater_equal(sequence.reduce_sum(element_times(labels, entity_id_matrix)), 1)
  cross_entropy = ops.negate(ops.reduce_sum(ops.element_times(expand_label, ops.log(expand_pred))))
  #cross_entroy = softmax_cross_entropy(expand_pred, expand_label, name='CrossEntropy')
  accuracy = accuracy_func(expand_pred, expand_label, name='accuracy')
  apply_loss = combine([cross_entropy, answers, labels, accuracy])
  return Block(apply_loss, 'AvgSoftMaxCrossEntropy', Record(labels=labels_raw))

def bind_data(func, data):
  bind = {}
  for arg in func.arguments:
    if arg.name == 'query':
      bind[arg] = data.streams.query
    if arg.name == 'context':
      bind[arg] = data.streams.context
    if arg.name == 'entities':
      bind[arg] = data.streams.entities
    if arg.name == 'labels':
      bind[arg] = data.streams.label
    if arg.name == 'entity_ids':
      bind[arg] = data.streams.entity_ids
  return bind

def evaluation(trainer, data, bind, minibatch_size, epoch_size):
  if epoch_size is None:
    epoch_size = 1
  for key in bind.keys():
    if key.name == 'labels':
      label_arg = key
      break
  eval_acc = 0
  eval_s = 0
  k = 0
  print("Start evaluation with {0} samples ...".format(epoch_size))
  while k < epoch_size:
    mb = data.next_minibatch(minibatch_size, input_map=bind)
    k += mb[label_arg].num_samples
    sm = mb[label_arg].num_sequences
    avg_acc = trainer.test_minibatch(mb)
    eval_acc += sm*avg_acc
    eval_s += sm 
    sys.stdout.write('.')
    sys.stdout.flush()
  eval_acc /= eval_s
  print("")
  log("Evaluation Acc: {0}, samples: {1}".format(eval_acc, eval_s))

def train(model, train_data, max_epochs=1, save_model_flag=False, epoch_size=270000, model_name='rsn', eval_data=None, eval_size=None, check_point_freq=0.1):
  # Criterion nodes
  global log_name
  if model_name is not None:
    log_name = model_name+'_log'
  #criterion_loss = contractive_reward(model)
  criterion_loss = loss(model)
  loss_func = criterion_loss.outputs[0]
  eval_func = criterion_loss.outputs[-1]

  def create_sgd_learner(ep_size, mb_size):
    schedule_unit=int(ep_size*0.1/mb_size)
    learning_rate = [(schedule_unit*1, 0.01), (schedule_unit*3, 0.005), (schedule_unit*5,0.001), (1, 0.0005)]
    lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.minibatch, 1)
    clipping_threshold_per_sample = 0.01
    #clipping_threshold_per_sample = 10.0
    gradient_clipping_with_truncation = True
    lr = learner.sgd(model.parameters, lr_schedule,
              gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
              gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    learner_desc = 'Alg: sgd, learning rage: {0}, gradient clip: {1}'.format(learning_rate, clipping_threshold_per_sample)
    log("Create learner. {0}".format(learner_desc))
    return lr

  def create_momentum_learner(ep_size, mb_size):
    schedule_unit=int(ep_size*0.1/mb_size)
    #learning_rate = [(schedule_unit*1, 0.05), (schedule_unit*1, 0.01)]
    #learning_rate = [(schedule_unit*1, 0.05), (schedule_unit*1, 0.01), (1,0.001)]
    learning_rate = 0.001
    lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.minibatch, 1)
    momentum = learner.momentum_schedule(0.99)
    clipping_threshold_per_sample = 1
    #clipping_threshold_per_sample = 10.0
    gradient_clipping_with_truncation = True
    lr = learner.momentum_sgd(model.parameters, lr_schedule, momentum,
              gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
              gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    learner_desc = 'Alg: monentum, learning rage: {0}, momentum: {1}, gradient clip: {2}'.format(learning_rate, momentum[0], clipping_threshold_per_sample)
    log("Create learner. {0}".format(learner_desc))
    return lr

  def create_adam_learner(ep_size, mb_size):
    #learning_rate = 0.0005
    #learning_rate = 0.00005
    learning_rate = 0.0001
    lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.sample)
    momentum = learner.momentum_schedule(0.90)
    clipping_threshold_per_sample = 10
    #clipping_threshold_per_sample = 10/32
    #gradient_clipping_with_truncation = False
    gradient_clipping_with_truncation = True
    momentum_var = learner.momentum_schedule(0.999)
    lr = learner.adam_sgd(model.parameters, lr_schedule, momentum, True, momentum_var,
            gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
            gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    learner_desc = 'Alg: Adam, learning rage: {0}, momentum: {1}, gradient clip: {2}'.format(learning_rate, momentum[0], clipping_threshold_per_sample)
    log("Create learner. {0}".format(learner_desc))
    return lr

  def create_nesterov_learner(ep_size, mb_size):
    schedule_unit=int(ep_size*0.1/mb_size)
    learning_rate = [(schedule_unit*1, 0.05), (schedule_unit*5, 0.01), (1,0.001)]
    lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.minibatch, 1)
    momentum = learner.momentum_schedule(0.99)
    clipping_threshold_per_sample = 10
    #clipping_threshold_per_sample = 10.0
    gradient_clipping_with_truncation = True
    learn = learner.nesterov(model.parameters, lr_schedule,momentum,
              gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
              gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    learner_desc = 'Alg: nesterov, learning rage: {0}, momentum: {1}, gradient clip: {2}'.format(learning_rate, momentum[0], clipping_threshold_per_sample)
    log("Create learner. {0}".format(learner_desc))
    return learn
  # Instantiate the trainer object to drive the model training
  #learn = learner.adagrad(model.parameters, lr_schedule, gradient_clipping_threshold_per_sample = clipping_threshold_per_sample, gradient_clipping_with_truncation = gradient_clipping_with_truncation)
  minibatch_size = 24000
  #minibatch_size = 30000
  #minibatch_size = 40000
  #minibatch_size = 20000
  lr = create_adam_learner(epoch_size, minibatch_size)
  #lr = create_momentum_learner(epoch_size, minibatch_size)
  #lr = create_nesterov_learner(epoch_size, minibatch_size)
  #lr = create_sgd_learner(epoch_size, minibatch_size)
  trainer = Trainer(model.outputs[-1], loss_func, eval_func, lr)
  # Get minibatches of sequences to train with and perform model training
  # bind inputs to data from readers
  train_bind = bind_data(criterion_loss, train_data)
  for k in train_bind.keys():
    if k.name == 'labels':
      label_key = k
      break
  eval_bind = bind_data(criterion_loss, eval_data)

  i = 0
  minibatch_count = 0
  training_progress_output_freq = 100
  check_point_interval = int(epoch_size*check_point_freq)
  check_point_id = 0
  total_samples = 0
  for epoch in range(max_epochs):
    epoch_loss = 0
    epoch_acc = 0
    epoch_samples = 0
    i = 0
    win_loss = 0
    win_acc = 0
    win_samples = 0
    chk_loss = 0
    chk_acc = 0
    chk_samples = 0
    while i < epoch_size:
      # get next minibatch of training data
      # TODO: Shuffle entities? @yelong
      mb_train = train_data.next_minibatch(minibatch_size, input_map=train_bind)
      i += mb_train[label_key].num_samples
      total_samples += mb_train[label_key].num_samples
      #test_acc = trainer.test_minibatch(mb_train)
      trainer.train_minibatch(mb_train)
      minibatch_count += 1
      sys.stdout.write('.')
      sys.stdout.flush()
      # collect epoch-wide stats
      samples = trainer.previous_minibatch_sample_count
      ls = trainer.previous_minibatch_loss_average * samples
      acc = trainer.previous_minibatch_evaluation_average * samples
      #print("Test acc:{0}, train acc: {1}".format(test_acc*samples, acc))
      epoch_loss += ls
      epoch_acc += acc
      win_loss += ls
      win_acc += acc
      chk_loss += ls
      chk_acc += acc
      epoch_samples += samples
      win_samples += samples
      chk_samples += samples
      if int(epoch_samples/training_progress_output_freq) != int((epoch_samples-samples)/training_progress_output_freq):
        print('')
        log("Lastest sample count = {}, Train Loss: {}, Evalualtion ACC: {}".format(win_samples, win_loss/win_samples, 
          win_acc/win_samples))
        log("Total sample count = {}, Train Loss: {}, Evalualtion ACC: {}".format(chk_samples, chk_loss/chk_samples, 
          chk_acc/chk_samples))
        win_samples = 0
        win_loss = 0
        win_acc = 0
      new_chk_id = int(total_samples/check_point_interval)
      if new_chk_id != check_point_id:
        check_point_id = new_chk_id
        print('')
        log("--- CHECKPOINT %d: samples=%d, loss = %.2f, acc = %.2f%% ---" % (check_point_id, chk_samples, chk_loss/chk_samples, 100.0*(chk_acc/chk_samples)))
        if eval_data:
          evaluation(trainer, eval_data, eval_bind, minibatch_size, eval_size)
        if save_model_flag:
          # save the model every epoch
          model_filename = os.path.join('model', "model_%s_%03d.dnn" % (model_name, check_point_id))
          model.save_model(model_filename)
          log("Saved model to '%s'" % model_filename)
        chk_samples = 0
        chk_loss = 0
        chk_acc = 0

    print('')
    log("--- EPOCH %d: samples=%d, loss = %.2f, acc = %.2f%% ---" % (epoch, epoch_samples, epoch_loss/epoch_samples, 100.0*(epoch_acc/epoch_samples)))
  if eval_data:
    evaluation(trainer, eval_data, eval_bind, minibatch_size, eval_size)
  if save_model_flag:
    # save the model every epoch
    model_filename = os.path.join('model', "model_%s_final.dnn" % (model_name))
    model.save_model(model_filename)
    log("Saved model to '%s'" % model_filename)
