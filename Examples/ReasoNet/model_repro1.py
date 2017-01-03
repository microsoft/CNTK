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
stabilize = Stabilizer()

logfile = 'log/train_{}.log'.format(datetime.now().strftime("%m-%d_%H.%M.%S"))
if not os.path.exists("log"):
  os.mkdir("log")
if not os.path.exists("model"):
  os.mkdir("model")
if os.path.exists(logfile):
  os.remove(logfile)

def log(message, toconsole=True):
  if toconsole:
     print(message)
  with open(logfile, 'a') as logf:
    logf.write("{}| {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))

def create_reader(path, vocab_dim, randomize, rand_size= io.DEFAULT_RANDOMIZATION_WINDOW, size=INFINITELY_REPEAT):
  return MinibatchSource(CTFDeserializer(path, StreamDefs(
    context  = StreamDef(field='C', shape=vocab_dim, is_sparse=True),
    query    = StreamDef(field='Q', shape=vocab_dim, is_sparse=True),
    entities  = StreamDef(field='E', shape=1, is_sparse=False),
    label   = StreamDef(field='L', shape=1, is_sparse=False)
    )), randomize=randomize, randomization_window = size)

def text_convolution(win_size, in_dim, out_dim):
  #activation = _resolve_activation(activation)
  output_channels_shape = _as_tuple(out_dim)
  output_rank = len(output_channels_shape)
  filter_shape = (win_size, in_dim)
  filter_rank = len(filter_shape)
  kernel_shape = _INFERRED + filter_shape # kernel := filter plus reductionDims

  # parameters bound to this Function
  init_kernel = glorot_uniform(filter_rank=filter_rank, output_rank=1)
  #init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-1))
  # BUGBUG: It is very confusing that output_rank is negative, esp. since that means count from the start. Solution: add a flag
  W = Parameter(output_channels_shape + kernel_shape,             init=init_kernel, name='W')                   # (K, C, H, W) aka [ W x H x C x K ]
  #w = np.reshape(np.array([[[[2, -1, 0, -1, 2],[1,1,2,-1,-1],[1,2,0,2,1]]]], dtype = np.float32), (1, 1, 3, 5))
  #W = constant(value=w)
  #b = Parameter(output_channels_shape + (1,) * len(filter_shape), init=init_bias,   name='b') if bias else None # (K,    1, 1) aka [ 1 x 1 x     K ]

  # expression
  x = Placeholder(name='convolution_arg')
  # TODO: update the parameter order of convolution() to match the optional ones as in here? (options order matches Keras)
  strides = (1, 1, in_dim)

  apply_x = convolution (W, x,
                         strides = _as_tuple(strides),
                         sharing = _as_tuple(True),
                         auto_padding = _as_tuple(False),
                         lower_pad = (0, win_size/2, 0),
                         upper_pad = (0, (win_size-1)/2, 0)
                         )
#                         # TODO: can we rename auto_padding to pad?
  #if bias:
  #    apply_x = apply_x + b
  apply_x = apply_x >> sigmoid
  return Block(apply_x, 'Convolution', Record(W=W))

  ########################
# define the model     #
########################

def gru_cell_opt(shape, init=init_default_or_glorot_uniform, name=''): # (x, (h,c))
  shape = _as_tuple(shape)

  if len(shape) != 1 :
    raise ValueError("gru_cell: shape must be vectors (rank-1 tensors)")

  # determine stacking dimensions
  cell_shape_stacked = shape * 2  # patched dims with stack_axis duplicated 4 times

  # parameters
  Wz = Parameter(cell_shape_stacked, init = init, name='Wz')
  Wr = Parameter(cell_shape_stacked, init = init, name='Wr')
  Wh = Parameter(cell_shape_stacked, init = init, name='Wh')
  #Uz = Parameter( _INFERRED + shape, init = init, name = 'Uz')
  #Ur = Parameter( _INFERRED + shape, init = init, name = 'Ur')
  #Uh = Parameter( _INFERRED + shape, init = init, name = 'Uh')
  U = Parameter( _INFERRED + (shape[0]*3,), init = init, name = 'Uzrh')

  def create_s_placeholder():
    # we pass the known dimensions here, which makes dimension inference easier
    return Placeholder(shape=shape, name='S') # (h, c)

  # parameters to model function
  x = Placeholder(name='gru_block_arg')
  prev_status = create_s_placeholder()

  # formula of model function
  Sn_1 = prev_status

  xU = times(x, U, name='x*Uzrh')
  xUz = ops.slice(xU, -1, 0, shape[0], name='x*Uz')
  xUr = ops.slice(xU, -1, shape[0], shape[0]*2, name='x*Ur')
  xUh = ops.slice(xU, -1, shape[0]*2, shape[0]*3, name='x*Uh')
  z = sigmoid(xUz + times(Sn_1, Wz, name='h[-1]*Wz'), name='z')
  r = sigmoid(xUr + times(Sn_1, Wr, name='h[-1]*Wr'), name='r')
  h = tanh(xUh + times(element_times(Sn_1, r, name='h[-1]@r'), Wh, name='Wh*(h[-1]@r)'), name='h')
  s = plus(element_times((1-z), h, name='(1-z)@h'), element_times(z, Sn_1, name='z@[h-1]'), name=name)
  apply_x_s = combine([s])
  apply_x_s.create_placeholder = create_s_placeholder
  return apply_x_s

def gru_cell(shape, init=init_default_or_glorot_uniform, name=''): # (x, (h,c))
  shape = _as_tuple(shape)

  if len(shape) != 1 :
    raise ValueError("gru_cell: shape must be vectors (rank-1 tensors)")

  # determine stacking dimensions
  cell_shape_stacked = shape * 2  # patched dims with stack_axis duplicated 4 times

  # parameters
  Wz = Parameter(cell_shape_stacked, init = init, name='Wz')
  Wr = Parameter(cell_shape_stacked, init = init, name='Wr')
  Wh = Parameter(cell_shape_stacked, init = init, name='Wh')
  Uz = Parameter( _INFERRED + shape, init = init, name = 'Uz')
  Ur = Parameter( _INFERRED + shape, init = init, name = 'Ur')
  Uh = Parameter( _INFERRED + shape, init = init, name = 'Uh')

  def create_s_placeholder():
    # we pass the known dimensions here, which makes dimension inference easier
    return Placeholder(shape=shape, name='S') # (h, c)

  # parameters to model function
  x = Placeholder(name='gru_block_arg')
  prev_status = create_s_placeholder()

  # formula of model function
  Sn_1 = prev_status

  z = sigmoid(times(x, Uz, name='x*Uz') + times(Sn_1, Wz, name='Sprev*Wz'), name='z')
  r = sigmoid(times(x, Ur, name='x*Ur') + times(Sn_1, Wr, name='Sprev*Wr'), name='r')
  h = tanh(times(x, Uh, name='x*Uh') + times(element_times(Sn_1, r, name='Sprev*r'), Wh), name='h')
  s = plus(element_times((1-z), h, name='(1-z)*h'), element_times(z, Sn_1, name='z*SPrev'), name=name)
  apply_x_s = combine([s])
  apply_x_s.create_placeholder = create_s_placeholder
  return apply_x_s

def bidirectionalLSTM(hidden_dim, x, splice_outputs=True):
  bwd = Recurrence(LSTM(hidden_dim), go_backwards=True ) (stabilize(x))
  if splice_outputs:
    # splice the outputs together
    hc = splice((fwd, bwd))
    return hc
  else:
    # return both (in cases where we want the 'final' hidden status)
    return (fwd, bwd)

def bidirectional_gru(hidden_dim, x, splice_outputs=True, name=''):
  fwd = Recurrence(gru_cell(hidden_dim), go_backwards=False) (stabilize(x))
  bwd = Recurrence(gru_cell(hidden_dim), go_backwards=True) (stabilize(x))
  if splice_outputs:
    # splice the outputs together
    hc = splice((fwd, bwd), name=name)
    return hc
  else:
    # return both (in cases where we want the 'final' hidden status)
    return (fwd, bwd)

def broadcast_as(op, seq_op, name=''):
  x=placeholder_variable(shape=op.shape, name='src_op')
  s=placeholder_variable(shape=seq_op.shape, name='tgt_seq')
  pd = sequence.scatter(x, sequence.is_first(s, name='isf_1'), name='sct')
  pout = placeholder_variable(op.shape, dynamic_axes=seq_op.dynamic_axes, name='pout')
  out = element_select(sequence.is_first(s, name='isf_2'), pd, past_value(pout, name='ptv'), name='br_sel')
  rlt = out.replace_placeholders({pout:sanitize_input(out), x:sanitize_input(op), s:sanitize_input(seq_op)})
  return combine([rlt], name=name)

def cosine_similarity(src, tgt, name=''):
  src_br = sequence.broadcast_as(src, tgt, name='cos_br')
  dot = ops.times_transpose(src_br, tgt)
  src_norm = ops.sqrt(ops.reduce_sum(ops.square(src_br)))
  tgt_norm = ops.sqrt(ops.reduce_sum(ops.square(tgt)))
  sim = ops.element_divide(dot, (src_norm*tgt_norm), name=name)
  return sim

def project_cosine_sim(status, memory, dim, init = init_default_or_glorot_uniform, name=''):
  cell_shape = (dim, dim)
  Wi = Parameter(cell_shape, init = init, name='Wi')
  Wm = Parameter(cell_shape, init = init, name='Wm')
  weighted_status = times(status, Wi, name = 'project_status')
  weighted_memory = times(memory, Wm, name = 'project_memory')
  return cosine_similarity(weighted_status, weighted_memory, name=name)

def termination_gate(status, dim, init = init_default_or_glorot_uniform, name=''):
  Wt = Parameter((dim, 1), init = init, name='Wt')
  return sigmoid(times(status, Wt), name=name)

def attention_rlunit(context_memory, query_memory, entity_memory, hidden_dim, init = init_default_or_glorot_uniform):
  status = Placeholder(name='status', shape=hidden_dim)
  context_attention_weight = project_cosine_sim(status, context_memory, hidden_dim, name='context_attention')
  query_attention_weight = project_cosine_sim(status, query_memory, hidden_dim, name='query_attetion')
  context_attention = sequence.reduce_sum(times(context_attention_weight, context_memory), name='C-Att')
  query_attention = sequence.reduce_sum(times(query_attention_weight, query_memory), name='Q-Att')
  attention = splice((query_attention, context_attention), name='att-sp')
  gru = gru_cell((hidden_dim, ), name='control_status')
  new_status = gru(attention, status).output
  termination_prob = termination_gate(new_status, dim=hidden_dim, name='terminate_prob')
  ans_attention = project_cosine_sim(new_status, entity_memory, hidden_dim, name='ans_attention')
  return combine([ans_attention, termination_prob, new_status], name='ReinforcementAttention')

#
# TODO: CNTK current will convert sparse variable to dense after reshape function
def create_model(vocab_dim, hidden_dim,  embedding_init=None,  embedding_dim=100, max_rl_iter=5, init=init_default_or_glorot_uniform):
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
    embedding = parameter(shape=(vocab_dim, embedding_dim), init=uniform(math.sqrt(6/(vocab_dim+embedding_dim))))
  else:
    embedding = parameter(shape=(vocab_dim, embedding_dim), init=None)
    embedding.value = embedding_init

  # TODO: Add dropout to embedding
  query_embedding  = times(query_sequence , embedding)
  context_embedding = times(context_sequence, embedding)

  # get source and context representations
  context_memory = bidirectional_gru(hidden_dim, context_embedding, name='Context_Mem')            # shape=(hidden_dim*2, *), *=context_seq_axis
  entity_condition = greater(entity_ids_mask, 0, name='condidion')
  entities_all = sequence.gather(entity_condition, entity_condition, name='entities_all')
  entity_memory = sequence.gather(context_memory, entity_condition, name='Candidate_Mem')
  #entity_memory = sequence.scatter(sequence.gather(context_memory, entity_condition, name='Candidate_Mem'), entities_all)
  qfwd, qbwd  = bidirectional_gru(hidden_dim, query_embedding, splice_outputs=False) # shape=(hidden_dim*2, *), *=query_seq_axis
  query_memory = splice((qfwd, qbwd), name='Query_SP')
  # get the source (aka 'query') representation
  init_status = splice((sequence.last(qfwd), sequence.first(qbwd)), name='Init_Status') # get last fwd status and first bwd status
  attention_rlu = attention_rlunit(context_memory, query_memory, entity_memory, hidden_dim*2, init)
  status_controls = []
  arlus = [None] * max_rl_iter
  answers = None
  probs = None
  for i in range(0, max_rl_iter):
    if i == 0:
      arlus[i] = attention_rlu(init_status)
    else:
      arlus[i] = attention_rlu(arlus[i-1].outputs[2])
    status_controls += list(arlus[i].outputs[0:2])
    if answers == None:
      answers = element_times(arlus[i].outputs[0], sequence.broadcast_as(arlus[i].outputs[1], arlus[i].outputs[0]))
      probs = arlus[i].outputs[1]
    else:
      answers += element_times(arlus[i].outputs[0], sequence.broadcast_as(arlus[i].outputs[1], arlus[i].outputs[0]))
      probs += arlus[i].outputs[1]
  final_answers = reshape(element_divide(answers, sequence.broadcast_as(probs, answers)), (1,), name='final_answers')
  result = combine([final_answers], name='ReasoNet')
  #result = combine(status_controls+[final_answers], name='ReasoNet')
  return Block(result, 'ReasoNet', Record(vocab_dim=vocab_dim, hidden_dim=hidden_dim, max_iter =max_rl_iter, context=context_sequence,
    query=query_sequence, entities=entity_ids_mask, entity_condition=entity_condition,
    entities_all=entities_all))

def pred(model):
  context = model.context
  entities = model.entities
  wordvocab_dim = model.vocab_dim
  entity_condition = model.entity_condition
  entities_all = model.entities_all
  answers = sequence.scatter(model.outputs[-1], entities_all, name='answers_prob')
  entity_ids = sequence.scatter(sequence.gather(reshape(model.context, wordvocab_dim), entity_condition), entities_all)
  item_preds = sequence.reduce_sum(times(reshape(answers, (1,)), entity_ids), name = 'item_preds')
  mask = sequence.reduce_sum(entity_ids, name='mask')
  probs = ops.element_select(mask, ops.exp(item_preds), 0, name='item_prob')
  return combine([mask, probs])

def accuracy_func(pred, label, name='accuracy'):
  pred_max = ops.hardmax(pred, name='pred_max')
  norm_label = ops.equal(label, [1], name='norm_label')
  acc = ops.times_transpose(pred_max, norm_label, name='accuracy')
  return acc

def seq_accuracy(pred, label, name=''):
  m = placeholder_variable(shape=(1,), dynamic_axes = pred.dynamic_axes, name='max')
  o = element_select(greater(pred, past_value(m)), pred, past_value(m))
  rlt = o.replace_placeholders({m:sanitize_input(o)})
  max_val = sequence.broadcast_as(sequence.last(rlt), rlt)
  first_max = sequence.first(sequence.where(ops.greater_equal(pred, max_val)))
  label_idx = sequence.first(sequence.where(ops.equal(label, 1), name='Label_idx'))
  return ops.equal(first_max, label_idx, name=name)

def seq_cross_entropy(pred, label, gama=10, name=''):
  #loss = ops.negate(sequence.reduce_sum(times(label, ops.log(pred_exp/(sequence.broadcast_as(sum_exp, pred))))), name = name)
  pred_exp = ops.exp(pred*gama, name='pred_exp')
  sum_exp = sequence.reduce_sum(pred_exp, name='sum_exp')
  pred_prob = element_divide(pred_exp, sequence.broadcast_as(sum_exp, pred), name='prob')
  log_prob = ops.log(pred_prob, name='log_prob')
  label_softmax = ops.element_times(label, log_prob, name = 'label_softmax')
  entropy = ops.negate(sequence.reduce_sum(label_softmax), name=name)
  accuracy = seq_accuracy(pred_prob, label, name='accuracy')
  return (entropy, accuracy)

def mask_cross_entropy(pred, label, mask, gama=10, name=''):
  pred_exp = element_select(mask, ops.exp(gama*pred), 0)
  label_msk = element_select(label, 1, 0)
  sum_exp = ops.reduce_sum(pred_exp)
  soft_max = ops.element_select(mask, ops.negate(ops.element_times(label_msk, ops.log(pred_exp/sum_exp))), 0)
  return ops.reduce_sum(soft_max, name=name)

def loss(model):
  context = model.context
  entities = model.entities
  wordvocab_dim = model.vocab_dim
  labels_raw = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context.dynamic_axes, name='labels')
  entity_condition = model.entity_condition
  entities_all = model.entities_all
  answers = model.outputs[-1]
  labels = sequence.gather(labels_raw, entity_condition, name='EntityLabels')
  #labels = sequence.scatter(sequence.gather(labels_raw, entity_condition, name='EntityLabels'), entities_all, name='seq_labels')
  cross_entroy, accuracy = seq_cross_entropy(answers, labels, name='CrossEntropyLoss')
  apply_loss = combine([cross_entroy, answers, labels, accuracy])
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

def train(model, train_data, max_epochs=1, save_model_flag=False, epoch_size=270000, model_name='rsn', eval_data=None, eval_size=None):
  # Criterion nodes
  criterion_loss = loss(model)
  loss_func = criterion_loss.outputs[0]
  eval_func = criterion_loss.outputs[-1]
  
  # Instantiate the trainer object to drive the model training
  learning_rate = 0.05
  lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.minibatch)
  minibatch_size = 12000
  momentum = learner.momentum_schedule(0.9) 
  momentum_var = learner.momentum_schedule(0.999)
  clipping_threshold_per_sample = 10.0
  gradient_clipping_with_truncation = True
  #learn = learner.adam_sgd(model.parameters, lr_schedule, momentum, momentum_var, 
  #           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
  #           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
  #learn = learner.adagrad(model.parameters, lr_schedule, gradient_clipping_threshold_per_sample = clipping_threshold_per_sample, gradient_clipping_with_truncation = gradient_clipping_with_truncation)
  learn = learner.momentum_sgd(model.parameters, lr_schedule, momentum, True, 
              gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
              gradient_clipping_with_truncation=gradient_clipping_with_truncation)

  trainer = Trainer(model.outputs[-1], loss_func, eval_func, learn)
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

  for epoch in range(max_epochs):
    loss_numer = 0
    metric_numer = 0
    total_samples = 0
    i = 0
    win_loss = 0
    win_acc = 0
    win_samples = 0
    while i < epoch_size:
      # get next minibatch of training data
      # TODO: Shuffle entities? @yelong
      mb_train = train_data.next_minibatch(minibatch_size, input_map=train_bind)
      i += mb_train[label_key].num_samples
      trainer.train_minibatch(mb_train)
      minibatch_count += 1
      sys.stdout.write('.')
      sys.stdout.flush()
      # collect epoch-wide stats
      samples = trainer.previous_minibatch_sample_count
      loss_numer += trainer.previous_minibatch_loss_average * samples
      metric_numer += trainer.previous_minibatch_evaluation_average * samples
      total_samples += samples
      win_samples += samples
      win_loss += trainer.previous_minibatch_loss_average * samples
      win_acc += trainer.previous_minibatch_evaluation_average * samples
      if int(total_samples/training_progress_output_freq) != int((total_samples-samples)/training_progress_output_freq):
        print('')
        log("Lastest sample count = {}, Train Loss: {}, Evalualtion ACC: {}".format(win_samples, win_loss/win_samples, 
          win_acc/win_samples))
        log("Total sample count = {}, Train Loss: {}, Evalualtion ACC: {}".format(total_samples, loss_numer/total_samples, 
          metric_numer/total_samples))
        win_samples = 0
        win_loss = 0
        win_acc = 0

    print('')
    log("--- EPOCH %d: samples=%d, loss = %.2f, acc = %.2f%% ---" % (epoch, total_samples, loss_numer/total_samples, 100.0*(metric_numer/total_samples)))
    if eval_data:
      evaluation(trainer, eval_data, eval_bind, minibatch_size, eval_size)

    if save_model_flag:
      # save the model every epoch
      model_filename = os.path.join('model', "model_%s_%03d.dnn" % (model_name, epoch))
      model.save_model(model_filename)
      print("Saved model to '%s'" % model_filename)
