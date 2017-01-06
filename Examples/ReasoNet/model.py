"""
ReasoNet model in CNTK
@author penhe@microsoft.com
"""
import sys
import os
import numpy as np
from cntk import Trainer, Axis, device, combine
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, reduce_sum, \
    parameter, times, element_times, past_value, plus, placeholder_variable, splice, reshape, constant, sigmoid, convolution, tanh, times_transpose, greater, cosine_distance, element_divide, element_select
import cntk.ops as ops
from cntk.blocks import LSTM, Stabilizer, _get_current_default_options, _is_given, _initializer_for, _resolve_activation, _INFERRED, Parameter, Placeholder, Block, init_default_or_glorot_uniform
from cntk.layers import Recurrence, Convolution
from cntk.initializer import uniform, glorot_uniform
from cntk.utils import get_train_eval_criterion, get_train_loss, Record, _as_tuple, sanitize_input, value_to_seq
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node


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

def create_reader(path, vocab_dim, randomize, size=INFINITELY_REPEAT):
  return MinibatchSource(CTFDeserializer(path, StreamDefs(
    context  = StreamDef(field='C', shape=vocab_dim, is_sparse=True),
    query    = StreamDef(field='Q', shape=vocab_dim, is_sparse=True),
    entities  = StreamDef(field='E', shape=1, is_sparse=False),
    label   = StreamDef(field='L', shape=1, is_sparse=False)
    )), randomize=randomize, epoch_size = size)

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
  fwd = Recurrence(LSTM(hidden_dim), go_backwards=False) (stabilize(x))
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
  src_br = broadcast_as(src, tgt, name='cos_br')
  sim = cosine_distance(src_br, tgt, name=name)
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

def attention_rlunit(context_memory, query_memory, candidate_memory, candidate_ids,hidden_dim, vocab_dim, init = init_default_or_glorot_uniform):
  status = Placeholder(name='status', shape=hidden_dim)
  context_attention_weight = project_cosine_sim(status, context_memory, hidden_dim, name='context_attention')
  query_attention_weight = project_cosine_sim(status, query_memory, hidden_dim, name='query_attetion')
  context_attention = sequence.reduce_sum(element_times(context_attention_weight, context_memory), name='C-Att')
  query_attention = sequence.reduce_sum(element_times(query_attention_weight, query_memory), name='Q-Att')
  attention = splice((query_attention, context_attention), name='att-sp')
  gru = gru_cell((hidden_dim, ), name='status')
  new_status = gru(attention, status).output
  termination_prob = termination_gate(new_status, dim=hidden_dim, name='prob')
  ans_attention = project_cosine_sim(new_status, candidate_memory, hidden_dim, name='ans_attention')
  answers = times(ans_attention, candidate_ids, name='answers')
  return combine([answers, termination_prob, new_status], name='ReinforcementAttention')

def seq_cross_entropy(pred, label, gama=10, name=''):
  pred_exp = ops.exp(pred, name='pred_exp')
  sum_exp = sequence.reduce_sum(pred_exp, name='sum_exp')
  pred_sum = element_divide(pred_exp, sequence.broadcast_as(sum_exp, pred), name='exp_divid')
  log_pred_sum = ops.log(pred_sum, name='log_pred')
  label_pred = times(label, log_pred_sum, name = 'label_softmax')
  entropy = ops.negate(sequence.reduce_sum(label_pred, name='sum_log'), name=name)
  #loss = ops.negate(sequence.reduce_sum(times(label, ops.log(pred_exp/(sequence.broadcast_as(sum_exp, pred))))), name = name)
  return entropy

def mask_cross_entropy(pred, label, mask, gama=10, name=''):
  pred_exp = element_select(mask, ops.exp(gama*pred), 0)
  label_msk = element_select(label, 1, 0)
  sum_exp = ops.reduce_sum(pred_exp)
  soft_max = ops.element_select(mask, ops.negate(ops.element_times(label_msk, ops.log(pred_exp/sum_exp))), 0)
  return ops.reduce_sum(soft_max, name=name)

def create_model(vocab_dim, hidden_dim, max_rl_iter=5, init=init_default_or_glorot_uniform):
  # Query and Doc/Context/Paragraph inputs to the model
  batch_axis = Axis.default_batch_axis()
  query_seq_axis = Axis('sourceAxis')
  context_seq_axis = Axis('contextAxis')
  query_dynamic_axes = [batch_axis, query_seq_axis]
  query_raw = input_variable(shape=(vocab_dim), is_sparse=True, dynamic_axes=query_dynamic_axes, name='query')
  context_dynamic_axes = [batch_axis, context_seq_axis]
  context_raw = input_variable(shape=(vocab_dim), is_sparse=True, dynamic_axes=context_dynamic_axes, name='context')
  candidate_dynamic_axes = [batch_axis, context_seq_axis]
  candidate_indicates = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context_dynamic_axes, name='entities')

  # Query sequences
  query_sequence = query_raw
  # Doc/Context sequences
  context_sequence = context_raw
  # embedding
  embed_dim = hidden_dim
  embedding = parameter(shape=(vocab_dim, embed_dim), init=uniform(1))

  # TODO: Use Golve to initialize the embedding
  # TODO: Add dropout to embedding
  query_embedding  = times(query_sequence , embedding)
  context_embedding = times(context_sequence, embedding)

  # get source and context representations
  context_memory = bidirectional_gru(hidden_dim, context_embedding, name='Context_Mem')            # shape=(hidden_dim*2, *), *=context_seq_axis
  candidate_filter = greater(candidate_indicates, 0)
  candidate_sc = sequence.gather(candidate_filter, candidate_filter)
  candidate_memory = sequence.scatter(sequence.gather(context_memory, candidate_filter, name='Candidate_Mem'), candidate_sc)
  candidate_ids = sequence.scatter(sequence.gather(candidate_indicates, candidate_filter, name = 'Candidate_Ids'), candidate_sc)
  entity_raw_ids = sequence.scatter(sequence.gather(reshape(context_raw, vocab_dim), candidate_filter), candidate_sc)
  qfwd, qbwd  = bidirectional_gru(hidden_dim, query_embedding, splice_outputs=False) # shape=(hidden_dim*2, *), *=query_seq_axis
  query_memory = splice((qfwd, qbwd), name='Query_SP')
  # get the source (aka 'query') representation
  status = splice((sequence.last(qfwd), sequence.first(qbwd)), name='Init_Status') # get last fwd status and first bwd status
  attention_rlu = attention_rlunit(context_memory, query_memory, candidate_memory, candidate_ids, hidden_dim*2, vocab_dim, init)
  status_controls = []
  arlus = [None] * max_rl_iter
  answers = None
  probs = None
  for i in range(0, max_rl_iter):
    arlus[i] = attention_rlu(status)
    status = arlus[i].outputs[2]
    status_controls += list(arlus[i].outputs[0:2])
    if answers == None:
      answers = element_times(arlus[i].outputs[0], sequence.broadcast_as(arlus[i].outputs[1], arlus[i].outputs[0]))
      probs = arlus[i].outputs[1]
    else:
      answers += element_times(arlus[i].outputs[0], sequence.broadcast_as(arlus[i].outputs[1], arlus[i].outputs[0]))
      probs += arlus[i].outputs[1]
  final_answers = element_divide(answers, sequence.broadcast_as(probs, answers), name='final_answers')
  result = combine(status_controls+[final_answers], name='ReasoNet')
  return Block(result, 'ReasoNet', Record(vocab_dim=vocab_dim, hidden_dim=hidden_dim, max_ite =max_rl_iter, context=context_raw,
    query=query_raw, entities=candidate_indicates, entity_masks=candidate_filter,
    entity_seqs=candidate_sc, entity_ids=entity_raw_ids))

def pred(model):
  context = model.context
  entities = model.entities
  wordvocab_dim = model.vocab_dim
  candidate_filter = model.entity_masks
  candidate_sc = model.entity_seqs
  answers = sequence.scatter(model.outputs[-1], candidate_sc, name='answers_prob')
  entity_ids = model.entity_ids
  item_preds = sequence.reduce_sum(times(reshape(answers, (1,)), entity_ids), name = 'item_preds')
  mask = sequence.reduce_sum(entity_ids, name='mask')
  probs = ops.element_select(mask, ops.exp(item_preds), 0, name='item_prob')
  return combine([mask, probs])

def loss(model):
  context = model.context
  entities = model.entities
  wordvocab_dim = model.vocab_dim
  labels_raw = input_variable(shape=(1,), is_sparse=False, dynamic_axes=context.dynamic_axes, name='labels')
  candidate_filter = model.entity_masks
  candidate_sc = model.entity_seqs
  answers = sequence.scatter(model.outputs[-1], candidate_sc, name='answers_prob')
  entity_ids = model.entity_ids
  item_preds = sequence.reduce_sum(times(reshape(answers, (1,)), entity_ids), name = 'item_preds')
  labels = sequence.scatter(sequence.gather(labels_raw, candidate_filter, name='EntityLabels'), candidate_sc, name='seq_labels')
  #cross_entroy = seq_cross_entroy(reshape(answers, (1,)), labels, name='CrossEntropyLoss')
  item_labels = sequence.reduce_sum(times(reshape(labels, (1,)), entity_ids), name='item_labels')
  mask = sequence.reduce_sum(entity_ids)
  cross_entroy = mask_cross_entropy(item_preds, item_labels, mask, name='CrossEntropyLoss')
  probs = ops.element_select(mask, ops.exp(item_preds), 0, name='item_probs')
  apply_loss = combine([cross_entroy, answers, labels, item_preds, probs])
  return Block(apply_loss, 'AvgSoftMaxCrossEntropy', Record(labels=item_labels))

#TODO: Add AUC for evaluation

def train(model, reader, max_epochs=1, save_model_flag=False, epoch_size=270000):
  # Criterion nodes
  criterion_loss = loss(model)
  loss_func = criterion_loss.outputs[0]
  eval_func = classification_error(criterion_loss.outputs[-1], criterion_loss.labels)
  
  # Instantiate the trainer object to drive the model training
  learning_rate = 0.005
  lr_per_sample = learning_rate_schedule(learning_rate, UnitType.minibatch)

  #minibatch_size = 30000 # max(sequence_length) --> so with avg length of context=1000 this is like 30 "full samples"
  minibatch_size = 5000

  momentum_time_constant = momentum_as_time_constant_schedule(1100)
  clipping_threshold_per_sample = 10.0
  gradient_clipping_with_truncation = True
  learner = momentum_sgd(model.parameters,
             lr_per_sample, momentum_time_constant,
             gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
             gradient_clipping_with_truncation=gradient_clipping_with_truncation)
  trainer = Trainer(model.outputs[-1], loss_func, eval_func, learner)

  # Get minibatches of sequences to train with and perform model training
  i = 0
  mbs = 0
  #epoch_size = 270000 # this number is in sequences -- need to fix (unfortunately has to be in 'elements' for now)
  # for ^^, we just need to keep adding up all the samples (1 per sequence) and end the epoch once we get to 270000
  training_progress_output_freq = 1

  # bind inputs to data from readers
  data_bind = {}
  label_key = None
  for arg in criterion_loss.arguments:
    if arg.name == 'query':
      data_bind[arg] = reader.streams.query
    if arg.name == 'context':
      data_bind[arg] = reader.streams.context
    if arg.name == 'entities':
      data_bind[arg] = reader.streams.entities
    if arg.name == 'labels':
      label_key = arg
      data_bind[arg] = reader.streams.label

  for epoch in range(max_epochs):
    loss_numer = 0
    metric_numer = 0
    denom = 0

    while i < (epoch+1) * epoch_size:

      # get next minibatch of training data
      #mb_train = train_reader.next_minibatch(minibatch_size_in_samples=minibatch_size, input_map=train_bind)
      # TODO: When will next_minibatch ended?
      # TODO: Shuffle entities? @yelong
      mb_train = reader.next_minibatch(1024, input_map=data_bind)
      trainer.train_minibatch(mb_train)

      # collect epoch-wide stats
      samples = trainer.previous_minibatch_sample_count
      loss_numer += trainer.previous_minibatch_loss_average * samples
      metric_numer += trainer.previous_minibatch_evaluation_average * samples
      denom += samples

      # debugging
      #print("previous minibatch sample count = %d" % samples)
      #print("mb_train[labels] num samples = %d" % mb_train[labels].num_samples)
      #print("previous minibatch loss average = %f" % trainer.previous_minibatch_loss_average)

      if mbs % training_progress_output_freq == 0:
        print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(mbs, 
            get_train_loss(trainer), get_train_eval_criterion(trainer)))
        print("previous minibatch sample count = %d" % samples)

      i += mb_train[label_key].num_samples
      mbs += 1

    print("--- EPOCH %d DONE: loss = %f, errs = %f ---" % (epoch, loss_numer/denom, 100.0*(metric_numer/denom)))

    if save_model_flag:
      # save the model every epoch
      model_filename = os.path.join('model', "model_epoch%d.dnn" % epoch)
      model.save_model(model_filename)
      print("Saved model to '%s'" % model_filename)
