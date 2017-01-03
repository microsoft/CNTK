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

def bidirectional_gru2(hidden_dim, x, splice_outputs=True, name=''):
  fwd = Recurrence(gru_cell(hidden_dim), go_backwards=False) (stabilize(x))
  bwd = Recurrence(gru_cell(hidden_dim), go_backwards=True) (stabilize(x))
  if splice_outputs:
    # splice the outputs together
    hc = splice((fwd, bwd), name=name)
    return hc
  else:
    # return both (in cases where we want the 'final' hidden status)
    return (fwd, bwd)

def bidirectional_gru(hidden_dim, x, splice_outputs=True, name=''):
  W = Parameter(_INFERRED +  _as_tuple(hidden_dim), init=glorot_uniform(), name='gru_params')
  dualstatus = ops.optimized_rnnstack(x, W, hidden_dim, 1, True, recurrent_op='gru', name=name)
  if splice_outputs:
    # splice the outputs together
    return dualstatus
  else:
    # return both (in cases where we want the 'final' hidden status)
    return (ops.slice(sequence.last(dualstatus), -1, 0, hidden_dim, name='fwd'), ops.slice(sequence.first(dualstatus), -1, hidden_dim, hidden_dim*2, name='bwd'), dualstatus) 

def broadcast_as(op, seq_op, name=''):
  x=placeholder_variable(shape=op.shape, name='src_op')
  s=placeholder_variable(shape=seq_op.shape, name='tgt_seq')
  pd = sequence.scatter(x, sequence.is_first(s, name='isf_1'), name='sct')
  pout = placeholder_variable(op.shape, dynamic_axes=seq_op.dynamic_axes, name='pout')
  out = element_select(sequence.is_first(s, name='isf_2'), pd, past_value(pout, name='ptv'), name='br_sel')
  rlt = out.replace_placeholders({pout:sanitize_input(out), x:sanitize_input(op), s:sanitize_input(seq_op)})
  return combine([rlt], name=name)

def cosine_similarity(src, tgt, name=''):
  src_dup = False
  if True:
  #if len(src.dynamic_axes)>0 and src.dynamic_axes[0].name=='defaultBatchAxis':
    src_br = sequence.broadcast_as(src, tgt, name='cos_br')
    src_dup = True
  else:
    src_br = src
  sim = cosine_distance(src_br, tgt, name)
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
  #return times(status, Wt, name=name)

def seq_max(x):
  m = placeholder_variable(shape=(1,), dynamic_axes = x.dynamic_axes, name='max')
  o = element_select(greater(x, ops.future_value(m)), x, ops.future_value(m))
  rlt = o.replace_placeholders({m:sanitize_input(o)})
  max_v = sequence.is_first(x)
  pv = placeholder_variable(shape=(1,), dynamic_axes = x.dynamic_axes, name='max_seq')
  max_seq = element_select(sequence.is_first(x), utils.sanitize_input(rlt), ops.past_value(pv))
  max_br = max_seq.replace_placeholders({pv:utils.sanitize_input(max_seq)})
  return utils.sanitize_input(max_br)

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
  softmax_norm = True
  if softmax_norm:
    #max_ans = sequence.broadcast_as(seq_max(ans_attention), ans_attention)
    #ans_exp = ops.exp(ans_attention*10)
    #max_ans = seq_max(ans_attention)
    ans_exp = ops.exp(ans_attention*10)
    ans_attention = element_divide(ans_exp, sequence.broadcast_as(sequence.reduce_sum(ans_exp), ans_exp), name='norm_ans')
  return combine([ans_attention, termination_prob, new_status], name='ReinforcementAttention')

#def attention_module(hidden_dim, max_steps = 5, init = init_default_or_glorot_uniform):
def attention_module(context_memory, query_memory, entity_memory, init_status, hidden_dim, max_steps = 5, init = init_default_or_glorot_uniform):
  c_mem = ops.placeholder_variable(shape=context_memory.shape, dynamic_axes=context_memory.dynamic_axes, name='c_mem')
  q_mem = ops.placeholder_variable(shape=query_memory.shape, dynamic_axes=query_memory.dynamic_axes, name='c_mem')
  e_mem = ops.placeholder_variable(shape=entity_memory.shape, dynamic_axes=entity_memory.dynamic_axes, name='c_mem')
  c_mem = ops.placeholder_variable(name='c_mem')
  q_mem = ops.placeholder_variable(name='q_mem')
  e_mem = ops.placeholder_variable(name='e_mem')
  i_status = ops.placeholder_variable(shape=(hidden_dim), name='init_status')
  gru = gru_cell((hidden_dim, ), name='control_status')
  status = ops.alias(i_status, name='status')
  output = [None]*max_steps*2
  sum_prob = None
  for step in range(max_steps):
    context_attention_weight = project_cosine_sim(status, c_mem, hidden_dim, name='context_attention')
    query_attention_weight = project_cosine_sim(status, q_mem, hidden_dim, name='query_attetion')
    context_attention = sequence.reduce_sum(times(context_attention_weight, c_mem), name='C-Att')
    query_attention = sequence.reduce_sum(times(query_attention_weight, q_mem), name='Q-Att')
    attention = splice((query_attention, context_attention), name='att-sp')
    status = gru(attention, status).output
    termination_prob = termination_gate(status, dim=hidden_dim, name='terminate_prob')
    ans_attention = project_cosine_sim(status, e_mem, hidden_dim, name='ans_attention')
    ans_exp = ops.exp(ans_attention*10)
    #ans_attention = element_divide(ans_exp, sequence.broadcast_as(sequence.reduce_sum(ans_exp), ans_exp), name='norm_ans')
    prob_exp = ops.exp(termination_prob*10)
    output[step*2+1] = prob_exp
    output[step*2] = ans_exp/sequence.broadcast_as(sequence.reduce_sum(ans_exp), ans_exp)
    if sum_prob is None:
      sum_prob = prob_exp
    else:
      sum_prob += prob_exp

  final_ans = None
  for step in range(max_steps):
    output[step*2+1] = element_divide(output[step*2+1], sum_prob)
    output[step*2] = ops.element_times(output[step*2], sequence.broadcast_as(output[step*2+1], output[step*2]))
    if final_ans is None:
      final_ans = output[step*2]
    else:
      final_ans += output[step*2]
  combine_func = combine([final_ans], name='Attention_func') 
  #combine_func = combine([final_ans], name='Attention_func') 
  #block_func = ops.as_block(combine_func, [], 'Attention_module', 'Attention_module')
  block_func = ops.as_block(combine_func, [(c_mem, ops.sanitize_input(context_memory)), (q_mem, ops.sanitize_input(query_memory)), (e_mem, ops.sanitize_input(entity_memory)), (i_status, ops.sanitize_input(init_status))], 'Attention_module', 'Attention_module')
  #return combine([ans_attention, termination_prob, new_status], name='ReinforcementAttention')
  #return combine_func.clone(ops.CloneMethod.share, {c_mem: ops.sanitize_input(context_memory), q_mem: ops.sanitize_input(query_memory), e_mem: ops.sanitize_input(entity_memory), i_status: ops.sanitize_input(init_status)})
  return block_func
#
# TODO: CNTK current will convert sparse variable to dense after reshape function
def create_model(vocab_dim, entity_dim, hidden_dim,  embedding_init=None,  embedding_dim=100, max_rl_iter=5, init=init_default_or_glorot_uniform):
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
  query_embedding  = ops.dropout(times(query_sequence , embedding), 0.2, name='query_embedding')
  context_embedding = ops.dropout(times(context_sequence, embedding), 0.2, name='context_embedding')

  # get source and context representations
  context_memory = bidirectional_gru(hidden_dim, context_embedding, name='Context_Mem')            # shape=(hidden_dim*2, *), *=context_seq_axis
  entity_condition = greater(entity_ids_mask, 0, name='condidion')
  entities_all = sequence.gather(entity_condition, entity_condition, name='entities_all')
  entity_memory = sequence.gather(context_memory, entity_condition, name='Candidate_Mem')
  entity_ids = input_variable(shape=(entity_dim), is_sparse=True, dynamic_axes=entity_memory.dynamic_axes, name='entity_ids')
  #entity_memory = sequence.scatter(sequence.gather(context_memory, entity_condition, name='Candidate_Mem'), entities_all)
  qfwd, qbwd, query_memory  = bidirectional_gru(hidden_dim, query_embedding, splice_outputs=False, name='Query_Mem') # shape=(hidden_dim*2, *), *=query_seq_axis
  init_status = splice((qfwd, qbwd), name='Init_Status') # get last fwd status and first bwd status
  # attention
  #attention_rlu = attention_rlunit(context_memory, query_memory, entity_memory, hidden_dim*2, init)
  #status_controls = [None]*(max_rl_iter*2)
  #arlus = [None] * max_rl_iter
  #answers = None
  #probs = None
  #term_prob_max = None
  #for i in range(0, max_rl_iter):
  #  if i == 0:
  #    arlus[i] = attention_rlu(init_status)
  #    term_prob_max = arlus[i].outputs[1]
  #  else:
  #    arlus[i] = attention_rlu(arlus[i-1].outputs[2])
  #    term_prob_max = ops.element_select(ops.greater(arlus[i].outputs[1], term_prob_max), arlus[i].outputs[1], term_prob_max)
  #for i in range(0, max_rl_iter):
  #  #term_prob_exp = ops.exp(arlus[i].outputs[1]*10)
  #  term_prob_exp = ops.exp(arlus[i].outputs[1]*10)
  #  step_ans = element_times(arlus[i].outputs[0], sequence.broadcast_as(term_prob_exp, arlus[i].outputs[0]))
  #  status_controls[2*i] = step_ans
  #  status_controls[2*i+1] = term_prob_exp
  #  if answers is None:
  #    answers = step_ans
  #    probs = term_prob_exp
  #  else:
  #    answers += step_ans
  #    probs += term_prob_exp
  ##final_answers = reshape(element_divide(answers, probs), (1,), name='final_answers')
  #prob_sum = sequence.broadcast_as(probs, answers)
  #final_answers = reshape(element_divide(answers, prob_sum), (1,), name='final_answers')
  #for i in range(0, max_rl_iter):
  #  status_controls[i*2+1] = reshape(element_divide(status_controls[i*2+1], probs), (1,))
  #  status_controls[i*2] = reshape(element_divide(status_controls[i*2], prob_sum),(1,))

  #result = combine([final_answers], name='ReasoNet')
  #result = combine(status_controls+[final_answers], name='ReasoNet')
  result = attention_module(context_memory, query_memory, entity_memory, init_status, hidden_dim*2, max_steps = max_rl_iter)
  #att_m = attention_module(hidden_dim*2, max_steps = max_rl_iter)
  #result = att_m(context_memory, query_memory, entity_memory, init_status)
  return Block(result, 'ReasoNet', Record(vocab_dim=vocab_dim, hidden_dim=hidden_dim, max_iter =max_rl_iter, context=context_sequence,
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
  return entropy

def mask_cross_entropy(pred, label, mask, gama=10, name=''):
  pred_exp = element_select(mask, ops.exp(gama*pred), 0)
  label_msk = element_select(label, 1, 0)
  sum_exp = ops.reduce_sum(pred_exp)
  soft_max = ops.element_select(mask, ops.negate(ops.element_times(label_msk, ops.log(pred_exp/sum_exp))), 0)
  return ops.reduce_sum(soft_max, name=name)

def softmax_cross_entropy(pred, label, gama=10, name=''):
  pred_exp = ops.exp(gama*pred)
  #pred_exp = ops.exp(gama*(pred-ops.reduce_max(pred)))
  sum_exp = ops.reduce_sum(pred_exp)
  soft_max = ops.negate(ops.element_times(label, ops.log(pred_exp/sum_exp)))
  return ops.reduce_sum(soft_max, name=name)

def contractive_reward(model):
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
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix))
  expand_label = ops.greater_equal(sequence.reduce_sum(element_times(labels, entity_id_matrix)), 1)
  max_rl_iter = int((len(model.outputs)-1)/2)
  base = None
  reward = [None]*max_rl_iter
  for i in range(0, max_rl_iter):
    if i == 0:
      base = model.outputs[i*2]
    else:
      base += model.outputs[i*2]
  reward_sum = None
  reward_exp = [None]*max_rl_iter
  base_sum = ops.times_transpose(sequence.reduce_sum(element_times(reshape(base, (1, )), entity_id_matrix)), expand_label)
  for i in range(0, max_rl_iter):
    reward[i] = ops.log(model.outputs[i*2+1])*ops.exp(ops.times_transpose(sequence.reduce_sum(element_times(reshape(model.outputs[i*2], (1,)), entity_id_matrix)), expand_label)/base_sum - 1)
    if i == 0:
      reward_sum = reward[i]
    else:
      reward_sum += reward[i]
  #cross_entroy = softmax_cross_entropy(expand_pred, expand_label, name='CrossEntropy')
  accuracy = accuracy_func(expand_pred, expand_label, name='accuracy')
  #apply_loss = combine([cross_entroy, answers, labels, accuracy])
  #apply_loss = combine([cross_entroy, answers, labels, reward_sum])
  apply_loss = combine([reward_sum, answers, labels, accuracy])
  return Block(apply_loss, 'AvgSoftMaxCrossEntropy', Record(labels=labels_raw))

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
  entity_id_matrix = ops.reshape(entity_ids, entity_dim)
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix))
  expand_label = ops.greater_equal(sequence.reduce_sum(element_times(labels, entity_id_matrix)), 1)
  cross_entroy = softmax_cross_entropy(expand_pred, expand_label, name='CrossEntropy')
  accuracy = accuracy_func(expand_pred, expand_label, name='accuracy')
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
  #criterion_loss = contractive_reward(model)
  criterion_loss = loss(model)
  loss_func = criterion_loss.outputs[0]
  eval_func = criterion_loss.outputs[-1]
  if model_name is not None:
    log_name = model_name+'_log'

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
    learning_rate = 0.0005
    lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.minibatch, 1)
    momentum = learner.momentum_schedule(0.90)
    clipping_threshold_per_sample = 10
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
  minibatch_size = 40000
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
