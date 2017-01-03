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
from cntk.blocks import LSTM, Stabilizer, _get_current_default_options, _is_given, _initializer_for, _resolve_activation, _INFERRED, Parameter, Placeholder, Block, init_default_or_glorot_uniform
from cntk.layers import Recurrence, Convolution
from cntk.initializer import uniform, glorot_uniform
from cntk.utils import get_train_eval_criterion, get_train_loss, Record, _as_tuple, sanitize_input
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node


########################
# variables and stuff  #
########################

data_dir = "./data"
model_dir = "./models"

# model dimensions
#vocab_dim = 40000
#embed_dim = 200
#hidden_dim = 256
#src_max_len = 120
#ctx_max_len = 1989

# stabilizer
stabilize = Stabilizer()

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

def create_reader(path, randomize, size=INFINITELY_REPEAT):
  return MinibatchSource(CTFDeserializer(path, StreamDefs(
    context  = StreamDef(field='C', shape=vocab_dim, is_sparse=True),
    query    = StreamDef(field='Q', shape=vocab_dim, is_sparse=True),
    answer   = StreamDef(field='A', shape=vocab_dim, is_sparse=True)
    )), randomize=randomize, epoch_size = size)

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
  return cosine_similarity(status, memory, name=name)

def project_dotprod_sim(status, memory, dim, init = init_default_or_glorot_uniform, name=''):
  cell_shape = (dim, dim)
  Wi = Parameter(cell_shape, init = init, name='Wi')
  Wm = Parameter(cell_shape, init = init, name='Wm')
  weighted_status = times(status, Wi, name = 'project_status')
  weighted_memory = times(memory, Wm, name = 'project_memory')
  return times_transpose(broadcast_as(weighted_status, weighted_memory), weighted_memory)

def termination_gate(status, dim, init = init_default_or_glorot_uniform, name=''):
  Wt = Parameter((dim, 1), init = init, name='Wt')
  return sigmoid(times(status, Wt), name=name)

def attention_rlunit(context_memory, query_memory, candidate_memory, candidate_ids,hidden_dim, vocab_dim, init = init_default_or_glorot_uniform):
  status = Placeholder(name='status', shape=hidden_dim)
  context_attention_weight = project_cosine_sim(status, context_memory, hidden_dim, name='context_attention')
  query_attention_weight = project_cosine_sim(status, query_memory, hidden_dim, name='query_attetion')
  context_attention = sequence.reduce_sum(times(context_attention_weight, context_memory), name='C-Att')
  query_attention = sequence.reduce_sum(times(query_attention_weight, query_memory), name='Q-Att')
  attention = splice((query_attention, context_attention), name='att-sp')
  gru = gru_cell((hidden_dim, ), name='status')
  new_status = gru(attention, status).output
  termination_prob = termination_gate(new_status, dim=hidden_dim, name='prob')
  ans_attention = project_cosine_sim(new_status, candidate_memory, hidden_dim, name='ans_attention')
  answers = times(ans_attention, candidate_ids, name='answers')
  return combine([answers, termination_prob, new_status], name='ReinforcementAttention')

def attention_rlunit2(hidden_dim, vocab_dim, init = init_default_or_glorot_uniform):
  status = Placeholder(name='status')
  context_memory = Placeholder(name='context_memory')
  query_memory = Placeholder(name='query_memory')
  candidate_memory = Placeholder(name='candidate_memory')
  candidate_ids = Placeholder(name='candidate_ids')
  context_attention_weight = project_cosine_sim(status, context_memory, hidden_dim)
  query_attention_weight = project_cosine_sim(status, query_memory, hidden_dim)
  context_attention = reduce_sum(element_times(context_attention_weight, context_memory), axis = 0)
  query_attention = reduce_sum(element_times(query_attention_weight, query_memory), axis = 0)
  attention = splice((query_attention, context_attention))
  gru = gru_cell((hidden_dim, ), name='status')
  new_status = gru(attention, status).output
  termination_prob = termination_gate(new_status, dim=hidden_dim, name='prob')
  ans_attention = project_cosine_sim(new_status, candidate_memory, hidden_dim)
  answers = times(ans_attention, candidate_ids, name='ans2')
  return combine([answers, termination_prob, new_status], name='ReinforcementAttention')

def set_dynamic_axes(dynamic_axes, shape):
  src = placeholder_variable(shape=shape, dynamic_axes = dynamic_axes)
  outp = placeholder_variable(shape=shape, dynamic_axes = dynamic_axes)

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
  candidate_filter = greater(candidate_indicates, 0)
  candidate_ids = sequence.gather(context_raw, candidate_filter)

  # Query sequences
  query_sequence = query_raw
  # Doc/Context sequences
  context_sequence = context_raw
  # embedding
  embed_dim = hidden_dim
  embedding = parameter(shape=(vocab_dim, embed_dim), init=uniform(1))

  query_embedding  = times(query_sequence , embedding)
  context_embedding = times(context_sequence, embedding)

  # get source and context representations
  context_memory   = bidirectional_gru(hidden_dim, context_embedding, name='Context_Mem')            # shape=(hidden_dim*2, *), *=context_seq_axis
  candidate_memory = sequence.gather(context_memory, candidate_filter, name='Candidate_Mem')
  
  qfwd, qbwd  = bidirectional_gru(hidden_dim, query_embedding, splice_outputs=False) # shape=(hidden_dim*2, *), *=query_seq_axis
  query_memory = splice((qfwd, qbwd), name='Query_SP')
  # get the source (aka 'query') representation
  status = splice((sequence.last(qfwd), sequence.first(qbwd)), name='Init_Status') # get last fwd status and first bwd status
  attention_rlu = attention_rlunit(context_memory, query_memory, context_memory.output, context_sequence, hidden_dim*2, vocab_dim, init)
  # TODO: the candidate_memory and candidate_ids are not in the same dynamic axes
  #attention_rlu = attention_rlunit(context_memory, query_memory, candidate_memory.output, candidate_ids.output, hidden_dim*2, vocab_dim, init)
  arlus = attention_rlu(status)
  return combine(arlus.outputs, name='ReasoNet')

def create_reader(path, vocab_dim, randomize, size=INFINITELY_REPEAT):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        context  = StreamDef(field='C', shape=vocab_dim, is_sparse=True),
        query    = StreamDef(field='Q', shape=vocab_dim, is_sparse=True),
        entities  = StreamDef(field='E', shape=1, is_sparse=False),
        label   = StreamDef(field='L', shape=1, is_sparse=False)
    )), randomize=randomize, epoch_size = size)
