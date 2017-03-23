"""
ReasoNet model in CNTK
@author penhe@microsoft.com
"""
import sys
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, DEFAULT_RANDOMIZATION_WINDOW
import cntk.ops as ops
from cntk.layers.blocks import _INFERRED, Parameter
#import cntk.utils as utils
from cntk.internal import _as_tuple, sanitize_input
import cntk.learner as learner
#import cntk.io as io
#import cntk.cntk_py as cntk_py
from .utils import *
from .layers import *

def create_reader(path, vocab_dim, entity_dim, randomize, rand_size= DEFAULT_RANDOMIZATION_WINDOW, size=INFINITELY_REPEAT):
  """
  Create data reader for the model
  Args:
    path: The data path
    vocab_dim: The dimention of the vocabulary
    entity_dim: The dimention of entities
    randomize: Where to shuffle the data before feed into the trainer
  """
  return MinibatchSource(CTFDeserializer(path, StreamDefs(
    context  = StreamDef(field='C', shape=vocab_dim, is_sparse=True),
    query    = StreamDef(field='Q', shape=vocab_dim, is_sparse=True),
    entities  = StreamDef(field='E', shape=1, is_sparse=False),
    label   = StreamDef(field='L', shape=1, is_sparse=False),
    entity_ids   = StreamDef(field='EID', shape=entity_dim, is_sparse=True)
    )), randomize=randomize)

def attention_model(context_memory, query_memory, init_status, hidden_dim, att_dim, max_steps = 5, init = glorot_uniform()):
  """
  Create the attention model for reasonet
  Args:
    context_memory: Context memory
    query_memory: Query memory
    init_status: Intialize status
    hidden_dim: The dimention of hidden state
    att_dim: The dimention of attention
    max_step: Maxuim number of step to revisit the context memory
  """
  gru = gru_cell((hidden_dim*2, ), name='control_status')
  status = init_status
  output = [None]*max_steps*2
  sum_prob = None
  context_cos_sim = project_cosine_sim(att_dim, name='context_attention')
  query_cos_sim = project_cosine_sim(att_dim, name='query_attention')
  ans_cos_sim = project_cosine_sim(att_dim, name='candidate_attention')
  stop_gate = termination_gate(name='terminate_prob')
  prev_stop = 0
  for step in range(max_steps):
    context_attention_weight = context_cos_sim(status, context_memory)
    query_attention_weight = query_cos_sim(status, query_memory)
    context_attention = sequence.reduce_sum(times(context_attention_weight, context_memory), name='C-Att')
    query_attention = sequence.reduce_sum(times(query_attention_weight, query_memory), name='Q-Att')
    attention = ops.splice(query_attention, context_attention, name='att-sp')
    status = gru(attention, status).output
    termination_prob = stop_gate(status)
    ans_attention = ans_cos_sim(status, context_memory)
    output[step*2] = ans_attention
    if step < max_steps -1:
      stop_prob = prev_stop + ops.log(termination_prob, name='log_stop')
    else:
      stop_prob = prev_stop
    output[step*2+1] = sequence.broadcast_as(ops.exp(stop_prob, name='exp_log_stop'), output[step*2], name='Stop_{0}'.format(step))
    prev_stop += ops.log(1-termination_prob, name='log_non_stop')

  final_ans = None
  for step in range(max_steps):
    if final_ans is None:
      final_ans = output[step*2] * output[step*2+1]
    else:
      final_ans += output[step*2] * output[step*2+1]
  combine_func = combine(output + [ final_ans ], name='Attention_func')
  return combine_func

class model_params:
  def __init__(self, vocab_dim, entity_dim, hidden_dim, embedding_dim=100, embedding_init=None, share_rnn_param=False, max_rl_steps=5, dropout_rate=None, init=glorot_uniform(), model_name='rsn'):
    self.vocab_dim = vocab_dim
    self.entity_dim = entity_dim
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.embedding_init = embedding_init
    self.max_rl_steps = max_rl_steps
    self.dropout_rate = dropout_rate
    self.init = init
    self.model_name = model_name
    self.share_rnn_param = share_rnn_param
    self.attention_dim = 384

def bind_data(func, data):
  """
  Bind data outputs to cntk function arguments based on the argument name
  """
  bind = {}
  for arg in func.arguments:
    if arg.name == 'query':
      bind[arg] = data.streams.query
    if arg.name == 'context':
      bind[arg] = data.streams.context
    if arg.name == 'entity_ids_mask':
      bind[arg] = data.streams.entities
    if arg.name == 'labels':
      bind[arg] = data.streams.label
    if arg.name == 'entity_ids':
      bind[arg] = data.streams.entity_ids
  return bind

def create_model(params : model_params):
  """
  Create ReasoNet model
  Args:
    params (class:`model_params`): The parameters used to create the model
  """
  logger.log("Create model: dropout_rate: {0}, init:{1}, embedding_init: {2}".format(params.dropout_rate, params.init, params.embedding_init))
  # Query and Doc/Context/Paragraph inputs to the model
  query_seq_axis = Axis('sourceAxis')
  context_seq_axis = Axis('contextAxis')
  query_sequence = sequence.input(shape=(params.vocab_dim), is_sparse=True, sequence_axis=query_seq_axis, name='query')
  context_sequence = sequence.input(shape=(params.vocab_dim), is_sparse=True, sequence_axis=context_seq_axis, name='context')
  entity_ids_mask = sequence.input(shape=(1,), is_sparse=False, sequence_axis=context_seq_axis, name='entity_ids_mask')
  # embedding
  if params.embedding_init is None:
    embedding_init = create_random_matrix(params.vocab_dim, params.embedding_dim)
  else:
    embedding_init = params.embedding_init
  embedding = parameter(shape=(params.vocab_dim, params.embedding_dim), init=None)
  embedding.value = embedding_init
  embedding_matrix = constant(embedding_init, shape=(params.vocab_dim, params.embedding_dim))

  if params.dropout_rate is not None:
    query_embedding  = ops.dropout(times(query_sequence , embedding), params.dropout_rate, name='query_embedding')
    context_embedding = ops.dropout(times(context_sequence, embedding), params.dropout_rate, name='context_embedding')
  else:
    query_embedding  = times(query_sequence , embedding, name='query_embedding')
    context_embedding = times(context_sequence, embedding, name='context_embedding')
  
  contextGruW = Parameter(_INFERRED +  _as_tuple(params.hidden_dim), init=glorot_uniform(), name='gru_params')
  queryGruW = Parameter(_INFERRED +  _as_tuple(params.hidden_dim), init=glorot_uniform(), name='gru_params')

  entity_embedding = ops.times(context_sequence, embedding_matrix, name='constant_entity_embedding')
  # Unlike other words in the context, we keep the entity vectors fixed as a random vector so that each vector just means an identifier of different entities in the context and it has no semantic meaning
  full_context_embedding = ops.element_select(entity_ids_mask, entity_embedding, context_embedding)
  context_memory = ops.optimized_rnnstack(full_context_embedding, contextGruW, params.hidden_dim, 1, True, recurrent_op='gru', name='context_mem')

  query_memory = ops.optimized_rnnstack(query_embedding, queryGruW, params.hidden_dim, 1, True, recurrent_op='gru', name='query_mem')
  qfwd = ops.slice(sequence.last(query_memory), -1, 0, params.hidden_dim, name='fwd')
  qbwd = ops.slice(sequence.first(query_memory), -1, params.hidden_dim, params.hidden_dim*2, name='bwd')
  init_status = ops.splice(qfwd, qbwd, name='Init_Status') # get last fwd status and first bwd status
  return attention_model(context_memory, query_memory, init_status, params.hidden_dim, params.attention_dim, max_steps = params.max_rl_steps)

def accuracy_func(prediction, label, name='accuracy'):
  """
  Compute the accuracy of the prediction
  """
  pred_max = ops.hardmax(prediction, name='pred_max')
  norm_label = ops.equal(label, [1], name='norm_label')
  acc = ops.times_transpose(pred_max, norm_label, name='accuracy')
  return acc

def contractive_reward(labels, predictions_and_stop_probabilities):
  """
  Compute the contractive reward loss in paper 'ReasoNet: Learning to Stop Reading in Machine Comprehension'
  Args:
    labels: The lables
    predictions_and_stop_probabilities: A list of tuples, each tuple contains the prediction and stop probability of the coresponding step.
  """
  base = None
  avg_rewards = None
  for step in range(len(predictions_and_stop_probabilities)):
    pred = predictions_and_stop_probabilities[step][0]
    stop = predictions_and_stop_probabilities[step][1]
    if base is None:
      base = ops.element_times(pred, stop)
    else:
      base = ops.plus(ops.element_times(pred, stop), base)
  avg_rewards = ops.stop_gradient(sequence.reduce_sum(base*labels))
  base_reward = sequence.broadcast_as(avg_rewards, base, name = 'base_line')
  # While  the learner will mimize the loss by default, we want it to maxiumize the rewards
  # Maxium rewards => minimal -rewards
  # So we use (1-r/b) as the rewards instead of (r/b-1)
  step_cr = ops.stop_gradient(1- ops.element_divide(labels, base_reward))
  normalized_contractive_rewards = ops.element_times(base, step_cr)
  rewards = sequence.reduce_sum(normalized_contractive_rewards) + avg_rewards
  return rewards

def loss(model, params:model_params):
  """
  Compute the loss and accuracy of the model output
  """
  model_args = {arg.name:arg for arg in model.arguments}
  context = model_args['context']
  entity_ids_mask = model_args['entity_ids_mask']
  entity_condition = greater(entity_ids_mask, 0, name='condidion')
  entities_all = sequence.gather(entity_condition, entity_condition, name='entities_all')
  entity_ids = input(shape=(params.entity_dim), is_sparse=True, dynamic_axes=entities_all.dynamic_axes, name='entity_ids')
  wordvocab_dim = params.vocab_dim
  labels_raw = input(shape=(1,), is_sparse=False, dynamic_axes=context.dynamic_axes, name='labels')
  answers = sequence.scatter(sequence.gather(model.outputs[-1], entity_condition), entities_all, name='Final_Ans')
  labels = sequence.scatter(sequence.gather(labels_raw, entity_condition), entities_all, name='EntityLabels')
  entity_id_matrix = ops.reshape(entity_ids, params.entity_dim)
  expand_pred = sequence.reduce_sum(element_times(answers, entity_id_matrix))
  expand_label = ops.greater_equal(sequence.reduce_sum(element_times(labels, entity_id_matrix)), 1)
  expand_candidate_mask = ops.greater_equal(sequence.reduce_sum(entity_id_matrix), 1)
  predictions_and_stop_probabilities=[]
  for step in range(int((len(model.outputs)-1)/2)):
    predictions_and_stop_probabilities += [(model.outputs[step*2], model.outputs[step*2+1])]
  loss_value = contractive_reward(labels_raw, predictions_and_stop_probabilities)
  accuracy = accuracy_func(expand_pred, expand_label, name='accuracy')
  apply_loss = combine([loss_value, answers, labels, accuracy], name='Loss')
  return apply_loss

def create_adam_learner(learn_params, learning_rate = 0.0005, gradient_clipping_threshold_per_sample=0.001):
  """
  Create adam learner
  """
  lr_schedule = learner.learning_rate_schedule(learning_rate, learner.UnitType.sample)
  momentum = learner.momentum_schedule(0.90)
  gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
  gradient_clipping_with_truncation = True
  momentum_var = learner.momentum_schedule(0.999)
  lr = learner.adam_sgd(learn_params, lr_schedule, momentum, True, momentum_var,
          low_memory = False,
          gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
          gradient_clipping_with_truncation = gradient_clipping_with_truncation)
  learner_desc = 'Alg: Adam, learning rage: {0}, momentum: {1}, gradient clip: {2}'.format(learning_rate, momentum[0], gradient_clipping_threshold_per_sample)
  logger.log("Create learner. {0}".format(learner_desc))
  return lr

def __evaluation(trainer, data, bind, minibatch_size, epoch_size):
  """
  Evaluate the loss and accurate of the evaluation data set during training stage
  """
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
    mbs = min(epoch_size - k, minibatch_size)
    mb = data.next_minibatch(mbs, input_map=bind)
    k += mb[label_arg].num_samples
    sm = mb[label_arg].num_sequences
    avg_acc = trainer.test_minibatch(mb)
    eval_acc += sm*avg_acc
    eval_s += sm
    sys.stdout.write('.')
    sys.stdout.flush()
  eval_acc /= eval_s
  print("")
  logger.log("Evaluation Acc: {0}, samples: {1}".format(eval_acc, eval_s))
  return eval_acc

def train(model, m_params:model_params, learner, train_data, max_epochs=1, save_model_flag=False, epoch_size=270000, eval_data=None, eval_size=None, check_point_freq=0.1, minibatch_size=50000, model_name='rsn'):
  """
  Train the model
  Args:
    model: The created model
    m_params: Model parameters
    learner: The learner used to train the model
  """
  criterion_loss = loss(model, m_params)
  loss_func = criterion_loss.outputs[0]
  eval_func = criterion_loss.outputs[-1]
  trainer = Trainer(model.outputs[-1], (loss_func, eval_func), learner)
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
  training_progress_output_freq = 500
  check_point_interval = int(epoch_size*check_point_freq)
  check_point_id = 0
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
      mbs = min(minibatch_size, epoch_size - i)
      mb_train = train_data.next_minibatch(minibatch_size, input_map=train_bind)
      i += mb_train[label_key].num_samples
      trainer.train_minibatch(mb_train)
      minibatch_count += 1
      sys.stdout.write('.')
      sys.stdout.flush()
      # collect epoch-wide stats
      samples = trainer.previous_minibatch_sample_count
      ls = trainer.previous_minibatch_loss_average * samples
      acc = trainer.previous_minibatch_evaluation_average * samples
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
        logger.log("Lastest sample count = {}, Train Loss: {}, Evalualtion ACC: {}".format(win_samples, win_loss/win_samples,
          win_acc/win_samples))
        logger.log("Total sample count = {}, Train Loss: {}, Evalualtion ACC: {}".format(chk_samples, chk_loss/chk_samples,
          chk_acc/chk_samples))
        win_samples = 0
        win_loss = 0
        win_acc = 0
      new_chk_id = int(i/check_point_interval)
      if new_chk_id != check_point_id and i < epoch_size :
        check_point_id = new_chk_id
        print('')
        logger.log("--- CHECKPOINT %d: samples=%d, loss = %.2f, acc = %.2f%% ---" % (check_point_id, chk_samples, chk_loss/chk_samples, 100.0*(chk_acc/chk_samples)))
        if eval_data:
          __evaluation(trainer, eval_data, eval_bind, minibatch_size, eval_size)
        if save_model_flag:
          # save the model every epoch
          model_filename = os.path.join('model', "model_%s_%03d.dnn" % (model_name, check_point_id))
          model.save_model(model_filename)
          logger.log("Saved model to '%s'" % model_filename)
        chk_samples = 0
        chk_loss = 0
        chk_acc = 0

    print('')
    logger.log("--- EPOCH %d: samples=%d, loss = %.2f, acc = %.2f%% ---" % (epoch, epoch_samples, epoch_loss/epoch_samples, 100.0*(epoch_acc/epoch_samples)))
  eval_acc = 0
  if eval_data:
    eval_acc = __evaluation(trainer, eval_data, eval_bind, minibatch_size, eval_size)
  if save_model_flag:
    # save the model every epoch
    model_filename = os.path.join('model', "model_%s_final.dnn" % (model_name))
    model.save_model(model_filename)
    logger.log("Saved model to '%s'" % model_filename)
  return (epoch_loss/epoch_samples, epoch_acc/epoch_samples, eval_acc)
