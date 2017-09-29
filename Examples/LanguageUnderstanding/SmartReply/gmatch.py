import cntk as C
import cntk.layers as cl
import time
import numpy as np
from common import options, encoder_type

def OptimizedRnnStack(hidden_dim, num_layers=1, recurrent_op='lstm', bidirectional=False, name=''):
    w = C.parameter((C.InferredDimension,hidden_dim), init=C.glorot_uniform())
    def func(x):
        return C.optimized_rnnstack(x, w, hidden_dim, num_layers, bidirectional, recurrent_op=recurrent_op, name=name)
    return func

def SummarizeRnnStack(hidden_dim, name=''):
    def func(x):
        first = C.sequence.first(x)
        last = C.sequence.last(x)
        combined = C.splice(first, last)
        return C.slice(combined, axis=0, begin_index=hidden_dim, end_index=3*hidden_dim, name=name)
    return func

def encoder_lstm(hidden_dims, column='x'):
    with C.default_options(init=C.normal(0.1)):
        return cl.Sequential([cl.Dropout(0.2),
                              OptimizedRnnStack(hidden_dims[0], bidirectional=True, name='LSTM_1_' + column),
                              SummarizeRnnStack(hidden_dims[0], name='Summary_' + column),
                              cl.Dropout(0.2)])


def encoder_ff(hidden_dims, column='x'):
    with C.default_options(init=C.normal(0.1)):
        return cl.Sequential([cl.Dropout(0.2),
                              cl.Dense(hidden_dims[0], activation=C.tanh, name='FF_1_' + column),
                              cl.Dropout(0.2),
                              cl.Dense(hidden_dims[1], activation=C.tanh, name='FF_2_' + column),
                              cl.Dropout(0.2),
                              cl.Dense(hidden_dims[2], activation=None,   name='FF_3_' + column),
                              cl.Dropout(0.2)])


def all_pairs_loss(message_hidden, reply_hidden):
    msg_matrix = C.unpack_batch(message_hidden)
    rep_matrix = C.unpack_batch(reply_hidden)
    all_inner_products = C.to_batch(C.times_transpose(rep_matrix, msg_matrix))

    positive_inner_products = C.reduce_sum(message_hidden * reply_hidden, axis=0)
    loss = C.reduce_log_sum_exp(all_inner_products) - positive_inner_products
    return loss


def create_model(emb_dim, hidden_dims, encoder_type='LSTM'):
    message = C.placeholder()
    reply = C.placeholder()
    with C.default_options(init=C.normal(0.1)):
        emb = cl.Embedding(emb_dim , name='Embedding')
    if encoder_type == 'LSTM':
        enc_model_x = encoder_lstm(hidden_dims, column='x')
        enc_model_y = encoder_lstm(hidden_dims, column='y')
    elif encoder_type == 'FF':
        enc_model_x = encoder_ff(hidden_dims, column='x')
        enc_model_y = encoder_ff(hidden_dims, column='y')
    else:
        raise ValueError('unknown encoder')

    enc_x= enc_model_x(emb(message))
    enc_y= enc_model_y(emb(reply))
    return C.combine((enc_x, enc_y), name='combined_output')


def train_loop(train_reader, model, criterion, learner):
    print('Train Loop Starting')
    max_epochs = options['max_epochs']
    minibatch_size = options['minibatch_size']
    epoch_size = options['epoch_size']

    total_samples=0
    progress_printer = C.logging.progress_print.ProgressPrinter(freq=0, tag='Training')
    if options['num_workers'] == 1:
        partition = 0
        num_partitions = 1
    else:
        partition = C.distributed.Communicator.rank()
        num_partitions = C.distributed.Communicator.num_workers()
        learner = C.distributed.data_parallel_distributed_learner(learner)
    trainer = C.Trainer(model, (criterion,criterion), learner)
    msg, reply = criterion.arguments
    input_map = {msg: train_reader.streams.msg, reply: train_reader.streams.reply}

    mb_train = train_reader.next_minibatch(minibatch_size * num_partitions, input_map=input_map,
                                           num_data_partitions=num_partitions, partition_index=partition)
    total_samples = mb_train[reply].num_sequences
    checkpoint = 1
    while mb_train:
        if total_samples > checkpoint * epoch_size:
            model.save("./model.%s.%.2d.cnt"%(encoder_type,checkpoint))
            checkpoint += 1

        trainer.train_minibatch(mb_train)
        progress_printer.update_with_trainer(trainer, with_metric=True)
        total_samples += mb_train[reply].num_sequences
        mb_train = train_reader.next_minibatch(minibatch_size * num_partitions, input_map=input_map,
                                               num_data_partitions=num_partitions, partition_index=partition)
    model.save("./model.%s.%.2d.cnt" % (encoder_type, checkpoint))
    C.distributed.Communicator.finalize()


if __name__ == '__main__':
    print(C.__version__)
    if encoder_type == 'LSTM':
        msg_features   = C.sequence.input_variable(shape=options['vocab_size'], sequence_axis=C.Axis('M'), is_sparse=True)
        reply_features = C.sequence.input_variable(shape=options['vocab_size'], sequence_axis=C.Axis('R'), is_sparse=True)
    elif encoder_type == 'FF':
        msg_features   = C.input_variable(shape=options['vocab_size'], is_sparse=True)
        reply_features = C.input_variable(shape=options['vocab_size'], is_sparse=True)
    else:
        raise ValueError('unknown encoder')

    z = create_model(options['emb_dim'], options['hidden'], encoder_type)
    z.replace_placeholders({z.placeholders[0]: msg_features, z.placeholders[1]: reply_features})

    loss = all_pairs_loss(z.outputs[0], z.outputs[1])

    learner = C.adadelta(z.parameters, gradient_clipping_threshold_per_sample=0.05, gradient_clipping_with_truncation=True)

    train_reader = C.io.MinibatchSource(C.io.CTFDeserializer(options['train'], C.io.StreamDefs(
        msg  =C.io.StreamDef(field='c', shape=options['vocab_size'], is_sparse=True),
        reply=C.io.StreamDef(field='a', shape=options['vocab_size'], is_sparse=True))), randomize=True, max_sweeps=options['max_epochs'])
    s = time.time()
    train_loop(train_reader, z, loss, learner)
    t = time.time()
    print(t-s)

