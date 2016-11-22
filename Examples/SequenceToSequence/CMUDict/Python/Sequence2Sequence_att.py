# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import argparse
import time
import cntk as C

from cntk import Trainer, Axis, save_model, load_model, distributed, persist
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, INFINITE_SAMPLES
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, slice, past_value, future_value, \
                     element_select, alias, hardmax, placeholder_variable, combine, parameter, plus, times, reduce_log_sum, exp, \
                     splice, constant, reshape, element_times, log, tanh
from cntk.ops.functions import CloneMethod
from cntk.graph import find_nodes_by_name

from localblocks import LSTM, Stabilizer

from cntk.initializer import glorot_uniform
from cntk.utils import get_train_eval_criterion, get_train_loss


########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../../"    # data resides in the CNTK folder
data_dir = cntk_dir + "Examples/SequenceToSequence/CMUDict/Data/"  # under Examples/SequenceToSequence
model_dir = "./Models"
input_vocab_size = 69
label_vocab_size = 69

# model dimensions
input_vocab_dim  = input_vocab_size
label_vocab_dim  = label_vocab_size
hidden_dim = 128
num_layers = 1
attention_dim = 128
attention_span = 20
use_attention = True

# stabilizer
stabilize = Stabilizer()

########################
# define the reader    #
########################

def create_reader(path, randomize, size=INFINITELY_REPEAT, distributed_after=INFINITE_SAMPLES):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
    )), randomize=randomize, epoch_size = size, distributed_after = distributed_after)


#########################################
# attention -- should go somewhere else #
#########################################

# create a past value window that returns two records: a value, shape=(N,dim), and a valid window, shape=(1,dim)
def past_value_window(N, input, axis=0):

    ones_like_input = plus(times(input, constant(0, shape=(input.shape[0],1))), constant(1, shape=(1)))
        
    last_value=[]
    last_valid=[]
    value = None
    valid = None

    for t in range(N):
        if t == 0:
            value = input
            valid = ones_like_input
        else:
            value = past_value(input, time_step=t)
            valid = past_value(ones_like_input, time_step=t)     
        
        last_value.append(sequence.last(value))
        last_valid.append(sequence.last(valid))

    # can't get splice to stack rows 'beside' each other, so stack on top and then reshape...
    value_a = splice(last_value, axis=axis)
    valid_a = splice(last_valid, axis=axis)

    # now reshape
    value = reshape(value_a, shape=(N, input.shape[0]))
    valid = reshape(valid_a, shape=(N, 1))

    # value[t] = value of t steps in the past; valid[t] = true if there was a value t steps in the past
    return value, valid

def my_softmax(z, axis):
    Z = reduce_log_sum(z, axis=axis) # reduce along axis
    P = exp(z - Z)
    
    return P

def create_attention_augment_hook(attention_dim, attention_span, decoder_dynamic_axis, encoder_outputH):

    # useful var
    encoder_output_dim = encoder_outputH.shape[0]

    # create the attention window
    (aw_value, aw_valid) = past_value_window(attention_span, encoder_outputH, axis=0)

    # setup the projection of the attention window to go into the tanh()
    def projected_attention_window_broadcast():
        W = parameter(shape=(attention_dim, encoder_output_dim), init=glorot_uniform())

        projected_value = sequence.broadcast_as(times(stabilize(element_times(aw_value, aw_valid)), W), 
                                                          decoder_dynamic_axis)
        value           = sequence.broadcast_as(aw_value, decoder_dynamic_axis)
        valid           = sequence.broadcast_as(aw_valid, decoder_dynamic_axis)

        # should be shape=(attention_span, attention_dim)
        return projected_value, value, valid

    # the function that gets passed to the LSTM function as the augment_input_hook parameter
    def augment_input_hook(input, prev_state):
        output_dim = prev_state.shape[0]
        W = parameter(shape=(output_dim, attention_dim), init=glorot_uniform())

        projectedH = times(stabilize(prev_state), W, output_rank=1)

        tanh_out = tanh(projectedH + projected_attention_window_broadcast()[0])  # (attention_span, attention_dim)
        
        # u = v * tanh(W1h + W2d)
        v = parameter(shape=(attention_dim, 1))
               
        u = times(stabilize(element_times(tanh_out, projected_attention_window_broadcast()[2])), v) # (attention_span, 1)
        u_valid = plus(u, log(projected_attention_window_broadcast()[2]), name='u_valid')

        attention_weights = alias(my_softmax(u_valid, 0), name='attention_weights')
        # the window should be shape=(attention_span, output_dim)
        weighted_attention_window = element_times(projected_attention_window_broadcast()[1], attention_weights, 
                                                  name='weighted_attention_window')

        ones = constant(value=1, shape=(attention_span))
        # weighted_attention_avg should be shape=(output_dim)
        weighted_attention_avg = times(ones, stabilize(weighted_attention_window), output_rank=1, 
                                       name='weighted_attention_avg')

        return weighted_attention_avg

    return augment_input_hook


########################
# define the model     #
########################

def LSTM_layer(input, output_dim, recurrence_hook_h=past_value, recurrence_hook_c=past_value, augment_input_hook=None):
    dh = placeholder_variable(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    dc = placeholder_variable(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    
    aux_input = None
    has_aux   = False
    if augment_input_hook != None:
        aux_input = augment_input_hook(input, dh)
        has_aux = True
       
    LSTM_cell = LSTM(output_dim, has_aux=has_aux)
    if has_aux:    
        f_x_h_c = LSTM_cell(input, (dh, dc), aux_input)
    else:
        f_x_h_c = LSTM_cell(input, (dh, dc))
    h_c = f_x_h_c.outputs
    
    h = recurrence_hook_h(h_c[0])
    c = recurrence_hook_c(h_c[1])

    replacements = { dh: h.output, dc: c.output }
    f_x_h_c.replace_placeholders(replacements)

    h = f_x_h_c.outputs[0]
    c = f_x_h_c.outputs[1]

    return combine([h]), combine([c])

def create_model():
    
    # Source and target inputs to the model
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    raw_input = input_variable(
        shape=(input_vocab_dim), dynamic_axes=input_dynamic_axes, name='raw_input')

    label_dynamic_axes = [batch_axis, label_seq_axis]
    raw_labels = input_variable(
        shape=(label_vocab_dim), dynamic_axes=label_dynamic_axes, name='raw_labels')

    # Instantiate the sequence to sequence translation model
    input_sequence = raw_input

    # Drop the sentence start token from the label, for decoder training
    label_sequence = sequence.slice(raw_labels, 1, 0, name='label_sequence') # <s> A B C </s> --> A B C </s>
    label_sentence_start = sequence.first(raw_labels)   # <s>

    # Setup primer for decoder
    is_first_label = sequence.is_first(label_sequence)  # 1 0 0 0 ...
    label_sentence_start_scattered = sequence.scatter(
        label_sentence_start, is_first_label)

    # Encoder
    encoder_output_h = stabilize(input_sequence)
    for i in range(0, num_layers):
        (encoder_output_h, encoder_output_c) = LSTM_layer(
            encoder_output_h.output, hidden_dim, future_value, future_value)

    # Prepare encoder output to be used in decoder
    thought_vector_h = sequence.first(encoder_output_h)
    thought_vector_c = sequence.first(encoder_output_c)

    thought_vector_broadcast_h = sequence.broadcast_as(
        thought_vector_h, label_sequence)
    thought_vector_broadcast_c = sequence.broadcast_as(
        thought_vector_c, label_sequence)

    # Decoder
    decoder_history_hook = alias(label_sequence, name='decoder_history_hook') # copy label_sequence

    decoder_input = element_select(is_first_label, label_sentence_start_scattered, past_value(
        decoder_history_hook))

    augment_input_hook = None
    if use_attention:
        augment_input_hook = create_attention_augment_hook(attention_dim, attention_span, 
                                                           label_sequence, encoder_output_h)

    decoder_output_h = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0) or use_attention:
            recurrence_hook_h = past_value
            recurrence_hook_c = past_value
        else:
            recurrence_hook_h = lambda operand: element_select(
                is_first_label, thought_vector_broadcast_h, past_value(operand))
            recurrence_hook_c = lambda operand: element_select(
                is_first_label, thought_vector_broadcast_c, past_value(operand))

        (decoder_output_h, decoder_output_c) = LSTM_layer(
            decoder_output_h.output, hidden_dim, recurrence_hook_h, recurrence_hook_c, 
            augment_input_hook)

    # Linear output layer
    W = parameter(shape=(decoder_output_h.shape[0], label_vocab_dim), init=glorot_uniform())
    B = parameter(shape=(label_vocab_dim), init=0)
    z = plus(B, times(stabilize(decoder_output_h), W))    
    
    return z

########################
# train action         #
########################

def train(distributed_trainer, train_reader, valid_reader, vocab, i2w, model, max_epochs):
    
    # do some hooks that we won't need in the future
    label_sequence = find_nodes_by_name(model, 'label_sequence')[0]    
    decoder_history_hook = find_nodes_by_name(model, 'decoder_history_hook')[0]  
        
    # Criterion nodes
    ce = cross_entropy_with_softmax(model, label_sequence)
    errs = classification_error(model, label_sequence)

    def clone_and_hook():
        # network output for decoder history
        net_output = hardmax(model)

        # make a clone of the graph where the ground truth is replaced by the network output
        return model.clone(CloneMethod.share, {decoder_history_hook.output : net_output.output})

    # get a new model that uses the past network output as input to the decoder
    new_model = clone_and_hook()

    # Instantiate the trainer object to drive the model training
    lr_per_sample = 0.007
    lr_schedule = learning_rate_schedule(lr_per_sample, UnitType.sample)
    minibatch_size = 72
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(model.parameters,
                           lr_schedule, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(model, ce, errs, learner, distributed_trainer)

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0
    epoch_size = 908241
    training_progress_output_freq = 100

    # bind inputs to data from readers
    train_bind = {
        find_arg_by_name('raw_input' , model) : train_reader.streams.features,
        find_arg_by_name('raw_labels', model) : train_reader.streams.labels
    }
    valid_bind = {
        find_arg_by_name('raw_input' , new_model) : valid_reader.streams.features,
        find_arg_by_name('raw_labels', new_model) : valid_reader.streams.labels
    }

    for epoch in range(max_epochs):
        loss_numer = 0
        metric_numer = 0
        denom = 0
        start_time = time.time()

        while i < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size, input_map=train_bind)
            import ipdb;ipdb.set_trace()  
            trainer.train_minibatch(mb_train)

            # collect epoch-wide stats
            samples = trainer.previous_minibatch_sample_count
            loss_numer += trainer.previous_minibatch_loss_average * samples
            metric_numer += trainer.previous_minibatch_evaluation_average * samples
            denom += samples

            # Print stats for the master
            # every N MBs evaluate on a test sequence to visually show how we're doing; also print training stats
            if distributed_trainer.communicator().current_worker().global_rank == 0:
               if mbs % training_progress_output_freq == 0:
                
                 print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(mbs, 
                      get_train_loss(trainer), get_train_eval_criterion(trainer)))
                
                 mb_valid = valid_reader.next_minibatch(minibatch_size, input_map=valid_bind)
                
                 e = new_model.eval(mb_valid)
                 print_sequences(e, i2w)

            i += mb_train[find_arg_by_name('raw_labels', model)].num_samples
            mbs += 1

        if distributed_trainer.communicator().current_worker().global_rank == 0:
            end_time = time.time()
            print("--- EPOCH %d DONE: loss = %f, errs = %f, time = %.2f ---" % (epoch, loss_numer/denom, 100.0*(metric_numer/denom), (end_time-start_time)))

            if save_model:
                # save the model every epoch
                model_filename = os.path.join(model_dir, "model_epoch%d.dnn" % epoch)
                save_model(new_model, model_filename)
                print("Saved model to '%s'" % model_filename)

########################
# helper functions     #
########################

def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
    vocab = [w.strip() for w in open(path).readlines()]
    i2w = { i:ch for i,ch in enumerate(vocab) }
    
    return (vocab, i2w)

# Given a vocab and tensor, print the output
def print_sequences(sequences, i2w):
    for s in sequences:
        print([i2w[np.argmax(w)] for w in s], sep=" ")

# helper function to find variables by name
# which is necessary when cloning or loading the model
def find_arg_by_name(name, expression):
    vars = [i for i in expression.arguments if i.name == name]
    assert len(vars) == 1
    return vars[0]



def write(reader, model_filename, vocab, i2w):
    
    model = load_model(model_filename)
    
    binder = {
        find_arg_by_name('raw_input' , model) : reader.streams.features,
        find_arg_by_name('raw_labels', model) : reader.streams.labels
    }

    for i in range(1):
        # get next minibatch of data
        mb = reader.next_minibatch(1, input_map=binder)
                
        e = model.eval(mb)
        print_sequences(e, i2w)

           



#############################
# main function boilerplate #
#############################
    
if __name__ == '__main__':

    C.set_default_device(best())

    # hook up data
    train_reader = create_reader(data_dir + "cmudict-0.7b.train-dev-1-21.ctf", False)
    valid_reader = create_reader(data_dir + "tiny.ctf", False)
    
    vocab, i2w = get_vocab(data_dir + "cmudict-0.7b.mapping")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quantize_bit', help='quantized bit', required=False, default='32')
    parser.add_argument('-b', '--block_size', help='block momentum block size, quantized bit would be ignored if this is set', required=False)
    parser.add_argument('-e', '--epochs', help='total epochs', required=False, default='20')
    parser.add_argument('-w', '--warm_start', help='number of samples to warm start before running distributed', required=False, default='10')
    args = vars(parser.parse_args())
    num_quantization_bits = int(args['quantize_bit'])
    epochs = int(args['epochs'])
    distributed_after_samples = int(args['warm_start'])
    
    if args['block_size']:
        block_size = int(args['block_size'])
        print("Start training:block_size = {}, epochs = {}, warm_start = {}".format(block_size, epochs, distributed_after_samples))
        distributed_trainer = distributed.block_momentum_distributed_trainer(
            block_size=block_size,
            distributed_after=distributed_after_samples)
    else:
        print("Start training: quantize_bit = {}, epochs = {}, warm_start = {}".format(num_quantization_bits, epochs, distributed_after_samples))
        distributed_trainer = distributed.data_parallel_distributed_trainer(
            num_quantization_bits=num_quantization_bits,
            distributed_after=distributed_after_samples) 
    
    # create model
    model = create_model()
    
    # train
    train(distributed_trainer, train_reader, valid_reader, vocab, i2w, model, max_epochs=10)
    distributed.Communicator.finalize()

    # write(valid_reader, "g2p_epoch0.dnn", vocab, i2w)
