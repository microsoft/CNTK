# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer, Axis, save_model, load_model
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.device import cpu, set_default_device
from cntk.learner import momentum_sgd, momentum_schedule
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, slice, past_value, future_value, element_select, alias, hardmax
from cntk.ops.functions import CloneMethod
from cntk.graph import find_nodes_by_name

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
sys.path.append(os.path.join(abs_path, "..", "..", "bindings", "python"))
from examples.common.nn import LSTMP_component_with_self_stabilization, stabilize, linear_layer, print_training_progress

########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."    # data resides in the CNTK folder
data_dir = cntk_dir + "/Examples/SequenceToSequence/CMUDict/Data/"  # under Examples/SequenceToSequence
model_dir = "./Models"
input_vocab_size = 69
label_vocab_size = 69

# model dimensions
input_vocab_dim  = input_vocab_size
label_vocab_dim  = label_vocab_size
hidden_dim = 256
num_layers = 1

########################
# define the reader    #
########################

def create_reader(path, randomize, size=INFINITELY_REPEAT):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_dim,  is_sparse=True)
    )), randomize=randomize, epoch_size = size)

########################
# define the model     #
########################

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
    label_sequence = slice(raw_labels, label_seq_axis, 
                           1, 0, name='label_sequence') # <s> A B C </s> --> A B C </s>
    label_sentence_start = sequence.first(raw_labels)   # <s>

    # Setup primer for decoder
    is_first_label = sequence.is_first(label_sequence)  # 1 0 0 0 ...
    label_sentence_start_scattered = sequence.scatter(
        label_sentence_start, is_first_label)

    # Encoder
    encoder_outputH = stabilize(input_sequence)
    for i in range(0, num_layers):
        (encoder_outputH, encoder_outputC) = LSTMP_component_with_self_stabilization(
            encoder_outputH.output, hidden_dim, hidden_dim, future_value, future_value)

    # Prepare encoder output to be used in decoder
    thought_vectorH = sequence.first(encoder_outputH)
    thought_vectorC = sequence.first(encoder_outputC)

    thought_vector_broadcastH = sequence.broadcast_as(
        thought_vectorH, label_sequence)
    thought_vector_broadcastC = sequence.broadcast_as(
        thought_vectorC, label_sequence)

    # Decoder
    decoder_history_hook = alias(label_sequence, name='decoder_history_hook') # copy label_sequence

    decoder_input = element_select(is_first_label, label_sentence_start_scattered, past_value(
        decoder_history_hook))

    decoder_outputH = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0):
            recurrence_hookH = past_value
            recurrence_hookC = past_value
        else:
            isFirst = sequence.is_first(label_sequence)
            recurrence_hookH = lambda operand: element_select(
                isFirst, thought_vector_broadcastH, past_value(operand))
            recurrence_hookC = lambda operand: element_select(
                isFirst, thought_vector_broadcastC, past_value(operand))

        (decoder_outputH, encoder_outputC) = LSTMP_component_with_self_stabilization(
            decoder_outputH.output, hidden_dim, hidden_dim, recurrence_hookH, recurrence_hookC)

    decoder_output = decoder_outputH

    # Softmax output layer
    z = linear_layer(stabilize(decoder_output), label_vocab_dim)
    
    return z

########################
# train action         #
########################

def train(train_reader, valid_reader, vocab, i2w, model, max_epochs):
    
    # do some hooks that we won't need in the future
    raw_labels = find_arg_by_name('raw_labels', model)    
    
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
    lr = 0.007
    minibatch_size = 72
    momentum_time_constant = 1100
    m_schedule = momentum_schedule(momentum_time_constant)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(model.parameters, lr, m_schedule, clipping_threshold_per_sample, gradient_clipping_with_truncation)
    trainer = Trainer(model, ce, errs, learner)

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0
    epoch_size = 908241
    training_progress_output_freq = 500

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

        while i < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size, input_map=train_bind)
            trainer.train_minibatch(mb_train)

            # collect epoch-wide stats
            samples = trainer.previous_minibatch_sample_count
            loss_numer += trainer.previous_minibatch_loss_average * samples
            metric_numer += trainer.previous_minibatch_evaluation_average * samples
            denom += samples

            # every N MBs evaluate on a test sequence to visually show how we're doing
            if mbs % training_progress_output_freq == 0:
                mb_valid = valid_reader.next_minibatch(minibatch_size, input_map=valid_bind)
                e = new_model.eval(mb_valid)
                print_sequences(e, i2w)

            print_training_progress(trainer, mbs, training_progress_output_freq)
            i += mb_train[raw_labels].num_samples
            mbs += 1

        print("--- EPOCH %d DONE: loss = %f, errs = %f ---" % (epoch, loss_numer/denom, 100.0*(metric_numer/denom)))

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


#############################
# main function boilerplate #
#############################
    
if __name__ == '__main__':

    # hook up data
    train_reader = create_reader(data_dir + "cmudict-0.7b.train-dev-20-21.ctf", True)
    valid_reader = create_reader("tiny.ctf", False)
    vocab, i2w = get_vocab(data_dir + "cmudict-0.7b.mapping")

    # create model
    model = create_model()
    
    # train
    train(train_reader, valid_reader, vocab, i2w, model, max_epochs=10)
