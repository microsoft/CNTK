# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.device import cpu, set_default_device
from cntk.learner import learning_rate_schedule, UnitType, momentum_sgd, momentum_as_time_constant_schedule
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, past_value, future_value, element_select, alias, hardmax
from cntk.ops.functions import CloneMethod

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "common"))
from nn import LSTMP_component_with_self_stabilization, stabilize, linear_layer, print_training_progress

# Given a vocab and tensor, print the output
def print_sequences(sequences, i2w):
    for s in sequences:
        print([i2w[np.argmax(w)] for w in s], sep=" ")

def create_reader(path, randomize, input_vocab_dim, label_vocab_dim, size=INFINITELY_REPEAT):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_dim,  is_sparse=True)
    )), randomize=randomize, epoch_size = size)

# helper function to find variables by name
# which is necessary when cloning or loading the model
def find_arg_by_name(name, expression):
    vars = [i for i in expression.arguments if i.name == name]
    assert len(vars) == 1
    return vars[0]

# Average of evaluation errors of all test minibatches
def translator_test_error(z, trainer, input_vocab_dim, label_vocab_dim, debug_output=False):
    # now setup a test run
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "cmudict-0.7b.test.ctf")

    test_reader = create_reader(test_path, False, input_vocab_dim, label_vocab_dim, FULL_DATA_SWEEP)

    test_bind = {
        find_arg_by_name('raw_input',z)  : test_reader.streams.features,
        find_arg_by_name('raw_labels',z) : test_reader.streams.labels
    }

    test_minibatch_size = 1024

    # Get minibatches of sequences to test and perform testing
    i = 0
    total_error = 0.0
    while True:
        mb = test_reader.next_minibatch(test_minibatch_size, input_map=test_bind)
        if not mb: break
        mb_error = trainer.test_minibatch(mb)
        total_error += mb_error

        if debug_output:
            print("Minibatch {}, Error {} ".format(i, mb_error))
        i += 1
    return total_error/i


# Creates and trains a sequence to sequence translation model

def sequence_to_sequence_translator(debug_output=False, run_test=False):

    input_vocab_dim = 69
    label_vocab_dim = 69

    # network complexity; initially low for faster testing
    hidden_dim = 256
    num_layers = 1

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
    label_sequence = sequence.slice(raw_labels, 1, 0) # <s> A B C </s> --> A B C </s>
    label_sentence_start = sequence.first(raw_labels)        # <s>

    is_first_label = sequence.is_first(label_sequence)       # <s> 0 0 0 ...
    label_sentence_start_scattered = sequence.scatter(
        label_sentence_start, is_first_label)

    # Encoder
    encoder_outputH = stabilize(input_sequence)
    for i in range(0, num_layers):
        (encoder_outputH, encoder_outputC) = LSTMP_component_with_self_stabilization(
            encoder_outputH.output, hidden_dim, hidden_dim, future_value, future_value)

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

    # Criterion nodes
    ce = cross_entropy_with_softmax(z, label_sequence)
    errs = classification_error(z, label_sequence)

    # network output for decoder history
    net_output = hardmax(z)

    # make a clone of the graph where the ground truth is replaced by the network output
    ng = z.clone(CloneMethod.share, {decoder_history_hook.output : net_output.output})

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.5, UnitType.minibatch)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(z.parameters, 
                           lr_per_minibatch, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample, 
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(z, ce, errs, learner)

    # setup data
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "cmudict-0.7b.train-dev-20-21.ctf")
    valid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "tiny.ctf")

    # readers
    randomize_data = True
    if run_test:
        randomize_data = False # because we want to get an exact error

    train_reader = create_reader(train_path, randomize_data, input_vocab_dim, label_vocab_dim)
    train_bind = {
        raw_input  : train_reader.streams.features,
        raw_labels : train_reader.streams.labels
    }

    # get the vocab for printing output sequences in plaintext
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "cmudict-0.7b.mapping")
    vocab = [w.strip() for w in open(vocab_path).readlines()]
    i2w = { i:ch for i,ch in enumerate(vocab) }

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0
    minibatch_size = 72
    epoch_size = 908241
    max_epochs = 10
    training_progress_output_freq = 500

    # make things more basic for running a quicker test
    if run_test:
        epoch_size = 5000
        max_epochs = 1
        training_progress_output_freq = 30

    valid_reader = create_reader(valid_path, False, input_vocab_dim, label_vocab_dim)
    valid_bind = {
            find_arg_by_name('raw_input',ng)  : valid_reader.streams.features,
            find_arg_by_name('raw_labels',ng) : valid_reader.streams.labels
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
                e = ng.eval(mb_valid)
                print_sequences(e, i2w)

            print_training_progress(trainer, mbs, training_progress_output_freq)
            i += mb_train[raw_labels].num_samples
            mbs += 1

        print("--- EPOCH %d DONE: loss = %f, errs = %f ---" % (epoch, loss_numer/denom, 100.0*(metric_numer/denom)))


    error1 = translator_test_error(z, trainer, input_vocab_dim, label_vocab_dim)

    z.save_model("seq2seq.dnn")
    z.restore_model("seq2seq.dnn")

    label_seq_axis = Axis('labelAxis')
    label_sequence = sequence.slice(find_arg_by_name('raw_labels',z), 1, 0)
    ce = cross_entropy_with_softmax(z, label_sequence)
    errs = classification_error(z, label_sequence)
    trainer = Trainer(z, ce, errs, [momentum_sgd(
                    z.parameters, lr_per_minibatch, momentum_time_constant, True,
                    clipping_threshold_per_sample, gradient_clipping_with_truncation)])

    error2 = translator_test_error(z, trainer, input_vocab_dim, label_vocab_dim)

    assert error1 == error2

    return error1

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    #set_default_device(cpu())

    error = sequence_to_sequence_translator(debug_output=False, run_test=True)
    print("Error: %f" % error)
