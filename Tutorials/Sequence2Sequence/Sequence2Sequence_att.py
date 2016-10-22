# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import logging
import math
import time
import cntk as C
from cntk import DeviceDescriptor, Trainer, Axis, text_format_minibatch_source, StreamConfiguration
from cntk.learner import momentum_sgd, momentum_schedule
from cntk.ops import input_variable, placeholder_variable, cross_entropy_with_softmax, classification_error, sequence, slice, past_value, future_value, element_select, times, hardmax, plus, transpose
from cntk.ops.functions import CloneMethod
from cntk.persist import save_model, load_model

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from nn import LSTMP_component_with_self_stabilization, stabilize, linear_layer, print_training_progress, create_attention_augment_hook

path_to_cntk = "../.."

# Creates and trains a sequence to sequence model
def sequence_to_sequence_translator(debug_output=False, save_model=False):

    # some model params
    input_vocab_dim = 69
    label_vocab_dim = 69

    hidden_dim = 128
    num_layers = 1

    use_attention = True
    attention_dim = 128
    attention_span = 20

    # Source and target inputs to the model
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    raw_input = input_variable(shape=(input_vocab_dim), dynamic_axes=input_dynamic_axes, name='raw_input')

    label_dynamic_axes = [batch_axis, label_seq_axis]
    raw_labels = input_variable(shape=(label_vocab_dim), dynamic_axes=label_dynamic_axes, name='raw_labels')

    # Instantiate the sequence to sequence translation model
    input_sequence = raw_input

    # Drop the sentence start token from the label, for decoder training
    label_sequence = slice(raw_labels, label_seq_axis, 1, 0)   # <s> A B C </s> --> A B C </s>
    label_sentence_start = sequence.first(raw_labels)          # <s>

    is_first_label = sequence.is_first(label_sequence)
    label_sentence_start_scattered = sequence.scatter(label_sentence_start, is_first_label) # <s> 0 0 0 ...

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
    decoder_history_hook = plus(label_sequence, 0, name='decoder_history_hook')   # work-around for 'alias'; we need a copy of label_sequence instead of a pointer

    decoder_input = element_select(is_first_label, label_sentence_start_scattered, past_value(decoder_history_hook))

    # Decoder params
    augment_input_hook = None
    if use_attention:
        augment_input_hook = create_attention_augment_hook(attention_dim, attention_span, label_sequence, encoder_outputH)

    decoder_outputH = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0) or use_attention:
            recurrence_hookH = past_value
            recurrence_hookC = past_value
        else:
            isFirst = sequence.is_first(label_sequence)
            recurrence_hookH = lambda operand: element_select(
                isFirst, thought_vector_broadcastH, past_value(operand))
            recurrence_hookC = lambda operand: element_select(
                isFirst, thought_vector_broadcastC, past_value(operand))

        (decoder_outputH, encoder_outputC) = LSTMP_component_with_self_stabilization(
            decoder_outputH.output, hidden_dim, hidden_dim, recurrence_hookH, recurrence_hookC,
            augment_input_hook, hidden_dim)

    decoder_output = decoder_outputH

    # Softmax output layer
    final = stabilize(decoder_output)
    z = linear_layer(final, label_vocab_dim)

    ce = cross_entropy_with_softmax(z, label_sequence)
    errs = classification_error(z, label_sequence)

    # network output for decoder history
    net_output = hardmax(z)

    # make a clone of the graph where the ground truth is replaced by the network output
    ng = z.clone(CloneMethod.share, {decoder_history_hook.output : net_output.output})

    # load the vocab
    vocab_path = path_to_cntk + "/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.mapping"
    vocab = [w.strip() for w in open(vocab_path).readlines()]
    i2w = { i:ch for i,ch in enumerate(vocab) }

    train_path = path_to_cntk + "/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.train-dev-20-21.ctf"
    test_path = r"tiny.ctf"

    # reader(s)
    train_reader = text_format_minibatch_source(train_path, [
                     StreamConfiguration('features', input_vocab_dim, is_sparse=True, stream_alias='S0'),
                     StreamConfiguration('labels',   label_vocab_dim, is_sparse=True, stream_alias='S1')
                   ], randomize=True)
    features_si_tr = train_reader.stream_info('features')
    labels_si_tr   = train_reader.stream_info('labels')

    test_reader =  text_format_minibatch_source(test_path, [
                     StreamConfiguration('features', input_vocab_dim, is_sparse=True, stream_alias='S0'),
                     StreamConfiguration('labels',   label_vocab_dim, is_sparse=True, stream_alias='S1')
                   ], randomize=False)
    features_si_te = test_reader.stream_info('features')
    labels_si_te   = test_reader.stream_info('labels')

    # trainer params
    epoch_size = 908241    # number of label words
    max_epochs = 10
    minibatch_size = 72
    training_progress_output_freq = 500
    lr = 0.007
    momentum_time_constant = 1100
    m_schedule = momentum_schedule(math.exp(-1.0 / momentum_time_constant))

    # setup trainer
    learner = momentum_sgd(z.parameters, lr, m_schedule, 
                           gradient_clipping_threshold_per_sample=2.3, gradient_clipping_with_truncation=True)

    trainer = Trainer(z, ce, errs, [learner])

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0
    for epoch in range(max_epochs):
        loss_numer = 0
        metric_numer = 0
        denom = 0

        while i < (epoch+1) * epoch_size:

            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)

            train_args = {'raw_input': mb_train[features_si_tr], 'raw_labels': mb_train[labels_si_tr]}
            trainer.train_minibatch(train_args)

            samples = trainer.previous_minibatch_sample_count
            loss_numer += trainer.previous_minibatch_loss_average * samples
            metric_numer += trainer.previous_minibatch_evaluation_average * samples
            denom += samples

            # every 500 MBs evaluate on a test sequence to visually show how we're doing
            if mbs % 500 == 0:

                mb_test = test_reader.next_minibatch(minibatch_size)
                test_args = {'raw_input': mb_test[features_si_te], 'raw_labels': mb_test[labels_si_te]}

                e = ng.eval(test_args)
                print_sequences(e, i2w)

                print_training_progress(trainer, mbs, training_progress_output_freq)

            i += mb_train[labels_si_tr].num_samples
            mbs += 1

        print("--- EPOCH %d DONE: loss = %f, errs = %f ---" % (epoch, loss_numer/denom, 100.0*(metric_numer/denom)))

        if save_model:
            # save the model every epoch
            model_filename = "model_epoch%d.dnn" % epoch
            save_model(z, model_filename)
            print("Saved model to '%s'" % model_filename)

    return 0


def dfs_walk(node, visitor, accum, visited):
    if node in visited:
        return
    visited.add(node)
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visitor, accum, visited)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visitor, accum, visited)

    if visitor(node):
        accum.append(node)

def visit(root_node, visitor):
    nodes = []
    dfs_walk(root_node, visitor, nodes, set())
    return nodes

def find_nodes_by_name(root_node, node_name):
    return visit(root_node, lambda x: x.name == node_name)


def write(model_filename):

    # params
    input_vocab_dim = 69
    label_vocab_dim = 69

    # load the model...
    model = load_model(np.float32, model_filename)

    # load the vocab
    vocab_path = path_to_cntk + "/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.mapping"
    vocab = [w.strip() for w in open(vocab_path).readlines()]
    i2w = { i:ch for i,ch in enumerate(vocab) }

    # setup data...
    rel_path = path_to_cntk + "/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.test.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    path = "tiny.ctf"

    test_reader = text_format_minibatch_source(path, [
        StreamConfiguration("features", input_vocab_dim, True, 'S0'),
        StreamConfiguration("labels"  , label_vocab_dim, True, 'S1')
    ], 0)
    features_si = test_reader.stream_info("features")
    labels_si   = test_reader.stream_info("labels")

    test_minibatch_size = 1024

    # get references to decoder history hook
    #decoder_history_hook = find_nodes_by_name(model, 'decoder_history_hook')[0]

    # clone graph and modify node
    #ng = model.clone(CloneMethod.share, {decoder_history_hook.output : hardmax(model).output})

    # Get minibatches of sequences to write with
    i = 0
    while True:
        mb = test_reader.next_minibatch(test_minibatch_size)
        if len(mb) == 0:
            break

        args = {'raw_input': mb[features_si], 'raw_labels': mb[labels_si]}

        #C.cntk_py.set_computation_network_trace_level(1000000)

        e = model.eval(args)
        print_sequences(e, i2w)


def print_sequences(sequences, i2w):
    for s in sequences:
        print([i2w[np.argmax(w)] for w in s], sep=" ")


if __name__ == '__main__':
    # Specify the target device to be used for computing
    target_device = DeviceDescriptor.gpu_device(0)
    # If it is crashing, probably you don't have a GPU, so try with CPU:
    #target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)

    # train
    sequence_to_sequence_translator(True)

    # write / decode
    model_filename = "model_epoch2.dnn"
    #write(model_filename)
