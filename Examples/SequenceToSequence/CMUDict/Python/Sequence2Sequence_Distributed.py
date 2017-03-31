# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
import argparse
import _cntk_py
import cntk

from cntk import Trainer, Axis
from cntk.device import try_set_default_device, gpu
from cntk.train.distributed import *
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learners import learning_rate_schedule, UnitType, momentum_sgd, momentum_as_time_constant_schedule
from cntk import input, cross_entropy_with_softmax, classification_error, sequence, element_select, alias, hardmax
from cntk.ops.functions import CloneMethod
from cntk.train.training_session import *
from cntk.logging import *

abs_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")
sys.path.append(os.path.join(abs_path, "..", "..", "..", "common"))
from nn import LSTMP_component_with_self_stabilization, stabilize, linear_layer, print_training_progress

default_quantization_bits = 32

def create_reader(path, randomize, input_vocab_dim, label_vocab_dim, size=INFINITELY_REPEAT):
    if not os.path.exists(path):
        raise RuntimeError("File '%s' does not exist." % (path))

    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_dim,  is_sparse=True)
    )), randomize=randomize, max_samples = size)

def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer):
    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.5, UnitType.minibatch)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True

    # Create learner
    if block_size is not None and num_quantization_bits != default_quantization_bits:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    local_learner = momentum_sgd(network['output'].parameters,
                           lr_per_minibatch, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)

    if block_size != None:
        learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    return Trainer(network['output'], (network['ce'], network['pe']), learner, progress_printer)

def create_network(input_vocab_dim, label_vocab_dim):
    # network complexity; initially low for faster testing
    hidden_dim = 256
    num_layers = 1

    # Source and target inputs to the model
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')
    raw_input = sequence.input(shape=(input_vocab_dim), sequence_axis=input_seq_axis, name='raw_input')
    raw_labels = sequence.input(shape=(label_vocab_dim), sequence_axis=label_seq_axis, name='raw_labels')

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
            encoder_outputH.output, hidden_dim, hidden_dim, sequence.future_value, sequence.future_value)

    thought_vectorH = sequence.first(encoder_outputH)
    thought_vectorC = sequence.first(encoder_outputC)

    thought_vector_broadcastH = sequence.broadcast_as(
        thought_vectorH, label_sequence)
    thought_vector_broadcastC = sequence.broadcast_as(
        thought_vectorC, label_sequence)

    # Decoder
    decoder_history_hook = alias(label_sequence, name='decoder_history_hook') # copy label_sequence

    decoder_input = element_select(is_first_label, label_sentence_start_scattered, sequence.past_value(
        decoder_history_hook))

    decoder_outputH = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0):
            recurrence_hookH = sequence.past_value
            recurrence_hookC = sequence.past_value
        else:
            isFirst = sequence.is_first(label_sequence)
            recurrence_hookH = lambda operand: element_select(
                isFirst, thought_vector_broadcastH, sequence.past_value(operand))
            recurrence_hookC = lambda operand: element_select(
                isFirst, thought_vector_broadcastC, sequence.past_value(operand))

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

    return {
        'raw_input' : raw_input,
        'raw_labels' : raw_labels,
        'ce' : ce,
        'pe' : errs,
        'ng' : ng,
        'output': z
    }

def train_and_test(network, trainer, train_reader, test_reader, epoch_size, minibatch_size):
    train_bind = {
        network['raw_input']  : train_reader.streams.features,
        network['raw_labels'] : train_reader.streams.labels
    }

    training_session(
        mb_source = train_reader,
        trainer=trainer,
        model_inputs_to_streams=train_bind,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=CheckpointConfig(frequency = epoch_size,
                                           filename = os.path.join(model_path, "SequenceToSequence"),
                                           restore = False),
        cv_config=CrossValidationConfig(source=test_reader, mb_size=minibatch_size)
    ).train()

def sequence_to_sequence_translator(train_data, test_data, epoch_size=908241, num_quantization_bits=default_quantization_bits, block_size=3200, warm_up=0, minibatch_size=72, max_epochs=10, randomize_data=False, log_to_file=None, num_mbs_per_log=10, gen_heartbeat=False):
    cntk.debugging.set_computation_network_trace_level(0)

    distributed_sync_report_freq = None
    if block_size is not None:
        distributed_sync_report_freq = 1

    progress_printer = ProgressPrinter(freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs,
        distributed_freq=distributed_sync_report_freq)

    input_vocab_dim = 69
    label_vocab_dim = 69

    network = create_network(input_vocab_dim, label_vocab_dim)
    trainer = create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer)

    train_reader = create_reader(train_data, randomize_data, input_vocab_dim, label_vocab_dim, size=max_epochs*epoch_size)

    test_reader = create_reader(test_data, False, input_vocab_dim, label_vocab_dim, size=cntk.io.FULL_DATA_SWEEP)

    train_and_test(network, trainer, train_reader, test_reader, epoch_size, minibatch_size)

if __name__ == '__main__':
    data_path  = os.path.join(abs_path, "..", "Data")

    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the CMUDict dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-e', '--epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-ms', '--minibatch_size', help='Minibatch size', type=int, required=False, default='16')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-es', '--epoch_size', help='Epoch size', type=int, required=False, default='64')
    parser.add_argument('-r', '--randomize_data', help='Randomize training data', type=bool, required=False, default=False)
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['device'] is not None:
        try_set_default_device(gpu(args['device']))

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    train_data = os.path.join(data_path, 'cmudict-0.7b.train-dev-20-21.ctf')
    test_data = os.path.join(data_path, 'cmudict-0.7b.test.ctf')

    try:
        sequence_to_sequence_translator(train_data, test_data,
                                        epoch_size=args['epoch_size'],
                                        num_quantization_bits=args['quantized_bits'],
                                        block_size=args['block_samples'],
                                        warm_up=args['distributed_after'],
                                        minibatch_size= args['minibatch_size'],
                                        max_epochs=args['epochs'],
                                        randomize_data=args['randomize_data'],
                                        log_to_file=args['logdir'],
                                        num_mbs_per_log=10)

    finally:
        Communicator.finalize()
