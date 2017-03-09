# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import cntk as C
import timeit
from cntk import Trainer, Axis
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk.ops.functions import load_model
from cntk.blocks import LSTM, Stabilizer
from cntk.layers import Recurrence, Dense
from cntk.models import For, Sequential
from cntk.utils import log_number_of_parameters, ProgressPrinter
from data_reader import DataReader
from math import log, exp
from cntk.device import set_default_device, cpu, gpu

# Setting global parameters
use_sampled_softmax = True
softmax_sample_size = 500 # Applies only when 'use_sampled_softmax = True'

use_sparse = True

hidden_dim = 200
num_layers = 2
num_epochs = 10
sequence_length = 40
sequences_per_batch = 10
alpha = 0.75
learning_rate = 0.002
momentum_as_time_constant = 10000
clipping_threshold_per_sample = 5.0
token_to_id_path        = './ptb/token2id.txt'
validation_file_path    = './ptb/valid.txt'
train_file_path         = './ptb/train.txt'
token_frequencies_file_path = './ptb/freq.txt'
segment_sepparator = '<eos>'
num_samples_between_progress_report = 100000


# reads a file with one number per line and returns the numbers as a list
def load_sampling_weights(sampling_weights_file_path):
    weights = []
    f = open(sampling_weights_file_path,'r')
    for line in f:
        if len(line) > 0:
            weights.append(float(line))
    return weights

# Creates model subgraph computing cross-entropy with softmax.
def cross_entropy_with_full_softmax(
    hidden_vector,  # Node providing the output of the recurrent layers
    target_vector,  # Node providing the expected labels (as sparse vectors)
    vocab_dim,      # Vocabulary size
    hidden_dim      # Dimension of the hidden vector
    ):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0)
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform)

    z = C.reshape( C.times_transpose(weights, hidden_vector) + bias, (1,vocab_dim))
    zT = C.times_transpose(z, target_vector)
    ce = C.reduce_log_sum_exp(z) - zT
    zMax = C.reduce_max(z)
    error_on_samples = C.less(zT, zMax)
    return (z, ce, error_on_samples)

# Creates model subgraph computing cross-entropy with sampled softmax.
def cross_entropy_with_sampled_softmax(
    hidden_vector,           # Node providing the output of the recurrent layers
    target_vector,           # Node providing the expected labels (as sparse vectors)
    vocab_dim,               # Vocabulary size
    hidden_dim,              # Dimension of the hidden vector
    num_samples,             # Number of samples to use for sampled softmax
    sampling_weights,        # Node providing weights to be used for the weighted sampling
    allow_duplicates = False # Boolean flag to control whether to use sampling with replacement (allow_duplicates == True) or without replacement.
    ):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0)
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform)

    sample_selector_sparse = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
    if use_sparse:
        sample_selector = sample_selector_sparse
    else:
        # Note: Sampled softmax with dense data is only supported for debugging purposes.
        # It might easily run into memory issues as the matrix 'I' below might be quite large.
        # In case we wan't to a dense representation for all data we have to convert the sample selector
        I = C.Constant(np.eye(vocab_dim, dtype=np.float32))
        sample_selector = C.times(sample_selector_sparse, I)

    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocab_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * vocab_dim]


    print("hidden_vector: "+str(hidden_vector.shape))
    wS = C.times(sample_selector, weights, name='wS') # [num_samples * hidden_dim]
    print("ws:"+str(wS.shape))
    zS = C.times_transpose(wS, hidden_vector, name='zS1') + C.times(sample_selector, bias, name='zS2') - C.times_transpose (sample_selector, log_prior, name='zS3')# [num_samples]

    # Getting the weight vector for the true label. Dimension hidden_dim
    wT = C.times(target_vector, weights, name='wT') # [1 * hidden_dim]
    zT = C.times_transpose(wT, hidden_vector, name='zT1') + C.times(target_vector, bias, name='zT2') - C.times_transpose(target_vector, log_prior, name='zT3') # [1]


    zSReduced = C.reduce_log_sum_exp(zS)

    # Compute the cross entropy that is used for training.
    # We don't check whether any of the classes in the random samples coincides with the true label, so it might happen that the true class is counted
    # twice in the normalizing denominator of sampled softmax.
    cross_entropy_on_samples = C.log_add_exp(zT, zSReduced) - zT

    # For applying the model we also output a node providing the input for the full softmax
    z = C.times_transpose(weights, hidden_vector) + bias
    z = C.reshape(z, shape = (vocab_dim))

    zSMax = C.reduce_max(zS)
    error_on_samples = C.less(zT, zSMax)
    return (z, cross_entropy_on_samples, error_on_samples)

def average_cross_entropy(full_cross_entropy_node, input_node, label_node, data):
    count = 0
    ce_sum = 0
    for features, labels, _ in data.minibatch_generator(validation_file_path, sequence_length, sequences_per_batch):
        arguments = ({input_node : features, label_node : labels})
        full_cross_entropy = full_cross_entropy_node.eval(arguments)
        for ce_list in full_cross_entropy:
            ce_sum += np.sum(ce_list)
            count += len(ce_list)

    return ce_sum / count

def create_model(input_sequence, label_sequence, vocab_dim, hidden_dim):
    # Create the rnn that computes the latent representation for the next token.
    rnn_with_latent_output = Sequential([
        C.Embedding(hidden_dim),   
        For(range(num_layers), lambda: 
            Sequential([Stabilizer(), Recurrence(LSTM(hidden_dim), go_backwards=False)])),
        ])

    
    # Apply it to the input sequence. 
    latent_vector = rnn_with_latent_output(input_sequence)

    # Connect the latent output to (sampled/full) softmax.
    if use_sampled_softmax:
        weights = load_sampling_weights(token_frequencies_file_path)
        smoothed_weights = np.float32( np.power(weights, alpha))
        sampling_weights = C.reshape(C.Constant(smoothed_weights), shape = (1,vocab_dim))
        z, ce, errs = cross_entropy_with_sampled_softmax(latent_vector, label_sequence, vocab_dim, hidden_dim, softmax_sample_size, sampling_weights)
    else:
        z, ce, errs = cross_entropy_with_full_softmax(latent_vector, label_sequence, vocab_dim, hidden_dim)

    return z, ce, errs


# Creates model inputs
def create_inputs(vocab_dim):
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]

    input_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = use_sparse)
    label_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = use_sparse)
    
    return input_sequence, label_sequence

def print_progress(samples_per_second, average_full_ce, total_samples, total_time):
    print("time=%.3f ce=%.3f perplexity=%.3f samples=%d samples/second=%.1f" % (total_time, average_full_ce, exp(average_full_ce), total_samples, samples_per_second))
    with open("log.txt", "a+") as myfile:
        myfile.write("%.3f\t%.3f\t%.3f\t%d\t%.1f\n" % (total_time, average_full_ce, exp(average_full_ce), total_samples, samples_per_second))


# Creates and trains an rnn language model.
def train_lm():
    data = DataReader(token_to_id_path, segment_sepparator)

    # Create model nodes for the source and target inputs
    input_sequence, label_sequence = create_inputs(data.vocab_dim)

    # Create the model. It has three output nodes
    # z: the input to softmax that  provides the latent representation of the next token
    # cross_entropy: this is used training criterion
    # error: this a binary indicator if the model predicts the correct token
    z, cross_entropy, error = create_model(input_sequence, label_sequence, data.vocab_dim, hidden_dim)

    # For measurement we use the (build in) full softmax.
    full_ce = C.cross_entropy_with_softmax(z, label_sequence)

    # print out some useful training information
    log_number_of_parameters(z) ; print()
    
    # Run the training loop
    num_trained_samples = 0
    num_trained_samples_since_last_report = 0

    # Instantiate the trainer object to drive the model training
    lr_schedule = learning_rate_schedule(learning_rate, UnitType.sample)
    momentum_schedule = momentum_as_time_constant_schedule(momentum_as_time_constant)
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(z.parameters, lr_schedule, momentum_schedule,
                            gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                            gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(z, (cross_entropy, error), learner)
  
    for epoch_count in range(num_epochs):
        for features, labels, token_count in data.minibatch_generator(train_file_path, sequence_length, sequences_per_batch):
            arguments = ({input_sequence : features, label_sequence : labels})

            t_start = timeit.default_timer()
            trainer.train_minibatch(arguments)
            t_end =  timeit.default_timer()

            samples_per_second = token_count / (t_end - t_start)

            # Print progress report every num_samples_between_progress_report samples

            if num_trained_samples_since_last_report >= num_samples_between_progress_report or num_trained_samples == 0:
                av_ce = average_cross_entropy(full_ce, input_sequence, label_sequence, data)
                print_progress(samples_per_second, av_ce, num_trained_samples, t_start)
                num_trained_samples_since_last_report = 0

            num_trained_samples += token_count
            num_trained_samples_since_last_report += token_count

        # after each epoch save the model
        model_filename = "models/lm_epoch%d.dnn" % epoch_count
        z.save_model(model_filename)
        print("Saved model to '%s'" % model_filename)


if __name__=='__main__':
    # train the LM
    train_lm()
