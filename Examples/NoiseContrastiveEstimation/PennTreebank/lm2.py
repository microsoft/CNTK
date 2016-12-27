# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
from cntk import reduce_log_sum, times_transpose
from cntk.blocks import *  # non-layer like building blocks such as LSTM()
from cntk.layers import *  # layer-like stuff such as Linear()
from cntk.models import *  # higher abstraction level, e.g. entire standard models and also operators like Sequential()
from cntk.utils import *
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk import Trainer, UnitType
from cntk.learner import adam_sgd, learning_rate_schedule, momentum_schedule, momentum_as_time_constant_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error

########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."  # data resides in the CNTK folder
data_dir = "c:/work/cntk/sampled_softmax/PennTreebank/data"
vocab_size = 10000 ;        # number of words in vocab

model_dir = "./Models"

# model dimensions
input_dim  = vocab_size
label_dim  = vocab_size
emb_dim    = 250
hidden_dim = 500

########################
# define the reader    #
########################

def create_reader(path):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        input_words = StreamDef(field='S0', shape=input_dim,   is_sparse=True),
        output_labels = StreamDef(field='S1', shape=label_dim,   is_sparse=True)
    )))

########################
# define the model     #
########################

def create_model():
  with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
    return Sequential([
        #Stabilizer(),
        Embedding(emb_dim),
        LayerStack(1,  lambda: [
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            #Stabilizer(),
            #Dropout(0.5)
        ]),
        Dense(label_dim)
    ])

########################
# train action         #
########################

def train(reader, model, max_epochs):
    # Input variables denoting the features and label data
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')
    
    input_dynamic_axes = [batch_axis, input_seq_axis]
    input_words = input_variable(
        shape=(input_dim), dynamic_axes=input_dynamic_axes, name='raw_input')

    label_dynamic_axes = [batch_axis, label_seq_axis]
    output_labels = input_variable(
        shape=(label_dim), dynamic_axes=label_dynamic_axes, name='raw_labels')

    #----------------------------------------------
    W = Parameter((input_dim,emb_dim), init=init_default_or_glorot_uniform, name='W')
    W2 = Parameter((emb_dim,input_dim), init=init_default_or_glorot_uniform, name='W2')
    #b = Parameter(        (emb_dim), init=init_default_or_glorot_uniform, name='b')

    #------------------------------------------------
    
    # apply model to input
    #z = model(input_words)
    x = sigmoid( times(W, input_words) )
    z = times( W2, x )

    # loss and metric
    ce = reduce_log_sum(z) - times_transpose(output_labels,z)
    #pe = classification_error(z, output_labels)
    pe = reduce_log_sum(z) - times_transpose(output_labels,z)

    # training config
    epoch_size = 10000
    minibatch_size = 50
    num_mbs_to_show_result = 10
    momentum_as_time_constant = momentum_as_time_constant_schedule(1 / -math.log(0.9))  # TODO: Change to round number. This is 664.39. 700?

    lr_per_sample = [0.003]*2+[0.0015]*12+[0.0003] # LR schedule over epochs (we don't run that mayn epochs, but if we did, these are good values)

    # trainer object
    lr_schedule = learning_rate_schedule(lr_per_sample, UnitType.sample, epoch_size)
    learner = adam_sgd(z.parameters,
                       lr_schedule, momentum_as_time_constant,
                       low_memory=True,
                       gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    trainer = Trainer(z, ce, pe, [learner])

    # define mapping from reader streams to network inputs
    input_map = {
        input_words : reader.streams.input_words,
        output_labels : reader.streams.output_labels
    }

    # process minibatches and perform model training
    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(freq=100, first=10, tag='Training') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Training')

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:               # loop over minibatches on the epoch
            # BUGBUG? The change of minibatch_size parameter vv has no effect.
            data = reader.next_minibatch(min(minibatch_size, epoch_end-t), input_map=input_map) # fetch minibatch
            trainer.train_minibatch(data)                                   # update model with it
            t += data[output_labels].num_samples                              # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
            #def trace_node(name):
            #    nl = [n for n in z.parameters if n.name() == name]
            #    if len(nl) > 0:
            #        print (name, np.asarray(nl[0].value))
            #trace_node('W')
            #trace_node('stabilizer_param')
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric

#############################
# main function boilerplate #
#############################

if __name__=='__main__':
    # TODO: leave these in for now as debugging aids; remove for beta
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    #set_computation_network_trace_level(1)  # TODO: remove debugging facilities once this all works
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    force_deterministic_algorithms()

    reader = create_reader(data_dir + "/ptb.train.txt.ctf")
    model = create_model()
    # train
    train(reader, model, max_epochs=8)
    # test (TODO)
    reader = create_reader(data_dir + "/ptb.test.txt.ctf")
    #test(reader, model_dir + "/slu.cmf")  # TODO: what is the correct pattern here?
