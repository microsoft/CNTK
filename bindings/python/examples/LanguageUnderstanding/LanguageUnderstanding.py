# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
from cntk.blocks import *  # non-layer like building blocks such as LSTM()
from cntk.layers import *  # layer-like stuff such as Linear()
from cntk.models import *  # higher abstraction level, e.g. entire standard models and also operators like Sequential()
from cntk.utils import *
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk import Trainer
from cntk.learner import adam_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error, splice
from cntk.trainer import create_trainer
from cntk.persist import load_model, save_model

########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."  # data resides in the CNTK folder
data_dir = cntk_dir + "/Examples/Tutorials/SLUHandsOn"                  # under Examples/Tutorials
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    # number of words in vocab, slot labels, and intent labels

model_dir = "./Models"

# model dimensions
emb_dim    = 150
hidden_dim = 300

########################
# define the reader    #
########################

def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        query         = StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
        intent_unused = StreamDef(field='S1', shape=num_intents, is_sparse=True),  # BUGBUG: unused, and should infer dim
        slot_labels   = StreamDef(field='S2', shape=num_labels,  is_sparse=True)
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

########################
# define the model     #
########################

def create_model_function():
  with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
    return Sequential([
        Embedding(emb_dim),
        Recurrence(LSTM(hidden_dim), go_backwards=False),
        Dense(num_labels)
    ])

########################
# define the criteria  #
########################

# compose model function and criterion primitives into a criterion function
#  takes:   Function: features -> prediction
#  returns: Function: (features, labels) -> (loss, metric)
# This function is generic and could be a stock function create_ce_classification_criter().
def create_criterion_function(model):
    labels = SymbolicArgument()    # aka Placeholder()
    ce   = cross_entropy_with_softmax(model, labels)
    errs = classification_error      (model, labels)
    return combine ([ce, errs]) # (features, labels) -> (loss, metric)

########################
# train action         #
########################

def train(reader, model, max_epochs):
    # criterion: (model args, labels) -> (loss, metric)
    #   here  (query, slot_labels) -> (ce, errs)
    criterion = create_criterion_function(model)

    # declare argument types
    #criterion.set_signature(variable_type_of(source.streams.query), variable_type_of(source.streams.slot_labels))
    criterion.set_signature(variable_of_type(vocab_size,  is_sparse=False), variable_of_type(num_labels,  is_sparse=True))

    # iteration config  --needed here because learner schedule needs it
    epoch_size = 36000
    minibatch_size = 70
    #epoch_size = 1000 ; max_epochs = 1 # uncomment for faster testing

    momentum_time_constant = momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9))  # TODO: Change to round number. This is 664.39. 700?

    lr_schedule = [0.003]*2+[0.0015]*12+[0.0003] # LR schedule over epochs (we don't run that mayn epochs, but if we did, these are good values)

    # trainer object
    lr_per_sample = learning_rate_schedule(lr_schedule, epoch_size)
    learner = adam_sgd(criterion.parameters,
                       lr=lr_per_sample, momentum=momentum_time_constant,
                       low_memory=True,
                       gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    # trainer
    trainer = create_trainer(model, criterion, learner)

    # process minibatches and perform model training
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(freq=100, first=10, tag='Training') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Training')

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:               # loop over minibatches on the epoch
            # BUGBUG? The change of minibatch_size parameter vv has no effect.
            data = reader.next_minibatch(min(minibatch_size, epoch_end-t))     # fetch minibatch
            trainer.train_minibatch_from_data(criterion, data[reader.streams.query], data[reader.streams.slot_labels])  # update model with it
            t += data[reader.streams.slot_labels].num_samples                  # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric # return values from last epoch


########################
# eval action          #
########################

def evaluate(reader, model):
    criterion = create_criterion_function(model)
    criterion.set_signature(None, variable_of_type(num_labels, is_sparse=True))

    # process minibatches and perform evaluation
    dummy_learner = adam_sgd(criterion.parameters, lr=1, momentum=0, low_memory=True) # BUGBUG: should not be needed
    evaluator = create_trainer(model, criterion, [dummy_learner])
    progress_printer = ProgressPrinter(freq=100, first=10, tag='Evaluation') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Evaluation')

    while True:
        minibatch_size = 1000
        data = reader.next_minibatch(minibatch_size) # fetch minibatch
        if not data:                                 # until we hit the end
            break
        metric = evaluator.test_minibatch_from_data(criterion, data[reader.streams.query], data[reader.streams.slot_labels])
        progress_printer.update(0, data[reader.streams.slot_labels].num_samples, metric) # log progress
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric

#############################
# main function boilerplate #
#############################

if __name__=='__main__':
    # TODO: leave these in for now as debugging aids; remove for beta
    # TODO: try cntk_py without _ (feedback from Willi)
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    #set_computation_network_trace_level(1)  # TODO: remove debugging facilities once this all works
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    force_deterministic_algorithms()

    reader = create_reader(data_dir + "/atis.train.ctf", is_training=True)
    model = create_model_function()
    # train
    train(reader, model, max_epochs=8)

    # save and load (as an illustration)
    path = data_dir + "/model.cmf"
    #save_model(model, path)
    # BUGBUG: fails with "AttributeError: 'super' object has no attribute 'save_model'"
    #model = load_model(path)

    # test
    reader = create_reader(data_dir + "/atis.test.ctf", is_training=False)
    evaluate(reader, model)
