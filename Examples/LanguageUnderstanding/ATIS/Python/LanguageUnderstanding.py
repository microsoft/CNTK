# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import math
from cntk.layers import *  # Layers library
from cntk.layers.typing import *
from cntk.utils import *
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk import Trainer, Value
from cntk.learners import fsadagrad, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from cntk import splice, relu
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import *

########################
# variables and stuff  #
########################

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")
data_dir   = os.path.join(abs_path, "..", "Data") # under Examples/LanguageUnderstanding/ATIS

vocab_size = 943 ; num_labels = 129 ; num_intents = 26    # number of words in vocab, slot labels, and intent labels

# model dimensions
emb_dim    = 150
hidden_dim = 300

########################
# define the reader    #
########################

def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        query         = StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
        intent_labels = StreamDef(field='S1', shape=num_intents, is_sparse=True),  # (used for intent classification variant)
        slot_labels   = StreamDef(field='S2', shape=num_labels,  is_sparse=True)
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

########################
# define the model     #
########################

# TODO: separate slot and intent tagging; maybe do multi-task learning
def create_model_function():
  from cntk.ops.sequence import last
  with default_options(enable_self_stabilization=True):  # inject an option to mimic the BS version identically; remove some day
    return Sequential([
        Label('input'),
        Embedding(emb_dim, name='embed'),
        Label('embedded_input'),
        Stabilizer(),
        Recurrence(LSTM(hidden_dim)),
        # For illustration, variations of Recurrence(LSTM(...)) include
        #  - GRU
        #  - RNN
        #  - bidirectional LSTM
        # which would be invoked as follows:
        #Recurrence(GRU(hidden_dim)),
        #Recurrence(RNNUnit(hidden_dim, activation=relu)),
        #(Recurrence(LSTM(hidden_dim)), Recurrence(LSTM(hidden_dim), go_backwards=True)), splice,
        Stabilizer(),
        Label('hidden_representation'),
        Dense(num_labels, name='out_projection')
        #last, Dense(num_intents)    # for intent classification
    ])


########################
# define the criteria  #
########################

# compose model function and criterion primitives into a criterion function
#  takes:   Function: features -> prediction
#  returns: Function: (features, labels) -> (loss, metric)
def create_criterion_function(model):
    #@Function
    #def criterion(query: Sequence[SparseTensor[vocab_size]], labels: Sequence[SparseTensor[num_labels]]):
    @Function
    @Signature(query = Sequence[SparseTensor[vocab_size]], labels = Sequence[SparseTensor[num_labels]])
    def criterion(query, labels):
        z = model(query)
        ce   = cross_entropy_with_softmax(z, labels)
        errs = classification_error      (z, labels)
        return (ce, errs)
    return criterion

###########################
# helper to try the model #
###########################

query_wl = None
slots_wl = None
query_dict = None
slots_dict = None

def peek(model, epoch):
    # load dictionaries
    global query_wl, slots_wl, query_dict, slots_dict
    if query_wl is None:
        query_wl = [line.rstrip('\n') for line in open(data_dir + "/../BrainScript/query.wl")]
        slots_wl = [line.rstrip('\n') for line in open(data_dir + "/../BrainScript/slots.wl")]
        query_dict = {query_wl[i]:i for i in range(len(query_wl))}
        slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}
    # run a sequence through
    seq = 'BOS flights from new york to seattle EOS'  # example string
    w = [query_dict[w] for w in seq.split()]          # convert to word indices
    z = model(Value.one_hot([w], vocab_size))               # run it through the model
    best = np.argmax(z,axis=2)                        # classify
    # show result
    print("Example Sentence After Epoch [{}]".format(epoch))
    for query, slot_label in zip(seq.split(),[slots_wl[s] for s in best[0]]):
        print("\t{}\t{}".format(query, slot_label))
    #print(model.embed.E.value)

########################
# train action         #
########################

def train(reader, model, max_epochs):

    # declare the model's input dimension, so that the saved model is usable
    model.update_signature(Sequence[SparseTensor[vocab_size]])
    #model.declare_args(vocab_size)

    # criterion: (model args, labels) -> (loss, metric)
    #   here  (query, slot_labels) -> (ce, errs)
    criterion = create_criterion_function(model)

    labels = reader.streams.slot_labels
    #labels = reader.streams.intent_labels  # for intent classification

    #from cntk.logging.graph import plot
    #plot(criterion, filename=data_dir + "/model.pdf")

    # iteration parameters  --needed here because learner schedule needs it
    epoch_size = 36000
    minibatch_size = 70
    #epoch_size = 1000 ; max_epochs = 1 # uncomment for faster testing

    # SGD parameters
    learner = fsadagrad(criterion.parameters,
                        lr         = learning_rate_schedule([0.003]*2+[0.0015]*12+[0.0003], UnitType.sample, epoch_size),
                        momentum   = momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9)),
                        gradient_clipping_threshold_per_sample = 15,
                        gradient_clipping_with_truncation = True)

    # trainer
    trainer = Trainer(None, criterion, learner)

    # process minibatches and perform model training
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(freq=100, first=10, tag='Training') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Training')

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        peek(model, epoch)                  # log some interesting info
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            # BUGBUG: The change of minibatch_size parameter vv has no effect.
            # TODO: change all examples to this pattern; then remove this comment
            data = reader.next_minibatch(min(minibatch_size, epoch_end-t))     # fetch minibatch
            #trainer.train_minibatch(data[reader.streams.query], data[labels])  # update model with it
            trainer.train_minibatch({criterion.arguments[0]: data[reader.streams.query], criterion.arguments[1]: data[labels]})  # update model with it
            t += data[labels].num_samples                                      # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)    # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric # return values from last epoch


########################
# eval action          #
########################

# helper function to create a dummy Trainer that one can call test_minibatch() on
# TODO: replace by a proper such class once available
def Evaluator(model, criterion):
    from cntk import Trainer
    from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
    loss, metric = Trainer._get_loss_metric(criterion)
    parameters = set(loss.parameters)
    if model:
        parameters |= set(model.parameters)
    if metric:
        parameters |= set(metric.parameters)
    dummy_learner = momentum_sgd(tuple(parameters), 
                                 lr = learning_rate_schedule(1, UnitType.minibatch),
                                 momentum = momentum_as_time_constant_schedule(0))
    return Trainer(model, (loss, metric), dummy_learner)

def evaluate(reader, model):
    criterion = create_criterion_function(model)

    # process minibatches and perform evaluation
    evaluator = Evaluator(None, criterion)

    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Evaluation') # more detailed logging
    progress_printer = ProgressPrinter(tag='Evaluation')
    while True:
        minibatch_size = 1000
        data = reader.next_minibatch(minibatch_size) # fetch minibatch
        if not data:                                 # until we hit the end
            break
        #metric = evaluator.test_minibatch(query=data[reader.streams.query], labels=data[reader.streams.slot_labels])
        # note: keyword syntax ^^ is optional; this is to demonstrate it
        metric = evaluator.test_minibatch({criterion.arguments[0]: data[reader.streams.query], criterion.arguments[1]: data[reader.streams.slot_labels]})
        progress_printer.update(0, data[reader.streams.slot_labels].num_samples, metric) # log progress
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric


#############################
# main function boilerplate #
#############################

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='total epochs', required=False, default='8')
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created', required=False, default=None)

    args = vars(parser.parse_args())
    max_epochs = int(args['epochs'])

    # TODO: leave these in for now as debugging aids; remove for beta
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms

    from _cntk_py import set_fixed_random_seed
    set_fixed_random_seed(1) # useful for testing

    reader = create_reader(data_dir + "/atis.train.ctf", is_training=True) 
    model = create_model_function()

    # train
    train(reader, model, max_epochs)

    # save and load (as an illustration)
    path = model_path + "/model.cmf"
    model.save_model(path)
    model = Function.load(path)

    # test
    reader = create_reader(data_dir + "/atis.test.ctf", is_training=False)
    evaluate(reader, model)
