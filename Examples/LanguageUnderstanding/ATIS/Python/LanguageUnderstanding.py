# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import math
import cntk
from cntk.layers import *  # Layers library
from cntk.layers.typing import *

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
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
        query         = cntk.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
        intent_labels = cntk.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),  # (used for intent classification variant)
        slot_labels   = cntk.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)
    )), randomize=is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)

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
        #Recurrence(RNNStep(hidden_dim, activation=cntk.relu)),
        #(Recurrence(LSTM(hidden_dim)), Recurrence(LSTM(hidden_dim), go_backwards=True)), cntk.splice,
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
    @cntk.Function.with_signature(query = Sequence[SparseTensor[vocab_size]], labels = Sequence[SparseTensor[num_labels]])
    def criterion(query, labels):
        z = model(query)
        ce = cntk.losses.cross_entropy_with_softmax(z, labels)
        errs = cntk.metrics.classification_error(z, labels)
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
    z = model(cntk.Value.one_hot([w], vocab_size))    # run it through the model
    best = np.argmax(z,axis=2)                        # classify
    # show result
    print("Example Sentence After {} Epochs".format(epoch))
    for query, slot_label in zip(seq.split(),[slots_wl[s] for s in best[0]]):
        print("\t{}\t{}".format(query, slot_label))
    #print(model.embed.E.value)

########################
# train action         #
########################

def train(reader, model, max_epochs):

    # declare the model's input dimension, so that the saved model is usable
    model.update_signature(Sequence[SparseTensor[vocab_size]])

    # criterion: (model args, labels) -> (loss, metric)
    #   here  (query, slot_labels) -> (ce, errs)
    criterion = create_criterion_function(model)

    labels = reader.streams.slot_labels
    #labels = reader.streams.intent_labels  # for intent classification

    #from cntk.logging.graph import plot
    #plot(criterion, filename=data_dir + "/model.pdf")

    # iteration parameters
    # Epoch size in CNTK refers to not entire data sweeps, but rather number of samples
    # between checkpointing and/or summarizing training progress.
    epoch_size = 36000
    minibatch_size = 70

    # SGD parameters
    learner = cntk.learners.fsadagrad(criterion.parameters,
                        lr = cntk.learners.learning_rate_schedule([0.003]*2+[0.0015]*12+[0.0003], cntk.learners.UnitType.sample, epoch_size),
                        momentum = cntk.learners.momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9)),
                        gradient_clipping_threshold_per_sample = 15,
                        gradient_clipping_with_truncation = True)

    # logging
    # We provide a progress_printer that logs loss and metric, as well as a callback
    # for additional logging after every epoch ("training summary"),  in which we run
    # an example through our model, to peek into how it improves.
    cntk.logging.log_number_of_parameters(model) ; print()
    progress_printer = cntk.logging.ProgressPrinter(freq=100, first=10, tag='Training') # more detailed logging
    #progress_printer = cntk.logging.ProgressPrinter(tag='Training')
    progress_callback = cntk.logging.TrainingSummaryProgressCallback(epoch_size, lambda epoch, *unused_args: peek(model, epoch+1))

    peek(model, 0)                  # see how the model is doing
    # train() will loop through the training data provided by 'reader', minibatch by minibatch,
    # and update the model. The progress_printer is used to print loss and metric periodically.
    # The progress_callback is another progress tracker we use to call into our peek() function,
    # which illustrates how the model becomes better with each epoch.
    progress = criterion.train(reader, streams=(reader.streams.query, reader.streams.slot_labels),
                               minibatch_size=minibatch_size, max_epochs=max_epochs, epoch_size=epoch_size,
                               parameter_learners=[learner],
                               callbacks=[progress_printer, progress_callback])
    return progress.epoch_summaries[-1].loss, progress.epoch_summaries[-1].metric # return loss and metric from last epoch


########################
# eval action          #
########################

def evaluate(reader, model):
    criterion = create_criterion_function(model)
    progress_printer = cntk.logging.ProgressPrinter(tag='Evaluation')
    # test() will loop through the data provided by the reader and accumulate the metirc value
    # of the criterion function. At the end, progress_printer will be used to show the average value.
    # test() returns an object that contains the average metric.
    metric = criterion.test(reader, streams=(reader.streams.query, reader.streams.slot_labels),
                            minibatch_size=1000, callbacks=[progress_printer]).metric
                               
    return metric

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
    model.save(path)
    model = Function.load(path)

    # test
    reader = create_reader(data_dir + "/atis.test.ctf", is_training=False)
    evaluate(reader, model)
