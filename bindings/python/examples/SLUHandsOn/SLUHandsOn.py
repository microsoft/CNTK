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
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk import Trainer
from cntk.learner import sgd, fsadagrad, learning_rate_schedule, momentum_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error
from examples.common.nn import print_training_progress

########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."  # data resides in the CNTK folder
data_dir = cntk_dir + "/Examples/Tutorials/SLUHandsOn"                  # under Examples/Tutorials
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    # number of words in vocab, slot labels, and intent labels

model_dir = "./Models"

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

########################
# define the reader    #
########################

def create_reader(path):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        query         = StreamDef(field='S0', shape=input_dim,   is_sparse=True),
        intent_unused = StreamDef(field='S1', shape=num_intents, is_sparse=True),  # BUGBUG: unused, and should infer dim
        slot_labels   = StreamDef(field='S2', shape=label_dim,   is_sparse=True)
    )))

########################
# define the model     #
########################

def create_model():
  # the following injects an option to mimic the BS version identically
  with default_options(enable_self_stabilization=False, initial_state=0.1):
    return Sequential([
        #Stabilizer(),
        Embedding(emb_dim),
        #BatchNormalization(),
        Recurrence(LSTM(hidden_dim), go_backwards=False),
        #Stabilizer(),
        #BatchNormalization(),
        Dense(label_dim)
    ])

########################
# train action         #
########################

def train(reader, model, max_epochs):
    # Input variables denoting the features and label data
    query       = Input(input_dim,  is_sparse=False)
    slot_labels = Input(num_labels, is_sparse=True)  # TODO: make sparse once it works

    # apply model to input
    z = model(query)
    #z = model(Stabilizer()(query))

    # loss and metric
    ce = cross_entropy_with_softmax(z, slot_labels)
    pe = classification_error      (z, slot_labels)

    # training config
    epoch_size = 36000
    minibatch_size = 70
    num_mbs_to_show_result = 100
    momentum_as_time_constant = minibatch_size / -math.log(0.9)

    lr_per_sample = [0.003]*2+[0.0015]*12+[0.0003]

    # trainer object
    lr_schedule = learning_rate_schedule(lr_per_sample, units=epoch_size)
    learner = fsadagrad(z.parameters, lr_schedule, momentum_as_time_constant,
                        targetAdagradAvDenom=1, gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)
    # BUGBUG: targetAdagradAvDenom must be removed from the interface, and default to 1

    trainer = Trainer(z, ce, pe, [learner])

    # define mapping from reader streams to network inputs
    input_map = {
        query       : reader.streams.query,
        slot_labels : reader.streams.slot_labels
    }

    # process minibatches and perform model training
    t = 0
    mbs = 0
    for epoch in range(max_epochs):
        loss_numer = 0  # TODO: find a nicer way of tracking, this is clumsy
        loss_denom = 0
        metric_numer = 0
        metric_denom = 0
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:
            data = reader.next_minibatch(min(minibatch_size, epoch_end-t), input_map=input_map)
            # BUGBUG? The change of minibatch_size parameter has no effect.
            #if data is None:
            #    break
            trainer.train_minibatch(data)
            #def trace_node(name):
            #    nl = [n for n in z.parameters if n.name() == name]
            #    if len(nl) > 0:
            #        print (name, np.asarray(nl[0].value))
            #trace_node('W')
            #trace_node('stabilizer_param')
            loss_numer += trainer.previous_minibatch_loss_average * trainer.previous_minibatch_sample_count  # too much code for something this simple
            loss_denom +=                                           trainer.previous_minibatch_sample_count
            metric_numer += trainer.previous_minibatch_evaluation_average * trainer.previous_minibatch_sample_count
            metric_denom +=                                                 trainer.previous_minibatch_sample_count
            print_training_progress(trainer, mbs if mbs > 10 else 0, num_mbs_to_show_result)
            t += data[slot_labels].num_samples
            mbs += 1
        print("--- EPOCH {} DONE: loss = {:0.6f} * {}, metric = {:0.1f}% * {} ---".format(epoch+1, loss_numer/loss_denom, loss_denom, metric_numer/metric_denom*100.0, metric_denom))

    return loss_numer/loss_denom, metric_numer/metric_denom

#############################
# main function boilerplate #
#############################

if __name__=='__main__':
    # TODO: get closure on Amit's feedback "Not the right pattern as we discussed over email. Please change to set_default_device(gpu(0))"
    #set_default_device(gpu(0))

    # TODO: leave these in for now as debugging aids; remove for beta
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1)  # TODO: remove debugging facilities once this all works
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    force_deterministic_algorithms()

    reader = create_reader(data_dir + "/atis.train.ctf")
    model = create_model()
    # train
    train(reader, model, max_epochs=8)
    # test (TODO)
    reader = create_reader(data_dir + "/atis.test.ctf")
    #test(reader, model_dir + "/slu.cmf")  # TODO: what is the correct pattern here?
