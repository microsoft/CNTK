# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
from cntk.blocks import *  # non-layer like building blocks such as LSTM()
from cntk.layers import *  # layer-like stuff such as Linear()
from cntk.models import *  # higher abstraction level, e.g. entire standard models and also operators like Sequential()
from cntk.utils import *
from cntk.io import CNTKTextFormatMinibatchSource, StreamDef
from cntk import Trainer
from cntk.learner import sgd, fsadagrad, learning_rates_per_sample, momentums_per_sample
from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error

#### some temporary bits that will become unnecessary in the future

from cntk import Axis
def _Infer(shape, axis):
    from utils import Record, _as_tuple
    return Record(shape=_as_tuple(shape), axis=axis, with_shape = lambda new_shape: _Infer(new_shape, axis))

# upgrade Trainer class, add new method
def _extend_Trainer(trainer):
    class TrainerEx(trainer.__class__):
        # new method get_next_minibatch()
        # TODO: make this a true static method so we can say Trainer.get_next_minibatch()
        # TODO: is the "_next" really necessary? Trainer.get_minibatch() seems sufficient
        @staticmethod
        def next_minibatch(source, minibatch_size, input_map):
            mb = reader.get_next_minibatch(minibatch_size)
            if len(mb) == 0:  # TODO: return None instead?
                return (None, 0)
            else:
                return ({ key : mb[value].m_data        for (key, value) in input_map.items() },
                        { key : mb[value].m_num_samples for (key, value) in input_map.items() })
    if hasattr(trainer, 'next_minibatch'):  # already extended: don't redo
        return trainer
    trainer.__class__ = TrainerEx
    return trainer

#### User code begins here

########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."  # data resides in the CNTK folder
data_dir = cntk_dir + "/Tutorials/SLUHandsOn"                           # under Tutorials
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

def Reader(path):
    return CNTKTextFormatMinibatchSource(path, streams=Record(
        query         = StreamDef(shape=input_dim,   is_sparse=True, alias='S0'),
        intent_unused = StreamDef(shape=num_intents, is_sparse=True, alias='S1'),  # BUGBUG: unused, and should infer dim
        slot_labels   = StreamDef(shape=label_dim,   is_sparse=True, alias='S2')
    ))

########################
# define the model     #
########################

def Model(_inf):  # TODO: all the _inf stuff will go away once dimension inference works
    return Sequential([
        Embedding(shape=emb_dim, _inf=_inf),
        Recurrence(over=LSTM(shape=hidden_dim, _inf=_inf.with_shape(emb_dim)), _inf=_inf.with_shape(emb_dim), go_backwards=False),
        Linear(shape=label_dim, _inf=_inf.with_shape(hidden_dim))
    ], _inf=_inf)

########################
# train action         #
########################

def train(reader, model):
    # Input variables denoting the features and label data
    query       = Input(input_dim,  is_sparse=False)  # TODO: make sparse once it works
    slot_labels = Input(num_labels, is_sparse=True)

    # apply model to input
    z = model(query)

    # loss and metric
    ce = cross_entropy_with_softmax(z, slot_labels)
    pe = classification_error      (z, slot_labels)

    # training config
    epoch_size = 36000
    max_epochs = 8
    lr = learning_rates_per_sample([0.003]*2+[0.0015]*12+[0.0003], units=epoch_size)
    momentums = momentums_per_sample([0])
    minibatch_size = 70
    num_mbs_to_show_result = 10

    # trainer object
    learner = fsadagrad(z.parameters(), lr, momentums,
                        targetAdagradAvDenom=1, clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    trainer = Trainer(z, ce, pe, [learner])
    _extend_Trainer(trainer)  # TODO: should be just baked in

    # define mapping from reader streams to network inputs
    input_map = {
        query       : reader.streams.query,
        slot_labels : reader.streams.slot_labels
    }

    # process minibatches and perform model training
    t = 0
    loss_numer = 0  # TODO: find a nicer way of tracking, this is clumsy
    loss_denom = 0
    metric_numer = 0
    metric_denom = 0
    for epoch in range(max_epochs):
        while t < (epoch+1) * epoch_size:
            data, num_samples = trainer.next_minibatch(reader, minibatch_size, input_map)
            if data is None:
                break
            trainer.train_minibatch(data)
            loss_numer += trainer.previous_minibatch_loss_average() * trainer.previous_minibatch_sample_count()  # too much code for something this simple
            loss_denom +=                                             trainer.previous_minibatch_sample_count()
            metric_numer += trainer.previous_minibatch_evaluation_average() * trainer.previous_minibatch_sample_count()
            metric_denom +=                                                   trainer.previous_minibatch_sample_count()
            t += num_samples[slot_labels]
            print_training_progress(trainer, t / minibatch_size, num_mbs_to_show_result)
        print("--- EPOCH {} DONE: loss = {:0.6f}, metric = {:0.1f}% ---".format(epoch+1, loss_numer/loss_denom, metric_numer/metric_denom*100.0))

#############################
# main function boilerplate #
#############################

if __name__=='__main__':
    set_gpu(0)
    reader = Reader(data_dir + "/atis.train.ctf")
    model = Model(_inf=_Infer(shape=input_dim, axis=[Axis.default_batch_axis(), Axis.default_dynamic_axis()]))
    # train
    # BUGBUG: Currently this fails with a mismatch error if axes ^^ are given in opposite order
    train(reader, model)
    # test (TODO)
    reader = Reader(data_dir + "/atis.test.ctf")
    #test(reader, model_dir + "/slu.cmf")  # TODO: what is the correct pattern here?
