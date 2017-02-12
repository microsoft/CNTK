# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import cntk

########################
# variables and stuff  #
########################

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."  # data resides in the CNTK folder
data_dir = cntk_dir + "/Examples/LanguageUnderstanding/ATIS/Data"       # under Examples/LanguageUnderstanding/ATIS
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

def create_reader(path, is_training=True):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
        query = cntk.io.StreamDef(field='S0', shape=input_dim,   is_sparse=True),
        intent_unused = cntk.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),  # BUGBUG: unused, and should infer dim
        slot_labels = cntk.io.StreamDef(field='S2', shape=label_dim,   is_sparse=True)
    )), randomize=is_training, epoch_size = cntk.io.INFINITELY_REPEAT if is_training else cntk.io.FULL_DATA_SWEEP)

########################
# define the model     #
########################

def create_model():
  with cntk.layers.default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
    return cntk.models.Sequential([
        cntk.layers.Embedding(emb_dim),
        cntk.layers.Recurrence(cntk.blocks.LSTM(hidden_dim), go_backwards=False),
        cntk.layers.Dense(label_dim)
    ])

########################
# train action         #
########################

def train(reader, model, max_epochs):
    # Input variables denoting the features and label data
    query = cntk.blocks.Input(input_dim,  is_sparse=False)
    slot_labels = cntk.blocks.Input(num_labels, is_sparse=True)  # TODO: make sparse once it works

    # apply model to input
    z = model(query)

    # loss and metric
    ce = cntk.ops.cross_entropy_with_softmax(z, slot_labels)
    pe = cntk.ops.classification_error(z, slot_labels)

    # training config
    epoch_size = 36000
    minibatch_size = 70
    num_mbs_to_show_result = 100

    # TODO: Change to round number. This is 664.39. 700?
    momentum_time_constant = cntk.learner.momentum_as_time_constant_schedule(minibatch_size / -math.log(0.9))  

    # LR schedule over epochs (we don't run that many epochs, but if we did, these are good values)
    lr_schedule = [0.003]*2+[0.0015]*12+[0.0003] 

    # trainer object
    lr_per_sample = cntk.learner.learning_rate_schedule(lr_schedule, cntk.learner.UnitType.sample, epoch_size)
    learner = cntk.learner.adam_sgd(z.parameters,
                                    lr=lr_per_sample, momentum=momentum_time_constant,
                                    low_memory=True,
                                    gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    trainer = cntk.Trainer(z, (ce, pe), [learner])

    # define mapping from reader streams to network inputs
    input_map = {
        query       : reader.streams.query,
        slot_labels : reader.streams.slot_labels
    }

    # process minibatches and perform model training
    cntk.utils.log_number_of_parameters(z) ; print()
    progress_printer = cntk.ProgressPrinter(freq=100, first=10, tag='Training') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Training')

    t = 0
 
    # loop over epochs
    for epoch in range(max_epochs):
        epoch_end = (epoch+1) * epoch_size

        # loop over minibatches on the epoch
        while t < epoch_end:
            # BUGBUG? The change of minibatch_size parameter vv has no effect.
            data = reader.next_minibatch(min(minibatch_size, epoch_end-t), input_map=input_map) # fetch minibatch
            trainer.train_minibatch(data)                                   # update model with it
            t += trainer.previous_minibatch_sample_count                    # count samples processed so far
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

    reader = create_reader(data_dir + "/atis.train.ctf")
    model = create_model()

    # train
    train(reader, model, max_epochs=8)

    # test (TODO)
    reader = create_reader(data_dir + "/atis.test.ctf")

    #test(reader, model_dir + "/slu.cmf")  # TODO: what is the correct pattern here?
