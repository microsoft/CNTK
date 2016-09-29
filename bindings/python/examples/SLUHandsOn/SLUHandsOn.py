# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
from cntk import DeviceDescriptor, Trainer, sgd_learner, Axis, text_format_minibatch_source, StreamConfiguration
from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error
import itertools
from cntk.ops.functions import Function

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import LSTMP_component_with_self_stabilization, embedding, linear_layer, select_last, print_training_progress


#### monkey-patching some missing stuff
def __matmul__(a,b):  # TODO: define @ once we have Python 3.5
    return times(a,b)
Function.__matmul__ = __matmul__  # should work in Python 3.5  --Function is not defined?

# apply a Function to a Variable or tuple of variables
def _apply(function, arg):
    if isinstance(arg, tuple):
        # BUGBUG: How about order of function arguments?
        # if input is a tuple then apply one arg at a time
        # note that arg may contain nested tuples as well, which get flattened this way
        # e.g. LSTM (x, (h, c))
        for one_arg in list(arg):
            function = _apply(function, one_arg)
        return function
    else:
        f = function.clone()
        inputs = f.inputs()  # TODO: This should be ordered, but it is a set (right?)
        placeholders = [p for p in inputs if p.is_placeholder()]
        return f.replace_placeholders({placeholders[0]: arg})  # replace first placeholder --TODO: order?
#Function.apply = _apply  # should really be __call__
Parameter = parameter   # these are factory methods for things with state
Input = input_variable

# Sequential -- composite that applies a sequence of functions onto an input
# Sequential ([F, G, H]) === F >> G >> H
def Sequential(arrayOfFunctions, _indim):
    r = Placeholder(_indim)
    for f in arrayOfFunctions:
        r = _apply(f, r)
    return r;

#### temporary layers lib, to be moved out
def Placeholder(_indim):
    return placeholder_variable(shape=layers._as_tuple(_indim))  # where to get the dynamic axis?
    #placeholder_variable(shape=(_indim), dynamic_axes=input.dynamic_axes())

class layers:
    # need to define everything indented by 4

    _INFERRED = 0   # TODO: use the predefined name for this

    # Linear -- create a fully-connected linear projection layer
    # Note: shape may describe a tensor as well.
    # TODO: change to new random-init descriptor
    @staticmethod
    def Linear(shape, _indim, bias=True, init='glorot_uniform', initValueScale=1, inputRank=None, mapRank=None):
        out_shape = layers._as_tuple(shape)
        in_shape = layers._as_tuple(_indim)  # TODO: INFERRED
        W = Parameter(in_shape + out_shape)
        b = Parameter(           out_shape) if bias else None
        x = Placeholder(_indim)
        apply = __matmul__(x, W) + b if bias else \
                __matmul__(x, W)
        return apply
        # TODO: how to break after the else?

    # type-cast a shape given as a scalar into a tuple
    @staticmethod
    def _as_tuple(x):
        return x if (isinstance(x,tuple)) else (x,)

    # Embedding -- create a linear embedding layer
    @staticmethod
    def Embedding(shape, _indim, init='glorot_uniform', initValueScale=1, embedding_path=None, transpose=False):
        shape = layers._as_tuple(shape)
        full_shape = (shape + _indim,) if transpose else ((_indim,) + shape)
        if embedding_path is None:
            # TODO: how to pass all optional args automatically in one go?
            return layers.Linear(shape, _indim, init=init, initValueScale=initValueScale)  # learnable
        else:
            E = Parameter(full_shape, initFromFilePath=embeddingPath, learningRateMultiplier=0)  # fixed from file
        x = Placeholder(_indim)
        apply = __matmul__(E, x) if transposed else \
                __matmul__(x, E)     # x is expected to be sparse one-hot
        return apply

    # TODO: We are stuck with two-argument functions
    @staticmethod
    def LSTM(hidden_shape, _indim, cell_shape=None): # (x, (h, c))
        if cell_shape is None:
            cell_shape = hidden_shape
        def create_hc_placeholder():
            return (Placeholder(hidden_shape), Placeholder(cell_shape)) # (h, c)
        x = Placeholder(_indim)
        prev_state = create_hc_placeholder()
        block = layers.Linear(hidden_shape, _indim)  # for now
        # return to caller a helper function to create placeholders for recurrence
        block.create_placeholder = create_hc_placeholder
        return block

    @staticmethod
    def Recurrence(block, _indim, go_backwards=False):
        # helper to compute previous value
        def previous_hook(state):
            if isinstance(state, tuple):  # if multiple then apply to each element
                return tuple([previous_hook(s) for s in list(state)])
            else: # not a tuple: must be a 'scalar', i.e. a single element
                return past_value(state) if not go_backwards else \
                       future_value(state)
        x = Placeholder(_indim)
        prev_state_forward = block.create_placeholder() # create a placeholder or a tuple of placeholders
        state = _apply (block, (x, prev_state_forward)) # apply the recurrent block
        prev_state = previous_hook(state) # recurrent memory. E.g. Previous or Next, with or without initial state, beam reordering etc.
        prev_state.replace_placeholders({key: value.output() for (key, value) in list(zip(list(prev_state_forward), list(prev_state)))})
        # TODO: Doe sthis ^^ work for tuples?
        return state

#### User code begins here

root_dir = "." ; data_dir = root_dir ; model_dir = data_dir + "/Models"

vocab_size = 943 ; num_labels = 129 ; num_intents = 26    # number of words in vocab, slot labels, and intent labels

def Reader(path):
    mb_source = text_format_minibatch_source(path, [
                    StreamConfiguration('query',      dim=vocab_size, is_sparse=True, stream_alias='S0'),
                    StreamConfiguration('slotLabels', dim=num_labels, is_sparse=True, stream_alias='S2') ], 36000)
    # what's that 10000 at the end?
    # stream_alias -> alias
    # if the dimension is 'dim', should it be called that in the other places as well, instead of 'shape'? Or change to 'shape'?

input_dim = vocab_size
emb_dim = 150
hidden_dim = 300
label_dim = num_labels

def Model(_indim):
    return Sequential([
        layers.Embedding(emb_dim, _indim=_indim),
        layers.Recurrence(layers.LSTM(hidden_dim, _indim=emb_dim), _indim=emb_dim, go_backwards=False),
        layers.Linear(label_dim, _indim=hidden_dim)
    ], _indim=_indim)

def train(reader, model):
    # Input variables denoting the features and label data
    query      = Input(input_dim)  # TODO: make sparse once it works
    slotLabels = Input(num_labels)

    # apply model to input
    z = _apply(model, query)

    # loss and metric
    ce = cross_entropy_with_softmax(z, slotLabels)
    pe = classification_error      (z, slotLabels)

    # training config
    lr = 0.003  # TODO: 0.003*2:0.0015*12:0.0003
    #gradUpdateType = "fsAdaGrad"
    #gradientClippingWithTruncation = True ; clippingThresholdPerSample = 15.0
    #first_mbs_to_show_result = 10
    minibatch_size = 70
    num_mbs_to_show_result = 10

    # trainer object
    trainer = Trainer(classifier_output, ce, pe, [sgd_learner(classifier_output.owner.parameters(), lr)])

    # process minibatches and perform model training
    for i in itertools.count():
        mb = mb_source.get_next_minibatch(minibatch_size)
        if len(mb) == 0:  # return None instead?
            break

        # monkey-patch in a helper to establish the mapping of input variables in the model to actual minibatch data to be trained with  --TODO: add method to MB class
        # TODO: should we use *args? Then no need for [ ]
        mb.data = lambda inputs: {input:mb[mb_source.stream_info(input)].m_data for input in inputs}

        trainer.train_minibatch(mb.data([query, slotLabels]))

        print_training_progress(trainer, i, num_mbs_to_show_result)

if __name__=='__main__':
    # Specify the target device to be used for computing
    target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)

    os.chdir('c:/work/CNTK/Tutorials/SLUHandsOn')

    reader = Reader(data_dir + "/atis.train.ctf")
    model = Model(_indim=input_dim)
    train(reader, model)
