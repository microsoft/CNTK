# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# models -- models or parts of models composed of multiple layers
#           e.g. Sequential(). We could also add some default models here (ResNet) or frameworks (seq-2-seq)

# TODO: clean up the dependencies
import numpy as np
import sys
import os
import time

from .utils.debughelpers import _name_node, _node_name, _node_description, _log_node
#from cntk.layers import *
from .utils import Record
from cntk import combine
from .blocks import identity, Block

# Sequential -- composite that applies a sequence of layers (or any functions) onto an input
# Sequential ([F, G, H]) === F >> G >> H
def Sequential(layers):
    if not isinstance(layers, list): # to support nested lists, run every item recursively through Sequential()
        # TODO: Is this confusing w.r.t. tuple which is parallel and list which is sequential?
        return layers
    from functools import reduce
    layers = [Sequential(layer) for layer in layers] # expand all layers recursively
    composed_function = reduce(lambda f, g: f >> g, layers, identity)
    return Block(composed_function, 'Sequential', Record(layers=layers))

# For(range(3), lambda i: Dense(2000))
# For(range(3), lambda: Dense(2000))
def For(range, constructor):
    #from inspect import signature
    #takes_arg = len(signature(constructor).parameters) > 0
    # Python 2.7 support requires us to use getargspec() instead
    from inspect import getargspec
    takes_arg = len(getargspec(constructor).args) > 0
    # helper to call the layer constructor
    def call(i):
        if takes_arg:
            return constructor(i)  # takes an arg: pass it
        else:
            return constructor()   # takes no arg: call without, that's fine too
    layers = [call(i) for i in range]
    apply_x = Sequential(layers)
    return Block(apply_x, 'For', Record(layers=layers))

# old name
def LayerStack(N, constructor):
    return For(range(N), constructor)

## attempts at attention wrapping--this is just a prototype
## create s2s model:
#def S2S(): # -> (lambda input, history -> z):
#    # --- the involved objects
#
#    # encoder as usual
#    encoder = Sequential([  # lambda input -> h_enc
#        Embedding(300),
#        Recurrence(GRU(300))
#    ])
#
#    # decoder_step(h_enc, prev_embedded_history, prev_h_dec) -> h_dec
#    # is like a recurrent unit (LSTM() etc.) except that it takes a second history input, the decoding history.
#    # Attention model is a (lambda h_enc: -> (lambda h_dec: context)), i.e. it is called as attention_model(h_enc)(prev_h_dec)
#
#    attention_model = AttentionModel(128, window=20) # -> lambda h_enc: --> f(h_dec) -> h_enc_aug
#    decoder_cell = GRU(300)
#
#    # s2s decoder layer is a recurrence over a decoder step
#    #  - training recurrence:
#    #     - takes input
#    #     - has no recurrence inside it, but a protocol
#    #     - S2STrainer(encoder, decoder_step) --> s2s trainable model input -> label
#    #  - deployment recurrence: does not take input, instead uses internal feedback
#    #     - S2SEvalGreedy(encoder, decoder_step) --> s2s greedy decoder input -> translation
#    #     - S2SEvalBeam(encoder, decoder_step)
#
#    # alternatively:
#    #  - notion of inner and outer recurrence
#    #     - technically two nested Recurrence() calls
#    #       How does that make sense?
#    #       Can this be written as a recursion? Two nested Recurrence() calls as a single recursion
#    h_dec(t) = rnn(x(t), h_dec(t-1))   # this is a scan()
#    # decoding:
#    x(t) = hardmax(h_dec(t-1))
#    h_dec(t) = rnn(hardmax(h_dec(t-1)), h_dec(t-1))  # two recurrences, but optimizable; not a scan()
#
#    # with output:
#    (out,h_dec)(t) = (proj(rnn(x(t), h_dec(t-1))), rnn(x(t), h_dec(t-1)))   # this is a scan()
#    x(t) = hardmax(out(t-1))
#    (out,h_dec)(t) = (proj(rnn(hardmax(out(t-1)), h_dec(t-1))), rnn(hardmax(out(t-1)), h_dec(t-1)))  # two recurrences, but optimizable; not a scan()
#
#    # with attention:
#    h_dec(t) = rnn(x(t), att(h_enc(*), h_dec(t-1)))   # this is a scan()
#    h_dec(t) = rnn(hardmax(h_dec(t-1)), att(h_enc(*), h_dec(t-1)))  # two recurrences, but optimizable; not a scan()
#
#    # an attention-based model is one where the recurrent feedback (h_dec) is augmented by a selected encoder h_enc
#    # i.e. the cell(x, h') -> h
#    # gets replaced by a new cell
#    # cell'(x, h') = cell(x, splice(h', att(h'))) -> h
#
#    def RecurrentDecoder(cell):
#        def decoder_recurrence(h_enc):
#            # how to feed back the output with scan()?
#            return FreeRecurrence(cell, initial_state=h_enc, )
#        return decoder_recurrence
#
#    def RecurrentDecoderWithAttention(cell, attention_model):
#        def decoder(h_enc):
#            att = attention_model(h_enc)
#            def att_cell(x,dh):
#                return cell(x,splice(dh,att(dh)))
#            return Recurrence(att_cell)
#        return decoder
#
#    decoder = RecurrentDecoderWithAttention(GRU(300), AttentionModel(128, window=20))
#
#    # decoder_step: (h_enc, prev_embedded_history, prev_state) --> 
#    @Function
#    def decoder_step(h_enc, prev_embedded_history, prev_state):
#        att_fn = attention(h_enc) # lambda h_dec -> h_enc_aug
#        h_enc_aug = att_fn(prev_state)
#        # feedback signal becomes another hidden state input
#        # TODO: is that correct? Seems not for GRU
#        return dec_cell(h_enc_aug, splice(prev_embedded_history, prev_state))
#
#    decoder_step = WithAttention(GRU(300), AttentionModel(128, window=20)),  # lambda (h_enc, prev_embedded_history, prev_state_vars) -> state_vars
#    # WithAttention() consumes the first argument, feeds it to the attention model,
#    # and then runs prev_state_vars[0] through the resulting lambda, and augments its output (=context) to the GRU's data input (=prev_embedded_history).
#    # It then runs the recurrent unit as usual.
#
#    decoder = Sequential([  # lambda h_enc, history -> z
#        (None, Embedding(300)),                                # lambda h_enc, history -> h_enc, embedded_history
#        Recurrence(decoder_step, has_aux=True, initial_aux=Parameter(300)),  # lambda h_enc, embedded_history -> h_dec
#        # has_aux means (1) there are two inputs both fed to the recurrent cell, and (2) the second gets delayed just like h_dec
#        Dense(30000)                                                         # h_dec -> z
#    ])
#
#    # --- the apply function
#    @Function
#    def s2s_fun(input, history):
#        return decoder(encoder(input), history)
#
#    # we return the apply function; all layer objects are captured, as usual
#    return s2s_fun  # lambda (input, history) -> z
#
## create criterion function from model
##def create_criterion(s2s_model: (lambda input, history -> z)) -> (lambda input, labels -> ce, errs):
#def create_criterion(s2s_model): # -> (lambda input, labels -> ce, errs):
#    @Function
#    def criterion(x, y):
#        z = s2s_model(x, y)
#        ce   = cross_entropy_with_softmax(z, y)
#        errs = classification_error      (z, y)
#        return Record(ce=ce, errs=errs)
#    return criterion
#
## create decoder from model
##def GreedyDecoder(s2s_model: (lambda input, history -> z)) -> (lambda input -> z):
#def GreedyDecoder(s2s_model):
#    # a greedy decoder is one where the chosen output is passed as the history
#    return Recurrence(s2s_model >> hardmax) (input)
#
#    #@Function
#    #def s2s_decode_greedily(input):
#    #    # TODO: is this also scan()? It is, except with an additional map function in the recurrent pass.
#    #    #       I.e. need to cast the s2s_model() as an RNN cell as well.
#    #    #       rnn_step: (input, prev_h) -> h
#    #    # s2s_model: (input, history) -> z
#    #    #@Function
#    #    #def step(input, prev_state):
#    #    #    embedded_history = hardmax(prev_state)
#    #    #    return s2s_model(input, embedded_history)
#    #    greedy_step = (identity, hardmax) >> s2s_model
#    #    return Recurrence(greedy_step) (input)
#    #    #history = ForwardReference()
#    #    #z = s2s_model(input, history)
#    #    #read_out = hardmax(z)
#    #    #history.value = read_out
#    #    #return read_out
#    #return s2s_decode_greedily        # lambda input -> z
#
## create model
#s2s_model = create_model()            # lambda (input, history) -> z
#
## train
#criterion = create_criterion(s2s_model)
#Trainer(s2s_model, criterion).train(...)
#
## decode greedily
#decode = GreedyDecoder(s2s_model) # lambda input -> read_out
#result = decode(input_sequence)
