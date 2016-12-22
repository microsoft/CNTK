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
# TODO: address this feedback: "I find this arbitrary. You can have Sequential as part of a bigger layer.  Or you can view a linear layer already as a model (which is part of the bigger model)."
# TODO: Willi had an idea how to use *layers to avoid the [ ]?
# Experimental: users can inject strings which name variables that are returned. Not pretty yet.
def Sequential(layers):
    if not isinstance(layers, list): # to support nested lists, run every item recursively through Sequential()
        # TODO: Is this confusing w.r.t. tuple which is parallel and list which is sequential?
        return layers
    #composed_function = identity
    #for layer in layers:
    #    def _is_string(obj):
    #        return isinstance(obj, str) # TODO: different in Python 2
    #    if _is_string(layer):
    #        UntestedBranchError("Sequential variable names") # BUGBUG: name gets lost in both Variable and resulting function once applied, so dict not usable for now for data, only for parameers
    #        composed_function = combine([composed_function.output], name=layer)
    #        attrs[layer] = composed_function
    #    else:
    #        composed_function = composed_function >> Sequential(layer)
    #attrs['layers'] = [layer for layer in layers if not _is_string(layer)]
    from functools import reduce
    layers = [Sequential(layer) for layer in layers] # expand all layers recursively
    composed_function = reduce(lambda f, g: f >> g, layers)
    # example ResNet layer:
    # rn_layer = (Conv(...) >> relu >> Conv(...) >> relu, None) >> plus
    # BUGBUG: In conjunction with alias(), this looses the placeholders somewhere; use this for debugging
    #composed_function = identity
    #for layer in layers:
    #    arg = composed_function
    #    composed_function = layer(composed_function)
    #    if len(composed_function.placeholders) != len(arg.placeholders):
    #        raise AssertionError('boom')
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

def LayerStack(N, constructor):
    return For(range(N), constructor)
