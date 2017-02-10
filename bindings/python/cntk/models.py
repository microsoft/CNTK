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
from .utils import Record
from cntk import combine
from .blocks import identity, Block

# Sequential -- composite that applies a sequence of layers (or any functions) onto an input
# Sequential ([F, G, H]) === F >> G >> H
# TODO: address this feedback: "I find this arbitrary. You can have Sequential as part of a bigger layer.  Or you can view a linear layer already as a model (which is part of the bigger model)."
# TODO: Willi had an idea how to use *layers to avoid the [ ]?
# Experimental: users can inject strings which name variables that are returned. Not pretty yet.
def Sequential(layers):
    if not isinstance(layers, (list,tuple)): # to support nested lists, run every item recursively through Sequential()
        return layers
    #apply_x = identity
    #for layer in layers:
    #    def _is_string(obj):
    #        return isinstance(obj, str) # TODO: different in Python 2
    #    if _is_string(layer):
    #        UntestedBranchError("Sequential variable names") # BUGBUG: name gets lost in both Variable and resulting function once applied, so dict not usable for now for data, only for parameers
    #        apply_x = combine([apply_x.output], name=layer)
    #        attrs[layer] = apply_x
    #    else:
    #        apply_x = apply_x >> Sequential(layer)
    #attrs['layers'] = [layer for layer in layers if not _is_string(layer)]
    from functools import reduce
    apply_x = reduce(lambda f, g: f >> Sequential(g), layers, identity)
    attrs = Record(layers=layers)
    return Block(apply_x, 'Sequential', members=attrs)

# For(range(3), lambda i: Dense(3))
# For(range(3), lambda: Dense(3))
def For(rng, constructor):
    '''
    Layer factory function to create a composite that applies a sequence of layers constructed with a constructor lambda(layer).
    E.g.
     For(range(3), lambda i: Dense(2000))
     For(range(3), lambda: Dense(2000))
    '''
    from inspect import getargspec
    takes_arg = len(getargspec(constructor).args) > 0
    # helper to call the layer constructor
    def call(i):
        if takes_arg:
            return constructor(i)  # takes an arg: pass it
        else:
            return constructor()   # takes no arg: call without, that's fine too
    layers = [call(i) for i in rng]
    apply_x = Sequential(layers)
    return Block(apply_x, 'For', members=Record(layers=layers))
