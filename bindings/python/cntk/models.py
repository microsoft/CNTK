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

def LayerStack(N, constructor):
    return For(range(N), constructor)
