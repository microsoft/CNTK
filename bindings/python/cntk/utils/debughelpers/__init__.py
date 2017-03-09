# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
from cntk import DeviceDescriptor, Trainer, Axis, StreamConfiguration
from cntk.learner import sgd, adam_sgd
from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error
import itertools
import cntk.utils


# helper to name nodes for printf debugging
_auto_node_names = dict()
_auto_name_count = dict()
def _name_node(n, name):
    if not n in _auto_node_names:     # only name a node once
        # strip _.*
        #name = name.split('[')[0]
        if not name in _auto_name_count: # count each type separately
            _auto_name_count[name] = 1
        else:
            _auto_name_count[name] += 1
        #name = name + "[{}]".format(_auto_name_count[name])
        name = name + ".{}".format(_auto_name_count[name])
        _auto_node_names[n] = name
    return n

# this gives a name to anything not yet named
def _node_name(n):
    global _auto_node_names, _auto_name_count
    if n in _auto_node_names:
        return _auto_node_names[n]
    try:
        name = n.what()
    except:
        name = n.name
    # internal node names (not explicitly named)
    if name == '':
        if hasattr(n, 'is_placeholder') and n.is_placeholder:
            name = '_'
        else:
            name = '_f'
    _name_node(n, name)
    return _node_name(n)

# -> node name (names of function args if any)
def _node_description(n):
    desc = _node_name(n)
    if hasattr(n, 'inputs'):
        inputs = n.inputs
        #desc = "{} [{}]".format(desc, ", ".join([_node_name(p) for p in inputs]))
        func_params = [input for input in inputs if input.is_parameter]
        func_args   = [input for input in inputs if input.is_placeholder]
        if func_params:
            desc = "{} {{{}}}".format(desc, ", ".join([_node_name(p) for p in func_params]))
        desc = "{} <{}>".format(desc, ", ".join([_node_name(func_arg) for func_arg in func_args]))
    return desc

def _log_node(n):
    print (_node_description(n))

#def dump_graph(f):
#    visited = set()
#    def r_dump_graph(f, indent):
#        if f in visited:  # don't double-print
#            return
#        visited.add(f)
#        # print a node
#        inputs = f.root_function.inputs()
#        s = "{} ( ".format(f.name())
#        for c in inputs:
#            s += _node_name(c) + " "
#        s += ")"
#        print(s)
#        # print its children
#        for c in inputs:
#            r_dump_graph(c, indent+2)
#    r_dump_graph (f, 0)
