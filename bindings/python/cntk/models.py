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
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
#from cntk.layers import *
from cntk.utils import Record
from cntk.blocks import identity, Block

# Sequential -- composite that applies a sequence of functions onto an input
# Sequential ([F, G, H]) === F >> G >> H
# TODO: address this feedback: "I find this arbitrary. You can have Sequential as part of a bigger layer.  Or you can view a linear layer already as a model (which is part of the bigger model)."
# TODO: can we use *arrayOfFunctions to avoid the [ ]?
# TODO: nested lists?
def Sequential(functions):
    from functools import reduce
    apply_x = reduce(lambda f, g: f >> g, functions, identity)
    #apply_x = _wrap_rename_Function(apply_x, 'Sequential')
    return Block(apply_x, 'Sequential', Record(functions=functions))
