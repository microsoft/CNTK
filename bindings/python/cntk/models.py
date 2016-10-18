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
#from cntk import DeviceDescriptor, Trainer, Axis, text_format_minibatch_source, StreamConfiguration
#from cntk.learner import sgd, fsadagrad
#from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.layers import *
from cntk.blocks import *
from cntk.blocks import _name_and_extend_Function, _wrap_rename_Function  # (debugging)

# Sequential -- composite that applies a sequence of functions onto an input
# Sequential ([F, G, H]) === F >> G >> H
# TODO: address this feedback: "I find this arbitrary. You can have Sequential as part of a bigger layer.  Or you can view a linear layer already as a model (which is part of the bigger model)."
def Sequential(arrayOfFunctions, _inf):
    import functools  # reduce()
    apply_x = functools.reduce(lambda f, g: f >> g, arrayOfFunctions, Identity(_inf=_inf))
    apply_x = _wrap_rename_Function(apply_x, 'Sequential')
    return apply_x;
