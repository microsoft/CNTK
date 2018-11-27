# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Customized Q function or (unnormalized) log of policy function.

If models from cntk.contrib.deeprl.agent.shared.models are not adequate, write
your own model as a function, which takes two required arguments
'shape_of_inputs', 'number_of_outputs', and two optional arguments
'loss_function', 'use_placeholder_for_input', and outputs a dictionary
containing 'inputs', 'outputs', 'f' and 'loss'. In the config file, set
QRepresentation or PolicyRepresentation to path (module_name.function_name) of
the function. QLearning/PolicyGradient will then automatically search for it.
"""

import cntk as C
import numpy as np


def conv_dqn(shape_of_inputs,
             number_of_outputs,
             loss_function=None,
             use_placeholder_for_input=False):
    """Example convolutional neural network for approximating the Q value function.

    This is the model used in the original DQN paper
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf.

    Args:
        shape_of_inputs: tuple of array (input) dimensions.
        number_of_outputs: dimension of output, equals the number of
            possible actions.
        loss_function: if not specified, use squared loss by default.
        use_placeholder_for_input: if true, inputs have to be replaced
            later with actual input_variable.

    Returns: a Python dictionary with string-valued keys including
        'inputs', 'outputs', 'loss' and 'f'.
    """
    # input/output
    inputs = C.ops.placeholder(shape=shape_of_inputs) \
        if use_placeholder_for_input \
        else C.ops.input_variable(shape=shape_of_inputs, dtype=np.float32)
    outputs = C.ops.input_variable(
        shape=(number_of_outputs,), dtype=np.float32)

    # network structure
    centered_inputs = inputs - 128
    scaled_inputs = centered_inputs / 256

    with C.layers.default_options(activation=C.ops.relu):
        q = C.layers.Sequential([
            C.layers.Convolution((8, 8), 32, strides=4),
            C.layers.Convolution((4, 4), 64, strides=2),
            C.layers.Convolution((3, 3), 64, strides=2),
            C.layers.Dense((512,)),
            C.layers.Dense(number_of_outputs, activation=None)
        ])(scaled_inputs)

    if loss_function is None:
        loss = C.losses.squared_error(q, outputs)
    else:
        loss = loss_function(q, outputs)

    return {
        'inputs': inputs,
        'outputs': outputs,
        'f': q,
        'loss': loss
    }
