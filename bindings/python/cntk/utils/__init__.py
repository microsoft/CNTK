# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import numpy as np


def _ones_like(batch, precision):
    '''
    Returns a new batch, which has the same format as ``batch`` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    from cntk.internal import sanitize_precision
    return [np.ones_like(sample, dtype=sanitize_precision(precision)) for sample in batch]


def eval(op, arguments=None, precision=None, device=None, backward_pass=False, expected_backward=None):
    '''
    It evaluates ``op`` on the data provided by the reader. This is useful
    mainly to explore the operators and for convenient unit testing.

    Args:
        op (:class:`Function`): operation to evaluate
        arguments: maps variables to their input data. The
         interpretation depends on the input type:

          * `dict`: keys are input variable or names, and values are the input data.

          * any other type: if node has a unique input, ``arguments`` is mapped to this input.
            For nodes with more than one input, only `dict` is allowed.

         In both cases, every sample in the data will be interpreted
         as a new sequence. To mark samples as continuations of the
         previous sequence, specify ``arguments`` as `tuple`: the
         first element will be used as ``arguments``, and the second one will
         be used as a list of bools, denoting whether a sequence is a new
         one (`True`) or a continuation of the previous one (`False`).
         Data should be either NumPy arrays or a
         :class:`~cntk.io.MinibatchData` instance.
        seq_starts (list of bools or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans that tell whether a sequence is a new sequence (`True`) or a
         continuation of the sequence in the same slot of the previous
         minibatch (`False`)
        precision (str or None): precision being 'float32', 'float64', or
         None, in which case it will be determined by inspecting the operator
         (costly)
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on
        backward_pass (`bool`, optional): whether a backward pass is performed
        expected_backward (`dict` or None): keys are variables for which to
         compute a backward ouptut. By default (None) all entries from
         'arguments' are used

    Returns:
        mapping of output variables to their values.
    '''

    if backward_pass:
        state, forward_output = op.forward(arguments, op.outputs, op.outputs,
                                           device=device)

        if expected_backward is None:
            expected_backward = arguments
        root_gradients = {v: _ones_like(o, precision) for v, o in
                          forward_output.items()}

        backward_output = op.backward(state, root_gradients, expected_backward)

        return forward_output, backward_output

    else:
        state, forward_output = op.forward(
            arguments, op.outputs, None, device=device)
        return forward_output, None


