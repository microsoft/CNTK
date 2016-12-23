# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Tests for Layers library. Currently inside Examples to make them easier to run standalone.
"""
# TODO: move to appropriate test.py (they don't run stand-alone, so cannot be debugged properly)

from __future__ import print_function
import os
import math
from cntk.blocks import *  # non-layer like building blocks such as LSTM()
from cntk.layers import *  # layer-like stuff such as Linear()
from cntk.models import *  # higher abstraction level, e.g. entire standard models and also orsisrators like Sequential()
from cntk.utils import *

# helper to create float32 arrays, to work around a bug that Function.eval() does not know how to cast its inputs
def array(vals):
    return np.array(vals, dtype=np.float32)

if __name__=='__main__':

    # ----------------------------------------------
    # Recurrence() over regular function
    # ----------------------------------------------

    from cntk.layers import Recurrence
    from cntk.ops import plus
    #r = Recurrence(lambda a,b:a+b)
    # BUGBUG: above fails when just passing plus, since it does not know how many args it has (due to optional args). Python has no way around.
    #r.update_signature(1)
    #data = [   # audio sequence
    #    array([[2], [6], [4], [8], [6]])
    #]
    #out = r(data)
    # BUGBUG: fails with "ValueError: Variable(Plus5_output) with unknown shape detected when compiling the Function graph!"
    #print(out)

    # ----------------------------------------------
    # audio convolution
    # ----------------------------------------------

    from cntk.layers import Convolution
    c = Convolution(3, init=array([4, 2, 1]), pad=True, reduction_rank=0, bias=False)
    c.dump()
    c.update_signature(5)
    c.dump()
    data = [   # audio sequence
        array([[2, 6, 4, 8, 6]])
    ]
    out = c(data)
    print(out)

    # ----------------------------------------------
    # per-sequence initial state
    # ----------------------------------------------

    data = [
        array([[31,42], [5,3]]),
        array([[13,42], [5,3], [3,2], [6,7], [12,5], [3,22]]),
        array([[14,43], [51,23], [2,1]])
    ]
    initial_state = [
        array([[7.1,8.1]]),
        array([[7.2,8.2]]),
        array([[7.3,8.3], [7.31, 8.31]]),
    ]
    from cntk.ops import past_value, future_value
    batch_axis = Axis.default_batch_axis()
    data_seq_axis = Axis('inputAxis')
    init_seq_axis = Axis('initAxis')
    f = past_value(Input(2, dynamic_axes=[batch_axis, data_seq_axis]), time_step=2, initial_state=Input(2, dynamic_axes=[batch_axis, init_seq_axis]))
    res = f(data, initial_state)
    print(res)

    # ----------------------------------------------
    # the end
    # ----------------------------------------------

    print("done")
