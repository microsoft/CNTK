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
from cntk.layers import *  # CNTK Layers library
from cntk.internal.utils import *

# helper to create float32 arrays, to work around a bug that Function.eval() does not know how to cast its inputs
def array(vals):
    return np.array(vals, dtype=np.float32)

if __name__=='__main__':

    # TODO: add all Layers tests here and use the correct pytest pattern

    # ----------------------------------------------
    # Recurrence() over regular function
    # ----------------------------------------------

    from cntk.layers import Recurrence
    from cntk.ops import plus
    from cntk.debugging import *
    r = Recurrence(plus)
    dump_function(r)
    r.update_signature(1)
    dump_function(r)
    data = [   # simple sequence
        array([[2], [6], [4], [8], [6]])
    ]
    #out = r(data)
    # BUGBUG: fails with "ValueError: Variable(Plus5_output) with unknown shape detected when compiling the Function graph!"
    #print(out)

    # ----------------------------------------------
    # sequential convolution without reduction dimension
    # ----------------------------------------------

    from cntk.layers import Convolution
    c = Convolution(3, init=array([4, 2, 1]), sequential=True, pad=False, reduction_rank=0, bias=False)
    dump_function(c)
    c.update_signature(1)
    dump_function(c)
    data = [   # audio sequence
        array([[2], [6], [4], [8], [6]])
    ]
    out = c(data)
    print(out)
    # [[[[ 24.  40.  38.]]]]

    # ----------------------------------------------
    # 1D convolution without reduction dimension
    # ----------------------------------------------

    from cntk.layers import Convolution
    c = Convolution(3, init=array([4, 2, 1]), pad=True, reduction_rank=0, bias=False)
    # BUGBUG: pad seems ignored??
    dump_function(c)
    c.update_signature(5)
    dump_function(c)
    data = [   # audio sequence
        array([[2, 6, 4, 8, 6]])
    ]
    out = c(data)
    print(out)
    # [[[[ 24.  40.  38.]]]]

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
    data_seq_axis = Axis('inputAxis')
    init_seq_axis = Axis('initAxis')
    f = sequence.past_value(sequence.input(2, sequence_axis=data_seq_axis), time_step=2, initial_state=sequence.input(2, sequence_axis=init_seq_axis))
    res = f(data, initial_state)
    print(res)

    # ----------------------------------------------
    # the end
    # ----------------------------------------------

    print("done")
