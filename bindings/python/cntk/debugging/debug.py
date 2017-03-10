# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import cntk_py

def save_as_legacy_model(root_op, filename):
    '''
    Save the network of ``root_op`` in ``filename``.
    For debugging purposes only, very likely to be deprecated in the future.

    Args:
        root_op (:class:`~cntk.functions.Function`): op of the graph to save
        filename (str): filename to store the model in.
    '''
    cntk_py.save_as_legacy_model(root_op, filename)

def set_computation_network_track_gap_nans(enable):
    '''
    Fill in NaNs in gaps of sequences to track unmasked uninitialized data.
    For debugging purposes only.

    Args:
        enable (Boolean): whether to enable gap nans tracking (with performance impact)
    '''
    cntk_py.set_computation_network_track_gap_nans(enable)

def set_computation_network_trace_level(level):
    '''
    Set trace level to the computation network. Currently supported values:
       0        : turn off trace
       1        : output nodes' dimensions and some other static info
       1000     : output each node's abs sum of elements in its value matrix for every forward/backward
       1000000  : output each node's full matrix for every forward/backward

    Args:
        level (int): trace level
    '''
    cntk_py.set_computational_network_trace_level(level)