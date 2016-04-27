# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Recurrent operations. These are nodes used with RNNs and make it possible to 
access, for example, `previous` or `next` elements in a sequence. For every 
operation we explain how the forward and backward passes are computed. For the 
backward pass we just explain the scalar case which is the building block for 
computing tensor gradients using the chain rule. For tensors, the backward pass 
of a node is computed as follows : for each element in the output tensor, its 
gradient with respect to the given input tensor is computed, then, the resulting 
tensors are added up.
"""

from cntk.ops.cntk1 import FutureValue, PastValue


def future_value(dims, x, time_step=1, default_hidden_activation=0.1, name=None):
    """
    This function returns the future value wrt `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the next logical sample. The `time_step` parameter is the number of steps 
    to look into the future and is 1 by default. If there is no future value (i.e. 
    the current sample is the last one in the tensor) then the `default_hidden_activation` 
    value is returned which is 0.1 by default.
    
    Example:
        >>> 
    
    Args:        
        dims: dimensions of the input `x`
        x: the tensor from which the future value is obtained
        time_step: the number of time steps to look into the future (default 1)
        default_hidden_activation: the default value to use when no future value 
        is available (default 0.1)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return FutureValue(dims, x, time_step, default_hidden_activation, var_name = name)
    
def past_value(dims, x, time_step=1, default_hidden_activation=0.1, name=None):
    """
    This function returns the past value wrt `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the previous logical sample. The `time_step` parameter is the number of steps 
    to look into the past and is 1 by default. If there is no past value (i.e. 
    the current sample is the first one in the tensor) then the `default_hidden_activation` 
    value is returned which is 0.1 by default.
    
    Example:
        >>> 
    
    Args:        
        dims: dimensions of the input `x`
        x: the tensor from which the past value is obtained
        time_step: the number of time steps to look into the past (default 1)
        default_hidden_activation: the default value to use when no past value 
        is available (default 0.1)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return PastValue(dims, x, time_step, default_hidden_activation, var_name = name)
