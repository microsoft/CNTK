# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Non-linear operations. For every operation we explain how the forward and backward
passes are computed. For the backward pass we just explain the scalar case which is the building 
block for computing tensor gradients using the chain rule. For tensors, the backward pass of a node 
is computed as follows : for each element in the output tensor, its gradient with respect to the
given input tensor is computed, then, the resulting tensors are added up.
"""

from cntk.ops.cntk1 import Clip

def clip(x, min_value, max_value, name=None):
    """
    Clips tensor values to fall between `min_value` and `max_value`.
    For the input tensor `x`, this node outputs a tensor of the same shape with 
    all of its values clipped to fall between `min_value` and `max_value`.
    The backward pass propagates the received gradient if no clipping occurred,
    and 0 if the value was clipped.
    
    Example:
        >>> clip([1., 2.1, 3.0, 4.1], 2., 4.)
        #[2.0, 2.1, 3.0, 4.0]
        
        >>> clip([-10., -5., 0., 5., 10.], [-5., -4., 0., 3., 5.], [5., 4., 1., 4., 9.])
        #[-5, -5., 0., 4., 9.]
    
    Args:
        x: tensor to be clipped
        min_value: the minimum value to clip element values to
        max_value: the maximum value to clip element values to
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return Clip(x, min_value, max_value, var_name = name)
