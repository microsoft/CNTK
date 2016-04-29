# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Reshaping operations. For every operation we explain how the forward and backward
passes are computed. For the backward pass we just explain the scalar case which is the building 
block for computing tensor gradients using the chain rule. For tensors, the backward pass of a node 
is computed as follows : for each element in the output tensor, its gradient with respect to the
given input tensor is computed, then, the resulting tensors are added up.
"""

def reshape(x, shape, beginAxis=0, endAxis=0, name=None):
    """
    Reinterpret input samples as having different tensor dimensions
    - just replaces metadata m_sampleLayout, does not change data values
    - one dimension may be specified as 0 and will be inferred
    - optional beginAxis/endAxis denote to only replace a sub-range of dims, for implementing ReshapeDimension() and FlattenRank()

    The output tensor has the same shape 'shape'.
    
    The backward pass propagates the received gradient for the output-shape to the input shape.
    
    Examples:
        >>> clip(2., 4., [1., 2.1, 3.0, 4.1])
        #[2.0, 2.1, 3.0, 4.0]
        
        >>> clip([-5., -4., 0., 3., 5.], [5., 4., 1., 4., 9.], [-10., -5., 0., 5., 10.])
        #[-5, -4., 0., 4., 9.]
    
    Args:        
        min_value: the minimum value to clip element values to
        max_value: the maximum value to clip element values to
        x: tensor to be clipped
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk.ops.cntk2 import Reshape
    return Reshape(x, shape, beginAxis, endAxis, var_name = name)