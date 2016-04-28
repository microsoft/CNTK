# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Evaluation operations. For every operation we explain how the forward and backward
passes are computed. For the backward pass we just explain the scalar case which is the building 
block for computing tensor gradients using the chain rule. For tensors, the backward pass of a node 
is computed as follows : for each element in the output tensor, its gradient with respect to the
given input tensor is computed, then, the resulting tensors are added up.
"""

from cntk.ops.cntk1 import CrossEntropyWithSoftmax

def crossentropy_with_softmax(target_values, feature_values, name=None):
    """
    This operator computes the cross entropy over the softmax of the `feature_values`.
    This op expects the `feature_values` as unscaled, it computes softmax over 
    the `feature_values` internally.  Any `feature_values` input over which softmax is 
    already computed before passing to this operator will be incorrect.
    
    Example:
        >>> crossentropy_with_softmax([0., 0., 0., 1.], [1., 1., 1., 1.])
        #[1.3862]
        
        >>> crossentropy_with_softmax([0.35, 0.15, 0.05, 0.45], [1, 2., 3., 4.])
        #[1.840]
    
    Args:
        target_values: the target valid probability distribution
        feature_values: the unscaled computed values from the network
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return CrossEntropyWithSoftmax(target_values, feature_values, var_name = name)