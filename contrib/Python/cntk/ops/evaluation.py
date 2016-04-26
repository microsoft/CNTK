# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""

"""

from cntk.ops.cntk1 import CrossEntropyWithSoftmax

def crossentropy_with_softmax(target_values, feature_values, name=None):
    """
    This operator computes the cross entropy over the softmax of the `feature_values`.
    This op expects the `feature_values` as unscaled, it computes softmax over 
    the `feature_values` internally. 
    
    Example:
        >>> crossentropywithsoftmax([0., 0., 0., 1.],[1., 1., 1., 1.])
        #[1.3862]
        
        >>> crossentropywithsoftmax([0.35, 0.15, 0.05, 0.45], [1, 2., 3., 4.])
        #[1.840]
    
    Args:
        target_values: the target valid probability distribution
        feature_values: the unscaled computed values from the network
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return CrossEntropyWithSoftmax(target_values, feature_values, var_name = name)