# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def eval(node):        
    """ 
    It evaluates a node that has taken a numpy array as input. Note that sequences
    are not supported yet by this method
    
    Examples:
        Plus with two matrices
        >>> print (cntk.eval(cntk.ops.plus([[-30.,40.], [1.,2.]], [[-30.,40.], [1.,2.]])))
        #   [array([[[-60., 80.], [2., 4.]]])]
        
        Times with broadcast of a scalar over a matrix
        >>> print (cntk.eval(cntk.ops.element_times([[-30.,40.], [1.,2.]], 5)))
        #   [array([[[-150., 200.], [5., 10.]]])]        

    Args:
        node (:class:`cntk.graph.ComputationNode`): the node to evaluate        

    Returns:
        NumPy array containing the result
    """    
    
    from cntk.context import get_new_context        
    from cntk.ops import input_numpy, constant
    from cntk.graph import ComputationNode, _InputComputationNodeBase
    import numpy as np
    
    # call a helper method to get a context
    with get_new_context() as ctx:
        first = True    
        
        # The params are passed as arryas, e.g. plus([1,2], [3,4]),  and we need to 
        # wrap them with input and parameter nodes.
        if node.params:
            for p in node.params:
                if p in node.inputs:
                    val = getattr(node, p)
                    if not isinstance(val, ComputationNode):
                        # One param needs to be an Input() node. This will be fixed in 
                        # CNTK soon, so that we can remove this workaround and evaluate a 
                        # network with no inputs.
                        if first:        
                            ir = input_numpy([val], alias=p, name=p)
                            setattr(node, p, ir)
                            first = False
                        else:
                            setattr(node, p, constant(getattr(node, p), name=p))
                    else:
                        if isinstance(val, _InputComputationNodeBase) and first:
                            first = False
                          
        return ctx.eval(node)
