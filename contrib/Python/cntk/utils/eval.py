# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def eval(node, clean_up=True):        
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
        clean_up (bool): whether the temporary directory should be removed when the context is left        

    Returns:
        NumPy array containing the result
    """    
    
    from cntk.context import get_new_context        
    
    # call a helper method to get a context
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        return ctx.eval(node)
