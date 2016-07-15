# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..utils import get_rank, sanitize_input, create_NDArrayViewPtr
from cntk import cntk_py

################################################################################
# convolution ops
################################################################################

################################################################################
# evaluation ops
################################################################################


def cross_entropy_with_softmax(target_vector, output_vector, name=''):
    """
    This operation computes the cross entropy over the softmax of the `output_vector`.
    It expects the `output_vector` as unscaled, and it computes softmax over 
    the `output_vector` internally.  Any `output_vector` input over which softmax is 
    already computed before passing to this operator will be incorrect.
    
    :math:`cross\_entropy\_with\_softmax(t, o) = {-{\sum_{i \in \{1,len(t)\}} t_i \log(softmax(o_i)) }}`
    
    Example:
        >>> C.eval(C.cross_entropy_with_softmax([0., 0., 0., 1.], [1., 1., 1., 50.]))
        #[0.]
        
        >>> C.eval(C.cross_entropy_with_softmax([0.35, 0.15, 0.05, 0.45], [1., 2., 3., 4.]))
        #[1.84]
    
    Args:
        target_vector: usually it is one-hot vector where the hot bit corresponds to the label index. But it can be any probability distribution over the labels.
        output_vector: the unscaled computed output values from the network
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import CrossEntropyWithSoftmax
    op = CrossEntropyWithSoftmax(target_vector, output_vector, name = name)
    op.rank = 0
    return op

def square_error(target_matrix, output_matrix, name=''):
    """
    This operation computes the sum of the squared difference between elements 
    in the two input matrices. The result is a scalar (i.e., one by one matrix). 
    This is often used as a training criterion node. 
    
    Example:
        >>> C.eval(C.square_error([4., 6.], [2., 1.]))
        #[29.]
        
        >>> C.eval(C.square_error([1., 2.], [1., 2.]))
        #[0.]
    
    Args:
        target_matrix: target matrix, it is usually a one-hot vector where the hot bit corresponds to the label index
        output_matrix: the output values from the network
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import SquareError
    target_matrix = sanitize_input(target_matrix)
    output_matrix = sanitize_input(output_matrix)
    op = SquareError(target_matrix, output_matrix, name = name)
    op.rank = 0
    return op
    
def error_prediction(target_vector, output_vector, name=''):
    """
    This operation computes the prediction error. It finds the index of the highest 
    value in the output_vector and compares it to the actual ground truth label
    (the index of the hot bit in the target vector). The result is a scalar 
    (i.e., one by one matrix). This is often used as an evaluation criterion. 
    It cannot be used as a training criterion though since the gradient is not
    defined for it.
    
    Example:
        >>> C.eval(C.error_prediction([0., 0., 0., 1.], [1., 2., 3., 4.]))
        #[0.]
        
        >>> C.eval(C.error_prediction([0., 0., 1., 0.], [1., 2., 3., 4.]))
        #[1.]
    
    Args:
        target_vector: it is one-hot vector where the hot bit corresponds to the label index
        output_vector: the output values from the network
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import ErrorPrediction
    target_vector = sanitize_input(target_vector)
    output_vector = sanitize_input(output_vector)
    op = ErrorPrediction(target_vector, output_vector, name = name)
    return op

################################################################################
# comparison ops
################################################################################

def less(left, right, name=''):
    """
    Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0. 

    Example:
       >>> C.eval(C.less([41., 42., 43.], [42., 42., 42.]))
         [array([[1., 0., 0.]])]
        
        >>> C.eval(C.eq([-1,0,1], [0]))
        [array([[1., 0., 0.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Less
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Less(left, right, name=name)
    return op

def equal(left, right, name=''):
    """
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise. 

    Example:
        >>> C.eval(C.equal([41., 42., 43.], [42., 42., 42.]))
        [array([[0., 1., 0.]])]
        
        >>> C.eval(C.eq([-1,0,1], [1]))
        [array([[0., 1., 0.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Equal
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Equal(left, right, name=name)
    return op

def greater(left, right, name=''):
    """
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0. 

    Example:
        >>> C.eval(C.greater([41., 42., 43.], [42., 42., 42.]))
        [array([[0., 0., 1.]])]
        
        >>> C.eval(C.greater([-1,0,1], [0]))
        [array([[1., 0., 1.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Greater
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Greater(left, right, name=name)
    return op

def greater_equal(left, right, name=''):
    """
    Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0. 

    Example:
        >>> C.eval(C.greater_equal([41., 42., 43.], [42., 42., 42.]))
        [array([[0., 1., 1.]])]
        
        >>> C.eval(C.greater_equal([-1,0,1], [0]))
        [array([[0., 1., 1.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import GreaterEqual
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = GreaterEqual(left, right, name=name)
    return op

def not_equal(left, right, name=''):
    """
    Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0. 

    Example:
        >>> C.eval(C.not_equal([41., 42., 43.], [42., 42., 42.]))
        [array([[1., 0., 1.]])]
        
        >>> C.eval(C.eq([-1,0,1], [0]))
        [array([[1., 0., 1.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import NotEqual
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = NotEqual(left, right, name=name)
    return op

def less_equal(left, right, name=''):
    """
    Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0. 

    Example:
        >>> C.eval(C.less_equal([41., 42., 43.], [42., 42., 42.]))
        [array([[1., 1., 0.]])]
        
        >>> C.eval(C.eq([-1,0,1], [0]))
        [array([[1., 1., 0.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import LessEqual
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = LessEqual(left, right, name=name)
    return op

################################################################################
# linear ops
################################################################################

# TODO 
def plus(left, right, name=''):
    """
    The output of this operation is the sum of the two input tensors. It supports broadcasting. 
    In case of scalars its backward pass propagates the received gradient. 
    The operator (+) has been overloaded and can equally be used instead of plus()

    Example:
        >>> C.eval(C.plus([1, 2, 3], [4, 5, 6]))
        [array([[ 5.,  7.,  9.]])]
        
        >>> C.eval(C.plus([-5, -4, -3, -2, -1], [10]))
        [array([[ 5.,  6.,  7.,  8.,  9.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    left = sanitize_input(left)
    right = sanitize_input(right)
    from ..cntk_py import Plus
    op = Plus(left, right, "hardcode name")
    return op

def minus(left, right, name=''):
    """
    The output of this operation is left minus right tensor. It supports broadcasting. 
    In case of scalars its backward pass propagates the received gradient. 
    The operator (-) has been overloaded and can equally be used instead of minus()

    Example:
        >>> C.eval(C.minus([1, 2, 3], [4, 5, 6]))
        [array([[-3., -3., -3.]])]
        
        >>> C.eval(C.minus([[1,2],[3,4]], 1))
        [array([[[ 0.,  1.],
                 [ 2.,  3.]]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    from cntk_py import Minus
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Minus(left, right, name=name)
    op.rank = max(op._.rank, op.y.rank)
    return op

def element_times(left, right, name=''):
    """
    The output of this operation is the element-wise product of the two input 
    tensors. It supports broadcasting. In case of scalars its backward pass to left propagates right 
    times the received gradient and vice versa.
    The operator (*) has been overloaded and can equally be used instead of element_times().    
    
    Example:
        >>> C.eval(C.element_times([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]))
        [array([[ 0.5  ,  0.25 ,  0.125,  0.   ]])]
        
        >>> C.eval(C.element_times([5., 10., 15., 30.], [2.]))
        [array([[ 10.,  20.,  30.,  60.]])]
    
    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import ElementTimes
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = ElementTimes(left, right, name=name)
    op.rank = max(op._.rank, op.y.rank)
    return op

def element_divide(left, right, name=''):
    """
    The output of this operation is the element-wise division of the two input 
    tensors. It supports broadcasting. In case of scalars its backward pass to 
    left propagates :math:`1/right` times the received gradient, and the backward 
    pass to right propagates. 
    The operator (/) has been overloaded and can equally be used instead of element_divide().
    :math:`(-left/right^2)` times the received gradient. 
    

    Example:
        >>> C.eval(C.element_divide([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]))
        [array([[ 2.,  4.,  8.,  0.]])]
        
        >>> C.eval(C.element_divide([5., 10., 15., 30.], [2.]))
        [array([[  2.5,   5. ,   7.5,  15. ]])]

    Args:
        left: left side tensor
        right: right side tensor
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import ElementDivide
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = ElementDivide(left, right, name=name)
    op.rank = max(op._.rank, op.y.rank)
    return op

def times(left, right, output_rank=1, name=''):
    """
    The output of this operation is the matrix product of the two input matrices.
    It supports broadcasting. Sparse is supported in the right operand, if it is a matrix.
    The operator '@' has been overloaded such that in Python 3.5 and later X @ W equals times(X, W).

    Example:
        >>> C.eval(C.times([[1,2],[3,4]], [5,6]))
        [array([[ 17.,  39.]])]
        
        >>> C.eval(cntk.times(np.reshape(np.arange(8), (2,2,2)),np.reshape(np.arange(8), (2,2,2)), output_rank=1))        
        [array([[[ 28.,  34.],
        [ 76.,  98.]]])]
        
        >>> C.eval(cntk.times(np.reshape(np.arange(8), (2,2,2)),np.reshape(np.arange(8), (2,2,2)), output_rank=2))        
        [array([[[[[  4.,   5.],
                   [  6.,   7.]],
                  [[ 12.,  17.],
                   [ 22.,  27.]]],
                 [[[ 20.,  29.],
                   [ 38.,  47.]],
                  [[ 28.,  41.],
                   [ 54.,  67.]]]]])]

    Args:
        left: left side matrix or tensor
        right: right side matrix or tensor
        output_rank (int): in case we have tensors as arguemnts, output_rank represents
            the number of axes to be collapsed in order to transform the tensors
            into matrices, perform the operation and then reshape back (explode the axes)
        name (str): the name of the node in the network            

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Times   
    # CNTK uses column vectors and column major representation, thus we reverse
    # params    
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Times(right, left, outputRank=output_rank, name=name)
    op.rank = op.x.rank + op.y.rank - 2    
    return op

def identity(x, name=''):
    """
    The identity function. It returns an identical tensor to the input tensor `x`: 

    :math:`pass_tensor(x) = x`

    Example:
        >>> C.eval(C.pass_tensor([0., 1.]))
        [array([[ 0.      ,  1.]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Identity
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Identity(x, name=name)
    op.rank = op._.rank   
    return op

################################################################################
# non_diff ops
################################################################################


def floor(arg, name=''):
    """
    The output of this operation is the element wise value rounded to the largest 
    integer less than or equal to the input.

    Example:
        >>> C.eval(C.floor([0.2, 1.3, 4., 5.5, 0.0]))
        [array([[ 0.,  1.,  4.,  5.,  0.]])]

        >>> C.eval(C.floor([[0.6, 3.3], [1.9, 5.6]]))
        [array([[[ 0.,  3.],
                 [ 1.,  5.]]])]

        >>> C.eval(C.floor([-5.5, -4.2, -3., -0.7, 0]))
        [array([[-6., -5., -3., -1.,  0.]])]

        >>> C.eval(C.floor([[-0.6, -4.3], [1.9, -3.2]]))
        [array([[[-1., -5.],
                 [ 1., -4.]]])]

    Args:
        arg: input tensor
        name (str): the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Floor
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Floor(arg, name = name)
    op.rank = op._.rank  
    return op

def ceil(arg, name=''):
    """
    The output of this operation is the element wise value rounded to the smallest 
    integer greater than or equal to the input.

    Example:
        >>> C.eval(C.ceil([0.2, 1.3, 4., 5.5, 0.0]))
        [array([[ 1.,  2.,  4.,  6.,  0.]])]
        
        >>> C.eval(C.ceil([[0.6, 3.3], [1.9, 5.6]]))
        [array([[[ 1.,  4.],
                 [ 2.,  6.]]])]

    Args:
        arg: input tensor
        name (str): the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Ceil
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Ceil(arg, name = name)
    op.rank = op._.rank  
    return op

def round(arg, name=''):
    """
    The output of this operation is the element wise value rounded to the nearest integer. 
    In case of tie, where element can have exact fractional part of 0.5
    this operation follows "round half-up" tie breaking strategy.
    This is different from the round operation of numpy which follows
    round half to even.

    Example:
        >>> C.eval(C.round([0.2, 1.3, 4., 5.5, 0.0]))
        [array([[ 0.,  1.,  4.,  6.,  0.]])]

        >>> C.eval(C.round([[0.6, 3.3], [1.9, 5.6]]))
        [array([[[ 1.,  3.],
                 [ 2.,  6.]]])]

        >>> C.eval(C.round([-5.5, -4.2, -3., -0.7, 0]))
        [array([[-5., -4., -3., -1.,  0.]])]

        >>> C.eval(C.round([[-0.6, -4.3], [1.9, -3.2]]))
        [array([[[-1., -4.],
                 [ 2., -3.]]])]

    Args:
        arg: input tensor
        name (str): the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Round
    left = sanitize_input(left)
    right = sanitize_input(right)
    op = Round(arg, name = name)
    op.rank = op._.rank  
    return op

################################################################################
# non_linear and nn ops
################################################################################


def clip(x, min_value, max_value, name=''):
    """
    Computes a tensor with all of its values clipped to fall
    between `min_value` and `max_value`, i.e.
    ``min(max(x, min_value), max_value)``.

    The output tensor has the same shape as `x`.
    
    The backward pass propagates the received gradient if no clipping occurred,
    and 0 if the value was clipped.
    
    Example:
        >>> C.eval(C.clip([1., 2.1, 3.0, 4.1], 2., 4.))
        [array([[ 2. ,  2.1,  3. ,  4. ]])]
        
        >>> C.eval(C.clip([-10., -5., 0., 5., 10.], [-5., -4., 0., 3., 5.], [5., 4., 1., 4., 9.]))
        [array([[-5., -4.,  0.,  4.,  9.]])]
    
    Args:        
        x: tensor to be clipped
        min_value: a scalar or a tensor which represents the minimum value to clip element values to
        max_value: a scalar or a tensor which represents the maximum value to clip element values to
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk_py import Clip
    x = sanitize_input(x)
    op = Clip(x, min_value, max_value, name = name)
    return op

def relu(x, name=''):
    """
    Rectified linear operation. Computes the element-wise rectified linear
    of `x`: ``max(x, 0)``

    The output tensor has the same shape as `x`.

    Example:
        >>> C.eval(C.relu([[-1, -0.5, 0, 1, 2]]))
        [array([[[ 0.,  0.,  0.,  1.,  2.]]])]
    
    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Relu
    x = sanitize_input(x)
    op = Relu(x, name=name)
    return op

def sigmoid(x, name=''):
    """
    Computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    The output tensor has the same shape as `x`.
    
    Example:
        >>> C.eval(C.sigmoid([-2, -1., 0., 1., 2.]))
        [array([[ 0.119203,  0.268941,  0.5     ,  0.731059,  0.880797]])]
    
    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Sigmoid
    x = sanitize_input(x)
    op = Sigmoid(x, name=name)
    return op

def tanh(x, name=''):
    """
    Computes the element-wise tanh of `x`: 

    The output tensor has the same shape as `x`.
    
    Example:
        >>> C.eval(C.tanh([[1,2],[3,4]]))
        [array([[[ 0.761594,  0.964028],
                 [ 0.995055,  0.999329]]])]
    
    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Tanh
    x = sanitize_input(x)
    op = Tanh(x, name=name)
    return op

def softmax(x, name=''):
    """
    Squashes the input values `x` such that they add up to 1: 

    :math:`softmax(x) = {\exp(x_i) - \max_{x_i \in x}(\exp(x_i)) \over {\sum_{x_i \in x} \exp(x_i)- \max_{x_i \in x}(\exp(x_i)) }}`

    The term :math:`\max_{x_i \in x}(\exp(x_i))` is subtracted for numerical
    stability.

    Example:
        >>> C.eval(C.softmax([[1, 1, 2, 3]]))
        [array([[[ 0.082595,  0.082595,  0.224515,  0.610296]]])]

        >>> C.eval(C.softmax([1, 1]))
        [array([[ 0.5,  0.5]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Softmax
    x = sanitize_input(x)
    op = Softmax(x)
    return op

def exp(x, name=''):
    """
    Computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Example:
        >>> C.eval(C.exp([0., 1.]))
        [array([[ 1.      ,  2.718282]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Exp
    x = sanitize_input(x)
    op = Exp(x, name=name)
    return op

def log(x, name=''):
    """
    Computes the element-wise the natural logarithm of `x`: 
    
    Example:
        >>> C.eval(C.log([1., 2.]))
        [array([[ 0.      ,  0.69314718056]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
                
    Note:
        CNTK returns -85.1 for log(x) if `x` is negative or zero. The reason is that 
        it uses 1e-37 (whose natural logarithm is -85.1) as the smallest float 
        number for `log`, because this is the only guaranteed precision across 
        platforms. This will be changed to return `NaN` and `-inf`.
    """
    from cntk_py import Log
    x = sanitize_input(x)
    op = Log(x, name=name)
    return op

def sqrt(x, name=''):
    """
    Computes the element-wise square-root of `x`: 

    :math:`sqrt(x) = {\sqrt[2]{x}}`

    Example:
        >>> C.eval(C.sqrt([0., 4.]))
        [array([[ 0.      ,  2.]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`        
        
    Note:
        CNTK returns zero for sqrt of negative nubmers, this will be changed to 
        retrun NaN
    """
    from cntk_py import Sqrt
    x = sanitize_input(x)
    op = Sqrt(x, name=name)
    return op

def square(x, name=''):
    """
    Computes the element-wise square of `x`:     

    Example:
        >>> C.eval(C.square([1., 10.]))
        [array([[ 1.      ,  100.]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Square
    x = sanitize_input(x)
    op = Square(x, name=name)
    return op

def abs(x, name=''):
    """
    Computes the element-wise absolute of `x`: 

    :math:`abs(x) = |x|`

    Example:
        >>> C.eval(C.abs([-1, 1, -2, 3]))
        [array([[ 1.,  1.,  2.,  3.]])]

    Args:
        x: numpy array or any :class:`cntk.graph.ComputationNode` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk_py import Abs
    x = sanitize_input(x)
    op = Abs(x, name=name)
    return op

def cond(flag, value_if_true, value_if_false, name=''):
    """
    return either value_if_true or value_if_false based on the value of flag.
    If flag != 0 value_if_true is returned, otherwise value_if_false.
    Behaves analogously to numpy.where(...).

    Example:
        >>> C.eval(C.cond([-10, -1, 0, 0.3, 100], [1, 10, 100, 1000, 10000], [ 2, 20, 200, 2000, 20000]))
        [array([[  1.00000000e+00,   1.00000000e+01,   2.00000000e+02,
                   1.00000000e+03,   1.00000000e+04]])]

    Args:
        flag: tensor
        value_if_true: tensor
        value_if_false: tensor
        name (str): the name of the node in the network          
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk_py import If
    flag = sanitize_input(flag)
    op = If(flag, value_if_true, value_if_false, name = name)
    return op
    
################################################################################
# recurrent ops
################################################################################

def future_value(shape, x, time_step=1, default_hidden_activation=0.1, name=''):
    """
    This function returns the future value wrt `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the next logical sample. The `time_step` parameter is the number of steps 
    to look into the future and is 1 by default. If there is no future value (i.e. 
    the current sample is the last one in the tensor) then the `default_hidden_activation` 
    value is returned which is 0.1 by default.
    
    Example:
        >>> data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        >>> t = C.dynamic_axis(name='t')
        >>> x = C.input_numpy([data], dynamic_axis=t)
        >>> with C.LocalExecutionContext('future_value') as ctx:
        ...     print(ctx.eval(C.future_value(0, x)))
        [array([[  5. ,   6. ,   7. ,   8. ],
                [  9. ,  10. ,  11. ,  12. ],
                [  0.1,   0.1,   0.1,   0.1]])]
    
    Args:        
        shape (tuple): dimensions of the input `x`, the shape will be inferred if zero is passed.
        x: the tensor (or its name) from which the future value is obtained. 
        time_step (int): the number of time steps to look into the future (default 1)
        default_hidden_activation (number): the default value to use when no future value is available (default 0.1)
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    from cntk_py import FutureValue
    x = sanitize_input(x)
    op = FutureValue(shape, x, time_step, default_hidden_activation, name = name)
    return op
    
def past_value(shape, x, time_step=1, default_hidden_activation=0.1, name=''):
    """
    This function returns the past value wrt `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the previous logical sample. The `time_step` parameter is the number of steps 
    to look into the past and is 1 by default. If there is no past value (i.e. 
    the current sample is the first one in the tensor) then the `default_hidden_activation` 
    value is returned which is 0.1 by default.
    
    Example:
        >>> data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        >>> t = C.dynamic_axis(name='t')
        >>> x = C.input_numpy([data], dynamic_axis=t)
        >>> with C.LocalExecutionContext('past_value') as ctx:
        ...     print(ctx.eval(C.past_value(0, x)))
        [array([[ 0.1,  0.1,  0.1,  0.1],
                [ 1. ,  2. ,  3. ,  4. ],
                [ 5. ,  6. ,  7. ,  8. ]])]
    
    Args:        
        shape (tuple): dimensions of the input `x`, the shape will be inferred if zero is passed.
        x: the tensor (or its name) from which the past value is obtained
        time_step (int): the number of time steps to look into the past (default 1)
        default_hidden_activation (number): the default value to use when no past value is available (default 0.1)
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    from cntk_py import PastValue
    x = sanitize_input(x)
    op = PastValue(shape, x, time_step, default_hidden_activation, name = name)
    return op

################################################################################
# reshaping ops
################################################################################


def reshape(x, shape, name=''):
    """
    Reinterpret input samples as having different tensor dimensions
    One dimension may be specified as 0 and will be inferred

    The output tensor has the same shape as 'shape'.
    
    The backward pass propagates the received gradient for the output-shape to the input shape.
    
    Examples:
        >>> C.eval(C.reshape([[0,1],[2,3],[4,5]], (2,3)))
        [array([[[ 0.,  4.,  3.],
                 [ 2.,  1.,  5.]]])]
            
    Args:        
        x: tensor to be reshaped
        shape (tuple): a tuple defining the resulting shape
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk_py import NewReshape
    if not np.isscalar(shape):
        # cntk uses column major, thus we reverse the shape    
        shape = tuple(reversed(shape))    
    x = sanitize_input(x)
    op = NewReshape(x, shape, 0, 0, name = name)
    return op
    
def transpose_dimensions(x, axis1, axis2, name=''):
    """
    Reverses two axes of the tensor. The output tensor has the same data but with
    axis1 and axis2 swapped.    
        
    Examples:
        >>> C.eval(C.transpose_dimensions([[0,1],[2,3],[4,5]], 1,2))
        [array([[[ 0.,  4.,  3.],
                 [ 2.,  1.,  5.]]])]
            
    Args:        
        x: tensor to be reshaped
        axis1 (int): the axis to swap with axis2
        axis2 (int): the axis to swap with axis1
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk_py import TransposeDimensions
    x = sanitize_input(x)
    op = TransposeDimensions(x, axis1, axis2, name = name)    
    #cntk uses column major, thus it will read the indices of data passed from 
    # python in reverse
    op.axis1 = abs(axis1) if axis1<0 else op.rank - axis1
    op.axis2 = abs(axis2) if axis2<0 else op.rank - axis2        
    return op

def _slice(x, begin_index, end_index, axis=0, name=''): 
    '''
    Slice the input along an axis.    

    Examples:
        >>> # create 2x3 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.asarray([[[1, 2, -3],
        ...                     [4, 5,  6]]])
        >>> x = C.input_numpy(data)
        >>> # slice index 1 (second) at first axis
        >>> C.eval(C.slice(x, 1, 2, 0))
        [array([[[ 4.,  5.,  6.]]])]
        >>> # slice index 0 (first) at second axis
        >>> C.eval(C.slice(x, 0, 1, 1))
        [array([[[ 1.],
                 [ 4.]]])]        

    NumPy's way of slicing works, too:

    Examples:
        >>> C.eval(x[1])
        [array([[[ 4.,  5.,  6.]]])]
        >>> C.eval(x[:,:2,:])
        [array([[[ 1.,  2.],
                 [ 4.,  5.]]])]

    Args:
        x: input tensor
        begin_index (int): the index along axis where the slicing starts
        end_index (int): the index along axis where the slicing ends
        axis (int or str): axis along which `begin_index` and `end_index` will be used. If axis is of type `str` then the time axis will be used.
        name (str): the name of the node in the network
        
    See also:
        Indexing in NumPy: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    from cntk_py import Slice
    x = sanitize_input(x)
    op = Slice(x, begin_index, end_index, axis, name=name)
    
    #cntk uses column major, thus it will read the indices of data passed from 
    # python in reverse
    if isinstance(axis, str):
        op.axis = -1 # time axis
    else:
        op.axis = abs(axis) if axis<0 else op.rank - axis
        
    return op
    
def splice(inputs, axis=0, name=''): 
    '''
    Concatenate the input tensors along an axis.    

    Examples:
        >>> # create 2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data1 = np.asarray([[[1, 2],
        ...                     [4, 5]]])
        >>> x = C.input_numpy(data1)
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data2 = np.asarray([[[10, 20],
        ...                     [30, 40],
        ...                     [50, 60]]])
        >>> y = C.input_numpy(data2)
        >>> # splice both inputs on axis=0 returns a 5x2 matrix
        >>> C.eval(C.splice((x,y), 0))
        [array([[[1, 2],
                 [4, 5],
                 [10, 20],
                 [30, 40],
                 [50, 60]]])]        

    Args:
        inputs (list): tuple of input tensors
        axis (int): axis along which the concatenation will be performed
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    from cntk_py import Splice
    inputs = sanitize_input(inputs)
    op = Splice(inputs, axis, name=name)

    #cntk uses column major, thus it will read the indices of data passed from 
    # python in reverse
    op.axis = abs(axis) if axis<0 else op.rank - axis

    # Splice is implemented using BrainScript code that results in nested nodes,
    # if it gets tag='output' the file name might differ depending on the execution
    # path in BS. Thus we wrap it by Identity to have a fixed name.
    return identity(op) 

################################################################################
# reduction ops
################################################################################

def reduce_sum(x, axis=0, name=''): 
    '''
    Computes the sum of the input tensor's elements across one axis. if `axis==rank`,
    then the sum will be computed over all axes, that is, the output is a scalar,
    which is the sum of tensor's elements.

    Examples:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[10, 20],[30, 40],[50, 60]]        
        
        >>> # reduce over the first axis
        >>> C.eval(C.reduce_sum(data, 0))
        [array([[[  90.,  120.]]])]     
        
        >>> # reduce over the second axis
        >>> C.eval(C.reduce_sum(data, 1))
        [array([[[  30.],
                 [  70.],
                 [ 110.]]])]        
        
        >>> # reduce over the all axes
        >>> C.eval(C.reduce_sum(data, 2))
        [array([[ 210.]])]       

    Args:
        x: input tensor
        axis (int): axis along which the reduction will be performed
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    from cntk_py import ReduceSum
    x = sanitize_input(x)
    op = ReduceSum(x, axis, name=name)
    # cntk uses column major, thus it will read the indices of data passed from 
    # python in reverse
    return op

def reduce_log_sum(inputs, name=''): 
    '''
    Computes the log sum of the input tensor's elements. The output is a scalar,
    which is the log sum of tensor's elements.

    Examples:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[10, 20],[30, 40],[50, 60]]        
        
        >>> # reduce over the all axes
        >>> C.eval(C.reduce_sum(data))
        [array([[ 60.000046]])]       

    Args:
        x: input tensor        
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    from cntk_py import ReduceLogSum
    inputs = sanitize_input(inputs)
    op = ReduceLogSum(inputs, 0, name=name)
    #TODO: Once axis != 0 is supported, expose it as argument, and compute the 
    #rank similar to reduce_sum
    #reduce_log_sum is implemented using BrainScript code that results in nested nodes,
    #We wrap it by Identity to guarantee that the tag 'output' is passed over and with a fixed name.
    return identity(op) 


################################################################################
# training ops
################################################################################
 
def dropout(x, name=''):
    """
    Compute a new tensor with `dropoutRate` perecent set to zero. The values 
    that are set to zero are randomly chosen. This is commonly used to prevent 
    overfitting during the training process.

    The output tensor has the same shape as `x`, but with `dropoutRate` of the
    elements set to zero (droped out).
            
    Args:        
        x: source tensor
        name (str): the name of the node in the network
                
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk_py import Dropout
    x = sanitize_input(x)
    op = Dropout(x, name = name)
    return op

################################################################################
# variables_and_parameters ops
################################################################################


def input(shape, dynamic_axis='', data_type=None, needs_gradient=True, name=''):
    """
    It creates an input node. The graph requires a separate reader that will be
    fed to this input.

    Args:
        shape (tuple): the shape of the input tensor
        dynamic_axis (str or output of :func:`cntk.ops.dynamic_axis`): the dynamic axis
        name (str): the name of the node in the network
        
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    if dynamic_axis:
        raise NotImplemented

    from .variables import Variable
    if not np.isscalar(shape):
        # cntk uses column major, thus we reverse the shape    
        shape = tuple(reversed(shape))

    # TODO dynamic axis
    op = Variable(shape, data_type, needs_gradient, name=name)
    
    return op

def sparse_input_numpy(indices, values, shape, alias=None, dynamic_axis='', name=''):
    '''
    Creates an input node from a sparse input tensors described by a list of indices
    and a list of values having a shape. The tensors represent one
    sample and can have sequences of different lengths. 

    Example:

        >>>
        # Creating a dense matrix 
        # [[ 10, 20]
        #  [ 30, 40]]
        # Note that we need to specify a batch of samples of sequences (all
        # having sequence length 1 in this example).
        >>> dense = C.input_numpy([[[10,20,30], 
                                    [40,50,60]]])
        # Creating a sparse array 
        # [0, 0.1, 0]
        >>> sparse = C.sparse_input_numpy(indices=[(1,)], values=[(0.1,)], shape=(3,))
        >>> C.eval(C.times(dense, sparse))
        [array([[ 2.,  5.]])]
        #  Creating a sparse matrix
        # [[0  ], 
           [0.1],
           [0  ]]
        >>> sparse = C.sparse_input_numpy(indices=[(1,)], values=[(0.1,)], shape=(3,1))
        >>> C.eval(C.times(dense, sparse))
        [array([[[ 2.],
                 [ 5.]]])]

    Args:
        indices (list): list (batch) of tuples (indices), which are positions of the values after flattening the tensor with `order='F'`
        values (list): list (batch) of tuples of values corresponding to indices
        shape (tuple): shape of the input
        alias (str): alias to be used in the data file
        dynamic_axis (str): whether the tensor has already the data
        alias (str): optional the alias to be used when serializing the data into an intermediate file
        name (str): the name of the node in the network
                
    Returns:
        :class:`cntk.graph.ComputationNode`
    '''

    op = sparse_input(shape, dynamic_axis=dynamic_axis, name=name)
    from ..reader import LazySparseInputReader
    op.reader = LazySparseInputReader(
        indices,
        values,
        shape,
        input_alias=alias,
        dynamic_axis=dynamic_axis,
        node=op)
    
    return op

def sparse_input(shape, dynamic_axis='', name=''):
    """
    It creates a sparse input node. The graph requires a separate reader that will be
    fed to this input.

    Args:
        shape (tuple): the shape of the input tensor
        dynamic_axis (str or output of :func:`cntk.ops.dynamic_axis`): the dynamic axis
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    from cntk_py import SparseInput
    if not np.isscalar(shape):
        # cntk uses column major, thus we reverse the shape    
        shape = tuple(reversed(shape))
    op = SparseInput(shape, dynamicAxis=dynamic_axis, name=name)
    op.rank = get_rank(shape)
    return op

def parameter(shape=None, value=None, learning_rate_multiplier=1.0,
        init_from_file_path=None, name=''):
    """
    It creates a parameter tensor. 

    Args:
        shape (tuple or int, optional): the shape of the input tensor. If not provided, it will be inferred from ``value``.
        value (scalar or NumPy array, optional): a scalar initial value that would be replicated for every element in the tensor or NumPy array. If ``None``, the tensor will be initialized uniformly random.
        learning_rate_multiplier (float): set to control the learning rate on this particular node
        init_from_file_path (str): the file that contains the initial tensor value. Used only if ``value=None``.
        name (str, optional): the name of the node in the network

    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    from variables import Parameter
    return Parameter(shape, value)
        
    """
    To be as generic as possible, we 
     - flatten the data 
     - initialize a ParameterTensor operator with it
     - ensure that the graph does not backprob to it.  
     - Finally we to reshape it.
    """

    from .. import utils
    if not (np.isscalar(value) or utils.is_tensor(value)):
        raise ValueError('value type is not supported: %s' % type(value))

    if isinstance(value, list) or np.isscalar(value):
        value = np.asarray(value)

    import scipy.sparse
    if scipy.sparse.issparse(value):
        raise ValueError('only dense data is supported')

    param_shape = value.shape if value.shape else (1,)
    
    # cntk uses column major, thus we reverse the shape    
    param_shape = tuple(reversed(param_shape))
    literal_shape = (param_shape[0], np.multiply.reduce(param_shape[1:]))

    # cntk expects data in reverse order, thus we transpose first 
    transposed_val = np.transpose(value)
    literal_array = np.reshape(transposed_val, literal_shape, order = 'F')    

    from io import BytesIO
    s = BytesIO()
    np.savetxt(s, literal_array, '%.4f')

    op = cntk1.ParameterTensor(
        dims=param_shape,
        learningRateMultiplier=learning_rate_multiplier,
        init='fromLiteral',
        initFromLiteral=s.getvalue().decode())

    op.rank = get_rank(param_shape)
    return op

def constant(value, name='', data_type=None, dev=None):
    """
    It creates a constant tensor initialized from a numpy array

    Args:
        value: the tensor constant passed as numpy array
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from .variables import Constant

    if dev is None:
        dev = cntk_py.DeviceDescriptor_CPUDevice()

    return Constant(value=value, name=name, data_type=data_type, dev=dev)

def dynamic_axis(name=''):
    """
    This function creates a dynamic axis object that can be connected to an input. 
    For sequence-based inputs, this allows the sequences to be of arbitrary lengths 
    and therefore allows networks to be setup without the need for padding.
    
    Example:
        See Examples/LSTM/seqcla.py for a use of :func:`cntk.ops.dynamic_axis`.
    
    Args:
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    
    from cntk_py import DynamicAxis
    op = DynamicAxis(name=name)
    op.rank = None
    return op


def reconcile_dynamic_axis(data_input, layout_input, name=''):
    """
    This function adapts the dynamic axis layout for `data_input` to match that 
    of `layout_input`. It allows these two tensors to be properly compared using, e.g. 
    a criterion node.
    
    Example:
        See Examples/LSTM/seqcla.py for a use of :func:`cntk.ops.reconcile_dynamic_axis`.    
    
    Args:
        data_input: the tensor to have its dynamic axis layout adapted
        layout_input: the tensor layout to use for adapting `data_input`s layout
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    
    from cntk_py import ReconcileDynamicAxis
    op = ReconcileDynamicAxis(data_input, layout_input, name=name)
    op.rank = data_input.rank
    return op

