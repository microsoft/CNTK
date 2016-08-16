# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..utils import sanitize_input, get_data_type, cntk_device
from cntk import cntk_py

def combine(operands, name=''):
    """
     Create a new Function instance which just combines the outputs of the specified list of 
     'operands' Functions such that the 'Outputs' of the new 'Function' are union of the
     'Outputs' of each of the specified 'operands' Functions. E.g. When creating a classification
     model, typically the CrossEntropy loss Function and the ClassificationError Function comprise
     the two roots of the computation graph which can be "Combine"d to create a single Function
     with 2 outputs; viz. CrossEntropy loss and ClassificationError output.    
    Args:
        operands (list): list of functions to combine
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import combine
    return combine(operands, name)

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
        target_vector: usually it is one-hot vector where the hot bit corresponds to the label index. 
        But it can be any probability distribution over the labels.
        output_vector: the unscaled computed output values from the network
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import cross_entropy_with_softmax
    target_vector = sanitize_input(target_vector, get_data_type(output_vector))
    output_vector = sanitize_input(output_vector, get_data_type(target_vector))
    return cross_entropy_with_softmax(target_vector, output_vector, name)

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import square_error
    target_matrix = sanitize_input(target_matrix, get_data_type(output_matrix))
    output_matrix = sanitize_input(output_matrix, get_data_type(target_matrix))
    return square_error(target_matrix, output_matrix, name)

def classification_error(target_vector, output_vector, name=''):
    """
    This operation computes the prediction error. It finds the index of the highest 
    value in the output_vector and compares it to the actual ground truth label
    (the index of the hot bit in the target vector). The result is a scalar 
    (i.e., one by one matrix). This is often used as an evaluation criterion. 
    It cannot be used as a training criterion though since the gradient is not
    defined for it.
    
    Example:
        >>> C.eval(C.classification_error([0., 0., 0., 1.], [1., 2., 3., 4.]))
        #[0.]
        
        >>> C.eval(C.classification_error([0., 0., 1., 0.], [1., 2., 3., 4.]))
        #[1.]
    
    Args:
        target_vector: it is one-hot vector where the hot bit corresponds to the label index
        output_vector: the output values from the network
        name (str): the name of the node in the network            
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import classification_error
    target_vector = sanitize_input(target_vector, get_data_type(output_vector))
    output_vector = sanitize_input(output_vector, get_data_type(target_vector))
    return classification_error(target_vector, output_vector, name)    

################################################################################
# convolution ops
################################################################################

def convolution(convolution_map, operand, strides=[1], sharing=[True], 
                auto_padding=[True], lower_pad=[0], upper_pad=[0], transpose=False, 
                max_temp_mem_size_in_samples=0, name=''):
    """
    TODO: 
    Args:        
        convolution_map:
        operand:
        strides:
        sharing:
        auto_padding:
        lower_pad:
        upper_pad:
        transpose:
        max_temp_mem_size_in_samples:
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import convolution
    operand = sanitize_input(operand)    
    return convolution(convolution_map, operand, strides, sharing, auto_padding, 
                        lower_pad, upper_pad, transpose, max_temp_mem_size_in_samples,
                        name) 

MAX_POOLING=1
AVG_POOLING=2

def pooling(operand, pooling_type, pooling_window_shape, strides=[1], auto_padding=[False], 
            lower_pad=[0], upper_pad=[0], name=''):
    """
    TODO: 
    Args:                
        operand:
        pooling_type:   
        pooling_window_shape:
        strides:
        auto_padding:
        lower_pad:
        upper_pad:
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import pooling
    operand = sanitize_input(operand)    
    return pooling(operand, pooling_type, pooling_window_shape, strides, auto_padding,
                    lower_pad, upper_pad, name) 

def batch_normalization(operand, scale, bias, running_mean, running_inv_std, special,
                        normalization_time_constant=0, blend_time_constant=0,
                        epsilon=0.00001, use_cudnn_engine=False, name=''):
    """
    TODO: 
    Args:                
        operand:
        scale:   
        bias:
        running_mean:
        running_inv_std:
        special:
        normalization_time_constant:
        blend_time_constant:
        epsilon:
        use_cudnn_engine:
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import batch_normalization
    operand = sanitize_input(operand)    
    return batch_normalization(operand, scale, bias, running_mean, running_inv_std, special,
                                normalization_time_constant, blend_time_constant,
                                epsilon, use_cudnn_engine, name) 

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import less
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return less(left, right, name)    

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import equal
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return equal(left, right, name)    

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import greater
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return greater(left, right, name)    

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import greater_equal
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return greater_equal(left, right, name)    

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import not_equal
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return not_equal(left, right, name)    

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import less_equal
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return less_equal(left, right, name)    

################################################################################
# linear ops
################################################################################

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import plus
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))   
    return plus(left, right, name)    

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
        :class:`cntk_py.Function`
    """

    from ..cntk_py import minus
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return minus(left, right, name)        

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import element_times
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return element_times(left, right, name)        

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import element_divide
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return element_divide(left, right, name)        

#TODO: PerDimMeanVarianceNormalize

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import times      
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return times(right, left, output_rank, name)        

#TOOD: enable when it is exposed in c++
def identity(x, name=''):
    """
    The identity function. It returns an identical tensor to the input tensor `x`: 

    :math:`pass_tensor(x) = x`

    Example:
        >>> C.eval(C.pass_tensor([0., 1.]))
        [array([[ 0.      ,  1.]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    raise NotImplementedError("identity is not implemented yet in V2")

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import floor
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return floor(arg, name)    

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import ceil
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return ceil(arg, name)

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
        :class:`cntk_py.Function`
    """
    from ..cntk_py import round
    left = sanitize_input(left, get_data_type(right))
    right = sanitize_input(right, get_data_type(left))
    return round(arg, name)

################################################################################
# non_linear and nn ops
################################################################################

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    """    
    raise NotImplementedError("clip is not implemented yet in V2")

def relu(x, name=''):
    """
    Rectified linear operation. Computes the element-wise rectified linear
    of `x`: ``max(x, 0)``

    The output tensor has the same shape as `x`.

    Example:
        >>> C.eval(C.relu([[-1, -0.5, 0, 1, 2]]))
        [array([[[ 0.,  0.,  0.,  1.,  2.]]])]
    
    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import relu
    x = sanitize_input(x)
    return relu(x, name)    

def sigmoid(x, name=''):
    """
    Computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    The output tensor has the same shape as `x`.
    
    Example:
        >>> C.eval(C.sigmoid([-2, -1., 0., 1., 2.]))
        [array([[ 0.119203,  0.268941,  0.5     ,  0.731059,  0.880797]])]
    
    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import sigmoid
    x = sanitize_input(x)
    return sigmoid(x, name)    

def tanh(x, name=''):
    """
    Computes the element-wise tanh of `x`: 

    The output tensor has the same shape as `x`.
    
    Example:
        >>> C.eval(C.tanh([[1,2],[3,4]]))
        [array([[[ 0.761594,  0.964028],
                 [ 0.995055,  0.999329]]])]
    
    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import tanh
    x = sanitize_input(x)
    return tanh(x, name)    

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
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import softmax
    x = sanitize_input(x)
    return softmax(x)    

def exp(x, name=''):
    """
    Computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Example:
        >>> C.eval(C.exp([0., 1.]))
        [array([[ 1.      ,  2.718282]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import exp
    x = sanitize_input(x)
    return exp(x, name)    

def log(x, name=''):
    """
    Computes the element-wise the natural logarithm of `x`: 
    
    Example:
        >>> C.eval(C.log([1., 2.]))
        [array([[ 0.      ,  0.69314718056]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
                
    Note:
        CNTK returns -85.1 for log(x) if `x` is negative or zero. The reason is that 
        it uses 1e-37 (whose natural logarithm is -85.1) as the smallest float 
        number for `log`, because this is the only guaranteed precision across 
        platforms. This will be changed to return `NaN` and `-inf`.
    """
    from ..cntk_py import log
    x = sanitize_input(x)
    return log(x, name)    

def sqrt(x, name=''):
    """
    Computes the element-wise square-root of `x`: 

    :math:`sqrt(x) = {\sqrt[2]{x}}`

    Example:
        >>> C.eval(C.sqrt([0., 4.]))
        [array([[ 0.      ,  2.]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`        
        
    Note:
        CNTK returns zero for sqrt of negative nubmers, this will be changed to 
        retrun NaN
    """
    from ..cntk_py import sqrt
    x = sanitize_input(x)
    return sqrt(x, name)    

def square(x, name=''):
    """
    Computes the element-wise square of `x`:     

    Example:
        >>> C.eval(C.square([1., 10.]))
        [array([[ 1.      ,  100.]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import square
    x = sanitize_input(x)
    return square(x, name)    

def abs(x, name=''):
    """
    Computes the element-wise absolute of `x`: 

    :math:`abs(x) = |x|`

    Example:
        >>> C.eval(C.abs([-1, 1, -2, 3]))
        [array([[ 1.,  1.,  2.,  3.]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import abs
    x = sanitize_input(x)
    return abs(x, name)    

def negate(x, name=''):
    """
    Computes the element-wise negation of `x`: 

    :math:`abs(x) = -x`

    Example:
        >>> C.eval(C.abs([-1, 1, -2, 3]))
        [array([[ 1.,  -1.,  2.,  -3.]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import negate
    x = sanitize_input(x)
    return negate(x, name)    

def reciprocal(x, name=''):
    """
    Computes the element-wise reciprocal of `x`: 

    Example:
        >>> C.eval(C.abs([-1/3, 1/5, -2, 3]))
        [array([[ -3.,  5.,  -1/2.,  1/3.]])]

    Args:
        x: numpy array or any :class:`cntk_py.Function` that outputs a tensor
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from ..cntk_py import reciprocal
    x = sanitize_input(x)
    return reciprocal(x, name)    

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    """    
    raise NotImplementedError("cond is not implemented yet in V2")
    
################################################################################
# recurrent ops
################################################################################

def future_value(initial_state, x, time_step=1, name=''):
    """
    This function returns the future value w.r.t. `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the next logical sample. The `time_step` parameter is the number of steps 
    to look into the future and is 1 by default. If there is no future value (i.e. 
    the current sample is the last one in the tensor) then the `initial_state` 
    value is returned.
    
    Example:
        TBA    
    Args:        
        initial_state: tensor or scalar representing the initial value to be
        used when the input tensor is shifted in time.
        x: the tensor (or its name) from which the future value is obtained. 
        time_step (int): the number of time steps to look into the future (default 1)        
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """    
    
    from ..cntk_py import future_value
    x = sanitize_input(x)
    return future_value(initial_state, x, time_step, name)    
    
def past_value(initial_state, x, time_step=1, default_hidden_activation=0.1, name=''):
    """
    This function returns the past value w.r.t. `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the previous logical sample. The `time_step` parameter is the number of steps 
    to look into the past and is 1 by default. If there is no past value (i.e. 
    the current sample is the first one in the tensor)  then the `initial_state` 
    value is returned.
    
    Example:
        TBA
    Args:        
        initial_state: tensor or scalar representing the initial value to be
        used when the input tensor is shifted in time.
        x: the tensor (or its name) from which the past value is obtained
        time_step (int): the number of time steps to look into the past (default 1)        
        name (str): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """    
    
    from ..cntk_py import past_value
    x = sanitize_input(x)
    return past_value(initial_state, x, time_step, name)    

################################################################################
# reshaping ops
################################################################################

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    """    
    raise NotImplementedError("reshape is not implemented yet in V2")

#TOOD: enable when it is exposed in c++  
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
        :class:`cntk_py.Function`
    """    
    raise NotImplementedError("transpose_dimensions is not implemented yet in V2")

#TOOD: enable when it is exposed in c++
def slice(x, begin_index, end_index, axis=0, name=''): 
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
        :class:`cntk_py.Function`
    '''
    raise NotImplementedError("slice is not implemented yet in V2")

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    '''
    raise NotImplementedError("splice is not implemented yet in V2") 

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
        :class:`cntk_py.Function`
    '''
    from ..cntk_py import reduce_sum
    x = sanitize_input(x)
    return reduce_sum(x, axis, name)    

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    '''
    raise NotImplementedError("reduce_log_sum is not implemented yet in V2")


################################################################################
# training ops
################################################################################

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    """    
    raise NotImplementedError("dropout is not implemented yet in V2")

################################################################################
# variables_and_parameters ops
################################################################################

def variable(shape, data_type=None, needs_gradient=True, name=''):
    """
    It creates an input node. The graph requires a separate reader that will be
    fed to this input.

    Args:
        shape (tuple): the shape of the input tensor     
        data_type: np.float32 or np.float64
        needs_gradients (bool): whether to back-propagates to it or not
        name (str): the name of the node in the network
        
    Returns:
        :class:`cntk_py.Function`
    """
    from .variables import Variable

    # TODO dynamic axis
    # TODO Sparse
    return Variable(shape, data_type, needs_gradient, name)

def placeholder(shape, name=''):
    """
    It creates a variable place holder for recurrence networks, when the network's dynamic axes
    are unfolded, the place holder will get assigned a variable along the correspondent dynamic axis.

    Args:
        shape (tuple): the shape of the variable tensor             
        name (str): the name of the node in the network
        
    Returns:
        :class:`cntk_py.Function`
    """
    from .variables import Placeholder

    # TODO dynamic axis?    
    return Placeholder(shape, name)
    
def parameter(shape=None, value=None, device_id=-1, name=''):
    """
    It creates a parameter tensor. 

    Args:
        shape (tuple or int, optional): the shape of the input tensor. If not provided, it will be inferred from ``value``.
        value (scalar or NumPy array, optional): a scalar initial value that would be replicated for every element in the tensor or NumPy array. 
        If ``None``, the tensor will be initialized uniformly random.
        device_id (int): device id, -1 for CPU, 0 or higher for GPU                
        name (str, optional): the name of the node in the network

    Returns:
        :class:`cntk_py.Function`
    """    
    #TODO: add swig support to this templatized constructor and reflect the change in this method:
    #   template<typename ElemType>
    #   Parameter(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), const std::wstring& name = L"")
    #       : Variable(shape, VariableKind::Parameter, AsDataType<ElemType>(), MakeSharedObject<NDArrayView>(initValue, shape, device), true, {}, name)

    from .variables import Parameter    
    return Parameter(shape, value, None, cntk_device(device_id), name)        

def constant(shape=None, value=None, device_id=-1, name=''):
    """
    It creates a constant tensor initialized from a numpy array

    Args:
        shape (tuple or int, optional): the shape of the input tensor. If not provided, it will be inferred from ``value``.
        value (scalar or NumPy array, optional): a scalar initial value that would be replicated for every element in the tensor or NumPy array. 
        If ``None``, the tensor will be initialized uniformly random.
        device_id (int): device id, -1 for CPU, 0 or higher for GPU                
        name (str, optional): the name of the node in the network
    Returns:
        :class:`cntk_py.Function`
    """
    from .variables import Constant, constant_from_scalar

    #TODO: add swig support to this templatized constructor and reflect the change in this method:
    #   template<typename ElemType>
    #   Constant(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), const std::wstring& name = L"")
    #       : Variable(shape, VariableKind::Constant, AsDataType<ElemType>(), MakeSharedObject<NDArrayView>(initValue, shape, device), true, {}, name)

    if np.isscalar(value):        
        return constant_from_scalar(shape, value, None, cntk_device(device_id), name)   
    return Constant(shape, value, None, cntk_device(device_id), name)

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    """
    
    raise NotImplementedError("dynamic_axis is not implemented yet in V2")

#TOOD: enable when it is exposed in c++
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
        :class:`cntk_py.Function`
    """
    
    raise NotImplementedError("reconcile_dynamic_axis is not implemented yet in V2")


