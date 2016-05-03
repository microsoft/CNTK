# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np

################################################################################
# convolution ops
################################################################################

################################################################################
# evaluation ops
################################################################################

def cross_entropy_with_softmax(target_vector, output_vector, name=None):
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
        target_vector: usually it is one-hot vector where the hot bit 
        corresponds to the label index. But it can be any probability distribution
        over the labels.
        output_vector: the unscaled computed output values from the network
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk1 import CrossEntropyWithSoftmax
    return CrossEntropyWithSoftmax(target_vector, output_vector, name = name)

def square_error(target_matrix, output_matrix, name=None):
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
        target_matrix: target matrix, it is usually a one-hot vector where the 
        hot bit corresponds to the label index
        output_matrix: the output values from the network
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk1 import SquareError
    return SquareError(target_matrix, output_matrix, name = name)

def error_prediction(target_vector, output_vector, name=None):
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
        target_vector: it is one-hot vector where the hot bit corresponds to the 
        label index
        output_vector: the output values from the network
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import ErrorPrediction
    return ErrorPrediction(target_vector, output_vector, name = name)


################################################################################
# linear ops
################################################################################

def plus(left, right, name=None):
    """
    Tensor addition operation. The output of this operation is the sum of the 
    two input tensors. It supports broadcasting. In case of scalars its backward
    pass propagates the received gradient. 

    Example:
        >>> C.eval(C.plus([1, 2, 3], [4, 5, 6]))
        [array([[ 5.,  7.,  9.]])]
        
        >>> C.eval(C.plus([-5, -4, -3, -2, -1], [10]))
        [array([[ 5.,  6.,  7.,  8.,  9.]])]

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Plus
    return Plus(left, right, name=name)


def minus(left, right, name=None):
    """
    Tensor subtraction operation. The output of this operation is left minus
    right tensor. It supports broadcasting. In case of scalars its backward
    pass propagates the received gradient. 

    Example:
        >>> C.eval(C.minus([1, 2, 3], [4, 5, 6]))
        [array([[-3., -3., -3.]])]
        
        >>> C.eval(C.minus([[1,2],[3,4]], 1))
        [array([[[ 0.,  1.],
                 [ 2.,  3.]]])]

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    from cntk.ops.cntk2 import Minus
    return Minus(left, right, name=name)


def element_times(left, right, name=None):
    """
    Element-wise multiplication operation. The output of this operation is the
    element-wise product of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left propagates right 
    times the received gradient and vice versa.
    
    Example:
        >>> C.eval(C.element_times([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]))
        [array([[ 0.5  ,  0.25 ,  0.125,  0.   ]])]
        
        >>> C.eval(C.element_times([5., 10., 15., 30.], [2.]))
        [array([[ 10.,  20.,  30.,  60.]])]
    
    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import ElementTimes
    return ElementTimes(left, right, name=name)


def element_divide(left, right, name=None):
    """
    Element-wise division operation. The output of this operation is the
    element-wise division of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left propagates :math:`1/right` 
    times the received gradient, and the backward pass to right propagates 
    :math:`(-left/right^2)` times the received gradient. 

    Example:
        >>> C.eval(C.element_divide([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]))
        [array([[ 2.,  4.,  8.,  0.]])]
        
        >>> C.eval(C.element_divide([5., 10., 15., 30.], [2.]))
        [array([[  2.5,   5. ,   7.5,  15. ]])]

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import ElementDivide
    return ElementDivide(left, right, name=name)


def times(left, right, name=None):
    """
    Tensor times operation. The output of this operation is the
    tensor product of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left propagates right
    times the received gradient and vice versa.

    Example:
        >>> C.eval(C.times([[1,2],[3,4]], [5,6]))
        [array([[ 17.,  39.]])]
        
        >>> C.eval(C.times([[1,2],[3,4],[5,6]], [[0.5,0.25],[0.25,0.5]]))
        [array([[[ 1.  ,  1.25],
                 [ 2.5 ,  2.75],
                 [ 4.  ,  4.25]]])]

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Times
    return Times(left, right, name=name)


################################################################################
# non_diff ops
################################################################################


def floor(arg, name=None):
    """
    Floor operation. The output of this operation is the
    element wise value rounded to the largest integer less than
    or equal to the input.

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
        name: the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Floor
    return Floor(arg, name = name)


def ceil(arg, name=None):
    """
    Ceil operation. The output of this operation is the
    element wise value rounded to the smallest integer greater than
    or equal to the input.

    Example:
        >>> C.eval(C.ceil([0.2, 1.3, 4., 5.5, 0.0]))
        [array([[ 1.,  2.,  4.,  6.,  0.]])]
        
        >>> C.eval(C.ceil([[0.6, 3.3], [1.9, 5.6]]))
        [array([[[ 1.,  4.],
                 [ 2.,  6.]]])]

    Args:
        arg: input tensor
        name: the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Ceil
    return Ceil(arg, name = name)


def round(arg, name=None):
    """
    Round operation. The output of this operation is the
    element wise value rounded to the nearest integer. In case
    of tie, where element can have exact fractional part of 0.5
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
        name: the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Round
    return Round(arg, name = name)


################################################################################
# non_linear ops
################################################################################


def clip(x, min_value, max_value, name=None):
    """
    Clip operation. Computes a tensor with all of its values clipped to fall
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
        min_value: the minimum value to clip element values to
        max_value: the maximum value to clip element values to
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk.ops.cntk2 import Clip
    return Clip(x, min_value, max_value, name = name)


def relu(x, name=None):
    """
    Rectified linear operation. Computes the element-wise rectified linear
    of `x`: ``max(x, 0)``

    The output tensor has the same shape as `x`.

    Example:
        >>> C.eval(C.relu([[-1, -0.5, 0, 1, 2]]))
        [array([[[ 0.,  0.,  0.,  1.,  2.]]])]
    
    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Relu
    return Relu(x, name=name)


def sigmoid(x, name=None):
    """
    Sigmoid operation. Computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    The output tensor has the same shape as `x`.
    
    Example:
        >>> C.eval(C.sigmoid([-2, -1., 0., 1., 2.]))
        [array([[ 0.119203,  0.268941,  0.5     ,  0.731059,  0.880797]])]
    
    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Sigmoid
    return Sigmoid(x, name=name)


def tanh(x, name=None):
    """
    Tanh operation. Computes the element-wise tanh of `x`: 

    The output tensor has the same shape as `x`.
    
    Example:
        >>> C.eval(C.tanh([[1,2],[3,4]]))
        [array([[[ 0.761594,  0.964028],
                 [ 0.995055,  0.999329]]])]
    
    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Tanh
    return Tanh(x, name=name)


def softmax(x, name=None):
    """
    Softmax operation. Squashes the input values `x` such that they add up to 1: 

    :math:`softmax(x) = {\exp(x_i) - \max_{x_i \in x}(\exp(x_i)) \over {\sum_{x_i \in x} \exp(x_i)- \max_{x_i \in x}(\exp(x_i)) }}`

    The term :math:`\max_{x_i \in x}(\exp(x_i))` is subtracted for numerical
    stability.

    Example:
        >>> C.eval(C.softmax([[1, 1, 2, 3]]))
        [array([[[ 0.082595,  0.082595,  0.224515,  0.610296]]])]

        >>> C.eval(C.softmax([1, 1]))
        [array([[ 0.5,  0.5]])]

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Softmax
    return Softmax(x)


def exp(x, name=None):
    """
    Exp operation. Computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Example:
        >>> C.eval(C.exp([0., 1.]))
        [array([[ 1.      ,  2.718282]])]

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk1 import Exp
    return Exp(x, name=name)


def abs(x, name=None):
    """
    Abs operation. Computes the element-wise absolute of `x`: 

    :math:`abs(x) = |x|`

    Example:
        >>> C.eval(C.abs([-1, 1, -2, 3]))
        [array([[ 1.,  1.,  2.,  3.]])]

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    from cntk.ops.cntk2 import Abs
    return Abs(x, name=name)


def cond(flag, value_if_true, value_if_false, name=None):
    """
    Return either value_if_true or value_if_false based on the value of flag.
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
        name: the name of the node in the network          
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk.ops.cntk1 import If
    return If(flag, value_if_true, value_if_false, name = name)


################################################################################
# recurrent ops
################################################################################




def future_value(dims, x, time_step=1, default_hidden_activation=0.1, name=None):
    """
    This function returns the future value wrt `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the next logical sample. The `time_step` parameter is the number of steps 
    to look into the future and is 1 by default. If there is no future value (i.e. 
    the current sample is the last one in the tensor) then the `default_hidden_activation` 
    value is returned which is 0.1 by default.
    
    Example:
        >>> future_value(0, [[1, 2], [3, 4], [5,6]], 1, 0.5)
        # [[3, 4], [5, 6], [0.5, 0.5]]
    
    Args:        
        dims: dimensions of the input `x`
        x: the tensor from which the future value is obtained
        time_step: the number of time steps to look into the future (default 1)
        default_hidden_activation: the default value to use when no future value 
        is available (default 0.1)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    from cntk.ops.cntk1 import FutureValue
    return FutureValue(dims, x, time_step, default_hidden_activation, name = name)
    
def past_value(dims, x, time_step=1, default_hidden_activation=0.1, name=None):
    """
    This function returns the past value wrt `x`. It is most often used when 
    creating RNNs. The resulting tensor has the same shape as the input but is 
    the previous logical sample. The `time_step` parameter is the number of steps 
    to look into the past and is 1 by default. If there is no past value (i.e. 
    the current sample is the first one in the tensor) then the `default_hidden_activation` 
    value is returned which is 0.1 by default.
    
    Example:
        >>> past_value(0, [[1, 2], [3, 4], [5,6]], 1, 0.5)
        # [[0.5, 0.5], [1, 2], [3, 4]]
    
    Args:        
        dims: dimensions of the input `x`
        x: the tensor from which the past value is obtained
        time_step: the number of time steps to look into the past (default 1)
        default_hidden_activation: the default value to use when no past value 
        is available (default 0.1)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    from cntk.ops.cntk1 import PastValue
    return PastValue(dims, x, time_step, default_hidden_activation, name = name)


################################################################################
# reshaping ops
################################################################################


def reshape(x, shape, name=None):
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
        shape: a tuple defining the resulting shape
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    from cntk.ops.cntk1 import NewReshape
    return NewReshape(x, shape, 0, 0, name = name)

################################################################################
# training ops
################################################################################

################################################################################
# variables_and_parameters ops
################################################################################

def input_numpy(value, alias=None, has_dynamic_axis=None, name=None):
    '''
    Creates an input node from a list of tensors. The tensors represent one
    sample and can have sequences of different lengths. 

    Example:
        >>> C.eval(C.input_numpy(np.ones((3, 2))))
        [array([[ 1.,  1.]]), array([[ 1.,  1.]]), array([[ 1.,  1.]])]

    Args:
        value (list): list of tensors potentially having sequences of different lengths.
        alias (str): alias to be used in the data file
        has_dynamic_axis (bool): If True, the outermost dimension is treated as the dynamic axis. If False, it will wrap each sample into its own 1-dimensional array.
        alias (str): optional the alias to be used when serializing the data into an intermediate file
    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    from .. import utils
    if utils.is_tensor_list(value) or utils.is_tensor(value):
        value = np.asarray(value)
        if has_dynamic_axis:
            cntk_shape = value[0].shape[1:]
        else:
            cntk_shape = value[0].shape

        node = input(cntk_shape)
        from ..reader import LazyInputReader
        node.reader = LazyInputReader(
            value,
            input_alias=alias,
            has_dynamic_axis=has_dynamic_axis,
            node=node)

        return node
    else:
        raise ValueError('value type is not supported: %s' % type(value))


def input(shape, dynamic_axis='', name=None):
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

    from cntk.ops.cntk1 import Input
    return Input(shape, dynamicAxis=dynamic_axis, name=name)


def parameter(shape=None, value=0, learning_rate_multiplier=1.0, init='uniform',
              init_value_scale=1, init_from_file_path='', init_from_literal=None,
              random_seed=-1, name=None):
    """
    It creates a parameter tensor. 

    Args:
        shape (tuple or int): the shape of the input tensor. If `init='fromLiteral'`, shape is not needed as it will be inferred from the literal.
        value: a scalar initial value that would be replicated for every element in the tensor
        learning_rate_multiplier (float): 
        init (str): 'uniform', 'fromFile' or 'fromLiteral' 
        init_value_scale (float): a scaling factor for the initial value
        init_from_file_path (str): the file that contains the initial tensor value
        init_from_literal (ndarray): the numpy array used to initialize the tensor parameter
        random_seed (float): the seed used for initialization
        name (str, optional): the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    from . import cntk1

    # if the parameter is initialized from a literal value
    if (init == 'fromLiteral'):
        """
        To be as generic as possible, we 
         - flatten the data 
         - initialize a ParameterTensor operator with it
         - ensure that the graph does not backprob to it.  
         - Finally we to reshape it.
        """

        value = init_from_literal

        from .. import utils
        if not (np.isscalar(value) or utils.is_tensor(value)):
            raise ValueError('value type is not supported: %s' % type(value))

        if isinstance(value, list) or np.isscalar(value):
            value = np.asarray(value)

        import scipy.sparse
        if scipy.sparse.issparse(value):
            raise ValueError('only dense data is supported')

        param_shape = value.shape if value.shape else (1,)
        literal_shape = (param_shape[0], np.multiply.reduce(param_shape[1:]))

        literal_array = np.reshape(value, literal_shape)

        from io import BytesIO
        s = BytesIO()
        np.savetxt(s, literal_array, '%.4f')

        return cntk1.ParameterTensor(
            dims=param_shape,
            learningRateMultiplier=learning_rate_multiplier,
            init='fromLiteral',
            initFromLiteral=s.getvalue().decode())

    else:
        return cntk1.ParameterTensor(shape, learning_rate_multiplier, init,
                                     init_value_scale, value, init_from_file_path,
                                     randomSeed=random_seed, name=name)


def constant(value, name=None):
    """
    It creates constant tensor initialized from a numpy array

    Args:
        value: the tensor constant passed as numpy array
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return parameter(name=name, init='fromLiteral', init_from_literal=value,
                     learning_rate_multiplier=0.0)


def dynamic_axis(name=None):
    """
    This function creates a dynamic axis object that can be connected to an input. 
    For sequence-based inputs, this allows the sequences to be of arbitrary lengths 
    and therefore allows networks to be setup without the need for padding.
    
    Example:
        See Examples/LSTM/seqcla.py for a use of :func:`cntk.ops.dynamic_axis`.
    
    Args:
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    
    from cntk.ops.cntk2 import DynamicAxis
    return DynamicAxis(name=name)


def reconcile_dynamic_axis(data_input, layout_input, name=None):
    """
    This function adapts the dynamic axis layout for `data_input` to match that 
    of `layout_input`. It allows these two tensors to be properly compared using, e.g. 
    a criterion node.
    
    Args:
        data_input: the tensor to have its dynamic axis layout adapted
        layout_input: the tensor layout to use for adapting `data_input`s layout
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    
    from cntk.ops.cntk1 import ReconcileDynamicAxis
    return ReconcileDynamicAxis(data_input, layout_input, name=name)
