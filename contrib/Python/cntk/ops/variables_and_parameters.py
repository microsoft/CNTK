# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


"""
This modules contains the interfaces to Inputs and Parameters. 
"""

import numpy as np
import scipy.sparse
from ..reader import CNTKTextFormatReader
from .. import utils


def input_reader(value, alias=None, has_dynamic_axis=True):
    '''
    Creates an input node from a list of tensors. The tensors represent one
    sample and can have sequences of different lengths. 

    Args:
        value (list): list of tensors potentially having sequences of different lengths.
        alias (str): alias to be used in the data file
        has_dynamic_axis (bool): If True, the outermost dimension is treated as the dynamic axis. If False, it will wrap each sample into its own 1-dimensional array.
        alias (str): optional the alias to be used when serializing the data into an intermediate file

    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    if utils.is_tensor_list(value) or utils.is_tensor(value):
        if has_dynamic_axis:
            cntk_shape = value[0][1:]
        else:
            cntk_shape = value[0]

        from ..ops import cntk1
        from ..reader import LazyInputReader
        node = cntk1.Input(cntk_shape)
        node.reader = LazyInputReader(
            value,
            input_alias=alias,
            has_dynamic_axis=has_dynamic_axis,
            node=node)

        return node
    else:
        raise ValueError('value type is not supported: %s' % type(value))


def input(dims, name=None):
    """
    It creates an input node. The graph requires a separate reader that will be
    fed to this input.

    Args:
        dims: the shape of the input tensor
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Input(dims, var_name=name)


def parameter(dims=None, name=None, learning_rate_multiplier=1.0, init='uniform',
              init_value_scale=1, value=0, init_from_file_path='', init_from_literal=None,
              random_seed=-1):
    """
    It creates a parameter tensor. 

    Args:
        dims (shape or int): the shape of the input tensor. If `init='fromLiteral'`, dims is not 
        needed as it will be inferred from the literal.
        name (str, optional): the name of the node in the network
        learning_rate_multiplier (float): 
        init (str): 'uniform', 'fromFile' or 'fromLiteral' 
        init_value_scale (float): a scaling factor for the initial value
        value: a scalar initial value that would be replicated for every element in the tensor
        init_from_file_path (str): the file that contains the initial tensor value
        init_from_literal (ndarray): the numpy array used to initialize the tensor parameter
        random_seed (float): the seed used for initialization
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

        if not (np.isscalar(value) or utils.is_tensor(value)):
            raise ValueError('value type is not supported: %s' % type(value))

        if isinstance(value, list) or np.isscalar(value):
            value = np.asarray(value)

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
        return cntk1.ParameterTensor(dims, learning_rate_multiplier, init,
                                     init_value_scale, value, init_from_file_path,
                                     randomSeed=random_seed, var_name=name)


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
