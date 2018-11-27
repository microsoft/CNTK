import cntk
from cntk.ops.functions import BlockFunction
from cntk.variables import Parameter
from cntk.ops import times
from cntk.internal import _as_tuple
from cntk.layers.blocks import _initializer_for, _INFERRED, identity
from cntk.layers.blocks import UntestedBranchError  # helpers
from cntk.default_options import is_default_override
from cntk.default_options import get_default_override, default_override_or

def svd_subprojection(matrix, k):
    '''
    Calculate svd of the matrix and produce a subprojection based on k

    Args:
        matrix : an input matrix        
        k (int): desired rank of the output matrix

    Returns:
        two matrices representing the original matrix after svd and 
        reducing them based on k.
    '''
    
    import numpy as np
    from numpy import dot, diag
    from numpy.linalg import svd

    # Decompose W into (U, s, V)
    U, s, V = svd(matrix, full_matrices=False)
          
    # Create two dense layers from this; one that takes U, one that takes
    # dot(s, V), but restrict them all to rank k, such that the result is a
    # k-rank subprojection
    W1 = np.ascontiguousarray(U[:, :k])
    W2 = dot(diag(s[:k]), V[:k, :])
   
    return W1, W2


def factor_dense(model, projection_function = None, filter_function = None, 
                 factor_function = None):
    '''
    Reduce the size of a dense model using the provided factor_function 
    and the projection_function. filter_function is used to select dense 
    layers to apply the reduction. If no factor_function is specified, 
    use svd decomposition. 

    Args:
        model               : dense model.
        projection_function : determin the new size of the dense model. It can 
                              be based on the shape of the weight matrix or 
                              other heuristics.
                              factor_function can choose to ignore the value k.
        filter_function     : filter layers in the model to apply the factorization
        factor_function     : factor the dense model (e.g. svd)   
                
    Returns:
        a model that is factored and reduced in size.
    '''
    if (factor_function == None and projection_function == None):
        raise ValueError("Dense: default factor function (svd) requires a projection_function.")
    
    dense_filter = (lambda x: type(x) == cntk.Function 
                                            and x.op_name == 'Dense' 
                                            and x.is_block
                                            and (filter_function(x) if filter_function else True))
   
    def dense_converter(model):        
        W, b = model.W.value, model.b.value

        ht, wdth = W.shape        
        # k is the rank of the output matrices. If a projection function is 
        # provided, then use it, otherwise assign min of two dimensions of
        # W to k.
        k = projection_function(W) if projection_function else min(ht, wdth)
        W1, W2 = factor_function(W, k) if factor_function else svd_subprojection(W, k)

        Ws = {'W1': W1, 'W2': W2}
        dfl = dense_factored((int(k), int(wdth)),
            init=Ws,
            activation=None,
            init_bias=b,
            name='DenseFactored')(model.inputs[2])
        return dfl

    return cntk.misc.convert(model, dense_filter, dense_converter)


def dense_factored(shapes, #(shape1, shape2)
                  activation=default_override_or(identity),
                  init={'W1':None, 'W2':None},
                  input_rank=None,
                  map_rank=None,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  name=''):
    '''
    Perform the new model creation using the factored inputs W1 and W2. 
    The returend function represents the new model.

    Args:
        shapes                  : dimensions of the input matrices.
        activation              : activation function used for the model.
        init                    : the two matrices corresponding to the factorization.
        input_rank              : rank of the input tensor.
        map_rank                : ???
        bias                    : bias for the model.
        init_bias               : initial bias value.
        name                    : name of the block function that creates the new model.
        
    Returns:
        a model that is factored and projected (reduced).
    '''

    # matthaip: Not sure how to handle input tensor of rank > 1
    # or selective flattening of ranks
    assert(input_rank is None and
           map_rank is None and
           all(isinstance(s,int) for s in list(shapes)))

    activation = get_default_override(cntk.layers.Dense, activation=activation)
    bias       = get_default_override(cntk.layers.Dense, bias=bias)
    init_bias  = get_default_override(cntk.layers.Dense, init_bias=init_bias)
    # how to use get_default_override for init parameeter?

    output_shape1 = _as_tuple(shapes[0])
    output_shape2 = _as_tuple(shapes[1])
    if input_rank is not None and map_rank is not None:
        raise ValueError("Dense: input_rank and map_rank cannot be specified at the same time.")


    # If input_rank not given then pass a single _INFERRED; 
    # map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED

    # parameters bound to this Function
    #    init_weights = _initializer_for(init, Record(output_rank=output_rank))
    init_weights = init
    W1 = Parameter(input_shape + output_shape1, init=init_weights['W1'], name='W1')
    W2 = Parameter(output_shape1 + output_shape2, init=init_weights['W2'], name='W2')
    b = Parameter(output_shape2, init=init_bias,    name='b') if bias else None

    # expression of this function
    @BlockFunction('DenseFactored', name)
    def dense(x):
        r = times(x, W1)
        r = times(r, W2)
        if b:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r
    return dense

# Reference for sklearn.tucker.hooi:
# https://hal.inria.fr/hal-01219316/document
