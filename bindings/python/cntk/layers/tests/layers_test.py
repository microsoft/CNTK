# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import cntk as C
from cntk import Axis, reshape, sigmoid, element_max, Function, BlockFunction, Constant, greater, default_options, default_options_for, \
                 get_default_override, default_override_or
from cntk.layers import SequentialConvolution, Convolution, Convolution1D, Convolution2D, Convolution3D, Dense, Embedding, Fold, For, \
                        MaxPooling, MaxUnpooling, LSTM, GRU, RNNStep, Sequential, Stabilizer, Dropout, Recurrence, \
                        RecurrenceFrom, LayerNormalization, ConvolutionTranspose
from cntk.layers.typing import Sequence, Signature, Tensor, SequenceOver

import pytest

# Note: We do not test gradients here, assuming that those are tested elsewhere.
# Forward outputs are tested to verify that the structure of the layer is as expected.

def test_layers_name():
    from cntk import placeholder
    I = placeholder(name='input')
    p = Dense(10, name='dense10')(I)

    assert(p.name == 'dense10')
    assert(I.name == 'input')
    assert(p.root_function.name == 'dense10')

    q = Convolution((3, 3), 3, name='conv33')(I)
    assert(q.name == 'conv33')
    assert(q.root_function.name == 'conv33')

    e = Embedding(0, name='emb')(I)
    assert(e.name == 'emb')
    assert(e.root_function.name == 'emb')

    e = Embedding(0, name='')(I)
    assert(e.name == '')
    assert(e.root_function.name == '')

def assert_list_of_arrays_equal(r, exp, err_msg):
    for r_i, exp_i in zip(r, exp): # note: must compare seq by seq due to differing lengths
        np.testing.assert_array_equal(r_i, exp_i, err_msg=err_msg)

####################################
# default options
####################################

def test_default_options():
    def Test(some_param=default_override_or(13)):
        some_param = get_default_override(Test, some_param=some_param)
        return some_param
    assert Test() == 13
    assert Test(42) == 42
    with default_options(some_param=1968):
        assert Test() == 1968
        assert Test(some_param=1976) == 1976
        with default_options_for(Test, some_param=2017):
            assert Test() == 2017
            assert Test(some_param=123) == 123
    with default_options_for(test_default_options, some_param=2017): # some other function (misusing test_default_options() as a placeholder)
        assert Test() == 13  # tests that default value does not apply since it is set for a different function
        assert Test(some_param=124) == 124

####################################
# @Function, @BlockFunction, types
####################################

def test_Function():

    ####################################################
    # Test 1: BlockFunction()
    ####################################################
    @BlockFunction('Square', 'block_name')
    def f(x):
        return x * x
    assert f.shape == (-2,)
    assert f.root_function.op_name == 'Square'
    assert f.name == 'block_name'

    ####################################################
    # Test 2: Function() with shapes and type
    ####################################################
    # For 2.7 compat, use @Function @Signature instead.
    # TODO: Once we no longer need to support 2.7, change back to @Function.
    #@Function
    #def g(x : Tensor[3,2]):
    @Function
    @Signature(Tensor[3,2])
    def g(x):
        return x * x
    assert g.shape == (3,2)
    r = g([[[2, 1], [5, 2], [1, 3]]])
    e = [np.array([[2, 1], [5, 2], [1, 3]]) ** 2]
    assert_list_of_arrays_equal(r, e, err_msg='@Function test failed')

    ####################################################
    # Test 3: Function() with shapes and type; short-cut
    ####################################################
    @Function.with_signature(Tensor[3,2])
    def g(x):
        return x * x
    assert g.shape == (3,2)
    r = g([[[2, 1], [5, 2], [1, 3]]])
    e = [np.array([[2, 1], [5, 2], [1, 3]]) ** 2]
    assert_list_of_arrays_equal(r, e, err_msg='@Function.with_signature test failed')

####################################
# . syntax for name lookup
####################################

def test_lookup():
    model = Sequential([ Dense(3, init=1, name='first'), Dense(2, init=2, name='second')])
    model.update_signature((2,))
    W1 = model.first.W.value
    W2 = model.second.W.value
    np.testing.assert_array_equal(W1, np.ones((2,3)),     err_msg='Error in lookup of Dense parameters')
    np.testing.assert_array_equal(W2, np.ones((3,2)) * 2, err_msg='Error in lookup of Dense parameters')

####################################
# Recurrence(), RecurrenceFrom()
####################################

def test_recurrence():
    inputAxis = Axis('inputAxis')
    stateAxis = Axis('stateAxis')
    InputSequence = SequenceOver[inputAxis]
    StateSequence = SequenceOver[stateAxis]

    # input and expected for both tests below
    x = np.reshape(np.arange(0,25, dtype=np.float32), (1,5,5))
    exp = [[ 0.239151, 0.239151, 0.239151, 0.239151, 0.239151],
           [ 0.338713, 0.338713, 0.338713, 0.338713, 0.338713],
           [ 0.367456, 0.367456, 0.367456, 0.367456, 0.367456],
           [ 0.375577, 0.375577, 0.375577, 0.375577, 0.375577],
           [ 0.377891, 0.377891, 0.377891, 0.377891, 0.377891]]

    ####################################################
    # Test 1: Recurrence(): initial state is constant
    ####################################################
    # Note: We cannot use random init of the GRU parameters because random numbers will
    # depend on what previous tests were run. Hence, use a constant (which is not realistic).
    # TODO: Find out how to reset the random generator, then remove the constant init.
    R = Recurrence(GRU(5, init=0.05), go_backwards=False, initial_state=0.1)
    @Function
    @Signature(InputSequence[Tensor[5]])
    def F(x):
        return R(x)
    rt = F(x)
    np.testing.assert_array_almost_equal(rt[0], exp, decimal=6, err_msg='Error in Recurrence(GRU()) forward')

    ####################################################
    # Test 2: RecurrenceFrom(): initial state is data input
    ####################################################
    RF = RecurrenceFrom(GRU(5, init=0.05), go_backwards=False)
    @Function
    @Signature(s = StateSequence[Tensor[5]], x = InputSequence[Tensor[5]])
    def FF(s, x):
        return RF(s, x)
    s = np.ones((1,5,5)) * 0.1 # we pass the same value as the constant in the previous test to make the result the same
    rt = FF(s, x)
    np.testing.assert_array_almost_equal(rt[0], exp, decimal=6, err_msg='Error in RecurrenceFrom(GRU()) forward')

def test_recurrence_step_fun():
    import cntk as C
    def step_f(prev1, x):
        return prev1 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, x):
        return prev1 * x, prev2 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, x):
        return prev1 * x, prev2 * x, prev3 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, prev5, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, prev5, prev6, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x, prev6 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, prev5, prev6, prev7, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x, prev6 * x, prev7 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, prev5, prev6, prev7, prev8, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x, prev6 * x, prev7 * x, prev8 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, prev5, prev6, prev7, prev8, prev9, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x, prev6 * x, prev7 * x, prev8 * x, prev9 * x
    rec = Recurrence(step_f)

    def step_f(prev1, prev2, prev3, prev4, prev5, prev6, prev7, prev8, prev9, prev10, x):
        return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x, prev6 * x, prev7 * x, prev8 * x, prev9 * x, prev10 * x
    rec = Recurrence(step_f)

    with pytest.raises(ValueError):
        def step_f(prev1, prev2, prev3, prev4, prev5, prev6, prev7, prev8, prev9, prev10, prev11, x):
            return prev1 * x, prev2 * x, prev3 * x, prev4 * x, prev5 * x, prev6 * x, prev7 * x, prev8 * x, prev9 * x, prev10 * x, prev11 * x
        rec = Recurrence(step_f)

    with pytest.raises(TypeError):
        v = C.input_variable((1), name='additional_input_variable')
        step_f = lambda prev, x: prev * v * x
        rec = Recurrence(step_f)

    with pytest.raises(TypeError):
        def step_f(prev1, x):
            p = C.Parameter((1))
            return prev1 * x * p
        rec = Recurrence(step_f)



####################################
# recurrence (Fold()) over regular function
####################################

def test_recurrence_fun():
    from cntk.layers import Recurrence
    from cntk.ops import plus

    ####################################################
    # Test 1: sum-reduction over sequence
    ####################################################
    r = Fold(plus)
    r.update_signature(Sequence[Tensor[1]])
    data = [np.array([[2], [6], [4], [8], [6]])]   # simple sequence
    out = r(data)
    exp = [sum(data[0])]
    np.testing.assert_array_equal(out, exp, err_msg='Error in recurrence over plus')

    ####################################################
    # Test 2: max-pool over sequence
    ####################################################
    r = Fold(element_max)
    r.update_signature(Sequence[Tensor[2]])
    data = [np.array([[2,1], [6,3], [4,2], [8,1], [6,0]])]   # simple sequence
    out = r(data)
    exp = [np.max(data[0], axis=0)]
    np.testing.assert_array_equal(out, exp, err_msg='Error in recurrence over element_max')

####################################
# UnfoldFrom()
####################################

def test_unfold():
    from cntk.layers import UnfoldFrom

    @Function
    def double_up(s):
        return s * 2
    x = [[[0],[0],[0]],
         [[0],[0],[0],[0],[0]]]

    ####################################################
    # Test 1: simple unfold
    ####################################################
    UF = UnfoldFrom(double_up)
    @Function
    @Signature(Sequence[Tensor[1]])
    def FU(x):
        return UF(Constant(1), x)
    r = FU(x)
    exp = [[ 2. , 4. , 8.],
	       [ 2. , 4. , 8. , 16. , 32. ]]
    assert_list_of_arrays_equal(r, exp, err_msg='Error in UnfoldFrom() forward')

    ####################################################
    # Test 2: unfold with length increase and terminating condition
    ####################################################
    UF = UnfoldFrom(double_up, until_predicate=lambda x: greater(x, 63), length_increase=1.6)
    @Function
    @Signature(Sequence[Tensor[1]])
    def FU(x):
        return UF(Constant(1), x)
    r = FU(x)
    exp = [[ 2 , 4 , 8 , 16 , 32 ],         # tests length_increase
           [ 2 , 4 , 8 , 16 , 32 , 64 ]] # tests early cut-off due to until_predicate

    assert_list_of_arrays_equal(r, exp, err_msg='Error in UnfoldFrom(..., until_predicate, length_increase, ...) forward')

####################################
# Test LSTM recurrence
####################################


RECURRENT_BLOCK_DATA = [ # block_type, block_outputs_count, block_size, W_mult, H_mult, outputs_count
                   # expected_res
                  (LSTM, 2, 5, 4, 4,
                   [[ 0.21532 , 0.21532 , 0.21532 , 0.21532 , 0.21532 ],
                    [ 0.760161, 0.760161, 0.760161, 0.760161, 0.760161],
                    [ 0.95975 , 0.95975 , 0.95975 , 0.95975 , 0.95975 ],
                    [ 0.993661, 0.993661, 0.993661, 0.993661, 0.993661]]),
                  (GRU, 1, 5, 3, 2,
                   [[ 0.1903  , 0.1903  , 0.1903  , 0.1903  , 0.1903  ],
                    [ 0.262537, 0.262537, 0.262537, 0.262537, 0.262537],
                    [ 0.276712, 0.276712, 0.276712, 0.276712, 0.276712],
                    [ 0.279545, 0.279545, 0.279545, 0.279545, 0.279545]]),
                  (RNNStep, 1, 5, 1, 1,
                   [[ 0.645656, 0.645656, 0.645656, 0.645656, 0.645656],
                    [ 0.925727, 0.925727, 0.925727, 0.925727, 0.925727],
                    [ 0.986114, 0.986114, 0.986114, 0.986114, 0.986114],
                    [ 0.997249, 0.997249, 0.997249, 0.997249, 0.997249]]),
]

@pytest.mark.parametrize("block_type, block_outputs_count, block_size, W_mult, H_mult, expected_res", RECURRENT_BLOCK_DATA)
def test_recurrent_block(block_type, block_outputs_count, block_size, W_mult, H_mult, expected_res):
    input_shape = 4

    sequenceAxis = Axis('sequenceAxis')

    y = C.input_variable(input_shape, dynamic_axes=[Axis.default_batch_axis(), sequenceAxis])
    data = np.reshape(np.arange(0,16, dtype=np.float32), (1,4,4))

    rnn_block = block_type(block_size, init=0.1)

    assert len(rnn_block.outputs) == block_outputs_count
    rnn_net = Recurrence(rnn_block)(y)

    assert rnn_net.b.shape == (W_mult*block_size,)
    assert rnn_net.W.shape == (input_shape, W_mult*block_size)
    assert rnn_net.H.shape == (block_size, H_mult*block_size)

    res = rnn_net.eval(data)
    expected = np.asarray(expected_res, dtype=np.float32)

    np.testing.assert_array_almost_equal(res[0], expected, decimal=6)

####################################
# Test dense layer for correctness
####################################

def test_layers_dense(device_id):
    y = C.input_variable(2)
    dat = np.array([[-1., 1.]], dtype=np.float32)

    ####################################################
    # Test 1: no activation
    ####################################################
    p = Dense(2, activation=None, name='foo')(y)
    res = p(y).eval({y: dat})

    npout = np.matrix(dat[0]) * p.foo.W.value + p.foo.b.value

    np.testing.assert_array_equal(res, npout, err_msg='Error in dense layer')

    ####################################################
    # Test 2: with activation
    ####################################################
    p = Dense(2, activation=sigmoid, name='foo')(y)
    res = p(y).eval({y: dat})

    def _sigmoid(x):
        return 1./(1 + np.exp(-x))

    npout = _sigmoid(np.matrix(dat[0]) * p.foo.W.value + p.foo.b.value)

    np.testing.assert_array_almost_equal(res, npout, decimal=7, err_msg='Error in dense layer with sigmoid')

    ####################################################
    # Test 3: 2-dense layer
    ####################################################
    p = Dense(3, activation=None, name='foo')(y)
    q = Dense(3, activation=None, name='bar')(p)
    res = q(y).eval({y: dat})

    npout1 = np.matrix(dat[0]) * p.foo.W.value + p.foo.b.value
    npout = npout1 * q.bar.W.value + q.bar.b.value

    np.testing.assert_array_almost_equal(res, npout, decimal=7, err_msg='Error in 2-dense layer')

    ####################################################
    # Test 4: Failing configuration
    ####################################################

    with pytest.raises(ValueError):
        Dense(2, input_rank=1, map_rank=1) # input_rank and map_rank can be specified at the same time

########################################
# Test Embedding layer for correctness
########################################
def test_layers_embedding():
    embDim = 3
    y = C.input_variable(2)

    # embedding base case
    e = Embedding(shape=embDim, name='foo')

    dat = np.array([[-1., 1.]], dtype=np.float32)
    res = e(y).eval({y: dat})

    npout = np.matrix(dat[0]) * e.E.value
    np.testing.assert_array_equal(res, npout, err_msg='Error in embedding layer')

    # embedding, initialized from a user-supplied starting point for the parameter
    e = Embedding(embDim, init=[[1, 3, 2], [3, 4, 1]], name='bar')

    dat = np.array([[-1., 1.]], dtype=np.float32)
    res = e(y).eval({y: dat})

    npout = np.matrix(dat[0]) * e.E.value
    np.testing.assert_array_equal(res, npout, err_msg='Error in constant embedding layer')

    # embedding, initialized from a user-supplied constant weight table
    e = Embedding(weights=[[1, 3, 2], [3, 4, 1]], name='baz')

    dat = np.array([[-1., 1.]], dtype=np.float32)
    res = e(y).eval({y: dat})

    npout = np.matrix(dat[0]) * e.E.value
    np.testing.assert_array_equal(res, npout, err_msg='Error in constant embedding layer')

    # Failing calls
    with pytest.raises(ValueError):
        Embedding(shape=None, init=1, weights=[1., 2., 3.])

    with pytest.raises(ValueError):
        Embedding(3, weights=[1., 2., 3.])

    with pytest.raises(ValueError):
        Embedding(name="embedding")

########################################
# Test Convolutional layer for shape correctness
########################################

def _getConvOutShape(inDim, kernelDim, zeroPad, strides):
    # First we tackle unit stride
    i, k, s = inDim, kernelDim, strides

    if zeroPad == False:
        if s == 1:
            return inDim - kernelDim + 1
        elif s > 1:
            return int(np.floor((i - k)/s) + 1)
        else:
            raise ValueError("Stride must be a non-zero positive number")
    else: #Zero padding == True
        if s == 1:
            return i
        elif s > 1:
            p = k//2
            return int(np.floor((i + 2 * p -k)/s) + 1)
        else:
            raise ValueError("Stride must be a non-zero positive number")

def test_layers_convolution_shape():
    # Get the output shape
    # i: input dimension
    # k: kernel dimension
    # p: number of zero padding
    # s: strides
    inC, inH, inW = 2, 6, 7
    y = C.input_variable((inC, inH, inW))
    in_filter_shape = (3, 2)
    out_num_filters = 4

    ##########################################################
    # Test convolutional layer for correctness (p=False s = 1)
    ##########################################################
    zeropad = False
    in_strides = 1

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    # shape should be
    model_shape = model(y).foo.shape

    expected_shape = (out_num_filters,
                      _getConvOutShape(inH, in_filter_shape[0], zeropad, in_strides),
                      _getConvOutShape(inW, in_filter_shape[1], zeropad, in_strides))

    np.testing.assert_array_equal(model_shape, expected_shape, \
        "Error in convolution with stride = 1 with no padding")

    ############################################################
    # Test convolutional layer for correctness (p=True s = (3,2))
    ############################################################
    zeropad = False
    in_strides_t = (2, 3)

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides_t, name='foo')
    # shape should be
    model_shape = model(y).foo.shape

    expected_shape = (out_num_filters,
                      _getConvOutShape(inH, in_filter_shape[0], zeropad, in_strides_t[0]),
                      _getConvOutShape(inW, in_filter_shape[1], zeropad, in_strides_t[1]))

    np.testing.assert_array_equal(model_shape, expected_shape, \
        "Error in convolution with stride>1 with no padding")

    ##########################################################
    # Test convolutional layer for correctness (pad=True s = 1)
    ##########################################################
    zeropad = True
    in_strides = 1

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    # shape should be
    model_shape = model(y).foo.shape

    expected_shape = (out_num_filters,
                      _getConvOutShape(inH, in_filter_shape[0], zeropad, in_strides),
                      _getConvOutShape(inW, in_filter_shape[1], zeropad, in_strides))

    np.testing.assert_array_equal(model_shape, expected_shape, \
        "Error in convolution with stride = 1 and padding")

    ##########################################################
    # Test convolutional layer for correctness (pad=True s = 2)
    ##########################################################
    zeropad = True
    in_strides = 2

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')

    # shape should be
    model_shape = model(y).foo.shape

    expected_shape = (out_num_filters,
                      _getConvOutShape(inH, in_filter_shape[0], zeropad, in_strides),
                      _getConvOutShape(inW, in_filter_shape[1], zeropad, in_strides))

    np.testing.assert_array_equal(model_shape, expected_shape, \
        "Error in convolution with stride > 1 and padding")

def test_layers_convolution_value():
    # Common parameters
    inC, inH, inW = 3, 10, 10
    in_filter_shape = (3, 3)
    out_num_filters = 1
    dat = np.ones([1, inC, inH, inW], dtype = np.float32)

    ##########################################################
    # Test convolutional layer for correctness (p=False s = 1)
    ##########################################################
    y = C.input_variable((inC, inH, inW))
    zeropad = False
    in_strides = 1

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = np.sum(model.foo.W.value)

    np.testing.assert_array_almost_equal(res[0][0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution computation with stride = 1 and zeropad = False")

    ##########################################################
    # Test convolutional layer for correctness (p=False s = 2)
    ##########################################################
    zeropad = False
    in_strides = 2

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = np.sum(model.foo.W.value)

    np.testing.assert_array_almost_equal(res[0][0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution computation with stride = 2 and zeropad = False")

    ##########################################################
    # Test convolutional layer for correctness (p=True s = 1)
    ##########################################################
    zeropad = True
    in_strides = 1

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = np.sum(model.foo.W.value)

    # Compare the center of the res with the sum of the weights
    np.testing.assert_array_almost_equal(res[0][0][1][1], expected_res, decimal=6, \
        err_msg="Error in convolution computation with stride = 1 and zeropad = True")

    ##########################################################
    # Test convolutional layer for second invocation/parameter sharing
    ##########################################################
    y1 = C.input_variable((inC, inH, inW))
    res = model(y1).eval({y1: dat}) # this re-clones 'model'

    # Extract the W weight matrix
    expected_res = np.sum(model.foo.W.value)

    # Compare the center of the res with the sum of the weights
    np.testing.assert_array_almost_equal(res[0][0][1][1], expected_res, decimal=6, \
        err_msg="Error in convolution computation with stride = 1 and zeropad = True, second invocation")

    ##########################################################
    # Test convolutional layer for correctness (p=True s = 2)
    ##########################################################
    zeropad = True
    in_strides = 2

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = np.sum(model.foo.W.value[0,:,1:,1:])

    # Compare at the top-left corner, to see the effect of zero-padding.
    np.testing.assert_array_almost_equal(res[0][0][0][0], expected_res, decimal=5,
        err_msg="Error in convolution computation with stride = 2 and zeropad = True")

def test_convolution_consistency_in_different_evals():
    inC, inH, inW = 1,4,4

    y = C.input_variable((inC,inH, inW))

    cMap = 1

    dat = np.arange(0,16, dtype=np.float32).reshape(1,1,4,4)

    conv = Convolution((2,2), cMap, pad=False, activation=None, name='foo' )(y)

    first_eval_result = conv(dat)

    np.testing.assert_array_almost_equal(conv(dat), first_eval_result, decimal=5,
        err_msg="Error in convolution consistency, different results for two runs")

def test_failing_convolution():
    with pytest.raises(ValueError):
        conv = Convolution((3,3), 1)
        conv.update_signature(5)

##########################################################
# Test convolutional 3D layer for correctness (p=False s = 1)
##########################################################
def test_layers_convolution_3d():
    inC, inH, inW, inD = 1, 3, 3, 3
    y = C.input_variable((inC,inH, inW, inD))
    dat = np.ones([1, inC, inH, inW, inD], dtype = np.float32)

    model = Convolution3D((3, 3, 3),
                          num_filters=1,
                          activation=None,
                          pad=False,
                          strides=1, name='foo')
    # shape should be
    model_shape = model(y).foo.shape

    np.testing.assert_array_equal(model_shape, (1, 1, 1, 1), \
        "Error in convolution3D with stride = 1 and padding")

    res = model(y).eval({y: dat})

    expected_res = np.sum(model.foo.W.value)

    np.testing.assert_array_almost_equal(res[0][0][0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution3D computation with stride = 1 and zeropad = True")

##########################################################
# Test convolutional 2D layer for correctness (p=False s = 1)
##########################################################
def test_layers_convolution_2d():
    inC, inH, inW = 1, 3, 3
    y = C.input_variable((inC,inH, inW))

    dat = np.ones([1, inC, inH, inW], dtype = np.float32)

    model = Convolution2D((3, 3),
                          num_filters=1,
                          activation=None,
                          pad=False,
                          strides=1, name='foo')
    # shape should be
    model_shape = model(y).foo.shape
    np.testing.assert_array_equal(model_shape, (1, 1, 1), \
        "Error in convolution2D with stride = 1 and padding")

    res = model(y).eval({y: dat})

    expected_res = np.sum(model.foo.W.value)

    np.testing.assert_array_almost_equal(res[0][0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution2D computation with stride = 1 and zeropad = True")

##########################################################
# Test convolutional 1D layer for correctness (p=False s = 1)
##########################################################
def test_layers_convolution_1d():
    inC, inW = 1, 3
    y = C.input_variable((inC, inW))

    dat = np.ones([1, inC, inW], dtype = np.float32)

    model = Convolution1D((3, ),
                          num_filters=1,
                          activation=None,
                          pad=False,
                          strides=1, name='foo')
    # shape should be
    model_shape = model(y).foo.shape
    np.testing.assert_array_equal(model_shape, (1, 1), \
        "Error in convolution1D with stride = 1 and padding")

    res = model(y).eval({y: dat})

    expected_res = np.sum(model.foo.W.value)

    np.testing.assert_array_almost_equal(res[0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution1D computation with stride = 1 and zeropad = True")

####################################
# sequential convolution 1D without reduction dimension (old)
####################################
def test_sequential_convolution_1d_without_reduction_dim_old():
    c = Convolution(3, init=np.array([4., 2., 1.], dtype=np.float32), sequential=True, pad=False, reduction_rank=0, bias=False)
    c.update_signature(Sequence[Tensor[()]])  # input is a sequence of scalars
    data = [np.array([2., 6., 4., 8., 6.])]   # like a short audio sequence, in the dynamic dimension
    out = c(data)
    exp = [[24., 40., 38.]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in sequential convolution without reduction dimension')

    # Filter shape (3, 1) instead of 3 should be more reasonable.
    # e.g. Input shape [#] x [1] matches filter shape [3, 1], where as input shape [#] x [] matches filter shape [3].
    #      Input shape [#] x [3] matches filter shape [3, 2].
    # This setup will not be supported in the newer version SequentialConvolution.
    c = Convolution(3, init=np.array([4., 2., 1.], dtype=np.float32), sequential=True, pad=False, reduction_rank=0, bias=False)
    c.update_signature(Sequence[Tensor[1]]) # input is a sequence of dim-1 vectors
    data = [np.array([[2.], [6], [4.], [8.], [6.]])]
    out = c(data)
    exp = [[[24.], [40.], [38]]] # not reducing; hence, output is also a sequence of dim-1 vectors
    np.testing.assert_array_equal(out, exp, err_msg='Error in sequential convolution without reduction dimension')

    # these cases failed before
    emb_dim = 10
    x = C.input_variable(**Sequence[Tensor[20]])
    m = Embedding(emb_dim)(x)
    m = Convolution(filter_shape=3, sequential=True)(m)

    # this one still fails
    # Reshape: Operand (sub-)dimensions '[3]' incompatible with desired replacement (sub-)dimensions '[]'. Number of elements must be the same..
    m = Embedding(emb_dim)(x)
    m = reshape(m, (emb_dim,1))
    m = Convolution(filter_shape=(3,1), num_filters=13, pad=True, sequential=True)(m)

    m = Embedding(emb_dim)(x)
    m = Convolution(filter_shape=3, pad=True, sequential=True)(m)

def test_sequential_convolution_1d_without_reduction_dim_old():
    # new SequentialConvolution
    c = SequentialConvolution(3, init=np.array([4., 2., 1.], dtype=np.float32), pad=False, reduction_rank=0, bias=False)
    c.update_signature(Sequence[Tensor[()]])
    data = [np.array([2., 6., 4., 8., 6.])]   # like a short audio sequence, in the dynamic dimension
    out = c(data)
    exp = [[24., 40., 38.]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in sequential convolution without reduction dimension')

    # Filter shape (3, 1) instead of 3 should be more reasonable.
    # e.g. Input shape [#] x [1] matches filter shape [3, 1], where as input shape [#] x [] matches filter shape [3].
    #      Input shape [#] x [3] matches filter shape [3, 2].
    # This setup will not be supported in the newer version SequentialConvolution.
    c = SequentialConvolution((3,1), init=np.array([4., 2., 1.], dtype=np.float32).reshape((3,1)), pad=False, reduction_rank=0, bias=False)
    c.update_signature(Sequence[Tensor[1]]) # input is a sequence of dim-1 vectors
    data = [np.array([[2.], [6], [4.], [8.], [6.]])]
    out = c(data)
    exp = [[[24.], [40.], [38]]] # not reducing; hence, output is also a sequence of dim-1 vectors
    np.testing.assert_array_equal(out, exp, err_msg='Error in sequential convolution without reduction dimension')

    # these cases failed before
    emb_dim = 10
    x = C.input_variable(**Sequence[Tensor[20]])
    m = Embedding(emb_dim)(x)
    m = SequentialConvolution(filter_shape=3)(m)

    # this one still fails
    # Reshape: Operand (sub-)dimensions '[3]' incompatible with desired replacement (sub-)dimensions '[]'. Number of elements must be the same..
    m = Embedding(emb_dim)(x)
    m = reshape(m, (emb_dim,1))
    m = SequentialConvolution(filter_shape=(3,1), num_filters=13, pad=True)(m)

    m = Embedding(emb_dim)(x)
    m = SequentialConvolution(filter_shape=3, pad=True)(m)


####################################
# sequential convolution 2D without reduction dimension
####################################
def test_sequential_convolution_2d_without_reduction_dim():
    data = np.asarray([[0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.3, 0.4, 0.5, 0.6], [2.5, 2.3, 2.1, 1.9, 1.7]], dtype=np.float32)
    data_ = data.reshape((1, 3, 5))

    c = SequentialConvolution((3,2), pad=False, bias=False, reduction_rank=0)
    c.update_signature(Sequence[Tensor[5]])

    out = c(data_)

    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j], data[j][i:i+2].reshape(2)) for i in range(4)]] for j in range(3)]
    exp = np.sum(np.asarray(exp), 0)

    np.testing.assert_array_almost_equal(out[0][0], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = False")

    c = SequentialConvolution((3,2), pad=True, bias=False, reduction_rank=0)
    x = C.input_variable(**Sequence[Tensor[5]])
    c = c(x)

    out = c.eval({x:data_})
    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j], data[j][i:i+2].reshape(2)) for i in range(4)]] + [c.W.value[0][0][j][0] * data[j][-1]] for j in range(3)]
    exp = np.sum(np.asarray(exp), 0)
    np.testing.assert_array_almost_equal(out[0][1], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = True")

    c = SequentialConvolution((3,2), pad=True, strides=2, bias=False, reduction_rank=0)
    c = c(x)

    out = c.eval({x:data_})
    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j+1], data[j][i:i+2].reshape(2)) for i in [0, 2]]] + [c.W.value[0][0][j+1][0] * data[j][-1]] for j in range(2)]
    exp = np.sum(np.asarray(exp), 0)

    np.testing.assert_array_almost_equal(out[0][0], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True, strides = 2 and zeropad = True")
    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j], data[j+1][i:i+2].reshape(2)) for i in [0, 2]]] + [c.W.value[0][0][j][0] * data[j+1][-1]] for j in range(2)]
    exp = np.sum(np.asarray(exp), 0)
    np.testing.assert_array_almost_equal(out[0][1], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True, strides = 2 and zeropad = True")


####################################
# sequential convolution 1D
####################################
def test_sequential_convolution_1d():
    data = np.asarray([0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
    data = data.reshape((5, 1))

    c = SequentialConvolution(3, pad=False, bias=False)
    c.update_signature(Sequence[Tensor[1]])  # input is a sequence of scalars
    out = c(data)
    exp = [np.dot(a, b) for a,b in [(c.W.value[0][0], data[i:i+3].reshape(3)) for i in range(3)]]

    np.testing.assert_array_almost_equal(out[0], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True and zeropad = False")

    c = SequentialConvolution(3, pad=True, bias=False)
    x = C.input_variable(**Sequence[Tensor[1]])
    c = c(x)

    out = c.eval({x:data})
    exp = [np.dot(c.W.value[0][0][1:3], data[0:2])] + [np.dot(a, b) for a,b in [(c.W.value[0][0], data[i:i+3].reshape(3)) for i in range(3)]] + [np.dot(c.W.value[0][0][0:2], data[3:5])]
    np.testing.assert_array_almost_equal(out[0], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True and zeropad = True")

    c = SequentialConvolution(2, pad=True, strides=2, bias=False)
    c = c(x)

    out = c.eval({x:data})
    exp = [np.dot(a, b) for a,b in [(c.W.value[0][0], data[i:i+2].reshape(2)) for i in [0, 2]]] + [np.dot(c.W.value[0][0][0], data[4])]

    np.testing.assert_array_almost_equal(out[0], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True, strides = 2 and zeropad = True")

    c = SequentialConvolution(2, num_filters=3, pad=True, bias=False)
    c = c(x)

    out = c.eval({x:data})
    out = out[0].transpose()
    exp = [np.dot(a, b) for a,b in [(c.W.value[0][0], data[i:i+2].reshape(2)) for i in range(4)]] + [np.dot(c.W.value[0][0][0], data[4])]
    np.testing.assert_array_almost_equal(out[0], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True, strides = 2 and zeropad = True")
    exp = [np.dot(a, b) for a,b in [(c.W.value[2][0], data[i:i+2].reshape(2)) for i in range(4)]] + [np.dot(c.W.value[2][0][0], data[4])]
    np.testing.assert_array_almost_equal(out[2], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True, strides = 2 and zeropad = True")


    c = SequentialConvolution(2, num_filters=3, pad=True, init_bias=np.asarray([1,2,3], dtype=np.float32))
    c = c(x)

    out = c.eval({x:data})
    out = out[0].transpose()
    exp = [np.dot(a, b) + 1 for a,b in [(c.W.value[0][0], data[i:i+2].reshape(2)) for i in range(4)]] + [np.dot(c.W.value[0][0][0], data[4]) + 1]
    np.testing.assert_array_almost_equal(out[0], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True, strides = 2, zeropad = True and bias = (1,2,3)")
    exp = [np.dot(a, b) + 3 for a,b in [(c.W.value[2][0], data[i:i+2].reshape(2)) for i in range(4)]] + [np.dot(c.W.value[2][0][0], data[4]) + 3]
    np.testing.assert_array_almost_equal(out[2], exp, decimal=5, \
        err_msg="Error in convolution1D computation with sequential = True, strides = 2, zeropad = True and bias = (1,2,3)")


####################################
# sequential convolution 1D with channel & filter size > 1
####################################
def test_sequential_convolution_1d_channel_filter():
    data = np.asarray([[[0.4, 0.2], [0.6, 0.1], [0.8, 1.5], [1.0, 3.2], [1.2, 1.8]]], dtype=np.float32)
    data = data.reshape((5, 2))
    x = C.input_variable(**Sequence[Tensor[2]])

    c = SequentialConvolution(3, num_filters=4, pad=True, bias=False)
    c = c(x)
    out = c.eval({x:data})

    data = data.transpose()

    for k in range(4):
        exp = [[np.dot(c.W.value[k][j][1:3], data[j][0:2])] + [np.dot(a, b) for a,b in [(c.W.value[k][j], data[j][i:i+3].reshape(3)) for i in range(3)]] + [np.dot(c.W.value[k][j][0:2], data[j][3:5])] for j in range(2)]
        exp = np.sum(np.asarray(exp), 0)

        np.testing.assert_array_almost_equal(out[0].transpose()[k], exp, decimal=5, \
            err_msg="Error in convolution1D computation with channel size 2, filter num 4, sequential = True and zeropad = True")

####################################
# sequential convolution 2D
####################################
def test_sequential_convolution_2d():
    data = np.asarray([[0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.3, 0.4, 0.5, 0.6], [2.5, 2.3, 2.1, 1.9, 1.7]], dtype=np.float32)
    data = data.reshape((3, 1, 5))

    c = SequentialConvolution((3,2), pad=False, bias=False)
    c.update_signature(Sequence[Tensor[1, 5]])

    out = c(data)

    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j], data[j][0][i:i+2].reshape(2)) for i in range(4)]] for j in range(3)]
    exp = np.sum(np.asarray(exp), 0)

    np.testing.assert_array_almost_equal(out[0][0], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = False")

    c = SequentialConvolution((3,2), pad=True, bias=False)
    x = C.input_variable(**Sequence[Tensor[1, 5]])
    c = c(x)

    out = c.eval({x:data})
    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j], data[j][0][i:i+2].reshape(2)) for i in range(4)]] + [c.W.value[0][0][j][0] * data[j][0][-1]] for j in range(3)]
    exp = np.sum(np.asarray(exp), 0)
    np.testing.assert_array_almost_equal(out[0][1], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = True")

    c = SequentialConvolution((3,2), pad=True, strides=2, bias=False)
    c = c(x)

    out = c.eval({x:data})
    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j+1], data[j][0][i:i+2].reshape(2)) for i in [0, 2]]] + [c.W.value[0][0][j+1][0] * data[j][0][-1]] for j in range(2)]
    exp = np.sum(np.asarray(exp), 0)

    np.testing.assert_array_almost_equal(out[0][0], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True, strides = 2 and zeropad = True")
    exp = [[np.dot(a, b) for a,b in [(c.W.value[0][0][j], data[j+1][0][i:i+2].reshape(2)) for i in [0, 2]]] + [c.W.value[0][0][j][0] * data[j+1][0][-1]] for j in range(2)]
    exp = np.sum(np.asarray(exp), 0)
    np.testing.assert_array_almost_equal(out[0][1], exp, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True, strides = 2 and zeropad = True")

    data = np.ones((3,1,5), dtype=np.float32)

    c = SequentialConvolution((3,2), pad=False, num_filters=4, init_bias=np.asarray([1,2,3,4], dtype=np.float32))
    c = c(x)
    out = c.eval({x:data})
    for i in range(4):
        np.testing.assert_array_almost_equal(out[0][0][i], [np.sum(c.W.value[i]) + j + 1 for j in range(4)], \
            err_msg="Error in convolution2D computation with sequential = True, num_filters = 4 and init_bias = [1,2,3,4]")

    c = SequentialConvolution(filter_shape=(3,2), pad=True, strides=(2,2), num_filters=4)
    c = c(x)
    out = c.eval({x:data})
    # output shape: [out_seq, out_num_filters(omitted if = 1), out_feats]
    np.testing.assert_equal(out[0].shape, (2,4,3), \
        err_msg="Error in convolution2D computation with sequential = True, num_filters = 4 and bias = True: wrong output shape")

####################################
# 1D convolution without reduction dimension
####################################
def test_1D_convolution_without_reduction_dim():
    c = Convolution1D(3, init=np.array([4, 2, 1]), pad=True, reduction_rank=0, bias=False)
    c.update_signature(5)
    data = [np.array([[2, 6, 4, 8, 6]])]   # like a audio sequence, in a static dimension
    out = c(data)
    exp = [[10, 24, 40, 38, 44]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in 1D convolution without reduction dimension')

    # Failing call
    with pytest.raises(ValueError):
        Convolution1D((2,3))


####################################
# 1D convolution with dilation
####################################
def test_1D_convolution_with_dilation(device_id):
    # Currently dilation is not supported on CPU. 
    if device_id == -1:
        return
    c = Convolution(3, init=np.array([4,2,1]), reduction_rank=0, bias=False, dilation=2, pad=True)
    c.update_signature(5)
    data = [np.array([[2, 6, 4, 8, 6]])]   # like a audio sequence, in a static dimension
    out = c(data)
    exp = [[8, 20, 22, 40, 28]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in 1D convolution with dilation = 2')

####################################
# 1D sequential convolution with dilation
####################################
def test_1D_sequential_convolution_with_dilation(device_id):
    # Currently dilation is not supported on CPU. 
    if device_id == -1:
        return
    c = SequentialConvolution(3, init=np.array([4,2,1]), reduction_rank=0, bias=False, dilation=2, pad=True)
    c.update_signature(Sequence[Tensor[()]])
    data = np.array([[2, 6, 4, 8, 6]])   # like a audio sequence, in a static dimension
    out = c(data)
    exp = [[8, 20, 22, 40, 28]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in 1D convolution with dilation = 2')

    c = SequentialConvolution((3,1), init=np.array([4,2,1]).reshape((3,1)), reduction_rank=0, bias=False, dilation=2, pad=True)
    c.update_signature(Sequence[Tensor[1]])
    data = np.array([[2, 6, 4, 8, 6]])   # like a audio sequence, in a static dimension
    data = data.reshape((5, 1))
    out = c(data)
    exp = [[[8], [20], [22], [40], [28]]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in 1D convolution with dilation = 2')

####################################
# 2D convolution with dilation
####################################
def test_2D_convolution_with_dilation(device_id):
    # Currently dilation is not supported on CPU. 
    if device_id == -1:
        return
    dilation = 2

    data = np.asarray([[0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.3, 0.4, 0.5, 0.6], [2.5, 2.3, 2.1, 1.9, 1.7]], dtype=np.float32)
    data = data.reshape((1, 1, 3, 5))

    c = Convolution((3,2), pad=True, bias=False, dilation=2)
    c.update_signature(Sequence[Tensor[1, 3, 5]])

    out = c(data)
    data = data.reshape((3, 5))
    exp_00 = np.dot([0, 0, 0, data[0][1], 0, data[2][1]], list(c.W.value[0][0][0]) + list(c.W.value[0][0][1]) + list(c.W.value[0][0][2]))
    exp_22 = np.dot([data[0][1], data[0][3], data[2][1], data[2][3], 0,0], list(c.W.value[0][0][0]) + list(c.W.value[0][0][1]) + list(c.W.value[0][0][2]))

    np.testing.assert_array_almost_equal(out[0][0][0][0], exp_00, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = False")
    np.testing.assert_array_almost_equal(out[0][0][2][2], exp_22, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = False")

####################################
# 2D sequential convolution with dilation
####################################
def test_2D_sequential_convolution_with_dilation(device_id):
    # Currently dilation is not supported on CPU. 
    if device_id == -1:
        return
    dilation = 2

    data = np.asarray([[0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.3, 0.4, 0.5, 0.6], [2.5, 2.3, 2.1, 1.9, 1.7]], dtype=np.float32)
    data = data.reshape((1, 3, 1, 5))

    c = SequentialConvolution((3,2), pad=True, bias=False, dilation=2)
    c.update_signature(Sequence[Tensor[1, 5]])

    out = c(data)
    data = data.reshape((3, 5))
    exp_00 = np.dot([0, 0, 0, data[0][1], 0, data[2][1]], list(c.W.value[0][0][0]) + list(c.W.value[0][0][1]) + list(c.W.value[0][0][2]))
    exp_22 = np.dot([data[0][1], data[0][3], data[2][1], data[2][3], 0,0], list(c.W.value[0][0][0]) + list(c.W.value[0][0][1]) + list(c.W.value[0][0][2]))

    np.testing.assert_array_almost_equal(out[0][0][0], exp_00, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = False")
    np.testing.assert_array_almost_equal(out[0][2][2], exp_22, decimal=5, \
        err_msg="Error in convolution2D computation with sequential = True and zeropad = False")

##########################################################
# Test Convolution Transpose layer for correctness
##########################################################
def test_layers_convolution_transpose():
    import pytest

    inC, inH, inW = 1, 3, 3
    in_filter_shape = (3, 3)
    out_num_filters = 1
    dat = np.ones([1, inC, inH, inW], dtype = np.float32)

    y = C.input_variable((inC, inH, inW))

    ##########################################################
    # Test convolutional layer for correctness (p=False s = 1)
    ##########################################################

    zeropad = False
    in_strides = 1

    model = ConvolutionTranspose(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = np.sum(model.W.value)

    np.testing.assert_array_almost_equal(res[0][0][2][2], expected_res, decimal=6, \
        err_msg="Error in convolution transpose computation with stride = 1 and zeropad = False")

    ##########################################################
    # Test convolutional transpose layer for correctness (p=False s = 2)
    ##########################################################
    zeropad = False
    in_strides = 2

    model = ConvolutionTranspose(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = model.W.value[0][0][:2,:2]

    np.testing.assert_array_almost_equal(res[0][0][:2,:2], expected_res, decimal=6, \
        err_msg="Error in convolution transpose computation with stride = 2 and zeropad = False")

    ##########################################################
    # Test convolutional transpose layer for correctness (p=True s = 1)
    ##########################################################
    zeropad = True
    in_strides = 1

    model = ConvolutionTranspose(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    expected_res = np.sum(model.W.value)

    np.testing.assert_array_almost_equal(res[0][0][1][1], expected_res, decimal=6, \
        err_msg="Error in convolution transpose computation with stride = 1 and zeropad = True")

    ##########################################################
    # Test convolutional transpose layer for correctness (p=True s = 2)
    ##########################################################
    zeropad = True
    in_strides = 2

    model = ConvolutionTranspose(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    expected_res = model.W.value[0][0][1][1]

    np.testing.assert_array_almost_equal(res[0][0][0][0], expected_res, decimal=6,
        err_msg="Error in convolution transpose computation with stride = 1 and zeropad = True")

def test_failing_convolution_transpose():
    with pytest.raises(ValueError):
        conv = ConvolutionTranspose((3,3), 1)
        conv.update_signature(5)

##########################################################
# Test Conv/Pooling/Unpooling/Deconvolution and layer for correctness
##########################################################
def test_layers_conv_pool_unpool_deconv():
    inC, inH, inW = 1,4,4

    y = C.input_variable((inC,inH, inW))

    cMap = 1

    zero_pad = True
    conv_init = 1
    filter_shape = (2,2)
    pooling_strides = (2,2)

    dat = np.arange(0,16, dtype=np.float32).reshape(1,1,4,4)

    conv = Convolution(filter_shape, cMap, pad=zero_pad, init=conv_init,activation=None)(y)

    pool = MaxPooling(filter_shape, pooling_strides)(conv)

    unpool = MaxUnpooling(filter_shape, pooling_strides)(pool, conv)

    z = ConvolutionTranspose(filter_shape, cMap, init=conv_init, pad=zero_pad)(unpool)

    assert z.shape == y.shape

    res = z(dat)

    expected_res = np.asarray([[30, 64, 34], [76, 160, 84], [46, 96, 50]], np.float32)

    np.testing.assert_array_almost_equal(res[0][0][1:,1:], expected_res, decimal=6,
        err_msg="Wrong values in conv/pooling/unpooling/conv_transposed")

##########################################################
# Test for dropout
##########################################################
def test_layers_dropout():
    dat = np.array([[1., 1., 1., 1.]], dtype=np.float32)
    y = C.input_variable(4)
    p = Dense(1, activation=None, name='foo')(y)
    z = Dropout(0.75, name='bar')(p)

    res =  z(y).eval({y: dat})
    expected_res = np.sum(p.foo.W.value)

    np.testing.assert_array_almost_equal(res, expected_res, decimal=7, \
        err_msg="Error in dropout computation")

    z = Dropout(keep_prob=0.25, name='bar')(p)
    res =  z(y).eval({y: dat})
    np.testing.assert_array_almost_equal(res, expected_res, decimal=7, \
        err_msg="Error in dropout computation with keep_prob")

    with pytest.raises(ValueError):
        z = Dropout(keep_prob=-1.5, name='bar')(p)

    with pytest.raises(ValueError):
        z = Dropout(1.5, name='bar')(p)

##########################################################
# Test for Stabilizer
##########################################################
def test_layers_stabilizer():
    y = C.input_variable(4)
    p = Stabilizer()(y)

    dat = np.array([[1.0,2.0,3.0,4.0]], dtype=np.float32)
    res = p(y).eval({y: dat})

    # a stabilizer starts with having no effect, hence input=output
    np.testing.assert_array_almost_equal(res, dat, decimal=6, \
        err_msg="Error in layer normalization computation")

##########################################################
# Test for LayerNormalization
##########################################################
def test_layers_layer_normalization():
    y = C.input_variable(4)
    p = LayerNormalization(name='foo')(y)

    dat = np.array([[1.0,2.0,3.0,4.0]], dtype=np.float32)
    res =  p(y).eval({y: dat})

    checkedBias = False
    checkedScale = False
    for param in p.parameters:
        if param.name == "bias":
            assert param.value.shape == y.shape
            checkedBias = True
        elif param.name == "scale":
            assert param.value.shape == y.shape
            checkedScale = True

    assert checkedBias and checkedScale

    mean_dat = np.mean(dat)
    x = dat-mean_dat
    std = np.sqrt(np.mean(x*x))
    epsilon = 0.00001

    np.testing.assert_array_almost_equal(res, x/(std + epsilon), decimal=6, \
        err_msg="Error in layer normalization computation")

##########################################################
# Test for BatchNormalization
##########################################################
# TESTTODO: Currently the result doesn't match the expected result
def test_layers_batch_normalization():
    pass
#    dat = np.array([[1.0,0.5,1.0,0.5]], dtype=np.float32)
#    y = C.input_variable(4)
#    p = BatchNormalization(init_scale=2
#                                     normalization_time_constant=0,
#                                     name ='foo')(y)
#
#    res =  p(y).eval({y: dat})
#
#    mean_dat = np.mean(dat)
#    x = dat-mean_dat
#    std = np.sqrt(np.mean(x*x))
#    bias = 0
#    scale = 2
#    expected_res = scale * (x/(std + 0.00001)) + bias
#    np.testing.assert_array_almost_equal(res[0], expected_res, decimal=5, \
#         err_msg="Error in BN computation")


def test_cloned_parameters_are_identical():
    inputs = 2
    outputs = 4
    hidden_dimension = 2

    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)

    def z():
        return Sequential([
            Dense(hidden_dimension, activation=C.relu),
            Dense(outputs)])

    def compare(f1, f2):
        for x, y in zip(f1.parameters, f2.parameters):
            assert np.all(x.value == y.value)

    def clone_and_compare(z):
            compare(z, z.clone('clone'))

    # first, verify that when cloning un-initialized parameters,
    # the seed value is properly copied, so that after the
    # initialization cloned and original value are the same.
    z1 = z()(features)
    clone_and_compare(z1)

    # now, do the same, but first force parameter initialization.
    z2 = z()(features)
    ignored = 0
    for x in z2.parameters:
        ignored += np.sum(x.value)
    clone_and_compare(z2)

    # now, test the layer's lib clone() method.
    z3 = z()
    z4 = z3.clone('clone')
    compare(z3(features), z4(features))