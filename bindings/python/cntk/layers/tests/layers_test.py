# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import *
from cntk.layers import *
from cntk.layers.typing import *

# Note: We do not test gradients here, assuming that those are tested elsewhere.
# Forward outputs are tested to verify that the structure of the layer is as expected.

def test_layers_name(device_id):
    from cntk import placeholder
    I = placeholder(name='input')
    p = Dense(10, name='dense10')(I)
    assert(I.name == 'input')
    assert(p.root_function.name == 'dense10')

    q = Convolution((3, 3), 3, name='conv33')(I)
    assert(q.root_function.name == 'conv33')

    e = Embedding(0, name='emb')(I)
    assert(e.root_function.name == 'emb')

    e = Embedding(0, name='')(I)
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

def test_Function(device_id):

    ####################################################
    # Test 1: BlockFunction()
    ####################################################
    @BlockFunction('Square', 'block_name')
    def f(x):
        return x * x
    assert f.shape == (-2,)
    #assert f.op_name == 'Square'   # BUGBUG: op_name is 'CompositeFunctionOpName'
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

####################################
# . syntax for name lookup
####################################

def test_lookup(device_id):
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

####################################
# recurrence (Fold()) over regular function
####################################

def test_recurrence_fun(device_id):
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

def test_unfold(device_id):
    from cntk.layers import UnfoldFrom

    @Function
    def double_up(s):
        return s * 2
    x = [[[0],[0],[0]],
         [[0],[0],[0],[0],[0]]]

    ####################################################
    # Test 1: simple unfold
    ####################################################
    UF = UnfoldFrom(double_up, initial_state=1)
    @Function
    @Signature(Sequence[Tensor[1]])
    def FU(x):
        return UF(x)
    r = FU(x)
    exp = [[[ 2 ], [ 4 ], [ 8 ]],
           [[ 2 ], [ 4 ], [ 8 ], [ 16 ], [ 32 ]]]
    assert_list_of_arrays_equal(r, exp, err_msg='Error in UnfoldFrom() forward')

    ####################################################
    # Test 2: unfold with length increase and terminating condition
    ####################################################
    UF = UnfoldFrom(double_up, until_predicate=lambda x: greater(x, 63),  initial_state=1, length_increase=1.6)
    @Function
    @Signature(Sequence[Tensor[1]])
    def FU(x):
        return UF(x)
    r = FU(x)
    exp = [[[ 2 ], [ 4 ], [ 8 ], [ 16 ], [ 32 ]],         # tests length_increase
           [[ 2 ], [ 4 ], [ 8 ], [ 16 ], [ 32 ], [ 64 ]]] # tests early cut-off due to until_predicate
    print(r)
    print(exp)
    assert_list_of_arrays_equal(r, exp, err_msg='Error in UnfoldFrom(..., until_predicate, length_increase, ...) forward')

####################################
# Test dense layer for correctness
####################################

def test_layers_dense(device_id):
    y = Input(2)
    dat = np.array([[-1., 1.]], dtype=np.float32)

    ####################################################
    # Test 1: no activation
    ####################################################
    p = Dense(2, activation=None, name='foo')(y)
    res = p(y).eval({y: dat})

    npout = np.matrix(dat[0]) * p.foo.W.value + p.foo.b.value
    print(res[0])
    print(npout)
    np.testing.assert_array_equal(res[0], npout, err_msg='Error in dense layer')

    ####################################################
    # Test 2: with activation
    ####################################################
    p = Dense(2, activation=sigmoid, name='foo')(y)
    res = p(y).eval({y: dat})

    def _sigmoid(x):
        return 1./(1 + np.exp(-x))

    npout = _sigmoid(np.matrix(dat[0]) * p.foo.W.value + p.foo.b.value)
    print(res[0])
    print(npout)

    np.testing.assert_array_almost_equal(res[0], npout, decimal=7, err_msg='Error in dense layer with sigmoid')

    ####################################################
    # Test 3: 2-dense layer
    ####################################################
    p = Dense(3, activation=None, name='foo')(y)
    q = Dense(3, activation=None, name='bar')(p)
    res = q(y).eval({y: dat})

    npout1 = np.matrix(dat[0]) * p.foo.W.value + p.foo.b.value
    npout = npout1 * q.bar.W.value + q.bar.b.value

    np.testing.assert_array_almost_equal(res[0], npout, decimal=7, err_msg='Error in 2-dense layer')

########################################
# Test Embedding layer for correctness
########################################
def test_layers_embedding(device_id):
    embDim = 3
    y = Input(2)

    # embedding base case
    e = Embedding(shape=embDim, name='foo')

    dat = np.array([[-1., 1.]], dtype=np.float32)
    res = e(y).eval({y: dat})

    npout = np.matrix(dat[0]) * e.E.value
    np.testing.assert_array_equal(res[0], npout, err_msg='Error in embedding layer')

    # embedding, initialized from a user-supplied starting point for the parameter
    e = Embedding(embDim, init=[[1, 3, 2], [3, 4, 1]], name='bar')

    dat = np.array([[-1., 1.]], dtype=np.float32)
    res = e(y).eval({y: dat})

    npout = np.matrix(dat[0]) * e.E.value
    np.testing.assert_array_equal(res[0], npout, err_msg='Error in constant embedding layer')

    # embedding, initialized from a user-supplied constant weight table
    e = Embedding(weights=[[1, 3, 2], [3, 4, 1]], name='baz')

    dat = np.array([[-1., 1.]], dtype=np.float32)
    res = e(y).eval({y: dat})

    npout = np.matrix(dat[0]) * e.E.value
    np.testing.assert_array_equal(res[0], npout, err_msg='Error in constant embedding layer')

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

def test_layers_convolution_shape(device_id):
    # Get the output shape
    # i: input dimension
    # k: kernel dimension
    # p: number of zero padding
    # s: strides
    inC, inH, inW = 2, 6, 7
    y = Input((inC, inH, inW))
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
    print(expected_shape)
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

def  test_layers_convolution_value(device_id):

    # Common parameters
    inC, inH, inW = 1, 3, 3
    in_filter_shape = (3, 3)
    out_num_filters = 1
    dat = np.ones([1, inC, inH, inW], dtype = np.float32)

    ##########################################################
    # Test convolutional layer for correctness (p=False s = 1)
    ##########################################################
    y = Input((inC, inH, inW))
    zeropad = False
    in_strides = 1

    model = Convolution(in_filter_shape,
                        num_filters=out_num_filters,
                        activation=None,
                        pad=zeropad,
                        strides=in_strides, name='foo')
    res = model(y).eval({y: dat})

    # Extract the W weight matrix
    expected_res = np.sum(getattr(model, 'foo').parameters[0].value)

    np.testing.assert_array_almost_equal(res[0][0][0][0][0], expected_res, decimal=7, \
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
    expected_res = np.sum(getattr(model, 'foo').parameters[0].value)

    np.testing.assert_array_almost_equal(res[0][0][0][0][0], expected_res, decimal=7, \
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
    expected_res = np.sum(getattr(model, 'foo').parameters[0].value)

    # Compare the center of the res with the sum of the weights
    np.testing.assert_array_almost_equal(res[0][0][0][1][1], expected_res, decimal=7, \
        err_msg="Error in convolution computation with stride = 1 and zeropad = True")

    ##########################################################
    # Test convolutional layer for correctness (p=True s = 1)
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
    W = getattr(model, 'foo').parameters[0].value
    expected_res = np.sum(W[0][0][1:,1:])

    # Compare the center of the res with the sum of the weights
    np.testing.assert_array_almost_equal(res[0][0][0][0][0], expected_res, decimal=5,
        err_msg="Error in convolution computation with stride = 1 and zeropad = True")

##########################################################
# Test convolutional 3D layer for correctness (p=False s = 1)
##########################################################
def  test_layers_convolution_3d(device_id):
    inC, inH, inW, inD = 1, 3, 3, 3
    y = Input((inC,inH, inW, inD))
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

    expected_res = np.sum(getattr(model, 'foo').parameters[0].value)

    np.testing.assert_array_almost_equal(res[0][0][0][0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution3D computation with stride = 1 and zeropad = True")

##########################################################
# Test convolutional 2D layer for correctness (p=False s = 1)
##########################################################
def test_layers_convolution_2d(device_id):
    inC, inH, inW = 1, 3, 3
    y = Input((inC,inH, inW))

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

    expected_res = np.sum(getattr(model, 'foo').parameters[0].value)

    np.testing.assert_array_almost_equal(res[0][0][0][0][0], expected_res, decimal=5, \
        err_msg="Error in convolution2D computation with stride = 1 and zeropad = True")

####################################
# sequential convolution without reduction dimension
####################################

def test_sequential_convolution_without_reduction_dim(device_id):
    c = Convolution(3, init=np.array([4, 2, 1]), sequential=True, pad=False, reduction_rank=0, bias=False)
    c.update_signature(Sequence[Tensor[()]]) # input is a sequence of scalars
    data = [np.array([2, 6, 4, 8, 6])]   # like a short audio sequence, in the dynamic dimension
    out = c(data)
    exp = [[24, 40, 38]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in sequential convolution without reduction dimension')

####################################
# 1D convolution without reduction dimension
####################################

def test_1D_convolution_without_reduction_dim(device_id):
    c = Convolution1D(3, init=np.array([4, 2, 1]), pad=True, reduction_rank=0, bias=False)
    ## BUGBUG: pad seems ignored? It looks like auto_padding=(False, False, True) gets passed on as m_audoPad = { false, false, true }
    c.update_signature(5)
    data = [np.array([[2, 6, 4, 8, 6]])]   # like a audio sequence, in a static dimension
    out = c(data)
    exp = [[24, 40, 38]]
    np.testing.assert_array_equal(out, exp, err_msg='Error in 1D convolution without reduction dimension')

##########################################################
# Test Deconvolution layer for correctness
##########################################################
# TESTTODO: Add the test for deconvolution once current bug with lower/upper pad is fixed
def test_layers_deconvolution(device_id):
    pass

##########################################################
# Test Conv/Pooling/Unpooling/Deconvolution and layer for correctness
##########################################################
# TESTTODO: Add the test for deconvolution once current bug with lower/upper pad is fixed
def test_layers_conv_pool_unpool_deconv(device_id):
    pass
#    inC, inH, inW = 1,4,4
#
#    y = Input((inC,inH, inW))
#
#    cMap =1
#
#
#    conv = Convolution((2,2), cMap, pad=True, activation=None, name='foo' )(y)
#
#    pool = MaxPooling((2,2), (2,2), name='bar')(conv)
#
#    unpool = MaxUnpooling ((4,4), (4,4), name ='baz')(pool, conv)
#
#    z = Deconvolution((2,2), inC, cMap,
#                                  lower_pad=(0,2,2),
#                                  upper_pad=(0,2,2),
#                                  bias=False,
#                                  init=glorot_uniform(0.001))(unpool,
#                                  name='faz')
#
#
#    print(z.faz.shape)
#
#    dat = np.arange(0,16, dtype=np.float32).reshape(1,1,4,4)
#    maxpool   = MaxPooling(filter_shape=(2,2), strides=(2,2), name='bar')
#    print(maxpool(y).shape)
#
#
#    res = maxpool(y).eval({y: dat})
#    print(res)
#
#    maxunpool = MaxUnpooling(filter_shape=(2,2),
#                             strides=(2,2),
#                             name='foo')((maxpool),(y))
#
#    # Add a few asserts (1 for value and other for shape once this is running)

##########################################################
# Test for dropout
##########################################################
def test_layers_dropout(device_id):
    dat = np.array([[1., 1., 1., 1.]], dtype=np.float32)
    y = Input(4)
    p = Dense(1, activation=None, name='foo')(y)
    z = Dropout(0.75, name='bar')(p)

    res =  z(y).eval({y: dat})
    expected_res = np.sum(p.foo.W.value)

    np.testing.assert_array_almost_equal(res, expected_res, decimal=7, \
        err_msg="Error in dropout computation")

##########################################################
# Test for Stabilizer
##########################################################
def test_layers_stabilizer(device_id):
    y = Input(4)
    p = Stabilizer()(y)

    dat = np.array([[1.0,2.0,3.0,4.0]], dtype=np.float32)
    res = p(y).eval({y: dat})

    # a stabilizer starts with having no effect, hence input=output
    np.testing.assert_array_almost_equal(res[0], dat, decimal=6, \
        err_msg="Error in layer normalization computation")

##########################################################
# Test for LayerNormalization
##########################################################
def test_layers_layer_normalization(device_id):
    y = Input(4)
    p = LayerNormalization(name='foo')(y)

    dat = np.array([[1.0,2.0,3.0,4.0]], dtype=np.float32)
    res =  p(y).eval({y: dat})

    mean_dat = np.mean(dat)
    x = dat-mean_dat
    std = np.sqrt(np.mean(x*x))
    epsilon = 0.00001

    np.testing.assert_array_almost_equal(res[0], x/(std + epsilon), decimal=6, \
        err_msg="Error in layer normalization computation")

##########################################################
# Test for BatchNormalization
##########################################################
# TESTTODO: Currently the result doesn't match the expected result
def test_layers_batch_normalization(device_id):
    pass
#    dat = np.array([[1.0,0.5,1.0,0.5]], dtype=np.float32)
#    y = Input(4)
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
