# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
from cntk.ops import abs, square, sqrt, cos, splice
from cntk import Axis, reshape, sigmoid, element_max, Function, BlockFunction, Constant, greater, default_options, default_options_for, \
                 get_default_override, default_override_or
from cntk.layers import SequentialConvolution, Convolution, Convolution1D, Convolution2D, Convolution3D, Dense, Embedding, Fold, For, \
                        MaxPooling, MaxUnpooling, LSTM, GRU, RNNStep, Sequential, Stabilizer, Dropout, Recurrence, \
                        RecurrenceFrom, LayerNormalization, ConvolutionTranspose
from cntk.layers.typing import Sequence, Signature, Tensor, SequenceOver
from cntk.layers import For, Dense, SequentialClique, ResNetBlock, Sequential
from cntk.layers.sequence import Delay

import numpy as np
import pytest
import os
import subprocess
import re
import shutil
import time
import tempfile

onnx = pytest.importorskip("onnx")
from onnx import numpy_helper

from .onnx_test_helper import find_onnx_value_info_proto_with_matching_name, save_cntk_data_as_onnx_tensor, save_test_data
from .onnx_test_helper import get_onnx_test_runner_callscript
from .onnx_verify_helper import get_onnx_test_runner_path_str, parse_verify_out_str

###############################
# Helpers
###############################
windows = os.getenv("OS")=="Windows_NT"

def save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None):
    if not windows:
        return;

    folder = os.path.join(str(tmpdir), "test_" + name)
    os.mkdir(folder)

    model_file_name = os.path.join(folder, name + ".onnx")

    model.save(model_file_name, format = C.ModelFormat.ONNX)

    input_dict = dict(zip(model.arguments, data))

    o0 = model.eval(input_dict, device=device)
    o0 = np.array(o0)

    onnx_model = onnx.load(model_file_name);

    test_data_path = os.path.join(folder, "test_data_set_0")

    os.mkdir(test_data_path)
    save_test_data(model, onnx_model, test_data_path, data, o0, name, tmpdir)

    onnx_test_runner_path_str = get_onnx_test_runner_path_str()    
    callargs = [onnx_test_runner_path_str, '-n', name, str(tmpdir)]
    process = subprocess.run(callargs, stdout=subprocess.PIPE)

    failures = parse_verify_out_str(process.stdout.decode('utf-8'))
    assert(failures == 0)
    print ("test passed: ", name)

def generate_sequence_data(batch_size, seq_len, feature_size, input_as_index = False):
    assert batch_size == 1
    np.random.seed(0)
    data = np.zeros((batch_size, seq_len)).astype(np.float32) if input_as_index else np.zeros((batch_size, seq_len, feature_size)).astype(np.float32) 
    for i in range(0,seq_len):
        one_hot_index = np.random.random_integers(0, feature_size - 1)
        if input_as_index:
            data[0][i] = one_hot_index
        else:
            data[0][i][one_hot_index] = 1
    return data

def generate_sequential_data(tensor_shape):
    total = np.prod(tensor_shape)
    return np.reshape(range(0, total), tensor_shape).astype(np.float32)

##################################################
# onnx test ported from CNTK\bindings\python\cntk\ops\tests\sequence_test.py
##################################################
# this model has input to LSTM with static shape (-3, 3) which is not supported in ONNX.
def test_lstm_over_lstm_thought_vectors(tmpdir):
    pytest.skip('test_lstm_over_lstm_thought_vectors test a model with input to a LSTM op with static shape (-3, 3) which is not supported in ONNX.')
    input_vocab_size=3
    emb_dim = 4
    hidden_dim = 5
    num_labels = 2
    x_seq_input = C.sequence.input_variable((C.FreeDimension, input_vocab_size), is_sparse=True, name='features')
    label_seq_input = C.sequence.input_variable(num_labels, is_sparse=True, sequence_axis=C.Axis('label_sequence'), name='labels')
    with C.default_options(initial_state=0.1):
        model = C.layers.Embedding(emb_dim, name='embed')(x_seq_input)
        model = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False)(model)
        model = C.sequence.last(model)
        model = C.to_sequence_like(model, label_seq_input)
        model = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False)(model)
        model = C.layers.Dense(num_labels, name='classify')(model)

    seq1_data = [[[0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]], [[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]], [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0]]]
    csr_seq1 = _to_csr(seq1_data)
    ndarrayview1 = C.NDArrayView.from_csr(csr_seq1, shape=(3, 4, 3), device=C.cpu())
    x_seq_data = C.Value.create(C.sequence.input_variable((4, 3), is_sparse=True), [ndarrayview1], device=C.cpu()).data

    seq1_label_data = [[1, 1], [1, 1], [1, 1],[1, 1]]
    label_seq_data = [_to_csr(seq1_label_data)]

    batch_size, sequence_len1, sequence_len2, input_vocab_size_ = np.shape(x_seq_data)
    sequence_len2_, num_labels_ = np.shape(seq1_label_data)
    assert input_vocab_size_ == input_vocab_size
    assert sequence_len2_ == sequence_len2
    assert num_labels_ == num_labels

    #res = model.eval({x_seq_input : x_seq_data, label_seq_input : label_seq_data})    
    model.save(tmpdir + "/test_lstm_over_lstm_thought_vectors.onnx", C.ModelFormat.ONNX)

def test_sequence_max(tmpdir):
    np.random.seed(0)
    a = np.float32(np.random.rand(1,100,8))
    src = C.sequence.input_variable(shape=(8), sequence_axis=C.Axis("Seq"))
    out = C.sequence.reduce_max(src)
    test_name = 'sequence_max'
    save_onnx_model_with_validation_data(tmpdir, out, a, test_name, device=None)

def test_neg_sequence_max(tmpdir):
    np.random.seed(0)
    a = np.float32(-np.random.rand(1,100,8))
    src = C.sequence.input_variable(shape=(8), sequence_axis=C.Axis("Seq"))
    out = C.sequence.reduce_max(src)
    test_name = 'neg_sequence_max'
    save_onnx_model_with_validation_data(tmpdir, out, a, test_name, device=None)

def test_sequence_softmax(tmpdir):
    np.random.seed(0)
    a = np.float32(np.random.rand(1,100,8))
    src = C.sequence.input_variable(shape=(8), sequence_axis=C.Axis("Seq"))
    out = C.sequence.softmax(src)
    test_name = 'sequence_softmax'
    save_onnx_model_with_validation_data(tmpdir, out, a, test_name, device=None)

def test_op_broadcast_as_in_loop(tmpdir):
    a_data = np.asarray([[1]], dtype=np.float32)
    b_data = np.asarray([[[2], [3], [4]]], dtype=np.float32)

    a = C.input_variable(shape=(1,), name='a')
    b = C.sequence.input_variable(shape=(1,), name='b')

    out_placeholder = C.placeholder()
    out_delayed = C.sequence.past_value(out_placeholder, time_step=5)
    out_delayed_plus_b = out_delayed + b
    out = C.sequence.broadcast_as(a + 1.2, out_delayed_plus_b)  # + 1.2  # adding a constant to make 'a' into main graph
    out.replace_placeholder(out)

    name = 'op_broadcast_as_in_loop'
    save_onnx_model_with_validation_data(tmpdir, out, [a_data, b_data], name, device=None)

def test_op_gather_dynamic_axes_equivalence(tmpdir):
    input_data1 = np.asarray([1], dtype=np.float32)
    input_data2 = np.asarray([2], dtype=np.float32)

    a = C.sequence.input_variable(shape=input_data1.shape, dtype=np.float32, name='a')
    b = C.sequence.input_variable(shape=input_data2.shape, dtype=np.float32, name='b')

    is_last_a = C.sequence.is_last(a)
    a_last = C.sequence.gather(a, is_last_a)
    b_last = C.sequence.gather(b, is_last_a)
    z = a_last + b_last

    # create batch
    input_data1.shape = (1, 1) + input_data1.shape
    input_data2.shape = (1, 1) + input_data2.shape

    name = 'gather_dynamic_axes_equivalence'
    save_onnx_model_with_validation_data(tmpdir, z, [input_data1, input_data2], name, device=None)

def test_op_gather_derived_dynamic_axes_equivalence(tmpdir):
    # CNTK Sequence.Gather returns [#](1,) instead of [#, *](1,).
    # this causes shape mismatch in onnxruntime/CompareTwoTensors.
    # investigation code pieces in Loop-scan.ipynb
    input_data1 = np.asarray([1], dtype=np.float32)
    input_data2 = np.asarray([2], dtype=np.float32)

    a = C.sequence.input_variable(shape=input_data1.shape,
                       dtype=np.float32,
                       name='a')
    b = C.sequence.input_variable(shape=input_data2.shape,
                       dtype=np.float32,
                       name='b')

    a_last = C.sequence.gather(a, C.sequence.is_last(a), new_sequence_axis_typeinfo=(0, 1))
    b_last = C.sequence.gather(b, C.sequence.is_last(b), new_sequence_axis_typeinfo=(0, 1))

    z = a_last + b_last

    input_data1.shape = (1, 1) + input_data1.shape
    input_data2.shape = (1, 1) + input_data2.shape

    name = 'op_gather_derived_dynamic_axes_equivalence'
    save_onnx_model_with_validation_data(tmpdir, z, [input_data1, input_data2], name, device=None)

def test_op_sequence_reduce_sum(tmpdir):
    a = C.sequence.input_variable(shape=(1,), dtype=np.float32, needs_gradient=True, name='a')

    sequence_sum_a_plus_sequence_sum_a = C.sequence.reduce_sum(a) + C.sequence.reduce_sum(a)

    a_data = np.asarray([[[2], [3], [4]]], dtype=np.float32)

    name = 'sequence_reduce_sum_1'
    save_onnx_model_with_validation_data(tmpdir, sequence_sum_a_plus_sequence_sum_a, a_data, name, device=None)

    # Verify that calling sequence reduction on a placeholder with known
    # shape but unknown dynamic axes does not result in a problem
    p = C.placeholder(shape=(1,))
    r = C.sequence.reduce_sum(p)
    r.replace_placeholder(a)

    name = 'sequence_reduce_sum_2'
    save_onnx_model_with_validation_data(tmpdir, r, a_data, name, device=None)

def test_sequence_unpack_with_convolution(tmpdir):
    batch_size = 1
    x = C.sequence.input((20, 20), dtype=np.float32)
    y = C.sequence.unpack(x, 0, no_mask_output=True)
    z = C.reshape(y, (3, 20, 20))
    kernel = C.constant(1.0, (4, 3, 3, 3), device=None)
    t = C.convolution(kernel, z, auto_padding=[False, True, True])
    val = np.random.random((batch_size, 3, 20, 20)).astype(np.float32)

    name = 'sequence_unpack_with_convolution'
    save_onnx_model_with_validation_data(tmpdir, t, val, name, device=None)

def test_sequence_unpack_with_broadcast_as(tmpdir):
    x = C.sequence.input_variable(5)
    a = C.sequence.input_variable(4, sequence_axis=C.Axis('a'))
    y, mask = C.sequence.unpack(x, 0).outputs
    bvm = C.sequence.broadcast_as(0 * C.reduce_sum(y) + mask, a)

    x1 = np.arange(7 * 5).reshape(1, 7, 5).astype('f')
    a1 = np.arange(3 * 4).reshape(1, 3, 4).astype('f')

    name = 'sequence_unpack_with_broadcast_as'
    save_onnx_model_with_validation_data(tmpdir, bvm, [x1, a1], name, device=None)

##################################################
# END onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\ops\tests\sequence_test.py
##################################################

##################################################
# BEGIN onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\layers\tests\layers_test.py
##################################################
def test_recurrence(tmpdir):
    inputAxis = C.Axis('inputAxis')
    stateAxis = C.Axis('stateAxis')
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

    name = 'recurrence_F'
    save_onnx_model_with_validation_data(tmpdir, F, x, name, device=None)

    ####################################################
    # Test 2: RecurrenceFrom(): initial state is data input
    ####################################################
    RF = RecurrenceFrom(GRU(5, init=0.05), go_backwards=False)
    @Function
    @Signature(s = StateSequence[Tensor[5]], x = InputSequence[Tensor[5]])
    def FF(s, x):
        return RF(s, x)
    s = np.ones((1,1,5), dtype=np.float32) * 0.1 # we pass the same value as the constant in the previous test to make the result the same
    rt = FF(s, x)

    name = 'recurrence_FF'
    save_onnx_model_with_validation_data(tmpdir, FF, [s, x], name, device=None)

def test_recurrence_step_fun(tmpdir):
    def test_rec_model(rec, name, tmpdir):
        batch, sequence_length, feature = 1, 3, 2
        model = rec(C.sequence.input_variable((feature,))) 
        data = generate_sequential_data((batch,sequence_length,feature))
        
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

    def step_f(prev1, x):
        return prev1 * x
    rec = Recurrence(step_f)
    test_rec_model(rec, "recurrence_step_fun1.onnx", tmpdir)

    # TODO: CNTK Forward shape error
    #def step_f(prev1, prev2, x):
    #    return prev1 * x, prev2 * x
    #rec = Recurrence(step_f)
    #test_rec_model(rec, "test_recurrence_step_fun2.onnx", tmpdir)

def test_recurrence_fun(tmpdir):
    from cntk.ops import plus
    ####################################################
    # Test 1: sum-reduction over sequence
    ####################################################
    r = Fold(plus)
    model = r(C.sequence.input_variable((1,)))
    data = np.array([[[2], [6], [4], [8], [6]]], dtype=np.float32)   # simple sequence
    name = "recurrence_fun_plus"
    save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

    ####################################################
    # Test 2: max-pool over sequence
    ####################################################
    r = Fold(element_max)
    model = r(C.sequence.input_variable((2,)))
    data = np.array([[[2,1], [6,3], [4,2], [8,1], [6,0]]], dtype=np.float32)   # simple sequence
    name = "recurrence_fun_element_max"
    save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_unfold(tmpdir):
    from cntk.layers import UnfoldFrom

    @Function
    def double_up(s):
        return s * 2 + 1
    x = np.array([[[0],[2],[7],[12],[10]]], dtype=np.float32)

    ####################################################
    # Test 1: simple unfold
    ####################################################
    x_axis_like = C.sequence.input_variable(shape=(1,), name="x_axis_like")
    UF = UnfoldFrom(double_up)(Constant(1, (), name="init_value"), x_axis_like)

    name = "unfold"
    save_onnx_model_with_validation_data(tmpdir, UF, x, name, device=None)

def test_recurrent_block(tmpdir): 
    def run_test_recurrent_block(block_type, block_outputs_count, block_size, W_mult, H_mult, expected_res, tmpdir):
        input_shape = 4

        sequenceAxis = Axis('sequenceAxis')

        y = C.input_variable(input_shape, dynamic_axes=[Axis.default_batch_axis(), sequenceAxis])
        data = np.reshape(np.arange(0,16, dtype=np.float32), (1,4,4))

        rnn_block = block_type(block_size, init=0.1)

        assert len(rnn_block.outputs) == block_outputs_count
        rnn_net = Recurrence(rnn_block)(y)

        name = "test_recurrent_block" + block_type.__name__
        save_onnx_model_with_validation_data(tmpdir, rnn_net, data, name, device=None)

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
    for c in RECURRENT_BLOCK_DATA:
        block_type, block_outputs_count, block_size, W_mult, H_mult, expected_res = c
        run_test_recurrent_block(block_type, block_outputs_count, block_size, W_mult, H_mult, expected_res, tmpdir)

def test_sequential_convolution_1d_without_reduction_dim_old(tmpdir):
    for pad in [False, True]:        
        conv = Convolution(3, init=np.array([4., 2., 1.], dtype=np.float32), sequential=True, pad=pad, reduction_rank=0, bias=False)
        model = conv(C.sequence.input_variable(()))
        name = "sequential_convolution_1d_without_reduction_dim_old_1_" + ("pad" if pad else "nopad")
        data = np.array([[2., 6., 4., 8., 6.]], dtype=np.float32)   # like a short audio sequence, in the dynamic dimension
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        # Filter shape (3, 1) instead of 3 should be more reasonable.
        # e.g. Input shape [#] x [1] matches filter shape [3, 1], where as input shape [#] x [] matches filter shape [3].
        #      Input shape [#] x [3] matches filter shape [3, 2].
        # This setup will not be supported in the newer version SequentialConvolution.
        conv = Convolution(3, init=np.array([4., 2., 1.], dtype=np.float32), sequential=True, pad=pad, reduction_rank=0, bias=False)
        model = conv(C.sequence.input_variable((1,)))
        name = "sequential_convolution_1d_without_reduction_dim_old_2" + ("pad" if pad else "nopad")
        data = np.array([[[2.], [6], [4.], [8.], [6.]]], dtype=np.float32)
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        batch, sequence, feature, emb_dim = 1, 7, 20, 10
        x = C.sequence.input_variable((feature,))
        m = Embedding(emb_dim)(x)
        model = Convolution(filter_shape=3, sequential=True, pad=pad)(m)
        name = "sequential_convolution_1d_without_reduction_dim_old_3" + ("pad" if pad else "nopad")
        data = generate_sequence_data(batch, sequence, feature)
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)


        batch, sequence, feature, emb_dim = 1, 7, 20, 10
        x = C.sequence.input_variable((feature,))
        m = Embedding(emb_dim)(x)
        m = reshape(m, (emb_dim,1))
        model = Convolution(filter_shape=(3,1), num_filters=13, pad=pad, sequential=True)(m)
        name = "sequential_convolution_1d_without_reduction_dim_old_4" + ("pad" if pad else "nopad")
        data = generate_sequence_data(batch, sequence, feature)
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_sequential_convolution_1d_without_reduction_dim_new(tmpdir):
    ### new SequentialConvolution
    for pad in [False, True]:
        conv = SequentialConvolution(3, init=np.array([4., 2., 1.], dtype=np.float32), pad=pad, reduction_rank=0, bias=False)
        model = conv(C.sequence.input_variable(()))

        data = np.array([[2., 6., 4., 8., 6.]], dtype=np.float32)   # like a short audio sequence, in the dynamic dimension
        name = "sequential_convolution_1d_without_reduction_dim_new_1" + ("pad" if pad else "nopad")

        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        ##Filter shape (3, 1) instead of 3 should be more reasonable.
        ##e.g. Input shape [#] x [1] matches filter shape [3, 1], where as input shape [#] x [] matches filter shape [3].
        ##     Input shape [#] x [3] matches filter shape [3, 2].
        ##This setup will not be supported in the newer version SequentialConvolution.

        conv = SequentialConvolution((3,1), init=np.array([4., 2., 1.], dtype=np.float32).reshape((3,1)), pad=pad, reduction_rank=0, bias=False)
        model = conv(C.sequence.input_variable((1,)))
        data = np.array([[[2], [6], [4], [8], [6]]], dtype=np.float32)
        name = "sequential_convolution_1d_without_reduction_dim_new_2" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        batch, sequence, feature, emb_dim = 1, 7, 20, 10
        x = C.input_variable(**Sequence[Tensor[feature]])
        m = Embedding(emb_dim)(x)
        model = SequentialConvolution(filter_shape=3, pad=pad)(m)

        data = generate_sequence_data(batch, sequence, feature)
    
        name = "sequential_convolution_1d_without_reduction_dim_new_3" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        batch, sequence, feature, emb_dim = 1, 7, 20, 10
        x = C.input_variable(**Sequence[Tensor[feature]])
        m = Embedding(emb_dim)(x)
        m = reshape(m, (emb_dim,1))
        model = SequentialConvolution(filter_shape=(3,1), num_filters=13, pad=pad)(m)

        data = generate_sequence_data(batch, sequence, feature)
        name = "sequential_convolution_1d_without_reduction_dim_new_4" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        batch, sequence, feature, emb_dim = 1, 7, 20, 10
        x = C.input_variable(**Sequence[Tensor[feature]])
        m = Embedding(emb_dim)(x)
        model = SequentialConvolution(filter_shape=3, pad=pad)(m)

        data = generate_sequence_data(batch, sequence, feature)
        name = "sequential_convolution_1d_without_reduction_dim_new_5" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)


def test_sequential_convolution_2d_without_reduction_dim(tmpdir):

    data = np.asarray([[0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.3, 0.4, 0.5, 0.6], [2.5, 2.3, 2.1, 1.9, 1.7]], dtype=np.float32)
    data = data.reshape((1, 3, 5))

    for pad in [False, True]:
        c = SequentialConvolution((3,2), pad=pad, bias=False, reduction_rank=0)
        model = c(C.sequence.input_variable((5,)))
        name = "sequential_convolution_2d_without_reduction_dim_1" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        c = SequentialConvolution((3,2), pad=pad, strides=2, bias=False, reduction_rank=0)
        model = c(C.sequence.input_variable((5,)))
        name = "sequential_convolution_2d_without_reduction_dim_2" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_sequential_convolution_1d(tmpdir):
    data = np.asarray([0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
    data = data.reshape((1, 5, 1))
    for pad in [False, True]:
        c = SequentialConvolution(3, pad=pad, bias=False)
        model = c(C.sequence.input_variable((1,)))  # input is a sequence of scalars
        name = "sequential_convolution_1d_1" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        c = SequentialConvolution(2, pad=pad, strides=2, bias=False)
        model = c(C.sequence.input_variable((1,)))  # input is a sequence of scalars
        name = "sequential_convolution_1d_2" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        c = SequentialConvolution(2, num_filters=3, pad=pad, bias=False)
        model = c(C.sequence.input_variable((1,)))
        name = "sequential_convolution_1d_3" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        c = SequentialConvolution(2, num_filters=3, pad=pad, init_bias=np.asarray([1,2,3], dtype=np.float32))
        model = c(C.sequence.input_variable((1,)))
        name = "sequential_convolution_1d_4" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_sequential_convolution_1d_channel_filter(tmpdir):
    data = np.asarray([[[0.4, 0.2], [0.6, 0.1], [0.8, 1.5], [1.0, 3.2], [1.2, 1.8]]], dtype=np.float32)
    data = data.reshape((1, 5, 2))
    for pad in [False, True]:
        x = C.sequence.input_variable((2,))

        c = SequentialConvolution(3, num_filters=4, pad=pad, bias=False)
        model = c(x)
        name = "sequential_convolution_1d_channel_filter" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_sequential_convolution_2d(tmpdir):
    data = np.asarray([[0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.3, 0.4, 0.5, 0.6], [2.5, 2.3, 2.1, 1.9, 1.7]], dtype=np.float32)
    data = data.reshape((1, 3, 1, 5))

    for pad in [False, True]:
        c = SequentialConvolution((3,2), pad=pad, bias=False)
        model = c(C.sequence.input_variable((1, 5)))

        name = "sequential_convolution_2d_1" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        c = SequentialConvolution((3,2), pad=pad, strides=2, bias=False)
        model = c(C.sequence.input_variable((1, 5)))
        name = "sequential_convolution_2d_2" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        data = np.ones((1,3,1,5), dtype=np.float32)
        init_bias = np.asarray([1,2,3,4,5], dtype=np.float32) if pad else np.asarray([1,2,3,4], dtype=np.float32)

        c = SequentialConvolution((3,2), pad=pad, num_filters=4, init_bias=init_bias)
        model = c(C.sequence.input_variable((1, 5)))

        name = "sequential_convolution_2d_3" + ("pad" if pad else "nopad")
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

        c = SequentialConvolution(filter_shape=(3,2), pad=pad, strides=(2,2), num_filters=4)
        model = c(C.sequence.input_variable((1, 5)))
        name = "sequential_convolution_2d_4" + ("pad" if pad else "nopad")
        data = np.ones((1,3,1,5), dtype=np.float32)
        save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)


##################################################
# END onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\layers\tests\layers_test.py
##################################################
##################################################
# BEGIN onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\layers\tests\higher_order_layers_test.py
##################################################
INPUT_DATA = [[4, 7, 9]]

def test_sequential_clique_with_functions(tmpdir):
    x = C.input_variable(len(INPUT_DATA[0]))
    seq_clique = SequentialClique([abs, sqrt, square], name="my_clique")(x)

    np_data = np.asarray(INPUT_DATA, np.float32)
    
    name = "sequential_clique_with_functions"
    save_onnx_model_with_validation_data(tmpdir, seq_clique, np_data, name, device=None)

def test_sequential_clique_with_layers(tmpdir):
    input_elements, expected = 5, 360.0
    x = C.input_variable(input_elements)

    np_data = [np.arange(input_elements, dtype=np.float32)]

    unit_dense = Dense(input_elements, activation=None, init=1)

    seq_clique = SequentialClique([unit_dense, unit_dense, unit_dense])(x)

    name = "sequential_clique_with_layers"
    save_onnx_model_with_validation_data(tmpdir, seq_clique, np_data, name, device=None)


def test_sequential_constructor(tmpdir):
    x = C.input_variable(len(INPUT_DATA[0]))
    np_data = np.asarray(INPUT_DATA, np.float32)

    seq_layers = Sequential([abs, sqrt, square, cos])(x)

    name = "sequential_constructor"
    save_onnx_model_with_validation_data(tmpdir, seq_layers, np_data, name, device=None)


##################################################
# END onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\layers\tests\higher_order_layers_test.py
##################################################

##################################################
# END onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\layers\tests\models_tests.py
##################################################
def test_attention_model(tmpdir):
    attention_dim = 128
    # this configuration settings causing deprecated compatible_attention_mode old_attention. It produces 
    # a splice node with errorous shape: [#](2) => Splice => [#](20, 1, 2). 
    #attention_span = 20
    #attention_axis = -3
    attention_span = None
    attention_axis = None
    feature_shape = 2

    att_model = C.layers.AttentionModel(attention_dim, attention_span, attention_axis, name='attention_model')
    model=att_model(C.sequence.input_variable((feature_shape,)), C.sequence.input_variable((feature_shape,)))

    batch_size, sequence_length = 1, 50
    data1 = generate_sequence_data(batch_size, sequence_length, feature_shape)
    data2 = generate_sequence_data(batch_size, sequence_length, feature_shape)

    name = "attention_model"
    save_onnx_model_with_validation_data(tmpdir, model, [data1, data2], name, device=None)

##################################################
# END onnx test ported from E:\cntk\CNTKRel2.2\CNTK\bindings\python\cntk\layers\tests\models_tests.py
##################################################

##################################################
# other sequence tests
##################################################

def test_Delay(tmpdir): 
    x = C.input_variable(**Sequence[Tensor[2]])
    x0 = np.reshape(np.arange(6,dtype=np.float32),(1,3,2))

    make_trigram = Sequential([tuple(Delay(T) for T in (-1,0,1)), splice])
    y = make_trigram(x)
    name = "Delay"
    save_onnx_model_with_validation_data(tmpdir, y, x0, name, device=None)

def test_SimpleRecurrence(tmpdir):
    def step_f(prev1, x):
        return prev1 + x
    rec = Recurrence(step_f, go_backwards = True)
    batch_size, seq_len, feature_size = 1, 2, 2
    model = rec(C.sequence.input_variable((feature_size,)))
    
    data = generate_sequential_data((batch_size, seq_len, feature_size))
    name = "SimpleRecurrence"
    save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_SimpleUniDirectionRNN(tmpdir):
    feature_shape = (1,)
    plus_constant = np.full(shape=feature_shape, fill_value=10, dtype=np.float32)
    data = np.array([[[1], [2], [3], [4]]]).astype(np.float32)

    x = C.sequence.input_variable(feature_shape, dtype=np.float32)
    p1 = C.placeholder()
    p3 = p1 + x
    p5 = C.sequence.past_value(p3, np.float32(1.3))

    name, directPastValueOutput, model = None, False, None
    if directPastValueOutput:
        name = "SimpleUniDirectionRNN_5"
        p6 = p5 + plus_constant
        model = p6.replace_placeholders({p1: p5})
    else:
        name = "SimpleUniDirectionRNN_6"
        p6 = p5 + np.array(plus_constant, dtype=np.float32);
        model = p6.replace_placeholders({p1: p6}) + plus_constant
    
    save_onnx_model_with_validation_data(tmpdir, model, data, name, device=None)

def test_SimpleBiDirectionRNN(tmpdir):
    ## expected exception. This is a desired behaivor. A CNTK scan loop with only have a single direction. 
    pytest.skip('test_SimpleBiDirectionRNN expected exception. This is a desired behaivor. A CNTK scan loop with only have a single direction.')
    x = C.sequence.input_variable((2,), dtype=np.float32)
    p1 = C.placeholder()
    p2 = C.placeholder()
    p3 = p1 + x
    p4 = p3 + p2
    p5 = C.sequence.past_value(p4)
    p6 = C.sequence.future_value(p5)
    p7 = p6 + x
    model = p7.replace_placeholders({p1: p5, p2: p6})
    
    plot_graph(model, tmpdir + "/SimpleBiDirectionRNN")

    data = np.array([[[1, 2], [2, 3], [3,4], [4,5]]]).astype(np.float32)

    # expected exception. This is a desired behaivor. A CNTK scan loop with only have a single direction. 
    # ValueError: It is not allowed to have multiple different stepping directions in the same loop (loop connected to PastValue349 PastValue operation).
    model.eval(data)

def test_SequenceDelay(tmpdir):
    v = C.sequence.input_variable(shape=(2))
    seq_data = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]], dtype=np.float32)

    future_v = C.sequence.delay(v, initial_state=0, time_step=-1)
    testName = "SequenceDelayFuture"
    save_onnx_model_with_validation_data(tmpdir, future_v, seq_data, testName, device=None)

    future_v = C.sequence.delay(v, initial_state=[5.0,6.0], time_step=-1)
    testName = "SequenceDelayFuture_init_tensor"
    save_onnx_model_with_validation_data(tmpdir, future_v, seq_data, testName, device=None)

    future_v = C.sequence.delay(v, initial_state=[5.0,6.0], time_step=-2)
    testName = "SequenceDelayFuture_init_tensor_offset2"
    save_onnx_model_with_validation_data(tmpdir, future_v, seq_data, testName, device=None)

    past_v = C.sequence.delay(v, initial_state=0, time_step=1)
    testName = "SequenceDelayPast"
    save_onnx_model_with_validation_data(tmpdir, future_v, seq_data, testName, device=None)

    past_v = C.sequence.delay(v, initial_state=[5.0,6.0], time_step=1)
    testName = "SequenceDelayPast_init_tensor"
    save_onnx_model_with_validation_data(tmpdir, future_v, seq_data, testName, device=None)

    past_v = C.sequence.delay(v, initial_state=[5.0,6.0], time_step=2)
    testName = "SequenceDelayPast_init_tensor_offset2"
    save_onnx_model_with_validation_data(tmpdir, future_v, seq_data, testName, device=None)

def test_PastFutureValue(tmpdir):    
    x = C.sequence.input_variable((3,2))
    x0 = np.reshape(np.arange(24,dtype=np.float32),(1,4,3,2))
    y = C.sequence.past_value(x)
    test_name = "PastValue_0"
    save_onnx_model_with_validation_data(tmpdir, y, x0, test_name, device=None)

    y = C.sequence.future_value(x)
    test_name = "FutureValue_0"
    save_onnx_model_with_validation_data(tmpdir, y, x0, test_name, device=None)

    s = C.input_variable((3,2))  # not a sequence, e.g. a final encoder hidden state
    s0 = np.reshape(np.arange(6,dtype=np.float32)/2,(1,3,2))

    y = C.sequence.past_value(x, initial_state=s)
    test_name = "PastValue_1"
    save_onnx_model_with_validation_data(tmpdir, y, [x0, s0], test_name, device=None)

    y = C.sequence.future_value(x, initial_state=s)
    test_name = "FutureValue_1"
    save_onnx_model_with_validation_data(tmpdir, y, [x0, s0], test_name, device=None)

    y = C.sequence.past_value(x, initial_state=np.float32(1.23))
    test_name = "PastValue_2"
    save_onnx_model_with_validation_data(tmpdir, y, x0, test_name, device=None)

    y = C.sequence.future_value(x, initial_state=np.float32(1.23))
    test_name = "FutureValue_2"
    save_onnx_model_with_validation_data(tmpdir, y, x0, test_name, device=None)

def test_SimpleLoopWithLSTM(tmpdir):
    hidden_dim = 3
    in_dim = 5
    x = C.sequence.input_variable((in_dim,))
    r = C.sequence.input_variable((1,)) # value of 0/1. 0 means reset
    merged_x = C.splice(x, r) # Recurrence only takes 1 input, so concatenate the two
    cell = C.layers.LSTM(hidden_dim) # (dh, dc, x) -> (h, c)
    y = C.layers.Recurrence(cell)(x)

    @C.Function
    def lstm_with_reset(dh, dc, xr):
        xx = xr[0:-1]
        rr = xr[-1]
        return cell(rr * dh, rr * dc, xx)

    yr = C.layers.Recurrence(lstm_with_reset)(merged_x)

    seq_len = [2,3,5]
    total_len = np.sum(seq_len)
    accum_seq_len = np.cumsum(seq_len)
    
    x_total_data = np.random.rand(1, total_len, in_dim).astype(np.float32)
    x_data = [np.squeeze(v) for v in np.split(x_total_data, accum_seq_len[0:-1], axis=1)]
    
    r_data = np.ones(accum_seq_len[-1])
    for i in np.nditer(accum_seq_len[0:-1]):
        r_data[i] = 0
    r_data = np.reshape(r_data, (1,-1,1)).astype(np.float32)

    model_name = "SimpleLoopWithLSTM"
    save_onnx_model_with_validation_data(tmpdir, yr, [x_total_data, r_data], model_name, device=None)

def test_SeqeunceGather(tmpdir):
    batch_size, sequence_length = 1, 3
    x0 = generate_sequential_data((batch_size, sequence_length, 2, 2))
    np.shape(x0)

    cond = np.array([[0,1,1]]).astype(np.float32)
    x = C.sequence.input_variable(shape=(2, 2))
    c = C.sequence.input_variable(shape=())
    model = C.sequence.gather(x,c)

    o0 = model.eval({x:x0, c:cond})
    o1 = np.compress(cond[0], x0, axis=1)

    np.allclose(o0, o1[0])

    test_name = "SeqeunceGather"
    save_onnx_model_with_validation_data(tmpdir, model, [x0, cond], test_name, device=None)

def test_AttLoopPastValueWindow(tmpdir):
    pytest.skip('test_AttLoopPastValueWindow expects error of SequenceSlice cannot be in a scan loop')

    batch_size, seq_len, feature = 1, 5, 1
    window_size, axis = 2, -2
    x = C.input_variable(**C.layers.typing.Sequence[C.layers.typing.Tensor[feature]])

    initial_state=C.constant([1])
    @C.Function
    def test_selfatt(x):
        with C.layers.default_options():
            @C.Function
            def att_window(d, x):
                att, _ = C.layers.PastValueWindow(window_size, axis)(x).outputs
                att = C.sequence.broadcast_as(att, x)
                return C.plus(C.minus(C.minus(C.plus(x, d), d), x), att) # preserve only att value
            y = Recurrence(att_window, initial_state=initial_state, go_backwards=False)(x)
            return y
    model = test_selfatt(x)
    x0 = generate_sequential_data((batch_size, seq_len, feature))
    o0 = model.eval(x0)
    test_name = 'AttLoopPastValueWindow'
    save_onnx_model_with_validation_data(tmpdir, model, x0, test_name, device=None)

def test_AttLoopPastValueWindow_Alternative(tmpdir):
    x = C.input_variable(**Sequence[Tensor[2]])
    def stack_window(x, window_size, go_backwards=False):
        input_shape = x.shape
        windowed_shape = (window_size,) + x.shape
        with C.layers.default_options():
            @C.Function
            def slice_and_splice(d, x):
                d = C.slice(d, 0, -window_size+1, 0) if not go_backwards else C.slice(d, 0, 0, window_size - 1)
                x = C.reshape(x, (1,) + input_shape, name="reshape_x")
                return C.splice(d, x, axis=0) if not go_backwards else C.splice(x, d, axis=0)
            y = Recurrence(slice_and_splice, initial_state=C.constant(np.zeros(windowed_shape), dtype=np.float32), go_backwards=go_backwards)(x)
            return y

    test_name = 'AttLoopPastValueWindow_Alternative'
    model = stack_window(x, 2)

    batch_size, seq_len, feature = 1, 17, 2
    x0 = generate_sequential_data((batch_size, seq_len, feature))
    save_onnx_model_with_validation_data(tmpdir, model, x0, test_name, device=None)

