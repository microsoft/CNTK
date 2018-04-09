import pytest
import numpy as np
import cntk as C
import tensorflow as tf
from cntk.contrib.paramtracker import *

import tempfile

def test_key_numpy_store_create():
    workdir = tempfile.gettempdir() + ".mytmp"
    store = KeyNumpyStore(workdir)
    store['a_b_c'] = np.ones((2, 2, 3))

    store2 = KeyNumpyStore(workdir)
    np.testing.assert_equal(store2['a_b_c'], np.ones((2, 2, 3)))
    KeyNumpyStore.clear(workdir)


def test_save_load_parameters():
    workdir = tempfile.gettempdir()
    params2 = ParameterTracker.get_instance('cntk')
    params1 = ParameterTracker.get_instance('tf')\
                            .set_workingpath(workdir)\
                            .share_values_to(params2)


    aa = np.zeros((2, 2, 3))
    bb = np.zeros((2, 4, 3))
    def get_param_value(p):
        return p

    def set_param_value(p, value):
        #should p.set_value(load_store[key])
        np.copyto(p, value)
        return [p]

    params1.track_parameter('aa', np.ones((2, 2, 3)), get_param_value, set_param_value)
    params1.track_parameter('bb', np.ones((2, 4, 3)), get_param_value, set_param_value)
    params1.save_parameters()

    params2.track_parameter('aa', aa, get_param_value, set_param_value)
    params2.track_parameter('bb', bb, get_param_value, set_param_value)
    params2.load_parameters()
    np.testing.assert_equal(aa, np.ones((2, 2, 3)))
    np.testing.assert_equal(bb, np.ones((2, 4, 3)))

def test_save_load_parameters_key_name_scope():
    workdir = tempfile.gettempdir()
    params2 = ParameterTracker.get_instance('cntk')
    params1 = ParameterTracker.get_instance('tf')\
                            .set_workingpath(workdir)\
                            .share_values_to(params2)


    aa = np.zeros((2, 2, 3))
    bb = np.zeros((2, 4, 3))
    aaa = np.zeros((2, 2, 5))
    bbb = np.zeros((2, 4, 5))

    def get_param_value(p):
        return p

    def set_param_value(p, value):
        #should p.set_value(load_store[key])
        np.copyto(p, value)
        return [p]

    params1.track_parameter('aa', np.ones((2, 2, 3)), get_param_value, set_param_value)
    with key_scope(params1, 'next_depth'):
        params1.track_parameter('aa', np.ones((2, 2, 5)), get_param_value, set_param_value)
        params1.track_parameter('bb', np.ones((2, 4, 5)), get_param_value, set_param_value)
    #test the exit of name scope:
    params1.track_parameter('bb', np.ones((2, 4, 3)), get_param_value, set_param_value)
    params1.save_parameters()

    params2.track_parameter('aa', aa, get_param_value, set_param_value)
    with key_scope(params2, 'next_depth'):
        params2.track_parameter('aa', aaa, get_param_value, set_param_value)
        params2.track_parameter('bb', bbb, get_param_value, set_param_value)
    # test the exit of name scope:
    params2.track_parameter('bb', bb, get_param_value, set_param_value)
    params2.load_parameters()
    np.testing.assert_equal(aa, np.ones((2, 2, 3)))
    np.testing.assert_equal(bb, np.ones((2, 4, 3)))
    np.testing.assert_equal(aaa, np.ones((2, 2, 5)))
    np.testing.assert_equal(bbb, np.ones((2, 4, 5)))


def test_tf2cntk_save_load_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)
    with tf.variable_scope("test_tf2cntk_save_load_parameters"):
        tf_aa = tf.get_variable('tf_weights',
                         [2,3,5],
                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    cntk_aa = C.Parameter((2,3,5), name='tf_weights', dtype=np.float32)


    params1.track_parameter('aa', tf_aa, get_value_func=get_tf_param_value)
    params2.track_parameter('aa', cntk_aa, set_value_func=set_cntk_param_value)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        params1.save_parameters(sess)
        params2.load_parameters()
        np.testing.assert_equal(get_tf_param_value(tf_aa, sess), (cntk_aa * 1.0).eval())

def test_cntk2tf_save_load_parameters():
    workdir = tempfile.gettempdir()

    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)
    with tf.variable_scope("test_cntk2tf_save_load_parameters"):
        tf_aa = tf.get_variable('tf_weights',
                         [2,3,5],
                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    cntk_aa = C.Parameter((2,3,5), init = np.ones((2,3,5)), name='tf_weights', dtype=np.float32)


    params1.track_parameter('aa', cntk_aa, get_value_func=get_cntk_param_value)
    params2.track_parameter('aa', tf_aa, set_value_func=set_tf_param_value)

    with tf.Session() as sess:
        params1.save_parameters()
        params2.load_parameters(sess)
        np.testing.assert_equal(get_tf_param_value(tf_aa, sess), get_cntk_param_value(cntk_aa))



def test_tf2cntk_save_load_embedding_parameters():
    workdir = tempfile.gettempdir()

    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)


    np_embed_matrix = np.array([[0.5, -0.6, -0.9, 0.2], [-1.0, -0.3, -0.6, 0.9],
                                [0.5, 2.0, 1.6, 0.5], [-0.1, 0.4, -1.6, 0.3],
                                [0.2, -1.0, 0.11, 0.5], [-1.7, 0.7, 0.1, 0.03]])
    with tf.variable_scope("test_tf2cntk_save_load_embedding_parameters"):
        tf_embed_param = tf.get_variable('tf_weights',
                           # np_embed_matrix.shape,
                         initializer=np_embed_matrix)
    cntk_input = C.input_variable(1, dtype=np.float32)
    cntk_embed = C.layers.Embedding( np_embed_matrix.shape[1])(C.one_hot(cntk_input, np_embed_matrix.shape[0]))

    x = tf.placeholder(tf.int32, [None, 1])
    tf_embed = tf.nn.embedding_lookup(tf_embed_param, x)
    init_op = tf.global_variables_initializer()
    exact_v = np_embed_matrix[[2,4][:]]

    params1.track_parameter('embedding', tf_embed_param, get_value_func=get_tf_param_value)
    track_cntk_embedding(params2, 'embedding', cntk_embed)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        input_data = [[2], [4]]
        tf_embedding_result = sess.run(tf_embed, {x: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()
        np.testing.assert_allclose(get_tf_param_value(tf_embed_param, sess), (cntk_embed.parameters[0] * 1.0).eval())

        cntk_embedding_result =  cntk_embed.eval({cntk_input: input_data})
        np.testing.assert_allclose(tf_embedding_result, cntk_embedding_result)


def test_tf2cntk_save_load_conv2d_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    filter_h = 3
    filter_w = 3
    in_channels = 3
    out_channels = 5
    spatial_shape = (15, 15)

    tf_input = tf.placeholder(tf.float32, [None] + list(spatial_shape) + [in_channels])
    with tf.variable_scope("test_tf2cntk_save_load_embedding_parameters"):
        conv_weights = tf.get_variable('conv_weights', [filter_h, filter_w, in_channels, out_channels],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        conv_bias = tf.get_variable('conv_bias', [out_channels], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    tf_conv = tf.tanh(tf.nn.conv2d(input=tf_input, filter=conv_weights, strides=[1, 1, 1, 1], padding='VALID') + conv_bias)

    cntk_input = C.input_variable((in_channels,) + spatial_shape, dtype=np.float32)
    cntk_conv = C.layers.Convolution2D(num_filters=out_channels, filter_shape=(filter_h, filter_w), strides=1, pad=False, activation=C.tanh)(cntk_input)

    init_op = tf.global_variables_initializer()

    # params1.track_parameter('conv.weights', conv_weights, get_value_func=get_tf_param_value)
    # params1.track_parameter('conv.bias', conv_bias, get_value_func=get_tf_param_value)
    #
    #
    # params2.track_parameter('conv.weights', cntk_conv.parameters[0], set_value_func=set_cntk_conv2d_weights_from_tf)
    # params2.track_parameter('conv.bias', cntk_conv.parameters[1], set_value_func=set_cntk_conv2d_bias_from_tf)
    params1.track_parameter('conv', (conv_weights, conv_bias), get_value_func=tf_get_conv2d_param)
    params2.track_parameter('conv', cntk_conv, set_value_func=set_cntk_conv2d_params_from_tf)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size,) + spatial_shape + (in_channels,)).astype(np.float32)
        tf_conv_result = sess.run(tf_conv, {tf_input: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()
        # tf filter  shape: [filter_height, filter_width, in_channels, out_channels]
        # cntk filter shape: [out_channels, in_channels, filter_height, filter_width]
        np.testing.assert_allclose(get_tf_param_value(conv_weights, sess).transpose(3, 2, 0, 1), (cntk_conv.parameters[0] * 1.0).eval())
        np.testing.assert_allclose(get_tf_param_value(conv_bias, sess).reshape(cntk_conv.parameters[1].shape), (cntk_conv.parameters[1] * 1.0).eval())

        cntk_conv_result = cntk_conv.eval({cntk_input: input_data.transpose(0, 3, 1, 2)})
        #tf output shape: [batch, h, w, out_channels]
        #cntk output shape: [batch, out_channels, h, w]
        np.testing.assert_array_almost_equal(tf_conv_result.transpose(0, 3, 1, 2), cntk_conv_result)

def test_tf2cntk_save_load_contrib_conv2d_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    filter_h = 3
    filter_w = 3
    in_channels = 3
    out_channels = 5
    spatial_shape = (15, 15)

    tf_input = tf.placeholder(tf.float32, [None] + list(spatial_shape) + [in_channels])
    conv2d_scope = "test_tf2cntk_save_load_embedding_parameters"
    with tf.variable_scope(conv2d_scope):
        tf_conv = tf.contrib.layers.conv2d(tf_input, out_channels, [filter_h, filter_w], stride=1, padding='VALID', activation_fn=tf.tanh)


    cntk_input = C.input_variable((in_channels,) + spatial_shape, dtype=np.float32)
    cntk_conv = C.layers.Convolution2D(num_filters=out_channels, filter_shape=(filter_h, filter_w), strides=1, pad=False, activation=C.tanh)(cntk_input)

    init_op = tf.global_variables_initializer()

    params1.track_parameter('conv', conv2d_scope, get_value_func=get_tf_contrib_conv2d_param_value)
    params2.track_parameter('conv', cntk_conv, set_value_func=set_cntk_conv2d_params_from_tf)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size,) + spatial_shape + (in_channels,)).astype(np.float32)
        tf_conv_result = sess.run(tf_conv, {tf_input: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()
        # tf filter  shape: [filter_height, filter_width, in_channels, out_channels]
        # cntk filter shape: [out_channels, in_channels, filter_height, filter_width]
        tf_param_value = get_tf_contrib_conv2d_param_value(conv2d_scope, sess)
        np.testing.assert_allclose(tf_param_value['W'].transpose(3, 2, 0, 1), (cntk_conv.parameters[0] * 1.0).eval())
        np.testing.assert_allclose(tf_param_value['b'].reshape(cntk_conv.parameters[1].shape), (cntk_conv.parameters[1] * 1.0).eval())


        cntk_conv_result = cntk_conv.eval({cntk_input: input_data.transpose(0, 3, 1, 2)})
        #tf output shape: [batch, h, w, out_channels]
        #cntk output shape: [batch, out_channels, h, w]
        np.testing.assert_array_almost_equal(tf_conv_result.transpose(0, 3, 1, 2), cntk_conv_result)

def test_tf2cntk_save_load_rnn_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    input_dim = 13
    seq_len = 10
    hiddm_dim = 4

    tf_input = tf.placeholder(tf.float32, [None, seq_len, input_dim])
    with tf.variable_scope("rnn1"):
        cell = tf.nn.rnn_cell.LSTMCell(hiddm_dim, forget_bias=0)
        tf_rnn, tf_rnn_state = tf.nn.dynamic_rnn(cell, tf_input, dtype=tf.float32)

    tf_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn')
    tf_weights = [p for p in tf_parameters if 'kernel' in p.name][0]
    tf_bias = [p for p in tf_parameters if 'bias' in p.name][0]


    cntk_input = C.sequence.input_variable(input_dim, dtype=np.float32)
    cntk_rnn = C.layers.Recurrence(C.layers.LSTM(hiddm_dim, init_bias=C.glorot_uniform()))(cntk_input)

    params1.track_parameter('rnn', ('rnn1', tf_rnn, tf_rnn_state), get_value_func=get_tf_lstm_param_value)
    params2.track_parameter('rnn', cntk_rnn, set_value_func=set_cntk_lstm_param_from_tf)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size, seq_len, input_dim)).astype(np.float32)
        tf_rnn_result = sess.run(tf_rnn, {tf_input: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()
        b, W, H = cntk_rnn.parameters
        cntk_weights = C.splice(W, H, axis=0)
        np.testing.assert_array_almost_equal(get_tf_param_value(tf_weights, sess),
                                   (cntk_weights * 1.0).eval())
        np.testing.assert_array_almost_equal(get_tf_param_value(tf_bias, sess),
                                   (b * 1.0).eval())

        cntk_rnn_result = cntk_rnn.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_result, cntk_rnn_result)


def test_tf2cntk_save_load_bidr_rnn_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    input_dim = 13
    seq_len = 10
    hiddm_dim = 4

    tf_input = tf.placeholder(tf.float32, [None, seq_len, input_dim])
    with tf.variable_scope("rnn2"):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddm_dim, forget_bias=0)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddm_dim, forget_bias=0)
        (fw, bw), (fws, bws) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, tf_input, dtype=tf.float32)

    tf_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn')
    tf_fw_weights = [p for p in tf_parameters if 'fw' in p.name and 'kernel' in p.name][0]
    tf_fw_bias = [p for p in tf_parameters if 'fw' in p.name and 'bias' in p.name][0]
    tf_bw_weights = [p for p in tf_parameters if 'bw' in p.name and 'kernel' in p.name][0]
    tf_bw_bias = [p for p in tf_parameters if 'bw' in p.name and 'bias' in p.name][0]


    cntk_input = C.sequence.input_variable(input_dim, dtype=np.float32)
    cntk_fw, cntk_bw = C.layers.Recurrence(C.layers.LSTM(hiddm_dim, init_bias=C.glorot_uniform()))(cntk_input), C.layers.Recurrence(C.layers.LSTM(hiddm_dim), go_backwards=True)(cntk_input)

    params1.track_parameter('rnn_fw', ('rnn2/bidirectional_rnn/fw', fw, fws), get_value_func=get_tf_lstm_param_value)
    params2.track_parameter('rnn_fw', cntk_fw, set_value_func=set_cntk_lstm_param_from_tf)

    params1.track_parameter('rnn_bw', ('rnn2/bidirectional_rnn/bw', bw, bws), get_value_func=get_tf_lstm_param_value)
    params2.track_parameter('rnn_bw', cntk_bw, set_value_func=set_cntk_lstm_param_from_tf)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size, seq_len, input_dim)).astype(np.float32)
        tf_rnn_fw_result = sess.run(fw, {tf_input: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()

        fw_b, fw_W, fw_H = cntk_fw.parameters
        cntk_fw_weights = C.splice(fw_W, fw_H, axis=0)
        np.testing.assert_array_almost_equal(get_tf_param_value(tf_fw_weights, sess),
                                   (cntk_fw_weights * 1.0).eval())
        np.testing.assert_array_almost_equal(get_tf_param_value(tf_fw_bias, sess),
                                   (fw_b * 1.0).eval())
        cntk_rnn_fw_result = cntk_fw.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_fw_result, cntk_rnn_fw_result)

        tf_rnn_bw_result = sess.run(bw, {tf_input: input_data})
        bw_b, bw_W, bw_H = cntk_bw.parameters
        cntk_bw_weights = C.splice(bw_W, bw_H, axis=0)
        np.testing.assert_array_almost_equal(get_tf_param_value(tf_bw_weights, sess),
                                   (cntk_bw_weights * 1.0).eval())
        np.testing.assert_array_almost_equal(get_tf_param_value(tf_bw_bias, sess),
                                   (bw_b * 1.0).eval())
        cntk_rnn_bw_result = cntk_bw.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_bw_result, cntk_rnn_bw_result)


def get_tf_gru_param_value(rnn_func, sess):
    '''

    Args:
        rnn_func: (rnn_var_scope, rnn_output, rnn_state) where runn_var_scope is a string to identify the scope of the LSTM weight and bias parameters.
        sess: tensorflow session

    Returns: Numpy array fo the rnn parameter values: (rnn_weights, rnn_bias)

    '''
    import tensorflow as tf
    rnn_var_scope, fw, fw_s = rnn_func
    tf_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, rnn_var_scope)
    rnn_weights = [p for p in tf_parameters if 'kernel' in p.name][0]
    rnn_bias = [p for p in tf_parameters if 'bias' in p.name][0]
    return sess.run({'W': rnn_weights, 'b': rnn_bias})


def test_tf2cntk_save_load_gru_rnn_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    input_dim = 13
    seq_len = 10
    hiddm_dim = 4

    tf_input = tf.placeholder(tf.float32, [None, seq_len, input_dim])
    with tf.variable_scope("rnn3"):
        #cell = tf.nn.rnn_cell.GRUCell(hiddm_dim)
        cell = tf.contrib.rnn.GRUBlockCell(hiddm_dim)
        tf_rnn, tf_rnn_state = tf.nn.dynamic_rnn(cell, tf_input, dtype=tf.float32)
    cntk_input = C.sequence.input_variable(input_dim, dtype=np.float32)
    cntk_rnn = C.layers.Recurrence(C.layers.GRU(hiddm_dim, init_bias=C.glorot_uniform()))(cntk_input)

    tf_parameters = get_tf_vars('rnn', ['w_ru', 'b_ru', 'w_c', 'b_c'])
    params1.track_parameter('rnn', ('rnn3', tf_rnn, hiddm_dim, input_dim), get_value_func=get_tf_contrib_gru_param_value)
    params2.track_parameter('rnn', cntk_rnn, set_value_func=set_cntk_gru_param_from_tf_contrib)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size, seq_len, input_dim)).astype(np.float32)
        tf_rnn_result = sess.run(tf_rnn, {tf_input: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()
        cntk_rnn_result = cntk_rnn.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_result, cntk_rnn_result)


def test_tf2cntk_save_load_gru_birnn_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    input_dim = 13
    seq_len = 10
    hiddm_dim = 4

    tf_input = tf.placeholder(tf.float32, [None, seq_len, input_dim])
    with tf.variable_scope("rnn4"):
        fw_cell =  tf.contrib.rnn.GRUBlockCell(hiddm_dim)
        bw_cell =  tf.contrib.rnn.GRUBlockCell(hiddm_dim)
        (fw, bw), (fws, bws) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, tf_input, dtype=tf.float32)

    cntk_input = C.sequence.input_variable(input_dim, dtype=np.float32)
    cntk_fw_cell = C.layers.GRU(hiddm_dim, init_bias=C.glorot_uniform())
    cntk_bw_cell = C.layers.GRU(hiddm_dim, init_bias=C.glorot_uniform())

    cntk_fw, cntk_bw = C.layers.Recurrence(cntk_fw_cell)(cntk_input), C.layers.Recurrence(cntk_bw_cell, go_backwards=True)(cntk_input)

    params1.track_parameter('rnn_fw', ('rnn4/bidirectional_rnn/fw', fw, hiddm_dim, input_dim), get_value_func=get_tf_contrib_gru_param_value)
    params2.track_parameter('rnn_fw', cntk_fw, set_value_func=set_cntk_gru_param_from_tf_contrib)

    params1.track_parameter('rnn_bw', ('rnn4/bidirectional_rnn/bw', bw, hiddm_dim, input_dim), get_value_func=get_tf_contrib_gru_param_value)
    params2.track_parameter('rnn_bw', cntk_bw, set_value_func=set_cntk_gru_param_from_tf_contrib)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size, seq_len, input_dim)).astype(np.float32)
        tf_rnn_fw_result = sess.run(fw, {tf_input: input_data})

        params1.save_parameters(sess)
        params2.load_parameters()

        cntk_rnn_fw_result = cntk_fw.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_fw_result, cntk_rnn_fw_result)

        tf_rnn_bw_result = sess.run(bw, {tf_input: input_data})
        cntk_rnn_bw_result = cntk_bw.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_bw_result, cntk_rnn_bw_result)

def test_tf2cntk_save_load_gru_birnn_single_tf_cell_template_parameters():
    workdir = tempfile.gettempdir()
    params1 = ParameterTracker(workdir)
    params2 = ParameterTracker(workdir)

    input_dim = 13
    seq_len = 10
    hiddm_dim = 4

    tf_input = tf.placeholder(tf.float32, [None, seq_len, input_dim])
    with tf.variable_scope("rnn5"):
        fw_cell =  tf.contrib.rnn.GRUBlockCell(hiddm_dim)
        bw_cell =  fw_cell
        #in tf, even with one template actually two instances are created
        (fw, bw), (fws, bws) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, tf_input, dtype=tf.float32)

    cntk_input = C.sequence.input_variable(input_dim, dtype=np.float32)
    cntk_fw_cell = C.layers.GRU(hiddm_dim, init_bias=C.glorot_uniform())
    cntk_bw_cell = C.layers.GRU(hiddm_dim, init_bias=C.glorot_uniform())

    cntk_fw, cntk_bw = C.layers.Recurrence(cntk_fw_cell)(cntk_input), C.layers.Recurrence(cntk_bw_cell, go_backwards=True)(cntk_input)

    params1.track_parameter('rnn_fw', ('rnn5/bidirectional_rnn/fw', fw, hiddm_dim, input_dim), get_value_func=get_tf_contrib_gru_param_value)
    params2.track_parameter('rnn_fw', cntk_fw, set_value_func=set_cntk_gru_param_from_tf_contrib)

    params1.track_parameter('rnn_bw', ('rnn5/bidirectional_rnn/bw', bw, hiddm_dim, input_dim), get_value_func=get_tf_contrib_gru_param_value)
    params2.track_parameter('rnn_bw', cntk_bw, set_value_func=set_cntk_gru_param_from_tf_contrib)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 3
        input_data = np.random.random((batch_size, seq_len, input_dim)).astype(np.float32)
        tf_rnn_fw_result = sess.run(fw, {tf_input: input_data})

        params1.save_parameters(sess)
        loaded_parms = params2.load_parameters()
        print('loaded cntk params: ', loaded_parms)

        cntk_rnn_fw_result = cntk_fw.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_fw_result, cntk_rnn_fw_result)

        tf_rnn_bw_result = sess.run(bw, {tf_input: input_data})
        cntk_rnn_bw_result = cntk_bw.eval({cntk_input: input_data})
        np.testing.assert_array_almost_equal(tf_rnn_bw_result, cntk_rnn_bw_result)