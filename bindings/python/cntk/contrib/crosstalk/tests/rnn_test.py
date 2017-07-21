import numpy as np
from cntk.contrib import crosstalk as cstk
import tempfile
workdir = tempfile.gettempdir()

data_cntk = [[[1,2,3,4,5],[2,2,3,4,5],[3,2,3,4,5]], [[4,2,3,4,5],[5,2,3,4,5]]]
data_tf   = [[[1,2,3,4,5],[2,2,3,4,5],[3,2,3,4,5]], [[4,2,3,4,5],[5,2,3,4,5],[0,0,0,0,0]]]
data_tf_len = [3,2]
max_seq_len = max(data_tf_len)
batch_size = len(data_tf)

in_dim = 5
dim = 3

def cntk_baseline_lstm():
    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance
    input_var = C.sequence.input_variable(shape=(in_dim))
    fwbw = C.splice(C.layers.Recurrence(C.layers.LSTM(dim, init_bias=C.glorot_uniform()))(input_var), C.layers.Recurrence(C.layers.LSTM(dim), go_backwards=True)(input_var))
    ci.watch(fwbw, 'birnn', var_type=cstk.RnnAttr,
          attr=cstk.RnnAttr(bidirectional=True, op_type='lstm', input_dim=in_dim, hidden_dim=dim, forget_bias=0))
    ci.watch(fwbw, 'birnn_out')

    data = {input_var:data_cntk}
    ci.set_data(data)
    ci.set_workdir(workdir)
    ci.fetch('birnn', save=True)
    ci.fetch('birnn_out', save=True)
    ci.reset()

def tf_baseline_lstm():
    import tensorflow as tf
    import cntk.contrib.crosstalk.crosstalk_tensorflow as crtf
    ci = crtf.instance

    tf.reset_default_graph()

    with tf.variable_scope("rnn"):
        x = tf.placeholder(tf.float32, [batch_size, max_seq_len, in_dim])
        l = tf.placeholder(tf.int32, [batch_size])

        if tf.VERSION.startswith('0.12'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(dim)
            (fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, x, l, dtype=tf.float32)
            scope = 'rnn/BiRNN'
        elif tf.VERSION.startswith('1'):
            (fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.BasicLSTMCell(dim), tf.contrib.rnn.BasicLSTMCell(dim), x, l, dtype=tf.float32)
            scope = 'rnn/bidirectional_rnn'
        else:
            raise Exception('only supports 0.12.* and 1.*')

        ci.watch(scope,
                  'birnn', var_type=cstk.RnnAttr,
                  attr=cstk.RnnAttr(bidirectional=True, op_type='lstm', input_dim=in_dim, hidden_dim=dim, forget_bias=1)) # tf default forget_bias==1

        if tf.VERSION.startswith('0.12'):
            output = tf.concat(2, [fw, bw])
        elif tf.VERSION.startswith('1'):
            output = tf.concat([fw,bw], 2)
        else:
            raise Exception('only supports 0.12.* and 1.*')

        ci.watch(output, 'birnn_out', var_type=crtf.VariableType)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = {x:data_tf, l:data_tf_len}
        ci.set_workdir(workdir)
        ci.set_data(sess, data)
        ci.fetch('birnn', save=True)
        ci.fetch('birnn_out', save=True)
        ci.reset()
        sess.close()

def test_cntk_cudnn():
    try:
        import tensorflow
        has_tensorflow = True
    except:
        has_tensorflow = False

    if has_tensorflow:
        tf_baseline_lstm()
    else:
        cntk_baseline_lstm()

    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance

    input_var = C.sequence.input_variable(shape=(in_dim))
    data = {input_var:data_cntk}
    ci.set_data(data)
    ci.set_workdir(workdir)

    W = C.parameter((-1,dim,), init=C.glorot_uniform())
    cudnn_fwbw = C.optimized_rnnstack(input_var, W, dim, 1, bidirectional=True, recurrent_op='lstm')
    ci.watch(cudnn_fwbw, 'cntk_birnn_cudnn', var_type=cstk.RnnAttr,
          attr=cstk.RnnAttr(bidirectional=True, op_type='lstm', input_dim=in_dim, hidden_dim=dim, forget_bias=0))
    ci.watch(cudnn_fwbw, 'cntk_birnn_cudnn_out')

    ci.assign('cntk_birnn_cudnn', load=True, load_name='birnn')
    assert ci.compare('cntk_birnn_cudnn_out', compare_name='birnn_out', rtol=1e-4, atol=1e-6)

    ci.fetch('cntk_birnn_cudnn', save=True)
    ci.assign('cntk_birnn_cudnn', load=True)
    assert ci.compare('cntk_birnn_cudnn_out', compare_name='birnn_out', rtol=1e-4, atol=1e-6)

    # test assign with value
    num_gates=4
    ci.assign('cntk_birnn_cudnn', value=cstk.RnnArgs(fw_W=np.random.random((in_dim,num_gates*dim)).astype(np.float32),
                                                     fw_H=np.random.random((dim,num_gates*dim)).astype(np.float32),
                                                     fw_b=np.random.random((num_gates*dim,)).astype(np.float32),
                                                     bw_W=np.random.random((in_dim,num_gates*dim)).astype(np.float32),
                                                     bw_H=np.random.random((dim,num_gates*dim)).astype(np.float32),
                                                     bw_b=np.random.random((num_gates*dim,)).astype(np.float32)))

    ci.reset()
