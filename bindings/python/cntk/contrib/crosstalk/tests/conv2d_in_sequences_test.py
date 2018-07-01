import numpy as np
from cntk.contrib import crosstalk as cstk
import tempfile
workdir = tempfile.gettempdir()

batch_size = 20
filter_width = 5
char_emb_dim = 8
num_chars = 16
seq_len = 4
sample_shape = (num_chars, char_emb_dim,)
input_data = np.random.random((batch_size,seq_len)+sample_shape).astype(np.float32)
filter_shape = (filter_width,char_emb_dim,)
num_filters = 100

def cntk_baseline_conv2d():
    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance
    input_var = C.sequence.input_variable(shape=sample_shape)
    input_reshaped = C.reshape(input_var, (1,)+sample_shape)
    conv_out = C.layers.Convolution2D(filter_shape, num_filters, init_bias=C.glorot_uniform())(input_reshaped)
    ci.watch(conv_out, 'conv2d', var_type=cstk.Conv2DAttr,
              attr=cstk.Conv2DAttr(filter_shape=filter_shape, num_filters=num_filters))
    ci.watch(conv_out, 'conv2d_out')

    data = {input_var:input_data}
    ci.set_data(data)
    ci.set_workdir(workdir)
    ci.fetch('conv2d', save=True)
    ci.fetch('conv2d_out', save=True)
    ci.reset()

def tf_baseline_conv2d():
    import tensorflow as tf
    import cntk.contrib.crosstalk.crosstalk_tensorflow as crtf
    ci = crtf.instance

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [batch_size, seq_len, num_chars, char_emb_dim])
    filter_bank = tf.get_variable("char_filter_bank",
                                  shape=[filter_width, char_emb_dim, num_filters],
                                  dtype=tf.float32)
    bias = tf.get_variable("char_filter_biases", shape=[num_filters], dtype=tf.float32)

    x_reshape = tf.reshape(x, [-1] + x.get_shape().as_list()[-2:])
    char_conv = tf.expand_dims(tf.transpose(tf.nn.conv1d(x_reshape, filter_bank, stride=1, padding='VALID') + bias, perm=[0,2,1]), -1)
    char_conv = tf.reshape(char_conv, [-1, seq_len] + char_conv.shape.as_list()[-3:])

    ci.watch(cstk.Conv2DArgs(W=crtf.find_trainable('char_filter_bank'), b=crtf.find_trainable('char_filter_biases')), 'conv2d', var_type=cstk.Conv2DAttr,
               attr=cstk.Conv2DAttr(filter_shape=(filter_width, char_emb_dim,), num_filters=num_filters))
    ci.watch(char_conv, 'conv2d_out', var_type=crtf.VariableType) # note the output is transposed to NCHW

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = {x:input_data}
        ci.set_workdir(workdir)
        ci.set_data(sess, data)
        ci.fetch('conv2d_out', save=True)
        ci.fetch('conv2d', save=True)
        ci.assign('conv2d', load=True)
        assert ci.compare('conv2d_out')
        ci.reset()
        sess.close()

def test_cntk_conv2d():
    try:
        import tensorflow
        has_tensorflow = True
    except:
        has_tensorflow = False

    if has_tensorflow:
        tf_baseline_conv2d()
    else:
        cntk_baseline_conv2d()

    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance

    input_var = C.sequence.input_variable(shape=sample_shape)
    input_reshaped = C.reshape(input_var, (1,)+sample_shape)
    conv_out = C.layers.Convolution2D(filter_shape, num_filters, activation=None)(input_reshaped)

    ci.watch(conv_out, 'conv2d', var_type=cstk.Conv2DAttr,
              attr=cstk.Conv2DAttr(filter_shape=filter_shape, num_filters=num_filters))
    ci.watch(conv_out, 'conv2d_out')

    data = {input_var:input_data}
    ci.set_data(data)
    ci.set_workdir(workdir)
    conv_out_values = conv_out.eval(data)

    # load parameters from crosstalk and verify results are the same
    ci.assign('conv2d', load=True)
    assert ci.compare('conv2d_out', rtol=1e-4, atol=1e-6)

    # test assign with value
    ci.assign('conv2d', value=cstk.Conv2DArgs(W=np.random.random((num_filters,) + filter_shape).astype(np.float32),
                                              b=np.random.random((num_filters,)).astype(np.float32)))

    ci.reset()
