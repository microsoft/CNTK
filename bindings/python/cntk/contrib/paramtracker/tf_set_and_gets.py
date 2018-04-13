from . import *

#The following are Tensorflow related.
def get_tf_vars(var_scope, param_names, name_extra_func=lambda name: name[name.rfind('/') + 1:-2]):
    import tensorflow as tf
    param_names = set(param_names) if isinstance(param_names, list) else {param_names}
    tf_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_scope)
    params = { name_extra_func(p.name): p for p in tf_parameters}
    params = {name: p for (name, p) in params.items() if name in param_names}
    return params

def get_tf_param_value(p, sess):
    return SavedParams(sess.run(p), [p])


def set_tf_param_value(p, value, sess):
    assign_op = p.assign(value)
    sess.run(assign_op)
    return [p]


def get_tf_conv2d_param_value(tf_contrib_name_scope, sess, param_keywords = ['weights', 'biases']):
    scope, keywords = tf_contrib_name_scope, param_keywords
    params = get_tf_vars(scope, keywords)
    v = sess.run(params)
    #data_format = 'NHWC'
    #tf filter  shape: [filter_height, filter_width, in_channels, out_channels]
    #cntk filter shape: [out_channels, in_channels, filter_height, filter_width]
    return SavedParams({'W': v[keywords[0]].transpose(3,2,0,1),
                        #cntk bias shape is [out_channells, 1, 1] but tf bias shpae is [out_channels,]
                        'b': v[keywords[1]].reshape(v[keywords[1]].shape + (1, 1))},
                       params.values())


def get_tf_contrib_conv2d_param_value(tf_contrib_name_scope, sess):
    return get_tf_conv2d_param_value(tf_contrib_name_scope, sess, param_keywords=['weights', 'biases'])


def get_tf_lstm_param_value(tf_lstm_info, sess, param_keywords=['kernel', 'bias']):
    '''
    Args:
        rnn_func: (rnn_var_scope, rnn_output, rnn_state) where runn_var_scope is a string to identify the scope of the LSTM weight and bias parameters.
        sess: tensorflow session

    Returns: Numpy array fo the rnn parameter values: (rnn_weights, rnn_bias)
    '''
    import tensorflow as tf
    tf_param_name_scope, hidden_dim = tf_lstm_info
    if isinstance(tf_param_name_scope, tuple):
        scope, keywords = tf_param_name_scope
    else:
        scope, keywords = tf_param_name_scope, param_keywords
    params = get_tf_vars(scope, keywords)
    v = sess.run(params)
    W = v[keywords[0]]
    b = v[keywords[1]]

    cntk_W = W[0:- hidden_dim, :]
    cntk_H = W[-hidden_dim:, :]
    return SavedParams({'W': cntk_W, 'H': cntk_H, 'b': b}, params.values())


def get_tf_contrib_gru_param_value(rnn_func_info, sess):
    '''

    Args:
        rnn_func_info: (rnn_var_scope, cell_dim, input_dim) where runn_var_scope is a string to identify the scope of the LSTM weight and bias parameters.
        sess: tensorflow session

    Returns: Numpy array fo the rnn parameter values: (rnn_weights, rnn_bias)

    '''
    rnn_var_scope, cell_dim, input_dim = rnn_func_info
    params = get_tf_vars(rnn_var_scope, ['w_ru', 'b_ru', 'w_c', 'b_c'])
    v = sess.run(params)
    w_ru, b_ru, w_c, b_c = v['w_ru'], v['b_ru'], v['w_c'], v['b_c']

    #decomposing tf's stacking of GRU paramters to the standard notation in paper: https://arxiv.org/abs/1701.05923
    value = {
        'W_z': w_ru[0:input_dim, cell_dim: 2* cell_dim],#TODO: double check accordingt to tf.contrib.gru document, it should be: -w_ru[0:input_dim, cell_dim: 2* cell_dim],
        'W_r': w_ru[0:input_dim, 0:cell_dim],
        'W_h': w_c[0:input_dim,:],
        'b_z': b_ru[cell_dim: 2* cell_dim],#TODO: double check accordingt to tf.contrib.gru document, -b_ru[cell_dim: 2* cell_dim],
        'b_r': b_ru[0:cell_dim],
        'b_h': b_c,
        'U_z': w_ru[input_dim:input_dim+cell_dim, cell_dim: 2* cell_dim],#TODO: double check accordingt to tf.contrib.gru document, -w_ru[input_dim:input_dim+cell_dim, cell_dim: 2* cell_dim],
        'U_r': w_ru[input_dim:input_dim+cell_dim, 0:cell_dim],
        'U_h': w_c[input_dim:input_dim+cell_dim,:]
    }
    return SavedParams(value, params.values())


def get_tf_gru_param_value(rnn_func_info, sess):
    '''

    Args:
        rnn_func_info: (rnn_var_scope, cell_dim, input_dim) where runn_var_scope is a string to identify the scope of the LSTM weight and bias parameters.
        sess: tensorflow session

    Returns: Numpy array fo the rnn parameter values: (rnn_weights, rnn_bias)

    '''
    rnn_var_scope, cell_dim, input_dim = rnn_func_info
    params = get_tf_vars(rnn_var_scope,
                         ['gates/kernel', 'gates/bias',
                          'candidate/kernel', 'candidate/bias'],
                         name_extra_func=lambda name: name[name.rfind('/', 0, name.rfind('/')) + 1:-2])
    #in tf rnn_cell_imply.py:
    #    gate_inputs = math_ops.matmul(
    #    array_ops.concat([inputs, state], 1), self._gate_kernel)
    #    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
    #
    #    candidate = math_ops.matmul(
    #    array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    #    candidate = nn_ops.bias_add(candidate, self._candidate_bias)
    #    value = math_ops.sigmoid(gate_inputs)
    #    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    #    r_state = r * state
    #    c = self._activation(candidate)
    #    new_h = u * state + (1 - u) * c

    v = sess.run(params)
    w_rz = v['gates/kernel']
    b_rz = v['gates/bias']
    w_hu = v['candidate/kernel']
    b_hu = v['candidate/bias']
    value = {
        'W_z': w_rz[0:input_dim, cell_dim: 2* cell_dim],
        'W_r': w_rz[0:input_dim, 0:cell_dim],
        'W_h': w_hu[0:input_dim,:],
        'b_z': b_rz[cell_dim: 2* cell_dim],
        'b_r': b_rz[0:cell_dim],
        'b_h': b_hu,
        'U_z': w_rz[input_dim:input_dim+cell_dim, cell_dim: 2* cell_dim],
        'U_r': w_rz[input_dim:input_dim+cell_dim, 0:cell_dim],
        'U_h': w_hu[input_dim:input_dim+cell_dim,:]
    }

    return SavedParams(value, params.values())
