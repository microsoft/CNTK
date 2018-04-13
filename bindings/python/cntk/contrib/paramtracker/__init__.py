# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Parameter tracker (Contrib) for taking snapshots in one framework and restore in another so as to debug/convert among
frameworks.
'''

import os
import numpy as np
import shutil
import collections
from contextlib import contextmanager


class KeyNumpyStore(object):
    """
    Key-value to_store on-disk for numpy arrays.
    """

    @staticmethod
    def clear(store_dir):
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir)

    def __init__(self, store_dir):
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)
        self.keystore_dir = store_dir

    def __getitem__(self, key):
        keypath = os.path.join(self.keystore_dir, key)
        keypath += ".npz"
        with np.load(keypath) as obj:
            keys = obj.keys()
            if len(keys) > 1:
                return {k:obj[k] for k in keys}
            else:
                return obj['arr_0']

    def __setitem__(self, key, arr):
        keypath = os.path.join(self.keystore_dir, key)
        if os.path.exists(keypath):
            shutil.rmtree(keypath)
        if isinstance(arr, dict):
            np.savez(keypath, **arr)
        else:
            np.savez(keypath, arr)
        #TODO: a code path to compress: np.savez_compressed(keypath, arr)

    def reset(self):
        KeyNumpyStore.clear(self.keystore_dir)


SavedParams = collections.namedtuple('SavedParams', 'value params')
SavedParams.__doc__ = '''\
Loaded parameter record.

value - the saved value which can be a single value, a dict of values, and lists depending on how the target framework consume the values.
params - a list of the saved params in the native framework for debug or bookkeeping, e.g. a list of tf.variable in tensorflow, and a list of Parameters in CNTK.
    '''

class ParameterTracker(object):
    instances = {}

    @contextmanager
    def name_scope(self, scope):
        '''
        Nested name scope for the parameter tracker. A systematic manner to organize the name space for the keys,
        especially when networks are created deep in a few nested function calls.
        Args:
            scope: the name of the nested level of scope.

        Returns: None

        '''
        self._enter_key_space(scope)
        yield
        self._exit_key_space()

    @staticmethod
    def get_instance(key):
        if key not in ParameterTracker.instances:
            ParameterTracker.instances[key] = ParameterTracker()
        return ParameterTracker.instances[key]

    def __init__(self, to_path=None, from_path=None):
        '''
        A tracker to save and load parameters for various frameworks, e.g. CNTK, Tensorflow, Caffe2 and MxNet.

        Args:
            to_path:
            from_path:
        '''
        self.entries = {}
        self._init_stores(to_path, from_path)
        self.name_space = []

    def _enter_key_space(self, name):
        assert(name is not None)
        self.name_space.append(name)

    def _exit_key_space(self):
        if len(self.name_space) > 0:
            self.name_space.pop()

    def _name_key(self, key):
        return self.name_space[-1] + "." + key if len(self.name_space) > 0 else key

    def _init_stores(self, to_store_path, from_store_path):
        self.to_store = KeyNumpyStore(to_store_path) if to_store_path is not None else None
        if from_store_path is None or to_store_path == from_store_path:
            #if no from_store is specified or its path is the same as the to_store, use the to_store
            self.from_store = self.to_store
        else:
            self.from_store = KeyNumpyStore(from_store_path)

    def reset(self, to_store_path=None, from_store_path=None):
        self._init_stores(to_store_path, from_store_path)
        self.entries = {}
        self.name_space = []
        return self

    def set_workingpath(self, to_store_path=None, from_store_path=None):
        self._init_stores(to_store_path, from_store_path)
        return self

    def set_frompath(self, from_store_path):
        self.from_store = KeyNumpyStore(from_store_path)
        return self

    def set_topath(self, to_store_path):
        self.to_store = KeyNumpyStore(to_store_path)
        return self

    def share_values_to(self, other_parameter_tracker):
        '''
        Set the other parameter tracker to share the same working director as this tracker so to cross talk. This is
        for in-process sharing. For cross-processes sharing, please set the source parameter tracker's to-working-path by
        `set_topath` to the target parameter tracker's from-working-path by `set_frompath`.
        Args:
            other_parameter_tracker: the other parameter tracker to be tied. The two parameter trackers will share the same working.
            directory so to crosstalk. 

        Returns:
            The self object.
        '''
        if isinstance(other_parameter_tracker, str):
            other_parameter_tracker = ParameterTracker.get_instance(other_parameter_tracker)
        if not isinstance(other_parameter_tracker, ParameterTracker):
            raise TypeError('The other parameter tracker must be the global name of tracker or a ParameterTracker instance.')
        other_parameter_tracker.set_frompath(self.to_store.keystore_dir)
        return self

    def track_parameter(self, key, param, get_value_func=None, set_value_func=None):
        '''
        Track a parameter or a collection of parameters under an identified key. To ensure correct tracking,
        different (collection) of parameters must be tracked under different keys.
        Args:
            key (str): The unique key of the parameter being tracked.  
            param: The parameter to be tracking.
            get_value_func: A function (param, context[option]) -> value which provides the value to be saved with optional
               context.
            set_value_func: A function (param, parameter_value, context[optional]) -> [list_of_set_parameters] which perform necessary
               transformatoin of the parameter value and set it to the appropriate parameter with optional context.
        Returns:
            A list of parameters which have been set.
        '''
        self.entries[self._name_key(key)] = (param, get_value_func, set_value_func)
        return self

    def load_parameters(self, context=None, key_criteria_func = None):
        '''
        Load the parameters that are saved in the from_working_path which is set by `set_frompath`.
        Args:
            context (optional): The context of framework which is needed to load the parameters. This context will be consumed
               by get_param_value function of the corresponding framework. For example, in Tensorflow it is usually the session;
               if necessary, it may contain additional context as needed by the specific set of get_param_value functions.
            key_criteria_func: A criteria to filter the keys of the parameters to be loaded.

        Returns:
            A list of parameters which have been set. If the parameters have been set more than once, there will be duplicated 
            items in the list for debugging purpose. 
        '''
        if self.from_store is None:
            raise(ValueError('Parameter tracker working path is not set. Please use set_workingpath to set the value.'))
        set_params = []
        for key, entry in self.entries.items():
            if key_criteria_func is None or key_criteria_func(key):
                param, _, set_value_func = entry
                if set_value_func is not None:
                    param_value = self.from_store[key]
                    if context:
                        params = set_value_func(param, param_value, context)
                        set_params.extend(params)
                    else:
                        params = set_value_func(param, param_value)
                        set_params.extend(params)
        return set_params

    def save_parameters(self, context = None, key_criteria_func = None):
        '''
        Save the parameters to the to_working_path which is set by `set_topath`.

        Args:
            context (optional): The context of framework which is needed to load the parameters. This context will be consumed
               by get_param_value function of the corresponding framework. For example, in Tensorflow it is usually the session;
               if necessary, it may contain additional context as needed by the specific set of get_param_value functions.
            key_criteria_func: A criteria to filter the keys of the parameters to be saved.

        Returns:
            A list of parameters which have been saved for loading. If the parameters have been saved more than once,
               there will be duplicated items in the list for debugging purpose.
        '''
        if self.to_store is None:
            raise (
            ValueError('Parameter tracker working path is not set. Please use set_workingpath to set the value.'))
        saved_params = []
        for key, entry in self.entries.items():
            if key_criteria_func is None or key_criteria_func(key):
                param, get_value_func, _ = entry
                if get_value_func is not None:
                    value, params = get_value_func(param, context) if context\
                             else get_value_func(param)
                    self.to_store[key] = value
                    saved_params.extend(params)
        return saved_params


def get_tf_vars(var_scope, param_names, name_extra_func=lambda name: name[name.rfind('/') + 1:-2]):
    import tensorflow as tf
    param_names = set(param_names) if isinstance(param_names, list) else {param_names}
    tf_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_scope)
    params = { name_extra_func(p.name): p for p in tf_parameters}
    params = {name: p for (name, p) in params.items() if name in param_names}
    return params

def get_tf_param_value(p, sess):
    return SavedParams(sess.run(p), [p])


def get_cntk_param_value(p):
    return SavedParams((p * 1.0).eval(), [p])

def set_cntk_param_value(p, value):
    p.value = value
    return [p]


def set_tf_param_value(p, value, sess):
    assign_op = p.assign(value)
    sess.run(assign_op)
    return [p]


def get_cntk_embedding(key, embedder):
    return get_cntk_param_value(embedder.parameters[0])


def set_cntk_embedding(embedder, value):
    return set_cntk_param_value(embedder.parameters[0], value)


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


def set_cntk_conv2d_params(conv, value):
    w_value, b_value = value['W'], value['b']
    params = {p.name: p for p in conv.parameters}
    return set_cntk_param_value(params['W'], w_value) + \
           set_cntk_param_value(params['b'], b_value)


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


def set_cntk_lstm_param(rnn_func, value):
    '''

    Args:
        rnn_func: a CNTK lstm recurrence function
        value:  a tuple of  weights and bias: (tf_weights, tf_bias) in numpy arrays

    Returns:

    '''
    b, W, H = rnn_func.parameters
    from_W, from_H, from_b = value['W'], value['H'], value['b']
    b.value = from_b
    W.value = from_W
    H.value = from_H
    return [W, H, b]



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


def set_cntk_gru_param(rnn_func, value):
    '''

    Args:
        rnn_func: a CNTK lstm recurrence function
        value:  a dictionary of GRU parameters according to paper:  https://arxiv.org/abs/1701.05923 and the original one:
          https://arxiv.org/abs/1406.1078. Echoing the https://arxiv.org/abs/1701.05923 notation, the string keys
           are 'W_z', 'W_r', 'W_h', 'b_z', 'b_r', 'b_h', 'U_z', 'U_r', 'U_h'.

    Returns:
        A list of set parameters.
    '''
    params = {p.name: p for p in rnn_func.parameters}
    params['W'].value = np.hstack([value['W_z'], value['W_r'], value['W_h']])
    params['b'].value = np.hstack([value['b_z'], value['b_r'], value['b_h']])
    params['H'].value = np.hstack([value['U_z'], value['U_r']])
    params['H1'].value = value['U_h']
    return list(params.values())
