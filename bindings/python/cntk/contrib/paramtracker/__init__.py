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


class ParameterTracker(object):
    instances = {}

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
        self.name_space.append(name)

    def _exit_key_space(self):
        if len(self.name_space) > 0:
            self.name_space.pop()

    def _name_key(self, key):
        return self.name_space[-1] + "." + key if self.name_space else key

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
        Set the other parameter tracker to share the same working director as this tracker so to cross talk.
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
        
        Args:
            context: 
            key_criteria_func: 

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
        if self.to_store is None:
            raise (
            ValueError('Parameter tracker working path is not set. Please use set_workingpath to set the value.'))
        for key, entry in self.entries.items():
            if key_criteria_func is None or key_criteria_func(key):
                param, get_value_func, _ = entry
                if get_value_func is not None:
                    param_value = get_value_func(param, context) if context\
                             else get_value_func(param)
                    self.to_store[key] = param_value
        return self

@contextmanager
def key_scope(param_tracker, scope):
    param_tracker._enter_key_space(scope)
    yield
    param_tracker._exit_key_space()


def get_tf_vars(var_scope, param_names, name_extra_func=lambda name: name[name.rfind('/') + 1:-2]):
    import tensorflow as tf
    param_names = set(param_names) if isinstance(param_names, list) else {param_names}
    tf_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_scope)
    params = { name_extra_func(p.name): p for p in tf_parameters}
    params = {name: p for (name, p) in params.items() if name in param_names}
    return params

def get_tf_param_value(p, sess):
    return sess.run(p)


def get_cntk_param_value(p):
    return (p * 1.0).eval()

def set_cntk_param_value(p, value):
    p.value = value
    return [p]


def set_tf_param_value(p, value, sess):
    assign_op = p.assign(value)
    sess.run(assign_op)
    return [p]

def track_cntk_embedding(tracker, key, embedder):
    tracker.track_parameter(key, embedder.parameters[0], get_value_func=get_cntk_param_value, set_value_func=set_cntk_param_value)

def set_cntk_conv2d_weights_from_tf(conv_weights, value):
    #data_format = 'NHWC'
    #tf filter  shape: [filter_height, filter_width, in_channels, out_channels]
    #cntk filter shape: [out_channels, in_channels, filter_height, filter_width]
    w_value = value.transpose(3,2,0,1)
    return set_cntk_param_value(conv_weights, w_value)

def set_cntk_conv2d_bias_from_tf(p, value):
    return set_cntk_param_value(p, value.reshape(p.shape))

def tf_get_conv2d_param(tf_conv_params, sess):
    w_value, b_value = sess.run(tf_conv_params)
    return {'W': w_value, 'b': b_value}

def set_cntk_conv2d_params_from_tf(conv, value):
    w_value, b_value = value['W'], value['b']
    params = {p.name: p for p in conv.parameters}
    return set_cntk_conv2d_weights_from_tf(params['W'], w_value) + \
           set_cntk_conv2d_bias_from_tf(params['b'], b_value)



def get_tf_lstm_param_value(rnn_func, sess):
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


def set_cntk_lstm_param_from_tf(rnn_func, value):
    '''

    Args:
        rnn_func: a CNTK lstm recurrence function
        value:  a tuple of  weights and bias: (tf_weights, tf_bias) in numpy arrays

    Returns:

    '''
    b, W, H = rnn_func.parameters
    tf_weights, tf_bias = value['W'], value['b']
    b.value = tf_bias
    W.value = tf_weights[0:W.shape[0], :]
    H.value = tf_weights[W.shape[0]:, :]
    return [W, H, b]

def get_tf_contrib_gru_param_value(rnn_func, sess):
    '''

    Args:
        rnn_func: (rnn_var_scope, rnn_output, rnn_state) where runn_var_scope is a string to identify the scope of the LSTM weight and bias parameters.
        sess: tensorflow session

    Returns: Numpy array fo the rnn parameter values: (rnn_weights, rnn_bias)

    '''
    rnn_var_scope, cell_func, cell_dim, input_dim = rnn_func
    params = get_tf_vars(rnn_var_scope, ['w_ru', 'b_ru', 'w_c', 'b_c'])
    v = sess.run(params)
    w_ru, b_ru, w_c, b_c = v['w_ru'], v['b_ru'], v['w_c'], v['b_c']

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
    return value

def set_cntk_gru_param_from_tf_contrib(rnn_func, value):
    '''

    Args:
        rnn_func: a CNTK lstm recurrence function
        value:  a tuple of  weights and bias: (tf_weights, tf_bias) in numpy arrays

    Returns:

    '''
    params = {p.name: p for p in rnn_func.parameters}
    params['W'].value = np.hstack([value['W_z'], value['W_r'], value['W_h']])
    params['b'].value = np.hstack([value['b_z'], value['b_r'], value['b_h']])
    params['H'].value = np.hstack([value['U_z'], value['U_r']])
    params['H1'].value = value['U_h']
    return list(params.values())


def set_cntk_gru_param_from_tf(rnn_func, value):
    '''

    Args:
        rnn_func: a CNTK lstm recurrence function
        value:  a tuple of  weights and bias: (tf_weights, tf_bias) in numpy arrays

    Returns:

    '''
    b, W, H = rnn_func.parameters
    tf_weights, tf_bias = value['W'], value['b']
    b.value = tf_bias
    W.value = tf_weights[0:W.shape[0], :]
    H.value = tf_weights[W.shape[0]:, :]
    return [W, H, b]
