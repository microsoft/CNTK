# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Parameter tracker (Contrib) for taking snapshots in one framework and restore in another so as to debug/convert among
frameworks.
'''


import os
import py
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
        if isinstance(store_dir, py._path.local.LocalPath):
            store_dir = str(store_dir)
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

    def __init__(self, to_path=None, from_path=None, key_value_store_create_func = lambda path: KeyNumpyStore(path)):
        '''
        A tracker to save and load parameters for various frameworks, e.g. CNTK, Tensorflow, Caffe2 and MxNet.
        
        To track parameters from one framework (say Tensorflow) to another framework (say CNTK), the steps to take:
        
            1. Create get_param_value functions for the source framework, e.g. ::
            
                def get_tf_param_value(p, sess):
                    return SavedParams(sess.run(p), [p])

            2. Create set_param_value functions for the target framework, e.g. ::
            
                def set_cntk_param_value(p, value):
                    p.value = value
                    return [p]
            
            3. Create the source parameter tracker and the target parameter tracker with shared working path:: 
            
                shared_working_path = ...
                source_params = ParameterTracker(to_path=shared_working_path)
                target_params = ParameterTracker(from_path=shared_working_path)
                
            4. Insert tracking codes when the networks are created, e.g. ::

                #In the source framework:
                tf_weights = tf_aa = tf.get_variable('tf_weights', [2,3,5],  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                source_params.track_parameter('weights', tf_weights, get_value_func=get_tf_param_value)
                
                #In the target framework: 
                cntk_weights = C.Parameter((2,3,5), name='cntk_weights', dtype=np.float32)
                target_params.track_parameter('weights', cntk_aa, set_value_func=set_cntk_param_value)

            5. Call `save_parameters` after the parameter values are ready in the source framework, e.g. ::

                 sess.run(init_op)
                 saved_param_list = source_params.save_parameters(sess)
                 
                 #or with CNTK source
                 saved_param_list = source_parameters.save_parameters() #note that the context parameter is optional

            6. Call `load_parameters` after the parameter values are available in the from_path, e.g. ::
                
                 target_params.load_parameters()
                 
            7. Verify with your customized code to check the parity of the models in two frameworks.
                
        Args:
            to_path:
            from_path:
        '''
        self.entries = {}
        self.key_value_store_create_func = key_value_store_create_func
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
        self.to_store = self.key_value_store_create_func (to_store_path) if to_store_path is not None else None
        if from_store_path is None or to_store_path == from_store_path:
            #if no from_store is specified or its path is the same as the to_store, use the to_store
            self.from_store = self.to_store
        else:
            self.from_store = self.key_value_store_create_func (from_store_path)

    def reset(self, to_store_path=None, from_store_path=None):
        self._init_stores(to_store_path, from_store_path)
        self.entries = {}
        self.name_space = []
        return self

    def set_workingpath(self, to_store_path=None, from_store_path=None):
        self._init_stores(to_store_path, from_store_path)
        return self

    def set_frompath(self, from_store_path):
        self.from_store = self.key_value_store_create_func (from_store_path)
        return self

    def set_topath(self, to_store_path):
        self.to_store = self.key_value_store_create_func (to_store_path)
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


def get_cntk_param_value(p):
    return SavedParams((p * 1.0).eval(), [p])


def set_cntk_param_value(p, value):
    p.value = value
    return [p]


def get_cntk_embedding(key, embedder):
    return get_cntk_param_value(embedder.parameters[0])


def set_cntk_embedding(embedder, value):
    return set_cntk_param_value(embedder.parameters[0], value)


def set_cntk_conv2d_params(conv, value):
    w_value, b_value = value['W'], value['b']
    params = {p.name: p for p in conv.parameters}
    return set_cntk_param_value(params['W'], w_value) + \
           set_cntk_param_value(params['b'], b_value)


def set_cntk_lstm_param(rnn_func, value):
    '''

    Args:
        rnn_func: a CNTK lstm recurrence function
        value:  a dictionary of  weights and bias: {W: ..., H: ..., b: ...}

    Returns:
        A list of set parameters.
    '''
    b, W, H = rnn_func.parameters
    from_W, from_H, from_b = value['W'], value['H'], value['b']
    b.value = from_b
    W.value = from_W
    H.value = from_H
    return [W, H, b]


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
