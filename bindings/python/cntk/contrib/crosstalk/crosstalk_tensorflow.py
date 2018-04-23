# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import tensorflow as tf
import numpy as np
from cntk.contrib import crosstalk as cstk

VariableType = 'Variable'
TrainableType = 'Trainable'
DictTrainableType = 'DictTrainable'

def find_trainable(name, scope=None):
    '''
    Find a single trainable variable in a function by name when the function has multiple parameters.
    
    Args:
        func: The function to search
        name (`str`): The name of the parameter
        scope (`str`): The scope of the search

    Returns:
        The trainable variable that is found
    '''
    found = [tp for tp in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) if name in tp.name]
    if len(found)==0:
        raise Exception('not found')
    elif len(found) > 1:
        raise Exception('more than 1 found')
    return found[0]

def _trainable_setter(sess):
    def _set(p, raw_value, attr=None):
        if p.get_shape() != raw_value.shape:
            v = raw_value.reshape(p.get_shape())
        else:
            v = raw_value
        tf.assign(p, v).eval(session=sess)
    return _set
    
def _trainable_getter(sess):
    def _get(p, attr=None):
        return p.eval(sess)
    return _get
    
def _dict_trainable_setter(sess):
    def _set(td, raw_value, attr=None):
        if len(td) != len(raw_value):
            raise Exception('mismatch len')
        if td.keys() != raw_value.keys():
            raise Exception('mismatch keys')
        for k in td.keys():
            _trainable_setter(sess)(td[k], raw_value[k])
    return _set

def _dict_trainable_getter(sess):
    def _get(td, attr=None):
        return {k : _trainable_getter(sess)(td[k]) for k in td.keys()}
    return _get

def _variable_getter(sess, data):
    def _get(p, attr=None):
        return p.eval(data, session=sess)
    return _get

def _conv2d_getter(sess):
    def _get(pd, attr):
        W = _trainable_getter(sess)(pd.W)
        #handling input with sequence axis:
        W_rank = len(W.shape)
        #the transpose from tf [H, W, C] to cntk's [C, H, W] happens at the tailing axes excluding the leading dynamic
        #axes (batch and sequence axes) in the data format:
        axis_perm = (list(range(W_rank - 3)) if W_rank > 3 else []) + [i + W_rank - 3 for i in [2,0,1]]
        if pd.b:
            b = _trainable_getter(sess)(pd.b)
        else:
            b = None
        return cstk.Conv2DArgs(W=W.transpose(axis_perm), b=b.reshape(attr.num_filters,))
    return _get

def _conv2d_setter(sess):
    def _set(pd, raw_value, attr):
        _trainable_setter(sess)(pd.W, raw_value.W.transpose(1,2,0))
        if pd.b:
            _trainable_setter(sess)(pd.b, raw_value.b)
    return _set
    
def _adjust_forget_bias(all_bias, hidden_dim, forget_bias):
    i,m,f,o = np.split(all_bias, 4)
    f += forget_bias
    return np.concatenate((i,m,f,o))
    
def _rnn_trainable_in_scope(scope):
    if tf.VERSION.startswith('0.12'):
        fw_M=find_trainable('Matrix', scope=scope+'/FW')
        fw_b=find_trainable('Bias',   scope=scope+'/FW')
        bw_M=find_trainable('Matrix', scope=scope+'/BW')
        bw_b=find_trainable('Bias',   scope=scope+'/BW')
    elif tf.VERSION.startswith('1'):
        if tf.VERSION.startswith('1.1'):
            fw_M=find_trainable('weights', scope=scope+'/fw')
            fw_b=find_trainable('biases',   scope=scope+'/fw')
            bw_M=find_trainable('weights', scope=scope+'/bw')
            bw_b=find_trainable('biases',   scope=scope+'/bw')
        else: # the following changes started with version '1.2' until as of version 1.7 for now
            fw_M = find_trainable('kernel', scope=scope + '/fw')
            fw_b = find_trainable('bias', scope=scope + '/fw')
            bw_M = find_trainable('kernel', scope=scope + '/bw')
            bw_b = find_trainable('bias', scope=scope + '/bw')
    else:
        raise Exception('only supports 0.12.* and 1.*')

    return fw_M, fw_b, bw_M, bw_b

def _rnn_getter(sess):
    def _get(scope, attr):
        if not attr.bidirectional:
            raise NotImplementedError()
        fw_M, fw_b, bw_M, bw_b = _rnn_trainable_in_scope(scope)
        fw_W, fw_H = np.split(_trainable_getter(sess)(fw_M), [attr.input_dim])
        fw_b = _adjust_forget_bias(_trainable_getter(sess)(fw_b), attr.hidden_dim, attr.forget_bias)
        bw_W, bw_H = np.split(_trainable_getter(sess)(bw_M), [attr.input_dim])
        bw_b  = _adjust_forget_bias(_trainable_getter(sess)(bw_b), attr.hidden_dim, attr.forget_bias)
        return cstk.RnnArgs(fw_W=fw_W, fw_H=fw_H, fw_b=fw_b, bw_W=bw_W, bw_H=bw_H, bw_b=bw_b)
    return _get

def _rnn_setter(sess):
    def _set(scope, raw_value, attr):
        fw_M, fw_b, bw_M, bw_b = _rnn_trainable_in_scope(scope)
        if not attr.bidirectional:
            raise NotImplementedError()
        _trainable_setter(sess)(fw_M, np.concatenate((raw_value.fw_W, raw_value.fw_H)))
        _trainable_setter(sess)(fw_b, _adjust_forget_bias(raw_value.fw_b, attr.hidden_dim, -attr.forget_bias))
        _trainable_setter(sess)(bw_M, np.concatenate((raw_value.bw_W, raw_value.bw_H)))
        _trainable_setter(sess)(bw_b, _adjust_forget_bias(raw_value.bw_b, attr.hidden_dim, -attr.forget_bias))
    return _set

def _embed_getter(sess):
    def _get(p, attr):
        map = {}
        value = _trainable_getter(sess)(p)
        for i in range(attr.input_dim):
            map[attr.dict[i]] = value[i,:]
        return map
    return _get
    
def _embed_setter(sess):
    def _set(p, raw_value, attr):
        out = [None]*attr.input_dim
        for w in raw_value.keys():
            out[attr.dict.index(w)] = raw_value[w] 
        _trainable_setter(sess)(p, np.asarray(out))
    return _set

class TensorFlowCrosstalk(cstk.Crosstalk):
    '''
    TensorFlow implementation for crosstalk
    '''
    def __init__(self):
        super(TensorFlowCrosstalk, self).__init__()

    def set_data(self, sess, data):
        '''
        Set session and mapped data for setter/getters
        
        Args:
            sess : The tensorflow session
            data : The input data feed dict for eval
        '''
        super(TensorFlowCrosstalk, self).register_funcs(TrainableType, setter=_trainable_setter(sess), getter=_trainable_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(DictTrainableType, setter=_dict_trainable_setter(sess), getter=_dict_trainable_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(VariableType, getter=_variable_getter(sess, data))
        super(TensorFlowCrosstalk, self).register_funcs(cstk.Conv2DAttr, setter=_conv2d_setter(sess), getter=_conv2d_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(cstk.RnnAttr, setter=_rnn_setter(sess), getter=_rnn_getter(sess))
        super(TensorFlowCrosstalk, self).register_funcs(cstk.EmbedAttr, setter=_embed_setter(sess), getter=_embed_getter(sess))
        
    def is_trainable(self, name):
        '''
        Check if variable with name is a trainable
        
        Args:
            name (`str`): Variable name to check
        '''
        var_type = self.vars[name].type
        return var_type != VariableType
        
    def load_all_trainables(self):
        '''
        Load all trainables from files in working directory
        '''
        self.load([n for n in self.vars.keys() if self.is_trainable(n)])

    def save_all_trainables(self):
        '''
        Save all trainables to files in working directory
        '''
        self.save([n for n in self.vars.keys() if self.is_trainable(n)])

instance = TensorFlowCrosstalk()