# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import numpy as np
from cntk.contrib import crosstalk as cstk

DictParameterType = 'DictParameter'

def find_func_param(func, name=None, shape=None, allow_not_found=False):
    '''
    Find a single parameter in a function by name or by shape when the function has multiple parameters.
    If the function only has one parameter it's directly returned.
    
    Args:
        func (:class:`~cntk.ops.functions.Function`): The function to search parameter for
        name (string) : The name of the parameter
        shape (tuple): The shape of the parameter
        allow_not_found (bool): Set to True to avoid raise exception when not found
    
    Returns:
        The :class:`~cntk.variables.Parameter` that is found
    '''
    if len(func.parameters) == 1:
        return func.parameters[0]
    found = [p for p in func.parameters if (shape and p.shape == shape) or name == p.name]
    if not found:
        if allow_not_found:
            return None
        else:
            raise Exception('param ({} {}) not found'.format(name, shape))
    if len(found) > 1:
        raise Exception('more than one found')
    return found[0]
    
def _parameter_setter(p, raw_value, attr=None):
    if p.shape != raw_value.shape:
        raise Exception('different shape, expected {} actual {}'.format(p.shape, raw_value.shape))
    p.value = raw_value
    
def _parameter_getter(p, attr=None):
    return p.value
    
def _dict_parameter_setter(pd, raw_value, attr=None):
    if len(pd) != len(raw_value):
        raise Exception('mismatch len')
    if pd.keys() != raw_value.keys():
        raise Exception('mismatch keys')
    for k in pd.keys():
        _parameter_setter(pd[k], raw_value[k])

def _dict_parameter_getter(pd, attr=None):
    return {k:pd[k].value for k in pd.keys()}

def _function_getter(data):
    def _get(f, attr=None):
        return f.eval(data)
    return _get

def _variable_getter(data):
    def _get(f, attr=None):
        return C.as_composite(f.owner).eval(data)[f]
    return _get

def _conv2d_getter(f, attr):
    W = _parameter_getter(find_func_param(f, shape=(attr.num_filters, 1,) + attr.filter_shape))
    bias_param = find_func_param(f, shape=(attr.num_filters, 1, 1,), allow_not_found=True)
    if bias_param:
        b = _parameter_getter(bias_param)
    else:
        b = None
    return cstk.Conv2DArgs(W=W[:,0,:,:], b=None if b is None else b.reshape(-1))

def _conv2d_setter(f, raw_value, attr):
    W = find_func_param(f, shape=(attr.num_filters, 1,) + attr.filter_shape)
    _parameter_setter(W, raw_value.W.reshape(W.shape))
    if raw_value.b is not None:
        b = find_func_param(f, shape=(attr.num_filters, 1, 1,))
        _parameter_setter(b, raw_value.b.reshape(b.shape))

def _get_rnn_gates(op_type):
    num_gates = 1
    if op_type == 'lstm':
        num_gates = 4
    elif op_type == 'gru':
        # NOTE that cudnn GRU implementation is different from standard one
        # that cell got projection/bias as well before element_times
        # from CUDNN doc watch out for the difference in h't calculation:
        #
        # it = sigmoid(Wixt + Riht-1 + bWi + bRu)
        # rt = sigmoid(Wrxt + Rrht-1 + bWr + bRr)
        # h't = tanh(Whxt + rt.*(Rhht-1 + bRh) + bWh)
        # ht = (1 - it) .* h't + it .* ht-1
        #
        # so to convert cudnn to CPU we need a different GRU
        num_gates = 3
    else:
        raise NotImplementedError()
    return num_gates

# return splitter for cudnn param of shape (_inferred, hidden_dim) along _inferred
def _get_cudnn_rnn_splitter(attr):
    in_dim = attr.input_dim
    h_dim = attr.hidden_dim
    gates = _get_rnn_gates(attr.op_type)
    # for unidirectional, W, H, b1, b2
    # for bidirectional, fw_W, fw_H, bw_W, bw_H, fw_b1, fw_b2, bw_b1, bw_b2
    multiplier = 2 if attr.bidirectional else 1
    splitter = [in_dim*h_dim*gates, h_dim*h_dim*gates] * multiplier + [h_dim*gates, h_dim*gates] * multiplier

    splitter = splitter[0:-1]
    return np.cumsum(splitter)

def _get_birnn_param(f):
    if f.root_function.op_name != 'Splice':
        raise NotImplementedError()
    # assuming forward/backward cell first/second input to Splice
    fw = f.root_function.inputs[0].owner
    bw = f.root_function.inputs[1].owner
    return cstk.RnnArgs(fw_W=find_func_param(fw, name='W'),
                        fw_H=find_func_param(fw, name='H'),
                        fw_b=find_func_param(fw, name='b'),
                        bw_W=find_func_param(bw, name='W'),
                        bw_H=find_func_param(bw, name='H'),
                        bw_b=find_func_param(bw, name='b'))

'''
cudnn lstm gate is in order of input/forget/mem/output,
while both CNTK and tensorflow is input/mem/forget/output
the saved model uses CNTK/tensorflow order so cudnn weights needs ajust
NOTE this function is identical to its reverse
'''            
def _adjust_lstm_gate_order(W):
    if len(W.shape) == 2:
        i,f,m,o = np.hsplit(W, 4)
        return np.concatenate((i,m,f,o), axis=1)
    elif len(W.shape) == 1:
        i,f,m,o = np.split(W, 4)
        return np.concatenate((i,m,f,o))
    else:
        raise Exception('invalid input')
    
def _rnn_getter(f, attr):
    if not attr.bidirectional:
        raise NotImplementedError()
    use_cudnn = (len(f.parameters) == 1) # CNTK has only 1 big fat parameter when using cudnn
    if use_cudnn:
        gates = _get_rnn_gates(attr.op_type)
        fw_Wt, fw_Ht, bw_Wt, bw_Ht, fw_b1, fw_b2, bw_b1, bw_b2 = np.split(f.parameters[0].value.reshape(-1), _get_cudnn_rnn_splitter(attr))
        return cstk.RnnArgs(fw_W=_adjust_lstm_gate_order(fw_Wt.reshape(gates*attr.hidden_dim, -1).transpose()),
                            fw_H=_adjust_lstm_gate_order(fw_Ht.reshape(gates*attr.hidden_dim, -1).transpose()),
                            fw_b=_adjust_lstm_gate_order(fw_b1 + fw_b2),
                            bw_W=_adjust_lstm_gate_order(bw_Wt.reshape(gates*attr.hidden_dim, -1).transpose()),
                            bw_H=_adjust_lstm_gate_order(bw_Ht.reshape(gates*attr.hidden_dim, -1).transpose()),
                            bw_b=_adjust_lstm_gate_order(bw_b1 + bw_b2))
    else:
        param = _get_birnn_param(f)
        return cstk.RnnArgs(fw_W=_parameter_getter(param.fw_W),
                            fw_H=_parameter_getter(param.fw_H),
                            fw_b=_parameter_getter(param.fw_b),
                            bw_W=_parameter_getter(param.bw_W),
                            bw_H=_parameter_getter(param.bw_H),
                            bw_b=_parameter_getter(param.bw_b))

def _rnn_setter(f, raw_value, attr):
    if not attr.bidirectional:
        raise NotImplementedError()
    use_cudnn = (len(f.parameters) == 1)
    if use_cudnn:
        gates = _get_rnn_gates(attr.op_type)
        _parameter_setter(f.parameters[0], 
                         np.concatenate((_adjust_lstm_gate_order(raw_value.fw_W).transpose().reshape(-1),
                                         _adjust_lstm_gate_order(raw_value.fw_H).transpose().reshape(-1),
                                         _adjust_lstm_gate_order(raw_value.bw_W).transpose().reshape(-1),
                                         _adjust_lstm_gate_order(raw_value.bw_H).transpose().reshape(-1),
                                         _adjust_lstm_gate_order(raw_value.fw_b).reshape(-1),
                                         np.zeros_like(raw_value.fw_b).reshape(-1),
                                         _adjust_lstm_gate_order(raw_value.bw_b).reshape(-1),
                                         np.zeros_like(raw_value.bw_b).reshape(-1)
                                        )).reshape(f.parameters[0].shape))
    else:
        param = _get_birnn_param(f)
        _parameter_setter(param.fw_W, raw_value.fw_W)
        _parameter_setter(param.fw_H, raw_value.fw_H)
        _parameter_setter(param.fw_b, raw_value.fw_b)
        _parameter_setter(param.bw_W, raw_value.bw_W)
        _parameter_setter(param.bw_H, raw_value.bw_H)
        _parameter_setter(param.bw_b, raw_value.bw_b)

def _embed_getter(p, attr):
    map = {}
    value =  _parameter_getter(p)
    for i in range(attr.input_dim):
        map[attr.dict[i]] = value[i,:]
    return map
    
def _embed_setter(p, raw_value, attr):
    out = [None]*attr.input_dim
    for w in raw_value.keys():
        out[attr.dict.index(w)] = raw_value[w] 
    _parameter_setter(p, np.asarray(out))

class CNTKCrosstalk(cstk.Crosstalk):
    '''
    CNTK implementation for crosstalk
    '''
    def __init__(self):
        super(CNTKCrosstalk, self).__init__()
        super(CNTKCrosstalk, self).register_funcs(C.variables.Parameter, setter=_parameter_setter, getter= _parameter_getter)
        super(CNTKCrosstalk, self).register_funcs(DictParameterType, setter=_dict_parameter_setter, getter=_dict_parameter_getter)
        super(CNTKCrosstalk, self).register_funcs(cstk.Conv2DAttr, setter=_conv2d_setter, getter=_conv2d_getter)
        super(CNTKCrosstalk, self).register_funcs(cstk.RnnAttr, setter=_rnn_setter, getter=_rnn_getter)
        super(CNTKCrosstalk, self).register_funcs(cstk.EmbedAttr, setter=_embed_setter, getter=_embed_getter)

    def set_data(self, data):
        '''
        Set mapped data for variable evaluation
        
        Args:
            data: The input data as arguments parameter in :func:`~cntk.ops.functions.Function.eval`
        '''
        super(CNTKCrosstalk, self).register_funcs(C.ops.functions.Function, getter=_function_getter(data))
        super(CNTKCrosstalk, self).register_funcs(C.variables.Variable, getter=_variable_getter(data))

    def is_param(self, name):
        '''
        Check if var with name is a parameter
        
        Args:
            name (`str`): Variable name to check
        '''
        var_type = self.vars[name].type
        return var_type not in [C.ops.functions.Function, C.variables.Variable]

    def load_all_params(self):
        '''
        Load all parameters from files in working directory
        '''
        super(CNTKCrosstalk, self).load([n for n in self.vars.keys() if self.is_param(n)])
        
    def save_all_params(self):
        '''
        Save all parameters to files in working directory
        '''
        super(CNTKCrosstalk, self).save([n for n in self.vars.keys() if self.is_param(n)])

instance = CNTKCrosstalk()