# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Crosstalk (Contrib) for debugging/conversion among toolkits

It is the utility to manage variables for debugging/conversion across scripts in different toolkits.
With crosstalk, user can define named watch points to variables or parameters, and setting up a work dir. 
Then crosstalk can save/load variables to corresponding files from python debugger, and compare values using numpy. 
'''

import os
import pickle
import numpy as np
from collections import namedtuple

class _VarInfo(namedtuple('_VarInfo', 'var type attr')):
    ''' Variable information

    var
        The variable object

    type
        The type of the variable object

    attr
        The attributes for variable setter/getter
    '''

class _FuncInfo(namedtuple('_FuncInfo', 'setter getter')):
    ''' Setter/getter functions for a given variable type
    
    setter
        The setter function

    getter
        The getter function
    '''

# attributes for high level blocks
class Conv2DAttr(namedtuple('Conv2DAttr', 'filter_shape num_filters')):
    ''' Attribute for Conv2D variable
    
    filter_shape
        The filter shape

    num_filters
        Number of filters
    '''

class Conv2DArgs(namedtuple('Conv2DArgs', 'W b')):
    ''' Args inside Conv2D variable. Conv2D output is in NCHW format

    W
        The numpy ndarray of filter parameter in shape (num_filters, filter_w, filter_h,))

    b
        The numpy ndarray of bias parameter in shape (num_filters,)
    '''

class RnnAttr(namedtuple('RnnAttr', 'bidirectional op_type input_dim hidden_dim forget_bias')):
    ''' Attribute for RNN variable
    
    bidirectional
        True for bidirectional RNN, False for unidirection. Currently only support bidirectional=True

    op_type
        RNN cell type, currently only support 'lstm'
        
    input_dim
        Input dimension

    hidden_dim
        Hidden dimension in RNN cell

    forget_bias
        forget gate bias in LSTM
    '''

class RnnArgs(namedtuple('RnnArgs', 'fw_W fw_H fw_b bw_W bw_H bw_b')):
    ''' Args inside RNN variable.

    fw_W
        The numpy ndarray of input projection parameter for RNN forward in shape (input_dim, num_gates * hidden_dim)

    fw_H
        The numpy ndarray of hidden projection parameter for RNN forward in shape (hidden_dim, num_gates * hidden_dim)

    fw_b
        The numpy ndarray of bias parameter for RNN forward in shape (num_gates * hidden_dim,)
        
    bw_W
        The numpy ndarray of input projection parameter for RNN backward in shape (input_dim, num_gates * hidden_dim)

    bw_H
        The numpy ndarray of hidden projection parameter for RNN backward in shape (hidden_dim, num_gates * hidden_dim)

    bw_b
        The numpy ndarray of bias parameter for RNN backward in shape (num_gates * hidden_dim,)
    '''

class EmbedAttr(namedtuple('EmbedAttr', 'dict input_dim')):
    '''
    Attribute for embedding variable
    
    dict
        The list of vocabulary of the embedding

    input_dim
        The input dimension of the embedding
    '''

def _compare_list_to_ndarray(list_value, ndarray_value, rtol, atol, equal_nan):
    if ndarray_value.shape[0] != len(list_value):
        raise Exception('mismatch batch size')
    if ndarray_value.shape[2:] != list_value[0].shape[1:]:
        raise Exception('mismatch sample shape') # it's OK for sequence axis to be different
    match = True
    for batch, list_item in enumerate(list_value):
        ndarray_item = ndarray_value[batch][:list_item.shape[0]] # note gt might be padded
        if not np.isclose(ndarray_item, list_item, rtol, atol, equal_nan).all():
            diff = (ndarray_item - list_item) ** 2
            diff_sum = diff.sum(axis=tuple(range(1,len(diff.shape))))
            print('mismatch found at item {} row {}'.format(batch, np.argmax(diff_sum)))
            match = False
            break
    return match

class Crosstalk(object):
    '''
    Base class of Crosstalk.
    Please refer to crosstalk unittests for examples of how to exchange/compare values between different toolkits.
    '''
    def __init__(self):
        self.funcs = {}
        self.reset()

    def set_workdir(self, dir):
        '''
        Set up a working directory for save/load numpy values(.npy) or python data (.pkl)
        
        Args:
            dir (`str`): Working directory
        '''
        self.work_dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)

    def next_pass(self):
        '''
        Bump up passes so save won't overwrite existing files
        '''
        self.passes += 1

    def watch(self, var, name, var_type=None, attr=None):
        '''
        Add variables to watch with a unique name.

        Args:
            var: Variable to watch. The type is toolkit specific
            name (`str`): A unique name of the watched variable
            var_type : Variable type for set/get value callback. would be type of var if None is specified
            attr : attributes for the variable that would be used when getting/setting values. Could be one of Conv2DAttr/EmbedAttr/RnnAttr
        '''
        if name in self.vars.keys():
            raise Exception('var with name {} already exists'.format(name))
        self.vars[name] = _VarInfo(var, var_type if var_type else type(var), attr)

    def register_funcs(self, var_type, setter=None, getter=None):
        '''
        Register setter/getter functions for a given variable type
        
        Args:
            var_type: Type of the variable
            setter: Lambda function to set value
            getter: Lambda function to get value
        '''
        self.funcs[var_type] = _FuncInfo(setter, getter)

    def _get_filename(self, name):
        return os.path.join(self.work_dir, '{}_{}'.format(self.passes, name))

    def load_raw_value(self, name):
        '''
        Load raw value from npy|pkl file in working directory
        
        Args:
            name (`str`): Name of the file to load
            
        Returns:
            loaded data in numpy ndarray or dict of numpy ndarray
        '''
        if os.path.exists(self._get_filename(name)+'.npy'):
            return np.load(self._get_filename(name)+'.npy')
        elif os.path.exists(self._get_filename(name)+'.pkl'):
            with open(self._get_filename(name)+'.pkl', 'rb') as pkl:
                return pickle.load(pkl)
        else:
            raise Exception('file not found for name {}'.format(name))

    def assign(self, name, value=None, load=False, load_name=None):
        '''
        Set value to var, with option to load from working directory
        
        Args:
            name (`str`): Name of the variable to assign
            value : Numpy ndarray of dict of numpy ndarray data to assign to the variable
            load (`bool`): True to Load the data from working directory with the matching name, instead of using value. value has to be None when load=True
            load_name (`str`): None to load data with the same name, otherwise load with overrided load_name
        '''
        if load and value:
            raise Exception('assign can only have one source')
        var, var_type, attr = self.vars[name]
        if load:
            value = self.load_raw_value(load_name if load_name else name)
        else:
            old_value = self.fetch(name)
            if type(old_value) != type(value):
                raise Exception('cannot assign with different types: original {}, new {}'.format(type(old_value), type(value)))
        self.funcs[var_type].setter(var, value, attr)

    def fetch(self, name, save=False):
        '''
        Fetch/evaluate var with name and optionally save to working directory
        
        Args:
            name (`str`): Name of the variable to fetch
            save (`bool`): Save the data to working directory
        '''
        var, var_type, attr = self.vars[name]
        raw_value = self.funcs[var_type].getter(var, attr)
        if save:
            if type(raw_value) == np.ndarray:
                np.save(self._get_filename(name)+'.npy', raw_value)
            else:
                with open(self._get_filename(name)+'.pkl', 'wb') as pkl:
                    pickle.dump(raw_value, pkl)
        return raw_value

    def compare(self, name, compare_name=None, rtol=1e-05, atol=1e-08, equal_nan=False):
        '''
        Compare var with name to value in file in working directory
        
        Args:
            name (`str`): Name of the variable to compare
            compare_name (`str`): Compare to file with compare_name if specified
            rtol (`float`): The relative tolerance parameter, as in numpy.isclose()
            atol (`float`): The absolute tolerance parameter, as in numpy.isclose()
            equal_nan (`bool`): Whether to compare NaNs as equal, as in numpy.isclose()
        '''
        var, var_type, attr = self.vars[name]
        raw_value = self.funcs[var_type].getter(var, attr)
        gt_value = self.load_raw_value(compare_name if compare_name else name)
        if type(raw_value) == np.ndarray:
            if type(gt_value) == np.ndarray:
                return np.isclose(gt_value, raw_value, rtol, atol, equal_nan).all()
            elif type(gt_value) == list:
                return _compare_list_to_ndarray(gt_value, raw_value, rtol, atol, equal_nan)
        elif type(raw_value) == list and all([type(x) == np.ndarray for x in raw_value]):
            if type(gt_value) == np.ndarray: # handle batch axis in ndarray as well as in list
                return _compare_list_to_ndarray(raw_value, gt_value, rtol, atol, equal_nan)
            else:
                if type(gt_value) != list or not all([type(x) == np.ndarray for x in gt_value]) or len(gt_value) != len(raw_value):
                    raise Exception('mismatch length or type')
                return all([np.isclose(gt, raw, rtol, atol, equal_nan).all() for (gt,raw) in zip(gt_value, raw_value)])
        elif type(raw_value) == dict:
            if type(gt_value) != dict or not all([type(x) == np.ndarray for x in gt_value.values()]) or len(gt_value) != len(raw_value):
                raise Exception('mismatch length or type')
            if gt_value.keys() != raw_value.keys():
                raise Exception('mismatch dict')
            return all([np.isclose(gt_value[w], raw_value[w], rtol, atol, equal_nan).all() for w in gt_value.keys()])
        else:
            raise Exception('can only compare numpy.ndarray, list of numpy.ndarray or dict of numpy.ndarray')

    def load(self, names):
        '''
        Load variables in list of names
        
        Args:
            names : List of `str` of variable names to load
        '''
        [self.assign(n, load=True) for n in names if n in self.vars.keys()]

    def save(self, names):
        '''
        Save variables in list of names

        Args:
            names : List of `str` of variable names to save
        '''
        [self.fetch(n, save=True) for n in names if n in self.vars.keys()]

    def save_all(self):
        '''
        Save all variables
        '''
        self.save(self.vars.keys())

    def reset(self):
        '''
        Reset all variables and passes, setter/getter functions for variable types are kept
        '''
        self.vars = {}
        self.passes = 0