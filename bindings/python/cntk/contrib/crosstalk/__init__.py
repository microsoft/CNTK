# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import pickle
import numpy as np
from collections import namedtuple

VarInfo = namedtuple('VarInfo', 'var type attr')
FuncInfo = namedtuple('FuncInfo', 'setter getter')

# attributes for high level blocks
Conv2DAttr = namedtuple('Conv2DAttr', 'filter_shape num_filters')
Conv2DArgs = namedtuple('Conv2DArgs', 'W b')
#Conv2D filter shape (num_filters, filter_w, filter_h)
#Conv2D bias shape (num_filters)

RnnAttr = namedtuple('RnnAttr', 'bidirectional op_type input_dim hidden_dim forget_bias')
RnnArgs = namedtuple('RnnArgs', 'fw_W fw_H fw_b bw_W bw_H bw_b')

# embed is a special trainable parameter that it has a dictionary to match
# for word w, the embedding is p[dict.index(w), :]
# crosstalk saves embedding as a dict of word -> weight
EmbedAttr = namedtuple('EmbedAttr', 'dict input_dim')

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
    Base class of Crosstalk. Crosstalk is a utility to manage variables for debugging/conversion across scripts in different toolkits.
    
    With crosstalk, user can define named watch points to variables or parameters, and setting up a work dir.
    Then crosstalk can save/load variables to corresponding files from python debugger, and compare values using numpy.
    Please refer to crosstalk unittests for examples of how to exchange/compare values between different toolkits.
    '''
    def __init__(self):
        self.funcs = {}
        self.reset()

    def set_workdir(self, dir):
        '''
        Set up a working directory for save/load numpy values(.npy) or python data (.pkl)
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
        Add variables to watch with a unique name
        File in working directory would be named <pass>_<name>.(npy|pkl)
        '''
        if name in self.vars.keys():
            raise Exception('var with name {} already exists')
        self.vars[name] = VarInfo(var, var_type if var_type else type(var), attr)

    def register_funcs(self, var_type, setter=None, getter=None):
        '''
        Register setter/getter functions for var_type
        '''
        self.funcs[var_type] = FuncInfo(setter, getter)

    def _get_filename(self, name):
        return os.path.join(self.work_dir, '{}_{}'.format(self.passes, name))

    def load_raw_value(self, name):
        '''
        Load raw value from npy|pkl file in work_dir
        '''
        if os.path.exists(self._get_filename(name)+'.npy'):
            return np.load(self._get_filename(name)+'.npy')
        elif os.path.exists(self._get_filename(name)+'.pkl'):
            with open(self._get_filename(name)+'.pkl', 'rb') as pkl:
                return pickle.load(pkl)

    def assign(self, name, value=None, load=False, load_name=None):
        '''
        Set value to var, with option to load from work_dir
        '''
        if load and value:
            raise Exception('set_var can only have one source')
        var, var_type, attr = self.vars[name]
        if load:
            value = self.load_raw_value(load_name if load_name else name)
        else:
            old_value = self.eval_var(name)
            if type(old_value) == np.ndarray:
                if old_value.shape != value.shape:
                    raise Exception('shape mismatch, required {}, actual {}'.format(old_value.shape, value.shape))
            elif type(old_value) == list:
                for (v0, v1) in zip(old_value, value):
                    if v0.shape != v1.shape:
                        raise Exception('shape mismatch, required {}, actual {}'.format(v0.shape, v1.shape))
            else:
                raise Exception('set_var can only work on ndarray')
        self.funcs[var_type].setter(var, value, attr)

    def fetch(self, name, save=False):
        '''
        Evaluate var with name and optionally save to work_dir
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
        Compare var with name to value in file in work_dir. Specify compare_name if the file to compare is with a different name of var.
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
        Load vars in list of names
        '''
        [self.assign(n, load=True) for n in names if n in self.vars.keys()]
            
    def save(self, names):
        '''
        Save vars in list
        '''
        [self.fetch(n, save=True) for n in names if n in self.vars.keys()]

    def save_all(self):
        '''
        Save all vars
        '''
        self.save(self.vars.keys())
        
    def reset(self):
        '''
        Reset all vars and passes, setter/getter are kept
        '''
        self.vars = {}
        self.passes = 0