import os
import pickle
import numpy as np
from collections import namedtuple

VarInfo = namedtuple('VarInfo', 'var type attr')
FuncInfo = namedtuple('FuncInfo', 'setter getter')

# attributes for high level blocks
Conv2DAttr = namedtuple('Conv2DAttr', 'filter_shape num_filters has_bias')
Conv2DArgs = namedtuple('Conv2DArgs', 'W b')

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

class Crosstalk:
    '''
    class to hold variables for debugging/conversion
    '''
    def __init__(self):
        self.vars = {}
        self.funcs = {}
        self.passes = 0

    def set_workdir(self, dir):
        self.work_dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)

    def next_pass(self):
        self.passes += 1

    '''
    add variables to watch
    '''
    def watch(self, var, name, var_type=None, attr=None):
        if name in self.vars.keys():
            raise Exception('var with name {} already exists')
        self.vars[name] = VarInfo(var, var_type if var_type else type(var), attr)

    '''
    register setter/getter functions for var_type
    '''
    def register_funcs(self, var_type, setter=None, getter=None):
        self.funcs[var_type] = FuncInfo(setter, getter)

    def _get_filename(self, name):
        return os.path.join(self.work_dir, '{}_{}'.format(self.passes, name))

    '''
    load raw value from npy file in work_dir
    '''
    def load_raw_value(self, name):
        if os.path.exists(self._get_filename(name)+'.npy'):
            return np.load(self._get_filename(name)+'.npy')
        elif os.path.exists(self._get_filename(name)+'.pkl'):
            with open(self._get_filename(name)+'.pkl', 'rb') as pkl:
                return pickle.load(pkl)

    '''
    set value to var, with option to load from disk
    '''
    def assign(self, name, value=None, load=False, load_name=None):
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

    '''
    evaluate var and optionally save to work_dir
    '''
    def fetch(self, name, save=False):
        var, var_type, attr = self.vars[name]
        raw_value = self.funcs[var_type].getter(var, attr)
        if save:
            if type(raw_value) == np.ndarray:
                np.save(self._get_filename(name)+'.npy', raw_value)
            else:
                with open(self._get_filename(name)+'.pkl', 'wb') as pkl:
                    pickle.dump(raw_value, pkl)
        return raw_value
        
    '''
    compare var value with work_dir
    '''
    def compare(self, name, compare_name=None, rtol=1e-05, atol=1e-08, equal_nan=False):
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

    '''
    load vars in list of names
    '''
    def load(self, names):
        [self.assign(n, load=True) for n in names if n in self.vars.keys()]
            
    '''
    save vars in list
    '''
    def save(self, names):
        [self.fetch(n, save=True) for n in names if n in self.vars.keys()]

    '''
    save all vars
    '''
    def save_all(self):
        self.save(self.vars.keys())