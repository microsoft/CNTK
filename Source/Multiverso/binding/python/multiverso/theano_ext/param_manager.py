#!/usr/bin/env python
# coding:utf8

import lasagne
import numpy as np
import multiverso as mv


class MVModelParamManager(object):
    '''
    MVModelParamManager is manager to make managing and synchronizing the
    variables in lasagne more easily
    '''
    def __init__(self, model):
        ''' The constructor of MVModelParamManager

        The constructor will associate the parameter with multiverso array
        table.  The initial value of ArrayTableHandler will be same as the
        parameters of model. If different parameters are used in different
        processes, the average of them will be used as the initial value
        '''
        self.shapes = []
        self.dtypes = []
        self.sizes = []
        self.all_param_list = []
        self.model = model
        for arr in self.get_all_param_values():
            self.shapes.append(arr.shape)
            # TODO: Now only float32 is supported in multiverso. So I store all
            # the parameters in a float32 array. This place need modification
            # after other types are supported
            assert(np.dtype("float32") == arr.dtype)
            self.dtypes.append(arr.dtype)
            self.sizes.append(arr.size)
            self.all_param_list.extend([i for i in np.nditer(arr)])
        self.all_param_list = np.array(self.all_param_list)

        self.tbh = mv.ArrayTableHandler(len(self.all_param_list), init_value=self.all_param_list)
        mv.barrier()  # add barrier to make sure the initial values have token effect
        self.all_param_list = self.tbh.get()
        self._set_all_param_to_model()

    def get_all_param_values(self):
        '''Get all param values of specific model

        Gets the parameters of the model. It should return a list of Numpy
        arrays with shapes and types matching the output of
        `set_all_param_values()`.
        '''
        raise NotImplemented()

    def set_all_param_values(self, params):
        '''Set all param values of specific model

        Sets the parameters of the model.  The `params` argument should be a
        list of Numpy arrays with shapes and types matching the output of
        `get_all_param_values()`.
        '''
        raise NotImplemented()

    def _set_all_param_to_model(self):
        n = 0
        params = []
        for i, size in enumerate(self.sizes):
            params.append(self.all_param_list[n:n + size].reshape(self.shapes[i]))
            n += size
        self.set_all_param_values(params)

    def sync_all_param(self):
        '''sync all parameters with multiverso server

        This function will
        1) calc all the delta of params in the model and add the delta to multiverso server
        2) get the latest value from the multiverso server
        '''
        cur_model_params = np.concatenate([
            arr.reshape(-1) for arr in self.get_all_param_values()])

        params_delta = cur_model_params - self.all_param_list
        self.tbh.add(params_delta)
        self.all_param_list = self.tbh.get()
        self._set_all_param_to_model()
