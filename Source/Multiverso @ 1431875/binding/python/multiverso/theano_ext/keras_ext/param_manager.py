#!/usr/bin/env python

from ..param_manager import MVModelParamManager


class KerasParamManager(MVModelParamManager):
    '''
    KerasParamManager is manager to make managing and synchronizing the
    variables in keras more easily
    '''

    def get_all_param_values(self):
        return self.model.get_weights()

    def set_all_param_values(self, params):
        self.model.set_weights(params)
