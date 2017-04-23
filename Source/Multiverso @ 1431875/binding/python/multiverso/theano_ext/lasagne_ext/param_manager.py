#!/usr/bin/env python
# coding:utf8

import lasagne
from ..param_manager import MVModelParamManager


class LasagneParamManager(MVModelParamManager):
    '''
    LasagneParamManager is manager to make managing and synchronizing the
    variables in lasagne more easily
    '''

    def get_all_param_values(self):
        return lasagne.layers.get_all_param_values(self.model)

    def set_all_param_values(self, params):
        lasagne.layers.set_all_param_values(self.model, params)
