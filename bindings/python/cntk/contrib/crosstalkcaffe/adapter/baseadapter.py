# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from abc import ABCMeta, abstractmethod


class Adapter(object):
    '''
     The abstact class of model reader
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def load_model(self, global_conf):
        '''
         load the network weights
        '''
        pass
