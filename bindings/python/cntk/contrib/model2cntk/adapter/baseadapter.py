# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from abc import ABCMeta, abstractmethod

# TODO: Add support of multiple outputs layers
# TODO: Add filter initialize setting


class Adapter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def load_description(self, solver_path, model_path): pass

    @abstractmethod
    def load_model(self, global_conf): pass


# To support multiple platform, should be implemented
class SetupParameters(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def convolution(native_parameters, input_info, cntk_layer_def): pass

