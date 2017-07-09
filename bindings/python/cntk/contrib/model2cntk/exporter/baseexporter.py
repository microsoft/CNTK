# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from abc import ABCMeta, abstractmethod


class BaseExporter(object):
    __metaclass__ = ABCMeta

    def __init__(self, uni_model_desc):
        self._uni_model_desc = uni_model_desc

    @abstractmethod
    def export_scripts(self, export_path): pass

    @abstractmethod
    def export_network(self, export_path, cntk_model_desc): pass
