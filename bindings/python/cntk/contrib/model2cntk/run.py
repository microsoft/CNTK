# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys

from . import adapter
from .unimodel import cntkinstance
from .utils import globalconf
from .validation import validcore


class ModelConverter(object):
    """
    The model converter includes two statis methods.
    convert_script(Coming soon): used to convert the training script
    convert_model: used to convert the trained model
    """
    @staticmethod
    def script_conversion(_):
        """
        The function is used to convert the scripts into CNTK python script (Coming soon)

        Args:
            conf (string): the path of configuration FileExistsError

        Returns:
            None
        """
        raise AssertionError('Discard for now')

    @staticmethod
    def convert_model(conf_path):
        """
        The function is used to convert the model of Caffe to CNTK

        Args:
            conf (string): the path of configuration file

        Returns:
            None
        """
        conf = globalconf.load_conf(conf_path)
        try:
            adapter_impl = adapter.ADAPTER_DICT[conf.source_solver.source]
        except KeyError:
            sys.stderr.write('un-implemented platform type\n')
        cntk_model_desc = adapter_impl.load_model(conf)

        instance = cntkinstance.CntkApiInstance(cntk_model_desc, global_conf=conf)
        instance.export_model()

        # valid the network
        validator = validcore.Validator(global_conf=conf, functions=instance.get_functions())
        if validator.val_network():
            validator.activate()
