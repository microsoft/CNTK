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


class CaffeConverter(object):
    '''
     Convert Caffe CNN into CNTK formats
    '''
    @staticmethod
    def from_model(conf_path):
        '''
         Convert a Caffe model to a CNTK model

        Args:
            conf_path (str): Path to the configuration file

        Returns:
            None
        '''
        conf = globalconf.load_conf(conf_path)
        try:
            adapter_impl = adapter.ADAPTER_DICT[conf.source_solver.source]
        except KeyError:
            sys.stderr.write('Platform type not implemented\n')
        cntk_model_desc = adapter_impl.load_model(conf)

        instance = cntkinstance.CntkApiInstance(cntk_model_desc, global_conf=conf)
        instance.export_model()

        # validate the network
        validator = validcore.Validator(global_conf=conf, functions=instance.get_functions())
        if validator.val_network():
            sys.stdout.write('Start validating model: %s\n' % conf.source_solver.model_path)
            validator.activate()
        else:
            # Since the validator needs the runtime caffe, it's an optional choice. 
            sys.stdout.write('Detect validator disable, ignore validating the network\n')
