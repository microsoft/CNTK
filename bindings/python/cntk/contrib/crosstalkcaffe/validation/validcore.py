# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
import numpy as np

from . import validcaffe

VALID_CORES = {
    'Caffe': validcaffe.CaffeValidCore(),
}


class Validator(object):
    '''
     The validation module to check the difference between original Caffe and converted CNTK models
    '''
    def __init__(self, global_conf, functions):
        self._source_solver = global_conf.source_solver
        self._valid_solver = global_conf.valid_solver
        self._functions = functions

    def val_network(self):
        '''
         The function executes CNTK forward and dumps watched nodes into temporary files with
            numpy.mat formats

        Args:
            None

        Return:
            bool: whether to validate the network
        '''
        if self._valid_solver.save_path is None:
            sys.stdout.write('ignore validation network...\n')
            return False
        val_inputs = [value for key, value in self._functions.items() if key \
            in self._valid_solver.val_inputs]
        val_nodes = [value for key, value in self._functions.items() if key \
            in self._valid_solver.val_nodes.keys()]
        if not os.path.exists(self._valid_solver.save_path):
            os.mkdir(self._valid_solver.save_path)

        def _parser_save_path(dir_path, node_name):
            file_name = '.'.join((node_name, 'npy'))
            file_name = file_name.replace('\\', '.')
            file_name = file_name.replace('/', '.')
            return os.path.join(dir_path, file_name)

        valid_augments = {}
        for val_input in val_inputs:
            source_val_input = self._valid_solver.val_inputs[val_input.name][0]
            if len(source_val_input) == 2:
                [lower, upper] = source_val_input
                input_array = (upper - lower) * np.random.random_sample((1, ) + val_input.shape)
            else:
                input_array = np.array(source_val_input).reshape((1, ) + val_input.shape)
            valid_augments[val_input.name] = input_array.astype(np.float32)
            save_path = _parser_save_path(self._valid_solver.save_path, val_input.name)
            np.save(save_path, input_array)
        for val_node in val_nodes:
            used_augments = {augment: valid_augments[augment.name] \
                for augment in val_node.arguments }
            val_results = val_node.forward(used_augments)
            output_array = list(val_results[1].values())[0]
            save_path = _parser_save_path(self._valid_solver.save_path, \
                                          self._valid_solver.val_nodes[val_node.name])
            np.save(save_path, output_array)

        return True

    def activate(self):
        '''
         The function executes target validation core
        
        Args:
            None
        
        Return:
            None
        '''
        VALID_CORES[self._source_solver.source].execute(
            self._source_solver, self._valid_solver.save_path)

