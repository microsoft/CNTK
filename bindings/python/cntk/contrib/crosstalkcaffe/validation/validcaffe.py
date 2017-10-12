# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys

from abc import ABCMeta, abstractmethod

import numpy as np
from cntk.contrib.crosstalkcaffe.adapter.bvlccaffe.caffeimpl import CaffeResolver

class ValidCore(object):
    '''
     The abstract class to support different validation methods
    '''
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def execute(source_solver, valid_dir):
        '''
         execute the validation
        '''
        pass


class CaffeValidCore(ValidCore):
    '''
     Validation module of Caffe side.
    '''
    @staticmethod
    def execute(source_solver, valid_dir):
        '''
         Execute the validation in Caffe side.

        Args:
            source_solver (:class:`~cntk.contrib.crosstalkcaffe.utils.globalconf.SourceSolverConf`):
                the source solver instanced from global configuration
            valid_dir (str): the path to save temporary CNTK forward results

        Return:
            None
        '''
        caffe_solver = CaffeResolver()
        caffe = caffe_solver.caffe
        if not caffe_solver.runtime():
            sys.__stdout__.write('No caffe runtime support, ignore validation...\n')
            return
        sys.__stdout__.write('Start valid feature map...\n')
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(source_solver.model_path, source_solver.weights_path, caffe.TEST)
        for name in net.inputs:
            input_blob = net.blobs[name]
            target_array = np.load(os.path.join(valid_dir, name + '.npy')).reshape(input_blob.data.shape)
            np.copyto(input_blob.data, target_array)
        net.forward()
        for file_name in os.listdir(valid_dir):
            target, _ = os.path.splitext(file_name)
            if target in net.inputs:
                continue
            gt_result = np.load(os.path.join(valid_dir, file_name))
            test_result = net.blobs[target].data
            power_error = np.power(gt_result.flatten() - test_result.flatten(), 2).sum()
            rsme_diff = np.sqrt(power_error / gt_result.size)
            sys.__stdout__.write(('Validating %s with RMSE = %s, MAX = %s, MIN = %s\n' %
                             (target, str(rsme_diff), str(gt_result.max()), str(gt_result.min()))))
        sys.__stdout__.write('Validation finished...\n')
        sys.__stdout__.flush()
