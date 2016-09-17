# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0'

import os

from . import ops
from . import cntk_py
from .utils import *
from .ops import *

# pulling elements from the wrapper into cntk namespace that do not require
# special wrapping
from .cntk_py import learning_rates_per_sample, sgd_learner, \
    DeviceDescriptor, Constant, StreamConfiguration, \
    text_format_minibatch_source, momentums_per_sample, momentum_sgd_learner, \
    Axis

import numpy as np

DATATYPE = np.float32

class Trainer(cntk_py.Trainer):
    '''
    Trainer to train the specified 'model' with the specified `training_loss` as the training criterion,
    the specified `evaluation_function` as the criterion for evaluating the trained model's quality, and using the specified set
    of `parameters` for updating the model's parameters using computed gradients.
    '''
    def __init__(self, model, loss_function, eval_function, parameters):
        if isinstance(model, cntk_py.Variable):
            model = model.owner
        if isinstance(loss_function, cntk_py.Variable):
            loss_function = loss_function.owner
        if isinstance(eval_function, cntk_py.Variable):
            eval_function = eval_function.owner
        super(Trainer, self).__init__(model, loss_function, eval_function, parameters)

    def train_minibatch(self, arguments, device=None):
        '''
        Optimize model parameters using the specified 'arguments' minibatch of training samples.
        Returns false if all parameter learners indicate end of learning (through their Update method's return value).
        Args:
            arguments (dict): map from input variables to the data, data should be either numpy
            arrays or cntk.Value instances returned by a minibatch source
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that contains the type and id of the device
        Returns:
            bool
        '''
        if not device:
            device=DeviceDescriptor.use_default_device()        
        arguments = sanitize_var_map(arguments, add_batch_axis=True)
        super(Trainer, self).train_minibatch(arguments, device)

    def test_minibatch(self, arguments, device=None):
        '''
        Test the model on the specified batch of samples using the evaluation Function specified during construction of the Trainer
        Returns the average evaluation criterion value per sample for the tested minibatch of samples
        Args:
            arguments (dict): map from input variables to the data, data should be either numpy
            arrays or cntk.Value instances returned by a minibatch source
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that contains the type and id of the device
        Returns:
            float
        '''
        if not device:
            device=DeviceDescriptor.use_default_device()        
        arguments = sanitize_var_map(arguments, add_batch_axis=True)
        super(Trainer, self).train_minibatch(arguments, device)

