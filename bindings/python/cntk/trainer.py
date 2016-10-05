# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .cntk_py import DeviceDescriptor
from .utils import sanitize_var_map

class Trainer(cntk_py.Trainer):
    '''
    Trainer to train the specified `model` with the specified `training_loss`
    as the training criterion, the specified `evaluation_function` as the
    criterion for evaluating the trained model's quality, and using the
    specified set of `parameter_learners` for updating the model's parameters
    using computed gradients.

    Args:
       model (`:class:cntk.ops.Function`): root node of the function to train
       loss_function (`:class:cntk.ops.Function`): loss function 
       eval_function (`:class:cntk.ops.Function`): evaluation function
       parameter_learners (`list`): list of learners from `:cntk:cntk.learner`
    '''
    def __init__(self, model, loss_function, eval_function, parameter_learners):
        if isinstance(model, cntk_py.Variable):
            model = model.owner
        self.model = model
        if isinstance(loss_function, cntk_py.Variable):
            loss_function = loss_function.owner
        if isinstance(eval_function, cntk_py.Variable):
            eval_function = eval_function.owner
        super(Trainer, self).__init__(model, loss_function, eval_function,
                parameter_learners)

    def train_minibatch(self, arguments, device=None):
        '''
        Optimize model parameters using the specified 'arguments' minibatch of training samples.
        Returns false if all parameter learners indicate end of learning (through their Update method's return value).

        Args:
            arguments (`dict` or `list`): map from input variables to the data
             or list of inputs in the order that the function expects. Data
             should be either NumPy arrays or cntk.Value instances returned by a
             minibatch source.
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool`: `True` if updates have been performed
        '''
        if not device:
            device=DeviceDescriptor.use_default_device()        
        arguments = sanitize_var_map(self.model.arguments(), arguments, add_batch_axis=True)

        return super(Trainer, self).train_minibatch(arguments, device)

    def test_minibatch(self, arguments, device=None):
        '''
        Test the model on the specified batch of samples using the evaluation
        Function specified during construction of the Trainer. 
        of samples.

        Args:
            arguments (`dict` or `list`): map from input variables to the data
             or list of inputs in the order that the function expects. Data
             should be either NumPy arrays or cntk.Value instances returned by a
             minibatch source.
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
        Returns:
            `float`: the average evaluation criterion value per sample for the
              tested minibatch.
        '''
        if not device:
            device=DeviceDescriptor.use_default_device()        
        arguments = sanitize_var_map(self.model.arguments(), arguments, add_batch_axis=True)

        return super(Trainer, self).test_minibatch(arguments, device)

