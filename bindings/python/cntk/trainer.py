# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .cntk_py import DeviceDescriptor
from .utils import sanitize_var_map, sanitize_function, typemap

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
        # TODO sanitizing should be removed once Swig's typemaps are in place
        model = sanitize_function(model)
        loss_function = sanitize_function(loss_function)
        eval_function = sanitize_function(eval_function)

        super(Trainer, self).__init__(model, loss_function, eval_function,
                parameter_learners)

    def train_minibatch(self, arguments, device=None):
        '''
        Optimize model parameters using the specified 'arguments' minibatch of training samples.
        Returns false if all parameter learners indicate end of learning (through their Update method's return value).

        Args:
            arguments (`dict` or `list` or single input): 
              * map from input variables to the data
              * list of inputs in the order that the function expects or 
              * a single input, if the function only has one argument. 
              Data should be either NumPy arrays or a `:class:cntk.io.MinibatchSource`
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool`: `True` if updates have been performed
        '''
        if not device:
            device=DeviceDescriptor.use_default_device()        
        arguments = sanitize_var_map(self.model().arguments(), arguments)

        return super(Trainer, self).train_minibatch(arguments, device)

    def test_minibatch(self, arguments, device=None):
        '''
        Test the model on the specified batch of samples using the evaluation
        Function specified during construction of the Trainer. 
        of samples.

        Args:
            arguments (`dict` or `list` or single input): 
              * map from input variables to the data
              * list of inputs in the order that the function expects or 
              * a single input, if the function only has one argument. 
              Data should be either NumPy arrays or a `:class:cntk.io.MinibatchSource`
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
        Returns:
            `float`: the average evaluation criterion value per sample for the
              tested minibatch.
        '''
        if not device:
            device=DeviceDescriptor.use_default_device()        
        arguments = sanitize_var_map(self.model().arguments(), arguments, add_batch_axis=True)

        return super(Trainer, self).test_minibatch(arguments, device)

    def save_checkpoint(self, filename):
        '''
        Saves a checkpoint of the model and other Trainer state at the
        specified file location.

        Args:
            filename (`str`): filename to store the checkpoint
        '''

        super(Trainer, self).save_checkpoint(filename)

    def restore_from_checkpoint(self, filename):
        '''
        Saves a checkpoint of the model and other Trainer state at the
        specified file location.

        Args:
            filename (`str`): filename to restore the checkpoint from
        '''

        super(Trainer, self).restore_from_checkpoint(filename)

    @typemap
    def model(self):
        '''
        Returns the model that the trainer is training.

        Returns:
            `:class:cntk.Function`
        '''
        return super(Trainer, self).model()
        
    @typemap
    def loss_function(self):
        '''
        Returns the loss function that the trainer is using.

        Returns:
            `:class:cntk.Function`
        '''
        return super(Trainer, self).loss_function()

    @typemap
    def evaluation_function(self):
        '''
        Returns the evaluation function that the trainer is using.

        Returns:
            `:class:cntk.Function`
        '''
        return super(Trainer, self).evaluation_function()

    @typemap
    def parameter_learners(self):
        '''
        Returns the parameter learners that the trainer is using.

        Returns:
            `list` of `:class:cntk.learner.Learner`
        '''
        return super(Trainer, self).parameter_learners()

    def previous_minibatch_loss_average(self):
        '''
        Returns the average training loss per sample for the last minibatch trained

        Returns:
            `double`
        '''
        return super(Trainer, self).previous_minibatch_loss_average()

    def previous_minibatch_evaluation_average(self):
        '''
        Returns the average evaluation criterion value per sample for the last minibatch trained

        Returns:
            `double`
        '''
        return super(Trainer, self).previous_minibatch_evaluation_average()

    def previous_minibatch_sample_count(self):
        '''
        Returns the number of samples in the last minibatch trained with

        Returns:
            `int`
        '''
        return super(Trainer, self).previous_minibatch_sample_count()

