
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .device import use_default_device
from .utils import sanitize_var_map, sanitize_function, typemap, value_to_seq
from .io import _py_dict_to_cntk_dict

__doc__= '''\
A trainer encapsulates the overall training process and employs one or more
:doc:`learners <cntk.learner>` to tune the parameters of a specified model
using gradients of parameters w.r.t. a training objective.
'''

class Trainer(cntk_py.Trainer):
    '''
    Trainer to train the specified ``model`` with the specified ``training_loss``
    as the training criterion, the specified ``evaluation_function`` as the
    criterion for evaluating the trained model's quality, and using the
    specified set of ``parameter_learners`` for updating the model's parameters
    using computed gradients.

    Args:
       model (:class:`~cntk.ops.functions.Function`): root node of the function to train
       loss_function (:class:`~cntk.ops.functions.Function`): loss function
       eval_function (:class:`~cntk.ops.functions.Function`): evaluation function
       parameter_learners (list): list of learners from :mod:`cntk.learner`
    '''
    def __init__(self, model, loss_function, eval_function, parameter_learners):
        # TODO sanitizing should be removed once Swig's typemaps are in place
        model = sanitize_function(model)
        loss_function = sanitize_function(loss_function)
        eval_function = sanitize_function(eval_function)
        if not isinstance(parameter_learners, list):
            parameter_learners = [parameter_learners]

        super(Trainer, self).__init__(model, loss_function, eval_function, parameter_learners)

    def train_minibatch(self, arguments, outputs=None, device=None):
        '''
        Optimize model parameters using the specified 'arguments' minibatch of training samples.

        Args:
            arguments: maps variables to their
             input data. Empty map signifies end of local training data.
             The interpretation depends on the input type:
               * `dict`: keys are input variable or names, and values are the input data.
                 See :meth:`~cntk.ops.functions.Function.forward` for details on passing input data.
               * any other type: if node has an unique input, ``arguments`` is mapped to this input.
                For nodes with more than one input, only `dict` is allowed.
             In both cases, every every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify ``arguments`` as `tuple`: the
             first element will be used as ``arguments``, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            outputs (iterable): outputs to fetch values for.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool` or `tuple`:
            If ``outputs`` have not been provided, the returned value is `True`
            if updates have been performed, `False` if all parameter learners
            indicate end of learning (through their update). Otherwise, the
            return value is a tuple of the that `bool` and a dictionary that
            maps the variables in `outputs` to their respective NumPy arrays.
        '''
        if not device:
            device = use_default_device()

        if arguments:
            arguments = sanitize_var_map(self.model.arguments, arguments)

        if outputs:
            output_map = {v: None for v in outputs}
            updated = super(Trainer, self).train_minibatch(arguments,
                    output_map, device)
            for k,v in output_map.items():
                output_map[k] = value_to_seq(v)

            return updated, output_map
        else:
            updated = super(Trainer, self).train_minibatch(arguments, device)

        return updated


    def test_minibatch(self, arguments, device=None):
        '''
        Test the model on the specified batch of samples using the evaluation
        Function specified during construction of the Trainer.

        Args:
            arguments: maps variables to their
             input data. The interpretation depends on the input type:

               * `dict`: keys are input variable or names, and values are the input data.
                 See :meth:`~cntk.ops.functions.Function.forward` for details on passing input data.
               * any other type: if node has an unique input, ``arguments`` is mapped to this input.
                For nodes with more than one input, only `dict` is allowed.
             In both cases, every every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify ``arguments`` as `tuple`: the
             first element will be used as ``arguments``, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
        Returns:
            `float`: the average evaluation criterion value per sample for the
              tested minibatch.
        '''
        if not device:
            device = use_default_device()
        arguments = sanitize_var_map(self.model.arguments, arguments)

        return super(Trainer, self).test_minibatch(arguments, device)

    def save_checkpoint(self, filename, external_state={}):
        '''
        Saves a checkpoint of the model and other Trainer state at the
        specified file location.

        Args:
            filename (str): filename to store the checkpoint.
        '''

        super(Trainer, self).save_checkpoint(filename, _py_dict_to_cntk_dict(external_state))

    def restore_from_checkpoint(self, filename):
        '''
        Saves a checkpoint of the model and other Trainer state at the
        specified file location.

        Args:
            filename (str): filename to restore the checkpoint from
        '''

        super(Trainer, self).restore_from_checkpoint(filename)

    @property
    @typemap
    def model(self):
        '''
        The model that the trainer is training.
        '''
        return super(Trainer, self).model()

    @property
    @typemap
    def loss_function(self):
        '''
        The loss function that the trainer is using.
        '''
        return super(Trainer, self).loss_function()

    @property
    @typemap
    def evaluation_function(self):
        '''
        The evaluation function that the trainer is using.
        '''
        return super(Trainer, self).evaluation_function()

    @property
    @typemap
    def parameter_learners(self):
        '''
        The parameter learners that the trainer is using.
        '''
        return super(Trainer, self).parameter_learners()

    @property
    def previous_minibatch_loss_average(self):
        '''
        The average training loss per sample for the last minibatch trained
        '''
        return super(Trainer, self).previous_minibatch_loss_average()

    @property
    def previous_minibatch_evaluation_average(self):
        '''
        The average evaluation criterion value per sample for the last minibatch trained
        '''
        return super(Trainer, self).previous_minibatch_evaluation_average()

    @property
    def previous_minibatch_sample_count(self):
        '''
        The number of samples in the last minibatch trained with
        '''
        return super(Trainer, self).previous_minibatch_sample_count()

    @property
    def total_number_of_samples_seen(self):
        '''
        The number of samples seen globally between all workers from the beginning of training.
        '''
        return super(Trainer, self).total_number_of_samples_seen()
