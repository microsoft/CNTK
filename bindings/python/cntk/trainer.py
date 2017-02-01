
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .device import use_default_device
from .utils import sanitize_var_map, sanitize_function, typemap, value_to_seq, variable_value_to_seq
from .io import _py_dict_to_cntk_dict, MinibatchData

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
       criterion (:class:`~cntk.ops.functions.Function`): Function with one or two outputs,
        representing loss and evaluation metric (in this order).
        Alternatively, a tuple(loss Function, evaluation Function) is also accepted.
       parameter_learners (list): list of learners from :mod:`cntk.learner`
    '''

    @staticmethod
    def _get_loss_metric(criterion): # helper to interpret criterion parameter
        if isinstance(criterion, cntk_py.Function): # input can be a tuple of Functions or a tuple-valued Function
            criterion = criterion.outputs           # break up tuple-valued Function into tuple of Functions
        # map Variable to Function
        from cntk import combine
        #criterion = tuple([sanitize_function(output) for output in criterion])
        # BUGBUG: for tuple-valued BlockFunctions, sanitize_function() returns the tuple; must use combine() instead
        criterion = tuple([combine([output], output.name) if isinstance(output, cntk_py.Variable) else output for output in criterion])
        if not isinstance(criterion, tuple): # input can be a single value or a tuple (loss, metric)
            criterion = (criterion, None)    # if single then pad with None for the metric
        if len(criterion) == 1:
            criterion = criterion + (None,) # tuple of 1 value: pad with None
        if len(criterion) != 2:
            raise ValueError("criterion parameter must be a singleton or a tuple of 2 elements")
        return criterion

    def __init__(self, model, criterion, parameter_learners):
        loss_function, eval_function = Trainer._get_loss_metric(criterion)
        # TODO sanitizing should be removed once Swig's typemaps are in place
        if model is not None:  # None means dummy model that is, e.g., the same as a criterion
            model = sanitize_function(model)
        loss_function = sanitize_function(loss_function)
        if eval_function is not None:
            eval_function = sanitize_function(eval_function)
        if not isinstance(parameter_learners, list):
            parameter_learners = [parameter_learners]

        trainer = cntk_py.trainer_impl(model, loss_function, eval_function, parameter_learners)
        # transplant into this class instance
        self.__dict__ = trainer.__dict__

    def _train_test_mb_map_args(self, *args, **kwargs):
        '''helper function for mimicking Python calling convention in train/test_minibatch()'''
        # one argument, which is an arg map
        if len(args) == 1 and isinstance(args[0], dict):
            return args[0]
        # map to function arguments
        args = self.loss_function.argument_map(*args, **kwargs)
        # in this use case, all must have the same inputs
        if self.model:
            if self.loss_function.arguments != self.model.arguments:
                raise ValueError("model function must have the same signature and inputs as the loss function")
        if self.evaluation_function:
            if self.loss_function.arguments != self.evaluation_function.arguments:
                raise ValueError("evaluation function must have the same signature and inputs as the loss function")
        return args

    def train_minibatch(self, *arguments, **kwargs):
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
                TODO: document more complex use case
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
        outputs, device, kwargs = (lambda outputs=None, device=None, **kwargs: (outputs, device, kwargs))(**kwargs) # Python 2.7 does not allow (self, *arguments, outputs=None, device=None, **kwargs)
        if not device:
            device = use_default_device()

        # mimic Python function-call semantics unique input or multiple arguments are given
        arguments = self._train_test_mb_map_args(*arguments, **kwargs)

        if arguments: # arguments must feed all inputs (model, loss, eval)
            all_args = set(self.loss_function.arguments)
            if self.model:
                all_args |= set(self.model.arguments)
            if self.evaluation_function:
                all_args |= set(self.evaluation_function.arguments)
            arguments = sanitize_var_map(tuple(all_args), arguments,
                extract_values_from_minibatch_data = False)

        contains_minibatch_data = False
        if (len(arguments) > 0):
            value = next(iter(arguments.values()))
            contains_minibatch_data = isinstance(value, MinibatchData)

        if outputs:
            output_map = {v: None for v in outputs}
            
            if contains_minibatch_data:
                updated = super(Trainer, self).train_minibatch_overload_for_minibatchdata(
                    arguments, output_map, device)
            else:    
                updated = super(Trainer, self).train_minibatch(arguments,
                    output_map, device)

            for k,v in output_map.items():
                output_map[k] = variable_value_to_seq(v, k)

            return updated, output_map
        else:

            if contains_minibatch_data:
                updated = super(Trainer, self).train_minibatch_overload_for_minibatchdata(
                    arguments, device)
            else:    
                updated = super(Trainer, self).train_minibatch(arguments,
                    device)

        return updated

    def test_minibatch(self, *arguments, **kwargs):
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
               TODO: update this
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
        device, kwargs = (lambda device=None, **kwargs: (device, kwargs))(**kwargs) # Python 2.7 does not allow (self, *arguments, device=None, **kwargs)
        if not device:
            device = use_default_device()

        # mimic Python function-call semantics unique input or multiple arguments are given
        arguments = self._train_test_mb_map_args(*arguments, **kwargs)

        # pass all args of all parts (model, loss, eval)
        all_args = set(self.loss_function.arguments)
        if self.model:
            all_args |= set(self.model.arguments)
        if self.evaluation_function:
            all_args |= set(self.evaluation_function.arguments)
        arguments = sanitize_var_map(tuple(all_args), arguments)

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
        Restores a checkpoint of the model and Trainer state from the
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


def Evaluator(model, criterion):
    '''
    Create an evaluator. This is really a Trainer object with dummy SGD parameters that we can call test_minibatch() on.
    '''
    loss, metric = Trainer._get_loss_metric(criterion)
    from .learner import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
    parameters = set(loss.parameters)
    if model:
        parameters |= set(model.parameters)
    if metric:
        parameters |= set(metric.parameters)
    dummy_learner = momentum_sgd(tuple(parameters), 
                                 lr = learning_rate_schedule(1, UnitType.minibatch),
                                 momentum = momentum_as_time_constant_schedule(0))
    return Trainer(model, (loss, metric), dummy_learner)
