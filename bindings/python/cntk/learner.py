# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
from . import cntk_py
from .utils import typemap
from enum import Enum, unique

@unique
class UnitType(Enum):
    '''
    Indicates whether the values in the schedule are specified on the per-sample or 
    per-minibatch basis.
    '''

    sample = 1
    '''
    Schedule contains per-sample values.
    '''

    minibatch = 2
    '''
    Schedule contains per-minibatch values (and need to be re-scaled by the learner
    using the actual minibatch size in samples).
    '''


__doc__='''
Learner tunes a set of parameters during the training process. One can use
different learners for different sets of parameters. Currently, CNTK supports
the following learning algorithms:

+------------------------+
| Learning algorithms    |
+========================+
| AdaGrad                |
+------------------------+
| Adam                   |
| (a low memory variant) |
+------------------------+
| MomentumSGD            |
+------------------------+
| Nesterov               |
+------------------------+
| RMSProp                |
+------------------------+
| SGD                    |
+------------------------+
'''

class Learner(cntk_py.Learner):
    '''
    Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    For e.g momentum, AdaGrad, RMSProp etc. are different types of learners with their own algorithms for
    learning parameter values using first order gradients.
    To instantiate a concrete learner, use the factory methods in this module.
    '''

    def update(self, gradient_values, training_sample_count):
        '''
        Update the parameters associated with this learner.

        Args:
            gradient_values (`dict`): maps :class:`cntk.variables.Parameter` to
             a NumPy array containing the first order gradient values for the
             Parameter w.r.t. the training objective.
            training_sample_count (`int`): training sample count

        Returns:
            `False` to indicate that learning has stopped for all of the parameters associated with this learner
        '''
        from .utils import create_NDArrayView_from_NumPy
        var_nd_map = { var:create_NDArrayView_from_NumPy(val) for var, val in
                gradient_values.items() }

        return super(Learner, self).update(var_nd_map, training_sample_count)

    @property
    @typemap
    def parameters(self):
        '''
        The set of parameters associated with this learner.
        '''
        return super(Learner, self).parameters()


    def reset_learning_rate(self, learning_rate):
        '''
        Resets the learning rate.

        Args:
            learning_rate (`float`, `list` or a training schedule): learning rate 
            to reset to
        '''
        learning_rate = learning_rate_schedule(learning_rate)
        return super(Learner, self).reset_learning_rate(learning_rate)

    def learning_rate(self, minibatch_size=1):
        '''
        The learning rate.

        Args:
            minibatch_size (`int`): minibatch size to re-scaled
            the learning rate to the per-sample value (in case when the schedule 
            was build with unit=UnitType.minibatch).
        '''
        return super(Learner, self).learning_rate(minibatch_size)

@typemap
def training_parameter_schedule(schedule, epoch_size=1, unit=UnitType.sample):
    '''
    Create a training parameter schedule containing either per-sample (default)
    or per-minibatch values.

    Examples:
        >>> # Use a fixed value 0.01 for all samples
        >>> s = training_parameter_schedule(0.01)
        >>> s[0], s[1]
        (0.01, 0.01)

        >>> # Use 0.01 for the first 1000 samples, then 0.001 for the remaining ones
        >>> s = training_parameter_schedule([0.01, 0.001], 1000)
        >>> s[0], s[1], s[1000], s[1001]
        (0.01, 0.01, 0.001, 0.001)

        >>> # Use 0.1 for the first 12 epochs, then 0.01 for the next 15,
        >>> # followed by 0.001 for the remaining ones, with a 100 samples in an epoch
        >>> s = training_parameter_schedule([(12, 0.1), (15, 0.01), (1, 0.001)], 100)
        >>> s[0], s[1199], s[1200], s[2699], s[2700], s[5000]
        (0.1, 0.1, 0.01, 0.01, 0.001, 0.001)

    Args:
        schedule (`float` or `list`): if `float`, is the parameter schedule to be used
         for all samples. In case of list, the elements are used as the
         values for ``epoch_size`` samples. If list contains pair, the second element is
         used as a value for (``epoch_size`` x first element) samples
        epoch_size (`int`): number of samples as a scheduling unit. Parameters in
         the schedule change their values every 'epoch_size' samples.
        unit (:class:`cntk.ops.functions.UnitType`): one of two

          * 'sample': the returned schedule contains per-sample values (default)
          * 'minibatch': the returned schedule contains per-minibatch values.

    Returns:
        training parameter schedule
    '''
    if not isinstance(unit, UnitType):
            raise ValueError('schedule unit "%s" is not supported' %
                    str(method))

    if isinstance(schedule, (cntk_py.training_parameter_per_sample_schedule, 
                             cntk_py.training_parameter_per_minibatch_schedule,
                             cntk_py.momentum_as_time_constant_schedule)):
        return schedule

    if isinstance(schedule, (int, float)):
        if unit is UnitType.sample:
            return cntk_py.training_parameter_per_sample_schedule(schedule)
        else:
            return cntk_py.training_parameter_per_minibatch_schedule(schedule)

    if isinstance(schedule, list):
        if unit is UnitType.sample:
            return cntk_py.training_parameter_per_sample_schedule(schedule, epoch_size)
        else:
            return cntk_py.training_parameter_per_minibatch_schedule(schedule, epoch_size)

    raise ValueError('schedule must be either a float or a list, not %s'%type(schedule))

@typemap
def learning_rate_schedule(lr, epoch_size=1, unit=UnitType.sample):
    '''
    Create a learning rate schedule (using the same semantics as 
    :func:`training_parameter_schedule`).

    Args:
        lr (`float` or `list`): see parameter ``schedule`` in 
         :func:`training_parameter_schedule`.
        epoch_size (`int`): see parameter ``epoch_size`` in 
         :func:`training_parameter_schedule`.
        unit (:class:`cntk.ops.functions.UnitType`): see parameter 
         ``unit`` in :func:`training_parameter_schedule`.

    Returns:
        learning rate schedule
    '''
    return training_parameter_schedule(lr, epoch_size, unit)

@typemap
def momentum_schedule(momentum, epoch_size=1, unit=UnitType.sample):
    '''
    Create a momentum schedule (using the same semantics as 
    :func:`training_parameter_schedule`).

    Args:
        momentum (`float` or `list`): see parameter ``schedule`` in 
         :func:`training_parameter_schedule`.
        epoch_size (`int`): see parameter ``epoch_size`` in 
         :func:`training_parameter_schedule`.
        unit (:class:`cntk.ops.functions.UnitType`): see parameter 
         ``unit`` in :func:`training_parameter_schedule`.

    If you want to provide momentum values in a sample/minibatch
    agnostic way, use :func:`momentum_as_time_constant_schedule`.

    Examples:
        >>> # Use a fixed momentum of 0.99 for all samples
        >>> m = momentum_schedule(0.99)

        >>> # Use the momentum value 0.99 for the first 1000 samples, 
        >>> # then 0.9 for the remaining ones
        >>> m = momentum_schedule([0.99,0.9], 1000)
        >>> m[0], m[999], m[1000], m[1001]
        (0.99, 0.99, 0.9, 0.9)
        
        >>> # Use the momentum value 0.99 for the first 999 samples, 
        >>> # then 0.88 for the next 888 samples, and 0.77 for the
        >>> # the remaining ones
        >>> m = momentum_schedule([(999,0.99),(888,0.88),(0, 0.77)])
        >>> m[0], m[998], m[999], m[999+888-1], m[999+888]
        (0.99, 0.99, 0.88, 0.88, 0.77)

    Args:
        momentum (`float` or `list`): see parameter ``schedule`` in 
         :func:`training_parameter_schedule`.
        epoch_size (`int`): see parameter ``epoch_size`` in 
         :func:`training_parameter_schedule`.
        unit (:class:`cntk.ops.functions.UnitType`): see parameter 
         ``unit`` in :func:`training_parameter_schedule`.

    Returns:
        momentum schedule
    '''
    return training_parameter_schedule(momentum, epoch_size, unit)

@typemap
def momentum_as_time_constant_schedule(momentum, epoch_size=1):
    '''
    Create a momentum schedule in a minibatch agnostic way (using the same 
    semantics as :func:`training_parameter_schedule`).

    Args:
        momentum (`float` or `list`): see parameter ``schedule`` in 
         :func:`training_parameter_schedule`.
        epoch_size (`int`): see parameter ``epoch_size`` in 
         :func:`training_parameter_schedule`.
        unit (:class:`cntk.ops.functions.UnitType`): see parameter 
         ``unit`` in :func:`training_parameter_schedule`.

    CNTK specifies momentum in a minibatch-size agnostic way as the time
    constant (in samples) of a unit-gain 1st-order IIR filter. The value
    specifies the number of samples after which a gradient has an effect of
    1/e=37%.

    If you want to specify the momentum per sample (or per minibatch),
    use :func:`momentum_schedule`.


    Examples:
        >>> # Use a fixed momentum of 1100 for all samples
        >>> m = momentum_as_time_constant_schedule(1100)

        >>> # Use the time constant 1100 for the first 1000 samples, 
        >>> # then 1500 for the remaining ones
        >>> m = momentum_as_time_constant_schedule([1100, 1500], 1000)

    Args:
        momentum (`float` or `list`): see parameter ``schedule`` in 
         :func:`training_parameter_schedule`.
        epoch_size (`int`): see parameter ``epoch_size`` in 
         :func:`training_parameter_schedule`.

    Returns:
        momentum as time constant schedule
    '''
    if isinstance(momentum, (cntk_py.training_parameter_per_sample_schedule, 
                             cntk_py.training_parameter_per_minibatch_schedule,
                             cntk_py.momentum_as_time_constant_schedule)):
        return momentum

    if isinstance(momentum, (int, float)):
        return cntk_py.momentum_as_time_constant_schedule(momentum)
    if isinstance(momentum, list):
        return cntk_py.momentum_as_time_constant_schedule(momentum, epoch_size)

    raise ValueError('momentum must be either a float or a list, not %s'%type(momentum))


# TODO figure out how to pass infty to C++ in a portable way
@typemap
def sgd(parameters, lr,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an SGD learner instance to learn the parameters.

    Args:
        parameters (`list` of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' method of the root
         operator.
        lr ('float', `list` or output of `:func:learning_rate_schedule`): learning rate 
         schedule. When the argument value is a `float` or a `list`, lr is 
         converted to a per-sample schedule by invoking `:func:learning_rate_schedule`.
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a :class:`cntk.learner.Learner` that can be passed to the :class:`cntk.trainer.Trainer`
    '''
    lr = learning_rate_schedule(lr)
    gaussian_noise_injection_std_dev = training_parameter_schedule(gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.sgd_learner(parameters, lr, additional_options)

@typemap
def momentum_sgd(parameters, lr, momentum,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates a Momemtum SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr ('float', `list` or output of `:func:learning_rate_schedule`): learning rate 
         schedule. When the argument value is a `float` or a `list`, lr is 
         converted to a per-sample schedule by invoking `:func:learning_rate_schedule`.
        momentum (`float`, `list` or output of `:func:momentum_schedule` or 
         `:func:momentum_as_time_constant_schedule`): momentum schedule. When the argument 
         value is a `float` or a `list`, momentum is converted to a per-sample schedule by 
         invoking `:func:momentum_schedule`. Refer to 
         https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a :class:`cntk.learner.Learner` that can be passed to the :class:`cntk.trainer.Trainer`
    '''
    lr = learning_rate_schedule(lr)
    momentum = momentum_schedule(momentum)
    gaussian_noise_injection_std_dev = training_parameter_schedule(gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.momentum_sgd_learner(parameters, lr, momentum,
            additional_options)

@typemap
def nesterov(parameters, lr, momentum,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates a Nesterov SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr ('float', `list` or output of `:func:learning_rate_schedule`): learning rate 
         schedule. When the argument value is a `float` or a `list`, lr is 
         converted to a per-sample schedule by invoking `:func:learning_rate_schedule`.
        momentum (`float`, `list` or output of `:func:momentum_schedule` or 
         `:func:momentum_as_time_constant_schedule`): momentum schedule. When the argument 
         value is a `float` or a `list`, momentum is converted to a per-sample schedule by 
         invoking `:func:momentum_schedule`. Refer to 
         https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a :class:`cntk.learner.Learner` that can be passed to the :class:`cntk.trainer.Trainer`
    '''
    lr = learning_rate_schedule(lr)
    momentum = momentum_schedule(momentum)
    gaussian_noise_injection_std_dev = training_parameter_schedule(gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.nesterov_learner(parameters, lr, momentum,
            additional_options)

@typemap
def adagrad(parameters, lr, need_ave_multiplier=True,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr ('float', `list` or output of `:func:learning_rate_schedule`): learning rate 
         schedule. When the argument value is a `float` or a `list`, lr is 
         converted to a per-sample schedule by invoking `:func:learning_rate_schedule`.
        need_ave_multiplier ('bool', default):
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a :class:`cntk.learner.Learner` that can be passed to the :class:`cntk.trainer.Trainer`
    '''
    lr = learning_rate_schedule(lr)
    gaussian_noise_injection_std_dev = training_parameter_schedule(gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.ada_grad_learner(parameters, lr, need_ave_multiplier,
            additional_options)

# TODO: unCamelCase and integrate upcoming CR
@typemap
def adam_sgd(parameters, lr, momentum,
        variance_momentum = momentum_as_time_constant_schedule(720000),
        low_memory=True,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an Adam learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr ('float', `list` or output of `:func:learning_rate_schedule`): learning rate 
         schedule. When the argument value is a `float` or a `list`, lr is 
         converted to a per-sample schedule by invoking `:func:learning_rate_schedule`.
        momentum (`float`, `list` or output of `:func:momentum_schedule` or 
         `:func:momentum_as_time_constant_schedule`): momentum schedule. When the argument 
         value is a `float` or a `list`, momentum is converted to a per-sample schedule by 
         invoking `:func:momentum_schedule`. Refer to 
         https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        variance_momentum (`float`, `list` or output of `:func:momentum_schedule` or 
         `:func:momentum_as_time_constant_schedule`): variance momentum schedule. When the argument 
         value is a `float` or a `list`, variance momentum is converted to a per-sample schedule by 
         invoking `:func:momentum_schedule`. Defaults to momentum_as_time_constant_schedule(720000).
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a :class:`cntk.learner.Learner` that can be passed to the :class:`cntk.trainer.Trainer`
    '''
    if not low_memory:
        raise NotImplementedError('adam: low_memory=True currently required')

    lr = learning_rate_schedule(lr)
    momentum = momentum_schedule(momentum)
    variance_momentum = momentum_schedule(variance_momentum)
    gaussian_noise_injection_std_dev = training_parameter_schedule(gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.adam_learner(parameters, lr, momentum,
            variance_momentum, low_memory, additional_options)

@typemap
def rmsprop(parameters, lr,
        gamma, inc, dec, max, min,
        need_ave_multiplier=True,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an RMSProp learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr ('float', `list` or output of `:func:learning_rate_schedule`): learning rate 
         schedule. When the argument value is a `float` or a `list`, lr is 
         converted to a per-sample schedule by invoking `:func:learning_rate_schedule`.
        gamma ('float'):
        inc ('float'):
        dec ('float'):
        max ('float'):
        min ('float'):
        need_ave_multiplier ('bool', default):
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a :class:`cntk.learner.Learner` that can be passed to the :class:`cntk.trainer.Trainer`
    '''
    lr = learning_rate_schedule(lr)
    gaussian_noise_injection_std_dev = training_parameter_schedule(gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.rmsprop_learner(parameters, lr, gamma, inc, dec, max, min,
            need_ave_multiplier, additional_options)

