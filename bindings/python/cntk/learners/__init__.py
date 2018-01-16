# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
A learner tunes a set of parameters during the training process. One can use
different learners for different sets of parameters. Currently, CNTK supports
the following learning algorithms:

- :func:`AdaDelta <adadelta>`
- :func:`AdaGrad <adagrad>`
- :func:`FSAdaGrad <fsadagrad>`
- :func:`Adam <adam>`
- :func:`MomentumSGD <momentum_sgd>`
- :func:`Nesterov <nesterov>`
- :func:`RMSProp <rmsprop>`
- :func:`SGD <sgd>`
- :func:`Learner with a customized update function <universal>`
'''


from enum import Enum, unique
import warnings
import numpy as np
import cntk.internal.utils as utils

from .. import cntk_py, NDArrayView, asarray
from cntk.internal import typemap
from ..internal.swig_helper import map_if_possible

@unique
class UnitType(Enum):
    '''
    Deprecated:: 2.2

    Indicates whether the values in the schedule are specified on the per-sample or
    per-minibatch basis.
    '''

    sample = 'sample'
    '''
    Schedule contains per-sample values.
    '''

    minibatch = 'minibatch'
    '''
    Schedule contains per-minibatch values (and need to be re-scaled by the learner
    using the actual minibatch size in samples).
    '''

def default_unit_gain_value():
    '''
    Returns true if by default momentum is applied in the unit-gain fashion.
    '''
    return cntk_py.default_unit_gain_value()


def set_default_unit_gain_value(value):
    '''
    Sets globally default unit-gain flag value.
    '''
    cntk_py.set_default_unit_gain_value(value)

# an internal method to verify that the learning rate schedule
# has a proper (per-sample or per-MB schedule) type and raise
# an exception otherwise


def _verify_learning_rate_type(learning_rate):
    if not isinstance(learning_rate,
                      cntk_py.training_double_parameter_schedule):

        raise ValueError('learning_rate type (%s) not supported. '
                         'learning_rate must be a training schedule '
                         '(output of learning_rate_schedule() function)'
                         % type(learning_rate))

# an internal method to verify that the mometum schedule
# has a proper (per-MB or time-constant schedule) type and raise
# an exception otherwise


def _verify_momentum_type(momentum):
    if not isinstance(momentum,
                      cntk_py.training_double_parameter_schedule):

        raise ValueError('momentum type (%s) not supported. '
                         'momentum must be a training schedule '
                         '(output of momentum_schedule() or '
                         'momentum_as_time_constant_schedule() function)'
                         % type(momentum))


class Learner(cntk_py.Learner):

    '''
    Abstraction for learning a subset of parameters of a learnable function using first order gradient values.
    For example momentum, AdaGrad, RMSProp, etc. are different types of learners with their own algorithms for
    learning parameter values using first order gradients.
    To instantiate a concrete learner, use the factory methods in this module.
    '''

    def update(self, gradient_values, training_sample_count, is_sweep_end=False):
        '''
        Update the parameters associated with this learner.

        Args:
            gradient_values (dict): maps :class:`~cntk.variables.Parameter` to
             a NumPy array containing the first order gradient values for the
             Parameter w.r.t. the training objective.
            training_sample_count (int): number of samples in the minibatch
            is_sweep_end (bool): a flag indicating whether it is at the end of a sweep of data

        Returns:
            bool: `False` to indicate that learning has stopped for all of the parameters associated with this learner
        '''
        var_nd_map = {var: NDArrayView.from_data(val) for var, val in
                      gradient_values.items()}

        return super(Learner, self)._update(var_nd_map, training_sample_count, is_sweep_end)

    @property
    @typemap
    def parameters(self):
        '''
        The set of parameters associated with this learner.
        '''
        return super(Learner, self).parameters()

    def reset_learning_rate(self, learning_rate):
        '''
        Resets the learning rate. The new schedule is adjusted to be relative
        to the current number of elapsed samples/sweeps: the 0 offset in
        the new schedule corresponds to the current value of elapsed samples/sweeps,
        and it takes effect from the current position in the training process onwards.

        Args:
            learning_rate (output of :func:`learning_parameter_schedule`)
             learning rate to reset to
        '''
        _verify_learning_rate_type(learning_rate)
        if not learning_rate.is_minibatch_size_explicitly_specified:
            #If the schedule minibatch size is not explicitly specified, the learner's specification will take over
            if self.minibatch_size is not None and self.minibatch_size != self.ignored_minibatch_size:
                learning_rate.minibatch_size = self.minibatch_size
        return super(Learner, self).reset_learning_rate(learning_rate)

    def learning_rate(self):
        '''
        Current learning rate schedule.
        '''
        return super(Learner, self).learning_rate()

IGNORE = Learner.ignored_minibatch_size
'''
Indicate that the minibatch size is ignored in learning's hyper-parameter schedule.
'''

class UserLearner(cntk_py.Learner):

    '''
    Base class of all user-defined learners. To implement your own learning
    algorithm, derive from this class and override the :meth:`update`.

    Certain optimizers (such as AdaGrad) require additional storage.
    This can be allocated and initialized during construction.
    '''

    def __init__(self, parameters, lr_schedule, as_numpy=True):
        super(UserLearner, self).__init__(parameters, lr_schedule)
        self.as_numpy = as_numpy
        self.__disown__()

    def _update(self, gradient_values, training_sample_count, sweep_end):
        '''
        Update the parameters and related state associated with this learner.

        Args:
            gradient_values (dict): maps :class:`~cntk.variables.Parameter`
             to a NumPy array containing the gradient for the Parameter w.r.t.
             the training objective.
            training_sample_count (int): number of samples in the minibatch
            sweep_end (bool): if the data is fed by a conforming reader, this
             indicates whether a full pass over the dataset has just occurred.

        Returns:
            bool: `False` to indicate that learning has stopped for all of the
            parameters associated with this learner
        '''
        map_if_possible(gradient_values)

        if self.as_numpy:
            var_nd_map = {var: asarray(gradient_values[var]) \
                          for var, val in gradient_values.items()}
        else:
            var_nd_map = gradient_values

        return self.update(gradient_values, training_sample_count, sweep_end)

    def update(self, gradient_values, training_sample_count, sweep_end):
        '''
        Update the parameters associated with this learner.

        Args:
            gradient_values (dict): maps :class:`~cntk.variables.Parameter` to
             a NumPy array containing the first order gradient values for the
             Parameter w.r.t. the training objective.
            training_sample_count (int): number of samples in the minibatch
            sweep_end (bool): if the data is fed by a conforming reader, this indicates
             whether a full pass over the dataset has just occurred.

        Returns:
            bool: `False` to indicate that learning has stopped for all of the
            parameters associated with this learner
        '''
        raise NotImplementedError('UserLearner.update must be overriden')

def _prepare_training_parameter_list(schedule):
    if isinstance(schedule, list):
        return [(1, v) if isinstance(v, (float, int)) else v for v in schedule]
    else:
        return schedule

@typemap
def training_parameter_schedule(schedule, unit=UnitType.minibatch, epoch_size=None):
    '''
    Deprecated:: 2.2

    Create a training parameter schedule containing either per-sample (default)
    or per-minibatch values.

    Examples:
        >>> # Use a fixed value 0.01 for all samples
        >>> s = training_parameter_schedule(0.01)
        >>> s[0], s[1]
        (0.01, 0.01)

        >>> # Use 0.01 for the first 1000 samples, then 0.001 for the remaining ones
        >>> s = training_parameter_schedule([0.01, 0.001], epoch_size=1000)
        >>> s[0], s[1], s[1000], s[1001]
        (0.01, 0.01, 0.001, 0.001)

        >>> # Use 0.1 for the first 12 epochs, then 0.01 for the next 15,
        >>> # followed by 0.001 for the remaining ones, with a 100 samples in an epoch
        >>> s = training_parameter_schedule([(12, 0.1), (15, 0.01), (1, 0.001)], epoch_size=100)
        >>> s[0], s[1199], s[1200], s[2699], s[2700], s[5000]
        (0.1, 0.1, 0.01, 0.01, 0.001, 0.001)

    Args:
        schedule (float or list): if float, is the parameter schedule to be used
         for all samples. In case of list, the elements are used as the
         values for ``epoch_size`` samples. If list contains pair, the second element is
         used as a value for (``epoch_size`` x first element) samples
        unit (:class:`UnitType`): one of two
          * ``sample``: the returned schedule contains per-sample values
          * ``minibatch``: the returned schedule contains per-minibatch values.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minbiatch size.
        epoch_size (optional, int): number of samples as a scheduling unit.
         Parameters in the schedule change their values every ``epoch_size``
         samples. If no ``epoch_size`` is provided, this parameter is substituted
         by the size of the full data sweep, in which case the scheduling unit is
         the entire data sweep (as indicated by the MinibatchSource) and parameters
         change their values on the sweep-by-sweep basis specified by the
         ``schedule``.

    Returns:
        training parameter schedule

    See also:
        :func:`learning_parameter_schedule`
    '''

    if unit == UnitType.sample:
        ref_minibatch_size = 1
    else: # unit == UnitType.minibatch
        ref_minibatch_size = cntk_py.training_double_parameter_schedule.ignored_minibatch_size

    if isinstance(schedule, cntk_py.training_double_parameter_schedule):
        schedule.is_minibatch_size_explicitly_specified = True #legacy learning parameter always have the specification
        return schedule

    if isinstance(schedule, (int, float)):
        if epoch_size is not None:
            warnings.warn('When providing the schedule as a number, epoch_size is ignored', RuntimeWarning)
        if UnitType(unit):
            schedule = cntk_py.training_double_parameter_schedule(*[schedule, ref_minibatch_size])
            schedule.is_minibatch_size_explicitly_specified = True  # legacy learning parameter always have the specification
            return schedule

    epoch_size = epoch_size if epoch_size is not None else cntk_py.training_double_parameter_schedule.full_data_sweep
    if isinstance(schedule, list) and UnitType(unit):
        schedule = _prepare_training_parameter_list(schedule)
        args = [schedule, epoch_size, ref_minibatch_size]
        schedule = cntk_py.training_double_parameter_schedule(*args)
        schedule.is_minibatch_size_explicitly_specified = True #legacy learning parameter always have the specification
        return schedule

    raise ValueError(
        'schedule must be either a float or a list, not %s' % type(schedule))

@typemap
def learning_parameter_schedule(schedule, minibatch_size=None, epoch_size=None):
    '''
    Create a learning parameter schedule.

    Args:
        schedule (float or list): if float, is the parameter schedule to be used
         for all samples. In case of list [p_1, p_2, .., p_n], the i-th parameter p_i in the list is used as the
         value from the (``epoch_size`` * (i-1) + 1)-th sample to the (``epoch_size`` * i)-th sample. If list contains 
         pair, i.e. [(num_epoch_1, p_1), (num_epoch_n, p_2), .., (num_epoch_n, p_n)], the i-th parameter is used as a 
         value from the (``epoch_size`` * (num_epoch_0 + ... + num_epoch_2 + ... + num_epoch_(i-1) + 1)-th sample to the 
         (``epoch_size`` * num_epoch_i)-th sample (taking num_epoch_0 = 0 as a special initialization).
        minibatch_size (int): an integer to specify the minibatch size that schedule are designed for. 
         CNTK will scale the schedule internally so as to simulate the behavior of the schedule as much as possible
         to match the designed effect. If it is not specified, CNTK will set to the special value :attr:`IGNORE`.
        epoch_size (optional, int): number of samples as a scheduling unit.
         Parameters in the schedule change their values every ``epoch_size``
         samples. If no ``epoch_size`` is provided, this parameter is substituted
         by the size of the full data sweep, in which case the scheduling unit is
         the entire data sweep (as indicated by the MinibatchSource) and parameters
         change their values on the sweep-by-sweep basis specified by the
         ``schedule``.

    Returns:
        learning parameter schedule
    '''
    if isinstance(schedule, cntk_py.training_double_parameter_schedule):
        return schedule

    is_minibatch_size_explicitly_specified = True
    if minibatch_size == None:
        is_minibatch_size_explicitly_specified = False
        minibatch_size = 0

    if isinstance(schedule, (int, float)):
        if epoch_size is not None:
            warnings.warn('When providing the schedule as a number, epoch_size is ignored', RuntimeWarning)
        schedule = cntk_py.training_double_parameter_schedule(*[schedule, minibatch_size])
        schedule.is_minibatch_size_explicitly_specified = is_minibatch_size_explicitly_specified
        return schedule

    epoch_size = epoch_size if epoch_size is not None else cntk_py.training_double_parameter_schedule.full_data_sweep
    if isinstance(schedule, list):
        schedule = _prepare_training_parameter_list(schedule)
        args = [schedule, epoch_size, minibatch_size]
        schedule = cntk_py.training_double_parameter_schedule(*args)
        schedule.is_minibatch_size_explicitly_specified = is_minibatch_size_explicitly_specified
        return schedule

    raise ValueError(
        'schedule must be either a float or a list, not %s' % type(schedule))


@typemap
def learning_parameter_schedule_per_sample(schedule, epoch_size=None):
    '''
    Create a learning parameter schedule as if the parameter is applied to minibatches of size 1. CNTK
    will scale the parameters accordingly with respect to the actual minibatch size.

    Args:
        schedule (float or list): if float, is the parameter schedule to be used
         for all samples. In case of list [p_1, p_2, .., p_n], the i-th parameter p_i in the list is used as the
         value from the (``epoch_size`` * (i-1) + 1)-th sample to the (``epoch_size`` * i)-th sample. If list contains
         pair, i.e. [(num_epoch_1, p_1), (num_epoch_n, p_2), .., (num_epoch_n, p_n)], the i-th parameter is used as a
         value from the (``epoch_size`` * (num_epoch_0 + ... + num_epoch_2 + ... + num_epoch_(i-1) + 1)-th sample to the
         (``epoch_size`` * num_epoch_i)-th sample (taking num_epoch_0 = 0 as a special initialization).
        epoch_size (optional, int): number of samples as a scheduling unit.
         Parameters in the schedule change their values every ``epoch_size``
         samples. If no ``epoch_size`` is provided, this parameter is substituted
         by the size of the full data sweep, in which case the scheduling unit is
         the entire data sweep (as indicated by the MinibatchSource) and parameters
         change their values on the sweep-by-sweep basis specified by the
         ``schedule``.

    Returns:
        learning parameter schedule as if it is applied to minibatches of size 1.
    '''
    return learning_parameter_schedule(schedule, minibatch_size=1, epoch_size=epoch_size)


@typemap
def learning_rate_schedule(lr, unit, epoch_size=None):
    '''
    Deprecated:: 2.2

    Create a learning rate schedule (using the same semantics as
    :func:`training_parameter_schedule`).

    Args:
        lr (float or list): see parameter ``schedule`` in
         :func:`training_parameter_schedule`.
        unit (:class:`UnitType`): see parameter
         ``unit`` in :func:`training_parameter_schedule`.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minbiatch size instead.
        epoch_size (int): see parameter ``epoch_size`` in
         :func:`training_parameter_schedule`.

    Returns:
        learning rate schedule

    See also:
        :func:`training_parameter_schedule`
    '''
    return training_parameter_schedule(lr, unit, epoch_size)


@typemap
def momentum_schedule(momentum, epoch_size=None, minibatch_size = None):
    '''
    Create a momentum schedule (using the same semantics as
    :func:`learning_parameter_schedule`) which applies the momentum 
    decay every N samples where N is specified by the argument `minibatch_size`.

    Args:
        momentum (float or list): see parameter ``schedule`` in
         :func:`training_parameter_schedule`.
        epoch_size (int): see parameter ``epoch_size`` in
         :func:`training_parameter_schedule`.
        minibatch_size (int): an integer to specify the reference minibatch size; 
          CNTK will scale the momentum internally so as to simulate the momentum decay of the specified minibatch 
          size while the actual minibatch sizes of the fed data can vary. In this way, momentum values can be provided 
          in a minibatch-size agnostic way (equal decay per sample). If minibatch_size is `None` (default), the momentum
          is applied to the whole minibatch regardless of the actual minibatch sizes (not in a minibatch-size agnostic way).

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

    Returns:
        momentum schedule
    '''
    return learning_parameter_schedule(momentum, minibatch_size, epoch_size)


@typemap
def momentum_schedule_per_sample(momentum, epoch_size=None):
    '''
    Create a per-sample momentum schedule (using the same semantics as
    :func:`momentum_schedule` but specializing in per sample momentum schedule).

    Args:
        momentum (float or list): see parameter ``schedule`` in
         :func:`training_parameter_schedule`.
        epoch_size (int): see parameter ``epoch_size`` in
         :func:`momentum_schedule`.
    Returns:
        momentum schedule
    '''
    return momentum_schedule(momentum, minibatch_size=1, epoch_size=epoch_size)


@typemap
def momentum_as_time_constant_schedule(momentum, epoch_size=None):
    '''
    Create a momentum schedule in a minibatch-size agnostic way
    (using the same semantics as :func:`training_parameter_schedule`
    with `unit=UnitType.sample`).

    Deprecated:: 2.2
        This is for legacy API.
        In this legacy API,::
        
            #assume the desired minibatch size invariant constant momentum rate is: momentum_rate
            momentum_time_constant = -minibatch_size/np.log(momentum_rate)
            momentum = momentum_as_time_constant_schedule(momentum_time_constant)

        The equivalent code in the latest API, ::

            momentum = momentum_schedule(momentum_rate, minibatch_size = minibatch_size)


    Args:
        momentum (float or list): see parameter ``schedule`` in
         :func:`training_parameter_schedule`.
        epoch_size (int): see parameter ``epoch_size`` in
         :func:`training_parameter_schedule`.

    CNTK specifies momentum in a minibatch-size agnostic way as the time
    constant (in samples) of a unit-gain 1st-order IIR filter. The value
    specifies the number of samples after which a gradient has an effect of
    1/e=37%.

    If you want to specify the momentum per N samples (or per minibatch),
    use :func:`momentum_schedule`.

    Examples:
        >>> # Use a fixed momentum of 1100 for all samples
        >>> m = momentum_as_time_constant_schedule(1100)

        >>> # Use the time constant 1100 for the first 1000 samples,
        >>> # then 1500 for the remaining ones
        >>> m = momentum_as_time_constant_schedule([1100, 1500], 1000)

    Returns:
        momentum as time constant schedule
    '''
    if isinstance(momentum, (cntk_py.training_double_parameter_schedule)):
        #the legacy momentum as time constant schedule: the ref minibatch size is always 1, so it is specified by definition
        momentum.is_minibatch_size_explicitly_specified = True
        return momentum

    if isinstance(momentum, (int, float)):
        if epoch_size is not None:
            warnings.warn('When providing the schedule as a number, epoch_size is ignored', RuntimeWarning)
        momentum = cntk_py.momentum_as_time_constant_schedule(momentum)
        momentum.is_minibatch_size_explicitly_specified = True
        return momentum

    epoch_size = epoch_size if epoch_size is not None else cntk_py.training_double_parameter_schedule.full_data_sweep
    if isinstance(momentum, list):
        momentum = _prepare_training_parameter_list(momentum)
        args = [momentum, epoch_size, 1] #momentum constant schedule's reference minibatch size is always per sample
        momentum = cntk_py.training_double_parameter_schedule(*args)
        momentum = cntk_py.momentum_as_time_constant_schedule(momentum)
        momentum.is_minibatch_size_explicitly_specified = True
        return momentum

    raise ValueError(
        'momentum must be either a float or a list, not %s' % type(momentum))

# TODO figure out how to pass infty to C++ in a portable way


def _infer_ref_minibatch_size_from_legacy_use_mean_gradient(ref_minibatch_size, use_mean_gradient):
    if (ref_minibatch_size, use_mean_gradient) == (None, None):
        #if ref_minibatch_size and the legacy use_mean_gradient are neither specified
        return None
    if ref_minibatch_size is not None:
        if use_mean_gradient == True and ref_minibatch_size != cntk_py.Learner.ignored_minibatch_size:
            Warning(
                'Learner reference minibatch size is specified while use_mean_gradient (depreated option) is specified to True. Learner reference minibatch size will override the mean gradient behavior')
        #if the ref_minibatch_size is specified, it overrides the legacay use_mean_gradient specification
        return ref_minibatch_size
    elif use_mean_gradient is not None:
        #if the ref_minibatch_size is NOT specified, the legacay use_mean_gradient specification take in the effect
        return cntk_py.Learner.ignored_minibatch_size if use_mean_gradient is True else None
    return None

def _infer_learning_parameter_schedule(number_or_schedule, ref_minibatch_size, epoch_size, use_mean_gradient=None):
    #the input is a number, create a new training parameter
    if isinstance(number_or_schedule, (int, float)) or \
            (isinstance(number_or_schedule, list) and all(isinstance(r, (int, float, tuple)) for r in number_or_schedule)):
        #default is per minibatch if the reference minibatch size is not specified.
        ref_minibatch_size = 0 if ref_minibatch_size is None else ref_minibatch_size
        schedule = learning_parameter_schedule(number_or_schedule, ref_minibatch_size, epoch_size)
        schedule.is_minibatch_size_explicitly_specified = ref_minibatch_size is not None
        return schedule
    elif isinstance(number_or_schedule,
                      cntk_py.training_double_parameter_schedule):
        if not number_or_schedule.is_minibatch_size_explicitly_specified and ref_minibatch_size is not None:
            #If the schedule minibatch size is not explicitly specified, the learner's specification will take over
            number_or_schedule.minibatch_size = ref_minibatch_size
        #for backward compatibility: use_mean_gradient = True and lr.unit = UnitType.sample
        #this combination was there to avoid the double-scaling of gradients when the gradients are already mean gradients
        if use_mean_gradient and number_or_schedule.minibatch_size == 1:
            #override the learning rate's minibatch_size to IGNORE
            number_or_schedule.minibatch_size = IGNORE
            Warning('use_mean_gradient=True and learning_rate_schedule.unit=UnitType.sample is a deprecated combination. '
                    'Please use the new learner APIs: see https://www.cntk.ai/pythondocs/cntk.learners.html for details.')
        return number_or_schedule
    else:
        raise ValueError('training parameter schedule type (%s) not supported. '
                         'training parameter schedule must be a training schedule '
                         % type(number_or_schedule))


def _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, ref_minibatch_size, schedule, epoch_size):
    #if non-None reference_minibatch_size will take precedence otherwise according use_mean_gradient if it is True
    ref_minibatch_size = _infer_ref_minibatch_size_from_legacy_use_mean_gradient(ref_minibatch_size, use_mean_gradient)
    #if minibatch_size is not None, any schedules that are with unspecified reference minibatch size will be overrided.
    schedule = _infer_learning_parameter_schedule(schedule, ref_minibatch_size, epoch_size, use_mean_gradient)
    _verify_learning_rate_type(schedule)
    return schedule, ref_minibatch_size

@typemap
def sgd(parameters, lr,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
        gradient_clipping_with_truncation=True, use_mean_gradient=None,
        minibatch_size=None, epoch_size=None):
    '''sgd(parameters, lr, l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates an SGD learner instance to learn the parameters. See [1] for more
    information on how to set the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' method of the root
         operator.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate. See also:  :func:`learning_parameter_schedule`


    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    See also:
        [1] L. Bottou. `Stochastic Gradient Descent Tricks
        <https://www.microsoft.com/en-us/research/publication/stochastic-gradient-tricks>`_. Neural
        Networks: Tricks of the Trade: Springer, 2012.
    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.sgd_learner(parameters, lr, additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def momentum_sgd(parameters, lr, momentum, unit_gain=default_unit_gain_value(),
                 l1_regularization_weight=0.0, l2_regularization_weight=0.0,
                 gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
                 gradient_clipping_with_truncation=True, use_mean_gradient=None,
                 minibatch_size=None, epoch_size=None):
    '''momentum_sgd(parameters, lr, momentum, unit_gain=default_unit_gain_value(), l1_regularization_weight=0.0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates a Momentum SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        momentum (float, list, output of :func:`momentum_schedule`): momentum schedule.
         For additional information, please refer to the :cntkwiki:`this CNTK Wiki article <BrainScript-SGD-Block#converting-learning-rate-and-momentum-parameters-from-other-toolkits>`.
        unit_gain: when ``True``, momentum is interpreted as a unit-gain filter. Defaults
         to the value returned by :func:`default_unit_gain_value`.
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate and momentum. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`
    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)
    momentum = _infer_learning_parameter_schedule(momentum, minibatch_size, epoch_size)
    _verify_momentum_type(momentum)
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.momentum_sgd_learner(parameters, lr, momentum, unit_gain,
                                        additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def nesterov(parameters, lr, momentum, unit_gain=default_unit_gain_value(),
             l1_regularization_weight=0.0, l2_regularization_weight=0.0,
             gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
             gradient_clipping_with_truncation=True, use_mean_gradient=None,
             minibatch_size=None, epoch_size=None):
    '''nesterov(parameters, lr, momentum, unit_gain=default_unit_gain_value(), l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates a Nesterov SGD learner instance to learn the parameters. This was
    originally proposed by Nesterov [1] in 1983 and then shown to work well in
    a deep learning context by Sutskever, et al. [2].

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        momentum (float, list, output of :func:`momentum_schedule`): momentum schedule.
         For additional information, please refer to the :cntkwiki:`this CNTK Wiki article <BrainScript-SGD-Block#converting-learning-rate-and-momentum-parameters-from-other-toolkits>`.
        unit_gain: when ``True``, momentum is interpreted as a unit-gain filter. Defaults
         to the value returned by :func:`default_unit_gain_value`.
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate and momentum. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    See also:
        [1] Y. Nesterov. A Method of Solving a Convex Programming Problem with Convergence Rate O(1/ sqrt(k)). Soviet Mathematics Doklady, 1983.

        [2] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. `On the
        Importance of Initialization and Momentum in Deep Learning
        <http://www.cs.toronto.edu/~fritz/absps/momentum.pdf>`_.  Proceedings
        of the 30th International Conference on Machine Learning, 2013.

    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)
    momentum = _infer_learning_parameter_schedule(momentum, minibatch_size, epoch_size)
    _verify_momentum_type(momentum)
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt=cntk_py.nesterov_learner(parameters, lr, momentum, unit_gain,
                                    additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt

@typemap
def adadelta(parameters, lr=learning_parameter_schedule_per_sample(1), rho=0.95, epsilon=1e-8,
             l1_regularization_weight=0.0, l2_regularization_weight=0.0,
             gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
             gradient_clipping_with_truncation=True, use_mean_gradient=None,
             minibatch_size=None, epoch_size=None):
    '''adadelta(parameters, lr, rho, epsilon, l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates an AdaDelta learner instance to learn the parameters. See [1] for
    more information.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        rho (float): exponential smooth factor for each minibatch.
        epsilon (float): epsilon for sqrt.
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    See also
        [1]  Matthew D. Zeiler, `ADADELTA: An Adaptive Learning Rate Method
        <https://arxiv.org/pdf/1212.5701.pdf>`_.
    '''
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    minibatch_size = _infer_ref_minibatch_size_from_legacy_use_mean_gradient(minibatch_size, use_mean_gradient)
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.ada_delta_learner(parameters, lr, rho, epsilon,
                                    additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def adagrad(parameters, lr, need_ave_multiplier=True,
            l1_regularization_weight=0.0, l2_regularization_weight=0.0,
            gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
            gradient_clipping_with_truncation=True, use_mean_gradient=None,
            minibatch_size=None, epoch_size=None):
    '''adagrad(parameters, lr, need_ave_multiplier=True, l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates an AdaGrad learner instance to learn the parameters. See [1] for
    more information.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        need_ave_multiplier (bool, default):
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    See also:
        [1]  J. Duchi, E. Hazan, and Y. Singer. `Adaptive Subgradient Methods
        for Online Learning and Stochastic Optimization
        <http://www.magicbroom.info/Papers/DuchiHaSi10.pdf>`_. The Journal of
        Machine Learning Research, 2011.
    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    minibatch_size = _infer_ref_minibatch_size_from_legacy_use_mean_gradient(minibatch_size, use_mean_gradient)
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.ada_grad_learner(parameters, lr, need_ave_multiplier,
                                    additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def fsadagrad(parameters, lr, momentum, unit_gain=default_unit_gain_value(),
              variance_momentum=momentum_schedule_per_sample(0.9999986111120757),
              l1_regularization_weight=0.0, l2_regularization_weight=0.0,
              gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
              gradient_clipping_with_truncation=True, use_mean_gradient=None,
              minibatch_size=None, epoch_size=None):
    '''fsadagrad(parameters, lr, momentum, unit_gain=default_unit_gain_value(), variance_momentum=momentum_schedule_per_sample(0.9999986111120757), l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates an FSAdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        momentum (float, list, output of :func:`momentum_schedule`): momentum schedule.
         For additional information, please refer to the :cntkwiki:`this CNTK Wiki article <BrainScript-SGD-Block#converting-learning-rate-and-momentum-parameters-from-other-toolkits>`.
        unit_gain: when ``True``, momentum is interpreted as a unit-gain filter. Defaults
         to the value returned by :func:`default_unit_gain_value`.
        variance_momentum (float, list, output of :func:`momentum_schedule`): variance momentum schedule. Defaults
         to ``momentum_schedule_per_sample(0.9999986111120757)``.
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate, momentum and variance_momentum. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)

    momentum = _infer_learning_parameter_schedule(momentum, minibatch_size, epoch_size)
    _verify_momentum_type(momentum)
    variance_momentum = _infer_learning_parameter_schedule(variance_momentum, minibatch_size, epoch_size)
    _verify_momentum_type(variance_momentum)
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    minibatch_size = _infer_ref_minibatch_size_from_legacy_use_mean_gradient(minibatch_size, use_mean_gradient)
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.fsada_grad_learner(parameters, lr, momentum, unit_gain,
                                      variance_momentum, additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def adam(parameters, lr, momentum, unit_gain=default_unit_gain_value(),
         variance_momentum=momentum_schedule_per_sample(0.9999986111120757),
         l1_regularization_weight=0.0, l2_regularization_weight=0.0,
         gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
         gradient_clipping_with_truncation=True, use_mean_gradient=None, epsilon=1e-8, adamax=False,
         minibatch_size=None, epoch_size=None):
    '''adam(parameters, lr, momentum, unit_gain=default_unit_gain_value(), variance_momentum=momentum_schedule_per_sample(0.9999986111120757), l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True, epsilon=1e-8, adamax=False)
    Creates an Adam learner instance to learn the parameters. See [1] for more
    information.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        momentum (float, list, output of :func:`momentum_schedule`): momentum schedule. Note that this is the beta1 parameter in the Adam paper [1]. 
         For additional information, please refer to the :cntkwiki:`this CNTK Wiki article <BrainScript-SGD-Block#converting-learning-rate-and-momentum-parameters-from-other-toolkits>`.
        unit_gain: when ``True``, momentum is interpreted as a unit-gain filter. Defaults
         to the value returned by :func:`default_unit_gain_value`.
        variance_momentum (float, list, output of :func:`momentum_schedule`): variance momentum schedule. 
         Note that this is the beta2 parameter in the Adam paper [1]. Defaults to ``momentum_schedule_per_sample(0.9999986111120757)``. 
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        epsilon (float, optional): numerical stability constant,
         defaults to 1e-8
        adamax: when ``True``, use infinity-norm variance momentum update instead of L2. Defaults
         to False
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate, momentum and variance_momentum. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    See also:
        [1] D. Kingma, J. Ba. `Adam: A Method for Stochastic Optimization
        <https://arxiv.org/abs/1412.6980>`_. International Conference for
        Learning Representations, 2015.
    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)

    momentum = _infer_learning_parameter_schedule(momentum, minibatch_size, epoch_size)
    _verify_momentum_type(momentum)
    variance_momentum = _infer_learning_parameter_schedule(variance_momentum, minibatch_size, epoch_size)
    _verify_momentum_type(variance_momentum)
    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.adam_learner(parameters, lr, momentum, unit_gain,
                                variance_momentum, epsilon, adamax, additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def rmsprop(parameters, lr,
            gamma, inc, dec, max, min,
            need_ave_multiplier=True,
            l1_regularization_weight=0.0, l2_regularization_weight=0.0,
            gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=np.inf,
            gradient_clipping_with_truncation=True, use_mean_gradient=None,
            minibatch_size=None, epoch_size=None):
    '''rmsprop(parameters, lr, gamma, inc, dec, max, min, need_ave_multiplier=True, l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True)
    Creates an RMSProp learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr (float, list, output of :func:`learning_parameter_schedule`): a learning rate in float, or a learning rate schedule.
         See also:  :func:`learning_parameter_schedule`
        gamma (float): Trade-off factor for current and previous gradients. Common value is 0.95. Should be in range (0.0, 1.0)
        inc (float): Increasing factor when trying to adjust current learning_rate. Should be greater than 1
        dec (float): Decreasing factor when trying to adjust current learning_rate. Should be in range (0.0, 1.0)
        max (float): Maximum scale allowed for the initial learning_rate. Should be greater than zero and min
        min (float): Minimum scale allowed for the initial learning_rate. Should be greater than zero
        need_ave_multiplier (bool, default ``True``):
        l1_regularization_weight (float, optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight (float, optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev (float, optional): the standard deviation
         of the Gaussian noise added to parameters post update, defaults to 0.0
        gradient_clipping_threshold_per_sample (float, optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation (bool, default ``True``): use gradient clipping
         with truncation
        use_mean_gradient (bool, optional): use averaged gradient as input to learner.

            deprecated:: 2.2
                Use minibatch_size parameter to specify the reference minibatch size.
        minibatch_size (int, default ``None``): The minibatch size that the learner's parameters are designed or pre-tuned for. This
         size is usually set to the same as the minibatch data source's size. CNTK will perform automatic scaling of the parameters
         to enable efficient model parameter update implementation while approximate the behavior of pre-designed and pre-tuned parameters.
         In case that minibatch_size is not specified, CNTK will inherit the minibatch size from the learning rate schedule;
         if the learning rate schedule does not specify the minibatch_size, CNTK will set it to :attr:`IGNORE`. Setting minibatch_size to :attr:`IGNORE`
         will have the learner apply as it is preventing CNTK performing any hyper-parameter scaling. See also:  :func:`learning_parameter_schedule`
        epoch_size (optional, int): number of samples as a scheduling unit for learning rate. See also:  :func:`learning_parameter_schedule`

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`
    '''
    lr, minibatch_size = _infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr, epoch_size)

    gaussian_noise_injection_std_dev = \
        training_parameter_schedule(
            gaussian_noise_injection_std_dev)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
    minibatch_size = _infer_ref_minibatch_size_from_legacy_use_mean_gradient(minibatch_size, use_mean_gradient)
    if minibatch_size is not None:
        additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(minibatch_size) #need this to make proper typed DictionaryValue

    opt = cntk_py.rmsprop_learner(parameters, lr, gamma, inc, dec, max, min,
                                   need_ave_multiplier, additional_options)
    opt.is_minibatch_size_explicitly_specified = minibatch_size is not None
    return opt


@typemap
def universal(update_func, parameters):
    '''
    Creates a learner which uses a CNTK function to update the parameters.

    Args:
        update_func: function that takes parameters and gradients as arguments and
         returns a :class:`~cntk.ops.functions.Function` that performs the
         desired updates. The returned function updates the parameters by
         means of containing :func:`~cntk.ops.assign` operations.
         If ``update_func`` does not contain :func:`~cntk.ops.assign` operations
         the parameters will not be updated.
        parameters (list): list of network parameters to tune.
         These can be obtained by the root operator's `parameters`.

    Returns:
        :class:`~cntk.learners.Learner`: learner instance that can be passed to
        the :class:`~cntk.train.trainer.Trainer`

    Examples:
        >>> def my_adagrad(parameters, gradients):
        ...     accumulators = [C.constant(0, shape=p.shape, dtype=p.dtype, name='accum') for p in parameters]
        ...     update_funcs = []
        ...     for p, g, a in zip(parameters, gradients, accumulators):
        ...         accum_new = C.assign(a, g * g)
        ...         update_funcs.append(C.assign(p, p - 0.01 * g / C.sqrt(accum_new + 1e-6)))
        ...     return C.combine(update_funcs)
        ...
        >>> x = C.input_variable((10,))
        >>> y = C.input_variable((2,))
        >>> z = C.layers.Sequential([C.layers.Dense(100, activation=C.relu), C.layers.Dense(2)])(x)
        >>> loss = C.cross_entropy_with_softmax(z, y)
        >>> learner = C.universal(my_adagrad, z.parameters)
        >>> trainer = C.Trainer(z, loss, learner)
        >>> # now trainer can be used as any other Trainer

    '''

    from .. import constant
    args, _ = utils.get_python_function_arguments(update_func)
    if len(args) != 2:
        raise ValueError('update_func must be a function that accepts two arguments (parameters, gradients)')
    gradients = []
    for p in parameters:
        if any(dim<0 for dim in p.shape):
            raise ValueError('parameter %s has inferred dimensions. Please create the learner after all parameter shapes have been determined'%str(p))
        gradients.append(constant(0, shape=p.shape, dtype=p.dtype, name='grad'))
    #TODO: add additional options and learning context to the parameters of the updat_func so that the update function
    #      can make use of the context and additional options
    result = update_func(parameters, gradients)

    return cntk_py.universal_learner(parameters, gradients, result)
