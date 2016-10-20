# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
from . import cntk_py
from .utils import typemap

__doc__='''
Learner tunes a set of parameters during the training process. One can use
different learners for different sets of parameters. Currently, CNTK supports
the following learning algorithms:

+------------------------+
| Learning algorithms    |
+========================+
| AdaGrad                |
+------------------------+
| FSAdaGrad              |
| (a variant of Adam)    |
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
            learning_rate (`float`): learning rate to reset to
        '''
        return super(Learner, self).reset_learning_rate()

    @property
    def learning_rate(self):
        '''
        The learning rate.
        '''
        return super(Learner, self).learning_rate()

@typemap
def learning_rate_schedule(lr, units=1):
    '''
    Create a learning rate schedule.

    Examples:
        >>> # Use a fixed learning rate 0.01 for all samples
        >>> lr = learning_rate_schedule(0.01)
        >>> lr[0], lr[1]
        (0.01, 0.01)

        >>> # Use the learning rate 0.01 for the first 1000 samples, then 0.001 for the remaining ones
        >>> lr = learning_rate_schedule([0.01, 0.001], 1000)
        >>> lr[0], lr[1], lr[1000], lr[1001]
        (0.01, 0.01, 0.001, 0.001)

    Args:
        lr (`float` or `list`): if `float`, it is the learning rate to be used
         for all samples. In case of list, the elements are used as the
         learning rates for ``units`` samples.
        units (`int`): unit for the learning rates to have effect

    Returns:
        schedule for learning rates per sample
    '''
    if isinstance(lr, float):
        return cntk_py.learning_rates_per_sample(lr)

    if not isinstance(lr, list):
        raise ValueError('lr must be either a float or a list, not %s'%type(lr))

    return cntk_py.learning_rates_per_sample(lr, units)

@typemap
def momentum_schedule(momentum, units=1):
    '''
    Create a momentum schedule in a minibatch agnostic way.

    CNTK specifies momentum in a minibatch-size agnostic way as the time
    constant (in samples) of a unit-gain 1st-order IIR filter. The value
    specifies the number of samples after which a gradient has an effect of
    1/e=37%.

    If you want to specify the momentum per sample,
    use :func:`momentums_per_sample`.


    Examples:
        >>> # Use a fixed momentum of 1100 for all samples
        >>> m = momentum_schedule(1100)

        >>> # Use the time constant 1100 for the first 1000 samples, then 1500 for the remaining ones
        >>> m = momentum_schedule([1100, 1500], 1000)

    Args:
        momentum (`float` or `list`): if `float`, it is the momentum to be used
         for all samples. In case of list, the elements are used as the
         momentum for ``units`` samples.
        units (`int`): unit for the momentum to have effect

    Returns:
        momentum schedule
    '''
    # FIXME: Swig does not see MomentumValuesAsTimeConstants as an inherited
    # type of MomentumValuesPerSample, so we are doing the conversion
    # explicitly for now. This has to be solved in the Swig layer eventually.
    to_per_sample = lambda x: 0 if x==0 else math.exp(-1.0 / x)

    if isinstance(momentum, (int, float)):
        return cntk_py.momentums_per_sample(to_per_sample(momentum))


    if not isinstance(momentum, list):
        raise ValueError('momentum must be either a float or a list, not %s'%type(momentum))

    return cntk_py.momentums_per_sample([to_per_sample(m) for m in momentum], units)

@typemap
def momentum_schedule_per_sample(momentum, units=1):
    '''
    Create a momentum schedule where the momentum is specified on a per-sample
    basis.

    If you want to provide momentum values in a sample/minibatch
    agnostic way, use :func:`momentum_schedule`.

    Examples:
        >>> # Use a fixed momentum of 0.99 for all samples
        >>> m = momentum_schedule_per_sample(0.99)

        >>> # Use the learning rate 0.99 for the first 1000 samples, then 0.9 for the remaining ones
        >>> m = momentum_schedule_per_sample([0.99,0.9], 1000)

    Args:
        momentum (`float` or `list`): if `float`, it is the momentum to be used
         for all samples. In case of list, the elements are used as the
         momentum for ``units`` samples.
        units (`int`): unit for the momentum to have effect

    Returns:
        schedule for momentum per sample
    '''
    if isinstance(momentum, (int, float)):
        return cntk_py.momentums_per_sample(momentum)

    if not isinstance(momentum, list):
        raise ValueError('momentum must be either a float or a list, not %s'%type(momentum))

    return cntk_py.momentums_per_sample(momentum, units)


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
        lr ('float' or output of :func:`learning_rate_schedule`): learning
         rates per sample.
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
    if isinstance(lr, (int, float)):
        lr = learning_rate_schedule(lr)

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
        lr ('float' or output of `:func:learning_rate_schedule`): learning
         rates per sample.
        momentum (`float` or output of `:func:momentum_schedule`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
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
    if isinstance(lr, (int, float)):
        lr = learning_rate_schedule(lr)

    if isinstance(momentum, (int, float)):
        momentum = momentum_schedule(momentum)

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
        lr ('float' or output of `:func:learning_rate_schedule`): learning
         rates per sample.
        momentum (`float` or output of `:func:momentum_schedule`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
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
    if isinstance(lr, (int, float)):
        lr = learning_rate_schedule(lr)

    if isinstance(momentum, (int, float)):
        momentum = momentum_schedule(momentum)

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
        lr ('float' or output of `:func:learning_rate_schedule`): learning
         rates per sample.
         allowed, but schedules will be added soon
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
    if isinstance(lr, (int, float)):
        lr = learning_rate_schedule(lr)

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
def fsadagrad(parameters, lr, momentum,
        targetAdagradAvDenom = 0.0025, varianceTimeConstant = 720000,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0,
        gaussian_noise_injection_std_dev=0.0, gradient_clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an FS AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the root operator's ``parameters``.
        lr ('float' or output of `:func:learning_rate_schedule`): learning
         rates per sample.
        momentum (`float` or output of `:func:momentum_schedule`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        targetAdagradAvDenom ('float', optional): FSAdaGrad magic number,
         defaults to 0.0025 (1/400)
        varianceTimeConstant ('long', optional): FSAdaGrad magic number,
         defaults to 720000 ( 2 * 3600 * 100)
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
    if isinstance(lr, (int, float)):
        lr = learning_rate_schedule(lr)

    if isinstance(momentum, (int, float)):
        momentum = momentum_schedule(momentum)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.fsada_grad_learner(parameters, lr, momentum,
            targetAdagradAvDenom, varianceTimeConstant,
            additional_options)

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
        lr ('float'): learning rate per sample. Currently, only float is
         allowed, but schedules will be added soon
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
    if isinstance(lr, (int, float)):
        lr = learning_rate_schedule(lr)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.rmsprop_learner(parameters, lr, gamma, inc, dec, max, min,
            need_ave_multiplier, additional_options)

