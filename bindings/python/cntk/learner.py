# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .utils import typemap

class Learner(cntk_py.Learner):
    '''
    Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    For e.g momentum, AdaGrad, RMSProp etc. are different types of learners with their own algorithms for
    learning parameter values using first order gradients.
    To instantiate a concreate learner, use the factory methods in this module.
    '''
        
    def update(gradient_values, training_sample_count):
        '''
        Update the parameters associated with this learner. 

        Args:
            gradient_values (`dict`): maps `:class:cntk.variables.Parameter` to a NumPy array
            training_sample_count (`int`): training sample count

        Returns:
            `False` to indicate that learning has stopped for all of the parameters associated with this learner
        '''
        return super(Learner, self).update(gradient_values, training_sample_count)

    @typemap
    def parameters(self):
        '''
        Returns:
            the set of parameters associated with this learner.
        '''
        return super(Learner, self).parameters()


    def reset_learning_rate(self, learning_rate):
        '''
        Resets the learning rate.

        Args:
            learning_rate (`float`): learning rate to reset to
        '''
        return super(Learner, self).reset_learning_rate()

    def learning_rate(self):
        '''
        Returns the learning rate.
        '''
        return super(Learner, self).learning_rate()

@typemap
def learning_rates_per_sample(lr, units=1):
    '''
    Create a learning rate schedule.

    Examples:
        >>> # Use the learning rate 0.7 for all samples
        >>> lr = learning_rates_per_sample(0.7)
        >>> [lr[i] for i in [0,1,2,3]]
        [0.7, 0.7, 0.7, 0.7]

        >>> # Use the learning rate 0.7 for the first 3 samples, then 0.3 for the remaining ones
        >>> lr = learning_rates_per_sample([0.7,0.3], 3)
        >>> [lr[i] for i in range(10)]
        [0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    Args:
        lr (`float` or `list`): if `float`, it is the learning rate to be used
         for all samples. In case of list, the elements are used as the
         learning rates for `units` samples.
        units (`int`): unit for the learning rates to have effect

    Returns:
        schedule for learning rates per sample
    '''
    if isinstance(lr, float):
        return cntk_py.learning_rates_per_sample(lr)
    
    if not isinstance(lr, list):
        raise ValueError('lr must be either a float or a list')

    return cntk_py.learning_rates_per_sample(lr, units)

@typemap
def momentums_per_sample(momentums, units=1):
    '''
    Create a momentums schedule.

    Examples:
        >>> # Use the learning rate 0.7 for all samples
        >>> lr = momentums_per_sample(0.7)
        >>> [lr[i] for i in [0,1,2,3]]
        [0.7, 0.7, 0.7, 0.7]

        >>> # Use the learning rate 0.7 for the first 3 samples, then 0.3 for the remaining ones
        >>> lr = momentums_per_sample([0.7,0.3], 3)
        >>> [lr[i] for i in range(10)]
        [0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    Args:
        momentums (`float` or `list`): if `float`, it is the momentum to be used
         for all samples. In case of list, the elements are used as the
         momentums for `units` samples.
        units (`int`): unit for the momentums to have effect

    Returns:
        schedule for momentums per sample
    '''
    if isinstance(momentums, float):
        return cntk_py.momentums_per_sample(momentums)
    
    if not isinstance(momentums, list):
        raise ValueError('momentum must be either a float or a list')

    return cntk_py.momentums_per_sample(momentums, units)


# TODO figure out how to pass infty to C++ in a portable way
@typemap
def sgd(parameters, lr, 
        l1_regularization_weight=0.0, l2_regularization_weight=0.0, 
        gaussian_noise_injection_std_dev=0.0, clipping_threshold_per_sample=1E10, 
        gradient_clipping_with_truncation=True):
    '''
    Creates an SGD learner instance to learn the parameters.

    Args:
        parameters (`list` of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' method of the root
         operator.
        lr ('float' or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation 
         of the Gaussian noise added to parameters post update, defaults to 0.0
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a `:class:cntk.learner.Learner` that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)
        
    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.clipping_threshold_per_sample = clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.sgd_learner(parameters, lr, additional_options)

@typemap
def momentum_sgd(parameters, lr, momentums, 
        l1_regularization_weight=0.0, l2_regularization_weight=0.0, 
        gaussian_noise_injection_std_dev=0.0, clipping_threshold_per_sample=1E10, 
        gradient_clipping_with_truncation=True):
    '''
    Creates a Momemtum SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        momentums (`float` or output of `:func:momentums_per_sample`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation 
         of the Gaussian noise added to parameters post update, defaults to 0.0
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a `:class:cntk.learner.Learner` that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    if type(momentums) == float:
        momentums = momentums_per_sample(momentums)
    
    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.clipping_threshold_per_sample = clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.momentum_sgd_learner(parameters, lr, momentums,
            additional_options)

@typemap
def nesterov(parameters, lr, momentums, 
        l1_regularization_weight=0.0, l2_regularization_weight=0.0, 
        gaussian_noise_injection_std_dev=0.0, clipping_threshold_per_sample=1E10, 
        gradient_clipping_with_truncation=True):
    '''
    Creates a Nesterov SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        momentums (`float` or output of `:func:momentums_per_sample`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation 
         of the Gaussian noise added to parameters post update, defaults to 0.0
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a `:class:cntk.learner.Learner` that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    if type(momentums) == float:
        momentums = momentums_per_sample(momentums)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.clipping_threshold_per_sample = clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
        
    return cntk_py.nesterov_learner(parameters, lr, momentums,
            additional_options)

@typemap
def adagrad(parameters, lr, need_ave_multiplier=True, 
        l1_regularization_weight=0.0, l2_regularization_weight=0.0, 
        gaussian_noise_injection_std_dev=0.0, clipping_threshold_per_sample=1E10, 
        gradient_clipping_with_truncation=True):
    '''
    Creates an AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
         allowed, but schedules will be added soon
        need_ave_multiplier ('bool', default): 
        l1_regularization_weight ('float', optional): the L1 regularization weight per sample,
         defaults to 0.0
        l2_regularization_weight ('float', optional): the L2 regularization weight per sample,
         defaults to 0.0
        gaussian_noise_injection_std_dev ('float', optional): the standard deviation 
         of the Gaussian noise added to parameters post update, defaults to 0.0
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a `:class:cntk.learner.Learner` that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.clipping_threshold_per_sample = clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation
        
    return cntk_py.ada_grad_learner(parameters, lr, need_ave_multiplier,
            additional_options)

@typemap
def fsadagrad(parameters, lr, momentums,
        targetAdagradAvDenom = 0.0025, varianceTimeConstant = 720000,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0, 
        gaussian_noise_injection_std_dev=0.0, clipping_threshold_per_sample=1E10, 
        gradient_clipping_with_truncation=True):
    '''
    Creates an FS AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        momentums (`float` or output of `:func:momentums_per_sample`): momentum values per sample.
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
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a `:class:cntk.learner.Learner` that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    if type(momentums) == float:
        momentums = momentums_per_sample(momentums)
        
    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.clipping_threshold_per_sample = clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.fsada_grad_learner(parameters, lr, momentums, 
            targetAdagradAvDenom, varianceTimeConstant,
            additional_options)

@typemap
def rmsprop(parameters, lr, 
        gamma, inc, dec, max, min,
        need_ave_multiplier=True,
        l1_regularization_weight=0.0, l2_regularization_weight=0.0, 
        gaussian_noise_injection_std_dev=0.0, clipping_threshold_per_sample=1E10, 
        gradient_clipping_with_truncation=True):
    '''
    Creates an RMSProp learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
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
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient clipping

    Returns:
        Instance of a `:class:cntk.learner.Learner` that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)
        
    additional_options = cntk_py.AdditionalLearningOptions()
    additional_options.l1_regularization_weight = l1_regularization_weight
    additional_options.l2_regularization_weight = l2_regularization_weight
    additional_options.gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev
    additional_options.clipping_threshold_per_sample = clipping_threshold_per_sample
    additional_options.gradient_clipping_with_truncation = gradient_clipping_with_truncation

    return cntk_py.rmsprop_learner(parameters, lr, gamma, inc, dec, max, min,
            need_ave_multiplier, additional_options)

