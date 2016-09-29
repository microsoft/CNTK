# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py

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


# TODO figure out how to pass infty to C++ in a portable way
def sgd(parameters, lr, 
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an SGD learner instance to learn the parameters.

    Args:
        parameters (`list` of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' method of the root
         operator.
        lr ('float' or list of `float`s or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', default `True`): gradient
         glipping

    Returns:
        Instance of a learner that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    return cntk_py.sgd_learner(parameters, lr, clipping_threshold_per_sample,
            gradient_clipping_with_truncation)

def momentum_sgd(parameters, lr, momentums,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates a Momemtum SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or list of `float`s or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        momentums (instance of `MomentumsPerSample`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    return cntk_py.momentum_sgd_learner(parameters, lr, momentums,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)

def nesterov(parameters, lr, momentums,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates a Nesterov SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or list of `float`s or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        momentums (instance of `MomentumsPerSample`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    return cntk_py.nesterov_learner(parameters, lr, momentums,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)


def adagrad(parameters, lr, 
        need_ave_multiplier=True,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or list of `float`s or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
         allowed, but schedules will be added soon
        need_ave_multiplier ('bool', default): 
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    return cntk_py.ada_grad_learner(parameters, lr, need_ave_multiplier,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)

def fsadagrad(parameters, lr, momentums,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an FS AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float' or list of `float`s or output of `:func:learning_rates_per_sample`): learning
         rates per sample.  
        momentums (instance of `MomentumsPerSample`): momentum values per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    return cntk_py.fsada_grad_learner(parameters, lr, momentums,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)

def rmsprop(parameters, lr, 
        gamma, inc, dec, max, min,
        need_ave_multiplier=True,
        clipping_threshold_per_sample=1E10,
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
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be passed to the `Trainer`
    '''
    if type(lr) == float:
        lr = learning_rates_per_sample(lr)

    return cntk_py.rmsprop_learner(parameters, lr, gamma, inc, dec, max, min,
            need_ave_multiplier, clipping_threshold_per_sample, gradient_clipping_with_truncation)

