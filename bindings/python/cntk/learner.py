# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py

# TODO figure out how to pass infty to C++ in a portable way
def sgd_learner(parameters, lr, 
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float'): learning rate per sample. Currently, only float is
         allowed, but schedules will be added soon
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be pased to the `Trainer`
    '''
    if type(lr) == float:
        lr = cntk_py.learning_rates_per_sample(lr)

    return cntk_py.sgd_learner(parameters, lr, clipping_threshold_per_sample,
            gradient_clipping_with_truncation)

def momentum_sgd_learner(parameters, lr, momentums,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates a Momemtum SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float'): learning rate per sample. Currently, only float is
         allowed, but schedules will be added soon.
        momentums (instance of `MomentumsPerSample`): momentums per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be pased to the `Trainer`
    '''
    if type(lr) == float:
        lr = cntk_py.learning_rates_per_sample(lr)

    return cntk_py.momentum_sgd_learner(parameters, lr, momentums,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)

def nesterov_learner(parameters, lr, momentums,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates a Nesterov SGD learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float'): learning rate per sample. Currently, only float is
         allowed, but schedules will be added soon
        momentums (instance of `MomentumsPerSample`): momentums per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be pased to the `Trainer`
    '''
    if type(lr) == float:
        lr = cntk_py.learning_rates_per_sample(lr)

    return cntk_py.nesterov_learner(parameters, lr, momentums,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)


def adagrad_learner(parameters, lr, 
        need_ave_multiplier=True,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float'): learning rate per sample. Currently, only float is
         allowed, but schedules will be added soon
        need_ave_multiplier ('bool', default): 
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be pased to the `Trainer`
    '''
    if type(lr) == float:
        lr = cntk_py.learning_rates_per_sample(lr)

    return cntk_py.ada_grad_learner(parameters, lr, need_ave_multiplier,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)

def fsadagrad_learner(parameters, lr, momentums,
        clipping_threshold_per_sample=1E10,
        gradient_clipping_with_truncation=True):
    '''
    Creates an FS AdaGrad learner instance to learn the parameters.

    Args:
        parameters (list of parameters): list of network parameters to tune.
         These can be obtained by the '.parameters()' function of 
        lr ('float'): learning rate per sample. Currently, only float is
         allowed, but schedules will be added soon
        momentums (instance of `MomentumsPerSample`): momentums per sample.
         Refer to https://github.com/Microsoft/CNTK/wiki/SGD-block#converting-learning-rate-and-momentum-parameters-from-other-toolkits
        clipping threshold per sample ('float', optional): clipping threshold
         per sample, defaults to infinity
        gradient_clipping_with_truncation ('bool', optional): defaults to True

    Returns:
        Instance of a learner that can be pased to the `Trainer`
    '''
    if type(lr) == float:
        lr = cntk_py.learning_rates_per_sample(lr)

    return cntk_py.fsada_grad_learner(parameters, lr, momentums,
            clipping_threshold_per_sample, gradient_clipping_with_truncation)

def rmsprop_learner(parameters, lr, 
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
        Instance of a learner that can be pased to the `Trainer`
    '''
    if type(lr) == float:
        lr = cntk_py.learning_rates_per_sample(lr)

    return cntk_py.rmsprop_learner(parameters, lr, gamma, inc, dec, max, min,
            need_ave_multiplier, clipping_threshold_per_sample, gradient_clipping_with_truncation)

