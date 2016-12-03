# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .cntk_py import default_param_init_scale as DefaultParamInitScale,\
        sentinel_value_for_infer_param_init_rank as SentinelValueForInferParamInitRank,\
        sentinel_value_for_auto_select_random_seed as SentinelValueForAutoSelectRandomSeed


def uniform(scale=DefaultParamInitScale, seed=None):
    '''
    Uniform initializer

    Args:
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to uniform distribution between `scale*[-0.05, 0.05]`
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.uniform_initializer(scale, seed)

def gaussian(output_rank=SentinelValueForInferParamInitRank, filter_rank=SentinelValueForInferParamInitRank, scale=DefaultParamInitScale, seed=None):
    '''
    Gaussian initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to Gaussian distribution with mean `0` and standard deviation `scale*0.2/sqrt(fanIn))`. 
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.gaussian_initializer(output_rank, filter_rank, scale, seed)

def xavier(output_rank=SentinelValueForInferParamInitRank, filter_rank=SentinelValueForInferParamInitRank, scale=DefaultParamInitScale, seed=None):
    '''
    Xavier initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to Gaussian distribution with mean `0` and standard deviation `scale*sqrt(3.0/fanIn)`
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.xavier_initializer(output_rank, filter_rank, scale, seed)

def glorot_uniform(output_rank=SentinelValueForInferParamInitRank, filter_rank=SentinelValueForInferParamInitRank, scale=DefaultParamInitScale, seed=None):
    '''
    Glorot initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to uniform distribution between `scale*sqrt(6.0/(fanIn+fanOut))*[-1,1]`
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.glorot_uniform_initializer(output_rank, filter_rank, scale, seed)

def glorot_normal(output_rank=SentinelValueForInferParamInitRank, filter_rank=SentinelValueForInferParamInitRank, scale=DefaultParamInitScale, seed=None):
    '''
    initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to Gaussian distribution with mean `0` and standard deviation `scale*sqrt(2.0/(fanIn+fanOut))`
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.glorot_normal_initializer(output_rank, filter_rank, scale, seed)

def he_uniform(output_rank=SentinelValueForInferParamInitRank, filter_rank=SentinelValueForInferParamInitRank, scale=DefaultParamInitScale, seed=None):
    '''
    initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to uniform distribution between `scale*sqrt(6.0/fanIn)*[-1,1]`
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.he_uniform_initializer(output_rank, filter_rank, scale, seed)

def he_normal(output_rank=SentinelValueForInferParamInitRank, filter_rank=SentinelValueForInferParamInitRank, scale=DefaultParamInitScale, seed=None):
    '''
    initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        initialized to Gaussian distribution with mean `0` and standard deviation `scale*sqrt(2.0/fanIn)`
    '''
    if seed is None:
        seed = SentinelValueForAutoSelectRandomSeed

    return cntk_py.he_normal_initializer(output_rank, filter_rank, scale, seed)

def bilinear(kernel_width, kernel_height):
    '''
    initializer

    Args:
        kernel_width (`int`): kernel width
        kernel_height (`int`): kernel height

    Returns:
        initializer for :class:`cntk.variables.Parameter`
        useful for deconvolution layer
    '''
    return cntk_py.bilinear_initializer(kernel_width, kernel_height)

def initializer_with_rank(initializer, output_rank=None, filter_rank=None):
    '''
    override output_rank and filter_rank specification in a random initializer
    constructed without an explciti output_rank and filter_rank specification

    Args:'
        initializer: initializer whose output_rank and filter_rank parameters are to be overriden
        output_rank (`int`): new output rank value
        filter_rank (`int`): new filter rank value

    Returns:
        new initializer for `:class:cntk.variables.Parameter` with specified output_rank and filter_rank
    '''
    if output_rank is None:
        output_rank = SentinelValueForInferParamInitRank
    if filter_rank is None:
        filter_rank = SentinelValueForInferParamInitRank
    return cntk_py.random_initializer_with_rank(initializer, output_rank, filter_rank)
