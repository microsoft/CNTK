# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .cntk_py import default_param_init_scale as DefaultParamInitScale,\
        default_param_init_output_rank as DefaultParamInitOutputRank,\
        default_param_init_filter_rank as DefaultParamInitFilterRank,\
        default_random_seed as DefaultRandomSeed


def uniform(scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    Uniform initializer

    Args:
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.uniform_initializer(scale, seed)

def gaussian(output_rank=DefaultParamInitOutputRank, filter_rank=DefaultParamInitFilterRank, scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    Gaussian initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.gaussian_initializer(output_rank, filter_rank, scale, seed)

def xavier(output_rank=DefaultParamInitOutputRank, filter_rank=DefaultParamInitFilterRank, scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    Xavier initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.xavier_initializer(output_rank, filter_rank, scale, seed)

def glorot_uniform(output_rank=DefaultParamInitOutputRank, filter_rank=DefaultParamInitFilterRank, scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    Glorot initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.glorot_uniform_initializer(output_rank, filter_rank, scale, seed)

def glorot_normal(output_rank=DefaultParamInitOutputRank, filter_rank=DefaultParamInitFilterRank, scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.glorot_normal_initializer(output_rank, filter_rank, scale, seed)

def he_uniform(output_rank=DefaultParamInitOutputRank, filter_rank=DefaultParamInitFilterRank, scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.he_uniform_initializer(output_rank, filter_rank, scale, seed)

def he_normal(output_rank=DefaultParamInitOutputRank, filter_rank=DefaultParamInitFilterRank, scale=DefaultParamInitScale, seed=DefaultRandomSeed):
    '''
    initializer

    Args:
        output_rank (`int`): output rank
        filter_rank (`int`): filter rank
        scale (`float`): scale
        seed (`int`): random seed

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.he_normal_initializer(output_rank, filter_rank, scale, seed)

def bilinear(kernel_width, kernel_height):
    '''
    initializer

    Args:
        kernel_width (`int`): kernel width
        kernel_height (`int`): kernel height

    Returns:
        initializer for `:class:cntk.variables.Parameter`
    '''
    return cntk_py.bilinear_initializer(kernel_width, kernel_height)
