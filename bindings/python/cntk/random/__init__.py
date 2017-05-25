# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
from __future__ import print_function
import numpy as np
from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as auto_select
from cntk.cntk_py import uniform_random, normal_random, gumbel_random, bernoulli_random
from cntk.default_options import default_override_or
from cntk.internal.swig_helper import typemap
from cntk.internal import sanitize_random_args, sanitize_input


@typemap
def uniform(shape, dtype=default_override_or(np.float32), low=0.0, high=1.0, seed=auto_select, name=''):
    """uniform(shape, dtype=default_override_or(np.float32), low=0.0, high=1.0, seed=auto_select, name='')
    Generates samples from the uniform distribution in the interval [`low`,`high`).

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        low (float): lower end of the range of the random numbers
        high (float): upper end of the range of the random numbers
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    shape, dtype = sanitize_random_args(shape, dtype)
    return uniform_random(shape, dtype, low, high, seed, name)


@typemap
def normal(shape, dtype=default_override_or(np.float32), mean=0.0, scale=1.0, seed=auto_select, name=''):
    """normal(shape, dtype=default_override_or(np.float32), mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the normal distribution with mean `mean` and standard deviation `scale`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        mean (float): mean of the distribution
        scale (float): scale (standard deviation) of the distribution
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    shape, dtype = sanitize_random_args(shape, dtype)
    return normal_random(shape, dtype, mean, scale, seed, name)


@typemap
def gumbel(shape, dtype=default_override_or(np.float32), loc=0.0, scale=1.0, seed=auto_select, name=''):
    """gumbel(shape, dtype=default_override_or(np.float32), loc=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the Gumbel distribution with location `loc` and scale `scale`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        loc (float): location of the distribution
        scale (float): scale of the distribution
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        `The Gumbel-Max Trick
        <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>`_.
    """
    shape, dtype = sanitize_random_args(shape, dtype)
    return gumbel_random(shape, dtype, loc, scale, seed, name)


@typemap
def bernoulli(shape, dtype=default_override_or(np.float32), mean=0.5, seed=auto_select, name=''):
    """bernoulli(shape, dtype=default_override_or(np.float32), mean=0.5, seed=auto_select, name='')
    Generates samples from the Bernoulli distribution with success probability `mean`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        mean (float): success probability
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    shape, dtype = sanitize_random_args(shape, dtype)
    return bernoulli_random(shape, dtype, mean, seed, name)


@typemap
def uniform_like(x, low=0.0, high=1.0, seed=auto_select, name=''):
    """uniform_like(x, low=0.0, high=1.0, seed=auto_select, name='')
    Generates samples from the uniform distribution in the interval [`low`,`high`).

    Args:
        x: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        low (float): lower end of the range of the random numbers
        high (float): upper end of the range of the random numbers
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    x = sanitize_input(x)
    return uniform_random(x, low, high, seed, name)


@typemap
def normal_like(x, mean=0.0, scale=1.0, seed=auto_select, name=''):
    """normal_like(x, mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the normal distribution with mean `mean` and standard deviation `scale`.

    Args:
        x: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        mean (float): mean of the distribution
        scale (float): scale (standard deviation) of the distribution
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    x = sanitize_input(x)
    return normal_random(x, mean, scale, seed, name)


@typemap
def gumbel_like(x, loc=0.0, scale=1.0, seed=auto_select, name=''):
    """gumbel_like(x, mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the Gumbel distribution with location `loc` and scale `scale`.

    Args:
        x: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        loc (float): location of the distribution
        scale (float): scale of the distribution
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        `The Gumbel-Max Trick
        <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>`_.
    """
    x = sanitize_input(x)
    return gumbel_random(x, loc, scale, seed, name)


@typemap
def bernoulli_like(x, mean=0.5, seed=auto_select, name=''):
    """bernoulli_like(x, mean=0.5, seed=auto_select, name='')
    Generates samples from the Bernoulli distribution with success probability `mean`.

    Args:
        x: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        mean (float): success probability
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    x = sanitize_input(x)
    return bernoulli_random(x, mean, seed, name)


