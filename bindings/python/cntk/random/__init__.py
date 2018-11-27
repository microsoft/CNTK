# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Functions that generate random numbers with respect to designated distributions.
"""


from __future__ import division
from __future__ import print_function
import numpy as np
from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as auto_select
from cntk.default_options import default_override_or
from cntk.internal.swig_helper import typemap
from cntk.internal import sanitize_random_args, sanitize_input


@typemap
def uniform(shape, dtype=default_override_or(np.float32), low=0.0, high=1.0, seed=auto_select, name=''):
    """uniform(shape, dtype=default_override_or(np.float32), low=0.0, high=1.0, seed=auto_select, name='')
    Generates samples from the uniform distribution in the interval [`low`,`high`).

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64 or np.float16): data type. Default is np.float32.
        low (float): lower end of the range of the random numbers
        high (float): upper end of the range of the random numbers
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Examples:
        >>> u = C.random.uniform((2,3), seed=98052)
        >>> u.eval(device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        array([[ 0.931785,  0.814722,  0.479606],
               [ 0.937468,  0.004351,  0.185131]], dtype=float32)

    """
    from cntk.cntk_py import uniform_random
    shape, dtype = sanitize_random_args(shape, dtype)
    return uniform_random(shape, dtype, low, high, seed, name)


@typemap
def normal(shape, dtype=default_override_or(np.float32), mean=0.0, scale=1.0, seed=auto_select, name=''):
    """normal(shape, dtype=default_override_or(np.float32), mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the normal distribution with mean `mean` and standard deviation `scale`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64 or np.float16): data type. Default is np.float32.
        mean (float): mean of the distribution
        scale (float): scale (standard deviation) of the distribution
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Examples:
        >>> z = C.random.normal((2,3), seed=98052)
        >>> z.eval(device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        array([[ 1.803254,  0.995395, -0.631974],
               [-1.736721,  0.005615, -0.340025]], dtype=float32)
    """
    from cntk.cntk_py import normal_random
    shape, dtype = sanitize_random_args(shape, dtype)
    return normal_random(shape, dtype, mean, scale, seed, name)


@typemap
def gumbel(shape, dtype=default_override_or(np.float32), loc=0.0, scale=1.0, seed=auto_select, name=''):
    """gumbel(shape, dtype=default_override_or(np.float32), loc=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the Gumbel distribution with location `loc` and scale `scale`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64 or np.float16): data type. Default is np.float32.
        loc (float): location of the distribution
        scale (float): scale of the distribution
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Examples:
        >>> g = C.random.gumbel((2,3), seed=98052)
        >>> g.eval(device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        array([[-0.987713, -0.522298,  0.425918],
               [-1.019599,  5.435177,  1.586071]], dtype=float32)

    See also:
        `The Gumbel-Max Trick
        <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>`__.
    """
    from cntk.cntk_py import gumbel_random
    shape, dtype = sanitize_random_args(shape, dtype)
    return gumbel_random(shape, dtype, loc, scale, seed, name)


@typemap
def bernoulli(shape, dtype=default_override_or(np.float32), mean=0.5, seed=auto_select, name=''):
    """bernoulli(shape, dtype=default_override_or(np.float32), mean=0.5, seed=auto_select, name='')
    Generates samples from the Bernoulli distribution with success probability `mean`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64 or np.float16): data type. Default is np.float32.
        mean (float): success probability
        seed (int): pseudo random number generator seed (default: automatically select a unique seed)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Examples:
        >>> b = C.random.bernoulli((2,3), seed=98052)
        >>> b.eval(device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        array([[ 1.,  1.,  0.],
               [ 1.,  0.,  0.]], dtype=float32)
    """
    from cntk.cntk_py import bernoulli_random
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

    Examples:
        >>> x = C.input_variable(4)
        >>> x0 = np.zeros((3,4), dtype=np.float32)
        >>> u = C.random.uniform_like(x, seed=98052)
        >>> u.eval({x:x0}, device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        array([[ 0.931785,  0.814722,  0.479606,  0.937468],
               [ 0.004351,  0.185131,  0.00632 ,  0.118901],
               [ 0.710054,  0.304273,  0.043126,  0.987818]], dtype=float32)
    """
    from cntk.cntk_py import uniform_random_like
    x = sanitize_input(x)
    return uniform_random_like(x, low, high, seed, name)


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

    Examples:
        >>> x = C.parameter((2,3,4))
        >>> z = C.random.normal_like(x, seed=98052)
        >>> z.eval(device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        array([[[ 1.803254,  0.995395, -0.631974, -1.736721],
                [ 0.005615, -0.340025, -0.011913, -0.236371],
                [-1.207685, -0.495846,  0.037022, -1.220596]],
        <BLANKLINE>
               [[ 0.872981,  0.654405, -0.111421, -0.544074],
                [ 1.543746, -0.63555 , -1.072869, -0.379701],
                [ 0.592069, -1.035192,  1.679303, -0.391963]]], dtype=float32)
    """
    from cntk.cntk_py import normal_random_like
    x = sanitize_input(x)
    return normal_random_like(x, mean, scale, seed, name)


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

    Examples:
        >>> x = C.constant(np.zeros((2,3,4), dtype=np.float32))
        >>> g = C.random.gumbel_like(x, seed=98052)
        >>> s = g.eval(device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        >>> np.round(s, 4)
        array([[[-0.9877, -0.5223,  0.4259, -1.0196],
                [ 5.4352,  1.5861,  5.0608,  2.0668],
                [-0.2135,  1.0139,  3.1217, -1.4834]],
        <BLANKLINE>
               [[ 0.4507,  0.6325,  2.1682,  0.4463],
                [-0.6583,  0.1147, -0.3144, -0.7925],
                [ 1.9773, -0.3627, -0.4566, -0.2368]]], dtype=float32)

    See also:
        `The Gumbel-Max Trick
        <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>`__.
    """
    from cntk.cntk_py import gumbel_random_like
    x = sanitize_input(x)
    return gumbel_random_like(x, loc, scale, seed, name)


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

    Examples:
        >>> p = C.placeholder()
        >>> bp = C.random.bernoulli_like(p, seed=98052)
        >>> x = C.sequence.input_variable(1)
        >>> bx = bp.replace_placeholders({p:x})
        >>> x0 = np.zeros((1,3,1), dtype=np.float32)
        >>> bx.eval({x:x0}, device=C.cpu()) # explicitly setting cpu because this is tested on multiple platforms; leave it unspecified in your code
        [array([[ 1.],
               [ 1.],
               [ 0.]], dtype=float32)]
    """
    from cntk.cntk_py import bernoulli_random_like
    x = sanitize_input(x)
    return bernoulli_random_like(x, mean, seed, name)


