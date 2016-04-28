# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unary operators which are not differentiable
"""

from cntk.ops.cntk1 import Floor, Ceil, Round

def floor(arg, name=None):
    """
    Floor operation. The output of this operation is the
    element wise value rounded to the largest integer less than
    or equal to the input.

    Example:
        >>> floor([0.2, 1.3, 4., 5.5, 0.0])
        #[0.0, 1.0, 4.0, 5.0, 0.0]

        >>> floor([[0.6, 3.3], [1.9, 5.6]])
        #[[0.0, 3.0], [1.0, 5.0]]

        >>> floor([-5.5, -4.2, -3., -0.7, 0])
        #[-6.0, -5.0, -3.0, -1.0, 0.0]

        >>> floor([[-0.6, -4.3], [1.9, -3.2]])
        #[[-1.0, -5.0], [1.0, -4.0]]

    Args:
        arg: input tensor
        name: the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Floor(arg, var_name = name)

def ceil(arg, name=None):
    """
    Ceil operation. The output of this operation is the
    element wise value rounded to the smallest integer greater than
    or equal to the input.

    Example:
        >>> ceil([0.2, 1.3, 4., 5.5, 0.0])
        #[1.0, 2.0, 4.0, 6.0, 0.0]

        >>> ceil([[0.6, 3.3], [1.9, 5.6]])
        #[[1.0, 4.0], [2.0, 6.0]]

        >>> ceil([-5.5, -4.2, -3., -0.7, 0])
        #[-5.0, -4.0, -3.0, 0.0, 0.0]

        >>> ceil([[-0.6, -4.3], [1.9, -3.2]])
        #[[0.0, -4.0], [2.0, -3.0]]

    Args:
        arg: input tensor
        name: the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Ceil(arg, var_name = name)

def round(arg, name=None):
    """
    Round operation. The output of this operation is the
    element wise value rounded to the nearest integer. In case
    of tie, where element can have exact fractional part of 0.5
    this operation follows "round half-up" tie breaking strategy.
    This is different from the round operation of numpy which follows
    round half to even.

    Example:
        >>> round([0.2, 1.3, 4., 5.5, 0.0])
        #[0.0, 1.0, 4.0, 6.0, 0.0]

        >>> round([[0.6, 3.3], [1.9, 5.6]])
        #[[1.0, 3.0], [2.0, 6.0]]

        >>> round([-5.5, -4.2, -3., -0.7, 0])
        #[-5.0, -4.0, -3.0, -1.0, 0.0]

        >>> round([[-0.6, -4.3], [1.9, -3.2]])
        #[[-1.0, -4.0], [2.0, -3.0]]

    Args:
        arg: input tensor
        name: the name of the node in the network (optional)
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Round(arg, var_name = name)

