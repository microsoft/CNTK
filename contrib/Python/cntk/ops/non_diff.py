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
    floor operation
    Args:
        arg: tensor input
    Returns:
        floor node
    """

    return Floor(arg, var_name = name)

def ceil(arg, name=None):
    """
    round operation
    Args:
        arg: tensor input
    Returns:
        ceil node
    """

    return Ceil(arg, var_name = name)

def round(arg, name=None):
    """
    round operation
    Args:
        arg: tensor input
    Returns:
        round node
    """

    return Round(arg, var_name = name)

