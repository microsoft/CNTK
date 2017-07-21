# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Utility functions."""

import cntk.ops as C


def huber_loss(output, target):
    r"""See https://en.wikipedia.org/wiki/Huber_loss for definition.

    \delta is set to 1. This is not the right definition if output and target
    differ in more than one dimension.
    """
    a = target - output
    return C.reduce_sum(C.element_select(
        C.less(C.abs(a), 1), C.square(a) * 0.5, C.abs(a) - 0.5))


def negative_of_entropy_with_softmax(p):
    """See https://en.wikipedia.org/wiki/Entropy_(information_theory)."""
    return C.reduce_sum(C.softmax(p) * p) - C.reduce_log_sum_exp(p)
