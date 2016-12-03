# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..axis import *

def test_axis():
    a = Axis(1)
    assert isinstance(a.is_static_axis, bool)
    assert a.is_static_axis == True
    assert a.static_axis_index() == 1


def test_dyn_axis():
    a = Axis.new_unique_dynamic_axis('x')
    assert isinstance(a.is_static_axis, bool)
    assert a.is_static_axis == False

def test_default_axis():
    assert isinstance(Axis.default_batch_axis(), Axis)
    assert isinstance(Axis.default_dynamic_axis(), Axis)

def test_axis_comparison():
    a1 = Axis.default_batch_axis()
    a2 = Axis.default_batch_axis()
    assert a1 == a2
