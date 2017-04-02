# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..axis import *
import cntk as C

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

def test_axis_str():
    i = C.sequence.input((1, 3))
    assert str(C.Axis.all_axes()) == "Axis('AllAxes')"
    assert str(C.Axis.all_static_axes()) == "Axis('AllStaticAxes')"
    assert str(C.Axis.unknown_dynamic_axes()) == "(Axis('UnknownAxes'),)"
    assert str(Axis(1)) == "Axis('staticAxisIdx=1')"
    assert str(Axis(-1)) == "Axis('staticAxisIdx=-1')"
    assert str(i.dynamic_axes) == "(Axis('defaultBatchAxis'), Axis('defaultDynamicAxis'))"
