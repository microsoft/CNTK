# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy
import pytest

from cntk.ops import *
from cntk.utils import *
from cntk import cntk_py

def _param():
    return cntk_py.Parameter((1,2), cntk_py.DataType_Float, 5.0)

def test_typemap():
    @typemap
    def returnParam():
        return _param()

    res = returnParam()
    assert res.__class__ == variables.Parameter

    @typemap
    def returnTuple():
        return 'some_string', _param()

    res_str, res_param = returnTuple()
    assert res_str.__class__ == str
    assert res_param.__class__ == variables.Parameter

    @typemap
    def returnList():
        return ['some_string', _param()]

    res_str, res_param = returnList()
    assert res_str.__class__ == str
    assert res_param.__class__ == variables.Parameter

    @typemap
    def returnSet():
        return set(['some_string', _param()])

    res = returnList()
    assert len(res) == 2
    res.remove('some_string')
    left_over = res.pop()
    assert left_over.__class__ == variables.Parameter

    @typemap
    def returnTupleWithDict():
        return (None, { _param(): _param() })
                
    res = returnTupleWithDict()
    assert len(res) == 2
    for k,v in res[1].items():
        assert k.__class__ == variables.Parameter
        assert v.__class__ == variables.Parameter

    @typemap
    def returnFunction():
        left_val = [[10,2]]
        right_val = [[2],[3]]

        p = placeholder(shape=(1,2))
        op = times(p, right_val)
        c = constant(left_val)

        return op.replace_placeholders({p:c})

    res = returnFunction()
    assert res.__class__ == functions.Function
