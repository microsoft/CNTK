# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

import cntk as C
from ..sanitize import sanitize_permutation


def test_sanitize_permutation():
    from itertools import permutations

    def perm_apply(x, perm):
        return [x[i] for i in perm]
    for i in range(7):
        test = [42 + j for j in range(i)]
        for p in permutations(range(i)):
            q = sanitize_permutation(p)
            assert(all(x == y for x, y in zip(
                list(reversed(perm_apply(test, p))), perm_apply(list(reversed(test)), q))))


def test_sanitize_number_of_parameters():
    i1 = C.input_variable(shape=(1,), needs_gradient=True)
    i2 = C.input_variable(shape=(1,), needs_gradient=True)
    res = i1 + i2

    with pytest.raises(ValueError):
        val = C.Value(C.NDArrayView.from_dense(np.asarray([[[1]]],
            dtype=np.float32)))
        res.forward(val)

    with pytest.raises(ValueError):
        res.eval()

    with pytest.raises(ValueError):
        res.forward({i1: [[1]]})
