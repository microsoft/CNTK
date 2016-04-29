# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from cntk import *


def test_eval_example():
    TOLERANCE_ABSOLUTE = 1E-06
    result = eval_example()
    assert np.allclose(result, np.asarray([[105., 107.5], [ 110., 112.5]]), atol=TOLERANCE_ABSOLUTE)

def eval_example():
    sample = [2, 3], [4, 5]
    sequence = np.asarray([sample])
    batch = [sequence]
    X = input_reader(batch)
    out = 2.5 * X + 100

    with LocalExecutionContext('demo', clean_up=True) as ctx:
        result = ctx.eval(out)
        return result

if __name__ == "__main__":
    print(eval_example())
    # outputs:
    # [[ 105.   107.5]
    #  [ 110.   112.5]]
