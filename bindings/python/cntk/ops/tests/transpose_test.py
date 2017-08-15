# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import transpose

def test_transpose():
    """
    Test for transpose()
    :return: Nothing
    """
    repeat_for = 5

    for repeat in range(repeat_for):
        for i in range(1, 5):
            permutation = np.random.permutation(i + 1)
            permutation = [int(p) for p in permutation]

            shape = [np.random.randint(2, 5) for _ in range(i + 1)]
            entries = np.product(shape)

            data = np.arange(entries)
            data.shape = shape

            np_transposed = np.transpose(np.copy(data), np.copy(permutation))
            by_transposeCNTK = transpose(np.ascontiguousarray(data), permutation).eval()

            assert np.alltrue(np_transposed == by_transposeCNTK)

if __name__ == "__main__":
    test_transpose()
