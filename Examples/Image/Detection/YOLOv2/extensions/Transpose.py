# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from cntk import reshape, swapaxes, alias
import numpy as np


def transpose_a_lot(tensor, permutation):
    """
    Function capable of transposing tensors with more than 5 unflattened inputs.
    :param tensor: tensor to be transposed
    :param permutation: Permutation to be applied to the axes. Tuple containing numbers starting from zero without duplications, length must be identical to the length of tensor.shape
    :return: the transposed tensor
    """
    nr_of_axes = len(permutation)

    # assertions on the input
    ## assert input dims are correct
    assert len(tensor.shape) == nr_of_axes, "The number of axes in the permutation does not match the input!"
    ## assert permutation is valid
    np_perm=np.asarray(permutation)
    for i in range(nr_of_axes):
        assert np.any(np_perm == i), "Axis " + str(i) + " is not set in the permutation!"

    # in the beginning the axes are sorted
    current_permutation = np.arange(nr_of_axes)
    tensor = alias(tensor, "Begin_TransposeAlot_"+str(permutation))

    for i in range(nr_of_axes - 1): # n-1 is sufficient since if 0..n-1 are correctly ordered than n  must be in the correct place, too
        # does the axis at the current position need to be swapped?
        if permutation[i] != current_permutation[i]:
            # search for current position of the axis to be placed at i!
            for j in range(i, nr_of_axes):
                if current_permutation[j] == permutation[i]:
                    break

            # swap these two axes
            tensor = swapaxes(tensor, i, j)
            current_permutation[[i, j]] = current_permutation[[j, i]]
            # print(current_permutation)

    return alias(tensor, "End_TranposeAlot_"+str(permutation))


def test_transpose_a_lot():
    """
    Test for transpose_a_lot()
    :return: Nothing
    """
    repeat_for = 5

    for repeat in range(repeat_for):
        for i in range(1, 12):
            permutation = np.random.permutation(i + 1)
            #shape = []
            #entries = 1
            #for j in range(i + 1):
            #    length = int(np.random.random_sample() * 4) + 2 # create random int [2,5]
            #    shape += [length]
            #    entries *= length

            shape = [np.random.randint(2, 5) for _ in range(i + 1)]
            entries = np.product(shape)

            data = np.arange(entries)
            data.shape = shape

            np_transposed = np.transpose(np.copy(data), np.copy(permutation))
            by_transposeAlot = transpose_a_lot(np.ascontiguousarray(np.copy(data)), np.copy(permutation)).eval()

            assert np.alltrue(np_transposed == by_transposeAlot)
