# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from cntk import reshape, swapaxes, alias
import numpy as np

def transposeAlot(tensor, permutation):
    nr_of_axes = len(permutation)

    #assertions on the input
    assert len(tensor.shape) == nr_of_axes, "The number of axes in the permutation does not match the input!"
    np_perm=np.asarray(permutation)
    for i in range(nr_of_axes):
        assert np.any(np_perm == i), "Axis " + str(i) + " is not set in the permutation!"

    current_permutation = np.arange(nr_of_axes)

    for i in range(nr_of_axes - 1):
        # does this position need to be changed?
        if permutation[i] != current_permutation[i]:
            # search for current position of the axis to be placed at i!
            for j in range(i, nr_of_axes):
                if current_permutation[j] == permutation[i]:
                    break

            # swap these two axes
            tensor = swapaxes(tensor, i, j)

            current_permutation[[i, j]] = current_permutation[[j, i]]
            #print(current_permutation)

    return alias(tensor, "End_TranposeAlot_"+str(permutation))


def depth_increasing_pooling(tensor, pooling_window_shape, input_shape=None):
    if input_shape is None: input_shape = tensor.shape
    # this is necessary in case that the previous model has not yet inferred its shape.
    # In order to perform the reshape properly the input_shape must be known!

    assert len(input_shape) == 3, "depth_increasing_pooling requires the input tensor to have 3 axes!"

    shape_tmp = (input_shape[0],
                 int(input_shape[1] / pooling_window_shape[0]),
                 pooling_window_shape[0],
                 int(input_shape[2] / pooling_window_shape[1]),
                 pooling_window_shape[1])
    final_shape = (input_shape[0] * pooling_window_shape[0] * pooling_window_shape[1],
                   input_shape[1] / pooling_window_shape[0],
                   input_shape[2] / pooling_window_shape[1])

    temp1 = reshape(tensor, shape_tmp)

    temp2 = transposeAlot(temp1, (0, 2, 4, 1, 3))

    temp3 = reshape(temp2, final_shape)

    return temp3


if __name__ == '__main__':
    """
    Here the functionality for the pooling_window_shape (2,2) is validated.
    """


    #Test for transposeAlot
    if False:
        for i in range(1,12):
            permutation = np.random.permutation(i+1)
            shape = []
            entries = 1
            for j in range(i+1):
                length = int(np.random.random_sample()*10)+1
                shape += [length]
                entries *= length

            data = np.arange(entries)
            data.shape = shape

            print(data.shape, permutation)

            np_transposed = np.transpose(np.copy(data), np.copy(permutation))
            by_transposeAlot = transposeAlot(np.ascontiguousarray(np.copy(data)), np.copy(permutation)).eval()

            assert np.alltrue(np_transposed == by_transposeAlot)


    # Test for depth_increasing_pooling

    def dip_np(volume, pooling_window_shape):
        shape_tmp = (volume.shape[0],
                     int(volume.shape[1] / pooling_window_shape[0]),
                     pooling_window_shape[0],
                     int(volume.shape[2] / pooling_window_shape[1]),
                     pooling_window_shape[1])
        final_shape = (volume.shape[0] * pooling_window_shape[0] * pooling_window_shape[1],
                       int(volume.shape[1] / pooling_window_shape[0]),
                       int(volume.shape[2] / pooling_window_shape[1]))

        volume.shape = shape_tmp
        temp = np.ascontiguousarray(np.transpose(volume, (0, 2, 4, 1, 3)))
        temp.shape = final_shape
        return temp


    def transposeAlotNp(model, permutation):
        nr_of_axes = len(permutation)
        current_permutation = np.arange(nr_of_axes)

        for i in range(nr_of_axes - 1):
            # does this position need to be changed?
            if permutation[i] != current_permutation[i]:
                # search for current position of the axis to be placed at i!
                for j in range(i, nr_of_axes):
                    if current_permutation[j] == permutation[i]:
                        break

                # swap these two axes
                model = np.ascontiguousarray(np.swapaxes(model, i, j))

                tmp = current_permutation[i]
                current_permutation[i] = current_permutation[j]
                current_permutation[j] = tmp
                #print(current_permutation)

        return model


    def dip_swap_np(volume, pooling_window_shape):
        shape_tmp = (volume.shape[0],
                     int(volume.shape[1] / pooling_window_shape[0]),
                     pooling_window_shape[0],
                     int(volume.shape[2] / pooling_window_shape[1]),
                     pooling_window_shape[1])
        final_shape = (volume.shape[0] * pooling_window_shape[0] * pooling_window_shape[1],
                       int(volume.shape[1] / pooling_window_shape[0]),
                       int(volume.shape[2] / pooling_window_shape[1]))

        volume.shape = shape_tmp
        temp = transposeAlotNp(volume, (0, 2, 4, 1, 3))
        temp.shape = final_shape
        return temp


    data = np.asarray(
        [[[0, 1, 2, 3],
          [10, 11, 12, 13],
          [20, 21, 22, 23],
          [30, 31, 32, 33]],

         [[100, 101, 102, 103],
          [110, 111, 112, 113],
          [120, 121, 122, 123],
          [130, 131, 132, 133]]
         ], dtype=float)

    print("data:")
    print(data)
    print(data.shape)

    expected = np.asarray(
        [[[0, 2],
          [20, 22]],

         [[1, 3],
          [21, 23]],

         [[10, 12],
          [30, 32]],

         [[11, 13],
          [31, 33]],

         [[100, 102],
          [120, 122]],

         [[101, 103],
          [121, 123]],

         [[110, 112],
          [130, 132]],

         [[111, 113],
          [131, 133]]], dtype=float)

    np_transposed_version = dip_np(np.copy(data), (2, 2))
    print("Succeeded with numpy.transpose:")
    print(np.alltrue(np_transposed_version == expected)) # true
    print()

    np_repeated_swap_version = dip_swap_np(np.copy(data), (2, 2))
    print("Succeeded with selfwritten transpose by numpy.swapaxes:")
    print(np.alltrue(np_repeated_swap_version == expected)) #true
    print()

    cntk_made = depth_increasing_pooling(np.copy(data), (2, 2)).eval()
    print("Succeeded with selfwritten cntk:")
    print(np.alltrue(cntk_made == expected)) #true
    print()

    print("Ouput:")
    print(cntk_made)
    print(cntk_made.shape)
