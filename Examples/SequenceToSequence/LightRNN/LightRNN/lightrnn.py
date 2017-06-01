# =============================================================================
# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


from cntk.ops import parameter, placeholder
from cntk.ops.sequence import past_value
from cntk.ops import times, sigmoid, tanh, slice
from cntk.initializer import glorot_uniform


def LSTMCell(x, y, dh, dc):
    '''The based LightLSTM Cell'''
    input_dim = x.shape[0]
    cell_dim = dh.shape[0]

    b = parameter(shape=(4 * cell_dim), init=0)
    W = parameter(shape=(input_dim, 4 * cell_dim), init=glorot_uniform())
    H = parameter(shape=(cell_dim, 4 * cell_dim), init=glorot_uniform())

    # projected contribution from input x, hidden, and bias
    proj4 = b + times(x, W) + times(dh, H)

    it_proj = slice(proj4, -1, 0 * cell_dim, 1 * cell_dim)
    bit_proj = slice(proj4, -1, 1 * cell_dim, 2 * cell_dim)
    ft_proj = slice(proj4, -1, 2 * cell_dim, 3 * cell_dim)
    ot_proj = slice(proj4, -1, 3 * cell_dim, 4 * cell_dim)

    it = sigmoid(it_proj)  # input gate
    bit = it * tanh(bit_proj)

    ft = sigmoid(ft_proj)  # forget gate
    bft = ft * dc

    ct = bft + bit
    ot = sigmoid(ot_proj)  # output gate
    ht = ot * tanh(ct)

    # projected contribution from input y, hidden, and bias
    proj4_2 = b + times(y, W) + times(ht, H)

    it_proj_2 = slice(proj4_2, -1, 0 * cell_dim, 1 * cell_dim)
    bit_proj_2 = slice(proj4_2, -1, 1 * cell_dim, 2 * cell_dim)
    ft_proj_2 = slice(proj4_2, -1, 2 * cell_dim, 3 * cell_dim)
    ot_proj_2 = slice(proj4_2, -1, 3 * cell_dim, 4 * cell_dim)

    it_2 = sigmoid(it_proj_2)  # input gate
    bit_2 = it_2 * tanh(bit_proj_2)

    ft_2 = sigmoid(ft_proj_2)  # forget gate
    bft_2 = ft_2 * ct

    ct2 = bft_2 + bit_2
    ot_2 = sigmoid(ot_proj_2)  # output gate
    ht2 = ot_2 * tanh(ct2)
    return (ht, ct, ht2, ct2)


def LSTM(input1, input2, cell_dim,
              recurrence_hookH=past_value, recurrence_hookC=past_value):
    '''Light-LSTM for language model'''
    dh = placeholder(shape=(cell_dim), dynamic_axes=input2.dynamic_axes)
    dc = placeholder(shape=(cell_dim), dynamic_axes=input2.dynamic_axes)

    Cell = LSTMCell(input1, input2, dh, dc)
    actualDh = recurrence_hookH(Cell[2])
    actualDc = recurrence_hookC(Cell[3])

    Cell[0].replace_placeholders(
            {dh: actualDh.output, dc: actualDc.output})

    return (Cell[0], Cell[1], Cell[2], Cell[3])
