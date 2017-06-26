# =============================================================================
# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# =============================================================================

from __future__ import division

from cntk.ops.sequence import past_value
from cntk.ops import parameter, placeholder
from cntk.ops import times, sigmoid, tanh, slice, splice
from cntk.initializer import glorot_uniform


def lightlstm(input_dim, cell_dim):
    x = placeholder(name='x')
    dh = placeholder(name='dh')
    dc = placeholder(name='dc')
    x1 = slice(x, -1, input_dim * 0, input_dim * 1)
    x2 = slice(x, -1, input_dim * 1, input_dim * 2)

    def LSTMCell(x, y, dh, dc):
        '''The based LightLSTM Cell'''

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

    Cell = LSTMCell(x1, x2, dh, dc)

    actualDh = past_value(Cell[2])
    actualDc = past_value(Cell[3])

    Cell[0].replace_placeholders(
        {dh: actualDh.output, dc: actualDc.output})
    return splice(Cell[0], Cell[2], axis=-1)
