# =============================================================================
# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# =============================================================================

from __future__ import division

import cntk as C

from cntk.ops.sequence import past_value
from cntk.initializer import glorot_uniform


def lightlstm(input_dim, cell_dim):
    x = C.placeholder(name='x')
    dh = C.placeholder(name='dh')
    dc = C.placeholder(name='dc')
    x1 = C.slice(x, -1, input_dim * 0, input_dim * 1)
    x2 = C.slice(x, -1, input_dim * 1, input_dim * 2)

    def LSTMCell(x, y, dh, dc):
        '''LightLSTM Cell'''

        b = C.parameter(shape=(4 * cell_dim), init=0)
        W = C.parameter(shape=(input_dim, 4 * cell_dim), init=glorot_uniform())
        H = C.parameter(shape=(cell_dim, 4 * cell_dim), init=glorot_uniform())

        # projected contribution from input x, hidden, and bias
        proj4 = b + C.times(x, W) + C.times(dh, H)

        it_proj = C.slice(proj4, -1, 0 * cell_dim, 1 * cell_dim)
        bit_proj = C.slice(proj4, -1, 1 * cell_dim, 2 * cell_dim)
        ft_proj = C.slice(proj4, -1, 2 * cell_dim, 3 * cell_dim)
        ot_proj = C.slice(proj4, -1, 3 * cell_dim, 4 * cell_dim)

        it = C.sigmoid(it_proj)  # input gate
        bit = it * C.tanh(bit_proj)

        ft = C.sigmoid(ft_proj)  # forget gate
        bft = ft * dc

        ct = bft + bit
        ot = C.sigmoid(ot_proj)  # output gate
        ht = ot * C.tanh(ct)

        # projected contribution from input y, hidden, and bias
        proj4_2 = b + C.times(y, W) + C.times(ht, H)

        it_proj_2 = C.slice(proj4_2, -1, 0 * cell_dim, 1 * cell_dim)
        bit_proj_2 = C.slice(proj4_2, -1, 1 * cell_dim, 2 * cell_dim)
        ft_proj_2 = C.slice(proj4_2, -1, 2 * cell_dim, 3 * cell_dim)
        ot_proj_2 = C.slice(proj4_2, -1, 3 * cell_dim, 4 * cell_dim)

        it_2 = C.sigmoid(it_proj_2)  # input gate
        bit_2 = it_2 * C.tanh(bit_proj_2)

        ft_2 = C.sigmoid(ft_proj_2)  # forget gate
        bft_2 = ft_2 * ct

        ct2 = bft_2 + bit_2
        ot_2 = C.sigmoid(ot_proj_2)  # output gate
        ht2 = ot_2 * C.tanh(ct2)
        return (ht, ct, ht2, ct2)

    Cell = LSTMCell(x1, x2, dh, dc)

    actualDh = past_value(Cell[2])
    actualDc = past_value(Cell[3])

    Cell[0].replace_placeholders(
        {dh: actualDh.output, dc: actualDc.output})
    return C.splice(Cell[0], Cell[2], axis=-1)
