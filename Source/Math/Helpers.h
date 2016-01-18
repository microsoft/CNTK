//
// <copyright file="Helpers.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

//helpful macros
// TODO: the file's name is too general to be included from outside; MathHelpers.h?

//iterators
#pragma once
#undef foreach_row
#undef foreach_column
#undef foreach_coord
#undef foreach_row_in_submat
#define foreach_row(_i, _m) for (long _i = 0; _i < (_m).GetNumRows(); _i++)
#define foreach_column(_j, _m) for (long _j = 0; _j < (_m).GetNumCols(); _j++)
#define foreach_coord(_i, _j, _m)                   \
    for (long _j = 0; _j < (_m).GetNumCols(); _j++) \
        for (long _i = 0; _i < (_m).GetNumRows(); _i++)
#define foreach_row_in_submat(_i, _istart, _iend, _m) for (long _i = _istart; _i < min(_iend, (_m).GetNumRows()); _i++)

//this functions returns the index of the first column element in the columnwise array representing matrix with _numRows rows
#define column_s_ind_colwisem(_colNum, _numRows) ((_numRows) * (_colNum))
