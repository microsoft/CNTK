//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#include "stdafx.h"

// This class implements a block handler based on the SSE intrinsics available on intel platforms.
// Since we don't have SSE on ARM64 (NEON has similar functionality but is not identical) we cannot
// use the BlockHandlerSSE implementation no ARM64.
// Therefore, exclude the implementation on ARM64 builds for now.
// TODO: In the future, we should provide a NEON based implementation instead.
#if !defined(__aarch64__)

#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>

#include "BlockHandlerSSE.h"
#include "BlockMultiplierMatrixUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

int BlockHandlerSSE::RowToColOffsetRewrittenA(int row, int kOffset, int blockSize, int rowsPerBlock, int origCols)
{
    int rowIdx = row / rowsPerBlock;
    int offsetFromBlockBeginning = row % rowsPerBlock;
    int colIdx = kOffset * rowsPerBlock * blockSize + (offsetFromBlockBeginning * blockSize);
    return (rowIdx * (origCols / blockSize) * rowsPerBlock * blockSize) + colIdx;
}


//col is the original column of B
//kOffset is the offset to the current block we are multiplying against (in absolute
int BlockHandlerSSE::RowToColOffsetRewrittenB(int col, int kOffset, int blockSize, int origCols)
{
    return (origCols *  blockSize * kOffset) + (col * blockSize);
}


}}}

#endif
