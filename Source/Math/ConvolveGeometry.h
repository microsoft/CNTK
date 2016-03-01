//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Notes:
// * ConvolveGeometry represents the application of one or more rectangular "kernels" (all of the same size)
//   to a rectangular input to produce a rectangular output.
// * A "cell" in the rectangular input is identified by a single coordinate called a "col" (for column).
// * A "cell" in the rectangular output is identified by a single coordinate called a "row".
// * The kernels may involve weights, in which case MpRowIwht indicates the starting index of the weights
//   used for a given output cell.
// The overall idea of ConvolveGeometry is to precompute maps that can be used to apply convolutions of
// arbitrary configuration and dimension. In such case the generic implementation becomes very simple and invariant
// wrt convolution configuration and dimensionality. For specific cases like 2D convolutions and full sharing,
// highly optimized implementations (e.g. cuDNN) are used.
class ConvolveGeometry final
{
public:
    using IntVec = std::vector<int>;

    // Maps from a "row" (index of output cell) to its base "col" (index of input cell). For a given row,
    // the cols that contribute to it are { MpRowCol[row] + Indices[i0 + 1 + i] | 0 <= i < Indices[i0] },
    // where i0 = MpRowIndices[row].
    const IntVec& MpRowCol() const { return m_mpRowCol; }

    // Maps from a "row" (index of output cell) to where to start in the weights array. Each run of weights
    // consists of KernelSize weights.
    const IntVec& MpRowIwht() const { return m_mpRowIwht; }

    // Maps from a "row" (index of output cell) to its starting index in Runs. A run consists of:
    // * skip count (to skip that many weights)
    // * item count
    // * relative indices into source (item count of these)
    // * masks (all 1's or all 0's) (item count of these)
    // For items that are masked out (0 mask), the index stored is the next valid index.
    // This ensures that accessing the corresponding neuron value doesn't fault and that
    // backprop operations write the correct value last (any previous writes won't change
    // the value).
    // NOTE: The first (zeroth) run is always the "full" kernel run. Also, MpRowRun can be empty,
    // indicating that all values are zero (all outputs use the "full" kernel run).
    const IntVec& MpRowRun() const { return m_mpRowRun; }
    const IntVec& Runs() const { return m_runs; }

    // Maps from a "row" (index of output cell) to its starting index in Indices. Note that "Runs" is intended
    // for kernels that have weights, while "Indices" is intended for kernels that don't need to access weights.
    // As a result, the encoding in Indices is simpler and more direct.
    // A run in Indices consists of:
    // * item count
    // * relative indices into source (item count of these)
    // NOTE: The first run of indices is always the "full" kernel run. Also, MpRowIndices can be empty,
    // indicating that all values are zero (all outputs use the "full" kernel run).
    const IntVec&  MpRowIndices() const { return m_mpRowIndices; }
    const IntVec&  Indices() const { return m_indices; }

    // The indices of the first ("top-left-most") "kernel-center" cell in the source.
    const IntVec&  Start() const { return m_start; }
    int StartIndex() const { return m_startIndex; }

    ConvolveGeometry(const TensorShape& input, const TensorShape& kernel)
    {
        assert(input.GetRank() == kernel.GetRank());
    }

private:
    IntVec m_mpRowCol;
    IntVec m_mpRowIwht;
    IntVec m_mpRowRun;
    IntVec m_runs;
    IntVec m_mpRowIndices;
    IntVec m_indices;
    IntVec m_start;
    int m_startIndex;
};

} } }
