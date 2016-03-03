//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "TensorShape.h"

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
    using BoolVec = std::vector<bool>;

    const TensorShape& InputShape() const { return m_inputShape; }
    const TensorShape& OutputShape() const { return m_outputShape; }
    const TensorShape& KernelShape() const { return m_kernelShape; }
    const TensorShape& MapCount() const { return m_mapCount; }
    const TensorShape& Stride() const { return m_stride; }
    const BoolVec& Sharing() const { return m_sharing; }
    const BoolVec& AutoPad() const { return m_autoPad; }
    const TensorShape& LowerPad() const { return m_lowerPad; }
    const TensorShape& UpperPad() const { return m_upperPad; }

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

    ConvolveGeometry(const TensorShape& inputShape, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& stride,
                     const BoolVec& sharing, const BoolVec& autoPad, const TensorShape& lowerPad, const TensorShape& upperPad)
                     : m_inputShape(inputShape), m_kernelShape(kernelShape), m_mapCount(mapCount), m_stride(stride), m_sharing(sharing),
                     m_autoPad(autoPad), m_lowerPad(lowerPad), m_upperPad(upperPad)
    {
        // Note: this ctor is a bit long so sit back and relax.

        assert(m_inputShape.GetRank() == m_kernelShape.GetRank());
        assert(m_mapCount.GetRank() == 1 || m_mapCount.GetRank() == m_inputShape.GetRank());
        assert(m_stride.GetRank() == 1 || m_stride.GetRank() == m_inputShape.GetRank());
        assert(m_sharing.size() == 1 || m_sharing.size() == m_inputShape.GetRank());
        assert(m_autoPad.size() == 1 || m_autoPad.size() == m_inputShape.GetRank());
        assert(m_lowerPad.GetRank() == 1 || m_lowerPad.GetRank() == m_inputShape.GetRank());
        assert(m_upperPad.GetRank() == 1 || m_upperPad.GetRank() == m_inputShape.GetRank());
        
        m_outputShape = ComputeOutputShape(m_inputShape, m_kernelShape, m_mapCount, m_stride,
                                           m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        assert(m_inputShape.GetRank() == m_outputShape.GetRank());

        size_t dimCount = inputShape.GetRank();
        size_t kernelSize = kernelShape.GetNumElements();

        // Compute the total number of kernels.
        size_t kernelCount = 1;
        for (size_t i = 0; i < dimCount; i++)
            kernelCount *= !GetSharing(i) ? m_outputShape[i] : GetMapCount(i);

        // Compute the "Start" indices.
        m_start.resize(dimCount);
        m_startIndex = 0;
        m_originIndex = 0;
        for (int i = (int)dimCount - 1; i >= 0; i--)
        {
            assert((m_outputShape[i] % GetMapCount(i)) == 0);
            int outPerMap = (int)(m_outputShape[i] / GetMapCount(i));
            // Number of cells between first and last "centers", inclusive.
            int cells = (int)((outPerMap - 1) * GetStride(i) + 1);
            assert(m_inputShape[i] >= cells);

            // Extra cells, to the left and right of "cells".
            int extra = (int)m_inputShape[i] - cells;
            assert(extra >= 0);

            // When LowerPad and/or UpperPad are specified, the Start[i] value is determined by those values.
            int lo = GetAutoPad(i) ? 0 : (int)m_lowerPad[m_lowerPad.size() == 1 ? 0 : i];
            int hi = GetAutoPad(i) ? 0 : (int)m_upperPad[m_upperPad.size() == 1 ? 0 : i];
            if (lo != 0 || hi != 0)
            {
                assert(extra + lo + hi + 1 == m_kernelShape[i]);
                // Compute the number of cells on the left and right parts of the kernel,
                // not counting the "kernel-center" cell. If m_kernelShape[i] is even, the extra cell is
                // placed on the right (the center is shifted to the left).
                int right = (int)m_kernelShape[i] - 1;
                int left = right / 2;
                right -= left;
                assert(left <= right);
                assert(right <= left + 1);

                assert(lo <= left);
                assert(hi <= right);
                m_start[i] = left - lo;
                assert(m_start[i] + cells + right == m_inputShape[i] + hi);
            }
            else
            {
                m_start[i] = extra / 2;
#ifdef _DEBUG
                // If we're padding then extra should be covered.
                bool padded = GetAutoPad(i);
                assert(!padded || extra + 1 <= m_kernelShape[i]);
                // If we're not padding then, we should stay within the input dimension.
                assert(padded || extra + 1 >= m_kernelShape[i]);

                // Compute the number of cells on the left and right parts of the kernel,
                // not counting the "kernel-center" cell. If m_kernelShape[i] is even, the extra cell is
                // placed on the right (the center is shifted to the left).
                int right = (int)m_kernelShape[i] - 1;
                int left = right / 2;
                right -= left;
                assert(0 <= left);
                assert(left <= right);
                assert(right <= left + 1);

                int min = m_start[i] - left;
                int max = m_start[i] + (int)cells + right;
                assert(!padded || min <= 0 && max >= m_inputShape[i]);
                assert(padded || min >= 0 && max <= m_inputShape[i]);

                int diff = min - ((int)m_inputShape[i] - max);
                assert(std::abs(diff) <= 1);
#endif
            }

            m_startIndex = m_startIndex * (int)m_inputShape[i] + m_start[i];
            m_originIndex = m_originIndex * (int)m_inputShape[i] + ((int)m_kernelShape[i] - 1) / 2;
        }
        
        // Compute support, mapping from the index into the kernel to offset into source.
        // Support consists of the column deltas of the kernels, as offsets from MpRowCol[row].
        IntVec support(kernelSize);
        std::vector<IntVec> kernelCoords(kernelSize);
        for (int idx = 0; idx < kernelSize; idx++)
        {
            kernelCoords[idx].resize(dimCount);
            int ivSrc = 0;
            int factor = 1;
            int cur = idx;
            for (size_t i = 0; i < dimCount; i++)
            {
                assert(cur >= 0);
                int d = (int)m_kernelShape[i];
                assert(d > 0);
                int coord = cur % d;
                cur /= d;
                kernelCoords[idx][i] = coord;
                ivSrc += factor * coord;
                factor *= (int)m_inputShape[i];
            }
            assert(cur == 0);
            assert(ivSrc < m_inputShape.GetNumElements());
            support[idx] = ivSrc - m_originIndex;
        }
        
        size_t outputSize = m_outputShape.GetNumElements();
        // Compute the mappings (where row = output node index, col = source node index):
        // * from row to the index of the first weight to use for that row.
        // * from row to the first input col. The rest are col + _support[i].
        m_mpRowIwht.resize(outputSize);
        m_mpRowCol.resize(outputSize);
        m_mpRowRun.resize(outputSize);
        m_mpRowIndices.resize(outputSize);

        // A "key" is an equivalence class of run/masks.
        // Calculate the key for an interior cell (for using all of support - when all masks are 1's).
        int keyInterior = 0;
        for (size_t i = 0; i < dimCount; i++)
        {
            int width = (int)m_kernelShape[i];
            keyInterior = keyInterior * width + (width - 1) / 2;
        }

        m_runs.resize(2 * kernelSize + 2, -1);
        m_indices.resize(kernelSize + 1);
        m_runs[0] = 0; // Skip count
        m_runs[1] = (int)kernelSize; // Count of entries
        m_indices[0] = (int)kernelSize;
        for (size_t i = 0; i < kernelSize; i++)
        {
            m_runs[2 + i] = support[i];
            m_indices[1 + i] = support[i];
        }

        // Working buffer for masks.
        IntVec masks(kernelSize);

        // Map from key to pair of starting locations in Runs and Indices.
        std::map<int, std::pair<int, int>>  mpkeystarts;
        mpkeystarts[keyInterior] = std::make_pair(0, 0);

        IntVec dkey(dimCount);
        for (size_t row = 0; row < outputSize; row++)
        {
            // Compute the kernel number, column, and key.
            // REVIEW alexeyk: Seems like there should be a simpler and faster way, without starting
            // from scratch for each output (row)....
            int kern = 0;
            int col = 0;
            int factorKern = 1;
            int factorCol = 1;
            int key = 0;
            int cur = (int)row;
            for (size_t i = 0; i < dimCount; i++)
            {
                int dim = (int)(m_outputShape[i] / GetMapCount(i));
                int coord = cur % dim;
                cur /= dim;

                // Kernel
                if (!GetSharing(i))
                {
                    kern += factorKern * coord;
                    factorKern *= dim;
                }

                int maps = (int)GetMapCount(i);
                if (maps > 1)
                {
                    kern += factorKern * (cur % maps);
                    cur /= maps;
                    factorKern *= maps;
                }

                // Transform coord to input index space.
                coord *= (int)GetStride(i);
                coord += m_start[i];

                col += factorCol * coord;
                factorCol *= (int)m_inputShape[i];

                int width = (int)m_kernelShape[i];
                int half = (width - 1) / 2;
                int min = coord - half;
                int lim = min + width;
                if (min < 0)
                    dkey[i] = min;
                else if (lim > m_inputShape[i])
                    dkey[i] = lim - (int)m_inputShape[i];
                else
                    dkey[i] = 0;
                int dk = dkey[i] + half;
                assert(0 <= dk);
                assert(dk < width);
                key = key * width + dk;
            }
            assert(cur == 0);
            assert(0 <= kern);
            assert(kern < kernelCount);
            assert(0 <= col);
            assert(col < m_inputShape.GetNumElements());

            auto startsIter = mpkeystarts.find(key);
            if (startsIter == mpkeystarts.end())
            {
                auto starts = std::make_pair((int)m_runs.size(), (int)m_indices.size());
                mpkeystarts[key] = starts;

                int indexCount = 0;
                for (int idx = 0; idx < kernelSize; idx++)
                {
                    const auto& coords = kernelCoords[idx];
                    int mask = 0;
                    for (int i = (int)dimCount; ; )
                    {
                        if (--i < 0)
                        {
                            // All OK.
                            mask = -1;
                            break;
                        }
                        int k = dkey[i] + coords[i];
                        if (k < 0)
                            break;
                        if (k >= m_kernelShape[i])
                            break;
                    }
                    assert(mask == 0 || mask == -1);
                    indexCount -= mask;
                    masks[idx] = mask;
                }

                int skip = 0;
                while (masks[skip] == 0)
                    skip++;
                int count = (int)kernelSize;
                while (masks[count - 1] == 0)
                    count--;

                count -= skip;
                m_runs.push_back(skip); // Skip count
                m_runs.push_back(count); // Count of entries
                m_indices.push_back(indexCount);
                for (int i = 0, iMin = 0; i < count; i++)
                {
                    int index = support[skip + i];
                    int mask = masks[skip + i];
                    if (mask != 0)
                    {
                        // Add "index" to runs for this slot and any immediately preceeding
                        // slots that have mask == 0.
                        assert(iMin <= i);
                        assert(m_runs.size() == starts.first + 2 + iMin);
                        for (; iMin <= i; iMin++)
                            m_runs.push_back(index);
                        assert(iMin == i + 1);
                        assert(m_runs.size() == starts.first + 2 + iMin);

                        m_indices.push_back(index);
                    }
                }
                for (int i = 0; i < count; i++)
                    m_runs.push_back(masks[skip + i]);
                assert(m_runs.size() == std::get<0>(starts) + 2 + 2 * count);
                assert(m_indices.size() == std::get<1>(starts) + 1 + indexCount);

                m_mpRowRun[row] = starts.first;
                m_mpRowIndices[row] = starts.second;
            }
            else
            {
                m_mpRowRun[row] = (*startsIter).second.first;
                m_mpRowIndices[row] = (*startsIter).second.second;
            }
            assert(0 <= kern);
            assert(kern < kernelCount);
            m_mpRowCol[row] = col;
            m_mpRowIwht[row] = kern * (int)kernelSize;
        }
    }

    static TensorShape ComputeOutputShape(const TensorShape& inputShape, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& stride,
                                          const BoolVec& sharing, const BoolVec& autoPad, const TensorShape& lowerPad, const TensorShape& upperPad)
    {
        if (inputShape.GetRank() != kernelShape.GetRank())
            InvalidArgument("Convolution input and kernel tensors must have the same rank.");
        if (mapCount.GetRank() != 1 && inputShape.GetRank() != mapCount.GetRank())
            InvalidArgument("Convolution map tensor must have rank 1 or the same as the input tensor.");
        if (stride.GetRank() != 1 && inputShape.GetRank() != stride.GetRank())
            InvalidArgument("Convolution stride tensor must have rank 1 or the same as the input tensor.");
        if (sharing.size() != 1 && inputShape.GetRank() != sharing.size())
            InvalidArgument("Convolution sharing tensor must have rank 1 or the same as the input tensor.");
        if (autoPad.size() != 1 && inputShape.GetRank() != autoPad.size())
            InvalidArgument("Convolution padding tensor must have rank 1 or the same as the input tensor.");
        if (lowerPad.GetRank() != 1 && inputShape.GetRank() != lowerPad.GetRank())
            InvalidArgument("Convolution lower pad tensor must have rank 1 or the same as the input tensor.");
        if (upperPad.GetRank() != 1 && inputShape.GetRank() != upperPad.GetRank())
            InvalidArgument("Convolution upper pad tensor must have rank 1 or the same as the input tensor.");

        SmallVector<size_t> dimsOutput(inputShape.GetRank());
        for (size_t i = 0; i < inputShape.GetRank(); i++)
        {
            assert(inputShape[i] >= 1);
            if (kernelShape[i] > inputShape[i])
                InvalidArgument("NDConvolution operation requires that kernel dim %d <= input dim %d.", (int)kernelShape[i], (int)inputShape[i]);

            size_t delta = stride[stride.GetRank() == 1 ? 0 : i];
            if (delta > kernelShape[i])
                InvalidArgument("NDConvolution operation requires that stride %d <= input dim %d.", (int)stride[i], (int)inputShape[i]);
            
            size_t dim = inputShape[i];
            bool autoPadCur = autoPad[autoPad.size() == 1 ? 0 : i];
            if (autoPadCur)
            {
                dim += kernelShape[i] - 1;
            }
            else
            {
                size_t lo = lowerPad[lowerPad.size() == 1 ? 0 : i];
                size_t hi = upperPad[upperPad.size() == 1 ? 0 : i];
                dim += lo + hi;
            }
            size_t dimOut = (dim - kernelShape[i]) / delta + 1;
            if (!autoPadCur)
            {
                // When LowerPad and/or UpperPad are specified, we insist that the kernel applications
                // fill the entire space.
                size_t size = (dimOut - 1) * delta + kernelShape[i];
                if (size != dim)
                    InvalidArgument("NDConvolution requires that kernel fills the entire space if auto-padding is disabled.");
            }
            if (mapCount.size() > 1)
                dimOut *= mapCount[i];
            else if (i == inputShape.GetRank() - 1)
                dimOut *= mapCount[0];

            dimsOutput[i] = dimOut;
        }

        auto dimsOut = TensorShape(dimsOutput);
        // Check the output dimensions.
        size_t mapCountTotal = mapCount.GetNumElements();
        size_t sizeOut = dimsOut.GetNumElements();
        assert((sizeOut % mapCountTotal) == 0);

        return dimsOut;
    }

    DISABLE_COPY_AND_MOVE(ConvolveGeometry);

private:

    size_t GetStride(size_t dim)
    {
        assert(m_stride.size() == 1 || dim < m_stride.size());
        return m_stride[m_stride.size() == 1 ? 0 : dim];
    }

    size_t GetMapCount(size_t dim)
    {
        assert(m_mapCount.size() == 1 || dim < m_mapCount.size());
        // If the whole map count tensor was specified explicitly - return requested component.
        if (m_mapCount.size() > 1)
            return m_mapCount[dim];
        // If map count tensor rank == 1 then assume it represents number of feature maps for the rightmost dimension.
        if (dim == m_inputShape.size() - 1)
            return m_mapCount[0];
        return 1;
    }

    bool GetSharing(size_t dim)
    {
        assert(m_sharing.size() == 1 || dim < m_sharing.size());
        return m_sharing[m_sharing.size() == 1 ? 0 : dim];
    }

    bool GetAutoPad(size_t dim)
    {
        assert(m_autoPad.size() == 1 || dim < m_autoPad.size());
        return m_autoPad[m_autoPad.size() == 1 ? 0 : dim];
    }

private:
    TensorShape m_inputShape;
    TensorShape m_outputShape;
    TensorShape m_kernelShape;
    TensorShape m_mapCount;
    TensorShape m_stride;
    BoolVec m_sharing;
    BoolVec m_autoPad;
    TensorShape m_lowerPad;
    TensorShape m_upperPad;

    // There are several reasons why int type is used here rather than size_t:
    // 1. Many of these vectors contain offsets which can be negative.
    // 2. Most of these vectors will be copied into device memory (GPU) so the smaller the size - the better.
    //    Also, 64-bit operations are slower on GPU.
    // 3. If you are still not convinced, we don't expect convolutions to be more than 2B in size anyway. 
    // See description to corresponding getter functions to understand what these are.
    IntVec m_mpRowCol;
    IntVec m_mpRowIwht;
    IntVec m_mpRowRun;
    IntVec m_runs;
    IntVec m_mpRowIndices;
    IntVec m_indices;
    // The indices of the first ("top-left-most") "kernel-center" cell in the source.
    IntVec m_start;
    int m_startIndex;
    // When the first kernel cell is aligned with the first source cell, this is the index of the input cell that
    // is aligned with the "kernel-center" cell. Indices in "Runs" and "Indices" are relative to OriginIndex.
    int m_originIndex;
};

} } }
