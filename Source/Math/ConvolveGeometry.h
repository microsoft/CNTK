//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "TensorShape.h"
#include <iterator>

namespace Microsoft { namespace MSR { namespace CNTK {

// Notes:
// * ConvolveGeometry represents the application of one or more rectangular "kernels" (all of the same size)
//   to a rectangular input to produce a rectangular output.
// * A "cell" in the rectangular input is identified by a single coordinate called a "col" (for column).
// * A "cell" in the rectangular output is identified by a single coordinate called a "row".
// * The kernels may involve weights, in which case MpRowIwht indicates the starting index of the weights
//   used for a given output cell.
// The overall idea of ConvolveGeometry is to precompute maps that can be used to apply convolutions of
// arbitrary configurations and dimensions. In such case the generic implementation becomes very simple and invariant
// wrt convolution configuration and dimensionality. For specific cases like 2D/3D convolutions and full sharing,
// highly optimized implementations (e.g. cuDNN) are used.
// TODO: rename to ConvolutionGeometry
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
    size_t Groups() const { return m_groups; }

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
    // In addition, all items in Indices are valid source indices so no masking is required in subsequent computation.
    const IntVec&  MpRowIndices() const { return m_mpRowIndices; }
    const IntVec&  Indices() const { return m_indices; }

    // Number of kernels (equal to MapCount if sharing is all true values).
    size_t KernelCount() const { return m_kernelCount; }

    ConvolveGeometry(const TensorShape& inputShape, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& stride,
        const BoolVec& sharing, const BoolVec& autoPad, const TensorShape& lowerPad, const TensorShape& upperPad, const TensorShape& dilation = TensorShape(1),
        const bool ceilOutDim = false, const size_t groups = 1)
        : m_inputShape(inputShape), m_kernelShape(kernelShape), m_mapCount(mapCount), m_stride(stride), m_sharing(sharing),
        m_autoPad(autoPad), m_lowerPad(lowerPad), m_upperPad(upperPad), m_dilation(dilation), m_groups(groups)
    {
        assert(m_inputShape.GetRank() == m_kernelShape.GetRank());
        assert(m_mapCount.GetRank() == 1 || m_mapCount.GetRank() == m_inputShape.GetRank());
        assert(m_stride.GetRank() == 1 || m_stride.GetRank() == m_inputShape.GetRank());
        assert(m_sharing.size() == 1 || m_sharing.size() == m_inputShape.GetRank());
        assert(m_autoPad.size() == 1 || m_autoPad.size() == m_inputShape.GetRank());
        assert(m_lowerPad.GetRank() == 1 || m_lowerPad.GetRank() == m_inputShape.GetRank());
        assert(m_upperPad.GetRank() == 1 || m_upperPad.GetRank() == m_inputShape.GetRank());

        m_outputShape = ComputeOutputShape(m_inputShape, m_kernelShape, m_mapCount, m_stride,
            m_sharing, m_autoPad, m_lowerPad, m_upperPad, m_dilation, m_groups, ceilOutDim);
        assert(m_inputShape.GetRank() == m_outputShape.GetRank());

        // Compute the total number of kernels.
        m_kernelCount = 1;
        for (size_t i = 0; i < inputShape.GetRank(); i++)
            m_kernelCount *= !GetSharing(i) ? m_outputShape[i] : GetMapCount(i);
    }

    bool ComputeConvGeometryExplicit()
    {
        size_t dimCount = m_inputShape.GetRank();
        size_t kernelSize = m_kernelShape.GetNumElements();

        // Compute the "Start" indices.
        m_start.resize(dimCount);
        m_startIndex = 0;
        m_originIndex = 0;
        for (int i = (int)dimCount - 1; i >= 0; i--)
        {
            assert((m_outputShape[i] % GetMapCount(i)) == 0);
            int outPerMap = (int)(m_outputShape[i] / GetMapCount(i));
            // Number of cells between first and last "centers", inclusive.
            int cells = (int)((outPerMap - 1) * GetStride(i) + 1); assert(m_inputShape[i] >= cells);
            // Extra cells, to the left and right of "cells".
            int extra = (int)m_inputShape[i] - cells;
            assert(extra >= 0);

            bool padded = GetAutoPad(i);
            if (padded)
            {
                m_start[i] = extra / 2;
            }
            else
            {
                m_start[i] = ((int)m_kernelShape[i] - 1) / 2;
                int lo = (int)m_lowerPad[m_lowerPad.size() == 1 ? 0 : i];
                int hi = (int)m_upperPad[m_upperPad.size() == 1 ? 0 : i];
                if (lo != 0 || hi != 0)
                {
                    m_start[i] -= lo;
                    assert(m_start[i] >= 0);
                    assert(m_start[i] + cells + (int)m_kernelShape[i] - 1 == m_inputShape[i] + hi + lo);
                }
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

                // 'key' is an offset that can be computed for each output cell.
                // It is used to map from the output cell, to the effective input kernel mask.
                //   (Pooling ignores padded NaN values, so kernel mask is used)
                // The most essential point about 'key' is that there must never be a collision.
                // That is, there must not exist two cells which have a different kernel mask, but producing the same key value.
                // E.g., consider case of input shape: [5], stride: [1], kernel: [3], pad: [1, 1].
                //          input:             [ x1 x2 x3 x4 x5 ]
                //          padded input:  [ NaN x1 x2 x3 x4 x5 NaN ]
                //          output:            [ y1 y2 y3 y4 y5 ]
                //      There are 3 kernel masks for this case.
                //      Case 1: For output cell y1, the effective kernel mask is
                //          kernel mask:    [ 0  1  1  ]
                //       => padded input:   [ __ x1 x2 ]
                //      Case 2: For output cell y2(and y3,y4), the effective kernel mask is
                //          kernel mask:    [ 1  1  1  ]
                //       => padded input:   [ x1 x2 x3 ]
                //      Case 3: For output cell y5, the effective kernel mask is
                //          kernel mask:    [ 1  1  0  ]
                //       => padded input:   [ x4 x5 __ ]
                //      Thus there will be a total of 3 different 'key' values. y2, y3 and y4 should produce the same 'key' values. 
                int width = (int)m_kernelShape[i];
                int half = (width - 1) / 2;
                // 'min' stands for the first input index along this axis that is covered by the current kernel.
                //  if negative, -min is equal to the number of padded cells that are covered.
                int min = coord - half;
                // 'lim' stands for the last input index + 1 along this axis that is covered by the current kernel.
                //  if lim > inputShape, lim - inputShape is equal to the number of padded cells that are covered.
                int lim = min + width;
                if (min < 0)
                    // Case 1.
                    // When min < 0, the current kernel covers (lower)padded values, thus the offset is recorded so that
                    // a map from 'key' to the kernel mask can be established.
                    dkey[i] = min;
                else if (lim > m_inputShape[i])
                    // Case 3.
                    // Similarly, when lim > inputShape, the current kernel covers (upper)padded values.
                    dkey[i] = lim - (int)m_inputShape[i];
                else
                    // Case 2.
                    // No padded values are covered, the kernel mask is all ones and is shared.
                    dkey[i] = 0;
                bool isAutoPad = GetAutoPad(i);
                // Ignore hi/lo values when auto padding is true.
                int hi = isAutoPad ? 0 : (int)m_upperPad[m_upperPad.size() == 1 ? 0 : i];
                int lo = isAutoPad ? 0 : (int)m_lowerPad[m_lowerPad.size() == 1 ? 0 : i];
                // dk contributes to the 'key' value for this particular axis.
                // There are two properties for dk that must be satisfied.
                //  1. dk \in (0, width + hi + lo].
                //  2. there must not exist two cells which have a different kernel mask, but producing the same dk value.
                // Both are required such that colision is avoided for 'key' when dk values for different axes are accumulated together.
                // With careful calculations, we can show
                // dk \in | [ half + 1, half + lo + 1)                      , min < 0
                //        | ( half + lo + 1, width + hi + lo - half + 1)    , lim - inputShape > 0
                //        | { half + lo + 1}                                , otherwise
                // where half = (width - 1) / 2 and width = kernelShape.
                // Since half \in [0, width), we can conclude dk \in [1, width + hi + lo + 1) = (0, width + hi + lo].
                int dk = dkey[i] + half + lo + 1;
                assert(0 < dk);
                assert(dk <= (width + hi + lo));
                key = key * (width + hi + lo) + dk;
            }
            assert(cur == 0);
            assert(0 <= kern);
            assert(kern < m_kernelCount);
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
            assert(kern < m_kernelCount);
            m_mpRowCol[row] = col;
            m_mpRowIwht[row] = kern * (int)kernelSize;
        }
        return true;
    }

    size_t GetStride(size_t dim) const
    {
        assert(m_stride.size() == 1 || dim < m_stride.size());
        return m_stride[m_stride.size() == 1 ? 0 : dim];
    }

    size_t GetDilation(size_t dim) const
    {
        assert(m_dilation.size() == 1 || dim < m_dilation.size());
        return m_dilation[m_dilation.size() == 1 ? 0 : dim];
    }

    size_t GetMapCount(size_t dim) const
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

    bool GetSharing(size_t dim) const
    {
        assert(m_sharing.size() == 1 || dim < m_sharing.size());
        return m_sharing[m_sharing.size() == 1 ? 0 : dim];
    }

    bool GetAutoPad(size_t dim) const
    {
        assert(m_autoPad.size() == 1 || dim < m_autoPad.size());
        return m_autoPad[m_autoPad.size() == 1 ? 0 : dim];
    }

    int GetLowerPad(size_t dim) const
    {
        if (!GetAutoPad(dim))
            return (int)m_lowerPad[m_lowerPad.size() == 1 ? 0 : dim];

        int dilation = (int)GetDilation(dim);
        int kernSize = ((int)m_kernelShape[dim] - 1) * dilation + 1;
        int inpSize = (int)m_inputShape[dim];
        int outSize = (int)m_outputShape[dim];
        int stride = (int)GetStride(dim);

        // Taken from computation in ConvolveGeometry ctor.
        // Number of cells between first and last "centers", inclusive.
        int cells = (outSize - 1) * stride + 1;
        // Extra cells, to the left and right of "cells".
        int extra = inpSize - cells;
        int center = extra / 2;
        return -(center - (kernSize - 1) / 2);
    }

    // GetUpperPad
    // 
    // There will be four cases
    //      kernelSize  extraSize   padSizes                cuDnn & MKL (they only support symmetric padding)
    // 1.   odd         even        lower = upper           supported
    // 2.   even        odd         lower = upper           supported
    // 3.   odd         odd         lower = upper + 1       supported with extra 1 padding on upperPad
    // 4.   even        even        lower = upper - 1       unsupported
    //
    // extra size = (dim - 1) % stride
    //
    // So for cases where lower = upper + 1. We can simply decide to pad one extra for upperPad, 
    // as it will yield the same shape and value results.
    // However, for cases where lower = upper - 1. We cannot pad the extra for lowerPad, 
    // as it will change the center of the kernel, and produce different value and maybe different shape results. 
    //
    // Parameter: 
    //  bool trySymmetricAutoPad : if set to true, this function will return symmetric padding for case 3 by padding 1 extra on upperPad.
    //                             This parameter is ignored if auto padding is not enabled. 
    int GetUpperPad(size_t dim, bool trySymmetricAutoPad = false) const
    {
        if (!GetAutoPad(dim))
            return (int)m_upperPad[m_upperPad.size() == 1 ? 0 : dim];
       
        int dilation = (int)GetDilation(dim);
        int kernSize = ((int)m_kernelShape[dim] - 1) * dilation + 1;
        int inpSize = (int)m_inputShape[dim];
        int outSize = (int)m_outputShape[dim];
        int stride = (int)GetStride(dim);

        // Taken from computation in ConvolveGeometry ctor.
        // Number of cells between first and last "centers", inclusive.
        int cells = (outSize - 1) * stride + 1;
        // Extra cells, to the left and right of "cells".
        int extra = inpSize - cells;
        int center = extra / 2;
        int upperPad = (kernSize - 1) - (kernSize - 1) / 2 - (extra - center);

        if (trySymmetricAutoPad && (kernSize % 2 == 1) && (extra % 2 == 1))
        {
            // case 3: pad extra 1 for upperPad to enable symmetric padding. 
            upperPad++;
        }
        return upperPad;
    }

    // Return if padding is enabled for input channel axis. 
    bool IsPaddingOverChannelAxis() const
    {
        size_t channelIdx = m_inputShape.GetRank() - 1;
        assert(m_inputShape.GetRank() >= 1);
        // check for lowerPad value. This value is incorrect when out channel size > 1. Check if channel stride is >= channel size in that case.
        return (GetLowerPad(channelIdx) > 0) && (GetStride(channelIdx) < m_inputShape[channelIdx]);
    }

    // Computes output shape given input shape and other convolution parameters.
    static TensorShape ComputeOutputShape(const TensorShape& inputShape, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& stride,
                                          const BoolVec& sharing, const BoolVec& autoPad, const TensorShape& lowerPad, const TensorShape& upperPad,
                                          const TensorShape& dilation=TensorShape(1), const size_t groups=1, const bool ceilOutDim = false, const bool needsDynamicValidation = false,
                                          const bool isFinalValidationPass = false)
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
            auto kernelShape_i = (i == kernelShape.GetRank() - 1) ? kernelShape[i] * groups : kernelShape[i];

            // If lowerPad and upperPad are specified, include them in the minimum size check
            // for input image that is done below. Otherwise (if autoPad is specified), just 
            // check against kernel size only.
            auto lowerPadValForSizeCheck = autoPad[autoPad.size() == 1 ? 0 : i] ? 0 : lowerPad[lowerPad.size() == 1 ? 0 : i];
            auto upperPadValForSizeCheck = autoPad[autoPad.size() == 1 ? 0 : i] ? 0 : upperPad[upperPad.size() == 1 ? 0 : i];
            if (kernelShape_i > (inputShape[i] + upperPadValForSizeCheck + lowerPadValForSizeCheck) )
            {
                if(isFinalValidationPass || !needsDynamicValidation)
                    InvalidArgument("Convolution operation requires that kernel dim %d <= input dim %d.", (int)kernelShape_i, (int)inputShape[i]);
                else
                {
                    dimsOutput[i] = 1; // 1 is a placeholder till all shapes are resolved.
                    continue;
                }
            }

            size_t delta = stride[stride.GetRank() == 1 ? 0 : i];
            size_t dim = inputShape[i];
            bool autoPadCur = autoPad[autoPad.size() == 1 ? 0 : i];
            size_t lo = lowerPad[lowerPad.size() == 1 ? 0 : i];
            size_t hi = upperPad[upperPad.size() == 1 ? 0 : i];
            size_t dil = dilation[dilation.GetRank() == 1 ? 0 : i];

            if (autoPadCur)
            {
                dim += dil * (kernelShape_i - 1);
            }
            else
            {
                dim += lo + hi;
            }

            size_t effectiveKernelShape = (kernelShape_i - 1) * dil + 1;
            float preciseDimOut = (float)(dim - effectiveKernelShape) / delta + 1;
            size_t dimOut = static_cast<size_t>(ceilOutDim ? ceil(preciseDimOut) : floor(preciseDimOut));
            // When LowerPad and/or UpperPad are specified (i.e. > 0), we insist that the kernel applications
            // fill the entire space.
            if (!autoPadCur && (lo > 0 || hi > 0))
            {
                size_t size = (dimOut - 1) * delta + kernelShape_i;
                // size must be >= (lo + inputShape[i]) to cover all original input space(excluding padding).
                if (size < (dim - hi))
                    InvalidArgument("Convolution requires that kernel fills the entire space if auto-padding is disabled.");
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
        UNUSED(mapCountTotal);
        UNUSED(sizeOut);

        return dimsOut;
    }

    // Computes input shape given output shape and other convolution parameters.
    // Used in deconvolution operation.
    static TensorShape ComputeInputShape(const TensorShape& outputShape, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& stride,
                                         const BoolVec& sharing, const BoolVec& autoPad, const TensorShape& lowerPad, const TensorShape& upperPad, 
                                         const TensorShape& dilation=TensorShape(1), const size_t groups=1, bool ceilOutDim = false, const bool needsDynamicValidation = false,
                                         const bool isFinalValidationPass = false)
    {
        UNUSED(ceilOutDim);
        UNUSED(dilation);
        UNUSED(groups);
        UNUSED(needsDynamicValidation);
        UNUSED(isFinalValidationPass);
        if (outputShape.GetRank() != kernelShape.GetRank())
            InvalidArgument("Convolution output and kernel tensors must have the same rank.");
        if (mapCount.GetRank() != 1 && outputShape.GetRank() != mapCount.GetRank())
            InvalidArgument("Convolution map tensor must have rank 1 or the same as the output tensor.");
        if (stride.GetRank() != 1 && outputShape.GetRank() != stride.GetRank())
            InvalidArgument("Convolution stride tensor must have rank 1 or the same as the output tensor.");
        if (sharing.size() != 1 && outputShape.GetRank() != sharing.size())
            InvalidArgument("Convolution sharing tensor must have rank 1 or the same as the output tensor.");
        if (autoPad.size() != 1 && outputShape.GetRank() != autoPad.size())
            InvalidArgument("Convolution padding tensor must have rank 1 or the same as the output tensor.");
        if (lowerPad.GetRank() != 1 && outputShape.GetRank() != lowerPad.GetRank())
            InvalidArgument("Convolution lower pad tensor must have rank 1 or the same as the output tensor.");
        if (upperPad.GetRank() != 1 && outputShape.GetRank() != upperPad.GetRank())
            InvalidArgument("Convolution upper pad tensor must have rank 1 or the same as the output tensor.");

        SmallVector<size_t> dimsInput(outputShape.GetRank());
        for (size_t i = 0; i < outputShape.GetRank(); i++)
        {
            assert(outputShape[i] >= 1);

            size_t delta = stride[stride.GetRank() == 1 ? 0 : i];
            size_t dim = outputShape[i];
            // Input dimension does not include output map count.
            size_t curMapCount = 1;
            if (mapCount.size() > 1)
                curMapCount = mapCount[i];
            else if (i == outputShape.GetRank() - 1)
                curMapCount = mapCount[0];
            assert((dim % curMapCount) == 0);
            dim /= curMapCount;

            bool autoPadCur = autoPad[autoPad.size() == 1 ? 0 : i];
            size_t lo = lowerPad[lowerPad.size() == 1 ? 0 : i];
            size_t hi = upperPad[upperPad.size() == 1 ? 0 : i];
            size_t dimIn = (dim - 1) * delta;
            // We need to be able to restore any input size from the output, not just the one
            // that does not require padding. For example, if output is 14, stride 2 and
            // desired input is 28 then padded input will be 31. In this case if autopadding is enabled,
            // the input will 27 as (27 - 1) / 2 + 1 == 14.
            if (autoPadCur)
                dimIn += 1;
            else
                dimIn += (int64_t)kernelShape[i] - (lo + hi);
            // When LowerPad and/or UpperPad are specified (i.e. > 0), we insist that the kernel applications
            // fill the entire space.
            if (!autoPadCur && (lo > 0 || hi > 0))
            {
                size_t size = (dimIn - kernelShape[i] + lo + hi) / delta + 1;
                if (size != dim)
                    InvalidArgument("Convolution requires that kernel fills the entire space if auto-padding is disabled.");
            }

            dimsInput[i] = dimIn;
        }

        return TensorShape(dimsInput);
    }

    // This is for a special case handling, where ceilOutDim = True and cntkAutoPadding = False.
    // In CNTK, no paddings should be generated since autoPadding is False. 
    // Yet due to ceilOutDim = True, the outputShape might end up 1 size larger, requiring
    // an input of dimension that actually exceeds the current input. 
    // E.g.
    //      input dim: 112, kernel size: 3, stride: 2
    // The output dim will end up 56. 
    // This will require an input dim of 113. 
    // This function returns the number of extra cells required.
    // I.e. 1 in the above example: 113 - 112 = 1. 
    int GetExtraCellsCount(size_t dim) const
    {
        int dilation = (int)GetDilation(dim);
        int kernSize = ((int)m_kernelShape[dim] - 1) * dilation + 1;
        int inpSize = (int)m_inputShape[dim];
        int outSize = (int)m_outputShape[dim];
        int stride = (int)GetStride(dim);

        return (outSize - 1) * stride + kernSize - inpSize;
    }

    // Used in unit tests and during debugging.
    operator std::string() const
    {
        std::ostringstream res;
        res << "Input: " << (string)InputShape();
        res << ", Output: " << (string)OutputShape();
        res << ", Kernel: " << (string)KernelShape();
        res << ", Map: " << (string)MapCount();
        res << ", Stride: " << (string)Stride();
        res << ", Sharing: (";
        std::copy(begin(Sharing()), end(Sharing()) - 1, std::ostream_iterator<bool>(res, ", "));
        res << Sharing().back() << ")";
        res << ", AutoPad: (";
        std::copy(begin(AutoPad()), end(AutoPad()) - 1, std::ostream_iterator<bool>(res, ", "));
        res << AutoPad().back() << ")";
        res << ", LowerPad: " << (string)LowerPad();
        res << ", UpperPad: " << (string)UpperPad();
        return res.str();
    }

    // For MKL, if auto padding is enabled, in some cases we can convert asymmetric padding to symmetric padding,
    // with the same output shape and value. 
    bool IsAsymmetricPadding(bool useMKL) const
    {
        for (size_t i = 0; i < KernelShape().size(); i++)
        {
            auto lowerPad = GetLowerPad(i);
            auto upperPad = GetUpperPad(i, useMKL);
            auto stride = GetStride(i);
            if ((lowerPad != upperPad) && (stride < InputShape()[i]))
            {
                return true;
            }
        }
        return false;
    }

    DISABLE_COPY_AND_MOVE(ConvolveGeometry);

private:
    TensorShape m_inputShape;
    TensorShape m_outputShape;
    TensorShape m_kernelShape;
    TensorShape m_mapCount;
    TensorShape m_stride;
    TensorShape m_dilation;
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

    size_t m_kernelCount;
    size_t m_groups;
};

using ConvolveGeometryPtr = std::shared_ptr<ConvolveGeometry>;

} } }
