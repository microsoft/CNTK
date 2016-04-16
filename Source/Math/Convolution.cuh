//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType>
__global__ void kConvolutionForward(int batchSize, const ElemType* __restrict__ kernel,
                                    const int* mpRowCol, const int* mpRowIwht,
                                    const int* mpRowRun, const int* __restrict__ runs,
                                    const ElemType* __restrict__ src, int srcVecSize,
                                    ElemType* dst, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dstVecSize)
        return;

    src += blockIdx.y * srcVecSize;
    dst += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];
        assert(0 <= colBase && colBase < srcVecSize);

        ElemType sum = 0;
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < srcVecSize);
            sum += kernel[ivBase + skip + i] * src[colBase + dcol];
        }
        dst[row] = sum;

        src += blockDim.y * srcVecSize;
        dst += blockDim.y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kConvolutionBackwardData(int batchSize, const ElemType* __restrict__ kernel,
                                         const int* mpRowCol, const int* mpRowIwht,
                                         const int* mpRowRun, const int* __restrict__ runs,
                                         const ElemType* __restrict__ srcGrad, int srcVecSize,
                                         ElemType* grad, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= srcVecSize)
        return;

    srcGrad += blockIdx.y * srcVecSize;
    grad += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];
        assert(0 <= colBase && colBase < dstVecSize);

        ElemType g = srcGrad[row];
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
            atomicAdd(&grad[colBase + dcol], g * kernel[ivBase + skip + i]);
        }

        srcGrad += blockDim.y * srcVecSize;
        grad += blockDim.y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kConvolutionBackwardKernel(int batchSize, int inVecSize, int outVecSize,
                                           const ElemType* __restrict__ in,
                                           const int* mpRowCol, const int* mpRowIwht,
                                           const int* mpRowRun, const int* __restrict__ runs,
                                           const ElemType* __restrict__ srcGrad,
                                           ElemType* kernelGrad)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outVecSize)
        return;

    in += blockIdx.y * inVecSize;
    srcGrad += blockIdx.y * outVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];
        assert(0 <= colBase && colBase < inVecSize);

        ElemType g = srcGrad[row];
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < inVecSize);
            atomicAdd(&kernelGrad[ivBase + skip + i], g * in[colBase + dcol]);
        }

        in += blockDim.y * inVecSize;
        srcGrad += blockDim.y * outVecSize;
    }
}

template <typename ElemType>
__global__ void kMaxPoolingForward(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                   const ElemType* __restrict__ src, int srcVecSize,
                                   ElemType* dst, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dstVecSize)
        return;

    src += blockIdx.y * srcVecSize;
    dst += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        assert(0 <= colBase && colBase < srcVecSize);

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        ElemType res = src[colBase + indices[i0]];
        for (int i = 1; i < size; i++)
        {
            int dcol = indices[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < srcVecSize);
            res = max(res, src[colBase + dcol]);
        }
        dst[row] = res;

        src += blockDim.y * srcVecSize;
        dst += blockDim.y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kMaxPoolingBackward(int batchSize, const ElemType* out, const ElemType* in,
                                    const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                    const ElemType* __restrict__ srcGrad, int srcVecSize,
                                    ElemType* grad, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= srcVecSize)
        return;

    in += blockIdx.y * dstVecSize;
    out += blockIdx.y * srcVecSize;
    srcGrad += blockIdx.y * srcVecSize;
    grad += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        assert(0 <= colBase && colBase < dstVecSize);

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        assert(size > 0);
        ElemType g = srcGrad[row];
        ElemType m = out[row];
        for (int i = 0; i < size; i++)
        {
            int dcol = indices[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
            if (in[colBase + dcol] >= m)
                atomicAdd(&grad[colBase + dcol], g);
        }

        in += blockDim.y * dstVecSize;
        out += blockDim.y * srcVecSize;
        srcGrad += blockDim.y * srcVecSize;
        grad += blockDim.y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kAveragePoolingForward(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                       const ElemType* __restrict__ src, int srcVecSize,
                                       ElemType* dst, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dstVecSize)
        return;

    src += blockIdx.y * srcVecSize;
    dst += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        assert(0 <= colBase && colBase < srcVecSize);

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        ElemType sum = 0;
        for (int i = 0; i < size; i++)
        {
            int dcol = indices[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < srcVecSize);
            sum += src[colBase + dcol];
        }
        dst[row] = sum / size;

        src += blockDim.y * srcVecSize;
        dst += blockDim.y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kAveragePoolingBackward(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                        const ElemType* __restrict__ srcGrad, int srcVecSize,
                                        ElemType* grad, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= srcVecSize)
        return;

    srcGrad += blockIdx.y * srcVecSize;
    grad += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        assert(0 <= colBase && colBase < dstVecSize);

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        assert(size > 0);
        ElemType g = srcGrad[row] / size;
        for (int i = 0; i < size; i++)
        {
            int dcol = indices[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
            atomicAdd(&grad[colBase + dcol], g);
        }

        srcGrad += blockDim.y * srcVecSize;
        grad += blockDim.y * dstVecSize;
    }
}

} } }
