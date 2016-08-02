//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// The file contains CUDA kernels that are used in reference convolution
// engine. All these kernels look very similar as they use the same
// idea of precomputed maps described in ConvolveGeometry.h
// That is, 'mpRowCol' maps each convolution output to the start of the
// input. 'mpRowIwht', 'mpRowRun' and 'runs' provide maps that allow
// to get indices of the active weight when applying the convolution.
// See ConvolveGeometry.h (MpRowCol, MpRowIwht etc) for more details.
// -----------------------------------------------------------------------

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
__global__ void kROIPoolingForward(const int total_iterations,
    const int num_rois, const int img_count,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const ElemType* src, 
    const ElemType* roi_data, ElemType* dst, ElemType* argmax)
{
    // index loops over all total_rois*c*pooled_height*pooled_width output locations.
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < (total_iterations); index += blockDim.x * gridDim.x) {
        
        // we want NCHW
        // (n, c, ph, pw) is an element in the pooled output
        // n is the global ROI index (the new batch index)
        int roi_size = channels * pooled_height * pooled_width;
        int n = index / roi_size;
        int effective_index = index % roi_size;
        int pw = effective_index % pooled_width;
        int ph = (effective_index / pooled_width) % pooled_height;
        int c = effective_index / pooled_height / pooled_width;

        roi_data += n * 4;

        // cast relative numbers into floats.
        float rx = static_cast<float>(roi_data[0]),
            ry = static_cast<float>(roi_data[1]),
            rw = static_cast<float>(roi_data[2]),
            rh = static_cast<float>(roi_data[3]);

        // roi data is relative to original image size
        int roi_start_w = static_cast<int>(roundf(rx * width));
        int roi_start_h = static_cast<int>(roundf(ry * height));
        int roi_width = static_cast<int>(max(roundf(rw * width), 1.0));
        int roi_height = static_cast<int>(max(roundf(rh * height), 1.0));
        
        float winH = static_cast<float>(roi_height)
            / static_cast<float>(pooled_height);
        float winW = static_cast<float>(roi_width)
            / static_cast<float>(pooled_width);
        
        // compute window for this output location.
        int hstart = static_cast<int>(floorf(static_cast<float>(ph) * winH));
        int wstart = static_cast<int>(floorf(static_cast<float>(pw) * winW));
        int hend = static_cast<int>(ceilf(static_cast<float>(ph + 1) * winH));
        int wend = static_cast<int>(ceilf(static_cast<float>(pw + 1) * winW));
        
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        
        bool isempty = (hend <= hstart) || (wend <= wstart);
        // Define an empty pooling region to be zero
        float maxval = isempty ? 0 : -FLT_MAX;
        int maxidx = -1;

        // img_idx = n (global roi index) / num_rois (rois per image)
        int img_idx = n / num_rois;
        src += img_idx * channels * height * width;
        
        for (int h = hstart; h < hend; h++) {
            for (int w = wstart; w < wend; w++) {
                int src_index = w + h*width + c*height*width;
                //int src_index = 0;
                if (src[src_index] > maxval) {
                    maxval = src[src_index];
                    maxidx = src_index;
                }
            }
        }
        dst[index] = maxval;
        argmax[index] = maxidx;
    }
}

template <typename ElemType>
__global__ void kROIPoolingBackward(const int total_iterations,
    const int num_rois, const int img_count,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const ElemType* pooled_grad,
    const ElemType* roi_data, ElemType* grad, const ElemType* argmax)
{
    // index loops over all input locations (locations in the original input tensor).
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < (total_iterations); index += blockDim.x * gridDim.x) {

        // we want NCHW
        // (n, c, h, w) is an element in the input image
        // n is the image index
        //int img_size = channels * width * height;
        //int n = index / img_size;
        //int effective_index = index % img_size;
        //int w = effective_index % width;
        //int h = (effective_index / width) % height;
        //int c = effective_index / width / height;

        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;

        // compute range of ROIs corresponding to this image:
        int roi_min = n * num_rois;
        int roi_max = (n + 1) * num_rois;

        ElemType gradient = 0;

        for (int roi_n = roi_min; roi_n < roi_max; roi_n++) {
            const ElemType* roi_offset = roi_data + roi_n * 4;
            float rx = static_cast<float>(roi_offset[0]),
                ry = static_cast<float>(roi_offset[1]),
                rw = static_cast<float>(roi_offset[2]),
                rh = static_cast<float>(roi_offset[3]);

            // roi data is relative to original image size
            int roi_start_w = static_cast<int>(roundf(rx * width));
            int roi_start_h = static_cast<int>(roundf(ry * height));
            int roi_width = static_cast<int>(max(roundf(rw * width), 1.0));
            int roi_height = static_cast<int>(max(roundf(rh * height), 1.0));

            // skip this ROI if it doesn't contain our input location.
            const bool in_roi = (w >= roi_start_w && w < roi_start_w + roi_width &&
                h >= roi_start_h && h < roi_start_h + roi_height);

            if (!in_roi) {
                continue;
            }

            float winH = static_cast<float>(roi_height)
                / static_cast<float>(pooled_height);
            float winW = static_cast<float>(roi_width)
                / static_cast<float>(pooled_width);

            // what pooled nodes in the output for this ROI could have pooled this input location?
            int phstart = static_cast<int>(floorf(static_cast<float>(h - roi_start_h) / winH));
            int phend = static_cast<int>(ceilf(static_cast<float>(h - roi_start_h + 1) / winH));
            int pwstart = static_cast<int>(floorf(static_cast<float>(w - roi_start_w) / winW));
            int pwend = static_cast<int>(ceilf(static_cast<float>(w - roi_start_w + 1) / winW));

            phstart = min(max(phstart, 0), pooled_height);
            phend = min(max(phend, 0), pooled_height);
            pwstart = min(max(pwstart, 0), pooled_width);
            pwend = min(max(pwend, 0), pooled_width);

            // go right up to this channel of this ROI.
            int offset = (roi_n * channels + c) * pooled_width * pooled_height;
            const ElemType* offset_pool_grad = pooled_grad + offset;
            const ElemType* offset_argmax = argmax + offset;

            for (int ph = phstart; ph < phend; ph++) {
                for (int pw = pwstart; pw < pwend; pw++) {
                    if (offset_argmax[ph * pooled_width + pw] == (h * width + w)) {
                        gradient += offset_pool_grad[ph * pooled_width + pw];
                    }
                }
            }
        }
        grad[index] = gradient;
    }
}

template <typename ElemType>
__global__ void kMaxUnpooling(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                              const ElemType* __restrict__ src, const ElemType* poolIn, int srcVecSize,
                              ElemType* dst, int dstVecSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= srcVecSize)
        return;

    src    += blockIdx.y * srcVecSize;
    poolIn += blockIdx.y * dstVecSize;
    dst    += blockIdx.y * dstVecSize;

    for (int sample = blockIdx.y; sample < batchSize; sample += gridDim.y)
    {
        int colBase = mpRowCol[row];
        assert(0 <= colBase && colBase < dstVecSize);

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        ElemType curMax = poolIn[colBase + indices[i0]];
        ElemType prevMax = curMax;
        int imax = 0;
        for (int i = 1; i < size; i++)
        {
            int dcol = indices[i0 + i];
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
            curMax = max(curMax, poolIn[colBase + dcol]);
            if (curMax > prevMax)
            {
                prevMax = curMax;
                imax = i;
            }

        }

        int dcol = indices[i0 + imax];
        assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);

        dst[colBase + dcol] = src[row];

        src    += blockIdx.y * srcVecSize;
        poolIn += blockIdx.y * dstVecSize;
        dst    += blockIdx.y * dstVecSize;
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

}}}
