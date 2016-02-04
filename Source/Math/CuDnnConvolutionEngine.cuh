//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <device_launch_parameters.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#pragma warning(disable : 4127)
#pragma warning(disable : 4201)
#pragma warning(disable : 4515)
#endif
#include <cub/cub.cuh>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

    size_t RoundUpToMultiple(size_t n, size_t blockSize)
    {
        return (n + blockSize - 1) / blockSize;
    }

    template <int UnrollFactor, typename T>
    __device__ __forceinline__ void LoadElements(const T* src, T dst[UnrollFactor])
    {
    #pragma unroll
        for (int i = 0; i < UnrollFactor; i++)
            dst[i] = src[i];
    }

    template <>
    __device__ __forceinline__ void LoadElements<2, float>(const float* src, float dst[2])
    {
        // src must be aligned at 8 bytes boundary.
        assert(reinterpret_cast<uintptr_t>(src) % (sizeof(dst)) == 0);
        auto v = *(const float2*)src;
        dst[0] = v.x;
        dst[1] = v.y;
    }

    template <>
    __device__ __forceinline__ void LoadElements<4, float>(const float* src, float dst[4])
    {
        // src must be aligned at 16 bytes boundary.
        assert(reinterpret_cast<uintptr_t>(src) % (sizeof(dst)) == 0);
        // Can do the following instead (use ld.global.nc.* on CC 3.5+):
        // asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(src));
        // Similar for shared memory (e.g. ld.shared.*)
        auto v = *(const float4*)src;
        dst[0] = v.x;
        dst[1] = v.y;
        dst[2] = v.z;
        dst[3] = v.w;
    }

    template <int UnrollFactor, typename T>
    __device__ __forceinline__ void StoreElements(const T src[UnrollFactor], T* dst)
    {
    #pragma unroll
        for (int i = 0; i < UnrollFactor; i++)
            dst[i] = src[i];
    }

    template <>
    __device__ __forceinline__ void StoreElements<2, float>(const float src[2], float* dst)
    {
        // dst must be aligned at 8 bytes boundary.
        assert(reinterpret_cast<uintptr_t>(dst) % (sizeof(src)) == 0);
        float2 v;
        v.x = src[0];
        v.y = src[1];
        *(reinterpret_cast<float2*>(dst)) = v;
    }

    template <>
    __device__ __forceinline__ void StoreElements<4, float>(const float src[4], float* dst)
    {
        // dst must be aligned at 16 bytes boundary.
        assert(reinterpret_cast<uintptr_t>(dst) % (sizeof(src)) == 0);
        float4 v;
        v.x = src[0];
        v.y = src[1];
        v.z = src[2];
        v.w = src[3];
        *(reinterpret_cast<float4*>(dst)) = v;
    }

    struct Operations
    {
        template <typename T>
        struct RSqrt
        {
        };

        template <>
        struct RSqrt<float>
        {
            __device__ float operator()(float a)
            {
                //return __frsqrt_rn(a);
                return rsqrtf(a);
            }
        };

        template <>
        struct RSqrt<double>
        {
            __device__ double operator()(double a)
            {
                return rsqrt(a);
            }
        };
    };

    // The kernel implements online, parallel and numerically stable algorithm 
    // for computing batch mean and variance (here inverse standard deviation) with one pass over the data.
    // It uses algorithm described in: http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    template <int BlockDimX, int BlockDimY, int UnrollFactor, typename T>
    __global__ void kComputeBatchMeanAndInvStdDev(int vectorSize, int batchSize, const T* x, double epsilon, T* xMean, T* xInvStdDev)
    {
        static_assert(BlockDimX * UnrollFactor == CUB_PTX_WARP_THREADS, "BlockDimX * UnrollFactor must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert((vectorSize % UnrollFactor) == 0);
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
        assert(isfinite(epsilon) && epsilon > 0);

        int irowSrcBase = (blockIdx.x * BlockDimX + threadIdx.x) * UnrollFactor;
        if (irowSrcBase >= vectorSize)
            return;
        assert(irowSrcBase + UnrollFactor <= vectorSize);

        int n = 0;
        T mean[UnrollFactor];
        T m2[UnrollFactor];
    #pragma unroll
        for (int k = 0; k < UnrollFactor; k++)
        {
            mean[k] = 0;
            m2[k] = 0;
        }

        int icolSrc = threadIdx.y;
        const T* psrc = x + static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
        for (; icolSrc < batchSize; icolSrc += BlockDimY)
        {
            n++;
            T curVal[UnrollFactor];
            LoadElements<UnrollFactor>(psrc, curVal);
            // No need for separate unrolling, SASS looks good.
    #pragma unroll
            for (int k = 0; k < UnrollFactor; k++)
            {
                T d = curVal[k] - mean[k];
                // REVIEW alexeyk: we enabled fast CUDA math in CNTK so division below will be approximate, is this a problem?
                // Using precise math slows down the code by about 40%.
                mean[k] += d / n;
                m2[k] += d * (curVal[k] - mean[k]);
            }
            psrc += vectorSize * BlockDimY;
        }

        const int tid = threadIdx.y * BlockDimX + threadIdx.x;
        const int laneId = tid & 0x1f;
        // First, reduce within warp using shuffle.
        if (n > 0)
        {
    #pragma unroll
            for (int i = 1; i < CUB_PTX_WARP_THREADS / BlockDimX; i *= 2)
            {
                int srcLane = laneId + BlockDimX * i;
                int n2 = cub::ShuffleIndex(n, srcLane);
                int nsum = n + n2;
                T d[UnrollFactor];
    #pragma unroll
                for (int k = 0; k < UnrollFactor; k++)
                {
                    d[k] = cub::ShuffleIndex(mean[k], srcLane) - mean[k];
                    T dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += cub::ShuffleIndex(m2[k], srcLane) + d[k] * n * dScaled;
                }
                n = nsum;
            }
        }

        // Storage for each warp in a thread block. First warp ("accumulator") holds 
        // final results so it does not need shared memory.
        const int cwarp = BlockDimX * BlockDimY / CUB_PTX_WARP_THREADS;
        __shared__ T meanRes[BlockDimX * UnrollFactor][cwarp - 1];
        __shared__ T m2Res[BlockDimX * UnrollFactor][cwarp - 1];
        __shared__ int nRes[cwarp - 1];

        // Each warp (except warp0) will write accumulated results to shared memory.
        const int iwarp = tid / CUB_PTX_WARP_THREADS;
        if (iwarp > 0 && laneId < BlockDimX)
        {
            if (laneId == 0)
                nRes[iwarp - 1] = n;
    #pragma unroll
            for (int k = 0; k < UnrollFactor; k++)
            {
                meanRes[laneId * UnrollFactor + k][iwarp - 1] = mean[k];
                m2Res[laneId * UnrollFactor + k][iwarp - 1] = m2[k];
            }
        }
        __syncthreads();

        // Accumulate and write final results.
        if (threadIdx.y == 0)
        {
            // Use simple loop as number of warps is small, 8 at max.
    #pragma unroll
            for (int i = 0; i < cwarp - 1; i++)
            {
                int n2 = nRes[i];
                int nsum = n + n2;
                T d[UnrollFactor];
    #pragma unroll
                for (int k = 0; k < UnrollFactor; k++)
                {
                    d[k] = meanRes[threadIdx.x * UnrollFactor + k][i] - mean[k];
                    T dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += m2Res[threadIdx.x * UnrollFactor + k][i] + d[k] * n * dScaled;
                }
                n = nsum;
            }
            size_t idxDstBase = (blockIdx.x * BlockDimX + threadIdx.x) * UnrollFactor;
            StoreElements<UnrollFactor>(mean, xMean + idxDstBase);
            Operations::RSqrt<T> rsqrtOp;
    #pragma unroll
            for (int k = 0; k < UnrollFactor; k++)
            {
                m2[k] = rsqrtOp(static_cast<T>(m2[k] / batchSize + epsilon));
            }
            StoreElements<UnrollFactor>(mean, xMean + idxDstBase);
            StoreElements<UnrollFactor>(m2, xInvStdDev + idxDstBase);
        }
    }

    template <int UnrollFactor, typename T>
    void ComputeBatchMeanAndInvStdDevImpl(size_t vectorSize, size_t batchSize, const T* x, double epsilon, T* xMean, T* xInvStdDev, cudaStream_t stream)
    {
        assert((vectorSize % UnrollFactor) == 0);

        const int BlockDimX = 32 / UnrollFactor;
        const int BlockDimY = 4 * UnrollFactor;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create grid with only one block in y(batch)-dimension as kernel uses striding.
        auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * UnrollFactor)));
        kComputeBatchMeanAndInvStdDev<BlockDimX, BlockDimY, UnrollFactor><<<gdim, bdim, 0, stream>>>(
            static_cast<int>(vectorSize), static_cast<int>(batchSize), x, epsilon, xMean, xInvStdDev);
    }

    template <typename T>
    cudaError_t ComputeBatchMeanAndInvStdDev(size_t vectorSize, size_t batchSize, const T* x, double epsilon, T* xMean, T* xInvStdDev, cudaStream_t stream)
    {
        if ((vectorSize % 4) == 0)
            ComputeBatchMeanAndInvStdDevImpl<4>(vectorSize, batchSize, x, epsilon, xMean, xInvStdDev, stream);
        else if ((vectorSize % 2) == 0)
            ComputeBatchMeanAndInvStdDevImpl<2>(vectorSize, batchSize, x, epsilon, xMean, xInvStdDev, stream);
        else
            ComputeBatchMeanAndInvStdDevImpl<1>(vectorSize, batchSize, x, epsilon, xMean, xInvStdDev, stream);
        return cudaGetLastError();
    }

    template <int BlockDimX, int BlockDimY, int UnrollFactor, typename T>
    __global__ void kNormalizeBatchTraining(int vectorSize, int batchSize, const T* x, T* y,
        const T* bnScale, const T* bnBias, const T* batchMean, const T* batchInvStdDev)
    {
        static_assert(BlockDimX * UnrollFactor == CUB_PTX_WARP_THREADS, "BlockDimX * UnrollFactor must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert((vectorSize % UnrollFactor) == 0);
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        int irowBase = (blockIdx.x * BlockDimX + threadIdx.x) * UnrollFactor;
        if (irowBase >= vectorSize)
            return;
        assert(irowBase + UnrollFactor <= vectorSize);

        __shared__ T meanS[BlockDimX * UnrollFactor];
        __shared__ T invStdDevS[BlockDimX * UnrollFactor];
        __shared__ T scaleS[BlockDimX * UnrollFactor];
        __shared__ T biasS[BlockDimX * UnrollFactor];
        int offs = threadIdx.x * UnrollFactor;
        // REVIEW alexeyk: optimize smem usage, reduce transaction count (not sure if it's worth it though).
        if (threadIdx.y == 0)
        {
            LoadElements<UnrollFactor>(batchMean + irowBase, meanS + offs);
            LoadElements<UnrollFactor>(batchInvStdDev + irowBase, invStdDevS + offs);
            LoadElements<UnrollFactor>(bnScale + irowBase, scaleS + offs);
            LoadElements<UnrollFactor>(bnBias + irowBase, biasS + offs);
        }
        __syncthreads();
        T mean[UnrollFactor];
        T invStdDev[UnrollFactor];
        T scale[UnrollFactor];
        T bias[UnrollFactor];
        LoadElements<UnrollFactor>(meanS + offs, mean);
        LoadElements<UnrollFactor>(invStdDevS + offs, invStdDev);
        LoadElements<UnrollFactor>(scaleS + offs, scale);
        LoadElements<UnrollFactor>(biasS + offs, bias);

        int icol = blockIdx.y * BlockDimY + threadIdx.y;
        size_t startOffs = static_cast<size_t>(icol) * vectorSize + irowBase;
        const T* psrc = x + startOffs;
        T* pdst = y + startOffs;
        size_t stride = static_cast<size_t>(gridDim.y * BlockDimY) * vectorSize;
        for (; icol < batchSize; icol += gridDim.y * BlockDimY, psrc += stride, pdst += stride)
        {
            T val[UnrollFactor];
            LoadElements<UnrollFactor>(psrc, val);
    #pragma unroll
            for (int k = 0; k < UnrollFactor; k++)
            {
                val[k] = scale[k] * (val[k] - mean[k]) * invStdDev[k] + bias[k];
            }
            StoreElements<UnrollFactor>(val, pdst);
        }
        //for (int k = 0; k < UnrollFactor; k++)
        //    printf("(%d, %d, %d): (%d, %f)\n", threadIdx.x, threadIdx.y, k, laneId, mean[k]);
    }

    template <int UnrollFactor, typename T>
    void NormalizeBatchTrainingImpl(size_t vectorSize, size_t batchSize, const T* x, T* y,
        const T* bnScale, const T* bnBias, const T* batchMean, const T* batchInvStdDev, cudaStream_t stream)
    {
        assert((vectorSize % UnrollFactor) == 0);

        const int BlockDimX = 32 / UnrollFactor;
        const int BlockDimY = 4 * UnrollFactor;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create a grid that has uses striding in y-dimension to cover whole mini-batch.
        auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * UnrollFactor)));
        kNormalizeBatchTraining<BlockDimX, BlockDimY, UnrollFactor><<<gdim, bdim, 0, stream>>>(
            static_cast<int>(vectorSize), static_cast<int>(batchSize), x, y, bnScale, bnBias,
            batchMean, batchInvStdDev);
    }

    template <typename T>
    cudaError_t NormalizeBatchTraining(size_t vectorSize, size_t batchSize, const T* x, T* y,
        const T* bnScale, const T* bnBias, const T* batchMean, const T* batchInvStdDev, cudaStream_t stream)
    {
        if ((vectorSize % 4) == 0)
            NormalizeBatchTrainingImpl<4>(vectorSize, batchSize, x, y, bnScale, bnBias, batchMean, batchInvStdDev, stream);
        else if ((vectorSize % 2) == 0)
            NormalizeBatchTrainingImpl<2>(vectorSize, batchSize, x, y, bnScale, bnBias, batchMean, batchInvStdDev, stream);
        else
            NormalizeBatchTrainingImpl<1>(vectorSize, batchSize, x, y, bnScale, bnBias, batchMean, batchInvStdDev, stream);
        return cudaGetLastError();
    }

    template <typename T>
    cudaError_t BatchNormalizationForwardTraining(size_t vectorSize, size_t batchSize, const T* x, T* y,
        const T* bnScale, const T* bnBias, double epsilon, T* saveMean, T* saveInvStdDev, cudaStream_t stream)
    {
        assert(nullptr != x);
        assert(nullptr != y);
        assert(std::isfinite(epsilon) && epsilon > 0);
        assert(nullptr != saveMean);
        assert(nullptr != saveInvStdDev);

        cudaError_t err;
        err = ComputeBatchMeanAndInvStdDev(vectorSize, batchSize, x, epsilon, saveMean, saveInvStdDev, stream);
        if (cudaSuccess != err)
            return err;

        err = NormalizeBatchTraining(vectorSize, batchSize, x, y, bnScale, bnBias, saveMean, saveInvStdDev, stream);
        return err;
    }

} } }
