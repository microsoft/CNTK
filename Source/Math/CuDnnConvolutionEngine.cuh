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

    using Tensor4D = ConvolutionTensor4D;

    size_t RoundUpToMultiple(size_t n, size_t blockSize)
    {
        return (n + blockSize - 1) / blockSize;
    }

    cudaError_t GetLastCudaError()
    {
        cudaError_t prelaunchErr = cudaGetLastError();
        assert(cudaSuccess == prelaunchErr);
        if (prelaunchErr != cudaSuccess)
            return prelaunchErr;
        
#ifndef NO_SYNC
        cudaError_t executionErr = cudaStreamSynchronize(GetStream());
        assert(cudaSuccess == executionErr);
        if (executionErr != cudaSuccess)
            return executionErr;
#endif
        return cudaSuccess;
    }

    template <int U, typename T>
    __device__ __forceinline__ void LoadValues(const T* src, T dst[U])
    {
#pragma unroll
        for (int i = 0; i < U; i++)
            dst[i] = src[i];
    }

    template <>
    __device__ __forceinline__ void LoadValues<2, float>(const float* src, float dst[2])
    {
        // src must be aligned at 8 bytes boundary.
        assert(reinterpret_cast<uintptr_t>(src) % (sizeof(dst)) == 0);
        auto v = *(const float2*)src;
        dst[0] = v.x;
        dst[1] = v.y;
    }

    template <>
    __device__ __forceinline__ void LoadValues<4, float>(const float* src, float dst[4])
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

    template <int U, typename T>
    __device__ __forceinline__ void StoreValues(const T src[U], T* dst)
    {
#pragma unroll
        for (int i = 0; i < U; i++)
            dst[i] = src[i];
    }

    template <>
    __device__ __forceinline__ void StoreValues<2, float>(const float src[2], float* dst)
    {
        // dst must be aligned at 8 bytes boundary.
        assert(reinterpret_cast<uintptr_t>(dst) % (sizeof(src)) == 0);
        float2 v;
        v.x = src[0];
        v.y = src[1];
        *(reinterpret_cast<float2*>(dst)) = v;
    }

    template <>
    __device__ __forceinline__ void StoreValues<4, float>(const float src[4], float* dst)
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
                // REVIEW alexeyk: rsqrtf is just one MUFU.RSQ instruction so it's faster than
                // __frsqrt_rn intrinsic which performs round-to-nearest-even rounding which adds ~10 other instructions.
                // __frsqrt_rn is unbiased rounding though, need to verify whether it is a better choice for BN implementation.
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

    // This function is used to select correct unroll factor.
    // REVIEW alexeyk: ask our C++ gurus (Marko/Amit) if there is better way.
    template <template <int> class Func, typename T, typename ...Targs>
    void Call(size_t vectorSize, Targs... args)
    {
        if ((vectorSize % 4) == 0)
            Func<4>::template Call<T>(args...);
        else if ((vectorSize % 2) == 0)
            Func<2>::template Call<T>(args...);
        else
            Func<1>::template Call<T>(args...);
    }

    //--------------------------------------------------------------------
    // Mean and variance computaion
    //--------------------------------------------------------------------

    // The kernel implements online, parallel and numerically stable algorithm 
    // for computing batch mean and variance (here inverse standard deviation) with one pass over the data.
    // It uses algorithm described in: http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    template <int BlockDimX, int BlockDimY, int U, typename T>
    __global__ void kComputeBatchMeanAndInvStdDev(int vectorSize, int batchSize, const T* x, T* runMean, T* runInvStdDev,
                                                  double epsilon, T* xMean, T* xInvStdDev)
    {
        static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert((vectorSize % U) == 0);
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
        assert(isfinite(epsilon) && epsilon > 0);

        int irowSrcBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
        if (irowSrcBase >= vectorSize)
            return;
        assert(irowSrcBase + U <= vectorSize);

        int n = 0;
        T mean[U];
        T m2[U];
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            mean[k] = 0;
            m2[k] = 0;
        }

        int icolSrc = threadIdx.y;
        const T* psrc = x + static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
        // Stride over all vectors in the batch.
        for (; icolSrc < batchSize; icolSrc += BlockDimY)
        {
            n++;
            T curVal[U];
            LoadValues<U>(psrc, curVal);
            // No need for separate unrolling, SASS looks good.
#pragma unroll
            for (int k = 0; k < U; k++)
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
                T d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
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
        __shared__ T meanRes[BlockDimX * U][cwarp - 1];
        __shared__ T m2Res[BlockDimX * U][cwarp - 1];
        __shared__ int nRes[cwarp - 1];

        // Each warp (except warp0) will write accumulated results to shared memory.
        const int iwarp = tid / CUB_PTX_WARP_THREADS;
        if (iwarp > 0 && laneId < BlockDimX)
        {
            if (laneId == 0)
                nRes[iwarp - 1] = n;
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                meanRes[laneId * U + k][iwarp - 1] = mean[k];
                m2Res[laneId * U + k][iwarp - 1] = m2[k];
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
                T d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    d[k] = meanRes[threadIdx.x * U + k][i] - mean[k];
                    T dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += m2Res[threadIdx.x * U + k][i] + d[k] * n * dScaled;
                }
                n = nsum;
            }
            size_t idxDstBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
            StoreValues<U>(mean, xMean + idxDstBase);
            StoreValues<U>(mean, runMean + idxDstBase);
            Operations::RSqrt<T> rsqrtOp;
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                m2[k] = rsqrtOp(static_cast<T>(m2[k] / batchSize + epsilon));
            }
            StoreValues<U>(m2, xInvStdDev + idxDstBase);
            StoreValues<U>(m2, runInvStdDev + idxDstBase);
        }
    }

    // This kernel is very similar to kComputeBatchMeanAndInvStdDev except it reduces not just over N (mini-batch)
    // but also W and H dimensions.
    // REVIEW alexeyk: is it possible to combine this and previous kernel into a single kernel without hurting performance/readability much?
    template <int BlockDimX, int BlockDimY, int U, typename T>
    __global__ void kComputeSpatialBatchMeanAndInvStdDev(int vectorSize, int spatialSize, int batchSize, const T* x, T* runMean, T* runInvStdDev,
                                                         double epsilon, T* xMean, T* xInvStdDev)
    {
        static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
        assert((spatialSize % U) == 0);
        assert((vectorSize % spatialSize) == 0);
        assert(isfinite(epsilon) && epsilon > 0);

        int irowSrcBase = blockIdx.x * spatialSize + threadIdx.x * U;
        if (irowSrcBase >= vectorSize)
            return;
        assert(irowSrcBase + U <= vectorSize);
        int irowSrcLim = (blockIdx.x + 1) * spatialSize;

        int n = 0;
        T mean[U];
        T m2[U];
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            mean[k] = 0;
            m2[k] = 0;
        }

        int icolSrc = threadIdx.y;
        const T* psrcBase = x + static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
        // Stride over all vectors in the batch.
        for (; icolSrc < batchSize; icolSrc += BlockDimY)
        {
            const T* psrc = psrcBase;
            // Stride over all values in feature map (W and H dimensions).
            for (int irowSrc = irowSrcBase; irowSrc < irowSrcLim; irowSrc += BlockDimX * U, psrc += BlockDimX * U)
            {
                n++;
                T curVal[U];
                LoadValues<U>(psrc, curVal);
                // No need for separate unrolling, SASS looks good.
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    T d = curVal[k] - mean[k];
                    // REVIEW alexeyk: we enabled fast CUDA math in CNTK so division below will be approximate, is this a problem?
                    // Using precise math slows down the code by about 40%.
                    mean[k] += d / n;
                    m2[k] += d * (curVal[k] - mean[k]);
                }
            }
            psrcBase += vectorSize * BlockDimY;
        }

        const int tid = threadIdx.y * BlockDimX + threadIdx.x;
        const int laneId = tid & 0x1f;
        // First, reduce within warp using shuffle.
        if (n > 0)
        {
#pragma unroll
            for (int i = 1; i < CUB_PTX_WARP_THREADS; i *= 2)
            {
                int srcLane = laneId + i;
                int n2 = cub::ShuffleIndex(n, srcLane);
                int nsum = n + n2;
                T d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
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
        __shared__ T meanRes[U][cwarp - 1];
        __shared__ T m2Res[U][cwarp - 1];
        __shared__ int nRes[cwarp - 1];

        // Each warp (except warp0) will write accumulated results to shared memory.
        const int iwarp = tid / CUB_PTX_WARP_THREADS;
        if (iwarp > 0 && laneId == 0)
        {
            nRes[iwarp - 1] = n;
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                meanRes[k][iwarp - 1] = mean[k];
                m2Res[k][iwarp - 1] = m2[k];
            }
        }
        __syncthreads();

        // One thread will accumulate and write final results.
        if (tid == 0)
        {
            // Use simple loop as number of warps is small, 8 at max.
#pragma unroll
            for (int i = 0; i < cwarp - 1; i++)
            {
                int n2 = nRes[i];
                int nsum = n + n2;
                T d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    d[k] = meanRes[k][i] - mean[k];
                    T dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += m2Res[k][i] + d[k] * n * dScaled;
                }
                n = nsum;
            }
            // Final step - accumlate results in mean[0] and m2[0].
            // REVIEW alexeyk: move outside of the loop, before storing values to smem.
#pragma unroll
            for (int k = 1; k < U; k++)
            {
                T d = mean[k] - mean[0];
                T dScaled = d * n / (n + k * n);
                mean[0] += dScaled;
                m2[0] += m2[k] + d * k * n * dScaled;
            }

            xMean[blockIdx.x] = mean[0];
            runMean[blockIdx.x] = mean[0];
            Operations::RSqrt<T> rsqrtOp;
            m2[0] = rsqrtOp(static_cast<T>(m2[0] / (batchSize * spatialSize) + epsilon));
            xInvStdDev[blockIdx.x] = m2[0];
            runInvStdDev[blockIdx.x] = m2[0];
        }
    }

    template <int U>
    struct ComputeBatchMeanAndInvStdDev
    {
        template <typename T>
        static void Call(size_t vectorSize, size_t batchSize, const T* x, T* runMean, T* runInvStdDev, double epsilon, T* xMean, T* xInvStdDev, cudaStream_t stream)
        {
            assert((vectorSize % U) == 0);

            const int BlockDimX = 32 / U;
            const int BlockDimY = 4 * U;
            auto bdim = dim3(BlockDimX, BlockDimY);
            // Create grid with only one block in y(batch)-dimension as kernel uses striding.
            auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)));
            kComputeBatchMeanAndInvStdDev<BlockDimX, BlockDimY, U><<<gdim, bdim, 0, stream>>>(
                static_cast<int>(vectorSize), static_cast<int>(batchSize), 
                x, runMean, runInvStdDev, epsilon, xMean, xInvStdDev);
        }
    };

    template <int U>
    struct ComputeSpatialBatchMeanAndInvStdDev
    {
        template <typename T>
        static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, const T* x, T* runMean, T* runInvStdDev, double epsilon,
                         T* xMean, T* xInvStdDev, cudaStream_t stream)
        {
            assert((vectorSize % spatialSize) == 0);
            assert((spatialSize % U) == 0);

            const int BlockDimX = 32 / U;
            const int BlockDimY = 4 * U;
            auto bdim = dim3(BlockDimX, BlockDimY);
            // Create grid with only one block in y(batch)-dimension as kernel uses striding.
            // Each thread block processes a single whole feature map independently (i.e. reduces over W, H and N dimensions).
            auto gdim = dim3(static_cast<unsigned int>(vectorSize / spatialSize));
            kComputeSpatialBatchMeanAndInvStdDev<BlockDimX, BlockDimY, U><<<gdim, bdim, 0, stream>>>(
                static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), 
                x, runMean, runInvStdDev,epsilon, xMean, xInvStdDev);
        }
    };

    //--------------------------------------------------------------------
    // Forward propagation
    //--------------------------------------------------------------------

    template <int BlockDimX, int BlockDimY, bool Spatial, int U, typename T>
    __global__ void kNormalizeBatchTraining(int vectorSize, int spatialSize, int batchSize, const T* x, T* y,
        const T* bnScale, const T* bnBias, const T* batchMean, const T* batchInvStdDev)
    {
        static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
        assert((vectorSize % U) == 0);
        assert(!Spatial || (spatialSize % U) == 0);
        assert((vectorSize % spatialSize) == 0);

        int irowBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
        if (irowBase >= vectorSize)
            return;
        assert(irowBase + U <= vectorSize);

        __shared__ T meanS[BlockDimX * U];
        __shared__ T invStdDevS[BlockDimX * U];
        __shared__ T scaleS[BlockDimX * U];
        __shared__ T biasS[BlockDimX * U];
        int offs = threadIdx.x * U;
        // REVIEW alexeyk: optimize smem usage, reduce transaction count (is it worth it?).
        if (threadIdx.y == 0)
        {
            if (Spatial)
            {
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    int imap = (irowBase + k) / spatialSize;
                    meanS[offs + k] = batchMean[imap];
                    invStdDevS[offs + k] = batchInvStdDev[imap];
                    scaleS[offs + k] = bnScale[imap];
                    biasS[offs + k] = bnBias[imap];
                }
            }
            else
            {
                LoadValues<U>(batchMean + irowBase, meanS + offs);
                LoadValues<U>(batchInvStdDev + irowBase, invStdDevS + offs);
                LoadValues<U>(bnScale + irowBase, scaleS + offs);
                LoadValues<U>(bnBias + irowBase, biasS + offs);
            }
        }
        __syncthreads();
        T mean[U];
        T invStdDev[U];
        T scale[U];
        T bias[U];
        LoadValues<U>(meanS + offs, mean);
        LoadValues<U>(invStdDevS + offs, invStdDev);
        LoadValues<U>(scaleS + offs, scale);
        LoadValues<U>(biasS + offs, bias);

        int icol = blockIdx.y * BlockDimY + threadIdx.y;
        size_t startOffs = static_cast<size_t>(icol) * vectorSize + irowBase;
        const T* psrc = x + startOffs;
        T* pdst = y + startOffs;
        size_t stride = static_cast<size_t>(gridDim.y * BlockDimY) * vectorSize;
        for (; icol < batchSize; icol += gridDim.y * BlockDimY, psrc += stride, pdst += stride)
        {
            T val[U];
            LoadValues<U>(psrc, val);
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                val[k] = scale[k] * (val[k] - mean[k]) * invStdDev[k] + bias[k];
            }
            StoreValues<U>(val, pdst);
        }
    }

    template <int U>
    struct NormalizeBatchTraining
    {
        template <typename T>
        static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, bool spatial, const T* x, T* y,
            const T* bnScale, const T* bnBias, const T* batchMean, const T* batchInvStdDev, cudaStream_t stream)
        {
            assert((vectorSize % U) == 0);

            const int BlockDimX = 32 / U;
            const int BlockDimY = 4 * U;
            auto bdim = dim3(BlockDimX, BlockDimY);
            // Create a grid that has uses striding in y-dimension to cover whole mini-batch.
            auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)));
            if (spatial)
            {
                kNormalizeBatchTraining<BlockDimX, BlockDimY, true, U><<<gdim, bdim, 0, stream>>>(
                    static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, y, bnScale, bnBias,
                    batchMean, batchInvStdDev);
            }
            else
            {
                kNormalizeBatchTraining<BlockDimX, BlockDimY, false, U><<<gdim, bdim, 0, stream>>>(
                    static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, y, bnScale, bnBias,
                    batchMean, batchInvStdDev);
            }
        }
    };

    template <typename T>
    cudaError_t BatchNormalizationForwardTraining(const Tensor4D& t, bool spatial, const T* x, T* y,
                                                  const T* bnScale, const T* bnBias, T* runMean, T* runInvStdDev,
                                                  double epsilon, T* saveMean, T* saveInvStdDev, cudaStream_t stream)
    {
        assert(nullptr != x);
        assert(nullptr != y);
        assert(nullptr != bnScale);
        assert(nullptr != bnBias);
        assert(std::isfinite(epsilon) && epsilon > 0);
        assert(nullptr != runMean);
        assert(nullptr != runInvStdDev);
        assert(nullptr != saveMean);
        assert(nullptr != saveInvStdDev);

        size_t vectorSize = t.w() * t.h() * t.c();
        size_t spatialSize = spatial ? t.w() * t.h() : 1;
        size_t batchSize = t.n();
        assert(0 < vectorSize && vectorSize <= std::numeric_limits<int>::max());
        assert(0 < batchSize  && batchSize  <= std::numeric_limits<int>::max());

        if (spatial)
        {
            Call<ComputeSpatialBatchMeanAndInvStdDev, T>(spatialSize, vectorSize, spatialSize, batchSize, x, 
                                                         runMean, runInvStdDev, epsilon, saveMean, saveInvStdDev, stream);
            cudaError_t err = GetLastCudaError();
            if (cudaSuccess != err)
                return err;

        }
        else
        {
            Call<ComputeBatchMeanAndInvStdDev, T>(vectorSize, vectorSize, batchSize, x,
                                                  runMean, runInvStdDev, epsilon, saveMean, saveInvStdDev, stream);
            cudaError_t err = GetLastCudaError();
            if (cudaSuccess != err)
                return err;
        }
        Call<NormalizeBatchTraining, T>(spatial ? spatialSize : vectorSize, vectorSize, spatialSize, batchSize,
                                        spatial, x, y, bnScale, bnBias, saveMean, saveInvStdDev, stream);
        return GetLastCudaError();
    }

    //--------------------------------------------------------------------
    // Backpropagation
    //--------------------------------------------------------------------

    template <int BlockDimX, int BlockDimY, int U, typename T>
    __global__ void kComputeScaleAndBiasGradients(int vectorSize, int batchSize, const T* x, const T* dy, T* dScale, T* dBias,
                                                  const T* saveMean, const T* saveInvStdDev)
    {
        static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        static_assert(((BlockDimY - 1) & BlockDimY) == 0, "BlockDimY must be a power of 2.");
        assert((vectorSize % U) == 0);
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // REVIEW alexeyk: first part looks very similar to kComputeBatchMeanAndInvStdDev, any chance to refactor?
        int irowSrcBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
        if (irowSrcBase >= vectorSize)
            return;
        assert(irowSrcBase + U <= vectorSize);

        T mean[U];
        T invStdDev[U];
        __shared__ T meanS[BlockDimX * U];
        __shared__ T invStdDevS[BlockDimX * U];
        // Read mean and inv std dev.
        if (threadIdx.y == 0)
        {
            LoadValues<U>(saveMean + irowSrcBase, mean);
            LoadValues<U>(saveInvStdDev + irowSrcBase, invStdDev);
            StoreValues<U>(mean, &meanS[threadIdx.x * U]);
            StoreValues<U>(invStdDev, &invStdDevS[threadIdx.x * U]);
        }
        __syncthreads();
        if (threadIdx.y != 0)
        {
            LoadValues<U>(&meanS[threadIdx.x * U], mean);
            LoadValues<U>(&invStdDevS[threadIdx.x * U], invStdDev);
        }

        T ds[U];
        T db[U];
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            ds[k] = 0;
            db[k] = 0;
        }

        int icolSrc = threadIdx.y;
        size_t startOffs = static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
        const T* px = x + startOffs;
        const T* pdy = dy + startOffs;
        size_t stride = static_cast<size_t>(vectorSize) * BlockDimY;
        // Stride over all vectors in the batch.
        for (; icolSrc < batchSize; icolSrc += BlockDimY, px += stride, pdy += stride)
        {
            T curX[U];
            T curdY[U];
            LoadValues<U>(px, curX);
            LoadValues<U>(pdy, curdY);
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                ds[k] += pdy[k] * (curX[k] - mean[k]) * invStdDev[k];
                db[k] += pdy[k];
            }
        }

        // Final reduction.
        __shared__ T dsS[BlockDimY][BlockDimX * U];
        __shared__ T dbS[BlockDimY][BlockDimX * U];
        StoreValues<U>(ds, &dsS[threadIdx.y][threadIdx.x * U]);
        StoreValues<U>(db, &dbS[threadIdx.y][threadIdx.x * U]);
        __syncthreads();
        // Very simple block reduction. As the block y dim is small (e.g. 16) then the loop
        // is executed very few times (e.g. 4) so the performance is good.
        // Can be potentially improved by using shuffle instructions (as in kComputeBatchMeanAndInvStdDev).
#pragma unroll
        for (int y = BlockDimY / 2; y > 0; y /= 2)
        {
            if (threadIdx.y < y)
            {
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    dsS[threadIdx.y][threadIdx.x * U + k] += dsS[threadIdx.y + y][threadIdx.x * U + k];
                    dbS[threadIdx.y][threadIdx.x * U + k] += dbS[threadIdx.y + y][threadIdx.x * U + k];
                }
                __syncthreads();
            }
        }

        // Write results.
        if (threadIdx.y == 0)
        {
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                dScale[irowSrcBase + k] = dsS[0][threadIdx.x * U + k];
                dBias[irowSrcBase + k] = dbS[0][threadIdx.x * U + k];
            }
        }
    }

    template <int BlockDimX, int BlockDimY, int U, typename T>
    __global__ void kComputeSpatialScaleAndBiasGradients(int vectorSize, int spatialSize, int batchSize, const T* x, const T* dy, T* dScale, T* dBias,
                                                         const T* saveMean, const T* saveInvStdDev)
    {
        static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
        assert((spatialSize % U) == 0);
        assert((vectorSize % spatialSize) == 0);

        int irowBase = blockIdx.x * spatialSize + threadIdx.x * U;
        if (irowBase >= vectorSize)
            return;
        assert(irowBase + U <= vectorSize);
        int irowLim = (blockIdx.x + 1) * spatialSize;

        T mean;
        T invStdDev;
        __shared__ T meanS;
        __shared__ T invStdDevS;
        const int tid = threadIdx.y * BlockDimX + threadIdx.x;
        // Read mean and inv std dev.
        if (tid == 0)
        {
            meanS = saveMean[blockIdx.x];
            invStdDevS = saveInvStdDev[blockIdx.x];
        }
        __syncthreads();
        if (tid != 0)
        {
            mean = meanS;
            invStdDev = invStdDevS;
        }

        T ds[U];
        T db[U];
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            ds[k] = 0;
            db[k] = 0;
        }

        int icolSrc = threadIdx.y;
        size_t startOffs = static_cast<size_t>(icolSrc) * vectorSize + irowBase;
        const T* pxBase = x + startOffs;
        const T* pdyBase = dy + startOffs;
        size_t stride = static_cast<size_t>(vectorSize) * BlockDimY;
        // Stride over all vectors in the batch.
        for (; icolSrc < batchSize; icolSrc += BlockDimY, pxBase += stride, pdyBase += stride)
        {
            const T* px = pxBase;
            const T* pdy = pdyBase;
            // Stride over all values in feature map (W and H dimensions).
            for (int irow = irowBase; irow < irowLim; irow += BlockDimX * U, px += BlockDimX * U, pdy += BlockDimX * U)
            {
                T curX[U];
                T curdY[U];
                LoadValues<U>(px, curX);
                LoadValues<U>(pdy, curdY);
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    ds[k] += pdy[k] * (curX[k] - mean) * invStdDev;
                    db[k] += pdy[k];
                }
            }
        }
        __syncthreads();
        using BlockReduce = cub::BlockReduce<T, BlockDimX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BlockDimY>;
        // Note: must use separate temp storages for each reduction.
        __shared__ typename BlockReduce::TempStorage tmp1;
        T dsRes = BlockReduce(tmp1).Sum(ds);
        __shared__ typename BlockReduce::TempStorage tmp2;
        T dbRes = BlockReduce(tmp2).Sum(db);
        if (tid == 0)
        {
            dScale[blockIdx.x] = dsRes;
            dBias[blockIdx.x] = dbRes;
        }
    }

    template <int U>
    struct ComputeScaleAndBiasGradients
    {
        template <typename T>
        static void Call(size_t vectorSize, size_t batchSize, const T* x, const T* dy,
            T* dScale, T* dBias, const T* saveMean, const T* saveInvStdDev, cudaStream_t stream)
        {
            assert((vectorSize % U) == 0);
            const int BlockDimX = 32 / U;
            const int BlockDimY = 4 * U;
            auto bdim = dim3(BlockDimX, BlockDimY);
            // Create a grid that has uses striding in y-dimension to cover whole mini-batch.
            auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)));
            kComputeScaleAndBiasGradients<BlockDimX, BlockDimY, U><<<gdim, bdim, 0, stream>>>(
                static_cast<int>(vectorSize), static_cast<int>(batchSize), x, dy, dScale, dBias, saveMean, saveInvStdDev);
        }
    };

    template <int U>
    struct ComputeSpatialScaleAndBiasGradients
    {
        template <typename T>
        static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, const T* x, const T* dy,
            T* dScale, T* dBias, const T* saveMean, const T* saveInvStdDev, cudaStream_t stream)
        {
            assert((spatialSize % U) == 0);
            assert((vectorSize % spatialSize) == 0);

            const int BlockDimX = 32 / U;
            const int BlockDimY = 4 * U;
            auto bdim = dim3(BlockDimX, BlockDimY);
            // Create a grid that has uses striding in y-dimension to cover whole mini-batch.
            auto gdim = dim3(static_cast<unsigned int>(vectorSize / spatialSize));
            kComputeSpatialScaleAndBiasGradients<BlockDimX, BlockDimY, U><<<gdim, bdim, 0, stream>>>(
                static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, dy, dScale, dBias, saveMean, saveInvStdDev);
        }
    };

    template <int BlockDimX, int BlockDimY, bool Spatial, int U, typename T>
    __global__ void kBackpropagateBatchNormGradients(int vectorSize, int spatialSize, int batchSize, const T* x, const T* dy, T* dx, const T* bnScale,
                                                     const T* dScale, const T* dBias, const T* saveMean, const T* saveInvStdDev)
    {
        static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
        static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
        assert(blockDim.x == BlockDimX);
        assert(blockDim.y == BlockDimY);
        assert(blockDim.z == 1);
        assert(gridDim.z == 1);
        assert((vectorSize % U) == 0);
        assert(Spatial || spatialSize == 1);
        assert(!Spatial || (spatialSize % U) == 0);
        assert((vectorSize % spatialSize) == 0);

        int irowBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
        if (irowBase >= vectorSize)
            return;
        assert(irowBase + U <= vectorSize);
        T scale[U];
        T ds[U];
        T db[U];
        T mean[U];
        T invStdDev[U];
        // REVIEW alexeyk: here we're wasting some bandwidth but this might be ok as it's a one-timer.
        if (Spatial)
        {
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                int imap = (irowBase + k) / spatialSize;
                scale[k] = bnScale[imap];
                ds[k] = dScale[imap];
                db[k] = dBias[imap];
                mean[k] = saveMean[imap];
                invStdDev[k] = saveInvStdDev[imap];
            }
        }
        else
        {
            LoadValues<U>(bnScale + irowBase, scale);
            LoadValues<U>(dScale + irowBase, ds);
            LoadValues<U>(dBias + irowBase, db);
            LoadValues<U>(saveMean + irowBase, mean);
            LoadValues<U>(saveInvStdDev + irowBase, invStdDev);
        }

        int icol = blockIdx.y * BlockDimY + threadIdx.y;
        size_t startOffs = static_cast<size_t>(icol) * vectorSize + irowBase;
        const T* px = x + startOffs;
        const T* pdy = dy + startOffs;
        T* pdx = dx + startOffs;
        size_t stride = static_cast<size_t>(gridDim.y * BlockDimY) * vectorSize;
        for (; icol < batchSize; icol += gridDim.y * BlockDimY, px += stride, pdy += stride, pdx += stride)
        {
            T xCur[U];
            T dyCur[U];
            T dxCur[U];
            LoadValues<U>(px, xCur);
            LoadValues<U>(pdy, dyCur);
            LoadValues<U>(pdx, dxCur);
            // From the BN paper, dL/dxi is a sum of three terms: dL/dxi = t1 + t2 + t3
            // After simplifcation, they become the following:
            // 1. t1 = scale * dL/dyi * invStdDev
            // 2. t2 = (-scale / m) * invStdDev * xHat * dL/dScale
            // 3. t3 = (-scale / m) * invStdDev * dL/dBias (for this one note that Sum(xHat) == 0)
            // Simplifying this a bit more, we get the formula below.
            T val[U];
            int m = Spatial ? batchSize * spatialSize : batchSize;
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                T xNorm = (xCur[k] - mean[k]) * invStdDev[k];
                val[k] = dxCur[k] + (scale[k] * invStdDev[k]) * (dyCur[k] - (xNorm * ds[k] + db[k]) / m);
            }
            StoreValues<U>(val, pdx);
        }
    }

    template <int U>
    struct BackpropagateBatchNormGradients
    {
        template <typename T>
        static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, bool spatial, const T* x, const T* dy, T* dx,
                         const T* bnScale, const T* dScale, const T* dBias, const T* saveMean, const T* saveInvStdDev, cudaStream_t stream)
        {
            assert((vectorSize % U) == 0);
            const int BlockDimX = 32 / U;
            const int BlockDimY = 4 * U;
            auto bdim = dim3(BlockDimX, BlockDimY);
            auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)),
                             static_cast<unsigned int>(RoundUpToMultiple(batchSize, BlockDimY)));
            if (spatial)
            {
                kBackpropagateBatchNormGradients<BlockDimX, BlockDimY, true, U><<<gdim, bdim, 0, stream>>>(
                    static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, dy, dx, bnScale, dScale, dBias, saveMean, saveInvStdDev);
            }
            else
            {
                kBackpropagateBatchNormGradients<BlockDimX, BlockDimY, false, U><<<gdim, bdim, 0, stream>>>(
                    static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, dy, dx, bnScale, dScale, dBias, saveMean, saveInvStdDev);
            }
        }
    };

    template <typename T>
    cudaError_t BatchNormalizationBackward(const Tensor4D& t, bool spatial, const T* x, const T* dy, T* dx, const T* bnScale,
                                           T* dScale, T* dBias, const T* saveMean, const T* saveInvStdDev, cudaStream_t stream)
    {
        assert(nullptr != x);
        assert(nullptr != dy);
        assert(nullptr != dx);
        assert(nullptr != bnScale);
        assert(nullptr != dScale);
        assert(nullptr != dBias);
        assert(nullptr != saveMean);
        assert(nullptr != saveInvStdDev);

        size_t vectorSize = t.w() * t.h() * t.c();
        size_t spatialSize = spatial ? t.w() * t.h() : 1;
        size_t batchSize = t.n();
        assert(0 < vectorSize && vectorSize <= std::numeric_limits<int>::max());
        assert(0 < batchSize  && batchSize  <= std::numeric_limits<int>::max());

        if (spatial)
        {
            Call<ComputeSpatialScaleAndBiasGradients, T>(spatialSize, vectorSize, spatialSize, batchSize, x, dy, dScale, dBias, 
                                                         saveMean, saveInvStdDev, stream);
            cudaError_t err = GetLastCudaError();
            if (cudaSuccess != err)
                return err;
        }
        else
        {
            Call<ComputeScaleAndBiasGradients, T>(vectorSize, vectorSize, batchSize, x, dy, dScale, dBias, 
                                                  saveMean, saveInvStdDev, stream);
            cudaError_t err = GetLastCudaError();
            if (cudaSuccess != err)
                return err;
        }
        Call<BackpropagateBatchNormGradients, T>(spatial ? spatialSize : vectorSize, vectorSize, spatialSize, batchSize, spatial, 
                                                 x, dy, dx, bnScale, dScale, dBias, saveMean, saveInvStdDev, stream);
        return GetLastCudaError();
    }
} } }
