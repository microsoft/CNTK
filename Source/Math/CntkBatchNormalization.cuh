//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

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
/*
template <int U, typename T>
__device__ __forceinline__ void LoadValues(const T* src, T dst[U])
{
#pragma unroll
    for (int i = 0; i < U; i++)
        dst[i] = src[i];
}
*/
template <int U, typename T1, typename T2>
__device__ __forceinline__ void LoadValues(const T1* src, T2 dst[U])
{
#pragma unroll
    for (int i = 0; i < U; i++)
        dst[i] = (T2)src[i];
}

template <>
__device__ __forceinline__ void LoadValues<2, float, float>(const float* src, float dst[2])
{
    // src must be aligned at 8 bytes boundary.
    assert(reinterpret_cast<uintptr_t>(src) % (sizeof(dst)) == 0);
    auto v = *(const float2*)src;
    dst[0] = v.x;
    dst[1] = v.y;
}

template <>
__device__ __forceinline__ void LoadValues<4, float, float>(const float* src, float dst[4])
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
/*
template <int U, typename T>
__device__ __forceinline__ void StoreValues(const T src[U], T* dst)
{
#pragma unroll
    for (int i = 0; i < U; i++)
        dst[i] = src[i];
}
*/
template <int U, typename T1, typename T2>
__device__ __forceinline__ void StoreValues(const T1 src[U], T2* dst)
{
#pragma unroll
    for (int i = 0; i < U; i++)
        dst[i] = (T2)src[i];
}

template <>
__device__ __forceinline__ void StoreValues<2, float, float>(const float src[2], float* dst)
{
    // dst must be aligned at 8 bytes boundary.
    assert(reinterpret_cast<uintptr_t>(dst) % (sizeof(src)) == 0);
    float2 v;
    v.x = src[0];
    v.y = src[1];
    *(reinterpret_cast<float2*>(dst)) = v;
}

template <>
__device__ __forceinline__ void StoreValues<4, float, float>(const float src[4], float* dst)
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

template <typename T>
__device__ __forceinline__ T Shuffle(T input, int srcLane, unsigned int mask)
{
#ifdef __CUDA_ARCH__
    // shfl is supported only on Kepler+
    static_assert(__CUDA_ARCH__ >= 300, "CNTK only supports only Kepler GPU architecture or newer.");
#if CUDA_VERSION >= 9000
    return cub::ShuffleIndex(input, srcLane, CUB_PTX_WARP_THREADS, mask); // Need cub > 1.7.0
#else
    return cub::ShuffleIndex(input, srcLane);
#endif
#else
    assert(false);
    return input; // keep compiler happy
#endif
}

namespace Operations
{
    __device__ float RSqrt(float a)
    {
        // REVIEW alexeyk: rsqrtf is just one MUFU.RSQ instruction so it's faster than
        // __frsqrt_rn intrinsic which performs round-to-nearest-even rounding which adds ~10 other instructions.
        // __frsqrt_rn is unbiased rounding though, need to verify whether it is a better choice for BN implementation.
        //return __frsqrt_rn(a);
        assert(::isfinite(a) && a > 0);
        return rsqrtf(a);
    }

    __device__ double RSqrt(double a)
    {
        assert(::isfinite(a) && a > 0);
        return rsqrt(a);
    }

    __device__ half RSqrt(half a)
    {
#if __CUDA_ARCH__ >= 600
        return hrsqrt(a);
#else
        return __float2half(rsqrtf(__half2float(a)));
#endif
    }
}

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

template <template <int> class Func, typename T1, typename T2, typename ...Targs>
void Call2(size_t vectorSize, Targs... args)
{
    if ((vectorSize % 4) == 0)
        Func<4>::template Call<T1, T2>(args...);
    else if ((vectorSize % 2) == 0)
        Func<2>::template Call<T1, T2>(args...);
    else
        Func<1>::template Call<T1, T2>(args...);
}

//--------------------------------------------------------------------
// Mean and variance computation
//--------------------------------------------------------------------

// The kernel implements online, parallel and numerically stable algorithm
// for computing batch mean and variance (and inverse standard deviation) with one pass over the data.
// It uses algorithms by Knuth/Welford and Chan et al (http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
// In short, algorithm has 2 steps:
// 1. Each thread strides over the input and computes mean and
//    m2 value (used to compute variance and inverse standard deviation at the end) - Welford algorithm.
// 2. Parallel reduction (Chan algorithm) performed by columns (note that
//    thread block and grid X dimensions go along the vector and Y dimension - along the batch).
//    As a result, each block has 2 * blockDim.x (mean and inverse stddev) values to write at the end.
//
// Running mean and variance will be averaged according to an exponential
// averaging factor (expAvgFactor), taking the running statistics with weight
// (1 - expAvgFactor).
// Batch mean and inverse standard deviation will be further averaged according
// to a blending factor (blendFactor), taking the running statistics with
// weight blendFactor.
// If (expAvgFactor = 0) && (blendFactor = 1), there is no need to call this
// function, since there no update based on batch data is involved (inference
// mode).
//
// Averaging into running variables (runMean, runVariance):
//     expAvgFactor == 0 - use running mean/var instead of the actual batch mean/var.
// 0 < expAvgFactor <  1 - average running mean/var with actual batch mean/var, e.g.,
//                         new runMean = expAvgFactor * actual batch mean + (1 - expAvgFactor) * runMean
//     expAvgFactor == 1 - use actual batch mean/var
//
// Blending into batch variables (based on new running statistics computed above):
//     blendFactor == 1 - use (new) running mean/var instead of the current actual batch mean/var.
// 0 < blendFactor <  1 - blend new running mean/var with averaged mean/var of the current minibatch, e.g.,
//                        new xMean = (1 - blendFactor) * actual batch mean + blendFactor * new runMean
//     blendFactor == 0 - use actual batch mean/var
template <int BlockDimX, int BlockDimY, int U, typename ElemType, typename StatType>
__global__ void kComputeBatchMeanAndInvStdDev(int vectorSize, int batchSize,
                                              const ElemType* x,                         // (in) input data
                                              double expAvgFactor, // TODO why not ElemType? same for the other parameters, functions?
                                              double blendFactor,
                                              StatType* runMean, StatType* runVariance,  // (in/out) running mean/variance, gets updated with current minibatch
                                              double epsilon,
                                              StatType* xMean, StatType* xInvStdDev)     // (out) this minibatch's mean and inverse stddev
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
    static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
    assert((vectorSize % U) == 0);
    assert(blockDim.x == BlockDimX);
    assert(blockDim.y == BlockDimY);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert(::isfinite(epsilon) && epsilon > 0);
    assert(::isfinite(expAvgFactor) && 0 <= expAvgFactor && expAvgFactor <= 1);
    assert(::isfinite(blendFactor) && 0 <= blendFactor && blendFactor <= 1);
    assert(batchSize >= 1);

    if (expAvgFactor != 0 || blendFactor != 1)
    {
        int irowSrcBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
        if (irowSrcBase >= vectorSize)
            return;
        assert(irowSrcBase + U <= vectorSize);

        // --- estimate this minibatch's mean/variance

        // first estimate mean over all data for this thread
        int n = 0;
        comp_t mean[U]; // this thread's part of the mean vector (stored as a normalized mean also during accumulation)
        comp_t m2[U];   // likewise for variance
        comp_t im2[U];  // and inverse stddev
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            mean[k] = 0;
            m2[k] = 0;
        }

        int icolSrc = threadIdx.y;
        const ElemType* psrc = x + static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
        // Stride over all vectors in the batch.
        for (; icolSrc < batchSize; icolSrc += BlockDimY)
        {
            n++;
            comp_t curVal[U];
            LoadValues<U>(psrc, curVal);
            // No need for separate unrolling, SASS looks good.
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                comp_t d = curVal[k] - mean[k];
                // REVIEW alexeyk: we enabled fast CUDA math in CNTK so division below will be approximate, is this a problem?
                // Using precise math slows down the code by about 40%.
                mean[k] += d / n; // mean_n = [mean_{n-1} * (n-1) + curVal] / n = mean_{n-1} *n/n - mean_{n-1} / n + curVal / n
                m2[k] += d * (curVal[k] - mean[k]);
            }
            psrc += vectorSize * BlockDimY;
        }

        // now reduce minibatch mean/variance across threads
        const int tid = threadIdx.y * BlockDimX + threadIdx.x;
        const int laneId = tid & 0x1f;

        unsigned int mask;
#if CUDA_VERSION >= 9000
        mask = __ballot_sync(0xffffffff, n);
#endif

        // First, reduce within warp using shuffle.
        if (n > 0)
        {
#pragma unroll
            for (int i = 1; i < CUB_PTX_WARP_THREADS / BlockDimX; i *= 2)
            {
                int srcLane = laneId + BlockDimX * i;
                int n2 = Shuffle(n, srcLane, mask);
                int nsum = n + n2;
                comp_t d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    d[k] = Shuffle(mean[k], srcLane, mask) - mean[k];
                    comp_t dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += Shuffle(m2[k], srcLane, mask) + d[k] * n * dScaled;
                }
                n = nsum;
            }
        }

        // Storage for each warp in a thread block. First warp ("accumulator") holds
        // final results so it does not need shared memory.
        const int cwarp = BlockDimX * BlockDimY / CUB_PTX_WARP_THREADS;
        __shared__ comp_t meanRes[BlockDimX * U][cwarp - 1];
        __shared__ comp_t m2Res[BlockDimX * U][cwarp - 1];
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

        // --- final reduction and update of running mean/variance

        // Accumulate and write final results.
        // REVIEW alexeyk: see if atomicAdd can be used instead, do perf comparison.
        if (threadIdx.y == 0)
        {
            // Use simple loop as number of warps is small, 8 at max.
#pragma unroll
            for (int i = 0; i < cwarp - 1; i++)
            {
                int n2 = nRes[i];
                int nsum = n + n2;
                comp_t d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    d[k] = meanRes[threadIdx.x * U + k][i] - mean[k];
                    comp_t dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += m2Res[threadIdx.x * U + k][i] + d[k] * n * dScaled;
                }
                n = nsum;
            }

            size_t idxDstBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
            comp_t run[U];
            comp_t x[U];

            // Compute running mean and batch mean.
            LoadValues<U>(runMean + idxDstBase, run);
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                run[k] = expAvgFactor * mean[k] + (1.0 - expAvgFactor) * run[k];
                x[k] = blendFactor * run[k] + (1.0 - blendFactor) * mean[k];
            }
            StoreValues<U>(run, runMean + idxDstBase);
            StoreValues<U>(x, xMean + idxDstBase);
            // At this point, runMean[] and xMean[] have been updated

            // Compute running variance and batch inverse standard deviation
            LoadValues<U>(runVariance + idxDstBase, run);
            // TODO add back special cases
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                // Compute batch inverse standard deviation and variance
                comp_t runVariance = batchSize == 1 ? 0 : m2[k] / (batchSize - 1);
                // Average
                run[k] = expAvgFactor * runVariance + (1.0 - expAvgFactor) * run[k];
                // Blend
                im2[k] = Operations::RSqrt(static_cast<comp_t>(m2[k] / batchSize + epsilon));
                if (blendFactor != 0)
                {
                    comp_t runInvStdDev = Operations::RSqrt(static_cast<comp_t>(run[k] + epsilon));
                    im2[k] = blendFactor * runInvStdDev + (1.0 - blendFactor) * im2[k];
                }
            }
            StoreValues<U>(run, runVariance + idxDstBase);
            StoreValues<U>(im2, xInvStdDev + idxDstBase);
            // at this point, runVariance[] xInvStdDev[] have been updated
        }
    }
    else if (threadIdx.y == 0)
    {
        size_t idxDstBase = (blockIdx.x * BlockDimX + threadIdx.x) * U;
        comp_t run[U];

        // Copy mean
        LoadValues<U>(runMean + idxDstBase, run);
        StoreValues<U>(run, xMean + idxDstBase);

        // Copy & convert variance
        LoadValues<U>(runVariance + idxDstBase, run);
#pragma unroll
        for (int k = 0; k < U; k++)
            run[k] = Operations::RSqrt(static_cast<comp_t>(run[k] + epsilon));
        StoreValues<U>(run, xInvStdDev + idxDstBase);
    }
}

// This kernel is very similar to kComputeBatchMeanAndInvStdDev except it reduces not just over N (minibatch)
// but also W and H dimensions.
// REVIEW alexeyk: is it possible to combine this and previous kernel into a single kernel without hurting performance/readability much?
template <int BlockDimX, int BlockDimY, int U, typename ElemType, typename StatType>
__global__ void kComputeSpatialBatchMeanAndInvStdDev(int vectorSize, int spatialSize, int batchSize, const ElemType* x,
                                                     double expAvgFactor, double blendFactor,
                                                     StatType* runMean, StatType* runVariance,
                                                     double epsilon, StatType* xMean, StatType* xInvStdDev)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    static_assert(BlockDimX * U == CUB_PTX_WARP_THREADS, "BlockDimX * U must be equal to warp size (32).");
    static_assert((BlockDimX * BlockDimY % CUB_PTX_WARP_THREADS) == 0, "Block size must be a multiple of warp size (32).");
    assert(blockDim.x == BlockDimX);
    assert(blockDim.y == BlockDimY);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert((spatialSize % U) == 0);
    assert((vectorSize % spatialSize) == 0);
    assert(::isfinite(expAvgFactor) && 0 <= expAvgFactor && expAvgFactor <= 1);
    assert(::isfinite(blendFactor) && 0 <= blendFactor && blendFactor <= 1);
    assert(::isfinite(epsilon) && epsilon > 0);
    assert(batchSize >= 1);

    if (expAvgFactor != 0 || blendFactor != 1)
    {
        int irowSrcBase = blockIdx.x * spatialSize + threadIdx.x * U;
        if (irowSrcBase >= vectorSize)
            return;
        assert(irowSrcBase + U <= vectorSize);
        int irowSrcLim = (blockIdx.x + 1) * spatialSize;

        int n = 0;
        comp_t mean[U];
        comp_t m2[U];
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            mean[k] = 0;
            m2[k] = 0;
        }

        int icolSrc = threadIdx.y;
        const ElemType* psrcBase = x + static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
        // Stride over all vectors in the batch.
        for (; icolSrc < batchSize; icolSrc += BlockDimY)
        {
            const ElemType* psrc = psrcBase;
            // Stride over all values in feature map (W and H dimensions).
            for (int irowSrc = irowSrcBase; irowSrc < irowSrcLim; irowSrc += BlockDimX * U, psrc += BlockDimX * U)
            {
                n++;
                comp_t curVal[U];
                LoadValues<U>(psrc, curVal);
                // No need for separate unrolling, SASS looks good.
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    comp_t d = curVal[k] - mean[k];
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
        unsigned int mask;
#if CUDA_VERSION >= 9000
        mask = __ballot_sync(0xffffffff, n);
#endif
        if (n > 0)
        {
#pragma unroll
            for (int i = 1; i < CUB_PTX_WARP_THREADS; i *= 2)
            {
                int srcLane = laneId + i;
                int n2 = Shuffle(n, srcLane, mask);
                int nsum = n + n2;
                comp_t d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    d[k] = Shuffle(mean[k], srcLane, mask) - mean[k];
                    comp_t dScaled = d[k] * n2 / nsum;
                    mean[k] += dScaled;
                    m2[k] += Shuffle(m2[k], srcLane, mask) + d[k] * n * dScaled;
                }
                n = nsum;
            }
        }

        // Storage for each warp in a thread block. First warp ("accumulator") holds
        // final results so it does not need shared memory.
        const int cwarp = BlockDimX * BlockDimY / CUB_PTX_WARP_THREADS;
        __shared__ comp_t meanRes[U][cwarp - 1];
        __shared__ comp_t m2Res[U][cwarp - 1];
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
                comp_t d[U];
#pragma unroll
                for (int k = 0; k < U; k++)
                {
                    d[k] = meanRes[k][i] - mean[k];
                    comp_t dScaled = d[k] * n2 / nsum;
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
                comp_t d = mean[k] - mean[0];
                comp_t dScaled = d * n / (n + k * n);
                mean[0] += dScaled;
                m2[0] += m2[k] + d * k * n * dScaled;
            }

            // TODO add back special cases
            runMean[blockIdx.x] = expAvgFactor * mean[0] + (1.0 - expAvgFactor) * runMean[blockIdx.x];
            xMean[blockIdx.x] = blendFactor * runMean[blockIdx.x] + (1.0 - blendFactor) * mean[0];

            comp_t runV = batchSize * spatialSize == 1 ? 0 : m2[0] / (batchSize * spatialSize - 1);
            runVariance[blockIdx.x] = expAvgFactor * runV + (1.0 - expAvgFactor) * runVariance[blockIdx.x];
            xInvStdDev[blockIdx.x] = Operations::RSqrt(static_cast<comp_t>(m2[0] / (batchSize * spatialSize) + (comp_t)epsilon));
            if (blendFactor != 0)
            {
                comp_t runInvStdDev = Operations::RSqrt(static_cast<comp_t>((comp_t)runVariance[blockIdx.x] + (comp_t)epsilon));
                xInvStdDev[blockIdx.x] = blendFactor * runInvStdDev + (1.0 - blendFactor) * xInvStdDev[blockIdx.x];
            }
        }
    }
    else if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        xMean[blockIdx.x] = runMean[blockIdx.x];
        xInvStdDev[blockIdx.x] = Operations::RSqrt(static_cast<comp_t>((comp_t)runVariance[blockIdx.x] + (comp_t)epsilon));
    }
}

// The struct is used by Call function to select proper template in runtime based on the size of the vector.
// The same pattern is used in other cases of similar structs.
template <int U>
struct ComputeBatchMeanAndInvStdDev
{
    template <typename ElemType, typename StatType>
    static void Call(size_t vectorSize, size_t batchSize,
                     const ElemType* x,                         // (in) input data
                     double expAvgFactor,
                     double blendFactor,
                     StatType* runMean, StatType* runVariance,  // (in/out) running mean/variance, gets updated with current minibatch
                     double epsilon,
                     StatType* xMean, StatType* xInvStdDev,     // (out) actual interpolated mean/stddev that are used to normalize. Returned since needed in backprop.
                     cudaStream_t stream)
    {
        assert((vectorSize % U) == 0);
        assert(batchSize >= 1);

        const int BlockDimX = 32 / U;
        const int BlockDimY = 4 * U;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create grid with only one block in y(batch)-dimension as kernel uses striding.
        auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)));
        kComputeBatchMeanAndInvStdDev<BlockDimX, BlockDimY, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
            static_cast<int>(vectorSize), static_cast<int>(batchSize),
            x, expAvgFactor, blendFactor, runMean, runVariance, epsilon, xMean, xInvStdDev);
    }
};

template <int U>
struct ComputeSpatialBatchMeanAndInvStdDev
{
    template <typename ElemType, typename StatType>
    static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, const ElemType* x,
                        double expAvgFactor, double blendFactor, StatType* runMean, StatType* runVariance,
                        double epsilon, StatType* xMean, StatType* xInvStdDev, cudaStream_t stream)
    {
        assert((vectorSize % spatialSize) == 0);
        assert((spatialSize % U) == 0);
        assert(batchSize >= 1);

        const int BlockDimX = 32 / U;
        const int BlockDimY = 4 * U;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create grid with only one block in y(batch)-dimension as kernel uses striding.
        // Each thread block processes a single whole feature map independently (i.e. reduces over W, H and N dimensions).
        auto gdim = dim3(static_cast<unsigned int>(vectorSize / spatialSize));
        kComputeSpatialBatchMeanAndInvStdDev<BlockDimX, BlockDimY, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
            static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize),
            x, expAvgFactor, blendFactor, runMean, runVariance, epsilon, xMean, xInvStdDev);
    }
};

//--------------------------------------------------------------------
// Forward propagation
// All functions accept input/outputs tensors in column-major format where each column is a vector of a minibatch.
// In convolutional case (i.e. spatial=true), each vector is in CHW format where W dimension has stride = 1.
// Tensors for biases and inverse stddevs have dimensions that equal to vector dimension in non-convolutional (i.e. spatial=false)
// or Cx1x1 in convolutional case.
//--------------------------------------------------------------------

template <int BlockDimX, int BlockDimY, bool Spatial, bool NormalizeRunningStats, int U, typename ElemType, typename StatType>
__global__ void kNormalizeBatchTraining(int vectorSize, int spatialSize, int batchSize,
    double epsilon,
    const ElemType* x, ElemType* y,
    const StatType* bnScale, const StatType* bnBias,
    const StatType* runningMean, const StatType* runningVariance,
    const StatType* batchMean, StatType* batchInvStdDev)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
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

    __shared__ comp_t meanS[BlockDimX * U];
    __shared__ comp_t invStdDevS[BlockDimX * U];
    __shared__ comp_t scaleS[BlockDimX * U];
    __shared__ comp_t biasS[BlockDimX * U];
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
                meanS[offs + k] = NormalizeRunningStats ? runningMean[imap] : batchMean[imap];
                invStdDevS[offs + k] = NormalizeRunningStats
                    ? Operations::RSqrt(static_cast<comp_t>((comp_t)runningVariance[imap] + (comp_t)epsilon))
                    : (comp_t)batchInvStdDev[imap];
                scaleS[offs + k] = bnScale[imap];
                biasS[offs + k] = bnBias[imap];
            }
        }
        else
        {
            LoadValues<U>((NormalizeRunningStats ? runningMean : batchMean) + irowBase, meanS + offs);
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                invStdDevS[offs + k] = NormalizeRunningStats
                    ? Operations::RSqrt(static_cast<comp_t>((comp_t)runningVariance[irowBase + k] + (comp_t)epsilon))
                    : (comp_t)batchInvStdDev[irowBase + k];
            }
            LoadValues<U>(bnScale + irowBase, scaleS + offs);
            LoadValues<U>(bnBias + irowBase, biasS + offs);
        }
    }
    __syncthreads();
    comp_t mean[U];
    comp_t invStdDev[U];
    comp_t scale[U];
    comp_t bias[U];
    LoadValues<U>(meanS + offs, mean);
    LoadValues<U>(invStdDevS + offs, invStdDev);
    LoadValues<U>(scaleS + offs, scale);
    LoadValues<U>(biasS + offs, bias);

    int icol = blockIdx.y * BlockDimY + threadIdx.y;
    size_t startOffs = static_cast<size_t>(icol) * vectorSize + irowBase;
    const ElemType* psrc = x + startOffs;
    ElemType* pdst = y + startOffs;
    size_t stride = static_cast<size_t>(gridDim.y * BlockDimY) * vectorSize;
    for (; icol < batchSize; icol += gridDim.y * BlockDimY, psrc += stride, pdst += stride)
    {
        comp_t val[U];
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
    template <typename ElemType, typename StatType>
    static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, bool spatial,
                     bool normalizeRunningStats, double epsilon,
                     const ElemType* x, ElemType* y,                               // (in, out) data to normalize -> normalized data
                     const StatType* bnScale, const StatType* bnBias,              // (in) scale/bias to denormalize with
                     const StatType* runningMean, const StatType* runningVariance, // (in) running mean/variance
                     const StatType* batchMean, StatType* batchInvStdDev,          // (in) batch mean/stddev to normalize with
                     cudaStream_t stream)
    {
        assert((vectorSize % U) == 0);
        assert(batchSize >= 1);

        const int BlockDimX = 32 / U;
        const int BlockDimY = 4 * U;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create a grid that has uses striding in y-dimension to cover whole minibatch.
        auto gdim = dim3((unsigned int)RoundUpToMultiple(vectorSize, BlockDimX * U));
        if (spatial)
        {
            if (normalizeRunningStats)
                kNormalizeBatchTraining<BlockDimX, BlockDimY, true, true, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
                    (int)vectorSize, (int)spatialSize, (int)batchSize,
                    epsilon,
                    x, y, bnScale, bnBias,
                    runningMean, runningVariance,
                    batchMean, batchInvStdDev);
            else
                kNormalizeBatchTraining<BlockDimX, BlockDimY, true, false, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
                    (int)vectorSize, (int)spatialSize, (int)batchSize,
                    epsilon,
                    x, y, bnScale, bnBias,
                    runningMean, runningVariance,
                    batchMean, batchInvStdDev);
        }
        else
        {
            if (normalizeRunningStats)
                kNormalizeBatchTraining<BlockDimX, BlockDimY, false, true, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
                    (int)vectorSize, (int)spatialSize, (int)batchSize,
                    epsilon,
                    x, y, bnScale, bnBias,
                    runningMean, runningVariance,
                    batchMean, batchInvStdDev);
            else
                kNormalizeBatchTraining<BlockDimX, BlockDimY, false, false, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
                    (int)vectorSize, (int)spatialSize, (int)batchSize,
                    epsilon,
                    x, y, bnScale, bnBias,
                    runningMean, runningVariance,
                    batchMean, batchInvStdDev);

        }
    }
};

//--------------------------------------------------------------------
// Backpropagation
// BatchNormalizationBackward back-propagates derivatives of batch normalization function
// with respect to the inputs and scale and bias parameters.
// All tensor dimensions and assumptions are the same as in case of forward propagation.
//--------------------------------------------------------------------

template <int BlockDimX, int BlockDimY, int U, typename ElemType, typename StatType>
__global__ void kComputeScaleAndBiasGradients(int vectorSize, int batchSize, const ElemType* x, const ElemType* dy, StatType* dScale, StatType* dBias,
                                              const StatType* savedMean, const StatType* savedInvStdDev)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
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

    comp_t mean[U];
    comp_t invStdDev[U];
    __shared__ comp_t meanS[BlockDimX * U];
    __shared__ comp_t invStdDevS[BlockDimX * U];
    // Read mean and inv std dev.
    if (threadIdx.y == 0)
    {
        LoadValues<U>(savedMean + irowSrcBase, mean);
        LoadValues<U>(savedInvStdDev + irowSrcBase, invStdDev);
        StoreValues<U>(mean, &meanS[threadIdx.x * U]);
        StoreValues<U>(invStdDev, &invStdDevS[threadIdx.x * U]);
    }
    __syncthreads();
    if (threadIdx.y != 0)
    {
        LoadValues<U>(&meanS[threadIdx.x * U], mean);
        LoadValues<U>(&invStdDevS[threadIdx.x * U], invStdDev);
    }

    comp_t ds[U];
    comp_t db[U];
#pragma unroll
    for (int k = 0; k < U; k++)
    {
        ds[k] = 0;
        db[k] = 0;
    }

    int icolSrc = threadIdx.y;
    size_t startOffs = static_cast<size_t>(icolSrc) * vectorSize + irowSrcBase;
    const ElemType* px = x + startOffs;
    const ElemType* pdy = dy + startOffs;
    size_t stride = static_cast<size_t>(vectorSize) * BlockDimY;
    // Stride over all vectors in the batch.
    for (; icolSrc < batchSize; icolSrc += BlockDimY, px += stride, pdy += stride)
    {
        comp_t curX[U];
        comp_t curdY[U];
        LoadValues<U>(px, curX);
        LoadValues<U>(pdy, curdY);
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            ds[k] += (comp_t)pdy[k] * (curX[k] - mean[k]) * invStdDev[k];
            db[k] += (comp_t)pdy[k];
        }
    }

    // Final reduction.
    __shared__ comp_t dsS[BlockDimY][BlockDimX * U];
    __shared__ comp_t dbS[BlockDimY][BlockDimX * U];
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

template <int BlockDimX, int BlockDimY, int U, typename ElemType, typename StatType>
__global__ void kComputeSpatialScaleAndBiasGradients(int vectorSize, int spatialSize, int batchSize, const ElemType* x, const ElemType* dy,
                                                        StatType* dScale, StatType* dBias, const StatType* savedMean, const StatType* savedInvStdDev)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
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

    comp_t mean;
    comp_t invStdDev;
    __shared__ comp_t meanS;
    __shared__ comp_t invStdDevS;
    const int tid = threadIdx.y * BlockDimX + threadIdx.x;
    // Read mean and inv std dev.
    if (tid == 0)
    {
        meanS = savedMean[blockIdx.x];
        invStdDevS = savedInvStdDev[blockIdx.x];
    }
    __syncthreads();
    if (tid != 0)
    {
        mean = meanS;
        invStdDev = invStdDevS;
    }

    comp_t ds[U];
    comp_t db[U];
#pragma unroll
    for (int k = 0; k < U; k++)
    {
        ds[k] = 0;
        db[k] = 0;
    }

    int icolSrc = threadIdx.y;
    size_t startOffs = static_cast<size_t>(icolSrc) * vectorSize + irowBase;
    const ElemType* pxBase = x + startOffs;
    const ElemType* pdyBase = dy + startOffs;
    size_t stride = static_cast<size_t>(vectorSize) * BlockDimY;
    // Stride over all vectors in the batch.
    for (; icolSrc < batchSize; icolSrc += BlockDimY, pxBase += stride, pdyBase += stride)
    {
        const ElemType* px = pxBase;
        const ElemType* pdy = pdyBase;
        // Stride over all values in feature map (W and H dimensions).
        for (int irow = irowBase; irow < irowLim; irow += BlockDimX * U, px += BlockDimX * U, pdy += BlockDimX * U)
        {
            comp_t curX[U];
            comp_t curdY[U];
            LoadValues<U>(px, curX);
            LoadValues<U>(pdy, curdY);
#pragma unroll
            for (int k = 0; k < U; k++)
            {
                ds[k] += (comp_t)pdy[k] * (curX[k] - mean) * invStdDev;
                db[k] += (comp_t)pdy[k];
            }
        }
    }
    __syncthreads();
    using BlockReduce = cub::BlockReduce<comp_t, BlockDimX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BlockDimY>;
    // Note: must use separate temp storages for each reduction.
    __shared__ typename BlockReduce::TempStorage tmp1;
    comp_t dsRes = BlockReduce(tmp1).Sum(ds);
    __shared__ typename BlockReduce::TempStorage tmp2;
    comp_t dbRes = BlockReduce(tmp2).Sum(db);
    if (tid == 0)
    {
        dScale[blockIdx.x] = dsRes;
        dBias[blockIdx.x] = dbRes;
    }
}

template <int U>
struct ComputeScaleAndBiasGradients
{
    template <typename ElemType, typename StatType>
    static void Call(size_t vectorSize, size_t batchSize, const ElemType* x, const ElemType* dy,
        StatType* dScale, StatType* dBias, const StatType* savedMean, const StatType* savedInvStdDev, cudaStream_t stream)
    {
        assert((vectorSize % U) == 0);
        assert(batchSize >= 1);
        const int BlockDimX = 32 / U;
        const int BlockDimY = 4 * U;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create a grid that has uses striding in y-dimension to cover whole minibatch.
        auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)));
        kComputeScaleAndBiasGradients<BlockDimX, BlockDimY, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
            static_cast<int>(vectorSize), static_cast<int>(batchSize), x, dy, dScale, dBias, savedMean, savedInvStdDev);
    }
};

template <int U>
struct ComputeSpatialScaleAndBiasGradients
{
    template <typename ElemType, typename StatType>
    static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, const ElemType* x, const ElemType* dy,
                     StatType* dScale, StatType* dBias, const StatType* savedMean, const StatType* savedInvStdDev, cudaStream_t stream)
    {
        assert((spatialSize % U) == 0);
        assert((vectorSize % spatialSize) == 0);
        assert(batchSize >= 1);

        const int BlockDimX = 32 / U;
        const int BlockDimY = 4 * U;
        auto bdim = dim3(BlockDimX, BlockDimY);
        // Create a grid that has uses striding in y-dimension to cover whole minibatch.
        auto gdim = dim3(static_cast<unsigned int>(vectorSize / spatialSize));
        kComputeSpatialScaleAndBiasGradients<BlockDimX, BlockDimY, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
            static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, dy, dScale, dBias, savedMean, savedInvStdDev);
    }
};

// mbStatsWeight is the weight with which current MB's stats were used (0 means not at all, locked model).
template <int BlockDimX, int BlockDimY, bool Spatial, int U, typename ElemType, typename StatType>
__global__ void kBackpropagateBatchNormGradients(int vectorSize, int spatialSize, int batchSize, const ElemType* x, const ElemType* dy, ElemType* dx,
                                                    const StatType* bnScale, StatType mbStatsWeight, const StatType* dScale, const StatType* dBias,
                                                    const StatType* savedMean, const StatType* savedInvStdDev)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
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
    comp_t scale[U];
    comp_t ds[U];
    comp_t db[U];
    comp_t mean[U];
    comp_t invStdDev[U];
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
            mean[k] = savedMean[imap];
            invStdDev[k] = savedInvStdDev[imap];
        }
    }
    else
    {
        LoadValues<U>(bnScale + irowBase, scale);
        LoadValues<U>(dScale + irowBase, ds);
        LoadValues<U>(dBias + irowBase, db);
        LoadValues<U>(savedMean + irowBase, mean);
        LoadValues<U>(savedInvStdDev + irowBase, invStdDev);
    }

    int icol = blockIdx.y * BlockDimY + threadIdx.y;
    size_t startOffs = static_cast<size_t>(icol) * vectorSize + irowBase;
    const ElemType* px = x + startOffs;
    const ElemType* pdy = dy + startOffs;
    ElemType* pdx = dx + startOffs;
    size_t stride = static_cast<size_t>(gridDim.y * BlockDimY) * vectorSize;
    for (; icol < batchSize; icol += gridDim.y * BlockDimY, px += stride, pdy += stride, pdx += stride)
    {
        comp_t xCur[U];
        comp_t dyCur[U];
        comp_t dxCur[U];
        LoadValues<U>(px, xCur);
        LoadValues<U>(pdy, dyCur);
        LoadValues<U>(pdx, dxCur);
        // From the BN paper, dL/dxi is a sum of three terms: dL/dxi = t1 + t2 + t3
        // The formulas for dBias and dScale happen to occur as subexpressions in this gradient as well.
        // Leveraging this, this gradient can be simplified to:
        //   t1 = scale * dL/dyi * invStdDev
        //   t2 = mbStatsWeight * (-scale / m) * invStdDev * xHat * dL/dScale
        //   t3 = mbStatsWeight * (-scale / m) * invStdDev * dL/dBias (for this one note that Sum(xHat) == 0)
        // with
        //   dBias = Reduce(dy)
        //   dScale = Reduce(dy * xHat)
        // Simplifying this a bit more, we get the formula below.
        comp_t val[U];
        int m = Spatial ? batchSize * spatialSize : batchSize;
#pragma unroll
        for (int k = 0; k < U; k++)
        {
            comp_t xNorm = (xCur[k] - mean[k]) * invStdDev[k]; // xHat
            // scale * invStdDev * (
            //   dL/dyi
            //   - mbStatsWeight * (xHat * dL/dScale + dL/dBias) / m
            // )
            val[k] = dxCur[k]   // (adding to gradient)
                     + (scale[k] * invStdDev[k]) * (
                        dyCur[k]
                        - (comp_t)mbStatsWeight * (xNorm * ds[k] + db[k]) / m);
        }
        StoreValues<U>(val, pdx);
    }
}

template <int U>
struct BackpropagateBatchNormGradients
{
    template <typename ElemType, typename StatType>
    static void Call(size_t vectorSize, size_t spatialSize, size_t batchSize, bool spatial, const ElemType* x, const ElemType* dy, ElemType* dx,
                     const StatType* bnScale, StatType mbStatsWeight, const StatType* dScale,
                     const StatType* dBias, const StatType* savedMean, const StatType* savedInvStdDev, cudaStream_t stream)
    {
        assert((vectorSize % U) == 0);
        assert(batchSize >= 1);
        const int BlockDimX = 32 / U;
        const int BlockDimY = 4 * U;
        auto bdim = dim3(BlockDimX, BlockDimY);
        auto gdim = dim3(static_cast<unsigned int>(RoundUpToMultiple(vectorSize, BlockDimX * U)),
                         static_cast<unsigned int>(RoundUpToMultiple(batchSize,  BlockDimY)));
        if (spatial)
        {
            kBackpropagateBatchNormGradients<BlockDimX, BlockDimY, true/*spatial*/, U, ElemType, StatType><<<gdim, bdim, 0, stream>>>(
                static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, dy, dx, bnScale, mbStatsWeight, dScale, dBias, savedMean, savedInvStdDev);
        }
        else
        {
            kBackpropagateBatchNormGradients<BlockDimX, BlockDimY, false/*not spatial*/, U><<<gdim, bdim, 0, stream>>>(
                static_cast<int>(vectorSize), static_cast<int>(spatialSize), static_cast<int>(batchSize), x, dy, dx, bnScale, mbStatsWeight, dScale, dBias, savedMean, savedInvStdDev);
        }
    }
};

}}}
