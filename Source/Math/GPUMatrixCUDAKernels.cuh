//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "BestGpu.h"

#ifndef CPUONLY

#pragma push_macro("TENSOR_OPS_DECL")
#define TENSOR_OPS_DECL __device__ __host__
#include "CommonMatrix.h"
#include "GPUMatrix.h"
#include "TensorOps.h" // for exp_() etc.
#include "device_functions.h"
#include <cuda_runtime.h>
#include <assert.h>
#include <float.h>
#pragma pop_macro("TENSOR_OPS_DECL")

// REVIEW alexeyk: disable warnings properly for GCC/clang
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100) // 'identifier': unreferenced formal parameter
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4458) // declaration of 'identifier' hides class member
#pragma warning(disable : 4515) // 'namespace': namespace uses itself
#endif
#include <cub/cub.cuh>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

// special markers in BlockId2ColOrRow()/ColOrRow2BlockId()
static const GPUSPARSE_INDEX_TYPE Id_NotAssigned = -1;
static const GPUSPARSE_INDEX_TYPE Id_Pending = INT_MAX;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing

// On older GPUs, CUDA atomicAdd() only exists for 'float'. This is the 'double' version.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// TODO: replace this with TensorOps.h LogAdd(). It differs in using ElemType throughout, while this one seems to use 'double' versions of exp() and log().
// The 'k' in the name is to avoid naming conflicts with various versions of logadd() that are defined throughout the codebase.
template <class ElemType>
static inline __device__ __host__ ElemType logaddk(ElemType x, ElemType y)
{
    ElemType temp, diff, z;

    if (x < y)
    {
        temp = x;
        x = y;
        y = temp;
    }
    diff = y - x;
    if (diff < MINLOGEXP)
    {
        return (x < LSMALL) ? LZERO : x;
    }
    else
    {
        z = exp(diff);
        return x + log(1.0 + z);
    }
}

namespace Microsoft { namespace MSR { namespace CNTK {

// ---------------------------------------------------------------------------
// GridDim -- helper to choose the CUDA grid dimensions
// ---------------------------------------------------------------------------

template <class INT, class INT2>
static INT CeilDiv(INT a, INT2 b) // ceil(a/b)
{
    return (INT)(((size_t) a + (size_t) b - 1) / (size_t) b); // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim
{
    static const CUDA_LONG maxThreadsPerBlock = 1024; // use this many threads per block
    static const CUDA_LONG maxWarpsPerBlock = 32;     // use this many warps per block. This means 1024 threads for warpSize=32

    // use these for launching
    //   GridDim grid(NN);
    //   kernel<<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, ...>>>(...)
    int m_blocksPerGrid, m_threadsPerBlock; // (these may in the future be extended to multi-dimensional ones)
    CUDA_LONG m_N;

    GridDim(CUDA_LONG N) // linear grid
    {
        m_N = N;
        if (N == 0) // CUDA will fail to launch with 0 blocks
            N = 1;

        // get device information
        const auto& props = GetDeviceProps();
        CUDA_LONG numProcs = props.multiProcessorCount;
        CUDA_LONG warpSize = props.warpSize;

        // distribute warps evenly over processors
        CUDA_LONG warpsPerProc = CeilDiv(N, numProcs * warpSize);

        // if too many warps per block then reduce #warps
        // This limits the number of threads to 512.
        if (warpsPerProc > maxWarpsPerBlock)
        {
            CUDA_LONG overBy = CeilDiv(warpsPerProc, maxWarpsPerBlock); // we are over by this factor
            warpsPerProc = CeilDiv(warpsPerProc, overBy);
        }

        // put it back together
        m_threadsPerBlock = warpsPerProc * warpSize;        // =a multiple of 32 that is as close to 1024 as makes sense given NN
        m_blocksPerGrid = CeilDiv(N, m_threadsPerBlock);
        if (m_blocksPerGrid == 1)
            m_threadsPerBlock = N; // don't launch more than necessary  --TODO: Does this make a difference at all?
        assert(m_blocksPerGrid * m_threadsPerBlock >= N);
    }

    static const std::vector<cudaDeviceProp>& GetCachedDeviceProps()
    {
        std::call_once(s_cachedDevicePropsInitFlag, [=]{
            int numDevices;
            CUDA_CALL(cudaGetDeviceCount(&numDevices));
            s_cachedDeviceProps.resize(numDevices);
            for (int i = 0; i < numDevices; i++)
                CUDA_CALL(cudaGetDeviceProperties(&s_cachedDeviceProps[i], i));
        });
       
        return s_cachedDeviceProps;
    }

    static size_t GetCurrentDeviceId()
    {
        int deviceId;
        cudaGetDevice(&deviceId);
        return (size_t)deviceId;
    }


    // get device properties of current device
    static const cudaDeviceProp& GetDeviceProps()
    {
        const auto& cachedDevicesProps = GetCachedDeviceProps();
        return cachedDevicesProps[GetCurrentDeviceId()];
    }

    // compute our location on the grid
    static __device__ CUDA_LONG GetLinearThreadId()
    {
        return blockDim.x * blockIdx.x + threadIdx.x;
    }

private: 
    // TODO: drop call_once and co. and make cached devices a local static, once we're on VS2015.
    static std::vector<cudaDeviceProp> s_cachedDeviceProps;
    static std::once_flag s_cachedDevicePropsInitFlag;
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
    CUDA_LONG id = GridDim::GetLinearThreadId();   \
    if (id >= N)                                   \
        return;

#ifdef __GNUC__
#define UNUSED_FUNCTION_ATTRIBUTE __attribute__((unused))
#else
#define UNUSED_FUNCTION_ATTRIBUTE
#endif

// ===========================================================================
// CUDA kernels follow, lots of them
// ===========================================================================

// _elementWise*() kernels
//
// Designed to operate on contiguous blocks of memory, where the output is a simple function of the inputs.
// The first parameters of every function are inputs, and the last two arguments to each function are always
// (ElemenType *res, CUDA_LONG N), a pointer and length of the output block. Each thread computes a function
// of the inputs for one value in the output.

template <class ElemType>
__global__ void _elementWisePowerOnCuda(
    const ElemType alpha,
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    if (alpha == 0)
    {
        res[id] = 1;
    }
    else if (alpha == 1)
    {
        res[id] = a[id];
    }
    else if (alpha == 2)
    {
        res[id] = a[id] * a[id];
    }
    else if (alpha == 3)
    {
        res[id] = a[id] * a[id] * a[id];
    }
    else
    {
        if (sizeof(ElemType) == sizeof(double))
        {
            res[id] = pow(a[id], alpha);
        }
        else
        {
            res[id] = powf(a[id], alpha);
        }
    }
};

// Note that this code is inefficient on CUDA due to diverging code paths.
// Use Sigmoid() in TensorOps.h instead, which solves this problem.
template <class ElemType>
__global__ void _elementWiseSigmoidOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
#if 0 // this computes the same thing but is twice as fast on CUDA
    res[id] = Microsoft::MSR::CNTK::Sigmoid(a[id]);
#else
    if (a[id] >= 0)
    {
        ElemType e = exp_(-a[id]);
        res[id] = 1 / (1 + e);
    }
    else
    {
        ElemType e = exp_(a[id]);
        res[id] = e / (1 + e);
    }
#endif
};

template <class ElemType>
__global__ void _assignSigmoidOf(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

// This function computes 1 / (1 + e^(-x)) which yields 1 / (1 + e^|x|) if x is negative,
// and e^x / (1 + e^x) if x is positive.
// BUGBUG: This does not invert the calculation when the exp argument becomes large, potentially causing overflows.
//         There is a second version of this function that does. That should be used.
#if 0 // this has the same speed now, although not identical accuracy
    res[id] = Microsoft::MSR::CNTK::Sigmoid(a[id]);
#else
    ElemType negElem = -a[id];
    ElemType e = exp_(negElem);

    res[id] = 1 / (e + 1);
#endif
};

template <class ElemType>
__global__ void _elementWiseLinRectDerivativeOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = (a[id] <= 0) ? 0 : 1;
}

template <class ElemType>
__global__ void _elementWiseSigmoidDerivativeOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = a[id] * (1 - a[id]);
}

template <class ElemType>
__global__ void _elementWiseTanhOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = tanh_(a[id]);
};

//to prevent negative values caused by floating operations, we force inputs to be >=0
//this may, however, hide problems in the caller.
template <class ElemType>
__global__ void _elementWiseSqrtOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = sqrt_(max((ElemType) 0, a[id]));
};

template <class ElemType>
__global__ void _elementWiseExpOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = exp_(a[id]);
};

template <class ElemType>
__global__ void _elementWiseLogOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = (a[id] < EPS_IN_LOG) ? LOG_OF_EPS_IN_LOG : log_(a[id]);
};

template <class ElemType>
__global__ void _elementWiseAbsOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = fabs_(a[id]);
};

template <class ElemType>
__global__ void _elementWiseCosineOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = cos_(a[id]);
};

template <class ElemType>
__global__ void _elementWiseNegativeSineOnCuda(
    const ElemType* a,
    ElemType* res,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    res[id] = -sin_(a[id]);
};

template <class ElemType>
__global__ void _setValue(
    ElemType* a,
    const ElemType v,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] = v;
};

template <class ElemType>
__global__ void _setValue(
    ElemType* a,
    const ElemType* d_v,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] = d_v[0];
};

template <class ElemType>
__global__ void _copyColumnsStrided(ElemType* dest, ElemType* src, CUDA_LONG N, CUDA_LONG numRows, CUDA_LONG destNumColsStride, CUDA_LONG srcNumColsStride)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG denseColIdx = id / numRows;
    CUDA_LONG rowIdx = id - (denseColIdx * numRows);

    dest[(denseColIdx * destNumColsStride * numRows) + rowIdx] = src[(denseColIdx * srcNumColsStride * numRows) + rowIdx];
}

template <class ElemType>
__global__ void _assignToRowSliceValuesOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / srcRows;
    CUDA_LONG row = id - (col * srcRows);

    dest[col * destRows + row + startIndex] = src[id];
}

template <class ElemType>
__global__ void _assignRowSliceValuesOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / destRows;
    CUDA_LONG row = id - (col * destRows);

    // dest[id] = src[col*srcRows + row + startIndex];
    dest[id] = src[IDX2C(row + startIndex, col, srcRows)];
}

template <class ElemType>
__global__ void _addToRowSliceValuesOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / srcRows; // src is the full matrix, rowslice is taken from the dest
    CUDA_LONG row = id - (col * srcRows);

    // dest[col*destRows + row + startIndex] += src[id];
    dest[IDX2C(row + startIndex, col, destRows)] += src[id];
}

template <class ElemType>
__global__ void _addWithRowSliceValuesOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG startIndex, const CUDA_LONG destRows, const CUDA_LONG srcRows)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / destRows; // dest is the full matrix, rowslice is taken from the src
    CUDA_LONG row = id - (col * destRows);

    dest[id] += src[IDX2C(row + startIndex, col, srcRows)];
}

template <class ElemType>
__global__ void _assignToDiagonalValuesOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG srcCols)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / srcCols;
    CUDA_LONG row = id - (col * srcCols);

    if (row == col)
        dest[row] = src[id];
}

template <class ElemType>
__global__ void _assignRowStackValuesOf(ElemType* dest, ElemType** srces, size_t* startRowIndeces, const CUDA_LONG numSrces, const CUDA_LONG N, const CUDA_LONG destRows, const CUDA_LONG destCols)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / destRows; // dest is the full matrix, rowslice is taken from the src
    CUDA_LONG row = id - (col * destRows);

    // can we replace the for loop with something better?
    int srcId = 0;
    for (; srcId < numSrces; srcId++)
    {
        if (startRowIndeces[srcId + 1] > row)
            break;
    }

    dest[id] = srces[srcId][IDX2C(row - startRowIndeces[srcId], col, startRowIndeces[srcId + 1] - startRowIndeces[srcId])];
}

template <class ElemType>
__global__ void _assignRepeatOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG srcRows, const CUDA_LONG srcCols, const CUDA_LONG destRows)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG destCol = id / destRows;
    CUDA_LONG destRow = id - (destCol * destRows);

    CUDA_LONG srcRow = destRow % srcRows;
    CUDA_LONG srcCol = destCol % srcCols;

    dest[id] = src[IDX2C(srcRow, srcCol, srcRows)];
}

template <class ElemType>
__global__ void _addToRowRepeatValuesOf(ElemType* dest, ElemType* src, const CUDA_LONG N, const CUDA_LONG srcRows, const CUDA_LONG srcCols, const CUDA_LONG destRows)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG col = id / srcRows;
    CUDA_LONG row = (id - (col * srcRows)) % destRows;

    // dest[col*destRows + row + startIndex] += src[id];
    dest[IDX2C(row, col, destRows)] += src[id];
}

template <class ElemType>
__global__ void _assignPositiveAndShiftedNegSample(ElemType* dest, const ElemType* src, const CUDA_LONG N, const CUDA_LONG srcRows, const CUDA_LONG srcCols, const CUDA_LONG destRows, const CUDA_LONG posNumber, const CUDA_LONG shiftNumber)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    CUDA_LONG destCol = id / destRows;
    CUDA_LONG destRow = id - (destCol * destRows);

    CUDA_LONG sampleInDestCol = destRow / srcRows;
    CUDA_LONG srcRow = destRow - srcRows * sampleInDestCol;
    CUDA_LONG srcCol = sampleInDestCol < posNumber ? destCol : (destCol + shiftNumber + sampleInDestCol - posNumber) % srcCols;

    dest[id] = src[IDX2C(srcRow, srcCol, srcRows)];
}

template <class ElemType>
__global__ void _addFoldedPositiveAndShiftedNegSample(ElemType* folded, const ElemType* unfolded, const CUDA_LONG unfoldedN, const CUDA_LONG unfoldedRows, const CUDA_LONG unfoldedCols, const CUDA_LONG foldedRows, const CUDA_LONG posNumber, const CUDA_LONG shiftNumber)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= unfoldedN)
        return;

    CUDA_LONG unfoldedCol = id / unfoldedRows;
    CUDA_LONG unfoldedRow = id - (unfoldedCol * unfoldedRows);

    CUDA_LONG sampleInUnfoldedCol = unfoldedRow / foldedRows;
    CUDA_LONG foldedRow = unfoldedRow - foldedRows * sampleInUnfoldedCol;
    CUDA_LONG foldedCol = sampleInUnfoldedCol < posNumber ? unfoldedCol : (unfoldedCol + shiftNumber + sampleInUnfoldedCol - posNumber) % unfoldedCols;

    atomicAdd(&folded[IDX2C(foldedRow, foldedCol, foldedRows)], unfolded[id]);
}

template <class ElemType>
__global__ void _assignDifferenceOf1(
    ElemType* us,
    const ElemType alpha,
    const ElemType* a,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    us[id] = alpha - a[id];
};

template <class ElemType>
__global__ void _assignDifferenceOf2(
    ElemType* us,
    const ElemType alpha,
    const ElemType* a,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    us[id] = a[id] - alpha;
};

///a is a scalar
template <class ElemType>
__global__ void _scaleAndAddScalar(
    ElemType* c,
    const CUDA_LONG N,
    const ElemType alpha,
    const ElemType* a,
    const ElemType* b)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    c[id] = alpha * a[0] + b[id];
};

template <class ElemType>
__global__ void _multiply1x1AndWeightedAdd(
    ElemType alpha, const ElemType* a, const ElemType* b, ElemType beta, ElemType* c, CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    ElemType f = alpha * *a; // scalar matrix
    if (beta == 0)           // don't even read the memory if beta is 0
        c[id] = b[id] * f;
    else
        c[id] = b[id] * f + c[id] * beta;
}

template <class ElemType>
__global__ void _addValue(
    ElemType* a,
    const ElemType v,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] += v;
};

template <class ElemType>
__global__ void _addValue(
    ElemType* a,
    const ElemType* d_v,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] += d_v[0];
};

template <class ElemType>
__global__ void _elemMul(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] *= b[id];
};

template <class ElemType>
__global__ void _assignElementProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    us[id] = a[id] * b[id];
}

template <class ElemType>
__global__ void _assignKhatriRaoProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG rowsA,
    const CUDA_LONG rowsB,
    const CUDA_LONG cols)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;

    const CUDA_LONG rows = rowsA * rowsB;
    const CUDA_LONG col = id / rows;
    if (col >= cols)
        return;

    const CUDA_LONG row = id % rows;
    const CUDA_LONG rowB = row / rowsA;
    const CUDA_LONG rowA = row % rowsA;

    us[id] = a[rowA + col * rowsA] * b[rowB + col * rowsB];
}

template <class ElemType>
__global__ void _addColumnReshapeProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG rowsB,
    const CUDA_LONG rowsC,
    const CUDA_LONG cols,
    const bool transposeAColumn)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;

    const CUDA_LONG col = id / rowsC;
    if (col >= cols)
        return;

    const CUDA_LONG row = id % rowsC;
    CUDA_LONG bBase = col * rowsB;
    CUDA_LONG aBase = bBase * rowsC;
    ElemType v = 0;

    if (transposeAColumn)
    {
        aBase += row * rowsB;
        for (CUDA_LONG i = 0; i < rowsB; i++)
        {
            v += a[aBase++] * b[bBase++];
        }
    }
    else
    {
        aBase += row;
        for (CUDA_LONG i = 0; i < rowsB; i++)
        {
            v += a[aBase] * b[bBase++];
            aBase += rowsC;
        }
    }
    us[row + col * rowsC] += v;
}

template <class ElemType>
__global__ void _assignElementDivisionOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    ElemType smallValue = EPS_IN_INVERSE;

    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    ElemType v = b[id];

    if (v < 0 && v > -smallValue)
        us[id] = a[id] / (-smallValue);
    else if (v >= 0 && v < smallValue)
        us[id] = a[id] / smallValue;
    else
        us[id] = a[id] / v;
}

template <class ElemType>
__global__ void _elemInverse(
    ElemType* us,
    const CUDA_LONG N)
{
    ElemType smallValue = EPS_IN_INVERSE;

    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    if (us[id] < 0 && us[id] > -smallValue)
        us[id] = 1 / -smallValue;
    else if (us[id] >= 0 && us[id] < smallValue)
        us[id] = 1 / smallValue;
    else
        us[id] = 1 / us[id];
}

template <class ElemType>
__global__ void _logSoftMaxColWise(
    ElemType* a,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows) // ld
{
    int col_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (col_id >= m_numCols)
        return;

    __shared__ ElemType maxV[GridDim::maxThreadsPerBlock];
    __shared__ ElemType Sum[GridDim::maxThreadsPerBlock];
    maxV[threadIdx.x] = a[IDX2C(0, col_id, m_numRows)];
    Sum[threadIdx.x] = 0;

    for (CUDA_LONG i = 0; i < m_numRows; ++i)
    {
        if (a[IDX2C(i, col_id, m_numRows)] > maxV[threadIdx.x])
        {
            maxV[threadIdx.x] = a[IDX2C(i, col_id, m_numRows)];
        }
    }

    for (CUDA_LONG i = 0; i < m_numRows; ++i)
    {
        ElemType tmp = a[IDX2C(i, col_id, m_numRows)] - maxV[threadIdx.x];
        Sum[threadIdx.x] += (sizeof(ElemType) == sizeof(float) ? expf(tmp) : exp(tmp));
    }
    Sum[threadIdx.x] = maxV[threadIdx.x] + (sizeof(ElemType) == sizeof(float) ? logf(Sum[threadIdx.x]) : log(Sum[threadIdx.x]));
    for (CUDA_LONG i = 0; i < m_numRows; ++i)
    {
        a[IDX2C(i, col_id, m_numRows)] -= Sum[threadIdx.x];
    }
}

//template<class ElemType>
//__global__ void _assignColumnwiseSoftmaxOf(
//    const ElemType *a,
//    ElemType* us,
//    const CUDA_LONG m_numCols,
//    const CUDA_LONG m_numRows) // thead per column
//{
//    int col_id = blockDim.x * blockIdx.x + threadIdx.x;
//    if (col_id>=m_numCols)
//        return;
//
//    __shared__ ElemType maxV[GridDim::maxThreadsPerBlock];
//    __shared__ ElemType Sum[GridDim::maxThreadsPerBlock];
//    maxV[threadIdx.x]=a[IDX2C(0,col_id,m_numRows)];
//    Sum[threadIdx.x]=0;
//
//    for (CUDA_LONG i=0;i<m_numRows;++i)
//    {
//        if (a[IDX2C(i,col_id,m_numRows)]>maxV[threadIdx.x])
//        {
//            maxV[threadIdx.x]=a[IDX2C(i,col_id,m_numRows)];
//        }
//    }
//
//    for (CUDA_LONG i=0;i<m_numRows;++i)
//    {
//        if (sizeof(ElemType)==sizeof(float))
//        {
//            us[IDX2C(i,col_id,m_numRows)] = expf(a[IDX2C(i,col_id,m_numRows)]-maxV[threadIdx.x]);
//        }
//        else
//        {
//            us[IDX2C(i,col_id,m_numRows)] = exp(a[IDX2C(i,col_id,m_numRows)]-maxV[threadIdx.x]);
//        }
//        Sum[threadIdx.x] +=  us[IDX2C(i,col_id,m_numRows)];
//    }
//
//    for (CUDA_LONG i=0;i<m_numRows;++i)
//    {
//        us[IDX2C(i,col_id,m_numRows)] /= Sum[threadIdx.x] ;
//    }
//}

// each block processes one column. There must be 512 threads in a block
template <class ElemType>
__global__ void _assignColumnwiseLogSoftmaxOf512Threads(
    const ElemType* a,
    ElemType* us,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows)
{
    // We first find max per column
    __shared__ ElemType partials[512];
    partials[threadIdx.x] = -10000000;

    for (int i = threadIdx.x; i < m_numRows; i += 512)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x], a[IDX2C(i, blockIdx.x, m_numRows)]);
    }
    __syncthreads();

    if (threadIdx.x < 256)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 256], partials[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x < 128)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 128], partials[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x < 64)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 64], partials[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x < 32)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 32], partials[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x < 16)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 16], partials[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 8], partials[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x < 4)
    {
        partials[threadIdx.x] = max(partials[threadIdx.x + 4], partials[threadIdx.x]);
    }
    __syncthreads();

    __shared__ ElemType colMax[1];
    if (threadIdx.x == 0)
    {
        colMax[0] = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
    }
    __syncthreads();
    partials[threadIdx.x] = 0.0f;

    // Now start finding sums
    for (int i = threadIdx.x; i < m_numRows; i += 512)
    {
        ElemType tmp = a[IDX2C(i, blockIdx.x, m_numRows)] - colMax[0];
        us[IDX2C(i, blockIdx.x, m_numRows)] = tmp;
        partials[threadIdx.x] += (sizeof(ElemType) == sizeof(float)) ? expf(tmp) : exp(tmp);
    }
    __syncthreads();

    if (threadIdx.x < 256)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 256];
    }
    __syncthreads();

    if (threadIdx.x < 128)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 128];
    }
    __syncthreads();

    if (threadIdx.x < 64)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 64];
    }
    __syncthreads();

    if (threadIdx.x < 32)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 32];
    }
    __syncthreads();

    if (threadIdx.x < 16)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 16];
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 8];
    }
    __syncthreads();

    if (threadIdx.x < 4)
    {
        partials[threadIdx.x] += partials[threadIdx.x + 4];
    }
    __syncthreads();

    __shared__ ElemType colSum[1];
    if (threadIdx.x == 0)
    {
        colSum[0] = partials[0] + partials[1] + partials[2] + partials[3];
        colSum[0] = (sizeof(ElemType) == sizeof(float)) ? logf(colSum[0]) : log(colSum[0]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < m_numRows; i += 512)
    {
        us[IDX2C(i, blockIdx.x, m_numRows)] -= colSum[0];
    }
}

template <class ElemType>
__global__ void _logSoftMaxRowWise(
    ElemType* a,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows) // ld
{
    int row_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (row_id >= m_numRows)
        return;

    __shared__ ElemType maxV[GridDim::maxThreadsPerBlock];
    __shared__ ElemType Sum[GridDim::maxThreadsPerBlock];
    maxV[threadIdx.x] = a[IDX2C(row_id, 0, m_numRows)];
    Sum[threadIdx.x] = 0;

    for (CUDA_LONG j = 0; j < m_numCols; ++j)
    {
        if (a[IDX2C(row_id, j, m_numRows)] > maxV[threadIdx.x])
        {
            maxV[threadIdx.x] = a[IDX2C(row_id, j, m_numRows)];
        }
    }

    for (CUDA_LONG j = 0; j < m_numCols; ++j)
    {
        ElemType tmp = a[IDX2C(row_id, j, m_numRows)] - maxV[threadIdx.x];
        Sum[threadIdx.x] += sizeof(ElemType) == sizeof(float) ? expf(tmp) : exp(tmp);
    }
    Sum[threadIdx.x] = maxV[threadIdx.x] + (sizeof(ElemType) == sizeof(float) ? logf(Sum[threadIdx.x]) : log(Sum[threadIdx.x]));
    for (CUDA_LONG j = 0; j < m_numCols; ++j)
    {
        a[IDX2C(row_id, j, m_numRows)] -= Sum[threadIdx.x];
    }
}

// each block processes one column. There must be 512 threads in a block
template <class ElemType>
__global__ void _assignColumnwiseHardmaxOf512Threads(
    const ElemType* a,
    ElemType* us,
    const CUDA_LONG m_numCols,
    const CUDA_LONG m_numRows)
{
    // We first find max per column
    __shared__ ElemType partials[512];
    __shared__ int colMaxI[512];
    int row = threadIdx.x % m_numRows;
    colMaxI[threadIdx.x] = row;
    partials[threadIdx.x] = a[IDX2C(row, blockIdx.x, m_numRows)];

    for (int i = threadIdx.x; i < m_numRows; i += 512)
    {
        if (partials[threadIdx.x] < a[IDX2C(i, blockIdx.x, m_numRows)])
        {
            partials[threadIdx.x] = a[IDX2C(i, blockIdx.x, m_numRows)];
            colMaxI[threadIdx.x] = i;
        }
    }
    __syncthreads();

    if (m_numRows > 256)
    {
        if (threadIdx.x < 256)
        {
            int other = threadIdx.x + 256;
            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (m_numRows > 128)
    {
        if (threadIdx.x < 128)
        {
            int other = threadIdx.x + 128;

            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (m_numRows > 64)
    {
        if (threadIdx.x < 64)
        {
            int other = threadIdx.x + 64;
            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (m_numRows > 32)
    {
        if (threadIdx.x < 32)
        {
            int other = threadIdx.x + 32;
            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (m_numRows > 16)
    {
        if (threadIdx.x < 16)
        {
            int other = threadIdx.x + 16;
            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (m_numRows > 8)
    {
        if (threadIdx.x < 8)
        {
            int other = threadIdx.x + 8;
            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (m_numRows > 4)
    {
        if (threadIdx.x < 4)
        {
            int other = threadIdx.x + 4;
            if (partials[threadIdx.x] < partials[other])
            {
                partials[threadIdx.x] = partials[other];
                colMaxI[threadIdx.x] = colMaxI[other];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int i = 1; i < 4 && i < m_numRows; i++)
        {
            if (partials[0] < partials[i])
            {
                partials[0] = partials[i];
                colMaxI[0] = colMaxI[i];
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < m_numRows; i += 512)
    {
        us[IDX2C(i, blockIdx.x, m_numRows)] = (i == colMaxI[0]) ? 1 : 0;
    }
}

template <class ElemType>
__global__ void _assignTruncateBottom(
    ElemType* us,
    const ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    us[id] = a[id] < threshold ? threshold : a[id];
}

template <class ElemType>
__global__ void _assignTruncateTop(
    ElemType* us,
    const ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    us[id] = a[id] > threshold ? threshold : a[id];
}

template <class ElemType>
__global__ void _setToZeroIfAbsLessThan(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    if (sizeof(ElemType) == sizeof(float))
    {
        if (fabsf(a[id]) < threshold)
            a[id] = 0;
    }
    else
    {
        if (fabs(a[id]) < threshold)
            a[id] = 0;
    }
}

template <class ElemType>
__global__ void _areEqual(
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N,
    const ElemType threshold,
    long* d_res)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    if (sizeof(ElemType) == sizeof(float))
    {
        if (fabsf(a[id] - b[id]) > threshold)
        {
            d_res[0] = 0;
        }
    }
    else
    {
        if (fabs(1.0 * a[id] - 1.0 * b[id]) > threshold)
        {
            d_res[0] = 0;
        }
    }
}

// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
template <class ElemType>
__global__ void _tensorShuffleScaleAndAdd(
    ElemType keepWeight, const ElemType* pa, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const ElemType* pb, ElemType* pc)
{
    size_t N = D * S * M * K * T;
    CUDA_LONG na = blockDim.x * blockIdx.x + threadIdx.x; // input tensor of dimension (D x S x M x K x T)
    if (na >= N)
        return;
    // recover the 5 indices from the loop counter
    size_t d = na % D;
    size_t s = (na / D) % S;
    size_t m = (na / D / S) % M;
    size_t k = (na / D / S / M) % K;
    size_t t = (na / D / S / M / K) % T;
    // compute index for the a and b/c tensors
    size_t nb = (((t * S + s) * M + m) * K + k) * D + d; // output tensor of dimension (D x K x M x S x T): k/K and s/S swapped
    // perform the computation
    ElemType cval = keepWeight ? keepWeight * pb[nb] : 0; // if weight is 0 then don't bother to read memory (efficiency) or to multiply (NaN-safe)
    cval += scaleFactor * pa[na];
    pc[nb] = cval;
}

// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
template <class ElemType>
__global__ void _tensorShuffleScaleAndAddRowSparse(
    const ElemType* anzValues, // source nz values
    const GPUSPARSE_INDEX_TYPE* aRowIndex,
    const GPUSPARSE_INDEX_TYPE* aColCSCIndex,
    ElemType* cnzValues, // target nz values
    GPUSPARSE_INDEX_TYPE* cRowIndex,
    GPUSPARSE_INDEX_TYPE* cColCSCIndex,
    size_t D, size_t S, size_t M, size_t K, size_t T,
    size_t nz)
{
    CUDA_LONG N = blockDim.x * blockIdx.x + threadIdx.x; // input tensor of dimension (D x S x M x K x T)
    if (N < aColCSCIndex[0] || N >= aColCSCIndex[T])
        return;

    size_t col;
    for (col = 0; col < T; col++)
    {
        if (aColCSCIndex[col + 1] > N)
            break;
    }

    size_t na = aRowIndex[N];
    int start = aColCSCIndex[col];
    int end = aColCSCIndex[col + 1];

    // recover the 5 indices from the loop counter
    size_t d = (na) % D;
    size_t s = (na / D) % S;
    size_t m = (na / D / S) % M;
    size_t k = (na / D / S / M) % K;

    // compute index for the a and b/c tensors
    size_t nc = ((s * M + m) * K + k) * D + d; // output tensor of dimension (D x K x M x S): k/K and s/S swapped

    int rowIdx = start;
    for (size_t j = start; j < end; j++)
    {
        // recover the 5 indices from the loop counter
        size_t na_i = aRowIndex[j];
        size_t d_i = (na_i) % D;
        size_t s_i = (na_i / D) % S;
        size_t m_i = (na_i / D / S) % M;
        size_t k_i = (na_i / D / S / M) % K;

        // compute index for the a and b/c tensors
        size_t nc_i = ((s_i * M + m_i) * K + k_i) * D + d_i; // output tensor of dimension (D x K x M x S): k/K and s/S swapped
        if (nc_i < nc)
        {
            rowIdx++;
        }
    }

    cnzValues[rowIdx] = anzValues[N];
    cRowIndex[rowIdx] = nc;

    if (N == 0)
    {
        for (int i = 0; i <= T; i++)
        {
            cColCSCIndex[i] = aColCSCIndex[i];
        }
    }
}

template <class ElemType>
__global__ void _hasElement(
    const ElemType* a,
    const CUDA_LONG N,
    ElemType* d_res // [2x1] vector. The first is the value to be compared and the second is the 0/1 to return
    )
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    if (a[id] == d_res[0])
    {
        d_res[1] = 1;
    }
}

template <class ElemType>
__global__ void _setDiagonalValue(
    ElemType* a,
    const ElemType v,
    const CUDA_LONG N,
    const CUDA_LONG ld)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[IDX2C(id, id, ld)] = v;
}

template <class ElemType>
__global__ void _setDiagonalValueFromVector(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[IDX2C(id, id, N)] = b[id];
}

template <class ElemType>
__global__ void _adagrad(
    ElemType* a,
    ElemType* d_v,
    const CUDA_LONG N,
    ElemType* multipliers)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    const ElemType floor = 1e-16f;

    a[id] += d_v[id] * d_v[id];
    ElemType temp = sqrt(a[id] + floor);
    d_v[id] /= temp;

    if (multipliers != nullptr)
        multipliers[id] = 1 / temp;
}

template <class ElemType>
__global__ void _adagrad4BlockSparse(
    ElemType* a,          // dense
    const size_t numRows, // number of rows in a and in d_v
    ElemType* d_v,        // block sparse
    const GPUSPARSE_INDEX_TYPE* blockId2ColOrRow,
    ElemType* multipliers,
    const bool colMajor,
    const size_t len,  // major dim, numRows in colMajor and numcols in rowMajor
    const CUDA_LONG N) // total number of non-zero values
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    const ElemType floor = 1e-16f;
    CUDA_LONG blockid = id / len;
    CUDA_LONG row = colMajor ? id - blockid * len : blockId2ColOrRow[blockid];
    CUDA_LONG col = colMajor ? blockId2ColOrRow[blockid] : id - blockid * len;

    size_t indexInA = row + col * numRows;
    a[indexInA] += d_v[id] * d_v[id];
    ElemType temp = sqrt(a[indexInA] + floor);
    d_v[id] /= temp;

    if (multipliers != nullptr)
        multipliers[id] = 1 / temp;
}

template <class ElemType>
__global__ void _fsadagrad(CUDA_LONG size, ElemType* grad, ElemType* smoothAda, ElemType* smoothMom, ElemType* val,
                           ElemType lr, ElemType mom, ElemType adaWeight, ElemType adaMul, bool unitGainMomentum)
{
    const ElemType unitGainFactor = unitGainMomentum ? (1.0 - mom) : 1.0;
    CUDA_LONG idx = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride)
    {
        ElemType g = grad[idx];
        ElemType adaSqr = adaWeight * smoothAda[idx] + (1.0f - adaWeight) * g * g;
        smoothAda[idx] = adaSqr;
        if (adaSqr != 0.0f)
        {
            ElemType w;
            if (sizeof(ElemType) == sizeof(double))
            {
                w = adaMul * rsqrt(adaSqr);
            }
            else
            {
                w = adaMul * rsqrtf(adaSqr);
            }

            if (w > 10.0f)
                w = 10.0f;
            g *= w;
        }

        if (mom > 0.0f)
        {
            g = mom * smoothMom[idx] + unitGainFactor * g;
            smoothMom[idx] = g;
        }

        g *= lr;
        val[idx] -= g;
    }
}

template<class ElemType>
inline __device__ ElemType _getvalue4BlockSparseCol(ElemType* v, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len, CUDA_LONG idx)
{
    CUDA_LONG col = idx / len;
    CUDA_LONG row = idx - col * len;
    CUDA_LONG blockid = colOrRow2blockId[col];
    return (blockid == Id_NotAssigned) ? 0 : v[blockid * len + row];
}

template<class ElemType>
inline __device__ void _scalevalue4BlockSparseCol(ElemType* v, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len, CUDA_LONG idx, ElemType s)
{
    CUDA_LONG col = idx / len;
    CUDA_LONG row = idx - col * len;
    CUDA_LONG blockid = colOrRow2blockId[col];
    if (blockid != Id_NotAssigned)
    {
        v[blockid * len + row] *= s;
    }
}

template <class ElemType>
__global__ void _fsadagrad4BlockSparseCol(CUDA_LONG size, 
    ElemType* grad_bsc, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len,
    ElemType* smoothAda, ElemType* smoothMom, ElemType* val,
    ElemType lr, ElemType mom, ElemType adaWeight, ElemType adaMul, bool unitGainMomentum)
{
    const ElemType unitGainFactor = unitGainMomentum ? (1.0 - mom) : 1.0;
    CUDA_LONG idx = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride)
    {
        ElemType g = _getvalue4BlockSparseCol(grad_bsc, colOrRow2blockId, len, idx);
        ElemType adaSqr = adaWeight * smoothAda[idx] + (1.0f - adaWeight) * g * g;
        smoothAda[idx] = adaSqr;
        if (adaSqr != 0.0f)
        {
            ElemType w;
            if (sizeof(ElemType) == sizeof(double))
            {
                w = adaMul * rsqrt(adaSqr);
            }
            else
            {
                w = adaMul * rsqrtf(adaSqr);
            }

            if (w > 10.0f)
                w = 10.0f;
            g *= w;
        }

        if (mom > 0.0f)
        {
            g = mom * smoothMom[idx] + unitGainFactor * g;
            smoothMom[idx] = g;
        }

        g *= lr;
        val[idx] -= g;
    }
}

template <class ElemType>
__global__ void _rmsprop_init(
    ElemType* avars, ElemType* signs, ElemType* steps,
    ElemType* curr_grad,
    const CUDA_LONG N)
{
    CUDA_LONG i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    ElemType tmp = curr_grad[i];
    avars[i] = tmp * tmp;
    signs[i] = ElemType(0.0);
    steps[i] = ElemType(0.02);
}

template <class ElemType>
__global__ void _rmsprop_init4BlockSparseCol(
    ElemType* avars, ElemType* signs, ElemType* steps,
    ElemType* curr_grad, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len,
    const CUDA_LONG N)
{
    CUDA_LONG i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    ElemType tmp = _getvalue4BlockSparseCol(curr_grad, colOrRow2blockId, len, i);

    avars[i] = tmp * tmp;
    signs[i] = ElemType(0.0);
    steps[i] = ElemType(0.02);
}

template <class ElemType>
__global__ void _rmsprop(
    ElemType* avars, ElemType* signs, ElemType* steps,
    ElemType* curr_grad,
    const CUDA_LONG N,
    ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN,
    ElemType floor,
    ElemType* upd_gpu,
    ElemType* multipliers)
{
    CUDA_LONG i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    avars[i] = RMS_GAMMA * avars[i] + (ElemType(1.0) - RMS_GAMMA) * (curr_grad[i] * curr_grad[i]);

    // // grad sign base 3: 0->neg, 1->zero, 2->pos
    // const int grad_sign = 1 + (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

    // // signs[i] contains three consecutive grad_sign
    // signs[i]  = 3*(int(signs[i]) % 9) + grad_sign;

    // // update according to the following table:
    // // (!pos,!pos,!pos) or (!neg,!neg,!neg): RMS_WGT_INC
    // // (!neg,!neg,neg) or (!pos,!pos,pos): RMS_WGT_DEC
    // // otherwise: no action

    // switch(int(upd_gpu[int(signs[i])]))
    // {
    // case 0:
    //    steps[i] = max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);
    //    break;
    // case 2:
    //    steps[i] = min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
    //    break;
    // }
    // curr_grad[i] *= steps[i] / sqrt(avars[i] + floor);

    const int grad_sign = (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

    if (signs[i] * grad_sign > 0)
        steps[i] = min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
    else
        steps[i] = max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);

    ElemType temp = steps[i] / sqrt(avars[i] + floor);
    curr_grad[i] *= temp;
    signs[i] = grad_sign;

    if (multipliers != nullptr)
        multipliers[i] = temp;
}

template <class ElemType>
__global__ void _rmsprop4BlockSparseCol(
    ElemType* avars, ElemType* signs, ElemType* steps,
    ElemType* grad_bsc, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len,
    const CUDA_LONG N,
    ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN,
    ElemType floor,
    ElemType* upd_gpu,
    ElemType* multipliers)
{
    CUDA_LONG i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    ElemType g = _getvalue4BlockSparseCol(grad_bsc, colOrRow2blockId, len, i);

    avars[i] = RMS_GAMMA * avars[i] + (ElemType(1.0) - RMS_GAMMA) * (g * g);

    // // grad sign base 3: 0->neg, 1->zero, 2->pos
    // const int grad_sign = 1 + (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

    // // signs[i] contains three consecutive grad_sign
    // signs[i]  = 3*(int(signs[i]) % 9) + grad_sign;

    // // update according to the following table:
    // // (!pos,!pos,!pos) or (!neg,!neg,!neg): RMS_WGT_INC
    // // (!neg,!neg,neg) or (!pos,!pos,pos): RMS_WGT_DEC
    // // otherwise: no action

    // switch(int(upd_gpu[int(signs[i])]))
    // {
    // case 0:
    //    steps[i] = max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);
    //    break;
    // case 2:
    //    steps[i] = min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
    //    break;
    // }
    // curr_grad[i] *= steps[i] / sqrt(avars[i] + floor);

    const int grad_sign = (ElemType(0) < g) - (g < ElemType(0));

    if (signs[i] * grad_sign > 0)
        steps[i] = min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
    else
        steps[i] = max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);

    ElemType temp = steps[i] / sqrt(avars[i] + floor);
    _scalevalue4BlockSparseCol(grad_bsc, colOrRow2blockId, len, i, temp);
    signs[i] = grad_sign;

    if (multipliers != nullptr)
        multipliers[i] = temp;
}

template <class ElemType>
__global__ void _rescaleToRange(
    ElemType* a,
    const CUDA_LONG N,
    const ElemType low,
    const ElemType high)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] = a[id] * (high - low) + low;
}

template <class ElemType>
__global__ void _setMaskAndScale(
    ElemType* a,
    const CUDA_LONG N,
    const ElemType maskRate,
    const ElemType scaleValue)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    a[id] = a[id] <= maskRate ? 0 : scaleValue;
}

template <class ElemType>
__global__ void _vectorSum(
    ElemType* c,       // output
    const ElemType* a, // input
    const CUDA_LONG n, // a.numRows
    const CUDA_LONG m, // a.numCols
    const bool isColWise)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= m) || (!isColWise && id >= n))
        return;

    ElemType sum = 0;

    if (isColWise)
    {
        for (CUDA_LONG i = 0; i < n; ++i)
        {
            sum += a[IDX2C(i, id, n)];
        }
    }
    else
    {
        for (CUDA_LONG j = 0; j < m; ++j)
        {
            sum += a[IDX2C(id, j, n)];
        }
    }
    c[id] = sum;
}

template <class ElemType>
__global__ void _vectorNorm1(
    ElemType* c,       // output
    const ElemType* a, // input
    const CUDA_LONG n, // a.numRows
    const CUDA_LONG m, // a.numCols
    const bool isColWise)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= m) || (!isColWise && id >= n))
        return;

    ElemType sum = 0;

    if (isColWise)
    {
        for (CUDA_LONG i = 0; i < n; ++i)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                sum += fabsf(a[IDX2C(i, id, n)]);
            }
            else
            {
                sum += fabs(a[IDX2C(i, id, n)]);
            }
        }
    }
    else
    {
        for (CUDA_LONG j = 0; j < m; ++j)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                sum += fabsf(a[IDX2C(id, j, n)]);
            }
            else
            {
                sum += fabs(a[IDX2C(id, j, n)]);
            }
        }
    }
    c[id] = sum;
}

//one column per thread
template <class ElemType>
__global__ void _vectorNorm2(
    ElemType* c,       // output
    const ElemType* a, // input
    const CUDA_LONG N, // a.GetNumRows();
    const CUDA_LONG M, // a.GetNumCols();
    const bool isColWise)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= M) || (!isColWise && id >= N))
        return;

    ElemType sum = 0;
    if (isColWise)
    {
        for (CUDA_LONG i = 0; i < N; ++i)
        {
            ElemType v = a[IDX2C(i, id, N)];
            sum += v * v;
        }
    }
    else
    {
        for (CUDA_LONG j = 0; j < M; ++j)
        {
            ElemType v = a[IDX2C(id, j, N)];
            sum += v * v;
        }
    }

    if (sizeof(ElemType) == sizeof(float))
        c[id] = sqrtf(sum);
    else
        c[id] = sqrt(sum);
}

template <class ElemType>
__global__ void _convertInd2ValsAdjustInd(
    ElemType* inds,
    const ElemType* M,
    ElemType* vals,
    const CUDA_LONG n, // number of cols
    const CUDA_LONG m, // number of rows
    const bool isColWise)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= n) || (!isColWise && id >= m))
        return;
    inds[id]--;
    if (isColWise)
    {
        vals[id] = M[IDX2C((int) inds[id], id, m)];
    }
    else
    {
        vals[id] = M[IDX2C(id, (int) inds[id], m)];
    }
}

//assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template <class ElemType>
__global__ void _assignPackedConvolutionInput(ElemType* packedMatrix, const ElemType* inputSubBatch, const CUDA_LONG batchSize,
                                              const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputChannels,
                                              const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputChannels,
                                              const CUDA_LONG kernelWidth, const CUDA_LONG kernelHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample, const bool zeroPadding)
{
    const CUDA_LONG inputHeightTimesChannel = inputHeight * inputChannels;
    const size_t inputDim = inputWidth * inputHeightTimesChannel;

    const CUDA_LONG idall = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = idall / inputDim;
    if (sample >= batchSize)
        return;

    const CUDA_LONG id = idall % inputDim;
    const CUDA_LONG y = id / inputHeightTimesChannel; // inputCol

    const size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
    const size_t packedInputColsPerSample = outputWidth * outputHeight; // output size per channel

    // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * inputChannels)
    // IN_ELEM_COLPOS = sample

    const CUDA_LONG nXC = id % inputHeightTimesChannel; // channel + inputRow*inputChannels
    const CUDA_LONG x = nXC / inputChannels;            // inputRow
    const CUDA_LONG c = nXC % inputChannels;            // channel

    ElemType currentInputValue = inputSubBatch[id + sample * inputDim];

    CUDA_LONG x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (zeroPadding)
    {
        const CUDA_LONG halfKernelWidth = kernelWidth / 2;
        const CUDA_LONG halfKernelHeight = kernelHeight / 2;

        x0 = max((ElemType)0.0f, ceil((x - (ElemType) kernelHeight + 1.0f + halfKernelHeight) / (ElemType) verticalSubsample)); // row : first wrow in which x is in
        x1 = x + halfKernelHeight - x0 * verticalSubsample;                                                           // first posxInKernel
        y0 = max((ElemType)0.0f, ceil((y - (ElemType) kernelWidth + 1.0f + halfKernelWidth) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
        y1 = y + halfKernelWidth - y0 * horizontalSubsample;                                                          // first posyInKernel
    }
    else
    {
        x0 = max((ElemType)0.0f, ceil((x - (ElemType) kernelHeight + 1) / (ElemType) verticalSubsample));  // row : first wrow in which x is in
        x1 = x - x0 * verticalSubsample;                                                         // first posxInKernel
        y0 = max((ElemType)0.0f, ceil((y - (ElemType) kernelWidth + 1) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
        y1 = y - y0 * horizontalSubsample;                                                       // first posyInKernel
    }

    // PACK_ELEM_ROWPOS(channel, posxInKernel, posyInKernel) = (channel * kernelWidth * kernelHeight + posxInKernel + posyInKernel * kernelHeight)
    // PACK_ELEM_COLPOS(sample, wrow, wcol) = (sample*packedInputColsPerSample + outputHeight*wcol + wrow

    CUDA_LONG packColBase = sample * packedInputColsPerSample + y0 * outputHeight;
    for (CUDA_LONG wcol = y0, posyInKernel = y1; wcol < outputWidth && posyInKernel >= 0; wcol++, posyInKernel -= horizontalSubsample)
    {
        CUDA_LONG packRowBase = c * kernelWidth * kernelHeight + posyInKernel * kernelHeight;
        for (CUDA_LONG wrow = x0, posxInKernel = x1; wrow < outputHeight && posxInKernel >= 0; wrow++, posxInKernel -= verticalSubsample)
        {
            const CUDA_LONG packRow = packRowBase + posxInKernel;
            const CUDA_LONG packCol = packColBase + wrow;
            packedMatrix[packRow + packCol * packedInputRows] = currentInputValue;
        }
        packColBase += outputHeight;
    }
}

//assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template <class ElemType>
__global__ void _unpackConvolutionInput(const ElemType* packedMatrix, ElemType* inputSubBatch, const CUDA_LONG batchSize,
                                        const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputChannels,
                                        const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputChannels,
                                        const CUDA_LONG kernelWidth, const CUDA_LONG kernelHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample, const bool zeroPadding)
{
    const CUDA_LONG inputHeightTimesChannel = inputHeight * inputChannels;
    const size_t inputDim = inputWidth * inputHeightTimesChannel;

    const CUDA_LONG idall = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = idall / inputDim;
    if (sample >= batchSize)
        return;

    const CUDA_LONG id = idall % inputDim;
    const CUDA_LONG y = id / inputHeightTimesChannel; // inputCol

    const size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
    const size_t packedInputColsPerSample = outputWidth * outputHeight; // output size per channel

    // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * inputChannels)
    // IN_ELEM_COLPOS = sample

    const CUDA_LONG nXC = id % inputHeightTimesChannel; // channel + inputRow*inputChannels
    const CUDA_LONG x = nXC / inputChannels;            // inputRow
    const CUDA_LONG c = nXC % inputChannels;            // channel

    CUDA_LONG x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (zeroPadding)
    {
        const CUDA_LONG halfKernelWidth = kernelWidth / 2;
        const CUDA_LONG halfKernelHeight = kernelHeight / 2;

        x0 = max(0.0f, ceil((x - (ElemType) kernelHeight + 1.0f + halfKernelHeight) / (ElemType) verticalSubsample)); // row : first wrow in which x is in
        x1 = x + halfKernelHeight - x0 * verticalSubsample;                                                           // first posxInKernel
        y0 = max(0.0f, ceil((y - (ElemType) kernelWidth + 1.0f + halfKernelWidth) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
        y1 = y + halfKernelWidth - y0 * horizontalSubsample;                                                          // first posyInKernel
    }
    else
    {
        x0 = max(0.0f, ceil((x - (ElemType) kernelHeight + 1) / (ElemType) verticalSubsample));  // row : first wrow in which x is in
        x1 = x - x0 * verticalSubsample;                                                         // first posxInKernel
        y0 = max(0.0f, ceil((y - (ElemType) kernelWidth + 1) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
        y1 = y - y0 * horizontalSubsample;                                                       // first posyInKernel
    }

    // PACK_ELEM_ROWPOS(channel, posxInKernel, posyInKernel) = (channel * kernelWidth * kernelHeight + posxInKernel + posyInKernel * kernelHeight)
    // PACK_ELEM_COLPOS(sample, wrow, wcol) = (sample*packedInputColsPerSample + outputHeight*wcol + wrow

    ElemType currentInputValue = inputSubBatch[id + sample * inputDim];
    CUDA_LONG packColBase = sample * packedInputColsPerSample + y0 * outputHeight;
    for (CUDA_LONG wcol = y0, posyInKernel = y1; wcol < outputWidth && posyInKernel >= 0; wcol++, posyInKernel -= horizontalSubsample)
    {
        CUDA_LONG packRowBase = c * kernelWidth * kernelHeight + posyInKernel * kernelHeight;
        for (CUDA_LONG wrow = x0, posxInKernel = x1; wrow < outputHeight && posxInKernel >= 0; wrow++, posxInKernel -= verticalSubsample)
        {
            const CUDA_LONG packRow = packRowBase + posxInKernel;
            const CUDA_LONG packCol = packColBase + wrow;
            currentInputValue += packedMatrix[packRow + packCol * packedInputRows];
        }
        packColBase += outputHeight;
    }

    inputSubBatch[id + sample * inputDim] = currentInputValue;
}

template <class ElemType>
__global__ void _assignMaxPoolingResult(ElemType* outputBatch, const ElemType* inputBatch, const CUDA_LONG batchSize, const CUDA_LONG channels,
                                        const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
                                        const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
                                        const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample)
{
    const CUDA_LONG outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = outputIndex / outputSizePerSample;
    if (sample >= batchSize)
        return;

    const CUDA_LONG outputIndexWithinSample = outputIndex % outputSizePerSample;
    const CUDA_LONG inputHeightTimesChannel = inputHeight * channels;
    const CUDA_LONG outputHeightTimesChannel = outputHeight * channels;

    // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
    // IN_ELEM_COLPOS = sample

    // OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
    // OUT_ELEM_COLPOS = sample

    const CUDA_LONG y = outputIndexWithinSample / outputHeightTimesChannel;   // wcol
    const CUDA_LONG nXC = outputIndexWithinSample % outputHeightTimesChannel; // channel + wrow*channels
    const CUDA_LONG x = nXC / channels;                                       // wrow
    const CUDA_LONG c = nXC % channels;                                       // channel

    const ElemType* inputBatchBase4Sample = inputBatch + sample * inputSizePerSample;
    register ElemType maxVal = -FLT_MAX;
    const CUDA_LONG rowInWindowBase = (x * verticalSubsample + y * horizontalSubsample * inputHeight) * channels + c;
    for (CUDA_LONG colInWindow = 0; colInWindow < windowWidth; colInWindow++)
    {
        CUDA_LONG rowInInput = rowInWindowBase + colInWindow * inputHeightTimesChannel;
        for (CUDA_LONG rowInWindow = 0; rowInWindow < windowHeight; rowInWindow++)
        {
            const ElemType val = inputBatchBase4Sample[rowInInput];
            maxVal = max(maxVal, val);
            rowInInput += channels;
        }
    }
    outputBatch[outputIndexWithinSample + sample * outputSizePerSample] = maxVal;
}

template <class ElemType>
__global__ void _addMaxPoolingGradient(ElemType* inputGradientBatch, const ElemType* outputGradientBatch, const ElemType* inputBatch, const ElemType* outputBatch,
                                       const CUDA_LONG batchSize, const CUDA_LONG channels,
                                       const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
                                       const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
                                       const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample)
{
    const CUDA_LONG inputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = inputIndex / inputSizePerSample;
    if (sample >= batchSize)
        return;

    const CUDA_LONG inputIndexWithinSample = inputIndex % inputSizePerSample;

    const CUDA_LONG inputHeightTimesChannel = inputHeight * channels;
    const CUDA_LONG outputHeightTimesChannel = outputHeight * channels;

    // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
    // IN_ELEM_COLPOS = sample

    // OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
    // OUT_ELEM_COLPOS = sample

    const CUDA_LONG y = inputIndexWithinSample / inputHeightTimesChannel;   // col in input
    const CUDA_LONG nXC = inputIndexWithinSample % inputHeightTimesChannel; // channel + row*chanels
    const CUDA_LONG x = nXC / channels;                                     // row in input
    const CUDA_LONG c = nXC % channels;                                     // channel

    CUDA_LONG startOutX = max(0.0f, ceil((x - (ElemType) windowHeight + 1) / (ElemType) verticalSubsample));     // inclusive start
    CUDA_LONG endOutX = (x / verticalSubsample < outputHeight - 1) ? x / verticalSubsample : outputHeight - 1;   // inclusive end
    CUDA_LONG startOutY = max(0.0f, ceil((y - (ElemType) windowWidth + 1) / (ElemType) horizontalSubsample));    // inclusive start
    CUDA_LONG endOutY = (y / horizontalSubsample < outputWidth - 1) ? y / horizontalSubsample : outputWidth - 1; // inclusive end

    ElemType* inputGradientBatchBase4Sample = inputGradientBatch + sample * inputSizePerSample;
    const ElemType* outputGradientBatchBase4Sample = outputGradientBatch + sample * outputSizePerSample;
    const ElemType* outputBatchBase4Sample = outputBatch + sample * outputSizePerSample;

    ElemType inputValue = inputBatch[inputIndexWithinSample + sample * inputSizePerSample];
    for (CUDA_LONG outY = startOutY; outY <= endOutY; outY++)
    {
        for (CUDA_LONG outX = startOutX; outX <= endOutX; outX++)
        {
            CUDA_LONG outputIndex = outY * outputHeightTimesChannel + outX * channels + c;
            if (inputValue == outputBatchBase4Sample[outputIndex])
                inputGradientBatchBase4Sample[inputIndexWithinSample] += outputGradientBatchBase4Sample[outputIndex];
        }
    }
}
template <class ElemType>
__global__ void _assignAveragePoolingResult(ElemType* outputBatch, const ElemType* inputBatch, const CUDA_LONG batchSize, const CUDA_LONG channels,
                                            const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
                                            const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
                                            const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample)
{
    const CUDA_LONG outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = outputIndex / outputSizePerSample;
    if (sample >= batchSize)
        return;

    const CUDA_LONG outputIndexWithinSample = outputIndex % outputSizePerSample;
    const CUDA_LONG inputHeightTimesChannel = inputHeight * channels;
    const CUDA_LONG outputHeightTimesChannel = outputHeight * channels;

    // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
    // IN_ELEM_COLPOS = sample

    // OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
    // OUT_ELEM_COLPOS = sample

    const CUDA_LONG y = outputIndexWithinSample / outputHeightTimesChannel;   // wcol
    const CUDA_LONG nXC = outputIndexWithinSample % outputHeightTimesChannel; // channel + wrow*channels
    const CUDA_LONG x = nXC / channels;                                       // wrow
    const CUDA_LONG c = nXC % channels;                                       // channel

    const ElemType* inputBatchBase4Sample = inputBatch + sample * inputSizePerSample;

    register ElemType average = 0;
    const CUDA_LONG rowInWindowBase = (x * verticalSubsample + y * horizontalSubsample * inputHeight) * channels + c;
    for (CUDA_LONG colInWindow = 0; colInWindow < windowWidth; colInWindow++)
    {
        CUDA_LONG rowInInput = rowInWindowBase + colInWindow * inputHeightTimesChannel;
        for (CUDA_LONG rowInWindow = 0; rowInWindow < windowHeight; rowInWindow++)
        {
            average += inputBatchBase4Sample[rowInInput];
            rowInInput += channels;
        }
    }

    outputBatch[outputIndexWithinSample + sample * outputSizePerSample] = average / windowWidth / windowHeight;
}

template <class ElemType>
__global__ void _addAveragePoolingGradient(ElemType* inputGradientBatch, const ElemType* outputGradientBatch,
                                           const CUDA_LONG batchSize, const CUDA_LONG channels,
                                           const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
                                           const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
                                           const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample)
{
    const CUDA_LONG inputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = inputIndex / inputSizePerSample;
    if (sample >= batchSize)
        return;

    const CUDA_LONG inputIndexWithinSample = inputIndex % inputSizePerSample;

    const CUDA_LONG inputHeightTimesChannel = inputHeight * channels;
    const CUDA_LONG outputHeightTimesChannel = outputHeight * channels;
    const CUDA_LONG windowSize = windowWidth * windowHeight;

    // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
    // IN_ELEM_COLPOS = sample

    // OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
    // OUT_ELEM_COLPOS = sample

    const CUDA_LONG y = inputIndexWithinSample / inputHeightTimesChannel;   // col in input
    const CUDA_LONG nXC = inputIndexWithinSample % inputHeightTimesChannel; // channel + row*chanels
    const CUDA_LONG x = nXC / channels;                                     // row in input
    const CUDA_LONG c = nXC % channels;                                     // channel

    CUDA_LONG startOutX = max(0.0f, ceil((x - (ElemType) windowHeight + 1) / (ElemType) verticalSubsample));     // inclusive start
    CUDA_LONG endOutX = (x / verticalSubsample < outputHeight - 1) ? x / verticalSubsample : outputHeight - 1;   // inclusive end
    CUDA_LONG startOutY = max(0.0f, ceil((y - (ElemType) windowWidth + 1) / (ElemType) horizontalSubsample));    // inclusive start
    CUDA_LONG endOutY = (y / horizontalSubsample < outputWidth - 1) ? y / horizontalSubsample : outputWidth - 1; // inclusive end

    ElemType* inputGradientBatchBase4Sample = inputGradientBatch + sample * inputSizePerSample;
    const ElemType* outputGradientBatchBase4Sample = outputGradientBatch + sample * outputSizePerSample;

    for (CUDA_LONG outY = startOutY; outY <= endOutY; outY++)
    {
        for (CUDA_LONG outX = startOutX; outX <= endOutX; outX++)
        {
            CUDA_LONG outputIndex = outY * outputHeightTimesChannel + outX * channels + c;
            inputGradientBatchBase4Sample[inputIndexWithinSample] += outputGradientBatchBase4Sample[outputIndex] / windowSize;
        }
    }
}

template <class ElemType>
__global__ void _addMaxPoolingGradientLoopOut(ElemType* inputGradientBatch, const ElemType* outputGradientBatch, const ElemType* inputBatch, const ElemType* outputBatch,
                                              const CUDA_LONG batchSize, const CUDA_LONG channels,
                                              const CUDA_LONG inputWidth, const CUDA_LONG inputHeight, const CUDA_LONG inputSizePerSample,
                                              const CUDA_LONG outputWidth, const CUDA_LONG outputHeight, const CUDA_LONG outputSizePerSample,
                                              const CUDA_LONG windowWidth, const CUDA_LONG windowHeight, const CUDA_LONG horizontalSubsample, const CUDA_LONG verticalSubsample)
{
    const CUDA_LONG outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG sample = outputIndex / outputSizePerSample;
    if (sample >= batchSize)
        return;

    const CUDA_LONG outputIndexWithinSample = outputIndex % outputSizePerSample;
    const CUDA_LONG inputWidthTimesChannel = inputWidth * channels;
    const CUDA_LONG outputWidthTimesChannel = outputWidth * channels;
    const CUDA_LONG y = outputIndexWithinSample / outputWidthTimesChannel;
    const CUDA_LONG nXC = outputIndexWithinSample % outputWidthTimesChannel;
    const CUDA_LONG x = nXC / channels;
    const CUDA_LONG c = nXC % channels;

    const CUDA_LONG offset0 = sample * inputSizePerSample + y * verticalSubsample * inputWidthTimesChannel + x * horizontalSubsample * channels;
    const ElemType* pCurWindow4Input = inputBatch + offset0; // pooling to current window's first input pixel
    ElemType* pCurWindow4InGradient = inputGradientBatch + offset0;
    for (CUDA_LONG yy = 0; yy < windowHeight; yy++)
    {
        const CUDA_LONG offset1 = yy * inputWidthTimesChannel + c;
        const ElemType* pf0 = pCurWindow4Input + offset1;
        ElemType* pf1 = pCurWindow4InGradient + offset1;
        for (CUDA_LONG xx = 0; xx < windowWidth; xx++)
        {
            const CUDA_LONG offset2 = xx * channels;
            if (pf0[offset2] == outputBatch[outputIndex])
            {
                pf1[offset2] += outputGradientBatch[outputIndex]; // need to be atomic however atomicAdd on double is not supported.
            }
        }
    }
}

template <class ElemType>
__global__ void _addElementProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    us[id] += (a[id] * b[id]);
}

template <class ElemType>
__global__ void _columnElementMultiplyWith(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, // a.GetNumRows();
    const CUDA_LONG M) // us.GetNumCols();
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    // __shared__ ElemType _a[GridDim::maxThreadsPerBlock];
    // _a[threadIdx.x]=a[id];
    ElemType mul = a[id];
    for (CUDA_LONG j = 0; j < M; ++j)
    {
        us[IDX2C(id, j, N)] = us[IDX2C(id, j, N)] * mul;
    }
}

template <class ElemType>
__global__ void _rowElementMultiplyWith(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, // us.GetNumRows();
    const CUDA_LONG M) // a.GetNumCols();
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= M)
        return;

    // __shared__ ElemType _a[GridDim::maxThreadsPerBlock];
    // _a[threadIdx.x]=a[id];
    ElemType mul = a[id];
    for (CUDA_LONG i = 0; i < N; ++i)
    {
        us[IDX2C(i, id, N)] = us[IDX2C(i, id, N)] * mul;
    }
}

template <class ElemType>
__global__ void _rowElementDivideBy(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, // us.GetNumRows();
    const CUDA_LONG M) // a.GetNumCols();
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= M)
        return;

    // __shared__ ElemType _a[GridDim::maxThreadsPerBlock];
    // _a[threadIdx.x]=a[id];
    ElemType v = a[id];
    if (v >= 0 && v < EPS_IN_INVERSE)
        v = EPS_IN_INVERSE;
    else if (v < 0 && v > -EPS_IN_INVERSE)
        v = (-EPS_IN_INVERSE);

    for (CUDA_LONG i = 0; i < N; ++i)
    {
        us[IDX2C(i, id, N)] = us[IDX2C(i, id, N)] / v;
    }
}

template <class ElemType>
__global__ void _ColumnElementDivideBy(
    ElemType* us,
    const ElemType* a,
    const CUDA_LONG N, // a.GetNumRows();
    const CUDA_LONG M) // us.GetNumCols();
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    ElemType smallValue = EPS_IN_INVERSE;

    // __shared__ ElemType _a[GridDim::maxThreadsPerBlock];
    // _a[threadIdx.x]=a[id];
    ElemType v = a[id];
    for (CUDA_LONG j = 0; j < M; ++j)
    {
        if (v < 0 && v > -smallValue)
            us[IDX2C(id, j, N)] /= (-smallValue);
        else if (v >= 0 && v < smallValue)
            us[IDX2C(id, j, N)] /= smallValue;
        else
            us[IDX2C(id, j, N)] /= v;
    }
}

template <class ElemType>
__global__ void _innerProduct(
    ElemType* c,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N, // a.GetNumRows();
    const CUDA_LONG M, // a.GetNumCols();
    const bool isColWise)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= M) || (!isColWise && id >= N))
        return;

    ElemType sum = 0;
    CUDA_LONG index;
    if (isColWise)
    {
        for (CUDA_LONG i = 0; i < N; ++i)
        {
            index = IDX2C(i, id, N);
            sum += a[index] * b[index];
        }
    }
    else
    {
        for (CUDA_LONG j = 0; j < M; ++j)
        {
            index = IDX2C(id, j, N);
            sum += a[index] * b[index];
        }
    }

    c[id] = sum;
}

template <class ElemType>
__global__ void _innerProduct4SparseCSC(
    ElemType* c,
    const ElemType* a,
    const GPUSPARSE_INDEX_TYPE* aRowIndex,
    const GPUSPARSE_INDEX_TYPE* aColCSCIndex,
    const ElemType* b,
    const CUDA_LONG M, // a.GetNumRows();
    const CUDA_LONG N, // a.GetNumCols();
    const bool isColWise)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= N) || (!isColWise && id >= M))
        return;

    ElemType sum = 0;
    CUDA_LONG index;

    if (isColWise)
    {
        for (CUDA_LONG i = aColCSCIndex[id]; i < aColCSCIndex[id+1]; i++)
        {
            index = IDX2C(aRowIndex[i], id, M);
            sum += a[i] * b[index];
        }
    }
    else
    {
        for (CUDA_LONG j = 0; j < N; ++j)
        {
            for (CUDA_LONG i = aColCSCIndex[j]; i < aColCSCIndex[j+1]; i++)
            {
                if (aRowIndex[i] == id)
                {
                    index = IDX2C(id, j, M);
                    sum += a[i] * b[index];
                    break;
                }
            }
        }
    }

    c[id] = sum;
}

template <class ElemType>
__global__ void _assignSignOf(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    ElemType v = b[id];
    a[id] = (v == (ElemType) 0 ? (ElemType) 0 : (v > 0 ? (ElemType) 1 : (ElemType)(-1)));
}

template <class ElemType>
__global__ void _addSignOf(
    ElemType* a,
    const ElemType* b,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    ElemType v = b[id];
    a[id] += (v == (ElemType) 0 ? (ElemType) 0 : (v > 0 ? (ElemType) 1 : (ElemType)(-1)));
}

// This function processes 1 column per block. this function needs 512 threads
template <class ElemType, bool IsMax>
__global__ void _vectorMaxMinReduce512Threads(
    const ElemType* us,
    ElemType* Indexes,
    ElemType* Values,
    const CUDA_LONG numRows,
    const CUDA_LONG numCols)
{
    // we first find max per column
    __shared__ ElemType partials[512];
    __shared__ int partialsInd[512];
    if (IsMax)
    {
        partials[threadIdx.x] = -10000000;
    }
    else
    {
        partials[threadIdx.x] = 10000000;
    }
    partialsInd[threadIdx.x] = -1;

    for (int i = threadIdx.x; i < numRows; i += 512)
    {
        if ((IsMax ? (us[IDX2C(i, blockIdx.x, numRows)] > partials[threadIdx.x]) : (us[IDX2C(i, blockIdx.x, numRows)] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = us[IDX2C(i, blockIdx.x, numRows)];
            partialsInd[threadIdx.x] = i;
        }
    }
    __syncthreads();

    if (threadIdx.x < 256)
    {
        if ((IsMax ? (partials[threadIdx.x + 256] > partials[threadIdx.x]) : (partials[threadIdx.x + 256] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 256];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 256];
        }
    }
    __syncthreads();

    if (threadIdx.x < 128)
    {
        if ((IsMax ? (partials[threadIdx.x + 128] > partials[threadIdx.x]) : (partials[threadIdx.x + 128] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 128];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 128];
        }
    }
    __syncthreads();

    if (threadIdx.x < 64)
    {
        if ((IsMax ? (partials[threadIdx.x + 64] > partials[threadIdx.x]) : (partials[threadIdx.x + 64] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 64];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 64];
        }
    }
    __syncthreads();

    if (threadIdx.x < 32)
    {
        if ((IsMax ? (partials[threadIdx.x + 32] > partials[threadIdx.x]) : (partials[threadIdx.x + 32] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 32];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 32];
        }
    }
    __syncthreads();

    if (threadIdx.x < 16)
    {
        if ((IsMax ? (partials[threadIdx.x + 16] > partials[threadIdx.x]) : (partials[threadIdx.x + 16] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 16];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 16];
        }
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        if ((IsMax ? (partials[threadIdx.x + 8] > partials[threadIdx.x]) : (partials[threadIdx.x + 8] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 8];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 8];
        }
    }
    __syncthreads();

    if (threadIdx.x < 4)
    {
        if ((IsMax ? (partials[threadIdx.x + 4] > partials[threadIdx.x]) : (partials[threadIdx.x + 4] < partials[threadIdx.x])) || (partialsInd[threadIdx.x] == -1))
        {
            partials[threadIdx.x] = partials[threadIdx.x + 4];
            partialsInd[threadIdx.x] = partialsInd[threadIdx.x + 4];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        ElemType mx = partials[0];
        int ind = partialsInd[0];
        if ((IsMax ? (mx < partials[1]) : (mx > partials[1])) || (ind == -1))
        {
            mx = partials[1];
            ind = partialsInd[1];
        }
        if ((IsMax ? (mx < partials[2]) : (mx > partials[2])) || (ind == -1))
        {
            mx = partials[2];
            ind = partialsInd[2];
        }
        if ((IsMax ? (mx < partials[3]) : (mx > partials[3])) || (ind == -1))
        {
            mx = partials[3];
            ind = partialsInd[3];
        }
        Values[blockIdx.x] = mx;
        Indexes[blockIdx.x] = ind;
    }
}

template <class ElemType>
__global__ void _vectorMax(
    const ElemType* us,
    ElemType* maxIndexes,
    ElemType* maxValues,
    const CUDA_LONG m, // number of rows
    const CUDA_LONG n, // number of cols
    const bool isColWise)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG maxInd = -1;
    ElemType maxVal = -100000;

    if (isColWise)
    {
        if (id >= n)
            return;

        for (CUDA_LONG i = 0; i < m; i++)
        {
            if (maxInd == -1 || us[IDX2C(i, id, m)] >= maxVal)
            {
                maxInd = i;
                maxVal = us[IDX2C(i, id, m)];
            }
        }
    }
    else
    {
        if (id >= m)
            return;

        for (CUDA_LONG j = 0; j < n; j++)
        {
            if (maxInd == -1 || us[IDX2C(id, j, m)] >= maxVal)
            {
                maxInd = j;
                maxVal = us[IDX2C(id, j, m)];
            }
        }
    }
    maxIndexes[id] = maxInd;
    maxValues[id] = maxVal;
}

template <class ElemType>
__global__ void _vectorMin(
    const ElemType* us,
    ElemType* minIndexes,
    ElemType* minValues,
    const CUDA_LONG m, // number of rows
    const CUDA_LONG n, // number of cols
    const bool isColWise)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG minInd = -1;
    ElemType minVal = -100000;

    if (isColWise)
    {
        if (id >= n)
            return;

        for (CUDA_LONG i = 0; i < m; i++)
        {
            if (minInd == -1 || us[IDX2C(i, id, m)] <= minVal)
            {
                minInd = i;
                minVal = us[IDX2C(i, id, m)];
            }
        }
    }
    else
    {
        if (id >= m)
            return;

        for (CUDA_LONG j = 0; j < n; j++)
        {
            if (minInd == -1 || us[IDX2C(id, j, m)] <= minVal)
            {
                minInd = j;
                minVal = us[IDX2C(id, j, m)];
            }
        }
    }
    minIndexes[id] = minInd;
    minValues[id] = minVal;
}

template <class ElemType>
__global__ void _matrixMatrixAddOnCuda(
    const ElemType alpha,
    const ElemType* a,
    const ElemType* b,
    ElemType* c,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    c[id] = alpha * a[id] + b[id];
}

template <class ElemType>
__global__ void _matrixVectorRowWiseAddWithThreadPerElem(
    const ElemType* a,
    const ElemType* b,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m, // number of rows
    const CUDA_LONG n) // number of cols
{
    CUDA_LONG N = m * n; // used in CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id,N) macro
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

    CUDA_LONG col = id / m;

    us[id] = alpha * a[col] + b[id];
}

//this implementation uses more threads but also more memory access
template <class ElemType>
__global__ void _matrixVectorColumnWiseAddWithThreadPerElem(
    const ElemType* a,
    const ElemType* b,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m, // number of rows
    const CUDA_LONG n) // number of cols
{
    CUDA_LONG N = m * n; // used in CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id,N) macro
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

    CUDA_LONG col = id / m;
    CUDA_LONG row = id - col * m;

    us[id] = alpha * a[row] + b[id];
}

template <class ElemType>
__global__ void _matrixVectorColumnWiseAddWithThreadPerRow(
    const ElemType* a,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m, // number of rows
    const CUDA_LONG n) // number of cols
{
#ifdef VALIDATION
    if (blockDim.x * blockIdx.x + threadIdx.x == 0)
    {
        printf("** _matrixVectorColumnWiseAdd on device:\na = %p, us = %p, alpha = %f, m = %ld, n = %ld\n",
               a, us, alpha, m, n);
        printf("us[0] = %f\n", us[0]);
        printf("a[0] = %f\n", a[0]);
    }
#endif
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= m)
        return;
    ElemType tmp = a[id];
#ifdef VALIDATION
    printf("  a[%d] = %f\n", id, tmp);
#endif
    for (CUDA_LONG j = 0; j < n; ++j)
    {
        us[j * m + id] += alpha * tmp;
    }
}

template <class ElemType>
__global__ void _matrixVectorColumnWiseAddBlockPerRow(
    const ElemType* a,
    ElemType* us,
    ElemType alpha,
    const CUDA_LONG m, // number of rows
    const CUDA_LONG n) // number of cols
{
    ElemType tmp;

    if (threadIdx.x == 0)
    {
        tmp = a[blockIdx.x];
    }
    __syncthreads();

    int loadPerThread = n / blockDim.x;

    for (int i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? n : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        us[m * blockIdx.x + i] += alpha * tmp;
    }
}

template <class ElemType>
__global__ void _addScaledDifference(
    ElemType alpha,
    ElemType* a,
    ElemType* b,
    ElemType* c,
    CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    c[id] = c[id] + (a[id] - b[id]) * (alpha);
}

template <class ElemType>
__global__ void _assignScaledDifference(
    ElemType alpha,
    ElemType* a,
    ElemType* b,
    ElemType* c,
    CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    c[id] = (a[id] - b[id]) * (alpha);
}

template <class ElemType>
__global__ void _addScaledDifference(
    ElemType* alpha,
    ElemType* a,
    ElemType* b,
    ElemType* c,
    CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    c[id] = c[id] + (a[id] - b[id]) * alpha[0];
}

template <class ElemType>
__global__ void _assignScaledDifference(
    ElemType* alpha,
    ElemType* a,
    ElemType* b,
    ElemType* c,
    CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    c[id] = (a[id] - b[id]) * alpha[0];
}

template <class ElemType>
__global__ void _addElementToElement(
    ElemType beta,
    const ElemType* a, CUDA_LONG indexA,
    ElemType* c, CUDA_LONG indexC)
{
    //CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;  // only one thread launched
    //if (id > 0)
    //    return;
    ElemType us = beta ? beta * c[indexC] : 0; // do not multiply if beta is 0, could be a NaN
    us += a[indexA];
    c[indexC] = us;
}

template <class ElemType>
__global__ void _assignNumOfDiff1024Threads(
    const ElemType* a,
    const ElemType* b,
    ElemType* c,
    CUDA_LONG N)
{
    __shared__ ElemType partialSums[1024];
    partialSums[threadIdx.x] = 0;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG loadPerThread = N / blockDim.x;
    for (CUDA_LONG i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        partialSums[threadIdx.x] += (a[i] != b[i]);
    }
    __syncthreads();

    // 512
    if (threadIdx.x < 512)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 512];
    }
    __syncthreads();

    // 256
    if (threadIdx.x < 256)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 256];
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 128];
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 64];
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 32];
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 16];
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 8];
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 4];
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        c[0] = partialSums[0] + partialSums[1] + partialSums[2] + partialSums[3];
    }
}

/*template<class ElemType>
__global__ void _assignNumOfDiff1024Threads(
ElemType *a,
ElemType *b,
ElemType *c,
CUDA_LONG N)
{
//TO DO: replace atomic operation with reduction

__shared__ int totalSum;
if (threadIdx.x == 0) totalSum = 0;
__syncthreads();

int id = blockDim.x * blockIdx.x + threadIdx.x;
if (id>=N)
return;

int localVal = (a[id] != b[id]);
atomicAdd(&totalSum, localVal);
__syncthreads();

c[id] = totalSum;
}*/

template <class ElemType>
__global__ void _scaleArray(
    ElemType alpha,
    ElemType* us,
    CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    us[id] = us[id] * alpha;
}

template <class ElemType>
__global__ void _sparseCSRPlusDense(
    ElemType alpha,
    const ElemType* m_dVal,
    const int* m_dRow,
    const int* m_dCol,
    ElemType* pArrayDev,
    CUDA_LONG M)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= M)
        return;
    int start = m_dRow[id];
    int end = m_dRow[id + 1];
    for (int _i = start; _i < end; ++_i) // _i is index in m_dVal and m_dCol
    {
        int j = m_dCol[_i];
        pArrayDev[IDX2C(id, j, M)] += (alpha * m_dVal[_i]);
    }
}

template <class ElemType>
__global__ void _sparseCSRElemMulDense(
    const ElemType* m_dVal,
    const int* m_dRow,
    const int* m_dCol,
    const ElemType* b,
    ElemType* c,
    CUDA_LONG M)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= M)
        return;
    int start = m_dRow[id];
    int end = m_dRow[id + 1];
    for (int _i = start; _i < end; ++_i) // _i is index in m_dVal and m_dCol
    {
        int j = m_dCol[_i];
        c[IDX2C(id, j, M)] = b[IDX2C(id, j, M)] * m_dVal[_i];
    }
}

template <class ElemType>
__global__ void _isValid(
    const GPUSPARSE_INDEX_TYPE* rowIndex,
    const GPUSPARSE_INDEX_TYPE* colCSCIndex,
    const int rows,
    const int cols,
    const int nz,
    long* d_res)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= cols || d_res[0] <= 0)
        return;

    int start = colCSCIndex[id];
    int end = colCSCIndex[id + 1];

    if (start > end)
    {
        if (d_res[0] > 0)
        {
            d_res[0] = -1;
            d_res[1] = id;
            d_res[2] = start;
            d_res[3] = end;
        }
    }
    else if (end > nz)
    {
        if (d_res[0] > 0)
        {
            d_res[0] = -2;
            d_res[1] = id + 1;
            d_res[2] = end;
            d_res[3] = nz;
        }
    }
    else
    {
        for (int j = start; j < end; j++) // j points to the value
        {
            if (rowIndex[j] >= rows)
            {
                if (d_res[0] > 0)
                {
                    d_res[0] = -3;
                    d_res[1] = j;
                    d_res[2] = rowIndex[j];
                    d_res[3] = rows;
                    break;
                }
            }
            if (j > start && rowIndex[j] < rowIndex[j - 1])
            {
                if (d_res[0] > 0)
                {
                    d_res[0] = -4;
                    d_res[1] = id;
                    d_res[2] = j;
                    d_res[3] = rowIndex[j];
                    break;
                }
            }
        }
    }
}

template <class ElemType>
__global__ void _shiftColCSCIndexFromSliceViewToAbsolute(
    GPUSPARSE_INDEX_TYPE* colCSCIndex,
    const int cols,
    const int nz)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= cols)
        return;

    colCSCIndex[id] = colCSCIndex[id] - colCSCIndex[0];

    if (id == cols - 1)
        colCSCIndex[cols] = nz;
}

//c = alpha * op(a) * op(b) + beta*c
// TODO: This function can be further improved by loading the kernel in shared memory
template <class ElemType>
__global__ void _dense1DConvMultSparseCSCAndWeightedAddToDense(
    const int m,                   // rowDense
    const int k,                   // colDense
    const int n,                   // colSparse
    const int numChannels,         // input num channels
    const int numSteps,            // convolution num steps
    const int horizontalSubsample, // convolution step size
    const bool channelwise,        // pixelwise for normal multiplication and channelwise for convolution operation
    const ElemType alpha,
    const ElemType* a, // dense
    const bool transposeA,
    const ElemType* bnzValues, // sparse nz values
    const GPUSPARSE_INDEX_TYPE* rowIndex,
    const GPUSPARSE_INDEX_TYPE* colCSCIndex,
    const ElemType beta,
    ElemType* c // dense target
    )
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= m * numSteps * n)
        return;

    int colInC = id / (m * numSteps);
    int rowInC = id % (m * numSteps);
    int stepIdx = rowInC / m;

    int start = colCSCIndex[colInC];
    int end = colCSCIndex[colInC + 1];

    ElemType s = 0;
    for (int j = start; j < end; j++) // j points to the value
    {
        int i = rowIndex[j] - (horizontalSubsample * numChannels * stepIdx); // offset row index by the convolution step

        if (i >= 0)
        {
            if (i >= k)
                break;

            // Convert to channelwise index.
            // This is similar to rowwise to columnwise conversion
            if (channelwise)
            {
                int pixel = i / numChannels;
                int channel = i % numChannels;
                int numPixels = k / numChannels;
                i = channel * numPixels + pixel;
            }

            if (!transposeA)
                s += a[IDX2C(rowInC % m, i, m)] * bnzValues[j];
            else
                s += a[IDX2C(i, rowInC % m, k)] * bnzValues[j];
        }
    }

    c[IDX2C(rowInC, colInC, m * numSteps)] = alpha * s + (beta == 0 ? 0 : beta * c[IDX2C(rowInC, colInC, m * numSteps)]); // If beta is zero then don't lookup c
}

/// c += alpha * a * b^T
template <class ElemType>
__global__ void _dense1DConvMultSparseCSCTransposeAndAddToDense(
    int m,                   // rowDense
    int k,                   // colDense
    int n,                   // colSparse
    int numChannels,         // input num channels
    int numSteps,            // convolution num steps
    int horizontalSubsample, // convolution step size
    bool channelwise,        // pixelwise for normal multiplication and channelwise for convolution operation
    int rowInB,              // row index of the sparse matrix
    ElemType alpha,
    const ElemType* a, // dense
    bool transposeA,
    const ElemType* bnzValues, // sparse nz values
    const GPUSPARSE_INDEX_TYPE* rowIndex,
    const GPUSPARSE_INDEX_TYPE* colCSCIndex,
    ElemType* c // dense target
    )
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= m * numSteps)
        return;

    int rowInC = id;
    int stepIdx = rowInC / m;
    int i = rowInB - (horizontalSubsample * numChannels * stepIdx); // offset row index by the convolution step

    if (i < 0 || i >= k)
        return;

    // Convert to channelwise index.
    // This is similar to rowwise to columnwise conversion
    if (channelwise)
    {
        int pixel = i / numChannels;
        int channel = i % numChannels;
        int numPixels = k / numChannels;
        i = channel * numPixels + pixel;
    }

    int start = colCSCIndex[rowInB];
    int end = colCSCIndex[rowInB + 1];

    ElemType s = 0;
    for (int j = start; j < end; j++) // j points to the value that are in the same row
    {
        int colInC = rowIndex[j]; // the column index because of transpose

        // bnzValues[j] = the b[][j] value
        if (!transposeA)
            s = a[IDX2C(rowInC % m, i, m)] * bnzValues[j];
        else
            s = a[IDX2C(i, rowInC % m, k)] * bnzValues[j];

        atomicAdd(&c[IDX2C(rowInC, colInC, m * numSteps)], alpha * s);
    }
}

template <class ElemType>
__global__ void _columnwiseScaleAndWeightedAdd(
    ElemType alpha,
    const ElemType* aData,
    const ElemType* vData,
    ElemType beta,
    ElemType* cData,
    int m, int n)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= m * n)
        return;

    CUDA_LONG col = id / m;

    if (beta == 0) // don't even read the memory if beta is 0
        cData[id] = alpha * vData[col] * aData[id];
    else
        cData[id] = alpha * vData[col] * aData[id] + beta * cData[id];
}

template <class ElemType>
__global__ void _columnwiseScaleAndWeightedAdd4CSC(
    ElemType alpha,
    const ElemType* aData, const GPUSPARSE_INDEX_TYPE* aSecondaryIndices, const GPUSPARSE_INDEX_TYPE* aMajorIndices,
    const ElemType* vData,
    ElemType beta,
    ElemType* cData,
    int m, int n)
{
    CUDA_LONG col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= n)
        return;

    GPUSPARSE_INDEX_TYPE start = aSecondaryIndices[col];
    GPUSPARSE_INDEX_TYPE end = aSecondaryIndices[col + 1];

    for (GPUSPARSE_INDEX_TYPE p = start; p < end; p++)
    {
        GPUSPARSE_INDEX_TYPE row = aMajorIndices[p];
        ElemType val = aData[p];

        if (beta == 0) // don't even read the memory if beta is 0
            cData[IDX2C(row, col, m)] = alpha * vData[col] * val;
        else
            cData[IDX2C(row, col, m)] = alpha * vData[col] * val + beta * cData[IDX2C(row, col, m)];
    }
}

template <class ElemType>
__global__ void _reshape(
    const int oldNumRows,                       // old row count
    const int oldNumCols,                       // old col count
    const int newNumRows,                       // new row count
    const int newNumCols,                       // new col count
    const GPUSPARSE_INDEX_TYPE* oldRowIndex,    // old row index array
    const GPUSPARSE_INDEX_TYPE* oldColumnIndex, // old column index array
    GPUSPARSE_INDEX_TYPE* newRowIndex,          // new row index array
    GPUSPARSE_INDEX_TYPE* newColumnIndex        // new column index array
    )
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= newNumCols)
        return;

    int currentCol = id;
    int oldColLower = (newNumRows * currentCol) / oldNumRows;

    // initialize to the end and then scan in the right direction in the for-loop
    int currentColStart = oldColumnIndex[oldNumCols];

    for (int oldCol = oldColLower; oldCol < oldNumCols; oldCol++)
    {
        int start = oldColumnIndex[oldCol];
        int end = oldColumnIndex[oldCol + 1];
        bool done = false;

        for (int j = start; j < end; j++) // j points to the value
        {
            int oldRow = oldRowIndex[j];
            int index = (oldCol * oldNumRows + oldRow);
            int newCol = index / newNumRows;
            int newRow = index % newNumRows;

            if (newCol == currentCol)
                newRowIndex[j] = newRow;

            if (newCol >= currentCol && currentColStart > j)
                currentColStart = j;

            if (newCol > currentCol)
            {
                done = true;
                break;
            }
        }

        if (done)
            break;
    }

    newColumnIndex[currentCol] = currentColStart;

    if (currentCol == (newNumCols - 1))
        newColumnIndex[newNumCols] = oldColumnIndex[oldNumCols]; // set end pointer
}

//called before _determineBlockIds and _denseMulSparseCSCTransposeToSparseBlockCol to determine which columns have values and
//what's the mapping from the column id in the resulted SparseBlockCol format to the column id in the dense format
//input: rowIndexes: the row indexes of the CSC sparse matrix to be multiplied with
//blockIDs: the blockID mapping in the resulting matrix;
//nnz: number of nonzero value or the size of rowIndexes;
template <class ElemType>
__global__ void _findColsWithValues(
    const GPUSPARSE_INDEX_TYPE* rowIndexes, GPUSPARSE_INDEX_TYPE* col2BlockIds, const size_t nnz)
{
    const size_t nzIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (nzIndex >= nnz)
        return;

    if (col2BlockIds[rowIndexes[nzIndex]] == Id_NotAssigned)
        col2BlockIds[rowIndexes[nzIndex]] = Id_Pending; // this row has value.
}

//called before _denseMulSparseCSCTransposeToSparseBlockCol and after _findColsWithValuesto determine which columns have values and
//what's the mapping from the column id in the resulted SparseBlockCol format to the column id in the dense format
//input: rowIndexes: the row indexes of the CSC sparse matrix to be multiplied with
//blockId2Col: the blockID to colum id mapping in the resulting matrix;
//col2BlockId: the col2BlockId to blockID mapping in the resulting matrix;
//numCols: number of columns in the resulting matrix or the size of blockIDs
//blockSize: return the blockSize with values
template <class ElemType>
__global__ void _determineBlockIds(
    GPUSPARSE_INDEX_TYPE* blockId2Col, GPUSPARSE_INDEX_TYPE* col2BlockId, size_t numCols, size_t* blockSize)
{
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= numCols)
        return;

    if (col2BlockId[col] == Id_Pending)
    {
        GPUSPARSE_INDEX_TYPE blockIndex = atomicAdd((unsigned int*)blockSize, (unsigned int)1);
        col2BlockId[col] = blockIndex;
        blockId2Col[blockIndex] = col;
    }
}

// backward pass from hidden layer to feature weight
//result (sparse BlockCol)= alpha * (lhs (dense) X rhs^T (sparse CSC)
//assume resultValues are 0-initialized
template <class ElemType>
__global__ void _denseMulSparseCSCTransposeToSparseBlockCol2(
    const ElemType alpha,
    const ElemType* lhsValues,
    const size_t numRowsLhs,
    const size_t numColsRhs,
    const ElemType* rhsNZValues,
    const GPUSPARSE_INDEX_TYPE* rhsRows,
    const GPUSPARSE_INDEX_TYPE* rhsCols,
    const GPUSPARSE_INDEX_TYPE* col2blockIds,
    ElemType* resultValues)
{
    const CUDA_LONG index = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG lhsCol = index / numRowsLhs; // rhsCol == lhsCol
    if (lhsCol >= numColsRhs)
        return;
    const CUDA_LONG lhsRow = index - numRowsLhs * lhsCol; // resultRow == lhsRow

    // each thread handles one [row, col] combination
    ElemType lhsValue = alpha * lhsValues[IDX2C(lhsRow, lhsCol, numRowsLhs)];

    CUDA_LONG start = rhsCols[lhsCol]; // rhsCol == lhsCol
    CUDA_LONG end = rhsCols[lhsCol + 1];

    for (CUDA_LONG p = start; p < end; p++)
    {
        CUDA_LONG rhsRow = rhsRows[p];
        ElemType rhsVal = rhsNZValues[p];
        CUDA_LONG resultCol = col2blockIds[rhsRow]; // resultCol == rhsRow maps to columnid

        // assume resultValues are 0-initialized
        atomicAdd(&resultValues[IDX2C(lhsRow, resultCol, numRowsLhs)], lhsValue * rhsVal);
    }
}

// backward pass from hidden layer to feature weight
//result (sparse BlockCol)= alpha * (lhs (dense) X rhs^T (sparse CSC)
//assume resultValues are 0-initialized
template <class ElemType>
__global__ void _denseMulSparseCSCTransposeToSparseBlockCol(
    const ElemType alpha,
    const ElemType* lhsValues,
    const size_t numRowsLhs,
    const size_t numColsRhs,                // The number of columns of rhs matrix before transpose. I.e. it is the 'conttacting' dimension in the matrix product to be computed.
    const ElemType* rhsNZValues,
    const GPUSPARSE_INDEX_TYPE* rhsRows,    // Mapping the ids of the non-zero values to their row index.
    const GPUSPARSE_INDEX_TYPE* rhsCols,    // Start id of each column.
    const GPUSPARSE_INDEX_TYPE* rhsRowIdx,  // Each non-zero row of the rhs sparse matrix get's an index (call it block-id). This array (size nnz) maps the nz-value row to the corresponding block-id.
    ElemType* resultValues,                 // Modified on return to contain values of the product.
    GPUSPARSE_INDEX_TYPE* blockId2Col       // Maps block-ids to column of the result matrix.
    )
{
    const CUDA_LONG index = blockIdx.x * blockDim.x + threadIdx.x;
    const CUDA_LONG lhsCol = index / numRowsLhs; // rhsCol == lhsCol
    if (lhsCol >= numColsRhs)
        return;
    const CUDA_LONG lhsRow = index - numRowsLhs * lhsCol; // resultRow == lhsRow

    // each thread handles one [row, col] combination of lhs
    ElemType lhsValue = alpha * lhsValues[IDX2C(lhsRow, lhsCol, numRowsLhs)];

    CUDA_LONG start = rhsCols[lhsCol]; // rhsCol == lhsCol
    CUDA_LONG end = rhsCols[lhsCol + 1];

    for (CUDA_LONG p = start; p < end; p++)
    {
        CUDA_LONG rhsRow = rhsRows[p];
        ElemType rhsVal = rhsNZValues[p];
        CUDA_LONG blockId = rhsRowIdx[p]; // resultCol == blockId
        blockId2Col[blockId] = rhsRow;    // indicate which colmn it actually points to

        // assume resultValues are 0-initialized
        atomicAdd(&resultValues[IDX2C(lhsRow, blockId, numRowsLhs)], lhsValue * rhsVal);
    }
}

// gradients update
template <class ElemType>
__global__ void _scaleSparseBlockAndAddToDense(
    const ElemType alpha,
    const bool blockCol, // true if blockRow
    const size_t numRows,
    const size_t numCols,
    const size_t numBlocks,
    const ElemType* lhsValues, // lhs is blockCol or blockRow
    const GPUSPARSE_INDEX_TYPE* blockIds,
    ElemType* rhs)
{
    const CUDA_LONG index = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG row, col;
    if (blockCol)
    {
        const CUDA_LONG blockId = index / numRows;
        if (blockId >= numBlocks)
            return;
        row = index - numRows * blockId;
        col = blockIds[blockId];
    }
    else
    {
        const CUDA_LONG blockId = index / numCols;
        if (blockId >= numBlocks)
            return;
        col = index - numCols * blockId;
        row = blockIds[blockId];
    }
    rhs[IDX2C(row, col, numRows)] += alpha * lhsValues[index];
}

#if 0
// compute predictions in cross entropy node
template <class ElemType>
__global__ void _computePrediction(
    int nv,
    const ElemType* a,
    int numrows,
    const ElemType* weight,
    int nrs,
    int labelSize,
    const GPUSPARSE_INDEX_TYPE* labelRow,
    const size_t* block2Id,
    const ElemType* cls,
    const ElemType* idx2cls,
    ElemType* val,
    GPUSPARSE_INDEX_TYPE* row,
    GPUSPARSE_INDEX_TYPE* pb)
{
    // get label block id
    int id = -1;
    int offset = -1;
    for (int i = 1; i < labelSize; i++)
    {
        if (blockIdx.x < block2Id[i])
        {
            id = i - 1;
            offset = blockIdx.x - block2Id[i - 1];
            break;
        }
    }
    if (id == -1)
    {
        id = labelSize - 1;
        offset = blockIdx.x - block2Id[labelSize - 1];
    }

    int t = labelRow[id];
    int iStt;
    int iEnd;
    if (t < nv)
    {
        int clsid = idx2cls[t];
        iStt = cls[IDX2C(0, clsid, 2)];
        iEnd = cls[IDX2C(1, clsid, 2)];
    }
    else
    {
        iStt = nv;
        iEnd = nrs;
    }
    int i = iStt + offset;
    int j = id / 2;

    int loadPerThread = (numrows + blockDim.x - 1) / blockDim.x;
    int tStart = loadPerThread * threadIdx.x;
    int tEnd = min((int) numrows, loadPerThread + tStart);

    ElemType v = 0.0;
    for (int h = tStart; h < tEnd; h++)
    {
        v += weight[IDX2C(i, h, nrs)] * a[IDX2C(h, j, numrows)];
    }
    atomicAdd(&val[blockIdx.x], v);
    row[blockIdx.x] = i;

    if (blockIdx.x == 0 && threadIdx.x == 0)
        pb[0] = 0;

    if ((threadIdx.x == 0) && (i == iEnd - 1) && (i >= nv))
        pb[j + 1] = blockIdx.x + 1;
}

// normalize predictions in cross entropy node
template <class ElemType>
__global__ void _normalizePrediction(
    const size_t labelSize,
    const size_t expandedLabelSize,
    const GPUSPARSE_INDEX_TYPE* labelRow,
    const size_t* block2Id,
    const GPUSPARSE_INDEX_TYPE* row,
    ElemType* val,
    ElemType* entropyScore)
{
    __shared__ ElemType partials[512];
    partials[threadIdx.x] = 0;

    int p = blockIdx.x;
    int t = labelRow[p];
    int start = block2Id[p];
    int end;
    if (p == labelSize - 1)
    {
        end = expandedLabelSize;
    }
    else
    {
        end = block2Id[p + 1];
    }
    int len = end - start;

    int loadPerThread = (len + blockDim.x - 1) / blockDim.x;
    int tStart = loadPerThread * threadIdx.x;
    int tLen = min((int) len, loadPerThread + tStart);

    for (int i = start + tStart; i < start + tLen; i++)
    {
        partials[threadIdx.x] += exp(val[i]);
    }

    __syncthreads();

    // now sum up the objective function
    int nTotalThreads = blockDim.x;

    while (nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);

        if (threadIdx.x < halfPoint)
            partials[threadIdx.x] += partials[threadIdx.x + halfPoint];

        __syncthreads();

        nTotalThreads = (nTotalThreads >> 1);
    }

    for (int i = start + tStart; i < start + tLen; i++)
    {
        val[i] = log(exp(val[i]) / partials[0]);
        if (row[i] == t)
        {
            atomicAdd(entropyScore, -val[i]);
            val[i] *= -1;
        }
    }
}

// compute prediction error in cross entropy node
template <class ElemType>
__global__ void _computePredictionError(
    ElemType* val,
    int N)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= N)
        return;

    if (val[p] < 0)
        val[p] = exp(val[p]); // negative;
    else
        val[p] = exp(-val[p]) - 1; // positive
}

// compute gradients of input in cross entropy node
template <class ElemType>
__global__ void _computeGradientOfInput(
    const ElemType* val,
    const GPUSPARSE_INDEX_TYPE* row,
    const GPUSPARSE_INDEX_TYPE* pb,
    ElemType* weight,
    size_t nrs,
    ElemType* grd,
    size_t numrows)
{
    int h = blockIdx.x % numrows;
    int j = blockIdx.x / numrows;

    int start = pb[j];
    int end = pb[j + 1];
    int len = end - start;

    int load = (len + blockDim.x - 1) / blockDim.x;
    int pStart = start + load * threadIdx.x;
    int pEnd = start + min(len, load * (threadIdx.x + 1));

    ElemType sum = 0;
    for (int p = pStart; p < pEnd; p++)
    {
        int i = row[p];
        sum += val[p] * weight[IDX2C(i, h, nrs)];
    }

    atomicAdd(&grd[IDX2C(h, j, numrows)], sum);
}
#endif

#if 0
template <class ElemType>
__global__ void computeNCEForwardProp512Threads(
    const ElemType* val,
    const int* col,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int numCols_a,
    const ElemType* b,
    ElemType* res)
{
    // val and col are in CSR format
    // val is an array contains log_Pn(w). To differentiate positive and negative samples,
    // we store log_Pn(w) as it is for positive samples, and -log_Pn(w) for negative samples
    // col is an array contains index of the word samples
    // a is a matrix in column major format contains output from hidden layer
    // b is the weight matrix for output layer
    // res is the buffer to store computed output (sparse)

    // follow the convention, this kernel must be run on 512 threads per block
    __shared__ ElemType partials[512];
    partials[threadIdx.x] = 0;

    // determine the elements to be handled by this block
    int total = numRows * sampleCount;
    int loadPerBlock = (total + gridDim.x - 1) / gridDim.x;

    int start = loadPerBlock * blockIdx.x;
    int end = min(total, loadPerBlock * (blockIdx.x + 1));

    for (int i = start; i < end; i++)
    {
        int colIndex = col[i];
        int rowIndex = i / sampleCount;

        int loadPerThread = (numCols_a + blockDim.x - 1) / blockDim.x;
        int tstart = loadPerThread * threadIdx.x;
        int tend = min(numCols_a, loadPerThread * (threadIdx.x + 1));

        for (int j = tstart; j < tend; j++)
            partials[threadIdx.x] = a[IDX2C(rowIndex, j, numRows)] * b[IDX2C(j, colIndex, numCols_a)];

        __syncthreads();

        // sum up
        int nTotalThreads = blockDim.x;

        while (nTotalThreads > 1)
        {
            int halfPoint = (nTotalThreads >> 1);

            if (threadIdx.x < halfPoint)
                partials[threadIdx.x] += partials[threadIdx.x + halfPoint];

            __syncthreads();

            nTotalThreads = (nTotalThreads >> 1);
        }

        if (threadIdx.x == 0)
            res[i] = partials[0];
    }
}
#endif

template <class ElemType>
__global__ void _computeNceOutputMax512Threads(
    const ElemType* col,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int numCols_a,
    const ElemType* b,
    const ElemType* bias,
    ElemType* res)
{
    // val and col are in CSR format
    // val is an array contains log_Pn(w). To differentiate positive and negative samples,
    // we store log_Pn(w) as it is for positive samples, and -log_Pn(w) for negative samples
    // col is an array contains index of the word samples
    // a is a matrix in column major format contains output from hidden layer
    // b is the weight matrix for output layer
    // res is the buffer to store computed output (sparse)

    // follow the convention, this kernel must be run on 512 threads per block
    __shared__ ElemType partials[512];
    partials[threadIdx.x] = 0;

    // threadIdx.x range from[0 ~ 512)
    // blockIdx.x range from[0 ~ nnz)
    // blockDim.x equal to 512
    // gridDim.x equal to nnz

    // determine the elements to be handled by this block
    int total = numRows * sampleCount;
    int loadPerBlock = (total + gridDim.x - 1) / gridDim.x;

    int start = loadPerBlock * blockIdx.x;
    int end = min(total, loadPerBlock * (blockIdx.x + 1));

    for (int i = start; i < end; i++)
    {
        int wid = (int) col[2 * i];
        int batchid = i / sampleCount;

        int loadPerThread = (numCols_a + blockDim.x - 1) / blockDim.x;
        int tstart = loadPerThread * threadIdx.x;
        int tend = min(numCols_a, loadPerThread * (threadIdx.x + 1));

        for (int j = tstart; j < tend; j++)
            partials[threadIdx.x] = a[IDX2C(j, batchid, numCols_a)] * b[IDX2C(j, wid, numCols_a)];

        __syncthreads();

        // sum up
        int nTotalThreads = blockDim.x;

        while (nTotalThreads > 1)
        {
            int halfPoint = (nTotalThreads >> 1);

            if (threadIdx.x < halfPoint)
                partials[threadIdx.x] += partials[threadIdx.x + halfPoint];

            __syncthreads();

            nTotalThreads = (nTotalThreads >> 1);
        }

        if (threadIdx.x == 0)
            res[i] = partials[0] + bias[wid];
    }
}

template <class ElemType>
__global__ void _assignSoftmaxSumMax512Threads(
    const ElemType* softmax,
    int sampleCount,
    const ElemType* a,
    ElemType* c) // run on 512 threads per block
{
    // val and col are in CSR format
    // val is an array contains log_Pn(w). To differentiate positive and negative samples,
    // we store log_Pn(w) as it is for positive samples, and -log_Pn(w) for negative samples
    // col is an array contains index of the word samples
    // a is a matrix in column major format contains output from hidden layer
    // b is the weight matrix for output layer
    // tmp is the buffer that stores NCE output calculated from _computeNceOutputMax512Threads
    // c is the matrix to store objective

    __shared__ ElemType partials[512];
    partials[threadIdx.x] = 0;

    int total = sampleCount;
    int loadPerThread = (total + blockDim.x - 1) / blockDim.x;

    // find out the items this thread is responsible for
    int start = loadPerThread * threadIdx.x;
    int end = min(total, loadPerThread * (threadIdx.x + 1));
    for (int i = start; i < end; i++)
    {
        int wid = (int) a[i];
        partials[threadIdx.x] += softmax[IDX2C(i, wid, sampleCount)];
    }

    __syncthreads();

    // now sum up the objective function
    int nTotalThreads = blockDim.x;

    while (nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);

        if (threadIdx.x < halfPoint)
            partials[threadIdx.x] += partials[threadIdx.x + halfPoint];

        __syncthreads();

        nTotalThreads = (nTotalThreads >> 1);
    }

    if (threadIdx.x == 0)
        c[0] = -partials[0];
}

template <class ElemType>
__global__ void _assignNoiseContrastiveEstimationMax512Threads(
    const ElemType* val,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int width, // number of columns in a
    const ElemType* b,
    ElemType* tmp,
    ElemType* c) // run on 512 threads per block
{
    // val and col are in CSR format
    // val is an array contains log_Pn(w). To differentiate positive and negative samples,
    // we store log_Pn(w) as it is for positive samples, and -log_Pn(w) for negative samples
    // col is an array contains index of the word samples
    // a is a matrix in column major format contains output from hidden layer
    // b is the weight matrix for output layer
    // tmp is the buffer that stores NCE output calculated from _computeNceOutputMax512Threads
    // c is the matrix to store objective

    __shared__ ElemType partials[512];
    partials[threadIdx.x] = 0;

    int total = numRows * sampleCount;
    int loadPerThread = (total + blockDim.x - 1) / blockDim.x;

    // find out the items this thread is responsible for
    int start = loadPerThread * threadIdx.x;
    int end = min(total, loadPerThread * (threadIdx.x + 1));

    ElemType log_num_noise_samples = log((ElemType)(sampleCount - 1));
    for (int i = start; i < end; i++)
    {
        ElemType prob = -val[2 * i + 1];
        bool positive = (prob > 0);
        if (positive)
            prob = -prob;
        ElemType score_noise = log_num_noise_samples + prob;
        ElemType z = logaddk(tmp[i], score_noise);
        ElemType logprob = tmp[i] - z;
        ElemType logprob_noise = score_noise - z;
        tmp[i] = -exp(logprob);
        if (positive)
            tmp[i] += 1;
        if (positive)
            partials[threadIdx.x] += logprob;
        else
            partials[threadIdx.x] += logprob_noise;
    }

    __syncthreads();

    // now sum up the objective function
    int nTotalThreads = blockDim.x;

    while (nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);

        if (threadIdx.x < halfPoint)
            partials[threadIdx.x] += partials[threadIdx.x + halfPoint];

        __syncthreads();

        nTotalThreads = (nTotalThreads >> 1);
    }

    if (threadIdx.x == 0)
        c[0] = -partials[0];
}

template <class ElemType>
__global__ void _assignNceDerivative(
    const ElemType* val,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int width, // number of columns in a
    const ElemType* b,
    const ElemType* tmp,
    ElemType* c,
    size_t inputIndex)
{
    // val and col are CSR format sparse matrix for label
    // val is an array contains log_Pn(w). To differentiate positive and negative samples
    // we store log_Pn(w) as it is for positive samples, and -log_Pn(w) for negative samples
    // col is an array contains index of the word samples
    // a is a matrix in column major format contains output from hidden layer
    // b is the weight matrix for output layer
    // tmp is a matrix of precalculated error
    // c is the output matrix to store calculated gradients

    int total = numRows * sampleCount;
    int loadPerBlock = (total + gridDim.x - 1) / gridDim.x;

    // find out the items this block is responsible for
    int start = loadPerBlock * blockIdx.x;
    int end = min(total, loadPerBlock * (blockIdx.x + 1));

    for (int i = start; i < end; i++)
    {
        int wid = (int) val[2 * i];
        int batchId = i / sampleCount;

        ElemType er = tmp[i]; // precalculated error for this output node

        // calculate gradients
        int loadPerThread = (width + blockDim.x - 1) / blockDim.x;
        int tstart = loadPerThread * threadIdx.x;
        int tend = min(width, loadPerThread * (threadIdx.x + 1));

        if (inputIndex == 1) // hidden layer output
        {
            for (int j = tstart; j < tend; j++)
            {
                ElemType val = -er * b[IDX2C(j, wid, width)];
                atomicAdd(&c[IDX2C(j, batchId, width)], val);
                // c[IDX2C(j, batchId, width)] += val;
                // c[IDX2C(batchId, j, numRows)] += val;
            }
        }
        else if (inputIndex == 2) // weight
        {
            for (int j = tstart; j < tend; j++)
            {
                ElemType val = -er * a[IDX2C(j, batchId, width)];
                atomicAdd(&c[IDX2C(j, wid, width)], val);
                // c[IDX2C(j, wid, width)] += val;
            }
        }
        else // bias vector
        {
            // ElemType val = -er;
            atomicAdd(&c[wid], -er);
            // c[wid] -= er;
        }
    }
}

template <class ElemType>
__global__ void _assignNceDerivativeNew(
    const ElemType* val,
    int numRows,
    int sampleCount,
    const ElemType* a,
    int width, // number of columns in a
    const ElemType* b,
    const ElemType* tmp,
    ElemType* c,
    size_t inputIndex)
{
    // val and col are CSR format sparse matrix for label
    // val is an array contains log_Pn(w). To differentiate positive and negative samples
    // we store log_Pn(w) as it is for positive samples, and -log_Pn(w) for negative samples
    // col is an array contains index of the word samples
    // a is a matrix in column major format contains output from hidden layer
    // b is the weight matrix for output layer
    // tmp is a matrix of precalculated error
    // c is the output matrix to store calculated gradients

    // logical single index for this thread
    int n = threadIdx.x + blockDim.x * blockIdx.x;

    int batchId = n / sampleCount;
    int total = numRows * sampleCount;
    // is thread in range for the addition
    if (n < total)
    {
        int wid = (int) val[2 * n];
        ElemType er = tmp[n];
        if (inputIndex == 1)
        {
            for (int i = 0; i < width; i++)
            {
                int j = (i + n) % width; // introduce randomization to avoid conflicts
                ElemType val = -er * b[IDX2C(j, wid, width)];
                atomicAdd(&c[IDX2C(j, batchId, width)], val);
            }
        }
        else if (inputIndex == 2)
        {
            for (int i = 0; i < width; i++)
            {
                int j = (i + n) % width; // introduce randomization to avoid conflicts
                ElemType val = -er * a[IDX2C(j, batchId, width)];
                atomicAdd(&c[IDX2C(j, wid, width)], val);
            }
        }
        else
            atomicAdd(&c[wid], -er);
    }
}

#if 0
// compute gradients of weights in cross entropy node
template <class ElemType>
__global__ void _computeGradientOfWeight(
    const ElemType* val,
    const GPUSPARSE_INDEX_TYPE* row,
    const GPUSPARSE_INDEX_TYPE* pb,
    size_t mb,
    size_t nv,
    const GPUSPARSE_INDEX_TYPE* labelRow,
    const size_t* labelBlock2UniqId,
    const ElemType* cls,
    const ElemType* idx2cls,
    ElemType* input,
    size_t nrs,
    ElemType* blockVal,
    GPUSPARSE_INDEX_TYPE* blockIds)
{
    int p = blockIdx.x;
    ElemType v = val[p];
    int i = row[p];
    int j = -1;
    for (int k = 1; k < mb; k++)
    {
        if (p < pb[k])
        {
            j = k - 1;
            break;
        }
    }
    if (j == -1)
    {
        j = mb - 1;
    }

    // figure out blocks
    int bId = i < nv ? 2 * j : 2 * j + 1;
    int t = labelRow[bId];
    int iStt;
    if (t < nv)
    {
        int clsid = idx2cls[t];
        iStt = cls[IDX2C(0, clsid, 2)];
    }
    else
    {
        iStt = nv;
    }
    int offset = i - iStt;
    int ii = labelBlock2UniqId[bId] + offset;

    int load = (nrs + blockDim.x - 1) / blockDim.x;
    int pStart = load * threadIdx.x;
    int pEnd = min((int) nrs, load + pStart);

    for (int h = pStart; h < pEnd; h++)
    {
        ElemType temp = v * input[IDX2C(h, j, nrs)];
        atomicAdd(&blockVal[ii * nrs + h], temp);
        blockIds[ii] = i;
    }
}
#endif

// used in clipping gradients
template <class ElemType>
__global__ void _inplaceTruncate(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)
    ElemType locThresholdPos = abs(threshold);
    ElemType locTHresholdNeg = -locThresholdPos;
    if (a[id] > locThresholdPos)
    {
        a[id] = locThresholdPos;
    }
    else if (a[id] < locTHresholdNeg)
    {
        a[id] = locTHresholdNeg;
    }
}

template <class ElemType>
__global__ void _inplaceSoftThreshold(
    ElemType* a,
    const ElemType threshold,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    if (a[id] > threshold)
    {
        a[id] -= threshold;
    }
    else if (a[id] < -threshold)
    {
        a[id] += threshold;
    }
    else
        a[id] = 0;
}

template <class ElemType>
__global__ void _normalGradForSparseBlock(
    const ElemType momentum,
    const bool blockCol, // true if blockRow
    const size_t numRows,
    const size_t numCols,
    const size_t numBlocks,
    ElemType* lhsValues, // lhs is blockCol or blockRow
    const GPUSPARSE_INDEX_TYPE* blockIds,
    ElemType* rhs,
    bool unitGainMomentum)
{
    const ElemType unitGainFactor = unitGainMomentum ? (1.0 - momentum) : 1.0;
    const CUDA_LONG index = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG row, col;
    if (blockCol)
    {
        const CUDA_LONG blockId = index / numRows;
        if (blockId >= numBlocks)
            return;
        row = index - numRows * blockId;
        col = blockIds[blockId];
    }
    else
    {
        const CUDA_LONG blockId = index / numCols;
        if (blockId >= numBlocks)
            return;
        col = index - numCols * blockId;
        row = blockIds[blockId];
    }
    rhs[IDX2C(row, col, numRows)] = unitGainFactor * lhsValues[index] + momentum * rhs[IDX2C(row, col, numRows)];
    lhsValues[index] = rhs[IDX2C(row, col, numRows)];
}

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template <class ElemType>
__global__ void _reductionSum1024Threads(
    const ElemType* data,
    ElemType* sum,
    CUDA_LONG N)
{

    __shared__ ElemType partialSums[1024];
    partialSums[threadIdx.x] = 0;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG loadPerThread = N / blockDim.x;
    for (CUDA_LONG i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        partialSums[threadIdx.x] += data[i];
    }
    __syncthreads();

    // 512
    if (threadIdx.x < 512)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 512];
    }
    __syncthreads();

    // 256
    if (threadIdx.x < 256)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 256];
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 128];
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 64];
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 32];
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 16];
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 8];
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 4];
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        sum[0] = partialSums[0] + partialSums[1] + partialSums[2] + partialSums[3];
    }
}

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template <class ElemType>
__global__ void _reductionSumAndAssign1024Threads(
    ElemType* toAssign,
    const ElemType* data,
    CUDA_LONG N, // length of data
    CUDA_LONG M) // length of toAssign
{
    __shared__ ElemType partialSums[1024];
    __shared__ ElemType res;
    partialSums[threadIdx.x] = 0;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG loadPerThread = N / blockDim.x;
    for (CUDA_LONG i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        partialSums[threadIdx.x] += data[i];
    }
    __syncthreads();

    // 512
    if (threadIdx.x < 512)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 512];
    }
    __syncthreads();

    // 256
    if (threadIdx.x < 256)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 256];
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 128];
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 64];
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 32];
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 16];
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 8];
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 4];
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        res = partialSums[0] + partialSums[1] + partialSums[2] + partialSums[3];
        for (CUDA_LONG i = 0; i < M; ++i)
            toAssign[i] = res;
    }
}

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template <class ElemType>
__global__ void _reductionSum21024Threads(
    const ElemType* data,
    ElemType* sum,
    CUDA_LONG N,
    bool takeSqrt = false)
{

    __shared__ ElemType partialSums[1024];
    partialSums[threadIdx.x] = 0;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG loadPerThread = N / blockDim.x;
    for (CUDA_LONG i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    // for (int i= threadIdx.x*loadPerThread; i<(threadIdx.x+1)*loadPerThread;++i)
    {
        partialSums[threadIdx.x] += (data[i] * data[i]);
    }
    __syncthreads();

    // 512
    if (threadIdx.x < 512)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 512];
    }
    __syncthreads();

    // 256
    if (threadIdx.x < 256)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 256];
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 128];
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 64];
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 32];
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 16];
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 8];
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums[threadIdx.x] += partialSums[threadIdx.x + 4];
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        sum[0] = partialSums[0] + partialSums[1] + partialSums[2] + partialSums[3];
        if (takeSqrt)
        {
            if (sizeof(ElemType) == sizeof(float))
                sum[0] = sqrtf(sum[0]);
            else
                sum[0] = sqrt(sum[0]);
        }
    }
}

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template <class ElemType>
__global__ void _reductionMatrixNormInf1024Threads(
    const ElemType* data,
    ElemType* maxAbs,
    CUDA_LONG N)
{

    __shared__ ElemType partialSums[1024];
    partialSums[threadIdx.x] = 0;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    int loadPerThread = N / blockDim.x;
    for (int i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            partialSums[threadIdx.x] = max(fabsf(data[i]), partialSums[threadIdx.x]);
        }
        else
        {
            partialSums[threadIdx.x] = max(fabs(data[i]), partialSums[threadIdx.x]);
        }
    }
    __syncthreads();

    // 512
    if (threadIdx.x < 512)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 512], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 256
    if (threadIdx.x < 256)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 256], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 128], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 64], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 32], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 16], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 8], partialSums[threadIdx.x]);
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums[threadIdx.x] = max(partialSums[threadIdx.x + 4], partialSums[threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        maxAbs[0] = max(max(partialSums[0], partialSums[1]), max(partialSums[2], partialSums[3]));
    }
}

//This function should be called with 1024 threads per block and 1 block
//THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION!!!
template <class ElemType>
__global__ void _reductionMatrixNorm01024Threads(
    const ElemType* data,
    ElemType* nz,
    CUDA_LONG N)
{

    __shared__ ElemType partialSums[1024];
    partialSums[threadIdx.x] = 0;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG loadPerThread = N / blockDim.x;
    for (CUDA_LONG i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        if (data[i] != 0)
            ++partialSums[threadIdx.x];
    }
    __syncthreads();

    // 512
    if (threadIdx.x < 512)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 512] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 256
    if (threadIdx.x < 256)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 256] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 128] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 64] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 32] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 16] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 8] + partialSums[threadIdx.x];
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums[threadIdx.x] = partialSums[threadIdx.x + 4] + partialSums[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        nz[0] = partialSums[0] + partialSums[1] + partialSums[2] + partialSums[3];
    }
}

template <class ElemType>
__global__ void _getSparseVectorRepresntationForCSCMatrix(
    const int* m_dRow,
    const int* m_dCol,
    int* vectArray,
    const CUDA_LONG M,
    const CUDA_LONG N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M)
        return;
    int start = m_dRow[i];
    int end = m_dRow[i + 1];
    for (int _i = start; _i < end; ++_i) // _i is index in m_dVal and m_dCol
    {
        int j = m_dCol[_i];
        vectArray[_i] = i * N + j;
    }
}

template <class ElemType>
__global__ void _lrHelper512Threads(
    const ElemType* data1,
    const ElemType* data2,
    const CUDA_LONG N,
    ElemType* d_res)
{
    __shared__ ElemType partialSums1[512];
    __shared__ ElemType partialSums2[512];
    partialSums1[threadIdx.x] = 0;
    partialSums2[threadIdx.x] = 0;

    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    int loadPerThread = N / blockDim.x;
    for (int i = threadIdx.x * loadPerThread; i < (threadIdx.x == blockDim.x - 1 ? N : (threadIdx.x + 1) * loadPerThread); ++i)
    {
        partialSums1[threadIdx.x] += (data1[i] * data1[i]);
        partialSums2[threadIdx.x] += (data2[i] * data2[i]);
    }
    __syncthreads();

    /*
    // 512
    if (threadIdx.x<512)
    {
    partialSums1[threadIdx.x]+=partialSums1[threadIdx.x+512];
    partialSums2[threadIdx.x]+=partialSums2[threadIdx.x+512];
    }
    __syncthreads();*/

    // 256
    if (threadIdx.x < 256)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 256];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 256];
    }
    __syncthreads();

    // 128
    if (threadIdx.x < 128)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 128];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 128];
    }
    __syncthreads();

    // 64
    if (threadIdx.x < 64)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 64];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 64];
    }
    __syncthreads();

    // 32
    if (threadIdx.x < 32)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 32];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 32];
    }
    __syncthreads();

    // 16
    if (threadIdx.x < 16)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 16];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 16];
    }
    __syncthreads();

    // 8
    if (threadIdx.x < 8)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 8];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 8];
    }
    __syncthreads();

    // 4
    if (threadIdx.x < 4)
    {
        partialSums1[threadIdx.x] += partialSums1[threadIdx.x + 4];
        partialSums2[threadIdx.x] += partialSums2[threadIdx.x + 4];
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        ElemType fns1 = partialSums1[0] + partialSums1[1] + partialSums1[2] + partialSums1[3];
        ElemType fns2 = partialSums2[0] + partialSums2[1] + partialSums2[2] + partialSums2[3];
        if (sizeof(ElemType) == sizeof(float))
        {
            d_res[0] = max((ElemType) 0, d_res[0] / max((ElemType) 1.0e-10, sqrtf(fns1)) / max((ElemType) 1.0e-10, sqrtf(fns2)));
        }
        else
        {
            d_res[0] = max((ElemType) 0, d_res[0] / max((ElemType) 1.0e-10, sqrt(fns1)) / max((ElemType) 1.0e-10, sqrt(fns2)));
        }
    }
}

/*
template<class ElemType>
__global__ void _lrHelper512Threads(
ElemType* d_tmp)
{
if (sizeof(ElemType)==sizeof(float))
{
d_tmp[0] = max((ElemType)0, d_tmp[0]/max((ElemType)1.0e-10,sqrtf(d_tmp[1]))/max((ElemType)1.0e-10,sqrtf(d_tmp[2])));
}
else
{
d_tmp[0] = max((ElemType)0, d_tmp[0]/max((ElemType)1.0e-10,sqrt(d_tmp[1]))/max((ElemType)1.0e-10,sqrt(d_tmp[2])));
}
}
*/

template <class ElemType>
__global__ void _assignElementProductOfWithShiftNeg(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int shift,
    const int NTPlusOne,
    const int BS)
{
    CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= NTPlusOne || idy >= BS)
        return;

    if (idx == 0)
    {
        // this is row-0. No need to shift
        us[IDX2C(idx, idy, NTPlusOne)] = a[idy] * b[idy];
    }
    else
    {
        int cs = shift + idx - 1;
        int tmpidy = (idy + cs) % BS;
        us[IDX2C(idx, idy, NTPlusOne)] = a[idy] * b[tmpidy];
    }
}

template <class ElemType>
__global__ void _innerProductWithShiftNeg(
    ElemType* c,
    const ElemType* a,
    const ElemType* b,
    const CUDA_LONG N, // a.GetNumRows();
    const CUDA_LONG M, // a.GetNumCols();
    const CUDA_LONG shift,
    const CUDA_LONG NTPlusOne)
{
    CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= NTPlusOne || idy >= M)
        return;

    ElemType sum = 0;
    CUDA_LONG index_a = 0;
    CUDA_LONG index_b = 0;
    CUDA_LONG col_a = 0;
    CUDA_LONG col_b = 0;
    if (idx == 0)
    {
        // this is row 0. No need to shift
        // the product of a(:,idy) dot b(:,idy)
        col_a = idy;
        for (CUDA_LONG i = 0; i < N; ++i)
        {
            index_a = IDX2C(i, col_a, N);
            sum += a[index_a] * b[index_a];
        }
    }
    else
    {
        int cs = shift + idx - 1;
        col_a = idy;
        col_b = (idy + cs) % M;
        for (int i = 0; i < N; ++i)
        {
            index_a = IDX2C(i, col_a, N);
            index_b = IDX2C(i, col_b, N);
            sum += a[index_a] * b[index_b];
        }
    }
    c[IDX2C(idx, idy, NTPlusOne)] = sum;
}

template <class ElemType>
__global__ void _getARowByIndex(
    ElemType* us,
    const ElemType* a,
    const int O, // a's rows
    const int P, // a's cols
    const int m  // the m-th row of a
    )
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= P)
        return;
    //    us[id] = a[id] * b[id];
    us[id] = a[IDX2C(m, id, O)];
}

template <class ElemType>
__global__ void _conductRowElementMultiplyWithShift(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int O, // b's rows
    const int P, // b's cols
    const int shift,
    const bool isafixed)
{
    CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
    CUDA_LONG idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= O || idy >= P)
        return;

    int tmpidy = (idy + shift) % P;
    if (isafixed)
    {
        // we fix a, and shift b
        us[IDX2C(idx, idy, O)] = a[idy] * b[IDX2C(idx, tmpidy, O)];
    }
    else
    {
        // we fix b, but shift a
        us[IDX2C(idx, idy, O)] = a[tmpidy] * b[IDX2C(idx, idy, O)];
    }
}

template <class ElemType>
__global__ void _assignElementProductOfWithShift(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int shift,
    const CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;

    int tmpidb = (id + shift) % N;
    us[id] = a[id] * b[tmpidb];
}

// minus 1 at a specific position
template <class ElemType>
__global__ void _minusOneAt(
    ElemType* c,
    CUDA_LONG position,
    CUDA_LONG N)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    if (id == position)
        c[id] = c[id] - 1.0;
}

// the kernel function for CRFLSTMNetwork  backward computation
// assume a column slice of input and output
// This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == 3 * iNumLab.
template <class ElemType>
__global__ void _rcrfBackwardComputeMax1024Labels(
    const size_t t, // time position
    const size_t iNumPos,
    const ElemType* galpha,       // column slice at current time t
    ElemType* gbeta,              // column slices with [row, 2] at current time t for [
    const ElemType* gzeta,        // column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores, // column slice at current time t
    const size_t iNumLab, const int shift)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ double sh_alpha_and_beta[]; // [id] or [id + iNumLab] or [id + 2 * iNumLab)]
    // need byte size = (iNumPos * iNumLab * 2 + iNumLab * iNumLab) * sizeof(ElemType)

    ElemType* alpha = (ElemType*) (sh_alpha_and_beta);
    ElemType* beta_t1 = (ElemType*) (alpha + iNumLab);
    ElemType* zeta = (ElemType*) (beta_t1 + iNumLab);
    ElemType pair_scores[1024];  // [j=0..iNumLab-1]

    if (id < 0 || id >= iNumLab)
        return;

    // copy global memory to shared memory to save time
    alpha[id] = galpha[IDX2C(id, t, iNumLab)];
    if (t < iNumPos - 1)
        beta_t1[id] = gbeta[IDX2C(id, t + 1, iNumLab)];
    zeta[id] = gzeta[id];

    __syncthreads();

    for (int j = 0; j < iNumLab; j++)
        pair_scores[j] = gpair_scores[IDX2C(j, id, iNumLab)];

    ElemType fTmp = LZERO;
    if (t == iNumPos - 1)
    {
        fTmp = alpha[id] - zeta[id];
    }
    else
    {
        for (int j = 0; j < iNumLab; j++)
        {
            fTmp = logaddk(fTmp, beta_t1[j] + alpha[id] + pair_scores[j] - zeta[j]);
        }
    }

    gbeta[IDX2C(id, t, iNumLab)] = fTmp;
}

// $\zeta_t(j) = {\sum_k exp(\delta_{t-1}(k) + a_{kj}(t))}$.
// This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
template <class ElemType>
__global__ void _rcrfBackwardComputeZetaMax1024Labels(
    const size_t t, // time position
    const size_t iNumPos,
    const ElemType* galpha, // column slice at current time t
    ElemType* gzeta,        // column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores,
    const size_t iNumLab, const int shift)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ double sh_alpha_and_beta[]; // [id]
    // need byte size = (iNumPos * iNumLab * 2 + iNumLab * iNumLab) * sizeof(ElemType)

    ElemType* alpha = (ElemType*) (sh_alpha_and_beta);
    ElemType pair_scores[1024]; // [j=0..iNumLab-1]

    if (id < 0 || id >= iNumLab)
        return;

    // copy global memory to shared memory to save time
    alpha[id] = galpha[IDX2C(id, t, iNumLab)];

    __syncthreads();

    for (int j = 0; j < iNumLab; j++)
        pair_scores[j] = gpair_scores[IDX2C(id, j, iNumLab)];

    ElemType fSum = LZERO;
    for (int m = 0; m < iNumLab; m++)
    {
        if (t == iNumPos - 1)
            fSum = logaddk(fSum, alpha[IDX2C(m, 0, iNumLab)]);
        else
            fSum = logaddk(fSum, alpha[IDX2C(m, 0, iNumLab)] + pair_scores[m]);
    }

    gzeta[id] = fSum;
}

/// $\zeta_t(j) = {\sum_k exp(\delta_{t-1}(k) + a_{kj}(t))}$.
// This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
template <class ElemType>
__global__ void _rcrfTransGrdComputeZetaMax1024Labels(
    const int t, // time position
    const size_t iNumPos,
    const ElemType* galpha, // column slice at current time t
    ElemType* gzeta,        // column slices with [row, 2] at current time t for [
    const ElemType* gpair_scores,
    const size_t iNumLab,
    const size_t start_lbl,
    const int shift)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ double sh_alpha_and_beta[]; // [id]
    // need byte size = (iNumPos * iNumLab * 2 + iNumLab * iNumLab) * sizeof(ElemType)

    ElemType* alpha = (ElemType*) (sh_alpha_and_beta);
    ElemType pair_scores[1024]; // [j=0..iNumLab-1]

    if (id < 0 || id >= iNumLab)
        return;

    // copy global memory to shared memory to save time
    if (t >= 0)
        alpha[id] = galpha[IDX2C(id, t, iNumLab)];

    __syncthreads();

    for (int j = 0; j < iNumLab; j++)
        pair_scores[j] = gpair_scores[IDX2C(id, j, iNumLab)];

    ElemType fSum = LZERO;
    ElemType fTmp;
    for (int m = 0; m < iNumLab; m++)
    {
        if (t < 0)
        {
            if (m == start_lbl)
                fTmp = 0;
            else
                fTmp = LZERO;
        }
        else
            fTmp = alpha[m];

        fSum = logaddk(fSum, pair_scores[m] + fTmp);
    }

    gzeta[id] = fSum;
}

// This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
template <class ElemType>
__global__ void _rcrfTransGrdComputeMax1024Labels(
    int t,
    const size_t start_lbl,
    const ElemType* galpha,
    const ElemType* gbeta,
    const ElemType* gzeta,
    const ElemType* gpair_scores,
    const ElemType* lbls,
    ElemType* grd,
    const size_t iNumPos,
    const size_t iNumLab,
    const int shift)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ double sh_alpha_and_beta[]; // [id]
    // need byte size = (iNumPos * iNumLab * 2 + iNumLab * iNumLab) * sizeof(ElemType)

    ElemType* alpha = (ElemType*) (sh_alpha_and_beta);
    ElemType* beta = (ElemType*) (alpha + iNumLab);
    ElemType* zeta = (ElemType*) (beta + iNumLab);
    ElemType pair_scores[1024]; // [j=0..iNumLab-1]

    if (id < 0 || id >= iNumLab)
        return;

    // copy global memory to shared memory to save time
    if (t > 0)
        alpha[id] = galpha[IDX2C(id, t - 1, iNumLab)];
    beta[id] = gbeta[IDX2C(id, t, iNumLab)];
    zeta[id] = gzeta[id];

    __syncthreads();

    for (int j = 0; j < iNumLab; j++)
        pair_scores[j] = gpair_scores[IDX2C(j, id, iNumLab)];

    ElemType fTmp;
    ElemType fTmp2;
    for (int j = 0; j < iNumLab; j++)
    {
        if (t == 0)
        {
            if (id == start_lbl)
                fTmp = 0;
            else
                fTmp = LZERO;
        }
        else
            fTmp = alpha[id];

        fTmp2 = fTmp + pair_scores[j] - zeta[j];
        assert(fTmp2 <= 0.0);
        fTmp2 += beta[j];

        fTmp = exp(fTmp2);
        grd[IDX2C(j, id, iNumLab)] += fTmp;
    }

    if ((t == 0 && id == start_lbl) || (t > 0 && t < iNumPos && lbls[IDX2C(id, t - 1, iNumLab)] != 0))
    {
        for (int ik = 0; ik < iNumLab; ik++)
        {
            if (lbls[IDX2C(ik, t, iNumLab)] != 0)
                grd[IDX2C(ik, id, iNumLab)] -= 1.0;
        }
    }
};

template <class ElemType>
__global__ void _reductionLogAddSum(
    const ElemType* data,
    ElemType* sum,
    const size_t sum_size,
    CUDA_LONG N)
{

    __shared__ ElemType partialLogAddSum[GridDim::maxThreadsPerBlock];

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if (id < N)
        partialLogAddSum[tid] = data[id];
    else
        partialLogAddSum[tid] = LZERO;

    __syncthreads();

    // do reduction on the shared memory
    size_t start_width = ceil((N + 0.0) / 2.0);
    for (size_t s = start_width; s > 0; s >>= 1)
    {
        ElemType lSum = LZERO;
        if (tid < s)
        {
            lSum = logaddk(partialLogAddSum[tid], partialLogAddSum[tid + s]);
            partialLogAddSum[tid] = lSum;
        }
    }
    __syncthreads();

    if (tid == 0)
        sum[0] = partialLogAddSum[0];
}

// set the value of certain columns to be zero
// the column is decided by threshhold value
// TODO: This kernel has very poor performace and needs to
// be optimized
template <class ElemType>
__global__ void _DropFrame(
    ElemType* a,
    const ElemType* label,
    const ElemType* gamma,
    const ElemType framedropthreshhold,
    const long m_numCols,
    const long m_numRows) // ld
{
    int col_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (col_id >= m_numCols)
        return;
    bool dropframe = false;
    // find the 1 in the one-hot representation of the labels
    // This is a linear scan--bad perf!
    for (long i = 0; i < m_numRows; ++i)
    {
        int idx = IDX2C(i, col_id, m_numRows);
        // printf("%u ", idx);
        if (fabs(label[idx] - 1.0) < 0.1) // we found the 1 in the vector
        {
            if (gamma[idx] < framedropthreshhold)
                dropframe = true;
            break;
        }
    }

    if (dropframe)
    {
        // printf("frame dropped %u ", col_id);
        for (long i = 0; i < m_numRows; ++i)
        {
            a[IDX2C(i, col_id, m_numRows)] = 0.0;
        }
    }
}

template <class ElemType>
__global__ void _AssignSequenceError(const ElemType hsmoothingWeight, ElemType* error, const ElemType* label,
                                     const ElemType* dnnoutput, const ElemType* gamma, ElemType alpha, const long N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    error[id] -= alpha * (label[id] - (1.0 - hsmoothingWeight) * dnnoutput[id] - hsmoothingWeight * gamma[id]);
    // change to ce
    // error[id] -= alpha * (label[id] - dnnoutput[id] );
}

template <class ElemType>
__global__ void _copyTopKResults(const uint64_t* indexes, const ElemType* values, ElemType* maxIndexes, ElemType* maxValues,
                                 CUDA_LONG crow, CUDA_LONG ccol, int topK)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= topK * ccol)
        return;
    CUDA_LONG irow = id % topK;
    CUDA_LONG icol = id / topK;
    maxIndexes[id] = static_cast<CUDA_LONG>(indexes[icol * crow + irow] >> 32);
    maxValues[id] = values[icol * crow + irow];
}

template <int BlockSize, class ElemType>
__global__ void _assignNumOfDiffCol(const ElemType* a, const ElemType* b, ElemType* c, CUDA_LONG crowB, CUDA_LONG ccol)
{
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);

    int cur = 0;
    CUDA_LONG icol = threadIdx.x;
    for (; icol < ccol; icol += blockDim.x)
    {
        ElemType key = a[icol];
        CUDA_LONG idxB = icol * crowB;
        CUDA_LONG irow = 0;
        for (; irow < crowB; irow++, idxB++)
        {
            if (b[idxB] == key)
                break;
        }

        cur += (irow == crowB);
    }

    using BlockReduceT = cub::BlockReduce<int, BlockSize>;
    __shared__ typename BlockReduceT::TempStorage tmp;

    int res = BlockReduceT(tmp).Sum(cur);
    if (threadIdx.x == 0)
        *c = res;
}

template <class ElemType>
__global__ void _maskColumnsValue(ElemType* a, const char* columnsMask, CUDA_LONG numCols, CUDA_LONG numRows, ElemType val, CUDA_LONG numColsPerMaskEntry)
{
    CUDA_LONG maskColIdx = blockIdx.x;
    CUDA_LONG matrixStartColIdx = maskColIdx * numColsPerMaskEntry;

    for (CUDA_LONG k = 0; k < numColsPerMaskEntry; ++k)
    {
        CUDA_LONG colIdx = matrixStartColIdx + k;
        if (colIdx > numCols)
            return;

        if (columnsMask[IDX2C(0, maskColIdx, 1)] == 1)
            return;

        CUDA_LONG rowIdx = threadIdx.x;
        for (; rowIdx < numRows; rowIdx += blockDim.x)
        {
            a[IDX2C(rowIdx, colIdx, numRows)] = val;
        }
    }
}

template <class ElemType>
__global__ void _adam(CUDA_LONG size, ElemType* grad, ElemType* smoothAda, ElemType* smoothMom, ElemType* val,
    ElemType lr, ElemType mom, ElemType adaWeight, ElemType adaMul, bool unitGainMomentum)
{
    const ElemType unitGainFactor = unitGainMomentum ? (1.0 - mom) : 1.0;
    CUDA_LONG idx = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride)
    {
        ElemType g = grad[idx];
        ElemType adaSqr = adaWeight * smoothAda[idx] + (1.0f - adaWeight) * g * g;
        smoothAda[idx] = adaSqr;
        ElemType w;
        if (sizeof(ElemType) == sizeof(double))
        {
            w = adaMul * rsqrt(adaSqr + 1e-8);
        }
        else
        {
            w = adaMul * rsqrtf(adaSqr + 1e-8);
        }

        g = mom * smoothMom[idx] + unitGainFactor * g;
        smoothMom[idx] = g;
        g = lr*g*w;
        val[idx] -= g;
    }
}

template <class ElemType>
__global__ void _adam4BlockSparseCol(CUDA_LONG size,
    ElemType* grad_bsc, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len,
    ElemType* smoothAda, ElemType* smoothMom, ElemType* val,
    ElemType lr, ElemType mom, ElemType adaWeight, ElemType adaMul, bool unitGainMomentum)
{
    const ElemType unitGainFactor = unitGainMomentum ? (1.0 - mom) : 1.0;
    CUDA_LONG idx = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride)
    {
        ElemType g = _getvalue4BlockSparseCol(grad_bsc, colOrRow2blockId, len, idx);
        ElemType adaSqr = adaWeight * smoothAda[idx] + (1.0f - adaWeight) * g * g;
        smoothAda[idx] = adaSqr;
        ElemType w;
        if (sizeof(ElemType) == sizeof(double))
        {
            w = adaMul * rsqrt(adaSqr + 1e-8);
        }
        else
        {
            w = adaMul * rsqrtf(adaSqr + 1e-8);
        }

        g = mom * smoothMom[idx] + unitGainFactor * g;
        smoothMom[idx] = g;
        g = lr*g*w;
        val[idx] -= g;
    }
}

template <class ElemType>
__global__ void _adadelta(CUDA_LONG size, ElemType* grad, ElemType* smoothAda, ElemType* smoothX2, ElemType* val,
    ElemType rho, ElemType epsilon)
{
    CUDA_LONG idx = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride)
    {
        ElemType g = grad[idx];
        ElemType adaSqr = rho * smoothAda[idx] + (1.0f - rho) * g * g;
        smoothAda[idx] = adaSqr;
        ElemType x2 = smoothX2[idx];
        ElemType deltaX;
        if (sizeof(ElemType) == sizeof(double))
        {
            deltaX = -sqrt(x2 + epsilon) * rsqrt(adaSqr + epsilon) * g;
        }
        else
        {
            deltaX = -sqrtf(x2 + epsilon) * rsqrtf(adaSqr + epsilon) * g;
        }

        smoothX2[idx] = rho * smoothX2[idx] + (1.0f - rho) * deltaX * deltaX;
        val[idx] += deltaX;
    }
}

template <class ElemType>
__global__ void _adadelta4BlockSparseCol(CUDA_LONG size,
    ElemType* grad_bsc, const GPUSPARSE_INDEX_TYPE* colOrRow2blockId, const size_t len,
    ElemType* smoothAda, ElemType* smoothX2, ElemType* val,
    ElemType rho, ElemType epsilon)
{
    CUDA_LONG idx = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_LONG stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride)
    {
        ElemType g = _getvalue4BlockSparseCol(grad_bsc, colOrRow2blockId, len, idx);
        ElemType adaSqr = rho * smoothAda[idx] + (1.0f - rho) * g * g;
        smoothAda[idx] = adaSqr;
        ElemType x2 = smoothX2[idx];
        ElemType deltaX;
        if (sizeof(ElemType) == sizeof(double))
        {
            deltaX = -sqrt(x2 + epsilon) * rsqrt(adaSqr + epsilon) * g;
        }
        else
        {
            deltaX = -sqrtf(x2 + epsilon) * rsqrtf(adaSqr + epsilon) * g;
        }

        smoothX2[idx] = rho * smoothX2[idx] + (1.0f - rho) * deltaX * deltaX;
        val[idx] += deltaX;
    }
}

// Calculate alpha in forward-backward calculation. equation (6), (7) in http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf
// GPU x dimension corresponds to utterances, y dimension corresponds to phone sequence in each utterance
// prob (input): the posterior output from the network
// alpha (output): alpha for forward-backward calculation. 
// phoneSeq (input): phone ID sequence for each utterance in this minibatch, each col is one utterance 
// phoneBound (input): phone boundary (frame index) of each phone for each utterance in this minibatch, each col is one utterance 
// uttToChanInd (input):  map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
// uttFrameNum (input): the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// uttBeginFrame(input): the positon of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
// uttPhoneNum (input): the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// numChannels (input): channel number in this minibatch
// uttNum (input): number of utterances
// t (input): time stamp to process
// maxPhoneNum (input): the max number of phones between utterances
// totalPhoneNum (input): the total number of phones of all utterances
// blankTokenId (input): id of the CTC blank token
// delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference.
//      Alpha and Beta scores outside of the delay boundary are set to zero.
//      Setting this parameter smaller will result in shorted delay between label output during decoding.
//      delayConstraint=-1 means no constraint
template<class ElemType>
__global__ void _assignAlphaScore(
    const ElemType *prob,
    ElemType *alphaScore,
    ElemType *phoneSeq,
    ElemType *phoneBound,
    const size_t *uttToChanInd,
    const size_t *uttFrameNum,
    const size_t *uttBeginFrame,
    const size_t *uttPhoneNum,
    size_t numChannels,
    const size_t uttNum,
    const size_t  t,
    const size_t maxPhoneNum, // Maximum length of utterance in this MB
    const size_t totalPhoneNum, // Total number of phones
    const size_t blankTokenId,
    const int delayConstraint)
{
    LONG64 uttId = blockDim.x * blockIdx.x + threadIdx.x;
    // Index of the label in the sequence
    LONG64 phoneSeqId = blockDim.y * blockIdx.y + threadIdx.y;

    // Number of phones and frames in this utterance
    LONG64 phoneNum = uttPhoneNum[uttId]; 
    LONG64 frameNum = uttFrameNum[uttId];

    if (uttId >= uttNum || phoneSeqId >= phoneNum - 1 || t >= frameNum || phoneSeqId == 0) return;

    // Current and previous phone indices in phoneSeq matrix
    LONG64 labelid = uttId*maxPhoneNum + phoneSeqId;
    LONG64 labelid_2 = labelid - 2;

    // Actual current phone label
    LONG64 phoneId = (LONG64)(phoneSeq[labelid]);

    // Index of the current frame in minibatch
    LONG64 timeId = (t + uttBeginFrame[uttId])*numChannels + uttToChanInd[uttId];

    // Index of probability of observing phoneId at frame timeId
    LONG64 probId = timeId*totalPhoneNum + phoneId;

    LONG64 alphaId = maxPhoneNum* timeId + phoneSeqId; // alpha_t(s)
    // Previous time frame
    LONG64 timeId_1 = timeId - numChannels; // Index corresponding to (t-1)
    LONG64 alphaId_0 = maxPhoneNum* timeId_1 + phoneSeqId; // alpha_{t-1}(s)
    LONG64 alphaId_1 = alphaId_0 - 1; // alpha_{t-1}(s-1)
    LONG64 alphaId_2 = alphaId_0 - 2; // alpha_{t-1}(s-2)

    if (t == 0)
    {
        // Initialize recursion
        if (phoneSeqId == 1 || phoneSeqId == 2)
        {
            alphaScore[alphaId] = prob[probId];
        }
    }
    else
    {
        if (phoneSeqId >= 1)
        {
            ElemType x = LZERO;

            ElemType ascore;
            if (phoneSeqId > 2)
            {
                // if current label is not blank and not equal prev non-blank label
                if ((LONG64)(phoneSeq[labelid]) != blankTokenId && phoneId != (LONG64)(phoneSeq[labelid_2]))
                {
                    x = logaddk(x, alphaScore[alphaId_2]);
                }
            }

            if (phoneSeqId > 1)
            {
                x = logaddk(x, alphaScore[alphaId_1]);
            }

            x = logaddk(x, alphaScore[alphaId_0]);

            if (phoneId != SIZE_MAX)
                ascore = prob[probId]; // Probability of observing given label at given time
            else
                ascore = 0;
            alphaScore[alphaId] = (ElemType)x + ascore;
            if (delayConstraint != -1)
            {
                LONG64 labelid_r = labelid + 2;
                LONG64 phoneBoundId_r = (LONG64)(phoneBound[labelid_r]);
                if (phoneId == blankTokenId)
                {
                    // only constraint right side
                    if (t > phoneBoundId_r + delayConstraint - 1)
                        alphaScore[alphaId] = LZERO;
                }
                else if (phoneId != blankTokenId)
                {
                    if (t > phoneBoundId_r + delayConstraint)
                        alphaScore[alphaId] = LZERO;
                }
            }
        }
    }
}

// Calculate beta in forward-backward calculation, equation (10), (11) in http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf 
// See _assignAlphaScore for the explanation of parameters
template<class ElemType>
__global__ void _assignBetaScore(
    const ElemType *prob,
    ElemType *betaScore,
    ElemType *phoneSeq,
    ElemType *phoneBound,
    const size_t *uttToChanInd,
    const size_t *uttFrameNum,
    const size_t *uttBeginFrame,
    const size_t *uttPhoneNum,
    const size_t numChannels,
    const size_t uttNum,
    const size_t  t,
    const size_t maxPhoneNum,
    const size_t totalPhoneNum,
    const size_t blankTokenId,
    const int delayConstraint)
{
    LONG64 uttId = blockDim.x * blockIdx.x + threadIdx.x;
    // Index of the label in the sequence
    LONG64 phoneSeqId = blockDim.y * blockIdx.y + threadIdx.y;
    LONG64 phoneNum = uttPhoneNum[uttId];
    LONG64 frameNum = uttFrameNum[uttId];

    if (uttId >= uttNum || phoneSeqId >= phoneNum - 1 || t >= frameNum || phoneSeqId == 0) return;

    LONG64 labelid = uttId*maxPhoneNum + phoneSeqId;
    LONG64 labelid_2 = labelid + 2;
    LONG64 phoneId = (LONG64)(phoneSeq[labelid]);
    LONG64 timeId = (t + uttBeginFrame[uttId])*numChannels + uttToChanInd[uttId];
    LONG64 probId = timeId*totalPhoneNum + phoneId;
    LONG64 betaid = maxPhoneNum* timeId + phoneSeqId;
    LONG64 timeId_1 = timeId + numChannels;
    LONG64 betaid_0 = maxPhoneNum* timeId_1 + phoneSeqId;
    LONG64 betaid_1 = betaid_0 + 1;
    LONG64 betaid_2 = betaid_0 + 2;

    if (t == frameNum - 1)
    {
        if (phoneSeqId == phoneNum - 3 || phoneSeqId == phoneNum - 2)
        {
            betaScore[betaid] = prob[probId];
        }
    }
    else
    {
        if (phoneSeqId >= 1)
        {
            ElemType x = LZERO;
            ElemType ascore;
            if (phoneSeqId < phoneNum - 3)
            {
                if (phoneSeq[labelid] != blankTokenId && phoneId != phoneSeq[labelid_2])
                {
                    x = logaddk(x, betaScore[betaid_2]);
                }
            }

            if (phoneSeqId < phoneNum - 2)
            {
                x = logaddk(x, betaScore[betaid_1]);
            }

            x = logaddk(x, betaScore[betaid_0]);

            if (phoneId != SIZE_MAX)
                ascore = prob[probId];
            else
                ascore = 0;
            betaScore[betaid] = (ElemType)x + ascore;
            if (delayConstraint != -1)
            {
                LONG64 phoneBoundId_r = (LONG64)(phoneBound[labelid_2]);
                if (phoneId == blankTokenId)
                {
                    if (t > phoneBoundId_r + delayConstraint - 1)
                        betaScore[betaid] = LZERO;
                }
                else if (phoneId != blankTokenId)
                {
                    if (t > phoneBoundId_r + delayConstraint)
                        betaScore[betaid] = LZERO;
                }
            }
        }
    }
}

// Calculate derivative, equation (15) in http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf
// See _assignAlphaScore for the explanation of parameters
template<class ElemType>
__global__ void _assignCTCScore(
    ElemType *CTCscore,
    ElemType *prob,
    ElemType *alphaScore,
    ElemType *betaScore,
    ElemType *phoneSeq,
    const size_t uttNum,
    const size_t *uttToChanInd,
    const size_t *uttBeginFrame,
    const size_t *uttPhoneNum,
    const size_t *uttFrameNum,
    const long numChannels,
    const long maxPhoneNum,
    const long totalPhoneNum)
{
    LONG64 uttId = blockDim.x * blockIdx.x + threadIdx.x;
    LONG64 t = blockDim.y * blockIdx.y + threadIdx.y;

    if (uttId < uttNum && t < uttFrameNum[uttId])
    {
        LONG64 phoneNum = uttPhoneNum[uttId];
        LONG64 alphaId_0 = (uttBeginFrame[uttId] * numChannels + uttToChanInd[uttId]) * maxPhoneNum;
        LONG64 timeId = (t + uttBeginFrame[uttId])*numChannels + uttToChanInd[uttId];
        ElemType P_lx = betaScore[alphaId_0];

        for (int s = 1; s < phoneNum - 1; s++)
        {
            long phoneId = phoneSeq[uttId*maxPhoneNum + s];
            LONG64 alphaId = maxPhoneNum* timeId + s;
            LONG64 probId = timeId*totalPhoneNum + phoneId;

            if (phoneId != SIZE_MAX)
            {
                ElemType logoccu = alphaScore[alphaId] + betaScore[alphaId] - prob[probId] - (ElemType)P_lx;
                CTCscore[probId] = logaddk(CTCscore[probId], logoccu);
            }
        }

        for (int s = 0; s < totalPhoneNum; s++)
        {
            LONG64 probId = timeId*totalPhoneNum + s;
            ElemType logoccu = CTCscore[probId];
            if (logoccu < LZERO)
                CTCscore[probId] = 0.0f;
            else
                CTCscore[probId] = exp(logoccu);
        }
    }
}

// Calculate CTC score. equation (8) in http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf 
template<class ElemType>
__global__ void _assignTotalScore(ElemType *betaScore,
    ElemType *totalScore,
    const size_t uttNum,
    const size_t *uttToChanInd,
    const size_t *uttBeginFrame,
    const size_t numChannels,
    const size_t maxPhoneNum)
{
    LONG64 uttId = blockIdx.x;
    if (uttId < uttNum)
    {
        LONG64 alphaId_0 = (uttBeginFrame[uttId] * numChannels + uttToChanInd[uttId]) * maxPhoneNum;

        betaScore[alphaId_0] = logaddk(betaScore[alphaId_0 + 1], betaScore[alphaId_0 + 2]);
        totalScore[uttId] = betaScore[alphaId_0];
    }
}

template<class ElemType>
__global__ void _assignOneHot(ElemType *indices,
                                  ElemType *targetBuffer,
                                  size_t num_class,
                                  size_t num_item,
                                  size_t num_element)
{
    const CUDA_LONG index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_element)
    {
        if (indices[index] >= 0 && indices[index] < num_class)
        {
            size_t block_id = index / num_item;
            size_t item_id = index % num_item;
            targetBuffer[block_id * num_class * num_item + item_id + num_item * (size_t)indices[index]] = 1;
        }
    }
}

template<class ElemType>
__global__ void _assignOneHotAsSparse(ElemType *indices,
                                      GPUSPARSE_INDEX_TYPE *secondaryIndices,
                                      GPUSPARSE_INDEX_TYPE *majorIndices,
                                      ElemType *targetBuffer,
                                      size_t num_class,
                                      int num_item,
                                      size_t num_elements)
{
    const CUDA_LONG index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements)
    {
        int block_id = index / num_item;
        int item_id = index % num_item;
        // for invalid indices, theorically they should not belong to nz elements.
        // but if we scan the indices to count the valid indices number,
        // it will be difficult for parallel calculation, especially on GPU.
        // here we chose to keep those elements in nz element list, but with value 0 at row 0
        if (indices[index] >= 0 && indices[index] < num_class)
        {
            targetBuffer[index] = 1;
            majorIndices[index] = ((int)indices[index] * num_item) + item_id;
        }
        else
        {
            targetBuffer[index] = 0;
            majorIndices[index] = item_id;
        }

        if (item_id == 0)
            secondaryIndices[block_id + 1] = num_item * (block_id + 1);

        if (index == 0)
            secondaryIndices[0] = 0;
    }
}

}}}

#endif // !CPUONLY
