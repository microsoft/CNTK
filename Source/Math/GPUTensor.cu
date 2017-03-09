//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Basics.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUTensor.h"
#include "GPUMatrix.h"
#include "GPUMatrixCUDAKernels.cuh"
#include "CommonMatrix.h"
#define TENSOR_OPS_DECL __device__ __host__
#include "TensorOps.h"
#include "fast_divmod.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <assert.h>
#include<limits.h>

// use fast divisor
#define USE_FAST_DIVMOD

#ifndef let
#define let const auto
#endif

#pragma comment(lib, "cudart.lib") // instruct linker to reference these libs
#pragma comment(lib, "cublas.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

#ifdef _WIN32
// thread local storage to access the current stream, initalize to default stream
__declspec(thread)
#endif
extern cudaStream_t t_stream;

namespace Microsoft { namespace MSR { namespace CNTK {

// =======================================================================
// TensorView support
// =======================================================================

// TensorView computes element-wise tensor operations.
//  - supports general strides
//  - input broadcasting is supported by stride=0
//  - the operation is denoted by an opCode
//  - reduction is supported, including summation, min, max (dual to broadcasting when computing gradients)
//  - reduction operation is given by an opCode: opSum, opMin, opMax and opLogSum.
//
// This library makes extensive use of templates and macros.
// Specifically, templates are used recursively to recurse over tensor dimensions.
// For example, a tensor op of rank K is computed by looping over the last dimension
// and then calling the same function template recursively with K-1.
// Template specializations exist in order to:
//  - terminate recursion
//  - optimize for thread-parallel reduction where elements are consecutive in memory
//
// The general algorithm is very straight forward:
//
//     for all output dimensions [###]:                                 // TensorOp()
//         output[###] *= beta
//         for all reduction dimensions [***]:                          // TensorOpWithReduction()
//             output[###] += op(input1[###,***], input1[###,***], ...) * alpha
//
// Indices and dimensions used throughout this code:
//  - N = ariness; number of arguments *including output* (binary op: N=3)
//  - K = rank of output elements, regularOpDims.size(). K=0 means scalar.
//  - k = -1..K-1 = recursion index
//  - M = reduction rank, reducingOpDims.size(). M=0 means no reduction.
//  - m = -1..M-1 = recursion index
//
// Other frequently used variable names:
//  - alpha, beta: BLAS-style weights: outVal = beta * outVal + alpha * f(inVals)
//                 where beta=0 is an assignment (0 * outVal := 0, even e.g. if outVal = NaN)
//  - pointers[N]:          pointer to first element, for each argument
//  - regularOpDims[K]:     tensor dimensions of output elements to produce
//  - regularStrides[N,K]:  strides; multiply index[k] with strides[n,k] to get element offset for this dimension
//                          Broadcasting of inputs is implemented by a stride being 0.
//  - reducingOpDims[M]:    tensor dimensions of input elements to reduce over
//  - reducingStrides[N,M]: strides for input reduction. Always 0 for output argument.
//
// This code uses two custom structs, FixedArray<> and FixedMatrix<>, which
// are templated equivalents to vector<> and vector<vector<>> for CUDA code.

// -----------------------------------------------------------------------
// simple fixed-size arrays for passing dimension information by value
// since CUDA can't just take our std::array and std::vector
// -----------------------------------------------------------------------

template <typename T, size_t N>
struct FixedArray
{
    T m_data[N];
    __device__ __host__ size_t size() const
    {
        return N;
    }
    __device__ __host__ T& operator[](size_t n)
    {
        return m_data[n];
    }
    __device__ __host__ T operator[](size_t n) const
    {
        return m_data[n];
    }
    template <class VEC>
    FixedArray(const VEC& data) // construct from CPU-side STL array or vector
    {
        assert(data.size() == N);
        for (size_t n = 0; n < N; n++)
        {
            m_data[n] = (T) data[n];
            if (m_data[n] != data[n]) // overflow check
                InvalidArgument("FixedArray: Dimensions out of range, too few bits.");
        }
    }
};
template <typename T> // specialized version for 0 elements
struct FixedArray<T, 0>
{
    __device__ __host__ size_t size() const
    {
        return 0;
    }
    template <class VEC>
    FixedArray(const VEC& data)
    {
        assert(data.size() == 0);
        UNUSED(data);
    }
    FixedArray()
    {
    }
};

template <typename T, size_t N, size_t K> // N = which input/output; K = index depth
struct FixedMatrix
{
    T m_data[N][K];
    __device__ __host__ size_t getNumRows() const
    {
        return N;
    }
    __device__ __host__ size_t getNumCols() const
    {
        return K;
    }
    __device__ __host__ T& operator()(size_t n, size_t k)
    {
        return m_data[n][k];
    }
    __device__ __host__ T operator()(size_t n, size_t k) const
    {
        return m_data[n][k];
    }
    template <typename U>
    FixedMatrix(const array<SmallVector<U>, N>& data) // construct from CPU-side array of vectors
    {
        assert(data.size() == N);
        for (size_t n = 0; n < N; n++)
        {
            assert(data[n].size() == K);
            for (size_t k = 0; k < K; k++)
            {
                m_data[n][k] = (T) data[n][k];
                if (m_data[n][k] != data[n][k]) // overflow check
                    InvalidArgument("FixedArray: Dimensions out of range, too few bits.");
            }
        }
    }
};
template <typename T, size_t N> // specialized version for 0 elements
struct FixedMatrix<T, N, 0>
{
    __device__ __host__ size_t getNumRows() const
    {
        return N;
    }
    __device__ __host__ size_t getNumCols() const
    {
        return 0;
    }
    template <typename U>
    FixedMatrix(const array<SmallVector<U>, N>& data)
    {
        assert(data.size() == N);
        for (size_t n = 0; n < N; n++)
            assert(data[n].size() == 0);
        UNUSED(data);
    }
    FixedMatrix()
    {
    }
};

// -----------------------------------------------------------------------
// function to actually compute a function of (N-1) inputs based on the opcode
// -----------------------------------------------------------------------

template <class ElemType>
struct TensorOps
{
    static __device__ ElemType Compute(const FixedArray<ElemType*, 1>& pointers, ElementWiseOperator op)
    {
#define CaseNullaryTensorOp(oper)       \
    case ElementWiseOperator::op##oper: \
        return Op##oper<ElemType>()
        switch (op)
        {
            ForAllNullaryOps(CaseNullaryTensorOp);
        default:
            return OpConstOne<ElemType>(); // (failure--we only have one nullary op, so use the same, maybe it will eliminate the switch altogether)
        }
    }
    static __device__ ElemType Compute(const FixedArray<ElemType*, 2>& pointers, ElementWiseOperator op)
    {
        ElemType a = *(pointers[0]);
#define CaseUnaryTensorOp(oper)         \
    case ElementWiseOperator::op##oper: \
        return Op##oper(a)
        switch (op)
        {
            ForAllUnaryOps(CaseUnaryTensorOp);
        default:
            return 0; // (failure)
        }
    }
    static __device__ ElemType Compute(const FixedArray<ElemType*, 3>& pointers, ElementWiseOperator op)
    {
        // const ElemType & a = *(pointers[0]);    // const & for opIndex--costs quite some code bloat
        ElemType a = *(pointers[0]);
        ElemType b = *(pointers[1]);
#define CaseBinaryTensorOp(oper)        \
    case ElementWiseOperator::op##oper: \
        return Op##oper(a, b)
        switch (op)
        {
            ForAllBinaryOps(CaseBinaryTensorOp); // note: this costs about 6% compared to having only a single case
        default:
            return 0; // (failure)
        }
    }
    static __device__ ElemType Compute(const FixedArray<ElemType*, 4>& pointers, ElementWiseOperator op)
    {
#define CaseTernaryTensorOp(oper)       \
    case ElementWiseOperator::op##oper: \
        return Op##oper(*(pointers[0]), *(pointers[1]), *(pointers[2])) // reading each time, which saves mem accesses for OpCond
        switch (op)
        {
            ForAllTernaryOps(CaseTernaryTensorOp);
        default:
            return 0; // (failure)
        }
    }
};

//----------------------------------------------------------------------------
// For reductions we need the neutral elements of the corresponding binary ops
//----------------------------------------------------------------------------

// NeutralValue seems to be dead code
template <typename ElemType> __device__ ElemType NeutralValue(ElementWiseOperator op)
{
    return 0; // error, only the explicit instantiations below should be used.
};

template<> __device__ float NeutralValue<float>(ElementWiseOperator op)
{
    switch (op)
    {
    case ElementWiseOperator::opSum:                return 0;
    case ElementWiseOperator::opLogSum:             return -INFINITY;
    case ElementWiseOperator::opMin:                return FLT_MAX;
    case ElementWiseOperator::opMax:                return -FLT_MAX;
    case ElementWiseOperator::opElementwiseProduct: return 1.0f;
    default:                                        return 0; // error
    }
};

template<> __device__ double NeutralValue<double>(ElementWiseOperator op)
{
    switch (op)
    {
    case ElementWiseOperator::opSum:                return 0;
    case ElementWiseOperator::opLogSum:             return -INFINITY;
    case ElementWiseOperator::opMin:                return DBL_MAX;
    case ElementWiseOperator::opMax:                return -DBL_MAX;
    case ElementWiseOperator::opElementwiseProduct: return 1.0;
    default:                                        return 0; // error
    }
};


// ----------------------------------------------------------------------------
// Function to update an aggregate value for the specifed reduction operation
// ----------------------------------------------------------------------------

template<typename ReductionType, class ElemType> __device__ void UpdateAggregate(ReductionType& aggregate, ElemType val, ElementWiseOperator reductionOp)
{
    switch (reductionOp)
    {
    case ElementWiseOperator::opSum:
        aggregate += val;
        break;
    case ElementWiseOperator::opLogSum:
        aggregate = OpLogSum(aggregate, val);
        break;
    case ElementWiseOperator::opMin:
        if (val < aggregate)
            aggregate = val;
        break;
    case ElementWiseOperator::opMax:
        if (val > aggregate)
            aggregate = val;
        break;
    case ElementWiseOperator::opElementwiseProduct:
        aggregate *= val;
        break;
    }
};


// -----------------------------------------------------------------------
// function to compute the value for a given output location (including reduction)
// -----------------------------------------------------------------------

//#define ReduceElemType double
#define ReduceElemType ElemType // (note: we could use 'double' here, but that would cause problems with CUDA cards that don't support double)

template <class ElemType, C_size_t N, C_int M, C_int m>
struct TensorOpReduce
{
    // this version for m >= 0
    static __device__ ElemType Compute(FixedArray<ElemType*, N> pointers, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                       const FixedArray<C_unsigned_int, M>& reducingOpDims, const FixedMatrix<C_int, N, M>& reducingStrides)
    {
        // start with index 0
        // We may use 'double' since we are memory-bound anyway.
        ReduceElemType aggregate = TensorOpReduce<ElemType, N, M, m - 1>::Compute(pointers, op, reductionOp, reducingOpDims, reducingStrides);
        // apply this index to the pointers
        C_size_t dim = reducingOpDims[m];
        for (C_size_t k = 1 /*done with k=0 already*/; k < dim; k++)
        {
            // bump the pointers
            #pragma unroll
            for (C_size_t i = 0; i < N - 1; i++) { // N-1 because output is not used here
                pointers[i] += reducingStrides(i, (C_size_t) m);
            }
            ElemType val = TensorOpReduce<ElemType, N, M, m - 1>::Compute(pointers, op, reductionOp, reducingOpDims, reducingStrides);
            UpdateAggregate<ReduceElemType, ElemType>(aggregate, val, reductionOp);
        }
        return (ElemType) aggregate;
    }
};

// this one terminates the template recursion over reduction dimensions
// The pointers are pointing to the input element.
template <class ElemType, C_size_t N, C_int M>
struct TensorOpReduce<ElemType, N, M, /*m=*/-1>
{
    // this version for m = -1
    // the pointers are pointing to the right location(s) to take the operation over
    static __device__ ElemType Compute(FixedArray<ElemType*, N> pointers, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                       const FixedArray<C_unsigned_int, M>& /*reducingOpDims*/, const FixedMatrix<C_int, N, M>& /*reducingStrides*/)
    {
        return TensorOps<ElemType>::Compute(pointers, op); // finally computing something!
    }
};

// -----------------------------------------------------------------------
// function to compute one constituent of the value for a given output location
// (reduction is not done here, but by calling into here multiple times)
// -----------------------------------------------------------------------

template <class ElemType, C_size_t N, C_int M, C_int m>
struct TensorOpParallelReduce
{
    // this version for m >= 0
    static __device__ ElemType Compute(CUDA_LONG id, FixedArray<ElemType*, N> pointers, ElementWiseOperator op,
                                       const FixedArray<C_unsigned_int, M>& reducingOpDims, const FixedMatrix<C_int, N, M>& reducingStrides,
                                       FixedArray<fast_divmod, M> reducingOpDimDivmod)
    {
        // map id (location on grid) to index[k]
        C_size_t stride = 1; // compute the stride. This seems expensive, but since we we only currently support M <= 2, this is just compile-time selection between 1 and reducingOpDims[0].
        #pragma unroll
        for (int i = 0; i < m; i++) {
            stride *= reducingOpDims[(C_size_t) i];
        }

        C_size_t index;
#ifndef USE_FAST_DIVMOD
        index = id / stride; // this dimension. For m=0, the stride is 1 and hence the division will be removed at compile time.
        // id = id % stride;             // remaining dimensions inside this. For m=0 this value is ignored and hence not even computed.
        id = id - stride*index;             // remaining dimensions inside this. For m=0 this value is ignored and hence not even computed.
#else
        if (m == 0) {
          index = id;
          id = 0;
        } else {
          reducingOpDimDivmod[m].divmod(id, index, id);
        }
#endif
        // apply this index to the pointers
        #pragma unroll
        for (C_size_t i = 0; i < N - 1; i++) {
            pointers[i] += index * reducingStrides(i, (C_size_t) m); // now this dimension is taken care of
        }
        return TensorOpParallelReduce<ElemType, N, M, m - 1>::Compute(id, pointers, op, reducingOpDims, reducingStrides, reducingOpDimDivmod);
    }
};

// this one terminates the template recursion over reduction dimensions
// The pointers are pointing to the input element.
template <class ElemType, C_size_t N, C_int M>
struct TensorOpParallelReduce<ElemType, N, M, /*m=*/-1>
{
    // this version for m = -1
    // the pointers are pointing to the right location(s) to take the operation over
    static __device__ ElemType Compute(CUDA_LONG /*id*/, FixedArray<ElemType*, N> pointers, ElementWiseOperator op,
                                       const FixedArray<C_unsigned_int, M>& /*reducingOpDims*/, const FixedMatrix<C_int, N, M>& /*reducingStrides*/,
                                       FixedArray<fast_divmod, M> reducingOpDimDivmod)
    {
        return TensorOps<ElemType>::Compute(pointers, op); // finally computing something!
    }
};

// -----------------------------------------------------------------------
// perform loop over regular index k for N-nary operations (N counting the output)
// -----------------------------------------------------------------------

// The 'pointers' only refer to a single element, so we will bump them in-place to perform indexing.
template <class ElemType, C_size_t N, C_int M, C_int K, bool parallelReduce, C_int k>
struct TensorOpElement
{
    // template-recursive version loops over indices
    static __device__ void Compute(CUDA_LONG id, ElemType beta, FixedArray<ElemType*, N>& pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const FixedArray<C_unsigned_int, K>& regularOpStrides, const FixedMatrix<C_int, N, K>& regularStrides,
                                   const FixedArray<C_unsigned_int, M>& reducingOpDims, const FixedMatrix<C_int, N, M>& reducingStrides,
                                   CUDA_LONG reductionBegin, CUDA_LONG reductionChunkSize,
                                   FixedArray<fast_divmod, K> regularOpStrideDivmod, FixedArray<fast_divmod, M> reducingOpDimDivmod)
    {
        // map id (location on grid) to index[k]
#ifndef USE_FAST_DIVMOD
        C_size_t stride = regularOpStrides[(C_size_t) k];
        C_size_t index = id / stride; // this dimension
        // id = id % stride;             // remaining dimensions inside this
        id = id - stride*index;             // remaining dimensions inside this
#else
        C_size_t index;
        regularOpStrideDivmod[k].divmod(id, index, id);
#endif
        // apply this index to the pointers
        #pragma unroll
        for (C_size_t i = 0; i < N; i++) {
            pointers[i] += index * regularStrides(i, (C_size_t) k); // now this dimension is taken care of
        }
        // process the previous index
        TensorOpElement<ElemType, N, M, K, parallelReduce, k - 1>::Compute(id, beta, pointers, alpha, op, reductionOp, regularOpStrides, regularStrides, reducingOpDims, reducingStrides, reductionBegin, reductionChunkSize,
                                                                           regularOpStrideDivmod, reducingOpDimDivmod);
    }
};

// specialization for k=0 where op stride is guaranteed to be 1
template <class ElemType, C_size_t N, C_int M, C_int K, bool parallelReduce>
struct TensorOpElement<ElemType, N, M, K, parallelReduce, /*k=*/0>
{
    // template-recursive version loops over indices
    static __device__ void Compute(CUDA_LONG id, ElemType beta, FixedArray<ElemType*, N>& pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const FixedArray<C_unsigned_int, K>& regularOpStrides, const FixedMatrix<C_int, N, K>& regularStrides,
                                   const FixedArray<C_unsigned_int, M>& reducingOpDims, const FixedMatrix<C_int, N, M>& reducingStrides,
                                   CUDA_LONG reductionBegin, CUDA_LONG reductionChunkSize,
                                   FixedArray<fast_divmod, K> regularOpStrideDivmod, FixedArray<fast_divmod, M> reducingOpDimDivmod)
    {
        // map id (location on grid) to index[k]
        C_size_t index = id; // this dimension
        // apply this index to the pointers
        #pragma unroll
        for (C_size_t i = 0; i < N; i++) {
            pointers[i] += index * regularStrides(i, 0); // now this dimension is taken care of
        }
        // process the previous index
        TensorOpElement<ElemType, N, M, K, parallelReduce, -1>::Compute(/*id*/ 0, beta, pointers, alpha, op, reductionOp, regularOpStrides, regularStrides, reducingOpDims, reducingStrides, reductionBegin, reductionChunkSize,
                                                                        regularOpStrideDivmod, reducingOpDimDivmod);
    }
};

// specialization for k = -1 terminates the template recursion, and computes reductions in a for loop
template <class ElemType, C_size_t N, C_int M, C_int K>
struct TensorOpElement<ElemType, N, M, K, /*parallelReduce=*/false, /*k=*/-1>
{
    // template-recursion-teminating version computes the actual value for this output location
    // now the output pointers point to the right element (input pointers may still iterate for reduction)
    static __device__ void Compute(CUDA_LONG /*id*/, ElemType beta, FixedArray<ElemType*, N>& pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const FixedArray<C_unsigned_int, K>& /*regularOpStrides*/, const FixedMatrix<C_int, N, K>& /*regularStrides*/,
                                   const FixedArray<C_unsigned_int, M>& reducingOpDims, const FixedMatrix<C_int, N, M>& reducingStrides, CUDA_LONG /*reductionBegin*/, CUDA_LONG /*reductionChunkSize*/,
                                   FixedArray<fast_divmod, K> regularOpStrideDivmod, FixedArray<fast_divmod, M> reducingOpDimDivmod)
    {
        // compute the operation for this output coordinate
        // This may still involve a reduction over inverse-broadcasting dimensions.
        ElemType val = TensorOpReduce<ElemType, N, M, M - 1>::Compute(pointers, op, reductionOp, reducingOpDims, reducingStrides);
        // scale
        val *= alpha;
        // combine with previous value in target matrix, then write it out
        if (N < 4 || val != 0 || beta != 1) // (skip memory access if not needed) (N<4: skip this test)
        {
            auto* pout = pointers[pointers.size() - 1];
            if (beta != 0) // (skip memory access if not needed, and allow for ignoring NaNs)
                val += beta * *pout;
            // save
            *pout = val;
        }
    }
};

#undef ALLOW_ATOMIC_REDUCTION // undefine to disable use of atomicAdd() below, for testing it

// specialization for k = -1 terminates the template recursion, and computes reductions in parallel
template <class ElemType, C_size_t N, C_int M, C_int K>
struct TensorOpElement<ElemType, N, M, K, /*parallelReduce=*/true, /*k=*/-1>
{
    // template-recursion-teminating version computes the actual value for this output location
    // now the output pointers point to the right element (input pointers may still iterate for reduction)
    static __device__ void Compute(CUDA_LONG /*id*/, ElemType beta, FixedArray<ElemType*, N>& pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const FixedArray<C_unsigned_int, K>& /*regularOpStrides*/, const FixedMatrix<C_int, N, K>& /*regularStrides*/,
                                   const FixedArray<C_unsigned_int, M>& reducingOpDims, const FixedMatrix<C_int, N, M>& reducingStrides, CUDA_LONG reductionBegin, CUDA_LONG reductionChunkSize,
                                   FixedArray<fast_divmod, K> regularOpStrideDivmod, FixedArray<fast_divmod, M> reducingOpDimDivmod)
    {
        CUDA_LONG reductionBlock = blockIdx.z; // reduction-block index  --larger reductions are split into blocks
        CUDA_LONG tid = threadIdx.x;           // thread index
        CUDA_LONG tids = blockDim.x;           // out of how many threads  --note: last block is partial

        // determine our range  --this is a single int mul, we can stomach it (we could alternatively pass in yet another parameter)
        CUDA_LONG reductionDim = (CUDA_LONG) reducingOpDims[0];
        for (C_size_t i = 1; i < reducingOpDims.size(); i++)
            reductionDim *= reducingOpDims[i];

        // determine the redId range that we operate on
        // Each thread takes a stride tid + (multiples of tids) within this range.
        reductionBegin += reductionChunkSize * reductionBlock;
        CUDA_LONG reductionEnd = min(reductionBegin + reductionChunkSize, reductionDim);

        // compute the operation for this input coordinate
        ReduceElemType aggregate = NeutralValue<ReduceElemType>(reductionOp);

        for (CUDA_LONG redId = reductionBegin + tid; redId < reductionEnd; redId += tids)
        {
            auto val = TensorOpParallelReduce<ElemType, N, M, M - 1>::Compute(redId, pointers, op, reducingOpDims, reducingStrides, reducingOpDimDivmod);
            UpdateAggregate<ReduceElemType, ElemType>(aggregate, val, reductionOp);
        }

        // reduce    --cf https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
        __shared__ ReduceElemType volatile accumulators[GridDim::maxThreadsPerBlock /*tids*/];
        accumulators[tid] = aggregate;
        __syncthreads();
        static_assert(GridDim::maxThreadsPerBlock <= 1024, "GridDim::maxThreadsPerBlock too large, need to add manually unrolled steps");
        for (CUDA_LONG i = 512; i; i >>= 1)
        {
            if (tid < i && tid + i < tids)
                UpdateAggregate<volatile ReduceElemType, volatile ReduceElemType>(accumulators[tid], accumulators[tid + i], reductionOp);

            if (0 + i < tids)
                __syncthreads(); // sync if condition true for at least one thread
            // TODO: use volatile* and then we can skip the __syncthreads() for the last 32 values. See Amit's allreduce() function implementation in MatrixQuantizer_kernel.cu.
        }

        // now set final value to output coordinate
        if (tid == 0)
        {
            ElemType val = (ElemType) accumulators[0];
            // scale
            val *= alpha;
            // combine with previous value in target matrix, then write it out
            if (N < 4 || val != 0 || beta != 1) // (skip memory access if not needed) (N<4: skip this test)
            {
                auto* pout = pointers[pointers.size() - 1];
#ifdef ALLOW_ATOMIC_REDUCTION
                CUDA_LONG reductionBlocks = gridDim.z; // number of reduction blocks. If >1 we need atomicAdd
                if (reductionBlocks > 1) // multiple blocks: need to use atomicAdd()
                {
                    // in this case, outer calling code must pass beta = 1
                    atomicAdd(pout, val);
                }
                else
#endif
                {
                    if (beta != 0)
                        val += beta * *pout;
                    // save
                    *pout = val;
                }
            }
        }
    }
};

// -----------------------------------------------------------------------
// kernel and launch  --no reduction
// -----------------------------------------------------------------------

// launch tensor op with CUDA
template <class ElemType, C_size_t N, C_int M, C_int K>
__global__ void _launchTensorOp(ElemType beta, FixedArray<ElemType*, N> pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                FixedArray<C_unsigned_int, K> regularOpStrides, FixedMatrix<C_int, N, K> regularStrides, CUDA_LONG numElements,
                                FixedArray<C_unsigned_int, M> reducingOpDims, FixedMatrix<C_int, N, M> reducingStrides,
                                FixedArray<fast_divmod, K> regularOpStrideDivmod, FixedArray<fast_divmod, M> reducingOpDimDivmod)
{
    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id < numElements) // note: there are no __syncthread() calls inside
        TensorOpElement<ElemType, N, M, K, false, K - 1>::Compute(id, beta, pointers, alpha, op, reductionOp, regularOpStrides, regularStrides, reducingOpDims, reducingStrides, 0, 0,
                                                                  regularOpStrideDivmod, reducingOpDimDivmod);
}

template <class ElemType, C_size_t N, C_int K>
static void LaunchTensorOp(ElemType beta, array<ElemType*, N> pointerVector, ElemType alpha, ElementWiseOperator op,
                           const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrideVectors)
{
    // copy all parameters to CUDA-compatible data structures
    FixedArray<ElemType*, N> pointers(pointerVector);
    SmallVector<C_size_t> regularOpStrideVector; // kernel needs the strides for converting thread index back to multi-dimensional tensor index
    C_size_t numElements = 1;
    // input divisors
    SmallVector<fast_divmod> regularOpStrideDivmodVector;
    for (C_size_t k = 0; k < regularOpDims.size(); k++)
    {
        regularOpStrideVector.push_back(numElements);
        // create fast division objects
        regularOpStrideDivmodVector.push_back(fast_divmod(numElements));
        numElements *= (C_size_t) regularOpDims[k];
    }

    SmallVector<fast_divmod> reducingOpDimDivmodVector;

    FixedArray<C_unsigned_int, K> regularOpStrides(regularOpStrideVector);
    FixedMatrix<C_int, N, K> regularStrides(regularStrideVectors);
    FixedArray<C_unsigned_int, /*M=*/0> reducingOpDims; // empty reduction dimensions
    FixedMatrix<C_int, N, /*M=*/0> reducingStrides;
    // reduced divisors
    FixedArray<fast_divmod,       K> regularOpStrideDivmod(regularOpStrideDivmodVector);
    FixedArray<fast_divmod, /*M=*/0> reducingOpDimDivmod;

    // launch the kernel
    CUDA_LONG NN = (CUDA_LONG) numElements; // linear space identifying each individual input element
    SyncGuard syncGuard;
    GridDim grid(NN);
    _launchTensorOp<ElemType, N, /*M=*/0, K> <<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream >>>(beta, pointers, alpha, op, (ElementWiseOperator)(-1) /* dummy reductionOp */, regularOpStrides, regularStrides,
                                                                                                              grid.m_N, reducingOpDims, reducingStrides,
                                                                                                              regularOpStrideDivmod, reducingOpDimDivmod);
}

// -----------------------------------------------------------------------
// kernel and launch  --with reduction
// -----------------------------------------------------------------------

template <class ElemType, C_size_t N, C_int M, C_int K>
__global__ void _launchTensorOpWithReduction(ElemType beta, FixedArray<ElemType*, N> pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                             FixedArray<C_unsigned_int, K> regularOpStrides, FixedMatrix<C_int, N, K> regularStrides, CUDA_LONG numElements,
                                             FixedArray<C_unsigned_int, M> reducingOpDims, FixedMatrix<C_int, N, M> reducingStrides,
                                             CUDA_LONG reductionBegin, CUDA_LONG reductionChunkSize,
                                             FixedArray<fast_divmod, K> regularOpStrideDivmod, FixedArray<fast_divmod, M> reducingOpDimDivmod)
{
    CUDA_LONG id = gridDim.x * blockIdx.y + blockIdx.x; // input dimensions are Y dimension of blocks in this case, so we can use thread dim for shared-memory/parallelization
#ifndef ALLOW_ATOMIC_REDUCTION
    CUDA_LONG reductionBlock = blockIdx.z;                         // reduction-block index  --larger reductions are split into blocks
    pointers[pointers.size() - 1] += numElements * reductionBlock; // the output tensor is dense (no gaps); and there is one copy for each reduction block (those get further reduced into one later)
#endif
    if (id < numElements) {                             // note: we have __syncthread() calls but only entire blocks in sync, so this is OK
        TensorOpElement<ElemType, N, M, K, true, K - 1>::Compute(id, beta, pointers, alpha, op, reductionOp, regularOpStrides, regularStrides, reducingOpDims, reducingStrides, reductionBegin, reductionChunkSize,
                                                                 regularOpStrideDivmod, reducingOpDimDivmod);
    }
}

// helper function to provide a reduction buffer
template <class ElemType>
static shared_ptr<ElemType> AllocateReductionBuffer(size_t N)
{
    ElemType* deviceBufferPtr;
    CUDA_CALL(cudaMalloc((void**)&deviceBufferPtr, sizeof(ElemType) * N));
    return shared_ptr<ElemType>(deviceBufferPtr, [](ElemType* deviceBufferPtr){ cudaFree((void*)deviceBufferPtr); });
}

template <class ElemType>
static shared_ptr<ElemType> GetReductionBuffer(size_t N)
{
    bool dontCache = false;         // (for debugging only)
    if (t_stream != 0 || dontCache) // we cache for the NULL stream but don't bother for others, since we only ever use the NULL stream currently
        return AllocateReductionBuffer<ElemType>(N);

    static shared_ptr<ElemType> reductionBuffersCache[32]; // cache of objects    --TODO: Do we have a #define the the max somewhere? Then also use it in CPUMatrix.cu GetOnesTensor()
    static size_t reductionBuffersCacheSize[_countof(reductionBuffersCache)] = { 0 };
    let deviceId = GridDim::GetCurrentDeviceId();
    if (deviceId >= _countof(reductionBuffersCache)) // index check w.r.t. our hard-coded dimensions
        return AllocateReductionBuffer<ElemType>(N); // out of bounds: don't cache
    if (!reductionBuffersCache[deviceId])
    {
        reductionBuffersCache[deviceId] = AllocateReductionBuffer<ElemType>(N);
        reductionBuffersCacheSize[deviceId] = N;
    }
    if (N > reductionBuffersCacheSize[deviceId]) // buffer size check
        LogicError("GetReductionBuffer: Must be called with the number of multiprocs, which may not change.");
    return reductionBuffersCache[deviceId];
}

// All dimensions (N-ariness, number of input dimensions K and number of reduction dimensions M) are bound to template parameters now.
template <class ElemType, C_size_t N, C_int M, C_int K>
static void LaunchTensorOpWithReduction(ElemType beta, array<ElemType*, N> pointerVector, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                        const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrideVectors,
                                        const SmallVector<size_t>& reducingOpDimVector, const array<SmallVector<ptrdiff_t>, N>& reducingStrideVectors)
{
    // copy all parameters to CUDA-compatible data structures
    FixedArray<ElemType*, N> pointers(pointerVector);
    SmallVector<C_size_t> regularOpStrideVector; // kernel needs the strides for converting thread index back to multi-dimensional tensor index
    C_size_t numElements = 1;
    // input divisors
    SmallVector<fast_divmod> regularOpStrideDivmodVector;
    for (C_size_t k = 0; k < regularOpDims.size(); k++)
    {
        regularOpStrideVector.push_back(numElements); // stride for dense representation of our output elements (if they were flattened)
        regularOpStrideDivmodVector.push_back(fast_divmod((unsigned int)numElements));
        numElements *= (C_size_t) regularOpDims[k];
    }
    // output divisors
    SmallVector<fast_divmod> reducingOpDimDivmodVector;
    C_size_t stride = 1;
    for (C_size_t k = 0; k < reducingOpDimVector.size(); ++k) {
      reducingOpDimDivmodVector.push_back(fast_divmod(stride));
      stride *= (C_size_t) reducingOpDimVector[k];
    }

    FixedArray<C_unsigned_int,    K> regularOpStrides(regularOpStrideVector);
    FixedMatrix<C_int,         N, K> regularStrides(regularStrideVectors);
    FixedArray<C_unsigned_int,    M> reducingOpDims(reducingOpDimVector);
    FixedMatrix<C_int,         N, M> reducingStrides(reducingStrideVectors);
    // reduced divisors
    FixedArray<fast_divmod,       K> regularOpStrideDivmod(regularOpStrideDivmodVector);
    FixedArray<fast_divmod,       M> reducingOpDimDivmod(reducingOpDimDivmodVector);

    // launch the kernel
    CUDA_LONG NN = (CUDA_LONG) numElements; // linear space identifying each individual output element
    SyncGuard syncGuard;

    // do some optimization for reductions
    //  - example: 30 GPU procs, warp size 32 --> 960 GPU cores
    //  - NN elements must be computed, each involving a reduction over reductionDim elements
    // Cases:
    //  - #output elements NN >= GPU cores  -->  use one proc per element, do reduction in inner loop
    //    E.g. if >=960 elements are computed, each gets its own GPU thread.
    //  - reduction dimension would benefit from multiple blocks  -->  multiple blocks work on a single output element
    //    E.g.
    //     - gradient of adding a bias: reducing to a bias, e.g. 512-dim
    //     - gradient of scalar multiplication: big elementwise product reduced to a scalar (big dot product, e.g. [1024 x 1024] = 1M elements)
    //     - softmax in seq-2-seq attention model: reduce over length of attention window (e.g. 20)
    //     - summation of criterion value: scalar reduction over a few hundred or thousand samples in the minibatch
    C_size_t reductionDim = 1; // number of elements to reduce over
    for (C_size_t k = 0; k < reducingOpDimVector.size(); k++)
        reductionDim *= (C_size_t) reducingOpDimVector[k];
    GridDim grid(NN);
    let& props = GridDim::GetDeviceProps();
    // === simple case: NN large, one thread per output element
    bool disableParallelReduction = false;                       // (for debugging)
    if (reductionDim == 1 ||                                     // no reduction
        grid.m_blocksPerGrid >= props.multiProcessorCount ||     // enough output elements to fill all multiprocs
        reductionDim * numElements <= 2 * props.warpSize ||      // trivial operation not worth the trouble (2* because the more complex one also needs 2 kernel launches)
        disableParallelReduction ||                              // (for debugging)
        reductionDim * numElements <= props.multiProcessorCount) // recursive call from reduction below
    {
        // we got enough elements to generate: do one element per thread, and reduction inside
        _launchTensorOp<ElemType, N, M, K><<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream>>>(
            beta, pointers, alpha, op, reductionOp,
            regularOpStrides, regularStrides, grid.m_N,
            reducingOpDims, reducingStrides,
            regularOpStrideDivmod, reducingOpDimDivmod);
    }
    // === optimization: simple case would not use all multiprocs
    else
    {
        // m_blocksPerGrid can be thought of NN / 512, with appropriate rounding

        // we are reducing and are underutilizing the multiprocs we have: get more parallelism by doing reduction in parallel
        // If we get here, then
        //  - the total number of outputs to produce is < #multiprocs * warpSize, e.g. < 960
        //  - each output has at least two inputs, but possibly millions
        // Examples:
        //  (a1) NN=900
        //        - each multiproc processes multiple elements concurrently, each reducing over its inputs inside
        //        - use one block per output element
        //  (a2) NN=30
        //        - same as (a1) except 30 multiprocs run only a single block each
        //  (a3) NN=16
        //        - same as (a1) except only 16 multiproc run one block
        //  (b1) NN=15
        //        - 2 blocks work together on a single output element
        //  (b2) NN=1    (NN < #multiprocs, e.g. NN < 30)
        //        - multiple blocks work together on a single output element
        //        - only this case requires memory, and only K * NN
        //          where K = blocks that work together,
        //          both K and NN < #multiprocs,
        //          and K * NN = on the order of NN, but generally a bit larger due to rounding.

        // By how much do we underutilize?
        // We increase #blocks by that factor by breaking reduction into that many chunks.
        let numReductionChunks = max(props.multiProcessorCount / NN, 1); // only >1 for NN < multiProcessorCount

        // distribute NN over block X and Y
        let blockXOverBy = CeilDiv(NN, props.maxGridSize[0]);
        let numBlocksX = CeilDiv(NN, blockXOverBy);
        let numBlocksY = CeilDiv(NN, numBlocksX);
        // while block Z is for multiple blocks working together on a single output element
        let numBlocksZ = numReductionChunks;
        // Block dim is now:
        //  - X, Y: such that X*Y covers NN
        //  - Z: reduction chunks

        // reduction goes into thread dim X
        let reductionChunkSize = CeilDiv(reductionDim, numReductionChunks);
        let numThreadsX = min(reductionChunkSize, GridDim::maxThreadsPerBlock); // any that's over will be done by looping inside the kernel

        // --- cases (a1) and (a2)
        // This involves no reduction across blocks.
        if (numReductionChunks == 1)
        {
            _launchTensorOpWithReduction<ElemType, N, M, K><<<dim3(numBlocksX, numBlocksY, numBlocksZ), numThreadsX, numThreadsX * sizeof(ReduceElemType), t_stream>>>(
                beta, pointers, alpha, op, reductionOp,
                regularOpStrides, regularStrides, NN,
                reducingOpDims, reducingStrides, /*reductionBegin*/ 0, reductionChunkSize,
                regularOpStrideDivmod, reducingOpDimDivmod);
        }
        // --- case (b)
        // Reduction across blocks. This is the difficult one.
#ifndef ALLOW_ATOMIC_REDUCTION // temporarily disabled to ensure it is not causing the non-reproducability
        else
        {
            // we get here if NN <= #multiprocs
            assert(NN <= props.multiProcessorCount && numBlocksX == NN && numBlocksY == 1);
            // dims are:
            //  - numBlocksZ = numReductionChunks = how many multiprocs work together to produce one output element
            //  - numBlocksX = NN = number of output elements
            //  - numThreadsX = reductionChunkSize clipped to 512; reductionChunkSize > 512 is handled by an inner for loop inside of the kernel

            // we need memory for block outputs of dimension [numBlocksX x numBlocksZ]
            //  - total elements = NN * Floor(#multiprocs / NN) = <= #multiprocs
            let reductionBufferSize = props.multiProcessorCount;
            assert(reductionBufferSize >= NN * numBlocksZ);
            shared_ptr<ElemType> reductionBuffer = GetReductionBuffer<ElemType>(reductionBufferSize);

            // 'pointers', 'regularOpStrides', and 'regularStrides' are set up to point to the target memory.
            // We need to reroute them to point to our reductionBuffer.
            //  - pointer[N-1] -> replace by reductionBuffer
            //  - regularStrides -> replace [N-1] by regularOpStrides which already represent the NN elements for a dense memory layout
            //  - beta -> 0 since we write into temp memory
            //  - kernel must use block.z as second index into the output buffer; add (block.z * NN) to the pointer
            FixedArray<ElemType*, N> pointers1 = pointers;
            pointers1[N - 1] = reductionBuffer.get();
            auto regularStrideVectors1 = regularStrideVectors;
            for (size_t k = 0; k < regularOpStrides.size(); k++)
                regularStrideVectors1[N - 1][k] = (ptrdiff_t)regularOpStrideVector[k];
            FixedMatrix<C_int, N, K> regularStrides1(regularStrideVectors1);
            ElemType beta1  = 0;
            ElemType alpha1 = 1;
            _launchTensorOpWithReduction<ElemType, N, M, K> << <dim3(numBlocksX, numBlocksY, numBlocksZ), numThreadsX, numThreadsX * sizeof(ReduceElemType), t_stream >> >(
                beta1, pointers1, alpha1, op, reductionOp,
                regularOpStrides, regularStrides1, NN,
                reducingOpDims, reducingStrides, /*reductionBegin*/0, reductionChunkSize,
                regularOpStrideDivmod, reducingOpDimDivmod);

#if 1
            // now reduce and redistribute
            // Create a new tensor task, and execute it recursively:
            //  - input  = reductionBuffer
            //  - output = true output
            //  - op dims/strides     = output elements
            //  - reduce dims/strides = numBlocksZ
            //  - op = opCopy
            array<ElemType*, 2>                    pointerVector2{         reductionBuffer.get(),        pointerVector[N - 1] };
            const array<SmallVector<ptrdiff_t>, 2> regularStrideVectors2{  regularStrideVectors1[N - 1], regularStrideVectors[N - 1] };
            const array<SmallVector<ptrdiff_t>, 2> reducingStrideVectors2{ SmallVector<ptrdiff_t>{ NN }, SmallVector<ptrdiff_t>{ 0 } };
            const SmallVector<size_t>              reducingOpDimVector2{ (size_t)numReductionChunks };
            LaunchTensorOpWithReduction<ElemType, /*N=*/2, /*M=*/1, K>(
                beta, pointerVector2, alpha, ElementWiseOperator::opCopy, reductionOp,
                regularOpDims, regularStrideVectors2,
                reducingOpDimVector2, reducingStrideVectors2);

            // (note: ^^this will have a nested syncGuard, which is fine)

#else
            _launchTensorOp<ElemType, N, M, K><<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream>>>(
                beta, pointers, alpha, op, reductionOp,
                regularOpStrides, regularStrides, grid.m_N,
                reducingOpDims, reducingStrides);
            //for (size_t z = 0; z < numBlocksZ; z++)
            //    _launchTensorOpWithReduction<ElemType, N, M, K><<<dim3(numBlocksX, numBlocksY, 1), numThreadsX, numThreadsX * sizeof(ReduceElemType), t_stream>>>(z == 0 ? beta : 1, pointers, alpha, op,
            //    regularOpStrides, regularStrides, NN,
            //    reducingOpDims, reducingStrides, reductionChunkSize * z, reductionChunkSize);
            vector<ElemType> peekPartial(NN * numBlocksZ, -42);
            vector<ElemType> peekFinal(NN, -42);
            CUDA_CALL(cudaMemcpy(peekPartial.data(), reductionBuffer,             sizeof(ElemType) * peekPartial.size(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(peekFinal.data(),   pointers[pointers.size()-1], sizeof(ElemType) * peekFinal.size(),   cudaMemcpyDeviceToHost));
            double s1 = 0, s2 = 0;
            for (auto v : peekPartial)
                s1 += v;
            for (auto v : peekFinal)
                s2 += v;
            sin(1.0);
#endif
        }
#else
        else if (beta == 1)
        {
            // no need to pre-scale; just add (common for gradients)
            _launchTensorOpWithReduction<ElemType, N, M, K><<<dim3(numBlocksX, numBlocksY, numBlocksZ), numThreadsX, numThreadsX * sizeof(ReduceElemType), t_stream>>>(beta, pointers, alpha, op, reductionOp, regularOpStrides,
                                                                   regularStrides, NN, reducingOpDims, reducingStrides, 0, reductionChunkSize,
                                                                   regularOpStrideDivmod, reducingOpDimDivmod);
            return;
        }
        else
        {
            // We need more than one chunk, we will use atomicAdd().
            // First reset/pre-multiply input; then do the remaining chunks using atomicAdd().
            _launchTensorOpWithReduction<ElemType, N, M, K><<<dim3(numBlocksX, numBlocksY, 1), numThreadsX, numThreadsX * sizeof(ReduceElemType), t_stream>>>(beta, pointers, alpha, op, reductionOp, regularOpStrides, regularStrides, NN, reducingOpDims, reducingStrides, 0, reductionChunkSize,
            regularOpStrideDivmod, reducingOpDimDivmod);
            // We will leave it like this for a while, but eventually need to revisit using temporary memory.
            _launchTensorOpWithReduction<ElemType, N, M, K><<<dim3(numBlocksX, numBlocksY, numBlocksZ - 1), numThreadsX, numThreadsX * sizeof(ReduceElemType), t_stream>>>(/*beta=*/1, pointers, alpha, op, reductionOp, regularOpStrides, regularStrides, NN, reducingOpDims, reducingStrides, reductionChunkSize, reductionChunkSize,
            regularOpStrideDivmod, reducingOpDimDivmod);
        }
#endif
    }
}

// -----------------------------------------------------------------------
// kernel and launch  --linear unary
// -----------------------------------------------------------------------

// for linear unary ops, we need to define a functor for every function for use as a template parameter (lambda syntax doesn't work in CUDA 7)
#define DefineUnaryTensorFunctor(oper)           \
    struct Functor##oper                         \
    {                                            \
        template <class ElemType>                \
        static __device__ ElemType f(ElemType a) \
        {                                        \
            return Op##oper(a);                  \
        }                                        \
    };
ForAllUnaryOps(DefineUnaryTensorFunctor);

// the top-level kernel for linear unary ops
// Note: If we have a beta, we have 2 memory accesses, so this optimization may no longer be needed as we are memory-bound.
template <class ElemType, class FN>
__global__ void _launchUnaryTensorOp(ElemType beta, const ElemType* pa, ElemType* pb, ElemType alpha, CUDA_LONG numElements)
{
    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id >= numElements)
        return;
    ElemType a = pa[id];
    ElemType val = FN::f(a);
    val *= alpha;
    if (beta != 0)
        val += beta * pb[id];
    pb[id] = val;
}
// version without beta and alpha
template <class ElemType, class FN>
__global__ void _launchUnaryTensorOp(const ElemType* pa, ElemType* pb, CUDA_LONG numElements)
{
    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id >= numElements)
        return;
    ElemType a = pa[id];
    ElemType val = FN::f(a);
    pb[id] = val;
}

// special case of linear unary operation
template <class ElemType>
void LaunchUnaryTensorOp(ElemType beta, const ElemType* pa, ElemType* pb, ElemType alpha, ElementWiseOperator op, size_t regularOpDim)
{
    CUDA_LONG NN = (CUDA_LONG) regularOpDim;

#define CaseLaunchUnaryTensorOp(oper)                                                                                                        \
    case ElementWiseOperator::op##oper:                                                                                                      \
        if (beta == 0 && alpha == 1)                                                                                                         \
            _launchUnaryTensorOp<ElemType, Functor##oper><<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream>>>(pa, pb, NN); \
        else                                                                                                                                 \
            _launchUnaryTensorOp<ElemType, Functor##oper><<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream>>>(beta, pa, pb, alpha, NN);\
        break;

    SyncGuard syncGuard;
    GridDim grid(NN);
    switch (op)
    {
        ForAllUnaryOps(CaseLaunchUnaryTensorOp);
    default:
        LogicError("LaunchTensorOp1: Unknown op code %d.", (int) op);
    }
}

// -----------------------------------------------------------------------
// map runtime parameters N to template parameters
// -----------------------------------------------------------------------

// tensor operation with k+1 dimensions (-1 means scalar)
template <class ElemType, C_size_t N, C_int K>
static void TensorOpWithRegularLoop(ElemType beta, const array<ElemType*, N>& pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
                                    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    size_t dims = reducingOpDims.size();
    switch (dims)
    {
    case 2:
        return LaunchTensorOpWithReduction<ElemType, N, 2, K>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 1:
        return LaunchTensorOpWithReduction<ElemType, N, 1, K>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 0:
        return LaunchTensorOp<ElemType, N, K>(beta, pointers, alpha, op, regularOpDims, regularStrides);
    default:
        LogicError("TensorOp: %d non-flattened reduction dimensions are not supported.", (C_int) dims);
    }
}

// tensor operation, generalized in number of arguments
// This function now expands into different k. It also eliminates the offsets by adding them to the pointers.
template <class ElemType, C_size_t N>
void TensorOpN(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
               const array<size_t, N>& offsets,
               const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
               const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    for (C_size_t i = 0; i < N; i++) // N = a small constant, this will be unrolled
        pointers[i] += offsets[i];
    size_t dims = regularOpDims.size();
    switch (dims)
    {
    case 4:
        return TensorOpWithRegularLoop<ElemType, N, 4>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 3:
        return TensorOpWithRegularLoop<ElemType, N, 3>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 2:
        return TensorOpWithRegularLoop<ElemType, N, 2>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 1:
        return TensorOpWithRegularLoop<ElemType, N, 1>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 0:
        return TensorOpWithRegularLoop<ElemType, N, 0>(beta, pointers, alpha, op, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    default:
        LogicError("TensorOp: %d non-flattened input dimensions are not supported.", (C_int) dims);
    }
}

//------------------------------------------------------------------------
// explicit instantiations--these are being called from GPUMatrix.cu
//------------------------------------------------------------------------

template void TensorOpN<float, 2>(float beta, array<float*, 2> pointers, float alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                  const array<size_t, 2>& offsets,
                                  const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                  const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);
template void TensorOpN<float, 3>(float beta, array<float*, 3> pointers, float alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                  const array<size_t, 3>& offsets,
                                  const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                                  const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides);
template void TensorOpN<float, 4>(float beta, array<float*, 4> pointers, float alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                  const array<size_t, 4>& offsets,
                                  const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                                  const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides);
template void TensorOpN<double, 2>(double beta, array<double*, 2> pointers, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 2>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);
template void TensorOpN<double, 3>(double beta, array<double*, 3> pointers, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 3>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides);
template void TensorOpN<double, 4>(double beta, array<double*, 4> pointers, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 4>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides);

template void LaunchUnaryTensorOp(float beta, const float* pa, float* pb, float alpha, ElementWiseOperator op, size_t regularOpDim);
template void LaunchUnaryTensorOp(double beta, const double* pa, double* pb, double alpha, ElementWiseOperator op, size_t regularOpDim);

}}}

#endif // CPUONLY
