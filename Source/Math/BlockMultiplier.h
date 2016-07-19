//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#pragma once
#include "BlockMultiplierPlatform.h"
#include <malloc.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <exception>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include "BlockMultiplierMatrixUtil.h"
#include "BlockHandlerSSE.h"
#ifdef SUPPORT_AVX2
#include "BlockHandlerAVX.h"
#endif
//#define STDTHREAD
#define OPENMPTHREAD
#ifdef STDTHREAD
#include "StdThreadPool.h"
#else 
#ifdef OPENMPTHREAD
#include <omp.h>
#endif
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// All of the information needed for the various BlockHandler functions.
// Packaged into a struct because in some cases we call these functions on a thread pool
// and most thread pool class interfaces aren't happy with lots of arguments to the worker
// thread function.
//
// Throughout the code, I use the following nomenclature:
// A is the LHS matrix for the multiplication, B is the RHS, and C is the result.
// (i.e. A * B = C).
// m, k, and n are the matrix dimensions.
// m is the number of rows in A and C.
// k is the common dimension (the number of columns in A and rows in B).
// n is the number of columns in B and C.
// In other words, A is (m x k), B is (k x n) and C is (m x n).
//
template<typename BlockHandlerT>struct HandlerArgs
{
    int startRow;
    int blocks;
    int m;
    int k;
    int n;
    typename BlockHandlerT::ScalarAT* newA;
    typename BlockHandlerT::ScalarBT* B;
    int32_t* transC;
    int rowsPerThread;
    int rowsPerBlock;
    typename BlockHandlerT::VectorT* pBlockPreparedB;
};


// BlockMultiplier is a GEMM implementation that is optimized by
// reordering matrices so that they will be accessed sequentially
// in memory at multiplication time. This reduces cache misses and
// results in faster-executing code on the CPU.
// This class handles the rewriting of the matrices and the top level
// multiplication. Blocks of A and B (the LHS and RHS of the multiplication)
// are then handed off to a class implementing the BlockHandlerT interface.
// Implementations are provided for multiplying 16-bit integer matrices using
// the SSE and AVX2 instruction sets. To compile for AVX2, you need to add the /arch:AVX2
// flag to the compiler. Note that the AVX2 code only runs on Haswell or better processors,
// will throw illegal instruction on other machines.
// To use the code, first call PrepareB, which rewrites B in block order and returns
// a pointer to the rewritten block (don't forget to call FreePreparedB on it when you're done
// multiplying by that matrix). Then you can call MultiplyMatrices().
// For details on how the block rewrite works, see the comments on RewriteBInBlockOrder and
// RewriteAInBlockOrder.
template<typename BlockHandlerT> class BlockMultiplier
{
    public:
        // The type of the element in the A (LHS) matrix
        typedef typename BlockHandlerT::ScalarAT ScalarAT;
        // The type of the element in the B (RHS) matrix
        typedef typename BlockHandlerT::ScalarBT ScalarBT;
        // Right now we always produce 32-bit integer results. This could be changed if necessary.
        typedef int32_t ScalarCT;
        // The vectorized type of the block multiplier (e.g. __m128i for SSE, __m256 for AVX2).
        typedef typename BlockHandlerT::VectorT VectorT;

    private:
        BlockHandlerT m_handler;
        int referenceKernel(ScalarAT* A, ScalarBT* B, int blockSize);
        void kernelsse8x2(__m128i xmmRow0, __m128i xmmRow1, short* B, short* return1, short* return2);
        int RowToColOffsetRewrittenB(int col, int kOffset, int blockSize, int origCols);
        int RowToColOffsetRewrittenA(int row, int kOffset, int blockSize, int rowsPerBlock, int origCols);
        void RewriteBInBlockOrder(ScalarBT* oldB, ScalarBT* newB, int k, int n);
        ScalarBT* RewriteBInBlockOrder(ScalarBT* oldB, ScalarBT* newB, int k, int n, int blockSize, int* kOffset);
        void RewriteAInBlockOrder(ScalarAT* A, ScalarAT* newA, int m, int k, int blockSize, int rowsPerBlock);
        ScalarAT* RewriteAInBlockOrder(ScalarAT* A, ScalarAT* newA, int m, int k, int blockSize, int rowsPerBlock, int* pKOffset);
        ScalarAT* RewriteAInBlockOrder2(ScalarAT* A, ScalarAT* newA, int m, int k, int blockSize, int rowsPerBlock, int* pKOffset);
        int m_blockSize;
        std::mutex m_MultiplyMut;

        // Function objects - thin wrappers around the thread functions (which know how to feed
        // blocks to the actualy dot product kernels implemented in BlockHandlerT).

        class BlockHandler128x4Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler128x4Thread(param); }
        };
        class BlockHandler64x4Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler64x4Thread(param); }
        };
        class BlockHandler32x4Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler32x4Thread(param); }
        };
        class BlockHandler16x4Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler16x4Thread(param); }
        };
        class BlockHandler8x4Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler8x4Thread(param); }
        };
        class BlockHandler128x1Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler128x1Thread(param); }
        };
        class BlockHandler64x1Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler64x1Thread(param); }
        };
        class BlockHandler32x1Fn 
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler32x1Thread(param); }
        };
        class BlockHandler16x1Fn
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler16x1Thread(param); }
        };
        class BlockHandler8x1Fn
        {
            public:
                void operator()(HandlerArgs<BlockHandlerT> param) { BlockHandler8x1Thread(param); }
        };

#ifdef STDTHREAD
        std::unique_ptr<StdThreadPool<HandlerArgs<BlockHandlerT>>> m_pPool;
#endif

        typename BlockHandlerT::VectorT* m_pBlockHandlerBInfo;


        // All of the BlockHandlerFooxBarThread functions work the same way.
        // FooxBar means we are processing Foo elements from the common dimension K,
        // Bar rows at a time. So 128x4 means we are processing four rows at a time, with
        // 128 entries of the common dimension (k) at a time.
        //
        // We walk through all of the blocks for this set of rows in the common dimension, accumulating
        // the partial sums in resultStorage as we go. These temporary data are then copied into the target
        // matrix (C). Accumulating directly in C is a disaster because you end up with read-write hazards
        // in multithreaded situations and end up with lots of pipeline stalls trying to reconcile the cache, so
        // this ends up being faster even through we're allocating the memory each time. I think we can
        // get rid of the allocation but I haven't had time to do this yet.
        static void BlockHandler128x4Thread(HandlerArgs<BlockHandlerT> ha)
        {
            // Accumulate full row results locally b/f writing to C
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * ha.rowsPerBlock * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * ha.rowsPerBlock * ha.n);
            const int blocksAtOnce = 2;

            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; currBlock += blocksAtOnce)
            {
                BlockHandlerT::HandleBlock128x4(currBlock, ha.startRow, ha.k, ha.n, ha.newA, 
                        ha.B, std::min(ha.blocks - currBlock, blocksAtOnce), resultStorage, ha.pBlockPreparedB);
            }

            int n = ha.n;
            {
                // This takes about the same amount of time as the memcpy version below.
                for (int c = 0; c < n; ++c)
                {
                    //_mm_prefetch((char*)&(transC[RowColToOffset(c, startRow, m)]), _MM_HINT_T1);
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    VectorT result2 = resultStorage[RowColToOffset(1, c, n)];
                    VectorT result3 = resultStorage[RowColToOffset(2, c, n)];
                    VectorT result4 = resultStorage[RowColToOffset(3, c, n)];
                    int32_t firstHorizontal  = my_hadd(result1);
                    int32_t secondHorizontal = my_hadd(result2);
                    int32_t thirdHorizontal  = my_hadd(result3);
                    int32_t fourthHorizontal = my_hadd(result4);
                    transC[RowColToOffset(ha.startRow, c, n)]     = firstHorizontal;
                    transC[RowColToOffset(ha.startRow + 1, c, n)] = secondHorizontal;
                    transC[RowColToOffset(ha.startRow + 2, c, n)] = thirdHorizontal;
                    transC[RowColToOffset(ha.startRow + 3, c, n)] = fourthHorizontal;
                }
            }

            ALIGNED_FREE(resultStorage);

        }

        static void BlockHandler64x4Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * 4 * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * 4 * ha.n);
            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock64x4(currBlock, ha.startRow, ha.k, ha.n, ha.newA, 
                        ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    VectorT result2 = resultStorage[RowColToOffset(1, c, n)];
                    VectorT result3 = resultStorage[RowColToOffset(2, c, n)];
                    VectorT result4 = resultStorage[RowColToOffset(3, c, n)];
                    int32_t firstHorizontal  = my_hadd(result1);
                    int32_t secondHorizontal = my_hadd(result2);
                    int32_t thirdHorizontal  = my_hadd(result3);
                    int32_t fourthHorizontal = my_hadd(result4);
                    transC[RowColToOffset(ha.startRow, c, n)]     += firstHorizontal;
                    transC[RowColToOffset(ha.startRow + 1, c, n)] += secondHorizontal;
                    transC[RowColToOffset(ha.startRow + 2, c, n)] += thirdHorizontal;
                    transC[RowColToOffset(ha.startRow + 3, c, n)] += fourthHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler32x4Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * 4 * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * 4 * ha.n);
            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock32x4(currBlock, ha.startRow, ha.k, ha.n, ha.newA, 
                        ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    VectorT result2 = resultStorage[RowColToOffset(1, c, n)];
                    VectorT result3 = resultStorage[RowColToOffset(2, c, n)];
                    VectorT result4 = resultStorage[RowColToOffset(3, c, n)];
                    int32_t firstHorizontal  = my_hadd(result1);
                    int32_t secondHorizontal = my_hadd(result2);
                    int32_t thirdHorizontal  = my_hadd(result3);
                    int32_t fourthHorizontal = my_hadd(result4);
                    transC[RowColToOffset(ha.startRow, c, n)]     += firstHorizontal;
                    transC[RowColToOffset(ha.startRow + 1, c, n)] += secondHorizontal;
                    transC[RowColToOffset(ha.startRow + 2, c, n)] += thirdHorizontal;
                    transC[RowColToOffset(ha.startRow + 3, c, n)] += fourthHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler16x4Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*) ALIGNED_ALLOC(sizeof(VectorT) * 4 * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * 4 * ha.n);
            int32_t* transC = ha.transC;
            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock16x4(currBlock, ha.startRow, ha.k, ha.n, ha.newA, 
                        ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    VectorT result2 = resultStorage[RowColToOffset(1, c, n)];
                    VectorT result3 = resultStorage[RowColToOffset(2, c, n)];
                    VectorT result4 = resultStorage[RowColToOffset(3, c, n)];
                    int32_t firstHorizontal  = my_hadd(result1);
                    int32_t secondHorizontal = my_hadd(result2);
                    int32_t thirdHorizontal  = my_hadd(result3);
                    int32_t fourthHorizontal = my_hadd(result4);
                    transC[RowColToOffset(ha.startRow, c, n)]     += firstHorizontal;
                    transC[RowColToOffset(ha.startRow + 1, c, n)] += secondHorizontal;
                    transC[RowColToOffset(ha.startRow + 2, c, n)] += thirdHorizontal;
                    transC[RowColToOffset(ha.startRow + 3, c, n)] += fourthHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler8x4Thread(HandlerArgs<BlockHandlerT> ha)
        {
            __m128i* resultStorage = (__m128i*)ALIGNED_ALLOC(sizeof(__m128i) * 4 * ha.n, 16);
            memset(resultStorage, 0, sizeof(__m128i) * 4 * ha.n);
            int32_t* transC = ha.transC;
            //_mm_prefetch((char*)&(transC[RowColToOffset(c, ha.startRow, m)]), _MM_HINT_T1);

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock8x4(currBlock, ha.startRow, ha.k, ha.n, ha.newA, 
                        ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    __m128i result1 = resultStorage[RowColToOffset(0, c, n)];
                    __m128i result2 = resultStorage[RowColToOffset(1, c, n)];
                    __m128i result3 = resultStorage[RowColToOffset(2, c, n)];
                    __m128i result4 = resultStorage[RowColToOffset(3, c, n)];
                    int32_t firstHorizontal  = my_hadd(result1);
                    int32_t secondHorizontal = my_hadd(result2);
                    int32_t thirdHorizontal  = my_hadd(result3);
                    int32_t fourthHorizontal = my_hadd(result4);
                    transC[RowColToOffset(ha.startRow, c, n)]     += firstHorizontal;
                    transC[RowColToOffset(ha.startRow + 1, c, n)] += secondHorizontal;
                    transC[RowColToOffset(ha.startRow + 2, c, n)] += thirdHorizontal;
                    transC[RowColToOffset(ha.startRow + 3, c, n)] += fourthHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }



        static void BlockHandler128x1Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * ha.rowsPerBlock * ha.n, 64);
            memset(resultStorage, 0, sizeof(VectorT) * ha.rowsPerBlock * ha.n);
            const int blocksAtOnce = 2;
            int32_t* transC = ha.transC;
            for (int currBlock = 0; currBlock < ha.blocks; currBlock += blocksAtOnce)
            {
                BlockHandlerT::HandleBlock128x1(currBlock, ha.startRow, ha.k, ha.n, 
                        ha.newA, ha.B, std::min(ha.blocks - currBlock, blocksAtOnce), resultStorage, ha.pBlockPreparedB);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    int32_t firstHorizontal = my_hadd(result1);
                    transC[RowColToOffset(ha.startRow, c, n)] = firstHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler64x1Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * ha.rowsPerBlock * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * ha.rowsPerBlock * ha.n);
            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock64x1(currBlock, ha.startRow, ha.k, 
                        ha.n, ha.newA, ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    int32_t firstHorizontal = my_hadd(result1);
                    transC[RowColToOffset(ha.startRow, c, n)] += firstHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler32x1Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * ha.rowsPerBlock * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * ha.rowsPerBlock * ha.n);
            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock32x1(currBlock, ha.startRow, ha.k,
                        ha.n, ha.newA, ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    int32_t firstHorizontal = my_hadd(result1);
                    transC[RowColToOffset(ha.startRow, c, n)] += firstHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler16x1Thread(HandlerArgs<BlockHandlerT> ha)
        {
            VectorT* resultStorage = (VectorT*)ALIGNED_ALLOC(sizeof(VectorT) * ha.rowsPerBlock * ha.n, 16);
            memset(resultStorage, 0, sizeof(VectorT) * ha.rowsPerBlock  * ha.n);
            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock16x1(currBlock, ha.startRow, ha.k, ha.n, 
                        ha.newA, ha.B, 1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    VectorT result1 = resultStorage[RowColToOffset(0, c, n)];
                    int32_t firstHorizontal = my_hadd(result1);
                    transC[RowColToOffset(ha.startRow, c, n)] += firstHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

        static void BlockHandler8x1Thread(HandlerArgs<BlockHandlerT> ha)
        {
            __m128i* resultStorage = (__m128i*)ALIGNED_ALLOC(sizeof(__m128i) * ha.rowsPerBlock * ha.n, 64);
            memset(resultStorage, 0, sizeof(__m128i) * ha.rowsPerBlock * ha.n);
            int32_t* transC = ha.transC;

            for (int currBlock = 0; currBlock < ha.blocks; ++currBlock)
            {
                BlockHandlerT::HandleBlock8x1(currBlock, ha.startRow, ha.k, ha.n, 
                        ha.newA, ha.B,  1, resultStorage);
            }

            int n = ha.n;
            {
                for (int c = 0; c < n; ++c)
                {
                    __m128i result1 = resultStorage[RowColToOffset(0, c, n)];
                    int32_t firstHorizontal = my_hadd(result1);
                    transC[RowColToOffset(ha.startRow, c, n)] += firstHorizontal;
                }
            }
            ALIGNED_FREE(resultStorage);
        }

    public:

        // Borrowed from [https://software.intel.com/en-us/forums/intel-isa-extensions/topic/285219]
        // Results in a saturated add, i.e. in the overflow case we return int_max, and in the
        // underflow we return int_min.
        // The code is clever - it uses the _mm_blendv_ps instruction to avoid branching.
        // It's based on the fact that you can only have overflow if the signs of the operands
        // are identical and different from the sign of the result.
        // The resulting code results in matrix multiplies that are a few percent slower than the non-saturating code,
        // but we avoid the wild inaccuracies resulting from over/underflow.
        FORCEINLINE static __m128i my_adds_epi32(__m128i a, __m128i b)
        {
            __m128i int_min = _mm_set1_epi32(0x80000000);
            __m128i int_max = _mm_set1_epi32(0x7FFFFFFF);
            __m128i res = _mm_add_epi32(a, b);
            __m128i sign_and = _mm_and_si128(a, b);
            __m128i sign_or = _mm_or_si128(a, b);
            __m128i min_sat_mask = _mm_andnot_si128(res, sign_and);
            __m128i max_sat_mask = _mm_andnot_si128(sign_or, res);
            __m128 res_temp = _mm_blendv_ps(_mm_castsi128_ps(res), _mm_castsi128_ps(int_min), _mm_castsi128_ps(min_sat_mask));
            return _mm_castps_si128(_mm_blendv_ps(res_temp, _mm_castsi128_ps(int_max), _mm_castsi128_ps(max_sat_mask)));
        }

        //Saturated horizontal add
        FORCEINLINE static int32_t my_hadd(__m128i hAddMe)
        {
            // We have ABCD, we want A + B + C +D.
            // Shuffle to get BADC
            __m128i shuff1 = _mm_shuffle_epi32(hAddMe, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i res1 = my_adds_epi32(hAddMe, shuff1);
            // Now we have
            // A+B B+A, C+D D+C, so shuffle and add once more
            __m128i shuff2 = _mm_shuffle_epi32(res1, _MM_SHUFFLE(0, 1, 2, 3));
            // This gives us
            // D+C A+B B+A C+D
            __m128i res2 = my_adds_epi32(res1, shuff2);
            return _mm_extract_epi32(res2, 0);
        }

#ifdef SUPPORT_AVX2
        //Same as above, for AVX registers
        FORCEINLINE static __m256i my_adds_epi32(__m256i a, __m256i b)
        {
            __m256i int_min = _mm256_set1_epi32(0x80000000);
            __m256i int_max = _mm256_set1_epi32(0x7FFFFFFF);
            __m256i res = _mm256_add_epi32(a, b);
            __m256i sign_and = _mm256_and_si256(a, b);
            __m256i sign_or = _mm256_or_si256(a, b);
            __m256i min_sat_mask = _mm256_andnot_si256(res, sign_and);
            __m256i max_sat_mask = _mm256_andnot_si256(sign_or, res);
            __m256 res_temp = _mm256_blendv_ps(_mm256_castsi256_ps(res), 
                    _mm256_castsi256_ps(int_min), _mm256_castsi256_ps(min_sat_mask));
            return _mm256_castps_si256(_mm256_blendv_ps(res_temp, _mm256_castsi256_ps(int_max), _mm256_castsi256_ps(max_sat_mask)));
        }

        //Same as above, for AVX registers
        FORCEINLINE static int32_t my_hadd(__m256i hAddMe)
        {
            __m256i shuff1 = _mm256_shuffle_epi32(hAddMe, _MM_SHUFFLE(1, 0, 3, 2));
            __m256i res1 = my_adds_epi32(hAddMe, shuff1);
            __m256i shuff2 = _mm256_shuffle_epi32(res1, _MM_SHUFFLE(0, 1, 2, 3));
            __m256i res2 = my_adds_epi32(res1, shuff2);
            __m256i shifted = _mm256_permute2f128_si256(res2, res2, 1);
            __m256i res3 = my_adds_epi32(res2, shifted);
            union {
                int32_t i[8]; __m256i v;
            } u;
            u.v = res3;
            return u.i[0];
        }
#endif


        int m_numThreads;

        BlockMultiplier(int numThreads = 1) 
        {
            SetNumThreads(numThreads);
        }

        void SetNumThreads(int threads)
        {
            m_numThreads = threads;
#ifdef STDTHREAD
            m_pPool.reset(new StdThreadPool<HandlerArgs<BlockHandlerT>>(threads));
#else
#ifdef OPENMPTHREAD
            m_oldNumThreads = omp_get_num_threads();
            omp_set_num_threads(threads);
#endif
#endif
        }

        ~BlockMultiplier()
        {
            BlockHandlerT::FreePreparedB(m_pBlockHandlerBInfo);
            omp_set_num_threads(m_oldNumThreads);
        }
        static ScalarAT* CreateMatrixA(int m, int n, ScalarAT initVal = 0);
        static ScalarBT* CreateMatrixB(int m, int n, ScalarBT initVal = 0);
        static int32_t* CreateMatrixC(int m, int n, int32_t initVal = 0);
        ScalarBT* PrepareB(ScalarBT* oldB, int k, int n);
        template<typename ScalarT> static void FreeMatrix(ScalarT* freeMe) { FreeAlignedMatrix<ScalarT>(freeMe); }
        // We assume B has been rewritten in block order.
        // For now we assume m, k and n are all multiples of kernelsize.
        void MultiplyMatrices(ScalarAT* A, int m, int k, ScalarBT* B, int n, int32_t* C, ScalarAT alpha = 1, ScalarBT beta = 0);
        static const int MAXRANGE = 1 << 13;
        int m_oldNumThreads;
};

// Instantiate block multipliers
template<> const int BlockMultiplier<BlockHandlerSSE>::MAXRANGE;
#ifdef SUPPORT_AVX2
template<> const int BlockMultiplier<BlockHandlerAVX>::MAXRANGE;
#endif


template<typename BlockHandlerT> typename BlockMultiplier<BlockHandlerT>::ScalarAT* BlockMultiplier<BlockHandlerT>::CreateMatrixA(int m, int n, ScalarAT initVal)
{
    return CreateAlignedMatrix<ScalarAT>(m, n, initVal);
}
template<typename BlockHandlerT> typename BlockMultiplier<BlockHandlerT>::ScalarBT* BlockMultiplier<BlockHandlerT>::CreateMatrixB(int m, int n, ScalarBT initVal)
{
    return CreateAlignedMatrix<ScalarBT>(m, n, initVal);
}
template<typename BlockHandlerT> int32_t* BlockMultiplier<BlockHandlerT>::CreateMatrixC(int m, int n, int32_t initVal)
{
    return CreateAlignedMatrix<int32_t>(m, n, initVal);
}


// reference kernel to get the algo right
// If you make any changes to the block algorithm, test it with this.
template<typename BlockHandlerT> FORCEINLINE int BlockMultiplier<BlockHandlerT>::referenceKernel(ScalarAT* A, 
        ScalarBT* B, int blockSize)
{
    // For now just use regular instructions
    // until we have the breakdown figured out.
    int accum = 0;
    for (int i = 0; i < blockSize; ++i)
    {
        accum += A[i] * B[i];
    }
    return accum;
}

// Rewrites B in Block order so that memory accesses to B will be sequential.
// See comments on RewriteBInBlockOrder for details.
template<typename BlockHandlerT> typename BlockMultiplier<BlockHandlerT>::ScalarBT* BlockMultiplier<BlockHandlerT>::PrepareB(ScalarBT* oldB, int k, int n)
{
    ScalarBT* newB = CreateMatrixB(k, n);
    int offset = 0;

    ScalarBT* next = RewriteBInBlockOrder(oldB, newB, k, n, 128, &offset);
    if (offset < k)
    {
        next = RewriteBInBlockOrder(oldB, next, k, n, 64, &offset);
    }
    if (offset < k)
    {
        next = RewriteBInBlockOrder(oldB, next, k, n, 32, &offset);
    }
    if (offset < k)
    {
        next = RewriteBInBlockOrder(oldB, next, k, n, 16, &offset);
    }
    if (offset < k)
    {
        next = RewriteBInBlockOrder(oldB, next, k, n, 8, &offset);
    }
    if (offset < k)
    {
        int blockSize = k - offset;
        next = RewriteBInBlockOrder(oldB, next, k, n, blockSize, &offset);
    }
    assert(next - newB == k * n);
    m_pBlockHandlerBInfo = BlockHandlerT::PrepareExtraB(newB, k, n);

    return newB;
}

// A version of RewriteBInBlockOrder that allows for multiple decreasing block sizes
// (Which is necessary for letting us handle arbitrarily-sized matrices). On the first call, set kOffset to zero, and
// newB to the beginning of the newly allocated B array.
// We return a pointer to the next writable element in newB, and kOffset gets the new offset into the shared dimension. 

// So if you have blocks of size 128 and 64, do something like this:
// short* newB = alloc_b();
// int kOffset=0;
// int retBlockSize = firstBlockSize;
// (Write out the blocks up to k=128)
// short* nextB = RewriteBInBlockOrder(oldB, newB, k, n, 128, *kOffset);
// short* lastB = RewriteBInBlockOrder(oldB, nextB, k, n, 64, &kOffset);

// For each block, the logic is as follows:
// Given a block size of 2 and a matrix that looks like this:
// B00 B01 B02 B03
// B10 B11 B12 B13
// B20 B21 B22 B23
// B30 B31 B32 B33
// .. we will rewrite it like this
// B00 B10 B01 B11 B02 B12 B03 B13 //The first block of columns 0,1,2, and 3.
// B20 B30 B21 B31 B22 B32 B23 B33 //The second block of columns, 0, 1, 2, and 3.

// So each "row" of the new matrix (block size * num_original_cols) contains all of the values that
// will be multiplied with block zero of a given row.
// Column offsets within the row can be computed by (block_size * col_offset).

// Given the original B matrix, we restructure it so that we will access all of its elements in block order.
// That way we can load in blocksize elements from A, then read in the corresponding elements from each
// column of B in sequential order.
// Old, replaced with fn below, keeping around for now for reference.
template<typename BlockHandlerT> typename BlockMultiplier<BlockHandlerT>::ScalarBT* BlockMultiplier<BlockHandlerT>::RewriteBInBlockOrder(ScalarBT* oldB, ScalarBT* newB, int k, 
        int n, int blockSize, int* kOffset)
{
    ScalarBT* curr = newB;
    int numBlocks = (k - *kOffset) / blockSize;
    for (int b = 0; b < numBlocks; ++b)
    {
        // Row offset to beginning of block
        int blockOffset = *kOffset + (b * blockSize);
        for (int c = 0; c < n; ++c)
        {
            for (int rowOffset = 0; rowOffset < blockSize; ++rowOffset)
            {
                int idx = RowColToOffset(blockOffset + rowOffset, c, n);
                *curr++ = oldB[idx];
            }
        }
    }
    // Bump offset for next level down
    *kOffset += (numBlocks * blockSize);
    // Return ptr to next value to be written
    return curr;
}

// We can also reorder A, pulling in multiple rows at once.
// So while the input looks like this in memory:
// Row1Block1 Row1Block2 Row1Block3
// Row2Block1 Row2Block2 Row2Block3
// Row3Block1 Row3Block2 Row3Block3
// Row4Block1 Row4Block2 Row4Block3
//
// we want the rewritten array to look like this (assuming 2 rows per block):
// Row1Block1 Row2Block1 Row1Block2 Row2Block2 Row1Block3 Row2Block3
// Row3Block1 Row4Block1 Row3Block2 Row4Block2 Row3Block3 Row4Block3

// length = 2 (rows per block) * 3 (blocks per row)

// So the first "row" contains the first rowsPerBlock rows in block order.
// We can index into the start of block b in a row via (b * blockSize * rowsPerBlock)
// blockSize is the size of our dot product kernel.
template<typename BlockHandlerT> void BlockMultiplier<BlockHandlerT>::RewriteAInBlockOrder(ScalarAT* A, ScalarAT* newA, int m,
        int k, int /*blockSize*/, int rowsPerBlock)
{
    int currOffset = 0;
    ScalarAT* next = RewriteAInBlockOrder(A, newA, m, k, 128, rowsPerBlock, &currOffset);
    if (currOffset < k)
    {
        next = RewriteAInBlockOrder(A, next, m, k, 64, rowsPerBlock, &currOffset);
    }
    if (currOffset < k)
    {
        next = RewriteAInBlockOrder(A, next, m, k, 32, rowsPerBlock, &currOffset);
    }
    if (currOffset < k)
    {
        next = RewriteAInBlockOrder(A, next, m, k, 16, rowsPerBlock, &currOffset);
    }
    if (currOffset < k)
    {
        next = RewriteAInBlockOrder(A, next, m, k, 8, rowsPerBlock, &currOffset);
    }
    if (currOffset < k)
    {
        int blockSize = k - currOffset;
        next = RewriteAInBlockOrder(A, next, m, k, blockSize, rowsPerBlock, &currOffset);
    }
}

template<typename BlockHandlerT> typename BlockMultiplier<BlockHandlerT>::ScalarAT* BlockMultiplier<BlockHandlerT>::RewriteAInBlockOrder(ScalarAT* A, ScalarAT* newA, int m, int k, 
        int blockSize, int rowsPerBlock, int* pKOffset)
{
    ScalarAT* currOut = newA;
    int blocksPerRow = (k - *pKOffset) / blockSize;
    for (int startRow = 0; startRow < m; startRow += rowsPerBlock)
    {
        for (int currBlock = 0; currBlock < blocksPerRow; ++currBlock)
        {
            for (int rowOffset = 0; rowOffset < rowsPerBlock; ++rowOffset)
            {
                int startCol = *pKOffset + (currBlock * blockSize);
                for (int c = 0; c < blockSize; ++c)
                {
                    *currOut++ = A[RowColToOffset(startRow + rowOffset, startCol + c, k)];
                }
            }
        }
    }
    *pKOffset += blocksPerRow * blockSize;
    return currOut;
}


// I tried parallelizing this and got no speedup - that doesn't really make sense though.
template<typename BlockHandlerT> typename BlockMultiplier<BlockHandlerT>::ScalarAT* BlockMultiplier<BlockHandlerT>::RewriteAInBlockOrder2(ScalarAT* A, ScalarAT* newA, 
        int m, int k, int blockSize, int rowsPerBlock, int* pKOffset)
{
    ScalarAT* currOut = newA;
    int blocksPerRow = (k - *pKOffset) / blockSize;
    //#pragma omp parallel for
    for (int startRow = 0; startRow < m; startRow += rowsPerBlock)
    {
        int offset3 = startRow * blocksPerRow * (rowsPerBlock/4) * blockSize;
        for (int currBlock = 0; currBlock < blocksPerRow; ++currBlock)
        {
            int outOffset2 = currBlock * blockSize * rowsPerBlock;
            for (int rowOffset = 0; rowOffset < rowsPerBlock; ++rowOffset)
            {
                int outOffset = rowOffset * blockSize;
                int startCol = *pKOffset + (currBlock * blockSize);

                for (int c = 0; c < blockSize; ++c)
                {
                    *(currOut + outOffset + outOffset2 + offset3 + c) = A[RowColToOffset(startRow + rowOffset, startCol + c, k)];
                }
            }
        }
    }

    currOut += blockSize * rowsPerBlock * blocksPerRow * (m/rowsPerBlock);
    *pKOffset += blocksPerRow * blockSize;
    return currOut;
}


template<typename BlockHandlerT>struct BlockInfo
{
    BlockInfo(int cnt, int setK, int offsetASet, int offsetBSet, std::function<void(HandlerArgs<BlockHandlerT>)> fourFnSet,
            std::function<void(HandlerArgs<BlockHandlerT>)> oneFnSet)
        : blockCnt(cnt), k(setK), offsetA(offsetASet), offsetB(offsetBSet), fourFn(fourFnSet), oneFn(oneFnSet) {}

    int blockCnt;
    int k;
    int offsetA;
    int offsetB;
    std::function<void (HandlerArgs<BlockHandlerT>)> fourFn;
    std::function<void (HandlerArgs<BlockHandlerT>)> oneFn;
};


// We assume B has been rewritten in block order.
// We assume C has been zeroed out.
template<typename BlockHandlerT> void BlockMultiplier<BlockHandlerT>::MultiplyMatrices(ScalarAT* A, int m, int k, ScalarBT* B, int n,
        int32_t* C, ScalarAT alpha, ScalarBT beta)
{

    // The lock here is required whenever you are calling MultiplyMatrices on the same BlockMultiplier object
    // (For instance you are simultaneously multiplying two different inputs by the same first layer weight matrix).
    // It is an artifact of the thread pool I am using when STDTHREAD is defined and may not be necessary
    // in the openmp case, but I have not tested this scenario so I'm leaving it in for now.
    std::lock_guard<std::mutex> lock(m_MultiplyMut);
    {
        // We want to multithread to the extent possible. When batch size is small this
        // means doing a row at a time so we can take advantage of multiple procs.
        // But only row sizes 1 and 4 are supported so far (should be fixed by codegen).
        int rowsPerBlock = m / m_numThreads;
        if (rowsPerBlock < 4)
            rowsPerBlock = 1;
        if (rowsPerBlock > 4)
            rowsPerBlock = 4;

        // Fall back to row at a time if we end up with an invalid # of rows at a time
        // TODO: We should always do 4 rows at a time if it makes sense from a threading standpoint
        // since it is significantly more efficient. So if we have e.g. 7 rows, we should do
        // one set of four rows at a time and then three single rows. This however will require
        // changes to this function, RewriteAInBlockOrder and RowToColOffsetRewrittenA, so for now
        // we are silently backing off to row at a time.
        if (m % rowsPerBlock != 0)
        {
            rowsPerBlock = 1;
        }

        if (alpha != 1 || beta != 0)
        {
            throw std::logic_error("alpha / beta not yet implemented for this class");
        }

        ScalarAT* newA = CreateMatrixA(m, k);

        RewriteAInBlockOrder(A, newA, m, k, m_blockSize, rowsPerBlock);

        int blocks128 = k / 128;
        int k128 = blocks128 * 128;
        int blocks64 = (k - k128) / 64;
        int k64 = blocks64 * 64;
        int blocks32 = (k - k128 - k64) / 32;
        int k32 = blocks32 * 32;
        int blocks16 = (k - k128 - k64 - k32) / 16;
        int k16 = blocks16 * 16;
        int blocks8 = (k - k128 - k64 - k32 - k16) / 8;
        int k8 = blocks8 * 8;
        int blocks1 = (k - k128 - k64 - k32 - k16 - k8);

        int offsetA64 = m * blocks128 * 128;
        int offsetB64 = n * blocks128 * 128;
        int offsetA32 = offsetA64 + (m * k64);
        int offsetB32 = offsetB64 + (n * k64);
        int offsetA16 = offsetA32 + (m * k32);
        int offsetB16 = offsetB32 + (n * k32);
        int offsetA8 = offsetA16 + (m * k16);
        int offsetB8 = offsetB16 + (n * k16);

        int offsetA1 = offsetA8 + (m * k8);
        int offsetB1 = offsetB8 + (n * k8);


        BlockHandler128x4Fn fn128x4;
        BlockHandler64x4Fn fn64x4;
        BlockHandler32x4Fn fn32x4;
        BlockHandler16x4Fn fn16x4;
        BlockHandler8x4Fn fn8x4;
        BlockHandler128x1Fn fn128x1;
        BlockHandler64x1Fn fn64x1;
        BlockHandler32x1Fn fn32x1;
        BlockHandler16x1Fn fn16x1;
        BlockHandler8x1Fn fn8x1;

        std::vector<BlockInfo<BlockHandlerT>*> blockInfos(5);

        BlockInfo<BlockHandlerT> bi128(blocks128, k128, 0, 0, fn128x4, fn128x1);
        BlockInfo<BlockHandlerT> bi64(blocks64, k64, offsetA64, offsetB64, fn64x4, fn64x1);
        BlockInfo<BlockHandlerT> bi32(blocks32, k32, offsetA32, offsetB32, fn32x4, fn32x1);
        BlockInfo<BlockHandlerT> bi16(blocks16, k16, offsetA16, offsetB16, fn16x4, fn16x1);
        BlockInfo<BlockHandlerT> bi8(blocks8, k8, offsetA8, offsetB8, fn8x4, fn8x1);

        blockInfos[0] = &bi128;
        blockInfos[1] = &bi64;
        blockInfos[2] = &bi32;
        blockInfos[3] = &bi16;
        blockInfos[4] = &bi8;
        for (int i = 0; i < blockInfos.size(); ++i)
        {
            BlockInfo<BlockHandlerT>& currBlockInfo = *(blockInfos[i]);
            if ( currBlockInfo.blockCnt > 0)
            {
                HandlerArgs<BlockHandlerT> ha;
                ha.blocks = currBlockInfo.blockCnt;
                ha.m = m;
                ha.k = currBlockInfo.k;
                ha.n = n;
                ha.newA = newA + currBlockInfo.offsetA;
                ha.B = B + currBlockInfo.offsetB;
                ha.transC = C;
                ha.rowsPerThread = m / m_numThreads;
                ha.rowsPerBlock = rowsPerBlock;
                ha.pBlockPreparedB = m_pBlockHandlerBInfo;

                if (rowsPerBlock == 4)
                {

#ifdef OPENMPTHREAD
#pragma omp parallel for
#endif
                    for (int startRow = 0; startRow < m; startRow += 4)
                    {
                        ha.startRow = startRow;
#ifdef STDTHREAD
                        m_pPool->QueueAndWake(ha, currBlockInfo.fourFn);
#else
#ifdef OPENMPTHREAD
                        currBlockInfo.fourFn(ha);
#endif
#endif
                    }

#ifdef STDTHREAD 
                    m_pPool->Drain();
#endif
                }

                else if (rowsPerBlock == 1)
                {
#ifdef OPENMPTHREAD
#pragma omp parallel for
#endif
                    for (int startRow = 0; startRow < m; ++startRow)
                    {
                        ha.startRow = startRow;
#ifdef STDTHREAD
                        m_pPool->QueueAndWake(ha, currBlockInfo.oneFn);
#else
#ifdef OPENMPTHREAD
                        currBlockInfo.oneFn(ha);
#endif
#endif
                    }
#ifdef STDTHREAD
                    m_pPool->Drain();
#endif
                }
                else
                {
                    throw std::runtime_error("Illegal setting for rowsPerBlock");
                }

            }

        }

        if (blocks1 > 0)
        {
            ScalarAT* pA = newA + offsetA1;
            for (int startRow = 0; startRow < m; startRow++)
            {
                ScalarBT* pB = B + offsetB1;
                for (int c = 0; c < n; ++c)
                {
                    C[RowColToOffset(startRow, c, n)] += referenceKernel(pA, pB, blocks1);
                    pB += blocks1;
                }
                pA += blocks1;

            }
        }

        FreeAlignedMatrix(newA);
    }
}



}}}// end namespace
