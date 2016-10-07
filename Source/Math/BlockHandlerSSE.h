//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#pragma once
#include "BlockMultiplierPlatform.h"
#include <emmintrin.h>
#include <cassert>
#include <cstdint>
#include <thread>
#include "BlockMultiplierMatrixUtil.h"
#define FOR_CNTK
#ifdef FOR_CNTK
#include "CommonMatrix.h"
#endif

namespace Microsoft { namespace MSR { namespace CNTK {
//Handles block multiplications using SSE2 instructions (128-bit data path)
//Utility class used by block matrix multiplier, which calls these functions for each
//block multiplication of various sizes.
class MATH_API BlockHandlerSSE
{
    private:
        FORCEINLINE static void kernelsse8x4(__m128i xmmRow0, __m128i xmmRow1, __m128i xmmRow2, __m128i xmmRow3,
                                             short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4);
        FORCEINLINE static void kernelsse16x4(__m128i xmmRow0B0a, __m128i xmmRow0B0b,
                                              __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow2B0a, __m128i xmmRow2B0b,
                                              __m128i xmmRow3B0a, __m128i xmmRow3B0b, short* B, 
                                              __m128i* return1, __m128i* return2, __m128i* return3, __m128i * return4);
        FORCEINLINE static void kernelsse32x4(__m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
                                              __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow1B0c, __m128i xmmRow1B0d,
                                              __m128i xmmRow2B0a, __m128i xmmRow2B0b, __m128i xmmRow2B0c, __m128i xmmRow2B0d,
                                              __m128i xmmRow3B0a, __m128i xmmRow3B0b, __m128i xmmRow3B0c, __m128i xmmRow3B0d,
                                              short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4);
        FORCEINLINE static void kernelsse64x4(
                                              __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
                                              __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
                                              __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow1B0c, __m128i xmmRow1B0d,
                                              __m128i xmmRow1B0e, __m128i xmmRow1B0f, __m128i xmmRow1B0g, __m128i xmmRow1B0h,
                                              __m128i xmmRow2B0a, __m128i xmmRow2B0b, __m128i xmmRow2B0c, __m128i xmmRow2B0d,
                                              __m128i xmmRow2B0e, __m128i xmmRow2B0f, __m128i xmmRow2B0g, __m128i xmmRow2B0h,
                                              __m128i xmmRow3B0a, __m128i xmmRow3B0b, __m128i xmmRow3B0c, __m128i xmmRow3B0d,
                                              __m128i xmmRow3B0e, __m128i xmmRow3B0f, __m128i xmmRow3B0g, __m128i xmmRow3B0h,
                                              short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4);
        FORCEINLINE static void kernelsse128x4(
                                               __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
                                               __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
                                               __m128i xmmRow0B0i, __m128i xmmRow0B0j, __m128i xmmRow0B0k, __m128i xmmRow0B0l,
                                               __m128i xmmRow0B0m, __m128i xmmRow0B0n, __m128i xmmRow0B0o, __m128i xmmRow0B0p,
                                               __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow1B0c, __m128i xmmRow1B0d,
                                               __m128i xmmRow1B0e, __m128i xmmRow1B0f, __m128i xmmRow1B0g, __m128i xmmRow1B0h,
                                               __m128i xmmRow1B0i, __m128i xmmRow1B0j, __m128i xmmRow1B0k, __m128i xmmRow1B0l,
                                               __m128i xmmRow1B0m, __m128i xmmRow1B0n, __m128i xmmRow1B0o, __m128i xmmRow1B0p,
                                               __m128i xmmRow2B0a, __m128i xmmRow2B0b, __m128i xmmRow2B0c, __m128i xmmRow2B0d,
                                               __m128i xmmRow2B0e, __m128i xmmRow2B0f, __m128i xmmRow2B0g, __m128i xmmRow2B0h,
                                               __m128i xmmRow2B0i, __m128i xmmRow2B0j, __m128i xmmRow2B0k, __m128i xmmRow2B0l,
                                               __m128i xmmRow2B0m, __m128i xmmRow2B0n, __m128i xmmRow2B0o, __m128i xmmRow2B0p,
                                               __m128i xmmRow3B0a, __m128i xmmRow3B0b, __m128i xmmRow3B0c, __m128i xmmRow3B0d,
                                               __m128i xmmRow3B0e, __m128i xmmRow3B0f, __m128i xmmRow3B0g, __m128i xmmRow3B0h,
                                               __m128i xmmRow3B0i, __m128i xmmRow3B0j, __m128i xmmRow3B0k, __m128i xmmRow3B0l,
                                               __m128i xmmRow3B0m, __m128i xmmRow3B0n, __m128i xmmRow3B0o, __m128i xmmRow3B0p,
                                               short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4);

        FORCEINLINE static void kernelsse8x1(__m128i xmmRow0,
                                             short* B, __m128i* return1);
        FORCEINLINE static void kernelsse16x1(__m128i xmmRow0B0a, __m128i xmmRow0B0b,
                                              short* B, __m128i* return1);
        FORCEINLINE static void kernelsse32x1(__m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
                                              short* B, __m128i* return1);
        FORCEINLINE static void kernelsse64x1(
                                               __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
                                               __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
                                               short* B, __m128i* return1);
        FORCEINLINE static void kernelsse128x1(
                                               __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
                                               __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
                                              __m128i xmmRow0B0i, __m128i xmmRow0B0j, __m128i xmmRow0B0k, __m128i xmmRow0B0l,
                                              __m128i xmmRow0B0m, __m128i xmmRow0B0n, __m128i xmmRow0B0o, __m128i xmmRow0B0p,
                                             short* B, __m128i* return1);

        //TODO: Should these be refactored somewhere else? Any BlockHandler will need access to these functions.
        //Separate class with static functions? Maybe move the Block rewriting functions as well as these to a new
        //static class.
        static int RowToColOffsetRewrittenB(int col, int kOffset, int blockSize, int origCols);
        static int RowToColOffsetRewrittenA(int row, int kOffset, int blockSize, int rowsPerBlock, int origCols);
    public:
        typedef __m128i VectorT;
        typedef int16_t ScalarAT;
        typedef int16_t ScalarBT;
        typedef int32_t ScalarCT;
        FORCEINLINE static void HandleBlock8x4(int currBlock, int startRow, int k, int n, short* newA, short* B, int blockCnt,
                                               __m128i* resultStorage);
        FORCEINLINE static void HandleBlock16x4(int currBlock, int startRow, int k, int n, short* newA, short* B,int blockCnt,
                                                __m128i* resultStorage);
        FORCEINLINE static void HandleBlock32x4(int currBlock, int startRow, int k, int n, short* newA, short* B, int blockCnt,
                                                __m128i* resultStorage);
        FORCEINLINE static void HandleBlock64x4(int currBlock, int startRow, int k, int n, short* newA, short* B, int blockCnt, 
                                                __m128i* resultStorage);
        FORCEINLINE static void HandleBlock128x4(int currBlock, int startRow, int k, int n, short* newA, short* B,
                                                 int blockCnt, __m128i* resultStorage, VectorT* subtractMe);
        FORCEINLINE static void HandleBlock128x1(int currBlock, int startRow, int k, int n, short* newA, short* B,
                                                 int blockCnt, __m128i* resultStorage, VectorT* subtractMe);
        FORCEINLINE static void HandleBlock8x1(int currBlock, int startRow, int k, int n, short* newA, short* B, int blockCnt,
                                               __m128i* resultStorage);
        FORCEINLINE static void HandleBlock16x1(int currBlock, int startRow, int k, int n, short* newA, short* B, int blockCnt,
                                                __m128i* resultStorage);
        FORCEINLINE static void HandleBlock32x1(int currBlock, int startRow, int k, int n, short* newA, short* B, int blockCnt,
                                                __m128i* resultStorage);
        FORCEINLINE static void HandleBlock64x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  int blockCnt, 
                                                __m128i* resultStorage);
        static VectorT* PrepareExtraB(const ScalarBT* prepareMe, int k, int n)
        {
            prepareMe;  k; n; //warning re. unreferenced params
            return nullptr;
        }
        static void FreePreparedB(VectorT* freeMe) { freeMe;  assert(nullptr == freeMe); }

};

//Unfortunately all of these static inline function definitions need to be in the header file.

//Load functions - these functions read in one block
//from four consecutive rows starting at currA

#define LOAD_8x1 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);


#define LOAD_8x4 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r1b0a = _mm_load_si128((__m128i*)currA + 1);\
__m128i r2b0a = _mm_load_si128((__m128i*)currA + 2);\
__m128i r3b0a = _mm_load_si128((__m128i*)currA + 3);\

#define LOAD_16x1 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);

#define LOAD_16x4 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r1b0a = _mm_load_si128((__m128i*)currA + 2);\
__m128i r1b0b = _mm_load_si128((__m128i*)currA + 3);\
__m128i r2b0a = _mm_load_si128((__m128i*)currA + 4);\
__m128i r2b0b = _mm_load_si128((__m128i*)currA + 5);\
__m128i r3b0a = _mm_load_si128((__m128i*)currA + 6);\
__m128i r3b0b = _mm_load_si128((__m128i*)currA + 7);


#define LOAD_32x1 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r0b0c = _mm_load_si128((__m128i*)currA + 2);\
__m128i r0b0d = _mm_load_si128((__m128i*)currA + 3);

#define LOAD_32x4 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r0b0c = _mm_load_si128((__m128i*)currA + 2);\
__m128i r0b0d = _mm_load_si128((__m128i*)currA + 3);\
\
__m128i r1b0a = _mm_load_si128((__m128i*)currA + 4);\
__m128i r1b0b = _mm_load_si128((__m128i*)currA + 5);\
__m128i r1b0c = _mm_load_si128((__m128i*)currA + 6);\
__m128i r1b0d = _mm_load_si128((__m128i*)currA + 7);\
\
__m128i r2b0a = _mm_load_si128((__m128i*)currA + 8);\
__m128i r2b0b = _mm_load_si128((__m128i*)currA + 9);\
__m128i r2b0c = _mm_load_si128((__m128i*)currA + 10);\
__m128i r2b0d = _mm_load_si128((__m128i*)currA + 11);\
\
__m128i r3b0a = _mm_load_si128((__m128i*)currA + 12);\
__m128i r3b0b = _mm_load_si128((__m128i*)currA + 13);\
__m128i r3b0c = _mm_load_si128((__m128i*)currA + 14);\
__m128i r3b0d = _mm_load_si128((__m128i*)currA + 15);


#define LOAD_64x1 \
__m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r0b0c = _mm_load_si128((__m128i*)currA + 2);\
__m128i r0b0d = _mm_load_si128((__m128i*)currA + 3);\
__m128i r0b0e = _mm_load_si128((__m128i*)currA + 4);\
__m128i r0b0f = _mm_load_si128((__m128i*)currA + 5);\
__m128i r0b0g = _mm_load_si128((__m128i*)currA + 6);\
__m128i r0b0h = _mm_load_si128((__m128i*)currA + 7);\

#define LOAD_64x4 \
__m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r0b0c = _mm_load_si128((__m128i*)currA + 2);\
__m128i r0b0d = _mm_load_si128((__m128i*)currA + 3);\
__m128i r0b0e = _mm_load_si128((__m128i*)currA + 4);\
__m128i r0b0f = _mm_load_si128((__m128i*)currA + 5);\
__m128i r0b0g = _mm_load_si128((__m128i*)currA + 6);\
__m128i r0b0h = _mm_load_si128((__m128i*)currA + 7);\
\
__m128i r1b0a = _mm_load_si128((__m128i*)currA + 8);\
__m128i r1b0b = _mm_load_si128((__m128i*)currA + 9);\
__m128i r1b0c = _mm_load_si128((__m128i*)currA + 10);\
__m128i r1b0d = _mm_load_si128((__m128i*)currA + 11);\
__m128i r1b0e = _mm_load_si128((__m128i*)currA + 12);\
__m128i r1b0f = _mm_load_si128((__m128i*)currA + 13);\
__m128i r1b0g = _mm_load_si128((__m128i*)currA + 14);\
__m128i r1b0h = _mm_load_si128((__m128i*)currA + 15);\
\
__m128i r2b0a = _mm_load_si128((__m128i*)currA + 16);\
__m128i r2b0b = _mm_load_si128((__m128i*)currA + 17);\
__m128i r2b0c = _mm_load_si128((__m128i*)currA + 18);\
__m128i r2b0d = _mm_load_si128((__m128i*)currA + 19);\
__m128i r2b0e = _mm_load_si128((__m128i*)currA + 20);\
__m128i r2b0f = _mm_load_si128((__m128i*)currA + 21);\
__m128i r2b0g = _mm_load_si128((__m128i*)currA + 22);\
__m128i r2b0h = _mm_load_si128((__m128i*)currA + 23);\
\
__m128i r3b0a = _mm_load_si128((__m128i*)currA + 24);\
__m128i r3b0b = _mm_load_si128((__m128i*)currA + 25);\
__m128i r3b0c = _mm_load_si128((__m128i*)currA + 26);\
__m128i r3b0d = _mm_load_si128((__m128i*)currA + 27);\
__m128i r3b0e = _mm_load_si128((__m128i*)currA + 28);\
__m128i r3b0f = _mm_load_si128((__m128i*)currA + 29);\
__m128i r3b0g = _mm_load_si128((__m128i*)currA + 30);\
__m128i r3b0h = _mm_load_si128((__m128i*)currA + 31);

#define LOAD3_128x4 \
    __m128i r0b0a3 = _mm_load_si128((__m128i*)currA3);\
__m128i r0b0b3 = _mm_load_si128((__m128i*)currA3 + 1);\
__m128i r0b0c3 = _mm_load_si128((__m128i*)currA3 + 2);\
__m128i r0b0d3 = _mm_load_si128((__m128i*)currA3 + 3);\
__m128i r0b0e3 = _mm_load_si128((__m128i*)currA3 + 4);\
__m128i r0b0f3 = _mm_load_si128((__m128i*)currA3 + 5);\
__m128i r0b0g3 = _mm_load_si128((__m128i*)currA3 + 6);\
__m128i r0b0h3 = _mm_load_si128((__m128i*)currA3 + 7);\
__m128i r0b0i3 = _mm_load_si128((__m128i*)currA3 + 8);\
__m128i r0b0j3 = _mm_load_si128((__m128i*)currA3 + 9);\
__m128i r0b0k3 = _mm_load_si128((__m128i*)currA3 + 10);\
__m128i r0b0l3 = _mm_load_si128((__m128i*)currA3 + 11);\
__m128i r0b0m3 = _mm_load_si128((__m128i*)currA3 + 12);\
__m128i r0b0n3 = _mm_load_si128((__m128i*)currA3 + 13);\
__m128i r0b0o3 = _mm_load_si128((__m128i*)currA3 + 14);\
__m128i r0b0p3 = _mm_load_si128((__m128i*)currA3 + 15);\
\
__m128i r1b0a3 = _mm_load_si128((__m128i*)currA3 + 16);\
__m128i r1b0b3 = _mm_load_si128((__m128i*)currA3 + 17);\
__m128i r1b0c3 = _mm_load_si128((__m128i*)currA3 + 18);\
__m128i r1b0d3 = _mm_load_si128((__m128i*)currA3 + 19);\
__m128i r1b0e3 = _mm_load_si128((__m128i*)currA3 + 20);\
__m128i r1b0f3 = _mm_load_si128((__m128i*)currA3 + 21);\
__m128i r1b0g3 = _mm_load_si128((__m128i*)currA3 + 22);\
__m128i r1b0h3 = _mm_load_si128((__m128i*)currA3 + 23);\
__m128i r1b0i3 = _mm_load_si128((__m128i*)currA3 + 24);\
__m128i r1b0j3 = _mm_load_si128((__m128i*)currA3 + 25);\
__m128i r1b0k3 = _mm_load_si128((__m128i*)currA3 + 26);\
__m128i r1b0l3 = _mm_load_si128((__m128i*)currA3 + 27);\
__m128i r1b0m3 = _mm_load_si128((__m128i*)currA3 + 28);\
__m128i r1b0n3 = _mm_load_si128((__m128i*)currA3 + 29);\
__m128i r1b0o3 = _mm_load_si128((__m128i*)currA3 + 30);\
__m128i r1b0p3 = _mm_load_si128((__m128i*)currA3 + 31);\
\
__m128i r2b0a3 = _mm_load_si128((__m128i*)currA3 + 32);\
__m128i r2b0b3 = _mm_load_si128((__m128i*)currA3 + 33);\
__m128i r2b0c3 = _mm_load_si128((__m128i*)currA3 + 34);\
__m128i r2b0d3 = _mm_load_si128((__m128i*)currA3 + 35);\
__m128i r2b0e3 = _mm_load_si128((__m128i*)currA3 + 36);\
__m128i r2b0f3 = _mm_load_si128((__m128i*)currA3 + 37);\
__m128i r2b0g3 = _mm_load_si128((__m128i*)currA3 + 38);\
__m128i r2b0h3 = _mm_load_si128((__m128i*)currA3 + 39);\
__m128i r2b0i3 = _mm_load_si128((__m128i*)currA3 + 40);\
__m128i r2b0j3 = _mm_load_si128((__m128i*)currA3 + 41);\
__m128i r2b0k3 = _mm_load_si128((__m128i*)currA3 + 42);\
__m128i r2b0l3 = _mm_load_si128((__m128i*)currA3 + 43);\
__m128i r2b0m3 = _mm_load_si128((__m128i*)currA3 + 44);\
__m128i r2b0n3 = _mm_load_si128((__m128i*)currA3 + 45);\
__m128i r2b0o3 = _mm_load_si128((__m128i*)currA3 + 46);\
__m128i r2b0p3 = _mm_load_si128((__m128i*)currA3 + 47);\
\
__m128i r3b0a3 = _mm_load_si128((__m128i*)currA3 + 48);\
__m128i r3b0b3 = _mm_load_si128((__m128i*)currA3 + 49);\
__m128i r3b0c3 = _mm_load_si128((__m128i*)currA3 + 50);\
__m128i r3b0d3 = _mm_load_si128((__m128i*)currA3 + 51);\
__m128i r3b0e3 = _mm_load_si128((__m128i*)currA3 + 52);\
__m128i r3b0f3 = _mm_load_si128((__m128i*)currA3 + 53);\
__m128i r3b0g3 = _mm_load_si128((__m128i*)currA3 + 54);\
__m128i r3b0h3 = _mm_load_si128((__m128i*)currA3 + 55);\
__m128i r3b0i3 = _mm_load_si128((__m128i*)currA3 + 56);\
__m128i r3b0j3 = _mm_load_si128((__m128i*)currA3 + 57);\
__m128i r3b0k3 = _mm_load_si128((__m128i*)currA3 + 58);\
__m128i r3b0l3 = _mm_load_si128((__m128i*)currA3 + 59);\
__m128i r3b0m3 = _mm_load_si128((__m128i*)currA3 + 60);\
__m128i r3b0n3 = _mm_load_si128((__m128i*)currA3 + 61);\
__m128i r3b0o3 = _mm_load_si128((__m128i*)currA3 + 62);\
__m128i r3b0p3 = _mm_load_si128((__m128i*)currA3 + 63);

#define DECL2_128x1 \
    __m128i r0b0a2;\
__m128i r0b0b2;\
__m128i r0b0c2;\
__m128i r0b0d2;\
__m128i r0b0e2;\
__m128i r0b0f2;\
__m128i r0b0g2;\
__m128i r0b0h2;\
__m128i r0b0i2;\
__m128i r0b0j2;\
__m128i r0b0k2;\
__m128i r0b0l2;\
__m128i r0b0m2;\
__m128i r0b0n2;\
__m128i r0b0o2;\
__m128i r0b0p2;

#define LOAD2_128x1 \
    r0b0a2 = _mm_load_si128((__m128i*)currA2);\
r0b0b2 = _mm_load_si128((__m128i*)currA2 + 1);\
r0b0c2 = _mm_load_si128((__m128i*)currA2 + 2);\
r0b0d2 = _mm_load_si128((__m128i*)currA2 + 3);\
r0b0e2 = _mm_load_si128((__m128i*)currA2 + 4);\
r0b0f2 = _mm_load_si128((__m128i*)currA2 + 5);\
r0b0g2 = _mm_load_si128((__m128i*)currA2 + 6);\
r0b0h2 = _mm_load_si128((__m128i*)currA2 + 7);\
r0b0i2 = _mm_load_si128((__m128i*)currA2 + 8);\
r0b0j2 = _mm_load_si128((__m128i*)currA2 + 9);\
r0b0k2 = _mm_load_si128((__m128i*)currA2 + 10);\
r0b0l2 = _mm_load_si128((__m128i*)currA2 + 11);\
r0b0m2 = _mm_load_si128((__m128i*)currA2 + 12);\
r0b0n2 = _mm_load_si128((__m128i*)currA2 + 13);\
r0b0o2 = _mm_load_si128((__m128i*)currA2 + 14);\
r0b0p2 = _mm_load_si128((__m128i*)currA2 + 15);

#define DECL2_128x4\
    __m128i r0b0a2;\
__m128i r0b0b2;\
__m128i r0b0c2;\
__m128i r0b0d2;\
__m128i r0b0e2;\
__m128i r0b0f2;\
__m128i r0b0g2;\
__m128i r0b0h2;\
__m128i r0b0i2;\
__m128i r0b0j2;\
__m128i r0b0k2;\
__m128i r0b0l2;\
__m128i r0b0m2;\
__m128i r0b0n2;\
__m128i r0b0o2;\
__m128i r0b0p2;\
\
__m128i r1b0a2;\
__m128i r1b0b2;\
__m128i r1b0c2;\
__m128i r1b0d2;\
__m128i r1b0e2;\
__m128i r1b0f2;\
__m128i r1b0g2;\
__m128i r1b0h2;\
__m128i r1b0i2;\
__m128i r1b0j2;\
__m128i r1b0k2;\
__m128i r1b0l2;\
__m128i r1b0m2;\
__m128i r1b0n2;\
__m128i r1b0o2;\
__m128i r1b0p2;\
\
__m128i r2b0a2;\
__m128i r2b0b2;\
__m128i r2b0c2;\
__m128i r2b0d2;\
__m128i r2b0e2;\
__m128i r2b0f2;\
__m128i r2b0g2;\
__m128i r2b0h2;\
__m128i r2b0i2;\
__m128i r2b0j2;\
__m128i r2b0k2;\
__m128i r2b0l2;\
__m128i r2b0m2;\
__m128i r2b0n2;\
__m128i r2b0o2;\
__m128i r2b0p2;\
\
__m128i r3b0a2;\
__m128i r3b0b2;\
__m128i r3b0c2;\
__m128i r3b0d2;\
__m128i r3b0e2;\
__m128i r3b0f2;\
__m128i r3b0g2;\
__m128i r3b0h2;\
__m128i r3b0i2;\
__m128i r3b0j2;\
__m128i r3b0k2;\
__m128i r3b0l2;\
__m128i r3b0m2;\
__m128i r3b0n2;\
__m128i r3b0o2;\
__m128i r3b0p2;

#define LOAD2_128x4 \
r0b0a2 = _mm_load_si128((__m128i*)currA2);\
r0b0b2 = _mm_load_si128((__m128i*)currA2 + 1);\
r0b0c2 = _mm_load_si128((__m128i*)currA2 + 2);\
r0b0d2 = _mm_load_si128((__m128i*)currA2 + 3);\
r0b0e2 = _mm_load_si128((__m128i*)currA2 + 4);\
r0b0f2 = _mm_load_si128((__m128i*)currA2 + 5);\
r0b0g2 = _mm_load_si128((__m128i*)currA2 + 6);\
r0b0h2 = _mm_load_si128((__m128i*)currA2 + 7);\
r0b0i2 = _mm_load_si128((__m128i*)currA2 + 8);\
r0b0j2 = _mm_load_si128((__m128i*)currA2 + 9);\
r0b0k2 = _mm_load_si128((__m128i*)currA2 + 10);\
r0b0l2 = _mm_load_si128((__m128i*)currA2 + 11);\
r0b0m2 = _mm_load_si128((__m128i*)currA2 + 12);\
r0b0n2 = _mm_load_si128((__m128i*)currA2 + 13);\
r0b0o2 = _mm_load_si128((__m128i*)currA2 + 14);\
r0b0p2 = _mm_load_si128((__m128i*)currA2 + 15);\
\
r1b0a2 = _mm_load_si128((__m128i*)currA2 + 16);\
r1b0b2 = _mm_load_si128((__m128i*)currA2 + 17);\
r1b0c2 = _mm_load_si128((__m128i*)currA2 + 18);\
r1b0d2 = _mm_load_si128((__m128i*)currA2 + 19);\
r1b0e2 = _mm_load_si128((__m128i*)currA2 + 20);\
r1b0f2 = _mm_load_si128((__m128i*)currA2 + 21);\
r1b0g2 = _mm_load_si128((__m128i*)currA2 + 22);\
r1b0h2 = _mm_load_si128((__m128i*)currA2 + 23);\
r1b0i2 = _mm_load_si128((__m128i*)currA2 + 24);\
r1b0j2 = _mm_load_si128((__m128i*)currA2 + 25);\
r1b0k2 = _mm_load_si128((__m128i*)currA2 + 26);\
r1b0l2 = _mm_load_si128((__m128i*)currA2 + 27);\
r1b0m2 = _mm_load_si128((__m128i*)currA2 + 28);\
r1b0n2 = _mm_load_si128((__m128i*)currA2 + 29);\
r1b0o2 = _mm_load_si128((__m128i*)currA2 + 30);\
r1b0p2 = _mm_load_si128((__m128i*)currA2 + 31);\
\
r2b0a2 = _mm_load_si128((__m128i*)currA2 + 32);\
r2b0b2 = _mm_load_si128((__m128i*)currA2 + 33);\
r2b0c2 = _mm_load_si128((__m128i*)currA2 + 34);\
r2b0d2 = _mm_load_si128((__m128i*)currA2 + 35);\
r2b0e2 = _mm_load_si128((__m128i*)currA2 + 36);\
r2b0f2 = _mm_load_si128((__m128i*)currA2 + 37);\
r2b0g2 = _mm_load_si128((__m128i*)currA2 + 38);\
r2b0h2 = _mm_load_si128((__m128i*)currA2 + 39);\
r2b0i2 = _mm_load_si128((__m128i*)currA2 + 40);\
r2b0j2 = _mm_load_si128((__m128i*)currA2 + 41);\
r2b0k2 = _mm_load_si128((__m128i*)currA2 + 42);\
r2b0l2 = _mm_load_si128((__m128i*)currA2 + 43);\
r2b0m2 = _mm_load_si128((__m128i*)currA2 + 44);\
r2b0n2 = _mm_load_si128((__m128i*)currA2 + 45);\
r2b0o2 = _mm_load_si128((__m128i*)currA2 + 46);\
r2b0p2 = _mm_load_si128((__m128i*)currA2 + 47);\
\
r3b0a2 = _mm_load_si128((__m128i*)currA2 + 48);\
r3b0b2 = _mm_load_si128((__m128i*)currA2 + 49);\
r3b0c2 = _mm_load_si128((__m128i*)currA2 + 50);\
r3b0d2 = _mm_load_si128((__m128i*)currA2 + 51);\
r3b0e2 = _mm_load_si128((__m128i*)currA2 + 52);\
r3b0f2 = _mm_load_si128((__m128i*)currA2 + 53);\
r3b0g2 = _mm_load_si128((__m128i*)currA2 + 54);\
r3b0h2 = _mm_load_si128((__m128i*)currA2 + 55);\
r3b0i2 = _mm_load_si128((__m128i*)currA2 + 56);\
r3b0j2 = _mm_load_si128((__m128i*)currA2 + 57);\
r3b0k2 = _mm_load_si128((__m128i*)currA2 + 58);\
r3b0l2 = _mm_load_si128((__m128i*)currA2 + 59);\
r3b0m2 = _mm_load_si128((__m128i*)currA2 + 60);\
r3b0n2 = _mm_load_si128((__m128i*)currA2 + 61);\
r3b0o2 = _mm_load_si128((__m128i*)currA2 + 62);\
r3b0p2 = _mm_load_si128((__m128i*)currA2 + 63);

#define LOAD_128x1 \
__m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r0b0c = _mm_load_si128((__m128i*)currA + 2);\
__m128i r0b0d = _mm_load_si128((__m128i*)currA + 3);\
__m128i r0b0e = _mm_load_si128((__m128i*)currA + 4);\
__m128i r0b0f = _mm_load_si128((__m128i*)currA + 5);\
__m128i r0b0g = _mm_load_si128((__m128i*)currA + 6);\
__m128i r0b0h = _mm_load_si128((__m128i*)currA + 7);\
__m128i r0b0i = _mm_load_si128((__m128i*)currA + 8);\
__m128i r0b0j = _mm_load_si128((__m128i*)currA + 9);\
__m128i r0b0k = _mm_load_si128((__m128i*)currA + 10);\
__m128i r0b0l = _mm_load_si128((__m128i*)currA + 11);\
__m128i r0b0m = _mm_load_si128((__m128i*)currA + 12);\
__m128i r0b0n = _mm_load_si128((__m128i*)currA + 13);\
__m128i r0b0o = _mm_load_si128((__m128i*)currA + 14);\
__m128i r0b0p = _mm_load_si128((__m128i*)currA + 15);



#define LOAD_128x4 \
__m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r0b0b = _mm_load_si128((__m128i*)currA + 1);\
__m128i r0b0c = _mm_load_si128((__m128i*)currA + 2);\
__m128i r0b0d = _mm_load_si128((__m128i*)currA + 3);\
__m128i r0b0e = _mm_load_si128((__m128i*)currA + 4);\
__m128i r0b0f = _mm_load_si128((__m128i*)currA + 5);\
__m128i r0b0g = _mm_load_si128((__m128i*)currA + 6);\
__m128i r0b0h = _mm_load_si128((__m128i*)currA + 7);\
__m128i r0b0i = _mm_load_si128((__m128i*)currA + 8);\
__m128i r0b0j = _mm_load_si128((__m128i*)currA + 9);\
__m128i r0b0k = _mm_load_si128((__m128i*)currA + 10);\
__m128i r0b0l = _mm_load_si128((__m128i*)currA + 11);\
__m128i r0b0m = _mm_load_si128((__m128i*)currA + 12);\
__m128i r0b0n = _mm_load_si128((__m128i*)currA + 13);\
__m128i r0b0o = _mm_load_si128((__m128i*)currA + 14);\
__m128i r0b0p = _mm_load_si128((__m128i*)currA + 15);\
\
__m128i r1b0a = _mm_load_si128((__m128i*)currA + 16);\
__m128i r1b0b = _mm_load_si128((__m128i*)currA + 17);\
__m128i r1b0c = _mm_load_si128((__m128i*)currA + 18);\
__m128i r1b0d = _mm_load_si128((__m128i*)currA + 19);\
__m128i r1b0e = _mm_load_si128((__m128i*)currA + 20);\
__m128i r1b0f = _mm_load_si128((__m128i*)currA + 21);\
__m128i r1b0g = _mm_load_si128((__m128i*)currA + 22);\
__m128i r1b0h = _mm_load_si128((__m128i*)currA + 23);\
__m128i r1b0i = _mm_load_si128((__m128i*)currA + 24);\
__m128i r1b0j = _mm_load_si128((__m128i*)currA + 25);\
__m128i r1b0k = _mm_load_si128((__m128i*)currA + 26);\
__m128i r1b0l = _mm_load_si128((__m128i*)currA + 27);\
__m128i r1b0m = _mm_load_si128((__m128i*)currA + 28);\
__m128i r1b0n = _mm_load_si128((__m128i*)currA + 29);\
__m128i r1b0o = _mm_load_si128((__m128i*)currA + 30);\
__m128i r1b0p = _mm_load_si128((__m128i*)currA + 31);\
\
__m128i r2b0a = _mm_load_si128((__m128i*)currA + 32);\
__m128i r2b0b = _mm_load_si128((__m128i*)currA + 33);\
__m128i r2b0c = _mm_load_si128((__m128i*)currA + 34);\
__m128i r2b0d = _mm_load_si128((__m128i*)currA + 35);\
__m128i r2b0e = _mm_load_si128((__m128i*)currA + 36);\
__m128i r2b0f = _mm_load_si128((__m128i*)currA + 37);\
__m128i r2b0g = _mm_load_si128((__m128i*)currA + 38);\
__m128i r2b0h = _mm_load_si128((__m128i*)currA + 39);\
__m128i r2b0i = _mm_load_si128((__m128i*)currA + 40);\
__m128i r2b0j = _mm_load_si128((__m128i*)currA + 41);\
__m128i r2b0k = _mm_load_si128((__m128i*)currA + 42);\
__m128i r2b0l = _mm_load_si128((__m128i*)currA + 43);\
__m128i r2b0m = _mm_load_si128((__m128i*)currA + 44);\
__m128i r2b0n = _mm_load_si128((__m128i*)currA + 45);\
__m128i r2b0o = _mm_load_si128((__m128i*)currA + 46);\
__m128i r2b0p = _mm_load_si128((__m128i*)currA + 47);\
\
__m128i r3b0a = _mm_load_si128((__m128i*)currA + 48);\
__m128i r3b0b = _mm_load_si128((__m128i*)currA + 49);\
__m128i r3b0c = _mm_load_si128((__m128i*)currA + 50);\
__m128i r3b0d = _mm_load_si128((__m128i*)currA + 51);\
__m128i r3b0e = _mm_load_si128((__m128i*)currA + 52);\
__m128i r3b0f = _mm_load_si128((__m128i*)currA + 53);\
__m128i r3b0g = _mm_load_si128((__m128i*)currA + 54);\
__m128i r3b0h = _mm_load_si128((__m128i*)currA + 55);\
__m128i r3b0i = _mm_load_si128((__m128i*)currA + 56);\
__m128i r3b0j = _mm_load_si128((__m128i*)currA + 57);\
__m128i r3b0k = _mm_load_si128((__m128i*)currA + 58);\
__m128i r3b0l = _mm_load_si128((__m128i*)currA + 59);\
__m128i r3b0m = _mm_load_si128((__m128i*)currA + 60);\
__m128i r3b0n = _mm_load_si128((__m128i*)currA + 61);\
__m128i r3b0o = _mm_load_si128((__m128i*)currA + 62);\
__m128i r3b0p = _mm_load_si128((__m128i*)currA + 63);


//Handler functions. These are called once for each row block.
//The row elements for one block of four rows are loaded into memory.
//Then we iterate over columns, adding partial dotproducts to the
//target matrix.
FORCEINLINE void BlockHandlerSSE::HandleBlock8x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int blockCnt, __m128i* resultStorage)
{
    //Avoid warning 3861
    blockCnt;
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 8, 4, k);
    short* currA = &newA[aOffset];
    LOAD_8x4;
    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 8, n)];
        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum2 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum3 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum4 = _mm_set_epi32(0, 0, 0, 0);
        kernelsse8x4(r0b0a, r1b0a, r2b0a, r3b0a,
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}

FORCEINLINE void BlockHandlerSSE::HandleBlock8x1(int currBlock, int startRow, int k, int n, short* newA, short* B, int /*blockCnt*/,
        __m128i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 8, 4, k);
    short* currA = &newA[aOffset];
    LOAD_8x1;
    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 8, n)];
        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        kernelsse8x1(r0b0a, 
                currB, &accum1);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
    }
}

FORCEINLINE void BlockHandlerSSE::HandleBlock16x4(int currBlock, int startRow, int k, int n, short* newA, short* B, int /*blockCnt*/,
        __m128i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 16, 1, k);
    short* currA = &newA[aOffset];
    LOAD_16x4;
    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 16, n)];

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum2 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum3 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum4 = _mm_set_epi32(0, 0, 0, 0);
        kernelsse16x4(
                r0b0a, r0b0b,
                r1b0a, r1b0b,
                r2b0a, r2b0b,
                r3b0a, r3b0b,
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}

FORCEINLINE void BlockHandlerSSE::HandleBlock16x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  int /*blockCnt*/,
        __m128i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 16, 1, k);
    short* currA = &newA[aOffset];
    LOAD_16x1;
    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 16, n)];

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);

        kernelsse16x1(
                r0b0a, r0b0b, 
                currB, &accum1);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
    }
}

FORCEINLINE void BlockHandlerSSE::HandleBlock32x4(int currBlock, int startRow, int k, int n, short* newA, short* B,  int /*blockCnt*/,
        __m128i* resultStorage)
{

    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 32, 1, k);
    short* currA = &newA[aOffset];
    LOAD_32x4;
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 32, n)];

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum2 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum3 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum4 = _mm_set_epi32(0, 0, 0, 0);
        kernelsse32x4(
                r0b0a, r0b0b, r0b0c, r0b0d,
                r1b0a, r1b0b, r1b0c, r1b0d,
                r2b0a, r2b0b, r2b0c, r2b0d,
                r3b0a, r3b0b, r3b0c, r3b0d,
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}

FORCEINLINE void BlockHandlerSSE::HandleBlock32x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  int /*blockCnt*/,
        __m128i* resultStorage)
{

    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 32, 1, k);
    short* currA = &newA[aOffset];
    LOAD_32x1;
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 32, n)];

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        kernelsse32x1(
                r0b0a, r0b0b, r0b0c, r0b0d,
                currB, &accum1);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
    }
}

FORCEINLINE void BlockHandlerSSE::HandleBlock64x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m128i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 64, 4, k);
    short* currA = &newA[aOffset];

    LOAD_64x4;


    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 64, n)];

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum2 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum3 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum4 = _mm_set_epi32(0, 0, 0, 0);

        kernelsse64x4(
                r0b0a, r0b0b, r0b0c, r0b0d, r0b0e, r0b0f, r0b0g, r0b0h,
                r1b0a, r1b0b, r1b0c, r1b0d, r1b0e, r1b0f, r1b0g, r1b0h,
                r2b0a, r2b0b, r2b0c, r2b0d, r2b0e, r2b0f, r2b0g, r2b0h,
                r3b0a, r3b0b, r3b0c, r3b0d, r3b0e, r3b0f, r3b0g, r3b0h,
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}


FORCEINLINE void BlockHandlerSSE::HandleBlock64x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  int /*blockCnt*/,
        __m128i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 64, 1, k);
    short* currA = &newA[aOffset];

    LOAD_64x1;

    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 64, n)];

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);

        kernelsse64x1(
                r0b0a, r0b0b, r0b0c, r0b0d, r0b0e, r0b0f, r0b0g, r0b0h,
                currB, &accum1);

        //Reverse write order for better locality, then transpose @ end
        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
    }
}






//I tried getting rid of the tiresome load macros and the huge number of arguments (=stack vars)
//by just passing in the base row pointer and doing the arithmetic in this function, but it is
//significantly slower so we have to live with it for now.

FORCEINLINE void BlockHandlerSSE::kernelsse8x4(__m128i xmmRow0, __m128i xmmRow1, __m128i xmmRow2, __m128i xmmRow3,
        short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4)
{
    __m128i xmmCol0 = _mm_load_si128((__m128i*)B);

    __m128i result1 = _mm_madd_epi16(xmmRow0, xmmCol0);
    __m128i result2 = _mm_madd_epi16(xmmRow1, xmmCol0);
    __m128i result3 = _mm_madd_epi16(xmmRow2, xmmCol0);
    __m128i result4 = _mm_madd_epi16(xmmRow3, xmmCol0);

    *return1 = result1;
    *return2 = result2;
    *return3 = result3;
    *return4 = result4;
}

FORCEINLINE void BlockHandlerSSE::kernelsse8x1(__m128i xmmRow0, 
        short* B, __m128i* return1)
{
    __m128i xmmCol0 = _mm_load_si128((__m128i*)B);

    __m128i result1 = _mm_madd_epi16(xmmRow0, xmmCol0);

    *return1 = result1;
}


FORCEINLINE void BlockHandlerSSE::kernelsse16x4(__m128i xmmRow0B0a, __m128i xmmRow0B0b,
        __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow2B0a, __m128i xmmRow2B0b,
        __m128i xmmRow3B0a, __m128i xmmRow3B0b, short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i * return4)
{
    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)B + 1);



    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i result1 = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);


    //Result for row 1
    __m128i r1b0axc0b0a = _mm_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m128i r1b0bxc0b0b = _mm_madd_epi16(xmmRow1B0b, xmmCol0B0b);
    __m128i result2 = _mm_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);

    //Result for row 2
    __m128i r2b0axc0b0a = _mm_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m128i r2b0bxc0b0b = _mm_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m128i result3 = _mm_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);

    //Result for row 3
    __m128i r3b0axc0b0a = _mm_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m128i r3b0bxc0b0b = _mm_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m128i result4 = _mm_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);

    //Now we can just add horizontally



    *return1 = result1;
    *return2 = result2;
    *return3 = result3;
    *return4 = result4;
}

FORCEINLINE void BlockHandlerSSE::kernelsse16x1(__m128i xmmRow0B0a, __m128i xmmRow0B0b,
        short* B, __m128i* return1)
{
    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)B + 1);



    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i result1 = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);

    //Now we can just add horizontally



    *return1 = result1;
}



FORCEINLINE void BlockHandlerSSE::kernelsse32x4(__m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
        __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow1B0c, __m128i xmmRow1B0d, __m128i xmmRow2B0a, __m128i xmmRow2B0b, __m128i xmmRow2B0c, __m128i xmmRow2B0d,
        __m128i xmmRow3B0a, __m128i xmmRow3B0b, __m128i xmmRow3B0c, __m128i xmmRow3B0d, short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i * return4)
{
    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)B + 1);
    __m128i xmmCol0B0c = _mm_load_si128((__m128i*)B + 2);
    __m128i xmmCol0B0d = _mm_load_si128((__m128i*)B + 3);



    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i r0b0cxc0b0c = _mm_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m128i r0b0dxc0b0d = _mm_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m128i result1a = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m128i result1b = _mm_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m128i result1ab = _mm_add_epi32(result1a, result1b);


    //Result for row 1
    __m128i r1b0axc0b0a = _mm_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m128i r1b0bxc0b0b = _mm_madd_epi16(xmmRow1B0b, xmmCol0B0b);
    __m128i r1b0cxc0b0c = _mm_madd_epi16(xmmRow1B0c, xmmCol0B0c);
    __m128i r1b0dxc0b0d = _mm_madd_epi16(xmmRow1B0d, xmmCol0B0d);
    __m128i result2a = _mm_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);
    __m128i result2b = _mm_add_epi32(r1b0cxc0b0c, r1b0dxc0b0d);
    __m128i result2ab = _mm_add_epi32(result2a, result2b);

    //Result for row 2
    __m128i r2b0axc0b0a = _mm_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m128i r2b0bxc0b0b = _mm_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m128i r2b0cxc0b0c = _mm_madd_epi16(xmmRow2B0c, xmmCol0B0c);
    __m128i r2b0dxc0b0d = _mm_madd_epi16(xmmRow2B0d, xmmCol0B0d);
    __m128i result3a = _mm_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);
    __m128i result3b = _mm_add_epi32(r2b0cxc0b0c, r2b0dxc0b0d);
    __m128i result3ab = _mm_add_epi32(result3a, result3b);

    //Result for row 3
    __m128i r3b0axc0b0a = _mm_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m128i r3b0bxc0b0b = _mm_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m128i r3b0cxc0b0c = _mm_madd_epi16(xmmRow3B0c, xmmCol0B0c);
    __m128i r3b0dxc0b0d = _mm_madd_epi16(xmmRow3B0d, xmmCol0B0d);
    __m128i result4a = _mm_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);
    __m128i result4b = _mm_add_epi32(r3b0cxc0b0c, r3b0dxc0b0d);
    __m128i result4ab = _mm_add_epi32(result4a, result4b);

    //Now we can just add horizontally



    *return1 = result1ab;
    *return2 = result2ab;
    *return3 = result3ab;
    *return4 = result4ab;
}

FORCEINLINE void BlockHandlerSSE::kernelsse32x1(__m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
        short* B, __m128i* return1)
{
    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)B + 1);
    __m128i xmmCol0B0c = _mm_load_si128((__m128i*)B + 2);
    __m128i xmmCol0B0d = _mm_load_si128((__m128i*)B + 3);



    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i r0b0cxc0b0c = _mm_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m128i r0b0dxc0b0d = _mm_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m128i result1a = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m128i result1b = _mm_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m128i result1ab = _mm_add_epi32(result1a, result1b);

    //Now we can just add horizontally

    *return1 = result1ab;
}




FORCEINLINE void BlockHandlerSSE::kernelsse64x4(
        __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
        __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
        __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow1B0c, __m128i xmmRow1B0d,
        __m128i xmmRow1B0e, __m128i xmmRow1B0f, __m128i xmmRow1B0g, __m128i xmmRow1B0h,
        __m128i xmmRow2B0a, __m128i xmmRow2B0b, __m128i xmmRow2B0c, __m128i xmmRow2B0d,
        __m128i xmmRow2B0e, __m128i xmmRow2B0f, __m128i xmmRow2B0g, __m128i xmmRow2B0h,
        __m128i xmmRow3B0a, __m128i xmmRow3B0b, __m128i xmmRow3B0c, __m128i xmmRow3B0d,
        __m128i xmmRow3B0e, __m128i xmmRow3B0f, __m128i xmmRow3B0g, __m128i xmmRow3B0h,
        short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4)
{
    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)(B + 8));
    __m128i xmmCol0B0c = _mm_load_si128((__m128i*)(B + 16));
    __m128i xmmCol0B0d = _mm_load_si128((__m128i*)(B + 24));
    __m128i xmmCol0B0e = _mm_load_si128((__m128i*)(B + 32));
    __m128i xmmCol0B0f = _mm_load_si128((__m128i*)(B + 40));
    __m128i xmmCol0B0g = _mm_load_si128((__m128i*)(B + 48));
    __m128i xmmCol0B0h = _mm_load_si128((__m128i*)(B + 56));



    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i r0b0cxc0b0c = _mm_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m128i r0b0dxc0b0d = _mm_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m128i r0b0exc0b0e = _mm_madd_epi16(xmmRow0B0e, xmmCol0B0e);
    __m128i r0b0fxc0b0f = _mm_madd_epi16(xmmRow0B0f, xmmCol0B0f);
    __m128i r0b0gxc0b0g = _mm_madd_epi16(xmmRow0B0g, xmmCol0B0g);
    __m128i r0b0hxc0b0h = _mm_madd_epi16(xmmRow0B0h, xmmCol0B0h);
    __m128i result1a = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m128i result1b = _mm_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m128i result1c = _mm_add_epi32(r0b0exc0b0e, r0b0fxc0b0f);
    __m128i result1d = _mm_add_epi32(r0b0gxc0b0g, r0b0hxc0b0h);
    __m128i result1ab = _mm_add_epi32(result1a, result1b);
    __m128i result1cd = _mm_add_epi32(result1c, result1d);
    __m128i result1abcd = _mm_add_epi32(result1ab, result1cd);



    //Result for row 1
    __m128i r1b0axc0b0a = _mm_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m128i r1b0bxc0b0b = _mm_madd_epi16(xmmRow1B0b, xmmCol0B0b);
    __m128i r1b0cxc0b0c = _mm_madd_epi16(xmmRow1B0c, xmmCol0B0c);
    __m128i r1b0dxc0b0d = _mm_madd_epi16(xmmRow1B0d, xmmCol0B0d);
    __m128i r1b0exc0b0e = _mm_madd_epi16(xmmRow1B0e, xmmCol0B0e);
    __m128i r1b0fxc0b0f = _mm_madd_epi16(xmmRow1B0f, xmmCol0B0f);
    __m128i r1b0gxc0b0g = _mm_madd_epi16(xmmRow1B0g, xmmCol0B0g);
    __m128i r1b0hxc0b0h = _mm_madd_epi16(xmmRow1B0h, xmmCol0B0h);
    __m128i result2a = _mm_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);
    __m128i result2b = _mm_add_epi32(r1b0cxc0b0c, r1b0dxc0b0d);
    __m128i result2c = _mm_add_epi32(r1b0exc0b0e, r1b0fxc0b0f);
    __m128i result2d = _mm_add_epi32(r1b0gxc0b0g, r1b0hxc0b0h);
    __m128i result2ab = _mm_add_epi32(result2a, result2b);
    __m128i result2cd = _mm_add_epi32(result2c, result2d);
    __m128i result2abcd = _mm_add_epi32(result2ab, result2cd);

    //Result for row 2
    __m128i r2b0axc0b0a = _mm_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m128i r2b0bxc0b0b = _mm_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m128i r2b0cxc0b0c = _mm_madd_epi16(xmmRow2B0c, xmmCol0B0c);
    __m128i r2b0dxc0b0d = _mm_madd_epi16(xmmRow2B0d, xmmCol0B0d);
    __m128i r2b0exc0b0e = _mm_madd_epi16(xmmRow2B0e, xmmCol0B0e);
    __m128i r2b0fxc0b0f = _mm_madd_epi16(xmmRow2B0f, xmmCol0B0f);
    __m128i r2b0gxc0b0g = _mm_madd_epi16(xmmRow2B0g, xmmCol0B0g);
    __m128i r2b0hxc0b0h = _mm_madd_epi16(xmmRow2B0h, xmmCol0B0h);
    __m128i result3a = _mm_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);
    __m128i result3b = _mm_add_epi32(r2b0cxc0b0c, r2b0dxc0b0d);
    __m128i result3c = _mm_add_epi32(r2b0exc0b0e, r2b0fxc0b0f);
    __m128i result3d = _mm_add_epi32(r2b0gxc0b0g, r2b0hxc0b0h);
    __m128i result3ab = _mm_add_epi32(result3a, result3b);
    __m128i result3cd = _mm_add_epi32(result3c, result3d);
    __m128i result3abcd = _mm_add_epi32(result3ab, result3cd);

    //Result for row 3
    __m128i r3b0axc0b0a = _mm_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m128i r3b0bxc0b0b = _mm_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m128i r3b0cxc0b0c = _mm_madd_epi16(xmmRow3B0c, xmmCol0B0c);
    __m128i r3b0dxc0b0d = _mm_madd_epi16(xmmRow3B0d, xmmCol0B0d);
    __m128i r3b0exc0b0e = _mm_madd_epi16(xmmRow3B0e, xmmCol0B0e);
    __m128i r3b0fxc0b0f = _mm_madd_epi16(xmmRow3B0f, xmmCol0B0f);
    __m128i r3b0gxc0b0g = _mm_madd_epi16(xmmRow3B0g, xmmCol0B0g);
    __m128i r3b0hxc0b0h = _mm_madd_epi16(xmmRow3B0h, xmmCol0B0h);
    __m128i result4a = _mm_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);
    __m128i result4b = _mm_add_epi32(r3b0cxc0b0c, r3b0dxc0b0d);
    __m128i result4c = _mm_add_epi32(r3b0exc0b0e, r3b0fxc0b0f);
    __m128i result4d = _mm_add_epi32(r3b0gxc0b0g, r3b0hxc0b0h);
    __m128i result4ab = _mm_add_epi32(result4a, result4b);
    __m128i result4cd = _mm_add_epi32(result4c, result4d);
    __m128i result4abcd = _mm_add_epi32(result4ab, result4cd);


    *return1 = result1abcd;
    *return2 = result2abcd;
    *return3 = result3abcd;
    *return4 = result4abcd;
}

FORCEINLINE void BlockHandlerSSE::kernelsse64x1(
        __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
        __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
        short* B, __m128i* return1)
{
    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)(B + 8));
    __m128i xmmCol0B0c = _mm_load_si128((__m128i*)(B + 16));
    __m128i xmmCol0B0d = _mm_load_si128((__m128i*)(B + 24));
    __m128i xmmCol0B0e = _mm_load_si128((__m128i*)(B + 32));
    __m128i xmmCol0B0f = _mm_load_si128((__m128i*)(B + 40));
    __m128i xmmCol0B0g = _mm_load_si128((__m128i*)(B + 48));
    __m128i xmmCol0B0h = _mm_load_si128((__m128i*)(B + 56));



    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i r0b0cxc0b0c = _mm_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m128i r0b0dxc0b0d = _mm_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m128i r0b0exc0b0e = _mm_madd_epi16(xmmRow0B0e, xmmCol0B0e);
    __m128i r0b0fxc0b0f = _mm_madd_epi16(xmmRow0B0f, xmmCol0B0f);
    __m128i r0b0gxc0b0g = _mm_madd_epi16(xmmRow0B0g, xmmCol0B0g);
    __m128i r0b0hxc0b0h = _mm_madd_epi16(xmmRow0B0h, xmmCol0B0h);
    __m128i result1a = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m128i result1b = _mm_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m128i result1c = _mm_add_epi32(r0b0exc0b0e, r0b0fxc0b0f);
    __m128i result1d = _mm_add_epi32(r0b0gxc0b0g, r0b0hxc0b0h);
    __m128i result1ab = _mm_add_epi32(result1a, result1b);
    __m128i result1cd = _mm_add_epi32(result1c, result1d);
    __m128i result1abcd = _mm_add_epi32(result1ab, result1cd);




    *return1 = result1abcd;
}


//Compiler issues bogus warning about uninitialized vars when blockSize > 1, but
//initialization takes place under the same condition as use so it's fine.
#pragma warning(push)
#pragma warning(disable: 4701)

FORCEINLINE void BlockHandlerSSE::HandleBlock128x1(int currBlock, int startRow, int k, int n, short* newA, short* B,
        int blockCnt, __m128i* resultStorage, VectorT* /*subtractMe*/)
{

    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 128, 1, k);
    int aOffset2 = RowToColOffsetRewrittenA(startRow, currBlock + 1, 128, 1, k);
    short* currA = &newA[aOffset];
    short* currA2 = &newA[aOffset2];

    LOAD_128x1;
    DECL2_128x1;
    if (blockCnt > 1)
    {
        LOAD2_128x1;
    }
    //LOAD3_128x4;
    for (int c = 0; c < n; ++c)
    {
        //This makes a small but noticable difference.
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 128, n)];
        short* currB2 = &B[RowToColOffsetRewrittenB(c, currBlock + 1, 128, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum2 = _mm_set_epi32(0, 0, 0, 0);




        kernelsse128x1(
                r0b0a, r0b0b, r0b0c, r0b0d, r0b0e, r0b0f, r0b0g, r0b0h,
                r0b0i, r0b0j, r0b0k, r0b0l, r0b0m, r0b0n, r0b0o, r0b0p,
                currB, &accum1);

        if (blockCnt > 1)
        {

            kernelsse128x1(
                    r0b0a2, r0b0b2, r0b0c2, r0b0d2, r0b0e2, r0b0f2, r0b0g2, r0b0h2,
                    r0b0i2, r0b0j2, r0b0k2, r0b0l2, r0b0m2, r0b0n2, r0b0o2, r0b0p2,
                    currB2, &accum2);
        }

        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], _mm_add_epi32(accum1, accum2));

    }


}




FORCEINLINE void BlockHandlerSSE::HandleBlock128x4(int currBlock, int startRow, int k, int n, short* newA, short* B,
        int blockCnt, __m128i* resultStorage, VectorT* /*subtractMe*/)
{

    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 128, 4, k);
    int aOffset2 = RowToColOffsetRewrittenA(startRow, currBlock + 1, 128, 4, k);
    short* currA = &newA[aOffset];
    short* currA2 = &newA[aOffset2];

    LOAD_128x4;
    DECL2_128x4;
    if (blockCnt > 1)
    {
        LOAD2_128x4;
    }

    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 128, n)];
        short* currB2 = &B[RowToColOffsetRewrittenB(c, currBlock + 1, 128, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m128i accum1 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum2 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum3 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum4 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum5 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum6 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum7 = _mm_set_epi32(0, 0, 0, 0);
        __m128i accum8 = _mm_set_epi32(0, 0, 0, 0);


        kernelsse128x4(
                r0b0a, r0b0b, r0b0c, r0b0d, r0b0e, r0b0f, r0b0g, r0b0h,
                r0b0i, r0b0j, r0b0k, r0b0l, r0b0m, r0b0n, r0b0o, r0b0p,
                r1b0a, r1b0b, r1b0c, r1b0d, r1b0e, r1b0f, r1b0g, r1b0h,
                r1b0i, r1b0j, r1b0k, r1b0l, r1b0m, r1b0n, r1b0o, r1b0p,
                r2b0a, r2b0b, r2b0c, r2b0d, r2b0e, r2b0f, r2b0g, r2b0h,
                r2b0i, r2b0j, r2b0k, r2b0l, r2b0m, r2b0n, r2b0o, r2b0p,
                r3b0a, r3b0b, r3b0c, r3b0d, r3b0e, r3b0f, r3b0g, r3b0h,
                r3b0i, r3b0j, r3b0k, r3b0l, r3b0m, r3b0n, r3b0o, r3b0p,
                currB, &accum1, &accum2, &accum3, &accum4);

        if (blockCnt > 1)
        {

            kernelsse128x4(
                    r0b0a2, r0b0b2, r0b0c2, r0b0d2, r0b0e2, r0b0f2, r0b0g2, r0b0h2,
                    r0b0i2, r0b0j2, r0b0k2, r0b0l2, r0b0m2, r0b0n2, r0b0o2, r0b0p2,
                    r1b0a2, r1b0b2, r1b0c2, r1b0d2, r1b0e2, r1b0f2, r1b0g2, r1b0h2,
                    r1b0i2, r1b0j2, r1b0k2, r1b0l2, r1b0m2, r1b0n2, r1b0o2, r1b0p2,
                    r2b0a2, r2b0b2, r2b0c2, r2b0d2, r2b0e2, r2b0f2, r2b0g2, r2b0h2,
                    r2b0i2, r2b0j2, r2b0k2, r2b0l2, r2b0m2, r2b0n2, r2b0o2, r2b0p2,
                    r3b0a2, r3b0b2, r3b0c2, r3b0d2, r3b0e2, r3b0f2, r3b0g2, r3b0h2,
                    r3b0i2, r3b0j2, r3b0k2, r3b0l2, r3b0m2, r3b0n2, r3b0o2, r3b0p2,
                    currB2, &accum5, &accum6, &accum7, &accum8);
        }



        resultStorage[RowColToOffset(0, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(0, c, n)], _mm_add_epi32(accum1, accum5));
        resultStorage[RowColToOffset(1, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(1, c, n)], _mm_add_epi32(accum2, accum6));
        resultStorage[RowColToOffset(2, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(2, c, n)], _mm_add_epi32(accum3, accum7));
        resultStorage[RowColToOffset(3, c, n)] = _mm_add_epi32(resultStorage[RowColToOffset(3, c, n)], _mm_add_epi32(accum4, accum8));
    }
}

#pragma warning(pop)

FORCEINLINE void BlockHandlerSSE::kernelsse128x4(
        __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
        __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
        __m128i xmmRow0B0i, __m128i xmmRow0B0j, __m128i xmmRow0B0k, __m128i xmmRow0B0l,
        __m128i xmmRow0B0m, __m128i xmmRow0B0n, __m128i xmmRow0B0o, __m128i xmmRow0B0p,
        __m128i xmmRow1B0a, __m128i xmmRow1B0b, __m128i xmmRow1B0c, __m128i xmmRow1B0d,
        __m128i xmmRow1B0e, __m128i xmmRow1B0f, __m128i xmmRow1B0g, __m128i xmmRow1B0h,
        __m128i xmmRow1B0i, __m128i xmmRow1B0j, __m128i xmmRow1B0k, __m128i xmmRow1B0l,
        __m128i xmmRow1B0m, __m128i xmmRow1B0n, __m128i xmmRow1B0o, __m128i xmmRow1B0p,
        __m128i xmmRow2B0a, __m128i xmmRow2B0b, __m128i xmmRow2B0c, __m128i xmmRow2B0d,
        __m128i xmmRow2B0e, __m128i xmmRow2B0f, __m128i xmmRow2B0g, __m128i xmmRow2B0h,
        __m128i xmmRow2B0i, __m128i xmmRow2B0j, __m128i xmmRow2B0k, __m128i xmmRow2B0l,
        __m128i xmmRow2B0m, __m128i xmmRow2B0n, __m128i xmmRow2B0o, __m128i xmmRow2B0p,
        __m128i xmmRow3B0a, __m128i xmmRow3B0b, __m128i xmmRow3B0c, __m128i xmmRow3B0d,
        __m128i xmmRow3B0e, __m128i xmmRow3B0f, __m128i xmmRow3B0g, __m128i xmmRow3B0h,
        __m128i xmmRow3B0i, __m128i xmmRow3B0j, __m128i xmmRow3B0k, __m128i xmmRow3B0l,
        __m128i xmmRow3B0m, __m128i xmmRow3B0n, __m128i xmmRow3B0o, __m128i xmmRow3B0p,
        short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4)
{

    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)B + 1);
    __m128i xmmCol0B0c = _mm_load_si128((__m128i*)B + 2);
    __m128i xmmCol0B0d = _mm_load_si128((__m128i*)B + 3);
    __m128i xmmCol0B0e = _mm_load_si128((__m128i*)B + 4);
    __m128i xmmCol0B0f = _mm_load_si128((__m128i*)B + 5);
    __m128i xmmCol0B0g = _mm_load_si128((__m128i*)B + 6);
    __m128i xmmCol0B0h = _mm_load_si128((__m128i*)B + 7);
    __m128i xmmCol0B0i = _mm_load_si128((__m128i*)B + 8);
    __m128i xmmCol0B0j = _mm_load_si128((__m128i*)B + 9);
    __m128i xmmCol0B0k = _mm_load_si128((__m128i*)B + 10);
    __m128i xmmCol0B0l = _mm_load_si128((__m128i*)B + 11);
    __m128i xmmCol0B0m = _mm_load_si128((__m128i*)B + 12);
    __m128i xmmCol0B0n = _mm_load_si128((__m128i*)B + 13);
    __m128i xmmCol0B0o = _mm_load_si128((__m128i*)B + 14);
    __m128i xmmCol0B0p = _mm_load_si128((__m128i*)B + 15);
    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i r0b0cxc0b0c = _mm_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m128i r0b0dxc0b0d = _mm_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m128i r0b0exc0b0e = _mm_madd_epi16(xmmRow0B0e, xmmCol0B0e);
    __m128i r0b0fxc0b0f = _mm_madd_epi16(xmmRow0B0f, xmmCol0B0f);
    __m128i r0b0gxc0b0g = _mm_madd_epi16(xmmRow0B0g, xmmCol0B0g);
    __m128i r0b0hxc0b0h = _mm_madd_epi16(xmmRow0B0h, xmmCol0B0h);
    __m128i r0b0ixc0b0i = _mm_madd_epi16(xmmRow0B0i, xmmCol0B0i);
    __m128i r0b0jxc0b0j = _mm_madd_epi16(xmmRow0B0j, xmmCol0B0j);
    __m128i r0b0kxc0b0k = _mm_madd_epi16(xmmRow0B0k, xmmCol0B0k);
    __m128i r0b0lxc0b0l = _mm_madd_epi16(xmmRow0B0l, xmmCol0B0l);
    __m128i r0b0mxc0b0m = _mm_madd_epi16(xmmRow0B0m, xmmCol0B0m);
    __m128i r0b0nxc0b0n = _mm_madd_epi16(xmmRow0B0n, xmmCol0B0n);
    __m128i r0b0oxc0b0o = _mm_madd_epi16(xmmRow0B0o, xmmCol0B0o);
    __m128i r0b0pxc0b0p = _mm_madd_epi16(xmmRow0B0p, xmmCol0B0p);
    __m128i result1a = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m128i result1b = _mm_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m128i result1c = _mm_add_epi32(r0b0exc0b0e, r0b0fxc0b0f);
    __m128i result1d = _mm_add_epi32(r0b0gxc0b0g, r0b0hxc0b0h);
    __m128i result1e = _mm_add_epi32(r0b0ixc0b0i, r0b0jxc0b0j);
    __m128i result1f = _mm_add_epi32(r0b0kxc0b0k, r0b0lxc0b0l);
    __m128i result1g = _mm_add_epi32(r0b0mxc0b0m, r0b0nxc0b0n);
    __m128i result1h = _mm_add_epi32(r0b0oxc0b0o, r0b0pxc0b0p);



    __m128i result1ab = _mm_add_epi32(result1a, result1b);
    __m128i result1cd = _mm_add_epi32(result1c, result1d);
    __m128i result1ef = _mm_add_epi32(result1e, result1f);
    __m128i result1gh = _mm_add_epi32(result1g, result1h);
    __m128i result1abcd = _mm_add_epi32(result1ab, result1cd);
    __m128i result1efgh = _mm_add_epi32(result1ef, result1gh);
    __m128i result1abcdefgh = _mm_add_epi32(result1abcd, result1efgh);

    //Result for row 1
    __m128i r1b0axc0b0a = _mm_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m128i r1b0bxc0b0b = _mm_madd_epi16(xmmRow1B0b, xmmCol0B0b);
    __m128i r1b0cxc0b0c = _mm_madd_epi16(xmmRow1B0c, xmmCol0B0c);
    __m128i r1b0dxc0b0d = _mm_madd_epi16(xmmRow1B0d, xmmCol0B0d);
    __m128i r1b0exc0b0e = _mm_madd_epi16(xmmRow1B0e, xmmCol0B0e);
    __m128i r1b0fxc0b0f = _mm_madd_epi16(xmmRow1B0f, xmmCol0B0f);
    __m128i r1b0gxc0b0g = _mm_madd_epi16(xmmRow1B0g, xmmCol0B0g);
    __m128i r1b0hxc0b0h = _mm_madd_epi16(xmmRow1B0h, xmmCol0B0h);
    __m128i r1b0ixc0b0i = _mm_madd_epi16(xmmRow1B0i, xmmCol0B0i);
    __m128i r1b0jxc0b0j = _mm_madd_epi16(xmmRow1B0j, xmmCol0B0j);
    __m128i r1b0kxc0b0k = _mm_madd_epi16(xmmRow1B0k, xmmCol0B0k);
    __m128i r1b0lxc0b0l = _mm_madd_epi16(xmmRow1B0l, xmmCol0B0l);
    __m128i r1b0mxc0b0m = _mm_madd_epi16(xmmRow1B0m, xmmCol0B0m);
    __m128i r1b0nxc0b0n = _mm_madd_epi16(xmmRow1B0n, xmmCol0B0n);
    __m128i r1b0oxc0b0o = _mm_madd_epi16(xmmRow1B0o, xmmCol0B0o);
    __m128i r1b0pxc0b0p = _mm_madd_epi16(xmmRow1B0p, xmmCol0B0p);



    __m128i result2a = _mm_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);
    __m128i result2b = _mm_add_epi32(r1b0cxc0b0c, r1b0dxc0b0d);
    __m128i result2c = _mm_add_epi32(r1b0exc0b0e, r1b0fxc0b0f);
    __m128i result2d = _mm_add_epi32(r1b0gxc0b0g, r1b0hxc0b0h);
    __m128i result2e = _mm_add_epi32(r1b0ixc0b0i, r1b0jxc0b0j);
    __m128i result2f = _mm_add_epi32(r1b0kxc0b0k, r1b0lxc0b0l);
    __m128i result2g = _mm_add_epi32(r1b0mxc0b0m, r1b0nxc0b0n);
    __m128i result2h = _mm_add_epi32(r1b0oxc0b0o, r1b0pxc0b0p);

    __m128i result2ab = _mm_add_epi32(result2a, result2b);
    __m128i result2cd = _mm_add_epi32(result2c, result2d);
    __m128i result2ef = _mm_add_epi32(result2e, result2f);
    __m128i result2gh = _mm_add_epi32(result2g, result2h);
    __m128i result2abcd = _mm_add_epi32(result2ab, result2cd);
    __m128i result2efgh = _mm_add_epi32(result2ef, result2gh);
    __m128i result2abcdefgh = _mm_add_epi32(result2abcd, result2efgh);

    //Result for row 2
    __m128i r2b0axc0b0a = _mm_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m128i r2b0bxc0b0b = _mm_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m128i r2b0cxc0b0c = _mm_madd_epi16(xmmRow2B0c, xmmCol0B0c);
    __m128i r2b0dxc0b0d = _mm_madd_epi16(xmmRow2B0d, xmmCol0B0d);
    __m128i r2b0exc0b0e = _mm_madd_epi16(xmmRow2B0e, xmmCol0B0e);
    __m128i r2b0fxc0b0f = _mm_madd_epi16(xmmRow2B0f, xmmCol0B0f);
    __m128i r2b0gxc0b0g = _mm_madd_epi16(xmmRow2B0g, xmmCol0B0g);
    __m128i r2b0hxc0b0h = _mm_madd_epi16(xmmRow2B0h, xmmCol0B0h);
    __m128i r2b0ixc0b0i = _mm_madd_epi16(xmmRow2B0i, xmmCol0B0i);
    __m128i r2b0jxc0b0j = _mm_madd_epi16(xmmRow2B0j, xmmCol0B0j);
    __m128i r2b0kxc0b0k = _mm_madd_epi16(xmmRow2B0k, xmmCol0B0k);
    __m128i r2b0lxc0b0l = _mm_madd_epi16(xmmRow2B0l, xmmCol0B0l);
    __m128i r2b0mxc0b0m = _mm_madd_epi16(xmmRow2B0m, xmmCol0B0m);
    __m128i r2b0nxc0b0n = _mm_madd_epi16(xmmRow2B0n, xmmCol0B0n);
    __m128i r2b0oxc0b0o = _mm_madd_epi16(xmmRow2B0o, xmmCol0B0o);
    __m128i r2b0pxc0b0p = _mm_madd_epi16(xmmRow2B0p, xmmCol0B0p);

    __m128i result3a = _mm_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);
    __m128i result3b = _mm_add_epi32(r2b0cxc0b0c, r2b0dxc0b0d);
    __m128i result3c = _mm_add_epi32(r2b0exc0b0e, r2b0fxc0b0f);
    __m128i result3d = _mm_add_epi32(r2b0gxc0b0g, r2b0hxc0b0h);
    __m128i result3e = _mm_add_epi32(r2b0ixc0b0i, r2b0jxc0b0j);
    __m128i result3f = _mm_add_epi32(r2b0kxc0b0k, r2b0lxc0b0l);
    __m128i result3g = _mm_add_epi32(r2b0mxc0b0m, r2b0nxc0b0n);
    __m128i result3h = _mm_add_epi32(r2b0oxc0b0o, r2b0pxc0b0p);

    __m128i result3ab = _mm_add_epi32(result3a, result3b);
    __m128i result3cd = _mm_add_epi32(result3c, result3d);
    __m128i result3ef = _mm_add_epi32(result3e, result3f);
    __m128i result3gh = _mm_add_epi32(result3g, result3h);
    __m128i result3abcd = _mm_add_epi32(result3ab, result3cd);
    __m128i result3efgh = _mm_add_epi32(result3ef, result3gh);
    __m128i result3abcdefgh = _mm_add_epi32(result3abcd, result3efgh);


    //Result for row 3
    __m128i r3b0axc0b0a = _mm_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m128i r3b0bxc0b0b = _mm_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m128i r3b0cxc0b0c = _mm_madd_epi16(xmmRow3B0c, xmmCol0B0c);
    __m128i r3b0dxc0b0d = _mm_madd_epi16(xmmRow3B0d, xmmCol0B0d);
    __m128i r3b0exc0b0e = _mm_madd_epi16(xmmRow3B0e, xmmCol0B0e);
    __m128i r3b0fxc0b0f = _mm_madd_epi16(xmmRow3B0f, xmmCol0B0f);
    __m128i r3b0gxc0b0g = _mm_madd_epi16(xmmRow3B0g, xmmCol0B0g);
    __m128i r3b0hxc0b0h = _mm_madd_epi16(xmmRow3B0h, xmmCol0B0h);
    __m128i r3b0ixc0b0i = _mm_madd_epi16(xmmRow3B0i, xmmCol0B0i);
    __m128i r3b0jxc0b0j = _mm_madd_epi16(xmmRow3B0j, xmmCol0B0j);
    __m128i r3b0kxc0b0k = _mm_madd_epi16(xmmRow3B0k, xmmCol0B0k);
    __m128i r3b0lxc0b0l = _mm_madd_epi16(xmmRow3B0l, xmmCol0B0l);
    __m128i r3b0mxc0b0m = _mm_madd_epi16(xmmRow3B0m, xmmCol0B0m);
    __m128i r3b0nxc0b0n = _mm_madd_epi16(xmmRow3B0n, xmmCol0B0n);
    __m128i r3b0oxc0b0o = _mm_madd_epi16(xmmRow3B0o, xmmCol0B0o);
    __m128i r3b0pxc0b0p = _mm_madd_epi16(xmmRow3B0p, xmmCol0B0p);

    __m128i result4a = _mm_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);
    __m128i result4b = _mm_add_epi32(r3b0cxc0b0c, r3b0dxc0b0d);
    __m128i result4c = _mm_add_epi32(r3b0exc0b0e, r3b0fxc0b0f);
    __m128i result4d = _mm_add_epi32(r3b0gxc0b0g, r3b0hxc0b0h);
    __m128i result4e = _mm_add_epi32(r3b0ixc0b0i, r3b0jxc0b0j);
    __m128i result4f = _mm_add_epi32(r3b0kxc0b0k, r3b0lxc0b0l);
    __m128i result4g = _mm_add_epi32(r3b0mxc0b0m, r3b0nxc0b0n);
    __m128i result4h = _mm_add_epi32(r3b0oxc0b0o, r3b0pxc0b0p);
    __m128i result4ab = _mm_add_epi32(result4a, result4b);
    __m128i result4cd = _mm_add_epi32(result4c, result4d);
    __m128i result4ef = _mm_add_epi32(result4e, result4f);
    __m128i result4gh = _mm_add_epi32(result4g, result4h);
    __m128i result4abcd = _mm_add_epi32(result4ab, result4cd);
    __m128i result4efgh = _mm_add_epi32(result4ef, result4gh);
    __m128i result4abcdefgh = _mm_add_epi32(result4abcd, result4efgh);

    *return1 = result1abcdefgh;
    *return2 = result2abcdefgh;
    *return3 = result3abcdefgh;
    *return4 = result4abcdefgh;
}

FORCEINLINE void BlockHandlerSSE::kernelsse128x1(
        __m128i xmmRow0B0a, __m128i xmmRow0B0b, __m128i xmmRow0B0c, __m128i xmmRow0B0d,
        __m128i xmmRow0B0e, __m128i xmmRow0B0f, __m128i xmmRow0B0g, __m128i xmmRow0B0h,
        __m128i xmmRow0B0i, __m128i xmmRow0B0j, __m128i xmmRow0B0k, __m128i xmmRow0B0l,
        __m128i xmmRow0B0m, __m128i xmmRow0B0n, __m128i xmmRow0B0o, __m128i xmmRow0B0p,
        short* B, __m128i* return1)
{

    __m128i xmmCol0B0a = _mm_load_si128((__m128i*)B);
    __m128i xmmCol0B0b = _mm_load_si128((__m128i*)B + 1);
    __m128i xmmCol0B0c = _mm_load_si128((__m128i*)B + 2);
    __m128i xmmCol0B0d = _mm_load_si128((__m128i*)B + 3);
    __m128i xmmCol0B0e = _mm_load_si128((__m128i*)B + 4);
    __m128i xmmCol0B0f = _mm_load_si128((__m128i*)B + 5);
    __m128i xmmCol0B0g = _mm_load_si128((__m128i*)B + 6);
    __m128i xmmCol0B0h = _mm_load_si128((__m128i*)B + 7);
    __m128i xmmCol0B0i = _mm_load_si128((__m128i*)B + 8);
    __m128i xmmCol0B0j = _mm_load_si128((__m128i*)B + 9);
    __m128i xmmCol0B0k = _mm_load_si128((__m128i*)B + 10);
    __m128i xmmCol0B0l = _mm_load_si128((__m128i*)B + 11);
    __m128i xmmCol0B0m = _mm_load_si128((__m128i*)B + 12);
    __m128i xmmCol0B0n = _mm_load_si128((__m128i*)B + 13);
    __m128i xmmCol0B0o = _mm_load_si128((__m128i*)B + 14);
    __m128i xmmCol0B0p = _mm_load_si128((__m128i*)B + 15);
    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m128i each (xmm registers))
    __m128i r0b0axc0b0a = _mm_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m128i r0b0bxc0b0b = _mm_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m128i r0b0cxc0b0c = _mm_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m128i r0b0dxc0b0d = _mm_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m128i r0b0exc0b0e = _mm_madd_epi16(xmmRow0B0e, xmmCol0B0e);
    __m128i r0b0fxc0b0f = _mm_madd_epi16(xmmRow0B0f, xmmCol0B0f);
    __m128i r0b0gxc0b0g = _mm_madd_epi16(xmmRow0B0g, xmmCol0B0g);
    __m128i r0b0hxc0b0h = _mm_madd_epi16(xmmRow0B0h, xmmCol0B0h);
    __m128i r0b0ixc0b0i = _mm_madd_epi16(xmmRow0B0i, xmmCol0B0i);
    __m128i r0b0jxc0b0j = _mm_madd_epi16(xmmRow0B0j, xmmCol0B0j);
    __m128i r0b0kxc0b0k = _mm_madd_epi16(xmmRow0B0k, xmmCol0B0k);
    __m128i r0b0lxc0b0l = _mm_madd_epi16(xmmRow0B0l, xmmCol0B0l);
    __m128i r0b0mxc0b0m = _mm_madd_epi16(xmmRow0B0m, xmmCol0B0m);
    __m128i r0b0nxc0b0n = _mm_madd_epi16(xmmRow0B0n, xmmCol0B0n);
    __m128i r0b0oxc0b0o = _mm_madd_epi16(xmmRow0B0o, xmmCol0B0o);
    __m128i r0b0pxc0b0p = _mm_madd_epi16(xmmRow0B0p, xmmCol0B0p);
    __m128i result1a = _mm_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m128i result1b = _mm_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m128i result1c = _mm_add_epi32(r0b0exc0b0e, r0b0fxc0b0f);
    __m128i result1d = _mm_add_epi32(r0b0gxc0b0g, r0b0hxc0b0h);
    __m128i result1e = _mm_add_epi32(r0b0ixc0b0i, r0b0jxc0b0j);
    __m128i result1f = _mm_add_epi32(r0b0kxc0b0k, r0b0lxc0b0l);
    __m128i result1g = _mm_add_epi32(r0b0mxc0b0m, r0b0nxc0b0n);
    __m128i result1h = _mm_add_epi32(r0b0oxc0b0o, r0b0pxc0b0p);



    __m128i result1ab = _mm_add_epi32(result1a, result1b);
    __m128i result1cd = _mm_add_epi32(result1c, result1d);
    __m128i result1ef = _mm_add_epi32(result1e, result1f);
    __m128i result1gh = _mm_add_epi32(result1g, result1h);
    __m128i result1abcd = _mm_add_epi32(result1ab, result1cd);
    __m128i result1efgh = _mm_add_epi32(result1ef, result1gh);
    __m128i result1abcdefgh = _mm_add_epi32(result1abcd, result1efgh);


    *return1 = result1abcdefgh;

}

}}}
