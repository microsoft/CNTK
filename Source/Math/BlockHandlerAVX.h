//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#pragma once
#include "BlockMultiplierPlatform.h"
#include <immintrin.h>
#include <emmintrin.h>
#include <assert.h>
#include <cstdint>
#define FOR_CNTK
#ifdef FOR_CNTK
#include "CommonMatrix.h"
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API BlockHandlerAVX
{

    private:
        //USE SSE for the blocks of 8, borrowed from BlockHandlerSSE
        FORCEINLINE static void kernelsse8x4(__m128i xmmRow0, __m128i xmmRow1, __m128i xmmRow2, __m128i xmmRow3, 
                short* B, __m128i* return1, __m128i* return2, __m128i* return3, __m128i* return4);
        FORCEINLINE static void kernelavx16x4(__m256i xmmRow0B0a, __m256i xmmRow1B0a, __m256i xmmRow2B0a, __m256i xmmRow3B0a,
                short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4);
        FORCEINLINE static void kernelavx32x4(
                __m256i xmmRow0B0a, __m256i xmmRow0B0b,
                __m256i xmmRow1B0a, __m256i xmmRow1B0b,
                __m256i xmmRow2B0a, __m256i xmmRow2B0b,
                __m256i xmmRow3B0a, __m256i xmmRow3B0b,
                short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4);
        FORCEINLINE static void kernelavx64x4(
                __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
                __m256i xmmRow1B0a, __m256i xmmRow1B0b, __m256i xmmRow1B0c, __m256i xmmRow1B0d,
                __m256i xmmRow2B0a, __m256i xmmRow2B0b, __m256i xmmRow2B0c, __m256i xmmRow2B0d,
                __m256i xmmRow3B0a, __m256i xmmRow3B0b, __m256i xmmRow3B0c, __m256i xmmRow3B0d,
                short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4);
        FORCEINLINE static void kernelavx128x4(
                __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
                __m256i xmmRow0B0e, __m256i xmmRow0B0f, __m256i xmmRow0B0g, __m256i xmmRow0B0h,
                __m256i xmmRow1B0a, __m256i xmmRow1B0b, __m256i xmmRow1B0c, __m256i xmmRow1B0d,
                __m256i xmmRow1B0e, __m256i xmmRow1B0f, __m256i xmmRow1B0g, __m256i xmmRow1B0h,
                __m256i xmmRow2B0a, __m256i xmmRow2B0b, __m256i xmmRow2B0c, __m256i xmmRow2B0d,
                __m256i xmmRow2B0e, __m256i xmmRow2B0f, __m256i xmmRow2B0g, __m256i xmmRow2B0h,
                __m256i xmmRow3B0a, __m256i xmmRow3B0b, __m256i xmmRow3B0c, __m256i xmmRow3B0d,
                __m256i xmmRow3B0e, __m256i xmmRow3B0f, __m256i xmmRow3B0g, __m256i xmmRow3B0h,
                short* B, __m256i* return1, __m256i* return2, __m256i* return3, __m256i* return4);

        FORCEINLINE static void kernelsse8x1(__m128i xmmRow0, 
                short* B, __m128i* return1);
        FORCEINLINE static void kernelavx16x1(__m256i xmmRow0B0a, 
                short* B, __m256i* return1 );
        FORCEINLINE static void kernelavx32x1(
                __m256i xmmRow0B0a, __m256i xmmRow0B0b,
                short* B, __m256i* return1);
        FORCEINLINE static void kernelavx64x1(
                __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
                short* B, __m256i* return1) ;
        FORCEINLINE static void kernelavx128x1(
                __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
                __m256i xmmRow0B0e, __m256i xmmRow0B0f, __m256i xmmRow0B0g, __m256i xmmRow0B0h,
                short* B, __m256i* return1);

        //TODO: Should these be refactored somewhere else? Any BlockHandler will need access to these functions.
        //Separate class with static functions? Maybe move the Block rewriting functions as well as these to a new
        //static class.
        static int RowToColOffsetRewrittenB(int col, int kOffset, int blockSize, int origCols);
        static int RowToColOffsetRewrittenA(int row, int kOffset, int blockSize, int rowsPerBlock, int origCols);
        static void DumpM256(__m256i dumpMe);
    public:
        typedef __m256i VectorT;
        typedef int16_t ScalarAT;
        typedef int16_t ScalarBT;
        typedef int32_t ScalarCT;
        FORCEINLINE static void HandleBlock8x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
                int blockCnt, __m128i* resultStorage);
        FORCEINLINE static void HandleBlock32x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
                int blockCnt, __m256i* resultStorage);
        FORCEINLINE static void HandleBlock64x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
                int blockCnt, __m256i* resultStorage);
        FORCEINLINE static void HandleBlock128x4(int currBlock, int startRow, int k, int n, short* newA, short* B,
                int blockCnt, __m256i* resultStorage, VectorT* subtractMe);

        FORCEINLINE static void HandleBlock8x1(int currBlock, int startRow, int k, int n, short* newA, short* B, 
                int blockCnt, __m128i* resultStorage);
        FORCEINLINE static void HandleBlock16x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  
                int blockCnt, __m256i* resultStorage);
        FORCEINLINE static void HandleBlock64x1(int currBlock, int startRow, int k, int n, short* newA, short* B, 
                int blockCnt, __m256i* resultStorage);
        FORCEINLINE static void HandleBlock128x1(int currBlock, int startRow, int k, int n, short* newA, short* B,
                int blockCnt, __m256i* resultStorage, VectorT* subtractMe);

        FORCEINLINE static void HandleBlock16x4(int currBlock, int startRow, int k, int n, short* newA, short* B,  
                int blockCnt, __m256i* resultStorage);



        //FORCEINLINE static void HandleBlock128x4(int currBlock, int startRow, int m, int k, int n, short* newA, short* B, 

        FORCEINLINE static void HandleBlock32x1(int currBlock, int startRow, int k, int n, short* newA, short* B, 
                int blockCnt, __m256i* resultStorage);

        static VectorT* PrepareExtraB(const ScalarBT* /*prepareMe*/, int /*k*/, int /*n*/)
        {
            return nullptr;
        }
        static void FreePreparedB(VectorT* freeMe) { freeMe;  assert(nullptr == freeMe); }
};

#define LOADAVX2_128x4 \
    __m256i r0b0a2 = _mm256_load_si256((__m256i*)currA2);		\
__m256i r0b0b2 = _mm256_load_si256((__m256i*)(currA2 + 16));	\
__m256i r0b0c2 = _mm256_load_si256((__m256i*)(currA2 + 32)); \
__m256i r0b0d2 = _mm256_load_si256((__m256i*)(currA2 + 48)); \
__m256i r0b0e2 = _mm256_load_si256((__m256i*)(currA2 + 64)); \
__m256i r0b0f2 = _mm256_load_si256((__m256i*)(currA2 + 80)); \
__m256i r0b0g2 = _mm256_load_si256((__m256i*)(currA2 + 96)); \
__m256i r0b0h2 = _mm256_load_si256((__m256i*)(currA2 + 112));\
\
__m256i r1b0a2 = _mm256_load_si256((__m256i*)(currA2 + 128));\
__m256i r1b0b2 = _mm256_load_si256((__m256i*)(currA2 + 144));\
__m256i r1b0c2 = _mm256_load_si256((__m256i*)(currA2 + 160));\
__m256i r1b0d2 = _mm256_load_si256((__m256i*)(currA2 + 176));\
__m256i r1b0e2 = _mm256_load_si256((__m256i*)(currA2 + 192));\
__m256i r1b0f2 = _mm256_load_si256((__m256i*)(currA2 + 208));\
__m256i r1b0g2 = _mm256_load_si256((__m256i*)(currA2 + 224));\
__m256i r1b0h2 = _mm256_load_si256((__m256i*)(currA2 + 240));\
\
__m256i r2b0a2 = _mm256_load_si256((__m256i*)(currA2 + 256));\
__m256i r2b0b2 = _mm256_load_si256((__m256i*)(currA2 + 272));\
__m256i r2b0c2 = _mm256_load_si256((__m256i*)(currA2 + 288));\
__m256i r2b0d2 = _mm256_load_si256((__m256i*)(currA2 + 304));\
__m256i r2b0e2 = _mm256_load_si256((__m256i*)(currA2 + 320));\
__m256i r2b0f2 = _mm256_load_si256((__m256i*)(currA2 + 336));\
__m256i r2b0g2 = _mm256_load_si256((__m256i*)(currA2 + 352));\
__m256i r2b0h2 = _mm256_load_si256((__m256i*)(currA2 + 368));\
\
__m256i r3b0a2 = _mm256_load_si256((__m256i*)(currA2 + 384));\
__m256i r3b0b2 = _mm256_load_si256((__m256i*)(currA2 + 400));\
__m256i r3b0c2 = _mm256_load_si256((__m256i*)(currA2 + 416));\
__m256i r3b0d2 = _mm256_load_si256((__m256i*)(currA2 + 432));\
__m256i r3b0e2 = _mm256_load_si256((__m256i*)(currA2 + 448));\
__m256i r3b0f2 = _mm256_load_si256((__m256i*)(currA2 + 464));\
__m256i r3b0g2 = _mm256_load_si256((__m256i*)(currA2 + 480));\
__m256i r3b0h2 = _mm256_load_si256((__m256i*)(currA2 + 496));\

#define LOADAVX2_128x1 \
    __m256i r0b0a2 = _mm256_load_si256((__m256i*)currA2);		\
__m256i r0b0b2 = _mm256_load_si256((__m256i*)(currA2 + 16));	\
__m256i r0b0c2 = _mm256_load_si256((__m256i*)(currA2 + 32)); \
__m256i r0b0d2 = _mm256_load_si256((__m256i*)(currA2 + 48)); \
__m256i r0b0e2 = _mm256_load_si256((__m256i*)(currA2 + 64)); \
__m256i r0b0f2 = _mm256_load_si256((__m256i*)(currA2 + 80)); \
__m256i r0b0g2 = _mm256_load_si256((__m256i*)(currA2 + 96)); \
__m256i r0b0h2 = _mm256_load_si256((__m256i*)(currA2 + 112));


#define LOADAVX_128x1 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r0b0b = _mm256_load_si256((__m256i*)(currA + 16));	\
__m256i r0b0c = _mm256_load_si256((__m256i*)(currA + 32)); \
__m256i r0b0d = _mm256_load_si256((__m256i*)(currA + 48)); \
__m256i r0b0e = _mm256_load_si256((__m256i*)(currA + 64)); \
__m256i r0b0f = _mm256_load_si256((__m256i*)(currA + 80)); \
__m256i r0b0g = _mm256_load_si256((__m256i*)(currA + 96)); \
__m256i r0b0h = _mm256_load_si256((__m256i*)(currA + 112));


#define LOADAVX_128x4 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r0b0b = _mm256_load_si256((__m256i*)(currA + 16));	\
__m256i r0b0c = _mm256_load_si256((__m256i*)(currA + 32)); \
__m256i r0b0d = _mm256_load_si256((__m256i*)(currA + 48)); \
__m256i r0b0e = _mm256_load_si256((__m256i*)(currA + 64)); \
__m256i r0b0f = _mm256_load_si256((__m256i*)(currA + 80)); \
__m256i r0b0g = _mm256_load_si256((__m256i*)(currA + 96)); \
__m256i r0b0h = _mm256_load_si256((__m256i*)(currA + 112));\
\
__m256i r1b0a = _mm256_load_si256((__m256i*)(currA + 128));\
__m256i r1b0b = _mm256_load_si256((__m256i*)(currA + 144));\
__m256i r1b0c = _mm256_load_si256((__m256i*)(currA + 160));\
__m256i r1b0d = _mm256_load_si256((__m256i*)(currA + 176));\
__m256i r1b0e = _mm256_load_si256((__m256i*)(currA + 192));\
__m256i r1b0f = _mm256_load_si256((__m256i*)(currA + 208));\
__m256i r1b0g = _mm256_load_si256((__m256i*)(currA + 224));\
__m256i r1b0h = _mm256_load_si256((__m256i*)(currA + 240));\
\
__m256i r2b0a = _mm256_load_si256((__m256i*)(currA + 256));\
__m256i r2b0b = _mm256_load_si256((__m256i*)(currA + 272));\
__m256i r2b0c = _mm256_load_si256((__m256i*)(currA + 288));\
__m256i r2b0d = _mm256_load_si256((__m256i*)(currA + 304));\
__m256i r2b0e = _mm256_load_si256((__m256i*)(currA + 320));\
__m256i r2b0f = _mm256_load_si256((__m256i*)(currA + 336));\
__m256i r2b0g = _mm256_load_si256((__m256i*)(currA + 352));\
__m256i r2b0h = _mm256_load_si256((__m256i*)(currA + 368));\
\
__m256i r3b0a = _mm256_load_si256((__m256i*)(currA + 384));\
__m256i r3b0b = _mm256_load_si256((__m256i*)(currA + 400));\
__m256i r3b0c = _mm256_load_si256((__m256i*)(currA + 416));\
__m256i r3b0d = _mm256_load_si256((__m256i*)(currA + 432));\
__m256i r3b0e = _mm256_load_si256((__m256i*)(currA + 448));\
__m256i r3b0f = _mm256_load_si256((__m256i*)(currA + 464));\
__m256i r3b0g = _mm256_load_si256((__m256i*)(currA + 480));\
__m256i r3b0h = _mm256_load_si256((__m256i*)(currA + 496));\

#define LOADAVX_64x4 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r0b0b = _mm256_load_si256((__m256i*)currA + 1);	\
__m256i r0b0c = _mm256_load_si256((__m256i*)currA + 2); \
__m256i r0b0d = _mm256_load_si256((__m256i*)currA + 3); \
\
__m256i r1b0a = _mm256_load_si256((__m256i*)currA + 4);\
__m256i r1b0b = _mm256_load_si256((__m256i*)currA + 5);\
__m256i r1b0c = _mm256_load_si256((__m256i*)currA + 6);\
__m256i r1b0d = _mm256_load_si256((__m256i*)currA + 7);\
\
__m256i r2b0a = _mm256_load_si256((__m256i*)currA + 8);\
__m256i r2b0b = _mm256_load_si256((__m256i*)currA + 9);\
__m256i r2b0c = _mm256_load_si256((__m256i*)currA + 10);\
__m256i r2b0d = _mm256_load_si256((__m256i*)currA + 11);\
\
__m256i r3b0a = _mm256_load_si256((__m256i*)currA + 12);\
__m256i r3b0b = _mm256_load_si256((__m256i*)currA + 13);\
__m256i r3b0c = _mm256_load_si256((__m256i*)currA + 14);\
__m256i r3b0d = _mm256_load_si256((__m256i*)currA + 15);

#define LOADAVX_64x1 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r0b0b = _mm256_load_si256((__m256i*)currA + 1);	\
__m256i r0b0c = _mm256_load_si256((__m256i*)currA + 2); \
__m256i r0b0d = _mm256_load_si256((__m256i*)currA + 3); 


#define LOADAVX_32x4 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r0b0b = _mm256_load_si256((__m256i*)currA + 1);	\
\
__m256i r1b0a = _mm256_load_si256((__m256i*)currA + 2);\
__m256i r1b0b = _mm256_load_si256((__m256i*)currA + 3);\
\
__m256i r2b0a = _mm256_load_si256((__m256i*)currA + 4);\
__m256i r2b0b = _mm256_load_si256((__m256i*)currA + 5);\
\
__m256i r3b0a = _mm256_load_si256((__m256i*)currA + 6);\
__m256i r3b0b = _mm256_load_si256((__m256i*)currA + 7);\

#define LOADAVX_32x1 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r0b0b = _mm256_load_si256((__m256i*)currA + 1);	



#define LOADAVX_16x4 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		\
__m256i r1b0a = _mm256_load_si256((__m256i*)currA + 1);\
__m256i r2b0a = _mm256_load_si256((__m256i*)currA + 2);\
__m256i r3b0a = _mm256_load_si256((__m256i*)currA + 3);\

#define LOADAVX_16x1 \
    __m256i r0b0a = _mm256_load_si256((__m256i*)currA);		

#define LOAD_8x4 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);\
__m128i r1b0a = _mm_load_si128((__m128i*)currA + 1);\
__m128i r2b0a = _mm_load_si128((__m128i*)currA + 2);\
__m128i r3b0a = _mm_load_si128((__m128i*)currA + 3);\

#define LOAD_8x1 \
    __m128i r0b0a = _mm_load_si128((__m128i*)currA);

FORCEINLINE void BlockHandlerAVX::HandleBlock8x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int blockCnt, __m128i* resultStorage)
{
    blockCnt; //warning 4100
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

FORCEINLINE void BlockHandlerAVX::HandleBlock8x1(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m128i* resultStorage)
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



FORCEINLINE void BlockHandlerAVX::HandleBlock16x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m256i* resultStorage) 
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 16, 4, k);
    short* currA = &newA[aOffset];
    LOADAVX_16x4;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 16, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);
        __m256i accum2 = _mm256_set1_epi16(0);
        __m256i accum3 = _mm256_set1_epi16(0);
        __m256i accum4 = _mm256_set1_epi16(0);
        kernelavx16x4(r0b0a, r1b0a, r2b0a, r3b0a,
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm256_add_epi32(resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm256_add_epi32(resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm256_add_epi32(resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}

FORCEINLINE void BlockHandlerAVX::HandleBlock16x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  
        int /*blockCnt*/, __m256i* resultStorage) 
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 16, 1, k);
    short* currA = &newA[aOffset];
    LOADAVX_16x1;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 16, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);

        kernelavx16x1(r0b0a, currB, &accum1);

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
    }

}



FORCEINLINE void BlockHandlerAVX::HandleBlock32x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m256i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 32, 4, k);
    short* currA = &newA[aOffset];
    LOADAVX_32x4;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 32, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);
        __m256i accum2 = _mm256_set1_epi16(0);
        __m256i accum3 = _mm256_set1_epi16(0);
        __m256i accum4 = _mm256_set1_epi16(0);				

        kernelavx32x4(
                r0b0a, r0b0b, 
                r1b0a, r1b0b, 
                r2b0a, r2b0b, 
                r3b0a, r3b0b, 
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}

FORCEINLINE void BlockHandlerAVX::HandleBlock32x1(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m256i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 32, 1, k);
    short* currA = &newA[aOffset];
    LOADAVX_32x1;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 32, n)];

        __m256i accum1 = _mm256_set1_epi16(0);


        kernelavx32x1(
                r0b0a, r0b0b, currB, &accum1);

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(0, c, n)], accum1);

    }
}

FORCEINLINE void BlockHandlerAVX::HandleBlock64x4(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m256i* resultStorage)
{

    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 64, 4, k);
    short* currA = &newA[aOffset];
    LOADAVX_64x4;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 64, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);
        __m256i accum2 = _mm256_set1_epi16(0);
        __m256i accum3 = _mm256_set1_epi16(0);
        __m256i accum4 = _mm256_set1_epi16(0);				

        kernelavx64x4(
                r0b0a, r0b0b, r0b0c, r0b0d, 
                r1b0a, r1b0b, r1b0c, r1b0d, 
                r2b0a, r2b0b, r2b0c, r2b0d, 
                r3b0a, r3b0b, r3b0c, r3b0d, 
                currB, &accum1, &accum2, &accum3, &accum4);

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(0, c, n)], accum1);
        resultStorage[RowColToOffset(1, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(1, c, n)], accum2);
        resultStorage[RowColToOffset(2, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(2, c, n)], accum3);
        resultStorage[RowColToOffset(3, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(3, c, n)], accum4);
    }
}

FORCEINLINE void BlockHandlerAVX::HandleBlock64x1(int currBlock, int startRow, int k, int n, short* newA, short* B, 
        int /*blockCnt*/, __m256i* resultStorage)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 64, 4, k);
    short* currA = &newA[aOffset];
    LOADAVX_64x1;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {

        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 64, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);


        kernelavx64x1(
                r0b0a, r0b0b, r0b0c, r0b0d,
                currB, &accum1);

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32(resultStorage[RowColToOffset(0, c, n)], accum1);
    }
}





FORCEINLINE void BlockHandlerAVX::HandleBlock128x4(int currBlock, int startRow, int k, int n, short* newA, short* B,  
        int blockCnt, __m256i* resultStorage, VectorT* /*subtractMe*/)
{

    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 128, 4, k);
    int aOffset2 = RowToColOffsetRewrittenA(startRow, currBlock + 1, 128, 4, k);
    short* currA = &newA[aOffset];
    short* currA2 = &newA[aOffset2];
    LOADAVX_128x4;
    LOADAVX2_128x4;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 128, n)];
        short* currB2 = &B[RowToColOffsetRewrittenB(c, currBlock + 1, 128, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);
        __m256i accum2 = _mm256_set1_epi16(0);
        __m256i accum3 = _mm256_set1_epi16(0);
        __m256i accum4 = _mm256_set1_epi16(0);
        __m256i accum5 = _mm256_set1_epi16(0);
        __m256i accum6 = _mm256_set1_epi16(0);
        __m256i accum7 = _mm256_set1_epi16(0);
        __m256i accum8 = _mm256_set1_epi16(0);

        kernelavx128x4(
                r0b0a, r0b0b, r0b0c, r0b0d, r0b0e, r0b0f, r0b0g, r0b0h,
                r1b0a, r1b0b, r1b0c, r1b0d, r1b0e, r1b0f, r1b0g, r1b0h,
                r2b0a, r2b0b, r2b0c, r2b0d, r2b0e, r2b0f, r2b0g, r2b0h,
                r3b0a, r3b0b, r3b0c, r3b0d, r3b0e, r3b0f, r3b0g, r3b0h,
                currB, &accum1, &accum2, &accum3, &accum4);

        if (blockCnt > 1)
        {
            kernelavx128x4(
                    r0b0a2, r0b0b2, r0b0c2, r0b0d2, r0b0e2, r0b0f2, r0b0g2, r0b0h2,
                    r1b0a2, r1b0b2, r1b0c2, r1b0d2, r1b0e2, r1b0f2, r1b0g2, r1b0h2,
                    r2b0a2, r2b0b2, r2b0c2, r2b0d2, r2b0e2, r2b0f2, r2b0g2, r2b0h2,
                    r3b0a2, r3b0b2, r3b0c2, r3b0d2, r3b0e2, r3b0f2, r3b0g2, r3b0h2,
                    currB2, &accum5, &accum6, &accum7, &accum8);
        }


        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(0, c, n)], _mm256_add_epi32(accum1,  accum5));
        resultStorage[RowColToOffset(1, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(1, c, n)], _mm256_add_epi32(accum2,  accum6));
        resultStorage[RowColToOffset(2, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(2, c, n)], _mm256_add_epi32(accum3,  accum7));
        resultStorage[RowColToOffset(3, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(3, c, n)], _mm256_add_epi32(accum4,  accum8));
    }
}


FORCEINLINE void BlockHandlerAVX::HandleBlock128x1(int currBlock, int startRow, int k, int n, short* newA, short* B,  
        int blockCnt, __m256i* resultStorage, VectorT* /*subtractMe*/)
{
    int aOffset = RowToColOffsetRewrittenA(startRow, currBlock, 128, 4, k);
    int aOffset2 = RowToColOffsetRewrittenA(startRow, currBlock + 1, 128, 4, k);
    short* currA = &newA[aOffset];
    short* currA2 = &newA[aOffset2];
    LOADAVX_128x1;
    LOADAVX2_128x1;
    //#pragma omp parallel for
    for (int c = 0; c < n; ++c)
    {
        short* currB = &B[RowToColOffsetRewrittenB(c, currBlock, 128, n)];
        short* currB2 = &B[RowToColOffsetRewrittenB(c, currBlock + 1, 128, n)];

        //The gain comes when we have all the row values loaded up
        //together and we multiply them all times each column, saving m_rowsPerBlock column
        //loads.

        __m256i accum1 = _mm256_set1_epi16(0);
        __m256i accum2 = _mm256_set1_epi16(0);
        kernelavx128x1(
                r0b0a, r0b0b, r0b0c, r0b0d, r0b0e, r0b0f, r0b0g, r0b0h,
                currB, &accum1);

        if (blockCnt > 1)
        {
            kernelavx128x1(
                    r0b0a2, r0b0b2, r0b0c2, r0b0d2, r0b0e2, r0b0f2, r0b0g2, r0b0h2,
                    currB2, &accum1);
        }

        resultStorage[RowColToOffset(0, c, n)] = _mm256_add_epi32( resultStorage[RowColToOffset(0, c, n)], _mm256_add_epi32(accum1,  accum2));
    }
}

FORCEINLINE void BlockHandlerAVX::kernelsse8x1(__m128i xmmRow0,
        short* B, __m128i* return1)
{
    __m128i xmmCol0 = _mm_load_si128((__m128i*)B);
    __m128i result1 = _mm_madd_epi16(xmmRow0, xmmCol0);
    *return1 = result1;
}


FORCEINLINE void BlockHandlerAVX::kernelsse8x4(__m128i xmmRow0, __m128i xmmRow1, __m128i xmmRow2, __m128i xmmRow3, 
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
FORCEINLINE void BlockHandlerAVX::kernelavx16x1(__m256i xmmRow0B0a, 
        short* B, __m256i* return1)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);

    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);


    *return1 = r0b0axc0b0a;
}




FORCEINLINE void BlockHandlerAVX::kernelavx16x4(__m256i xmmRow0B0a, __m256i xmmRow1B0a, __m256i xmmRow2B0a, __m256i xmmRow3B0a, 
        short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);

    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    //Result for row 1
    __m256i r1b0axc0b0a = _mm256_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    //Result for row 2
    __m256i r2b0axc0b0a = _mm256_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    //Result for row 3
    __m256i r3b0axc0b0a = _mm256_madd_epi16(xmmRow3B0a, xmmCol0B0a);

    *return1 = r0b0axc0b0a;
    *return2 = r1b0axc0b0a;
    *return3 = r2b0axc0b0a;
    *return4 = r3b0axc0b0a;
}

FORCEINLINE void BlockHandlerAVX::kernelavx32x1(
        __m256i xmmRow0B0a, __m256i xmmRow0B0b, 
        short* B, __m256i* return1)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);
    __m256i xmmCol0B0b = _mm256_load_si256((__m256i*)B + 1);

    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m256i r0b0bxc0b0b = _mm256_madd_epi16(xmmRow0B0b, xmmCol0B0b);

    __m256i result1a = _mm256_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);


    *return1 = result1a;
}



FORCEINLINE void BlockHandlerAVX::kernelavx32x4(
        __m256i xmmRow0B0a, __m256i xmmRow0B0b, 
        __m256i xmmRow1B0a, __m256i xmmRow1B0b, 
        __m256i xmmRow2B0a, __m256i xmmRow2B0b, 
        __m256i xmmRow3B0a, __m256i xmmRow3B0b, 
        short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);
    __m256i xmmCol0B0b = _mm256_load_si256((__m256i*)B + 1);

    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m256i r0b0bxc0b0b = _mm256_madd_epi16(xmmRow0B0b, xmmCol0B0b);

    __m256i result1a = _mm256_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);



    //Result for row 1
    __m256i r1b0axc0b0a = _mm256_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m256i r1b0bxc0b0b = _mm256_madd_epi16(xmmRow1B0b, xmmCol0B0b);

    __m256i result2a = _mm256_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);

    //Result for row 2
    __m256i r2b0axc0b0a = _mm256_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m256i r2b0bxc0b0b = _mm256_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m256i result3a = _mm256_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);



    //Result for row 3
    __m256i r3b0axc0b0a = _mm256_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m256i r3b0bxc0b0b = _mm256_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m256i result4a = _mm256_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);




    *return1 = result1a;
    *return2 = result2a;
    *return3 = result3a;
    *return4 = result4a;
}

FORCEINLINE void BlockHandlerAVX::kernelavx64x1(
        __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
        short* B, __m256i* return1)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);
    __m256i xmmCol0B0b = _mm256_load_si256((__m256i*)B + 1);
    __m256i xmmCol0B0c = _mm256_load_si256((__m256i*)B + 2);
    __m256i xmmCol0B0d = _mm256_load_si256((__m256i*)B + 3);
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m256i r0b0bxc0b0b = _mm256_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m256i r0b0cxc0b0c = _mm256_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m256i r0b0dxc0b0d = _mm256_madd_epi16(xmmRow0B0d, xmmCol0B0d);

    __m256i result1a = _mm256_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m256i result1b = _mm256_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);


    __m256i result1ab = _mm256_add_epi32(result1a, result1b);


    *return1 = result1ab;
    //std::cout << "Returning " << u.i[0] << " + " << u.i[4] << "(" << u.i[0] + u.i[4] << ") for first row" << std::endl;
}



FORCEINLINE void BlockHandlerAVX::kernelavx64x4(
        __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
        __m256i xmmRow1B0a, __m256i xmmRow1B0b, __m256i xmmRow1B0c, __m256i xmmRow1B0d,
        __m256i xmmRow2B0a, __m256i xmmRow2B0b, __m256i xmmRow2B0c, __m256i xmmRow2B0d,
        __m256i xmmRow3B0a, __m256i xmmRow3B0b, __m256i xmmRow3B0c, __m256i xmmRow3B0d,
        short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);
    __m256i xmmCol0B0b = _mm256_load_si256((__m256i*)B + 1);
    __m256i xmmCol0B0c = _mm256_load_si256((__m256i*)B + 2);
    __m256i xmmCol0B0d = _mm256_load_si256((__m256i*)B + 3);

    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m256i r0b0bxc0b0b = _mm256_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m256i r0b0cxc0b0c = _mm256_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m256i r0b0dxc0b0d = _mm256_madd_epi16(xmmRow0B0d, xmmCol0B0d);

    __m256i result1a = _mm256_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m256i result1b = _mm256_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);


    __m256i result1ab = _mm256_add_epi32(result1a, result1b);

    //Result for row 1
    __m256i r1b0axc0b0a = _mm256_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m256i r1b0bxc0b0b = _mm256_madd_epi16(xmmRow1B0b, xmmCol0B0b);
    __m256i r1b0cxc0b0c = _mm256_madd_epi16(xmmRow1B0c, xmmCol0B0c);
    __m256i r1b0dxc0b0d = _mm256_madd_epi16(xmmRow1B0d, xmmCol0B0d);

    __m256i result2a = _mm256_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);
    __m256i result2b = _mm256_add_epi32(r1b0cxc0b0c, r1b0dxc0b0d);
    __m256i result2ab = _mm256_add_epi32(result2a, result2b);

    //Result for row 2
    __m256i r2b0axc0b0a = _mm256_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m256i r2b0bxc0b0b = _mm256_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m256i r2b0cxc0b0c = _mm256_madd_epi16(xmmRow2B0c, xmmCol0B0c);
    __m256i r2b0dxc0b0d = _mm256_madd_epi16(xmmRow2B0d, xmmCol0B0d);
    __m256i result3a = _mm256_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);
    __m256i result3b = _mm256_add_epi32(r2b0cxc0b0c, r2b0dxc0b0d);

    __m256i result3ab = _mm256_add_epi32(result3a, result3b);


    //Result for row 3
    __m256i r3b0axc0b0a = _mm256_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m256i r3b0bxc0b0b = _mm256_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m256i r3b0cxc0b0c = _mm256_madd_epi16(xmmRow3B0c, xmmCol0B0c);
    __m256i r3b0dxc0b0d = _mm256_madd_epi16(xmmRow3B0d, xmmCol0B0d);

    __m256i result4a = _mm256_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);
    __m256i result4b = _mm256_add_epi32(r3b0cxc0b0c, r3b0dxc0b0d);
    __m256i result4ab = _mm256_add_epi32(result4a, result4b);

    *return1 = result1ab;
    *return2 = result2ab;
    *return3 = result3ab;
    *return4 = result4ab;
}

FORCEINLINE void BlockHandlerAVX::kernelavx128x1(
        __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
        __m256i xmmRow0B0e, __m256i xmmRow0B0f, __m256i xmmRow0B0g, __m256i xmmRow0B0h,
        short* B, __m256i* return1)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);
    __m256i xmmCol0B0b = _mm256_load_si256((__m256i*)(B + 16));
    __m256i xmmCol0B0c = _mm256_load_si256((__m256i*)(B + 32));
    __m256i xmmCol0B0d = _mm256_load_si256((__m256i*)(B + 48));
    __m256i xmmCol0B0e = _mm256_load_si256((__m256i*)(B + 64));
    __m256i xmmCol0B0f = _mm256_load_si256((__m256i*)(B + 80));
    __m256i xmmCol0B0g = _mm256_load_si256((__m256i*)(B + 96));
    __m256i xmmCol0B0h = _mm256_load_si256((__m256i*)(B + 112));
    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m256i r0b0bxc0b0b = _mm256_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m256i r0b0cxc0b0c = _mm256_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m256i r0b0dxc0b0d = _mm256_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m256i r0b0exc0b0e = _mm256_madd_epi16(xmmRow0B0e, xmmCol0B0e);
    __m256i r0b0fxc0b0f = _mm256_madd_epi16(xmmRow0B0f, xmmCol0B0f);
    __m256i r0b0gxc0b0g = _mm256_madd_epi16(xmmRow0B0g, xmmCol0B0g);
    __m256i r0b0hxc0b0h = _mm256_madd_epi16(xmmRow0B0h, xmmCol0B0h);
    __m256i result1a = _mm256_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m256i result1b = _mm256_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m256i result1c = _mm256_add_epi32(r0b0exc0b0e, r0b0fxc0b0f);
    __m256i result1d = _mm256_add_epi32(r0b0gxc0b0g, r0b0hxc0b0h);


    __m256i result1ab = _mm256_add_epi32(result1a, result1b);
    __m256i result1cd = _mm256_add_epi32(result1c, result1d);
    __m256i result1abcd = _mm256_add_epi32(result1ab, result1cd);

    *return1 = result1abcd;
    //std::cout << "Returning " << u.i[0] << " + " << u.i[4] << "(" << u.i[0] + u.i[4] << ") for first row" << std::endl;
}

FORCEINLINE void BlockHandlerAVX::kernelavx128x4(
        __m256i xmmRow0B0a, __m256i xmmRow0B0b, __m256i xmmRow0B0c, __m256i xmmRow0B0d,
        __m256i xmmRow0B0e, __m256i xmmRow0B0f, __m256i xmmRow0B0g, __m256i xmmRow0B0h,
        __m256i xmmRow1B0a, __m256i xmmRow1B0b, __m256i xmmRow1B0c, __m256i xmmRow1B0d,
        __m256i xmmRow1B0e, __m256i xmmRow1B0f, __m256i xmmRow1B0g, __m256i xmmRow1B0h,
        __m256i xmmRow2B0a, __m256i xmmRow2B0b, __m256i xmmRow2B0c, __m256i xmmRow2B0d,
        __m256i xmmRow2B0e, __m256i xmmRow2B0f, __m256i xmmRow2B0g, __m256i xmmRow2B0h,
        __m256i xmmRow3B0a, __m256i xmmRow3B0b, __m256i xmmRow3B0c, __m256i xmmRow3B0d,
        __m256i xmmRow3B0e, __m256i xmmRow3B0f, __m256i xmmRow3B0g, __m256i xmmRow3B0h,
        short* B, __m256i* return1, __m256i* return2, __m256i * return3, __m256i* return4)
{

    __m256i xmmCol0B0a = _mm256_load_si256((__m256i*)B);
    __m256i xmmCol0B0b = _mm256_load_si256((__m256i*)(B + 16));
    __m256i xmmCol0B0c = _mm256_load_si256((__m256i*)(B + 32));
    __m256i xmmCol0B0d = _mm256_load_si256((__m256i*)(B + 48));
    __m256i xmmCol0B0e = _mm256_load_si256((__m256i*)(B + 64));
    __m256i xmmCol0B0f = _mm256_load_si256((__m256i*)(B + 80));
    __m256i xmmCol0B0g = _mm256_load_si256((__m256i*)(B + 96));
    __m256i xmmCol0B0h = _mm256_load_si256((__m256i*)(B + 112));
    //Result for row 0
    //Nomenclature:
    //r0b0axc0b0a  means "Row zero block zero part A times column zero block zero part A. (Blocks > 8 take up > 1 __m256i each (xmm registers))
    __m256i r0b0axc0b0a = _mm256_madd_epi16(xmmRow0B0a, xmmCol0B0a);
    __m256i r0b0bxc0b0b = _mm256_madd_epi16(xmmRow0B0b, xmmCol0B0b);
    __m256i r0b0cxc0b0c = _mm256_madd_epi16(xmmRow0B0c, xmmCol0B0c);
    __m256i r0b0dxc0b0d = _mm256_madd_epi16(xmmRow0B0d, xmmCol0B0d);
    __m256i r0b0exc0b0e = _mm256_madd_epi16(xmmRow0B0e, xmmCol0B0e);
    __m256i r0b0fxc0b0f = _mm256_madd_epi16(xmmRow0B0f, xmmCol0B0f);
    __m256i r0b0gxc0b0g = _mm256_madd_epi16(xmmRow0B0g, xmmCol0B0g);
    __m256i r0b0hxc0b0h = _mm256_madd_epi16(xmmRow0B0h, xmmCol0B0h);
    __m256i result1a = _mm256_add_epi32(r0b0axc0b0a, r0b0bxc0b0b);
    __m256i result1b = _mm256_add_epi32(r0b0cxc0b0c, r0b0dxc0b0d);
    __m256i result1c = _mm256_add_epi32(r0b0exc0b0e, r0b0fxc0b0f);
    __m256i result1d = _mm256_add_epi32(r0b0gxc0b0g, r0b0hxc0b0h);


    __m256i result1ab = _mm256_add_epi32(result1a, result1b);
    __m256i result1cd = _mm256_add_epi32(result1c, result1d);
    __m256i result1abcd = _mm256_add_epi32(result1ab, result1cd);

    //Result for row 1
    __m256i r1b0axc0b0a = _mm256_madd_epi16(xmmRow1B0a, xmmCol0B0a);
    __m256i r1b0bxc0b0b = _mm256_madd_epi16(xmmRow1B0b, xmmCol0B0b);
    __m256i r1b0cxc0b0c = _mm256_madd_epi16(xmmRow1B0c, xmmCol0B0c);
    __m256i r1b0dxc0b0d = _mm256_madd_epi16(xmmRow1B0d, xmmCol0B0d);
    __m256i r1b0exc0b0e = _mm256_madd_epi16(xmmRow1B0e, xmmCol0B0e);
    __m256i r1b0fxc0b0f = _mm256_madd_epi16(xmmRow1B0f, xmmCol0B0f);
    __m256i r1b0gxc0b0g = _mm256_madd_epi16(xmmRow1B0g, xmmCol0B0g);
    __m256i r1b0hxc0b0h = _mm256_madd_epi16(xmmRow1B0h, xmmCol0B0h);

    __m256i result2a = _mm256_add_epi32(r1b0axc0b0a, r1b0bxc0b0b);
    __m256i result2b = _mm256_add_epi32(r1b0cxc0b0c, r1b0dxc0b0d);
    __m256i result2c = _mm256_add_epi32(r1b0exc0b0e, r1b0fxc0b0f);
    __m256i result2d = _mm256_add_epi32(r1b0gxc0b0g, r1b0hxc0b0h);
    __m256i result2ab = _mm256_add_epi32(result2a, result2b);
    __m256i result2cd = _mm256_add_epi32(result2c, result2d);
    __m256i result2abcd = _mm256_add_epi32(result2ab, result2cd);

    //Result for row 2
    __m256i r2b0axc0b0a = _mm256_madd_epi16(xmmRow2B0a, xmmCol0B0a);
    __m256i r2b0bxc0b0b = _mm256_madd_epi16(xmmRow2B0b, xmmCol0B0b);
    __m256i r2b0cxc0b0c = _mm256_madd_epi16(xmmRow2B0c, xmmCol0B0c);
    __m256i r2b0dxc0b0d = _mm256_madd_epi16(xmmRow2B0d, xmmCol0B0d);
    __m256i r2b0exc0b0e = _mm256_madd_epi16(xmmRow2B0e, xmmCol0B0e);
    __m256i r2b0fxc0b0f = _mm256_madd_epi16(xmmRow2B0f, xmmCol0B0f);
    __m256i r2b0gxc0b0g = _mm256_madd_epi16(xmmRow2B0g, xmmCol0B0g);
    __m256i r2b0hxc0b0h = _mm256_madd_epi16(xmmRow2B0h, xmmCol0B0h);
    __m256i result3a = _mm256_add_epi32(r2b0axc0b0a, r2b0bxc0b0b);
    __m256i result3b = _mm256_add_epi32(r2b0cxc0b0c, r2b0dxc0b0d);
    __m256i result3c = _mm256_add_epi32(r2b0exc0b0e, r2b0fxc0b0f);
    __m256i result3d = _mm256_add_epi32(r2b0gxc0b0g, r2b0hxc0b0h);

    __m256i result3ab = _mm256_add_epi32(result3a, result3b);
    __m256i result3cd = _mm256_add_epi32(result3c, result3d);
    __m256i result3abcd = _mm256_add_epi32(result3ab, result3cd);


    //Result for row 3
    __m256i r3b0axc0b0a = _mm256_madd_epi16(xmmRow3B0a, xmmCol0B0a);
    __m256i r3b0bxc0b0b = _mm256_madd_epi16(xmmRow3B0b, xmmCol0B0b);
    __m256i r3b0cxc0b0c = _mm256_madd_epi16(xmmRow3B0c, xmmCol0B0c);
    __m256i r3b0dxc0b0d = _mm256_madd_epi16(xmmRow3B0d, xmmCol0B0d);
    __m256i r3b0exc0b0e = _mm256_madd_epi16(xmmRow3B0e, xmmCol0B0e);
    __m256i r3b0fxc0b0f = _mm256_madd_epi16(xmmRow3B0f, xmmCol0B0f);
    __m256i r3b0gxc0b0g = _mm256_madd_epi16(xmmRow3B0g, xmmCol0B0g);
    __m256i r3b0hxc0b0h = _mm256_madd_epi16(xmmRow3B0h, xmmCol0B0h);

    __m256i result4a = _mm256_add_epi32(r3b0axc0b0a, r3b0bxc0b0b);
    __m256i result4b = _mm256_add_epi32(r3b0cxc0b0c, r3b0dxc0b0d);
    __m256i result4c = _mm256_add_epi32(r3b0exc0b0e, r3b0fxc0b0f);
    __m256i result4d = _mm256_add_epi32(r3b0gxc0b0g, r3b0hxc0b0h);
    __m256i result4ab = _mm256_add_epi32(result4a, result4b);
    __m256i result4cd = _mm256_add_epi32(result4c, result4d);
    __m256i result4abcd = _mm256_add_epi32(result4ab, result4cd);

    //Now we can just add horizontally




    *return1 = result1abcd;
    *return2 = result2abcd;
    *return3 = result3abcd;
    *return4 = result4abcd;
}


}}}
