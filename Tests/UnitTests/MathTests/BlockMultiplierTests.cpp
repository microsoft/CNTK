//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#include "stdafx.h"
#include "../../../Source/Math/BlockMultiplier.h"

namespace Microsoft { namespace MSR { namespace CNTK { namespace TEST {

//The simplest possible matrix multiplier, used here as a check.
template<typename ScalarAT, typename ScalarBT, typename ScalarCT, int MAXRANGE = 1 << ((8 * sizeof(ScalarAT)) - 3)> class ReferenceMultiplier
{
    public:

        typedef ScalarAT ScalarAT;
        typedef ScalarBT ScalarBT;
        typedef ScalarCT ScalarCT;

        static const int MAXRANGE = MAXRANGE;

        ScalarBT* PrepareB(ScalarBT* oldB, int k, int n) { return oldB; }
        static ScalarAT* CreateMatrixA(int m, int n)
        {
            return CreateMatrix<ScalarAT>(m, n);
        }
        static ScalarBT* CreateMatrixB(int m, int n)
        {
            return CreateMatrix<ScalarBT>(m, n);
        }
        static ScalarCT* CreateMatrixC(int m, int n)
        {
            return CreateMatrix<ScalarCT>(m, n);
        }
        template<typename ScalarT> static ScalarT* CreateMatrix(int m, int n, ScalarT initVal = ScalarT())
        {

            ScalarT* ret = new ScalarT[m*n];
            if (initVal != ScalarT())
            {
                for (int i = 0; i < m * n; ++i)
                {
                    ret[i] = initVal;
                }
            }
            return ret;
        }

        template<typename ScalarT> static void FreeMatrix(ScalarT* destroyMe)
        {
            delete[] destroyMe;
        }

        void MultiplyMatrices(ScalarAT* A, int m, int k, ScalarBT* B, int n, ScalarCT* C, ScalarAT alpha = (ScalarAT)1, ScalarBT beta = (ScalarBT)0)
        {

            alpha;
            beta;
            for (int r = 0; r < m; ++r)
            {
                for (int c = 0; c < n; ++c)
                {
                    ScalarCT accum = (ScalarCT)0;
                    for (int d = 0; d < k; ++d)
                    {
                        ScalarCT prod = (ScalarCT)(A[(k * r) + d]) * (ScalarCT)(B[(n*d) + c]);
                        bool signsIdentical = ((accum > 0) == (prod > 0));
                        //signed overflow occurs iff signs identical and sum different in sign from operators.
                        accum += prod;
                        if (signsIdentical && (accum > 0) != (prod > 0))
                        {
                            throw std::runtime_error("overflow!");
                        }
                    }
                    C[(r * n) + c] = accum;
                }
            }
        }
};

    template<typename ScalarAT, typename ScalarBT, typename ScalarCT, typename MultiplierT>static void TestMultiplierSub(
            int m, int k, int n, MultiplierT& testMult, int numThreads = 1, ScalarCT epsilon = ScalarCT())

{
    epsilon;
    testMult.SetNumThreads(numThreads);
    ReferenceMultiplier<ScalarAT, ScalarBT, ScalarCT> refMult;


    ScalarAT* refA = refMult.CreateMatrixA(m, k);
    ScalarBT* refB = refMult.CreateMatrixB(k, n);
    ScalarCT* refC = refMult.CreateMatrixC(m, n);
    ScalarAT* testA = testMult.CreateMatrixA(m, k);
    ScalarBT* testB = testMult.CreateMatrixB(k, n);
    ScalarCT* testC = testMult.CreateMatrixC(m, n);

    RandInitIntMatrix<ScalarAT>(refA, m, k, 63);
    RandInitIntMatrix<ScalarBT>(refB, k, n, 63);
    memcpy(testA, refA, sizeof(ScalarAT) * m * k);
    memcpy(testB, refB, sizeof(ScalarBT) * k * n);


    ScalarBT* testBPrepared = testMult.PrepareB(testB, k, n);

    refMult.MultiplyMatrices(refA, m, k, refB, n, refC);

    //Make sure we can multiply twice on the same matrix correctly.
    for (int i = 0; i < 2; ++i)
    {
        testMult.MultiplyMatrices(testA, m, k, testBPrepared, n, testC);

        //This will cause test failure and dump matrix to Octave format for debugging if they don't match
        CompareMatricesAndDump(refC, testC, m, k, n);
        memset(testC, (ScalarCT)0, sizeof(ScalarCT) * m * n);
    }

    refMult.FreeMatrix(refA);
    refMult.FreeMatrix(refB);
    refMult.FreeMatrix(refC);
    testMult.FreeMatrix(testA);
    testMult.FreeMatrix(testB);
    testMult.FreeMatrix(testC);
    if (testBPrepared != testB)
    {
        testMult.FreeMatrix(testBPrepared);
    }

}


    template<typename ScalarAT, typename ScalarBT, typename ScalarCT, typename MultiplierT>static void TestMultiplierSub(
            int m, int k, int n, int numThreads = 1, ScalarCT epsilon = ScalarCT())
{
    MultiplierT testMult;
    TestMultiplierSub<ScalarAT, ScalarBT, ScalarCT, MultiplierT>(m, k, n, testMult, numThreads, epsilon);
}

template<typename ScalarCT> void CompareMatricesAndDump(const ScalarCT* ref, const ScalarCT* test,
        int m, int /*k*/, int n)
{
    for (int i = 0; i < m * n; ++i)
    {
        BOOST_CHECK_EQUAL(ref[i], test[i]);
    }
}

BOOST_AUTO_TEST_SUITE(BlockMultiplierSuite)

BOOST_AUTO_TEST_CASE(BlockMultiplyTest)
{

    int m = 8;
    int k = 128;
    int n = 8;
    TestMultiplierSub<int16_t, int16_t, int32_t, BlockMultiplier<BlockHandlerSSE>>(m, k, n);
}

BOOST_AUTO_TEST_CASE(BlockMultiplyTestRowColMajor)
{
    int m = 4, k = 10, n = 8;
    TestMultiplierSub<int16_t, int16_t, int32_t, BlockMultiplier<BlockHandlerSSE>>(m, k, n);
    BlockMultiplier<BlockHandlerSSE> mult;
    
    // mult is row-major. Since A and B are col-major, deal with this by swapping arguments.
    // Effectively we're flipping the whole computation around a 45 degree angle
   
    int16_t* matA = mult.CreateMatrixA(m, k);
    int16_t* matB = mult.CreateMatrixB(k, n);
    int32_t* matC = mult.CreateMatrixC(m, n);
    float alpha = 1, beta = 0;
    for (int i = 0; i < 4; ++i)
    {
        matA[i] = matB[i] = (int16_t)i + 1;
    }
    int16_t* newA = mult.PrepareB(matA, k, m);
    // Flip!  m <-> n  and A <-> B
    mult.MultiplyMatrices(matB, n, k, newA, m, matC, (int16_t)alpha, (int16_t)beta);

    for (int i = 0; i < 4; ++i)
    {
        BOOST_CHECK_EQUAL(i + 1, matC[i]);
    }
    for (int i = 4; i < m * n; ++i)
    {
        BOOST_CHECK_EQUAL(0, matC[i]);
    }

    mult.FreeMatrix(matA);
    mult.FreeMatrix(matB);
    mult.FreeMatrix(matC);
    if (newA != matA)
        mult.FreeMatrix(newA);
}

BOOST_AUTO_TEST_SUITE_END()
}}}} //end namespaces
