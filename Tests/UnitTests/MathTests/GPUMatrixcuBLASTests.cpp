//
// <copyright file="GPUMatrixcuBLASTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPUMatrix Cuda BLAS library tests should go here
//
// Note from original repo: *_NoExceptionOnly_Test kind of tests tests only that the method runs without exception. They don't test correctness
// TODO: test correctness? (no need to test cuda internals), add tests for other cuBLAS methods? (check which ones are used)
//
#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include "..\..\..\Math\Math\GPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                BOOST_AUTO_TEST_SUITE(GPUMatrixcuBLASSuite)

				BOOST_AUTO_TEST_CASE(GPUBlasMultiplyAndWeightedAdd_NoExceptionOnly)
				{
					float alpha = 0.1f;
					float beta = 0.4f;
					GPUMatrix<float> M0_GPU(12, 5, 0 /*deviceId*/);
					GPUMatrix<float> M1_GPU(5, 11, 0 /*deviceId*/);
					GPUMatrix<float> M2_GPU(12, 11, 0 /*deviceId*/);
					GPUMatrix<float>::MultiplyAndWeightedAdd(alpha, M0_GPU, false, M1_GPU, false, beta, M2_GPU);

					// original test didn't have any checks, tests need to be improved
					BOOST_CHECK(!M0_GPU.IsEmpty());
				}

				BOOST_AUTO_TEST_CASE(GPUBlasScale_NoExceptionOnly)
				{
					float scale = 0.5f;
					GPUMatrix<float> M0_GPU(12, 53, 0 /*deviceId*/);
					GPUMatrix<float>::Scale(scale, M0_GPU);

					// original test didn't have any checks, tests need to be improved
					BOOST_CHECK(!M0_GPU.IsEmpty());
				}

				BOOST_AUTO_TEST_CASE(GPUBlasInnerProduct_NoExceptionOnly)
				{
					float *arr = new float[100];
					for (int i = 0; i<100; i++) arr[i] = 1.0f;
					GPUMatrix<float> AG(10, 10, 0 /*deviceId*/, arr, matrixFlagNormal);
					GPUMatrix<float> BG(10, 10, 0 /*deviceId*/, arr, matrixFlagNormal);
					GPUMatrix<float> CG(1, 10, 0 /*deviceId*/, arr, matrixFlagNormal);
					GPUMatrix<float>::InnerProduct(AG, BG, CG, true);

					// original test didn't have any checks, tests need to be improved
					BOOST_CHECK(!AG.IsEmpty());
				}

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}