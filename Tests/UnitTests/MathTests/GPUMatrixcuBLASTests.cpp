//
// <copyright file="GPUMatrixcuBLASTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPUMatrix Unit tests should go here
//
// Note from original repo: *_NoExceptionOnly_Test kind of tests tests only that the method runs without exception. They don't test correctness
// TODO: test correctness?, add tests for other cuBLAS methods (check which ones are used)
//
#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include "..\..\..\Math\Math\GPUMatrix.h"

#pragma warning (disable: 4244 4245 4305)       // conversions and truncations; we don't care in this test project

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
					float alpha = 0.4;
					GPUMatrix<float> M0_GPU(12, 5, 0 /*deviceId*/);
					GPUMatrix<float> M1_GPU(5, 11, 0 /*deviceId*/);
					GPUMatrix<float> M2_GPU(12, 11, 0 /*deviceId*/);
					GPUMatrix<float>::MultiplyAndWeightedAdd(0.1, M0_GPU, false, M1_GPU, false, alpha, M2_GPU);

					// original test didn'thave any checks, tests need to be improved
					BOOST_CHECK(!M0_GPU.IsEmpty());
				}

				BOOST_AUTO_TEST_CASE(GPUBlasScale_NoExceptionOnly)
				{
					float scale = 0.5;
					GPUMatrix<float> M0_GPU(12, 53, 0 /*deviceId*/);
					GPUMatrix<float> M1_GPU(12, 53, 0 /*deviceId*/);
					GPUMatrix<float>::Scale(scale, M0_GPU);

					// original test didn'thave any checks, tests need to be improved
					BOOST_CHECK(!M0_GPU.IsEmpty());
				}

				BOOST_AUTO_TEST_CASE(GPUBlasInnerProduct_NoExceptionOnly)
				{
					float *arr = new float[100];
					for (int i = 0; i<100; i++) arr[i] = 1;
					GPUMatrix<float> AG(10, 10, 0 /*deviceId*/, arr, matrixFlagNormal);
					GPUMatrix<float> BG(10, 10, 0 /*deviceId*/, arr, matrixFlagNormal);
					GPUMatrix<float> CG(1, 10, 0 /*deviceId*/, arr, matrixFlagNormal);
					GPUMatrix<float>::InnerProduct(AG, BG, CG, true);

					// original test didn'thave any checks, tests need to be improved
					BOOST_CHECK(!AG.IsEmpty());
				}

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}