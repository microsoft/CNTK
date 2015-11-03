//
// <copyright file="GPUMatrixTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPUMatrix unit tests should go here
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
                BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

				BOOST_AUTO_TEST_CASE(GPUMatrixDummy)
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

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}