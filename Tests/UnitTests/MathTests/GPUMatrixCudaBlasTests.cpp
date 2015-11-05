//
// <copyright file="GPUMatrixCudaBlasTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPUMatrix CUDA BLAS library tests should go here
//
#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "../../../Math/Math/GPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                const int deviceId = 0;
                const float epsilon = 0.0001f;

                BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

				BOOST_AUTO_TEST_CASE(GPUBlasMultiplyAndWeightedAdd)
				{
					const float alpha = 2.0f;
					const float beta = 0.42f;
					GPUMatrix<float> m0(12, 5, deviceId);
                    m0.SetValue(1.0f);
					GPUMatrix<float> m1(5, 11, deviceId);
                    m1.SetValue(1.0f);
                    GPUMatrix<float> m2(12, 11, deviceId);
                    m2.SetValue(1.0f);

                    // m2 = alpha * m0 * m1 + beta * m2
                    GPUMatrix<float>::MultiplyAndWeightedAdd(alpha, m0, false, m1, false, beta, m2);

                    GPUMatrix<float> mr(12, 11, deviceId);
                    mr.SetValue(10.42f);
                    BOOST_CHECK(m2.IsEqualTo(mr, epsilon));
				}

				BOOST_AUTO_TEST_CASE(GPUBlasScale)
				{
					const float scale = 0.5f;
					GPUMatrix<float> m0(12, 53, deviceId);
                    m0.SetValue(4.2f);
					GPUMatrix<float>::Scale(scale, m0);

                    GPUMatrix<float> mr(12, 53, deviceId);
                    mr.SetValue(2.1f);
                    BOOST_CHECK(m0.IsEqualTo(mr, epsilon));
                }

				BOOST_AUTO_TEST_CASE(GPUBlasInnerProduct)
				{
					float *arr = new float[100];
                    for (int i = 0; i < 100; i++)
                    {
                        arr[i] = 2.0f;
                    }

                    GPUMatrix<float> m0(10, 10, deviceId, arr, matrixFlagNormal);
					GPUMatrix<float> m1(10, 10, deviceId, arr, matrixFlagNormal);
					GPUMatrix<float> m2(1, 10, deviceId, arr, matrixFlagNormal);

                    GPUMatrix<float>::InnerProduct(m0, m1, m2, true);
                    GPUMatrix<float> mr(1, 10, deviceId);
                    mr.SetValue(40.0f);
                    BOOST_CHECK(m2.IsEqualTo(mr, epsilon));

                    GPUMatrix<float>::InnerProduct(m0, m1, m2, false);
                    BOOST_CHECK(m2.IsEqualTo(mr.Transpose(), epsilon));

                    delete[] arr;
                }

                // TODO: add tests for other CUDA BLAS methods?

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}