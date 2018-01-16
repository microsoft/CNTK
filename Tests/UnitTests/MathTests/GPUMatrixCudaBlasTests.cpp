//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// GPUMatrix CUDA BLAS library tests should go here
//
#include "stdafx.h"
#include "../../../Source/Math/GPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(GPUBlasMultiplyAndWeightedAdd, RandomSeedFixture)
{
    const float alpha = 2.0f;
    const float beta = 0.42f;
    GPUMatrix<float> m0(12, 5, c_deviceIdZero);
    m0.SetValue(1);
    GPUMatrix<float> m1(5, 11, c_deviceIdZero);
    m1.SetValue(1);
    GPUMatrix<float> m2(12, 11, c_deviceIdZero);
    m2.SetValue(1);

    // m2 = alpha * m0 * m1 + beta * m2
    GPUMatrix<float>::MultiplyAndWeightedAdd(alpha, m0, false, m1, false, beta, m2);

    GPUMatrix<float> mr(12, 11, c_deviceIdZero);
    mr.SetValue(10.42f);
    BOOST_CHECK(m2.IsEqualTo(mr, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUBlasBatchMatMul, RandomSeedFixture)
{

    GPUMatrix<float> m0(6, 5, c_deviceIdZero);
    m0.SetValue(1.0f);
    GPUMatrix<float> m1(6, 5, c_deviceIdZero);
    m1.SetValue(1.0f);
    GPUMatrix<float> m2(4, 5, c_deviceIdZero);
    m2.SetValue(std::nanf(""));
    GPUMatrix<float>::BatchMatMul(0.0f, m0, false, 2, m1, false, 2, m2, true);
    GPUMatrix<float> mr(4, 5, c_deviceIdZero);
    mr.SetValue(3.0f);
    BOOST_CHECK(m2.IsEqualTo(mr, c_epsilonFloatE4));

    GPUMatrix<float>::BatchMatMul(1.0f, m0, false, 2, m1, false, 2, m2, true);
    mr.SetValue(6.0f);
    BOOST_CHECK(m2.IsEqualTo(mr, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUBlasScale, RandomSeedFixture)
{
    const float scale = 0.5f;
    GPUMatrix<float> m0(12, 53, c_deviceIdZero);
    m0.SetValue(4.2f);
    GPUMatrix<float>::Scale(scale, m0);

    GPUMatrix<float> mr(12, 53, c_deviceIdZero);
    mr.SetValue(2.1f);
    BOOST_CHECK(m0.IsEqualTo(mr, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUBlasInnerProduct, RandomSeedFixture)
{
    GPUMatrix<float> m0(10, 10, c_deviceIdZero);
    GPUMatrix<float> m1(10, 10, c_deviceIdZero);
    GPUMatrix<float> m2(1, 10, c_deviceIdZero);
    m0.SetValue(2);
    m1.SetValue(2);
    m2.SetValue(2);

    GPUMatrix<float>::InnerProduct(m0, m1, m2, true);
    GPUMatrix<float> mr(1, 10, c_deviceIdZero);
    mr.SetValue(40);
    BOOST_CHECK(m2.IsEqualTo(mr, c_epsilonFloatE4));

    GPUMatrix<float>::InnerProduct(m0, m1, m2, false);
    BOOST_CHECK(m2.IsEqualTo(mr.Transpose(), c_epsilonFloatE4));
}

// TODO: add tests for other CUDA BLAS methods?

BOOST_AUTO_TEST_SUITE_END()
}
} } }
