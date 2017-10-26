//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Half data type GPU tests should go here
//
#include "stdafx.h"
#include "../../../Source/Math/GPUMatrix.h"
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/half.hpp"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

// Tests from GPUMatrixCudaBlasTests.cpp
BOOST_AUTO_TEST_SUITE(Half_GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(GPUBlasMultiplyAndWeightedAdd, RandomSeedFixture)
{
    const half alpha = 2.0f;
    const half beta = 0.42f;
    GPUMatrix<half> m0(12, 5, c_deviceIdZero);
    m0.SetValue(1);
    GPUMatrix<half> m1(5, 11, c_deviceIdZero);
    m1.SetValue(1);
    GPUMatrix<half> m2(12, 11, c_deviceIdZero);
    m2.SetValue(1);

    // m2 = alpha * m0 * m1 + beta * m2
    GPUMatrix<half>::MultiplyAndWeightedAdd(alpha, m0, false, m1, false, beta, m2);

    GPUMatrix<half> mr(12, 11, c_deviceIdZero);
    mr.SetValue(10.42f);
    BOOST_CHECK(m2.IsEqualTo(mr, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUBlasScale, RandomSeedFixture)
{
    const half scale = 0.5f;
    GPUMatrix<half> m0(12, 53, c_deviceIdZero);
    m0.SetValue(4.2f);
    GPUMatrix<half>::Scale(scale, m0);

    GPUMatrix<half> mr(12, 53, c_deviceIdZero);
    mr.SetValue(2.1f);
    BOOST_CHECK(m0.IsEqualTo(mr, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUBlasInnerProduct, RandomSeedFixture)
{
    GPUMatrix<half> m0(10, 10, c_deviceIdZero);
    GPUMatrix<half> m1(10, 10, c_deviceIdZero);
    GPUMatrix<half> m2(1, 10, c_deviceIdZero);
    m0.SetValue(2);
    m1.SetValue(2);
    m2.SetValue(2);

    GPUMatrix<half>::InnerProduct(m0, m1, m2, true);
    GPUMatrix<half> mr(1, 10, c_deviceIdZero);
    mr.SetValue(40);
    BOOST_CHECK(m2.IsEqualTo(mr, c_epsilonFloatE4));

    GPUMatrix<half>::InnerProduct(m0, m1, m2, false);
    BOOST_CHECK(m2.IsEqualTo(mr.Transpose(), c_epsilonFloatE4));
}

// TODO: add tests for other CUDA BLAS methods?

BOOST_AUTO_TEST_SUITE_END()

// Tests from GPUMatrixTests.cpp

BOOST_AUTO_TEST_SUITE(Half_GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(MatrixCopyAssignAcrossDevices, RandomSeedFixture)
{
    bool hasTwoGpus = false;
#ifndef CPUONLY
    auto gpus = GetAllGpusData();
    hasTwoGpus = (gpus.size() > 1);
#endif
    std::array<half, 6> array = { 1, 2, 3, 4, 5, 6 };

    {
        Matrix<half> m_gpu(2, 3, array.data(), c_deviceIdZero, matrixFlagNormal);
        Matrix<half> m_copy_gpu_0(m_gpu, c_deviceIdZero);
        if (hasTwoGpus)
            Matrix<half> m_copy_gpu_1(m_gpu, c_deviceIdZero + 1);
        Matrix<half> m_copy_cpu(m_gpu, -1);
    }

    {
        Matrix<half> m_cpu(2, 3, array.data(), -1, matrixFlagNormal);
        Matrix<half> m_copy_gpu_0(m_cpu, c_deviceIdZero);
        if (hasTwoGpus)
            Matrix<half> m_copy_gpu_1(m_cpu, c_deviceIdZero + 1);
        Matrix<half> m_copy_cpu(m_cpu, -1);
    }

    {
        Matrix<half> m_gpu(2, 3, array.data(), c_deviceIdZero, matrixFlagNormal);
        Matrix<half> m_copy_gpu_0(c_deviceIdZero);
        m_copy_gpu_0.AssignValuesOf(m_gpu);
        if (hasTwoGpus)
        {
            Matrix<half> m_copy_gpu_1(c_deviceIdZero + 1);
            m_copy_gpu_1.AssignValuesOf(m_gpu);
        }
        Matrix<half> m_copy_cpu(-1);
        m_copy_cpu.AssignValuesOf(m_gpu);
    }

    if (hasTwoGpus)
    {

        Matrix<half> m_gpu_0(2, 3, array.data(), c_deviceIdZero, matrixFlagNormal);
        Matrix<half> m_gpu_1(2, 3, c_deviceIdZero + 1, m_gpu_0.GetMatrixType(), m_gpu_0.GetFormat());
        try
        {
            // TODO: fix this!
            m_gpu_1.AssignValuesOf(m_gpu_0);
            BOOST_TEST(false, "Expected AssignValuesOf to fail.");
        }
        catch (...)
        {
        }
    }
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixConstructorNoFlag, RandomSeedFixture)
{
    // TODO: consider splitting into several tests
    GPUMatrix<half> m0(c_deviceIdZero);
    BOOST_CHECK(m0.IsEmpty());

    GPUMatrix<half> m1(12, 53, c_deviceIdZero);
    BOOST_CHECK_EQUAL(12, m1.GetNumRows());
    BOOST_CHECK_EQUAL(53, m1.GetNumCols());
    BOOST_CHECK_EQUAL(12 * 53, m1.GetNumElements());

    std::array<half, 2> array = {1, 14};
    m1.SetValue(1, 2, c_deviceIdZero, array.data());

    unique_ptr<half[]> result(m1.CopyToArray());
    BOOST_CHECK_EQUAL_COLLECTIONS(result.get(), result.get() + 2, array.begin(), array.end());

    GPUMatrix<half> m1Copy(m1);
    BOOST_CHECK(m1.IsEqualTo(m1Copy));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixConstructorFlagNormal, RandomSeedFixture)
{
    std::array<half, 6> array = {1, 2, 3, 4, 5, 6};
    GPUMatrix<half> m(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    unique_ptr<half[]> result(m.CopyToArray());
    BOOST_CHECK_EQUAL_COLLECTIONS(result.get(), result.get() + 6, array.begin(), array.end());
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixIdentityAndZero, RandomSeedFixture)
{
    // TODO: consider splitting into two separate tests?
    const int size = 60;
    GPUMatrix<half> m0(GPUMatrix<half>::Eye(size, c_deviceIdZero));
    unique_ptr<half[]> result0(m0.CopyToArray());

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            BOOST_CHECK_CLOSE(result0[i * size + j], i == j, 0.01);
        }
    }

    GPUMatrix<half> m1(GPUMatrix<half>::Zeros(size, size, c_deviceIdZero));
    unique_ptr<half[]> result1(m1.CopyToArray());
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            BOOST_CHECK_CLOSE(result1[i * size + j], 0.0f, 0.01);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixElementWiseOperations, RandomSeedFixture)
{
    const half val = 3.0;
    const int rows = 16;
    const int cols = 23;

    GPUMatrix<half> m0(rows, cols, c_deviceIdZero);
    m0.SetValue(val);
    GPUMatrix<half> m1(rows, cols, c_deviceIdZero);
    GPUMatrix<half> mr(rows, cols, c_deviceIdZero);

    // test element wise power
    half alpha = 2.0f;
    GPUMatrix<half>::ElementWisePower(alpha, m0, m1);
    mr.SetValue(std::pow(val, alpha));
    BOOST_CHECK(mr.IsEqualTo(m1, c_epsilonFloatE4));

    alpha = 0.234f;
    GPUMatrix<half>::ElementWisePower(alpha, m0, m1);
    mr.SetValue(std::pow(val, alpha));
    BOOST_CHECK(mr.IsEqualTo(m1, c_epsilonFloatE4));

    // test element wise absolute value
    m0.SetValue(-val);
    m1.AssignAbsOf(m0);
    mr.SetValue(val);
    BOOST_CHECK(mr.IsEqualTo(m1));

    // TODO: add other element wise operations?
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixInplaceOperations, RandomSeedFixture)
{
    const half val = 0.42f;
    const int rows = 16;
    const int cols = 23;

    GPUMatrix<half> m(rows, cols, c_deviceIdZero);
    GPUMatrix<half> mr(rows, cols, c_deviceIdZero);

    m.SetValue(val);
    m.InplaceExp();
    mr.SetValue(std::exp(val));
    BOOST_CHECK(mr.IsEqualTo(m, c_epsilonFloatE4));

    m.SetValue(val);
    m.InplaceLog();
    mr.SetValue(std::log(val));
    BOOST_CHECK(mr.IsEqualTo(m, c_epsilonFloatE4));

    m.SetValue(val);
    m.InplaceTanh();
    mr.SetValue(std::tanh(val));
    BOOST_CHECK(mr.IsEqualTo(m, c_epsilonFloatE4));

    m.SetValue(-val);
    m.InplaceAbs();
    mr.SetValue(val);
    BOOST_CHECK(mr.IsEqualTo(m, c_epsilonFloatE4));

    m.SetValue(val);
    m.InplaceSqrt();
    mr.SetValue(std::sqrt(val));
    BOOST_CHECK(mr.IsEqualTo(m, c_epsilonFloatE4));

    m.SetValue(val);
    m.InplaceSigmoid();
    mr.SetValue(1 / (std::exp(-val) + 1));
    BOOST_CHECK(mr.IsEqualTo(m, c_epsilonFloatE4));

    // TODO: there are two more inplace operations. Test these? compare to CPU results?
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAddAndSub, RandomSeedFixture)
{
    std::array<half, 6> array0 = {1, 2, 3, 4, 5, 6};
    GPUMatrix<half> m0(2, 3, c_deviceIdZero, array0.data(), matrixFlagNormal);

    std::array<half, 6> array1 = {11, 12, 13, 14, 15, 16};
    GPUMatrix<half> m1(2, 3, c_deviceIdZero, array1.data(), matrixFlagNormal);

    std::array<half, 6> array2 = {12, 14, 16, 18, 20, 22};
    GPUMatrix<half> m2(2, 3, c_deviceIdZero, array2.data(), matrixFlagNormal);

    std::array<half, 3> arrayCRS = {10, 10, 10};
    GPUMatrix<half> mc(2, 1, c_deviceIdZero, arrayCRS.data(), matrixFlagNormal);
    GPUMatrix<half> mr(1, 3, c_deviceIdZero, arrayCRS.data(), matrixFlagNormal);
    GPUMatrix<half> ms(1, 1, c_deviceIdZero, arrayCRS.data(), matrixFlagNormal);

    GPUMatrix<half> m3 = m2 - m0;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 += m0;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3 = m0 + 10;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 -= 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 = m1 + m0;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3 -= m0;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 = m1 - 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 += 10;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 -= mc;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 += mc;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 -= mr;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 += mr;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3.AssignDifferenceOf(m3, ms);
    BOOST_CHECK(m3.IsEqualTo(m0));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixNorms, RandomSeedFixture)
{
    std::array<half, 6> array = {
        1, 4, 2,
        5, 3, 6};
    GPUMatrix<half> m0(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<half> m3(c_deviceIdZero);
    m0.VectorNorm1(m3, true);
    array[0] = 5;
    array[1] = 7;
    array[2] = 9;
    GPUMatrix<half> m2(1, 3, c_deviceIdZero, array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m0.VectorNorm1(m3, false);
    m2.Resize(2, 1);
    array[0] = 6;
    array[1] = 15;
    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m0.VectorNorm2(m3, true);
    m2.Resize(1, 3);
    array[0] = 4.1231f;
    array[1] = 5.3852f;
    array[2] = 6.7082f;
    m2.SetValue(1, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloat5E4));

    m0.VectorNorm2(m3, false);
    m2.Resize(2, 1);
    array[0] = 3.7417f;
    array[1] = 8.7750f;
    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloat5E4));

    array[0] = 1;
    array[2] = 2;
    array[4] = 3;
    array[1] = 4;
    array[3] = 5;
    array[5] = 6;
    GPUMatrix<half> m00(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<half> m1(c_deviceIdZero);
    m00.VectorMax(m1, m3, true);
    m2.Resize(1, 3);
    array[0] = 4;
    array[1] = 5;
    array[2] = 6;
    m2.SetValue(1, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m00.VectorMax(m1, m3, false);
    m2.Resize(2, 1);
    array[0] = 3.;
    array[1] = 6;
    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m0.VectorNormInf(m3, true);
    m2.Resize(1, 3);
    array[0] = 4;
    array[1] = 5;
    array[2] = 6;
    m2.SetValue(1, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m0.VectorNormInf(m3, false);
    m2.Resize(2, 1);
    array[0] = 3.;
    array[1] = 6;
    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK(m3.IsEqualTo(m2));

    array[0] = 1;
    array[2] = 2;
    array[4] = 3;
    array[1] = 4;
    array[3] = 5;
    array[5] = 6;
    m00.SetValue(2, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
    BOOST_CHECK_EQUAL(6, m00.MatrixNormInf());

    BOOST_CHECK(abs(m0.FrobeniusNorm() - 9.5394) < c_epsilonFloatE3); // HALF_PRECISION
    BOOST_CHECK(abs(m0.MatrixNormInf() - 6) < c_epsilonFloatE4);
    BOOST_CHECK_EQUAL(21, m00.MatrixNorm1());

    GPUMatrix<half> a = GPUMatrix<half>::Eye(4096, c_deviceIdZero);
    BOOST_CHECK_EQUAL(4096, a.MatrixNorm0());

    GPUMatrix<half> b = GPUMatrix<half>::Eye(5, c_deviceIdZero);
    BOOST_CHECK_EQUAL(5, b.MatrixNorm0());
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixRandomUniform, RandomSeedFixture)
{
    const half low = -0.035f;
    const half high = 0.035f;
    auto m = GPUMatrix<half>::RandomUniform(768, 50, c_deviceIdZero, low, high, IncrementCounter());
    unique_ptr<half[]> result(m.CopyToArray());

    for (int i = 0; i < 768 * 50; ++i)
    {
        BOOST_CHECK_LE(result[i], high);
        // NV_TODO: change from GT to GE for now
        BOOST_CHECK_GE(result[i], low); // HALF_PRECISION
    }
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixColumnSlice, RandomSeedFixture)
{
    std::array<half, 6> array = {
        1, 4, 2,
        5, 3, 6};
    GPUMatrix<half> m0(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);
    GPUMatrix<half> m1(2, 2, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<half> m2 = m0.ColumnSlice(0, 2);
    BOOST_CHECK(m2.IsEqualTo(m1));

    std::array<half, 4> array3 = {array[2], array[3], array[4], array[5]};
    GPUMatrix<half> m3(2, 2, c_deviceIdZero, array3.data(), matrixFlagNormal);

    m2 = m0.ColumnSlice(1, 2);
    BOOST_CHECK(m2.IsEqualTo(m3));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixRowSlice, RandomSeedFixture)
{
    std::array<half, 15> array0 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15};
    GPUMatrix<half> m0(5, 3, c_deviceIdZero, array0.data(), matrixFlagNormal);

    std::array<half, 6> array1 = {
        3, 4, 8,
        9, 13, 14};
    GPUMatrix<half> m1(2, 3, c_deviceIdZero, array1.data(), matrixFlagNormal);

    GPUMatrix<half> m2(c_deviceIdZero);
    m2.AssignRowSliceValuesOf(m0, 2, 2);
    BOOST_CHECK(m2.IsEqualTo(m1));

    std::array<half, 15> array3 = {
        0, 0, 3,
        4, 0, 0,
        0, 8, 9,
        0, 0, 0,
        13, 14, 0};
    GPUMatrix<half> m3(5, 3, c_deviceIdZero, array3.data(), matrixFlagNormal);

    m3 += m0;
    m0.AddToRowSliceValuesOf(m1, 2, 2);
    BOOST_CHECK(m3.IsEqualTo(m0));

    m2.AddWithRowSliceValuesOf(m1, 0, 2);
    std::array<half, 6> array4 = {
        6, 8, 16,
        18, 26, 28};
    GPUMatrix<half> m4(2, 3, c_deviceIdZero, array4.data(), matrixFlagNormal);
    BOOST_CHECK(m2.IsEqualTo(m4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixKhatriRaoProduct, RandomSeedFixture)
{
    std::array<half, 12> arrayA = {
        0.8147f, 0.9058f, 0.1270f, 0.9134f,
        0.6324f, 0.0975f, 0.2785f, 0.5469f,
        0.9575f, 0.9649f, 0.1576f, 0.9706f};
    GPUMatrix<half> a(3, 4, c_deviceIdZero, arrayA.data());

    std::array<half, 8> arrayB = {
        0.9572f, 0.4854f, 0.8003f, 0.1419f,
        0.4218f, 0.9157f, 0.7922f, 0.9595f};
    GPUMatrix<half> b(2, 4, c_deviceIdZero, arrayB.data());

    std::array<half, 24> arrayD = {
        0.7798f, 0.8670f, 0.1215f, 0.3954f,
        0.4396f, 0.0616f, 0.7310f, 0.5061f,
        0.0781f, 0.1296f, 0.0897f, 0.0138f,
        0.1175f, 0.2307f, 0.4038f, 0.2550f,
        0.5008f, 0.8768f, 0.7644f, 0.1249f,
        0.7689f, 0.9258f, 0.1512f, 0.9313f};
    GPUMatrix<half> d(6, 4, c_deviceIdZero, arrayD.data());

    GPUMatrix<half> c(c_deviceIdZero);
    c.AssignKhatriRaoProductOf(a, b);

    BOOST_CHECK(c.IsEqualTo(d, c_epsilonFloatE3)); // HALF_PRECISION
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAddColumnReshapeProductOf, RandomSeedFixture)
{
    // tests column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
    std::array<half, 12> arrayA = {
        0.6557f, 0.0357f,
        0.8491f, 0.9340f,
        0.6787f, 0.7577f,
        0.7431f, 0.3922f,
        0.6555f, 0.1712f,
        0.7060f, 0.0318f,
    };
    GPUMatrix<half> a(6, 2, c_deviceIdZero, arrayA.data());

    std::array<half, 6> arrayB = {
        0.2769f, 0.0462f,
        0.0971f, 0.8235f,
        0.6948f, 0.3171f};
    GPUMatrix<half> b(3, 2, c_deviceIdZero, arrayB.data());

    std::array<half, 4> arrayD0 = {
        0.2867f, 0.1266f,
        1.2913f, 0.4520f};
    GPUMatrix<half> d0(2, 2, c_deviceIdZero, arrayD0.data());

    std::array<half, 4> arrayD1 = {
        0.2657f, 0.3636f,
        1.0923f, 0.6416f};
    GPUMatrix<half> d1(2, 2, c_deviceIdZero, arrayD1.data());

    GPUMatrix<half> c(2, 2, c_deviceIdZero);
    c.SetValue(0.0f);
    c.AddColumnReshapeProductOf(a, b, false);
    BOOST_CHECK(c.IsEqualTo(d0, c_epsilonFloatE4));

    c.SetValue(0.0f);
    c.AddColumnReshapeProductOf(a, b, true);
    BOOST_CHECK(c.IsEqualTo(d1, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixInnerProduct, RandomSeedFixture)
{
    std::array<half, 6> array = {
        1, 4, 2,
        5, 3, 6};
    GPUMatrix<half> m0(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<half> m1(c_deviceIdZero), m2(c_deviceIdZero);
    m1.AssignInnerProductOf(m0, m0, true);
    m2.AssignVectorNorm2Of(m0, true);
    m1.InplaceSqrt();
    BOOST_CHECK(m1.IsEqualTo(m2));

    m1.AssignInnerProductOf(m0, m0, false);
    m2.AssignVectorNorm2Of(m0, false);
    m1.InplaceSqrt();
    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAssignRepeatOf, RandomSeedFixture)
{
    std::array<half, 6> array0 = {
        1, 2,
        6, 7,
        11, 12};
    GPUMatrix<half> m0(2, 3, c_deviceIdZero, array0.data(), matrixFlagNormal);

    GPUMatrix<half> m1(c_deviceIdZero);
    m1.AssignRepeatOf(m0, 1, 1);
    BOOST_CHECK(m1.IsEqualTo(m0));

    std::array<half, 36> array2 = {
        1, 2, 1, 2, 1, 2,
        6, 7, 6, 7, 6, 7,
        11, 12, 11, 12, 11, 12,
        1, 2, 1, 2, 1, 2,
        6, 7, 6, 7, 6, 7,
        11, 12, 11, 12, 11, 12};
    GPUMatrix<half> m2(6, 6, c_deviceIdZero, array2.data(), matrixFlagNormal);

    m1.AssignRepeatOf(m0, 3, 2);
    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixRowElementOperations, RandomSeedFixture)
{
    GPUMatrix<half> m0 = GPUMatrix<half>::RandomUniform(20, 28, c_deviceIdZero, -1, 1, IncrementCounter());
    GPUMatrix<half> m1 = GPUMatrix<half>::RandomUniform(1, 28, c_deviceIdZero, 1, 2, IncrementCounter());

    GPUMatrix<half> m2(c_deviceIdZero);
    m2.SetValue(m0);
    m2.RowElementMultiplyWith(m1);
    m2.RowElementDivideBy(m1);

    BOOST_CHECK(m0.IsEqualTo(m2, c_epsilonFloatE3)); // HALF_PRECISION
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixColumnElementOperations, RandomSeedFixture)
{
    GPUMatrix<half> m0 = GPUMatrix<half>::RandomUniform(20, 28, c_deviceIdZero, -1, 1, IncrementCounter());
    GPUMatrix<half> m1 = GPUMatrix<half>::RandomUniform(20, 1, c_deviceIdZero, 1, 2, IncrementCounter());

    GPUMatrix<half> m2(c_deviceIdZero);
    m2.SetValue(m0);
    m2.ColumnElementMultiplyWith(m1);
    m2.ColumnElementDivideBy(m1);

    BOOST_CHECK(m0.IsEqualTo(m2, c_epsilonFloatE3)); // HALF_PRECISION
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixCurandSeedingHalf, RandomSeedFixture)
{
    const half low = 0;
    const half high = 1;
    const unsigned long seedUsed = 1;
    const unsigned long seedIgnored = 4711;

    // The current GPUMatrix implementation uses a static RNG.

    GPUMatrix<half>::ResetCurandObject(seedUsed, __FUNCTION__);
    auto m1 = GPUMatrix<half>::RandomUniform(16, 16, c_deviceIdZero, low, high, seedIgnored);

    GPUMatrix<half>::ResetCurandObject(seedUsed, __FUNCTION__);
    auto m2 = GPUMatrix<half>::RandomUniform(16, 16, c_deviceIdZero, low, high, seedIgnored);

    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAdam, RandomSeedFixture)
{
    GPUMatrix<half> adamMatrix(c_deviceIdZero);
    GPUMatrix<half> gradients(2, 1, c_deviceIdZero);
    GPUMatrix<half> parameters(2, 1, c_deviceIdZero);
    GPUMatrix<half> expectedParameters(2, 1, c_deviceIdZero);
    GPUMatrix<half> expectedStates(2, 2, c_deviceIdZero);
    half gradientValues[] = { 0.1, -0.1 };
    half paramValues[] = { 0.1, 0.1 };
    half expectedValues[] = { -0.05811338, 0.25811338 };
    half expectedStateValues[] = { 1e-5, 0.01, 1e-5, -0.01 };
    gradients.SetValue(2, 1, c_deviceIdZero, gradientValues, matrixFormatRowMajor);
    parameters.SetValue(2, 1, c_deviceIdZero, paramValues, matrixFormatRowMajor);
    expectedParameters.SetValue(2, 1, c_deviceIdZero, expectedValues, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, c_deviceIdZero, expectedStateValues, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, 1e-8, 0.1);

    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-2));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-2));

    half expectedValues2[] = { -0.27059249, 0.47059249 };
    half expectedStateValues2[] = { 2e-05, 0.019, 2e-05, -0.019 };
    expectedParameters.SetValue(2, 1, c_deviceIdZero, expectedValues2, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, c_deviceIdZero, expectedStateValues2, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, 1e-8, 0.1);

    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-2));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAdamVarEpsilon, RandomSeedFixture)
{
    GPUMatrix<half> adamMatrix(c_deviceIdZero);
    GPUMatrix<half> gradients(2, 1, c_deviceIdZero);
    GPUMatrix<half> parameters(2, 1, c_deviceIdZero);
    GPUMatrix<half> expectedParameters(2, 1, c_deviceIdZero);
    GPUMatrix<half> expectedStates(2, 2, c_deviceIdZero);
    half gradientValues[] = { 0.1, -0.1 };
    half paramValues[] = { 0.1, 0.1 };
    half expectedValues[] = { 0.0951532672, 0.1048467328 };
    half expectedStateValues[] = { 1e-5, 0.01, 1e-5, -0.01 };
    half epsilon = 0.1;

    gradients.SetValue(2, 1, c_deviceIdZero, gradientValues, matrixFormatRowMajor);
    parameters.SetValue(2, 1, c_deviceIdZero, paramValues, matrixFormatRowMajor);
    expectedParameters.SetValue(2, 1, c_deviceIdZero, expectedValues, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, c_deviceIdZero, expectedStateValues, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, epsilon, 0.1);

    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-3));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-3));

    half expectedValues2[] = { 0.0860598361, 0.1139401639 };
    half expectedStateValues2[] = { 2e-05, 0.019, 2e-05, -0.019 };
    expectedParameters.SetValue(2, 1, c_deviceIdZero, expectedValues2, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, c_deviceIdZero, expectedStateValues2, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, epsilon, 0.1);

    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-3));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-3));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixOneHot, RandomSeedFixture)
{
    GPUMatrix<half> result(c_deviceIdZero);
    const size_t num_class = 6;

    half data[4] = {1,2,3,4};
    GPUMatrix<half> m0(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, data, matrixFormatRowMajor);

    half exp_data[24];
    memset(&exp_data[0], 0, sizeof(half) * 24);
    exp_data[1] = exp_data[9] = exp_data[14] = exp_data[22] = 1;
    GPUMatrix<half> exp(12, 2, c_deviceIdZero);
    exp.SetValue(12, 2, c_deviceIdZero, exp_data, matrixFormatColMajor);

    vector<size_t> shape(3);
    shape[0] = num_class; shape[1] = 2; shape[2] = 2;

    result.AssignOneHot(m0, shape, 0);

    BOOST_CHECK(result.GetNumCols() == 2);
    BOOST_CHECK(result.GetNumRows() == 12);
    BOOST_CHECK(result.IsEqualTo(exp, 1e-6));

    half exp_data2[24];
    memset(&exp_data2[0], 0, sizeof(half) * 24);
    exp_data2[2] = exp_data2[7] = exp_data2[16] = exp_data2[21] = 1;
    GPUMatrix<half> exp2(12, 2, c_deviceIdZero);
    exp2.SetValue(12, 2, c_deviceIdZero, exp_data2, matrixFormatColMajor);

    vector<size_t> shape2(3);
    shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;
    GPUMatrix<half> result2(c_deviceIdZero);
    result2.AssignOneHot(m0, shape2, 1);

    BOOST_CHECK(result2.GetNumCols() == 2);
    BOOST_CHECK(result2.GetNumRows() == 12);
    BOOST_CHECK(result2.IsEqualTo(exp2, 1e-6));

    half dirty_data[4] = {1,-1,7,4};
    GPUMatrix<half> dirty_m(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, dirty_data, matrixFormatRowMajor);

    half dirty_exp_data[24];
    memset(&dirty_exp_data[0], 0, sizeof(half) * 24);
    dirty_exp_data[1] = dirty_exp_data[22] = 1;
    GPUMatrix<half> dirty_exp(12, 2, c_deviceIdZero);
    dirty_exp.SetValue(12, 2, c_deviceIdZero, dirty_exp_data, matrixFormatColMajor);

    GPUMatrix<half> dirty_result(c_deviceIdZero);
    dirty_result.AssignOneHot(m0, shape, 0);

    BOOST_CHECK(dirty_result.GetNumCols() == 2);
    BOOST_CHECK(dirty_result.GetNumRows() == 12);
    BOOST_CHECK(dirty_result.IsEqualTo(dirty_exp, 1e-6));
}

/*
// Disable, broken because of half atomic
BOOST_FIXTURE_TEST_CASE(GPUMatrixScatterToIndices, RandomSeedFixture)
{
    const size_t row_elements = 2;

    half data[4] = {1,2,2,4};
    GPUMatrix<half> m0(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, data, matrixFormatRowMajor);

    half target[12];
    memset(&target[0], 0, sizeof(half) * 12);
    target[2] = target[3] = 4;
    target[4] = target[5] = 3;
    target[6] = target[7] = 2;
    target[8] = target[9] = 1;
    GPUMatrix<half> m1(row_elements, 6, c_deviceIdZero);
    m1.SetValue(row_elements, 6, c_deviceIdZero, target, matrixFormatColMajor);

    half m3_data[8];
    memset(&m3_data[0], 0, sizeof(half) * 8);
    m3_data[0] = 1;
    m3_data[1] = 2;
    m3_data[2] = 3;
    m3_data[3] = 4;
    m3_data[4] = 5;
    m3_data[5] = 6;
    m3_data[6] = 7;
    m3_data[7] = 8;
    GPUMatrix<half> m3(4, 2, c_deviceIdZero);
    m3.SetValue(4, 2, c_deviceIdZero, m3_data, matrixFormatColMajor);
    m1.ScatterToIndices(m3, m0, row_elements);

    half expect[12];
    memset(&expect[0], 0, sizeof(half) * 12);
    expect[2] = 5;
    expect[3] = 6;
    expect[4] = 11;
    expect[5] = 13;
    expect[6] = 2;
    expect[7] = 2;
    expect[8] = 8;
    expect[9] = 9;
    GPUMatrix<half> m_expect(row_elements, 6, c_deviceIdZero);
    m_expect.SetValue(row_elements, 6, c_deviceIdZero, expect, matrixFormatColMajor);

    BOOST_CHECK(m1.IsEqualTo(m_expect, 1e-6));
}
*/

BOOST_FIXTURE_TEST_CASE(GPUMatrixGatherFromTarget, RandomSeedFixture)
{
    const size_t row_elements = 2;

    half data[4] = {1,2,3,4};
    GPUMatrix<half> m0(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, data, matrixFormatRowMajor);

    half target[12];
    memset(&target[0], 0, sizeof(half) * 12);
    target[2] = target[3] = 4;
    target[4] = target[5] = 3;
    target[6] = target[7] = 2;
    target[8] = target[9] = 1;
    GPUMatrix<half> m1(row_elements, 6, c_deviceIdZero);
    m1.SetValue(row_elements, 6, c_deviceIdZero, target, matrixFormatColMajor);

    half exp_data[8];
    memset(&exp_data[0], 0, sizeof(half) * 8);
    exp_data[0] = exp_data[1] = 4;
    exp_data[2] = exp_data[3] = 2;
    exp_data[4] = exp_data[5] = 3;
    exp_data[6] = exp_data[7] = 1;
    GPUMatrix<half> expect(4, 2, c_deviceIdZero);
    expect.SetValue(4, 2, c_deviceIdZero, exp_data, matrixFormatColMajor);

    GPUMatrix<half> m2(c_deviceIdZero);
    m2.GatherFromTarget(m0, m1, row_elements);
    BOOST_CHECK(m2.GetNumRows() == 4);
    BOOST_CHECK(m2.GetNumCols() == 2);
    BOOST_CHECK(m2.IsEqualTo(expect, 1e-6));
}

#if 0 // Temporarily disabling
BOOST_FIXTURE_TEST_CASE(GPUMatrixLargeInequality, RandomSeedFixture)
{
    const int rows = 33553921;
    const int cols = 1;

    auto m0 = GPUMatrix<half>::Zeros(rows, cols, c_deviceIdZero);
    auto m1 = GPUMatrix<half>::Ones(rows, cols, c_deviceIdZero);

    BOOST_CHECK(!m1.IsEqualTo(m0, c_epsilonFloatE5));
}
#endif

BOOST_AUTO_TEST_SUITE_END()

// Tests from MatrixDataSynchronizationTests.cpp

BOOST_AUTO_TEST_SUITE(Half_GPUMatrixSuite)

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_DefaultBehaviorTestForConstructors, RandomSeedFixture)
{
    const HalfMatrix matrixA1(c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA1.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(0, matrixA1.GetNumCols());
    BOOST_CHECK_EQUAL(0, matrixA1.GetNumRows());

    const HalfMatrix matrixA2(CPUDEVICE);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA2.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(0, matrixA2.GetNumCols());
    BOOST_CHECK_EQUAL(0, matrixA2.GetNumRows());

    const HalfMatrix matrixA3(13, 12, c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA3.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(12, matrixA3.GetNumCols());
    BOOST_CHECK_EQUAL(13, matrixA3.GetNumRows());

    half arr[5 * 45];
    const HalfMatrix matrixA4(5, 45, arr, c_deviceIdZero, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA4.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(45, matrixA4.GetNumCols());
    BOOST_CHECK_EQUAL(5, matrixA4.GetNumRows());

    const HalfMatrix matrixA5(45, 5, arr, CPUDEVICE, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA5.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(5, matrixA5.GetNumCols());
    BOOST_CHECK_EQUAL(45, matrixA5.GetNumRows());
}

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_AccessPatternAndTransferTest, RandomSeedFixture)
{
    half arr[5 * 45];
    const HalfMatrix matrixA(5, 45, arr, c_deviceIdZero, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());

    // GetValue calls operator() const, leaving the matrix in the BOTH state
    half x = matrixA.GetValue(0, 0);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixA.GetCurrentMatrixLocation());
    foreach_coord(i, j, matrixA)
    {
        x = matrixA.GetValue(i, j);
        BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixA.GetCurrentMatrixLocation());
    }

    HalfMatrix matrixB(15, 15, arr, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());

    // non-const operator leaves it in CPU state so that writing to it is valid
    half& y = matrixB(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    matrixB(4, 2) = y;
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    foreach_coord (i, j, matrixB)
    {
        y = matrixB(i, j);
        matrixB(j, i) = y;
        BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    }

    matrixB.TransferFromDeviceToDevice(CPUDEVICE, c_deviceIdZero, false);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(c_deviceIdZero, CPUDEVICE, false);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(CPUDEVICE, c_deviceIdZero, true);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(c_deviceIdZero, CPUDEVICE, true);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
}

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_GravitatingTowardsPreferredDevice, RandomSeedFixture)
{
    HalfMatrix matrixA = HalfMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, IncrementCounter());
    HalfMatrix matrixB = HalfMatrix::Eye(23, c_deviceIdZero);

    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());

    // Set the current matrix location by reading a value of the matrix (via non-const operator())
    half& x = matrixA(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA.GetCurrentMatrixLocation());
    x = matrixB(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());

    const HalfMatrix matrixC = matrixA * matrixB;
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixC.GetCurrentMatrixLocation());
}

BOOST_AUTO_TEST_SUITE_END()

// Tests from MatrixFileWriteReadTests.cpp

BOOST_AUTO_TEST_SUITE(Half_GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(GPUMatrixFileWriteRead, RandomSeedFixture)
{
    GPUMatrix<half> matrixGpu = GPUMatrix<half>::RandomUniform(43, 10, c_deviceIdZero, -26.3f, 30.2f, IncrementCounter());
    GPUMatrix<half> matrixGpuCopy = matrixGpu;

    std::wstring filenameGpu(L"MGPU.txt");
    File fileGpu(filenameGpu, fileOptionsText | fileOptionsReadWrite);

    fileGpu << matrixGpu;
    fileGpu.SetPosition(0);

    GPUMatrix<half> matrixGpuRead(c_deviceIdZero);
    fileGpu >> matrixGpuRead;

    BOOST_CHECK(matrixGpuCopy.IsEqualTo(matrixGpuRead, c_epsilonFloatE5));
}

BOOST_AUTO_TEST_SUITE_END()

}
} } }
