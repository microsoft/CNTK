//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// GPUMatrix unit tests should go here
//
#include "stdafx.h"
#include "../../../Source/Math/GPUMatrix.h"
#include "../../../Source/Math/Matrix.h"
#include "BestGpu.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(MatrixCopyAssignAccrossDevices, RandomSeedFixture)
{
    bool hasTwoGpus = false;
#ifndef CPUONLY
    auto gpus = GetAllGpusData();
    hasTwoGpus = (gpus.size() > 1);
#endif
    std::array<float, 6> array = { 1, 2, 3, 4, 5, 6 };

    {
        Matrix<float> m_gpu(2, 3, array.data(), c_deviceIdZero, matrixFlagNormal);
        Matrix<float> m_copy_gpu_0(m_gpu, c_deviceIdZero);
        if (hasTwoGpus)
            Matrix<float> m_copy_gpu_1(m_gpu, c_deviceIdZero + 1);
        Matrix<float> m_copy_cpu(m_gpu, -1);
    }

    {
        Matrix<float> m_cpu(2, 3, array.data(), -1, matrixFlagNormal);
        Matrix<float> m_copy_gpu_0(m_cpu, c_deviceIdZero);
        if (hasTwoGpus)
            Matrix<float> m_copy_gpu_1(m_cpu, c_deviceIdZero + 1);
        Matrix<float> m_copy_cpu(m_cpu, -1);
    }

    {
        Matrix<float> m_gpu(2, 3, array.data(), c_deviceIdZero, matrixFlagNormal);
        Matrix<float> m_copy_gpu_0(c_deviceIdZero);
        m_copy_gpu_0.AssignValuesOf(m_gpu);
        if (hasTwoGpus)
        {
            Matrix<float> m_copy_gpu_1(c_deviceIdZero + 1);
            m_copy_gpu_1.AssignValuesOf(m_gpu);
        }
        Matrix<float> m_copy_cpu(-1);
        m_copy_cpu.AssignValuesOf(m_gpu);
    }

    if (hasTwoGpus)
    {

        Matrix<float> m_gpu_0(2, 3, array.data(), c_deviceIdZero, matrixFlagNormal);
        Matrix<float> m_gpu_1(2, 3, c_deviceIdZero + 1, m_gpu_0.GetMatrixType(), m_gpu_0.GetFormat());
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
    GPUMatrix<float> m0(c_deviceIdZero);
    BOOST_CHECK(m0.IsEmpty());

    GPUMatrix<float> m1(12, 53, c_deviceIdZero);
    BOOST_CHECK_EQUAL(12, m1.GetNumRows());
    BOOST_CHECK_EQUAL(53, m1.GetNumCols());
    BOOST_CHECK_EQUAL(12 * 53, m1.GetNumElements());

    std::array<float, 2> array = {1, 14};
    m1.SetValue(1, 2, c_deviceIdZero, array.data());

    unique_ptr<float[]> result(m1.CopyToArray());
    BOOST_CHECK_EQUAL_COLLECTIONS(result.get(), result.get() + 2, array.begin(), array.end());

    GPUMatrix<float> m1Copy(m1);
    BOOST_CHECK(m1.IsEqualTo(m1Copy));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixConstructorFlagNormal, RandomSeedFixture)
{
    std::array<float, 6> array = {1, 2, 3, 4, 5, 6};
    GPUMatrix<float> m(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    unique_ptr<float[]> result(m.CopyToArray());
    BOOST_CHECK_EQUAL_COLLECTIONS(result.get(), result.get() + 6, array.begin(), array.end());
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixIdentityAndZero, RandomSeedFixture)
{
    // TODO: consider splitting into two separate tests?
    const int size = 60;
    GPUMatrix<float> m0(GPUMatrix<float>::Eye(size, c_deviceIdZero));
    unique_ptr<float[]> result0(m0.CopyToArray());

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            BOOST_CHECK_CLOSE(result0[i * size + j], i == j, 0.01);
        }
    }

    GPUMatrix<float> m1(GPUMatrix<float>::Zeros(size, size, c_deviceIdZero));
    unique_ptr<float[]> result1(m1.CopyToArray());
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
    const float val = 3.0;
    const int rows = 16;
    const int cols = 23;

    GPUMatrix<float> m0(rows, cols, c_deviceIdZero);
    m0.SetValue(val);
    GPUMatrix<float> m1(rows, cols, c_deviceIdZero);
    GPUMatrix<float> mr(rows, cols, c_deviceIdZero);

    // test element wise power
    float alpha = 2.0f;
    GPUMatrix<float>::ElementWisePower(alpha, m0, m1);
    mr.SetValue(std::pow(val, alpha));
    BOOST_CHECK(mr.IsEqualTo(m1, c_epsilonFloatE4));

    alpha = 0.234f;
    GPUMatrix<float>::ElementWisePower(alpha, m0, m1);
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
    const float val = 0.42f;
    const int rows = 16;
    const int cols = 23;

    GPUMatrix<float> m(rows, cols, c_deviceIdZero);
    GPUMatrix<float> mr(rows, cols, c_deviceIdZero);

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
    std::array<float, 6> array0 = {1, 2, 3, 4, 5, 6};
    GPUMatrix<float> m0(2, 3, c_deviceIdZero, array0.data(), matrixFlagNormal);

    std::array<float, 6> array1 = {11, 12, 13, 14, 15, 16};
    GPUMatrix<float> m1(2, 3, c_deviceIdZero, array1.data(), matrixFlagNormal);

    std::array<float, 6> array2 = {12, 14, 16, 18, 20, 22};
    GPUMatrix<float> m2(2, 3, c_deviceIdZero, array2.data(), matrixFlagNormal);

    std::array<float, 3> arrayCRS = {10, 10, 10};
    GPUMatrix<float> mc(2, 1, c_deviceIdZero, arrayCRS.data(), matrixFlagNormal);
    GPUMatrix<float> mr(1, 3, c_deviceIdZero, arrayCRS.data(), matrixFlagNormal);
    GPUMatrix<float> ms(1, 1, c_deviceIdZero, arrayCRS.data(), matrixFlagNormal);

    GPUMatrix<float> m3 = m2 - m0;
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
    std::array<float, 6> array = {
        1, 4, 2,
        5, 3, 6};
    GPUMatrix<float> m0(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<float> m3(c_deviceIdZero);
    m0.VectorNorm1(m3, true);
    array[0] = 5;
    array[1] = 7;
    array[2] = 9;
    GPUMatrix<float> m2(1, 3, c_deviceIdZero, array.data(), matrixFlagNormal);
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
    GPUMatrix<float> m00(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<float> m1(c_deviceIdZero);
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

    BOOST_CHECK(abs(m0.FrobeniusNorm() - 9.5394) < c_epsilonFloatE4);
    BOOST_CHECK(abs(m0.MatrixNormInf() - 6) < c_epsilonFloatE4);
    BOOST_CHECK_EQUAL(21, m00.MatrixNorm1());

    GPUMatrix<float> a = GPUMatrix<float>::Eye(4096, c_deviceIdZero);
    BOOST_CHECK_EQUAL(4096, a.MatrixNorm0());

    GPUMatrix<float> b = GPUMatrix<float>::Eye(5, c_deviceIdZero);
    BOOST_CHECK_EQUAL(5, b.MatrixNorm0());
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixRandomUniform, RandomSeedFixture)
{
    const float low = -0.035f;
    const float high = 0.035f;
    auto m = GPUMatrix<float>::RandomUniform(768, 50, c_deviceIdZero, low, high, IncrementCounter());
    unique_ptr<float[]> result(m.CopyToArray());

    for (int i = 0; i < 768 * 50; ++i)
    {
        BOOST_CHECK_LE(result[i], high);
        BOOST_CHECK_GT(result[i], low);
    }
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixColumnSlice, RandomSeedFixture)
{
    std::array<float, 6> array = {
        1, 4, 2,
        5, 3, 6};
    GPUMatrix<float> m0(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);
    GPUMatrix<float> m1(2, 2, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<float> m2 = m0.ColumnSlice(0, 2);
    BOOST_CHECK(m2.IsEqualTo(m1));

    std::array<float, 4> array3 = {array[2], array[3], array[4], array[5]};
    GPUMatrix<float> m3(2, 2, c_deviceIdZero, array3.data(), matrixFlagNormal);

    m2 = m0.ColumnSlice(1, 2);
    BOOST_CHECK(m2.IsEqualTo(m3));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixRowSlice, RandomSeedFixture)
{
    std::array<float, 15> array0 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15};
    GPUMatrix<float> m0(5, 3, c_deviceIdZero, array0.data(), matrixFlagNormal);

    std::array<float, 6> array1 = {
        3, 4, 8,
        9, 13, 14};
    GPUMatrix<float> m1(2, 3, c_deviceIdZero, array1.data(), matrixFlagNormal);

    GPUMatrix<float> m2(c_deviceIdZero);
    m2.AssignRowSliceValuesOf(m0, 2, 2);
    BOOST_CHECK(m2.IsEqualTo(m1));

    std::array<float, 15> array3 = {
        0, 0, 3,
        4, 0, 0,
        0, 8, 9,
        0, 0, 0,
        13, 14, 0};
    GPUMatrix<float> m3(5, 3, c_deviceIdZero, array3.data(), matrixFlagNormal);

    m3 += m0;
    m0.AddToRowSliceValuesOf(m1, 2, 2);
    BOOST_CHECK(m3.IsEqualTo(m0));

    m2.AddWithRowSliceValuesOf(m1, 0, 2);
    std::array<float, 6> array4 = {
        6, 8, 16,
        18, 26, 28};
    GPUMatrix<float> m4(2, 3, c_deviceIdZero, array4.data(), matrixFlagNormal);
    BOOST_CHECK(m2.IsEqualTo(m4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixKhatriRaoProduct, RandomSeedFixture)
{
    std::array<float, 12> arrayA = {
        0.8147f, 0.9058f, 0.1270f, 0.9134f,
        0.6324f, 0.0975f, 0.2785f, 0.5469f,
        0.9575f, 0.9649f, 0.1576f, 0.9706f};
    GPUMatrix<float> a(3, 4, c_deviceIdZero, arrayA.data());

    std::array<float, 8> arrayB = {
        0.9572f, 0.4854f, 0.8003f, 0.1419f,
        0.4218f, 0.9157f, 0.7922f, 0.9595f};
    GPUMatrix<float> b(2, 4, c_deviceIdZero, arrayB.data());

    std::array<float, 24> arrayD = {
        0.7798f, 0.8670f, 0.1215f, 0.3954f,
        0.4396f, 0.0616f, 0.7310f, 0.5061f,
        0.0781f, 0.1296f, 0.0897f, 0.0138f,
        0.1175f, 0.2307f, 0.4038f, 0.2550f,
        0.5008f, 0.8768f, 0.7644f, 0.1249f,
        0.7689f, 0.9258f, 0.1512f, 0.9313f};
    GPUMatrix<float> d(6, 4, c_deviceIdZero, arrayD.data());

    GPUMatrix<float> c(c_deviceIdZero);
    c.AssignKhatriRaoProductOf(a, b);
    BOOST_CHECK(c.IsEqualTo(d, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAddColumnReshapeProductOf, RandomSeedFixture)
{
    // tests column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
    std::array<float, 12> arrayA = {
        0.6557f, 0.0357f,
        0.8491f, 0.9340f,
        0.6787f, 0.7577f,
        0.7431f, 0.3922f,
        0.6555f, 0.1712f,
        0.7060f, 0.0318f,
    };
    GPUMatrix<float> a(6, 2, c_deviceIdZero, arrayA.data());

    std::array<float, 6> arrayB = {
        0.2769f, 0.0462f,
        0.0971f, 0.8235f,
        0.6948f, 0.3171f};
    GPUMatrix<float> b(3, 2, c_deviceIdZero, arrayB.data());

    std::array<float, 4> arrayD0 = {
        0.2867f, 0.1266f,
        1.2913f, 0.4520f};
    GPUMatrix<float> d0(2, 2, c_deviceIdZero, arrayD0.data());

    std::array<float, 4> arrayD1 = {
        0.2657f, 0.3636f,
        1.0923f, 0.6416f};
    GPUMatrix<float> d1(2, 2, c_deviceIdZero, arrayD1.data());

    GPUMatrix<float> c(2, 2, c_deviceIdZero);
    c.SetValue(0.0f);
    c.AddColumnReshapeProductOf(a, b, false);
    BOOST_CHECK(c.IsEqualTo(d0, c_epsilonFloatE4));

    c.SetValue(0.0f);
    c.AddColumnReshapeProductOf(a, b, true);
    BOOST_CHECK(c.IsEqualTo(d1, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixInnerProduct, RandomSeedFixture)
{
    std::array<float, 6> array = {
        1, 4, 2,
        5, 3, 6};
    GPUMatrix<float> m0(2, 3, c_deviceIdZero, array.data(), matrixFlagNormal);

    GPUMatrix<float> m1(c_deviceIdZero), m2(c_deviceIdZero);
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
    std::array<float, 6> array0 = {
        1, 2,
        6, 7,
        11, 12};
    GPUMatrix<float> m0(2, 3, c_deviceIdZero, array0.data(), matrixFlagNormal);

    GPUMatrix<float> m1(c_deviceIdZero);
    m1.AssignRepeatOf(m0, 1, 1);
    BOOST_CHECK(m1.IsEqualTo(m0));

    std::array<float, 36> array2 = {
        1, 2, 1, 2, 1, 2,
        6, 7, 6, 7, 6, 7,
        11, 12, 11, 12, 11, 12,
        1, 2, 1, 2, 1, 2,
        6, 7, 6, 7, 6, 7,
        11, 12, 11, 12, 11, 12};
    GPUMatrix<float> m2(6, 6, c_deviceIdZero, array2.data(), matrixFlagNormal);

    m1.AssignRepeatOf(m0, 3, 2);
    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixRowElementOperations, RandomSeedFixture)
{
    GPUMatrix<float> m0 = GPUMatrix<float>::RandomUniform(20, 28, c_deviceIdZero, -1, 1, IncrementCounter());
    GPUMatrix<float> m1 = GPUMatrix<float>::RandomUniform(1, 28, c_deviceIdZero, 1, 2, IncrementCounter());

    GPUMatrix<float> m2(c_deviceIdZero);
    m2.SetValue(m0);
    m2.RowElementMultiplyWith(m1);
    m2.RowElementDivideBy(m1);

    BOOST_CHECK(m0.IsEqualTo(m2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixColumnElementOperations, RandomSeedFixture)
{
    GPUMatrix<float> m0 = GPUMatrix<float>::RandomUniform(20, 28, c_deviceIdZero, -1, 1, IncrementCounter());
    GPUMatrix<float> m1 = GPUMatrix<float>::RandomUniform(20, 1, c_deviceIdZero, 1, 2, IncrementCounter());

    GPUMatrix<float> m2(c_deviceIdZero);
    m2.SetValue(m0);
    m2.ColumnElementMultiplyWith(m1);
    m2.ColumnElementDivideBy(m1);

    BOOST_CHECK(m0.IsEqualTo(m2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixCurandSeedingFloat, RandomSeedFixture)
{
    const float low = 0;
    const float high = 1;
    const unsigned long seedUsed = 1;
    const unsigned long seedIgnored = 4711;

    // The current GPUMatrix implementation uses a static RNG.

    GPUMatrix<float>::ResetCurandObject(seedUsed, __FUNCTION__);
    auto m1 = GPUMatrix<float>::RandomUniform(16, 16, c_deviceIdZero, low, high, seedIgnored);

    GPUMatrix<float>::ResetCurandObject(seedUsed, __FUNCTION__);
    auto m2 = GPUMatrix<float>::RandomUniform(16, 16, c_deviceIdZero, low, high, seedIgnored);

    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixCurandSeedingDouble, RandomSeedFixture)
{
    const double low = 0;
    const double high = 1;
    const unsigned long seedUsed = 1;
    const unsigned long seedIgnored = 4711;

    // The current GPUMatrix implementation uses a static RNG.

    GPUMatrix<double>::ResetCurandObject(seedUsed, __FUNCTION__);
    auto m1 = GPUMatrix<double>::RandomUniform(16, 16, c_deviceIdZero, low, high, seedIgnored);

    GPUMatrix<double>::ResetCurandObject(seedUsed, __FUNCTION__);
    auto m2 = GPUMatrix<double>::RandomUniform(16, 16, c_deviceIdZero, low, high, seedIgnored);

    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixAdam, RandomSeedFixture)
{
    GPUMatrix<double> adamMatrix(c_deviceIdZero);
    GPUMatrix<double> gradients(2, 1, c_deviceIdZero);
    GPUMatrix<double> parameters(2, 1, c_deviceIdZero);
    GPUMatrix<double> expectedParameters(2, 1, c_deviceIdZero);
    GPUMatrix<double> expectedStates(2, 2, c_deviceIdZero);
    double gradientValues[] = { 0.1, -0.1 };
    double paramValues[] = { 0.1, 0.1 };
    double expectedValues[] = { -0.05803489, 0.25803488 };
    double expectedStateValues[] = { 1e-5, 0.01, 1e-5, -0.01 };
    gradients.SetValue(2, 1, c_deviceIdZero, gradientValues, matrixFormatRowMajor);
    parameters.SetValue(2, 1, c_deviceIdZero, paramValues, matrixFormatRowMajor);
    expectedParameters.SetValue(2, 1, c_deviceIdZero, expectedValues, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, c_deviceIdZero, expectedStateValues, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, true);
    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-6));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-6));

    double expectedValues2[] = { -0.27046135, 0.47046134 };
    double expectedStateValues2[] = { 2e-05, 0.019, 2e-05, -0.019 };
    expectedParameters.SetValue(2, 1, c_deviceIdZero, expectedValues2, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, c_deviceIdZero, expectedStateValues2, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, true);
    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-6));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-6));
}

BOOST_FIXTURE_TEST_CASE(GPUMatrixOneHot, RandomSeedFixture)
{
    GPUMatrix<double> result(c_deviceIdZero);
    const size_t num_class = 6;

    double data[4] = { 1,2,3,4 };
    GPUMatrix<double> m0(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, data, matrixFormatRowMajor);

    double exp_data[24];
    memset(&exp_data[0], 0, sizeof(double) * 24);
    exp_data[1] = exp_data[9] = exp_data[14] = exp_data[22] = 1;
    GPUMatrix<double> exp(12, 2, c_deviceIdZero);
    exp.SetValue(12, 2, c_deviceIdZero, exp_data, matrixFormatColMajor);
    
    vector<size_t> shape(3);
    shape[0] = num_class; shape[1] = 2; shape[2] = 2;

    result.AssignOneHot(m0, shape, 0);
    
    BOOST_CHECK(result.GetNumCols() == 2);
    BOOST_CHECK(result.GetNumRows() == 12);
    BOOST_CHECK(result.IsEqualTo(exp, 1e-6));

    double exp_data2[24];
    memset(&exp_data2[0], 0, sizeof(double) * 24);
    exp_data2[2] = exp_data2[7] = exp_data2[16] = exp_data2[21] = 1;
    GPUMatrix<double> exp2(12, 2, c_deviceIdZero);
    exp2.SetValue(12, 2, c_deviceIdZero, exp_data2, matrixFormatColMajor);

    vector<size_t> shape2(3);
    shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;
    GPUMatrix<double> result2(c_deviceIdZero);
    result2.AssignOneHot(m0, shape2, 1);

    BOOST_CHECK(result2.GetNumCols() == 2);
    BOOST_CHECK(result2.GetNumRows() == 12);
    BOOST_CHECK(result2.IsEqualTo(exp2, 1e-6));

    double dirty_data[4] = {1,-1,7,4};
    GPUMatrix<double> dirty_m(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, dirty_data, matrixFormatRowMajor);

    double dirty_exp_data[24];
    memset(&dirty_exp_data[0], 0, sizeof(double) * 24);
    dirty_exp_data[1] = dirty_exp_data[22] = 1;
    GPUMatrix<double> dirty_exp(12, 2, c_deviceIdZero);
    dirty_exp.SetValue(12, 2, c_deviceIdZero, dirty_exp_data, matrixFormatColMajor);

    GPUMatrix<double> dirty_result(c_deviceIdZero);
    dirty_result.AssignOneHot(m0, shape, 0);

    BOOST_CHECK(dirty_result.GetNumCols() == 2);
    BOOST_CHECK(dirty_result.GetNumRows() == 12);
    BOOST_CHECK(dirty_result.IsEqualTo(dirty_exp, 1e-6));
}

#if 0 // Temporarily disabling
BOOST_FIXTURE_TEST_CASE(GPUMatrixLargeInequality, RandomSeedFixture)
{
    const int rows = 33553921;
    const int cols = 1;

    auto m0 = GPUMatrix<float>::Zeros(rows, cols, c_deviceIdZero);
    auto m1 = GPUMatrix<float>::Ones(rows, cols, c_deviceIdZero);

    BOOST_CHECK(!m1.IsEqualTo(m0, c_epsilonFloatE5));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
}
} } }
