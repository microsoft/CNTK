//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/CPUMatrix.h"
#include "../../../Source/Math/Helpers.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing

#define SIGNUM(v) ((v) > 0.0f ? 1.0f : -1.0f)
#define SIGNUMZ(v) ((v) == 0.0f ? 0.0f : (SIGNUM(v)))

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

 // TODO: do this for all math tests!
 // BOOST_GLOBAL_FIXTURE(DeterministicCPUAlgorithmsFixture);

BOOST_AUTO_TEST_SUITE(MatrixUnitTests)

BOOST_FIXTURE_TEST_CASE(MatrixConstructors, RandomSeedFixture)
{
    SingleMatrix a0(0);
    SingleMatrix a1(0);
    SingleMatrix a2(CPUDEVICE);
    SingleMatrix a3(13, 12, 0);

    BOOST_CHECK_EQUAL(0, a0.GetNumRows());
    BOOST_CHECK_EQUAL(0, a0.GetNumCols());
    BOOST_CHECK_EQUAL(0, a1.GetNumRows());
    BOOST_CHECK_EQUAL(0, a1.GetNumCols());
    BOOST_CHECK_EQUAL(13, a3.GetNumRows());
    BOOST_CHECK_EQUAL(12, a3.GetNumCols());

    BOOST_CHECK_EQUAL(a0.GetDeviceId(), c_deviceIdZero);
    BOOST_CHECK_EQUAL(a1.GetDeviceId(), c_deviceIdZero);
    BOOST_CHECK_EQUAL(a2.GetDeviceId(), CPUDEVICE);
    BOOST_CHECK_EQUAL(a3.GetDeviceId(), c_deviceIdZero);
}

BOOST_FIXTURE_TEST_CASE(MatrixMoveTest1, RandomSeedFixture)
{
    // no moves required
    SingleMatrix a(c_deviceIdZero);
    SingleMatrix b(c_deviceIdZero);
    b.Resize(50, 100);

    BOOST_CHECK_EQUAL(b.GetNumRows(), 50);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 100);
    BOOST_CHECK_EQUAL(a.GetDeviceId(), c_deviceIdZero);
    BOOST_CHECK_EQUAL(b.GetDeviceId(), c_deviceIdZero);

    std::swap(a, b);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 50);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 100);
    BOOST_CHECK_EQUAL(a.GetDeviceId(), c_deviceIdZero);
    BOOST_CHECK_EQUAL(b.GetDeviceId(), c_deviceIdZero);
}

BOOST_FIXTURE_TEST_CASE(MatrixMoveTest2, RandomSeedFixture)
{
    // potentially a move is required
    SingleMatrix a(c_deviceIdZero);
    SingleMatrix b(c_deviceIdZero);
    b.Resize(50, 100);
    BOOST_CHECK_EQUAL(b.GetNumRows(), 50);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 100);
    BOOST_CHECK_EQUAL(a.GetDeviceId(), 0);
    BOOST_CHECK_EQUAL(b.GetDeviceId(), 0);

    b(12, 13) = 14; // this will move whole matrix B from GPU to CPU
    BOOST_CHECK_EQUAL(b.GetDeviceId(), -1);

    std::swap(a, b); // this will not only swap A and B but will put them to their preferred device (GPU if present)
    BOOST_CHECK_EQUAL(a.GetNumRows(), 50);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 100);
    BOOST_CHECK_EQUAL(b.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 0);
    BOOST_CHECK_EQUAL(a.GetDeviceId(), -1);
    BOOST_CHECK_EQUAL(b.GetDeviceId(), 0);
}

BOOST_FIXTURE_TEST_CASE(MatrixDeepCopy, RandomSeedFixture)
{
    // This is deep copy, not move
    SingleMatrix a(c_deviceIdZero);
    SingleMatrix b(c_deviceIdZero);

    b.Resize(50, 100);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 0);
    BOOST_CHECK_EQUAL(b.GetNumRows(), 50);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 100);

    b.SetValue(a);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 0);
    BOOST_CHECK_EQUAL(b.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 0);

    b.Resize(50, 100);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 0);
    BOOST_CHECK_EQUAL(b.GetNumRows(), 50);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 100);

    b(2, 3) = 9;
    BOOST_CHECK_EQUAL(b(2, 3), 9);

    b.SetValue(a);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 0);
    BOOST_CHECK_EQUAL(b.GetNumRows(), 0);
    BOOST_CHECK_EQUAL(b.GetNumCols(), 0);
}

BOOST_FIXTURE_TEST_CASE(MatrixInitZero, RandomSeedFixture)
{
    SingleMatrix a = SingleMatrix::Zeros(12, 32, c_deviceIdZero);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 12);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 32);
    foreach_coord (i, j, a)
    {
        BOOST_CHECK_EQUAL(a(i, j), 0.0);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixInitEye, RandomSeedFixture)
{
    SingleMatrix a = SingleMatrix::Eye(56, c_deviceIdZero);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 56);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 56);

    foreach_coord (i, j, a)
    {
        if (i != j)
        {
            BOOST_CHECK_EQUAL(a(i, j), 0.0);
        }
        else
        {
            BOOST_CHECK_EQUAL(a(i, j), 1.0);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixInitOnes, RandomSeedFixture)
{
    SingleMatrix a = SingleMatrix::Ones(12, 56, c_deviceIdZero);
    BOOST_CHECK_EQUAL(a.GetNumRows(), 12);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 56);
    foreach_coord (i, j, a)
    {
        BOOST_CHECK_EQUAL(a(i, j), 1.0);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixInitGaussianRand, RandomSeedFixture)
{
    SingleMatrix a = SingleMatrix::RandomGaussian(640, 230, c_deviceIdZero, 0.0f, 2.0f, IncrementCounter());
    BOOST_CHECK_EQUAL(a.GetNumRows(), 640);
    BOOST_CHECK_EQUAL(a.GetNumCols(), 230);

    float avg = 0;
    foreach_coord (i, j, a)
    {
        avg += a(i, j);
    }
    avg /= (640 * 230);

    float std = 0;
    foreach_coord (i, j, a)
    {
        std += ((a(i, j) - avg) * (a(i, j) - avg));
    }
    std = sqrt(std / (640 * 230));

    BOOST_CHECK_LE(fabs(avg), c_epsilonFloatE1);
    BOOST_CHECK_LE(fabs(std - 2), c_epsilonFloatE1);
}

BOOST_FIXTURE_TEST_CASE(MatrixInitRandomUniform, RandomSeedFixture)
{
    const float low = -26.3f;
    const float high = 30.2f;
    SingleMatrix a = SingleMatrix::RandomUniform(435, 100, c_deviceIdZero, low, high, IncrementCounter());
    bool has_small = false;
    bool has_big = false;
    foreach_coord (i, j, a)
    {
        BOOST_CHECK_GE(a(i, j), low);
        BOOST_CHECK_LE(a(i, j), high);
        if (a(i, j) < -3)
        {
            has_small = true;
        }
        if (a(i, j) > 3)
        {
            has_big = true;
        }
    }
    BOOST_CHECK(has_small);
    BOOST_CHECK(has_big);
}

BOOST_FIXTURE_TEST_CASE(MatrixInitRandomUniformSeed, RandomSeedFixture)
{
    const float low = -0.01f;
    const float high = 0.01f;
    SingleMatrix a = SingleMatrix::RandomUniform(429, 1024, c_deviceIdZero, low, high, IncrementCounter());
    foreach_coord (i, j, a)
    {
        BOOST_CHECK_GE(a(i, j), low);
        BOOST_CHECK_LE(a(i, j), high);
    }

    // SingleMatrix b = SingleMatrix::RandomUniform(429, 1024, (float)-0.01, (float) 0.01, IncrementCounter());
    // BOOST_CHECK(a.IsEqualTo(b));
}

BOOST_FIXTURE_TEST_CASE(MatrixSetValueMethods, RandomSeedFixture)
{
    // void SetValue(const ElemType v);
    SingleMatrix a(32, 12, c_deviceIdZero);
    BOOST_CHECK_EQUAL(32, a.GetNumRows());
    BOOST_CHECK_EQUAL(12, a.GetNumCols());
    BOOST_CHECK_EQUAL(12 * 32, a.GetNumElements());
    const float v = -32.3451f;
    a.SetValue(v);
    foreach_coord (i, j, a)
    {
        BOOST_CHECK_EQUAL(v, a(i, j));
    }

    // void SetValue(const Matrix<ElemType>& deepCopyFrom);
    SingleMatrix b(c_deviceIdZero);
    b.SetValue(a);
    foreach_coord (i, j, b)
    {
        BOOST_CHECK_EQUAL(v, b(i, j));
    }

    // void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, const bool srcIsColMajor);
    std::array<float, 7> arrVector = {123.0f, 0.23f, -22.0f, 63.0f, 43.42f, 324.3f, 99912.0f};

    float *arr = arrVector.data();
    b.SetValue(2, 3, b.GetDeviceId(), arr, matrixFlagNormal);

    SingleMatrix b1(c_deviceIdZero);
    b1.SetValue(2, 3, b.GetDeviceId(), arr);
    foreach_coord (i, j, b1)
    {
        BOOST_CHECK_EQUAL(arr[IDX2C(i, j, 2)], b(i, j));
        BOOST_CHECK_EQUAL(arr[IDX2C(i, j, 2)], b1(i, j));
    }

    SingleMatrix bbbb = SingleMatrix::Zeros(6, 8, c_deviceIdZero);
    bbbb.SetColumn(arr, 3);
    for (int i = 0; i < 6; ++i)
    {
        BOOST_CHECK_EQUAL(arr[i], bbbb(i, 3));
    }

    // void SetDiagonalValue(const ElemType v);
    SingleMatrix c(4, 4, c_deviceIdZero);
    const float val = -0.00332f;
    c.SetDiagonalValue(val);
    foreach_coord (i, j, c)
    {
        if (i == j)
            BOOST_CHECK_EQUAL(val, c(i, j));
        else
            BOOST_CHECK_EQUAL(0, c(i, j));
    }

    // void SetDiagonalValue(const Matrix<ElemType>& vector);
    SingleMatrix d(4, 1, c_deviceIdZero);
    const float val1 = 43.324f;
    d.SetValue(val1);
    c.SetDiagonalValue(d);
    foreach_coord (i, j, c)
    {
        if (i == j)
            BOOST_CHECK_EQUAL(val1, c(i, j));
        else
            BOOST_CHECK_EQUAL(0, c(i, j));
    }

    SingleMatrix c1(5, 5, c_deviceIdZero);
    SingleMatrix d1(1, 5, c_deviceIdZero);
    float val2 = 0.53f;
    d1 = d1.Transpose();
    d1.SetValue(val2);
    c1.SetDiagonalValue(d1);
    foreach_coord (i, j, c1)
    {
        if (i == j)
            BOOST_CHECK_EQUAL(val2, c1(i, j));
        else
            BOOST_CHECK_EQUAL(0, c1(i, j));
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixTransposeTest, RandomSeedFixture)
{
    SingleMatrix a = SingleMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, IncrementCounter());
    BOOST_CHECK_EQUAL(64, a.GetNumRows());
    BOOST_CHECK_EQUAL(23, a.GetNumCols());

    SingleMatrix b = a.Transpose();

    BOOST_CHECK_EQUAL(23, b.GetNumRows());
    BOOST_CHECK_EQUAL(64, b.GetNumCols());

    foreach_coord (i, j, a)
    {
        BOOST_CHECK_EQUAL(a(i, j), b(j, i));
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixMultiAndDiv, RandomSeedFixture)
{
    SingleMatrix m0(2, 3, c_deviceIdZero);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    SingleMatrix m00(2, 3, c_deviceIdZero);
    m00(0, 0) = 10;
    m00(0, 1) = 20;
    m00(0, 2) = 30;
    m00(1, 0) = 40;
    m00(1, 1) = 50;
    m00(1, 2) = 60;

    SingleMatrix m1(2, 3, c_deviceIdZero);
    m1.Reshape(3, 2);
    m1(0, 0) = 11;
    m1(0, 1) = 15;
    m1(1, 0) = 14;
    m1(1, 1) = 13;
    m1(2, 0) = 12;
    m1(2, 1) = 16;

    SingleMatrix m2(2, 2, c_deviceIdZero);
    m2(0, 0) = 75;
    m2(0, 1) = 89;
    m2(1, 0) = 186;
    m2(1, 1) = 221;

    SingleMatrix m3 = m0 * m1;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3 = m0 * 10;
    BOOST_CHECK(m3.IsEqualTo(m00));

    m3 = m3 / 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 *= 10;
    BOOST_CHECK(m3.IsEqualTo(m00));

    m3 /= 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    SingleMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, false, 0, m3);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m1.Reshape(2, 3);
    SingleMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, true, 0, m3);
    m2(0, 0) = 74;
    m2(0, 1) = 92;
    m2(1, 0) = 182;
    m2(1, 1) = 227;
    BOOST_CHECK(m3.IsEqualTo(m2));

    SingleMatrix::MultiplyAndWeightedAdd(10, m0, false, m1, true, 2, m3);
    m2(0, 0) = 888;
    m2(0, 1) = 1104;
    m2(1, 0) = 2184;
    m2(1, 1) = 2724;
    BOOST_CHECK(m3.IsEqualTo(m2));

    SingleMatrix::MultiplyAndWeightedAdd(1, m0, true, m1, false, 0, m3);
    m2.Resize(3, 3);
    m2(0, 0) = 67;
    m2(0, 1) = 72;
    m2(0, 2) = 77;
    m2(1, 0) = 92;
    m2(1, 1) = 99;
    m2(1, 2) = 106;
    m2(2, 0) = 117;
    m2(2, 1) = 126;
    m2(2, 2) = 135;
    BOOST_CHECK(m3.IsEqualTo(m2));

    // Multiplications of arbitrary matrix with 1x1 matrix

    SingleMatrix a(2, 3, c_deviceIdZero);
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;
    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    SingleMatrix b = SingleMatrix::Eye(1, c_deviceIdZero);

    SingleMatrix c = a * b;
    BOOST_CHECK(c.IsEqualTo(a));
    c = b * a;
    BOOST_CHECK(c.IsEqualTo(a));
    b(0, 0) = 0.5;
    b.InplaceAbs();
    c = a * b;

    SingleMatrix d(2, 3, c_deviceIdZero);
    d(0, 0) = 0.5;
    d(0, 1) = 1;
    d(0, 2) = 1.5;
    d(1, 0) = 2;
    d(1, 1) = 2.5;
    d(1, 2) = 3;
    BOOST_CHECK(c.IsEqualTo(d));
}

BOOST_FIXTURE_TEST_CASE(MatrixTranspose, RandomSeedFixture)
{
    SingleMatrix m0(2, 3, c_deviceIdZero);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    SingleMatrix m1(3, 2, c_deviceIdZero);
    m1(0, 0) = 1;
    m1(0, 1) = 4;
    m1(1, 0) = 2;
    m1(1, 1) = 5;
    m1(2, 0) = 3;
    m1(2, 1) = 6;

    SingleMatrix m2 = m0.Transpose();
    BOOST_CHECK(m2.IsEqualTo(m1, c_epsilonFloatE4));

    m2.AssignTransposeOf(m1);
    BOOST_CHECK(m2.IsEqualTo(m0, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(MatrixAddAndSub, RandomSeedFixture)
{
    SingleMatrix m0(2, 3, c_deviceIdZero);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    SingleMatrix m1(2, 3, c_deviceIdZero);
    m1(0, 0) = 11;
    m1(0, 1) = 12;
    m1(0, 2) = 13;
    m1(1, 0) = 14;
    m1(1, 1) = 15;
    m1(1, 2) = 16;

    SingleMatrix m2(2, 3, c_deviceIdZero);
    m2(0, 0) = 12;
    m2(0, 1) = 14;
    m2(0, 2) = 16;
    m2(1, 0) = 18;
    m2(1, 1) = 20;
    m2(1, 2) = 22;

    SingleMatrix m3 = m2 - m0;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 += m0;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3 = m0 + 10;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 -= 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 = m1 + m0;
    BOOST_CHECK(m3.IsEqualTo(m2));
    SingleMatrix m4 = SingleMatrix::Eye(3, c_deviceIdZero);

    m3 -= m0;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 = m1 - 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    SingleMatrix m33(m3.DeepClone());
    m3 += 10;
    BOOST_CHECK(m3.IsEqualTo(m1));

    SingleMatrix m55 = SingleMatrix::Eye(1, c_deviceIdZero);
    m55(0, 0) = 10;
    m55.InplaceAbs();
    m33 += m55;
    BOOST_CHECK(m33.IsEqualTo(m1));
    m33 -= 10;
    m33 = m33 + 10;
    BOOST_CHECK(m33.IsEqualTo(m1));
}

BOOST_FIXTURE_TEST_CASE(MatrixElementOps, RandomSeedFixture)
{
    SingleMatrix m0(2, 3, c_deviceIdZero);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    SingleMatrix m00(2, 3, c_deviceIdZero);
    m00(0, 0) = 1.0f;
    m00(0, 1) = static_cast<float>(1 / 2.0);
    m00(0, 2) = static_cast<float>(1 / 3.0);
    m00(1, 0) = static_cast<float>(1 / 4.0);
    m00(1, 1) = static_cast<float>(1 / 5.0);
    m00(1, 2) = static_cast<float>(1 / 6.0);

    SingleMatrix m1(2, 3, c_deviceIdZero);
    m1(0, 0) = 1;
    m1(0, 1) = 1;
    m1(0, 2) = 1;
    m1(1, 0) = 1;
    m1(1, 1) = 1;
    m1(1, 2) = 1;

    SingleMatrix m3(c_deviceIdZero);
    m3.AssignElementProductOf(m0, m00);
    BOOST_CHECK(m3.IsEqualTo(m1, c_epsilonFloatE4));

    SingleMatrix m4 = SingleMatrix::Zeros(2, 3, c_deviceIdZero);
    m4.SetValue(m4.AddElementProductOf(m0, m00));
    BOOST_CHECK(m4.IsEqualTo(m1, c_epsilonFloatE4));

    m3 = m0 ^ 4;
    SingleMatrix m2(2, 3, c_deviceIdZero);
    m2(0, 0) = 1;
    m2(0, 1) = 16;
    m2(0, 2) = 81;
    m2(1, 0) = 256;
    m2(1, 1) = 625;
    m2(1, 2) = 1296;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE3));

    m3.SetValue(m0);
    m3 ^= 4;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE3));

    m3.SetValue(m0);
    m3.ElementMultiplyWith(m00);
    BOOST_CHECK(m3.IsEqualTo(m1, c_epsilonFloatE3));

    m3.SetValue(m0);
    m3.ElementInverse();
    BOOST_CHECK(m3.IsEqualTo(m00, c_epsilonFloatE3));

    m2(0, 0) = 0.7311f;
    m2(0, 1) = 0.8808f;
    m2(0, 2) = 0.9526f;
    m2(1, 0) = 0.9820f;
    m2(1, 1) = 0.9933f;
    m2(1, 2) = 0.9975f;
    m3.AssignElementDivisionOf(m2, m0);
    m2.ElementMultiplyWith(m00);
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceSigmoid();
    m2(0, 0) = 0.7311f;
    m2(0, 1) = 0.8808f;
    m2(0, 2) = 0.9526f;
    m2(1, 0) = 0.9820f;
    m2(1, 1) = 0.9933f;
    m2(1, 2) = 0.9975f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceTanh();
    m2(0, 0) = 0.7616f;
    m2(0, 1) = 0.9640f;
    m2(0, 2) = 0.9951f;
    m2(1, 0) = 0.9993f;
    m2(1, 1) = 0.9999f;
    m2(1, 2) = 1.0000f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceLogSoftmax(true);
    m3.InplaceExp();
    m2(0, 0) = 0.0474f;
    m2(0, 1) = 0.0474f;
    m2(0, 2) = 0.0474f;
    m2(1, 0) = 0.9526f;
    m2(1, 1) = 0.9526f;
    m2(1, 2) = 0.9526f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceLogSoftmax(false);
    m3.InplaceExp();
    m2(0, 0) = 0.0900f;
    m2(0, 1) = 0.2447f;
    m2(0, 2) = 0.6652f;
    m2(1, 0) = 0.0900f;
    m2(1, 1) = 0.2447f;
    m2(1, 2) = 0.6652f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceHardmax(true);
    m2(0, 0) = 0.0f;
    m2(0, 1) = 0.0f;
    m2(0, 2) = 0.0f;
    m2(1, 0) = 1.0f;
    m2(1, 1) = 1.0f;
    m2(1, 2) = 1.0f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceSqrt();
    m2(0, 0) = 1.0f;
    m2(0, 1) = 1.4142f;
    m2(0, 2) = 1.7321f;
    m2(1, 0) = 2.0f;
    m2(1, 1) = 2.2361f;
    m2(1, 2) = 2.4495f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceExp();
    m2(0, 0) = 2.7183f;
    m2(0, 1) = 7.3891f;
    m2(0, 2) = 20.0855f;
    m2(1, 0) = 54.5982f;
    m2(1, 1) = 148.4132f;
    m2(1, 2) = 403.4288f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceExp();
    m2(0, 0) = 2.7183f;
    m2(0, 1) = 7.3891f;
    m2(0, 2) = 20.0855f;
    m2(1, 0) = 54.5982f;
    m2(1, 1) = 148.4132f;
    m2(1, 2) = 403.4288f;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.InplaceLog();
    BOOST_CHECK(m3.IsEqualTo(m0, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceTruncateBottom(2);
    m2(0, 0) = 2;
    m2(0, 1) = 2;
    m2(0, 2) = 3;
    m2(1, 0) = 4;
    m2(1, 1) = 5;
    m2(1, 2) = 6;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE3));

    m3.SetValue(m0);
    m3.InplaceTruncateTop(4);
    m2(0, 0) = 1;
    m2(0, 1) = 2;
    m2(0, 2) = 3;
    m2(1, 0) = 4;
    m2(1, 1) = 4;
    m2(1, 2) = 4;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE3));
}

BOOST_FIXTURE_TEST_CASE(MatrixColumnElementMultiply, RandomSeedFixture)
{
    CPUMatrix<float> mcpu = CPUMatrix<float>::RandomUniform(429, 1024, -3.4f, 1, IncrementCounter());
    CPUMatrix<float> acpu = CPUMatrix<float>::Ones(429, 1);
    CPUMatrix<float> mcpuCopy(mcpu);

    mcpu.ColumnElementMultiplyWith(acpu);
    BOOST_CHECK(mcpuCopy.IsEqualTo(mcpu, c_epsilonFloatE4));

    Matrix<float> m = Matrix<float>::RandomUniform(429, 1024, c_deviceIdZero, -3.4f, 1, IncrementCounter());
    Matrix<float> a = Matrix<float>::Ones(429, 1, c_deviceIdZero);
    Matrix<float> mCopy(m.DeepClone());

    m.ColumnElementMultiplyWith(a);
    BOOST_CHECK(mCopy.IsEqualTo(m, c_epsilonFloatE4));

    CPUMatrix<float> mc1 = CPUMatrix<float>::RandomUniform(429, 1024, -3.4f, 1, IncrementCounter());
    CPUMatrix<float> mc2 = CPUMatrix<float>::RandomUniform(429, 1, 0, 3, IncrementCounter());
    mc1.ColumnElementMultiplyWith(mc2);

    Matrix<float> m1(mc1.GetNumRows(), mc1.GetNumCols(), mc1.Buffer(), matrixFlagNormal);
    Matrix<float> m2(mc2.GetNumRows(), mc2.GetNumCols(), mc2.Buffer(), matrixFlagNormal);
    m1.ColumnElementMultiplyWith(m2);

    foreach_coord (i, j, m2)
    {
        BOOST_CHECK_LT(fabs(m2(i, j) - mc2(i, j)), c_epsilonFloatE5);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixAssignXOf, RandomSeedFixture)
{
    // AssignDifferenceOf
    Matrix<float> a = Matrix<float>::RandomUniform(429, 1024, c_deviceIdZero, 5, 32, IncrementCounter());
    Matrix<float> b = Matrix<float>::RandomUniform(429, 1024, c_deviceIdZero, 5, 32, IncrementCounter());
    Matrix<float> c(c_deviceIdZero);

    c.AssignDifferenceOf(a, b);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), a(i, j) - b(i, j));
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    float x = 234.2f;
    c.AssignDifferenceOf(a, x);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), a(i, j) - x);
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    c.AssignDifferenceOf(x, a);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), x - a(i, j));
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    c.AssignDifferenceOf(1, a);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), 1 - a(i, j));
    }
    // 
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    
    // AssignSumOf
    c.AssignSumOf(a, b);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), a(i, j) + b(i, j));
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    

    // Check for self-assignment (c = c + b)
    auto tolerance = 5e-5;
    c.AssignSumOf(c, b);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_CLOSE(c(i, j), a(i, j) + 2 * b(i, j), tolerance);
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    
    // Check for self-assignment (c = b + c) 
    c.AssignSumOf(b, c);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_CLOSE(c(i, j), a(i, j) + 3 * b(i, j), tolerance);
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    // Check for self-assignment (c = c + a .* c)
    c.AssignSumOf(a, b);
    c.AddElementProductOf(a, c);
    foreach_coord(i, j, c)
    {
        BOOST_CHECK_CLOSE(c(i, j), (1 + a(i, j)) * (a(i, j) + b(i, j)), tolerance);
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    // Check for self-assignment (c = c + c .* a) 
    c.AssignSumOf(a, b);
    c.AddElementProductOf(c, a);
    foreach_coord(i, j, c)
    {
        BOOST_CHECK_CLOSE(c(i, j), (1 + a(i, j)) * (a(i, j) + b(i, j)), tolerance);
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    // Check for self-assignment (c = c + c .* c)
    c.AssignSumOf(a, b);
    c.AddElementProductOf(c, c);
    foreach_coord(i, j, c)
    {
        BOOST_CHECK_CLOSE(c(i, j), (1 + a(i, j) + b(i, j)) * (a(i, j) + b(i, j)), tolerance);
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

    // AssignElementProductOf
    c.AssignElementProductOf(a, b);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), a(i, j) * b(i, j));
    }
    a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
    c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);


    // AddElementProductOf
    Matrix<float> c_copy(c.DeepClone());
    c.AddElementProductOf(a, b);
    foreach_coord (i, j, c)
    {
        BOOST_CHECK_EQUAL(c(i, j), c_copy(i, j) + a(i, j) * b(i, j));
    }

    // AssignSigmoidOf
    CPUMatrix<float> ac = CPUMatrix<float>::RandomUniform(429, 1024, 5, 32, IncrementCounter());
    CPUMatrix<float> bc = CPUMatrix<float>::RandomUniform(429, 1024, -5, 12, IncrementCounter());
    Matrix<float> d(ac.GetNumRows(), ac.GetNumCols(), ac.Buffer(), matrixFlagNormal);
    Matrix<float> e(bc.GetNumRows(), bc.GetNumCols(), bc.Buffer(), matrixFlagNormal);
    ac.AssignSigmoidOf(bc);
    d.AssignSigmoidOf(e);
    foreach_coord (i, j, ac)
    {
        BOOST_CHECK_LT(fabs(ac(i, j) - d(i, j)), c_epsilonFloatE5);
    }

    // AssignSignOf
    Matrix<float> m1 = Matrix<float>::RandomUniform(42, 12, c_deviceIdZero, -5, 12, IncrementCounter());
    Matrix<float> m2(4, 5, c_deviceIdZero);
    m2.AssignSignOf(m1);
    foreach_coord (i, j, m1)
    {
        float v = m1(i, j);
        float expected = SIGNUMZ(v);
        float actual = m2(i, j);
        BOOST_CHECK_EQUAL(expected, actual);
    }

    Matrix<float> m3 = Matrix<float>::RandomUniform(42, 12, c_deviceIdZero, -5, 2, IncrementCounter());
    Matrix<float> m4(m3.DeepClone());
    m3.AddSignOf(m1);
    foreach_coord (i, j, m3)
    {
        float v = m1(i, j);
        BOOST_CHECK_EQUAL(m4(i, j) + SIGNUMZ(v), m3(i, j));
    }

    // AssignTruncateBottom and Top
    Matrix<float> m5(2, 2, c_deviceIdZero);
    m5(0, 0) = 1;
    m5(0, 1) = 2;
    m5(1, 0) = 3;
    m5(1, 1) = 4;

    Matrix<float> m6(c_deviceIdZero);
    m6.AssignTruncateBottomOf(m5, 3);
    BOOST_CHECK_EQUAL(3, m6(0, 0));
    BOOST_CHECK_EQUAL(3, m6(0, 1));
    BOOST_CHECK_EQUAL(3, m6(1, 0));
    BOOST_CHECK_EQUAL(4, m6(1, 1));

    Matrix<float> m7(c_deviceIdZero);
    m7.AssignTruncateTopOf(m5, 3);
    BOOST_CHECK_EQUAL(1, m7(0, 0));
    BOOST_CHECK_EQUAL(2, m7(0, 1));
    BOOST_CHECK_EQUAL(3, m7(1, 0));
    BOOST_CHECK_EQUAL(3, m7(1, 1));
}

BOOST_FIXTURE_TEST_CASE(MatrixSumOfElements, RandomSeedFixture)
{
    Matrix<float> m = Matrix<float>::Ones(429, 1024, 0);
    float sum = m.SumOfElements();
    BOOST_CHECK_EQUAL(429 * 1024, sum);

    CPUMatrix<float> mcpu = CPUMatrix<float>::Ones(429, 1024);
    float sumCPU = mcpu.SumOfElements();
    BOOST_CHECK_EQUAL(429 * 1024, sumCPU);

    Matrix<float> m1 = Matrix<float>::Ones(42, 332, c_deviceIdZero);
    m1 *= -1;
    float sum1 = m1.SumOfElements();
    BOOST_CHECK_EQUAL(-1 * 42 * 332, sum1);

    Matrix<float> m2 = Matrix<float>::Ones(3, 2, c_deviceIdZero);
    m2 *= -1;
    float sum2 = m2.SumOfElements();
    BOOST_CHECK_EQUAL(-1 * 3 * 2, sum2);
}

BOOST_FIXTURE_TEST_CASE(MatrixColumnSlice, RandomSeedFixture)
{
    std::array<float, 6> arr = {1, 2, 3, 4, 5, 6};
    auto *fArray = arr.data();

    Matrix<float> m0(2, 3, fArray, matrixFlagNormal);

    Matrix<float> m1(2, 2, fArray, matrixFlagNormal);

    Matrix<float> m2 = m0.ColumnSlice(0, 2);
    BOOST_CHECK(m2.IsEqualTo(m1, c_epsilonFloatE4));

    Matrix<float> m3(2, 2, fArray + 2, matrixFlagNormal);

    m2 = m0.ColumnSlice(1, 2);
    BOOST_CHECK(m2.IsEqualTo(m3, c_epsilonFloatE4));

    size_t k = 100, n = 20, m = 50;

    Matrix<float> ag(k, n, c_deviceIdZero);
    ag.SetUniformRandomValue(-1, 1, IncrementCounter());

    Matrix<float> bg(n, m, c_deviceIdZero);
    bg.SetUniformRandomValue(-1, 1, IncrementCounter());

    Matrix<float> cg(k, m, c_deviceIdZero);
    cg.SetUniformRandomValue(-1, 1, IncrementCounter());

    Matrix<float> dg(k, m, c_deviceIdZero);
    dg.AssignValuesOf(cg);

    Matrix<float>::MultiplyAndAdd(ag, false, bg, false, dg);

    for (int i = 0; i < m; i++)
    {
        Matrix<float> colBg = bg.ColumnSlice(i, 1);
        Matrix<float> colCg = cg.ColumnSlice(i, 1);
        Matrix<float>::MultiplyAndAdd(ag, false, colBg, false, colCg);
    }
    BOOST_CHECK(cg.IsEqualTo(dg, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(MatrixKhatriRaoProduct, RandomSeedFixture)
{
    std::array<float, 24> arr =
        {0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0};

    auto *fArray = arr.data();
    fArray[0] = 0.8147f;
    fArray[3] = 0.9134f;
    fArray[6] = 0.2785f;
    fArray[9] = 0.9649f;
    fArray[1] = 0.9058f;
    fArray[4] = 0.6324f;
    fArray[7] = 0.5469f;
    fArray[10] = 0.1576f;
    fArray[2] = 0.1270f;
    fArray[5] = 0.0975f;
    fArray[8] = 0.9575f;
    fArray[11] = 0.9706f;
    Matrix<float> a(3, 4, fArray, c_deviceIdZero);

    fArray[0] = 0.9572f;
    fArray[2] = 0.8003f;
    fArray[4] = 0.4218f;
    fArray[6] = 0.7922f;
    fArray[1] = 0.4854f;
    fArray[3] = 0.1419f;
    fArray[5] = 0.9157f;
    fArray[7] = 0.9595f;
    Matrix<float> b(2, 4, fArray, c_deviceIdZero);

    fArray[0] = 0.7798f;
    fArray[6] = 0.7310f;
    fArray[12] = 0.1175f;
    fArray[18] = 0.7644f;
    fArray[1] = 0.8670f;
    fArray[7] = 0.5061f;
    fArray[13] = 0.2307f;
    fArray[19] = 0.1249f;
    fArray[2] = 0.1215f;
    fArray[8] = 0.0781f;
    fArray[14] = 0.4038f;
    fArray[20] = 0.7689f;
    fArray[3] = 0.3954f;
    fArray[9] = 0.1296f;
    fArray[15] = 0.2550f;
    fArray[21] = 0.9258f;
    fArray[4] = 0.4396f;
    fArray[10] = 0.0897f;
    fArray[16] = 0.5008f;
    fArray[22] = 0.1512f;
    fArray[5] = 0.0616f;
    fArray[11] = 0.0138f;
    fArray[17] = 0.8768f;
    fArray[23] = 0.9313f;
    Matrix<float> d(6, 4, fArray, c_deviceIdZero);

    Matrix<float> c(c_deviceIdZero);
    c.AssignKhatriRaoProductOf(a, b);
    BOOST_CHECK(c.IsEqualTo(d, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(MatrixAddColumnReshapeProductOf, RandomSeedFixture)
{
    std::array<float, 12> arr =
        {0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0};

    auto *fArray = arr.data();
    fArray[0] = 0.6557f;
    fArray[6] = 0.7431f;
    fArray[1] = 0.0357f;
    fArray[7] = 0.3922f;
    fArray[2] = 0.8491f;
    fArray[8] = 0.6555f;
    fArray[3] = 0.9340f;
    fArray[9] = 0.1712f;
    fArray[4] = 0.6787f;
    fArray[10] = 0.7060f;
    fArray[5] = 0.7577f;
    fArray[11] = 0.0318f;
    Matrix<float> a(6, 2, fArray, c_deviceIdZero);

    fArray[0] = 0.2769f;
    fArray[3] = 0.8235f;
    fArray[1] = 0.0462f;
    fArray[4] = 0.6948f;
    fArray[2] = 0.0971f;
    fArray[5] = 0.3171f;
    Matrix<float> b(3, 2, fArray, c_deviceIdZero);

    fArray[0] = 0.2867f;
    fArray[2] = 1.2913f;
    fArray[1] = 0.1266f;
    fArray[3] = 0.4520f;
    Matrix<float> d0(2, 2, fArray, c_deviceIdZero);

    fArray[0] = 0.2657f;
    fArray[2] = 1.0923f;
    fArray[1] = 0.3636f;
    fArray[3] = 0.6416f;
    Matrix<float> d1(2, 2, fArray, c_deviceIdZero);

    Matrix<float> c(2, 2, c_deviceIdZero);
    c.SetValue(0.0f);
    c.AddColumnReshapeProductOf(a, b, false);
    BOOST_CHECK(c.IsEqualTo(d0, c_epsilonFloatE4));

    c.SetValue(0.0f);
    c.AddColumnReshapeProductOf(a, b, true);
    BOOST_CHECK(c.IsEqualTo(d1, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(MatrixCopy, RandomSeedFixture)
{
    const size_t crow = 3;
    const size_t ccol = 2;
    // Matrices are stored as column-major so below is 3x2 matrix.
    float src[] = {
        1.0f, 3.0f, 4.0f,
        6.0f, 2.0f, 5.0f};

    Matrix<float> srcM(crow, ccol, src, matrixFlagNormal, c_deviceIdZero);
    // Test full copy.
    CPUMatrix<float> actualM(crow, ccol);
    srcM.CopySection(actualM.GetNumRows(), actualM.GetNumCols(), actualM.Data(), actualM.GetNumRows());

    std::vector<float> expected = {
        1.0f, 3.0f, 4.0f,
        6.0f, 2.0f, 5.0f};
    BOOST_CHECK(actualM.IsEqualTo(CPUMatrix<float>(actualM.GetNumRows(), actualM.GetNumCols(), expected.data(), matrixFlagNormal)));

    // Test tile copy.
    actualM.Resize(crow - 1, ccol - 1);
    actualM.SetValue(std::numeric_limits<float>::quiet_NaN());
    srcM.CopySection(actualM.GetNumRows(), actualM.GetNumCols(), actualM.Data(), actualM.GetNumRows());

    expected = {1.0f, 3.0f};
    BOOST_CHECK(actualM.IsEqualTo(CPUMatrix<float>(actualM.GetNumRows(), actualM.GetNumCols(), expected.data(), matrixFlagNormal)));
}

BOOST_FIXTURE_TEST_CASE(MatrixHasElement, RandomSeedFixture)
{
    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        const size_t size = 3;
        float src[size] = {0.0f, 1.0f, 2.0f};
        SingleMatrix m1(1, size, src, deviceId, matrixFlagNormal);
        BOOST_CHECK(SingleMatrix::HasElement(m1, 1.0f));
        BOOST_CHECK(!SingleMatrix::HasElement(m1, -1.0f));

        auto qnan = std::numeric_limits<float>::quiet_NaN();
        BOOST_CHECK(!SingleMatrix::HasElement(m1, qnan));
        auto posInf = std::numeric_limits<float>::infinity();
        BOOST_CHECK(!SingleMatrix::HasElement(m1, posInf));

        m1(0, 1) = qnan;
        BOOST_CHECK(SingleMatrix::HasElement(m1, qnan));

        m1(0, 1) = posInf;
        BOOST_CHECK(SingleMatrix::HasElement(m1, posInf));
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixVectorMax, RandomSeedFixture)
{
    // Matrices are stored as column-major so below is 3x2 matrix.
    float src[] = {
        1.0f, 3.0f, 4.0f,
        6.0f, 2.0f, 5.0f};

    float expectedIdx[] = {
        2.0f, 1.0f,
        0.0f, 2.0f};

    float expectedVal[] = {
        4.0f, 3.0f,
        6.0f, 5.0f};

    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        Matrix<float> expIdx(2, 2, expectedIdx, deviceId, matrixFlagNormal);
        Matrix<float> expVal(2, 2, expectedVal, deviceId, matrixFlagNormal);

        Matrix<float> actual(3, 2, src, deviceId, matrixFlagNormal);
        Matrix<float> actualIdx(deviceId);
        Matrix<float> actualVal(deviceId);

        auto topK = 2;
        actual.VectorMax(actualIdx, actualVal, true, topK);
        BOOST_CHECK(actualIdx.IsEqualTo(expIdx));
        BOOST_CHECK(actualVal.IsEqualTo(expVal));
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixAssignNumOfDiff, RandomSeedFixture)
{
    float labels[] = {1.0f, 2.0f, 3.0f};

    // Matrices are stored as column-major so below is 2x3 matrix.
    float topKResults[] = {
        1.0f, 3.0f,
        4.0f, 6.0f,
        2.0f, 3.0f};

    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        Matrix<float> lbl(1, 3, labels, deviceId, matrixFlagNormal);
        Matrix<float> topKRes(2, 3, topKResults, deviceId, matrixFlagNormal);

        Matrix<float> actual(deviceId);
        actual.AssignNumOfDiff(lbl, topKRes, true);

        float expectedDiff = 1.0;
        BOOST_CHECK_EQUAL(expectedDiff, actual.Get00Element());
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixScale, RandomSeedFixture)
{
    const float low = -1.0f;
    const float high = 1.0f;
    float alpha = 0.7713f;
    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        auto a1 = SingleMatrix::RandomUniform(7, 11, deviceId, low, high, IncrementCounter());
        auto a2 = a1.DeepClone();
        BOOST_ASSERT(a1.IsEqualTo(a2));

        auto b1 = SingleMatrix::RandomUniform(7, 11, deviceId, low, high, IncrementCounter());
        auto b2 = b1.DeepClone();
        BOOST_ASSERT(b1.IsEqualTo(b2));

        Matrix<float>::ScaleAndAdd(alpha, b1, a1);

        Matrix<float>::Scale(alpha, b2);
        a2 += b2;

        // BUGBUG: this test currently fails on GPU.
        if (deviceId != CPUDEVICE)
            continue;
        
        // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
        // BOOST_CHECK(a1.IsEqualTo(a2));
        BOOST_CHECK(a1.IsEqualTo(a2, c_epsilonFloatE5));
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixSGDUpdate, RandomSeedFixture)
{
    const float low = -1.0f;
    const float high = 1.0f;
    float lr = 0.77f;
    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto p2 = p1.DeepClone();
        BOOST_ASSERT(p1.IsEqualTo(p2));

        auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto g2 = g1.DeepClone();
        BOOST_ASSERT(g1.IsEqualTo(g2));
        
        auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto sg2 = sg1.DeepClone();
        BOOST_ASSERT(sg1.IsEqualTo(sg2));

        for (; lr > 0.01; lr = lr / 2)
        {
            if (deviceId != CPUDEVICE)
            {
                // g1 is modified inside the GPU version of SGDUpdate, restore the original value here.
                g1.SetValue(g2);
            }

            p1.SGDUpdate(g1, lr);
            p2.MomentumSGDUpdate(g2, sg2, lr, 0.0);

            // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
            BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));

            if (deviceId != CPUDEVICE)
                continue;
            
            // GPU version of SGDUpdate scales gradient by the learning rate, this check will fail.
            // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
            BOOST_CHECK(g1.IsEqualTo(g2, c_epsilonFloatE5));
        }

        lr = std::pow(lr, lr);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixMomentumSGDUpdate_WithAndWithout_UnitGain, RandomSeedFixture)
{
    const float low = -1.0f;
    const float high = 1.0f;
    float lr = 0.77f;
    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto p2 = p1.DeepClone();
        BOOST_ASSERT(p1.IsEqualTo(p2));

        auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto g2 = g1.DeepClone();
        BOOST_ASSERT(g1.IsEqualTo(g2));
        
        auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto sg2 = sg1.DeepClone();
        BOOST_ASSERT(sg1.IsEqualTo(sg2));

        for (; lr > 0.01; lr = lr / 2)
        {
            p1.MomentumSGDUpdate(g1, sg1, lr, 0.0, true);
            p2.MomentumSGDUpdate(g2, sg2, lr, 0.0, false);
            // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
            BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));
        }

        for (lr = 1.0; lr > 0.03; lr = lr / 2)
        {
            p1.MomentumSGDUpdate(g1, sg1, lr, 0.5, true);
            p2.MomentumSGDUpdate(g2, sg2, lr/2, 0.5, false);
            // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
            BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));
        }

        // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
        BOOST_CHECK(g1.IsEqualTo(g2, c_epsilonFloatE5));
        BOOST_CHECK(sg1.IsEqualTo(sg2, c_epsilonFloatE5));

        p1.MomentumSGDUpdate(g1, sg1, lr, 0.5, true);
        p2.MomentumSGDUpdate(g2, sg2, lr, 0.5, false);
        // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
        BOOST_CHECK(!p1.IsEqualTo(p2, c_epsilonFloatE5));

        lr = std::pow(lr, lr);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixNesterovAcceleratedMomentumSGDUpdate_WithAndWithout_UnitGain, RandomSeedFixture)
{
    const float low = -1.0f;
    const float high = 1.0f;
    float lr = 0.77f;
    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto p2 = p1.DeepClone();
        BOOST_ASSERT(p1.IsEqualTo(p2));

        auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto g2 = g1.DeepClone();
        BOOST_ASSERT(g1.IsEqualTo(g2));
        
        auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto sg2 = sg1.DeepClone();
        BOOST_ASSERT(sg1.IsEqualTo(sg2));

        for (; lr > 0.01; lr = lr / 2)
        {
            p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.0, true);
            p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr, 0.0, false);
            // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
            BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));
        }

        for (lr = 1.0; lr > 0.03; lr = lr / 2)
        {
            p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.5, true);
            p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr/2, 0.5, false);
            // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
            BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));
        }

        // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
        BOOST_CHECK(g1.IsEqualTo(g2));
        BOOST_CHECK(sg1.IsEqualTo(sg2));

        p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.5, true);
        p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr, 0.5, false);

        // TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
        BOOST_CHECK(!p1.IsEqualTo(p2, c_epsilonFloatE5));

        lr = std::pow(lr, lr);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixFSAdagradUpdate_WithAndWithout_UnitGain, RandomSeedFixture)
{
    const float low = -1.0f;
    const float high = 1.0f;
    float lr = 0.77f;
    for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
    {
        auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto p2 = p1.DeepClone();
        BOOST_ASSERT(p1.IsEqualTo(p2));

        auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto g2 = g1.DeepClone();
        BOOST_ASSERT(g1.IsEqualTo(g2));
        
        auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, IncrementCounter());
        auto sg2 = sg1.DeepClone();
        BOOST_ASSERT(sg1.IsEqualTo(sg2));

        for (; lr > 0.01; lr = lr / 2)
        {
            size_t mbSize = 100;
            double smoothedCount = 10 / lr;
            double targetAdagradAvDenom = 1.0;
            double varMomentum = 1.0 - lr;

            sg1.FSAdagradUpdate(mbSize, g1, p1, smoothedCount, lr, targetAdagradAvDenom, 0.0, varMomentum, true);
            sg2.FSAdagradUpdate(mbSize, g2, p2, smoothedCount, lr, targetAdagradAvDenom, 0.0, varMomentum, true /*false*/);
            // BUGBUG: at the moment this fails even with identical arguments.
            // BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));
        }

        sg2.SetValue(sg1);
        BOOST_ASSERT(sg1.IsEqualTo(sg2));

        for (lr = 1.0; lr > 0.03; lr = lr / 2)
        {
            size_t mbSize = 100;
            double smoothedCount = 10 / lr;
            double targetAdagradAvDenom = 1.0;
            double varMomentum = 1.0 - lr;

            sg1.FSAdagradUpdate(mbSize, g1, p1, smoothedCount, lr, targetAdagradAvDenom, 0.5, varMomentum, true);
            sg2.FSAdagradUpdate(mbSize, g2, p2, smoothedCount, lr /*lr/2*/, targetAdagradAvDenom, 0.5, varMomentum, true /*false*/);
            // BUGBUG: at the moment this fails even with identical arguments.
            // BOOST_CHECK(p1.IsEqualTo(p2, c_epsilonFloatE5));
        }

        lr = std::pow(lr, lr);
    }
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
