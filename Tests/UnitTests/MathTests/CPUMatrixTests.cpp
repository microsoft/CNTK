//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/CPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

typedef CPUDoubleMatrix DMatrix;
typedef CPUSingleMatrix SMatrix;

// TODO: consider using PI from some library, e.g. boost
const double pi = 3.14159265358979323846264338327950288419716939937510;

BOOST_AUTO_TEST_SUITE(CPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(CPUMatrixConstructorNoFlags, RandomSeedFixture)
{
    DMatrix m;
    BOOST_CHECK(m.IsEmpty());

    m.Resize(2, 3);
    BOOST_CHECK(!m.IsEmpty());
    BOOST_CHECK_EQUAL(m.GetNumRows(), 2);
    BOOST_CHECK_EQUAL(m.GetNumCols(), 3);
    BOOST_CHECK_EQUAL(m.GetNumElements(), 6);

    m(0, 0) = 1;
    m(1, 2) = 2;
    BOOST_CHECK_EQUAL(m(0, 0), 1);
    BOOST_CHECK_EQUAL(m(1, 2), 2);

    DMatrix m1(m);
    BOOST_CHECK(m1.IsEqualTo(m));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixConstructorFlagNormal, RandomSeedFixture)
{
    std::array<float, 6> array = {1, 2, 3, 4, 5, 6};
    SMatrix m(2, 3, array.data(), matrixFlagNormal);
    BOOST_CHECK_EQUAL(m(0, 0), 1);
    BOOST_CHECK_EQUAL(m(0, 1), 3);
    BOOST_CHECK_EQUAL(m(0, 2), 5);
    BOOST_CHECK_EQUAL(m(1, 0), 2);
    BOOST_CHECK_EQUAL(m(1, 1), 4);
    BOOST_CHECK_EQUAL(m(1, 2), 6);
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixConstructorFormatRowMajor, RandomSeedFixture)
{
    std::array<double, 6> array = {7, 8, 9, 10, 11, 12};
    DMatrix m(2, 3, array.data(), matrixFormatRowMajor);
    BOOST_CHECK_EQUAL(m(0, 0), 7);
    BOOST_CHECK_EQUAL(m(0, 1), 8);
    BOOST_CHECK_EQUAL(m(0, 2), 9);
    BOOST_CHECK_EQUAL(m(1, 0), 10);
    BOOST_CHECK_EQUAL(m(1, 1), 11);
    BOOST_CHECK_EQUAL(m(1, 2), 12);
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixAddAndSub, RandomSeedFixture)
{
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    DMatrix m1(2, 3);
    m1(0, 0) = 11;
    m1(0, 1) = 12;
    m1(0, 2) = 13;
    m1(1, 0) = 14;
    m1(1, 1) = 15;
    m1(1, 2) = 16;

    DMatrix m2(2, 3);
    m2(0, 0) = 12;
    m2(0, 1) = 14;
    m2(0, 2) = 16;
    m2(1, 0) = 18;
    m2(1, 1) = 20;
    m2(1, 2) = 22;

    DMatrix mC(2, 1);
    mC(0, 0) = 10;
    mC(1, 0) = 10;

    DMatrix mR(1, 3);
    mR(0, 0) = 10;
    mR(0, 1) = 10;
    mR(0, 2) = 10;

    DMatrix mS(1, 1);
    mS(0, 0) = 10;

    DMatrix m3 = m2 - m0;
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

    m3 -= mC;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 += mC;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3 -= mR;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 += mR;
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3.AssignDifferenceOf(m3, mS);
    BOOST_CHECK(m3.IsEqualTo(m0));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixMultiplyAndDiv, RandomSeedFixture)
{
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    DMatrix m00(2, 3);
    m00(0, 0) = 10;
    m00(0, 1) = 20;
    m00(0, 2) = 30;
    m00(1, 0) = 40;
    m00(1, 1) = 50;
    m00(1, 2) = 60;

    // TODO: consider separate reshape test
    DMatrix m1(2, 3);
    m1.Reshape(3, 2);
    m1(0, 0) = 11;
    m1(0, 1) = 15;
    m1(1, 0) = 14;
    m1(1, 1) = 13;
    m1(2, 0) = 12;
    m1(2, 1) = 16;

    DMatrix m2(2, 2);
    m2(0, 0) = 75;
    m2(0, 1) = 89;
    m2(1, 0) = 186;
    m2(1, 1) = 221;

    DMatrix m3 = m0 * m1;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3 = m0 * 10;
    BOOST_CHECK(m3.IsEqualTo(m00));

    m3 = m3 / 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    m3 *= 10;
    BOOST_CHECK(m3.IsEqualTo(m00));

    m3 /= 10;
    BOOST_CHECK(m3.IsEqualTo(m0));

    DMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, false, 0, m3);
    BOOST_CHECK(m3.IsEqualTo(m2));

    m1.Reshape(2, 3);
    DMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, true, 0, m3);
    m2(0, 0) = 74;
    m2(0, 1) = 92;
    m2(1, 0) = 182;
    m2(1, 1) = 227;
    BOOST_CHECK(m3.IsEqualTo(m2));

    DMatrix::MultiplyAndWeightedAdd(10, m0, false, m1, true, 2, m3);
    m2(0, 0) = 888;
    m2(0, 1) = 1104;
    m2(1, 0) = 2184;
    m2(1, 1) = 2724;
    BOOST_CHECK(m3.IsEqualTo(m2));

    DMatrix::MultiplyAndWeightedAdd(1, m0, true, m1, false, 0, m3);
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
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixElementOperations, RandomSeedFixture)
{
    // TODO: consider splitting this large test
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    DMatrix m0Inverse(2, 3);
    m0Inverse(0, 0) = 1.0;
    m0Inverse(0, 1) = 1 / 2.0;
    m0Inverse(0, 2) = 1 / 3.0;
    m0Inverse(1, 0) = 1 / 4.0;
    m0Inverse(1, 1) = 1 / 5.0;
    m0Inverse(1, 2) = 1 / 6.0;

    DMatrix m1(2, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 1;
    m1(0, 2) = 1;
    m1(1, 0) = 1;
    m1(1, 1) = 1;
    m1(1, 2) = 1;

    DMatrix m3;
    m3.AssignElementProductOf(m0, m0Inverse);
    BOOST_CHECK(m3.IsEqualTo(m1, c_epsilonFloatE4));

    m3 = m0 ^ 4;
    DMatrix m2(2, 3);
    m2(0, 0) = 1;
    m2(0, 1) = 16;
    m2(0, 2) = 81;
    m2(1, 0) = 256;
    m2(1, 1) = 625;
    m2(1, 2) = 1296;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3.SetValue(m0);
    m3 ^= 4;
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3.SetValue(m0);
    m3.ElementMultiplyWith(m0Inverse);
    BOOST_CHECK(m3.IsEqualTo(m1));

    m3.SetValue(m0);
    m3.ElementInverse();
    BOOST_CHECK(m3.IsEqualTo(m0Inverse));

    m2(0, 0) = 0.7311;
    m2(0, 1) = 0.8808;
    m2(0, 2) = 0.9526;
    m2(1, 0) = 0.9820;
    m2(1, 1) = 0.9933;
    m2(1, 2) = 0.9975;
    m3.AssignElementDivisionOf(m2, m0);
    m2.ElementMultiplyWith(m0Inverse);
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceSigmoid();
    m2(0, 0) = 0.7311;
    m2(0, 1) = 0.8808;
    m2(0, 2) = 0.9526;
    m2(1, 0) = 0.9820;
    m2(1, 1) = 0.9933;
    m2(1, 2) = 0.9975;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceTanh();
    m2(0, 0) = 0.7616;
    m2(0, 1) = 0.9640;
    m2(0, 2) = 0.9951;
    m2(1, 0) = 0.9993;
    m2(1, 1) = 0.9999;
    m2(1, 2) = 1.0000;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceLogSoftmax(true);
    m3.InplaceExp();
    m2(0, 0) = 0.0474;
    m2(0, 1) = 0.0474;
    m2(0, 2) = 0.0474;
    m2(1, 0) = 0.9526;
    m2(1, 1) = 0.9526;
    m2(1, 2) = 0.9526;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceLogSoftmax(false);
    m3.InplaceExp();
    m2(0, 0) = 0.0900;
    m2(0, 1) = 0.2447;
    m2(0, 2) = 0.6652;
    m2(1, 0) = 0.0900;
    m2(1, 1) = 0.2447;
    m2(1, 2) = 0.6652;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceHardmax(true);
    m2(0, 0) = 0.0;
    m2(0, 1) = 0.0;
    m2(0, 2) = 0.0;
    m2(1, 0) = 1.0;
    m2(1, 1) = 1.0;
    m2(1, 2) = 1.0;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceHardmax(false);
    m2(0, 0) = 0.0;
    m2(0, 1) = 0.0;
    m2(0, 2) = 1.0;
    m2(1, 0) = 0.0;
    m2(1, 1) = 0.0;
    m2(1, 2) = 1.0;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceSqrt();
    m2(0, 0) = 1;
    m2(0, 1) = 1.4142;
    m2(0, 2) = 1.7321;
    m2(1, 0) = 2;
    m2(1, 1) = 2.2361;
    m2(1, 2) = 2.4495;
    BOOST_CHECK(m3.IsEqualTo(m2, c_epsilonFloatE4));

    m3.SetValue(m0);
    m3.InplaceExp();
    m2(0, 0) = 2.7183;
    m2(0, 1) = 7.3891;
    m2(0, 2) = 20.0855;
    m2(1, 0) = 54.5982;
    m2(1, 1) = 148.4132;
    m2(1, 2) = 403.4288;
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
    BOOST_CHECK(m3.IsEqualTo(m2));

    m3.SetValue(m0);
    m3.InplaceTruncateTop(4);
    m2(0, 0) = 1;
    m2(0, 1) = 2;
    m2(0, 2) = 3;
    m2(1, 0) = 4;
    m2(1, 1) = 4;
    m2(1, 2) = 4;
    BOOST_CHECK(m3.IsEqualTo(m2));

    DMatrix m_Trig(2, 3);
    m_Trig(0, 0) = 0;
    m_Trig(0, 1) = pi / 2.0;
    m_Trig(0, 2) = pi;
    m_Trig(1, 0) = 3.0 * pi / 2.0;
    m_Trig(1, 1) = 2.0 * pi;
    m_Trig(1, 2) = 5.0 * pi / 2.0;

    DMatrix m_Cos(2, 3);
    m_Cos.SetValue(m_Trig);

    DMatrix m_Cos_expected(2, 3);
    m_Cos_expected(0, 0) = 1;
    m_Cos_expected(0, 1) = 0;
    m_Cos_expected(0, 2) = -1;
    m_Cos_expected(1, 0) = 0;
    m_Cos_expected(1, 1) = 1;
    m_Cos_expected(1, 2) = 0;

    m_Cos.InplaceCosine();
    BOOST_CHECK(m_Cos.IsEqualTo(m_Cos_expected, c_epsilonFloatE4));

    m_Cos.SetValue(m_Trig);
    m_Cos.AssignCosineOf(m_Trig);
    BOOST_CHECK(m_Cos.IsEqualTo(m_Cos_expected, c_epsilonFloatE4));

    DMatrix m_NegSine(2, 3);
    m_NegSine.SetValue(m_Trig);

    DMatrix m_NegSine_expected(2, 3);
    m_NegSine_expected(0, 0) = 0;
    m_NegSine_expected(0, 1) = -1;
    m_NegSine_expected(0, 2) = 0;
    m_NegSine_expected(1, 0) = 1;
    m_NegSine_expected(1, 1) = 0;
    m_NegSine_expected(1, 2) = -1;

    m_NegSine.InplaceNegativeSine();
    BOOST_CHECK(m_NegSine.IsEqualTo(m_NegSine_expected, c_epsilonFloatE4));

    m_NegSine.SetValue(m_Trig);
    m_NegSine.AssignNegativeSineOf(m_Trig);
    BOOST_CHECK(m_NegSine.IsEqualTo(m_NegSine_expected, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixNorms, RandomSeedFixture)
{
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    DMatrix mResult;
    m0.VectorNorm1(mResult, true);
    DMatrix m2(1, 3);
    m2(0, 0) = 5;
    m2(0, 1) = 7;
    m2(0, 2) = 9;
    BOOST_CHECK(mResult.IsEqualTo(m2));

    m0.VectorNorm1(mResult, false);
    m2.Resize(2, 1);
    m2(0, 0) = 6;
    m2(1, 0) = 15;
    BOOST_CHECK(mResult.IsEqualTo(m2));

    m0.VectorNorm2(mResult, true);
    m2.Resize(1, 3);
    m2(0, 0) = 4.1231;
    m2(0, 1) = 5.3852;
    m2(0, 2) = 6.7082;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));

    m0.VectorNorm2(mResult, false);
    m2.Resize(2, 1);
    m2(0, 0) = 3.7417;
    m2(1, 0) = 8.7750;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));

    m0.VectorNormInf(mResult, true);
    m2.Resize(1, 3);
    m2(0, 0) = 4;
    m2(0, 1) = 5;
    m2(0, 2) = 6;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));

    m0.VectorNormInf(mResult, false);
    m2.Resize(2, 1);
    m2(0, 0) = 3;
    m2(1, 0) = 6;
    BOOST_CHECK(mResult.IsEqualTo(m2));

    BOOST_CHECK(abs(m0.FrobeniusNorm() - 9.5394) < c_epsilonFloatE4);
    BOOST_CHECK(abs(m0.MatrixNormInf() - 6) < c_epsilonFloatE4);

    DMatrix m1;
    m0.VectorMax(m1, mResult, true);
    m2.Resize(1, 3);
    m2(0, 0) = 4;
    m2(0, 1) = 5;
    m2(0, 2) = 6;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));

    m0.VectorMax(m1, mResult, false);
    m2.Resize(2, 1);
    m2(0, 0) = 3;
    m2(1, 0) = 6;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));

    m0.VectorMin(m1, mResult, true);
    m2.Resize(1, 3);
    m2(0, 0) = 1;
    m2(0, 1) = 2;
    m2(0, 2) = 3;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));

    m0.VectorMin(m1, mResult, false);
    m2.Resize(2, 1);
    m2(0, 0) = 1;
    m2(1, 0) = 4;
    BOOST_CHECK(mResult.IsEqualTo(m2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixSetValues, RandomSeedFixture)
{
    DMatrix m0(3, 3);
    m0(0, 0) = 10;
    m0(1, 1) = 10;
    m0(2, 2) = 10;

    DMatrix m1(3, 3);
    m1.SetDiagonalValue(10);
    BOOST_CHECK(m1.IsEqualTo(m0, c_epsilonFloatE4));

    DMatrix m2(3, 1);
    m2(0, 0) = 10;
    m2(1, 0) = 10;
    m2(2, 0) = 10;
    m1.SetDiagonalValue(m2);
    BOOST_CHECK(m1.IsEqualTo(m0, c_epsilonFloatE4));

    m1.SetUniformRandomValue(-0.01, 0.01, IncrementCounter());
    foreach_coord (i, j, m1)
    {
        BOOST_CHECK(m1(i, j) >= -0.01 && m1(i, j) < 0.01);
    }

    m1.Resize(20, 20);
    m1.SetGaussianRandomValue(1.0, 0.01, IncrementCounter());
    BOOST_CHECK_CLOSE(m1.SumOfElements(), static_cast<double>(m1.GetNumElements()), 1);
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixTranspose, RandomSeedFixture)
{
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    DMatrix m1(3, 2);
    m1(0, 0) = 1;
    m1(0, 1) = 4;
    m1(1, 0) = 2;
    m1(1, 1) = 5;
    m1(2, 0) = 3;
    m1(2, 1) = 6;

    DMatrix m2 = m0.Transpose();
    BOOST_CHECK(m2.IsEqualTo(m1, c_epsilonFloatE4));

    m2.AssignTransposeOf(m1);
    BOOST_CHECK(m2.IsEqualTo(m0, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixColumnSlice, RandomSeedFixture)
{
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(0, 2) = 3;
    m0(1, 0) = 4;
    m0(1, 1) = 5;
    m0(1, 2) = 6;

    DMatrix m1(2, 2);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(1, 0) = 4;
    m1(1, 1) = 5;

    DMatrix m2 = m0.ColumnSlice(0, 2);
    BOOST_CHECK(m2.IsEqualTo(m1, c_epsilonFloatE4));

    m1(0, 0) = 2;
    m1(0, 1) = 3;
    m1(1, 0) = 5;
    m1(1, 1) = 6;

    m2 = m0.ColumnSlice(1, 2);
    BOOST_CHECK(m2.IsEqualTo(m1, c_epsilonFloatE4));

    // TODO: this fails due to access violation (at least on desktop machine of pkranen)
    // size_t k = 100, n = 20, m = 50;
    // reducing sizes to 2, 20, 5
    size_t k = 2;
    size_t n = 20;
    size_t m = 5;

    DMatrix mA(k, n);
    mA.SetUniformRandomValue(-1, 1, IncrementCounter());

    DMatrix mB(n, m);
    mB.SetUniformRandomValue(-1, 1, IncrementCounter());

    DMatrix mC(k, m);
    mC.SetUniformRandomValue(-1, 1, IncrementCounter());

    DMatrix mD(k, m);
    mD.SetValue(mC);

    DMatrix::MultiplyAndAdd(mA, false, mB, false, mD);

    for (int i = 0; i < m; i++)
    {
        DMatrix colMB = mB.ColumnSlice(i, 1);
        DMatrix colMC = mC.ColumnSlice(i, 1);
        DMatrix::MultiplyAndAdd(mA, false, colMB, false, colMC);
    }

    BOOST_CHECK(mC.IsEqualTo(mD, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUKhatriRaoProduct, RandomSeedFixture)
{
    DMatrix mA(3, 4);
    mA(0, 0) = 0.8147;
    mA(0, 1) = 0.9134;
    mA(0, 2) = 0.2785;
    mA(0, 3) = 0.9649;
    mA(1, 0) = 0.9058;
    mA(1, 1) = 0.6324;
    mA(1, 2) = 0.5469;
    mA(1, 3) = 0.1576;
    mA(2, 0) = 0.1270;
    mA(2, 1) = 0.0975;
    mA(2, 2) = 0.9575;
    mA(2, 3) = 0.9706;

    DMatrix mB(2, 4);
    mB(0, 0) = 0.9572;
    mB(0, 1) = 0.8003;
    mB(0, 2) = 0.4218;
    mB(0, 3) = 0.7922;
    mB(1, 0) = 0.4854;
    mB(1, 1) = 0.1419;
    mB(1, 2) = 0.9157;
    mB(1, 3) = 0.9595;

    DMatrix mD(6, 4);
    mD(0, 0) = 0.7798;
    mD(0, 1) = 0.7310;
    mD(0, 2) = 0.1175;
    mD(0, 3) = 0.7644;
    mD(1, 0) = 0.8670;
    mD(1, 1) = 0.5061;
    mD(1, 2) = 0.2307;
    mD(1, 3) = 0.1249;
    mD(2, 0) = 0.1215;
    mD(2, 1) = 0.0781;
    mD(2, 2) = 0.4038;
    mD(2, 3) = 0.7689;
    mD(3, 0) = 0.3954;
    mD(3, 1) = 0.1296;
    mD(3, 2) = 0.2550;
    mD(3, 3) = 0.9258;
    mD(4, 0) = 0.4396;
    mD(4, 1) = 0.0897;
    mD(4, 2) = 0.5008;
    mD(4, 3) = 0.1512;
    mD(5, 0) = 0.0616;
    mD(5, 1) = 0.0138;
    mD(5, 2) = 0.8768;
    mD(5, 3) = 0.9313;

    DMatrix mC;
    mC.AssignKhatriRaoProductOf(mA, mB);
    BOOST_CHECK(mC.IsEqualTo(mD, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUAddColumnReshapeProductOf, RandomSeedFixture)
{
    DMatrix mA(6, 2);
    mA(0, 0) = 0.6557;
    mA(0, 1) = 0.7431;
    mA(1, 0) = 0.0357;
    mA(1, 1) = 0.3922;
    mA(2, 0) = 0.8491;
    mA(2, 1) = 0.6555;
    mA(3, 0) = 0.9340;
    mA(3, 1) = 0.1712;
    mA(4, 0) = 0.6787;
    mA(4, 1) = 0.7060;
    mA(5, 0) = 0.7577;
    mA(5, 1) = 0.0318;

    DMatrix mB(3, 2);
    mB(0, 0) = 0.2769;
    mB(0, 1) = 0.8235;
    mB(1, 0) = 0.0462;
    mB(1, 1) = 0.6948;
    mB(2, 0) = 0.0971;
    mB(2, 1) = 0.3171;

    DMatrix mD(2, 2);
    mD(0, 0) = 0.2867;
    mD(0, 1) = 1.2913;
    mD(1, 0) = 0.1266;
    mD(1, 1) = 0.4520;

    DMatrix mE(2, 2);
    mE(0, 0) = 0.2657;
    mE(0, 1) = 1.0923;
    mE(1, 0) = 0.3636;
    mE(1, 1) = 0.6416;

    DMatrix mC(2, 2);
    mC.SetValue(0);
    mC.AddColumnReshapeProductOf(mA, mB, false);
    BOOST_CHECK(mC.IsEqualTo(mD, c_epsilonFloatE4));

    mC.SetValue(0);
    mC.AddColumnReshapeProductOf(mA, mB, true);
    BOOST_CHECK(mC.IsEqualTo(mE, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixRowSliceAndStack, RandomSeedFixture)
{
    DMatrix m0(5, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 6;
    m0(0, 2) = 11;
    m0(1, 0) = 2;
    m0(1, 1) = 7;
    m0(1, 2) = 12;
    m0(2, 0) = 3;
    m0(2, 1) = 8;
    m0(2, 2) = 13;
    m0(3, 0) = 4;
    m0(3, 1) = 9;
    m0(3, 2) = 14;
    m0(4, 0) = 5;
    m0(4, 1) = 10;
    m0(4, 2) = 15;

    DMatrix m1(2, 3);
    m1(0, 0) = 3;
    m1(0, 1) = 8;
    m1(0, 2) = 13;
    m1(1, 0) = 4;
    m1(1, 1) = 9;
    m1(1, 2) = 14;

    DMatrix m2;
    m2.AssignRowSliceValuesOf(m0, 2, 2);
    BOOST_CHECK(m2.IsEqualTo(m1, c_epsilonFloatE4));

    DMatrix m3(5, 3);
    m3(0, 0) = 0;
    m3(0, 1) = 0;
    m3(0, 2) = 0;
    m3(1, 0) = 0;
    m3(1, 1) = 0;
    m3(1, 2) = 0;
    m3(2, 0) = 3;
    m3(2, 1) = 8;
    m3(2, 2) = 13;
    m3(3, 0) = 4;
    m3(3, 1) = 9;
    m3(3, 2) = 14;
    m3(4, 0) = 0;
    m3(4, 1) = 0;
    m3(4, 2) = 0;

    m3 += m0;
    m0.AddToRowSliceValuesOf(m1, 2, 2);
    BOOST_CHECK(m3.IsEqualTo(m0, c_epsilonFloatE4));

    m2.AddWithRowSliceValuesOf(m1, 0, 2);
    DMatrix m4(2, 3);
    m4(0, 0) = 6;
    m4(0, 1) = 16;
    m4(0, 2) = 26;
    m4(1, 0) = 8;
    m4(1, 1) = 18;
    m4(1, 2) = 28;
    BOOST_CHECK(m2.IsEqualTo(m4, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUAssignRepeatOf, RandomSeedFixture)
{
    DMatrix m0(2, 3);
    m0(0, 0) = 1;
    m0(0, 1) = 6;
    m0(0, 2) = 11;
    m0(1, 0) = 2;
    m0(1, 1) = 7;
    m0(1, 2) = 12;

    DMatrix m1;
    m1.AssignRepeatOf(m0, 1, 1);
    BOOST_CHECK(m1.IsEqualTo(m0, c_epsilonFloatE4));

    DMatrix m2(6, 6);
    m2(0, 0) = 1;
    m2(0, 1) = 6;
    m2(0, 2) = 11;
    m2(0, 3) = 1;
    m2(0, 4) = 6;
    m2(0, 5) = 11;
    m2(1, 0) = 2;
    m2(1, 1) = 7;
    m2(1, 2) = 12;
    m2(1, 3) = 2;
    m2(1, 4) = 7;
    m2(1, 5) = 12;
    m2(2, 0) = 1;
    m2(2, 1) = 6;
    m2(2, 2) = 11;
    m2(2, 3) = 1;
    m2(2, 4) = 6;
    m2(2, 5) = 11;
    m2(3, 0) = 2;
    m2(3, 1) = 7;
    m2(3, 2) = 12;
    m2(3, 3) = 2;
    m2(3, 4) = 7;
    m2(3, 5) = 12;
    m2(4, 0) = 1;
    m2(4, 1) = 6;
    m2(4, 2) = 11;
    m2(4, 3) = 1;
    m2(4, 4) = 6;
    m2(4, 5) = 11;
    m2(5, 0) = 2;
    m2(5, 1) = 7;
    m2(5, 2) = 12;
    m2(5, 3) = 2;
    m2(5, 4) = 7;
    m2(5, 5) = 12;

    m1.AssignRepeatOf(m0, 3, 2);
    BOOST_CHECK(m1.IsEqualTo(m2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPURowElementOperations, RandomSeedFixture)
{
    DMatrix m0 = DMatrix::RandomUniform(20, 28, -1, 1, IncrementCounter());
    DMatrix m1 = DMatrix::RandomUniform(1, 28, 1, 2, IncrementCounter());

    DMatrix m2;
    m2.SetValue(m0);
    m2.RowElementMultiplyWith(m1);
    m2.RowElementDivideBy(m1);

    BOOST_CHECK(m0.IsEqualTo(m2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUColumnElementOperations, RandomSeedFixture)
{
    DMatrix m0 = DMatrix::RandomUniform(20, 28, -1, 1, IncrementCounter());
    DMatrix m1 = DMatrix::RandomUniform(20, 1, 1, 2, IncrementCounter());

    DMatrix m2;
    m2.SetValue(m0);
    m2.ColumnElementMultiplyWith(m1);
    m2.ColumnElementDivideBy(m1);

    BOOST_CHECK(m0.IsEqualTo(m2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixSeedingFloat, RandomSeedFixture)
{
    const float low = 0;
    const float high = 1;
    const unsigned long seed = 4711;

    auto m1 = CPUMatrix<float>::RandomUniform(16, 16, low, high, seed);
    auto m2 = CPUMatrix<float>::RandomUniform(16, 16, low, high, seed);

    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixSeedingDouble, RandomSeedFixture)
{
    const double low = 0;
    const double high = 1;
    const unsigned long seed = 4711;

    auto m1 = CPUMatrix<double>::RandomUniform(16, 16, low, high, seed);
    auto m2 = CPUMatrix<double>::RandomUniform(16, 16, low, high, seed);

    BOOST_CHECK(m1.IsEqualTo(m2));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixAdam, RandomSeedFixture)
{
    CPUMatrix<double> adamMatrix;
    CPUMatrix<double> gradients(2, 1);
    CPUMatrix<double> parameters(2, 1);
    CPUMatrix<double> expectedParameters(2, 1);
    CPUMatrix<double> expectedStates(2, 2);
    double gradientValues[] = { 0.1, -0.1 };
    double paramValues[] = { 0.1, 0.1 };
    double expectedValues[] = { -0.05811338, 0.25811338 };
    double expectedStateValues[] = {1e-5, 0.01, 1e-5, -0.01};
    gradients.SetValue(2, 1, gradientValues, matrixFormatRowMajor);
    parameters.SetValue(2, 1, paramValues, matrixFormatRowMajor);
    expectedParameters.SetValue(2, 1, expectedValues, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, expectedStateValues, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, true);

    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-6));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-6));

    double expectedValues2[] = { -0.27059249, 0.47059249 };
    double expectedStateValues2[] = { 2e-05, 0.019, 2e-05, -0.019 };
    expectedParameters.SetValue(2, 1, expectedValues2, matrixFormatRowMajor);
    expectedStates.SetValue(2, 2, expectedStateValues2, matrixFormatRowMajor);
    adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, true);

    BOOST_CHECK(parameters.IsEqualTo(expectedParameters, 1e-6));
    BOOST_CHECK(adamMatrix.IsEqualTo(expectedStates, 1e-6));
}

BOOST_FIXTURE_TEST_CASE(CPUMatrixOneHot, RandomSeedFixture)
{
    const size_t num_class = 6;
    
    DMatrix m0(2, 2);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(1, 0) = 3;
    m0(1, 1) = 4;

    DMatrix expect(12, 2);
    expect(1, 0) = 1;
    expect(9, 0) = 1;
    expect(2, 1) = 1;
    expect(10, 1) = 1;

    vector<size_t> shape(3);
    shape[0] = num_class; shape[1] = 2; shape[2] = 2;
    DMatrix m1;
    m1.AssignOneHot(m0, shape, 0);
    BOOST_CHECK(m1.GetNumRows() == 12);
    BOOST_CHECK(m1.GetNumCols() == 2);
    BOOST_CHECK(m1.IsEqualTo(expect, 1e-6));

    DMatrix expect2(12, 2);
    expect2(2, 0) = 1;
    expect2(7, 0) = 1;
    expect2(4, 1) = 1;
    expect2(9, 1) = 1;

    vector<size_t> shape2(3);
    shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;
    DMatrix m2;
    m2.AssignOneHot(m0, shape2, 1);
    BOOST_CHECK(m2.GetNumRows() == 12);
    BOOST_CHECK(m2.GetNumCols() == 2);
    BOOST_CHECK(m2.IsEqualTo(expect2, 1e-6));

    DMatrix dirtyMatrix(2, 2);
    dirtyMatrix(0, 0) = 1;
    dirtyMatrix(0, 1) = -1;
    dirtyMatrix(1, 0) = 7;
    dirtyMatrix(1, 1) = 4;

    DMatrix dirtyExpect(12, 2);
    dirtyExpect(1, 0) = 1;
    dirtyExpect(9, 0) = 0;
    dirtyExpect(2, 1) = 0;
    dirtyExpect(10, 1) = 1;

    DMatrix dirty_m;
    dirty_m.AssignOneHot(dirtyMatrix, shape, 0);
    BOOST_CHECK(dirty_m.GetNumRows() == 12);
    BOOST_CHECK(dirty_m.GetNumCols() == 2);
    BOOST_CHECK(dirty_m.IsEqualTo(dirtyExpect, 1e-6));
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
