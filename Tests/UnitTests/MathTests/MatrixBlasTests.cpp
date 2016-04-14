//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <math.h>
#include <crtdefs.h>
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/CPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

const size_t c_value_1 = 1;
const size_t c_value_2 = 2;
const size_t c_value_3 = 3;

BOOST_AUTO_TEST_SUITE(CPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(MatrixMultiplyTest, RandomSeedFixture)
{
    // Part 1: Multiply with identity matrix
    SingleMatrix matrixA = SingleMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, IncrementCounter());
    SingleMatrix matrixB = SingleMatrix::Eye(23, c_deviceIdZero);
    SingleMatrix matrixC = matrixA * matrixB;
    foreach_coord (i, j, matrixA)
    {
        BOOST_CHECK_EQUAL(matrixA(i, j), matrixC(i, j));
    }

    SingleMatrix matrix0(3, 3, c_deviceIdZero);
    matrix0(0, 0) = 1;
    matrix0(0, 1) = 2;
    matrix0(0, 2) = 3;
    matrix0(1, 0) = 4;
    matrix0(1, 1) = 5;
    matrix0(1, 2) = 6;
    matrix0(2, 0) = 7;
    matrix0(2, 1) = 8;
    matrix0(2, 2) = 9;

    // Part 2: Compare with Octave on toy example
    SingleMatrix matrix1(3, 4, c_deviceIdZero);
    matrix1(0, 0) = 8;
    matrix1(0, 1) = 9;
    matrix1(0, 2) = 3;
    matrix1(0, 3) = 45;
    matrix1(1, 0) = 3;
    matrix1(1, 1) = 4;
    matrix1(1, 2) = 56;
    matrix1(1, 3) = 1;
    matrix1(2, 0) = -3;
    matrix1(2, 1) = 5;
    matrix1(2, 2) = 2;
    matrix1(2, 3) = 6;
    SingleMatrix matrix2 = matrix0 * matrix1;
    BOOST_CHECK_EQUAL(5, matrix2(0, 0));
    BOOST_CHECK_EQUAL(32, matrix2(0, 1));
    BOOST_CHECK_EQUAL(121, matrix2(0, 2));
    BOOST_CHECK_EQUAL(65, matrix2(0, 3));
    BOOST_CHECK_EQUAL(29, matrix2(1, 0));
    BOOST_CHECK_EQUAL(86, matrix2(1, 1));
    BOOST_CHECK_EQUAL(304, matrix2(1, 2));
    BOOST_CHECK_EQUAL(221, matrix2(1, 3));
    BOOST_CHECK_EQUAL(53, matrix2(2, 0));
    BOOST_CHECK_EQUAL(140, matrix2(2, 1));
    BOOST_CHECK_EQUAL(487, matrix2(2, 2));
    BOOST_CHECK_EQUAL(377, matrix2(2, 3));
}

BOOST_FIXTURE_TEST_CASE(MatrixMultiplyAndPlusAndMinus, RandomSeedFixture)
{
    // Part 1: Multiply with identity matrix
    SingleMatrix matrixA = SingleMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, IncrementCounter());
    SingleMatrix matrixB = SingleMatrix::Eye(23, c_deviceIdZero);
    SingleMatrix matrixB1 = SingleMatrix::RandomUniform(64, 23, c_deviceIdZero, -95.23f, 43.5f, IncrementCounter());
    SingleMatrix matrixB2 = SingleMatrix::RandomUniform(64, 23, c_deviceIdZero, 23.23f, 143.5f, IncrementCounter());
    SingleMatrix C = ((matrixA * matrixB) + matrixB1) - matrixB2;
    foreach_coord (i, j, matrixA)
    {
        BOOST_CHECK_EQUAL(matrixA(i, j) + matrixB1(i, j) - matrixB2(i, j), C(i, j));
    }

    // Part 3: compare with CPUMatrix results
    // TODO: Split into separate test case WI# 82
    CPUMatrix<float> cpuMatrix1 = CPUMatrix<float>::RandomUniform(429, 1024, -1, 1, IncrementCounter());
    CPUMatrix<float> cpuMatrix2 = CPUMatrix<float>::RandomUniform(429, 1024, -2, 2, IncrementCounter());
    CPUMatrix<float> cpuMatrix3 = CPUMatrix<float>::RandomUniform(1024, 1024, -3, 1, IncrementCounter());
    CPUMatrix<float> copyCpuMatrix3(cpuMatrix3);
    CPUMatrix<float>::MultiplyAndAdd(cpuMatrix1, true, cpuMatrix2, false, cpuMatrix3);

    SingleMatrix singleMatrix1(429, 1024, cpuMatrix1.Buffer(), matrixFlagNormal);
    SingleMatrix singleMatrix2(429, 1024, cpuMatrix2.Buffer(), matrixFlagNormal);
    SingleMatrix singleMatrix3(1024, 1024, copyCpuMatrix3.Buffer(), matrixFlagNormal);
    SingleMatrix::MultiplyAndAdd(singleMatrix1, true, singleMatrix2, false, singleMatrix3);
    foreach_coord (i, j, singleMatrix3)
    {
        BOOST_CHECK(fabs(singleMatrix3(i, j) - cpuMatrix3(i, j)) < c_epsilonFloat5E4);
    }

    // TODO: Split into separate test case WI# 82
    SingleMatrix singleMatrix4 = SingleMatrix::Ones(8, 9, c_deviceIdZero);
    SingleMatrix singleMatrix5 = SingleMatrix::Ones(8, 1, c_deviceIdZero);
    singleMatrix5(4, 0) = -5.5;
    SingleMatrix::ScaleAndAdd(1, singleMatrix5, singleMatrix4);
    foreach_coord (i, j, singleMatrix4)
    {
        if (i != 4)
        {
            BOOST_CHECK_EQUAL(2, singleMatrix4(i, j));
        }
        else
        {
            BOOST_CHECK_EQUAL(-4.5, singleMatrix4(i, j));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixScaleAndAdd, RandomSeedFixture)
{
    const int seed = rand();
    const SingleMatrix singleMatrixA = SingleMatrix::RandomUniform(1024, 512, c_deviceIdZero , - 12.34f, 55.2312f, seed + 0);
    const SingleMatrix singleMatrixB = SingleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34f, 55.2312f, seed + 1);
    SingleMatrix singleMatrixC(singleMatrixB.DeepClone());
    const float alpha = 0.34213f;
    SingleMatrix::ScaleAndAdd(alpha, singleMatrixA, singleMatrixC);
    foreach_coord (i, j, singleMatrixC)
    {
        BOOST_CHECK(fabsf(singleMatrixC(i, j) - (alpha * singleMatrixA(i, j) + singleMatrixB(i, j))) < c_epsilonFloatE5);
    }

    // Test 2
    // TODO: Split into separate test case WI# 82
    const SingleMatrix singleMatrixA1 = SingleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34f, 55.2312f, seed + 2);
    const SingleMatrix singleMatrixB1 = SingleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34f, 55.2312f, seed + 3);
    SingleMatrix singleMatrixC1(singleMatrixB1.DeepClone()); // C1==B1
    const float beta = -1.4654f;
    SingleMatrix::ScaleAndAdd(alpha, singleMatrixA1, beta, singleMatrixC1); // C1=alpha*A1+beta*C1
    foreach_coord (i, j, singleMatrixC1)
    {
        BOOST_CHECK(fabsf(singleMatrixC1(i, j) - (alpha * singleMatrixA1(i, j) + beta * singleMatrixB1(i, j))) < c_epsilonFloatE5);
    }

    // Test 3 - columnwise
    // TODO: Split into separate test case WI# 82
    const SingleMatrix singleMatrixA2 = SingleMatrix::RandomUniform(1024, 1, c_deviceIdZero, -12.34f, 55.2312f, seed + 4);
    const SingleMatrix singleMatrixB2 = SingleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34f, 55.2312f, seed + 5); // Column
    SingleMatrix singleMatrixC2(singleMatrixB2.DeepClone());                                                                // C2==B2
    const float betaOne = 1;
    SingleMatrix::ScaleAndAdd(alpha, singleMatrixA2, betaOne, singleMatrixC2); // C2=alpha*A1+beta*C1
    foreach_coord (i, j, singleMatrixC2)
    {
        float x = singleMatrixC2(i, j);
        float y = (alpha * singleMatrixA2(i, 0) + betaOne * singleMatrixB2(i, j));
        BOOST_CHECK(fabsf(x - y) < c_epsilonFloatE5);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixScaleAndAdd_double, RandomSeedFixture)
{
    const int seed = rand();
    DoubleMatrix matrixA = DoubleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34, 55.2312, seed + 0);
    DoubleMatrix matrixB = DoubleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34, 55.2312, seed + 1);
    DoubleMatrix matrixC(matrixB.DeepClone());
    const float alpha = 0.34213f;
    DoubleMatrix::ScaleAndAdd(alpha, matrixA, matrixC);
    foreach_coord (i, j, matrixC)
    {
        BOOST_CHECK(fabsf(static_cast<float>(matrixC(i, j) - (alpha * matrixA(i, j) + matrixB(i, j)))) < c_epsilonDoubleE11);
    }

    // Test 2
    // TODO: Split into separate test case WI# 82
    DoubleMatrix matrixA1 = DoubleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34f, 55.2312f, seed + 2);
    DoubleMatrix matrixB1 = DoubleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34f, 55.2312f, seed + 3);
    DoubleMatrix matrixC1(matrixB1.DeepClone()); // C1==B1
    const float beta = -1.4654f;
    DoubleMatrix::ScaleAndAdd(alpha, matrixA1, beta, matrixC1); // C1=alpha*A1+beta*C1
    foreach_coord (i, j, matrixC1)
    {
        BOOST_CHECK(fabsf(static_cast<float>(matrixC1(i, j) - (alpha * matrixA1(i, j) + beta * matrixB1(i, j)))) < c_epsilonDoubleE11);
    }

    // Test 3 - columnwise
    // TODO: Split into separate test case WI# 82
    DoubleMatrix matrixA2 = DoubleMatrix::RandomUniform(1024, 1, c_deviceIdZero, -12.34, 55.2312, seed + 4);
    DoubleMatrix matrixB2 = DoubleMatrix::RandomUniform(1024, 512, c_deviceIdZero, -12.34, 55.2312, seed + 5); // Column
    DoubleMatrix matrixC2(matrixB2.DeepClone());                                                              // C2==B2
    const float betaOne = 1;
    DoubleMatrix::ScaleAndAdd(alpha, matrixA2, betaOne, matrixC2); // C2=alpha*A1+beta*C1
    foreach_coord (i, j, matrixC2)
    {
        float x = static_cast<float>(matrixC2(i, j));
        float y = static_cast<float>((alpha * matrixA2(i, 0)) + (betaOne * matrixB2(i, j)));
        BOOST_CHECK(fabsf(x - y) < c_epsilonDoubleE11);
    }
}

BOOST_FIXTURE_TEST_CASE(MatrixNorms, RandomSeedFixture)
{
    SingleMatrix matrix0(c_value_2, c_value_3, c_deviceIdZero);
    matrix0(0, 0) = 1;
    matrix0(0, 1) = 2;
    matrix0(0, 2) = 3;
    matrix0(1, 0) = 4;
    matrix0(1, 1) = 5;
    matrix0(1, 2) = 6;

    SingleMatrix matrix3(c_deviceIdZero);
    matrix0.VectorNorm1(matrix3, true);
    SingleMatrix matrix2(c_value_1, c_value_3, c_deviceIdZero);
    matrix2(0, 0) = 5;
    matrix2(0, 1) = 7;
    matrix2(0, 2) = 9;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2));

    matrix0.VectorNorm1(matrix3, false);
    matrix2.Resize(2, 1);
    matrix2(0, 0) = 6;
    matrix2(1, 0) = 15;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2));

    matrix0.VectorNorm2(matrix3, true);
    matrix2.Resize(1, 3);
    matrix2(0, 0) = 4.1231f;
    matrix2(0, 1) = 5.3852f;
    matrix2(0, 2) = 6.7082f;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2, c_epsilonFloat5E4));

    matrix0.VectorNorm2(matrix3, false);
    matrix2.Resize(2, 1);
    matrix2(0, 0) = 3.7417f;
    matrix2(1, 0) = 8.7750f;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2, c_epsilonFloat5E4));

    SingleMatrix matrix00(c_value_2, c_value_3, c_deviceIdZero);
    matrix00(0, 0) = 1;
    matrix00(0, 1) = 2;
    matrix00(0, 2) = 3;
    matrix00(1, 0) = 4;
    matrix00(1, 1) = 5;
    matrix00(1, 2) = 6;
    SingleMatrix matrix1(c_deviceIdZero);
    matrix00.VectorMax(matrix1, matrix3, true);
    matrix2.Resize(1, 3);
    matrix2(0, 0) = 4;
    matrix2(0, 1) = 5;
    matrix2(0, 2) = 6;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2, c_epsilonFloatE4));

    matrix00.VectorMax(matrix1, matrix3, false);
    matrix2.Resize(2, 1);
    matrix2(0, 0) = 3;
    matrix2(1, 0) = 6;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2, c_epsilonFloatE4));

    matrix0.VectorNormInf(matrix3, true);
    matrix2.Resize(1, 3);
    matrix2(0, 0) = 4;
    matrix2(0, 1) = 5;
    matrix2(0, 2) = 6;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2, c_epsilonFloatE4));

    matrix0.VectorNormInf(matrix3, false);
    matrix2.Resize(2, 1);
    matrix2(0, 0) = 3;
    matrix2(1, 0) = 6;
    BOOST_CHECK(matrix3.IsEqualTo(matrix2));

    matrix00(0, 0) = 1;
    matrix00(0, 1) = 2;
    matrix00(0, 2) = 3;
    matrix00(1, 0) = 4;
    matrix00(1, 1) = 5;
    matrix00(1, 2) = 6;
    BOOST_CHECK_EQUAL(6, matrix00.MatrixNormInf());

    BOOST_CHECK(abs(matrix0.FrobeniusNorm() - 9.5394f) < c_epsilonFloatE4);
    BOOST_CHECK(abs(matrix0.MatrixNormInf() - 6) < c_epsilonFloatE4);
    BOOST_CHECK_EQUAL(21, matrix00.MatrixNorm1());

    Matrix<float> matrixA = Matrix<float>::Eye(4096, c_deviceIdZero);
    BOOST_CHECK_EQUAL(4096, matrixA.MatrixNorm0());

    Matrix<float> matrixB = Matrix<float>::Eye(5, c_deviceIdZero);
    BOOST_CHECK_EQUAL(5, matrixB.MatrixNorm0());
}

BOOST_FIXTURE_TEST_CASE(MatrixInnerProductOfMatrices, RandomSeedFixture)
{
    SingleMatrix vector1(c_value_2, c_value_3, c_deviceIdZero);
    vector1(0, 0) = 1;
    vector1(0, 1) = 2;
    vector1(0, 2) = 3;
    vector1(1, 0) = 4;
    vector1(1, 1) = 5;
    vector1(1, 2) = 6;
    SingleMatrix vector2(c_value_2, c_value_3, c_deviceIdZero);
    vector2(0, 0) = 7;
    vector2(0, 1) = 8;
    vector2(0, 2) = 9;
    vector2(1, 0) = 10;
    vector2(1, 1) = 11;
    vector2(1, 2) = 12;
    const float ip = SingleMatrix::InnerProductOfMatrices(vector1, vector2);
    BOOST_CHECK_EQUAL(217, ip);
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
