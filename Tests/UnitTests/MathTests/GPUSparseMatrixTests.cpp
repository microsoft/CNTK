//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// GPUSparseMatrix Unit tests should go here
//
#include "stdafx.h"
#include <math.h>
#ifdef _WIN32
#include <crtdefs.h>
#endif
#include "../../../Source/Math/GPUSparseMatrix.h"
#include "common.h"

using namespace Microsoft::MSR::CNTK;

// TODO: Move warning suppression to stdafx if necessary WI#: 83
//#pragma warning (disable: 4244 4245 4305)       // conversions and truncations; we don't care in this test project

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

const float c_v[9] = {1, 4, 2, 3, 5, 7, 8, 9, 6};
const int c_i[5] = {0, 2, 4, 7, 9};
const int c_j[9] = {0, 1, 1, 2, 0, 3, 4, 2, 4};
const int c_rowCount = 4;
const int c_colCount = 5;
const int c_size = c_rowCount + c_colCount;

BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixConstructorsAndInitializers, RandomSeedFixture)
{
    GPUSparseMatrix<float> m(c_deviceIdZero);
    BOOST_CHECK(m.IsEmpty());

    m.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);
    BOOST_CHECK_EQUAL(c_rowCount, m.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, m.GetNumCols());
    BOOST_CHECK(!m.IsEmpty());

    const GPUSparseMatrix<float> m1 = m;
    BOOST_CHECK_EQUAL(c_rowCount, m1.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, m1.GetNumCols());
    BOOST_CHECK(!m1.IsEmpty());

    const GPUSparseMatrix<float> m2(m);
    BOOST_CHECK_EQUAL(c_rowCount, m2.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, m2.GetNumCols());
    BOOST_CHECK(!m2.IsEmpty());
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixScaleAndAdd, RandomSeedFixture)
{
    const int m = 4;
    const int n = 5;    

    float a[m * n];
    float b[m * n];
    std::mt19937 rng(0);
    for (int i = 0; i < m * n; i++)
    {
        a[i] = static_cast<float>(rng());
        b[i] = static_cast<float>(rng());
    }

    const GPUMatrix<float> denseMatrixA(m, n, c_deviceIdZero, a, MatrixFlags::matrixFlagNormal);
    const GPUMatrix<float> denseMatrixB(m, n, c_deviceIdZero, b, MatrixFlags::matrixFlagNormal);

    GPUSparseMatrix<float> sparseMatrixA(c_deviceIdZero);
    sparseMatrixA.SetValue(denseMatrixA);
    GPUSparseMatrix<float> sparseMatrixB(c_deviceIdZero);
    sparseMatrixB.SetValue(denseMatrixB);

    GPUSparseMatrix<float> sparseMatrixC(c_deviceIdZero);
    const float alpha = 2;
    const float beta = 3;
    GPUSparseMatrix<float>::ScaleAndAdd(alpha, sparseMatrixA, beta, sparseMatrixB, sparseMatrixC);

    GPUSparseMatrix<float>::Scale(alpha, sparseMatrixC);

    const GPUMatrix<float> denseMatrixC = sparseMatrixC.CopyToDenseMatrix();
    unique_ptr<float[]> c(denseMatrixC.CopyToArray());
    for (int i = 0; i < m * n; i++)
    {        
        float res1 = alpha * (alpha * a[i] + beta * b[i]);
        float res2 = c[i];
        BOOST_REQUIRE_MESSAGE(AreEqual(res1, res2, Err<float>::Rel, Err<float>::Abs), 
                              "first mismatch at " << i << ", " << res1 << "!=" << res2 << ", relErr=" << (std::abs(res1 - res2) / std::max(std::abs(res1), \
                              std::abs(res2))) << ", absErr = " << std::abs(res1 - res2));
    }
}

BOOST_FIXTURE_TEST_CASE(GPUSparseDensePlusSparse, RandomSeedFixture)
{
    GPUSparseMatrix<float> sparseMatrix(c_deviceIdZero);

    sparseMatrix.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    GPUMatrix<float> denseMatrixB = GPUMatrix<float>::RandomUniform(c_rowCount, c_colCount, c_deviceIdZero, -2, 45, IncrementCounter());
    GPUMatrix<float> resultMatrix(denseMatrixB.GetNumRows(), denseMatrixB.GetNumCols(), c_deviceIdZero);

    const float alpha = 0.53f;
    const float beta = 1.0f;

    GPUSparseMatrix<float>::ScaleAndAdd(alpha, sparseMatrix, beta, denseMatrixB, resultMatrix);
    GPUMatrix<float>::ScaleAndAdd(alpha, sparseMatrix.CopyToDenseMatrix(), denseMatrixB);

    BOOST_CHECK(denseMatrixB.IsEqualTo(resultMatrix, c_epsilonFloatE5));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseElementwiseTimesDense, RandomSeedFixture)
{
    GPUSparseMatrix<float> lhs(c_deviceIdZero);
    lhs.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    const GPUMatrix<float> rhs = GPUMatrix<float>::RandomUniform(c_rowCount, c_colCount, c_deviceIdZero, -2, 45, IncrementCounter());

    const GPUMatrix<float> product = GPUSparseMatrix<float>::ElementProductOf(lhs, rhs);

    GPUMatrix<float> denseProduct(c_deviceIdZero);
    denseProduct.AssignElementProductOf(lhs.CopyToDenseMatrix(), rhs);

    BOOST_CHECK(product.IsEqualTo(denseProduct));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseTimesDense, RandomSeedFixture)
{
    GPUSparseMatrix<float> lhs(c_deviceIdZero);
    BOOST_CHECK(lhs.IsEmpty());
    lhs.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);
    BOOST_CHECK_EQUAL(c_rowCount, lhs.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, lhs.GetNumCols());
    BOOST_CHECK(!lhs.IsEmpty());

    const GPUMatrix<float> rhs = GPUMatrix<float>::Eye(c_colCount, c_deviceIdZero);
    GPUMatrix<float> result = GPUMatrix<float>::Ones(c_rowCount, c_colCount, c_deviceIdZero);

    GPUSparseMatrix<float>::MultiplyAndWeightedAdd(1, lhs, false, rhs, false, 1, result);

    float *arr = result.CopyToArray();
    CPUMatrix<float> ccpu(c_rowCount, c_colCount, arr, MatrixFlags::matrixFlagNormal);
    delete[] arr;

    BOOST_CHECK_EQUAL(1 + 1, ccpu(0, 0));
    BOOST_CHECK_EQUAL(4 + 1, ccpu(0, 1));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(0, 2));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(0, 3));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(0, 4));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(1, 0));
    BOOST_CHECK_EQUAL(2 + 1, ccpu(1, 1));
    BOOST_CHECK_EQUAL(3 + 1, ccpu(1, 2));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(1, 3));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(1, 4));
    BOOST_CHECK_EQUAL(5 + 1, ccpu(2, 0));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(2, 1));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(2, 2));
    BOOST_CHECK_EQUAL(7 + 1, ccpu(2, 3));
    BOOST_CHECK_EQUAL(8 + 1, ccpu(2, 4));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(3, 0));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(3, 1));
    BOOST_CHECK_EQUAL(9 + 1, ccpu(3, 2));
    BOOST_CHECK_EQUAL(0 + 1, ccpu(3, 3));
    BOOST_CHECK_EQUAL(6 + 1, ccpu(3, 4));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseTimesDenseRandom, RandomSeedFixture)
{
    const bool matrixTransB = false;
    for (auto sparseFormat : {MatrixFormat::matrixFormatSparseCSR, MatrixFormat::matrixFormatSparseCSC})
    {
        for (bool matrixTransA : {false, true})
        {
            for (size_t m : {1, 10, 100})
            {
                for (size_t k : {1, 10, 100})
                {
                    for (size_t n : {1, 10, 100})
                    {
                        size_t dim1 = m, dim2 = k, dim3 = k, dim4 = n;
                        if (matrixTransA)
                        {
                            dim1 = k;
                            dim2 = m;
                        }
                        // SPARSE
                        GPUMatrix<float> lhsDense(c_deviceIdZero);
                        lhsDense.AssignTruncateBottomOf(GPUMatrix<float>::RandomUniform(dim1, dim2, c_deviceIdZero, -3.0f, 1.0f, IncrementCounter()), 0);
                        GPUSparseMatrix<float> lhs(lhsDense, sparseFormat);

                        // DENSE
                        GPUMatrix<float> rhs = GPUMatrix<float>::RandomUniform(dim3, dim4, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());

                        // RESULT of Dense x Sparse
                        GPUMatrix<float> resultSP = GPUMatrix<float>::Ones(m, n, c_deviceIdZero);
                        GPUSparseMatrix<float>::MultiplyAndWeightedAdd(1, lhs, matrixTransA, rhs, matrixTransB, 1, resultSP);

                        // RESULT of Dense x Dense
                        GPUMatrix<float> resultDS = GPUMatrix<float>::Ones(m, n, c_deviceIdZero);
                        GPUMatrix<float>::MultiplyAndWeightedAdd(1, lhsDense, matrixTransA, rhs, matrixTransB, 1, resultDS);

                        float *arrSP = resultSP.CopyToArray();
                        CPUMatrix<float> resultSP_arr(m, n, arrSP, MatrixFlags::matrixFlagNormal);
                        delete[] arrSP;

                        float *arrDS = resultDS.CopyToArray();
                        CPUMatrix<float> resultDS_arr(m, n, arrDS, MatrixFlags::matrixFlagNormal);
                        delete[] arrDS;

                        // Check correctness with CPU results
                        for (int i = 0; i < m; i++) {
                            for (int j = 0; j < n; j++) {
                                BOOST_CHECK(fabs(resultSP_arr(i, j) - resultDS_arr(i, j)) < c_epsilonFloatE4);
                            }
                        }
                    }
                }
            }
        }
    }
}

#if 0
// TODO commented temporarily, this test (or underlying code) needs fixes

BOOST_FIXTURE_TEST_CASE(GPUDenseTimesSparse, RandomSeedFixture)
{
    GPUSparseMatrix<float> sparseMatrixA(matrixFormatSparseCSR, 0);
    BOOST_CHECK(sparseMatrixA.IsEmpty());

    sparseMatrixA.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);
    BOOST_CHECK_EQUAL(c_rowCount, sparseMatrixA.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, sparseMatrixA.GetNumCols());
    BOOST_CHECK(!sparseMatrixA.IsEmpty());

    const GPUSparseMatrix<float> sparseTransposeA = sparseMatrixA.Transpose();
    const GPUMatrix<float> denseTransposeA = sparseTransposeA.CopyToDenseMatrix();

    float arrA_times_transposeA[19] = { 17, 8, 5, 0, 8, 13, 0, 27, 5, 0, 138, 48, 0, 27, 48, 117 };

    const GPUMatrix<float> cet(c_rowCount, c_rowCount, c_deviceIdZero, arrA_times_transposeA, MatrixFlags::matrixFlagNormal);
    GPUMatrix<float> cres(c_rowCount, c_rowCount, c_deviceIdZero);
    GPUSparseMatrix<float>::Multiply(sparseMatrixA, denseTransposeA, cres);
    BOOST_CHECK(cres.IsEqualTo(cet));

    float arrAT_times_matrixA[25] = { 26, 4, 0, 35, 40, 4, 20, 6, 0, 0, 0, 6, 90, 0, 54, 35, 0, 0, 49, 56, 40, 0, 54, 56, 100 };
    const GPUMatrix<float> cet1(c_colCount, c_colCount, c_deviceIdZero, arrAT_times_matrixA, MatrixFlags::matrixFlagNormal);
    GPUMatrix<float> cres1(c_colCount, c_colCount, c_deviceIdZero);
    GPUSparseMatrix<float>::Multiply(denseTransposeA, sparseMatrixA, cres1);
    BOOST_CHECK(cres1.IsEqualTo(cet1));

    const GPUMatrix<float> matrixB = GPUMatrix<float>::RandomUniform(c_rowCount + c_colCount, c_rowCount, c_deviceIdZero, -100, 100, IncrementCounter());
    GPUMatrix<float> matrixC(c_size, c_colCount, c_deviceIdZero);
    GPUSparseMatrix<float>::Multiply(matrixB, sparseMatrixA, matrixC); // C=BA

    const GPUMatrix<float> transposeB = matrixB.Transpose();
    const GPUSparseMatrix<float> transposeA = sparseMatrixA.Transpose();
    GPUMatrix<float> transposeC(c_colCount, c_size, c_deviceIdZero);
    GPUSparseMatrix<float>::Multiply(transposeA, transposeB, transposeC); // CT = AT*BT = (BA)T
    const GPUMatrix<float> twiceTransposeC = transposeC.Transpose(); // CTT = C;

    BOOST_CHECK(twiceTransposeC.IsEqualTo(matrixC, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseTimesSparse, RandomSeedFixture)
{
    GPUSparseMatrix<float> matrixA;
    BOOST_CHECK(matrixA.IsEmpty());
    matrixA.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    GPUSparseMatrix<float> transposeA = matrixA.Transpose();

    GPUSparseMatrix<float> product;
    GPUSparseMatrix<float>::Multiply(transposeA, false, matrixA, false, product);

    const float expectedProduct[25] = { 26, 4, 0, 35, 40, 4, 20, 6, 0, 0, 0, 6, 90, 0, 54, 35, 0, 0, 49, 56, 40, 0, 54, 56, 100 };  // A * AT
    const float * const arr = product.CopyToDenseMatrix().CopyToArray();
    BOOST_CHECK_EQUAL_COLLECTIONS(&expectedProduct[0], &expectedProduct[25], &arr[0], &arr[25]);

    delete[] arr;
}

#endif

BOOST_FIXTURE_TEST_CASE(GPUSparseElementWise, RandomSeedFixture)
{
    GPUSparseMatrix<float> matrixA(c_deviceIdZero);
    BOOST_CHECK(matrixA.IsEmpty());
    matrixA.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    GPUSparseMatrix<float> poweredMatrix(c_deviceIdZero);
    const float somePower = 3.14f;
    poweredMatrix.ResizeAsAndCopyIndexFrom(matrixA);
    matrixA.ElementWisePower(somePower, matrixA, poweredMatrix);

    float *arr = nullptr;
    int *ii = nullptr;
    int *jj = nullptr;
    size_t ea, nz, nr, nc;
    poweredMatrix.GetMatrixFromCSRFormat(ii, jj, arr, ea, nz, nr, nc);

    for (int index = 0; index < c_size; ++index)
    {
        float y = powf(c_v[index], somePower);
        BOOST_CHECK(fabsf(y - arr[index]) < c_epsilonFloatE3);
    }

    delete[] arr;
    delete[] ii;
    delete[] jj;
}

BOOST_FIXTURE_TEST_CASE(GPUSparseIsEqual, RandomSeedFixture)
{
    GPUSparseMatrix<float> firstMatrix(c_deviceIdZero);
    firstMatrix.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    GPUSparseMatrix<float> secondMatrix(c_deviceIdZero);
    secondMatrix.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);
    BOOST_CHECK(secondMatrix.IsEqualTo(firstMatrix));

    GPUSparseMatrix<float> emptyMatrix(c_deviceIdZero);
    BOOST_CHECK(!emptyMatrix.IsEqualTo(firstMatrix));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseDenseConversions, RandomSeedFixture)
{
    GPUSparseMatrix<float> sparseMatrixA(c_deviceIdZero);
    BOOST_CHECK(sparseMatrixA.IsEmpty());
    sparseMatrixA.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);
    BOOST_CHECK_EQUAL(c_rowCount, sparseMatrixA.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, sparseMatrixA.GetNumCols());
    BOOST_CHECK(!sparseMatrixA.IsEmpty());

    const GPUMatrix<float> denseMatrixA = sparseMatrixA.CopyToDenseMatrix();
    BOOST_CHECK_EQUAL(c_rowCount, denseMatrixA.GetNumRows());
    BOOST_CHECK_EQUAL(c_colCount, denseMatrixA.GetNumCols());

    float *arr = denseMatrixA.CopyToArray();
    CPUMatrix<float> cpuMatrix(denseMatrixA.GetNumRows(), denseMatrixA.GetNumCols(), arr, MatrixFlags::matrixFlagNormal);
    delete[] arr;

    BOOST_CHECK_EQUAL(1, cpuMatrix(0, 0));
    BOOST_CHECK_EQUAL(4, cpuMatrix(0, 1));
    BOOST_CHECK_EQUAL(0, cpuMatrix(0, 2));
    BOOST_CHECK_EQUAL(0, cpuMatrix(0, 3));
    BOOST_CHECK_EQUAL(0, cpuMatrix(0, 4));
    BOOST_CHECK_EQUAL(5, cpuMatrix(2, 0));
    BOOST_CHECK_EQUAL(0, cpuMatrix(2, 1));
    BOOST_CHECK_EQUAL(0, cpuMatrix(2, 2));
    BOOST_CHECK_EQUAL(7, cpuMatrix(2, 3));
    BOOST_CHECK_EQUAL(8, cpuMatrix(2, 4));

    GPUSparseMatrix<float> sparseMatrixB(c_deviceIdZero);
    sparseMatrixB.SetValue(denseMatrixA);
    const GPUMatrix<float> denseMatrixB = sparseMatrixB.CopyToDenseMatrix();
    arr = denseMatrixB.CopyToArray();
    const CPUMatrix<float> cpuMatrixB(denseMatrixB.GetNumRows(), denseMatrixB.GetNumCols(), arr, MatrixFlags::matrixFlagNormal);
    delete[] arr;

    BOOST_CHECK_EQUAL(1, cpuMatrixB(0, 0));
    BOOST_CHECK_EQUAL(4, cpuMatrixB(0, 1));
    BOOST_CHECK_EQUAL(0, cpuMatrixB(0, 2));
    BOOST_CHECK_EQUAL(0, cpuMatrixB(0, 3));
    BOOST_CHECK_EQUAL(0, cpuMatrixB(0, 4));
    BOOST_CHECK_EQUAL(5, cpuMatrixB(2, 0));
    BOOST_CHECK_EQUAL(0, cpuMatrixB(2, 1));
    BOOST_CHECK_EQUAL(0, cpuMatrixB(2, 2));
    BOOST_CHECK_EQUAL(7, cpuMatrixB(2, 3));
    BOOST_CHECK_EQUAL(8, cpuMatrixB(2, 4));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseTranspose, RandomSeedFixture)
{
    GPUSparseMatrix<float> sparseMatrix(c_deviceIdZero);
    BOOST_CHECK(sparseMatrix.IsEmpty());
    sparseMatrix.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    const GPUSparseMatrix<float> transposeMatrixA = sparseMatrix.Transpose();
    GPUSparseMatrix<float> inplaceTranposeMatrix(sparseMatrix);
    inplaceTranposeMatrix.InplaceTranspose();
    BOOST_CHECK(inplaceTranposeMatrix.IsEqualTo(transposeMatrixA));

    const GPUSparseMatrix<float> tranposeMatrixB = sparseMatrix.Transpose();
    GPUSparseMatrix<float> assignedTransposeMatrix(c_deviceIdZero);
    assignedTransposeMatrix.AssignTransposeOf(tranposeMatrixB);
    BOOST_CHECK(assignedTransposeMatrix.IsEqualTo(sparseMatrix));

    sparseMatrix.InplaceTranspose();
    BOOST_CHECK(!assignedTransposeMatrix.IsEqualTo(sparseMatrix));

    sparseMatrix.InplaceTranspose();
    BOOST_CHECK(assignedTransposeMatrix.IsEqualTo(sparseMatrix));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseNormTests, RandomSeedFixture)
{
    GPUSparseMatrix<float> matrix(c_deviceIdZero);
    BOOST_CHECK(matrix.IsEmpty());
    matrix.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);

    const float frobenius = matrix.FrobeniusNorm();
    BOOST_CHECK(fabsf(16.882f - frobenius) < c_epsilonFloatE4);

    const float ninf = matrix.MatrixNormInf();
    BOOST_CHECK_EQUAL(9, ninf);

    const float n1 = matrix.MatrixNorm1();
    BOOST_CHECK_EQUAL(45, n1);
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixInnerProduct, RandomSeedFixture)
{
    GPUSparseMatrix<float> matrixOp1(c_deviceIdZero);
    BOOST_CHECK(matrixOp1.IsEmpty());
    matrixOp1.SetMatrixFromCSRFormat(c_i, c_j, c_v, c_size, c_rowCount, c_colCount);
    GPUMatrix<float> matrixOp1Dense = matrixOp1.CopyToDenseMatrix();

    const GPUMatrix<float> matrixOp2(GPUMatrix<float>::RandomUniform(c_rowCount, c_colCount, c_deviceIdZero, -3, 4, IncrementCounter()));
    const float x = GPUSparseMatrix<float>::InnerProductOfMatrices(matrixOp1, matrixOp2);
    const float y = GPUMatrix<float>::InnerProductOfMatrices(matrixOp1Dense, matrixOp2);
    BOOST_CHECK(fabsf(x - y) < c_epsilonFloatE5);

    GPUSparseMatrix<float> matrixOp1CSC(matrixOp1Dense, matrixFormatSparseCSC);
    const float x1 = GPUSparseMatrix<float>::InnerProductOfMatrices(matrixOp1CSC, matrixOp2);

    BOOST_CHECK(fabsf(x1 - y) < c_epsilonFloatE5);

    for(bool isColWise : {true, false})
    {
        GPUMatrix<float> matrixInnerProductDense(c_deviceIdZero);
        GPUMatrix<float> matrixInnerProductSparse(c_deviceIdZero);

        GPUMatrix<float>::InnerProduct(matrixOp1Dense, matrixOp2, matrixInnerProductDense, isColWise);
        GPUSparseMatrix<float>::InnerProduct(matrixOp1CSC, matrixOp2, matrixInnerProductSparse, isColWise);

        BOOST_CHECK(matrixInnerProductDense.IsEqualTo(matrixInnerProductSparse, c_epsilonFloatE5));
    }
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixColumnSlice, RandomSeedFixture)
{
    float values[6] = {1, 4, 2, 5, 3, 6};
    GPUMatrix<float> denseA(2, 3, c_deviceIdZero, values, MatrixFlags::matrixFlagNormal);
    GPUSparseMatrix<float> sparseA(c_deviceIdZero, MatrixFormat::matrixFormatSparseCSC);
    sparseA.SetValue(denseA);

    const GPUMatrix<float> sliceColumn1 = denseA.ColumnSlice(0, 2);
    const GPUMatrix<float> denseColumn1 = sparseA.ColumnSlice(0, 2).CopyColumnSliceToDense(0, 2);
    BOOST_CHECK(sliceColumn1.IsEqualTo(denseColumn1, c_epsilonFloatE4));

    const GPUMatrix<float> sliceColumn2 = denseA.ColumnSlice(1, 2);
    const GPUMatrix<float> sparseColumn2 = sparseA.ColumnSlice(1, 2).CopyColumnSliceToDense(0, 2);
    BOOST_CHECK(sliceColumn2.IsEqualTo(sparseColumn2, c_epsilonFloatE4));

    BOOST_CHECK(!sliceColumn1.IsEqualTo(sparseColumn2, c_epsilonFloatE4));
    BOOST_CHECK(!sliceColumn2.IsEqualTo(denseColumn1, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixCopyColumnSliceToDense, RandomSeedFixture)
{
    float values[6] = {1, 4, 2, 5, 3, 6};
    GPUMatrix<float> denseA(2, 3, c_deviceIdZero, values, MatrixFlags::matrixFlagNormal);
    GPUSparseMatrix<float> sparseA(c_deviceIdZero, MatrixFormat::matrixFormatSparseCSC);
    sparseA.SetValue(denseA);

    const GPUMatrix<float> sliceColumn1 = denseA.ColumnSlice(0, 2);
    const GPUMatrix<float> denseColumn1 = sparseA.CopyColumnSliceToDense(0, 2);
    BOOST_CHECK(sliceColumn1.IsEqualTo(denseColumn1, c_epsilonFloatE4));

    const GPUMatrix<float> sliceColumn2 = denseA.ColumnSlice(1, 2);
    const GPUMatrix<float> sparseColumn2 = sparseA.CopyColumnSliceToDense(1, 2);
    BOOST_CHECK(sliceColumn2.IsEqualTo(sparseColumn2, c_epsilonFloatE4));

    BOOST_CHECK(!sliceColumn1.IsEqualTo(sparseColumn2, c_epsilonFloatE4));
    BOOST_CHECK(!sliceColumn2.IsEqualTo(denseColumn1, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrix1DConvolutionFixedInit, RandomSeedFixture)
{
    const bool zeroPadding = false;
    const int horizontalSubsample = 1;
    const int verticalSubsample = 1;
    const int inputWidth = 6;
    const int inputHeight = 1;
    const int inputChannels = 1;
    const int kernelWidth = 2;
    const int kernelHeight = inputHeight;
    const int outputWidth = zeroPadding ? inputWidth : (inputWidth >= kernelWidth ? 1 + (inputWidth - kernelWidth) / horizontalSubsample : 0);
    const int outputHeight = inputHeight;
    const int outputChannels = 3;
    const int batchSize = 1;
    const int m = outputChannels;
    const int k = kernelWidth * inputChannels;
    const int l = inputWidth * inputChannels;
    const int n = batchSize;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float weights[9] = {1, 1, 1, 1, 1, 1};
    float values[9] = {1, 2, 3, 4, 5, 6};
    GPUMatrix<float> denseMatrixA(m, k, 0, weights, matrixFlagNormal);
    ;
    GPUMatrix<float> denseMatrixB(l, n, 0, values, matrixFlagNormal);
    ;
    GPUSparseMatrix<float> sparseMatrixB(c_deviceIdZero, matrixFormatSparseCSC);
    GPUMatrix<float> denseMatrixTemp(1, 1, c_deviceIdZero);  // this should get resized automatically
    GPUMatrix<float> resultMatrixBase(1, 1, c_deviceIdZero); // this should get resized automatically
    GPUMatrix<float> resultMatrixExp(1, 1, c_deviceIdZero);  // this should get resized automatically

    sparseMatrixB.SetValue(denseMatrixB);

    denseMatrixTemp.AssignPackedConvolutionInput(denseMatrixB,
                                                 inputWidth, inputHeight, inputChannels,
                                                 outputWidth, outputHeight, outputChannels,
                                                 kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);

    GPUMatrix<float>::MultiplyAndWeightedAdd(alpha, denseMatrixA, false, denseMatrixTemp, false, beta, resultMatrixBase);
    resultMatrixBase.Reshape(outputWidth * outputChannels, batchSize);

    GPUSparseMatrix<float>::ConvolveAndWeightedAdd(alpha, denseMatrixA, false, sparseMatrixB, false, beta, resultMatrixExp, inputChannels, horizontalSubsample, zeroPadding, true);

    BOOST_CHECK(resultMatrixExp.IsEqualTo(resultMatrixBase, c_epsilonFloatE5));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrix1DConvolutionRandomInit, RandomSeedFixture)
{
    for (auto transposeA : {false, true})
    {
        for (auto transposeB : {false, true})
        {
            for (auto zeroPadding : {false}) // TODO: There's a discrepancy w.r.t. padding - needs to be fixed!!!
            {
                for (auto inputChannels : {1, 10})
                {
                    for (auto horizontalSubsample : {1, 2, 3})
                    {
                        const int verticalSubsample = 1;
                        const int inputWidth = 10;
                        const int inputHeight = 1;
                        const int kernelWidth = 3;
                        const int kernelHeight = inputHeight;
                        const int outputWidth = zeroPadding ? inputWidth : (inputWidth >= kernelWidth ? 1 + (inputWidth - kernelWidth) / horizontalSubsample : 0);
                        const int outputHeight = inputHeight;
                        const int outputChannels = 100;
                        const int batchSize = 10;
                        const int m = outputChannels;
                        const int k = kernelWidth * inputChannels;
                        const int l = inputWidth * inputChannels;
                        const int n = batchSize;
                        const float alpha = 0.53f;
                        const float beta = transposeB ? 1.0f : 0.0f;
                        GPUMatrix<float> denseMatrixA = GPUMatrix<float>::RandomUniform(m, k, c_deviceIdZero, -1, 1, IncrementCounter());
                        GPUMatrix<float> denseMatrixB = GPUMatrix<float>::RandomUniform(l, n, c_deviceIdZero, -5, 5, IncrementCounter());
                        GPUMatrix<float> denseMatrixAT(k, m, c_deviceIdZero);
                        GPUMatrix<float> denseMatrixBT(l, n, c_deviceIdZero);
                        GPUSparseMatrix<float> sparseMatrixB(c_deviceIdZero, matrixFormatSparseCSC);
                        GPUMatrix<float> denseMatrixTemp(1, 1, c_deviceIdZero); // this should get resized automatically
                        GPUMatrix<float> resultMatrixBase(outputChannels, batchSize * outputWidth * outputHeight, c_deviceIdZero);
                        GPUMatrix<float> resultMatrixExp(outputWidth * outputHeight * outputChannels, batchSize, c_deviceIdZero);

                        if (transposeA)
                        {
                            denseMatrixAT.AssignTransposeOf(denseMatrixA);
                        }

                        if (transposeB)
                        {
                            denseMatrixBT.AssignTransposeOf(denseMatrixB);
                            sparseMatrixB.SetValue(denseMatrixBT);
                        }
                        else
                        {
                            sparseMatrixB.SetValue(denseMatrixB);
                        }

                        denseMatrixTemp.AssignPackedConvolutionInput(denseMatrixB,
                                                                     inputWidth, inputHeight, inputChannels,
                                                                     outputWidth, outputHeight, outputChannels,
                                                                     kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);

                        GPUMatrix<float>::MultiplyAndWeightedAdd(alpha, denseMatrixA, false, denseMatrixTemp, false, beta, resultMatrixBase);
                        resultMatrixBase.Reshape(outputWidth * outputChannels, batchSize);

                        GPUSparseMatrix<float>::ConvolveAndWeightedAdd(alpha, (transposeA ? denseMatrixAT : denseMatrixA), transposeA, sparseMatrixB, transposeB, beta, resultMatrixExp, inputChannels, horizontalSubsample, zeroPadding, true);

                        BOOST_CHECK(resultMatrixExp.IsEqualTo(resultMatrixBase, c_epsilonFloatE2));
                    }
                }
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrix1DConvolutionBackprop, RandomSeedFixture)
{
    const int inChannels = 50;
    const int inWidth = 10;
    const int inHeight = 1;
    const int batchSize = 20;
    const int kernelWidth = 3;
    const int kernelHeight = inHeight;
    const int horizontalSubsample = 1;
    const int verticalSubsample = 1;
    const bool zeroPadding = false;
    const int outChannels = 3;
    const int outWidth = zeroPadding ? (inWidth / horizontalSubsample) : (inWidth >= kernelWidth ? 1 + (inWidth - kernelWidth) / horizontalSubsample : 0);
    const int outHeight = inHeight;
    const float randomInitLowerBound = -1.0f;
    const float randomInitUpperBound = 1.0f;
    Matrix<float> outputGradientSubBatch = Matrix<float>::RandomUniform(outChannels, batchSize * outWidth, c_deviceIdZero, randomInitLowerBound, randomInitUpperBound, IncrementCounter());
    Matrix<float> inputSubBatch = Matrix<float>::RandomUniform(inChannels * inWidth, batchSize, c_deviceIdZero, randomInitLowerBound, randomInitUpperBound, IncrementCounter());
    Matrix<float> tempMatrix(1, 1, c_deviceIdZero);
    Matrix<float> inputGradientValues1 = Matrix<float>::Zeros(outChannels, inChannels * kernelWidth, c_deviceIdZero);
    Matrix<float> inputGradientValues2 = Matrix<float>::Zeros(outChannels, inChannels * kernelWidth, c_deviceIdZero);

    // Baseline
    tempMatrix.Resize(kernelWidth * kernelHeight * inChannels, outWidth * outHeight * batchSize);
    tempMatrix.AssignPackedConvolutionInput(inputSubBatch,
                                            inWidth, inHeight, inChannels, outWidth, outHeight, outChannels,
                                            kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);
    Matrix<float>::MultiplyAndAdd(outputGradientSubBatch, false, tempMatrix, true, inputGradientValues1);

    // Optimized code path for 1-D convolution on GPU + Sparse
    Matrix<float> inputSubBatchSparseReordered(batchSize * inWidth, inChannels, c_deviceIdZero, MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC);
    inputSubBatch.SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, true);
    inputSubBatch.Reshape(inChannels, batchSize * inWidth);
    inputSubBatch.InplaceTranspose();
    Matrix<float>::TensorShuffleScaleAndAdd(0.0f, inputSubBatch, 1, inWidth, 1, batchSize, inChannels, 1.0f, inputSubBatchSparseReordered, inputSubBatchSparseReordered);

    Matrix<float> outputGradientSubBatchReshaped = Matrix<float>::Zeros(outChannels, batchSize * outWidth, c_deviceIdZero);
    Matrix<float> outputGradientSubBatchReordered = Matrix<float>::Zeros(batchSize * outWidth, outChannels, c_deviceIdZero);
    outputGradientSubBatchReshaped = outputGradientSubBatch.Transpose();
    Matrix<float>::TensorShuffleScaleAndAdd(0.0f, outputGradientSubBatchReshaped, 1, outWidth, 1, batchSize, outChannels, 1.0f, outputGradientSubBatchReordered, outputGradientSubBatchReordered);

    inputGradientValues2.Reshape(outChannels * kernelWidth, inChannels);
    Matrix<float>::ConvolveAndWeightedAdd(1, outputGradientSubBatchReordered, true, inputSubBatchSparseReordered, false, 1, inputGradientValues2, batchSize, horizontalSubsample, zeroPadding, false);
    inputGradientValues2.Reshape(outChannels, inChannels * kernelWidth);

    BOOST_CHECK(inputGradientValues2.IsEqualTo(inputGradientValues1, c_epsilonFloatE2));
}

BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixReshape, RandomSeedFixture)
{
    const int oldRowCount = 10;
    const int oldColCount = 3;
    const int newRowCount = 5;
    const int newColCount = 6;
    GPUMatrix<float> denseMatrixA = GPUMatrix<float>::RandomUniform(oldRowCount, oldColCount, c_deviceIdZero, -1, 1, IncrementCounter());
    GPUMatrix<float> denseMatrixB(denseMatrixA);
    GPUMatrix<float> denseMatrixC(newRowCount, newColCount, c_deviceIdZero);
    GPUSparseMatrix<float> sparseMatrix(c_deviceIdZero, matrixFormatSparseCSC);

    denseMatrixB.Reshape(newRowCount, newColCount);

    sparseMatrix.SetValue(denseMatrixA);
    sparseMatrix.Reshape(newRowCount, newColCount);
    sparseMatrix.CopyToDenseMatrix(denseMatrixC);

    BOOST_CHECK(denseMatrixC.IsEqualTo(denseMatrixB, c_epsilonFloatE5));
    BOOST_CHECK(!denseMatrixC.IsEqualTo(denseMatrixA, c_epsilonFloatE5));
    BOOST_CHECK(sparseMatrix.IsValid());
}

BOOST_FIXTURE_TEST_CASE(GPUSparseTensorShuffleScaleAndAdd, RandomSeedFixture)
{
    size_t D = 13, S = 11, M = 7, K = 15, T = 8;
    GPUMatrix<float> denseMatrixA = GPUMatrix<float>::RandomUniform(D * S * M * K, T, c_deviceIdZero, -1, 1, IncrementCounter());
    GPUMatrix<float> denseMatrixB(D * S * M * K, T, c_deviceIdZero);
    GPUMatrix<float> denseMatrixC(D * S * M * K, T, c_deviceIdZero);
    GPUSparseMatrix<float> sparseMatrixA(c_deviceIdZero, matrixFormatSparseCSC);
    GPUSparseMatrix<float> sparseMatrixB(c_deviceIdZero, matrixFormatSparseCSC);
    sparseMatrixA.SetValue(denseMatrixA);

    GPUMatrix<float>::TensorShuffleScaleAndAdd(0, denseMatrixA, D, S, M, K, T, 1, denseMatrixB, denseMatrixB);
    GPUSparseMatrix<float>::TensorShuffleScaleAndAdd(0, sparseMatrixA, D, S, M, K, T, 1, sparseMatrixB, sparseMatrixB);
    sparseMatrixB.CopyToDenseMatrix(denseMatrixC);

    BOOST_CHECK(denseMatrixC.IsEqualTo(denseMatrixB, c_epsilonFloatE5));
    BOOST_CHECK(sparseMatrixB.IsValid());
}


BOOST_FIXTURE_TEST_CASE(GPUSparseMatrixOneHot, RandomSeedFixture)
{
    GPUSparseMatrix<double> result(c_deviceIdZero, matrixFormatSparseCSC);
    const size_t num_class = 6;

    double data[4] = { 1,2,3,4 };
    GPUMatrix<double> m0(2, 2, c_deviceIdZero);
    m0.SetValue(2, 2, c_deviceIdZero, data, matrixFormatRowMajor);

    double exp_data[24];
    memset(&exp_data[0], 0, sizeof(double) * 24);
    exp_data[1] = exp_data[9] = exp_data[14] = exp_data[22] = 1;
    GPUMatrix<double> exp(6, 4, c_deviceIdZero);
    exp.SetValue(6, 4, c_deviceIdZero, exp_data, matrixFormatColMajor);

    GPUSparseMatrix<double> exp_sparse(c_deviceIdZero, matrixFormatSparseCSC);
    exp_sparse.SetValue(exp);

    vector<size_t> shape(3);
    shape[0] = num_class; shape[1] = 2; shape[2] = 2;
    result.AssignOneHot(m0, shape, 0);

    BOOST_CHECK(result.IsValid());
    BOOST_CHECK(result.IsEqualTo(exp_sparse, 1e-6));

    vector<size_t> shape2(3);
    shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;

    GPUMatrix<double> exp2(12, 2, c_deviceIdZero);
    exp2.AssignOneHot(m0, shape2, 1);

    GPUSparseMatrix<double> result2(c_deviceIdZero, matrixFormatSparseCSC);
    result2.AssignOneHot(m0, shape2, 1);

    BOOST_CHECK(result2.IsValid());
    BOOST_CHECK(result2.IsEqualTo(exp2, 1e-6));

    double dirty_data[4] = { 1,-1,7,4 };
    GPUMatrix<double> dirty_m(2, 2, c_deviceIdZero);
    dirty_m.SetValue(2, 2, c_deviceIdZero, dirty_data, matrixFormatRowMajor);

    double dirty_exp_data[24];
    memset(&dirty_exp_data[0], 0, sizeof(double) * 24);
    dirty_exp_data[1] = dirty_exp_data[22] = 1;
    GPUMatrix<double> dirty_exp(6, 4, c_deviceIdZero);
    dirty_exp.SetValue(6, 4, c_deviceIdZero, dirty_exp_data, matrixFormatColMajor);

    GPUSparseMatrix<double> dirty_result(c_deviceIdZero, matrixFormatSparseCSC);
    dirty_result.AssignOneHot(dirty_m, shape, 0);

    GPUMatrix<double> dirty_dense = dirty_result.CopyToDenseMatrix();

    BOOST_CHECK(dirty_result.IsValid());
    BOOST_CHECK(dirty_dense.IsEqualTo(dirty_exp, 1e-6));

    double data2[2] = { 3,0 };
    GPUMatrix<double> m3(1, 2, c_deviceIdZero);
    m3.SetValue(1, 2, c_deviceIdZero, data2, matrixFormatRowMajor);

    double exp_data2[12];
    memset(&exp_data2[0], 0, sizeof(double) * 12);
    exp_data2[3] = exp_data2[6] = 1;
    GPUMatrix<double> exp_m(6, 2, c_deviceIdZero);
    exp_m.SetValue(6, 2, c_deviceIdZero, exp_data2, matrixFormatColMajor);

    GPUSparseMatrix<double> exp_m_sparse(c_deviceIdZero, matrixFormatSparseCSC);
    exp_m_sparse.SetValue(exp_m);

    vector<size_t> shape3(3);
    shape3[0] = num_class; shape3[1] = 2; shape3[2] = 1;

    GPUSparseMatrix<double> result_m(c_deviceIdZero, matrixFormatSparseCSC);
    result_m.AssignOneHot(m3, shape3, 0);

    BOOST_CHECK(result_m.IsValid());
    BOOST_CHECK(result_m.IsEqualTo(exp_m_sparse, 1e-6));
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
