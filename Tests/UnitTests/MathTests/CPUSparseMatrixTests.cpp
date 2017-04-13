//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#ifdef _WIN32
#include <crtdefs.h>
#endif
#include "../../../Source/Math/CPUSparseMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

typedef CPUDoubleSparseMatrix SparseMatrix;
typedef CPUDoubleMatrix DenseMatrix;

BOOST_AUTO_TEST_SUITE(CPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixColumnSlice, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;
    DenseMatrix dm0(m, n);
    SparseMatrix sm0(MatrixFormat::matrixFormatSparseCSC, m, n, 0);

    dm0.SetUniformRandomValue(-1, 1, IncrementCounter());

    foreach_coord (row, col, dm0)
    {
        sm0.SetValue(row, col, dm0(row, col));
    }

    const size_t start = 10;
    const size_t numCols = 20;
    DenseMatrix dm1 = dm0.ColumnSlice(start, numCols);
    DenseMatrix dm2 = sm0.ColumnSlice(start, numCols).CopyColumnSliceToDense(0, numCols);

    BOOST_CHECK(dm1.IsEqualTo(dm2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixCopyColumnSliceToDense, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;
    DenseMatrix dm0(m, n);
    SparseMatrix sm0(MatrixFormat::matrixFormatSparseCSC, m, n, 0);

    dm0.SetUniformRandomValue(-1, 1, IncrementCounter());

    foreach_coord (row, col, dm0)
    {
        sm0.SetValue(row, col, dm0(row, col));
    }

    const size_t start = 10;
    const size_t numCols = 20;
    DenseMatrix dm1 = dm0.ColumnSlice(start, numCols);
    DenseMatrix dm2 = sm0.CopyColumnSliceToDense(start, numCols);

    BOOST_CHECK(dm1.IsEqualTo(dm2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixMultiplyAndAdd, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;
    
    DenseMatrix dm0(m, n);
    dm0.SetUniformRandomValue(-1, 1, IncrementCounter());

    DenseMatrix dm1(m, n);
    dm1.SetUniformRandomValue(-300, 1, IncrementCounter());
    dm1.InplaceTruncateBottom(0);

    SparseMatrix sm1(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    foreach_coord(row, col, dm1)
    {
        if (dm1(row, col) != 0)
        {
            sm1.SetValue(row, col, dm1(row, col));
        }
    }

    DenseMatrix dm2(m, n);
    dm2.SetUniformRandomValue(-200, 1, IncrementCounter());
    dm2.InplaceTruncateBottom(0);

    SparseMatrix sm2(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    foreach_coord(row, col, dm2)
    {
        if (dm2(row, col) != 0)
        {
            sm2.SetValue(row, col, dm2(row, col));
        }
    }

    // generate SparseBlockCol matrix
    DenseMatrix dmMul(m, m);
    DenseMatrix::MultiplyAndAdd(dm0, false, dm1, true, dmMul);

    SparseMatrix smMul(MatrixFormat::matrixFormatSparseBlockCol, m, m, 0);
    SparseMatrix::MultiplyAndAdd(1, dm0, false, sm1, true, smMul);

    foreach_coord(row, col, dmMul)
    {
        BOOST_CHECK(abs(smMul(row, col) - dmMul(row, col)) < c_epsilonFloatE4);
    }

    SparseMatrix::MultiplyAndAdd(1, dm0, false, sm2, true, smMul);
    DenseMatrix::MultiplyAndAdd(dm0, false, dm2, true, dmMul);

    foreach_coord(row, col, dmMul)
    {
        BOOST_CHECK(abs(smMul(row, col) - dmMul(row, col)) < c_epsilonFloatE4);
    }
}

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixDoGatherColumnsOf, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;

    DenseMatrix dm(m, n);
    dm.SetUniformRandomValue(-200, 1, IncrementCounter());
    dm.InplaceTruncateBottom(0);

    SparseMatrix sm(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    foreach_coord(row, col, dm)
    {
        if (dm(row, col) != 0)
        {
            sm.SetValue(row, col, dm(row, col));
        }
    }

    std::vector<double> indexValue(n);
    for(size_t i = 0; i < n; i++) indexValue[i] = i % 3 ? (double)i : -1;
    DenseMatrix index(1, n, indexValue.data());

    SparseMatrix sm2(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    sm2.DoGatherColumnsOf(0, index, sm, 1);

    for (size_t i = 1; i < sm2.GetNumCols() + 1; i++)
    {
        BOOST_CHECK(sm2.ColLocation()[i] >= sm2.ColLocation()[i-1]);
    }
}

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixOneHot, RandomSeedFixture)
{
    const size_t num_class = 6;
    DenseMatrix m0(2, 2);
    m0(0, 0) = 1;
    m0(0, 1) = 2;
    m0(1, 0) = 3;
    m0(1, 1) = 4;
    
    vector<size_t> shape(3);
    shape[0] = num_class; shape[1] = 2; shape[2] = 2;

    SparseMatrix sm(MatrixFormat::matrixFormatSparseCSC);
    sm.AssignOneHot(m0, shape, 0);
    
    BOOST_CHECK(sm.NzCount() == 4);
    BOOST_CHECK(sm(1, 0) == 1);
    BOOST_CHECK(sm(2, 2) == 1);
    BOOST_CHECK(sm(3, 1) == 1);
    BOOST_CHECK(sm(4, 3) == 1);

    vector<size_t> shape2(3);
    shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;
    SparseMatrix sm2(MatrixFormat::matrixFormatSparseCSC);
    sm2.AssignOneHot(m0, shape2, 1);

    BOOST_CHECK(sm2.NzCount() == 4);
    BOOST_CHECK(sm2(2, 0) == 1);
    BOOST_CHECK(sm2(4, 1) == 1);
    BOOST_CHECK(sm2(7, 0) == 1);
    BOOST_CHECK(sm2(9, 1) == 1);

    DenseMatrix dirtyMatrix(2, 2);
    dirtyMatrix(0, 0) = 1;
    dirtyMatrix(0, 1) = -1;
    dirtyMatrix(1, 0) = 7;
    dirtyMatrix(1, 1) = 4;
    
    SparseMatrix sm3(MatrixFormat::matrixFormatSparseCSC);
    sm3.AssignOneHot(dirtyMatrix, shape, 0);

    BOOST_CHECK(sm3.NzCount() == 4);
    BOOST_CHECK(sm3(1, 0) == 1);
    BOOST_CHECK(sm3(0, 2) == 0);
    BOOST_CHECK(sm3(0, 1) == 0);
    BOOST_CHECK(sm3(4, 3) == 1);
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
