//
// <copyright file="CPUSparseMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Math\CPUSparseMatrix.h"
#define DEBUG_FLAG 1
using namespace Microsoft::MSR::CNTK;

#pragma warning (disable: 4305)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNTKMathTest
{    
    TEST_CLASS(CPUSparseMatrixUnitTest)
    {
        typedef CPUDoubleSparseMatrix SparseMatrix;
        typedef CPUDoubleMatrix DenseMatrix;

    public:

        TEST_METHOD(CPUSparseMatrixColumnSlice)
        {
            size_t m = 100, n = 50;
            size_t start = 10, numCols = 20;
            DenseMatrix DM0(m, n);
            SparseMatrix SM0(MatrixFormat::matrixFormatSparseCSC, m, n, 0);

            DM0.SetUniformRandomValue(-1, 1);

            foreach_coord(row, col, DM0)
            {
                auto val = DM0(row, col);
                SM0.SetValue(row, col, val);
            }

            DenseMatrix DM1 = DM0.ColumnSlice(start, numCols);
            DenseMatrix DM2 = SM0.ColumnSlice(start, numCols).CopyColumnSliceToDense(0, numCols);

            Assert::IsTrue(DM1.IsEqualTo(DM2, 0.0001));
        }

        TEST_METHOD(CPUSparseMatrixCopyColumnSliceToDense)
        {
            size_t m = 100, n = 50;
            size_t start = 10, numCols = 20;
            DenseMatrix DM0(m, n);
            SparseMatrix SM0(MatrixFormat::matrixFormatSparseCSC, m, n, 0);

            DM0.SetUniformRandomValue(-1, 1);

            foreach_coord(row, col, DM0)
            {
                auto val = DM0(row, col);
                SM0.SetValue(row, col, val);
            }

            DenseMatrix DM1 = DM0.ColumnSlice(start, numCols);
            DenseMatrix DM2 = SM0.CopyColumnSliceToDense(start, numCols);

            Assert::IsTrue(DM1.IsEqualTo(DM2, 0.0001));
        }
    };
}