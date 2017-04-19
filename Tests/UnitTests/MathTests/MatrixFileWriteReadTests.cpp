//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../Common/Include/Basics.h"
#include "../../../Source/Math/CPUMatrix.h"
#include "../../../Source/Math/GPUMatrix.h"
#include "../../Common/Include/fileutil.h"
#include "../../Common/Include/File.h"

#include <string>

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(CPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(CPUMatrixFileWriteRead, RandomSeedFixture)
{
    CPUMatrix<float> matrixCpu = CPUMatrix<float>::RandomUniform(43, 10, -26.3f, 30.2f, IncrementCounter());
    CPUMatrix<float> matrixCpuCopy = matrixCpu;

    std::wstring fileNameCpu(L"MCPU.txt");
    File fileCpu(fileNameCpu, fileOptionsText | fileOptionsReadWrite);

    fileCpu << matrixCpu;
    fileCpu.SetPosition(0);

    CPUMatrix<float> matrixCpuRead;
    fileCpu >> matrixCpuRead;

    BOOST_CHECK(matrixCpuCopy.IsEqualTo(matrixCpuRead, c_epsilonFloatE5));
}

BOOST_FIXTURE_TEST_CASE(MatrixFileWriteRead, RandomSeedFixture)
{
    // Test Matrix in Dense mode
    Matrix<float> matrix = Matrix<float>::RandomUniform(43, 10, c_deviceIdZero, - 26.3f, 30.2f, IncrementCounter());
    Matrix<float> matrixCopy = matrix.DeepClone();

    std::wstring fileName(L"M.txt");
    File file(fileName, fileOptionsText | fileOptionsReadWrite);

    file << matrix;
    file.SetPosition(0);

    Matrix<float> matrixRead(c_deviceIdZero);
    file >> matrixRead;

    BOOST_CHECK(matrixRead.IsEqualTo(matrixCopy, c_epsilonFloatE5));

    // Test Matrix in Sparse mode
    Matrix<float> matrixSparse = Matrix<float>::RandomUniform(43, 10, c_deviceIdZero, - 26.3f, 30.2f, IncrementCounter());
    Matrix<float> matrixSparseCopy = matrixSparse.DeepClone();

    matrixSparse.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSR, true);

    std::wstring filenameSparse(L"MS.txt");
    File fileSparse(filenameSparse, fileOptionsText | fileOptionsReadWrite);

    fileSparse << matrixSparse;
    fileSparse.SetPosition(0);

    Matrix<float> matrixSparseRead(c_deviceIdZero);
    fileSparse >> matrixSparseRead;

    BOOST_CHECK(MatrixType::SPARSE == matrixSparseRead.GetMatrixType());
    BOOST_CHECK(matrixSparseRead.IsEqualTo(matrixSparseCopy, c_epsilonFloatE5));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(GPUMatrixFileWriteRead, RandomSeedFixture)
{
    GPUMatrix<float> matrixGpu = GPUMatrix<float>::RandomUniform(43, 10, c_deviceIdZero, -26.3f, 30.2f, IncrementCounter());
    GPUMatrix<float> matrixGpuCopy = matrixGpu;

    std::wstring filenameGpu(L"MGPU.txt");
    File fileGpu(filenameGpu, fileOptionsText | fileOptionsReadWrite);

    fileGpu << matrixGpu;
    fileGpu.SetPosition(0);

    GPUMatrix<float> matrixGpuRead(c_deviceIdZero);
    fileGpu >> matrixGpuRead;

    BOOST_CHECK(matrixGpuCopy.IsEqualTo(matrixGpuRead, c_epsilonFloatE5));
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
