//
// <copyright file="MatrixFileWriteReadTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "../../Common/Include/Basics.h"
#include "../../Math/Math/CPUMatrix.h"
#include "../../Math/Math/GPUMatrix.h"
#include "../../Common/Include/fileutil.h"
#include "../../Common/Include/File.h"
// ToDo: CPP files directly included, use common library in the future if possible 
#include "../../Common/File.cpp"
#include "../../Common/fileutil.cpp"

#include <string>

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
	namespace MSR
	{
		namespace CNTK
		{
			namespace Test
			{
				BOOST_AUTO_TEST_SUITE(CPUMatrixSuite)

				BOOST_AUTO_TEST_CASE(CPUMatrixFileWriteRead)
				{
					CPUMatrix<float> matrixCpu = CPUMatrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
					CPUMatrix<float> matrixCpuCopy = matrixCpu;
					
					std::wstring fileNameCpu(L"MCPU.txt");
					File fileCpu(fileNameCpu, fileOptionsText | fileOptionsReadWrite);

					fileCpu << matrixCpu;
					fileCpu.SetPosition(0);

					CPUMatrix<float> matrixCpuRead;
					fileCpu >> matrixCpuRead;

					BOOST_CHECK(matrixCpuCopy.IsEqualTo(matrixCpuRead, 0.00001f));
				}

				BOOST_AUTO_TEST_CASE(MatrixFileWriteRead)
				{
					//Test Matrix in Dense mode
					Matrix<float> matrix = Matrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
					Matrix<float> matrixCopy = matrix;

					std::wstring fileName(L"M.txt");
					File file(fileName, fileOptionsText | fileOptionsReadWrite);

					file << matrix;
					file.SetPosition(0);

					Matrix<float> matrixRead;
					file >> matrixRead;

					BOOST_CHECK(matrixRead.IsEqualTo(matrixCopy, 0.00001f));

					//Test Matrix in Sparse mode
					Matrix<float> matrixSparse = Matrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
					Matrix<float> matrixSparseCopy = matrixSparse;

					matrixSparse.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSR, true);

					std::wstring filenameSparse(L"MS.txt");
					File fileSparse(filenameSparse, fileOptionsText | fileOptionsReadWrite);

					fileSparse << matrixSparse;
					fileSparse.SetPosition(0);

					Matrix<float> matrixSparseRead;
					fileSparse >> matrixSparseRead;

					BOOST_CHECK(MatrixType::SPARSE == matrixSparseRead.GetMatrixType());
					BOOST_CHECK(matrixSparseRead.IsEqualTo(matrixSparseCopy, 0.00001f));
				}

				BOOST_AUTO_TEST_SUITE_END()

				BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

				BOOST_AUTO_TEST_CASE(GPUMatrixFileWriteRead)
				{
					GPUMatrix<float> matrixGpu = GPUMatrix<float>::RandomUniform(43, 10, c_deviceIdZero, -26.3f, 30.2f);
					GPUMatrix<float> matrixGpuCopy = matrixGpu;

					std::wstring filenameGpu(L"MGPU.txt");
					File fileGpu(filenameGpu, fileOptionsText | fileOptionsReadWrite);

					fileGpu << matrixGpu;
					fileGpu.SetPosition(0);

					GPUMatrix<float> matrixGpuRead(c_deviceIdZero);
					fileGpu >> matrixGpuRead;

					BOOST_CHECK(matrixGpuCopy.IsEqualTo(matrixGpuRead, 0.00001f));
				}
				
				BOOST_AUTO_TEST_SUITE_END()

			}
		}
	}
}