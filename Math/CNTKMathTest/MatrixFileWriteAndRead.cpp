//
// <copyright file="MatrixFileWriteAndRead.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include <string>
#include "CppUnitTest.h"
#include "..\Math\Matrix.h"
#include "..\..\common\include\basetypes.h"
#include "..\..\common\include\fileutil.h"
#include "..\..\common\include\file.h"
#include "..\..\common\file.cpp"
#include "..\..\common\fileutil.cpp"



#define epsilon 0.000001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace CNTKMathTest
{    
    TEST_CLASS(MatrixAndFile)
    {        

    public:

        TEST_METHOD(CPUMatrixFileWriteAndRead)
        {
            //Test CPUMatrix
            CPUMatrix<float> M = CPUMatrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
            CPUMatrix<float> Mcopy=M;
            std::wstring filename(L"MCPU.txt");            
            File file(filename,fileOptionsText|fileOptionsReadWrite);            
            file<<M;
            CPUMatrix<float> M1;
            file.SetPosition(0);
            file>>M1;            
            Assert::IsTrue(Mcopy.IsEqualTo(M1,0.00001f));
        }
        
        TEST_METHOD(GPUMatrixFileWriteAndRead)
        {
            //Test GPUMatrix
            GPUMatrix<float> MG = GPUMatrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
            GPUMatrix<float> McopyG=MG;
            std::wstring filenameGPU(L"MGPU.txt");            
            File fileGPU(filenameGPU,fileOptionsText|fileOptionsReadWrite);            
            fileGPU<<MG;
            GPUMatrix<float> M1G;
            fileGPU.SetPosition(0);
            fileGPU>>M1G;            
            Assert::IsTrue(McopyG.IsEqualTo(M1G,0.00001f));
        }

        TEST_METHOD(MatrixFileWriteAndRead)
        {
            //Test Matrix in Dense mode
            Matrix<float> M = Matrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
            Matrix<float> Mcopy=M;
            std::wstring filename(L"M.txt");            
            File file(filename,fileOptionsText|fileOptionsReadWrite);            
            file<<M;
            Matrix<float> M1;
            file.SetPosition(0);
            file>>M1; 
            //float x=
            M1(0,0);
            Assert::IsTrue(M1.IsEqualTo(Mcopy,0.00001f));

            //Test Matrix in Sparse mode
            Matrix<float> MS = Matrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
            Matrix<float> MScopy=MS;
            MS.SwitchToMatrixType(MatrixType::SPARSE);            
            std::wstring filenameS(L"MS.txt");            
            File fileS(filenameS,fileOptionsText|fileOptionsReadWrite);            
            fileS<<MS;
            Matrix<float> M1S;            
            fileS.SetPosition(0);
            fileS>>M1S; 
            Assert::IsTrue(MatrixType::SPARSE==M1S.GetMatrixType());            
            Assert::IsTrue(M1S.IsEqualTo(MScopy,0.00001f));
        }
    };
}