//
// <copyright file="MatrixDataSynchronizationTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Math\Matrix.h"
#include "..\Math\CPUMatrix.h"
#include "..\Math\GPUMatrix.h"
#include "..\Math\CPUSparseMatrix.h"
#include "..\Math\GPUSparseMatrix.h"
#include "..\Math\Helpers.h"

#define epsilon 0.000001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace CNTKMathTest
{    
    TEST_CLASS(MatrixDataSynchronizationTests)
    {        

    public:

        //This tests defaulting to GPU. Your machine has to have CUDA5 GPU to pass it
        TEST_METHOD(MatrixDataSynchronization_DefaultBehaiviourTestForConstructors)
        {
            SingleMatrix A;
            Assert::IsTrue(CurrentDataLocation::GPU==A.GetCurrentMatrixLocation());            
            Assert::AreEqual<size_t>(0,A.GetNumCols());
            Assert::AreEqual<size_t>(0,A.GetNumRows());
            SingleMatrix A1(AUTOPLACEMATRIX);
            Assert::IsTrue(CurrentDataLocation::GPU==A1.GetCurrentMatrixLocation());            
            Assert::AreEqual<size_t>(0,A1.GetNumCols());
            Assert::AreEqual<size_t>(0,A1.GetNumRows());
            SingleMatrix A2((size_t)13,(size_t)12);
            Assert::IsTrue(CurrentDataLocation::GPU==A2.GetCurrentMatrixLocation());    
            Assert::AreEqual<size_t>(12,A2.GetNumCols());
            Assert::AreEqual<size_t>(13,A2.GetNumRows());
            float* arr = new float[225];
            SingleMatrix A3(5,45,arr,matrixFlagNormal);
            Assert::IsTrue(CurrentDataLocation::GPU==A3.GetCurrentMatrixLocation());
            Assert::AreEqual<size_t>(45,A3.GetNumCols());
            Assert::AreEqual<size_t>(5,A3.GetNumRows());
        }

        //This tests defaulting to GPU and transfering. Your machine has to have CUDA5 GPU to pass it
        TEST_METHOD(MatrixDataSynchronization_AceessPatternAndTransferTest)
        {
            float* arr = new float[225];
            const SingleMatrix A(5,45,arr,matrixFlagNormal);
            Assert::IsTrue(CurrentDataLocation::GPU==A.GetCurrentMatrixLocation());
            Assert::AreEqual<size_t>(45,A.GetNumCols());
            Assert::AreEqual<size_t>(5,A.GetNumRows());

            float x=A(1,1);            
            Assert::IsTrue(A.GetCurrentMatrixLocation()!=CurrentDataLocation::GPU);
            foreach_coord(i,j,A)
            {
                x=A(i,j);
                Assert::IsTrue(A.GetCurrentMatrixLocation()!=CurrentDataLocation::GPU);
            }

            SingleMatrix B(15,15,arr,matrixFlagNormal);
            Assert::IsTrue(CurrentDataLocation::GPU==B.GetCurrentMatrixLocation());
            Assert::AreEqual<size_t>(15,B.GetNumCols());
            Assert::AreEqual<size_t>(15,B.GetNumRows());

            x=B(1,1);            
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);
            B(4,2)=x;
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);
            foreach_coord(i,j,B)
            {
                x=B(i,j);
                B(j,i)=x;
                Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);
            }
            
            B.TransferFromDeviceToDevice(-1,0,false);
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::BOTH);
            B.TransferFromDeviceToDevice(0,-1,false);
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::BOTH);
            B.TransferFromDeviceToDevice(-1,0,true);
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::GPU);
            B.TransferFromDeviceToDevice(0,-1,true);
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);

            SingleMatrix C(-1);
            Assert::IsTrue(C.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);
            SingleMatrix D(15,15,arr,matrixFlagNormal,-1);
            Assert::IsTrue(D.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);
        }

        //This tests defaulting to GPU, transfering and gravitating towards preferred device. 
        //Your machine has to have CUDA5 GPU to pass it
        TEST_METHOD(MatrixDataSynchronization_GravitatingTowardsPreferredDevice)
        {
            SingleMatrix T(3,4,AUTOPLACEMATRIX);
            SingleMatrix BR(T);


            SingleMatrix A= SingleMatrix::RandomGaussian(64,23,0,2);               
            SingleMatrix B = SingleMatrix::Eye(23);

            Assert::IsTrue(A.GetCurrentMatrixLocation()==CurrentDataLocation::GPU);
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::GPU);

            float x = A(1,1); x;
            Assert::IsTrue(A.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);
            float y = B(1,1); y;
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::CPU);

            SingleMatrix C = A*B;
            Assert::IsTrue(A.GetCurrentMatrixLocation()==CurrentDataLocation::GPU);
            Assert::IsTrue(B.GetCurrentMatrixLocation()==CurrentDataLocation::GPU);
            Assert::IsTrue(C.GetCurrentMatrixLocation()==CurrentDataLocation::GPU);
        }
    };
}