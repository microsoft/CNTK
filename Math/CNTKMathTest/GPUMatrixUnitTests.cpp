//
// <copyright file="GPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
//GPUMatrix Unit tests should go here
#include "stdafx.h"
#include "CppUnitTest.h"
#include <math.h>
#include "..\Math\CPUMatrix.h"
#include "..\Math\GPUMatrix.cuh"
#define epsilon 0.00001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

#define DEBUG_FLAG 1
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Microsoft::MSR::CNTK;
namespace CNTKMathTest
{        
    
    TEST_CLASS(GPUMatrixUnitTests)
    {
    public:

        static void DebugPrint(FILE* gpuDebugFile, const GPUMatrix<float>& M, const char* str, const bool colwiseVec)
        {
            fprintf(gpuDebugFile, "\n %s\n", str);
            const int matNumRow = (int)M.GetNumRows();
            const int matNumCol = (int)M.GetNumCols();
            const int elemNum = (int)M.GetNumElements();

            if (colwiseVec)
            {
                GPUMatrix<float> M1 = M.Transpose();
                float* mArray = M1.CopyToArray();
                for (int i = 0; i < elemNum; i++)
                {
                    //if (i % matNumCol == 0)
                    // fprintf(gpuDebugFile, "Column Id %d: ", i/matNumRow);
                    fprintf(gpuDebugFile, "%3d ", (int)mArray[i]);
                    if ( (i+1)% matNumCol == 0)
                        fprintf(gpuDebugFile, "\n");
                }
            }
            else
            {
                float* mArray = M.CopyToArray();
                for (int i = 0; i < elemNum; i++)
                {
                    fprintf(gpuDebugFile, "%3d ", (int)mArray[i]);
                    if ( (i+1)% matNumCol == 0)
                        fprintf(gpuDebugFile, "\n");
                }
            }
            fprintf(gpuDebugFile, "\n");
        }        
        TEST_METHOD(GPUMatrixConsturctors)
        {
            GPUMatrix<float> M0;
            Assert::IsTrue(M0.IsEmpty());                        

            GPUMatrix<float> M1(12,53);
            Assert::AreEqual<size_t>(12,M1.GetNumRows());
            Assert::AreEqual<size_t>(53,M1.GetNumCols());   

            float *fArray = new float[6];
            fArray[0] = 1; fArray[1] = 2; fArray[2] = 3; 
            fArray[3] = 4; fArray[4] = 5; fArray[5] = 6; 
            GPUMatrix<float> M2(2, 3, fArray, matrixFlagNormal);

            double *dArray = new double[6];
            dArray[0] = 1; dArray[1] = 2; dArray[2] = 3; 
            dArray[3] = 4; dArray[4] = 5; dArray[5] = 6; 
            GPUMatrix<double> M3(2, 3, dArray, matrixFlagNormal);                        
        }

        TEST_METHOD(GPUMatrix_ElementOps_NoExceptionOnly_Test)
        {
            GPUMatrix<float> A(123,459);        
            GPUMatrix<float> C(123,459);          
            GPUMatrix<float>::ElementWisePower(2,A,C);        
            float alpha = 0.234f;        
            GPUMatrix<float>::ElementWisePower(alpha,A,C);        

            GPUMatrix<float> M(2,2);
            GPUMatrix<float> X = M.AssignAbsOf(M);

            GPUMatrix<float> Y = GPUMatrix<float>::Eye(600);
            float x=Y.Get00Element();
            Assert::AreEqual<float>(1,x);

            GPUMatrix<float> Z = GPUMatrix<float>::Zeros(3,4);
            x=Z.Get00Element();
            Assert::AreEqual<float>(0,x);
        }

        TEST_METHOD(GPUMatrix_InplaceOperations_NoExceptionOnly_Test)
        {
            GPUMatrix<float> A(42,69);                       
            A.InplaceExp();                     
            A.InplaceLog();        
            A.InplaceTanh();        
            A.InplaceAbs();        
            A.InplaceSqrt();
            A.InplaceSigmoid();        
        }

        TEST_METHOD(GPUMatrixAddAndSub)
        {
            float *fArray = new float[6];
            fArray[0] = 1; fArray[2] = 2; fArray[4] = 3; 
            fArray[1] = 4; fArray[3] = 5; fArray[5] = 6; 
            GPUMatrix<float> M0(2, 3, fArray, matrixFlagNormal);

            fArray[0] = 11; fArray[2] = 12; fArray[4] = 13; 
            fArray[1] = 14; fArray[3] = 15; fArray[5] = 16; 
            GPUMatrix<float> M1(2, 3, fArray, matrixFlagNormal);

            fArray[0] = 12; fArray[2] = 14; fArray[4] = 16; 
            fArray[1] = 18; fArray[3] = 20; fArray[5] = 22; 
            GPUMatrix<float> M2(2, 3, fArray, matrixFlagNormal);

            fArray[0] = 10; 
            fArray[1] = 10; 
            GPUMatrix<float> MC(2, 1, fArray, matrixFlagNormal);

            fArray[0] = 10; fArray[1] = 10; fArray[2] = 10; 
            GPUMatrix<float> MR(1, 3, fArray, matrixFlagNormal);

            fArray[0] = 10; 
            GPUMatrix<float> MS(1, 1, fArray, matrixFlagNormal);

            GPUMatrix<float>  M3 = M2 - M0;
            Assert::IsTrue(M3.IsEqualTo(M1)); 

            M3 += M0;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M3 = M0 + 10;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3 -= 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 = M1 + M0;
            Assert::IsTrue(M3.IsEqualTo(M2));  

            M3 -= M0;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3 = M1 - 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 += 10;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3 -= MC;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 += MC;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3 -= MR;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 += MR;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3.AssignDifferenceOf(M3, MS);
            Assert::IsTrue(M3.IsEqualTo(M0));
        }

        TEST_METHOD(GPUMatrixNorms)
        {
            float *fArray = new float[6];
            fArray[0] = 1; fArray[2] = 2; fArray[4] = 3; 
            fArray[1] = 4; fArray[3] = 5; fArray[5] = 6; 
            GPUMatrix<float> M0(2, 3, fArray, matrixFlagNormal);

            GPUMatrix<float> M3;
            M0.VectorNorm1(M3, true);
            fArray[0] = 5; fArray[1] = 7; fArray[2] = 9; 
            GPUMatrix<float> M2(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm1(M3, false);
            M2.Resize(2,1);
            fArray[0] = 6; fArray[1] = 15;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm2(M3, true);
            M2.Resize(1, 3);
            fArray[0] = 4.1231; fArray[1] = 5.3852; fArray[2] = 6.7082; 
            M2.SetValue(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0005)); 

            M0.VectorNorm2(M3, false);
            M2.Resize(2,1);
            fArray[0] =  3.7417; fArray[1] = 8.7750;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0005)); 

            fArray[0] = 1; fArray[2] = 2; fArray[4] = 3; 
            fArray[1] = 4; fArray[3] = 5; fArray[5] = 6; 
            GPUMatrix<float> M00(2, 3, fArray, matrixFlagNormal);

            GPUMatrix<float> M1;
            M00.VectorMax(M1, M3, true);
            M2.Resize(1, 3);
            fArray[0] = 4; fArray[1] = 5; fArray[2] = 6; 
            M2.SetValue(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M00.VectorMax(M1, M3, false);
            M2.Resize(2,1);
            fArray[0] =  3.; fArray[1] = 6;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorNormInf(M3, true);
            M2.Resize(1, 3);
            fArray[0] = 4; fArray[1] = 5; fArray[2] = 6; 
            M2.SetValue(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorNormInf(M3, false);
            M2.Resize(2,1);
            fArray[0] =  3.; fArray[1] = 6;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            fArray[0] = 1; fArray[2] = 2; fArray[4] = 3; 
            fArray[1] = 4; fArray[3] = 5; fArray[5] = 6; 
            M00.SetValue(2, 3, fArray, matrixFlagNormal);
            Assert::AreEqual<float>(6,M00.MatrixNormInf());

            Assert::IsTrue(abs(M0.FrobeniusNorm() - 9.5394) < 0.0001);
            Assert::IsTrue(abs(M0.MatrixNormInf() - 6) < 0.0001);
            Assert::AreEqual<float>(21,M00.MatrixNorm1());

            GPUMatrix<float> A = GPUMatrix<float>::Eye(4096);
            Assert::AreEqual<long>(4096,A.MatrixNorm0());

            GPUMatrix<float> B = GPUMatrix<float>::Eye(5);
            Assert::AreEqual<long>(5,B.MatrixNorm0());
        }
        TEST_METHOD(GPUMatrixRandomUniform)
        {
            GPUMatrix<float> A = GPUMatrix<float>::RandomUniform(768,50,-0.035f,0.035f,1);                       
            float* arr = A.CopyToArray();

            for (int i = 0; i<768*50;++i)
            {
                Assert::IsTrue(arr[i]<=0.035);
                Assert::IsTrue(arr[i]>-0.035);
            }

            delete[] arr;
        }

        TEST_METHOD(GPUMatrixColumnSlice)
        {
            float *fArray = new float[6];
            fArray[0] = 1; fArray[1] = 4; fArray[2] = 2; 
            fArray[3] = 5; fArray[4] = 3; fArray[5] = 6; 
            GPUMatrix<float> M0(2, 3, fArray, matrixFlagNormal);

            GPUMatrix<float> M1(2, 2, fArray, matrixFlagNormal);

            GPUMatrix<float> M2 = M0.ColumnSlice(0,2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001f)); 

            GPUMatrix<float> M3(2, 2, fArray+2, matrixFlagNormal);

            M2 = M0.ColumnSlice(1,2);
            Assert::IsTrue(M2.IsEqualTo(M3, 0.0001f)); 
        }

		TEST_METHOD(GPUMatrixRowSlice)
		{
            float *fArray = new float[15];
            fArray[0] = 1; fArray[5] = 6; fArray[10] = 11;
            fArray[1] = 2; fArray[6] = 7; fArray[11] = 12;
            fArray[2] = 3; fArray[7] = 8; fArray[12] = 13;
            fArray[3] = 4; fArray[8] = 9; fArray[13] = 14;
            fArray[4] = 5; fArray[9] = 10; fArray[14] = 15;
            GPUMatrix<float> M0(5, 3, fArray, matrixFlagNormal);

            float *fArray1 = new float[6];
            fArray1[0] = 3; fArray1[2] = 8; fArray1[4] = 13;
            fArray1[1] = 4; fArray1[3] = 9; fArray1[5] = 14;
            GPUMatrix<float> M1(2, 3, fArray1, matrixFlagNormal);

            GPUMatrix<float> M2;
            M2.AssignRowSliceValuesOf(M0, 2, 2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 

            float *fArray3 = new float[15];
            fArray3[0] = 0; fArray3[5] = 0; fArray3[10] = 0;
            fArray3[1] = 0; fArray3[6] = 0; fArray3[11] = 0;
            fArray3[2] = 3; fArray3[7] = 8; fArray3[12] = 13;
            fArray3[3] = 4; fArray3[8] = 9; fArray3[13] = 14;
            fArray3[4] = 0; fArray3[9] = 0; fArray3[14] = 0;
            GPUMatrix<float> M3(5, 3, fArray3, matrixFlagNormal);

            M3 += M0;
            M0.AddToRowSliceValuesOf(M1, 2,2);
            Assert::IsTrue(M3.IsEqualTo(M0, 0.0001)); 
		}

        TEST_METHOD(GPUKhatriRaoProduct)
        {
            float *fArray = new float[24];
            fArray[0] = 0.8147f; fArray[3] = 0.9134f; fArray[6] = 0.2785f; fArray[9] = 0.9649f;
            fArray[1] = 0.9058f; fArray[4] = 0.6324f; fArray[7] = 0.5469f; fArray[10] = 0.1576f;
            fArray[2] = 0.1270f; fArray[5] = 0.0975f; fArray[8] = 0.9575f; fArray[11] = 0.9706f;
            GPUMatrix<float> A(3,4,fArray);

            fArray[0] = 0.9572f; fArray[2] = 0.8003f; fArray[4] = 0.4218f; fArray[6] = 0.7922f;
            fArray[1] = 0.4854f; fArray[3] = 0.1419f; fArray[5] = 0.9157f; fArray[7] = 0.9595f;
            GPUMatrix<float> B(2,4,fArray);

            fArray[0] = 0.7798f; fArray[6] =  0.7310f; fArray[12] = 0.1175f; fArray[18] = 0.7644f;
            fArray[1] = 0.8670f; fArray[7] =  0.5061f; fArray[13] = 0.2307f; fArray[19] = 0.1249f;
            fArray[2] = 0.1215f; fArray[8] =  0.0781f; fArray[14] = 0.4038f; fArray[20] = 0.7689f;
            fArray[3] = 0.3954f; fArray[9] =  0.1296f; fArray[15] = 0.2550f; fArray[21] = 0.9258f;
            fArray[4] = 0.4396f; fArray[10] = 0.0897f; fArray[16] = 0.5008f; fArray[22] = 0.1512f;
            fArray[5] = 0.0616f; fArray[11] = 0.0138f; fArray[17] = 0.8768f; fArray[23] = 0.9313f;
            GPUMatrix<float> D(6,4, fArray);

            GPUMatrix<float> C;
            C.AssignKhatriRaoProductOf(A, B);
            Assert::IsTrue(C.IsEqualTo(D, 0.0001f)); 

        }

        TEST_METHOD(GPUAddColumnReshapeProductOf)
        {
            float *fArray = new float[12];
            fArray[0] = 0.6557f; fArray[6] =  0.7431f; 
            fArray[1] = 0.0357f; fArray[7] =  0.3922f; 
            fArray[2] = 0.8491f; fArray[8] =  0.6555f; 
            fArray[3] = 0.9340f; fArray[9]  = 0.1712f; 
            fArray[4] = 0.6787f; fArray[10] = 0.7060f; 
            fArray[5] = 0.7577f; fArray[11] = 0.0318f; 
            GPUMatrix<float> A(6,2,fArray);

            fArray[0] = 0.2769f; fArray[3] = 0.8235f; 
            fArray[1] = 0.0462f; fArray[4] = 0.6948f; 
            fArray[2] = 0.0971f; fArray[5] = 0.3171f; 
            GPUMatrix<float> B(3,2,fArray);

            fArray[0] = 0.2867f; fArray[2] = 1.2913f; 
            fArray[1] = 0.1266f; fArray[3] = 0.4520f; 
            GPUMatrix<float> D0(2,2,fArray);

            fArray[0] = 0.2657f; fArray[2] = 1.0923f; 
            fArray[1] = 0.3636f; fArray[3] = 0.6416f; 
            GPUMatrix<float> D1(2,2,fArray);

            GPUMatrix<float> C(2,2);
            C.SetValue(0.0f);
            C.AddColumnReshapeProductOf(A, B, false);
            Assert::IsTrue(C.IsEqualTo(D0, 0.0001f)); 

            C.SetValue(0.0f);
            C.AddColumnReshapeProductOf(A, B, true);
            Assert::IsTrue(C.IsEqualTo(D1, 0.0001f)); 
        }

        TEST_METHOD(GPUInnerProduct)
        {
            float *fArray = new float[6];
            fArray[0] = 1; fArray[2] = 2; fArray[4] = 3; 
            fArray[1] = 4; fArray[3] = 5; fArray[5] = 6; 
            GPUMatrix<float> M0(2, 3, fArray, matrixFlagNormal);

            GPUMatrix<float> M1, M2;
            M1.AssignInnerProductOf(M0, M0, true);
            M2.AssignVectorNorm2Of(M0, true);
            M1.InplaceSqrt();
            Assert::IsTrue(M1.IsEqualTo(M2)); 

            M1.AssignInnerProductOf(M0, M0, false);
            M2.AssignVectorNorm2Of(M0, false);
            M1.InplaceSqrt();
            Assert::IsTrue(M1.IsEqualTo(M2));         
        }
    };
}