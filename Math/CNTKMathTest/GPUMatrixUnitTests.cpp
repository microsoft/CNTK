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
#include "..\Math\GPUMatrix.h"
#define epsilon 0.00001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

#pragma warning (disable: 4244 4245 4305)       // conversions and truncations; we don't care in this test project

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
            GPUMatrix<float> M0(0 /*deviceId*/);
            Assert::IsTrue(M0.IsEmpty());                        

            GPUMatrix<float> M1(12, 53, 0 /*deviceId*/);
            Assert::AreEqual<size_t>(12,M1.GetNumRows());
            Assert::AreEqual<size_t>(53,M1.GetNumCols());   

            float *fArray = new float[6];
            fArray[0] = 1; fArray[1] = 2; fArray[2] = 3; 
            fArray[3] = 4; fArray[4] = 5; fArray[5] = 6; 
            GPUMatrix<float> M2(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            double *dArray = new double[6];
            dArray[0] = 1; dArray[1] = 2; dArray[2] = 3; 
            dArray[3] = 4; dArray[4] = 5; dArray[5] = 6; 
            GPUMatrix<double> M3(2, 3, 0 /*deviceId*/, dArray, matrixFlagNormal);
        }

        TEST_METHOD(GPUMatrix_ElementOps_NoExceptionOnly_Test)
        {
            GPUMatrix<float> A(123, 459, 0 /*deviceId*/);
            GPUMatrix<float> C(123, 459, 0 /*deviceId*/);
            GPUMatrix<float>::ElementWisePower(2,A,C);        
            float alpha = 0.234f;        
            GPUMatrix<float>::ElementWisePower(alpha,A,C);        

            GPUMatrix<float> M(2, 2, 0 /*deviceId*/);
            GPUMatrix<float> X = M.AssignAbsOf(M);

            GPUMatrix<float> Y(GPUMatrix<float>::Eye(600, 0 /*deviceId*/));
            float x=Y.Get00Element();
            Assert::AreEqual<float>(1,x);

            GPUMatrix<float> Z(GPUMatrix<float>::Zeros(3, 4, 0 /*deviceId*/));
            x=Z.Get00Element();
            Assert::AreEqual<float>(0,x);
        }

        TEST_METHOD(GPUMatrix_InplaceOperations_NoExceptionOnly_Test)
        {
            GPUMatrix<float> A(42, 69, 0 /*deviceId*/);
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
            GPUMatrix<float> M0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            fArray[0] = 11; fArray[2] = 12; fArray[4] = 13; 
            fArray[1] = 14; fArray[3] = 15; fArray[5] = 16; 
            GPUMatrix<float> M1(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            fArray[0] = 12; fArray[2] = 14; fArray[4] = 16; 
            fArray[1] = 18; fArray[3] = 20; fArray[5] = 22; 
            GPUMatrix<float> M2(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            fArray[0] = 10; 
            fArray[1] = 10; 
            GPUMatrix<float> MC(2, 1, 0 /*deviceId*/, fArray, matrixFlagNormal);

            fArray[0] = 10; fArray[1] = 10; fArray[2] = 10; 
            GPUMatrix<float> MR(1, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            fArray[0] = 10; 
            GPUMatrix<float> MS(1, 1, 0 /*deviceId*/, fArray, matrixFlagNormal);

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
            GPUMatrix<float> M0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            GPUMatrix<float> M3(0 /*deviceId*/);
            M0.VectorNorm1(M3, true);
            fArray[0] = 5; fArray[1] = 7; fArray[2] = 9; 
            GPUMatrix<float> M2(1, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm1(M3, false);
            M2.Resize(2,1);
            fArray[0] = 6; fArray[1] = 15;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm2(M3, true);
            M2.Resize(1, 3);
            fArray[0] = 4.1231f; fArray[1] = 5.3852f; fArray[2] = 6.7082f; 
            M2.SetValue(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0005f)); 

            M0.VectorNorm2(M3, false);
            M2.Resize(2,1);
            fArray[0] =  3.7417f; fArray[1] = 8.7750f;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0005f)); 

            fArray[0] = 1; fArray[2] = 2; fArray[4] = 3; 
            fArray[1] = 4; fArray[3] = 5; fArray[5] = 6; 
            GPUMatrix<float> M00(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            GPUMatrix<float> M1(0 /*deviceId*/);
            M00.VectorMax(M1, M3, true);
            M2.Resize(1, 3);
            fArray[0] = 4; fArray[1] = 5; fArray[2] = 6; 
            M2.SetValue(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001f)); 

            M00.VectorMax(M1, M3, false);
            M2.Resize(2,1);
            fArray[0] =  3.; fArray[1] = 6;
            M2.SetValue(2, 1, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001f)); 

            M0.VectorNormInf(M3, true);
            M2.Resize(1, 3);
            fArray[0] = 4; fArray[1] = 5; fArray[2] = 6; 
            M2.SetValue(1, 3, fArray, matrixFlagNormal);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001f)); 

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

            GPUMatrix<float> A = GPUMatrix<float>::Eye(4096, 0 /*deviceId*/);
            Assert::AreEqual<long>(4096,A.MatrixNorm0());

            GPUMatrix<float> B = GPUMatrix<float>::Eye(5, 0 /*deviceId*/);
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
            GPUMatrix<float> M0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            GPUMatrix<float> M1(2, 2, 0 /*deviceId*/, fArray, matrixFlagNormal);

            GPUMatrix<float> M2 = M0.ColumnSlice(0,2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001f)); 

            GPUMatrix<float> M3(2, 2, 0 /*deviceId*/, fArray + 2, matrixFlagNormal);

            M2 = M0.ColumnSlice(1,2);
            Assert::IsTrue(M2.IsEqualTo(M3, 0.0001f)); 
        }

        TEST_METHOD(GPUMatrixRowSliceAndStack)
        {
            float *fArray = new float[15];
            fArray[0] = 1; fArray[5] = 6; fArray[10] = 11;
            fArray[1] = 2; fArray[6] = 7; fArray[11] = 12;
            fArray[2] = 3; fArray[7] = 8; fArray[12] = 13;
            fArray[3] = 4; fArray[8] = 9; fArray[13] = 14;
            fArray[4] = 5; fArray[9] = 10; fArray[14] = 15;
            GPUMatrix<float> M0(5, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            float *fArray1 = new float[6];
            fArray1[0] = 3; fArray1[2] = 8; fArray1[4] = 13;
            fArray1[1] = 4; fArray1[3] = 9; fArray1[5] = 14;
            GPUMatrix<float> M1(2, 3, 0 /*deviceId*/, fArray1, matrixFlagNormal);

            GPUMatrix<float> M2(0 /*deviceId*/);
            M2.AssignRowSliceValuesOf(M0, 2, 2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 

            float *fArray3 = new float[15];
            fArray3[0] = 0; fArray3[5] = 0; fArray3[10] = 0;
            fArray3[1] = 0; fArray3[6] = 0; fArray3[11] = 0;
            fArray3[2] = 3; fArray3[7] = 8; fArray3[12] = 13;
            fArray3[3] = 4; fArray3[8] = 9; fArray3[13] = 14;
            fArray3[4] = 0; fArray3[9] = 0; fArray3[14] = 0;
            GPUMatrix<float> M3(5, 3, 0 /*deviceId*/, fArray3, matrixFlagNormal);

            M3 += M0;
            M0.AddToRowSliceValuesOf(M1, 2,2);
            Assert::IsTrue(M3.IsEqualTo(M0, 0.0001)); 

            M2.AddWithRowSliceValuesOf(M1, 0, 2);
            float *fArray4 = new float[6];
            fArray4[0] = 6; fArray4[2] = 16; fArray4[4] = 26;
            fArray4[1] = 8; fArray4[3] = 18; fArray4[5] = 28;
            GPUMatrix<float> M4(2, 3, 0 /*deviceId*/, fArray4, matrixFlagNormal);
            Assert::IsTrue(M2.IsEqualTo(M4, 0.0001));

#if 0
            GPUMatrix<float>  M5, M6, M7, M8;
            M5.AssignRowSliceValuesOf(M0, 0, 2);
            M6.AssignRowSliceValuesOf(M0, 2, 1);
            M7.AssignRowSliceValuesOf(M0, 3, 2);

            std::vector<const GPUMatrix<float> *> inputMatrices;
            inputMatrices.resize(3);
            inputMatrices[0] = &M5;
            inputMatrices[1] = &M6;
            inputMatrices[2] = &M7;
            M8.AssignRowStackValuesOf(inputMatrices, 0, 3);

            Assert::IsTrue(M8.IsEqualTo(M0, 0.0001));
#endif
        }

        TEST_METHOD(GPUKhatriRaoProduct)
        {
            float *fArray = new float[24];
            fArray[0] = 0.8147f; fArray[3] = 0.9134f; fArray[6] = 0.2785f; fArray[9] = 0.9649f;
            fArray[1] = 0.9058f; fArray[4] = 0.6324f; fArray[7] = 0.5469f; fArray[10] = 0.1576f;
            fArray[2] = 0.1270f; fArray[5] = 0.0975f; fArray[8] = 0.9575f; fArray[11] = 0.9706f;
            GPUMatrix<float> A(3, 4, 0 /*deviceId*/, fArray);

            fArray[0] = 0.9572f; fArray[2] = 0.8003f; fArray[4] = 0.4218f; fArray[6] = 0.7922f;
            fArray[1] = 0.4854f; fArray[3] = 0.1419f; fArray[5] = 0.9157f; fArray[7] = 0.9595f;
            GPUMatrix<float> B(2, 4, 0 /*deviceId*/, fArray);

            fArray[0] = 0.7798f; fArray[6] =  0.7310f; fArray[12] = 0.1175f; fArray[18] = 0.7644f;
            fArray[1] = 0.8670f; fArray[7] =  0.5061f; fArray[13] = 0.2307f; fArray[19] = 0.1249f;
            fArray[2] = 0.1215f; fArray[8] =  0.0781f; fArray[14] = 0.4038f; fArray[20] = 0.7689f;
            fArray[3] = 0.3954f; fArray[9] =  0.1296f; fArray[15] = 0.2550f; fArray[21] = 0.9258f;
            fArray[4] = 0.4396f; fArray[10] = 0.0897f; fArray[16] = 0.5008f; fArray[22] = 0.1512f;
            fArray[5] = 0.0616f; fArray[11] = 0.0138f; fArray[17] = 0.8768f; fArray[23] = 0.9313f;
            GPUMatrix<float> D(6, 4, 0 /*deviceId*/, fArray);

            GPUMatrix<float> C(0 /*deviceId*/);
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
            GPUMatrix<float> A(6, 2, 0 /*deviceId*/, fArray);

            fArray[0] = 0.2769f; fArray[3] = 0.8235f; 
            fArray[1] = 0.0462f; fArray[4] = 0.6948f; 
            fArray[2] = 0.0971f; fArray[5] = 0.3171f; 
            GPUMatrix<float> B(3, 2, 0 /*deviceId*/, fArray);

            fArray[0] = 0.2867f; fArray[2] = 1.2913f; 
            fArray[1] = 0.1266f; fArray[3] = 0.4520f; 
            GPUMatrix<float> D0(2, 2, 0 /*deviceId*/, fArray);

            fArray[0] = 0.2657f; fArray[2] = 1.0923f; 
            fArray[1] = 0.3636f; fArray[3] = 0.6416f; 
            GPUMatrix<float> D1(2, 2, 0 /*deviceId*/, fArray);

            GPUMatrix<float> C(2, 2, 0 /*deviceId*/);
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
            GPUMatrix<float> M0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            GPUMatrix<float> M1(0 /*deviceId*/), M2(0 /*deviceId*/);
            M1.AssignInnerProductOf(M0, M0, true);
            M2.AssignVectorNorm2Of(M0, true);
            M1.InplaceSqrt();
            Assert::IsTrue(M1.IsEqualTo(M2)); 

            M1.AssignInnerProductOf(M0, M0, false);
            M2.AssignVectorNorm2Of(M0, false);
            M1.InplaceSqrt();
            Assert::IsTrue(M1.IsEqualTo(M2));         
        }
        TEST_METHOD(GPUAssignRepeatOf)
        {
            float *fArray = new float[36];
            fArray[0] = 1; fArray[2] = 6; fArray[4] = 11;
            fArray[1] = 2; fArray[3] = 7; fArray[5] = 12;
            GPUMatrix<float> M0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

            GPUMatrix<float>  M1(0 /*deviceId*/);
            M1.AssignRepeatOf(M0, 1, 1);
            Assert::IsTrue(M1.IsEqualTo(M0, 0.0001));

            fArray[0] = 1; fArray[0 + 6] = 6; fArray[0 + 12] = 11; fArray[0 + 18] = 1; fArray[0 + 24] = 6; fArray[0 + 30] = 11;
            fArray[1] = 2; fArray[1 + 6] = 7; fArray[1 + 12] = 12; fArray[1 + 18] = 2; fArray[1 + 24] = 7; fArray[1 + 30] = 12;
            fArray[2] = 1; fArray[2 + 6] = 6; fArray[2 + 12] = 11; fArray[2 + 18] = 1; fArray[2 + 24] = 6; fArray[2 + 30] = 11;
            fArray[3] = 2; fArray[3 + 6] = 7; fArray[3 + 12] = 12; fArray[3 + 18] = 2; fArray[3 + 24] = 7; fArray[3 + 30] = 12;
            fArray[4] = 1; fArray[4 + 6] = 6; fArray[4 + 12] = 11; fArray[4 + 18] = 1; fArray[4 + 24] = 6; fArray[4 + 30] = 11;
            fArray[5] = 2; fArray[5 + 6] = 7; fArray[5 + 12] = 12; fArray[5 + 18] = 2; fArray[5 + 24] = 7; fArray[5 + 30] = 12;
            GPUMatrix<float> M3(6, 6, 0 /*deviceId*/, fArray, matrixFlagNormal);

            M1.AssignRepeatOf(M0, 3, 2);
            Assert::IsTrue(M1.IsEqualTo(M3, 0.0001));
        }

        TEST_METHOD(GPURowElementOperations)
        {
            GPUMatrix<float>   M0 = GPUMatrix<float>::RandomUniform(20, 28, 0 /*deviceId*/, -1, 1);
            GPUMatrix<float>   M1 = GPUMatrix<float>::RandomUniform(1, 28, 0 /*deviceId*/, 1, 2);

            GPUMatrix<float>   M3(0 /*deviceId*/);
            M3.SetValue(M0);
            M3.RowElementMultiplyWith(M1);
            M3.RowElementDivideBy(M1);

            Assert::IsTrue(M0.IsEqualTo(M3, 0.0001));
        }
        TEST_METHOD(GPUColumnElementOperations)
        {
            GPUMatrix<float>   M0 = GPUMatrix<float>::RandomUniform(20, 28, 0 /*deviceId*/, -1, 1);
            GPUMatrix<float>   M1 = GPUMatrix<float>::RandomUniform(20, 1, 0 /*deviceId*/, 1, 2);

            GPUMatrix<float>   M3(0 /*deviceId*/);
            M3.SetValue(M0);
            M3.ColumnElementMultiplyWith(M1);
            M3.ColumnElementDivideBy(M1);

            Assert::IsTrue(M0.IsEqualTo(M3, 0.0001));
        }
    };
}