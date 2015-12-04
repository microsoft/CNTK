//
// <copyright file="CPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Math\CPUMatrix.h"
#define DEBUG_FLAG 1
using namespace Microsoft::MSR::CNTK;

#pragma warning (disable: 4305)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNTKMathTest
{    
    TEST_CLASS(CPUMatrixUnitTest)
    {
        //typedef CPUSingleMatrix Matrix;  
        typedef CPUDoubleMatrix Matrix;  

    public:
        static void DebugPrint(FILE* gpuDebugFile, Matrix M, const char* str, const bool colwiseVec = true)
        {
            fprintf(gpuDebugFile, "\n %s\n", str);
            const size_t matNumCol = M.GetNumCols();
            const size_t elemNum = M.GetNumElements();
            Matrix M1 = M.Transpose();
            double* pArray = M1.GetArray();
            if (colwiseVec)
            {
                for (size_t i = 0; i < elemNum; i++)
                {

                    fprintf(gpuDebugFile, "%3d ", (int)pArray[i]);
                    if ( (i+1)% matNumCol == 0)
                        fprintf(gpuDebugFile, "\n");
                }
            }
            //const size_t matNumRow = M.GetNumRows();
            //for (int i = 0; i < matNumRow; i++)
            //{
            //    for (int j = 0; j < matNumCol; j++)
            //    {
            //        fprintf(gpuDebugFile, "%3d ", M(i,j));
            //        //if ( (j+1)% matNumCol == 0)
            //    }
            //    fprintf(gpuDebugFile, "\n");
            //}
        }    
        TEST_METHOD(CPUMatrixConsturctors)
        {
            Matrix M0;
            Assert::IsTrue(M0.IsEmpty());

            M0.Resize(2,3);
            Assert::IsFalse(M0.IsEmpty());
            Assert::AreEqual<size_t>(2,M0.GetNumRows());
            Assert::AreEqual<size_t>(3,M0.GetNumCols());
            Assert::AreEqual<size_t>(6,M0.GetNumElements());

            M0(0,0) = 1; M0(1,2) = 2;
            Assert::IsTrue(M0(0,0) == 1);
            Assert::IsTrue(M0(1,2) == 2);

            Matrix M1(12,53);
            Assert::AreEqual<size_t>(12,M1.GetNumRows());
            Assert::AreEqual<size_t>(53,M1.GetNumCols());   


            float *fArray = new float[6];
            fArray[0] = 1; fArray[1] = 2; fArray[2] = 3; 
            fArray[3] = 4; fArray[4] = 5; fArray[5] = 6; 
            CPUMatrix<float> M2(2, 3, fArray, matrixFlagNormal);
            Assert::AreEqual<float>(M2(0,0), 1);
            Assert::AreEqual<float>(M2(0,1), 3);
            Assert::AreEqual<float>(M2(0,2), 5);
            Assert::AreEqual<float>(M2(1,0), 2);
            Assert::AreEqual<float>(M2(1,1), 4);
            Assert::AreEqual<float>(M2(1,2), 6);

            double *dArray = new double[6];
            dArray[0] = 1; dArray[1] = 2; dArray[2] = 3; 
            dArray[3] = 4; dArray[4] = 5; dArray[5] = 6; 
            CPUMatrix<double> M3(2, 3, dArray, matrixFormatRowMajor);
            Assert::AreEqual<double>(M3(0,0), 1);
            Assert::AreEqual<double>(M3(0,1), 2);
            Assert::AreEqual<double>(M3(0,2), 3);
            Assert::AreEqual<double>(M3(1,0), 4);
            Assert::AreEqual<double>(M3(1,1), 5);
            Assert::AreEqual<double>(M3(1,2), 6);

            Matrix M4(M0);
            Assert::IsTrue(M4.IsEqualTo(M0));

            Matrix M5 = M0;
            Assert::IsTrue(M5.IsEqualTo(M0));
        }

        TEST_METHOD(CPUMatrixAddAndSub)
        {
            Matrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            Matrix M1(2,3);
            M1(0,0) = 11; M1(0,1) = 12; M1(0,2) = 13;
            M1(1,0) = 14; M1(1,1) = 15; M1(1,2) = 16;

            Matrix M2(2,3);
            M2(0,0) = 12; M2(0,1) = 14; M2(0,2) = 16;
            M2(1,0) = 18; M2(1,1) = 20; M2(1,2) = 22;

            Matrix MC(2,1);
            MC(0,0) = 10; 
            MC(1,0) = 10; 

            Matrix MR(1,3);
            MR(0,0) = 10; MR(0,1) = 10; MR(0,2) = 10; 

            Matrix MS(1,1);
            MS(0,0) = 10; 

            Matrix M3 = M2 - M0;
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

        TEST_METHOD(CPUMatrixMultiAndDiv)
        {
            Matrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            Matrix M00(2,3);
            M00(0,0) = 10; M00(0,1) = 20; M00(0,2) = 30;
            M00(1,0) = 40; M00(1,1) = 50; M00(1,2) = 60;

            Matrix M1(2,3);
            M1.Reshape(3,2);
            M1(0,0) = 11; M1(0,1) = 15; 
            M1(1,0) = 14; M1(1,1) = 13; 
            M1(2,0) = 12; M1(2,1) = 16; 

            Matrix M2(2,2);
            M2(0,0) = 75; M2(0,1) = 89; 
            M2(1,0) = 186; M2(1,1) = 221; 

            Matrix M3 = M0 * M1;
            Assert::IsTrue(M3.IsEqualTo(M2));  

            M3 = M0 * 10;
            Assert::IsTrue(M3.IsEqualTo(M00));  

            M3 = M3 / 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 *= 10;
            Assert::IsTrue(M3.IsEqualTo(M00));  

            M3 /= 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            Matrix::MultiplyAndWeightedAdd(1, M0, false, M1, false, 0, M3);
            Assert::IsTrue(M3.IsEqualTo(M2));  

            M1.Reshape(2,3);
            Matrix::MultiplyAndWeightedAdd(1, M0, false, M1, true, 0, M3);
            M2(0,0) = 74; M2(0,1) = 92; 
            M2(1,0) = 182; M2(1,1) = 227; 
            Assert::IsTrue(M3.IsEqualTo(M2));  

            Matrix::MultiplyAndWeightedAdd(10, M0, false, M1, true, 2, M3);
            M2(0,0) = 888; M2(0,1) = 1104; 
            M2(1,0) = 2184; M2(1,1) = 2724; 
            Assert::IsTrue(M3.IsEqualTo(M2));  

            Matrix::MultiplyAndWeightedAdd(1, M0, true, M1, false, 0, M3);
            M2.Resize(3,3);
            M2(0,0) = 67; M2(0,1) = 72; M2(0,2) = 77; 
            M2(1,0) = 92; M2(1,1) = 99; M2(1,2) = 106; 
            M2(2,0) = 117; M2(2,1) = 126; M2(2,2) = 135; 
            Assert::IsTrue(M3.IsEqualTo(M2));  
        }

        TEST_METHOD(CPUMatrixElementOps)
        {
            Matrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            Matrix M00(2,3);
            M00(0,0) = 1.0; M00(0,1) = 1/2.0; M00(0,2) = 1/3.0;
            M00(1,0) = 1/4.0; M00(1,1) = 1/5.0; M00(1,2) = 1/6.0;

            Matrix M1(2,3);
            M1(0,0) = 1; M1(0,1) = 1; M1(0,2) = 1;
            M1(1,0) = 1; M1(1,1) = 1; M1(1,2) = 1;

            Matrix M3;
            M3.AssignElementProductOf(M0, M00);
            Assert::IsTrue(M3.IsEqualTo(M1, 0.0001)); 

            M3 = M0 ^ 4;
            Matrix M2(2,3);
            M2(0,0) = 1; M2(0,1) = 16; M2(0,2) = 81;
            M2(1,0) = 256; M2(1,1) = 625; M2(1,2) = 1296;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M3.SetValue(M0);
            M3 ^= 4;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M3.SetValue(M0);
            M3.ElementMultiplyWith(M00);
            Assert::IsTrue(M3.IsEqualTo(M1)); 

            M3.SetValue(M0);
            M3.ElementInverse();
            Assert::IsTrue(M3.IsEqualTo(M00)); 

            M2(0,0) = 0.7311; M2(0,1) = 0.8808; M2(0,2) = 0.9526;
            M2(1,0) = 0.9820; M2(1,1) = 0.9933; M2(1,2) = 0.9975;
            M3.AssignElementDivisionOf(M2, M0);
            M2.ElementMultiplyWith(M00);
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M3.SetValue(M0);
            M3.InplaceSigmoid();
            M2(0,0) = 0.7311; M2(0,1) = 0.8808; M2(0,2) = 0.9526;
            M2(1,0) = 0.9820; M2(1,1) = 0.9933; M2(1,2) = 0.9975;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.SetValue(M0);
            M3.InplaceTanh();
            M2(0,0) = 0.7616; M2(0,1) = 0.9640; M2(0,2) = 0.9951;
            M2(1,0) = 0.9993; M2(1,1) = 0.9999; M2(1,2) = 1.0000;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.SetValue(M0);
            M3.InplaceLogSoftmax(true);
            M3.InplaceExp();
            M2(0,0) = 0.0474; M2(0,1) = 0.0474; M2(0,2) = 0.0474;
            M2(1,0) = 0.9526; M2(1,1) = 0.9526; M2(1,2) = 0.9526;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.SetValue(M0);
            M3.InplaceLogSoftmax(false);
            M3.InplaceExp();
            M2(0,0) = 0.0900; M2(0,1) = 0.2447; M2(0,2) = 0.6652;
            M2(1,0) = 0.0900; M2(1,1) = 0.2447; M2(1,2) = 0.6652;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M3.SetValue(M0);
            M3.InplaceHardmax(true);
            M2(0, 0) = 0.0; M2(0, 1) = 0.0; M2(0, 2) = 0.0;
            M2(1, 0) = 1.0; M2(1, 1) = 1.0; M2(1, 2) = 1.0;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001));

            M3.SetValue(M0);
            M3.InplaceHardmax(false);
            M2(0, 0) = 0.0; M2(0, 1) = 0.0; M2(0, 2) = 1.0;
            M2(1, 0) = 0.0; M2(1, 1) = 0.0; M2(1, 2) = 1.0;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001));

            M3.SetValue(M0);
            M3.InplaceSqrt();
            M2(0,0) = 1; M2(0,1) = 1.4142; M2(0,2) = 1.7321;
            M2(1,0) = 2; M2(1,1) = 2.2361; M2(1,2) = 2.4495;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.SetValue(M0);
            M3.InplaceExp();
            M2(0,0) = 2.7183; M2(0,1) = 7.3891; M2(0,2) = 20.0855;
            M2(1,0) = 54.5982; M2(1,1) = 148.4132; M2(1,2) = 403.4288;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.SetValue(M0);
            M3.InplaceExp();
            M2(0,0) = 2.7183; M2(0,1) = 7.3891; M2(0,2) = 20.0855;
            M2(1,0) = 54.5982; M2(1,1) = 148.4132; M2(1,2) = 403.4288;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.InplaceLog();
            Assert::IsTrue(M3.IsEqualTo(M0, 0.0001)); 

            M3.SetValue(M0);
            M3.InplaceTruncateBottom(2);
            M2(0,0) = 2; M2(0,1) = 2; M2(0,2) = 3;
            M2(1,0) = 4; M2(1,1) = 5; M2(1,2) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M3.SetValue(M0);
            M3.InplaceTruncateTop(4);
            M2(0,0) = 1; M2(0,1) = 2; M2(0,2) = 3;
            M2(1,0) = 4; M2(1,1) = 4; M2(1,2) = 4;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            double pi = 3.14159265358979323846264338327950288419716939937510;

            Matrix M_Trig(2,3);
            M_Trig(0,0) = 0; M_Trig(0,1) = pi/2.0; M_Trig(0,2) = pi;
            M_Trig(1,0) = 3.0*pi/2.0; M_Trig(1,1) = 2.0*pi; M_Trig(1,2) = 5.0*pi/2.0;

            Matrix M_Cos(2,3);
            M_Cos.SetValue(M_Trig);

            Matrix M_Cos_expected(2,3);
            M_Cos_expected(0,0) = 1; M_Cos_expected(0,1) = 0; M_Cos_expected(0,2) = -1;
            M_Cos_expected(1,0) = 0; M_Cos_expected(1,1) = 1; M_Cos_expected(1,2) =  0;

            M_Cos.InplaceCosine();
            Assert::IsTrue(M_Cos.IsEqualTo(M_Cos_expected, 0.0001)); 

            M_Cos.SetValue(M_Trig);
            M_Cos.AssignCosineOf(M_Trig);
            Assert::IsTrue(M_Cos.IsEqualTo(M_Cos_expected, 0.0001)); 

            Matrix M_NegSine(2,3);
            M_NegSine.SetValue(M_Trig);

            Matrix M_NegSine_expected(2,3);
            M_NegSine_expected(0,0) = 0; M_NegSine_expected(0,1) = -1; M_NegSine_expected(0,2) =  0;
            M_NegSine_expected(1,0) = 1; M_NegSine_expected(1,1) =  0; M_NegSine_expected(1,2) = -1;

            M_NegSine.InplaceNegativeSine();
            Assert::IsTrue(M_NegSine.IsEqualTo(M_NegSine_expected, 0.0001)); 

            M_NegSine.SetValue(M_Trig);
            M_NegSine.AssignNegativeSineOf(M_Trig);
            Assert::IsTrue(M_NegSine.IsEqualTo(M_NegSine_expected, 0.0001));
        }

        TEST_METHOD(CPUMatrixNorms)
        {
            Matrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            Matrix M3;
            M0.VectorNorm1(M3, true);
            Matrix M2(1, 3);
            M2(0,0) = 5; M2(0,1) = 7; M2(0,2) = 9;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm1(M3, false);
            M2.Resize(2,1);
            M2(0,0) = 6;
            M2(1,0) = 15;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm2(M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 4.1231; M2(0,1) = 5.3852; M2(0,2) = 6.7082;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorNorm2(M3, false);
            M2.Resize(2,1);
            M2(0,0) = 3.7417;
            M2(1,0) = 8.7750;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorNormInf(M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 4; M2(0,1) = 5; M2(0,2) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorNormInf(M3, false);
            M2.Resize(2,1);
            M2(0,0) = 3;
            M2(1,0) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            Assert::IsTrue(abs(M0.FrobeniusNorm() - 9.5394) < 0.0001);
            Assert::IsTrue(abs(M0.MatrixNormInf() - 6) < 0.0001);

            Matrix M1;
            M0.VectorMax(M1, M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 4; M2(0,1) = 5; M2(0,2) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorMax(M1, M3, false);
            M2.Resize(2,1);
            M2(0,0) = 3;
            M2(1,0) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorMin(M1, M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 1; M2(0,1) = 2; M2(0,2) = 3;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorMin(M1, M3, false);
            M2.Resize(2,1);
            M2(0,0) = 1;
            M2(1,0) = 4;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001));            
        }

        TEST_METHOD(CPUMatrixSetValues)
        {
            Matrix M0(3,3);
            M0(0,0) = 10; M0(1,1) = 10; M0(2,2) = 10;

            Matrix M1(3,3);
            M1.SetDiagonalValue(10);
            Assert::IsTrue(M1.IsEqualTo(M0, 0.0001)); 

            Matrix M2(3,1);
            M2(0,0) = 10; M2(1,0) = 10; M2(2,0) = 10;
            M1.SetDiagonalValue(M2);
            Assert::IsTrue(M1.IsEqualTo(M0, 0.0001)); 

            M1.SetUniformRandomValue(-0.01, 0.01);
            for (int i=0; i<M1.GetNumRows(); i++)
                for (int j=0; j<M1.GetNumCols(); j++)
                    Assert::IsTrue(M1(i,j) >= -0.01 && M1(i,j) < 0.01);

            M1.SetGaussianRandomValue(0, 0.01);
        }

        TEST_METHOD(CPUMatrixTranspose)
        {
            Matrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            Matrix M1(3,2);
            M1(0,0) = 1; M1(0,1) = 4; 
            M1(1,0) = 2; M1(1,1) = 5;
            M1(2,0) = 3; M1(2,1) = 6;

            Matrix M2 = M0.Transpose();
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 
           
            M2.AssignTransposeOf(M1);
            Assert::IsTrue(M2.IsEqualTo(M0, 0.0001)); 
        }

        TEST_METHOD(CPUMatrixColumnSlice)
        {
            Matrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            Matrix M1(2,2);
            M1(0,0) = 1; M1(0,1) = 2;
            M1(1,0) = 4; M1(1,1) = 5;

            Matrix M2 = M0.ColumnSlice(0,2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 

            M1(0,0) = 2; M1(0,1) = 3;
            M1(1,0) = 5; M1(1,1) = 6;

            M2 = M0.ColumnSlice(1,2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 

            size_t k=100, n=20, m=50;

            Matrix AG((size_t)k,(size_t)n);
            AG.SetUniformRandomValue(-1,1);

            Matrix BG((size_t)n,(size_t)m);
            BG.SetUniformRandomValue(-1,1);

            Matrix CG((size_t)k,(size_t)m);
            CG.SetUniformRandomValue(-1,1);
            Matrix DG((size_t)k,(size_t)m);
            DG.SetValue(CG);

            Matrix::MultiplyAndAdd(AG, false, BG, false, DG);

            for (int i=0; i<m; i++)
            {
                Matrix col_BG = BG.ColumnSlice(i,1);
                Matrix col_CG = CG.ColumnSlice(i,1);
                Matrix::MultiplyAndAdd(AG, false, col_BG, false, col_CG);
            }
            Assert::IsTrue(CG.IsEqualTo(DG, 0.0001)); 
        }

        TEST_METHOD(CPUKhatriRaoProduct)
        {
            Matrix A(3,4);
            A(0,0) = 0.8147; A(0,1) = 0.9134; A(0,2) = 0.2785; A(0,3) = 0.9649;
            A(1,0) = 0.9058; A(1,1) = 0.6324; A(1,2) = 0.5469; A(1,3) = 0.1576;
            A(2,0) = 0.1270; A(2,1) = 0.0975; A(2,2) = 0.9575; A(2,3) = 0.9706;

            Matrix B(2,4);
            B(0,0) = 0.9572; B(0,1) = 0.8003; B(0,2) = 0.4218; B(0,3) = 0.7922;
            B(1,0) = 0.4854; B(1,1) = 0.1419; B(1,2) = 0.9157; B(1,3) = 0.9595;

            Matrix D(6,4);
            D(0,0) = 0.7798; D(0,1) = 0.7310; D(0,2) = 0.1175; D(0,3) = 0.7644;
            D(1,0) = 0.8670; D(1,1) = 0.5061; D(1,2) = 0.2307; D(1,3) = 0.1249;
            D(2,0) = 0.1215; D(2,1) = 0.0781; D(2,2) = 0.4038; D(2,3) = 0.7689;
            D(3,0) = 0.3954; D(3,1) = 0.1296; D(3,2) = 0.2550; D(3,3) = 0.9258;
            D(4,0) = 0.4396; D(4,1) = 0.0897; D(4,2) = 0.5008; D(4,3) = 0.1512;
            D(5,0) = 0.0616; D(5,1) = 0.0138; D(5,2) = 0.8768; D(5,3) = 0.9313;

            Matrix C;
            C.AssignKhatriRaoProductOf(A, B);
            Assert::IsTrue(C.IsEqualTo(D, 0.0001)); 

        }

        TEST_METHOD(CPUAddColumnReshapeProductOf)
        {
            Matrix A(6,2);
            A(0,0) = 0.6557; A(0,1) = 0.7431; 
            A(1,0) = 0.0357; A(1,1) = 0.3922; 
            A(2,0) = 0.8491; A(2,1) = 0.6555; 
            A(3,0) = 0.9340; A(3,1) = 0.1712; 
            A(4,0) = 0.6787; A(4,1) = 0.7060; 
            A(5,0) = 0.7577; A(5,1) = 0.0318; 

            Matrix B(3,2);
            B(0,0) = 0.2769; B(0,1) = 0.8235; 
            B(1,0) = 0.0462; B(1,1) = 0.6948; 
            B(2,0) = 0.0971; B(2,1) = 0.3171; 

            Matrix D0(2,2);
            D0(0,0) = 0.2867; D0(0,1) = 1.2913; 
            D0(1,0) = 0.1266; D0(1,1) = 0.4520; 

            Matrix D1(2,2);
            D1(0,0) = 0.2657; D1(0,1) = 1.0923; 
            D1(1,0) = 0.3636; D1(1,1) = 0.6416; 

            Matrix C(2,2);
            C.SetValue(0);
            C.AddColumnReshapeProductOf(A, B, false);
            Assert::IsTrue(C.IsEqualTo(D0, 0.0001)); 

            C.SetValue(0);
            C.AddColumnReshapeProductOf(A, B, true);
            Assert::IsTrue(C.IsEqualTo(D1, 0.0001)); 
        }

        TEST_METHOD(CPUMatrixRowSliceAndStack)
        {
            Matrix M0(5,3);
            M0(0,0) = 1; M0(0,1) = 6; M0(0,2) = 11;
            M0(1,0) = 2; M0(1,1) = 7; M0(1,2) = 12;
            M0(2,0) = 3; M0(2,1) = 8; M0(2,2) = 13;
            M0(3,0) = 4; M0(3,1) = 9; M0(3,2) = 14;
            M0(4,0) = 5; M0(4,1) = 10; M0(4,2) = 15;

            Matrix M1(2,3);
            M1(0,0) = 3; M1(0,1) = 8; M1(0,2) = 13;
            M1(1,0) = 4; M1(1,1) = 9; M1(1,2) = 14;

            Matrix M2;
            M2.AssignRowSliceValuesOf(M0, 2, 2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 

            Matrix M3(5,3);
            M3(0,0) = 0; M3(0,1) = 0; M3(0,2) = 0;
            M3(1,0) = 0; M3(1,1) = 0; M3(1,2) = 0;
            M3(2,0) = 3; M3(2,1) = 8; M3(2,2) = 13;
            M3(3,0) = 4; M3(3,1) = 9; M3(3,2) = 14;
            M3(4,0) = 0; M3(4,1) = 0; M3(4,2) = 0;

            M3 += M0;
            M0.AddToRowSliceValuesOf(M1, 2,2);
            Assert::IsTrue(M3.IsEqualTo(M0, 0.0001)); 

            M2.AddWithRowSliceValuesOf(M1, 0, 2);
            Matrix M4(2, 3);
            M4(0, 0) = 6; M4(0, 1) = 16; M4(0, 2) = 26;
            M4(1, 0) = 8; M4(1, 1) = 18; M4(1, 2) = 28;
            Assert::IsTrue(M2.IsEqualTo(M4, 0.0001));

#if 0
            Matrix M5, M6, M7, M8;
            M5.AssignRowSliceValuesOf(M0, 0, 2);
            M6.AssignRowSliceValuesOf(M0, 2, 1);
            M7.AssignRowSliceValuesOf(M0, 3, 2);

            std::vector<const Matrix*> inputMatrices;
            inputMatrices.resize(3);
            inputMatrices[0] = &M5;
            inputMatrices[1] = &M6;
            inputMatrices[2] = &M7;
            M8.AssignRowStackValuesOf(inputMatrices, 0, 3);
            
            Assert::IsTrue(M8.IsEqualTo(M0, 0.0001));
#endif
        }

        TEST_METHOD(CPUAssignRepeatOf)
        {
            Matrix M0(2, 3);
            M0(0, 0) = 1; M0(0, 1) = 6; M0(0, 2) = 11;
            M0(1, 0) = 2; M0(1, 1) = 7; M0(1, 2) = 12;

            Matrix M1;
            M1.AssignRepeatOf(M0, 1, 1);
            Assert::IsTrue(M1.IsEqualTo(M0, 0.0001));

            Matrix M3(6, 6);
            M3(0, 0) = 1; M3(0, 1) = 6; M3(0, 2) = 11; M3(0, 3) = 1; M3(0, 4) = 6; M3(0, 5) = 11;
            M3(1, 0) = 2; M3(1, 1) = 7; M3(1, 2) = 12; M3(1, 3) = 2; M3(1, 4) = 7; M3(1, 5) = 12;
            M3(2, 0) = 1; M3(2, 1) = 6; M3(2, 2) = 11; M3(2, 3) = 1; M3(2, 4) = 6; M3(2, 5) = 11;
            M3(3, 0) = 2; M3(3, 1) = 7; M3(3, 2) = 12; M3(3, 3) = 2; M3(3, 4) = 7; M3(3, 5) = 12;
            M3(4, 0) = 1; M3(4, 1) = 6; M3(4, 2) = 11; M3(4, 3) = 1; M3(4, 4) = 6; M3(4, 5) = 11;
            M3(5, 0) = 2; M3(5, 1) = 7; M3(5, 2) = 12; M3(5, 3) = 2; M3(5, 4) = 7; M3(5, 5) = 12;

            M1.AssignRepeatOf(M0, 3, 2);
            Assert::IsTrue(M1.IsEqualTo(M3, 0.0001));
        }

        TEST_METHOD(CPURowElementOperations)
        {
            Matrix M0 = Matrix::RandomUniform(20, 28, -1, 1);
            Matrix M1 = Matrix::RandomUniform(1, 28, 1, 2);

            Matrix M3;
            M3.SetValue(M0);
            M3.RowElementMultiplyWith(M1);
            M3.RowElementDivideBy(M1);

            Assert::IsTrue(M0.IsEqualTo(M3, 0.0001));
        }
        TEST_METHOD(CPUColumnElementOperations)
        {
            Matrix M0 = Matrix::RandomUniform(20, 28, -1, 1);
            Matrix M1 = Matrix::RandomUniform(20, 1, 1, 2);

            Matrix M3;
            M3.SetValue(M0);
            M3.ColumnElementMultiplyWith(M1);
            M3.ColumnElementDivideBy(M1);

            Assert::IsTrue(M0.IsEqualTo(M3, 0.0001));
        }

		TEST_METHOD(CPUAssignMatrixByColumnSlice)
		{
			printf("starts here\n");
			Matrix M0 = Matrix::RandomUniform(400, 50, -100, 100); 


			vector<size_t> columnrange = { 0, 3, 5, 4 };
			Matrix M1; 
			try
			{
				M1.AssignMatrixByColumnSlice(M0, columnrange);
			}
			catch (exception& e)
			{
				printf("%s\n", e.what()); 
				Assert::Fail(); 
			}
		

			for (size_t des = 0; des < columnrange.size(); des ++)
			{
				size_t src = columnrange[des]; 

				double err = 0; 
				for (size_t r = 0; r < 400; r++)
				{
					double diff = (M0(r, src) - M1(r, des)); 
					diff *= diff; 
					err += diff; 
				}
				Assert::AreEqual(err, 0, 1e-7);
			}

		}

    };
}