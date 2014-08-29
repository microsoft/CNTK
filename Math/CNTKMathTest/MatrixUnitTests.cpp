//
// <copyright file="MatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Math\Matrix.h"

#define epsilon 0.000001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace CNTKMathTest
{    
    TEST_CLASS(MatrixUnitTest)
    {        

    public:

        //This test should fail if you don't have CUDA GPU (or working under remote desktop)
        TEST_METHOD(MatrixConsturctors)
        {
            SingleMatrix A0;
            SingleMatrix A1(AUTOPLACEMATRIX);
            SingleMatrix A2(-1);
            SingleMatrix A3((size_t)13,(size_t)12,(short)0);

            Assert::AreEqual<size_t>(0,A0.GetNumRows());
            Assert::AreEqual<size_t>(0,A0.GetNumCols());            
            Assert::AreEqual<size_t>(0,A1.GetNumRows());
            Assert::AreEqual<size_t>(0,A2.GetNumCols());            
            Assert::AreEqual<size_t>(13,A3.GetNumRows());
            Assert::AreEqual<size_t>(12,A3.GetNumCols());            
        }

        TEST_METHOD(MatrixMoveTest)
        {
            //no moves required
            SingleMatrix a;
            SingleMatrix b;
            b.Resize(100,100);
            std::swap(a,b);

            //potentially a move is required
            SingleMatrix A;
            SingleMatrix B;
            B.Resize(100,100);
            B(12,13)=14; //this will move whole matrix B from GPU to CPU
            std::swap(A,B);   // this will not only swap A and B but will put them to their preferred device (GPU if present)

            //This is deep copy, not move
            SingleMatrix a1;
            SingleMatrix b1;
            b1.Resize(12,34);
            b1=a1;

            SingleMatrix a2;
            SingleMatrix b2;
            b2.Resize(12,34);
            b2(2,3)=9;
            b2=a2;
        }


        TEST_METHOD(MatrixAssignmentOperatorsAndInitializers)
        {   
            //Zeros    
            SingleMatrix A0 = SingleMatrix::Zeros(12,32);           
            Assert::AreEqual<size_t>(12,A0.GetNumRows());
            Assert::AreEqual<size_t>(32,A0.GetNumCols());      
            foreach_coord(i,j,A0)
            {
                Assert::AreEqual<float>(0,A0(i,j));
            }

            //Eye
            SingleMatrix A1 = SingleMatrix::Eye(56);
            Assert::AreEqual<size_t>(56,A1.GetNumRows());
            Assert::AreEqual<size_t>(56,A1.GetNumCols());      
            foreach_coord(i,j,A1)
            {
                if (i!=j)
                {
                    float x = A1(i,j);
                    Assert::AreEqual<float>(0,A1(i,j));
                }
                else 
                {
                    float x = A1(i,j);
                    Assert::AreEqual<float>(1,A1(i,j));
                }
            }        

            //Ones
            SingleMatrix A2  = SingleMatrix::Ones(12,56);
            Assert::AreEqual<size_t>(12,A2.GetNumRows());
            Assert::AreEqual<size_t>(56,A2.GetNumCols());      
            foreach_coord(i,j,A2)
            {
                Assert::AreEqual<float>(1,A2(i,j));
            }            
                      
            //SetGaussianRandomValue            
            SingleMatrix A3= SingleMatrix::RandomGaussian(640,230,0,2,-1);            
            float avg = 0;
            foreach_coord(i,j,A3)
            {
                avg+=A3(i,j);
            }
            avg/=(640*230);
            float std=0;
            foreach_coord(i,j,A3)
            {
                std+=((A3(i,j)-avg)*(A3(i,j)-avg)/(640*230));
            }            
            std=sqrt(std);            
            //Assert::IsTrue(fabs(avg-0)<0.01);
            //WARNING: The deviance seems to be off for both CPU and GPU implementations
            //Assert::IsTrue(fabs(std-2)<0.01);

            //RandomUniform
            SingleMatrix A4 = SingleMatrix::RandomUniform(435, 100, -26.3, 30.2);
            bool has_small=false;
            bool has_big=false;
            foreach_coord(i,j,A4)
            {
                float x = A4(i,j);
                Assert::IsTrue((A4(i,j)>=-26.3)&&(A4(i,j)<30.2));
                if (A4(i,j)<-3)
                    has_small=true;
                if (A4(i,j)>3)
                    has_big=true;
            }
            Assert::IsTrue(has_small);
            Assert::IsTrue(has_big);

            //RandomUniform
            SingleMatrix A5(429,1024);
            A5.SetUniformRandomValue(-0.01,0.01);
            foreach_coord(i,j,A5)
            {
                Assert::IsTrue(A5(i,j)<=0.01&&A5(i,j)>=-0.01);
            }

            //Check that seed allows results reproduce
            //!!!WE NOW INITIALIZE SEED PER PROCESS!!!
            /*
            SingleMatrix A6 = SingleMatrix::RandomUniform(5,6,-1,3,1234);
            SingleMatrix A7 = SingleMatrix::RandomUniform(5,6,-1,3,1234);
            SingleMatrix A8 = SingleMatrix::RandomUniform(5,6,-1,3,12346);
            SingleMatrix A9 = SingleMatrix::RandomUniform(5,6,-1,3);

            Assert::IsTrue(A6.IsEqualTo(A7));
            //Assert::IsTrue(!A8.IsEqualTo(A7));  */

        }

        TEST_METHOD(MatrixSetValueMethods)
        {
            //void SetValue(const ElemType v);
            SingleMatrix A((size_t)32,(size_t)12);
            Assert::AreEqual<size_t>(32,A.GetNumRows());
            Assert::AreEqual<size_t>(12,A.GetNumCols());
            Assert::AreEqual<size_t>(12*32,A.GetNumElements());
            float v= -32.3451;
            A.SetValue(v);
            foreach_coord(i,j,A)
            {
                Assert::AreEqual<float>(v,A(i,j));
            }

            //void SetValue(const Matrix<ElemType>& deepCopyFrom);
            SingleMatrix B;
            B.SetValue(A);            
            foreach_coord(i,j,B)
            {
                Assert::AreEqual<float>(v,B(i,j));
            }

            //void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, const bool srcIsColMajor);
            float *arr = new float[6];
            arr[0]=123;arr[1]=0.23;arr[2]=-22;arr[3]=63;arr[4]=43.42;
            arr[5]=324.3;arr[6]=99912;
            B.SetValue(2,3,arr,matrixFlagNormal);

            SingleMatrix B1;
            B1.SetValue(2,3,arr);
            foreach_coord(i,j,B)
            {
                Assert::AreEqual<float>(arr[IDX2C(i,j,2)],B(i,j));
                Assert::AreEqual<float>(arr[IDX2C(i,j,2)],B1(i,j));
            }

            SingleMatrix BBBB = SingleMatrix::Zeros(6,8);
            BBBB.SetColumn(arr,3);
            for (int i=0;i<6;++i)
            {
                Assert::AreEqual<float>(arr[i],BBBB(i,3));
            }
            

            //void SetDiagonalValue(const ElemType v);            
            SingleMatrix C(4,4,AUTOPLACEMATRIX);
            float val = -0.00332;
            C.SetDiagonalValue(val);
            foreach_coord(i,j,C)
            {
                if (i==j)
                    Assert::AreEqual<float>(val,C(i,j));
                else
                    Assert::AreEqual<float>(0,C(i,j));
            }
            
            //void SetDiagonalValue(Matrix<ElemType>& vector);
            SingleMatrix D(4,1,AUTOPLACEMATRIX);
            float val1=43.324;
            D.SetValue(val1);
            C.SetDiagonalValue(D);
            foreach_coord(i,j,C)
            {
                if (i==j)
                    Assert::AreEqual<float>(val1,C(i,j));
                else
                    Assert::AreEqual<float>(0,C(i,j));
            }         

            SingleMatrix C1(5,5,AUTOPLACEMATRIX);
            SingleMatrix D1(1,5,AUTOPLACEMATRIX);
            float val2=0.53;            
            D1=D1.Transpose();
            D1.SetValue(val2);
            C1.SetDiagonalValue(D1);
            foreach_coord(i,j,C1)
            {
                if (i==j)
                    Assert::AreEqual<float>(val2,C1(i,j));
                else
                    Assert::AreEqual<float>(0,C1(i,j));
            }
        }

        TEST_METHOD(MatrixTransposeTest)
        {            
            SingleMatrix A= SingleMatrix::RandomGaussian(64,23,0,2);   
            Assert::AreEqual<size_t>(64,A.GetNumRows());
            Assert::AreEqual<size_t>(23,A.GetNumCols());

            SingleMatrix B=A.Transpose();

            Assert::AreEqual<size_t>(23,B.GetNumRows());
            Assert::AreEqual<size_t>(64,B.GetNumCols());

            foreach_coord(i,j,A)
            {
                Assert::AreEqual<size_t>(A(i,j),B(j,i));
            }
        }

        TEST_METHOD(MatrixMultiAndDiv)
        {
            SingleMatrix M0(2,3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            SingleMatrix M00(2,3);
            M00(0,0) = 10; M00(0,1) = 20; M00(0,2) = 30;
            M00(1,0) = 40; M00(1,1) = 50; M00(1,2) = 60;

            SingleMatrix M1((size_t)2,(size_t)3);
            M1.Reshape(3,2);
            M1(0,0) = 11; M1(0,1) = 15; 
            M1(1,0) = 14; M1(1,1) = 13; 
            M1(2,0) = 12; M1(2,1) = 16; 

            SingleMatrix M2((size_t)2,(size_t)2);
            M2(0,0) = 75; M2(0,1) = 89; 
            M2(1,0) = 186; M2(1,1) = 221; 

            SingleMatrix M3 = M0 * M1;
            Assert::IsTrue(M3.IsEqualTo(M2));  

            M3 = M0 * 10;
            Assert::IsTrue(M3.IsEqualTo(M00));  

            M3 = M3 / 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 *= 10;
            Assert::IsTrue(M3.IsEqualTo(M00));  

            M3 /= 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            SingleMatrix::MultiplyAndWeightedAdd(1, M0, false, M1, false, 0, M3);
            Assert::IsTrue(M3.IsEqualTo(M2));  

            M1.Reshape(2,3);
            SingleMatrix::MultiplyAndWeightedAdd(1, M0, false, M1, true, 0, M3);
            M2(0,0) = 74; M2(0,1) = 92; 
            M2(1,0) = 182; M2(1,1) = 227; 
            Assert::IsTrue(M3.IsEqualTo(M2));  

            SingleMatrix::MultiplyAndWeightedAdd(10, M0, false, M1, true, 2, M3);
            M2(0,0) = 888; M2(0,1) = 1104; 
            M2(1,0) = 2184; M2(1,1) = 2724; 
            Assert::IsTrue(M3.IsEqualTo(M2));  

            SingleMatrix::MultiplyAndWeightedAdd(1, M0, true, M1, false, 0, M3);
            M2.Resize(3,3);
            M2(0,0) = 67; M2(0,1) = 72; M2(0,2) = 77; 
            M2(1,0) = 92; M2(1,1) = 99; M2(1,2) = 106; 
            M2(2,0) = 117; M2(2,1) = 126; M2(2,2) = 135; 
            Assert::IsTrue(M3.IsEqualTo(M2));    

            //Multiplications of arbitrary matrix with 1x1 matrix 

            SingleMatrix A((size_t)2,(size_t)3);
            A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
            A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;

            SingleMatrix B = SingleMatrix::Eye(1);
            
            SingleMatrix C=A*B;
            Assert::IsTrue(C.IsEqualTo(A));
            C=B*A;
            Assert::IsTrue(C.IsEqualTo(A));
            B(0,0)=0.5;
            B.InplaceAbs();
            C=A*B;
            
            SingleMatrix D((size_t)2,(size_t)3);
            D(0,0) = 0.5; D(0,1) = 1; D(0,2) = 1.5;
            D(1,0) = 2; D(1,1) = 2.5; D(1,2) = 3;
            Assert::IsTrue(C.IsEqualTo(D));       
        }

        TEST_METHOD(MatrixTranspose)
        {
            SingleMatrix M0((size_t)2,(size_t)3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            SingleMatrix M1((size_t)3,(size_t)2);
            M1(0,0) = 1; M1(0,1) = 4; 
            M1(1,0) = 2; M1(1,1) = 5;
            M1(2,0) = 3; M1(2,1) = 6;

            SingleMatrix M2 = M0.Transpose();
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 
           
            M2.AssignTransposeOf(M1);
            Assert::IsTrue(M2.IsEqualTo(M0, 0.0001)); 
        }

        TEST_METHOD(MatrixAddAndSub)
        {
            SingleMatrix M0((size_t)2,(size_t)3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            SingleMatrix M1((size_t)2,(size_t)3);
            M1(0,0) = 11; M1(0,1) = 12; M1(0,2) = 13;
            M1(1,0) = 14; M1(1,1) = 15; M1(1,2) = 16;

            SingleMatrix M2((size_t)2,(size_t)3);
            M2(0,0) = 12; M2(0,1) = 14; M2(0,2) = 16;
            M2(1,0) = 18; M2(1,1) = 20; M2(1,2) = 22;
            
            SingleMatrix M3 = M2 - M0;            
            Assert::IsTrue(M3.IsEqualTo(M1)); 

            M3 += M0;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M3 = M0 + 10;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3 -= 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            M3 = M1 + M0;
            Assert::IsTrue(M3.IsEqualTo(M2)); 
            SingleMatrix M4=SingleMatrix::Eye(3);
            SingleMatrix M5 = M0*M4 + M1;

            M3 -= M0;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            M3 = M1 - 10;
            Assert::IsTrue(M3.IsEqualTo(M0));  

            SingleMatrix M33(M3); //M4==M3
            M3 += 10;
            Assert::IsTrue(M3.IsEqualTo(M1));  

            SingleMatrix M55=SingleMatrix::Eye(1);
            M55(0,0)=10;
            M55.InplaceAbs();
            M33 += M55;//M5 is 1x1 matrix
            Assert::IsTrue(M33.IsEqualTo(M1));  
            M33 -= 10;
            M33 = M33 + 10;
            Assert::IsTrue(M33.IsEqualTo(M1));  
        }

        TEST_METHOD(MatrixElementOps)
        {
            SingleMatrix M0((size_t)2,(size_t)3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            SingleMatrix M00((size_t)2,(size_t)3);
            M00(0,0) = 1.0; M00(0,1) = 1/2.0; M00(0,2) = 1/3.0;
            M00(1,0) = 1/4.0; M00(1,1) = 1/5.0; M00(1,2) = 1/6.0;

            SingleMatrix M1(2,3);
            M1(0,0) = 1; M1(0,1) = 1; M1(0,2) = 1;
            M1(1,0) = 1; M1(1,1) = 1; M1(1,2) = 1;

            SingleMatrix M3;
            M3.AssignElementProductOf(M0, M00);
            Assert::IsTrue(M3.IsEqualTo(M1, 0.0001)); 

            SingleMatrix M4=SingleMatrix::Zeros(2,3);            
            M4 = M4.AddElementProductOf(M0,M00);
            Assert::IsTrue(M4.IsEqualTo(M1,0.0001));            

            M3 = M0 ^ 4;
            SingleMatrix M2((size_t)2,(size_t)3);
            M2(0,0) = 1; M2(0,1) = 16; M2(0,2) = 81;
            M2(1,0) = 256; M2(1,1) = 625; M2(1,2) = 1296;
            Assert::IsTrue(M3.IsEqualTo(M2,0.001)); 

            M3.SetValue(M0);
            M3 ^= 4;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.001)); 

            M3.SetValue(M0);
            M3.ElementMultiplyWith(M00);
            Assert::IsTrue(M3.IsEqualTo(M1, 0.001)); 

            M3.SetValue(M0);
            M3.ElementInverse();
            Assert::IsTrue(M3.IsEqualTo(M00, 0.001)); 

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
            M3.InplaceSoftmax(true);
            M2(0,0) = 0.0474; M2(0,1) = 0.0474; M2(0,2) = 0.0474;
            M2(1,0) = 0.9526; M2(1,1) = 0.9526; M2(1,2) = 0.9526;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 
           
            M3.SetValue(M0);
            M3.InplaceSoftmax(false);
            M2(0,0) = 0.0900; M2(0,1) = 0.2447; M2(0,2) = 0.6652;
            M2(1,0) = 0.0900; M2(1,1) = 0.2447; M2(1,2) = 0.6652;
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
            Assert::IsTrue(M3.IsEqualTo(M2, 0.001)); 

            M3.SetValue(M0);
            M3.InplaceTruncateTop(4);
            M2(0,0) = 1; M2(0,1) = 2; M2(0,2) = 3;
            M2(1,0) = 4; M2(1,1) = 4; M2(1,2) = 4;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.001)); 
        }

        TEST_METHOD(MatrixColumnElementMultiplyWithVsCPUMatrix)
        {
            CPUMatrix<float> MCPU = CPUMatrix<float>::RandomUniform(429,1024,-3.4,1);
            CPUMatrix<float> ACPU = CPUMatrix<float>::Ones(429,1);
            CPUMatrix<float> MCPU_copy(MCPU);
            MCPU.ColumnElementMultiplyWith(ACPU);
            Assert::IsTrue(MCPU_copy.IsEqualTo(MCPU,0.0001));

            //            
            Matrix<float> M = Matrix<float>::RandomUniform(429,1024,-3.4,1);
            Matrix<float> A = Matrix<float>::Ones(429,1);
            Matrix<float> M_copy(M);
            M.ColumnElementMultiplyWith(A);
            Assert::IsTrue(M_copy.IsEqualTo(M,0.0001));                    

            CPUMatrix<float> MC1 = CPUMatrix<float>::RandomUniform(429,1024,-3.4,1);
            CPUMatrix<float> MC2 = CPUMatrix<float>::RandomUniform(429,1,0,3);
            MC1.ColumnElementMultiplyWith(MC2);

            Matrix<float> M1(MC1.GetNumRows(),MC1.GetNumCols(),MC1.GetArray(),matrixFlagNormal);
            Matrix<float> M2(MC2.GetNumRows(),MC2.GetNumCols(),MC2.GetArray(),matrixFlagNormal);
            M1.ColumnElementMultiplyWith(M2);
            
            foreach_coord(i,j,M2)
            {
                 Assert::IsTrue(fabs(M2(i,j)-MC2(i,j))<0.00001);
            }
        }

        TEST_METHOD(MatrixAssignXOf)
        {
            //AssignDifferenceOf
            Matrix<float> A = Matrix<float>::RandomUniform(429,1024,5,32);
            Matrix<float> B = Matrix<float>::RandomUniform(429,1024,5,32);
            Matrix<float> C;
            C.AssignDifferenceOf(A,B);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(C(i,j)==A(i,j)-B(i,j));
            }
            float x=234.2;
            C.AssignDifferenceOf(A,x);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(C(i,j)==A(i,j)-x);
            }

            C.AssignDifferenceOf(x,A);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(C(i,j)==x-A(i,j));
            }
            
            C.AssignDifferenceOf(1,A);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(C(i,j)==1-A(i,j));
            }
            //

            //AssignElementProductOf
            C.AssignElementProductOf(A,B);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(C(i,j)=A(i,j)*B(i,j));
            }

            //AddElementProductOf
            Matrix<float> C_copy(C);
            C.AddElementProductOf(A,B);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(C(i,j)==C_copy(i,j)+A(i,j)*B(i,j));
            }

            //AssignSigmoidOf
            CPUMatrix<float> AC = CPUMatrix<float>::RandomUniform(429,1024,5,32);
            CPUMatrix<float> BC = CPUMatrix<float>::RandomUniform(429,1024,-5,12);
            Matrix<float> D(AC.GetNumRows(),AC.GetNumCols(),AC.GetArray(),matrixFlagNormal);
            Matrix<float> E(BC.GetNumRows(),BC.GetNumCols(),BC.GetArray(),matrixFlagNormal);
            AC.AssignSigmoidOf(BC);
            D.AssignSigmoidOf(E);
            foreach_coord(i,j,AC)
            {                
                Assert::IsTrue(fabs(AC(i,j)-D(i,j))<0.00001);
            }

            //AssignSignOf
            Matrix<float> M1 = Matrix<float>::RandomUniform(42,12,-5,12);
            Matrix<float> M2(4,5);
            M2.AssignSignOf(M1);
            foreach_coord(i,j,M1)
            {
                float v = M1(i,j);
                float expected = (v == (float)0? (float)0 : (v > 0? (float)1 : (float)(-1)));
                float actual = M2(i,j);
                Assert::AreEqual<float>(expected, actual);
            }

            Matrix<float> M3 = Matrix<float>::RandomUniform(42,12,-5,2);;
            Matrix<float> M4(M3);
            M3.AddSignOf(M1);
            foreach_coord(i,j,M3)
            {
                float v = M1(i,j);
                Assert::AreEqual<float>(M4(i,j)+(v == (float)0? (float)0 : (v > 0? (float)1 : (float)(-1))),M3(i,j));
            }
            

            //AssignTruncateBottom and Top
            Matrix<float> M5(2,2);
            M5(0,0)=1;     M5(0,1)=2;                    
            M5(1,0)=3;     M5(1,1)=4;

            Matrix<float> M6;
            M6.AssignTruncateBottomOf(M5,3);
            Assert::AreEqual<float>(3,M6(0,0));
            Assert::AreEqual<float>(3,M6(0,1));
            Assert::AreEqual<float>(3,M6(1,0));
            Assert::AreEqual<float>(4,M6(1,1));

            
            Matrix<float> M7;
            M7.AssignTruncateTopOf(M5,3);
            Assert::AreEqual<float>(1,M7(0,0));
            Assert::AreEqual<float>(2,M7(0,1));
            Assert::AreEqual<float>(3,M7(1,0));
            Assert::AreEqual<float>(3,M7(1,1));
        }

        TEST_METHOD(MatrixSum)
        {
            Matrix<float> M=Matrix<float>::Ones(429,1024,0);
            float sum = M.SumOfElements();
            Assert::AreEqual<float>(429*1024,sum);

            CPUMatrix<float> MCPU=CPUMatrix<float>::Ones(429,1024);
            float sumCPU = MCPU.SumOfElements();
            Assert::AreEqual<float>(429*1024,sumCPU);

            Matrix<float> M1=Matrix<float>::Ones(42,332);
            M1*=-1;
            float sum1 = M1.SumOfElements();
            Assert::AreEqual<float>(-1*42*332,sum1);

            Matrix<float> M2=Matrix<float>::Ones(3,2);
            M2*=-1;
            float sum2 = M2.SumOfElements();
            Assert::AreEqual<float>(-1*3*2,sum2);
        }

        TEST_METHOD(MatrixColumnSlice)
        {
            float *fArray = new float[6];
            fArray[0] = 1; fArray[1] = 4; fArray[2] = 2; 
            fArray[3] = 5; fArray[4] = 3; fArray[5] = 6; 
           Matrix<float> M0(2, 3, fArray, matrixFlagNormal);

            Matrix<float> M1(2, 2, fArray, matrixFlagNormal);

            Matrix<float> M2 = M0.ColumnSlice(0,2);
            Assert::IsTrue(M2.IsEqualTo(M1, 0.0001)); 

            Matrix<float> M3(2, 2, fArray+2, matrixFlagNormal);

            M2 = M0.ColumnSlice(1,2);
            Assert::IsTrue(M2.IsEqualTo(M3, 0.0001)); 


            size_t k=100, n=20, m=50;

            Matrix<float> AG((size_t)k,(size_t)n);
            AG.SetUniformRandomValue(-1,1);

            Matrix<float> BG((size_t)n,(size_t)m);
            BG.SetUniformRandomValue(-1,1);

            Matrix<float> CG((size_t)k,(size_t)m);
            CG.SetUniformRandomValue(-1,1);
            Matrix<float> DG((size_t)k,(size_t)m);
            DG.SetValue(CG);

            Matrix<float>::MultiplyAndAdd(AG, false, BG, false, DG);

            for (int i=0; i<m; i++)
            {
                Matrix<float> col_BG = BG.ColumnSlice(i,1);
                Matrix<float> col_CG = CG.ColumnSlice(i,1);
                Matrix<float>::MultiplyAndAdd(AG, false, col_BG, false, col_CG);
            }
            Assert::IsTrue(CG.IsEqualTo(DG, 0.0001)); 
        }

        TEST_METHOD(MatrixKhatriRaoProduct)
        {
            float *fArray = new float[24];
            fArray[0] = 0.8147f; fArray[3] = 0.9134f; fArray[6] = 0.2785f; fArray[9] = 0.9649f;
            fArray[1] = 0.9058f; fArray[4] = 0.6324f; fArray[7] = 0.5469f; fArray[10] = 0.1576f;
            fArray[2] = 0.1270f; fArray[5] = 0.0975f; fArray[8] = 0.9575f; fArray[11] = 0.9706f;
            Matrix<float> A(3,4,fArray);

            fArray[0] = 0.9572f; fArray[2] = 0.8003f; fArray[4] = 0.4218f; fArray[6] = 0.7922f;
            fArray[1] = 0.4854f; fArray[3] = 0.1419f; fArray[5] = 0.9157f; fArray[7] = 0.9595f;
            Matrix<float> B(2,4,fArray);

            fArray[0] = 0.7798f; fArray[6] =  0.7310f; fArray[12] = 0.1175f; fArray[18] = 0.7644f;
            fArray[1] = 0.8670f; fArray[7] =  0.5061f; fArray[13] = 0.2307f; fArray[19] = 0.1249f;
            fArray[2] = 0.1215f; fArray[8] =  0.0781f; fArray[14] = 0.4038f; fArray[20] = 0.7689f;
            fArray[3] = 0.3954f; fArray[9] =  0.1296f; fArray[15] = 0.2550f; fArray[21] = 0.9258f;
            fArray[4] = 0.4396f; fArray[10] = 0.0897f; fArray[16] = 0.5008f; fArray[22] = 0.1512f;
            fArray[5] = 0.0616f; fArray[11] = 0.0138f; fArray[17] = 0.8768f; fArray[23] = 0.9313f;
            Matrix<float> D(6,4, fArray);

            Matrix<float> C;
            C.AssignKhatriRaoProductOf(A, B);
            Assert::IsTrue(C.IsEqualTo(D, 0.0001f)); 
        }
        TEST_METHOD(MatrixAddColumnReshapeProductOf)
        {
            float *fArray = new float[12];
            fArray[0] = 0.6557f; fArray[6] =  0.7431f; 
            fArray[1] = 0.0357f; fArray[7] =  0.3922f; 
            fArray[2] = 0.8491f; fArray[8] =  0.6555f; 
            fArray[3] = 0.9340f; fArray[9]  = 0.1712f; 
            fArray[4] = 0.6787f; fArray[10] = 0.7060f; 
            fArray[5] = 0.7577f; fArray[11] = 0.0318f; 
            Matrix<float> A(6,2,fArray);

            fArray[0] = 0.2769f; fArray[3] = 0.8235f; 
            fArray[1] = 0.0462f; fArray[4] = 0.6948f; 
            fArray[2] = 0.0971f; fArray[5] = 0.3171f; 
            Matrix<float> B(3,2,fArray);

            fArray[0] = 0.2867f; fArray[2] = 1.2913f; 
            fArray[1] = 0.1266f; fArray[3] = 0.4520f; 
            Matrix<float> D0(2,2,fArray);

            fArray[0] = 0.2657f; fArray[2] = 1.0923f; 
            fArray[1] = 0.3636f; fArray[3] = 0.6416f; 
            Matrix<float> D1(2,2,fArray);

            Matrix<float> C(2,2);
            C.SetValue(0.0f);
            C.AddColumnReshapeProductOf(A, B, false);
            Assert::IsTrue(C.IsEqualTo(D0, 0.0001f)); 

            C.SetValue(0.0f);
            C.AddColumnReshapeProductOf(A, B, true);
            Assert::IsTrue(C.IsEqualTo(D1, 0.0001f)); 
        }
    };
}