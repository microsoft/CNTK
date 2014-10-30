//
// <copyright file="MatrixBLASTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Math\Matrix.h"

#pragma warning (disable: 4244 4245 4305)       // conversions and truncations; we don't care in this test project

#define epsilon 0.000001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace CNTKMathTest
{    
    TEST_CLASS(MatrixBLASTests)
    {        

    public:

        TEST_METHOD(MatrixMultiplyAndWeightedAddTest)
        {
            //Part 1: Multiply with identity matrix
            SingleMatrix A = SingleMatrix::RandomGaussian(64,23,0,2); 
            SingleMatrix B = SingleMatrix::Eye(23);
            SingleMatrix C = A*B;
            foreach_coord(i,j,A)
            {
                Assert::AreEqual<float>(A(i,j),C(i,j));
            }

            SingleMatrix M(3,3,AUTOPLACEMATRIX);
            M(0,0)=1; M(0,1)=2; M(0,2)=3;
            M(1,0)=4; M(1,1)=5; M(1,2)=6;
            M(2,0)=7; M(2,1)=8; M(2,2)=9;

            //Part 2: Compare with Octave on toy example
            SingleMatrix M1(3,4,AUTOPLACEMATRIX);
            M1(0,0)=8; M1(0,1)=9; M1(0,2)=3; M1(0,3)=45; 
            M1(1,0)=3; M1(1,1)=4; M1(1,2)=56; M1(1,3)=1; 
            M1(2,0)=-3; M1(2,1)=5; M1(2,2)=2; M1(2,3)=6;
            SingleMatrix M2 = M*M1;
            Assert::AreEqual<float>(5,M2(0,0)); Assert::AreEqual<float>(32,M2(0,1)); Assert::AreEqual<float>(121,M2(0,2)); Assert::AreEqual<float>(65,M2(0,3));
            Assert::AreEqual<float>(29,M2(1,0)); Assert::AreEqual<float>(86,M2(1,1)); Assert::AreEqual<float>(304,M2(1,2)); Assert::AreEqual<float>(221,M2(1,3));
            Assert::AreEqual<float>(53,M2(2,0)); Assert::AreEqual<float>(140,M2(2,1)); Assert::AreEqual<float>(487,M2(2,2)); Assert::AreEqual<float>(377,M2(2,3));


            //Big MultiplyTest
            size_t sz=5000;
            SingleMatrix BM1 = SingleMatrix::RandomUniform(sz,sz,-1,1);
            SingleMatrix BM2 = SingleMatrix::RandomUniform(sz,1,-2,1);
            SingleMatrix BM3 = SingleMatrix::RandomUniform(sz,1,-1,2);
            SingleMatrix::MultiplyAndWeightedAdd(1,BM1,false,BM2,false,1,BM3);
        }

        TEST_METHOD(MatrixMultiplyAndPlusAndMinus)
        {
            //Part 1: Multiply with identity matrix
            SingleMatrix A = SingleMatrix::RandomGaussian(64,23,0,2); 
            SingleMatrix B = SingleMatrix::Eye(23);
            SingleMatrix B1 = SingleMatrix::RandomUniform(64,23,-95.23,43.5);
            SingleMatrix B2 = SingleMatrix::RandomUniform(64,23,23.23,143.5);
            SingleMatrix C = A*B+B1-B2;
            foreach_coord(i,j,A)
            {
                Assert::AreEqual<float>(A(i,j)+B1(i,j)-B2(i,j),C(i,j));
            }           
            //Part 3: compare with CPUMatrix results
            CPUMatrix<float> M1C = CPUMatrix<float>::RandomUniform(429,1024,-1,1);
            CPUMatrix<float> M2C = CPUMatrix<float>::RandomUniform(429,1024,-2,2);
            CPUMatrix<float> M3C = CPUMatrix<float>::RandomUniform(429,1024,-3,1);
            CPUMatrix<float> M3C_copy(M3C);
            CPUMatrix<float>::MultiplyAndAdd(M1C,true,M2C,false,M3C);

            SingleMatrix M1(429,1024,M1C.GetArray(),matrixFlagNormal);
            SingleMatrix M2(429,1024,M2C.GetArray(),matrixFlagNormal);
            SingleMatrix M3(429,1024,M3C_copy.GetArray(),matrixFlagNormal);
            SingleMatrix::MultiplyAndAdd(M1,true,M2,false,M3);
            foreach_coord(i,j,M3)
            {
                Assert::IsTrue(fabs(M3(i,j)-M3C(i,j))<0.0005);
            }

            SingleMatrix AA = SingleMatrix::Ones(8,9);
            SingleMatrix BB = SingleMatrix::Ones(8,1);
            BB(4,0)=-5.5;
            SingleMatrix::ScaleAndAdd(1,BB,AA);
            foreach_coord(i,j,AA)
            {
                if (i!=4)
                {
                    Assert::AreEqual<float>(2,AA(i,j));
                }
                else
                {
                    Assert::AreEqual<float>(-4.5,AA(i,j));
                }
            }
        }

        TEST_METHOD(MatrixScaleAndAdd)//This test will fail without GPU
        {
            int seed = rand();
            float precision = 0.00001;
            SingleMatrix A = SingleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+0,0);
            SingleMatrix B = SingleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+1,0);
            SingleMatrix C(B);
            float alpha = 0.34213;
            SingleMatrix::ScaleAndAdd(alpha,A,C);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(fabsf(C(i,j)-(alpha*A(i,j)+B(i,j)))<precision);                
            }
            //Test 2
            SingleMatrix A1 = SingleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+2,0);
            SingleMatrix B1 = SingleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+3,0);
            SingleMatrix C1(B1); //C1==B1            
            float beta = -1.4654;
            SingleMatrix::ScaleAndAdd(alpha,A1,beta,C1); //C1=alpha*A1+beta*C1
            foreach_coord(i,j,C1)
            {                
                Assert::IsTrue(fabsf(C1(i,j)-(alpha*A1(i,j)+beta*B1(i,j)))<precision);    
                //Assert::AreEqual<float>(C1(i,j),alpha*A(i,j)+beta*B(i,j));
            }

            //precision = 0.0001;
            //Test 3 - columnwise
            SingleMatrix A2 = SingleMatrix::RandomUniform(1024,1,-12.34,55.2312,seed+4,0);
            SingleMatrix B2 = SingleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+5,0); //Column
            SingleMatrix C2(B2); //C2==B2            
            beta = 1;
            SingleMatrix::ScaleAndAdd(alpha,A2,beta,C2); //C1=alpha*A1+beta*C1
            foreach_coord(i,j,C2)
            {
                float x = C2(i,j);
                float y = (alpha*A2(i,0)+beta*B2(i,j));
                Assert::IsTrue(fabsf(x-y)<precision);    
            }        
        }

        TEST_METHOD(MatrixScaleAndAdd_double)//This test will fail without GPU
        {
            int seed = rand();
            double precision = 0.00000000001;
            DoubleMatrix A = DoubleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+0,0);
            DoubleMatrix B = DoubleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+1,0);
            DoubleMatrix C(B);
            float alpha = 0.34213;
            DoubleMatrix::ScaleAndAdd(alpha,A,C);
            foreach_coord(i,j,C)
            {
                Assert::IsTrue(fabsf(C(i,j)-(alpha*A(i,j)+B(i,j)))<precision);                
            }
            //Test 2
            DoubleMatrix A1 = DoubleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+2,0);
            DoubleMatrix B1 = DoubleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+3,0);
            DoubleMatrix C1(B1); //C1==B1            
            float beta = -1.4654;
            DoubleMatrix::ScaleAndAdd(alpha,A1,beta,C1); //C1=alpha*A1+beta*C1
            foreach_coord(i,j,C1)
            {                
                Assert::IsTrue(fabsf(C1(i,j)-(alpha*A1(i,j)+beta*B1(i,j)))<precision);    
                //Assert::AreEqual<float>(C1(i,j),alpha*A(i,j)+beta*B(i,j));
            }

            //precision = 0.0001;
            //Test 3 - columnwise
            DoubleMatrix A2 = DoubleMatrix::RandomUniform(1024,1,-12.34,55.2312,seed+4,0);
            DoubleMatrix B2 = DoubleMatrix::RandomUniform(1024,512,-12.34,55.2312,seed+5,0); //Column
            DoubleMatrix C2(B2); //C2==B2            
            beta = 1;
            DoubleMatrix::ScaleAndAdd(alpha,A2,beta,C2); //C1=alpha*A1+beta*C1
            foreach_coord(i,j,C2)
            {
                float x = C2(i,j);
                float y = (alpha*A2(i,0)+beta*B2(i,j));
                Assert::IsTrue(fabsf(x-y)<precision);    
            }        
        }


        TEST_METHOD(MatrixNorms)
        {
            SingleMatrix M0((size_t)2,(size_t)3);
            M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
            M0(1,0) = 4; M0(1,1) = 5; M0(1,2) = 6;

            SingleMatrix M3;
            M0.VectorNorm1(M3, true);
            SingleMatrix M2((size_t)1, (size_t)3);
            M2(0,0) = 5; M2(0,1) = 7; M2(0,2) = 9;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm1(M3, false);
            M2.Resize(2,1);
            M2(0,0) = 6;
            M2(1,0) = 15;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M0.VectorNorm2(M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 4.1231; 
            M2(0,1) = 5.3852; 
            M2(0,2) = 6.7082;            
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0005)); 

            M0.VectorNorm2(M3, false);
            M2.Resize(2,1);
            M2(0,0) = 3.7417;
            M2(1,0) = 8.7750;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0005)); 

            SingleMatrix M00((size_t)2,(size_t)3);            
            M00(0,0) = 1; M00(0,1) = 2; M00(0,2) = 3;
            M00(1,0) = 4; M00(1,1) = 5; M00(1,2) = 6;            
            SingleMatrix M1;
            M00.VectorMax(M1, M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 4; M2(0,1) = 5; M2(0,2) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M00.VectorMax(M1, M3, false);
            M2.Resize(2,1);
            M2(0,0) = 3;
            M2(1,0) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            /*M0.VectorMin(M1, M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 1; M2(0,1) = 2; M2(0,2) = 3;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorMin(M1, M3, false);
            M2.Resize(2,1);
            M2(0,0) = 1;
            M2(1,0) = 4;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001));*/

            M0.VectorNormInf(M3, true);
            M2.Resize(1, 3);
            M2(0,0) = 4; M2(0,1) = 5; M2(0,2) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2, 0.0001)); 

            M0.VectorNormInf(M3, false);
            M2.Resize(2,1);
            M2(0,0) = 3;
            M2(1,0) = 6;
            Assert::IsTrue(M3.IsEqualTo(M2)); 

            M00(0,0) = 1; M00(0,1) = 2; M00(0,2) = 3;
            M00(1,0) = 4; M00(1,1) = 5; M00(1,2) = 6; 
            Assert::AreEqual<float>(6,M00.MatrixNormInf());

            Assert::IsTrue(abs(M0.FrobeniusNorm() - 9.5394) < 0.0001);
            Assert::IsTrue(abs(M0.MatrixNormInf() - 6) < 0.0001);
            Assert::AreEqual<float>(21,M00.MatrixNorm1());

            Matrix<float> A = Matrix<float>::Eye(4096);
            Assert::AreEqual<long>(4096,A.MatrixNorm0());

            Matrix<float> B = Matrix<float>::Eye(5);
            Assert::AreEqual<long>(5,B.MatrixNorm0());
        }

        TEST_METHOD(MatrixInnerProductOfMatrices)
        {
            SingleMatrix V1(2,3);
            V1(0,0)=1; V1(0,1)=2; V1(0,2)=3;
            V1(1,0)=4; V1(1,1)=5; V1(1,2)=6;
            SingleMatrix V2(2,3);
            V2(0,0)=7; V2(0,1)=8; V2(0,2)=9;
            V2(1,0)=10; V2(1,1)=11; V2(1,2)=12;
            float ip = SingleMatrix::InnerProductOfMatrices(V1,V2);
            Assert::AreEqual<float>(217,ip);
        }
    };
}


