//
// <copyright file="MatrixUnitTests.cpp" company="Microsoft">
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
    TEST_CLASS(MatrixUnitTest)
    {        

    public:

        //This test should fail if you don't have CUDA GPU (or working under remote desktop)
        TEST_METHOD(MatrixChangeModesBetweenDenseAndSparseTests_Simple)
        {
            Matrix<float> A;
            A.AssignTruncateBottomOf(Matrix<float>::RandomUniform(4096,2048,-3,0.1,0),0);
            long n0 = A.MatrixNorm0();
            Assert::IsTrue(MatrixType::DENSE==A.GetMatrixType()); 
            A.SwitchToMatrixType(MatrixType::SPARSE);
            Assert::IsTrue(MatrixType::SPARSE==A.GetMatrixType());
            long n1 = A.MatrixNorm0();
            Assert::AreEqual<long>(n0,n1);
            A.SwitchToMatrixType(MatrixType::DENSE);
            Assert::IsTrue(MatrixType::DENSE==A.GetMatrixType());            
        }

        TEST_METHOD(MatrixSparseTimesDense)
        {
            Matrix<float> Ad; //DENSE
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(4096,2048,-3,0.1,0),0);//DENSE
            Matrix<float> As(Ad);//DENSE
            As.SwitchToMatrixType(MatrixType::SPARSE);  //!!! MATRIX As becomes sparse
            Matrix<float> B = Matrix<float>::RandomGaussian(2048,128,1,4); //DENSE
            Matrix<float> C = Matrix<float>::RandomGaussian(4096,128,1,2); //DENSE
            Matrix<float> C1(C); //DENSE

            float alpha = 0.3, beta = 2;
            bool transposeA=false, transposeB=false;
            Matrix<float>::MultiplyAndWeightedAdd(alpha,Ad,transposeA,B,transposeB,beta,C); // DENSE*DENSE
            Matrix<float>::MultiplyAndWeightedAdd(alpha,As,transposeA,B,transposeB,beta,C1);// SPARSE*DENSE            
            Assert::IsTrue(C1.IsEqualTo(C,0.00001));            
        }

        TEST_METHOD(MatrixDenseTimesSparse)
        {
            Matrix<float> Ad;
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-3,0.1,0),0);
            Matrix<float> As(Ad);
            As.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC);

            Matrix<float> B = Matrix<float>::RandomGaussian(2048,1024,1,4);
            Matrix<float> C = Matrix<float>::RandomGaussian(2048,2048,1,2);
            Matrix<float> C1(C);

            float alpha = 0.3, beta = 0;
            bool transposeA=false, transposeB=false;
            Matrix<float>::MultiplyAndWeightedAdd(alpha,B,transposeA,Ad,transposeB,beta,C);
            Matrix<float>::MultiplyAndWeightedAdd(alpha,B,transposeA,As,transposeB,beta,C1);            
            Assert::IsTrue(C1.IsEqualTo(C,0.0001));  

            alpha = 3.3, beta = 1.3;            
            Matrix<float>::MultiplyAndWeightedAdd(alpha,B,transposeA,Ad,transposeB,beta,C);
            Matrix<float>::MultiplyAndWeightedAdd(alpha,B,transposeA,As,transposeB,beta,C1);            
            Assert::IsTrue(C1.IsEqualTo(C,0.00005)); //Seems like bad precision
        }

        TEST_METHOD(CPUMatrixDenseTimesSparse)
        {
            Matrix<float> Ad(CPUDEVICE);
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024, 2048, -3, 0.1, 0), 0);
            Matrix<float> As(Ad);
            As.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC);

            Matrix<float> B = Matrix<float>::RandomGaussian(2048, 1024, 1, 4, USE_TIME_BASED_SEED, CPUDEVICE);
            Matrix<float> C = Matrix<float>::RandomGaussian(2048, 2048, 1, 2, USE_TIME_BASED_SEED, CPUDEVICE);
            Matrix<float> C1(C);

            float alpha = 0.3, beta = 0;
            bool transposeA = false, transposeB = false;
            Matrix<float>::MultiplyAndWeightedAdd(alpha, B, transposeA, Ad, transposeB, beta, C);
            Matrix<float>::MultiplyAndWeightedAdd(alpha, B, transposeA, As, transposeB, beta, C1);
            Assert::IsTrue(C1.IsEqualTo(C, 0.0001));

            alpha = 3.3, beta = 1.3;
            Matrix<float>::MultiplyAndWeightedAdd(alpha, B, transposeA, Ad, transposeB, beta, C);
            Matrix<float>::MultiplyAndWeightedAdd(alpha, B, transposeA, As, transposeB, beta, C1);

            // TODO IsEqualTo NYI
            // Assert::IsTrue(C1.IsEqualTo(C, 0.00005));
        }
        
        TEST_METHOD(CPUMatrixDenseTimesSparseAsSparse)
        {
            Matrix<float> Ad(CPUDEVICE);
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(2048, 1024, -3, 0.1, 0), 0);

            Matrix<float> As(Ad);
            As.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC);

            Matrix<float> B = Matrix<float>::RandomGaussian(2048, 1024, 1, 4, USE_TIME_BASED_SEED, CPUDEVICE);
            Matrix<float> AsCsc = Matrix<float>::RandomGaussian(2048, 2048, 1, 2, USE_TIME_BASED_SEED, CPUDEVICE);
            Matrix<float> AsBlock(CPUDEVICE);
            AsBlock.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol);

            float alpha = 0.3, beta = 0;
            bool transposeA = false, transposeB = true;
            Matrix<float>::MultiplyAndWeightedAdd(alpha, B, transposeA, As, transposeB, beta, AsBlock);
            Matrix<float>::MultiplyAndWeightedAdd(alpha, B, transposeA, As, transposeB, beta, AsCsc);

            // TODO IsEqualTo NYI
            // Assert::IsTrue(AsBlock.IsEqualTo(AsCsc, 0.0001));
        }

        TEST_METHOD(MatrixSparseTimesSparse)
        {
            Matrix<float> Ad;
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-3,0.1,0),0);
            Matrix<float> As(Ad);

            Matrix<float> Bd;
            Bd.AssignTruncateBottomOf(Matrix<float>::RandomUniform(2048,1024,-5,0.4,0),0);
            Matrix<float> Bs(Bd);

            Matrix<float> Cd;
            Cd.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,1024,-4,0.2,0),0);
            Matrix<float> Cs(Cd);

            float alpha = 2.4, beta=0;
            bool transposeA = false, transposeB=false;
            Matrix<float>::MultiplyAndWeightedAdd(alpha,Ad,transposeA,Bd,transposeB,beta,Cd);

            As.SwitchToMatrixType(MatrixType::SPARSE);
            Bs.SwitchToMatrixType(MatrixType::SPARSE);
            Cs.SwitchToMatrixType(MatrixType::SPARSE);

            Matrix<float>::MultiplyAndWeightedAdd(alpha,As,transposeA,Bs,transposeB,beta,Cs);
            Cs.SwitchToMatrixType(MatrixType::DENSE);
            Assert::IsTrue(Cs.IsEqualTo(Cd,0.00001));  


            alpha = 2.4, beta=3.4; 
            Cs.SwitchToMatrixType(MatrixType::SPARSE);            
            Matrix<float>::MultiplyAndWeightedAdd(alpha,Ad,transposeA,Bd,transposeB,beta,Cd);

            As.SwitchToMatrixType(MatrixType::SPARSE);
            Bs.SwitchToMatrixType(MatrixType::SPARSE);
            Cs.SwitchToMatrixType(MatrixType::SPARSE);

            Matrix<float>::MultiplyAndWeightedAdd(alpha,As,transposeA,Bs,transposeB,beta,Cs);
            Cs.SwitchToMatrixType(MatrixType::DENSE);
            Assert::IsTrue(Cs.IsEqualTo(Cd,0.00001)); 
        }

        TEST_METHOD(MatrixSparsePlusSparse)
        {
            Matrix<float> Ad;
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-3,0.1,0),0);
            Matrix<float> As(Ad);

            Matrix<float> Bd;
            Bd.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-5,0.4,0),0);
            Matrix<float> Bs(Bd);

            float alpha = 1.0*rand() / RAND_MAX;
            Matrix<float>::ScaleAndAdd(alpha,Ad,Bd);

            As.SwitchToMatrixType(MatrixType::SPARSE);
            Bs.SwitchToMatrixType(MatrixType::SPARSE);
            Matrix<float>::ScaleAndAdd(alpha,As,Bs);

            Bs.SwitchToMatrixType(MatrixType::DENSE);
            Assert::IsTrue(Bs.IsEqualTo(Bd,0.00001));
        }

        TEST_METHOD(MatrixDensePlusSparse)
        {
            Matrix<float> Ad;
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-3,0.1,0),0);            

            Matrix<float> Bd;
            Bd.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-5,0.4,0),0);
            Matrix<float> Bs(Bd);

            float alpha = 1.0*rand() / RAND_MAX;
            Matrix<float>::ScaleAndAdd(alpha,Ad,Bd);

            Bs.SwitchToMatrixType(MatrixType::SPARSE);
            Matrix<float>::ScaleAndAdd(alpha,Ad,Bs);

            Bs.SwitchToMatrixType(MatrixType::DENSE);
            Assert::IsTrue(Bs.IsEqualTo(Bd,0.00001));
        }

        TEST_METHOD(MatrixSparsePlusDense)
        {
            Matrix<float> Ad;
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-3,0.1,0),0);            
            Matrix<float> As(Ad);

            Matrix<float> Bd;
            Bd.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-5,0.4,0),0);
            Matrix<float> Bd1(Bd);

            float alpha = 1.0*rand() / RAND_MAX;
            Matrix<float>::ScaleAndAdd(alpha,Ad,Bd);

            As.SwitchToMatrixType(MatrixType::SPARSE);
            Matrix<float>::ScaleAndAdd(alpha,As,Bd1);

            Assert::IsTrue(Bd1.IsEqualTo(Bd,0.00001));
        }

        TEST_METHOD(MatrixSparseElementWisePower)
        {
            Matrix<float> Ad;
            Ad.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-3,0.1,0),0);            
            Matrix<float> As(Ad);
            As.SwitchToMatrixType(MatrixType::SPARSE);
            
            Matrix<float> Bd;
            Bd.AssignTruncateBottomOf(Matrix<float>::RandomUniform(1024,2048,-5,0.4,0),0);
            Matrix<float> Bs(Bd);
            Bs.SwitchToMatrixType(MatrixType::SPARSE);

            Ad^=2.3; //DENSE
            As^=2.3; //SPARSE
            Assert::IsTrue(As.IsEqualTo(Ad,0.00001));
            Assert::IsTrue(Ad.IsEqualTo(As,0.00001));

            Bd.AssignElementPowerOf(Ad,3.2);
            Bs.AssignElementPowerOf(As,3.2);
#ifdef CHECK
            Bs.SwitchToMatrixType(DENSE);
            Bd.TransferFromDeviceToDevice(0,CPUDEVICE);
            Bs.TransferFromDeviceToDevice(0,CPUDEVICE);
            for (int r = 0; r < Bd.GetNumRows(); ++r)
                for (int c = 0; c < Bd.GetNumCols(); ++c)
                {
                    float dVal = Bd(r,c);
                    float sVal = Bs(r,c);
                    float diff = sVal - dVal;
                    if (fabs(diff) > 0.001)
                        cout << "[" << r << ", " << c << "]: " << sVal << " and " << dVal;
                }
#endif
            Assert::IsTrue(Bs.IsEqualTo(Bd,0.0001));
            Assert::IsTrue(Bd.IsEqualTo(Bs,0.0001));
        }
    };
}