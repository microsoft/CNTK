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
#include "..\Math\GPUSparseMatrix.cuh"
#define epsilon 0.00001
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Microsoft::MSR::CNTK;

#define ID_2C(i,j,ld) (((i)*(ld))+(j)) // 0 based indexing

namespace CNTKMathTest
{        

    TEST_CLASS(GPUSparseMatrixUnitTests)
    {
    public:

        TEST_METHOD(GPUSparseMatrixConsturctorsAndInitializers)
        {
            GPUSparseMatrix<float> M;
            Assert::IsTrue(M.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            M.SetMatrixFromCSRFormat(i,j,v,9,4,5);
            Assert::AreEqual<size_t>(4,M.GetNumRows());
            Assert::AreEqual<size_t>(5,M.GetNumCols());
            Assert::IsTrue(!M.IsEmpty());

            GPUSparseMatrix<float> M1=M;
            Assert::AreEqual<size_t>(4,M1.GetNumRows());
            Assert::AreEqual<size_t>(5,M1.GetNumCols());
            Assert::IsTrue(!M1.IsEmpty());

            GPUSparseMatrix<float> M2(M);
            Assert::AreEqual<size_t>(4,M2.GetNumRows());
            Assert::AreEqual<size_t>(5,M2.GetNumCols());
            Assert::IsTrue(!M2.IsEmpty());
        }

        TEST_METHOD(GPUSparseMatrixScaleAndAdd)
        {
            int m = 4;
            int n = 5;

            float alpha = 2;
            float beta = 3;

            float *a = new float[m*n];
            float *b = new float[m*n];
            for (int i=0;i<m*n;i++)
            {
                a[i]=rand();
                b[i]=rand();
            }

            GPUMatrix<float> A_d(m,n,a,matrixFlagNormal);
            GPUMatrix<float> B_d(m,n,b,matrixFlagNormal);

            GPUSparseMatrix<float> A;
            A.SetValue(A_d);
            GPUSparseMatrix<float> B;
            B.SetValue(B_d);

            GPUSparseMatrix<float> C;
            GPUSparseMatrix<float>::ScaleAndAdd(alpha,A,beta,B,C);

            GPUSparseMatrix<float>::Scale(alpha,C);

            GPUMatrix<float> C_d = C.CopyToDenseMatrix();
            float *c = C_d.CopyToArray();
            for (int i=0;i<m*n;i++)
            {
                Assert::AreEqual<float>(alpha*(alpha*a[i]+beta*b[i]),c[i]);                
            }
            delete[] a;
            delete[] b;
            delete[] c;            
        }

        TEST_METHOD(GPUSparseDensePlusSparse)
        {
            GPUSparseMatrix<float> M;            
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            M.SetMatrixFromCSRFormat(i,j,v,9,4,5);

            GPUMatrix<float> Bd = GPUMatrix<float>::RandomUniform(4,5,-2,45);
            GPUMatrix<float> Cs(Bd.GetNumRows(),Bd.GetNumCols());            

            float alpha = 0.53;
            float beta = 1;

            GPUSparseMatrix<float>::ScaleAndAdd(alpha,M,beta,Bd,Cs);
            GPUMatrix<float>::ScaleAndAdd(alpha,M.CopyToDenseMatrix(),Bd);

            Assert::IsTrue(Bd.IsEqualTo(Cs,0.00001));
        }

        TEST_METHOD(GPUSparseElemenwiseTimesDense)
        {
            GPUSparseMatrix<float> M;            
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            M.SetMatrixFromCSRFormat(i,j,v,9,4,5);

            GPUMatrix<float> Bd = GPUMatrix<float>::RandomUniform(4,5,-2,45);

            GPUMatrix<float> C1 = GPUSparseMatrix<float>::ElementProductOf(M,Bd);
            GPUMatrix<float> C2;
            C2.AssignElementProductOf(M.CopyToDenseMatrix(),Bd);
            Assert::IsTrue(C1.IsEqualTo(C2));
        }

        TEST_METHOD(GPUSSparseTimesDense)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);
            Assert::AreEqual<size_t>(4,A.GetNumRows());
            Assert::AreEqual<size_t>(5,A.GetNumCols());
            Assert::IsTrue(!A.IsEmpty());

            GPUMatrix<float> B = GPUMatrix<float>::Eye(5);
            GPUMatrix<float> C = GPUMatrix<float>::Ones(4,5);

            GPUSparseMatrix<float>::MultiplyAndWeightedAdd(1,A,false,B,1,C);

            float* arr = C.CopyToArray();
            CPUMatrix<float> CCPU(4,5,arr,matrixFlagNormal);
            delete[] arr;

            Assert::AreEqual<float>(1+1,CCPU(0,0));Assert::AreEqual<float>(4+1,CCPU(0,1));Assert::AreEqual<float>(0+1,CCPU(0,2));Assert::AreEqual<float>(0+1,CCPU(0,3));Assert::AreEqual<float>(0+1,CCPU(0,4));
            Assert::AreEqual<float>(0+1,CCPU(1,0));Assert::AreEqual<float>(2+1,CCPU(1,1));Assert::AreEqual<float>(3+1,CCPU(1,2));Assert::AreEqual<float>(0+1,CCPU(1,3));Assert::AreEqual<float>(0+1,CCPU(1,4));
            Assert::AreEqual<float>(5+1,CCPU(2,0));Assert::AreEqual<float>(0+1,CCPU(2,1));Assert::AreEqual<float>(0+1,CCPU(2,2));Assert::AreEqual<float>(7+1,CCPU(2,3));Assert::AreEqual<float>(8+1,CCPU(2,4));
            Assert::AreEqual<float>(0+1,CCPU(3,0));Assert::AreEqual<float>(0+1,CCPU(3,1));Assert::AreEqual<float>(9+1,CCPU(3,2));Assert::AreEqual<float>(0+1,CCPU(3,3));Assert::AreEqual<float>(6+1,CCPU(3,4));
        }

        TEST_METHOD(GPUSDenseTimesSparse)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);
            Assert::AreEqual<size_t>(4,A.GetNumRows());
            Assert::AreEqual<size_t>(5,A.GetNumCols());
            Assert::IsTrue(!A.IsEmpty());

            GPUSparseMatrix<float> ATs = A.Transpose();
            GPUMatrix<float> ATd = ATs.CopyToDenseMatrix();

            float* arrTd = ATd.CopyToArray();

            float arrA_times_AT[19] = {17,8,5,0,8,13,0,27,5,0,138,48,0,27,48,117}; 
            GPUMatrix<float> Cet(4,4,arrA_times_AT,matrixFlagNormal);
            GPUMatrix<float> Cres(4,4);
            GPUSparseMatrix<float>::Multiply(A,ATd,Cres);  //Sparse times dense
            Assert::IsTrue(Cres.IsEqualTo(Cet));

            float arrAT_times_A[25] = {26,4,0,35,40,4,20,6,0,0,0,6,90,0,54,35,0,0,49,56,40,0,54,56,100};
            GPUMatrix<float> Cet1(5,5,arrAT_times_A,matrixFlagNormal);
            GPUMatrix<float> Cres1(5,5);
            GPUSparseMatrix<float>::Multiply(ATd,A,Cres1);  //Dense times sparse

            float* arr = Cres1.CopyToArray();

            Assert::IsTrue(Cres1.IsEqualTo(Cet1));

            GPUMatrix<float> B = GPUMatrix<float>::RandomUniform(9,4,-100,100,0);
            GPUMatrix<float> C(9,5);
            GPUSparseMatrix<float>::Multiply(B,A,C); //C=BA

            GPUMatrix<float> BT = B.Transpose();
            GPUSparseMatrix<float> AT = A.Transpose();
            GPUMatrix<float> CT(5,9);
            GPUSparseMatrix<float>::Multiply(AT,BT,CT); // CT=AT*BT  = (BA)T
            GPUMatrix<float> CCT = CT.Transpose(); //CCT = C;

            /*            float* arr1 = C.CopyToArray();
            float* arr2 = CCT.CopyToArray(); */           

            Assert::IsTrue(CCT.IsEqualTo(C,0.0001));         
        }


        TEST_METHOD(GPUSSparseTimesSparse)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);
            GPUSparseMatrix<float> AT=A.Transpose();

            GPUSparseMatrix<float> C;
            GPUSparseMatrix<float>::Multiply(AT,false,A,false,C);

            float arrAT_times_A[25] = {26,4,0,35,40,4,20,6,0,0,0,6,90,0,54,35,0,0,49,56,40,0,54,56,100};
            float *arr = C.CopyToDenseMatrix().CopyToArray();
            for (int i=0;i<25;++i)
            {
                Assert::AreEqual<float>(arrAT_times_A[i],arr[i]);
            }        
            delete[] arr;
        }

        TEST_METHOD(GPUSSparseElementWise)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v1[9] = {1,4,2,3,5,7,8,9,6};
            int i1[5] = {0,2,4,7,9};
            int j1[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i1,j1,v1,9,4,5);

            GPUSparseMatrix<float> C;
            float p = 3.14;
            C.ResizeAs(A);
            A.ElementWisePower(p,A,C);

            float *arr = NULL;
            int *ii = NULL;
            int *jj = NULL;
            size_t nz,nr,nc;
            C.GetMatrixFromCSRFormat(ii,jj,arr,nz,nr,nc);

            for (int i=0;i<9;++i)
            {
                float y = powf(v1[i],p);
                Assert::IsTrue(fabsf(y-arr[i])<0.001);
            }
            delete[] arr;
            delete[] ii;
            delete[] jj;
        }


        TEST_METHOD(GPUSSparseIsEqual)
        {
            GPUSparseMatrix<float> A;            
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);

            GPUSparseMatrix<float> B;
            B.SetMatrixFromCSRFormat(i,j,v,9,4,5);
            Assert::IsTrue(B.IsEqualTo(A));

            GPUSparseMatrix<float> C;
            Assert::IsFalse(C.IsEqualTo(A));
        }

        TEST_METHOD(GPUSSparseDenseConversions)
        {           

            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);
            Assert::AreEqual<size_t>(4,A.GetNumRows());
            Assert::AreEqual<size_t>(5,A.GetNumCols());
            Assert::IsTrue(!A.IsEmpty());

            GPUMatrix<float> A_dense = A.CopyToDenseMatrix();
            Assert::AreEqual<size_t>(4,A_dense.GetNumRows());
            Assert::AreEqual<size_t>(5,A_dense.GetNumCols());

            float *arr = A_dense.CopyToArray();
            CPUMatrix<float> A_cpu(A_dense.GetNumRows(),A_dense.GetNumCols(),arr,matrixFlagNormal);            
            delete[] arr;

            Assert::AreEqual<float>(1,A_cpu(0,0)); 
            Assert::AreEqual<float>(4,A_cpu(0,1)); 
            Assert::AreEqual<float>(0,A_cpu(0,2)); 
            Assert::AreEqual<float>(0,A_cpu(0,3)); 
            Assert::AreEqual<float>(0,A_cpu(0,4));
            Assert::AreEqual<float>(5,A_cpu(2,0)); 
            Assert::AreEqual<float>(0,A_cpu(2,1)); 
            Assert::AreEqual<float>(0,A_cpu(2,2)); 
            Assert::AreEqual<float>(7,A_cpu(2,3)); 
            Assert::AreEqual<float>(8,A_cpu(2,4));

            GPUSparseMatrix<float> B;
            B.SetValue(A_dense);
            GPUMatrix<float> B_dense = B.CopyToDenseMatrix();
            arr = B_dense.CopyToArray();
            CPUMatrix<float> B_cpu(B_dense.GetNumRows(),B_dense.GetNumCols(),arr,matrixFlagNormal);
            delete[] arr;

            Assert::AreEqual<float>(1,B_cpu(0,0)); 
            Assert::AreEqual<float>(4,B_cpu(0,1)); 
            Assert::AreEqual<float>(0,B_cpu(0,2)); 
            Assert::AreEqual<float>(0,B_cpu(0,3)); 
            Assert::AreEqual<float>(0,B_cpu(0,4));
            Assert::AreEqual<float>(5,B_cpu(2,0)); 
            Assert::AreEqual<float>(0,B_cpu(2,1)); 
            Assert::AreEqual<float>(0,B_cpu(2,2)); 
            Assert::AreEqual<float>(7,B_cpu(2,3)); 
            Assert::AreEqual<float>(8,B_cpu(2,4));

            float dV[10] = {0,1,0,1,0,0,0,0,3,0};
            GPUMatrix<float> DenseVector(10,1,dV);
            GPUSparseMatrix<float> SparseVector;
            SparseVector.SetValue(DenseVector);
            float *dVal=NULL;
            int* Col=NULL;
            int* Row=NULL;
            size_t nz, colnum, rowind;
            SparseVector.GetMatrixFromCSRFormat(Row,Col,dVal,nz,rowind,colnum);

            float a[9] = { 1, 0, 4, 0, 0, 5, 4, 0, 0};
            GPUMatrix<float> A4(3,3,a,matrixFlagNormal);
            GPUSparseMatrix<float> A4s(A4);

            delete[] dVal; dVal=NULL;
            delete[] Col; Col=NULL;
            delete[] Row; Row=NULL;
            A4s.GetMatrixFromCSRFormat(Row,Col,dVal,nz,rowind,colnum);
            delete[] dVal;
            delete[] Col; 
            delete[] Row;
        }

        TEST_METHOD(GPUSSparseTranspose)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);

            GPUSparseMatrix<float> A1 = A.Transpose();
            GPUSparseMatrix<float> A2(A);
            A2.InplaceTranspose();
            Assert::IsTrue(A2.IsEqualTo(A1));

            GPUSparseMatrix<float> B = A.Transpose();
            GPUSparseMatrix<float> C;// = B.Transpose();
            C.AssignTransposeOf(B);

            Assert::IsTrue(C.IsEqualTo(A));
            A.InplaceTranspose();
            Assert::IsFalse(C.IsEqualTo(A));
            A.InplaceTranspose();          
            Assert::IsTrue(C.IsEqualTo(A));
        }

        TEST_METHOD(GPUSSparseNormTests)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);

            float frobenius = A.FrobeniusNorm();
            Assert::IsTrue(fabsf(16.882-frobenius)<0.0001);

            float ninf = A.MatrixNormInf();
            Assert::AreEqual<float>(9,ninf);

            float n1 = A.MatrixNorm1();
            Assert::AreEqual<float>(45,n1);
        }

        TEST_METHOD(GPUSSparseMatrixInnerProduct)
        {
            GPUSparseMatrix<float> A;
            Assert::IsTrue(A.IsEmpty());
            float v[9] = {1,4,2,3,5,7,8,9,6};
            int i[5] = {0,2,4,7,9};
            int j[9] = {0,1,1,2,0,3,4,2,4};
            A.SetMatrixFromCSRFormat(i,j,v,9,4,5);

            GPUMatrix<float> B = GPUMatrix<float>::RandomUniform(4,5,-3,4);
            float x = GPUSparseMatrix<float>::InnerProductOfMatrices(A,B);
            float y = GPUMatrix<float>::InnerProductOfMatrices(A.CopyToDenseMatrix(),B);
            Assert::IsTrue(fabsf(x-y)<0.00001);            
        }
    };
}

