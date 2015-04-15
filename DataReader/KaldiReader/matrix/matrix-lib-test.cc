// matrix/matrix-lib-test.cc

// Copyright 2009-2012   Microsoft Corporation;  Mohit Agarwal;  Lukas Burget;
//                       Ondrej Glembek;  Saarland University (Author: Arnab Ghoshal);
//                       Go Vivace Inc.;  Yanmin Qian;  Jan Silovsky;
//                       Johns Hopkins University (Author: Daniel Povey);
//                       Haihua Xu

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "matrix/matrix-lib.h"
#include <numeric>
#include <time.h> // This is only needed for UnitTestSvdSpeed, you can
// comment it (and that function) out if it causes problems.

namespace kaldi {

template<typename Real>
void RandPosdefSpMatrix(MatrixIndexT dim, SpMatrix<Real> *matrix) {
  MatrixIndexT dim2 = dim + (rand() % 3);  // slightly higher-dim.
  // generate random (non-singular) matrix
  Matrix<Real> tmp(dim, dim2);
  while (1) {
    tmp.SetRandn();
    if (tmp.Cond() < 100) break;
    KALDI_LOG << "Condition number of random matrix large "
              << static_cast<float>(tmp.Cond())
              << ", trying again (this is normal)";
  }
  // tmp * tmp^T will give positive definite matrix
  matrix->AddMat2(1.0, tmp, kNoTrans, 0.0);
  
  // Checks that the matrix is indeed pos-def
  TpMatrix<Real> sqrt(dim);
  sqrt.Cholesky(*matrix);
}


template<typename Real> static void InitRand(TpMatrix<Real> *M) {
  // start:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  (*M)(i, j) = RandGauss();
}



template<typename Real> static void InitRand(Vector<Real> *v) {
  for (MatrixIndexT i = 0;i < v->Dim();i++)
	(*v)(i) = RandGauss();
}

template<typename Real> static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0;i < v->Dim();i++)
	(*v)(i) = RandGauss();
}

template<typename Real> static void InitRand(MatrixBase<Real> *M) {
start:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
    for (MatrixIndexT j = 0;j < M->NumCols();j++)
      (*M)(i, j) = RandGauss();
  if (M->NumRows() != 0 && M->Cond() > 100) {
    printf("Condition number of random matrix large %f, trying again (this is normal)\n",
           (float) M->Cond());
    goto start;
  }
}


template<typename Real> static void InitRand(SpMatrix<Real> *M) {
start:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  (*M)(i, j) = RandGauss();
  if (M->NumRows() != 0 && M->Cond() > 100)
    goto start;
}

/*
  HERE(2)
template<typename Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j < A.NumCols();j++) {
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
}

template<typename Real> static void AssertEqual(const SpMatrix<Real> &A,
                                             const SpMatrix<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
}
*/
/*
  HERE:
template<typename Real>
static bool ApproxEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  SpMatrix<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}
*/

/* was:
   template<typename Real>
   bool ApproxEqual(SpMatrix<Real> &A, SpMatrix<Real> &B, float tol = 0.001) {
   KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
   for (MatrixIndexT i = 0;i < A.NumRows();i++)
   for (MatrixIndexT j = 0;j<=i;j++)
   if (std::abs(A(i, j)-B(i, j)) > tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j))))) return false;
   return true;
   }
*/

/*
  HERE
template<typename Real> static void AssertEqual(Vector<Real> &A, Vector<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0;i < A.Dim();i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}

template<typename Real> static bool ApproxEqual(Vector<Real> &A, Vector<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0;i < A.Dim();i++)
    if (std::abs(A(i)-B(i)) > tol) return false;
  return true;
}
*/

template<typename Real> static void CholeskyUnitTestTr() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = 2 + rand() % 10;
    Matrix<Real> M(dimM, dimM);
    InitRand(&M);
    SpMatrix<Real> S(dimM);
    S.AddMat2(1.0, M, kNoTrans, 0.0);
    TpMatrix<Real> C(dimM);
    C.Cholesky(S);
    Matrix<Real> CM(C);
    TpMatrix<Real> Cinv(C);
    Cinv.Invert();
    {
      Matrix<Real> A(C), B(Cinv), AB(dimM, dimM);
      AB.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
      KALDI_ASSERT(AB.IsUnit());
    }
    SpMatrix<Real> S2(dimM);
    S2.AddMat2(1.0, CM, kNoTrans, 0.0);
    AssertEqual(S, S2);
    C.Invert();
    Matrix<Real> CM2(C);
    CM2.Invert();
    SpMatrix<Real> S3(dimM);
    S3.AddMat2(1.0, CM2, kNoTrans, 0.0);
    AssertEqual(S, S3);
  }
}

template<typename Real> static void SlowMatMul() {
  MatrixIndexT N = 1000;
  Matrix<Real> M(N, N), P(N, N), Q(N, N);
  for (MatrixIndexT i = 0; i < 10000; i++) {
    Q.AddMatMat(1.0, M, kNoTrans, P, kNoTrans, 0.0);
  }
}  

template<typename Real> static void UnitTestAddToDiagMatrix() {
  for (int p = 0; p < 2; p++) {
    MatrixIndexT dimM = 10 + rand() % 2, dimN = 1 + rand() % 5;
    Matrix<Real> M(dimM, dimN), Mcopy(M);
    BaseFloat alpha = 0.35;
    M.AddToDiag(alpha);
    for (MatrixIndexT i = 0; i < dimM && i < dimN; i++)
      Mcopy(i, i) += alpha;
    AssertEqual(M, Mcopy);
  }
}
  

template<typename Real> static void UnitTestAddDiagVecMat() {
  for (int p = 0; p < 2; p++) {
    MatrixIndexT dimM = 100 + rand() % 255, dimN = 100 + rand() % 255;
    Real alpha = 0.43243, beta = 1.423;
    Matrix<Real> M(dimM, dimN), N(dimM, dimN);
    M.SetRandn();
    N.SetRandn();
    MatrixTransposeType trans = (p % 2 == 0 ? kNoTrans : kTrans);
    if (trans == kTrans)
      N.Transpose();
    
    Vector<Real> V(dimM);
    V.SetRandn();

    Matrix<Real> Mcheck(M);
    for (int32 r = 0; r < dimM; r++) {
      SubVector<Real> Mcheckrow(Mcheck, r);
      Vector<Real> Nrow(dimN);
      if (trans == kTrans) Nrow.CopyColFromMat(N, r);
      else Nrow.CopyFromVec(N.Row(r));
      Mcheckrow.Scale(beta);
      Mcheckrow.AddVec(alpha * V(r), Nrow);
    }

    M.AddDiagVecMat(alpha, V, N, trans, beta);
    AssertEqual(M, Mcheck);
    KALDI_ASSERT(M.Sum() != 0.0);
  }
}


template<typename Real> static void UnitTestAddSp() {
  for (MatrixIndexT i = 0;i< 10;i++) {
    MatrixIndexT dimM = 10+rand()%10;
    SpMatrix<Real> S(dimM);
    InitRand(&S);
    Matrix<Real> M(S), N(S);
    N.AddSp(2.0, S);
    M.Scale(3.0);
    AssertEqual(M, N);
  }
}

template<typename Real, typename OtherReal>
static void UnitTestSpAddVec() {
  for (MatrixIndexT i = 0;i< 10;i++) {
    BaseFloat alpha = (i<5 ? 1.0 : 0.5);
    MatrixIndexT dimM = 10+rand()%10;
    SpMatrix<Real> S(dimM);
    InitRand(&S);
    SpMatrix<Real> T(S);
    Vector<OtherReal> v(dimM);
    InitRand(&v);
    S.AddVec(alpha, v);
    for (MatrixIndexT i = 0; i < dimM; i++)
      T(i, i) += alpha * v(i);
    AssertEqual(S, T);
  }
}


template<typename Real>
static void UnitTestSpAddVecVec() {
  for (MatrixIndexT i = 0;i< 10;i++) {
    BaseFloat alpha = (i<5 ? 1.0 : 0.5);
    MatrixIndexT dimM = 10+rand()%10;
    SpMatrix<Real> S(dimM);
    InitRand(&S);
    Matrix<Real> T(S);

    Vector<Real> v(dimM), w(dimM);
    InitRand(&v);
    InitRand(&w);
    S.AddVecVec(alpha, v, w);
    T.AddVecVec(alpha, v, w);
    T.AddVecVec(alpha, w, v);
    Matrix<Real> U(S);
    AssertEqual(U, T);
  }
}


template<typename Real> static void UnitTestCopyRowsAndCols() {
  // Test other mode of CopyRowsFromVec, and CopyColsFromVec,
  // where vector is duplicated.
  for (MatrixIndexT i = 0; i < 30; i++) {
    MatrixIndexT dimM = 1 + rand() % 5, dimN = 1 + rand() % 5;
    Vector<float> w(dimN); // test cross-type version of
    // CopyRowsFromVec.
    Vector<Real> v(dimM);
    Matrix<Real> M(dimM, dimN), N(dimM, dimN);
    InitRand(&v);
    InitRand(&w);
    M.CopyColsFromVec(v);
    N.CopyRowsFromVec(w);
    for (MatrixIndexT r = 0; r < dimM; r++) {
      for (MatrixIndexT c = 0; c < dimN; c++) {
        KALDI_ASSERT(M(r, c) == v(r));
        KALDI_ASSERT(N(r, c) == w(c));
      }
    }    
  }
}

template<typename Real> static void UnitTestSpliceRows() {

  for (MatrixIndexT i = 0;i< 10;i++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;

    Vector<Real> V(dimM*dimN), V10(dimM*dimN);
    Vector<Real> Vs(std::min(dimM, dimN)), Vs10(std::min(dimM, dimN));
    InitRand(&V);
    Matrix<Real> M(dimM, dimN);
    M.CopyRowsFromVec(V);
    V10.CopyRowsFromMat(M);
    AssertEqual(V, V10);        
    
    for (MatrixIndexT i = 0;i < dimM;i++)
      for (MatrixIndexT  j = 0;j < dimN;j++)
        KALDI_ASSERT(M(i, j) == V(i*dimN + j));

    {
      Vector<Real> V2(dimM), V3(dimM);
      InitRand(&V2);
      MatrixIndexT x;
      M.CopyColFromVec(V2, x = (rand() % dimN));
      V3.CopyColFromMat(M, x);
      AssertEqual(V2, V3);
    }

    {
      Vector<Real> V2(dimN), V3(dimN);
      InitRand(&V2);
      MatrixIndexT x;
      M.CopyRowFromVec(V2, x = (rand() % dimM));
      V3.CopyRowFromMat(M, x);
      AssertEqual(V2, V3);
    }

    {
      M.CopyColsFromVec(V);
      V10.CopyColsFromMat(M);
      AssertEqual(V, V10);
    }

    {
      M.CopyDiagFromVec(Vs);
      Vs10.CopyDiagFromMat(M);
      AssertEqual(Vs, Vs10);
    }

  }
}

template<typename Real> static void UnitTestRemoveRow() {

  // this is for matrix
  for (MatrixIndexT p = 0;p< 10;p++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);
    MatrixIndexT i = rand() % dimM;  // Row to remove.
    Matrix<Real> N(M);
    N.RemoveRow(i);
    for (MatrixIndexT j = 0;j < i;j++) {
      for (MatrixIndexT k = 0;k < dimN;k++) {
        KALDI_ASSERT(M(j, k) == N(j, k));
      }
    }
    for (MatrixIndexT j = i+1;j < dimM;j++) {
      for (MatrixIndexT k = 0;k < dimN;k++) {
        KALDI_ASSERT(M(j, k) == N(j-1, k));
      }
    }
  }

  // this is for vector
  for (MatrixIndexT p = 0;p< 10;p++) {
    MatrixIndexT dimM = 10+rand()%10;
    Vector<Real> V(dimM);
    InitRand(&V);
    MatrixIndexT i = rand() % dimM;  // Element to remove.
    Vector<Real> N(V);
    N.RemoveElement(i);
    for (MatrixIndexT j = 0;j < i;j++) {
      KALDI_ASSERT(V(j) == N(j));
    }
    for (MatrixIndexT j = i+1;j < dimM;j++) {
      KALDI_ASSERT(V(j) == N(j-1));
    }
  }

}


template<typename Real> static void UnitTestSubvector() {

  Vector<Real> V(100);
  InitRand(&V);
  Vector<Real> V2(100);
  for (MatrixIndexT i = 0;i < 10;i++) {
    SubVector<Real> tmp(V, i*10, 10);
    SubVector<Real> tmp2(V2, i*10, 10);
    tmp2.CopyFromVec(tmp);
  }
  AssertEqual(V, V2);
}

// just need this for testing function below.  Compute n!!
static int32 DoubleFactorial(int32 i) {
  if (i <= 0) { return 1; } else { return i * DoubleFactorial(i - 2); }
}

template <typename Real>
static void UnitTestSetRandn() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT rows = 100 + rand() % 50, cols = 100 + rand() % 50;
    Matrix<Real> M(rows, cols);
    M.SetRandn();

    for (MatrixIndexT pow = 1; pow < 5; pow++) {
      // test moments 1 through 4 of
      // the distribution.
      Matrix<Real> Mpow(M);
      Mpow.ApplyPow(pow);
      Real observed_moment = Mpow.Sum() / (rows * cols);
      // see http://en.wikipedia.org/wiki/Normal_distribution#Moments,
      // note that mu = 0 and sigma = 1.
      Real expected_moment = (pow % 2 == 1 ? 0 : DoubleFactorial(pow - 1));
      Real k = 10.0; // This is just a constant we use to give us some wiggle
                     // room before rejecting the distribution... e.g. 10 sigma,
                     // quite approximately.
      Real allowed_deviation = k * pow / sqrt(static_cast<Real>(rows * cols));
      // give it a bit more wiggle room for higher powers.. this is quite
      // unscientific, it would be better to involve the absolute moments or
      // something like that, and use one of those statistical inequalities,
      // but it involves the gamma function and it's too much hassle to implement.
      Real lower_bound = expected_moment - allowed_deviation,
          upper_bound = expected_moment + allowed_deviation;
      KALDI_ASSERT(observed_moment >= lower_bound && observed_moment <= upper_bound);
    }
  }
}


template <typename Real>
static void UnitTestSetRandUniform() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT rows = 200 + rand() % 50, cols = 200 + rand() % 50;
    Matrix<Real> M(rows, cols);
    M.SetRandUniform();

    M.Add(-0.5); // we'll be testing the central moments, so
    // center it around zero first.
    // Got these moments from http://mathworld.wolfram.com/UniformDistribution.html
    Vector<Real> central_moments(5);
    central_moments(0) = 0.0;
    central_moments(1) = 0.0;
    central_moments(2) = 1.0 / 12; // times (b - a)^2, which equals 1.
    central_moments(3) = 0.0;
    central_moments(4) = 1.0 / 80; // times (b - a)^4, which equals 1.

    for (MatrixIndexT pow = 1; pow < central_moments.Dim(); pow++) {
      Matrix<Real> Mpow(M);
      Mpow.ApplyPow(pow);
      Real observed_moment = Mpow.Sum() / (rows * cols);
      // see http://en.wikipedia.org/wiki/Normal_distribution#Moments,
      // note that mu = 0 and sigma = 1.
      Real expected_moment = central_moments(pow);
      Real k = 10.0; // This is just a constant we use to give us some wiggle
                     // room before rejecting the distribution... e.g. 10 sigma,
                     // quite approximately.
      Real allowed_deviation = k / sqrt(static_cast<Real>(rows * cols));
      Real lower_bound = expected_moment - allowed_deviation,
          upper_bound = expected_moment + allowed_deviation;
      KALDI_ASSERT(observed_moment >= lower_bound && observed_moment <= upper_bound);
    }
  }
}


template <typename Real>
static void UnitTestSimpleForVec() {  // testing some simple operaters on vector

  for (MatrixIndexT i = 0; i < 5; i++) {
    Vector<Real> V(100), V1(100), V2(100), V3(100), V4(100);
    InitRand(&V);
    if (i % 2 == 0) {
      V1.SetZero();
      V1.Add(1.0);
    } else {
      V1.Set(1.0);
    }

    V2.CopyFromVec(V);
    V3.CopyFromVec(V1);
    V2.InvertElements();
    V3.DivElements(V);
    V4.CopyFromVec(V3);
    V4.AddVecDivVec(1.0, V1, V, 0.0);
    AssertEqual(V3, V2);
    AssertEqual(V4, V3);
    V4.MulElements(V);
    AssertEqual(V4, V1);
    V3.AddVecVec(1.0, V, V2, 0.0);
    AssertEqual(V3, V1);

    Vector<Real> V5(V), V6(V1), V7(V1), V8(V);
    V5.AddVec(1.0, V);
    V8.Scale(2.0);
    V6.AddVec2(1.0, V);
    V7.AddVecVec(1.0, V, V, 1.0);
    AssertEqual(V6, V7);
    AssertEqual(V5, V8);
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);
    Vector<Real> Vr(dimN), Vc(dimM);
    Vr.AddRowSumMat(0.4, M); 
    Vr.AddRowSumMat(0.3, M, 0.5); // note: 0.3 + 0.4*0.5 = 0.5.
    Vc.AddColSumMat(0.4, M);
    Vc.AddColSumMat(0.3, M, 0.5); // note: 0.3 + 0.4*0.5 = 0.5.
    Vr.Scale(2.0);
    Vc.Scale(2.0);

    Vector<Real> V2r(dimN), V2c(dimM);
    for (MatrixIndexT k = 0; k < dimM; k++) {
      V2r.CopyRowFromMat(M, k);
      AssertEqual(V2r.Sum(), Vc(k));
    }
    for (MatrixIndexT j = 0; j < dimN; j++) {
      V2c.CopyColFromMat(M, j);
      AssertEqual(V2c.Sum(), Vr(j));
    }
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    Vector<Real> V(100), V1(100), V2(100);
    InitRand(&V);
    
    V1.CopyFromVec(V);
    V1.ApplyExp();
    Real a = V.LogSumExp();
    V2.Set(exp(V.LogSumExp()));
    V1.DivElements(V2);
    Real b = V.ApplySoftMax();
    AssertEqual(V1, V);
    AssertEqual(a, b);
    KALDI_LOG << "a = " << a << ", b = " << b;
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimV = 10 + rand() % 10;
    Real p = 0.5 + RandUniform() * 4.5;
    Vector<Real> V(dimV), V1(dimV), V2(dimV);
    InitRand(&V);
    V1.AddVecVec(1.0, V, V, 0.0);  // V1:=V.*V.
    V2.CopyFromVec(V1);
    AssertEqual(V1.Norm(p), V2.Norm(p));
    AssertEqual(sqrt(V1.Sum()), V.Norm(2.0));
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimV = 10 + rand() % 10;
    Real p = RandUniform() * 1.0e-5;
    Vector<Real> V(dimV);
    V.Set(p);
    KALDI_ASSERT(V.IsZero(p));
    KALDI_ASSERT(!V.IsZero(p*0.9));
  }

  // Test ApplySoftMax() for matrix.
  Matrix<Real> M(10, 10);
  InitRand(&M);
  M.ApplySoftMax();
  KALDI_ASSERT( fabs(1.0 - M.Sum()) < 0.01);

}

template<typename Real>
static void UnitTestVectorMax() {
  int32 dimM = 1 + rand() % 10;
  Vector<Real> V(dimM);
  V.SetRandn();
  Real m = V(0);
  for (int32 i = 1; i < dimM; i++) m = std::max(m, V(i));
  KALDI_ASSERT(m == V.Max());
  MatrixIndexT i;
  KALDI_ASSERT(m == V.Max(&i));
  KALDI_ASSERT(m == V(i));
}

template<typename Real>
static void UnitTestVectorMin() {
  int32 dimM = 1 + rand() % 10;
  Vector<Real> V(dimM);
  V.SetRandn();
  Real m = V(0);
  for (int32 i = 1; i < dimM; i++) m = std::min(m, V(i));
  KALDI_ASSERT(m == V.Min());
  MatrixIndexT i;
  KALDI_ASSERT(m == V.Min(&i));
  KALDI_ASSERT(m == V(i));
}

template<typename Real>  
static void UnitTestReplaceValue(){
  MatrixIndexT dim = 10 + rand() % 2;
  Real orig = 0.1 * (rand() % 100), changed = 0.1 * (rand() % 50);
  Vector<Real> V(dim);
  V.SetRandn();
  V(dim / 2) = orig;
  Vector<Real> V1(V);
  for (MatrixIndexT i = 0; i < V1.Dim(); i ++) {
    if (V1(i) == orig) V1(i) = changed;
  }
  V.ReplaceValue(orig, changed);
  AssertEqual(V, V1);
}


template<typename Real>
static void UnitTestNorm() {  // test some simple norm properties: scaling.  also ApproxEqual test.

  for (MatrixIndexT p = 0; p < 10; p++) {
    Real scalar = RandGauss();
    if (scalar == 0.0) continue;
    if (scalar < 0) scalar *= -1.0;
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);
    SpMatrix<Real> S(dimM);
    InitRand(&S);
    Vector<Real> V(dimN);
    InitRand(&V);

    Real Mnorm = M.FrobeniusNorm(),
        Snorm = S.FrobeniusNorm(),
        Vnorm1 = V.Norm(1.0),
        Vnorm2 = V.Norm(2.0),
        Vnorm3 = V.Norm(3.0);
    M.Scale(scalar);
    S.Scale(scalar);
    V.Scale(scalar);
    KALDI_ASSERT(ApproxEqual(M.FrobeniusNorm(), Mnorm*scalar));
    KALDI_ASSERT(ApproxEqual(S.FrobeniusNorm(), Snorm*scalar));
    KALDI_ASSERT(ApproxEqual(V.Norm(1.0), Vnorm1 * scalar));
    KALDI_ASSERT(ApproxEqual(V.Norm(2.0), Vnorm2 * scalar));
    KALDI_ASSERT(ApproxEqual(V.Norm(3.0), Vnorm3 * scalar));

    KALDI_ASSERT(V.ApproxEqual(V));
    KALDI_ASSERT(M.ApproxEqual(M));
    KALDI_ASSERT(S.ApproxEqual(S));
    SpMatrix<Real> S2(S); S2.Scale(1.1);  KALDI_ASSERT(!S.ApproxEqual(S2));  KALDI_ASSERT(S.ApproxEqual(S2, 0.15));
    Matrix<Real> M2(M); M2.Scale(1.1);  KALDI_ASSERT(!M.ApproxEqual(M2));  KALDI_ASSERT(M.ApproxEqual(M2, 0.15));
    Vector<Real> V2(V); V2.Scale(1.1);  KALDI_ASSERT(!V.ApproxEqual(V2));  KALDI_ASSERT(V.ApproxEqual(V2, 0.15));
  }
}


template<typename Real>
static void UnitTestCopyRows() {
  for (MatrixIndexT p = 0; p < 10; p++) {
    MatrixIndexT num_rows1 = 10 + rand() % 10,
        num_rows2 = 10 + rand() % 10,
        num_cols = 10 + rand() % 10;
    Matrix<Real> M(num_rows1, num_cols);
    InitRand(&M);
    
    Matrix<Real> N(num_rows2, num_cols), O(num_rows2, num_cols);
    std::vector<int32> reorder(num_rows2);
    for (int32 i = 0; i < num_rows2; i++)
      reorder[i] = -1 + (rand() % (num_rows1 + 1));
    
    N.CopyRows(M, reorder);

    for (int32 i = 0; i < num_rows2; i++)
      for (int32 j = 0; j < num_cols; j++)
        if (reorder[i] < 0) O(i, j) = 0;
        else O(i, j) = M(reorder[i], j);
    
    KALDI_LOG << "M is " << M;
    KALDI_LOG << "N is " << N;
    KALDI_LOG << "O is " << O;
    AssertEqual(N, O);
  }
}

template<typename Real>
static void UnitTestCopyCols() {
  for (MatrixIndexT p = 0; p < 10; p++) {
    MatrixIndexT num_cols1 = 10 + rand() % 10,
        num_cols2 = 10 + rand() % 10,
        num_rows = 10 + rand() % 10;
    Matrix<Real> M(num_rows, num_cols1);
    InitRand(&M);
    
    Matrix<Real> N(num_rows, num_cols2), O(num_rows, num_cols2);
    std::vector<int32> reorder(num_cols2);
    for (int32 i = 0; i < num_cols2; i++)
      reorder[i] = -1 + (rand() % (num_cols1 + 1));
    
    N.CopyCols(M, reorder);
    
    for (int32 i = 0; i < num_rows; i++)
      for (int32 j = 0; j < num_cols2; j++)
        if (reorder[j] < 0) O(i, j) = 0;
        else O(i, j) = M(i, reorder[j]);
    KALDI_LOG << "M is " << M;
    KALDI_LOG << "N is " << N;
    KALDI_LOG << "O is " << O;
    AssertEqual(N, O);
  }
}


template<typename Real>
static void UnitTestSimpleForMat() {  // test some simple operates on all kinds of matrix

  for (MatrixIndexT p = 0; p < 10; p++) {
    // for FrobeniousNorm() function
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);
    {
      Matrix<Real> N(M);
      Real a = M.LogSumExp(), b = N.ApplySoftMax();
      AssertEqual(a, b);
      AssertEqual(1.0, N.Sum());
    }
    {
      Matrix<Real> N(M);
      N.Add(2.0);
      for (MatrixIndexT m = 0; m < dimM; m++)
        for (MatrixIndexT n = 0; n < dimN; n++)
          N(m, n) -= 2.0;
      AssertEqual(M, N);
    }

    Matrix<Real> N(M), M1(M);
    M1.MulElements(M);
    Real tmp1 = sqrt(M1.Sum());
    Real tmp2 = N.FrobeniusNorm();
    KALDI_ASSERT(std::abs(tmp1 - tmp2) < 0.00001);

    // for LargestAbsElem() function
    Vector<Real> V(dimM);
    for (MatrixIndexT i = 0; i < dimM; i++) {
      for (MatrixIndexT j = 0; j < dimN; j++) {
        M(i, j) = std::abs(M(i, j));
      }
      std::sort(M.RowData(i), M.RowData(i) + dimN);
      V(i) = M(i, dimN - 1);
    }
    std::sort(V.Data(), V.Data() + dimM);
    KALDI_ASSERT(std::abs(V(dimM - 1) - N.LargestAbsElem()) < 0.00001);
  }

  SpMatrix<Real> x(3);
  x.SetZero();

  std::stringstream ss;

  ss << "DP 3\n";
  ss << "4.6863" << '\n';
  ss << "3.7062 4.6032" << '\n';
  ss << "3.4160 3.7256  5.2474" << '\n';

  ss >> x;
  KALDI_ASSERT(x.IsPosDef() == true);  // test IsPosDef() function

  TpMatrix<Real> y(3);
  y.SetZero();
  y.Cholesky(x);

  KALDI_LOG << "Matrix y is a lower triangular Cholesky decomposition of x:"
            << '\n';
  KALDI_LOG << y << '\n';

  // test sp-matrix's LogPosDefDet() function
  Matrix<Real> B(x);
  Real tmp;
  Real *DetSign = &tmp;
  KALDI_ASSERT(std::abs(B.LogDet(DetSign) - x.LogPosDefDet()) < 0.00001);

  for (MatrixIndexT p = 0; p < 10; p++) {  // test for sp and tp matrix's AddSp() and AddTp() function
    MatrixIndexT dimM = 10 + rand() % 10;
    SpMatrix<Real> S(dimM), S1(dimM);
    TpMatrix<Real> T(dimM), T1(dimM);
    InitRand(&S);
    InitRand(&S1);
    InitRand(&T);
    InitRand(&T1);
    Matrix<Real> M(S), M1(S1), N(T), N1(T1);

    S.AddSp(1.0, S1);
    T.AddTp(1.0, T1);
    M.AddMat(1.0, M1);
    N.AddMat(1.0, N1);
    Matrix<Real> S2(S);
    Matrix<Real> T2(T);

    AssertEqual(S2, M);
    AssertEqual(T2, N);
  }

  for (MatrixIndexT i = 0; i < 10; i++) {  // test for sp matrix's AddVec2() function
    MatrixIndexT dimM = 10 + rand() % 10;
    SpMatrix<Real> M(dimM);
    Vector<Real> V(dimM);

    InitRand(&M);
    SpMatrix<double> Md(M);
    InitRand(&V);
    SpMatrix<Real> Sorig(M);
    M.AddVec2(0.5, V);
    Md.AddVec2(static_cast<Real>(0.5), V);
    for (MatrixIndexT i = 0; i < dimM; i++)
      for (MatrixIndexT j = 0; j < dimM; j++) {
        KALDI_ASSERT(std::abs(M(i, j) - (Sorig(i, j)+0.5*V(i)*V(j))) < 0.001);
        KALDI_ASSERT(std::abs(Md(i, j) - (Sorig(i, j)+0.5*V(i)*V(j))) < 0.001);
      }
  }
}


template<typename Real> static void UnitTestRow() {

  for (MatrixIndexT p = 0;p< 10;p++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);

    MatrixIndexT i = rand() % dimM;  // Row to get.

    Vector<Real> V(dimN);
    V.CopyRowFromMat(M, i);  // get row.
    for (MatrixIndexT k = 0;k < dimN;k++) {
      AssertEqual(M(i, k), V(k));
    }

    {
      SpMatrix<Real> S(dimN);
      InitRand(&S);
      Vector<Real> v1(dimN), v2(dimN);
      Matrix<Real> M(S);
      MatrixIndexT dim2 = rand() % dimN;
      v1.CopyRowFromSp(S, dim2);
      v2.CopyRowFromMat(M, dim2);
      AssertEqual(v1, v2);
    }
    
    MatrixIndexT j = rand() % dimN;  // Col to get.
    Vector<Real> W(dimM);
    W.CopyColFromMat(M, j);  // get row.
    for (MatrixIndexT k = 0;k < dimM;k++) {
      AssertEqual(M(k, j), W(k));
    }

  }
}

template<typename Real> static void UnitTestAxpy() {

  for (MatrixIndexT i = 0;i< 10;i++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;
    Matrix<Real> M(dimM, dimN), N(dimM, dimN), O(dimN, dimM);

    InitRand(&M); InitRand(&N); InitRand(&O);
    Matrix<Real> Morig(M);
    M.AddMat(0.5, N);
    for (MatrixIndexT i = 0;i < dimM;i++)
      for (MatrixIndexT j = 0;j < dimN;j++)
        KALDI_ASSERT(std::abs(M(i, j) - (Morig(i, j)+0.5*N(i, j))) < 0.1);
    M.CopyFromMat(Morig);
    M.AddMat(0.5, O, kTrans);
    for (MatrixIndexT i = 0;i < dimM;i++)
      for (MatrixIndexT j = 0;j < dimN;j++)
        KALDI_ASSERT(std::abs(M(i, j) - (Morig(i, j)+0.5*O(j, i))) < 0.1);
    {
      float f = 0.5 * (float) (rand() % 3);
      Matrix<Real> N(dimM, dimM);
      InitRand(&N);

      Matrix<Real> N2(N);
      Matrix<Real> N3(N);
      N2.AddMat(f, N2, kTrans);
      N3.AddMat(f, N, kTrans);
      AssertEqual(N2, N3);  // check works same with self as arg.
    }
  }
}

template<typename Real> static void UnitTestCopySp() {
  // Checking that the various versions of copying
  // matrix to SpMatrix work the same in the symmetric case.
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    int32 dim = 5 + rand() %  10;
    SpMatrix<Real> S(dim), T(dim);
    S.SetRandn();
    Matrix<Real> M(S);
    T.CopyFromMat(M, kTakeMeanAndCheck);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeMean);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeLower);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeUpper);
    AssertEqual(S, T);
  }
}


template<typename Real> static void UnitTestPower() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    // this is for matrix-pow
    MatrixIndexT dimM = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimM), N(dimM, dimM);
    InitRand(&M);
    N.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);  // N:=M*M^T.
    SpMatrix<Real> S(dimM);
    S.CopyFromMat(N);  // symmetric so should not crash.
    S.ApplyPow(0.5);
    S.ApplyPow(2.0);
    M.CopyFromSp(S);
    AssertEqual(M, N);

    // this is for vector-pow
    MatrixIndexT dimV = 10 + rand() % 10;
    Vector<Real> V(dimV), V1(dimV), V2(dimV);
    InitRand(&V);
    V1.AddVecVec(1.0, V, V, 0.0);  // V1:=V.*V.
    V2.CopyFromVec(V1);
    V2.ApplyPow(0.5);
    V2.ApplyPow(2.0);
    AssertEqual(V1, V2);
  }
}

template<typename Real> static void UnitTestHeaviside() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN), N(dimM, dimN);
    InitRand(&M);
    N = M;
    N.ApplyHeaviside();
    for (MatrixIndexT r = 0; r < dimM; r++) {
      for (MatrixIndexT c = 0; c < dimN; c++) {
        Real x = M(r, c), y = N(r, c);
        if (x < 0.0) KALDI_ASSERT(y == 0.0);
        if (x > 0.0) KALDI_ASSERT(y == 1.0);
        if (x == 0.0) { KALDI_ASSERT(y >= 0.0 && y <= 1.0); }
      }
    }
  }
}


template<typename Real> static void UnitTestAddOuterProductPlusMinus() {
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    MatrixIndexT dimM = 10 + rand() % 10;
    MatrixIndexT dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN), Plus(dimM, dimN), Minus(dimM, dimN),
        M2(dimM, dimN);
    Vector<Real> v1(dimM), v2(dimN);

    for (MatrixIndexT i = 0; i < 5; i++) {
      InitRand(&v1);
      InitRand(&v2);
      Real alpha = 0.333 * ((rand() % 10) - 5);
      M.AddVecVec(alpha, v1, v2);
         
      AddOuterProductPlusMinus(alpha, v1, v2, &Plus, &Minus);
      M2.SetZero();
      M2.AddMat(-1.0, Minus);
      M2.AddMat(1.0, Plus);
      AssertEqual(M, M2);
      KALDI_ASSERT(Minus.Min() >= 0);
      KALDI_ASSERT(Plus.Min() >= 0);
    }
  }
}

template<typename Real> static void UnitTestSger() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    MatrixIndexT dimM = 10 + rand() % 10;
    MatrixIndexT dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN), M2(dimM, dimN);
    Vector<Real> v1(dimM); InitRand(&v1);
    Vector<Real> v2(dimN); InitRand(&v2);
    Vector<double> v1d(v1), v2d(v2);
    M.AddVecVec(1.0f, v1, v2);
    M2.AddVecVec(1.0f, v1, v2);
    for (MatrixIndexT m = 0;m < dimM;m++)
      for (MatrixIndexT n = 0;n < dimN;n++) {
        KALDI_ASSERT(M(m, n) - v1(m)*v2(n) < 0.01);
        KALDI_ASSERT(M(m, n) - M2(m, n) < 0.01);
      }
  }
}



template<typename Real> static void UnitTestDeterminant() {  // also tests matrix axpy and IsZero() and TraceOfProduct{, T}
  for (MatrixIndexT iter = 0;iter < 5;iter++) {  // First test the 2 det routines are the same
	int dimM = 10 + rand() % 10;
	Matrix<Real> M(dimM, dimM), N(dimM, dimM);
	InitRand(&M);
	N.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);  // N:=M*M^T.
	for (MatrixIndexT i = 0;i < (MatrixIndexT)dimM;i++) N(i, i) += 0.0001;  // Make sure numerically +ve det-- can by chance be almost singular the way we initialized it (I think)
	SpMatrix<Real> S(dimM);
	S.CopyFromMat(N);  // symmetric so should not crash.
	Real logdet = S.LogPosDefDet();
	Real logdet2, logdet3, sign2, sign3;
	logdet2 = N.LogDet(&sign2);
    logdet3 = S.LogDet(&sign3);
	KALDI_ASSERT(sign2 == 1.0 && sign3 == 1.0 && std::abs(logdet2-logdet) < 0.1 && std::abs(logdet2 - logdet3) < 0.1);
	Matrix<Real> tmp(dimM, dimM); tmp.SetZero();
	tmp.AddMat(1.0, N);
	tmp.AddMat(-1.0, N, kTrans);
	// symmetric so tmp should be zero.
	if ( ! tmp.IsZero(1.0e-04)) {
	  printf("KALDI_ERR: matrix is not zero\n");
	  KALDI_LOG << tmp;
	  KALDI_ASSERT(0);
	}

	Real a = TraceSpSp(S, S), b = TraceMatMat(N, N), c = TraceMatMat(N, N, kTrans);
	KALDI_ASSERT(std::abs(a-b) < 0.1 && std::abs(b-c) < 0.1);
  }
}


template<typename Real> static void UnitTestDeterminantSign() {

  for (MatrixIndexT iter = 0;iter < 20;iter++) {  // First test the 2 det routines are the same
	int dimM = 10 + rand() % 10;
	Matrix<Real> M(dimM, dimM), N(dimM, dimM);
	InitRand(&M);
	N.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);  // N:=M*M^T.
	for (MatrixIndexT i = 0;i < (MatrixIndexT)dimM;i++) N(i, i) += 0.0001;  // Make sure numerically +ve det-- can by chance be almost singular the way we initialized it (I think)
	SpMatrix<Real> S(dimM);
	S.CopyFromMat(N);  // symmetric so should not crash.
	Real logdet = S.LogPosDefDet();
	Real logdet2, logdet3, sign2, sign3;
	logdet2 = N.LogDet(&sign2);
    logdet3 = S.LogDet(&sign3);
	KALDI_ASSERT(sign2 == 1.0 && sign3 == 1.0 && std::abs(logdet2-logdet) < 0.01 && std::abs(logdet2 - logdet3) < 0.01);

    MatrixIndexT num_sign_changes = rand() % 5;
    for (MatrixIndexT change = 0; change < num_sign_changes; change++) {
      // Change sign of S's det by flipping one eigenvalue, and N by flipping one row.
      {
        Matrix<Real> M(S);
        Matrix<Real> U(dimM, dimM), Vt(dimM, dimM);
        Vector<Real> s(dimM);
        M.Svd(&s, &U, &Vt);  // SVD: M = U diag(s) Vt
        s(rand() % dimM) *= -1;
        U.MulColsVec(s);
        M.AddMatMat(1.0, U, kNoTrans, Vt, kNoTrans, 0.0);
        S.CopyFromMat(M);
      }
      // change sign of N:
      N.Row(rand() % dimM).Scale(-1.0);
    }

    // add in a scaling factor too.
    Real tmp = 1.0 + ((rand() % 5) * 0.01);
    Real logdet_factor = dimM * log(tmp);
    N.Scale(tmp);
    S.Scale(tmp);

    Real logdet4, logdet5, sign4, sign5;
	logdet4 = N.LogDet(&sign4);
    logdet5 = S.LogDet(&sign5);
    AssertEqual(logdet4, logdet+logdet_factor, 0.01);
    AssertEqual(logdet5, logdet+logdet_factor, 0.01);
    if (num_sign_changes % 2 == 0) {
      KALDI_ASSERT(sign4 == 1);
      KALDI_ASSERT(sign5 == 1);
    } else {
      KALDI_ASSERT(sign4 == -1);
      KALDI_ASSERT(sign5 == -1);
    }
  }
}

template<typename Real> static void UnitTestSpVec() {
  // Test conversion back and forth between SpMatrix and Vector.
  for (MatrixIndexT iter = 0;iter < 1;iter++) {
	MatrixIndexT dimM =10;  // 20 + rand()%10;
    SpMatrix<Real> A(dimM), B(dimM);
    SubVector<Real> vec(A);
    B.CopyFromVec(vec);
    AssertEqual(A, B);
  }
}


template<typename Real> static void UnitTestSherman() {
  for (MatrixIndexT iter = 0;iter < 1;iter++) {
	MatrixIndexT dimM =10;  // 20 + rand()%10;
	MatrixIndexT dimK =2;  // 20 + rand()%dimM;
	Matrix<Real> A(dimM, dimM), U(dimM, dimK), V(dimM, dimK);
	InitRand(&A);
	InitRand(&U);
	InitRand(&V);
	for (MatrixIndexT i = 0;i < (MatrixIndexT)dimM;i++) A(i, i) += 0.0001;  // Make sure numerically +ve det-- can by chance be almost singular the way we initialized it (I think)

	Matrix<Real> tmpL(dimM, dimM);
	tmpL.AddMatMat(1.0, U, kNoTrans, V, kTrans, 0.0);  // tmpL =U *V.
	tmpL.AddMat(1.0, A);
	tmpL.Invert();

	Matrix<Real> invA(dimM, dimM);
	invA.CopyFromMat(A);
	invA.Invert();

	Matrix<Real> tt(dimK, dimM), I(dimK, dimK);
	tt.AddMatMat(1.0, V, kTrans, invA, kNoTrans, 0.0);  // tt0 = trans(V) *inv(A)

	Matrix<Real> tt1(dimM, dimK);
	tt1.AddMatMat(1.0, invA, kNoTrans, U, kNoTrans, 0.0);  // tt1=INV A *U
	Matrix<Real> tt2(dimK, dimK);
	tt2.AddMatMat(1.0, V, kTrans, tt1, kNoTrans, 0.0);
	for (MatrixIndexT i = 0;i < dimK;i++)
      for (MatrixIndexT j = 0;j < dimK;j++)
      {
        if (i == j)
          I(i, j) = 1.0;
        else
          I(i, j) = 0.0;
      }
	tt2.AddMat(1.0, I);   // I = identity
	tt2.Invert();

	Matrix<Real> tt3(dimK, dimM);
	tt3.AddMatMat(1.0, tt2, kNoTrans, tt, kNoTrans, 0.0);	// tt = tt*tran(V)
	Matrix<Real> tt4(dimM, dimM);
	tt4.AddMatMat(1.0, U, kNoTrans, tt3, kNoTrans, 0.0);	// tt = U*tt
	Matrix<Real> tt5(dimM, dimM);
	tt5.AddMatMat(1.0, invA, kNoTrans, tt4, kNoTrans, 0.0);	// tt = inv(A)*tt

	Matrix<Real> tmpR(dimM, dimM);
	tmpR.CopyFromMat(invA);
	tmpR.AddMat(-1.0, tt5);
	// printf("#################%f###############################.%f##############", tmpL, tmpR);

	AssertEqual<Real>(tmpL, tmpR);	// checks whether LHS = RHS or not...

  }
}


template<typename Real> static void UnitTestTraceProduct() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {  // First test the 2 det routines are the same
	int dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
	Matrix<Real> M(dimM, dimN), N(dimM, dimN);

	InitRand(&M);
	InitRand(&N);
	Matrix<Real> Nt(N, kTrans);
	Real a = TraceMatMat(M, Nt), b = TraceMatMat(M, N, kTrans);
	printf("m = %d, n = %d\n", dimM, dimN);
	KALDI_LOG << a << " " << b << '\n';
	KALDI_ASSERT(std::abs(a-b) < 0.1);
  }
}

template<typename Real> static void UnitTestSvd() {
  MatrixIndexT Base = 3, Rand = 2, Iter = 25;
  for (MatrixIndexT iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, std::min(dimM, dimN)), Vt(std::min(dimM, dimN), dimN);
    Vector<Real> s(std::min(dimM, dimN));
	InitRand(&M);
	if (iter < 2) KALDI_LOG << "M " << M;
	Matrix<Real> M2(dimM, dimN); M2.CopyFromMat(M);
	M.Svd(&s, &U, &Vt);
	if (iter < 2) {
	  KALDI_LOG << " s " << s;
	  KALDI_LOG << " U " << U;
	  KALDI_LOG << " Vt " << Vt;
	}
    MatrixIndexT min_dim = std::min(dimM, dimN);
    Matrix<Real> S(min_dim, min_dim);
    S.CopyDiagFromVec(s);
	Matrix<Real> Mtmp(dimM, dimN);
    Mtmp.SetZero();
    Mtmp.AddMatMatMat(1.0, U, kNoTrans, S, kNoTrans, Vt, kNoTrans, 0.0);
    AssertEqual(Mtmp, M2);
  }
}

template<typename Real> static void UnitTestSvdBad() {
  MatrixIndexT N = 20;
  {
    Matrix<Real> M(N, N);
    // M.Set(1591.3614306764898);
    M.Set(1.0);
    M(0, 0) *= 1.000001;
    Matrix<Real> U(N, N), V(N, N);
    Vector<Real> l(N);
    M.Svd(&l, &U, &V);
  }
  SpMatrix<Real> M(N);
  for(MatrixIndexT i =0; i < N; i++)
    for(MatrixIndexT j = 0; j <= i; j++)
      M(i, j) = 1591.3614306764898;
  M(0, 0) *= 1.00001;
  M(10, 10) *= 1.00001;
  Matrix<Real> U(N, N);
  Vector<Real> l(N);
  M.SymPosSemiDefEig(&l, &U);
}


template<typename Real> static void UnitTestSvdZero() {
  MatrixIndexT Base = 3, Rand = 2, Iter = 30;
  for (MatrixIndexT iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;  // M>=N.
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, dimM), Vt(dimN, dimN); Vector<Real> v(std::min(dimM, dimN));
    if (iter%2 == 0) M.SetZero();
    else M.Unit();
	if (iter < 2) KALDI_LOG << "M " << M;
	Matrix<Real> M2(dimM, dimN); M2.CopyFromMat(M);
    bool ans = M.Svd(&v, &U, &Vt);
	KALDI_ASSERT(ans);  // make sure works with zero matrix.
  }
}





template<typename Real> static void UnitTestSvdNodestroy() {
  MatrixIndexT Base = 3, Rand = 2, Iter = 25;
  for (MatrixIndexT iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimN = Base + rand() % Rand, dimM =  dimN + rand() % Rand;  // M>=N, as required by JAMA Svd.
    MatrixIndexT minsz = std::min(dimM, dimN);
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, minsz), Vt(minsz, dimN); Vector<Real> v(minsz);
	InitRand(&M);
	if (iter < 2) KALDI_LOG << "M " << M;
	M.Svd(&v, &U, &Vt);
	if (iter < 2) {
	  KALDI_LOG << " v " << v;
	  KALDI_LOG << " U " << U;
	  KALDI_LOG << " Vt " << Vt;
	}

    for (MatrixIndexT it = 0;it < 2;it++) {
      Matrix<Real> Mtmp(minsz, minsz);
      for (MatrixIndexT i = 0;i < v.Dim();i++) Mtmp(i, i) = v(i);
      Matrix<Real> Mtmp2(minsz, dimN);
      Mtmp2.AddMatMat(1.0, Mtmp, kNoTrans, Vt, kNoTrans, 0.0);
      Matrix<Real> Mtmp3(dimM, dimN);
      Mtmp3.AddMatMat(1.0, U, kNoTrans, Mtmp2, kNoTrans, 0.0);
      for (MatrixIndexT i = 0;i < Mtmp.NumRows();i++) {
        for (MatrixIndexT j = 0;j < Mtmp.NumCols();j++) {
          KALDI_ASSERT(std::abs(Mtmp3(i, j) - M(i, j)) < 0.0001);
        }
      }

      SortSvd(&v, &U, &Vt);  // and re-check...
      for (MatrixIndexT i = 0; i + 1 < v.Dim(); i++) // check SortSvd is working.
        KALDI_ASSERT(std::abs(v(i+1)) <= std::abs(v(i)));
    }
  }
}


/*
  template<typename Real> static void UnitTestSvdVariants() {  // just make sure it doesn't crash if we call it but don't want left or right singular vectors. there are KALDI_ASSERTs inside the Svd.
  #ifndef HAVE_ATLAS
  MatrixIndexT Base = 10, Rand = 5, Iter = 25;
  for (MatrixIndexT iter = 0;iter < Iter;iter++) {
  MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;
  // if (dimM<dimN) std::swap(dimM, dimN);  // M>=N.
  Matrix<Real> M(dimM, dimN);
  Matrix<Real> U(dimM, dimM), Vt(dimN, dimN); Vector<Real> v(std::min(dimM, dimN));
  Matrix<Real> Utmp(dimM, 1); Matrix<Real> Vttmp(1, dimN);
  InitRand(&M);
  M.Svd(v, U, Vttmp, "A", "N");
  M.Svd(v, Utmp, Vt, "N", "A");
  Matrix<Real> U2(dimM, dimM), Vt2(dimN, dimN); Vector<Real> v2(std::min(dimM, dimN));
  M.Svd(v, U2, Vt2, "A", "A");
  AssertEqual(U, U2); AssertEqual(Vt, Vt2);

  }
  #endif
  }*/

template<typename Real> static void UnitTestSvdJustvec() {  // Making sure gives same answer if we get just the vector, not the eigs.
  MatrixIndexT Base = 10, Rand = 5, Iter = 25;
  for (MatrixIndexT iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;  // M>=N.
    MatrixIndexT minsz = std::min(dimM, dimN);

	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, minsz), Vt(minsz, dimN); Vector<Real> v(minsz);
	M.Svd(&v, &U, &Vt);
    Vector<Real> v2(minsz);
    M.Svd(&v2);
    AssertEqual(v, v2);
  }
}

template<typename Real> static void UnitTestEigSymmetric() {

  for (MatrixIndexT iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10;
    SpMatrix<Real> S(dimM);
    InitRand(&S);
    Matrix<Real> M(S);  // copy to regular matrix.
    Matrix<Real> P(dimM, dimM);
    Vector<Real> real_eigs(dimM), imag_eigs(dimM);
    M.Eig(&P, &real_eigs, &imag_eigs);
    KALDI_ASSERT(imag_eigs.Sum() == 0.0);
    // should have M = P D P^T
    Matrix<Real> tmp(P); tmp.MulColsVec(real_eigs);  // P * eigs
    Matrix<Real> M2(dimM, dimM);
    M2.AddMatMat(1.0, tmp, kNoTrans, P, kTrans, 0.0);  // M2 = tmp * Pinv = P * eigs * P^T
    AssertEqual(M, M2);  // check reconstruction worked.
  }
}

template<typename Real> static void UnitTestEig() {

  for (MatrixIndexT iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 1 + iter;
    /*    if (iter < 10)
          dimM = 1 + rand() % 6;
          else
          dimM = 5 + rand()%10; */
    Matrix<Real> M(dimM, dimM);
    InitRand(&M);
    Matrix<Real> P(dimM, dimM);
    Vector<Real> real_eigs(dimM), imag_eigs(dimM);
    M.Eig(&P, &real_eigs, &imag_eigs);

    {  // Check that the eigenvalues match up with the determinant.
      BaseFloat logdet_check = 0.0, logdet = M.LogDet();
      for (MatrixIndexT i = 0; i < dimM ; i++)
        logdet_check += 0.5 * log(real_eigs(i)*real_eigs(i) + imag_eigs(i)*imag_eigs(i));
      AssertEqual(logdet_check, logdet);
    }
    Matrix<Real> Pinv(P);
    Pinv.Invert();
    Matrix<Real> D(dimM, dimM);
    CreateEigenvalueMatrix(real_eigs, imag_eigs, &D);
    
    // check that M = P D P^{-1}.
    Matrix<Real> tmp(dimM, dimM);
    tmp.AddMatMat(1.0, P, kNoTrans, D, kNoTrans, 0.0);  // tmp = P * D
    Matrix<Real> M2(dimM, dimM);
    M2.AddMatMat(1.0, tmp, kNoTrans, Pinv, kNoTrans, 0.0);  // M2 = tmp * Pinv = P * D * Pinv.

    {  // print out some stuff..
      Matrix<Real> MP(dimM, dimM);
      MP.AddMatMat(1.0, M, kNoTrans, P, kNoTrans, 0.0);
      Matrix<Real> PD(dimM, dimM);
      PD.AddMatMat(1.0, P, kNoTrans, D, kNoTrans, 0.0);

      Matrix<Real> PinvMP(dimM, dimM);
      PinvMP.AddMatMat(1.0, Pinv, kNoTrans, MP, kNoTrans, 0.0);
      AssertEqual(MP, PD);
    }

    AssertEqual(M, M2);  // check reconstruction worked.
  }
}


template<typename Real> static void UnitTestEigSp() {
  // Test the Eig function with SpMatrix.
  // We make sure to test pathological cases, that have
  // either large zero eigenspaces, or two large
  // eigenspaces with the same absolute value but +ve
  // and -ve.  Also zero matrix.
  
  for (MatrixIndexT iter = 0; iter < 100; iter++) {
	MatrixIndexT dimM = 1 + (rand() % 10);
    SpMatrix<Real> S(dimM);

    switch (iter % 5) {
      case 0: // zero matrix.
        break;
      case 1: // general random symmetric matrix.
        InitRand(&S);
        break;
      default:
        { // start with a random matrix; do decomposition; change the eigenvalues to
          // try to cover the problematic cases; reconstruct.
          InitRand(&S);
          Vector<Real> s(dimM); Matrix<Real> P(dimM, dimM);
          S.Eig(&s, &P);
          // We on purpose create a problematic case where
          // some eigs are either zero or share a value (+ve or -ve)
          // with some other eigenvalue.
          for (MatrixIndexT i = 0; i < dimM; i++) {
            if (rand() % 10 == 0) s(i) = 0; // set that eig to zero.
            else if (rand() % 10 < 2) {
              // set that eig to some other randomly chosen eig,
              // times random sign.
              s(i) = (rand()%2 == 0 ? 1 : -1) * s(rand() % dimM);
            }
          }
          // Reconstruct s from the eigenvalues "made problematic."
          S.AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
          Real *data = s.Data();
          std::sort(data, data+dimM);
          KALDI_LOG << "Real eigs are: " << s;

        }
    }
    Vector<Real> s(dimM); Matrix<Real> P(dimM, dimM);
    S.Eig(&s, &P);
    KALDI_LOG << "Found eigs are: " << s;
    SpMatrix<Real> S2(dimM);
    S2.AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
    {
      SpMatrix<Real> diff(S);
      diff.AddSp(-1.0, S2);
      Vector<Real> s(dimM); Matrix<Real> P(dimM, dimM);
      diff.Eig(&s, &P);
      KALDI_LOG << "Eigs of difference are " << s;
    }
    KALDI_ASSERT(S.ApproxEqual(S2, 1.0e-03f));
  }    
}

// TEMP!
template<typename Real>
static Real NonOrthogonality(const MatrixBase<Real> &M, MatrixTransposeType transM) {
  SpMatrix<Real> S(transM == kTrans ? M.NumCols() : M.NumRows());
  S.SetUnit();
  S.AddMat2(-1.0, M, transM, 1.0);
  Real max = 0.0;
  for (MatrixIndexT i = 0; i < S.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      max = std::max(max, std::abs(S(i, j)));
  return max;
}

template<typename Real>
static Real NonDiagonalness(const SpMatrix<Real> &S) {
  Real max_diag = 0.0, max_offdiag = 0.0;
  for (MatrixIndexT i = 0; i < S.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++) {
      if (i == j) { max_diag = std::max(max_diag, std::abs(S(i, j))); }
      else {  max_offdiag = std::max(max_offdiag, std::abs(S(i, j))); }
    }
  if (max_diag == 0.0) {
    if (max_offdiag == 0.0) return 0.0; // perfectly diagonal.
    else return 1.0; // perfectly non-diagonal.
  } else {
    return max_offdiag / max_diag;
  }
}


template<typename Real>
static Real NonUnitness(const SpMatrix<Real> &S) {
  SpMatrix<Real> tmp(S.NumRows());
  tmp.SetUnit();
  tmp.AddSp(-1.0, S);
  Real max = 0.0;
  for (MatrixIndexT i = 0; i < tmp.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      max = std::max(max, std::abs(tmp(i, j)));
  return max;
}

template<typename Real>
static void UnitTestTridiagonalize() {

  {
    float tmp[5];
    tmp[4] = 1.0;
    cblas_Xspmv(1, 0.0, tmp+2,
                tmp+1, 1, 0.0, tmp+4, 1);
    KALDI_ASSERT(tmp[4] == 0.0);
  }
  for (MatrixIndexT i = 0; i < 4; i++) {
    MatrixIndexT dim = 40 + rand() % 4;
    SpMatrix<Real> S(dim), S2(dim), R(dim), S3(dim);
    Matrix<Real> Q(dim, dim);
    InitRand(&S);
    SpMatrix<Real> T(S);
    T.Tridiagonalize(&Q);
    KALDI_LOG << "S trace " << S.Trace() << ", T trace " << T.Trace();
    //KALDI_LOG << S << "\n" << T;
    AssertEqual(S.Trace(), T.Trace());
    // Also test Trace().
    Real ans = 0.0;
    for (MatrixIndexT j = 0; j < dim; j++) ans += T(j, j);
    AssertEqual(ans, T.Trace());
    AssertEqual(T.LogDet(), S.LogDet());
    R.AddMat2(1.0, Q, kNoTrans, 0.0);
    KALDI_LOG << "Non-unit-ness of R is " << NonUnitness(R);
    KALDI_ASSERT(R.IsUnit(0.01)); // Check Q is orthogonal.
    S2.AddMat2Sp(1.0, Q, kTrans, T, 0.0);
    S3.AddMat2Sp(1.0, Q, kNoTrans, S, 0.0);
    //KALDI_LOG << "T is " << T;
    //KALDI_LOG << "S is " << S;
    //KALDI_LOG << "S2 (should be like S) is " << S2;
    //KALDI_LOG << "S3 (should be like T) is " << S3;
    AssertEqual(S, S2);
    AssertEqual(T, S3);        
  }
}

template<typename Real>
static void UnitTestTridiagonalizeAndQr() {

  {
    float tmp[5];
    tmp[4] = 1.0;
   // cblas_sspmv(CblasRowMajor, CblasLower, 1, 0.0, tmp+2,
   //             tmp+1, 1, 0.0, tmp+4, 1);
    cblas_Xspmv(1, 0.0, tmp+2,
                tmp+1, 1, 0.0, tmp+4, 1);

    KALDI_ASSERT(tmp[4] == 0.0);
  }
  for (MatrixIndexT i = 0; i < 4; i++) {
    MatrixIndexT dim = 50 + rand() % 4;
    SpMatrix<Real> S(dim), S2(dim), R(dim), S3(dim), S4(dim);
    Matrix<Real> Q(dim, dim);
    InitRand(&S);
    SpMatrix<Real> T(S);
    T.Tridiagonalize(&Q);
    KALDI_LOG << "S trace " << S.Trace() << ", T trace " << T.Trace();
    // KALDI_LOG << S << "\n" << T;
    AssertEqual(S.Trace(), T.Trace());
    // Also test Trace().
    Real ans = 0.0;
    for (MatrixIndexT j = 0; j < dim; j++) ans += T(j, j);
    AssertEqual(ans, T.Trace());
    AssertEqual(T.LogDet(), S.LogDet());
    R.AddMat2(1.0, Q, kNoTrans, 0.0);
    KALDI_LOG << "Non-unit-ness of R after tridiag is " << NonUnitness(R);
    KALDI_ASSERT(R.IsUnit(0.001)); // Check Q is orthogonal.
    S2.AddMat2Sp(1.0, Q, kTrans, T, 0.0);
    S3.AddMat2Sp(1.0, Q, kNoTrans, S, 0.0);
    //KALDI_LOG << "T is " << T;
    //KALDI_LOG << "S is " << S;
    //KALDI_LOG << "S2 (should be like S) is " << S2;
    //KALDI_LOG << "S3 (should be like T) is " << S3;
    AssertEqual(S, S2);
    AssertEqual(T, S3);
    SpMatrix<Real> T2(T);
    T2.Qr(&Q);
    R.AddMat2(1.0, Q, kNoTrans, 0.0);
    KALDI_LOG << "Non-unit-ness of R after QR is " << NonUnitness(R);
    KALDI_ASSERT(R.IsUnit(0.001)); // Check Q is orthogonal.
    AssertEqual(T.Trace(), T2.Trace());
    KALDI_ASSERT(T2.IsDiagonal());
    AssertEqual(T.LogDet(), T2.LogDet());
    S4.AddMat2Sp(1.0, Q, kTrans, T2, 0.0);
    //KALDI_LOG << "S4 (should be like S) is " << S4;
    AssertEqual(S, S4);
  }
}


template<typename Real> static void UnitTestMmul() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10, dimO = 20 + rand()%10;  // dims between 10 and 20.
	// MatrixIndexT dimM = 2, dimN = 3, dimO = 4;
	Matrix<Real> A(dimM, dimN), B(dimN, dimO), C(dimM, dimO);
	InitRand(&A);
	InitRand(&B);
	//
	// KALDI_LOG <<"a = " << A;
	// KALDI_LOG<<"B = " << B;
	C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);  // C = A * B.
	// 	KALDI_LOG << "c = " << C;
	for (MatrixIndexT i = 0;i < dimM;i++) {
	  for (MatrixIndexT j = 0;j < dimO;j++) {
		double sum = 0.0;
		for (MatrixIndexT k = 0;k < dimN;k++) {
		  sum += A(i, k) * B(k, j);
		}
		KALDI_ASSERT(std::abs(sum - C(i, j)) < 0.0001);
	  }
	}
  }
}


template<typename Real> static void UnitTestMmulSym() {

  // Test matrix multiplication on symmetric matrices.
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10;

	Matrix<Real> A(dimM, dimM), B(dimM, dimM), C(dimM, dimM), tmp(dimM, dimM), tmp2(dimM, dimM);
    SpMatrix<Real> sA(dimM), sB(dimM), sC(dimM), stmp(dimM);
    
	InitRand(&A); InitRand(&B); InitRand(&C);
    // Make A, B, C symmetric.
    tmp.CopyFromMat(A); A.AddMat(1.0, tmp, kTrans);
    tmp.CopyFromMat(B); B.AddMat(1.0, tmp, kTrans);
    tmp.CopyFromMat(C); C.AddMat(1.0, tmp, kTrans);

    sA.CopyFromMat(A);
    sB.CopyFromMat(B);
    sC.CopyFromMat(C);


    tmp.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);  // tmp = A * B.
    tmp2.AddSpSp(1.0, sA, sB, 0.0);  // tmp = sA*sB.
    AssertEqual(tmp, tmp2);
    tmp2.AddSpSp(1.0, sA, sB, 0.0);  // tmp = sA*sB.
    AssertEqual(tmp, tmp2);
  }
}


template<typename Real> static void UnitTestAddVecVec() {
  for (int32 i = 0; i < 20; i++) {
    int32 dimM = 5 + rand() % 10, dimN = 5 + rand() % 10;
    
    Matrix<Real> M(dimM, dimN);
    M.SetRandn();
    Matrix<Real> N(M);
    Vector<float> v(dimM), w(dimN);
    v.SetRandn();
    w.SetRandn();
    float alpha = 0.2 * (rand() % 10);
    M.AddVecVec(alpha, v, w);
    for (int32 j = 0; j < 20; j++) {
      int32 dimX = rand() % dimM, dimY = rand() % dimN;
      AssertEqual(M(dimX, dimY),
                  N(dimX, dimY) + alpha * v(dimX) * w(dimY));
    }
  }    
}


template<typename Real> static void UnitTestVecmul() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    MatrixTransposeType trans = (iter % 2 == 0 ? kTrans : kNoTrans);
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10;  // dims between 10 and 20.
    Real alpha = 0.333, beta = 0.5;
	Matrix<Real> A(dimM, dimN);
    if (trans == kTrans) A.Transpose();
	InitRand(&A);
	Vector<Real> x(dimM), y(dimN);
    //x.SetRandn();
	y.SetRandn();
    Vector<Real> orig_x(x), x2(x);

	x.AddMatVec(alpha, A, trans, y, beta);  // x = A * y + beta*x.
    x2.AddMatSvec(alpha, A, trans, y, beta);  // x = A * y + beta*x
    
	for (MatrixIndexT i = 0; i < dimM; i++) {
	  double sum = beta * orig_x(i);
	  for (MatrixIndexT j = 0; j < dimN; j++) {
        if (trans == kNoTrans) {
          sum += alpha * A(i, j) * y(j);
        } else {
          sum += alpha * A(j, i) * y(j);
        }
	  }
	  KALDI_ASSERT(std::abs(sum - x(i)) < 0.0001);
	}
    AssertEqual(x, x2);
  }

}

template<typename Real> static void UnitTestInverse() {
  for (MatrixIndexT iter = 0;iter < 10;iter++) {
	MatrixIndexT dimM = 20 + rand()%10;
	Matrix<Real> A(dimM, dimM), B(dimM, dimM), C(dimM, dimM);
	InitRand(&A);
	B.CopyFromMat(A);
    B.Invert();

	C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);  // C = A * B.


	for (MatrixIndexT i = 0;i < dimM;i++)
	  for (MatrixIndexT j = 0;j < dimM;j++)
		KALDI_ASSERT(std::abs(C(i, j) - (i == j?1.0:0.0)) < 0.1);
  }
}




template<typename Real> static void UnitTestMulElements() {
  for (MatrixIndexT iter = 0; iter < 5; iter++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10;
	Matrix<Real> A(dimM, dimN), B(dimM, dimN), C(dimM, dimN);
	InitRand(&A);
	InitRand(&B);

	C.CopyFromMat(A);
	C.MulElements(B);  // C = A .* B (in Matlab, for example).

	for (MatrixIndexT i = 0;i < dimM;i++)
	  for (MatrixIndexT j = 0;j < dimN;j++)
		KALDI_ASSERT(std::abs(C(i, j) - (A(i, j)*B(i, j))) < 0.0001);
  }
}


template<typename Real> static void UnitTestSpLogExp() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10;

    Matrix<Real> M(dimM, dimM); InitRand(&M);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, M, kNoTrans, 0.0);  // B = M*M^T -> positive definite (since M nonsingular).

    SpMatrix<Real> B2(B);
    B2.Log();
    B2.Exp();
    AssertEqual(B, B2);

    SpMatrix<Real> B3(B);
    B3.Log();
    B3.Scale(0.5);
    B3.Exp();
    Matrix<Real> sqrt(B3);
    SpMatrix<Real> B4(B.NumRows());
    B4.AddMat2(1.0, sqrt, kNoTrans, 0.0);
    AssertEqual(B, B4);
  }
}

template<typename Real> static void UnitTestDotprod() {
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 200 + rand()%100;
	Vector<Real> v(dimM), w(dimM);

	InitRand(&v);
	InitRand(&w);
    Vector<double> wd(w);

	Real f = VecVec(w, v), f2 = VecVec(wd, v), f3 = VecVec(v, wd);
	Real sum = 0.0;
	for (MatrixIndexT i = 0;i < dimM;i++) sum += v(i)*w(i);
	KALDI_ASSERT(std::abs(f-sum) < 0.0001);
    KALDI_ASSERT(std::abs(f2-sum) < 0.0001);
    KALDI_ASSERT(std::abs(f3-sum) < 0.0001);
  }
}

template<typename Real>
static void UnitTestResize() {
  for (size_t i = 0; i < 10; i++) {
    MatrixIndexT dimM1 = rand() % 10, dimN1 = rand() % 10,
        dimM2 = rand() % 10, dimN2 = rand() % 10;
    if (dimM1*dimN1 == 0) dimM1 = dimN1 = 0;
    if (dimM2*dimN2 == 0) dimM2 = dimN2 = 0;
    for (MatrixIndexT j = 0; j < 3; j++) {
      MatrixResizeType resize_type = static_cast<MatrixResizeType>(j);
      Matrix<Real> M(dimM1, dimN1);
      InitRand(&M);
      Matrix<Real> Mcopy(M);
      Vector<Real> v(dimM1);
      InitRand(&v);
      Vector<Real> vcopy(v);
      SpMatrix<Real> S(dimM1);
      InitRand(&S);
      SpMatrix<Real> Scopy(S);
      M.Resize(dimM2, dimN2, resize_type);
      v.Resize(dimM2, resize_type);
      S.Resize(dimM2, resize_type);
      if (resize_type == kSetZero) {
        KALDI_ASSERT(S.IsZero());
        KALDI_ASSERT(v.Sum() == 0.0);
        KALDI_ASSERT(M.IsZero());
      } else if (resize_type == kCopyData) {
        for (MatrixIndexT i = 0; i < dimM2; i++) {
          if (i < dimM1) AssertEqual(v(i), vcopy(i));
          else KALDI_ASSERT(v(i) == 0);
          for (MatrixIndexT j = 0; j < dimN2; j++) {
            if (i < dimM1 && j < dimN1) AssertEqual(M(i, j), Mcopy(i, j));
            else AssertEqual(M(i, j), 0.0);
          }
          for (MatrixIndexT i2 = 0; i2 < dimM2; i2++) {
            if (i < dimM1 && i2 < dimM1) AssertEqual(S(i, i2), Scopy(i, i2));
            else AssertEqual(S(i, i2), 0.0);
          }
        }
      }
    }
  }
}


template<typename Real>
static void UnitTestTp2Sp() {
  // Tests AddTp2Sp()
  for (MatrixIndexT iter = 0; iter < 4; iter++) {
	MatrixIndexT dimM = 10 + rand()%3;    
  
    TpMatrix<Real> T(dimM);
    InitRand(&T);
    SpMatrix<Real> S(dimM);
    InitRand(&S);

    Matrix<Real> M(T);
    for ( MatrixIndexT i = 0; i < dimM; i++)
      for (MatrixIndexT j = 0; j < dimM; j++) {
        if (j <= i) AssertEqual(T(i,j), M(i,j));
        else AssertEqual(M(i,j), 0.0);
      }

    SpMatrix<Real> A(dimM), B(dimM);
    A.AddTp2Sp(0.5, T, (iter < 2 ? kNoTrans : kTrans), S, 0.0);
    B.AddMat2Sp(0.5, M, (iter < 2 ? kNoTrans : kTrans), S, 0.0);
    AssertEqual(A, B);
  }
}

template<typename Real>
static void UnitTestTp2() {
  // Tests AddTp2()
  for (MatrixIndexT iter = 0; iter < 4; iter++) {
	MatrixIndexT dimM = 10 + rand()%3;    
  
    TpMatrix<Real> T(dimM);
    InitRand(&T);

    Matrix<Real> M(T);

    SpMatrix<Real> A(dimM), B(dimM);
    A.AddTp2(0.5, T, (iter < 2 ? kNoTrans : kTrans), 0.0);
    B.AddMat2(0.5, M, (iter < 2 ? kNoTrans : kTrans), 0.0);
    AssertEqual(A, B);
  }
}

template<typename Real>
static void UnitTestAddDiagMat2() {
  for (MatrixIndexT iter = 0; iter < 4; iter++) {
	MatrixIndexT dimM = 10 + rand() % 3,
                 dimN = 1 + rand() % 4;
    Vector<Real> v(dimM);
    InitRand(&v);
    Vector<Real> w(v);
    if (iter % 2 == 1) {
      Matrix<Real> M(dimM, dimN);
      InitRand(&M);
      v.AddDiagMat2(0.5, M, kNoTrans, 0.3);
      Matrix<Real> M2(dimM, dimM);
      M2.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);
      Vector<Real> diag(dimM);
      diag.CopyDiagFromMat(M2);
      w.Scale(0.3);
      w.AddVec(0.5, diag);
      AssertEqual(w, v);
    } else {
      Matrix<Real> M(dimN, dimM);
      InitRand(&M);
      v.AddDiagMat2(0.5, M, kTrans, 0.3);
      Matrix<Real> M2(dimM, dimM);
      M2.AddMatMat(1.0, M, kTrans, M, kNoTrans, 0.0);
      Vector<Real> diag(dimM);
      diag.CopyDiagFromMat(M2);
      w.Scale(0.3);
      w.AddVec(0.5, diag);
      AssertEqual(w, v);
    }
  }
}

template<typename Real>
static void UnitTestAddDiagMatMat() {
  for (MatrixIndexT iter = 0; iter < 4; iter++) {
    BaseFloat alpha = 0.432 + rand() % 5, beta = 0.043 + rand() % 2;
	MatrixIndexT dimM = 10 + rand() % 3,
                 dimN = 5 + rand() % 4;
    Vector<Real> v(dimM);
    Matrix<Real> M_orig(dimM, dimN), N_orig(dimN, dimM);
    M_orig.SetRandn();
    N_orig.SetRandn();
    MatrixTransposeType transM = (iter % 2 == 0 ? kNoTrans : kTrans);
    MatrixTransposeType transN = ((iter/2) % 2 == 0 ? kNoTrans : kTrans);
    Matrix<Real> M(M_orig, transM), N(N_orig, transN);
    
    InitRand(&v);
    Vector<Real> w(v);

    w.AddDiagMatMat(alpha, M, transM, N, transN, beta);
    
    {
      Vector<Real> w2(v);
      Matrix<Real> MN(dimM, dimM);
      MN.AddMatMat(1.0, M, transM, N, transN, 0.0);
      Vector<Real> d(dimM);
      d.CopyDiagFromMat(MN);
      w2.Scale(beta);
      w2.AddVec(alpha, d);
      AssertEqual(w, w2);
    }
  }
}

template<typename Real>
static void UnitTestOrthogonalizeRows() {
  for (MatrixIndexT iter = 0; iter < 100; iter++) {
    MatrixIndexT dimM = 4 + rand() % 5, dimN = dimM + (rand() % 2);
    Matrix<Real> M(dimM, dimN);
    for (MatrixIndexT i = 0; i < dimM; i++) {
      if (rand() % 5 != 0) M.Row(i).SetRandn();
    }
    if (rand() % 2 != 0) { // Multiply by a random square matrix;
      // keeps it low rank but will be correlated.  Harder
      // test case.
      Matrix<Real> N(dimM, dimM);
      N.SetRandn();
      Matrix<Real> tmp(dimM, dimN);
      tmp.AddMatMat(1.0, N, kNoTrans, M, kNoTrans, 0.0);
      M.CopyFromMat(tmp);
    }
    M.OrthogonalizeRows();
    Matrix<Real> I(dimM, dimM);
    I.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);
    KALDI_ASSERT(I.IsUnit(1.0e-05));
  }  
}

template<typename Real>
static void UnitTestTransposeScatter() {
  for (MatrixIndexT iter = 0;iter < 10;iter++) {

	MatrixIndexT dimA = 10 + rand()%3;
	MatrixIndexT dimO = 10 + rand()%3;
    Matrix<Real>   Af(dimA, dimA);
	SpMatrix<Real> Ap(dimA);
	Matrix<Real>   M(dimO, dimA);
    Matrix<Real>   Of(dimO, dimO);
	SpMatrix<Real> Op(dimO);
    Matrix<Real>   A_MT(dimA, dimO);

	for (MatrixIndexT i = 0;i < Ap.NumRows();i++) {
	  for (MatrixIndexT j = 0; j<=i; j++) {
        Ap(i, j) = RandGauss();
	  }
	}
	for (MatrixIndexT i = 0;i < M.NumRows();i++) {
	  for (MatrixIndexT j = 0; j < M.NumCols(); j++) {
        M(i, j) = RandGauss();
	  }
	}
    /*
      std::stringstream ss("1 2 3");
      ss >> Ap;
      ss.clear();
      ss.str("5 6 7 8 9 10");
      ss >> M;
    */

    Af.CopyFromSp(Ap);
    A_MT.AddMatMat(1.0, Af, kNoTrans, M, kTrans, 0.0);
    Of.AddMatMat(1.0, M, kNoTrans, A_MT, kNoTrans, 0.0);
    Op.AddMat2Sp(1.0, M, kNoTrans, Ap, 0.0);


    //    KALDI_LOG << "A" << '\n' << Af << '\n';
    //    KALDI_LOG << "M" << '\n' << M << '\n';
    //    KALDI_LOG << "Op" << '\n' << Op << '\n';

    for (MatrixIndexT i = 0; i < dimO; i++) {
	  for (MatrixIndexT j = 0; j<=i; j++) {
		KALDI_ASSERT(std::abs(Of(i, j) - Op(i, j)) < 0.0001);
      }
    }

    A_MT.Resize(dimO, dimA);
    A_MT.AddMatMat(1.0, Of, kNoTrans, M, kNoTrans, 0.0);
    Af.AddMatMat(1.0, M, kTrans, A_MT, kNoTrans, 1.0);
    Ap.AddMat2Sp(1.0, M, kTrans, Op, 1.0);

    //    KALDI_LOG << "Ap" << '\n' << Ap << '\n';
    //    KALDI_LOG << "Af" << '\n' << Af << '\n';

    for (MatrixIndexT i = 0; i < dimA; i++) {
	  for (MatrixIndexT j = 0; j<=i; j++) {
		KALDI_ASSERT(std::abs(Af(i, j) - Ap(i, j)) < 0.01);
      }
    }
  }
}


template<typename Real>
static void UnitTestRankNUpdate() {
  for (MatrixIndexT iter = 0;iter < 10;iter++) {
	MatrixIndexT dimA = 10 + rand()%3;
	MatrixIndexT dimO = 10 + rand()%3;
    Matrix<Real>   Af(dimA, dimA);
	SpMatrix<Real> Ap(dimA);
	SpMatrix<Real> Ap2(dimA);
	Matrix<Real>   M(dimO, dimA);
    InitRand(&M);
	Matrix<Real>   N(M, kTrans);
    Af.AddMatMat(1.0, M, kTrans, M, kNoTrans, 0.0);
    Ap.AddMat2(1.0, M, kTrans, 0.0);
    Ap2.AddMat2(1.0, N, kNoTrans, 0.0);
    Matrix<Real> Ap_f(Ap);
    Matrix<Real> Ap2_f(Ap2);
    AssertEqual(Ap_f, Af);
    AssertEqual(Ap2_f, Af);
  }
}

template<typename Real> static void  UnitTestSpInvert() {
  for (MatrixIndexT i = 0;i < 30;i++) {
	MatrixIndexT dimM = 6 + rand()%20;
	SpMatrix<Real> M(dimM);
	for (MatrixIndexT i = 0;i < M.NumRows();i++)
	  for (MatrixIndexT j = 0;j<=i;j++) M(i, j) = RandGauss();
	SpMatrix<Real> N(dimM);
	N.CopyFromSp(M);
    if (rand() % 2 == 0)
      N.Invert();
    else
      N.InvertDouble();
	Matrix<Real> Mm(dimM, dimM), Nm(dimM, dimM), Om(dimM, dimM);
	Mm.CopyFromSp(M); Nm.CopyFromSp(N);
	Om.AddMatMat(1.0, Mm, kNoTrans, Nm, kNoTrans, 0.0);
    KALDI_ASSERT(Om.IsUnit( 0.01*dimM ));
  }
}


template<typename Real> static void  UnitTestTpInvert() {
  for (MatrixIndexT i = 0;i < 30;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
	TpMatrix<Real> M(dimM);
	for (MatrixIndexT i = 0;i < M.NumRows();i++) {
	  for (MatrixIndexT j = 0;j < i;j++) M(i, j) = RandGauss();
      M(i, i) = 20 * std::max((Real)0.1, (Real) RandGauss());  // make sure invertible by making it diagonally dominant (-ish)
    }
	TpMatrix<Real> N(dimM);
	N.CopyFromTp(M);
	N.Invert();
    TpMatrix<Real> O(dimM);

	Matrix<Real> Mm(dimM, dimM), Nm(dimM, dimM), Om(dimM, dimM);
	Mm.CopyFromTp(M); Nm.CopyFromTp(N);

	Om.AddMatMat(1.0, Mm, kNoTrans, Nm, kNoTrans, 0.0);
    KALDI_ASSERT(Om.IsUnit(0.001));
  }
}


template<typename Real> static void  UnitTestLimitCondInvert() {
  for (MatrixIndexT i = 0;i < 10;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
    MatrixIndexT dimN = dimM + 1 + rand()%10;

    SpMatrix<Real> B(dimM);
    Matrix<Real> X(dimM, dimN); InitRand(&X);
    B.AddMat2(1.0, X, kNoTrans, 0.0);  // B = X*X^T -> positive definite (almost certainly), since N > M.


    SpMatrix<Real> B2(B);
    B2.LimitCond(1.0e+10, true);  // Will invert.

    Matrix<Real> Bf(B), B2f(B2);
    Matrix<Real> I(dimM, dimM); I.AddMatMat(1.0, Bf, kNoTrans, B2f, kNoTrans, 0.0);
    KALDI_ASSERT(I.IsUnit(0.1));
  }
}


template<typename Real> static void  UnitTestFloorChol() {
  for (MatrixIndexT i = 0;i < 10;i++) {
	MatrixIndexT dimM = 20 + rand()%10;


	MatrixIndexT dimN = 20 + rand()%10;
    Matrix<Real> X(dimM, dimN); InitRand(&X);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, X, kNoTrans, 0.0);  // B = X*X^T -> positive semidefinite.

    float alpha = (rand() % 10) + 0.5;
	Matrix<Real> M(dimM, dimM);
    InitRand(&M);
    SpMatrix<Real> C(dimM);
    C.AddMat2(1.0, M, kNoTrans, 0.0);  // C:=M*M^T
    InitRand(&M);
    C.AddMat2(1.0, M, kNoTrans, 1.0);  // C+=M*M^T (after making new random M)
    if (i%2 == 0)
      C.Scale(0.001);  // so it's not too much bigger than B (or it's trivial)
    SpMatrix<Real> BFloored(B); BFloored.ApplyFloor(C, alpha);
    
    
    for (MatrixIndexT j = 0;j < 10;j++) {
      Vector<Real> v(dimM);
      InitRand(&v);
      Real ip_b = VecSpVec(v, B, v);
      Real ip_a = VecSpVec(v, BFloored, v);
      Real ip_c = alpha * VecSpVec(v, C, v);
      if (i < 3) KALDI_LOG << "alpha = " << alpha << ", ip_a = " << ip_a << " ip_b = " << ip_b << " ip_c = " << ip_c <<'\n';
      KALDI_ASSERT(ip_a>=ip_b*0.999 && ip_a>=ip_c*0.999);
    }
  }
}




template<typename Real> static void  UnitTestFloorUnit() {
  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
    MatrixIndexT dimN = 20 + rand()%10;
    float floor = (rand() % 10) - 3;

    Matrix<Real> M(dimM, dimN); InitRand(&M);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, M, kNoTrans, 0.0);  // B = M*M^T -> positive semidefinite.

    SpMatrix<Real> BFloored(B); BFloored.ApplyFloor(floor);
    

    Vector<Real> s(dimM); Matrix<Real> P(dimM, dimM); B.SymPosSemiDefEig(&s, &P);
    Vector<Real> s2(dimM); Matrix<Real> P2(dimM, dimM); BFloored.SymPosSemiDefEig(&s2, &P2);

    KALDI_ASSERT ( (s.Min() >= floor && std::abs(s2.Min()-s.Min())<0.01) || std::abs(s2.Min()-floor)<0.01);
  }
}


template<typename Real> static void  UnitTestFloorCeiling() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10;
    Vector<Real> v(dimM);
    v.SetRandn();
    Real pivot = v(5);
    Vector<Real> f(v), f2(v), c(v), c2(v);
    MatrixIndexT floored2 = f.ApplyFloor(pivot),
        ceiled2 = c.ApplyCeiling(pivot);
    MatrixIndexT floored = 0, ceiled = 0;
    for (MatrixIndexT d = 0; d < dimM; d++) {
      if (f2(d) < pivot) { f2(d) = pivot; floored++; }
      if (c2(d) > pivot) { c2(d) = pivot; ceiled++; }
    }
    AssertEqual(f, f2);
    AssertEqual(c, c2);
    KALDI_ASSERT(floored == floored2);
    KALDI_ASSERT(ceiled == ceiled2);
  }
}
    
template<typename Real> static void  UnitTestMat2Vec() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10;

    Matrix<Real> M(dimM, dimM); InitRand(&M);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, M, kNoTrans, 0.0);  // B = M*M^T -> positive definite (since M nonsingular).

    Matrix<Real> P(dimM, dimM);
    Vector<Real> s(dimM);

    B.SymPosSemiDefEig(&s, &P);
    SpMatrix<Real> B2(dimM);
    B2.CopyFromSp(B);
    B2.Scale(0.25);

    // B2 <-- 2.0*B2 + 0.5 * P * diag(v)  * P^T
    B2.AddMat2Vec(0.5, P, kNoTrans, s, 2.0);  // 2.0 * 0.25 + 0.5 = 1.
    AssertEqual(B, B2);

    SpMatrix<Real> B3(dimM);
    Matrix<Real> PT(P, kTrans);
    B3.AddMat2Vec(1.0, PT, kTrans, s, 0.0);
    AssertEqual(B, B3);
  }
}

template<typename Real> static void  UnitTestLimitCond() {
  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
    SpMatrix<Real> B(dimM);
    B(1, 1) = 10000;
    KALDI_ASSERT(B.LimitCond(1000) == (dimM-1));
    KALDI_ASSERT(std::abs(B(2, 2) - 10.0) < 0.01);
    KALDI_ASSERT(std::abs(B(3, 0)) < 0.001);
  }
}

template<typename Real> static void  UnitTestTanh() {
  for (MatrixIndexT i = 0; i < 10; i++) {
    MatrixIndexT dimM = 5 + rand() % 10, dimN = 5 + rand() % 10;
    Matrix<Real> M(dimM, dimN), P(dimM, dimN), Q(dimM, dimN), R(dimM, dimN);
    M.SetRandn();
    P.SetRandn();
    Matrix<Real> N(M);
    for(int32 r = 0; r < dimM; r++) {
      for (int32 c = 0; c < dimN; c++) {
        Real x = N(r, c);
        if (x > 0.0) {
          x = -1.0 + 2.0 / (1.0 + exp(-2.0 * x));
        } else {
          x = 1.0 - 2.0 / (1.0 + exp(2.0 * x));
        }
        N(r, c) = x;
        Real out_diff = P(r, c), in_diff = out_diff * (1.0 - x * x);
        Q(r, c) = in_diff;
      }
    }
    M.Tanh(M);
    R.DiffTanh(N, P);
    AssertEqual(M, N);
    AssertEqual(Q, R);
  }
}

template<typename Real> static void  UnitTestSigmoid() {
  for (MatrixIndexT i = 0; i < 10; i++) {
    MatrixIndexT dimM = 5 + rand() % 10, dimN = 5 + rand() % 10;
    Matrix<Real> M(dimM, dimN), P(dimM, dimN), Q(dimM, dimN), R(dimM, dimN);
    M.SetRandn();
    P.SetRandn();
    Matrix<Real> N(M);
    for(int32 r = 0; r < dimM; r++) {
      for (int32 c = 0; c < dimN; c++) {
        Real x = N(r, c),
            y = 1.0 / (1 + exp(-x));
        N(r, c) = y;
        Real out_diff = P(r, c), in_diff = out_diff * y * (1.0 - y);
        Q(r, c) = in_diff;
      }
    }
    M.Sigmoid(M);
    R.DiffSigmoid(N, P);
    AssertEqual(M, N);
    AssertEqual(Q, R);
  }
}

template<typename Real> static void  UnitTestSoftHinge() {
  for (MatrixIndexT i = 0; i < 10; i++) {
    MatrixIndexT dimM = 5 + rand() % 10, dimN = 5 + rand() % 10;
    Matrix<Real> M(dimM, dimN), N(dimM, dimN), O(dimM, dimN);
    M.SetRandn();
    M.Scale(20.0);

    for(int32 r = 0; r < dimM; r++) {
      for (int32 c = 0; c < dimN; c++) {
        Real x = M(r, c);
        Real &y = N(r, c);
        if (x > 10.0) y = x;
        else y = log(1.0 + exp(x));
      }
    }
    O.SoftHinge(M);
    AssertEqual(N, O);
  }
}


template<typename Real> static void  UnitTestSimple() {
  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%20;
    Matrix<Real> M(dimM, dimN);
    M.SetUnit();
    KALDI_ASSERT(M.IsUnit());
    KALDI_ASSERT(!M.IsZero());
    KALDI_ASSERT(M.IsDiagonal());

    SpMatrix<Real> S(dimM); InitRand(&S);
    Matrix<Real> N(S);
    KALDI_ASSERT(!N.IsDiagonal());  // technically could be diagonal, but almost infinitely unlikely.
    KALDI_ASSERT(N.IsSymmetric());
    KALDI_ASSERT(!N.IsUnit());
    KALDI_ASSERT(!N.IsZero());

    M.SetZero();
    KALDI_ASSERT(M.IsZero());
    Vector<Real> V(dimM*dimN); InitRand(&V);
    Vector<Real> V2(V), V3(dimM*dimN);
    V2.ApplyExp();
    AssertEqual(V.Sum(), V2.SumLog());
	V3.ApplyLogAndCopy(V2);
    V2.ApplyLog();
    AssertEqual(V, V2);
	AssertEqual(V3, V2);

    {
      Vector<Real> V2(V);
      for (MatrixIndexT i = 0; i < V2.Dim(); i++)
        V2(i) = exp(V2(i));
      V.ApplyExp();
      AssertEqual(V, V2);
    }
    {
      Matrix<Real> N2(N), N3(N);
      for (MatrixIndexT i = 0; i < N.NumRows(); i++)
        for (MatrixIndexT j = 0; j < N.NumCols(); j++)
          N2(i, j) = exp(N2(i, j));
      N3.ApplyExp();
      AssertEqual(N2, N3);
    }
    KALDI_ASSERT(!S.IsDiagonal());
    KALDI_ASSERT(!S.IsUnit());
    N.SetUnit();
    S.CopyFromMat(N);
    KALDI_ASSERT(S.IsDiagonal());
    KALDI_ASSERT(S.IsUnit());
    N.SetZero();
    S.CopyFromMat(N);
    KALDI_ASSERT(S.IsZero());
    KALDI_ASSERT(S.IsDiagonal());
  }
}



template<typename Real> static void UnitTestIo() {

  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 1;
    MatrixIndexT dimN = rand()%10 + 1;
    bool binary = (i%2 == 0);

    if (i == 0) {
      dimM = 0;dimN = 0;  // test case when both are zero.
    }
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);
    Matrix<Real> N;
    Vector<Real> v1(dimM);
    InitRand(&v1);
    Vector<Real> v2(dimM);

    SpMatrix<Real> S(dimM);
    InitRand(&S);
    KALDI_LOG << "SpMatrix IO: " << S;
    SpMatrix<Real> T(dimM);

    {
      std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
      InitKaldiOutputStream(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v1.Write(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v1.Write(outs, binary);
    }

	{
      bool binary_in;
      bool either_way = (i%2 == 0);
      std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
      InitKaldiInputStream(ins, &binary_in);
      N.Resize(0, 0);
      T.Resize(0);
      v2.Resize(0);
      N.Read(ins, binary_in, either_way);
      T.Read(ins, binary_in, either_way);
      v2.Read(ins, binary_in, either_way);
      if (i%2 == 0)
        ((MatrixBase<Real>&)N).Read(ins, binary_in, true);  // add
      else
        N.Read(ins, binary_in, true);
      T.Read(ins, binary_in, true);  // add
      if (i%2 == 0)
        ((VectorBase<Real>&)v2).Read(ins, binary_in, true);  // add
      else
        v2.Read(ins, binary_in, true);
    }
    N.Scale(0.5);
    v2.Scale(0.5);
    T.Scale(0.5);
    AssertEqual(M, N);
    AssertEqual(v1, v2);
    AssertEqual(S, T);
  }
}


template<typename Real> static void UnitTestIoCross() {  // across types.

  typedef typename OtherReal<Real>::Real Other;  // e.g. if Real == float, Other == double.
  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 1;
    MatrixIndexT dimN = rand()%10 + 1;
    bool binary = (i%2 == 0);
    if (i == 0) {
      dimM = 0;dimN = 0;  // test case when both are zero.
    }
    Matrix<Real> M(dimM, dimN);
    Matrix<Other> MO;
    InitRand(&M);
    Matrix<Real> N(dimM, dimN);
    Vector<Real> v(dimM);
    Vector<Other> vO;
    InitRand(&v);
    Vector<Real> w(dimM);

    SpMatrix<Real> S(dimM);
    SpMatrix<Other> SO;
    InitRand(&S);
    SpMatrix<Real> T(dimM);

    {
      std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
      InitKaldiOutputStream(outs, binary);

      M.Write(outs, binary);
      S.Write(outs, binary);
      v.Write(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v.Write(outs, binary);
    }
	{
      std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
      bool binary_in;
      InitKaldiInputStream(ins, &binary_in);

      MO.Read(ins, binary_in);
      SO.Read(ins, binary_in);
      vO.Read(ins, binary_in);
      MO.Read(ins, binary_in, true);
      SO.Read(ins, binary_in, true);
      vO.Read(ins, binary_in, true);
      N.CopyFromMat(MO);
      T.CopyFromSp(SO);
      w.CopyFromVec(vO);
    }
    N.Scale(0.5);
    w.Scale(0.5);
    T.Scale(0.5);
    AssertEqual(M, N);
    AssertEqual(v, w);
    AssertEqual(S, T);
  }
}


template<typename Real> static void UnitTestHtkIo() {

  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 10;
    MatrixIndexT dimN = rand()%10 + 10;

    HtkHeader hdr;
    hdr.mNSamples = dimM;
    hdr.mSamplePeriod = 10000;  // in funny HTK units-- can set it arbitrarily
    hdr.mSampleSize = sizeof(float)*dimN;
    hdr.mSampleKind = 8;  // Mel spectrum.
    
    Matrix<Real> M(dimM, dimN);
    InitRand(&M);

    {
      std::ofstream os("tmpf", std::ios::out|std::ios::binary);
      WriteHtk(os, M, hdr);
    }

    Matrix<Real> N;
    HtkHeader hdr2;
    {
      std::ifstream is("tmpf", std::ios::in|std::ios::binary);
      ReadHtk(is, &N, &hdr2);
    }

    AssertEqual(M, N);
    KALDI_ASSERT(hdr.mNSamples == hdr2.mNSamples);
    KALDI_ASSERT(hdr.mSamplePeriod == hdr2.mSamplePeriod);
    KALDI_ASSERT(hdr.mSampleSize == hdr2.mSampleSize);
    KALDI_ASSERT(hdr.mSampleKind == hdr2.mSampleKind);
  }

}



template<typename Real> static void UnitTestRange() {  // Testing SubMatrix class.

  // this is for matrix-range
  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = (rand()%10) + 10;

    Matrix<Real> M(dimM, dimN);
    InitRand(&M);
    MatrixIndexT dimMStart = rand() % 5;
    MatrixIndexT dimNStart = rand() % 5;

    MatrixIndexT dimMEnd = dimMStart + 1 + (rand()%10); if (dimMEnd > dimM) dimMEnd = dimM;
    MatrixIndexT dimNEnd = dimNStart + 1 + (rand()%10); if (dimNEnd > dimN) dimNEnd = dimN;


    SubMatrix<Real> sub(M, dimMStart, dimMEnd-dimMStart, dimNStart, dimNEnd-dimNStart);

    KALDI_ASSERT(sub.Sum() == M.Range(dimMStart, dimMEnd-dimMStart, dimNStart, dimNEnd-dimNStart).Sum());

    for (MatrixIndexT i = dimMStart;i < dimMEnd;i++)
      for (MatrixIndexT j = dimNStart;j < dimNEnd;j++)
        KALDI_ASSERT(M(i, j) == sub(i-dimMStart, j-dimNStart));

    InitRand(&sub);

    KALDI_ASSERT(sub.Sum() == M.Range(dimMStart, dimMEnd-dimMStart, dimNStart, dimNEnd-dimNStart).Sum());

    for (MatrixIndexT i = dimMStart;i < dimMEnd;i++)
      for (MatrixIndexT j = dimNStart;j < dimNEnd;j++)
        KALDI_ASSERT(M(i, j) == sub(i-dimMStart, j-dimNStart));
  }

  // this if for vector-range
  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT length = (rand()%10) + 10;

    Vector<Real> V(length);
    InitRand(&V);
    MatrixIndexT lenStart = rand() % 5;

    MatrixIndexT lenEnd = lenStart + 1 + (rand()%10); if (lenEnd > length) lenEnd = length;

    SubVector<Real> sub(V, lenStart, lenEnd-lenStart);

    KALDI_ASSERT(sub.Sum() == V.Range(lenStart, lenEnd-lenStart).Sum());

    for (MatrixIndexT i = lenStart;i < lenEnd;i++)
      KALDI_ASSERT(V(i) == sub(i-lenStart));

    InitRand(&sub);

    KALDI_ASSERT(sub.Sum() == V.Range(lenStart, lenEnd-lenStart).Sum());

    for (MatrixIndexT i = lenStart;i < lenEnd;i++)
      KALDI_ASSERT(V(i) == sub(i-lenStart));
  }
}

template<typename Real> static void UnitTestScale() {

  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = (rand()%10) + 10;

    Matrix<Real> M(dimM, dimN);

    Matrix<Real> N(M);
    float f = (float)((rand()%10)-5);
    M.Scale(f);
    KALDI_ASSERT(M.Sum() == f * N.Sum());

    {
      // now test scale_rows
      M.CopyFromMat(N);  // make same.
      Vector<Real> V(dimM);
      InitRand(&V);
      M.MulRowsVec(V);
      for (MatrixIndexT i = 0; i < dimM;i++)
        for (MatrixIndexT j = 0;j < dimN;j++)
          KALDI_ASSERT(M(i, j) - N(i, j)*V(i) < 0.0001);
    }

    {
      // now test scale_cols
      M.CopyFromMat(N);  // make same.
      Vector<Real> V(dimN);
      InitRand(&V);
      M.MulColsVec(V);
      for (MatrixIndexT i = 0; i < dimM;i++)
        for (MatrixIndexT j = 0;j < dimN;j++)
          KALDI_ASSERT(M(i, j) - N(i, j)*V(j) < 0.0001);
    }

  }
}

template<typename Real> static void UnitTestMul() {
  for (MatrixIndexT x = 0; x<=1; x++) {
    MatrixTransposeType trans = (x == 1 ? kTrans: kNoTrans);
    for (MatrixIndexT i = 0;i < 5;i++) {
      float alpha = 1.0, beta =0;
      if (i%3 == 0) beta = 0.5;
      if (i%5 == 0) alpha = 0.7;
      MatrixIndexT dimM = (rand()%10) + 10;
      Vector<Real> v(dimM); InitRand(&v);
      TpMatrix<Real> T(dimM); InitRand(&T);
      Matrix<Real> M(dimM, dimM);
      if (i%2 == 1)
        M.CopyFromTp(T, trans);
      else
        M.CopyFromTp(T, kNoTrans);
      Vector<Real> v2(v);
      Vector<Real> v3(v);
      v2.AddTpVec(alpha, T, trans, v, beta);
      if (i%2 == 1)
        v3.AddMatVec(alpha, M, kNoTrans, v, beta);
      else
        v3.AddMatVec(alpha, M, trans, v, beta);

      v.AddTpVec(alpha, T, trans, v, beta);
      AssertEqual(v2, v3);
      AssertEqual(v, v2);
    }

    for (MatrixIndexT i = 0;i < 5;i++) {
      float alpha = 1.0, beta =0;
      if (i%3 == 0) beta = 0.5;
      if (i%5 == 0) alpha = 0.7;

      MatrixIndexT dimM = (rand()%10) + 10;
      Vector<Real> v(dimM); InitRand(&v);
      SpMatrix<Real> T(dimM); InitRand(&T);
      Matrix<Real> M(T);
      Vector<Real> v2(dimM);
      Vector<Real> v3(dimM);
      v2.AddSpVec(alpha, T, v, beta);
      v3.AddMatVec(alpha, M, i%2 ? kNoTrans : kTrans, v, beta);
      AssertEqual(v2, v3);
    }
  }
}


template<typename Real> static void UnitTestInnerProd() {

  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N);
  InitRand(&S);
  Vector<Real> v(N);
  InitRand(&v);
  Real prod = VecSpVec(v, S, v);
  Real f2=0.0;
  for (MatrixIndexT i = 0; i < N; i++)
    for (MatrixIndexT j = 0; j < N; j++) {
      f2 += v(i) * v(j) * S(i, j);
    }
  AssertEqual(prod, f2);
}


template<typename Real> static void UnitTestAddToDiag() {
  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N);
  InitRand(&S);
  SpMatrix<Real> S2(S);
  Real x = 0.5;
  S.AddToDiag(x);
  for (MatrixIndexT i = 0; i  < N; i++) S2(i, i) += x;
  AssertEqual(S, S2);
}

template<typename Real> static void UnitTestScaleDiag() {

  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N);
  InitRand(&S);
  SpMatrix<Real> S2(S);
  S.ScaleDiag(0.5);
  for (MatrixIndexT i = 0; i  < N; i++) S2(i, i) *= 0.5;
  AssertEqual(S, S2);
}


template<typename Real> static void UnitTestSetDiag() {
  
  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N), T(N);
  S.SetUnit();
  S.ScaleDiag(0.5);
  T.SetDiag(0.5);
  AssertEqual(S, T);

}

template<typename Real> static void UnitTestTraceSpSpLower() {

  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N), T(N);
  InitRand(&S);
  InitRand(&T);

  SpMatrix<Real> Sfast(S);
  Sfast.Scale(2.0);
  Sfast.ScaleDiag(0.5);

  AssertEqual(TraceSpSp(S, T), TraceSpSpLower(Sfast, T));
}

// also tests AddSmatMat
template<typename Real> static void UnitTestAddMatSmat() {
  for (MatrixIndexT i = 0; i < 6; i++) {
    MatrixIndexT dimM = (rand()%10) + 1,
        dimN = (rand()%10 + 1),
        dimO = (rand()%10 + 1);
    MatrixTransposeType transB = (i % 2 == 0 ? kTrans : kNoTrans),
        transC = (i % 3 == 0 ? kTrans : kNoTrans);
    Matrix<Real> A(dimM, dimN),
        B(dimM, dimO), C(dimO, dimN);
    A.SetRandn(); B.SetRandn(); C.SetRandn();
    if (transB == kTrans) B.Transpose();
    if (transC == kTrans) C.Transpose();
    Matrix<Real> A2(A), A3(A);
    BaseFloat beta = 0.333, alpha = 0.5;
    A.AddMatMat(alpha, B, transB, C, transC, beta);
    A2.AddMatSmat(alpha, B, transB, C, transC, beta);
    A3.AddSmatMat(alpha, B, transB, C, transC, beta);
    AssertEqual(A, A2);
    AssertEqual(A, A3);
  }
}

// Also tests AddSmat2Sp
template<typename Real> static void UnitTestAddMat2Sp() {
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = (rand()%10) + 1,
        dimN = (rand()%10 + 1);
    BaseFloat alpha = 0.8, beta = 0.9;
    SpMatrix<Real> S(dimM), T(dimN);
    S.SetRandn();
    T.SetRandn();
    Matrix<Real> M(dimM, dimN);
    M.SetRandn();
    MatrixTransposeType trans = (i % 2 == 1 ? kTrans: kNoTrans);
    if (trans == kTrans) M.Transpose();
    SpMatrix<Real> S2(S), S3(S);
    S.AddMat2Sp(alpha, M, trans, T, beta);
    S3.AddSmat2Sp(alpha, M, trans, T, beta);
    
    // M[trans?] * T.
    Matrix<Real> A(dimM, dimN);
    A.AddMatSp(1.0, M, trans, T, 0.0);
    Matrix<Real> B(dimM, dimM); // M[trans] * T * M[trans]'
    B.AddMatMat(1.0, A, kNoTrans, M, trans == kTrans ? kNoTrans : kTrans, 0.0);
    SpMatrix<Real> tmp(B);
    S2.Scale(beta);
    S2.AddSp(alpha, tmp);
    AssertEqual(S, S2);
    AssertEqual(S, S3);
  }
}

template<typename Real> static void UnitTestAddMatSelf() {
  MatrixIndexT dimM = (rand() % 10) + 1;
  Matrix<Real> M(dimM, dimM), N(dimM, dimM);
  M.SetRandn();
  N.AddMat(1.5, M);
  M.AddMat(0.5, M);
  AssertEqual(M, N);
  N.AddMat(0.5, M, kTrans);
  M.AddMat(0.5, M, kTrans);
  AssertEqual(M, N);
}

template<typename Real> static void UnitTestAddMat2() {
  MatrixIndexT extra = 1;
  // Test AddMat2 function of SpMatrix.
  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = (rand()%10) + extra,
        dimN = (rand() % 10) + extra;
    Real alpha = 0.2 * (rand() % 6),
        beta = 0.2 * (rand() % 6);
    SpMatrix<Real> S(dimM);
    S.SetRandn();
    MatrixTransposeType trans = (i % 2 == 1 ? kTrans: kNoTrans),
        other_trans = (trans == kTrans ? kNoTrans : kTrans);
    Matrix<Real> M;
    if (trans == kNoTrans) M.Resize(dimM, dimN);
    else M.Resize(dimN, dimM);
    M.SetRandn();

    Matrix<Real> Sfull(S), Sfull2(S);
    
    S.AddMat2(alpha, M, trans, beta);
    
    Sfull.AddMatMat(alpha, M, trans, M, other_trans, beta);

    Sfull2.SymAddMat2(alpha, M, trans, beta);

    // now symmetrize.
    SpMatrix<Real> Sfull2_copy(Sfull2, kTakeLower);
    Sfull2.CopyFromSp(Sfull2_copy);

    Matrix<Real> Sfull3(S);
    AssertEqual(Sfull, Sfull3);
    AssertEqual(Sfull2, Sfull3);
  }
}





template<typename Real> static void UnitTestSolve() {

  for (MatrixIndexT i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = dimM - (rand()%3);  // slightly lower-dim.

    SpMatrix<Real> H(dimM);
    Matrix<Real> M(dimM, dimN); InitRand(&M);
    H.AddMat2(1.0, M, kNoTrans, 0.0);  // H = M M^T

    Vector<Real> x(dimM);
    InitRand(&x);
    Vector<Real> tmp(dimM);
    tmp.SetRandn();
    Vector<Real> g(dimM);
    g.AddSpVec(1.0, H, tmp, 0.0); // Limit to subspace that H is in.
    Vector<Real> x2(x), x3(x);
    SolverOptions opts2, opts3;
    opts2.diagonal_precondition = rand() % 2;
    opts2.optimize_delta = rand() % 2;
    opts3.diagonal_precondition = rand() % 2;
    opts3.optimize_delta = rand() % 2;
    
    double ans2 =  SolveQuadraticProblem(H, g, opts2, &x2),
        ans3 = SolveQuadraticProblem(H, g, opts3, &x3);

    double observed_impr2 = (VecVec(x2, g) -0.5* VecSpVec(x2, H, x2)) -
        (VecVec(x, g) -0.5* VecSpVec(x, H, x)),
        observed_impr3 =  (VecVec(x3, g) -0.5* VecSpVec(x3, H, x3)) -
        (VecVec(x, g) -0.5* VecSpVec(x, H, x));
    AssertEqual(observed_impr2, ans2);
    AssertEqual(observed_impr3, ans3);
    KALDI_ASSERT(ans2 >= 0);
    KALDI_ASSERT(ans3 >= 0);
    KALDI_ASSERT(abs(ans2 - ans3) / std::max(ans2, ans3) < 0.01);
    //AssertEqual(x2, x3);
    //AssertEqual(ans1, ans2);
  }


  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = (rand() % 10) + 10;
    MatrixIndexT dimN = dimM - (rand() % 3); // slightly lower-dim.
    MatrixIndexT dimO = (rand() % 10) + 10;

    SpMatrix<Real> Q(dimM), SigmaInv(dimO);
    Matrix<Real> Mtmp(dimM, dimN);
    InitRand(&Mtmp);
    Q.AddMat2(1.0, Mtmp, kNoTrans, 0.0); // H = M M^T

    Matrix<Real> Ntmp(dimO, dimN);
    InitRand(&Ntmp);
    SigmaInv.AddMat2(1.0, Ntmp, kNoTrans, 0.0); // H = M M^T

    Matrix<Real> M(dimO, dimM), Y(dimO, dimM);
    InitRand(&M);
    InitRand(&Y);

    Matrix<Real> M2(M);

    SpMatrix<Real> Qinv(Q);
    if (Q.Cond() < 1000.0) Qinv.Invert();

    SolverOptions opts;
    opts.optimize_delta = rand() % 2;
    opts.diagonal_precondition = rand() % 2;
    double ans = SolveQuadraticMatrixProblem(Q, Y, SigmaInv, opts, &M2);
    
    Matrix<Real> M3(M);
    M3.AddMatSp(1.0, Y, kNoTrans, Qinv, 0.0);
    if (Q.Cond() < 1000.0) {
      AssertEqual(M2, M3);  // This equality only holds if SigmaInv full-rank,
      // which is overwhelmingly likely if dimO > dimM
    }

    {
      Real a1 = TraceMatSpMat(M2, kTrans, SigmaInv, Y, kNoTrans),
          a2 = TraceMatSpMatSp(M2, kNoTrans, Q, M2, kTrans, SigmaInv),
          b1 = TraceMatSpMat(M, kTrans, SigmaInv, Y, kNoTrans),
          b2 = TraceMatSpMatSp(M, kNoTrans, Q, M, kTrans, SigmaInv),
          a3 = a1 - 0.5 * a2,
          b3 = b1 - 0.5 * b2;
      KALDI_ASSERT(a3 >= b3);
      AssertEqual(a3 - b3, ans);
      // KALDI_LOG << "a3 = " << a3 << ", b3 = " << b3 << ", c3 = " << c3;
    }  // Check objf not decreased.
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = (rand() % 10) + 10;
    MatrixIndexT dimO = (rand() % 10) + 10;

    SpMatrix<Real> Q1(dimM), Q2(dimM), P1(dimO), P2(dimO);
    RandPosdefSpMatrix(dimM, &Q1);
    RandPosdefSpMatrix(dimM, &Q2);
    RandPosdefSpMatrix(dimO, &P1);
    RandPosdefSpMatrix(dimO, &P1);

    Matrix<Real> M(dimO, dimM), G(dimO, dimM);
    M.SetRandn();
    G.SetRandn();
    //    InitRand(&M);
    //    InitRand(&G);

    Matrix<Real> M2(M);

    SolverOptions opts;
    opts.optimize_delta = rand() % 2;    
    SolveDoubleQuadraticMatrixProblem(G, P1, P2, Q1, Q2, opts, &M2);

    {
      Real a1 = TraceMatMat(M2, G, kTrans),
          a2 = TraceMatSpMatSp(M2, kNoTrans, Q1, M2, kTrans, P1),
          a3 = TraceMatSpMatSp(M2, kNoTrans, Q2, M2, kTrans, P2),
          b1 = TraceMatMat(M, G, kTrans),
          b2 = TraceMatSpMatSp(M, kNoTrans, Q1, M, kTrans, P1),
          b3 = TraceMatSpMatSp(M, kNoTrans, Q2, M, kTrans, P2),
          a4 = a1 - 0.5 * a2 - 0.5 * a3,
          b4 = b1 - 0.5 * b2 - 0.5 * b3;
      KALDI_LOG << "a4 = " << a4 << ", b4 = " << b4;
      KALDI_ASSERT(a4 >= b4);
    }  // Check objf not decreased.
  }
}

template<typename Real> static void UnitTestMax2() {
  for (MatrixIndexT i = 0; i < 2; i++) {
    MatrixIndexT M = 1 + rand() % 10, N = 1 + rand() % 10;
    Matrix<Real> A(M, N), B(M, N), C(M, N), D(M, N);
    A.SetRandn();
    B.SetRandn();
    for (MatrixIndexT r = 0; r < M; r++)
      for (MatrixIndexT c = 0; c < N; c++)
        C(r, c) = std::max(A(r, c), B(r, c));
    D.CopyFromMat(A);
    D.Max(B);
    AssertEqual(C, D);
  }
}

template<typename Real> static void UnitTestMaxAbsEig() {
  for (MatrixIndexT i = 0; i < 1; i++) {
    SpMatrix<Real> M(10);
    M.SetRandn();
    Matrix<Real> P(10, 10);
    Vector<Real> s(10);
    M.Eig(&s, (i == 0 ? static_cast<Matrix<Real>*>(NULL) : &P));
    Real max_eig = std::max(-s.Min(), s.Max());
    AssertEqual(max_eig, M.MaxAbsEig());
  }
}

template<typename Real> static void UnitTestLbfgs() {
  MatrixIndexT temp = g_kaldi_verbose_level;
  g_kaldi_verbose_level = 4;
  for (MatrixIndexT iter = 0; iter < 3; iter++) {
    bool minimize = (iter % 2 == 0);
    MatrixIndexT dim = 1 + rand() % 30;
    SpMatrix<Real> S(dim);
    RandPosdefSpMatrix(dim, &S);
    Vector<Real> v(dim);
    InitRand(&v);
    // Function will be f = exp(0.1 * [ x' v  -0.5 x' S x ])
    // This is to maximize; we negate it when minimizing.
    
    //Vector<Real> hessian(dim);
    //hessian.CopyDiagFromSp(S);
    
    SpMatrix<Real> Sinv(S);
    Sinv.Invert();
    Vector<Real> x_opt(dim);
    x_opt.AddSpVec(1.0, Sinv, v, 0.0); // S^{-1} v-- the optimum.

    Vector<Real> init_x(dim);
    InitRand(&init_x);

    LbfgsOptions opts;
    opts.minimize = minimize; // This objf has a maximum, not a minimum.
    OptimizeLbfgs<Real> opt_lbfgs(init_x, opts);
    MatrixIndexT num_iters = 0;
    Real c = 0.01;
    Real sign = (minimize ? -1.0 : 1.0); // function has a maximum not minimum..
    while (opt_lbfgs.RecentStepLength() > 1.0e-04) {
      KALDI_VLOG(2) << "Last step length is " << opt_lbfgs.RecentStepLength();
      const VectorBase<Real> &x = opt_lbfgs.GetProposedValue();
      Real logf = VecVec(x, v) - 0.5 * VecSpVec(x, S, x);
      Vector<Real> dlogf_dx(v); //  derivative of log(f) w.r.t. x.
      dlogf_dx.AddSpVec(-1.0, S, x, 1.0);
      KALDI_VLOG(2) << "Gradient magnitude is " << dlogf_dx.Norm(2.0);
      Real f = exp(c * logf);
      Vector<Real> df_dx(dlogf_dx);
      df_dx.Scale(f * c); // comes from derivative of the exponential function.
      f *= sign;
      df_dx.Scale(sign);
      opt_lbfgs.DoStep(f, df_dx);
      num_iters++;
    }
    Vector<Real> x (opt_lbfgs.GetValue());
    Vector<Real> diff(x);
    diff.AddVec(-1.0, x_opt);
    KALDI_VLOG(2) << "L-BFGS finished after " << num_iters << " function evaluations.";
    /*
    if (sizeof(Real) == 8) {
      KALDI_ASSERT(diff.Norm(2.0) < 0.5);
    } else {
      KALDI_ASSERT(diff.Norm(2.0) < 2.0);
      } */
  }
  g_kaldi_verbose_level = temp;
}


template<typename Real> static void UnitTestMaxMin() {

  MatrixIndexT M = 1 + rand() % 10, N = 1 + rand() % 10;
  {
    Vector<Real> v(N);
    InitRand(&v);
    Real min = 1.0e+10, max = -1.0e+10;
    for (MatrixIndexT i = 0; i< N; i++) {
      min = std::min(min, v(i));
      max = std::max(max, v(i));
    }
    AssertEqual(min, v.Min());
    AssertEqual(max, v.Max());
  }
  {
    SpMatrix<Real> S(N);
    InitRand(&S);
    Real min = 1.0e+10, max = -1.0e+10;
    for (MatrixIndexT i = 0; i< N; i++) {
      for (MatrixIndexT j = 0; j <= i; j++) {
        min = std::min(min, S(i, j));
        max = std::max(max, S(i, j));
      }
    }
    AssertEqual(min, S.Min());
    AssertEqual(max, S.Max());
  }
  {
    Matrix<Real> mat(M, N);
    InitRand(&mat);
    Real min = 1.0e+10, max = -1.0e+10;
    for (MatrixIndexT i = 0; i< M; i++) {
      for (MatrixIndexT j = 0; j < N; j++) {
        min = std::min(min, mat(i, j));
        max = std::max(max, mat(i, j));
      }
    }
    AssertEqual(min, mat.Min());
    AssertEqual(max, mat.Max());
  }
}

template<typename Real>
static bool approx_equal(Real a, Real b) {
  return  ( std::abs(a-b) <= 1.0e-03 * (std::abs(a)+std::abs(b)));
}

template<typename Real> static void UnitTestTrace() {

  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10, dimO = 20 + rand()%10, dimP = dimM;
    Matrix<Real> A(dimM, dimN), B(dimN, dimO), C(dimO, dimP);
    InitRand(&A);     InitRand(&B);     InitRand(&C);
    Matrix<Real> AT(dimN, dimM), BT(dimO, dimN), CT(dimP, dimO);
    AT.CopyFromMat(A, kTrans); BT.CopyFromMat(B, kTrans); CT.CopyFromMat(C, kTrans);

    Matrix<Real> AB(dimM, dimO);
    AB.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
    Matrix<Real> BC(dimN, dimP);
    BC.AddMatMat(1.0, B, kNoTrans, C, kNoTrans, 0.0);
    Matrix<Real> ABC(dimM, dimP);
    ABC.AddMatMat(1.0, A, kNoTrans, BC, kNoTrans, 0.0);

    Real
        t1 = TraceMat(ABC),
        t2 = ABC.Trace(),
        t3 = TraceMatMat(A, BC),
        t4 = TraceMatMat(AT, BC, kTrans),
        t5 = TraceMatMat(BC, AT, kTrans),
        t6 = TraceMatMatMat(A, kNoTrans, B, kNoTrans, C, kNoTrans),
        t7 = TraceMatMatMat(AT, kTrans, B, kNoTrans, C, kNoTrans),
        t8 = TraceMatMatMat(AT, kTrans, BT, kTrans, C, kNoTrans),
        t9 = TraceMatMatMat(AT, kTrans, BT, kTrans, CT, kTrans);

    Matrix<Real> ABC1(dimM, dimP);  // tests AddMatMatMat.
    ABC1.AddMatMatMat(1.0, A, kNoTrans, B, kNoTrans, C, kNoTrans, 0.0);
    AssertEqual(ABC, ABC1);

    Matrix<Real> ABC2(dimM, dimP);  // tests AddMatMatMat.
    ABC2.AddMatMatMat(0.25, A, kNoTrans, B, kNoTrans, C, kNoTrans, 0.0);
    ABC2.AddMatMatMat(0.25, AT, kTrans, B, kNoTrans, C, kNoTrans, 2.0);  // the extra 1.0 means another 0.25.
    ABC2.AddMatMatMat(0.125, A, kNoTrans, BT, kTrans, C, kNoTrans, 1.0);
    ABC2.AddMatMatMat(0.125, A, kNoTrans, B, kNoTrans, CT, kTrans, 1.0);
    AssertEqual(ABC, ABC2);

    Real tol = 0.001;
    KALDI_ASSERT((std::abs(t1-t2) < tol) && (std::abs(t2-t3) < tol) && (std::abs(t3-t4) < tol)
                 && (std::abs(t4-t5) < tol) && (std::abs(t5-t6) < tol) && (std::abs(t6-t7) < tol)
                 && (std::abs(t7-t8) < tol) && (std::abs(t8-t9) < tol));
  }

  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10;
    SpMatrix<Real> S(dimM), T(dimN);
    InitRand(&S); InitRand(&T);
    Matrix<Real> M(dimM, dimN), O(dimM, dimN);
    InitRand(&M); InitRand(&O);
    Matrix<Real> sM(S), tM(T);

    Real x1 = TraceMatMat(tM, tM);
    Real x2 = TraceSpMat(T, tM);
    KALDI_ASSERT(approx_equal(x1, x2) || fabs(x1-x2) < 0.1);

    Real t1 = TraceMatMatMat(M, kNoTrans, tM, kNoTrans, M, kTrans);
    Real t2 = TraceMatSpMat(M, kNoTrans, T, M, kTrans);
    KALDI_ASSERT(approx_equal(t1, t2) || fabs(t1-12) < 0.1);

    Real u1 = TraceMatSpMatSp(M, kNoTrans, T, O, kTrans, S);
    Real u2 = TraceMatMatMatMat(M, kNoTrans, tM, kNoTrans, O, kTrans, sM, kNoTrans);
    KALDI_ASSERT(approx_equal(u1, u2) || fabs(u1-u2) < 0.1);
  }

}


template<typename Real> static void UnitTestComplexFt() {

  // Make sure it inverts properly.
  for (MatrixIndexT d = 0; d < 10; d++) {
    MatrixIndexT N = rand() % 100, twoN = 2*N;
    Vector<Real> v(twoN), w(twoN), x(twoN);
    InitRand(&v);
    ComplexFt(v, &w, true);
    ComplexFt(w, &x, false);
    if (N>0) x.Scale(1.0/static_cast<Real>(N));
    AssertEqual(v, x);
  }
}

template<typename Real> static void UnitTestDct() {

  // Check that DCT matrix is orthogonal (i.e. M^T M = I);
  for (MatrixIndexT i = 0; i < 10; i++) {
    MatrixIndexT N = 1 + rand() % 10;
    Matrix<Real> M(N, N);
    ComputeDctMatrix(&M);
    Matrix<Real> I(N, N);
    I.AddMatMat(1.0, M, kTrans, M, kNoTrans, 0.0);
    KALDI_ASSERT(I.IsUnit());
  }

}
template<typename Real> static void UnitTestComplexFft() {

  // Make sure it inverts properly.
  for (MatrixIndexT N_ = 0; N_ < 100; N_+=3) {
    MatrixIndexT N = N_;
    if (N>=95) {
      N = ( rand() % 150);
      N = N*N;  // big number.
    }

    MatrixIndexT twoN = 2*N;
    Vector<Real> v(twoN), w_base(twoN), w_alg(twoN), x_base(twoN), x_alg(twoN);

    InitRand(&v);

    if (N< 100) ComplexFt(v, &w_base, true);
    w_alg.CopyFromVec(v);
    ComplexFft(&w_alg, true);
    if (N< 100) AssertEqual(w_base, w_alg, 0.01*N);

    if (N< 100) ComplexFt(w_base, &x_base, false);
    x_alg.CopyFromVec(w_alg);
    ComplexFft(&x_alg, false);

    if (N< 100) AssertEqual(x_base, x_alg, 0.01*N);
    x_alg.Scale(1.0/N);
    AssertEqual(v, x_alg, 0.001*N);
  }
}


template<typename Real> static void UnitTestSplitRadixComplexFft() {

  // Make sure it inverts properly.
  for (MatrixIndexT N_ = 0; N_ < 30; N_+=3) {
    MatrixIndexT logn = 1 + rand() % 10;
    MatrixIndexT N = 1 << logn;

    MatrixIndexT twoN = 2*N;
    SplitRadixComplexFft<Real> srfft(N);
    for (MatrixIndexT p = 0; p < 3; p++) {
      Vector<Real> v(twoN), w_base(twoN), w_alg(twoN), x_base(twoN), x_alg(twoN);

      InitRand(&v);

      if (N< 100) ComplexFt(v, &w_base, true);
      w_alg.CopyFromVec(v);
      srfft.Compute(w_alg.Data(), true);

      if (N< 100) AssertEqual(w_base, w_alg, 0.01*N);

      if (N< 100) ComplexFt(w_base, &x_base, false);
      x_alg.CopyFromVec(w_alg);
      srfft.Compute(x_alg.Data(), false);

      if (N< 100) AssertEqual(x_base, x_alg, 0.01*N);
      x_alg.Scale(1.0/N);
      AssertEqual(v, x_alg, 0.001*N);
    }
  }
}



template<typename Real> static void UnitTestTranspose() {

  Matrix<Real> M(rand() % 5 + 1, rand() % 10 + 1);
  InitRand(&M);
  Matrix<Real> N(M, kTrans);
  N.Transpose();
  AssertEqual(M, N);
}

template<typename Real> static void UnitTestAddVecToRows() {
  Matrix<Real> M(rand() % 5 + 1, rand() % 10 + 1);
  InitRand(&M);
  Vector<float> v(M.NumCols());
  InitRand(&v);
  Matrix<Real> N(M);
  Vector<float> ones(M.NumRows());
  ones.Set(1.0);
  M.AddVecToRows(0.5, v);
  N.AddVecVec(0.5, ones, v);
  AssertEqual(M, N);
}

template<typename Real> static void UnitTestAddVec2Sp() {
  for (int32 i = 0; i < 10; i++) {
    int32 dim = rand() % 5;
    SpMatrix<Real> S(dim);
    S.SetRandn();
    Vector<Real> v(dim);
    v.SetRandn();
    Matrix<Real> M(dim, dim);
    M.CopyDiagFromVec(v);
    
    SpMatrix<Real> T1(dim);
    T1.SetRandn();
    SpMatrix<Real> T2(T1);
    Real alpha = 0.33, beta = 4.5;
    T1.AddVec2Sp(alpha, v, S, beta);
    T2.AddMat2Sp(alpha, M, kNoTrans, S, beta);
    AssertEqual(T1, T2);
  }
}    


template<typename Real> static void UnitTestAddVecToCols() {
  Matrix<Real> M(rand() % 5 + 1, rand() % 10 + 1);
  InitRand(&M);
  Vector<float> v(M.NumRows());
  InitRand(&v);
  Matrix<Real> N(M);
  Vector<float> ones(M.NumCols());
  ones.Set(1.0);
  M.AddVecToCols(0.5, v);
  N.AddVecVec(0.5, v, ones);
  AssertEqual(M, N);
}

template<typename Real> static void UnitTestComplexFft2() {

  // Make sure it inverts properly.
  for (MatrixIndexT pos = 0; pos < 10; pos++) {
    for (MatrixIndexT N_ = 2; N_ < 15; N_+=2) {
      if ( pos < N_) {
        MatrixIndexT N = N_;
        Vector<Real> v(N), vorig(N), v2(N);
        v(pos)  = 1.0;
        vorig.CopyFromVec(v);
        // KALDI_LOG << "Original v:\n" << v;
        ComplexFft(&v, true);
        // KALDI_LOG << "one fft:\n" << v;
        ComplexFt(vorig, &v2, true);
        // KALDI_LOG << "one fft[baseline]:\n" << v2;
        if (!ApproxEqual(v, v2) ) {
          ComplexFft(&vorig, true);
          KALDI_ASSERT(0);
        }
        ComplexFft(&v, false);
        // KALDI_LOG << "one more:\n" << v;
        v.Scale(1.0/(N/2));
        if (!ApproxEqual(v, vorig)) {
          ComplexFft(&vorig, true);
          KALDI_ASSERT(0);
        }// AssertEqual(v, vorig);
      }
    }
  }
}


template<typename Real> static void UnitTestSplitRadixComplexFft2() {

  // Make sure it inverts properly.
  for (MatrixIndexT p = 0; p < 30; p++) {
    MatrixIndexT logn = 1 + rand() % 10;
    MatrixIndexT N = 1 << logn;
    SplitRadixComplexFft<Real> srfft(N);
    for (MatrixIndexT q = 0; q < 3; q++) {
      Vector<Real> v(N*2), vorig(N*2);
      InitRand(&v);
      vorig.CopyFromVec(v);
      srfft.Compute(v.Data(), true);  // forward
      srfft.Compute(v.Data(), false);  // backward
      v.Scale(1.0/N);
      KALDI_ASSERT(ApproxEqual(v, vorig));
    }
  }
}


template<typename Real> static void UnitTestRealFft() {

  // First, test RealFftInefficient.
  for (MatrixIndexT N_ = 2; N_ < 100; N_ += 6) {
    MatrixIndexT N = N_;
    if (N >90) N *= rand() % 60;
    Vector<Real> v(N), w(N), x(N), y(N);
    InitRand(&v);
    w.CopyFromVec(v);
    RealFftInefficient(&w, true);
    y.CopyFromVec(v);
    RealFft(&y, true);  // test efficient one.
    // KALDI_LOG <<"v = "<<v;
    // KALDI_LOG << "Inefficient real fft of v is: "<< w;
    // KALDI_LOG << "Efficient real fft of v is: "<< y;
    AssertEqual(w, y, 0.01*N);
    x.CopyFromVec(w);
    RealFftInefficient(&x, false);
    RealFft(&y, false);
    // KALDI_LOG << "Inefficient real fft of v twice is: "<< x;
    if (N != 0) x.Scale(1.0/N);
    if (N != 0) y.Scale(1.0/N);
    AssertEqual(v, x, 0.001*N);
    AssertEqual(v, y, 0.001*N);  // ?
  }
}


template<typename Real> static void UnitTestSplitRadixRealFft() {

  for (MatrixIndexT p = 0; p < 30; p++) {
    MatrixIndexT logn = 2 + rand() % 11,
        N = 1 << logn;

    SplitRadixRealFft<Real> srfft(N);
    for (MatrixIndexT q = 0; q < 3; q++) {
      Vector<Real> v(N), w(N), x(N), y(N);
      InitRand(&v);
      w.CopyFromVec(v);
      RealFftInefficient(&w, true);
      y.CopyFromVec(v);
      srfft.Compute(y.Data(), true);  // test efficient one.
      // KALDI_LOG <<"v = "<<v;
      // KALDI_LOG << "Inefficient real fft of v is: "<< w;
      // KALDI_LOG << "Efficient real fft of v is: "<< y;
      AssertEqual(w, y, 0.01*N);
      x.CopyFromVec(w);
      RealFftInefficient(&x, false);
      srfft.Compute(y.Data(), false);
      // KALDI_LOG << "Inefficient real fft of v twice is: "<< x;
      x.Scale(1.0/N);
      y.Scale(1.0/N);
      AssertEqual(v, x, 0.001*N);
      AssertEqual(v, y, 0.001*N);  // ?
    }
  }
}



template<typename Real> static void UnitTestRealFftSpeed() {

  // First, test RealFftInefficient.
  KALDI_LOG << "starting. ";
  MatrixIndexT sz = 512;  // fairly typical size.
  for (MatrixIndexT i = 0; i < 3000; i++) {
    if (i % 1000 == 0) KALDI_LOG << "done 1000 [ == ten seconds of speech]\n";
    Vector<Real> v(sz);
    RealFft(&v, true);
  }
}

template<typename Real> static void UnitTestSplitRadixRealFftSpeed() {
  KALDI_LOG << "starting. ";
  MatrixIndexT sz = 512;  // fairly typical size.
  SplitRadixRealFft<Real> srfft(sz);
  for (MatrixIndexT i = 0; i < 6000; i++) {
    if (i % 1000 == 0)
      KALDI_LOG << "done 1000 [ == ten seconds of speech, split-radix]\n";
    Vector<Real> v(sz);
    srfft.Compute(v.Data(), true);
  }
}

template<typename Real>
void UnitTestComplexPower() {
  // This tests a not-really-public function that's used in Matrix::Power().

  for (MatrixIndexT i = 0; i < 10; i++) {
    Real power = RandGauss();
    Real x = 2.0, y = 0.0;
    bool ans = AttemptComplexPower(&x, &y, power);
    KALDI_ASSERT(ans);
    AssertEqual(std::pow(static_cast<Real>(2.0), power), x);
    AssertEqual(y, 0.0);
  }
  {
    Real x, y;
    x = 0.5; y = -0.3;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(2.21));
    KALDI_ASSERT(ans);
    ans = AttemptComplexPower(&x, &y, static_cast<Real>(1.0/2.21));
    KALDI_ASSERT(ans);
    AssertEqual(x, 0.5);
    AssertEqual(y, -0.3);
  }
  {
    Real x, y;
    x = 0.5; y = -0.3;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(2.0));
    KALDI_ASSERT(ans);
    AssertEqual(x, 0.5*0.5 - 0.3*0.3);
    AssertEqual(y, -0.3*0.5*2.0);
  }

  {
    Real x, y;
    x = 1.0/std::sqrt(2.0); y = -1.0/std::sqrt(2.0);
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(-1.0));
    KALDI_ASSERT(ans);
    AssertEqual(x, 1.0/std::sqrt(2.0));
    AssertEqual(y, 1.0/std::sqrt(2.0));
  }

  {
    Real x, y;
    x = 0.0; y = 0.0;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(-2.0));
    KALDI_ASSERT(!ans);  // zero; negative pow.
  }
  {
    Real x, y;
    x = -2.0; y = 0.0;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(1.5));
    KALDI_ASSERT(!ans);  // negative real case
  }
}
template<typename Real>
void UnitTestNonsymmetricPower() {

  for (MatrixIndexT iter = 0; iter < 30; iter++) {
    MatrixIndexT dimM = 1 + rand() % 20;
    Matrix<Real> M(dimM, dimM);
    InitRand(&M);

    Matrix<Real> MM(dimM, dimM);
    MM.AddMatMat(1.0, M, kNoTrans, M, kNoTrans, 0.0);  // MM = M M.
    Matrix<Real> MMMM(dimM, dimM);
    MMMM.AddMatMat(1.0, MM, kNoTrans, MM, kNoTrans, 0.0);

    Matrix<Real> MM2(MM);
    bool b = MM2.Power(1.0);
    KALDI_ASSERT(b);
    AssertEqual(MM2, MM);
    Matrix<Real> MMMM2(MM);
    b = MMMM2.Power(2.0);
    KALDI_ASSERT(b);
    AssertEqual(MMMM2, MMMM);
  }
  for (MatrixIndexT iter = 0; iter < 30; iter++) {
    MatrixIndexT dimM = 1 + rand() % 20;
    Matrix<Real> M(dimM, dimM);
    InitRand(&M);

    Matrix<Real> MM(dimM, dimM);
    MM.AddMatMat(1.0, M, kNoTrans, M, kNoTrans, 0.0);  // MM = M M.
    // This ensures there are no real, negative eigenvalues.

    Matrix<Real> MMMM(dimM, dimM);
    MMMM.AddMatMat(1.0, MM, kNoTrans, MM, kNoTrans, 0.0);

    Matrix<Real> MM2(M);
    if (!MM2.Power(2.0)) {  // possibly had negative eigenvalues
      KALDI_LOG << "Could not take matrix to power (not an error)\n";
    } else {
      AssertEqual(MM2, MM);
    }
    Matrix<Real> MMMM2(M);
    if (!MMMM2.Power(4.0)) {  // possibly had negative eigenvalues
      KALDI_LOG << "Could not take matrix to power (not an error)\n";
    } else {
      AssertEqual(MMMM2, MMMM);
    }
    Matrix<Real> MMMM3(MM);
    if (!MMMM3.Power(2.0)) {
      KALDI_ERR << "Could not take matrix to power (should have been able to)\n";
    } else {
      AssertEqual(MMMM3, MMMM);
    }

    Matrix<Real> MM4(MM);
    if (!MM4.Power(-1.0))
      KALDI_ERR << "Could not take matrix to power (should have been able to)\n";
    MM4.Invert();
    AssertEqual(MM4, MM);
  }
}

void UnitTestAddVecCross() {

  Vector<float> v(5);
  InitRand(&v);
  Vector<double> w(5);
  InitRand(&w);

  Vector<float> wf(w);

  for (MatrixIndexT i = 0; i < 2; i++) {
    float f = 1.0;
    if (i == 0) f = 2.0;

    {
      Vector<float> sum1(5);
      Vector<double> sum2(5);
      Vector<float> sum3(5);
      sum1.AddVec(f, v); sum1.AddVec(f, w);
      sum2.AddVec(f, v); sum2.AddVec(f, w);
      sum3.AddVec(f, v); sum3.AddVec(f, wf);
      Vector<float> sum2b(sum2);
      AssertEqual(sum1, sum2b);
      AssertEqual(sum1, sum3);
    }

    {
      Vector<float> sum1(5);
      Vector<double> sum2(5);
      Vector<float> sum3(5);
      sum1.AddVec2(f, v); sum1.AddVec2(f, w);
      sum2.AddVec2(f, v); sum2.AddVec2(f, w);
      sum3.AddVec2(f, v); sum3.AddVec2(f, wf);
      Vector<float> sum2b(sum2);
      AssertEqual(sum1, sum2b);
      AssertEqual(sum1, sum3);
    }
  }
}

template<typename Real> static void UnitTestMatrixExponential() {

  for (MatrixIndexT p = 0; p < 10; p++) {
    MatrixIndexT dim = 1 + rand() % 5;
    Matrix<Real> M(dim, dim);
    InitRand(&M);
    {  // work out largest eig.
      Real largest_eig = 0.0;
      Vector<Real> real_eigs(dim), imag_eigs(dim);
      M.Eig(NULL, &real_eigs, &imag_eigs);
      for (MatrixIndexT i = 0; i < dim; i++) {
        Real abs_eig =
            std::sqrt(real_eigs(i)*real_eigs(i) + imag_eigs(i)*imag_eigs(i));
        largest_eig = std::max(largest_eig, abs_eig);
      }
      if (largest_eig > 0.5) {  // limit largest eig to 0.5,
        // so Taylor expansion will converge.
        M.Scale(0.5 / largest_eig);
      }
    }

    Matrix<Real> expM(dim, dim);
    Matrix<Real> cur_pow(dim, dim);
    cur_pow.SetUnit();
    Real i_factorial = 1.0;
    for (MatrixIndexT i = 0; i < 52; i++) {  // since max-eig = 0.5 and 2**52 == dbl eps.
      if (i > 0) i_factorial *= i;
      expM.AddMat(1.0/i_factorial, cur_pow);
      Matrix<Real> tmp(dim, dim);
      tmp.AddMatMat(1.0, cur_pow, kNoTrans, M, kNoTrans, 0.0);
      cur_pow.CopyFromMat(tmp);
    }
    Matrix<Real> expM2(dim, dim);
    MatrixExponential<Real> mexp;
    mexp.Compute(M, &expM2);
    AssertEqual(expM, expM2);
  }
}


static void UnitTestMatrixExponentialBackprop() {
  for (MatrixIndexT p = 0; p < 10; p++) {
    MatrixIndexT dim = 1 + rand() % 5;
    // f is tr(N^T exp(M)).  backpropagating derivative
    // of this function.
    Matrix<double> M(dim, dim), N(dim, dim), delta(dim, dim);
    InitRand(&M);
    // { SpMatrix<double> tmp(dim); InitRand(&tmp); M.CopyFromSp(tmp); }
    InitRand(&N);
    InitRand(&delta);
    // { SpMatrix<double> tmp(dim); InitRand(&tmp); delta.CopyFromSp(tmp); }
    delta.Scale(0.00001);


    Matrix<double> expM(dim, dim);
    MatrixExponential<double> mexp;
    mexp.Compute(M, &expM);

    Matrix<double> expM2(dim, dim);
    {
      Matrix<double> M2(M);
      M2.AddMat(1.0, delta, kNoTrans);
      MatrixExponential<double> mexp2;
      mexp2.Compute(M2, &expM2);
    }
    double f_before = TraceMatMat(expM, N, kTrans);
    double f_after  = TraceMatMat(expM2, N, kTrans);

    Matrix<double> Mdash(dim, dim);
    mexp.Backprop(N, &Mdash);
    // now Mdash should be df/dM

    double f_diff = f_after - f_before;  // computed using method of differnces
    double f_diff2 = TraceMatMat(Mdash, delta, kTrans);  // computed "analytically"
    KALDI_LOG << "f_diff = " << f_diff << "\n";
    KALDI_LOG << "f_diff2 = " << f_diff2 << "\n";

    AssertEqual(f_diff, f_diff2);
  }
}
template<typename Real>
static void UnitTestPca(bool full_test) {
  // We'll test that we can exactly reconstruct the vectors, if
  // the PCA dim is <= the "real" dim that the vectors live in.
  for (MatrixIndexT i = 0; i < 10; i++) {
    bool exact = i % 2 == 0;
    MatrixIndexT true_dim = (full_test ? 200 : 50) + rand() % 5, //dim of subspace points live in
        feat_dim = true_dim + rand() % 5,  // dim of feature space
        num_points = true_dim + rand() % 5, // number of training points.
        G = std::min(feat_dim,
                     std::min(num_points,
                              static_cast<MatrixIndexT>(true_dim + rand() % 5)));

    Matrix<Real> Proj(feat_dim, true_dim);
    Proj.SetRandn();
    Matrix<Real> true_X(num_points, true_dim);
    true_X.SetRandn();
    Matrix<Real> X(num_points, feat_dim);
    X.AddMatMat(1.0, true_X, kNoTrans, Proj, kTrans, 0.0);

    Matrix<Real> U(G, feat_dim); // the basis
    Matrix<Real> A(num_points, G); // projection of points into the basis..
    ComputePca(X, &U, &A, true, exact);
    {
      SpMatrix<Real> I(G);
      I.AddMat2(1.0, U, kNoTrans, 0.0);
      KALDI_LOG << "Non-unit-ness of U is " << NonUnitness(I);
      KALDI_ASSERT(I.IsUnit(0.001));
    }
    Matrix<Real> X2(num_points, feat_dim);
    X2.AddMatMat(1.0, A, kNoTrans, U, kNoTrans, 0.0);
    // Check reproduction.
    KALDI_LOG << "A.Sum() " << A.Sum() << ", U.Sum() " << U.Sum();
    AssertEqual(X, X2, 0.01);
    // Check basis is orthogonal.
    Matrix<Real> tmp(G, G);
    tmp.AddMatMat(1.0, U, kNoTrans, U, kTrans, 0.0);
    KALDI_ASSERT(tmp.IsUnit(0.01));
  }
}


/* UnitTestPca2 test the same function, but it's more geared towards
    the 'inexact' method and for when you want less than the full number
    of PCA dimensions.
 */

template<typename Real>
static void UnitTestPca2(bool full_test) {
  for (MatrixIndexT i = 0; i < 5; i++) {
    // MatrixIndexT feat_dim = 600, num_points = 300;
    MatrixIndexT feat_dim = (full_test ? 600 : 100),
        num_points = (i%2 == 0 ? feat_dim * 2 : feat_dim / 2); // test
    // both branches of PCA code,  inner + outer.

    Matrix<Real> X(num_points, feat_dim);
    X.SetRandn();
    
    MatrixIndexT pca_dim = 30;

    Matrix<Real> U(pca_dim, feat_dim); // rows PCA directions.
    bool print_eigs = true, exact = false;
    ComputePca(X, &U, static_cast<Matrix<Real>*>(NULL), print_eigs, exact);

    Real non_orth = NonOrthogonality(U, kNoTrans);
    KALDI_ASSERT(non_orth < 0.001);
    KALDI_LOG << "Non-orthogonality of U is " << non_orth;
    Matrix<Real> U2(pca_dim, feat_dim);

    {
      SpMatrix<Real> Scatter(feat_dim);
      Scatter.AddMat2(1.0, X, kTrans, 0.0);
      Matrix<Real> V(feat_dim, feat_dim);
      Vector<Real> l(feat_dim);
      Scatter.Eig(&l, &V); // cols of V are eigenvectors.
      SortSvd(&l, &V);  // Get top dims.
      U2.CopyFromMat(SubMatrix<Real>(V, 0, feat_dim, 0, pca_dim), kTrans);
      Real non_orth = NonOrthogonality(U2, kNoTrans);
      KALDI_ASSERT(non_orth < 0.001);
      KALDI_LOG << "Non-orthogonality of U2 is " << non_orth;

      SpMatrix<Real> ScatterProjU(pca_dim), ScatterProjU2(pca_dim);
      ScatterProjU.AddMat2Sp(1.0, U, kNoTrans, Scatter, 0.0);
      ScatterProjU2.AddMat2Sp(1.0, U2, kNoTrans, Scatter, 0.0);
      KALDI_LOG << "Non-diagonality of proj with U is "
                << NonDiagonalness(ScatterProjU);
      KALDI_LOG << "Non-diagonality of proj with U2 is "
                << NonDiagonalness(ScatterProjU2);
      KALDI_ASSERT(ScatterProjU.IsDiagonal(0.01)); // Algorithm is statistical,
      // so it ends up being less accurate.
      KALDI_ASSERT(ScatterProjU2.IsDiagonal());
      KALDI_LOG << "Trace proj with U is " << ScatterProjU.Trace()
                << " with U2 is " << ScatterProjU2.Trace();
      AssertEqual(ScatterProjU.Trace(), ScatterProjU2.Trace(), 0.1);// Algorithm is
      // statistical so give it some leeway.
    }
  }
}

template<typename Real>
static void UnitTestSvdSpeed() {
  std::vector<MatrixIndexT> sizes;
  sizes.push_back(100);
  sizes.push_back(150);
  sizes.push_back(200);
  sizes.push_back(300);
  sizes.push_back(500);
  sizes.push_back(750);
  sizes.push_back(1000);
  sizes.push_back(2000);
  time_t start, end;
  for (size_t i = 0; i < sizes.size(); i++) {
    MatrixIndexT size = sizes[i];
    {
      start = time(NULL);
      SpMatrix<Real> S(size);
      Vector<Real> l(size);
      S.Eig(&l);
      end = time(NULL);
      double diff = difftime(end, start);
      KALDI_LOG << "For size " << size << ", Eig without eigenvectors took " << diff
                << " seconds.";
    }
    {
      start = time(NULL);
      SpMatrix<Real> S(size);
      S.SetRandn();
      Vector<Real> l(size);
      Matrix<Real> P(size, size);
      S.Eig(&l, &P);
      end = time(NULL);
      double diff = difftime(end, start);
      KALDI_LOG << "For size " << size << ", Eig with eigenvectors took " << diff
                << " seconds.";
    }
    {
      start = time(NULL);
      Matrix<Real> M(size, size);
      M.SetRandn();
      Vector<Real> l(size);
      M.Svd(&l, NULL, NULL);
      end = time(NULL);
      double diff = difftime(end, start);
      KALDI_LOG << "For size " << size << ", SVD without eigenvectors took " << diff
                << " seconds.";
    }
    {
      start = time(NULL);
      Matrix<Real> M(size, size), U(size, size), V(size, size);
      M.SetRandn();
      Vector<Real> l(size);
      M.Svd(&l, &U, &V);
      end = time(NULL);
      double diff = difftime(end, start);
      KALDI_LOG << "For size " << size << ", SVD with eigenvectors took " << diff
                << " seconds.";
    }
  }
}

template<typename Real> static void UnitTestCompressedMatrix() {
  // This is the basic test.

  CompressedMatrix empty_cmat;  // some tests on empty matrix
  KALDI_ASSERT(empty_cmat.NumRows() == 0);
  KALDI_ASSERT(empty_cmat.NumCols() == 0);

  MatrixIndexT num_failure = 0, num_tot = 10;
  for (MatrixIndexT n = 0; n < num_tot; n++) {
    MatrixIndexT num_rows = 10 * (rand() % 3), num_cols = rand() % 15;
    if (num_rows * num_cols == 0) {
      num_rows = 0;
      num_cols = 0;
    }
    Matrix<Real> M(num_rows, num_cols);
    if (rand() % 3 != 0) InitRand(&M);
    else {
      M.Add(RandGauss());
    }
    if (rand() % 2 == 0 && num_rows != 0) {  // set one row to all the same value,
      // which is one possible pathology.
      // Give it large dynamic range to increase chance that it
      // is the largest or smallest value in the matrix.
      M.Row(rand() % num_rows).Set(RandGauss() * 4.0);
    }
    double rand_val = RandGauss() * 4.0;
    // set a bunch of elements to all one value: increases
    // chance of pathologies.
    MatrixIndexT modulus = 1 + rand() % 5;
    for (MatrixIndexT r = 0; r < num_rows; r++)
      for (MatrixIndexT c = 0; c < num_cols; c++)
        if (rand() % modulus == 0) M(r, c) = rand_val;

    CompressedMatrix cmat(M);
    KALDI_ASSERT(cmat.NumRows() == num_rows);
    KALDI_ASSERT(cmat.NumCols() == num_cols);

    Matrix<Real> M2(cmat.NumRows(), cmat.NumCols());
    cmat.CopyToMat(&M2);

    Matrix<Real> diff(M2);
    diff.AddMat(-1.0, M);

    { // Check that when compressing a matrix that has already been compressed,
      // and uncompressing, we get the same answer.
      CompressedMatrix cmat2(M2);
      Matrix<Real> M3(cmat.NumRows(), cmat.NumCols());
      cmat2.CopyToMat(&M3);
      if (!M2.ApproxEqual(M3, 1.0e-05)) {
        KALDI_LOG << "cmat is: ";
        cmat.Write(std::cout, false);
        KALDI_LOG << "cmat2 is: ";
        cmat2.Write(std::cout, false);
        KALDI_ERR << "Matrices differ " << M2 << " vs. " << M3 << ", M2 range is "
                  << M2.Min() << " to " << M2.Max() << ", M3 range is " 
                  << M3.Min() << " to " << M3.Max();
      }
    }
    
    // test CopyRowToVec
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      Vector<Real> V(num_cols);
      cmat.CopyRowToVec(i, &V);  // get row.
      for (MatrixIndexT k = 0; k < num_cols; k++) {
        AssertEqual(M2(i, k), V(k));
      }
    }
    
    // test CopyColToVec
    for (MatrixIndexT i = 0; i < num_cols; i++) {
      Vector<Real> V(num_rows);
      cmat.CopyColToVec(i, &V);  // get col.
      for (MatrixIndexT k = 0;k < num_rows;k++) {
        AssertEqual(M2(k, i), V(k));
      }
    }

    //test of getting a submatrix
    if(num_rows != 0 && num_cols != 0){
      MatrixIndexT sub_row_offset = (num_rows == 1 ? 0 : rand() % (num_rows-1)),
          sub_col_offset = (num_cols == 1 ? 0 : rand() % (num_cols-1));
      // to make sure we don't mod by zero
      MatrixIndexT num_subrows = rand() % (num_rows-sub_row_offset),
          num_subcols = rand() % (num_cols-sub_col_offset);
      if(num_subrows == 0 || num_subcols == 0){  // in case we randomized to
        // empty matrix, at least make it correct
        num_subrows = 0;
        num_subcols = 0;
      }
      Matrix<Real> Msub(num_subrows, num_subcols);
      cmat.CopyToMat(sub_row_offset, sub_col_offset, &Msub);
      for (MatrixIndexT i = 0; i < num_subrows; i++) {
        for (MatrixIndexT k = 0;k < num_subcols;k++) {
          AssertEqual(M2(i+sub_row_offset, k+sub_col_offset), Msub(i, k));
        }
      }
    }

    if (n < 5) {  // test I/O.
      bool binary = (n % 2 == 1);
      {
        std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
        InitKaldiOutputStream(outs, binary);
        cmat.Write(outs, binary);
      }
      CompressedMatrix cmat2;
      {
        bool binary_in;
        std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
        InitKaldiInputStream(ins, &binary_in);
        cmat2.Read(ins, binary_in);
      }
#if 1
      { // check that compressed-matrix can be read as matrix.
        bool binary_in;
        std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
        InitKaldiInputStream(ins, &binary_in);
        Matrix<Real> mat1;
        mat1.Read(ins, binary_in);
        Matrix<Real> mat2(cmat2);
        AssertEqual(mat1, mat2);
      }
#endif


      { // check that matrix can be read as compressed-matrix.
        Matrix<Real> mat1(cmat);
        {
          std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
          InitKaldiOutputStream(outs, binary);
          mat1.Write(outs, binary);
        }
        bool binary_in;
        std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
        InitKaldiInputStream(ins, &binary_in);
        CompressedMatrix cmat2;
        cmat2.Read(ins, binary_in);
        Matrix<Real> mat2(cmat2);
        AssertEqual(mat1, mat2);
      }
      
      
      Matrix<Real> M3(cmat2.NumRows(), cmat2.NumCols());
      cmat2.CopyToMat(&M3);
      AssertEqual(M2, M3); // tests I/O of CompressedMatrix.

      CompressedMatrix cmat3(cmat2); // testing self-constructor, which
      // tests assignment operator.
      Matrix<Real> M4(cmat3.NumRows(), cmat3.NumCols());
      cmat3.CopyToMat(&M4);
      AssertEqual(M2, M4);
    }
    KALDI_LOG << "M = " << M;
    KALDI_LOG << "M2 = " << M2;
    double tot = M.FrobeniusNorm(), err = diff.FrobeniusNorm();
    KALDI_LOG << "Compressed matrix, tot = " << tot << ", diff = "
              << err;
    if (err > 0.01 * tot) {
      KALDI_WARN << "Failure in compressed-matrix test.";
      num_failure++;
    }
  }
  if (num_failure > 1)
    KALDI_ERR << "Too many failures in compressed matrix test.";
}
  

template<typename Real>
static void UnitTestTridiag() {
  SpMatrix<Real> A(3);
  A(1,1) = 1.0;
  A(1, 2) = 1.0;
  KALDI_ASSERT(A.IsTridiagonal());
  A(0, 2) = 1.0;
  KALDI_ASSERT(!A.IsTridiagonal());
  A(0, 2) = 0.0;
  A(0, 1) = 1.0;
  KALDI_ASSERT(A.IsTridiagonal());
}

template<typename Real>
static void UnitTestRandCategorical() {
  int32 N = 1 + rand()  % 10;
  Vector<Real> vec(N);
  for (int32 n = 0; n < N; n++)
    vec(n) = rand() % 3;
  if (vec.Sum() == 0)
    vec(0) = 2.0;
  Real sum = vec.Sum();
  int32 num_samples = 100000;
  std::vector<int32> counts(N, 0);
  for (int32 i = 0; i < num_samples; i++)
    counts[vec.RandCategorical()]++;
  for (int32 n = 0; n < N; n++) {
    Real a = counts[n] / (1.0 * num_samples),
        b = vec(n) / sum;
    KALDI_LOG << "a = " << a << ", b = " << b;
    KALDI_ASSERT(fabs(a - b) <= 0.1); // pretty arbitrary.  Will increase #samp if fails.
  }
}

template<class Real>
static void UnitTestTopEigs() {
  for (MatrixIndexT i = 0; i < 2; i++) {
    // Previously tested with this but takes too long.
    MatrixIndexT dim = 400, num_eigs = 100;
    SpMatrix<Real> mat(dim);
    for (MatrixIndexT i = 0; i < dim; i++)
      for (MatrixIndexT j = 0; j <= i; j++)
        mat(i, j) = RandGauss();

    Matrix<Real> P(dim, num_eigs);
    Vector<Real> s(num_eigs);
    mat.TopEigs(&s, &P);
    { // P should have orthogonal columns.  Check this.
      SpMatrix<Real> S(num_eigs);
      S.AddMat2(1.0, P, kTrans, 0.0);
      KALDI_LOG << "Non-unit-ness of S is " << NonUnitness(S);
      KALDI_ASSERT(S.IsUnit(1.0e-04));
    }
    // Note: we call the matrix "mat" by the name "S" below.
    Matrix<Real> SP(dim, num_eigs); // diag of P^T SP should be eigs.
    SP.AddSpMat(1.0, mat, P, kNoTrans, 0.0);
    Matrix<Real> PSP(num_eigs, num_eigs);
    PSP.AddMatMat(1.0, P, kTrans, SP, kNoTrans, 0.0);
    Vector<Real> s2(num_eigs);
    s2.CopyDiagFromMat(PSP);
    AssertEqual(s, s2);
    // Now check that eigs are close to real top eigs.
    {
      Matrix<Real> fullP(dim, dim);
      Vector<Real> fulls(dim);
      mat.Eig(&fulls, &fullP);
      KALDI_LOG << "Approximate eigs are " << s;
      // find sum of largest-abs-value eigs.
      fulls.Abs();
      std::sort(fulls.Data(), fulls.Data() + dim);
      SubVector<Real> tmp(fulls, dim - num_eigs, num_eigs);
      KALDI_LOG << "abs(real eigs) are " << tmp;
      BaseFloat real_sum = tmp.Sum();
      s.Abs();
      BaseFloat approx_sum = s.Sum();
      KALDI_LOG << "real sum is " << real_sum << ", approx_sum = " << approx_sum;
    }
  }
}

template<typename Real> static void MatrixUnitTest(bool full_test) {
  UnitTestAddMatSmat<Real>();
  UnitTestFloorChol<Real>();
  UnitTestFloorUnit<Real>();
  UnitTestAddMat2Sp<Real>();
  UnitTestLbfgs<Real>();
  // UnitTestSvdBad<Real>(); // test bug in Jama SVD code.
  UnitTestCompressedMatrix<Real>();
  UnitTestResize<Real>();
  UnitTestMatrixExponentialBackprop();
  UnitTestMatrixExponential<Real>();
  UnitTestNonsymmetricPower<Real>();
  UnitTestEigSymmetric<Real>();
  KALDI_LOG << " Point A";
  UnitTestComplexPower<Real>();
  UnitTestEig<Real>();
  UnitTestEigSp<Real>();
  UnitTestTridiagonalize<Real>();
  UnitTestTridiagonalizeAndQr<Real>();  
  // commenting these out for now-- they test the speed, but take a while.
  // UnitTestSplitRadixRealFftSpeed<Real>();
  // UnitTestRealFftSpeed<Real>();   // won't exit!/
  UnitTestComplexFt<Real>();
  KALDI_LOG << " Point B";
  UnitTestComplexFft2<Real>();
  UnitTestComplexFft<Real>();
  UnitTestSplitRadixComplexFft<Real>();
  UnitTestSplitRadixComplexFft2<Real>();
  UnitTestDct<Real>();
  UnitTestRealFft<Real>();
  KALDI_LOG << " Point C";
  UnitTestSplitRadixRealFft<Real>();
  UnitTestSvd<Real>();
  UnitTestSvdNodestroy<Real>();
  UnitTestSvdJustvec<Real>();
  UnitTestSpAddVec<Real, float>();
  UnitTestSpAddVec<Real, double>();
  UnitTestSpAddVecVec<Real>();
  UnitTestSpInvert<Real>();
  KALDI_LOG << " Point D";
  UnitTestTpInvert<Real>();
  UnitTestIo<Real>();
  UnitTestIoCross<Real>();
  UnitTestHtkIo<Real>();
  UnitTestScale<Real>();
  UnitTestTrace<Real>();
  KALDI_LOG << " Point E";
  CholeskyUnitTestTr<Real>();
  UnitTestAxpy<Real>();
  UnitTestSimple<Real>();
  UnitTestMmul<Real>();
  UnitTestMmulSym<Real>();
  UnitTestVecmul<Real>();
  UnitTestInverse<Real>();
  UnitTestMulElements<Real>();
  UnitTestDotprod<Real>();
  // UnitTestSvdVariants<Real>();
  UnitTestPower<Real>();
  UnitTestHeaviside<Real>();
  UnitTestCopySp<Real>();
  UnitTestDeterminant<Real>();
  KALDI_LOG << " Point F";
  UnitTestDeterminantSign<Real>();
  UnitTestSger<Real>();
  UnitTestAddOuterProductPlusMinus<Real>();
  UnitTestTraceProduct<Real>();
  UnitTestTransposeScatter<Real>(); 
  UnitTestRankNUpdate<Real>();
  UnitTestSherman<Real>();
  UnitTestSpVec<Real>();
  UnitTestLimitCondInvert<Real>();
  KALDI_LOG << " Point G";
  UnitTestLimitCond<Real>();
  UnitTestMat2Vec<Real>();
  UnitTestFloorCeiling<Real>();
  UnitTestSpLogExp<Real>();
  KALDI_LOG << " Point H";
  UnitTestCopyRowsAndCols<Real>();
  UnitTestSpliceRows<Real>();
  UnitTestAddSp<Real>();
  UnitTestRemoveRow<Real>();
  UnitTestRow<Real>();
  UnitTestSubvector<Real>();
  UnitTestRange<Real>();
  UnitTestSimpleForVec<Real>();
  UnitTestSetRandn<Real>();
  UnitTestSetRandUniform<Real>();
  UnitTestVectorMax<Real>();
  UnitTestVectorMin<Real>();
  UnitTestSimpleForMat<Real>();
  UnitTestTanh<Real>();
  UnitTestSigmoid<Real>();
  UnitTestSoftHinge<Real>();
  UnitTestNorm<Real>();
  UnitTestCopyCols<Real>();
  UnitTestCopyRows<Real>();
  UnitTestMul<Real>();
  KALDI_LOG << " Point I";
  UnitTestSolve<Real>();
  UnitTestAddMat2<Real>();
  UnitTestAddMatSelf<Real>();
  UnitTestMaxMin<Real>();
  UnitTestInnerProd<Real>();
  UnitTestScaleDiag<Real>();
  UnitTestSetDiag<Real>();
  UnitTestSetRandn<Real>();
  KALDI_LOG << " Point J";
  UnitTestTraceSpSpLower<Real>();
  UnitTestTranspose<Real>();
  UnitTestAddVec2Sp<Real>();
  UnitTestAddVecToRows<Real>();
  UnitTestAddVecToCols<Real>();
  UnitTestAddVecCross();
  UnitTestTp2Sp<Real>();
  UnitTestTp2<Real>();
  UnitTestAddDiagMat2<Real>();
  UnitTestAddDiagMatMat<Real>();
  UnitTestOrthogonalizeRows<Real>();
  UnitTestTopEigs<Real>();
  UnitTestRandCategorical<Real>();
  UnitTestTridiag<Real>();
  UnitTestTridiag<Real>();
  //  SlowMatMul<Real>();
  UnitTestAddDiagVecMat<Real>();
  UnitTestAddToDiagMatrix<Real>();
  UnitTestAddToDiag<Real>();
  UnitTestMaxAbsEig<Real>();
  UnitTestMax2<Real>();
  UnitTestPca<Real>(full_test);
  UnitTestPca2<Real>(full_test);
  UnitTestAddVecVec<Real>();
  UnitTestReplaceValue<Real>();
  // The next one is slow.  The upshot is that Eig is up to ten times faster
  // than SVD. 
  // UnitTestSvdSpeed<Real>();
}



}


int main() {
  using namespace kaldi;
  bool full_test = false;
  kaldi::MatrixUnitTest<double>(full_test);
  kaldi::MatrixUnitTest<float>(full_test);
  KALDI_LOG << "Tests succeeded.\n";

}

