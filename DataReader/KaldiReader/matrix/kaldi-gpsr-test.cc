// matrix/kaldi-gpsr-test.cc

// Copyright 2012   Arnab Ghoshal

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "gmm/model-test-common.h"
#include "matrix/kaldi-gpsr.h"
#include "util/kaldi-io.h"

using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

namespace kaldi {

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
      KALDI_WARN << "Condition number of random matrix large" << M->Cond()
                 << ": trying again (this is normal)";
      goto start;
    }
}

template<typename Real> static void InitRand(SpMatrix<Real> *M) {
 start_sp:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
    for (MatrixIndexT j = 0;j<=i;j++)
      (*M)(i, j) = RandGauss();
  if (M->NumRows() != 0 && M->Cond() > 100) {
    KALDI_WARN << "Condition number of random matrix large" << M->Cond()
               << ": trying again (this is normal)";
    goto start_sp;
  }
}

template<typename Real> static void UnitTestGpsr() {
  for (int32 i = 0; i < 5; i++) {
    MatrixIndexT dim1 = (rand() % 10) + 10;
    MatrixIndexT dim2 = (rand() % 10) + 10;

    Matrix<Real> M(dim1, dim2);
    InitRand(&M);
    SpMatrix<Real> H(dim2);
    H.AddMat2(1.0, M, kTrans, 0.0);  // H = M^T M
//    InitRand(&H);
//    KALDI_LOG << "dim 1 " << dim1 << "; dim 2 " << dim2 << " LD " << H.LogDet()
//              << " Cond " << H.Cond() << "\nH " << H;
//    KALDI_ASSERT(H.IsPosDef());

    Vector<Real> x(dim2);
    InitRand(&x);
    Vector<Real> g(dim2);
    InitRand(&g);
    GpsrConfig opts;
    opts.debias = (rand()%2 == 0);
    Real objf_old = 0.5* VecSpVec(x, H, x) - VecVec(x, g) +
        opts.gpsr_tau * x.Norm(1.0);
    GpsrBasic(opts, H, g, &x);
    Real objf_new = 0.5* VecSpVec(x, H, x) - VecVec(x, g) +
        opts.gpsr_tau * x.Norm(1.0);
    KALDI_ASSERT(objf_old >= objf_new);  // since we are minimizing
    KALDI_LOG << "GPSR-basic: objf old = " << objf_old << "; new = " << objf_new;
    Vector<Real> x2(x);
    GpsrBB(opts, H, g, &x);
    Real objf_new_bb = 0.5* VecSpVec(x, H, x) - VecVec(x, g) +
        opts.gpsr_tau * x.Norm(1.0);
    KALDI_ASSERT(objf_old >= objf_new_bb);  // since we are minimizing
    KALDI_LOG << "GPSR-BB: objf old = " << objf_old << "; new = " << objf_new_bb;
  }
}

}

int main() {
  kaldi::g_kaldi_verbose_level = 1;
  kaldi::UnitTestGpsr<float>();
  kaldi::UnitTestGpsr<double>();
  std::cout << "Test OK.\n";
  return 0;
}
