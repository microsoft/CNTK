// base/kaldi-error-test.cc

// Copyright 2009-2011  Microsoft Corporation

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


#include "base/kaldi-common.h"

// testing that we get the stack trace.
namespace kaldi {

void MyFunction2() {
  KALDI_ERR << "Ignore this error";
}

void MyFunction1() {
  MyFunction2();
}

void UnitTestError() {
  {
    std::cerr << "Ignore next error:\n";
    MyFunction1();
  }
}


}  // end namespace kaldi.

int main() {
  kaldi::g_program_name = "/foo/bar/kaldi-error-test";
  try {
    kaldi::UnitTestError();
    KALDI_ASSERT(0);  // should not happen.
  } catch (std::runtime_error &r) {
    std::cout << "UnitTestError: the error we generated was: " << r.what();
  }
}

