// base/kaldi-utils.cc
// Copyright 2009-2011   Karel Vesely;  Yanmin Qian;  Microsoft Corporation

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

#include <string>
#include "base/kaldi-common.h"

namespace kaldi {

std::string CharToString(const char &c) {
  char buf[20];
  if (std::isprint(c))
    sprintf(buf, "\'%c\'", c);
  else
    sprintf(buf, "[character %d]", (int) c);
  return (std::string) buf;
}

}  // end namespace kaldi
