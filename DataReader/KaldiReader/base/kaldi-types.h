// base/kaldi-types.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Jan Silovsky;  Yanmin Qian

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

#ifndef KALDI_BASE_KALDI_TYPES_H_
#define KALDI_BASE_KALDI_TYPES_H_ 1

namespace kaldi {
// TYPEDEFS ..................................................................
#if (KALDI_DOUBLEPRECISION != 0)
typedef double  BaseFloat;
#else
typedef float   BaseFloat;
#endif
}

#ifdef _MSC_VER
namespace kaldi {
typedef unsigned __int16 uint16;
typedef unsigned __int32 uint32;
typedef __int16          int16;
typedef __int32          int32;
typedef __int64          int64;
typedef unsigned __int64 uint64;
typedef float          float32;
typedef double        double64;
}
#else
// we can do this a different way if some platform
// we find in the future lacks stdint.h
#include <stdint.h>

namespace kaldi {
typedef uint16_t        uint16;
typedef uint32_t        uint32;
typedef uint64_t        uint64;
typedef int16_t         int16;
typedef int32_t         int32;
typedef int64_t         int64;
typedef float           float32;
typedef double         double64;
}  // end namespace kaldi
#endif

#endif  // KALDI_BASE_KALDI_TYPES_H_
