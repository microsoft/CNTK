// util/timer.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation

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
#ifndef KALDI_UTIL_TIMER_H_
#define KALDI_UTIL_TIMER_H_

#if defined(_MSC_VER) || defined(MINGW)

# include <windows.h>

namespace kaldi
{

class Timer {
 public:
  Timer() { Reset(); }
  void Reset() {
    QueryPerformanceCounter(&time_start_);
  }
  double Elapsed() {
    LARGE_INTEGER time_end;
    LARGE_INTEGER freq;
    QueryPerformanceCounter(&time_end);
    if (QueryPerformanceFrequency(&freq) == 0) return 0.0;  // Hardware does not support this.
    return ((double)time_end.QuadPart - (double)time_start_.QuadPart) /
        ((double)freq.QuadPart);
  }
 private:
  LARGE_INTEGER time_start_;
};
}

#else

# include <sys/time.h>
# include <unistd.h>
namespace kaldi
{
class Timer
{
 public:
  Timer() { Reset(); }

  void Reset() { gettimeofday(&this->time_start_, &time_zone_); }

  /// Returns time in seconds.
  double Elapsed() {
    struct timeval time_end;
    gettimeofday(&time_end, &time_zone_);
    double t1, t2;
    t1 =  (double)time_start_.tv_sec +
        (double)time_start_.tv_usec/(1000*1000);
    t2 =  (double)time_end.tv_sec + (double)time_end.tv_usec/(1000*1000);
    return t2-t1;
  }

 private:
  struct timeval time_start_;
  struct timezone time_zone_;
};
}

#endif


#endif

