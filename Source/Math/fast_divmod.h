//
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <cstdio>
#include <cassert>
#include <cmath>

namespace Microsoft { namespace MSR { namespace CNTK {

namespace detail {

__host__ __device__ __forceinline__
int mulhi(const int M, const int n) {
#ifdef __CUDA_ARCH__
  return __mulhi(M, n);
#else
  return (((unsigned long long)((long long)M * (long long)n)) >> 32);
#endif
}

}

// Based on code from Chapter 10 of "Hacker's Delight, 2nd ed."
class fast_divmod {
 public:
  fast_divmod(int d = 1) : d_(d), a_(0) {
    find_magic_numbers();
  }
  __host__ __device__
  fast_divmod(const fast_divmod& other) : d_(other.d_), a_(other.a_), s_(other.s_), M_(other.M_) {};
  __host__ __device__
  int div(int n) {
    // get high 32 bits of M * n
    int q = detail::mulhi(M_, n);
  
    // deal with add / subs if needed
    q += a_ * n;

    // shift if necessary
    if (s_ >= 0) {
      q >>= s_;
      q += ((unsigned int)q >> 31);
    }

    return q;
  }
  __host__ __device__
  void divmod(int n, int& q, int& r) {
    // handle special cases
    if (d_ == 1) {
      q = n;
      r = 0;
    } else if (d_ == -1) {
      q = -n;
      r = 0;
    } else {
      // general case
      q = div(n);
      r = n - q*d_;
    }
  }
  // Needed for overflow checks in FixedArray / FixedMatrix
  bool operator!=(const fast_divmod& other) {
    return (d_ != other.d_) || (M_ != other.M_) || (s_ != other.s_) || (a_ != other.a_);
  }

 public:
  int d_, M_, s_, a_;

 private:
  // Based on code from Hacker's delight 2.ed
  // Chapter 10, figure 10-1
  void find_magic_numbers() {
    // special case for d = 1, -1
    if (d_ == 1) {
      M_ = 0;
      s_ = 0;
      a_ = 1;
      return;
    } else if (d_ == -1) {
      M_ = 0;
      s_ = -1;
      a_ = -1;
      return;
    }
    // general case
    const unsigned two31 = 0x80000000;
    unsigned abs_d = (d_ == 0) ? 1 : abs(d_);
    unsigned t = two31 + ((unsigned)d_ >> 31); // t = 2^31 + (d < 0) ? 1 : 0
    unsigned abs_nc = t - 1 - (t % abs_d); // |n_c| = t - 1 - rem(t, |d|)
    int p = 31;
    unsigned q1 = two31 / abs_nc;      // Init q_1 = 2^31 / |n_c|
    unsigned r1 = two31 - q1 * abs_nc; // Init r_1 = rem(q_1, |n_c|)
    unsigned q2 = two31 / abs_d;       // Init q_2 = 2^31 / |d|
    unsigned r2 = two31 - q2 * abs_d;  // Init r_2 = rem(q_2, |d|)

    unsigned delta;
    // iterate p until
    // 2^p < n_c * (d - rem(2^p, d)) is satisfied
    do {
      ++p;
      q1 *= 2;
      r1 *= 2;

      if (r1 >= abs_nc) {
        q1 += 1;
        r1 -= abs_nc;
      }
      q2 *= 2;
      r2 *= 2;

      if (r2 >= abs_d) {
        q2 += 1;
        r2 -= abs_d;
      }
      delta = abs_d - r2;
    } while (q1 < delta ||
             (q1 == delta && r1 == 0));

    // store magic numbers
    M_ = q2 + 1;
    if (d_ < 0) M_ = -M_;
    s_ = p - 32;

    // generate sentinel for correct adds / subs
    // "generate the add if d > 0 and M < 0"
    if ((d_ > 0) && (M_ < 0)) a_ = 1;
    //  "generate the sub if d < 0 and M > 0"
    else if ((d_ < 0) && (M_ > 0)) a_ = -1;
    // Otherwise no add / sub needed
    else a_ = 0;
  }
};

}}}  // End namespaces
