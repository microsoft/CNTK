//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// define half type since __half is device only
// TODO: investigate performance of implementation, function signature and efficiency

#pragma once

#ifndef CPUONLY

#include <cuda_fp16.h> // ASSUME CUDA9

#if defined(__CUDACC__)
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else
#define __CUDA_HOSTDEVICE__
#endif

void halfbits2float(const unsigned short*, float*);
void float2halfbits(float*, unsigned short*);
class alignas(2) half : public __half {
public:
    half() = default;
    __CUDA_HOSTDEVICE__ half(const half& other) { __x = other.__x; }
    __CUDA_HOSTDEVICE__ half& operator=(const half& other) { __x = other.__x; return *this; }
    __CUDA_HOSTDEVICE__ half(half&& other) { *this = std::move(other); }
    __CUDA_HOSTDEVICE__ half& operator=(half&& other) { *this = std::move(other); return *this; }

    // convert from __half
    __CUDA_HOSTDEVICE__ half(const __half& other) : __half(other) {}
    __CUDA_HOSTDEVICE__ half& operator=(const __half& other) { *this = half(other); return *this; }

    // construction from build-in types
    __CUDA_HOSTDEVICE__ half(float f) {
#ifndef __CUDA_ARCH__
        float2halfbits(&f, &__x);
#else
        *this = half(__float2half(f));
#endif
    }

    __CUDA_HOSTDEVICE__ half& operator=(float f) {
#ifndef __CUDA_ARCH__
        float2halfbits(&f, &__x); return *this;
#else
        *this = half(__float2half(f)); return *this;
#endif
    }

    // cast to build-in types
    __CUDA_HOSTDEVICE__ operator float() const {
#ifndef __CUDA_ARCH__
        float f;
        halfbits2float(&__x, &f);
        return f;
#else
        return __half2float(*this);
#endif
    }

    __CUDA_HOSTDEVICE__ operator bool() const { return (__x & 0x7FFF) != 0; }
};

// Host functions for converting between FP32 and FP16 formats

inline void halfbits2float(const unsigned short* src, float* res)
{
    unsigned h = *src;
    unsigned sign = ((h >> 15) & 1);
    unsigned exponent = ((h >> 10) & 0x1f);
    unsigned mantissa = ((h & 0x3ff) << 13);

    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    *(unsigned*)res = ((sign << 31) | (exponent << 23) | mantissa);
}

inline void float2halfbits(float* src, unsigned short* dest)
{
    unsigned x = *(unsigned*)src;
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        *dest = 0x7fffU;
        return ;
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        *dest = sign | 0x7c00U;
        return;
    }
    if (u < 0x33000001) {
        *dest = (sign | 0x0000);
        return;
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    *dest = (sign | (exponent << 10) | mantissa);
}

#undef __CUDA_HOSTDEVICE__

#endif //CPUONLY
