//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#ifndef CPUONLY

#include <cuda_fp16.h> // ASSUME CUDA9

#if defined(__CUDACC__)
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else
#define __CUDA_HOSTDEVICE__
#endif

class alignas(2) half : public __half {
public:
    half() = default;
    __CUDA_HOSTDEVICE__ half(const half& other) { __x = other.__x; }
    __CUDA_HOSTDEVICE__ half& operator=(const half& other) { __x = other.__x; return *this; }
    __CUDA_HOSTDEVICE__ half(half&& other) { *this = std::move(other); }
    __CUDA_HOSTDEVICE__ half& operator=(half&& other) { *this = std::move(other); return *this; }

    // convert to/from __half
    __CUDA_HOSTDEVICE__ half(const __half& other) : __half(other) {}
    __CUDA_HOSTDEVICE__ half& operator=(const __half& other) { *this = half(other); return *this; }
    //__CUDA_HOSTDEVICE__ operator __half() const { __half ret; ret.__x = __x; return ret; }

    // construction from build-in types
    __CUDA_HOSTDEVICE__ half(float f) {
#ifndef __CUDA_ARCH__
        __x = 0U; //fake
#else
        //__x = __float2half(f).__x;
        //*this = reinterpret_cast<half>(__float2half(f))
        *this = half(__float2half(f));
#endif
    }

    // cast to build-in types
    __CUDA_HOSTDEVICE__ half& operator=(float f) {
#ifndef __CUDA_ARCH__
        __x = 0U; return *this; //fake
#else
        //__x = __float2half(f).__x; return *this;
        *this = half(__float2half(f)); return *this;
#endif
    }
    __CUDA_HOSTDEVICE__ operator float() const {
#ifndef __CUDA_ARCH__
        return 0.0f; //fake
#else
        return __half2float(*this);
#endif
    }

    __CUDA_HOSTDEVICE__ operator bool() const { return (__x & 0x7FFF) != 0; }
};

#undef __CUDA_HOSTDEVICE__

#endif //CPUONLY
