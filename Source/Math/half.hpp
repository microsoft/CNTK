//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// define half type since __half is device only
// TODO: investigate performance of implementation, function signature and efficiency

#pragma once

#include "../CNTKv2LibraryDll/API/HalfConverter.hpp"

#if !defined(CPUONLY) && __has_include("cuda_fp16.h")
#include <cuda_fp16.h> // ASSUME CUDA9
#else
class alignas(2) __half
{
protected:
    unsigned short __x;
};
#endif

#if defined(__CUDACC__)
#define __CUDA_HOSTDEVICE__ __host__ __device__
#define __INLINE__ __forceinline__
#else
#define __CUDA_HOSTDEVICE__
#define __INLINE__ inline
#endif

#define __FP16_DECL__ __INLINE__ __CUDA_HOSTDEVICE__

class alignas(2) half : public __half {
public:
    half() = default;
    __FP16_DECL__ half(const half& other) { __x = other.__x; }
    __FP16_DECL__ half& operator=(const half& other) { __x = other.__x; return *this; }
    __FP16_DECL__ half(half&& other) { *this = std::move(other); }

    //warning C4717 : 'half::operator=' : recursive on all control paths, function will cause runtime stack overflow
    //__CUDA_HOSTDEVICE__ half& operator=(half&& other) { *this = std::move(other); return *this; }

    // convert from __half
    __FP16_DECL__ half(const __half& other) : __half(other) {}
    __FP16_DECL__ half& operator=(const __half& other) { *this = half(other); return *this; }

    // construction from build-in types
    __FP16_DECL__ half(float f) {
#ifndef __CUDA_ARCH__
        CNTK::float2halfbits(&f, &__x);
#else
        *this = half(__float2half(f));
#endif
    }

    __FP16_DECL__ half& operator=(float f) {
#ifndef __CUDA_ARCH__
        CNTK::float2halfbits(&f, &__x); return *this;
#else
        *this = half(__float2half(f)); return *this;
#endif
    }

    __FP16_DECL__ half(double d) : half((float)d) {}

    __FP16_DECL__ half& operator=(double d) {
        *this = ((float)d);
        return *this;
    }

    __FP16_DECL__ half(int i) : half((float)i) {}

    __FP16_DECL__ half& operator=(int i) {
        *this = ((float)i);
        return *this;
    }

    __FP16_DECL__ half(size_t u) : half((float)u) {}

    __FP16_DECL__ half& operator=(size_t u) {
        *this = ((float)u);
        return *this;
    }

    // cast to build-in types
    __FP16_DECL__ operator float() const {
#ifndef __CUDA_ARCH__
        float f;
        CNTK::halfbits2float(&__x, &f);
        return f;
#else
        return __half2float(*this);
#endif
    }

//    __CUDA_HOSTDEVICE__ operator bool() const { return (__x & 0x7FFF) != 0; }
};

/* A selector used in kernels to get compute type base on ElemType(storage) */
/* default case, compute type == ElemType */
template <typename ElemType>
struct TypeSelector
{
    typedef ElemType comp_t;
};

/* Specialization for half. Kernels uses this wants io in half while compute in float */
template <>
struct TypeSelector<half>
{
    typedef float comp_t;
};

/* operators to write to/read from files for half */
inline Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, half& h)
{
    stream >> *(short *)&h;
    return stream;
}
inline Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const half& h)
{
    stream << *(short *)&h;
    return stream;
}

/* Some basic arithmetic operations expected of a builtin */
__FP16_DECL__ half operator+(const half &lh, const half &rh) { return (half)((float)lh + (float)rh); }
__FP16_DECL__ half operator-(const half &lh, const half &rh) { return (half)((float)lh - (float)rh); }
__FP16_DECL__ half operator*(const half &lh, const half &rh) { return (half)((float)lh * (float)rh); }
__FP16_DECL__ half operator/(const half &lh, const half &rh) { return (half)((float)lh / (float)rh); }

__FP16_DECL__ half &operator+=(half &lh, const half &rh) { lh = lh + rh; return lh; }
__FP16_DECL__ half &operator-=(half &lh, const half &rh) { lh = lh - rh; return lh; }
__FP16_DECL__ half &operator*=(half &lh, const half &rh) { lh = lh * rh; return lh; }
__FP16_DECL__ half &operator/=(half &lh, const half &rh) { lh = lh / rh; return lh; }

__FP16_DECL__ half &operator++(half &h)      { h += half(1.0f); return h; }
__FP16_DECL__ half &operator--(half &h)      { h -= half(1.0f); return h; }
__FP16_DECL__ half  operator++(half &h, int) { half ret = h; h += half(1.0f); return ret; }
__FP16_DECL__ half  operator--(half &h, int) { half ret = h; h -= half(1.0f); return ret; }

/* Unary plus and inverse operators */
__FP16_DECL__ half operator+(const half &h) { return h; }
__FP16_DECL__ half operator-(const half &h) { return half(0.0f) - h; }

/* Some basic comparison operations to make it look like a builtin */
__FP16_DECL__ bool operator==(const half &lh, const half &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const half &lh, const half &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const half &lh, const half &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const half &lh, const half &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const half &lh, const half &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const half &lh, const half &rh) { return (float)lh <= (float)rh; }

// overload binary operators between 'half' and build-in type. TODO: This should be handled in a better way
// int
__FP16_DECL__ float operator+(const int &lh, const half &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const int &lh, const half &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const int &lh, const half &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const int &lh, const half &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const int &lh, const half &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const int &lh, const half &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const int &lh, const half &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const int &lh, const half &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const int &lh, const half &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const int &lh, const half &rh) { return (float)lh <= (float)rh; }

__FP16_DECL__ float operator+(const half &lh, const int &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const half &lh, const int &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const half &lh, const int &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const half &lh, const int &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const half &lh, const int &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const half &lh, const int &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const half &lh, const int &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const half &lh, const int &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const half &lh, const int &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const half &lh, const int &rh) { return (float)lh <= (float)rh; }

// double
__FP16_DECL__ double operator+(const double &lh, const half &rh) { return (double)lh + (double)rh; }
__FP16_DECL__ double operator-(const double &lh, const half &rh) { return (double)lh - (double)rh; }
__FP16_DECL__ double operator*(const double &lh, const half &rh) { return (double)lh * (double)rh; }
__FP16_DECL__ double operator/(const double &lh, const half &rh) { return (double)lh / (double)rh; }
__FP16_DECL__ bool operator==(const double &lh, const half &rh) { return (double)lh == (double)rh; }
__FP16_DECL__ bool operator!=(const double &lh, const half &rh) { return (double)lh != (double)rh; }
__FP16_DECL__ bool operator> (const double &lh, const half &rh) { return (double)lh > (double)rh; }
__FP16_DECL__ bool operator< (const double &lh, const half &rh) { return (double)lh < (double)rh; }
__FP16_DECL__ bool operator>=(const double &lh, const half &rh) { return (double)lh >= (double)rh; }
__FP16_DECL__ bool operator<=(const double &lh, const half &rh) { return (double)lh <= (double)rh; }

__FP16_DECL__ double operator+(const half &lh, const double &rh) { return (double)lh + (double)rh; }
__FP16_DECL__ double operator-(const half &lh, const double &rh) { return (double)lh - (double)rh; }
__FP16_DECL__ double operator*(const half &lh, const double &rh) { return (double)lh * (double)rh; }
__FP16_DECL__ double operator/(const half &lh, const double &rh) { return (double)lh / (double)rh; }
__FP16_DECL__ bool operator==(const half &lh, const double &rh) { return (double)lh == (double)rh; }
__FP16_DECL__ bool operator!=(const half &lh, const double &rh) { return (double)lh != (double)rh; }
__FP16_DECL__ bool operator> (const half &lh, const double &rh) { return (double)lh > (double)rh; }
__FP16_DECL__ bool operator< (const half &lh, const double &rh) { return (double)lh < (double)rh; }
__FP16_DECL__ bool operator>=(const half &lh, const double &rh) { return (double)lh >= (double)rh; }
__FP16_DECL__ bool operator<=(const half &lh, const double &rh) { return (double)lh <= (double)rh; }

// float
__FP16_DECL__ float operator+(const float &lh, const half &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const float &lh, const half &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const float &lh, const half &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const float &lh, const half &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const float &lh, const half &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const float &lh, const half &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const float &lh, const half &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const float &lh, const half &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const float &lh, const half &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const float &lh, const half &rh) { return (float)lh <= (float)rh; }

__FP16_DECL__ float operator+(const half &lh, const float &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const half &lh, const float &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const half &lh, const float &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const half &lh, const float &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const half &lh, const float &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const half &lh, const float &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const half &lh, const float &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const half &lh, const float &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const half &lh, const float &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const half &lh, const float &rh) { return (float)lh <= (float)rh; }

// size_t
__FP16_DECL__ float operator+(const size_t &lh, const half &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const size_t &lh, const half &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const size_t &lh, const half &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const size_t &lh, const half &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const size_t &lh, const half &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const size_t &lh, const half &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const size_t &lh, const half &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const size_t &lh, const half &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const size_t &lh, const half &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const size_t &lh, const half &rh) { return (float)lh <= (float)rh; }

__FP16_DECL__ float operator+(const half &lh, const size_t &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const half &lh, const size_t &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const half &lh, const size_t &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const half &lh, const size_t &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const half &lh, const size_t &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const half &lh, const size_t &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const half &lh, const size_t &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const half &lh, const size_t &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const half &lh, const size_t &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const half &lh, const size_t &rh) { return (float)lh <= (float)rh; }

// LONG64(one place use this)
__FP16_DECL__ bool operator!=(const LONG64 &lh, const half &rh) { return (float)lh != (float)rh; }


// long int used by cpu matrix
__FP16_DECL__ float operator+(const long int &lh, const half &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const long int &lh, const half &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const long int &lh, const half &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const long int &lh, const half &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const long int &lh, const half &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const long int &lh, const half &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const long int &lh, const half &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const long int &lh, const half &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const long int &lh, const half &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const long int &lh, const half &rh) { return (float)lh <= (float)rh; }

__FP16_DECL__ float operator+(const half &lh, const long int &rh) { return (float)lh + (float)rh; }
__FP16_DECL__ float operator-(const half &lh, const long int &rh) { return (float)lh - (float)rh; }
__FP16_DECL__ float operator*(const half &lh, const long int &rh) { return (float)lh * (float)rh; }
__FP16_DECL__ float operator/(const half &lh, const long int &rh) { return (float)lh / (float)rh; }
__FP16_DECL__ bool operator==(const half &lh, const long int &rh) { return (float)lh == (float)rh; }
__FP16_DECL__ bool operator!=(const half &lh, const long int &rh) { return (float)lh != (float)rh; }
__FP16_DECL__ bool operator> (const half &lh, const long int &rh) { return (float)lh > (float)rh; }
__FP16_DECL__ bool operator< (const half &lh, const long int &rh) { return (float)lh < (float)rh; }
__FP16_DECL__ bool operator>=(const half &lh, const long int &rh) { return (float)lh >= (float)rh; }
__FP16_DECL__ bool operator<=(const half &lh, const long int &rh) { return (float)lh <= (float)rh; }

// half overload of some std function
namespace std
{
inline bool isfinite(half arg)
{
    return isfinite((float)arg);
}

inline bool isinf(half arg)
{
    return isinf((float)arg);
}

inline bool isnan(half arg)
{
    return isnan((float)arg);
}

inline half floor(half arg)
{
    return half(floor((float)arg));
}

inline half exp(half arg)
{
    return half(exp((float)arg));
}

inline half max(const half& lhs, const half& rhs)
{
    return half(max((float)lhs, (float)rhs));
}

inline half abs(half arg)
{
    return half(abs((float)arg));
}

}

#undef __CUDA_HOSTDEVICE__