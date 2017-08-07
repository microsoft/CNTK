//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types


#pragma once


#ifndef CPUONLY

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cub/cub.cuh>
#include <half.hpp>

/* Global-space operator functions are only available to nvcc compilation */
#if defined(__CUDACC__)
/* Arithmetic FP16 operations only supported on arch >= 5.3 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
/* Some basic arithmetic operations expected of a builtin */
__host__ __device__ __forceinline__ half operator+(const half &lh, const half &rh) { return (half)((float)lh + (float)rh); }
__host__ __device__ __forceinline__ half operator-(const half &lh, const half &rh) { return (half)((float)lh - (float)rh); }
__host__ __device__ __forceinline__ half operator*(const half &lh, const half &rh) { return (half)((float)lh * (float)rh); }
__host__ __device__ __forceinline__ half operator/(const half &lh, const half &rh) { return (half)((float)lh / (float)rh); }

__host__ __device__ __forceinline__ half &operator+=(half &lh, const half &rh) { lh = lh + rh; return lh; }
__host__ __device__ __forceinline__ half &operator-=(half &lh, const half &rh) { lh = lh - rh; return lh; }
__host__ __device__ __forceinline__ half &operator*=(half &lh, const half &rh) { lh = lh * rh; return lh; }
__host__ __device__ __forceinline__ half &operator/=(half &lh, const half &rh) { lh = lh / rh; return lh; }

/* Note for increment and decrement we use the raw value 0x3C00 equating to half(1.0f), to avoid the extra conversion */
__host__ __device__ __forceinline__ half &operator++(half &h)      { __half_raw one; one.x = 0x3C00; h += (__half)one; return h; }
__host__ __device__ __forceinline__ half &operator--(half &h)      { __half_raw one; one.x = 0x3C00; h -= (__half)one; return h; }
__host__ __device__ __forceinline__ half  operator++(half &h, int) { half ret = h; __half_raw one; one.x = 0x3C00; h += (__half)one; return ret; }
__host__ __device__ __forceinline__ half  operator--(half &h, int) { half ret = h; __half_raw one; one.x = 0x3C00; h -= (__half)one; return ret; }

/* Unary plus and inverse operators */
__host__ __device__ __forceinline__ half operator+(const half &h) { return h; }
__host__ __device__ __forceinline__ half operator-(const half &h) { half zero; zero = __float2half(0.0); return zero - h; }

/* Some basic comparison operations to make it look like a builtin */
__host__ __device__ __forceinline__ bool operator==(const half &lh, const half &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const half &lh, const half &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const half &lh, const half &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const half &lh, const half &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const half &lh, const half &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const half &lh, const half &rh) { return (float)lh <= (float)rh; }
#endif /* __CUDA_ARCH__ < 530 */
#endif /* defined(__CUDACC__) */

// overload binary operators between 'half' and build-in type. TODO: This should be handled in a better way
// int
__host__ __device__ __forceinline__ float operator+(const int &lh, const half &rh) { return (float)lh + (float)rh; }
__host__ __device__ __forceinline__ float operator-(const int &lh, const half &rh) { return (float)lh - (float)rh; }
__host__ __device__ __forceinline__ float operator*(const int &lh, const half &rh) { return (float)lh * (float)rh; }
__host__ __device__ __forceinline__ float operator/(const int &lh, const half &rh) { return (float)lh / (float)rh; }
__host__ __device__ __forceinline__ bool operator==(const int &lh, const half &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const int &lh, const half &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const int &lh, const half &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const int &lh, const half &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const int &lh, const half &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const int &lh, const half &rh) { return (float)lh <= (float)rh; }

__host__ __device__ __forceinline__ float operator+(const half &lh, const int &rh) { return (float)lh + (float)rh; }
__host__ __device__ __forceinline__ float operator-(const half &lh, const int &rh) { return (float)lh - (float)rh; }
__host__ __device__ __forceinline__ float operator*(const half &lh, const int &rh) { return (float)lh * (float)rh; }
__host__ __device__ __forceinline__ float operator/(const half &lh, const int &rh) { return (float)lh / (float)rh; }
__host__ __device__ __forceinline__ bool operator==(const half &lh, const int &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const half &lh, const int &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const half &lh, const int &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const half &lh, const int &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const half &lh, const int &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const half &lh, const int &rh) { return (float)lh <= (float)rh; }

// double
__host__ __device__ __forceinline__ double operator+(const double &lh, const half &rh) { return (double)lh + (double)rh; }
__host__ __device__ __forceinline__ double operator-(const double &lh, const half &rh) { return (double)lh - (double)rh; }
__host__ __device__ __forceinline__ double operator*(const double &lh, const half &rh) { return (double)lh * (double)rh; }
__host__ __device__ __forceinline__ double operator/(const double &lh, const half &rh) { return (double)lh / (double)rh; }
__host__ __device__ __forceinline__ bool operator==(const double &lh, const half &rh) { return (double)lh == (double)rh; }
__host__ __device__ __forceinline__ bool operator!=(const double &lh, const half &rh) { return (double)lh != (double)rh; }
__host__ __device__ __forceinline__ bool operator> (const double &lh, const half &rh) { return (double)lh > (double)rh; }
__host__ __device__ __forceinline__ bool operator< (const double &lh, const half &rh) { return (double)lh < (double)rh; }
__host__ __device__ __forceinline__ bool operator>=(const double &lh, const half &rh) { return (double)lh >= (double)rh; }
__host__ __device__ __forceinline__ bool operator<=(const double &lh, const half &rh) { return (double)lh <= (double)rh; }

__host__ __device__ __forceinline__ double operator+(const half &lh, const double &rh) { return (double)lh + (double)rh; }
__host__ __device__ __forceinline__ double operator-(const half &lh, const double &rh) { return (double)lh - (double)rh; }
__host__ __device__ __forceinline__ double operator*(const half &lh, const double &rh) { return (double)lh * (double)rh; }
__host__ __device__ __forceinline__ double operator/(const half &lh, const double &rh) { return (double)lh / (double)rh; }
__host__ __device__ __forceinline__ bool operator==(const half &lh, const double &rh) { return (double)lh == (double)rh; }
__host__ __device__ __forceinline__ bool operator!=(const half &lh, const double &rh) { return (double)lh != (double)rh; }
__host__ __device__ __forceinline__ bool operator> (const half &lh, const double &rh) { return (double)lh > (double)rh; }
__host__ __device__ __forceinline__ bool operator< (const half &lh, const double &rh) { return (double)lh < (double)rh; }
__host__ __device__ __forceinline__ bool operator>=(const half &lh, const double &rh) { return (double)lh >= (double)rh; }
__host__ __device__ __forceinline__ bool operator<=(const half &lh, const double &rh) { return (double)lh <= (double)rh; }

// float
__host__ __device__ __forceinline__ float operator+(const float &lh, const half &rh) { return (float)lh + (float)rh; }
__host__ __device__ __forceinline__ float operator-(const float &lh, const half &rh) { return (float)lh - (float)rh; }
__host__ __device__ __forceinline__ float operator*(const float &lh, const half &rh) { return (float)lh * (float)rh; }
__host__ __device__ __forceinline__ float operator/(const float &lh, const half &rh) { return (float)lh / (float)rh; }
__host__ __device__ __forceinline__ bool operator==(const float &lh, const half &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const float &lh, const half &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const float &lh, const half &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const float &lh, const half &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const float &lh, const half &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const float &lh, const half &rh) { return (float)lh <= (float)rh; }

__host__ __device__ __forceinline__ float operator+(const half &lh, const float &rh) { return (float)lh + (float)rh; }
__host__ __device__ __forceinline__ float operator-(const half &lh, const float &rh) { return (float)lh - (float)rh; }
__host__ __device__ __forceinline__ float operator*(const half &lh, const float &rh) { return (float)lh * (float)rh; }
__host__ __device__ __forceinline__ float operator/(const half &lh, const float &rh) { return (float)lh / (float)rh; }
__host__ __device__ __forceinline__ bool operator==(const half &lh, const float &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const half &lh, const float &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const half &lh, const float &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const half &lh, const float &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const half &lh, const float &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const half &lh, const float &rh) { return (float)lh <= (float)rh; }

// size_t
__host__ __device__ __forceinline__ float operator+(const size_t &lh, const half &rh) { return (float)lh + (float)rh; }
__host__ __device__ __forceinline__ float operator-(const size_t &lh, const half &rh) { return (float)lh - (float)rh; }
__host__ __device__ __forceinline__ float operator*(const size_t &lh, const half &rh) { return (float)lh * (float)rh; }
__host__ __device__ __forceinline__ float operator/(const size_t &lh, const half &rh) { return (float)lh / (float)rh; }
__host__ __device__ __forceinline__ bool operator==(const size_t &lh, const half &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const size_t &lh, const half &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const size_t &lh, const half &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const size_t &lh, const half &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const size_t &lh, const half &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const size_t &lh, const half &rh) { return (float)lh <= (float)rh; }

__host__ __device__ __forceinline__ float operator+(const half &lh, const size_t &rh) { return (float)lh + (float)rh; }
__host__ __device__ __forceinline__ float operator-(const half &lh, const size_t &rh) { return (float)lh - (float)rh; }
__host__ __device__ __forceinline__ float operator*(const half &lh, const size_t &rh) { return (float)lh * (float)rh; }
__host__ __device__ __forceinline__ float operator/(const half &lh, const size_t &rh) { return (float)lh / (float)rh; }
__host__ __device__ __forceinline__ bool operator==(const half &lh, const size_t &rh) { return (float)lh == (float)rh; }
__host__ __device__ __forceinline__ bool operator!=(const half &lh, const size_t &rh) { return (float)lh != (float)rh; }
__host__ __device__ __forceinline__ bool operator> (const half &lh, const size_t &rh) { return (float)lh > (float)rh; }
__host__ __device__ __forceinline__ bool operator< (const half &lh, const size_t &rh) { return (float)lh < (float)rh; }
__host__ __device__ __forceinline__ bool operator>=(const half &lh, const size_t &rh) { return (float)lh >= (float)rh; }
__host__ __device__ __forceinline__ bool operator<=(const half &lh, const size_t &rh) { return (float)lh <= (float)rh; }

// LONG64(one place use this)
__host__ __device__ __forceinline__ bool operator!=(const LONG64 &lh, const half &rh) { return (float)lh != (float)rh; }

// Generalize cublas calls
inline cublasStatus_t cublasgeamHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, float *alpha, float *A, int lda, float *beta, float *B, int ldb, float *C, int ldc){ return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, B, ldc); }
inline cublasStatus_t cublasgeamHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, double *alpha, double *A, int lda, double *beta, double *B, int ldb, double *C, int ldc){ return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, B, ldc); }
inline cublasStatus_t cublasgeamHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, half *alpha, half *A, int lda, half *beta, half *B, int ldb, half *C, int ldc){ RuntimeError("Unsupported template argument(half) in cublasgeamHelper"); }

inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, float *outputPtr, size_t num){ return curandGenerateUniform(generator, outputPtr, num); }
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, double *outputPtr, size_t num){ return curandGenerateUniformDouble(generator, outputPtr, num); }
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, half *outputPtr, size_t num){ RuntimeError("Unsupported template argument in GPUMat"); }

inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev){ return curandGenerateNormal(generator, outputPtr, n, mean, stddev); }
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev){ return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev); }
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, half *outputPtr, size_t n, half mean, half stddev){ RuntimeError("Unsupported template argument(half) in curandGenerateNormalHelper"); }

inline cublasStatus_t cublasasumHelper(cublasHandle_t handle, int n, const float *x, int incx, float *result){ return cublasSasum(handle, n, x, incx, result); }
inline cublasStatus_t cublasasumHelper(cublasHandle_t handle, int n, const double *x, int incx, double *result){ return cublasDasum(handle, n, x, incx, result); }
inline cublasStatus_t cublasasumHelper(cublasHandle_t handle, int n, const half *x, int incx, half *result){ RuntimeError("Unsupported template argument(half) in cublasasumHelper"); }

inline cublasStatus_t cublasIamaxHelper(cublasHandle_t handle, int n, const float *x, int incx, int *result){ return cublasIsamax(handle, n, x, incx, result); }
inline cublasStatus_t cublasIamaxHelper(cublasHandle_t handle, int n, const double *x, int incx, int *result){ return cublasIdamax(handle, n, x, incx, result); }
inline cublasStatus_t cublasIamaxHelper(cublasHandle_t handle, int n, const half *x, int incx, int *result){ RuntimeError("Unsupported template argument(half) in cublasIamaxHelper"); }

inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const float *alpha, float *x, int incx){ return cublasSscal(handle, n, alpha, x, incx); }
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const double *alpha, double *x, int incx){ return cublasDscal(handle, n, alpha, x, incx); }
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const half *alpha, half *x, int incx){ RuntimeError("Unsupported template argument(half) in cublasscalHelper"); }
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const char *alpha, char *x, int incx){ RuntimeError("Unsupported template argument(char) in cublasscalHelper"); }
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const short *alpha, short *x, int incx){ RuntimeError("Unsupported template argument(short) in cublasscalHelper"); }


inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result){ return cublasSdot(handle, n, x, incx, y, incy, result); }
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result){ return cublasDdot(handle, n, x, incx, y, incy, result); }
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const half *x, int incx, const half *y, int incy, half *result){ RuntimeError("Unsupported template argument(half) in cublasdotHelper"); }

// Generalize cuSparse calls
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float *A, int lda){ return cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda); }
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double *A, int lda){ return cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda); }
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const short *csrValA, const int *csrRowPtrA, const int *csrColIndA, short *A, int lda){ RuntimeError("Unsupported template argument(short) in GPUSparseMatrix"); }
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const char *csrValA, const int *csrRowPtrA, const int *csrColIndA, char *A, int lda){ RuntimeError("Unsupported template argument(char) in GPUSparseMatrix"); }

inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *cscValA, const int *cscRowIndA, const int *cscColPtrA, float *A, int lda){ return cusparseScsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda); }
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *cscValA, const int *cscRowIndA, const int *cscColPtrA, double *A, int lda){ return cusparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda); }
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const short *cscValA, const int *cscRowIndA, const int *cscColPtrA, short *A, int lda){ RuntimeError("Unsupported template argument(short) in GPUSparseMatrix"); }
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const char *cscValA, const int *cscRowIndA, const int *cscColPtrA, char *A, int lda){ RuntimeError("Unsupported template argument(char) in GPUSparseMatrix"); }

inline cusparseStatus_t cusparsecsr2cscHelper(cusparseHandle_t handle, int m, int n, int nnz, const float *csrVal, const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscRowInd, int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase){ return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase); }
inline cusparseStatus_t cusparsecsr2cscHelper(cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal, const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscRowInd, int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase){ return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase); }

inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr){ return cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr); }
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr){ return cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr); }
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const short *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr){ RuntimeError("Unsupported template argument(short) in GPUSparseMatrix"); }
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const char *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr){ RuntimeError("Unsupported template argument(char) in GPUSparseMatrix"); }

inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *A, int lda, const int *nnzPerRow, float *csrValA, int *csrRowPtrA, int *csrColIndA){ return cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA); }
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *A, int lda, const int *nnzPerRow, double *csrValA, int *csrRowPtrA, int *csrColIndA){ return cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA); }
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const short *A, int lda, const int *nnzPerRow, short *csrValA, int *csrRowPtrA, int *csrColIndA){ RuntimeError("Unsupported template argument(short) in GPUSparseMatrix"); }
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const char *A, int lda, const int *nnzPerRow, char *csrValA, int *csrRowPtrA, int *csrColIndA){ RuntimeError("Unsupported template argument(char) in GPUSparseMatrix"); }

inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *A, int lda, const int *nnzPerCol, float *cscValA, int *cscRowIndA, int *cscColPtrA){ return cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA); }
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *A, int lda, const int *nnzPerCol, double *cscValA, int *cscRowIndA, int *cscColPtrA){ return cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA); }
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const short *A, int lda, const int *nnzPerCol, short *cscValA, int *cscRowIndA, int *cscColPtrA){ RuntimeError("Unsupported template argument(short) in GPUSparseMatrix"); }
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const char *A, int lda, const int *nnzPerCol, char *cscValA, int *cscRowIndA, int *cscColPtrA){ RuntimeError("Unsupported template argument(char) in GPUSparseMatrix"); }

inline cusparseStatus_t cusparsecsrmmHelper(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const float *alpha, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *B, int ldb, const float *beta, float *C, int ldc){ return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc); }
inline cusparseStatus_t cusparsecsrmmHelper(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const double *alpha, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *B, int ldb, const double *beta, double *C, int ldc){ return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc); }

inline cusparseStatus_t cusparsecsrgemmHelper(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, const int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cusparseMatDescr_t descrB, const int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, float *csrValC, const int *csrRowPtrC, int *csrColIndC){ return cusparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC); }
inline cusparseStatus_t cusparsecsrgemmHelper(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, const int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cusparseMatDescr_t descrB, const int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, double *csrValC, const int *csrRowPtrC, int *csrColIndC){ return cusparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC); }

inline cusparseStatus_t cusparsecsrgeamHelper(cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA, int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *beta, const cusparseMatDescr_t descrB, int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, float *csrValC, int *csrRowPtrC, int *csrColIndC){ return cusparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC); }
inline cusparseStatus_t cusparsecsrgeamHelper(cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA, int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *beta, const cusparseMatDescr_t descrB, int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, double *csrValC, int *csrRowPtrC, int *csrColIndC){ return cusparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC); }

inline cusparseStatus_t cusparsedotiHelper(cusparseHandle_t handle, int nnz, const float *xVal, const int *xInd, const float *y, float *resultDevHostPtr, cusparseIndexBase_t idxBase){ return cusparseSdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase); }
inline cusparseStatus_t cusparsedotiHelper(cusparseHandle_t handle, int nnz, const double *xVal, const int *xInd, const double *y, double *resultDevHostPtr, cusparseIndexBase_t idxBase){ return cusparseDdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase); }


// Generalize cub calls
inline cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_keys_in, float *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream){ return cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream); }
inline cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_keys_in, double *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream){ return cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream); }
inline cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const half *d_keys_in, half *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream){ RuntimeError("Unsupported template argument(half) in SortPairsDescending"); }

#endif // CPUONLY
