//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add half2
// NV_TODO: investigate cub support for half

#pragma once


#ifndef CPUONLY

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <time.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100) // 'identifier': unreferenced formal parameter
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4458) // declaration of 'identifier' hides class member
#pragma warning(disable : 4515) // 'namespace': namespace uses itself
#pragma warning(disable : 4706) // assignment within conditional expression
#endif
#include <cub/cub.cuh>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "half.hpp"
#define TRANS_TILE_DIM 32
#define BLOCK_ROWS 8
#define COPY_TILE_DIM 1024
#define COPY_BLOCK_DIM 256

// kernel(s) for half functions with no library support
namespace {
__global__ void transposeNoOverlap(half *odata, const half *idata, const int m, const int n)
{
    __shared__ half tile[TRANS_TILE_DIM][TRANS_TILE_DIM+1];

    int x = blockIdx.x * TRANS_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANS_TILE_DIM + threadIdx.y;

    for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*m + x];

    __syncthreads();

    x = blockIdx.y * TRANS_TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TRANS_TILE_DIM + threadIdx.y;

    if(x >= n) return;

    for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS){
        if((y+j) >= m) return;
        odata[(y+j)*n + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
// set up curand state, need to move up layer to remove calling for each generate call
__global__ void setup_state(curandState *state, unsigned long long seed)
{
    curand_init(seed, 0, 0, state);
}

__global__ void GenerateUniformHalf(curandState *state, half *result, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;

    curandState localState = *state;

    float x;
    skipahead(id, &localState);
    x = curand_uniform(&localState);

    result[id] = x;
    if(id == n-1) *state = localState;
}

__global__ void GenerateNormalHalf(curandState *state, half *result, int n, half mean, half stddev)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;

    curandState localState = *state;

    float x;
    skipahead(id, &localState);
    x = curand_normal(&localState);

    result[id] = (float)mean + (float)stddev * x;
    if(id == n-1) *state = localState;
}

// kernels can convert matrix between half and float. speed currently not optimized, may need to add half2
/*
__global__ void copyHalf2Float(float *odata, const half *idata, const int n)
{
    float tmp[COPY_TILE_DIM/COPY_BLOCK_DIM];

    int x = blockIdx.x * COPY_TILE_DIM + threadIdx.x;

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        tmp[j] = (float) idata[x + j*COPY_BLOCK_DIM];

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        if(x + j*COPY_BLOCK_DIM < n) odata[x + j*COPY_BLOCK_DIM] = tmp[j];
}

__global__ void copyFloat2Half(half *odata, const float *idata, const int n)
{
    float tmp[COPY_TILE_DIM/COPY_BLOCK_DIM];

    int x = blockIdx.x * COPY_TILE_DIM + threadIdx.x;

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        tmp[j] = idata[x + j*COPY_BLOCK_DIM];

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        if(x + j*COPY_BLOCK_DIM < n) odata[x + j*COPY_BLOCK_DIM] = tmp[j];
}
*/

}

// Generalize library calls to be use in template functions

// gemm
inline cublasStatus_t cublasgemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasgemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasgemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* alpha, const half* A, int lda, const half* B, int ldb, const half* beta, half* C, int ldc)
{
    // This does true FP16 computation which is slow for non-Volta GPUs
    //return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // This does pseudo FP16 computation (input/output in fp16, computation in fp32)
    float h_a = *alpha;
    float h_b = *beta;
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &h_b, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT);
}
inline cublasStatus_t cublasgemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const int *alpha, const int *A, int lda, const int *B, int ldb, const int *beta, int *C, int ldc)
{
    RuntimeError("Unsupported template argument(int) in cublasgemmHelper");
}
inline cublasStatus_t cublasgemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const short *alpha, const short *A, int lda, const short *B, int ldb, const short *beta, short *C, int ldc)
{
    RuntimeError("Unsupported template argument(short) in cublasgemmHelper");
}
inline cublasStatus_t cublasgemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const char *alpha, const char *A, int lda, const char *B, int ldb, const char *beta, char *C, int ldc)
{
    RuntimeError("Unsupported template argument(char) in cublasgemmHelper");
}

// batched gemm
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float *Aarray[], int lda, const float *Barray[], int ldb, const float *beta, float *Carray[], int ldc, int batchCount)
{
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double *Aarray[], int lda, const double *Barray[], int ldb, const double *beta, double *Carray[], int ldc, int batchCount)
{
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* alpha, const half *Aarray[], int lda, const half *Barray[], int ldb, const half *beta, half *Carray[], int ldc, int batchCount)
{
    return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, (const __half**)Aarray, lda, (const __half**)Barray, ldb, beta, (__half**)Carray, ldc, batchCount);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const int *alpha, const int *Aarray[], int lda, const int *Barray[], int ldb, const int *beta, int *Carray[], int ldc, int batchCount)
{
    RuntimeError("Unsupported template argument(int) in cublasGemmBatchedHelper");
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const short *alpha, const short *Aarray[], int lda, const short *Barray[], int ldb, const short *beta, short *Carray[], int ldc, int batchCount)
{
    RuntimeError("Unsupported template argument(short) in cublasGemmBatchedHelper");
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const char *alpha, const char *Aarray[], int lda, const char *Barray[], int ldb, const char *beta, char *Carray[], int ldc, int batchCount)
{
    RuntimeError("Unsupported template argument(char) in cublasGemmBatchedHelper");
}

// axpy
inline cublasStatus_t cublasaxpyHelper(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasaxpyHelper(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasaxpyHelper(cublasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy)
{
    float tmp_alpha = *alpha;
    return cublasAxpyEx(handle, n, (void*)&tmp_alpha, CUDA_R_32F, (void*)x, CUDA_R_16F, incx, (void*)y, CUDA_R_16F, incy, CUDA_R_32F);
}
inline cublasStatus_t cublasaxpyHelper(cublasHandle_t handle, int n, const int *alpha, const int *x, int incx, int *y, int incy)
{
    RuntimeError("Unsupported template argument(int) in cublasaxpyHelper");
}
inline cublasStatus_t cublasaxpyHelper(cublasHandle_t handle, int n, const short *alpha, const short *x, int incx, short *y, int incy)
{
    RuntimeError("Unsupported template argument(short) in cublasaxpyHelper");
}
inline cublasStatus_t cublasaxpyHelper(cublasHandle_t handle, int n, const char *alpha, const char *x, int incx, char *y, int incy)
{
    RuntimeError("Unsupported template argument(char) in cublasaxpyHelper");
}

// transpose using geam
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, float *alpha, float *A, int lda, float *beta, float *B, int ldb, float *C, int ldc)
{
    return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, double *alpha, double *A, int lda, double *beta, double *B, int ldb, double *C, int ldc)
{
    return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, half *, half *A, int, half *, half *, int, half *C, int)
{
    if(C != A)
    {
        dim3 dimGrid((n+TRANS_TILE_DIM-1)/TRANS_TILE_DIM, (m+TRANS_TILE_DIM-1)/TRANS_TILE_DIM, 1);
        dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

        transposeNoOverlap<<<dimGrid, dimBlock>>>(C, A, n, m);
    }
    else
        RuntimeError("In place transpose(half) not supported."); // cublas do not support this either. There might be bug if this actually get called.
    return (cublasStatus_t) 0;
}
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, int *, int *A, int, int *, int *, int, int *C, int)
{
    RuntimeError("Unsupported template argument(int) in cublasTransposeHelper");
}
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, short *, short *A, int, short *, short *, int, short *C, int)
{
    RuntimeError("Unsupported template argument(short) in cublasTransposeHelper");
}
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, char *, char *A, int, char *, char *, int, char *C, int)
{
    RuntimeError("Unsupported template argument(char) in cublasTransposeHelper");
}

// asum
inline cublasStatus_t cublasasumHelper(cublasHandle_t handle, int n, const float *x, int incx, float *result)
{
    return cublasSasum(handle, n, x, incx, result);
}
inline cublasStatus_t cublasasumHelper(cublasHandle_t handle, int n, const double *x, int incx, double *result)
{
    return cublasDasum(handle, n, x, incx, result);
}
inline cublasStatus_t cublasasumHelper(cublasHandle_t, int n, const half *x, int incx, half *result)
{
    // pass in cudnn handle/descriptor to remove overhead?
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnReduceTensorDescriptor_t reduceTensorDesc;

    cudnnCreate(&cudnnHandle);
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&dstTensorDesc);
    cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);

    cudnnSetTensor4dDescriptorEx(srcTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
    cudnnSetTensor4dDescriptorEx(dstTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
    cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                   CUDNN_REDUCE_TENSOR_NORM1,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_NOT_PROPAGATE_NAN,
                                   CUDNN_REDUCE_TENSOR_NO_INDICES,
                                   CUDNN_32BIT_INDICES);

    void *workspace = NULL;
    size_t workspaceSizeInBytes = 0;
    cudnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
    if(workspaceSizeInBytes > 0) cudaMalloc(&workspace, workspaceSizeInBytes);

    float alpha = 1.0f;
    float beta = 0.0f;

    void *d_res;
    cudaMalloc(&d_res, sizeof(half));

    cudnnReduceTensor(cudnnHandle,
                      reduceTensorDesc,
                      NULL,
                      0,
                      workspace,
                      workspaceSizeInBytes,
                      &alpha,
                      srcTensorDesc,
                      (void*)x,
                      &beta,
                      dstTensorDesc,
                      d_res);

    cudaMemcpy((void *)result, d_res, sizeof(half), cudaMemcpyDeviceToHost);

    cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    cudnnDestroyTensorDescriptor(srcTensorDesc);
    cudnnDestroyTensorDescriptor(dstTensorDesc);
    cudnnDestroy(cudnnHandle);
    cudaFree(d_res);
    cudaFree(workspace);

    return (cublasStatus_t) 0;
}
inline cublasStatus_t cublasasumHelper(cublasHandle_t, int n, const int *x, int incx, int *result)
{
    RuntimeError("Unsupported template argument(int) in cublasasumHelper");
}
inline cublasStatus_t cublasasumHelper(cublasHandle_t, int n, const short *x, int incx, short *result)
{
    RuntimeError("Unsupported template argument(short) in cublasasumHelper");
}
inline cublasStatus_t cublasasumHelper(cublasHandle_t, int n, const char *x, int incx, char *result)
{
    RuntimeError("Unsupported template argument(char) in cublasasumHelper");
}


// amax
inline cublasStatus_t cublasamaxHelper(cublasHandle_t handle, int n, const float *x, int incx, int *result)
{
    return cublasIsamax(handle, n, x, incx, result);
}
inline cublasStatus_t cublasamaxHelper(cublasHandle_t handle, int n, const double *x, int incx, int *result)
{
    return cublasIdamax(handle, n, x, incx, result);
}
inline cublasStatus_t cublasamaxHelper(cublasHandle_t, int n, const half *x, int incx, int *result)
{
    unsigned int h_result_uint = 0;
    // pass in cudnn handle/descriptor to remove overhead?
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnReduceTensorDescriptor_t reduceTensorDesc;

    cudnnCreate(&cudnnHandle);
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&dstTensorDesc);
    cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);

    cudnnSetTensor4dDescriptorEx(srcTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
    cudnnSetTensor4dDescriptorEx(dstTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
    cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                   CUDNN_REDUCE_TENSOR_AMAX,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_NOT_PROPAGATE_NAN,
                                   CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
                                   CUDNN_32BIT_INDICES);

    void *workspace = NULL;
    size_t workspaceSizeInBytes = 0;
    cudnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
    if(workspaceSizeInBytes > 0) cudaMalloc(&workspace, workspaceSizeInBytes);

    float alpha = 1.0f;
    float beta = 0.0f;
    void *d_max;
    cudaMalloc(&d_max, sizeof(half));
    void *d_result_uint;
    cudaMalloc(&d_result_uint, sizeof(unsigned int));

    cudnnReduceTensor(cudnnHandle,
                      reduceTensorDesc,
                      d_result_uint,
                      sizeof(unsigned int),
                      workspace,
                      workspaceSizeInBytes,
                      &alpha,
                      srcTensorDesc,
                      (void*)x,
                      &beta,
                      dstTensorDesc,
                      d_max);

    cudaMemcpy(&h_result_uint, d_result_uint, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    cudnnDestroyTensorDescriptor(srcTensorDesc);
    cudnnDestroyTensorDescriptor(dstTensorDesc);
    cudnnDestroy(cudnnHandle);
    cudaFree(workspace);
    cudaFree(d_max);
    cudaFree(d_result_uint);

    *result = (int) h_result_uint;
    return (cublasStatus_t) 0;
}
inline cublasStatus_t cublasamaxHelper(cublasHandle_t, int n, const int *x, int incx, int *result)
{
    RuntimeError("Unsupported template argument(int) in cublasasumHelper");
}
inline cublasStatus_t cublasamaxHelper(cublasHandle_t, int n, const short *x, int incx, int *result)
{
    RuntimeError("Unsupported template argument(short) in cublasasumHelper");
}
inline cublasStatus_t cublasamaxHelper(cublasHandle_t, int n, const char *x, int incx, int *result)
{
    RuntimeError("Unsupported template argument(char) in cublasasumHelper");
}

// scal
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const float *alpha, float *x, int incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const double *alpha, double *x, int incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t cublasscalHelper(cublasHandle_t handle, int n, const half *alpha, half *x, int incx)
{
    float tmp_alpha = *alpha;
    return cublasScalEx(handle, n, (void*)&tmp_alpha, CUDA_R_32F, (void*)x, CUDA_R_16F, incx, CUDA_R_32F);
}
inline cublasStatus_t cublasscalHelper(cublasHandle_t, int, const int *, int *, int)
{
    RuntimeError("Unsupported template argument(int) in cublas_scal");
}
inline cublasStatus_t cublasscalHelper(cublasHandle_t, int, const short *, short *, int)
{
    RuntimeError("Unsupported template argument(short) in cublas_scal");
}
inline cublasStatus_t cublasscalHelper(cublasHandle_t, int, const char *, char *, int)
{
    RuntimeError("Unsupported template argument(char) in cublas_scal");
}

// dot
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const half *x, int incx, const half *y, int incy, half *result)
{
    return cublasDotEx(handle, n, (void*)x, CUDA_R_16F, incx, (void*)y, CUDA_R_16F, incy, (void*)result, CUDA_R_16F, CUDA_R_32F);
}
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const int *x, int incx, const int *y, int incy, int *result)
{
    RuntimeError("Unsupported template argument(int) in cublasdotHelper");
}
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const short *x, int incx, const short *y, int incy, short *result)
{
    RuntimeError("Unsupported template argument(short) in cublasdotHelper");
}
inline cublasStatus_t cublasdotHelper(cublasHandle_t handle, int n, const char *x, int incx, const char *y, int incy, char *result)
{
    RuntimeError("Unsupported template argument(char) in cublasdotHelper");
}

// curand
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, float *outputPtr, size_t num)
{
    return curandGenerateUniform(generator, outputPtr, num);
}
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, double *outputPtr, size_t num)
{
    return curandGenerateUniformDouble(generator, outputPtr, num);
}
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t, half *outputPtr, size_t num)
{
    curandState *devStates;
    cudaMalloc((void **)&devStates, sizeof(curandState));
    setup_state<<<1,1>>>(devStates, time(NULL)); // What does curandGenerateUniform actually doing? should also pass in state here

    dim3 dimGrid((unsigned int)(num+COPY_BLOCK_DIM-1)/COPY_BLOCK_DIM, 1, 1);
    dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
    GenerateUniformHalf<<<dimGrid, dimBlock>>>(devStates, outputPtr, (int)num);

    return (curandStatus_t) 0;
}
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t, int *, size_t)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t, short *, size_t)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t, char *, size_t)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
{
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
{
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t, half *outputPtr, size_t n, half mean, half stddev)
{
    curandState *devStates;
    cudaMalloc((void **)&devStates, sizeof(curandState));
    setup_state<<<1,1>>>(devStates, time(NULL)); // What does curandGenerateUniform actually doing? should also pass in state here

    dim3 dimGrid((unsigned int)(n+COPY_BLOCK_DIM-1)/COPY_BLOCK_DIM, 1, 1);
    dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
    GenerateNormalHalf<<<dimGrid, dimBlock>>>(devStates, outputPtr, (int)n, mean, stddev);

    return (curandStatus_t) 0;
}
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t, int *, size_t, int, int)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t, short *, size_t, short, short)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t, char *, size_t, char, char)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

#pragma warning(push) 
#pragma warning(disable : 4996) // Deprecated methods cusparse<T>csr2dense
// cusparse
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float *A, int lda)
{
    return cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double *A, int lda)
{
    return cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const half *, const int *, const int *, half *, int)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const int *, const int *, const int *, int *, int)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const short *, const int *, const int *, short *, int)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsecsr2denseHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const char *, const int *, const int *, char *, int)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *cscValA, const int *cscRowIndA, const int *cscColPtrA, float *A, int lda)
{
    return cusparseScsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
}
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *cscValA, const int *cscRowIndA, const int *cscColPtrA, double *A, int lda)
{
    return cusparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
}
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const half *, const int *, const int *, half *, int)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const int *, const int *, const int *, int *, int)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const short *, const int *, const int *, short *, int)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsecsc2denseHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const char *, const int *, const int *, char *, int)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

// 2020.12.10 - mj.jo
// cuda 10.0
//inline cusparseStatus_t cusparsecsr2cscHelper(cusparseHandle_t handle, int m, int n, int nnz, const float *csrVal, const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscRowInd, int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
//{
//    return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
//}
//inline cusparseStatus_t cusparsecsr2cscHelper(cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal, const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscRowInd, int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
//{
//    return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
//}
//inline cusparseStatus_t cusparsecsr2cscHelper(cusparseHandle_t, int, int, int, const half *, const int *, const int *, half *, int *, int *, cusparseAction_t, cusparseIndexBase_t)
//{
//    RuntimeError("Unsupported template argument(half) in cusparsecsr2cscHelper");
//}

// 2020.12.10 - mj.jo
// cuda 11.1
inline cusparseStatus_t cusparseCsr2cscEx2_bufferSizeHelper(cusparseHandle_t handle, int m, int n, int nnz, const float *csrVal, const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscColPtr, int *cscRowInd, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, size_t *bufferSize)
{
    return cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, CUDA_R_32F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, bufferSize);
}
inline cusparseStatus_t cusparseCsr2cscEx2_bufferSizeHelper(cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal, const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscColPtr, int *cscRowInd, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, size_t *bufferSize)
{
    return cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, CUDA_R_64F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, bufferSize);
}
inline cusparseStatus_t cusparseCsr2cscEx2_bufferSizeHelper(cusparseHandle_t, int, int, int, const half *, const int *, const int *, half *, int *, int *, cusparseAction_t, cusparseIndexBase_t, size_t *)
{
    RuntimeError("Unsupported template argument(half) in cusparseCsr2cscEx2_bufferSizeHelper");
}
inline cusparseStatus_t cusparseCsr2cscEx2_bufferSizeHelper(cusparseHandle_t, int, int, int, const int *, const int *, const int *, int *, int *, int *, cusparseAction_t, cusparseIndexBase_t, size_t *)
{
    RuntimeError("Unsupported template argument(int) in cusparseCsr2cscEx2_bufferSizeHelper");
}
inline cusparseStatus_t cusparseCsr2cscEx2_bufferSizeHelper(cusparseHandle_t, int, int, int, const short *, const int *, const int *, short *, int *, int *, cusparseAction_t, cusparseIndexBase_t, size_t *)
{
    RuntimeError("Unsupported template argument(short) in cusparseCsr2cscEx2_bufferSizeHelper");
}
inline cusparseStatus_t cusparseCsr2cscEx2_bufferSizeHelper(cusparseHandle_t, int, int, int, const char *, const int *, const int *, char *, int *, int *, cusparseAction_t, cusparseIndexBase_t, size_t *)
{
    RuntimeError("Unsupported template argument(char) in cusparseCsr2cscEx2_bufferSizeHelper");
}
inline cusparseStatus_t cusparseCsr2cscEx2Helper(cusparseHandle_t handle, int m, int n, int nnz, const float *csrVal, const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscColPtr, int *cscRowInd, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *buffer)
{
    return cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, CUDA_R_32F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, buffer);
}
inline cusparseStatus_t cusparseCsr2cscEx2Helper(cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal, const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscColPtr, int *cscRowInd, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *buffer)
{
    return cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, CUDA_R_64F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, buffer);
}
inline cusparseStatus_t cusparseCsr2cscEx2Helper(cusparseHandle_t, int, int, int, const half *, const int *, const int *, half *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *)
{
    RuntimeError("Unsupported template argument(half) in cusparseCsr2cscEx2Helper");
}
inline cusparseStatus_t cusparseCsr2cscEx2Helper(cusparseHandle_t, int, int, int, const int *, const int *, const int *, int *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *)
{
    RuntimeError("Unsupported template argument(int) in cusparseCsr2cscEx2Helper");
}
inline cusparseStatus_t cusparseCsr2cscEx2Helper(cusparseHandle_t, int, int, int, const short *, const int *, const int *, short *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *)
{
    RuntimeError("Unsupported template argument(short) in cusparseCsr2cscEx2Helper");
}
inline cusparseStatus_t cusparseCsr2cscEx2Helper(cusparseHandle_t, int, int, int, const char *, const int *, const int *, char *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *)
{
    RuntimeError("Unsupported template argument(char) in cusparseCsr2cscEx2Helper");
}
#pragma warning(pop)

inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)
{
    return cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
}
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)
{
    return cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
}
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t,cusparseDirection_t,int,int , const cusparseMatDescr_t, const half *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const int *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const short *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsennzHelper(cusparseHandle_t,cusparseDirection_t,int,int , const cusparseMatDescr_t, const char *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

#pragma warning(push)
#pragma warning(disable : 4996) // Deprecated methods cusparse<T>csr2dense
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *A, int lda, const int *nnzPerRow, float *csrValA, int *csrRowPtrA, int *csrColIndA)
{
    return cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
}
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *A, int lda, const int *nnzPerRow, double *csrValA, int *csrRowPtrA, int *csrColIndA)
{
    return cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
}
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const half *, int, const int *, half *, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const int *, int, const int *, int *, int *, int *)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const short *, int, const int *, short *, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsedense2csrHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const char *, int, const int *, char *, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *A, int lda, const int *nnzPerCol, float *cscValA, int *cscRowIndA, int *cscColPtrA)
{
    return cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
}
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *A, int lda, const int *nnzPerCol, double *cscValA, int *cscRowIndA, int *cscColPtrA)
{
    return cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
}
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const half *, int, const int *, half *, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const int *, int, const int *, int *, int *, int *)
{
    RuntimeError("Unsupported template argument(int) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t, int, int, const cusparseMatDescr_t, const short *, int, const int *, short *, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline cusparseStatus_t cusparsedense2cscHelper(cusparseHandle_t,int,int,const cusparseMatDescr_t, const char *, int, const int *, char *, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}
#pragma warning(pop)

// 2020.12.10 - mj.jo
// cuda 10.0
//inline cusparseStatus_t cusparsecsrmmHelper(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const float *alpha, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *B, int ldb, const float *beta, float *C, int ldc)
//{
//    return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
//}
//inline cusparseStatus_t cusparsecsrmmHelper(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const double *alpha, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *B, int ldb, const double *beta, double *C, int ldc)
//{
//    return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
//}
//inline cusparseStatus_t cusparsecsrmmHelper(cusparseHandle_t, cusparseOperation_t, int, int, int, int, const half *, const cusparseMatDescr_t, const half *, const int *, const int *, const half *, int, const half *, half *, int)
//{
//    RuntimeError("Unsupported template argument(half) in cusparsecsrmmHelper");
//}

// 2020.12.10 - mj.jo
// cuda 11.1
inline cusparseStatus_t cusparseCreateCsrHelper(cusparseSpMatDescr_t *spMatDescr, int m, int k, int nnz, cusparseIndexBase_t idxBase, float *csrVal, int *csrRowPtr, int *csrColInd)
{
    return cusparseCreateCsr(spMatDescr, m, k, nnz, csrRowPtr, csrColInd, csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, idxBase, CUDA_R_32F);
}
inline cusparseStatus_t cusparseCreateCsrHelper(cusparseSpMatDescr_t *spMatDescr, int m, int k, int nnz, cusparseIndexBase_t idxBase, double *csrVal, int *csrRowPtr, int *csrColInd)
{
    return cusparseCreateCsr(spMatDescr, m, k, nnz, csrRowPtr, csrColInd, csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, idxBase, CUDA_R_64F);
}
inline cusparseStatus_t cusparseCreateCsrHelper(cusparseSpMatDescr_t *, int, int, int, cusparseIndexBase_t, half *, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in cusparseCreateCsrHelper");
}
inline cusparseStatus_t cusparseCreateCsrHelper(cusparseSpMatDescr_t *, int, int, int, cusparseIndexBase_t, int *, int *, int *)
{
    RuntimeError("Unsupported template argument(int) in cusparseCreateCsrHelper");
}
inline cusparseStatus_t cusparseCreateCsrHelper(cusparseSpMatDescr_t *, int, int, int, cusparseIndexBase_t, short *, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in cusparseCreateCsrHelper");
}
inline cusparseStatus_t cusparseCreateCsrHelper(cusparseSpMatDescr_t *, int, int, int, cusparseIndexBase_t, char *, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in cusparseCreateCsrHelper");
}
inline cusparseStatus_t cusparseCreateDnMatHelper(cusparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t Id, float *values, cusparseOrder_t order)
{
    return cusparseCreateDnMat(dnMatDescr, rows, cols, Id, values, CUDA_R_32F, order);
}
inline cusparseStatus_t cusparseCreateDnMatHelper(cusparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t Id, double *values, cusparseOrder_t order)
{
    return cusparseCreateDnMat(dnMatDescr, rows, cols, Id, values, CUDA_R_64F, order);
}
inline cusparseStatus_t cusparseCreateDnMatHelper(cusparseDnMatDescr_t *, int64_t, int64_t, int64_t, half *, cusparseOrder_t)
{
    RuntimeError("Unsupported template argument(half) in cusparseCreateDnMatHelper");
}
inline cusparseStatus_t cusparseCreateDnMatHelper(cusparseDnMatDescr_t *, int64_t, int64_t, int64_t, int *, cusparseOrder_t)
{
    RuntimeError("Unsupported template argument(int) in cusparseCreateDnMatHelper");
}
inline cusparseStatus_t cusparseCreateDnMatHelper(cusparseDnMatDescr_t *, int64_t, int64_t, int64_t, short *, cusparseOrder_t)
{
    RuntimeError("Unsupported template argument(short) in cusparseCreateDnMatHelper");
}
inline cusparseStatus_t cusparseCreateDnMatHelper(cusparseDnMatDescr_t *, int64_t, int64_t, int64_t, char *, cusparseOrder_t)
{
    RuntimeError("Unsupported template argument(char) in cusparseCreateDnMatHelper");
}
inline cusparseStatus_t cusparseSpMM_bufferSizeHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const float *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const float *beta, cusparseDnMatDescr_t matC, size_t *bufferSize)
{
    return cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, bufferSize);
}
inline cusparseStatus_t cusparseSpMM_bufferSizeHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const double *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const double *beta, cusparseDnMatDescr_t matC, size_t *bufferSize)
{
    return cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, bufferSize);
}
inline cusparseStatus_t cusparseSpMM_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const half *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const half *, cusparseDnMatDescr_t, size_t *)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpMM_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpMM_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const int *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const int *, cusparseDnMatDescr_t, size_t *)
{
    RuntimeError("Unsupported template argument(int) in cusparseSpMM_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpMM_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const short *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const short *, cusparseDnMatDescr_t, size_t *)
{
    RuntimeError("Unsupported template argument(short) in cusparseSpMM_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpMM_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const char *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const char *, cusparseDnMatDescr_t, size_t *)
{
    RuntimeError("Unsupported template argument(char) in cusparseSpMM_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpMMHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const float *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const float *beta, cusparseDnMatDescr_t matC, void *externalBuffer)
{
    return cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, externalBuffer);
}
inline cusparseStatus_t cusparseSpMMHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const double *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const double *beta, cusparseDnMatDescr_t matC, void *externalBuffer)
{
    return cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, externalBuffer);
}
inline cusparseStatus_t cusparseSpMMHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const half *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const half *, cusparseDnMatDescr_t, void *)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpMMHelper");
}
inline cusparseStatus_t cusparseSpMMHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const int *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const int *, cusparseDnMatDescr_t, void *)
{
    RuntimeError("Unsupported template argument(int) in cusparseSpMMHelper");
}
inline cusparseStatus_t cusparseSpMMHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const short *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const short *, cusparseDnMatDescr_t, void *)
{
    RuntimeError("Unsupported template argument(short) in cusparseSpMMHelper");
}
inline cusparseStatus_t cusparseSpMMHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const char *, cusparseSpMatDescr_t, cusparseDnMatDescr_t, const char *, cusparseDnMatDescr_t, void *)
{
    RuntimeError("Unsupported template argument(char) in cusparseSpMMHelper");
}

// 2020.12.11 - mj.jo
// cuda 10.0
//inline cusparseStatus_t cusparsecsrgemmHelper(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, const int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cusparseMatDescr_t descrB, const int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, float *csrValC, const int *csrRowPtrC, int *csrColIndC)
//{
//    return cusparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
//}
//inline cusparseStatus_t cusparsecsrgemmHelper(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, const int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cusparseMatDescr_t descrB, const int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, double *csrValC, const int *csrRowPtrC, int *csrColIndC)
//{
//    return cusparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
//}
//inline cusparseStatus_t cusparsecsrgemmHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, int, int, int, const cusparseMatDescr_t, const int, const half *, const int *, const int *, const cusparseMatDescr_t, const int, const half *, const int *, const int *, const cusparseMatDescr_t, half *, const int *, int *)
//{
//    RuntimeError("Unsupported template argument(half) in cusparsecsrgemmHelper");
//}

// 2020.12.11 - mj.jo
// cuda 11.1
inline cusparseStatus_t cusparseSpGEMM_workEstimationHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const float *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const float *beta, cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize1, void *externalBuffer1)
{
    return cusparseSpGEMM_workEstimation(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, bufferSize1, externalBuffer1);
}
inline cusparseStatus_t cusparseSpGEMM_workEstimationHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const double *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const double *beta, cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize1, void *externalBuffer1)
{
    return cusparseSpGEMM_workEstimation(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, bufferSize1, externalBuffer1);
}
inline cusparseStatus_t cusparseSpGEMM_workEstimationHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const half *, cusparseSpMatDescr_t, cusparseSpMatDescr_t, const half *, cusparseSpMatDescr_t, cusparseSpGEMMDescr_t, size_t *, void *)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpGEMM_workEstimationHelper");
}
inline cusparseStatus_t cusparseSpGEMM_computeHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const float *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const float *beta, cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2, void *externalBuffer2)
{
    return cusparseSpGEMM_compute(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, bufferSize2, externalBuffer2);
}
inline cusparseStatus_t cusparseSpGEMM_computeHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const double *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const double *beta, cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2, void *externalBuffer2)
{
    return cusparseSpGEMM_compute(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, bufferSize2, externalBuffer2);
}
inline cusparseStatus_t cusparseSpGEMM_computeHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const half *, cusparseSpMatDescr_t, cusparseSpMatDescr_t, const half *, cusparseSpMatDescr_t, cusparseSpGEMMDescr_t, size_t *, void *)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpGEMM_computeHelper");
}
inline cusparseStatus_t cusparseSpGEMM_copyHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const float *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const float *beta, cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDescr)
{
    return cusparseSpGEMM_copy(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);
}
inline cusparseStatus_t cusparseSpGEMM_copyHelper(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const double *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const double *beta, cusparseSpMatDescr_t matC, cusparseSpGEMMDescr_t spgemmDescr)
{
    return cusparseSpGEMM_copy(handle, opA, opB, alpha, matA, matB, beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);
}
inline cusparseStatus_t cusparseSpGEMM_copyHelper(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const half *, cusparseSpMatDescr_t, cusparseSpMatDescr_t, const half *, cusparseSpMatDescr_t, cusparseSpGEMMDescr_t)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpGEMM_copyHelper");
}

// 2020.12.11 - mj.jo
// cuda 10.0
//inline cusparseStatus_t cusparsecsrgeamHelper(cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA, int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *beta, const cusparseMatDescr_t descrB, int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, float *csrValC, int *csrRowPtrC, int *csrColIndC)
//{
//    return cusparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
//}
//inline cusparseStatus_t cusparsecsrgeamHelper(cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA, int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *beta, const cusparseMatDescr_t descrB, int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB, const cusparseMatDescr_t descrC, double *csrValC, int *csrRowPtrC, int *csrColIndC)
//{
//    return cusparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
//}
//inline cusparseStatus_t cusparsecsrgeamHelper(cusparseHandle_t, int, int, const half *, const cusparseMatDescr_t, int, const half *, const int *, const int *, const half *, const cusparseMatDescr_t, int, const half *, const int *, const int *, const cusparseMatDescr_t, half *, int *, int *)
//{
//    RuntimeError("Unsupported template argument(half) in cusparsecsrgeamHelper");
//}

// 2020.12.11 - mj.jo
// cuda 11.1
inline cusparseStatus_t cusparsecsrgeam2_bufferSizeExtHelper(cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta, const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    return cusparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}
inline cusparseStatus_t cusparsecsrgeam2_bufferSizeExtHelper(cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *beta, const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    return cusparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}
inline cusparseStatus_t cusparsecsrgeam2_bufferSizeExtHelper(cusparseHandle_t, int, int, const half *, const cusparseMatDescr_t, int, const half *, const int *, const int *, const half *, const cusparseMatDescr_t, int, const half *, const int *, const int *, const cusparseMatDescr_t, const half *, const int *, const int *, size_t *)
{
    RuntimeError("Unsupported template argument(half) in cusparsecsrgeam2_bufferSizeExtHelper");
}
inline cusparseStatus_t cusparsecsrgeam2_bufferSizeExtHelper(cusparseHandle_t, int, int, const int *, const cusparseMatDescr_t, int, const int *, const int *, const int *, const int *, const cusparseMatDescr_t, int, const int *, const int *, const int *, const cusparseMatDescr_t, const int *, const int *, const int *, size_t *)
{
    RuntimeError("Unsupported template argument(int) in cusparsecsrgeam2_bufferSizeExtHelper");
}
inline cusparseStatus_t cusparsecsrgeam2_bufferSizeExtHelper(cusparseHandle_t, int, int, const short *, const cusparseMatDescr_t, int, const short *, const int *, const int *, const short *, const cusparseMatDescr_t, int, const short *, const int *, const int *, const cusparseMatDescr_t, const short *, const int *, const int *, size_t *)
{
    RuntimeError("Unsupported template argument(short) in cusparsecsrgeam2_bufferSizeExtHelper");
}
inline cusparseStatus_t cusparsecsrgeam2_bufferSizeExtHelper(cusparseHandle_t, int, int, const char *, const cusparseMatDescr_t, int, const char *, const int *, const int *, const char *, const cusparseMatDescr_t, int, const char *, const int *, const int *, const cusparseMatDescr_t, const char *, const int *, const int *, size_t *)
{
    RuntimeError("Unsupported template argument(char) in cusparsecsrgeam2_bufferSizeExtHelper");
}
inline cusparseStatus_t cusparsecsrgeam2Helper(cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta, const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC, float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    return cusparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}
inline cusparseStatus_t cusparsecsrgeam2Helper(cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *beta, const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC, double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    return cusparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}
inline cusparseStatus_t cusparsecsrgeam2Helper(cusparseHandle_t, int, int, const half *, const cusparseMatDescr_t, int, const half *, const int *, const int *, const half *, const cusparseMatDescr_t, int, const half *, const int *, const int *, const cusparseMatDescr_t, half *, int *, int *, void *)
{
    RuntimeError("Unsupported template argument(half) in cusparsecsrgeam2Helper");
}
inline cusparseStatus_t cusparsecsrgeam2Helper(cusparseHandle_t, int, int, const int *, const cusparseMatDescr_t, int, const int *, const int *, const int *, const int *, const cusparseMatDescr_t, int, const int *, const int *, const int *, const cusparseMatDescr_t, int *, int *, int *, void *)
{
    RuntimeError("Unsupported template argument(int) in cusparsecsrgeam2Helper");
}
inline cusparseStatus_t cusparsecsrgeam2Helper(cusparseHandle_t, int, int, const short *, const cusparseMatDescr_t, int, const short *, const int *, const int *, const short *, const cusparseMatDescr_t, int, const short *, const int *, const int *, const cusparseMatDescr_t, short *, int *, int *, void *)
{
    RuntimeError("Unsupported template argument(short) in cusparsecsrgeam2Helper");
}
inline cusparseStatus_t cusparsecsrgeam2Helper(cusparseHandle_t, int, int, const char *, const cusparseMatDescr_t, int, const char *, const int *, const int *, const char *, const cusparseMatDescr_t, int, const char *, const int *, const int *, const cusparseMatDescr_t, char *, int *, int *, void *)
{
    RuntimeError("Unsupported template argument(char) in cusparsecsrgeam2Helper");
}

// 2020.12.14 - mj.jo
// cuda 10.0
//inline cusparseStatus_t cusparsedotiHelper(cusparseHandle_t handle, int nnz, const float *xVal, const int *xInd, const float *y, float *resultDevHostPtr, cusparseIndexBase_t idxBase)
//{
//    return cusparseSdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase);
//}
//inline cusparseStatus_t cusparsedotiHelper(cusparseHandle_t handle, int nnz, const double *xVal, const int *xInd, const double *y, double *resultDevHostPtr, cusparseIndexBase_t idxBase)
//{
//    return cusparseDdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase);
//}
//inline cusparseStatus_t cusparsedotiHelper(cusparseHandle_t, int, const half *, const int *, const half *, half *, cusparseIndexBase_t)
//{
//    RuntimeError("Unsupported template argument(half) in cusparsedotiHelper");
//}

// 2020.12.14 - mj.jo
// cuda 11.1
inline cusparseStatus_t cusparseCreateSpVecHelper(cusparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz, int *indices, float *values, cusparseIndexBase_t idxBase)
{
    return cusparseCreateSpVec(spVecDescr, size, nnz, indices, values, CUSPARSE_INDEX_64I, idxBase, CUDA_R_32F);
}
inline cusparseStatus_t cusparseCreateSpVecHelper(cusparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz, int *indices, double *values, cusparseIndexBase_t idxBase)
{
    return cusparseCreateSpVec(spVecDescr, size, nnz, indices, values, CUSPARSE_INDEX_64I, idxBase, CUDA_R_64F);
}
inline cusparseStatus_t cusparseCreateSpVecHelper(cusparseSpVecDescr_t *, int64_t, int64_t, int *, half *, cusparseIndexBase_t)
{
    RuntimeError("Unsupported template argument(half) in cusparseCreateSpVecHelper");
}
inline cusparseStatus_t cusparseCreateSpVecHelper(cusparseSpVecDescr_t *, int64_t, int64_t, int *, int *, cusparseIndexBase_t)
{
    RuntimeError("Unsupported template argument(int) in cusparseCreateSpVecHelper");
}
inline cusparseStatus_t cusparseCreateSpVecHelper(cusparseSpVecDescr_t *, int64_t, int64_t, int *, short *, cusparseIndexBase_t)
{
    RuntimeError("Unsupported template argument(short) in cusparseCreateSpVecHelper");
}
inline cusparseStatus_t cusparseCreateSpVecHelper(cusparseSpVecDescr_t *, int64_t, int64_t, int *, char *, cusparseIndexBase_t)
{
    RuntimeError("Unsupported template argument(char) in cusparseCreateSpVecHelper");
}
inline cusparseStatus_t cusparseCreateDnVecHelper(cusparseDnVecDescr_t *dnVecDescr, int64_t size, float *values)
{
    return cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_32F);
}
inline cusparseStatus_t cusparseCreateDnVecHelper(cusparseDnVecDescr_t *dnVecDescr, int64_t size, double *values)
{
    return cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_64F);
}
inline cusparseStatus_t cusparseCreateDnVecHelper(cusparseDnVecDescr_t *, int64_t, half *)
{
    RuntimeError("Unsupported template argument(half) in cusparseCreateDnVecHelper");
}
inline cusparseStatus_t cusparseCreateDnVecHelper(cusparseDnVecDescr_t *, int64_t, int *)
{
    RuntimeError("Unsupported template argument(int) in cusparseCreateDnVecHelper");
}
inline cusparseStatus_t cusparseCreateDnVecHelper(cusparseDnVecDescr_t *, int64_t, short *)
{
    RuntimeError("Unsupported template argument(short) in cusparseCreateDnVecHelper");
}
inline cusparseStatus_t cusparseCreateDnVecHelper(cusparseDnVecDescr_t *, int64_t, char *)
{
    RuntimeError("Unsupported template argument(char) in cusparseCreateDnVecHelper");
}
inline cusparseStatus_t cusparseSpVV_bufferSizeHelper(cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, float *result, size_t *bufferSize)
{
    return cusparseSpVV_bufferSize(handle, opX, vecX, vecY, result, CUDA_R_32F, bufferSize);
}
inline cusparseStatus_t cusparseSpVV_bufferSizeHelper(cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, double *result, size_t *bufferSize)
{
    return cusparseSpVV_bufferSize(handle, opX, vecX, vecY, result, CUDA_R_64F, bufferSize);
}
inline cusparseStatus_t cusparseSpVV_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, half *, size_t *)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpVV_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpVV_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, int *, size_t *)
{
    RuntimeError("Unsupported template argument(int) in cusparseSpVV_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpVV_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, short *, size_t *)
{
    RuntimeError("Unsupported template argument(short) in cusparseSpVV_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpVV_bufferSizeHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, char *, size_t *)
{
    RuntimeError("Unsupported template argument(char) in cusparseSpVV_bufferSizeHelper");
}
inline cusparseStatus_t cusparseSpVVHelper(cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, float *result, void *externalBuffer)
{
    return cusparseSpVV(handle, opX, vecX, vecY, result, CUDA_R_32F, externalBuffer);
}
inline cusparseStatus_t cusparseSpVVHelper(cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, double *result, void *externalBuffer)
{
    return cusparseSpVV(handle, opX, vecX, vecY, result, CUDA_R_64F, externalBuffer);
}
inline cusparseStatus_t cusparseSpVVHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, half *, void *)
{
    RuntimeError("Unsupported template argument(half) in cusparseSpVV");
}
inline cusparseStatus_t cusparseSpVVHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, int *, void *)
{
    RuntimeError("Unsupported template argument(int) in cusparseSpVV");
}
inline cusparseStatus_t cusparseSpVVHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, short *, void *)
{
    RuntimeError("Unsupported template argument(short) in cusparseSpVV");
}
inline cusparseStatus_t cusparseSpVVHelper(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, char *, void *)
{
    RuntimeError("Unsupported template argument(char) in cusparseSpVV");
}


// Generalize cub calls
inline cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_keys_in, float *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream)
{
    return cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
}
inline cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_keys_in, double *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream)
{
    return cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
}
inline cudaError_t SortPairsDescending(void *, size_t, const half *, half *, const uint64_t *, uint64_t *, int, int, int, cudaStream_t)
{
    RuntimeError("Unsupported template argument(half) in SortPairsDescending");
}
inline cudaError_t SortPairsDescending(void *, size_t, const int *, int *, const uint64_t *, uint64_t *, int, int, int, cudaStream_t)
{
    RuntimeError("Unsupported template argument(int) in SortPairsDescending");
}
inline cudaError_t SortPairsDescending(void *, size_t, const short *, short *, const uint64_t *, uint64_t *, int, int, int, cudaStream_t)
{
    RuntimeError("Unsupported template argument(short) in SortPairsDescending");
}
inline cudaError_t SortPairsDescending(void *, size_t, const char *, char *, const uint64_t *, uint64_t *, int, int, int, cudaStream_t)
{
    RuntimeError("Unsupported template argument(char) in SortPairsDescending");
}

#endif // CPUONLY
