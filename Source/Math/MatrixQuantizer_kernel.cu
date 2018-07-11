#ifndef __MATRIX_QUANTIZER_KERNEL_CUH__
#define __MATRIX_QUANTIZER_KERNEL_CUH__
#include <float.h>
#include <cuda.h>
#include <curand_kernel.h> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#ifdef _MSC_VER
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4515) // 'namespace': namespace uses itself
#endif
#include <cub/cub.cuh>

#include "Constants.h"
#include "ValueQuantizer.h"
#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// =======================================================================
// thread layout helpers
// =======================================================================

// --- distribute array elements naively over threads
__host__ static void ParallelizeOverRangeDim(size_t size, dim3& griddim, dim3& blockdim, const size_t warpsize = 64)
{
    // <<< griddim, blockdim, sharedmemsize, stream >>>
    griddim = (unsigned int) ((size + warpsize - 1) / warpsize); // 'warpsize' threads on each block (-> threadIdx.x)
    blockdim = (unsigned int) warpsize;                          // -> blockIdx.x
}
// get the array index for the current thread
__device__ __inline__ static size_t ParallelizeOverRangeIndex()
{
    return threadIdx.x + (blockIdx.x * blockDim.x);
}

// =======================================================================
// quantization
// =======================================================================

// helper to reduce all T across all threads of a block
template <typename T, int BLOCKSIZE>
__device__ void allreduce(T& var)
{
    __shared__ T buf[BLOCKSIZE];
    volatile T* vBuf = buf;

    buf[threadIdx.x] = var;
    __syncthreads();

    // We assume BLOCKSIZE is a power of 2
    if (BLOCKSIZE >= 1024)
    {
        if (threadIdx.x < 512)
        {
            var = var + buf[threadIdx.x + 512];
            buf[threadIdx.x] = var;
        }
        __syncthreads();
    }

    if (BLOCKSIZE >= 512)
    {
        if (threadIdx.x < 256)
        {
            var = var + buf[threadIdx.x + 256];
            buf[threadIdx.x] = var;
        }
        __syncthreads();
    }

    if (BLOCKSIZE >= 256)
    {
        if (threadIdx.x < 128)
        {
            var = var + buf[threadIdx.x + 128];
            buf[threadIdx.x] = var;
        }
        __syncthreads();
    }

    if (BLOCKSIZE >= 128)
    {
        if (threadIdx.x < 64)
        {
            var = var + buf[threadIdx.x + 64];
            buf[threadIdx.x] = var;
        }
        __syncthreads();
    }

    // Intra warp reduce
    if ((BLOCKSIZE >= 64) && (threadIdx.x < 32))
    {
        var = var + vBuf[threadIdx.x + 32];
        vBuf[threadIdx.x] = var;
    }

    if ((BLOCKSIZE >= 32) && (threadIdx.x < 16))
    {
        var = var + vBuf[threadIdx.x + 16];
        vBuf[threadIdx.x] = var;
    }

    if ((BLOCKSIZE >= 16) && (threadIdx.x < 8))
    {
        var = var + vBuf[threadIdx.x + 8];
        vBuf[threadIdx.x] = var;
    }

    if ((BLOCKSIZE >= 8) && (threadIdx.x < 4))
    {
        var = var + vBuf[threadIdx.x + 4];
        vBuf[threadIdx.x] = var;
    }

    if ((BLOCKSIZE >= 4) && (threadIdx.x < 2))
    {
        var = var + vBuf[threadIdx.x + 2];
        vBuf[threadIdx.x] = var;
    }

    if ((BLOCKSIZE >= 2) && (threadIdx.x == 0))
    {
        var = var + vBuf[1];
        vBuf[0] = var;
    }

    __syncthreads();

    var = buf[0];
}

#define REDUCTION_BLOCK_SIZE 128 // 256 is much worse; 64 is somewhat worse

// version optimized for collated memory access
template <class ElemType, bool ZeroThresholdFor1Bit>
__global__ void _ComputeQuantiStatParj(const ElemType* us, const ElemType* inResidual, long M, long N, size_t ldNbits, char* qpackage)
{
    size_t subset = threadIdx.x; // first thread computes 0, 64, 128; second thread 1, 65, 129 etc.
    size_t j = blockIdx.x;       // we process one column per *block*, j=column index; note: j is never out of range

    size_t rows = M; // we compute from 0..rows-1
    size_t bits = 1 << ldNbits;
    const size_t colSizeByte = Microsoft::MSR::CNTK::QuantizedColumn<ElemType>::QuantizedColumnSize(bits, rows);
    auto& qcol = *(Microsoft::MSR::CNTK::QuantizedColumn<ElemType>*) &qpackage[colSizeByte * j];

    Microsoft::MSR::CNTK::ColumnQuantizer<ElemType>::ComputeRangeStatColjSubset<ZeroThresholdFor1Bit>(us, inResidual, M, j, bits, qcol.lower, qcol.upper,
                                                                                                      subset, REDUCTION_BLOCK_SIZE, allreduce<ElemType, REDUCTION_BLOCK_SIZE>, allreduce<unsigned int, REDUCTION_BLOCK_SIZE>);
}

//caller: griddim and blockdim should be both 1d
//total thread number is: totalNumQWordsAlMatrix = numCols() * numQWordsPerCol
//called to quantize a GPU matrix
template <class ElemType, bool ZeroThresholdFor1Bit>
__global__ void _QuantizeStripjOneQWord(
    const ElemType* us,
    ElemType* curResidual,
    long M, long N,
    char* qMat,
    size_t qColSize,
    size_t numQWordsPerCol,
    size_t ldNbits,
    ElemType* newResidual)
{
    // map our thread index into a linear index
    const size_t linindex = ParallelizeOverRangeIndex();

    // map to (QWord index, column index)
    const size_t j = linindex / numQWordsPerCol;
    if (j >= N) // out of col range
        return;

    const size_t iQWord = linindex % numQWordsPerCol;

    // get data pointers to the quantized column
    auto& qCol = *(Microsoft::MSR::CNTK::QuantizedColumn<ElemType>*) &qMat[qColSize * j];

    // and quantizer
    const Microsoft::MSR::CNTK::ColumnQuantizer<ElemType> q(ldNbits, qCol.lower, qCol.upper);

    // quantize one QWord to qCol[iQWord]
    qCol.bits[iQWord] = q.QuantizeOneQWord<ZeroThresholdFor1Bit>(us, curResidual, M, iQWord, M, numQWordsPerCol, j, newResidual);
}

template <class ElemType>
__global__ void UnquantizeStripejOneQWord(ElemType* us, const long M, const long N, const char* qpackage, size_t colsize, size_t numQWordsPerCol, size_t ldNbits, bool add)
{
    // this follows the same as  quantizestripej()
    // map our thread index into a linear index
    const size_t linindex = ParallelizeOverRangeIndex();
    // map to (QWord index, column index)
    const size_t j = linindex / numQWordsPerCol;

    if (j >= N) // out of col range
        return;

    const size_t iQWord = linindex % numQWordsPerCol;

    // get data pointers and quantizer
    const auto& qcol = *(const Microsoft::MSR::CNTK::QuantizedColumn<ElemType>*) &qpackage[colsize * j];
    const ElemType lower = qcol.lower;
    const ElemType upper = qcol.upper;
    Microsoft::MSR::CNTK::ColumnQuantizer<ElemType> q(ldNbits, lower, upper);
    // unquantize from this one QWord
    q.UnquantizeOneQWord(us, M, iQWord, M, numQWordsPerCol, j, qcol.bits[iQWord], add);
}

//maybe should move out into another class?
template <class ElemType>
void _QuantizeMatrix(
    const ElemType* us,
    ElemType* curResidual,
    long M, long N,
    char* qPackage,
    size_t Nbits,
    cudaStream_t stream,
    ElemType* newResidual,
    bool zeroThresholdFor1Bit)
{

    /* verify buffer allocation size
        if (msra::math::matrixquantizer::buffersize(bits, rows(), cols()) != gpubuffer.size())
        LogicError("quantizestripe: dimension of patch to be quantized does not match allocated buffer size for quantized data");
        if (rows() != curresidual.rows() || cols() != curresidual.cols()
        || rows() != newresidual.rows() || cols() != newresidual.cols())
        LogicError("quantizestripe: dimension of patch to be quantized does not match residual buffer");
        if (gpubuffer.size() == 0)      // empty buffer: empty matrix, we are done (explicit test needed since launch will fail with 0 threads)
        return;*/
    // determine mean and variance -> value range (stored in quant package)   --for 1 bit, refine it in a second pass
    const size_t ldNbits = ValueQuantizer<ElemType>::ld(Nbits);

    size_t nRow = M;
    size_t nCol = N;
    dim3 mvgriddim, mvblockdim;
    // using specialized CUDA code (not shared with CPU) for collated memory access
    // each thread column computes 'warpsize' elements
    mvgriddim = (unsigned int) nCol; // column number
    mvblockdim = REDUCTION_BLOCK_SIZE;

    if (zeroThresholdFor1Bit)
    {
        _ComputeQuantiStatParj<ElemType, true><<<mvgriddim, mvblockdim, 0, stream>>>(us, curResidual, M, N, ldNbits, qPackage);
    }
    else
    {
        _ComputeQuantiStatParj<ElemType, false><<<mvgriddim, mvblockdim, 0, stream>>>(us, curResidual, M, N, ldNbits, qPackage);
    }

    // quantize data (also computing the residual at once)
    // optimizing for collated memory access:
    //  - each 32-bit word represents an interleaved (not consecutive) set of floats -> parallel threads can do collated accesses
    // example:
    //  - total number of 32-bit words(1-bit quant): 1100 * 2048 / 32 = 70k
    //  - thread x dimension: index into 32-bit word (e.g. 1100/32 = 35 threads)
    //  - thread y dimension and thread position: column (e.g. 2048)
    //  - using 128 threads on one proc -> 70k/128 = 550 blocks
    //  - threads are indexed by a global index into quantized 32-bit words in increasing order; each thread must
    //     - re-linearize block index and thread index
    //     - map to (i,j) coordinate (start of the set of floats)

    const size_t numQWordsPerCol = Microsoft::MSR::CNTK::ColumnQuantizer<ElemType>::QWordsPerCol(nRow, Nbits);
    const size_t totalQWords = nCol * numQWordsPerCol;

    const size_t colsizebyte = Microsoft::MSR::CNTK::QuantizedColumn<ElemType>::QuantizedColumnSize(Nbits, nRow);

    dim3 griddim, blockdim;
    ParallelizeOverRangeDim(totalQWords, griddim, blockdim, 256);
    if (zeroThresholdFor1Bit)
    {
        _QuantizeStripjOneQWord<ElemType, true><<<griddim, blockdim, 0, stream>>>(us, curResidual, M, N, qPackage, colsizebyte, numQWordsPerCol, ldNbits, newResidual);
    }
    else
    {
        _QuantizeStripjOneQWord<ElemType, false><<<griddim, blockdim, 0, stream>>>(us, curResidual, M, N, qPackage, colsizebyte, numQWordsPerCol, ldNbits, newResidual);
    }
}

// unquantize
// Process the quantization package to recover (unquantize) the matrix patch.
template <class ElemType>
void _UnquantizeMatrix(const char* gpuBuffer, size_t gpuBufferSize,
                       ElemType* us, long M, long N,
                       size_t nBits, bool add, cudaStream_t stream)
{
    // verify buffer allocation size
    /*if (msra::math::matrixquantizer::buffersize(bits, rows(), cols()) != gpubuffer.size())
            LogicError("unquantizestripe: dimension of patch to be unquantized does not match size of quantized data");
        if (gpubuffer.size() == 0)      // empty buffer: empty matrix, we are done (explicit test needed since launch will fail with 0 threads)
            return;
        */
    size_t qSize = QuantizedColumn<ElemType>::QuantizedColumnSize(nBits, M) * N;
    if (qSize != gpuBufferSize)
        LogicError("unquantizestripe: dimension of patch to be unquantized does not match size of quantized data");
    if (gpuBufferSize == 0) // empty buffer: empty matrix, we are done (explicit test needed since launch will fail with 0 threads)
        return;

    // #bits must be a power of two; we operate on shift values
    const size_t ldNbits = ValueQuantizer<ElemType>::ld(nBits);
    // unquantize in the same thread layout as quantize(), see there
    const size_t numQWordsPerCol = ColumnQuantizer<ElemType>::QWordsPerCol(M, nBits);
    const size_t totalQWords = N * numQWordsPerCol;

    const size_t colsize = QuantizedColumn<ElemType>::QuantizedColumnSize(nBits, M);

    dim3 griddim, blockdim;
    ParallelizeOverRangeDim(totalQWords, griddim, blockdim, 256);
    UnquantizeStripejOneQWord<<<griddim, blockdim, 0, stream>>>(us, M, N, gpuBuffer, colsize, numQWordsPerCol, ldNbits, add);
}

const unsigned long long int constBitOff = 32;
const unsigned long long int constBitMaskAll1 = (1LL << constBitOff) - 1;

__device__ void _packValueAndPosition(unsigned long long int integerRepresentation, unsigned int position, unsigned long long int &packed)
{
    packed = (integerRepresentation << constBitOff) | position;
}

__device__ void _unpackValueAndPosition(unsigned int &integerRepresentation, unsigned int &position, unsigned long long int packed)
{
    integerRepresentation = (packed >> constBitOff) & constBitMaskAll1;
    position = packed & constBitMaskAll1;
}

//caller: griddim and blockdim should be both 1d
//total thread number is: totalNumQWordsAlMatrix = numCols() * numQWordsPerCol
//called to quantize a GPU matrix
#define ITEMS_PER_THREAD 4 // TODO: As arguemnt ?!?
#define WARP_SIZE (DEFAULT_BUCKET_SIZE / ITEMS_PER_THREAD) // Best with that value!
template <class ElemType>
__global__ void _selectK(
    const ElemType* us,
    ElemType* curResidual,
    long numBuckets,
    char* buffer,
    size_t topK,
    ElemType* newResidual,
    size_t totalNumElements)
{
    // map our thread index into a linear index
    const size_t linindex = ParallelizeOverRangeIndex();

    // map to (QWord index, column index)
    const size_t currBucket = blockIdx.x;

    // printf("I am here %ld\n",currBucket);

    // get data pointers to the k elements of the bucket
    size_t qColSize = sizeof(unsigned long long int);
    size_t bucketOffset = currBucket * topK * qColSize;

    // and quantizer
    size_t realNumRows = min(DEFAULT_BUCKET_SIZE, totalNumElements - currBucket * DEFAULT_BUCKET_SIZE);

    //typedef cub::BlockLoad<int, WARP_SIZE, ITEMS_PER_THREAD> BlockLoadT;
    typedef cub::BlockScan<int, WARP_SIZE> BlockScanT;
    typedef cub::BlockReduce<ElemType, WARP_SIZE> BlockReduceT;

    // Shared memory
    __shared__ union
    {
        typename BlockScanT::TempStorage scan;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    // Calculate inf and 1 norm
    ElemType absVals[ITEMS_PER_THREAD];

    int offset = (threadIdx.x * ITEMS_PER_THREAD);
    int idx = (blockIdx.x * DEFAULT_BUCKET_SIZE) + offset;
    // TODO use cub::BlockLoad??
    for(int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(idx + i < totalNumElements) {
            absVals[i] = fabs(curResidual[idx + i] + us[idx + i]);
        } else {
            absVals[i] = 0.0;
        }
    }

    __syncthreads();

    // Calc statistics
    //__shared__ ElemType infNorm;
    __shared__ ElemType oneNorm;

    //ElemType redVal = BlockReduceT(temp_storage.reduce).Reduce(absVals, cub::Max());
    //if(threadIdx.x == 0) {
    //    infNorm = redVal;
    //    //printf("Inf-Norm: %f\n", infNorm);
    //}
    //__syncthreads();

    ElemType redVal = BlockReduceT(temp_storage.reduce).Sum(absVals);
    if(threadIdx.x == 0) {
        oneNorm = redVal;
        //printf("One-Norm: %f\n", oneNorm);
    }
    __syncthreads();

    //// Calc epsilon if needed
    //if(infNorm > oneNorm / topK) {
    //    eps = (topK * infNorm - oneNorm) / (DEFAULT_BUCKET_SIZE - topK);
    //    //if(threadIdx.x == 0)
    //    //    printf("EPSILON: %f\n", eps);
    //}

    curandState state;
    curand_init((unsigned long long)clock() + linindex, 0, 0, &state);
    float prob = 0.0;
    if (oneNorm > 1e-5) {
      // > 0
      prob = topK / oneNorm;
      //prob = topK / (oneNorm + DEFAULT_BUCKET_SIZE * eps);
    }

    int take[ITEMS_PER_THREAD];
    int indices[ITEMS_PER_THREAD];

    for(int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(idx + i < totalNumElements) {
            take[i] = (curand_uniform(&state) <= (.1 * prob * absVals[i])) ? 1 : 0;
            //take[i] = (curand_uniform(&state) <= (prob * (absVals[i] + eps))) ? 1 : 0;
            //printf("Taking: %d - %d\n", idx + i, take[i]);
        } else {
            take[i] = 0;
        }
    }

    // TODO Hacky! (change to not only select the first k elements!)

    __syncthreads();

    BlockScanT(temp_storage.scan).InclusiveSum(take, indices);

    for(int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(idx + i < totalNumElements) {
            size_t ij = idx + i;
            if(take[i] > 0 && indices[i] <= topK) {
                ElemType val = curResidual[ij] + us[ij];

                auto& qItem = *(struct s_item<unsigned, ElemType> *) &buffer[bucketOffset + ((indices[i]-1) * (sizeof(unsigned) + sizeof(ElemType)))];
                qItem.idx = ij;
                qItem.val = val;

                newResidual[ij] = 0;
            } else {
                newResidual[ij] = curResidual[ij] + us[ij];
            }
        }
    }

    __syncthreads();
    if(threadIdx.x == blockDim.x -1) {
        // Last threads fills up with the last possible index in order for the sparse sum to work!

        unsigned lastIdx = (blockIdx.x * DEFAULT_BUCKET_SIZE) + realNumRows - 1;
        for(size_t i = indices[ITEMS_PER_THREAD-1]; i < topK; ++i) {
            auto& qItem = *(struct s_item<unsigned, ElemType> *) &buffer[bucketOffset + (i * (sizeof(unsigned) + sizeof(ElemType)))];
            qItem.idx = lastIdx;
            qItem.val = 0.0;
        }
    }
}

template <class ElemType>
__global__ void _reset(ElemType* us, size_t totalNumElements)
{
    int offset = (threadIdx.x * ITEMS_PER_THREAD);
    int idx = (blockIdx.x * DEFAULT_BUCKET_SIZE) + offset;
    // TODO use cub::BlockLoad??
    for(int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(idx + i < totalNumElements) {
            us[idx + i] = 0.0;
        }
    }
}

template <class ElemType>
__global__ void _writeBackK(ElemType* us, const char* buffer, size_t totalNumElements, bool dense)
{
    // this follows the same as  quantizestripej()
    // map our thread index into a linear index
    const size_t linindex = ParallelizeOverRangeIndex();

    if(linindex < totalNumElements) {
        if(dense) {
            auto& qItem = *(ElemType *) &buffer[linindex * sizeof(ElemType)];
            us[linindex] = qItem;
        } else {
            auto& qItem = *(struct s_item<unsigned, ElemType> *) &buffer[linindex * (sizeof(unsigned) + sizeof(ElemType))];
            us[qItem.idx] = qItem.val;
        }
    }
}

template <class ElemType>
void _TopKMatrix(
    const ElemType* us,
    ElemType* curResidual,
    long M, long N,
    char* buffer,
    size_t topK,
    cudaStream_t stream,
    ElemType* newResidual)
{
    size_t numElements = M * N;
    size_t numBuckets = (numElements + (DEFAULT_BUCKET_SIZE - 1)) / DEFAULT_BUCKET_SIZE;

    dim3 griddim, blockdim;
#if defined(_MSC_VER)
    griddim = (unsigned int) numBuckets;
#else
    griddim = numBuckets;
#endif
    blockdim = WARP_SIZE;

#if defined(_MSC_VER)
    _selectK<ElemType><<<griddim, blockdim, 0, stream>>>(us, curResidual, (long)numBuckets, buffer, topK, newResidual, numElements);
#else
    _selectK<ElemType><<<griddim, blockdim, 0, stream >>>(us, curResidual, numBuckets, buffer, topK, newResidual, numElements);
#endif
}

template <class ElemType>
void _UnTopKMatrix(const char* gpuBuffer, size_t nofItems,
                       ElemType* us, long M, long N,
                       cudaStream_t stream)
{
    if (nofItems == 0) // empty buffer: empty matrix, we are done (explicit test needed since launch will fail with 0 threads)
        return;

    // TODO If dense: let threads be responsible for more than 1 item
    size_t numerOfWarps = (nofItems + (WARP_SIZE - 1)) / WARP_SIZE;

    dim3 griddim, blockdim;
#if defined(_MSC_VER)
    griddim = (unsigned int)numerOfWarps;
#else
    griddim = numerOfWarps;
#endif
    blockdim = WARP_SIZE;
    _writeBackK<<<griddim, blockdim, 0, stream>>>(us, gpuBuffer, nofItems, M*N == nofItems);
}
}
}
}

#endif
