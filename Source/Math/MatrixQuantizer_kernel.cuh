#ifndef __MATRIX_QUANTIZER_KERNEL_CUH__
#define __MATRIX_QUANTIZER_KERNEL_CUH__
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

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

//#define NEW_COMPRESSION // experimental

#ifdef NEW_COMPRESSION
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif
template<class ElemType>
struct Item
{
    static const CUDA_LONG K = 8*sizeof(ElemType);
    ElemType val;
    CUDA_LONG pos; // position
};
__device__ float  fabs_(float  x) { return fabsf(x); }
__device__ double fabs_(double x) { return fabs (x); }
__device__ float  ImplantPos(float  val, CUDA_LONG pos) { return __int_as_float      ((__float_as_int      (val) & -Item<float >::K) | pos); }
__device__ double ImplantPos(double val, CUDA_LONG pos) { return __longlong_as_double((__double_as_longlong(val) & -Item<double>::K) | pos); }
__device__ CUDA_LONG ExtractPos(float  val) { return __float_as_int      (val) & ~-Item<float >::K; }
__device__ CUDA_LONG ExtractPos(double val) { return __double_as_longlong(val) & ~-Item<double>::K; }
template<class ElemType>
__global__ void CompressKernel(const ElemType* us, ElemType* curResidual, CUDA_LONG NN,
                               ElemType* compressed, ElemType* newResidual)
{
    static const CUDA_LONG K = Item<ElemType>::K;
    __shared__ Item<ElemType> items[K];
    CUDA_LONG pos = threadIdx.x; // 0..K-1
    CUDA_LONG blk = blockIdx.x;  // 0..#elements / K -1
    CUDA_LONG i = K/*=blockDim.x*/ * blk + pos; // absolute index in linearized matrix
    auto val = i < NN ? us[i] + curResidual[i] : 0; // get the matrix value
    // reduce-argmax
    items[pos].pos = pos;
    items[pos].val = val;
    for (CUDA_LONG step = 2; step <= K; step *= 2)
    {
        CUDA_LONG k1 = pos * step;
        CUDA_LONG k2 = k1 + (step / 2);
        if (k1 < K)
        {
            if (fabs_(items[k2].val) > fabs_(items[k1].val))
                items[k1] = items[k2];
            // now only the items at multiples of 'step' are valid
        }
        __syncthreads();
    }
    // now the index of the max item is in items[0]
    ElemType err;
    if (pos == items[0].pos) // we are the survivor
    {
        // implant the index in the least significant mantissa bits
        ElemType itemVal = ImplantPos(val, pos);
        // and store result in target buffer
        compressed[blk] = itemVal;
        err = val - itemVal; // we are missing this much (this also considers the implanted index)
    }
    else
        err = val; // all other values are completely missing
    // and finally compute the residual
    if (i < NN)
        newResidual[i] = err;
}
template <class ElemType>
__global__ void UncompressKernel(const ElemType* compressed, ElemType* us, const long NN, bool add)
{
    CUDA_LONG pos = threadIdx.x;
    CUDA_LONG blk = blockIdx.x * blockDim.y + threadIdx.y;
    CUDA_LONG i = blk * blockDim.x + pos;

    //CUDA_LONG NN = (CUDA_LONG)(M * N);
    //CUDA_LONG K = Item<ElemType>::K;
    //CUDA_LONG numWarps = 512 / K * K; // (must be a multiple of K)
    //UncompressKernel<ElemType> << <(NN + numWarps - 1) / numWarps, dim3(K, numWarps / K), 0, stream >> >((const ElemType*)gpuBuffer, us, NN, add);


    if (i >= NN)
        return;
    ElemType itemVal = compressed[blk];
    CUDA_LONG itemPos = ExtractPos(itemVal);
    if (itemPos != pos)
        itemVal = 0; // not the selected value: compressed as "0"
    if (!add)
        us[i] = itemVal;
    else if (itemVal)
        us[i] += itemVal;
}
#endif

// quantize an [M x N] matrix into qPackage in units of Nbits per value, and maintain residual
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
#ifdef NEW_COMPRESSION
    if (Nbits == 1)
    {
        CUDA_LONG NN = (CUDA_LONG)(M * N);
        CUDA_LONG K = Item<ElemType>::K;
        CompressKernel<ElemType><<<(NN + K - 1) / K, K, K * sizeof(Item<ElemType>), stream>>>(us, curResidual, NN, (ElemType*)qPackage, newResidual);
        //if (NN >= 320)
        //{
        //    ElemType usData[320];
        //    ElemType resData[320];
        //    ElemType compressed[10];
        //    cudaMemcpy(usData ,    us,                  sizeof(ElemType) * 320, cudaMemcpyDeviceToHost);
        //    cudaMemcpy(resData,    newResidual,         sizeof(ElemType) * 320, cudaMemcpyDeviceToHost);
        //    cudaMemcpy(compressed, (ElemType*)qPackage, sizeof(ElemType) * 10 , cudaMemcpyDeviceToHost);
        //    fprintf(stderr, "");
        //}
        return;
    }
#endif

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
#ifdef NEW_COMPRESSION
    if (nBits == 1)
    {
        CUDA_LONG NN = (CUDA_LONG)(M * N);
        CUDA_LONG K = Item<ElemType>::K;
        CUDA_LONG numWarps = 512 / K * K; // (must be a multiple of K)
        UncompressKernel<ElemType><<<(NN + numWarps - 1) / numWarps, dim3(K, numWarps / K), 0, stream>>>((const ElemType*)gpuBuffer, us, NN, add);
        return;
    }
#endif
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

}}} // namespace

#endif