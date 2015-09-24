// cudamatrix.cu(.h) -- CUDA kernels for matrix ops. Consider this a .cu/.cpp file.
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudamatrixops.cu.h $
// 
// 87    8/19/13 9:36p Ruizhao
// add function to remove "bad" frames in sequential DNN training
//
// 86    2/20/13 4:55p Dongyu
// fix the undefined constant error by including float.h
// 
// 85    1/09/13 9:21p V-hansu
// (remove part of assert)
// 
// 84    1/09/13 5:19p V-hansu
// fix last update
// 
// 83    1/09/13 5:10p V-hansu
// setbackpropagationerrorsignalhsmoothingi() work for fsmbr + smbr mode
// now (mode 2)
// 
// 82    1/09/13 3:29p V-hansu
// add setbackpropagationerrorsignalhsmoothing() and related kernel to
// prepare for cuda based hsmoothing
// 
// 81    1/02/13 9:12p Adame
// Update maxPool routines to handle large minibatch sizes
// 
// 80    12/07/12 5:15a Adame
// convolution/maxpool support
// 
// 79    11/27/12 6:38p V-hansu
// change senone2keepmodelupdate to a row vector
// 
// 78    11/27/12 4:10p V-hansu
// add senone2keepmodelupdate to setbackpropagationerrorsignal()
// 
// 77    11/14/12 12:47p V-hansu
// add macro NO_SIL_UPDATE, not defined
// 
// 76    11/13/12 7:29p V-hansu
// add code to disable silence error setting, disabled
// 
// 75    11/04/12 7:52a Fseide
// new class matrixaccumulator
// 
// 74    11/02/12 4:31p T-simonw
// code formatting and documentation
// 
// 73    11/01/12 6:15p T-simonw
// fix compilation issue with cuda 1.3, where atomicAdd is not available
// 
// 72    10/29/12 3:48p T-simonw
// add own kernels for
// weighteddotproduct
// special puprose method: sethessianvectorsignal
// elementwise operations (square & division)
// 
// 71    10/17/12 8:21p Fseide
// (fixed a comment)
// 
// 70    10/16/12 11:25a Fseide
// two new methods dropout() and scale(), for implementing Hinton's
// drop-out method
// 
// 69    10/11/12 3:45p V-hansu
// (fix a comment)
// 
// 68    9/27/12 12:28a V-hansu
// change setzero into setvalue
// 
// 67    9/24/12 3:25p Fseide
// adadenom() no longer takes numsummands, as it has become obsolete
// 
// 66    9/24/12 3:03p Fseide
// updated adagradientdenom() to now take two avdenoms, the actual (for
// clipping) and the target (for scaling)
// 
// 65    9/21/12 3:33p Fseide
// implemented nosoftmax option for posteriorstats()
// 
// 64    9/18/12 11:16a Fseide
// new method adagradientfromdenom()
// 
// 63    9/18/12 11:06a Fseide
// implemented asum() and adadenom()
// 
// 62    9/18/12 10:06a Fseide
// accumulatesqr() now implements the IIR filter, and has a new argument
// 'keepweight' for that
// 
// 61    9/17/12 6:18p Fseide
// implemented accumulatesqr()
// 
// 60    9/16/12 4:37p Fseide
// setzero() no longer touches padding values
// 
// 59    9/16/12 4:35p Fseide
// new function setzero()
// 
// 58    9/07/12 5:45p V-hansu
// modify gemsi to check if otherweight == 0
// 
// 57    8/28/12 6:27p Fseide
// moved some non-CUDA code from the .cu.h to the .cu file, which now also
// includes cudalatticeops.cu.h
// 
// 56    8/28/12 5:39p Fseide
// hack-implementation of somedataoperation() for testing CUDA vectors
// 
// 55    6/06/12 5:12p Adame
// Copy Sync update
// 
// 54    5/16/12 11:44p V-xieche
// modify the commant for softmaxstep2 function for more than 2 cuda
// devices stripe.
// 
// 53    5/14/12 10:36p V-xieche
// modify code for softmaxstep2 to support more than 2 cuda devices in top
// layer.
// 
// 52    4/06/12 6:27p V-xieche
// modify codes for posteriorstats function for striped top layer. not
// finished yet.
// 
// 51    4/05/12 9:51p V-xieche
// add functions for posteriorstats in striped toplayer pipeline training.
// not finished yet.
// 
// 50    4/02/12 11:03a Fseide
// (moved a fuction, now it compiles again)
// 
// 49    4/02/12 11:02a Fseide
// some factoring--this will not build, will need to move a function
// around. Checking in first for easier comparison.
// 
// 48    4/02/12 10:45a Fseide
// factorized the parallelized softmax
// 
// 47    4/02/12 10:11a Fseide
// updated the striped softmax function, step 1, to properly use coalesced
// memory accesses
// 
// 46    4/01/12 9:52p Fseide
// a further attempt at speed-up
// 
// 45    4/01/12 9:36p Fseide
// changed softmax to row-wise parallelization
// 
// 44    4/01/12 4:45p V-xieche
// delete an wrong assert.
// 
// 43    4/01/12 2:06p Fseide
// seterrorsignal now takes an offset parameter so that it can work for
// vertical stripes;
// fixed an unnoticed syntax error in partial softmax
// 
// 42    4/01/12 13:55 Fseide
// bug fix in partial softmax
// 
// 41    4/01/12 11:53a Fseide
// fixed the partial softmax
// 
// 40    4/01/12 11:47a Fseide
// update to softmax partial
// 
// 39    4/01/12 11:37a Fseide
// stripedsoftmaxstep1j() now uses some optimization
// 
// 38    4/01/12 11:23a V-xieche
// add code for striped softmax computation in 2 gpu.
// 
// 37    12/06/11 5:44p Dongyu
// fixed bugs in reshapecolumnproduct
// 
// 36    11/28/11 5:55p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 35    11/23/11 1:15p Dongyu
// add KhatriRaoProducti
// 
// 34    11/04/11 14:58 Fseide
// added new argument 'otherweight' to addrowsum() to allow unscaled
// gradients w.r.t. momentum
// 
// 33    10/28/11 14:52 Fseide
// cleaned up confusing and inconsistent alpha and beta parameters in
// gemm-like functions, now calling them 'thisscale' and 'otherweight' to
// make it crystal-clear
// 
// 32    10/25/11 5:16p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 31    10/06/11 5:16p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 30    6/21/11 13:48 Fseide
// (added a comment and a runtime check to patchasblockdiagonal() which
// does not support multi-device operation at this point in time)
// 
// 29    6/21/11 13:40 Fseide
// added frame for new function patchasblockdiagonal(), but inner loop not
// implemented yet
// 
// 28    3/03/11 8:15a Dongyu
// added weight sparseness support in training.
// 
// 27    2/26/11 9:04p Fseide
// now using hierarchical summation in softmax(), hoping to save 5 bits of
// accuracy  --it seems very noisy
// 
// 26    2/26/11 4:50p Fseide
// new method softmax()
// 
// 25    2/25/11 5:51p Fseide
// (cosmetic change)
// 
// 24    2/25/11 2:06p Fseide
// gems() no longer implemented using CUBLAS but with our own kernel to
// support non-contiguous memory
// 
// 23    2/10/11 4:03p Fseide
// (added a comment)
// 
// 22    2/10/11 3:20p Fseide
// bug fix: posteriorstats() now working
// 
// 21    2/10/11 1:53p Fseide
// new method posteriorstats() (although it does not work correctly yet)
// 
// 20    2/10/11 11:32a Fseide
// new method mulbydsigm()
// 
// 19    2/10/11 11:17a Fseide
// new method setbackpropagationerrorsignal()
// 
// 18    2/09/11 1:55p Fseide
// minor fix in new checklaunch() function
// 
// 17    2/09/11 1:54p Fseide
// added an error check to all kernel launches
// 
// 16    2/08/11 11:50p Fseide
// fixed the incorrect usage of <<< >>>
// 
// 15    2/07/11 9:52p Fseide
// llstats() implemented
// 
// 14    2/07/11 7:08p Fseide
// new method addtoallcolumns()
// 
// 13    2/07/11 6:52p Fseide
// implemented samplebinary()
// 
// 12    2/07/11 6:27p Fseide
// implemented addrowsum() and sigmoid()
// 
// 11    1/31/11 10:12p Fseide
// weighting factors fixed
// 
// 10    1/31/11 9:59p Fseide
// (added a comment)
// 
// 9     1/31/11 9:47p Fseide
// factored getpatch()
// 
// 8     1/31/11 9:05p Fseide
// (some optimization that did not help)
// 
// 7     1/31/11 8:41p Fseide
// fixed the +1 in matrixpatch
// 
// 6     1/31/11 8:35p Fseide
// gemm() now seems to work, now stress-testing
// 
// 5     1/31/11 7:18p Fseide
// first version oif gemm() that does something useful (on small matrix;
// to be fully tested)
// 
// 4     1/31/11 6:23p Fseide
// gemm() partially implemented
// 
// 3     1/31/11 4:56p Fseide
// first implementation of a CUDA kernel
// 
// 2     1/31/11 3:32p Fseide
// stub for execution
// 
// 1     1/30/11 7:30p Fseide
// created as empty placeholders
#if 0
#endif

#undef NO_SIL_UPDATE    // [v-hansu] set error to 0 if relate to silence

#include <cuda.h>
#include "cudalib.h"
#include "cudabasetypes.h"
#include "cudamatrixops.h"
#include <stdexcept>
#include <assert.h>
#include "stdio.h"
#include <float.h>

#pragma push_macro ("atomicAdd")
#define atomicAdd(address,value) (*(address)+=(value))  // don't forget to #undef (#praga pop_macro)! Otherwise CUDA might compile with this...

namespace msra { namespace cuda {

    // ======================================================================
    // matrix functions
    // ======================================================================

    // this = this * thisscale + rowsum(othercols)
    // threadIdx.x is row index. Otherwise we stupidly loop. Not very efficient, but avoids data copying.
    __global__ void addrowsumi (matrixref<float> us, float thisscale, const matrixref<float> othercols, float otherweight)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= othercols.rows())
            return;
        // compute the row sum
        float sumi = 0.0f;
        for (size_t j = 0; j < othercols.cols(); j++)
            sumi += othercols(i,j);
        // scale it
        sumi *= otherweight;
        // add to 'this'
        for (size_t j = 0; j < us.cols(); j++)
        {
            float usij = sumi;
            if (thisscale != 0.0f)
                usij += us(i,j) * thisscale;
            us(i,j) = usij;
        }
    }
    void cudamatrixops::addrowsum (float thisscale, const cudamatrixops & othercols, float otherweight)
    {
        assert (rows() == othercols.rows());
        addrowsumi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, thisscale, othercols, otherweight);
        checklaunch ("addrowsum");
    }

    // this = this * thisscale + rowsum(othercols)
    // threadIdx.x is row index. Otherwise we stupidly loop. Not very efficient, but avoids data copying.
    __global__ void addrowsumpooli (matrixref<float> us, float thisscale, const matrixref<float> othercols, float otherweight, size_t poolSize, size_t bands, size_t kernels)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute the row sum
        float sumi = 0.0f;
        size_t iPool = i;
        for (size_t ip=0; ip < poolSize; ++ip, iPool += bands*kernels)
        {
            for (size_t j = 0; j < othercols.cols(); j++)
                sumi += othercols(iPool,j);
        }
        // scale it
        sumi *= otherweight;
        // add to 'this'
        for (size_t j = 0; j < us.cols(); j++)
        {
            float usij = sumi;
            if (thisscale != 0.0f)
                usij += us(i,j) * thisscale;
            us(i,j) = usij;
        }
    }

    void cudamatrixops::addrowsumpool (float thisscale, const cudamatrixops & othercols, float otherweight, size_t poolSize, size_t bands, size_t kernels)
    {
        assert (rows()*poolSize == othercols.rows());
        addrowsumpooli<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, thisscale, othercols, otherweight, poolSize, bands, kernels);
        checklaunch ("addrowsumpool");
    }

   __global__ void transposei (matrixref<float> to, const matrixref<float> from, size_t minibatchSize)
    {
        // threadIdx.x is thread index, blockDim.x is blockSize
        // blockIdx.x is kernal index, gridIndx.x is max
        // blockIndx.y is band index, gridIndx.y is max
        const size_t iyIn = blockIdx.z*blockDim.x + threadIdx.x;
        if (iyIn >= minibatchSize)
            return;

        const size_t ixIn = blockIdx.y + (blockIdx.x * gridDim.y);
        //assert(from.cols() == blockDim.x);
        //assert(from.rows() == gridDim.x * gridDim.y);
        const size_t ixOut = iyIn;
        const size_t iyOut = ixIn;
        //assert(to.rows() == blockDim.x);
        //assert(to.cols() == gridDim.x * gridDim.y);
        to(ixOut, iyOut) = from(ixIn, iyIn);
    }
    // reorder the matrix, which is a combination of transposing it, and swapping the kernel and band dimensions
    __global__ void reorderi (matrixref<float> to, const matrixref<float> from, size_t minibatchSize)
    {
        // threadIdx.x is thread index, blockDim.x is blockSize
        // blockIdx.x is kernal index, gridIndx.x is max
        // blockIndx.y is band index, gridIndx.y is max
        const size_t iyIn = blockIdx.z*blockDim.x + threadIdx.x;
        if (iyIn >= minibatchSize)
            return;

        const size_t ixIn = blockIdx.y + (blockIdx.x * gridDim.y);
        //assert(from.cols() == blockDim.x);
        //assert(from.rows() == gridDim.x * gridDim.y);
        const size_t ixOut = blockIdx.x + (blockIdx.y * gridDim.x);
        const size_t iyOut = iyIn;
        //assert(to.rows() == blockDim.x);
        //assert(to.cols() == gridDim.x * gridDim.y);
        to(ixOut, iyOut) = from(ixIn, iyIn);
    }
     // reorder the matrix, which is a combination of transposing it, and swapping the kernel and band dimensions
    __global__ void reorderti (matrixref<float> to, const matrixref<float> from, size_t minibatchSize)
    {
        // threadIdx.x is thread index, blockDim.x is blockSize
        // blockIdx.x is kernal index, gridIndx.x is max
        // blockIndx.y is band index, gridIndx.y is max
        const size_t iyIn = blockIdx.z*blockDim.x + threadIdx.x;
        if (iyIn >= minibatchSize)
            return;

        const size_t ixIn = blockIdx.y + (blockIdx.x * gridDim.y);
        //assert(from.cols() == blockDim.x);
        //assert(from.rows() == gridDim.x * gridDim.y);
        const size_t ixOut = iyIn;
        const size_t iyOut = blockIdx.x + (blockIdx.y * gridDim.x);
        //assert(to.rows() == blockDim.x);
        //assert(to.cols() == gridDim.x * gridDim.y);
        to(ixOut, iyOut) = from(ixIn, iyIn);
    }

    // reorder transpose
    void cudamatrixops::reorder (cudamatrixops & to, size_t minibatchSize, size_t kernels, size_t bands, bool input) const
    {
        // assert dimensions are transposed as we expect
        const size_t blockSize = 256;
        assert(cols() == to.cols());
        assert(rows() == to.rows());
        //this->dump("input to reorder");
        if (!input){
            //BUGBUG: do not do transpose here
            transposei<<<dim3(kernels, bands, (minibatchSize+blockSize-1)/blockSize), blockSize, 0, GetCurrentStream()>>> (to, *this, minibatchSize);
        }
        else{
            reorderi<<<dim3(kernels, bands, (minibatchSize+blockSize-1)/blockSize), blockSize, 0, GetCurrentStream()>>> (to, *this, minibatchSize);
        }
        //to.dump("reordered output");
        checklaunch ("reorderi");
    }

    #define THREADS_PER_BLOCK 256 
    #if __CUDA_ARCH__ < 200 
        #define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK 
        #define MY_KERNEL_MIN_BLOCKS 2 
    #else    
        #define MY_KERNEL_MAX_THREADS (2 * THREADS_PER_BLOCK) 
        #define MY_KERNEL_MIN_BLOCKS 3 
    #endif

#if 1
    __global__ void 
    //__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
    activateCNN(matrixref<float> wM, matrixref<float> bM, matrixref<float> vM, matrixref<float> hM, int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize) {
        //w: wieght matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
        //b: bias vector. 1 * nKernels*nBands
        //v: layer input. batchSize * nPrevKernels*nPrevBands
        //h: layer output. batchSize * nBands*nKernels*poolSize, transposed for dbn model
        //nPrevBands includes the energy band (which is at the end of input frequency bands)
        //Each kernel processes one output. Memory access is sequential though batches. Each block uses the same weights and consecutive input through a mini-batch
        //Each block generates output of the same filter with the same shift for a different sample.
        //Consider sharing weight matrix
    
        unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
        const int nBands = (nPrevBands) / poolingBandShift;
        const int outRows = nBands*nKernels*poolSize;
        const int vInc = vM.getcolstride();
        const int N = batchSize*outRows;
    
        for(; xIndex < N; xIndex += blockDim.x * gridDim.x) { 
        
            //Extract counters
            int iSample = xIndex % batchSize;
            //int iKernel = (xIndex / batchSize) % (nBands*poolSize);
            //int iBand = (xIndex / (batchSize*nKernels)) % poolSize;
            int iBand = (xIndex / batchSize) % nBands;
            int iKernel = (xIndex / (batchSize*nBands)) % nKernels;
            int iShift = xIndex / (batchSize*nBands*nKernels);

            int iOutWeight = iBand + iKernel*nBands;

            float a = 0;

            //All bands

            int bandShift = iBand * poolingBandShift + iShift - poolSize/2 - filterSize/2;
            int startBand = 0;//iBand - filterSize/ 2 + shift - poolSize / 2;
            if (startBand + bandShift < 0) startBand = -bandShift;
            int endBand = filterSize;
            if (endBand + bandShift> nPrevBands-1) //-1 here because the last band is energy
                endBand = nPrevBands - 1 - bandShift;

            int startNdx = startBand * nPrevKernels;// + iOutWeight*weightRows;
            int endNdx   = endBand  * nPrevKernels;// + iOutWeight*weightRows;
            float *inputPtr = &vM(0,0) + (startBand+bandShift) * vInc * nPrevKernels + iSample;
            //int inputNdx = (startBand+bandShift) * batchSize * nPrevKernels + iSample;

            float *w = &wM(0, iOutWeight);
            for(int i = startNdx; i < endNdx; i++) {
                a += (*inputPtr) * w[i];
                inputPtr += vInc;
                //inputNdx += batchSize;
            }

#if 0 // energy is always zero, so not lots of point to do this
            //Energy bands
            startNdx = filterSize * nPrevKernels;// + iOutWeight*nPrevKernels*(filterSize+1);
            endNdx = (filterSize+1)*nPrevKernels;// + iOutWeight*nPrevKernels*(filterSize+1);
            inputPtr = v + (nPrevBands - 1) * vInc * nPrevKernels + iSample;

            for(int i = startNdx; i < endNdx; i++) {
                a += (*inputPtr) * w[i];
                inputPtr += vInc;
            }
#endif

            //Add bias
            a = a + bM(iOutWeight,0);

            //Activation Function
            a = 1.0f / (1.0f + __expf(-a));
            hM(xIndex/batchSize, xIndex%batchSize) = a;
        }
    }
#else


        __global__ void 
        __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
        activateCNN(matrixref<float> wM, matrixref<float> bM, matrixref<float> vM, matrixref<float> hM, const int nPrevBands, const int nPrevKernels, const int batchSize) {
            //w: wieght matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
            //b: bias vector. 1 * nKernels*nBands
            //v: layer input. batchSize * nPrevKernels*nPrevBands
            //h: layer output. batchSize * nBands*nKernels*poolSize, transposed for dbn model
            //nPrevBands includes the energy band (which is at the end of input frequency bands)
            //Each kernel processes one output. Memory access is sequential though batches. Each block uses the same weights and consecutive input through a mini-batch
            //Each block generates output of the same filter with the same shift for a different sample.
            //Consider sharing weight matrix

            // replacing parameters and variables with grid and block variables to get register count down
            #define nKernels gridDim.y 
            #define poolSize gridDim.z
            #define poolingBandShift blockDim.y
            #define filterSize blockDim.z
            #define iShift blockIdx.z
            #define iKernel blockIdx.y
            

            // reconstruct a single index from the multi dimesional inputs
            unsigned int xIndex = threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*(threadIdx.z + blockDim.z*(blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*(blockIdx.z + gridDim.z))))); 
            const int nBands = (nPrevBands) / poolingBandShift;
            const int vInc = vM.getcolstride();
            const int N = batchSize*nBands*nKernels*poolSize;
        
            //for(; xIndex < N; xIndex += blockDim.x * gridDim.x) { 
            if (xIndex < N)
            {
                //Extract counters
                int iSample = xIndex % batchSize;
                //int iKernel = (xIndex / batchSize) % (nBands*poolSize);
                //int iBand = (xIndex / (batchSize*nKernels)) % poolSize;
                int iBand = (xIndex / batchSize) % nBands;
                //int iKernel = (xIndex / (batchSize*nBands)) % nKernels;
                //int iShift = xIndex / (batchSize*nBands*nKernels);
    
                int iOutWeight = iBand + iKernel*nBands;
    
                float a = 0;
    
                //All bands
    
                int bandShift = iBand * poolingBandShift + iShift - poolSize/2 - filterSize/2;
                int startBand = 0;//iBand - filterSize/ 2 + shift - poolSize / 2;
                if (startBand + bandShift < 0) startBand = -bandShift;
                int endBand = filterSize;
                if (endBand + bandShift> nPrevBands-1) //-1 here because the last band is energy
                    endBand = nPrevBands - 1 - bandShift;
    
                int startNdx = startBand * nPrevKernels;// + iOutWeight*weightRows;
                int endNdx   = endBand  * nPrevKernels;// + iOutWeight*weightRows;
                float *inputPtr = &vM(iSample,(startBand+bandShift)*nPrevKernels);
                //int inputNdx = (startBand+bandShift) * batchSize * nPrevKernels + iSample;
    
                float *w = &wM(0, iOutWeight);
                for(int i = startNdx; i < endNdx; i++) {
                    a += (*inputPtr) * w[i];
                    inputPtr += vInc;
                    //inputNdx += batchSize;
                }
    
#if 0 // energy is always zero, so not lots of point to do this
                //Energy bands
                startNdx = filterSize * nPrevKernels;// + iOutWeight*nPrevKernels*(filterSize+1);
                endNdx = (filterSize+1)*nPrevKernels;// + iOutWeight*nPrevKernels*(filterSize+1);
                inputPtr = v + (nPrevBands - 1) * vInc * nPrevKernels + iSample;
    
                for(int i = startNdx; i < endNdx; i++) {
                    a += (*inputPtr) * w[i];
                    inputPtr += vInc;
                }
#endif
    
                //Add bias
                a = a + bM(iOutWeight,0);
    
                //Activation Function
                a = 1.0f / (1.0f + __expf(-a));
                hM(xIndex/batchSize, xIndex%batchSize) = a;
            }
#undef nKernels //gridDim.y 
#undef poolSize //gridDim.z
#undef poolingBandShift //blockDim.y
#undef filterSize //blockDim.z
#undef iShift //blockIdx.z
#undef iKernel //blockIdx.y
            
        }
#endif    

    void cudamatrixops::convolutionForward(cudamatrixops & out, const cudamatrixops & weight, const cudamatrixops & bias, size_t minibatchSize, size_t prevkernels, size_t prevbands, size_t kernels, size_t poolingBandShift, size_t poolSize, size_t filterSize)
    {
        int nBands = (prevbands) / poolingBandShift;
        assert(out.rows() == nBands*kernels*poolSize);
        assert(weight.rows() == prevkernels*(filterSize+1));
        assert(rows() == minibatchSize);

        size_t bands = prevbands/poolingBandShift;
        size_t N = minibatchSize*bands*kernels*poolSize;
        size_t blockSize = 512; //(minibatchSize+1)/2;

        // to reduce register count and increase occupancy, we are putting all parameters into grid and block dimensions
        assert(blockSize%(poolingBandShift*filterSize) == 0);
        dim3 block(blockSize/(poolingBandShift*filterSize), poolingBandShift, filterSize);
        size_t gridBlock = (kernels*poolSize*blockSize);
        size_t gridX = (N+gridBlock-1)/gridBlock;
        dim3 grid(gridX, kernels, poolSize);
        
        //weight.dump("weight matrix");
        //bias.dump("bias matrix");
        //activateCNN<<<grid, block, 0, GetCurrentStream()>>>(weight, bias, *this, out, prevbands, prevkernels, minibatchSize);       
                // don't let it exceed the limit 64*1024 
        // if limited the kernel loops internally until done
        int gridSize = (N+blockSize-1)/blockSize;
        while (gridSize >= 64*1024-1)
        {
            gridSize = (gridSize+1)/2;
        }

        activateCNN<<<gridSize, blockSize, 0, GetCurrentStream()>>>(weight, bias, *this, out, prevbands, prevkernels, poolingBandShift, kernels, minibatchSize, poolSize, filterSize);       
        //out.dump("convolution output");
        checklaunch ("activateCNN");        
    }


#if 0 // experiment, ended up being WAY slower
    // optimized version that doesn't worry about energy (since it's zero anyway), and uses the tread index as an index into the weight matrix
    __global__ void activateCNN_opt(matrixref<float> wM, matrixref<float> bM, matrixref<float> vM, matrixref<float> hM, int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize) {
        float *b = &bM(0,0);
        float *v = &vM(0,0);
        float *h = &hM(0,0);
        //w: wieght matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
        //b: bias vector. 1 * nKernels*nBands
        //v: layer input. nPrevKernels*nPrevBands * batchSize
        //h: layer output. batchSize * nBands*nKernels*poolSize, transposed for dbn model
        //nPrevBands includes the energy band (which is at the end of input frequency bands)
        //Each kernel processes one output. Memory access is sequential though batches. Each block uses the same weights and consecutive input through a mini-batch
        //Each block generates output of the same filter with the same shift for a different sample.
        //Consider sharing weight matrix

        const unsigned int nBands = (nPrevBands) / poolingBandShift;
        const int outRows = nBands*nKernels*poolSize;
        //const int N = batchSize*outRows;

        extern __shared__ float accumSum[]; //accumulator for the sum
        // 
        
        //for(; xIndex < N; xIndex += gridDim.x * gridDim.y * gridDim.z) { 
        
            //Extract counters
            int iSample = blockIdx.x % batchSize;
            //int iKernel = (xIndex / batchSize) % (nBands*poolSize);
            //int iBand = (xIndex / (batchSize*nKernels)) % poolSize;
            int iBand = (blockIdx.x/ batchSize);
            int iKernel = blockIdx.y;
            int iShift = blockIdx.z;

            int iOutWeight = iBand + iKernel*nBands;

            //float a = 0;

            //All bands

            int bandShift = iBand * poolingBandShift + iShift - poolSize/2 - filterSize/2;
            int startBand = 0;//iBand - filterSize/ 2 + shift - poolSize / 2;
            if (startBand + bandShift < 0) startBand = -bandShift;
            int endBand = filterSize;
            if (endBand + bandShift> nPrevBands-1) //-1 here because the last band is energy
                endBand = nPrevBands - 1 - bandShift;

            //int startNdx = startBand * nPrevKernels;// + iOutWeight*weightRows;
            int endNdx   = endBand  * nPrevKernels;// + iOutWeight*weightRows;
            float input = vM((startBand+bandShift)* nPrevKernels,iSample);
            //int inputNdx = (startBand+bandShift) * batchSize * nPrevKernels + iSample;

            float *w = &wM(0, iOutWeight);
            if (blockIdx.x < endNdx)    //previously: for(int i = startNdx; i < endNdx; i++) {
            {
                accumSum[blockIdx.x] = input * w[blockIdx.x];
            }

            // wait for accumulator to fill up
            __syncthreads();

            //NOTE: In current implementation the energy bands are always zero so no need to worry about them
#if 0            
            startNdx = filterSize * nPrevKernels;// + iOutWeight*nPrevKernels*(filterSize+1);
            endNdx = (filterSize+1)*nPrevKernels;// + iOutWeight*nPrevKernels*(filterSize+1);
            inputPtr = v + (nPrevBands - 1) * vInc * nPrevKernels + iSample;

            for(int i = startNdx; i < endNdx; i++) {
                a += (*inputPtr) * w[i];
                inputPtr += vInc;
            }
#endif

            ////////////////////////////////////////////////////////////////////////
            // Perform tree-like reduction of accumulators' results.
            ////////////////////////////////////////////////////////////////////////
            float power = __log2f(endNdx);
            int largest2 = int(power);
            int pow2 = 1 << largest2;

            // if there is a decimal after the integer, then largest power of two is one larger
            if (power-largest2 > 0.0001)
            {
                pow2 <<= 1;
            }
             for(int stride = pow2>>1; stride > 0; stride >>= 1){
                 __syncthreads();
                 int iAccum = threadIdx.x; 
                 int iTarget = iAccum + stride;
                 if (iAccum < stride && iTarget < endNdx)
                     accumSum[iAccum] += accumSum[iTarget];
             }

            // thread zero takes care of wrapping things up
            if (threadIdx.x == 0)
            {
                //Add bias
                float a = accumSum[0] + b[iOutWeight];

                //Activation Function
                a = 1.0f / (1.0f + __expf(-a));
                unsigned xIndex =  (blockIdx.z * gridDim.y*gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x; 

                hM(xIndex/batchSize, xIndex%batchSize) = a;
            }
        //}
    }

    void cudamatrixops::convolutionForward(cudamatrixops & out, const cudamatrixops & weight, const cudamatrixops & bias, size_t minibatchSize, size_t prevkernels, size_t prevbands, size_t kernels, size_t poolingBandShift, size_t poolSize, size_t filterSize)
    {
        int nBands = (prevbands) / poolingBandShift;
        assert(out.rows() == nBands*kernels*poolSize);
        assert(weight.rows() == prevkernels*(filterSize+1));
        assert(cols() == minibatchSize);
        dim3 gridSize(minibatchSize*nBands,kernels,poolSize);
        unsigned blockSize = prevkernels*filterSize; 
        
       //weight.dump("weight matrix");
        //bias.dump("bias matrix");
        activateCNN_opt<<<gridSize, blockSize, blockSize*sizeof(float), GetCurrentStream()>>>(weight, bias, *this, out, prevbands, prevkernels, poolingBandShift, kernels, minibatchSize, poolSize, filterSize);       
        //out.dump("convolution output");
        checklaunch ("activateCNN");        
    }
#endif

    __global__ void ComputeCnnDeltaW_mod(matrixref<float> deltaM, matrixref<float> vM, matrixref<float> dwM, float thisscale, float vhscale, 
        int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize) {
        //delta: delta signal below transposed:  nBands*nKernels*poolSize * batchSize
        //v: layer input:  batchSize * nPrevKernels*nPrevBands
        //dw: wieght delta matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
        float *delta = &deltaM(0,0);
        float *dw = &dwM(0,0);
        float *v = &vM(0,0);
        // dw = dw*thisscale + delta1 * vhscale
        unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
        int nBands = (nPrevBands) / poolingBandShift;
        int dwRows = nPrevKernels*(filterSize+1);
        int N = dwRows * nBands*nKernels;
        int vInc = vM.getcolstride(); // nPrevKernels*nPrevBands;
    
        extern __shared__ float sd[]; //shared delta, size determined by calling function (reserves shared memory for this CUDA code)
    
        for(; xIndex < N; xIndex += blockDim.x * gridDim.x) { 

            int iPrevKernel = xIndex % nPrevKernels;
            int iWeightBand = (xIndex/nPrevKernels) % (filterSize+1);
            int iBand = (xIndex / dwRows) % nBands;
            int iKernel = xIndex/ (dwRows*nBands );
            float a = 0;

            //int nPoolingShifts = 0;
        
            for(int iShift = 0; iShift < poolSize; iShift++) {
                int iPrevBand;
                if (iWeightBand == filterSize)
                    iPrevBand = nPrevBands-1;
                else {
                    iPrevBand = iBand*poolingBandShift + iShift - filterSize/2 - poolSize/2 + iWeightBand;
                    if (iPrevBand < 0 || iPrevBand >= nPrevBands -1)
                        continue;
                }

                //nPoolingShifts++;
                int idelta = iShift * nKernels*nBands + iKernel*nBands + iBand;
                //float *pDelta = delta + iShift * nKernels*nBands*batchSize + iKernel*nBands*batchSize + iBand*batchSize; 
                //Fill sd
                //int deltaInc = nBands*nKernels*poolSize;
                __syncthreads();
               for(int i = iPrevKernel; i < batchSize; i+= blockDim.x)
                    sd[i] = deltaM(idelta, i);
                __syncthreads();

                float *pV = &vM(0,iPrevBand * nPrevKernels + iPrevKernel);

                // int vInc = nPrevKernels*nPrevBands;
    //            float *pVEnd = pV + batchSize * vInc;
        
                //for(int iSample = 0; iSample < batchSize ; iSample++) {
                //    a += delta[iSample* nKernels*nBands*poolSize + iShift * nKernels*nBands + iBand * nKernels + iKernel]
                //    * v[iSample * nPrevKernels*nPrevBands + iPrevBand * nPrevKernels + iPrevKernel];
                //}
                for(int iSample = 0; iSample < batchSize ; iSample++) {
                    a += sd[iSample] * (*pV++);
                }
    //            while( pV < pVEnd) {
    //                a += *pDelta * *pV;
    //                pDelta += deltaInc;
    //                pV += vInc;
    //            }
            }
            float& dwf = dwM(xIndex%dwRows, xIndex/dwRows);
            //assert(!isnan(a) && !isnan(dwf));// != 0x7fffffff);
            dwf = dwf*thisscale + a * vhscale; // / nPoolingShifts;

            //dw[xIndex] = dw[xIndex]*thisscale + a * vhscale; // / nPoolingShifts;
        }
    }

/* originally:
__global__ void computeCnnDeltaW(float *delta, float *v, float *dw, int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize) {
    //delta: delta signal below activation fn transposed. batchSize * nBands*nKernels*poolSize
    //v: layer input transposed. nPrevKernels*nPrevBands * batchSize
    //dw: wieght delta matrix. nPrevKernels*(filterSize+1) * nBands*nKernels

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
    int nBands = (nPrevBands) / poolingBandShift;
    int N = nPrevKernels*(filterSize+1) * nBands*nKernels;
    
    extern __shared__ float sd[]; //shared delta
    
    for(; xIndex < N; xIndex += blockDim.x * gridDim.x) { 

        int iPrevKernel = xIndex % nPrevKernels;
        int iWeightBand = (xIndex/nPrevKernels) % (filterSize+1);
        int iBand = (xIndex / (nPrevKernels*(filterSize+1))) % nBands;
        int iKernel = xIndex/ (nPrevKernels*(filterSize+1)*nBands );

        float a = 0;

        //int nPoolingShifts = 0;
        
        for(int iShift = 0; iShift < poolSize; iShift++) {
            int iPrevBand;
            if (iWeightBand == filterSize)
                iPrevBand = nPrevBands-1;
            else {
                iPrevBand = iBand*poolingBandShift + iShift - filterSize/2 - poolSize/2 + iWeightBand;
                if (iPrevBand < 0 || iPrevBand >= nPrevBands -1)
                    continue;
            }

            //nPoolingShifts++;

            float *pDelta = delta + iShift * nKernels*nBands*batchSize + iKernel*nBands*batchSize + iBand*batchSize;
            //Fill sd
            //int deltaInc = nBands*nKernels*poolSize;
            __syncthreads();
           for(int i = iPrevKernel; i < batchSize; i+= nPrevKernels)
                sd[i] = pDelta[i];
            __syncthreads();

            float *pV = v + iPrevBand * nPrevKernels + iPrevKernel;
            int vInc = nPrevKernels*nPrevBands;
//            float *pVEnd = pV + batchSize * vInc;
        
            //for(int iSample = 0; iSample < batchSize ; iSample++) {
            //    a += delta[iSample* nKernels*nBands*poolSize + iShift * nKernels*nBands + iBand * nKernels + iKernel]
            //    * v[iSample * nPrevKernels*nPrevBands + iPrevBand * nPrevKernels + iPrevKernel];
            //}
            for(int iSample = 0; iSample < batchSize ; iSample++) {
                a += sd[iSample] * (*pV);
                pV += vInc;
            }
//            while( pV < pVEnd) {
//                a += *pDelta * *pV;
//                pDelta += deltaInc;
//                pV += vInc;
//            }
            
        }

        dw[xIndex] = a; // / nPoolingShifts;
    }

}

*/
__global__ void ComputeCnnDeltaW(matrixref<float> deltaM, matrixref<float> vM, matrixref<float> dwM, float thisscale, float vhscale, 
    int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize) 
    {
        //delta: delta signal below activation fn. batchSize * nBands*nKernels*poolSize
        //v: layer input transposed. nPrevKernels*nPrevBands * batchSize
        //dw: wieght delta matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
        float *delta = &deltaM(0,0);
        float *dw = &dwM(0,0);
        float *v = &vM(0,0);
        // dw = dw*thisscale + delta1 * vhscale

        unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
        int nBands = (nPrevBands) / poolingBandShift;
        int dwRows = nPrevKernels*(filterSize+1);
        int N = dwRows * nBands*nKernels;
        int vInc = vM.getcolstride(); // nPrevKernels*nPrevBands;
        int dInc = deltaM.getcolstride();
    
        extern __shared__ float sd[]; //shared delta
    
        for(; xIndex < N; xIndex += blockDim.x * gridDim.x) 
        { 

            int iPrevKernel = xIndex % nPrevKernels;
            int iWeightBand = (xIndex/nPrevKernels) % (filterSize+1);
            int iBand = (xIndex / dwRows) % nBands;
            int iKernel = xIndex/ (dwRows*nBands );

            float a = 0;

            //int nPoolingShifts = 0;
        
            for(int iShift = 0; iShift < poolSize; iShift++) {
                int iPrevBand;
                if (iWeightBand == filterSize)
                    iPrevBand = nPrevBands-1;
                else {
                    iPrevBand = iBand*poolingBandShift + iShift - filterSize/2 - poolSize/2 + iWeightBand;
                    if (iPrevBand < 0 || iPrevBand >= nPrevBands -1)
                        continue;
                }

                //nPoolingShifts++;

                float *pDelta = delta + (((iShift * nKernels) + iKernel)*nBands + iBand)*dInc;
                //Fill sd
                //int deltaInc = nBands*nKernels*poolSize;
                __syncthreads();
               for(int i = threadIdx.x; i < batchSize; i+= blockDim.x)
                    sd[i] = pDelta[i];
                __syncthreads();

                float *pV = v + iPrevBand * nPrevKernels + iPrevKernel;
    //            float *pVEnd = pV + batchSize * vInc;
        
                //for(int iSample = 0; iSample < batchSize ; iSample++) {
                //    a += delta[iSample* nKernels*nBands*poolSize + iShift * nKernels*nBands + iBand * nKernels + iKernel]
                //    * v[iSample * nPrevKernels*nPrevBands + iPrevBand * nPrevKernels + iPrevKernel];
                //}
                for(int iSample = 0; iSample < batchSize ; iSample++) {
                    a += sd[iSample] * (*pV);
                    pV += vInc;
                }
    //            while( pV < pVEnd) {
    //                a += *pDelta * *pV;
    //                pDelta += deltaInc;
    //                pV += vInc;
    //            }
            
            }

            float& dwf = dwM(xIndex%dwRows, xIndex/dwRows);
                //assert(!isnan(a) && !isnan(dwf));// != 0x7fffffff);
            dwf = dwf*thisscale + a * vhscale; // / nPoolingShifts;

            //dw[xIndex] = dw[xIndex]*thisscale + a * vhscale; // / nPoolingShifts;
        }
    }

__global__ void ComputeCnnDeltaW_old(matrixref<float> deltaM, matrixref<float> vM, matrixref<float> dwM, float thisscale, float vhscale,
    int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize) {
    //delta: delta signal below activation fn transposed. nBands*nKernels*poolSize * batchSize
    //v: layer input transposed. nPrevKernels*nPrevBands * batchSize
    //dw: wieght delta matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
    float *delta = &deltaM(0,0);
    float *dw = &dwM(0,0);
    float *v = &vM(0,0);

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
    int nBands = (nPrevBands) / poolingBandShift;
    int dwRows = nPrevKernels*(filterSize+1);
    int N = dwRows * nBands*nKernels;
    int deltaInc = deltaM.getcolstride(); //nBands*nKernels*poolSize;
    int vInc = vM.getcolstride(); // nPrevKernels*nPrevBands;
            
    for(; xIndex < N; xIndex += blockDim.x * gridDim.x) { 

        int iPrevKernel = xIndex % nPrevKernels;
        int iWeightBand = (xIndex/nPrevKernels) % (filterSize+1);
        int iBand = (xIndex / (nPrevKernels*(filterSize+1))) % nBands;
        int iKernel = xIndex/ (nPrevKernels*(filterSize+1)*nBands );

        float a = 0;

        //int nPoolingShifts = 0;
        
        for(int iShift = 0; iShift < poolSize; iShift++) {
            int iPrevBand;
            if (iWeightBand == filterSize)
                iPrevBand = nPrevBands-1;
            else {
                iPrevBand = iBand*poolingBandShift + iShift - filterSize/2 - poolSize/2 + iWeightBand;
                if (iPrevBand < 0 || iPrevBand >= nPrevBands -1)
                    continue;
            }

            //nPoolingShifts++;

            float *pDelta = delta + iShift * nKernels*nBands + iKernel*nBands + iBand;
            
            float *pV = v + iPrevBand * nPrevKernels + iPrevKernel;
            float *pVEnd = pV + batchSize * vInc;
        
            //for(int iSample = 0; iSample < batchSize ; iSample++) {
            //    a += delta[iSample* nKernels*nBands*poolSize + iShift * nKernels*nBands + iBand * nKernels + iKernel]
            //    * v[iSample * nPrevKernels*nPrevBands + iPrevBand * nPrevKernels + iPrevKernel];
            //}
            while( pV < pVEnd) {
                a += *pDelta * *pV;
                pDelta += deltaInc;
                pV += vInc;
            }
        }

        //dw[xIndex] = a; // / nPoolingShifts;
        float& dwf = dwM(xIndex%dwRows, xIndex/dwRows);
        //assert(!isnan(a) && !isnan(dwf));// != 0x7fffffff);
        dwf = dwf*thisscale + a * vhscale; // / nPoolingShifts;
        }
    }

    __global__ void 
    __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
    ComputeCnnDeltaW_new(matrixref<float> deltaM, matrixref<float> vM, matrixref<float> dwM, const float thisscale, const float vhscale,
        const int nPrevBands, const int batchSize, const int poolSize, const int poolingBandShift) {
        //delta: delta signal below activation fn transposed. nBands*nKernels*poolSize * batchSize
        //v: layer input transposed. nPrevKernels*nPrevBands * batchSize
        //dw: wieght delta matrix. nPrevKernels*(filterSize+1) * nBands*nKernels
        float *delta = &deltaM(0,0);
        float *v = &vM(0,0);

        // replacing parameters and variables with grid and block variables to get register count down
        #define nKernels gridDim.y 
        #define nBands gridDim.x
        #define nPrevKernels blockDim.x
        #define filterSizePlus1 blockDim.y
        #define filterSize (blockDim.y-1)
        #define iBand blockIdx.x
        #define iKernel blockIdx.y
        #define iPrevKernel threadIdx.x
        #define iWeightBand threadIdx.y

        // reconstruct a single index from the multi dimesional inputs
        //unsigned int xIndex = threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*(blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y))); 
        
        //unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
        //int nBands = (nPrevBands) / poolingBandShift;
        //const int dwRows = nPrevKernels*(filterSize+1);
        //const int N = nPrevKernels*filterSizePlus1*nBands*nKernels;
        const int deltaInc = deltaM.getcolstride(); //nBands*nKernels*poolSize;
        const int vInc = vM.getcolstride(); // nPrevKernels*nPrevBands;

                
        //for(; xIndex < N; xIndex += blockDim.x * gridDim.x) { 
        //if (xIndex < N)
        {
            //int iPrevKernel = xIndex % nPrevKernels;
            //int iWeightBand = (xIndex/nPrevKernels) % (filterSize+1);
            //int iBand = (xIndex / (nPrevKernels*(filterSize+1))) % nBands;
            //int iKernel = xIndex/ (nPrevKernels*(filterSize+1)*nBands );
    
            float a = 0;
    
            //int nPoolingShifts = 0;
            
            for(int iShift = 0; iShift < poolSize; iShift++) {
                int iPrevBand;
                if (iWeightBand == filterSize)
                    iPrevBand = nPrevBands-1;
                else {
                    iPrevBand = iBand*poolingBandShift + iShift - filterSize/2 - poolSize/2 + iWeightBand;
                    if (iPrevBand < 0 || iPrevBand >= nPrevBands -1)
                        continue;
                }
    
                //nPoolingShifts++;
    
                float *pDelta = delta + iShift * nKernels*nBands + iKernel*nBands + iBand;
                
                float *pV = v + iPrevBand * nPrevKernels + iPrevKernel;
                float *pVEnd = pV + batchSize * vInc;
            
                //for(int iSample = 0; iSample < batchSize ; iSample++) {
                //    a += delta[iSample* nKernels*nBands*poolSize + iShift * nKernels*nBands + iBand * nKernels + iKernel]
                //    * v[iSample * nPrevKernels*nPrevBands + iPrevBand * nPrevKernels + iPrevKernel];
                //}
                while( pV < pVEnd) {
                    a += *pDelta * *pV;
                    pDelta += deltaInc;
                    pV += vInc;
                }
            }
    
            //dw[xIndex] = a; // / nPoolingShifts;
            float& dwf = dwM(iPrevKernel + nPrevKernels*iWeightBand, iBand + nBands*iKernel);
            //assert(!isnan(a) && !isnan(dwf));// != 0x7fffffff);
            dwf = dwf*thisscale + a * vhscale; // / nPoolingShifts;
            }
        #undef nKernels //gridDim.y 
        #undef nBands //gridDim.x
        #undef nPrevKernels //blockDim.x
        #undef filterSizePlus1 //blockDim.y
        #undef filterSize //(blockDim.y-1)
        #undef iBand //blockIdx.x
        #undef iKernel //blockIdx.y
        #undef iPrevKernel //threadIdx.x
        #undef iWeightBand //threadIdx.y
        
        }

    static const int TILE_DIM = 32;
    static const int BLOCK_ROWS = 16;
    // Each block transposes a tile of TILE_DIM x TILE_DIM elements
    // using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
    // TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

    __global__ void Transpose (const matrixref<float> in, matrixref<float> out)
    {
        //__shared__ float tile[TILE_DIM][TILE_DIM+1];  // the +1 on the second dimension is to prevent bank conflicts

        int col = blockIdx.x * TILE_DIM + threadIdx.x;
        // limit check
        if (col >= in.cols())
            return;

        int row = blockIdx.y * TILE_DIM + threadIdx.y; 
        // limit check
        if (row >= in.rows())
            return;

        // check element to ensure it's in range
        if (col >= in.cols() || row >= in.rows())
            return;

        for (int i=0; i<TILE_DIM && row+i < in.rows(); i+=BLOCK_ROWS) 
        {
            out(col, row+i) = in(row+i, col);
        }
        return;
#if 0
        int colOut = row;
        int rowOut = col;

        // special case the common one 
        if (TILE_DIM == BLOCK_ROWS)
        {
            tile[threadIdx.y][threadIdx.x] = in(row,col);
            __syncthreads();
            out(rowOut, colOut) = tile[threadIdx.x][threadIdx.y];
        }
        else
        {
            // populated shared memory tile, checking for end edge of matrix
            for (int i=0; i<TILE_DIM && row+i < in.rows(); i+=BLOCK_ROWS) 
            {
                tile[threadIdx.y+i][threadIdx.x] = in(row+i,col);
            }

            __syncthreads();

            for (int i=0; i<TILE_DIM && colOut+i < out.cols(); i+=BLOCK_ROWS)
            {
                out(rowOut, colOut+i) = tile[threadIdx.x][threadIdx.y+i];
            }
        }
#endif
    }

    // transpose a matrix
    void cudamatrixops::transpose(cudamatrixops & out) const
    {
        assert(rows() == out.cols());
        assert(cols() == out.rows());

        // dimensions for transpose
        dim3 grid((cols()+TILE_DIM-1)/TILE_DIM, (rows()+TILE_DIM-1)/TILE_DIM);
        dim3 threads(TILE_DIM,BLOCK_ROWS);

        //this->dump("input pre-transpose");
        Transpose<<<grid, threads>>>(*this, out);
        //out.dump("transposed matrix");
    }

    // convolution back propogation delta computations
    void cudamatrixops::computeCnnDeltaW(const cudamatrixops & deltaM, const cudamatrixops & vM, cudamatrixops & deltatM, cudamatrixops & vtM, float thisscale, float vhscale, 
        size_t nPrevBands, size_t nPrevKernels, size_t poolingBandShift, size_t nKernels, size_t batchSize, size_t poolSize, size_t filterSize)
    {   // make sure all the dimensions are what we are expecting
        assert(vM.rows() == batchSize);
        int nBands = (nPrevBands) / poolingBandShift;
        assert(deltaM.rows() == nBands*nKernels*poolSize);
        assert(rows() == nPrevKernels*(filterSize+1));

        // transpose the matricies to make ComputeCnnDeltaW more efficient
        vM.transpose(vtM);
        //deltaM.transpose(deltatM);
#if 0 // showed minimal gain (10%)
    dim3 block(nPrevKernels, filterSize+1);
    dim3 grid(nBands, nKernels);

    //deltaM.dump("delta matrix");
    //this->dump("pre convolution");
    ComputeCnnDeltaW_new<<<grid, block, 0, GetCurrentStream()>>>(deltaM, vtM, *this, thisscale, vhscale,
        nPrevBands, batchSize, poolSize, poolingBandShift);
#else
        const int blockSize = 256;
        int gridSize = (nPrevBands*nKernels*poolSize*batchSize+blockSize-1)/blockSize;

        // don't let it exceed the limit 64*1024 
        // if limited the kernel loops internally until done
        while (gridSize >= 64*1024-1)
        {
            gridSize = (gridSize+1)/2;
        }
        //deltaM.dump("delta matrix");
        //this->dump("pre convolution");
        ComputeCnnDeltaW_old<<<gridSize, blockSize, batchSize*sizeof(float), GetCurrentStream()>>>(deltaM, vtM, *this, thisscale, vhscale,
            nPrevBands, nPrevKernels, poolingBandShift, nKernels, batchSize, poolSize, filterSize);
#endif        
        //this->dump("post convolution");
        checklaunch ("ComputeCnnDeltaW");        
    }


    //find the max value of each group of poolSize elements, and store it in the output matrix
    // it also stores the maxIndex so back propogation knows which element to update
    // in - dimensions are nBands*nKernels*poolSize * batchSize
    // out, maxIndex - dimensions are nBands*nKernels * batchSize
    __global__ void maxpoolForwardi (const matrixref<float> in, matrixref<float> out, matrixref<float> maxIndex, size_t poolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        // get two dimension coordinates
        // column major, so rows increment fastest
        size_t i = blockIdx.y;
        size_t j = threadIdx.x + (blockIdx.x * blockDim.x);

        // out of range check
        if (j >= minibatchsize)
            return;
        
        // find the maximum value in the given poolSize
        float maxValue = -FLT_MAX; //FLT_MIN;
        size_t maxIdx = 0;
        
        size_t iIn = i;       
        for (size_t ip=0;ip<poolSize; ++ip, iIn+=bands*kernels) 
        {
            const float value = in(iIn, j);
            if (value > maxValue)
            {
                maxValue = value;
                maxIdx = iIn;
            }
        }

        // transfer the max value to the output matrix
        out(i, j) = maxValue;

        // save the index for the back propogation
        maxIndex(i,j) = maxIdx;
    }

    void cudamatrixops::maxpoolForward(cudamatrixops & out, cudamatrixops & maxIndex, size_t poolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        assert(rows() == out.rows()*poolSize);
        assert(out.rows() == bands*kernels);
        assert (cols() == out.cols());
        const int blockSize = 256;
        size_t minibatchBlocks = (minibatchsize + blockSize-1)/blockSize;
        dim3 grid(minibatchBlocks, bands*kernels);

        maxpoolForwardi<<<grid, blockSize, 0, GetCurrentStream()>>> (*this, out, maxIndex, poolSize, bands, kernels, minibatchsize);
        //out.dump("maxpool output");
        //maxIndex.dump("maxIndex out");
        checklaunch ("maxpoolForward");        
    }
        //find the max value of each group of poolSize elements, and store it in the output matrix
    // it also stores the maxIndex so back propogation knows which element to update
    // in - dimensions are nBands*nKernels*poolSize * batchSize
    // out, maxIndex - dimensions are nBands*nKernels*nsubbands * batchSize
    __global__ void submaxpoolForwardi (const matrixref<float> in, matrixref<float> out, matrixref<float> maxIndex, size_t subpoolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        // get two dimension coordinates
        // column major, so rows increment fastest
        size_t i = blockIdx.y;
        size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        size_t k = blockIdx.z;

        // out of range check
        if (j >= minibatchsize)
            return;
        
        // find the maximum value in the given poolSize
        //BUGBUG below??, FLT_MIN is a min. positive value, not the lowest number float can represent. 
        // Was this rectifier-like op intentional? Probably not as is incorrect for functions having also negative domain...
        //float maxValue = FLT_MIN; 
        float maxValue = -FLT_MAX;
        size_t maxIdx = 0;
        
        size_t iIn = i + k*(subpoolSize*bands*kernels);       
        for (size_t ip=0;ip<subpoolSize; ++ip, iIn+=bands*kernels) 
        {
            const float value = in(iIn, j);
            if (value > maxValue)
            {
                maxValue = value;
                maxIdx = iIn;
            }
        }

        // transfer the max value to the output matrix
        out(i+ k*(bands*kernels), j) = maxValue;

        // save the index for the back propogation
        maxIndex(i+ k*(bands*kernels),j) = maxIdx;
    }
    void cudamatrixops::submaxpoolForward(cudamatrixops & out, cudamatrixops & maxIndex, size_t poolSize, size_t subpoolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        size_t nsubbands = poolSize/subpoolSize;
        assert(rows() == out.rows()*subpoolSize);
        assert(out.rows() == bands*kernels*nsubbands);
        assert (cols() == out.cols());	// 256
        const int blockSize = 256;
        size_t minibatchBlocks = (minibatchsize + blockSize-1)/blockSize;
        //dim3 block(1,blockSize);
        dim3 grid(minibatchBlocks, bands*kernels, nsubbands);

        submaxpoolForwardi<<<grid, blockSize, 0, GetCurrentStream()>>> (*this, out, maxIndex, subpoolSize, bands, kernels, minibatchsize);
        //out.dump("maxpool output");
        //maxIndex.dump("maxIndex out");
        checklaunch ("submaxpoolForward");        
    }
    // find the max value of each group of poolSize elements, and store it in the output matrix
    // it also stores the maxIndex so back propogation knows which element to update
    // in, maxIndex - dimensions are batchSize * nBands*nKernels
    // out - dimensions are batchSize * nBands*nKernels*poolSize
    __global__ void maxpoolBacki (const matrixref<float> in, matrixref<float> out, matrixref<float> maxIndex, size_t poolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        // get two dimension coordinates
        // column major, so rows increment fastest
        size_t i = blockIdx.y;
        size_t j = threadIdx.x + (blockIdx.x * blockDim.x);

        // out of range check
        if (j >= minibatchsize)
            return;

        // zero out all the values in the pool
        size_t iOut = i;       
        for (size_t ip=0;ip<poolSize; ++ip, iOut+=bands*kernels) 
        {
            out(iOut,j) = 0.0;
        }

        // update the error delta values only for the max value
        out(maxIndex(i,j),j) = in(i,j);
    }

    
    void cudamatrixops::maxpoolBack(cudamatrixops & out, const cudamatrixops & maxIndex, size_t poolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        //assert(this->cols()*poolSize == out.cols());
        //assert(this->cols() == bands*kernels);
        //assert (rows() == out.rows());
        assert(rows()*poolSize == out.rows());
        assert(rows() == bands*kernels);
        assert (cols() == out.cols());
        const int blockSize = 256;
        size_t minibatchBlocks = (minibatchsize + blockSize-1)/blockSize;
        dim3 grid(minibatchBlocks, bands*kernels);

        //this->dump("maxpoolBack input");
        //maxIndex.dump("maxIndex matrix");
        maxpoolBacki<<<grid, blockSize, 0, GetCurrentStream()>>> (*this, out, maxIndex, poolSize, bands, kernels, minibatchsize);
        //out.dump("maxpoolBack output"); will be dumped by transpose
        checklaunch ("maxpoolBack");
    }
        __global__ void submaxpoolBacki (const matrixref<float> in, matrixref<float> out, matrixref<float> maxIndex, size_t poolSize, size_t subpoolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
         size_t nsubbands = poolSize/subpoolSize;
        // get two dimension coordinates
        size_t i = blockIdx.y;
        size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        size_t k = blockIdx.z;

        // out of range check
        if (j >= minibatchsize)
            return;

        // zero out all the values in the pool
        size_t iOut = i + k*(bands*kernels*subpoolSize);       
        for (size_t ip=0;ip<subpoolSize; ++ip, iOut+=bands*kernels) 
        {
            out(iOut,j) = 0.0;
        }

        // update the error delta values only for the max value
        out(maxIndex(i+ k*(bands*kernels),j),j) = in(i+ k*(bands*kernels),j);
    }

    
    void cudamatrixops::submaxpoolBack(cudamatrixops & out, const cudamatrixops & maxIndex, size_t poolSize, size_t subpoolSize, size_t bands, size_t kernels, size_t minibatchsize)
    {
        size_t nsubbands = poolSize/subpoolSize;
        assert(rows()*subpoolSize == out.rows());
        assert(out.rows() == bands*kernels*poolSize);
        assert (cols() == out.cols());	// 256
        const int blockSize = 256;
        size_t minibatchBlocks = (minibatchsize + blockSize-1)/blockSize;
        //dim3 block(blockSize, nsubbands);
        dim3 grid(minibatchBlocks, bands*kernels, nsubbands);

        //this->dump("maxpoolBack input");
        //maxIndex.dump("maxIndex matrix");
        submaxpoolBacki<<<grid, blockSize, 0, GetCurrentStream()>>> (*this, out, maxIndex, poolSize, subpoolSize, bands, kernels, minibatchsize);
        //out.dump("maxpoolBack output"); will be dumped by transpose
        checklaunch ("submaxpoolBack");
    }
    // dump the values of the matrix m to the stdout on the host
    // only supported for CUDA 2.0 and higher
    __global__ void dumpi (matrixref<float> m, char* name)
    {
#if defined(DEBUG) //defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        for(int i=0;i < m.rows();i++)
        {
            for(int j=0;j < m.cols();j++)
                printf ("%.10g\t",  m(i,j));
            printf ("\n");
        }
#endif
    }
   
    void cudamatrixops::dump(char *name) const
    {
#if defined(DEBUG)// defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        static bool checkedSize = false;
        if (!checkedSize)
        {
            size_t sizeOfBuff;
            cudaDeviceGetLimit(&sizeOfBuff,cudaLimitPrintfFifoSize);
            if (sizeOfBuff < 1024*1024*1536) // check against 1.5GB
            {
                cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*1536);
            }
            checkedSize = true;
        }
        printf ("\n###### begin - %s ######\n", name);
        printf ("###### (%d, %d) - stride %d ######\n", rows(), cols(), getcolstride());
        fflush(stdout);
        dumpi<<<1, 1, 0, GetCurrentStream()>>> (*this, name);
        checklaunch ("dump");
        cudaDeviceSynchronize();
        printf ("\n###### end ######\n");
        fflush(stdout);
#endif
    }

        //both m1 and m2 are passed in normal form
    __global__ void KhatriRaoProducti (matrixref<float> us, const matrixref<float> m1, const matrixref<float> m2)
    {
        const size_t k = threadIdx.x + (blockIdx.x * blockDim.x);
        if (k >= us.cols())
            return;

        size_t jj = 0;
        for (size_t j=0; j<m2.rows(); j++) 
        {
            for (size_t i=0; i<m1.rows(); i++)
            {
                us(jj++, k) = m1(i,k) * m2(j,k);
            }
        }
    }

    void cudamatrixops::KhatriRaoProduct (const cudamatrixops & m1, const cudamatrixops & m2)
    {
        assert(m1.cols() == m2.cols());
        assert (rows() == m1.rows() * m2.rows());

        KhatriRaoProducti<<<dim3((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, m1, m2);
        checklaunch ("KhatriRaoProduct");
    }

    __global__ void reshapecolumnproducti (matrixref<float> hnew, const matrixref<float> eh, const matrixref<float> h, const bool isehtransposed)
    {
        const size_t t = threadIdx.x + (blockIdx.x * blockDim.x);
        if (t >= hnew.cols())
            return;

        if (isehtransposed)
        {
            // find nrows and ncols of the reshpaed eh
            size_t nrows = h.rows();
            size_t ncols = eh.rows() / nrows;

            size_t k=0;
            for (size_t j=0; j<ncols; j++)   // row and col is transposed
            {
                hnew(j,t) = 0.0f;
                for (size_t i=0; i<nrows; i++)
                {
                    hnew(j,t) += eh(k,t) * h(i,t);
                    k++;
                }
            }         
        }
        else
        {
            size_t ncols = h.rows();
            size_t nrows = eh.rows() / ncols;

            size_t k=0;
            for (size_t j=0; j<ncols; j++)
            {
                for (size_t i=0; i<nrows; i++)
                {
                    if (j == 0) 
                        hnew(i,t) = eh(k,t) * h(j,t);
                    else 
                        hnew(i,t) += eh(k,t) * h(j,t);
                    k++;
                }
            }
        }
    }

    void cudamatrixops::reshapecolumnproduct (const cudamatrixops & eh, const cudamatrixops & h, const bool isehtransposed)
    {
        assert(eh.cols() == h.cols() && cols() == h.cols());
        assert (eh.rows() == h.rows() * rows());

        reshapecolumnproducti<<<dim3((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, eh, h, isehtransposed);
        checklaunch ("reshapecolumnproduct");
    }

    // this = sigmoid(this)
    __global__ void sigmoidij (matrixref<float> us) // thread = one per coordinate (yeah, not too efficient)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
        if (i >= us.rows() || j >= us.cols())
            return;
        float exponent = us(i,j);
        if (exponent < -30.0f)
            us(i,j) = 0.0f;
        else
            us(i,j) = 1.0f / (1.0f + expf (-exponent));
    }
    void cudamatrixops::sigmoid()
    {
        dim3 t (32,16);
        dim3 b ((((unsigned int)rows())+t.x-1)/t.x,(((unsigned int)cols())+t.y-1)/t.y);
        sigmoidij<<<b, t, 0, GetCurrentStream()>>> (*this);
        checklaunch ("sigmoid");
    }

    // samplebinary --return 1.0 with a probability of P(i,j)
    __device__ int rand (unsigned long & holdrand)    // from rand.c
    {
        return( ((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0x7fff );
    }
    __global__ void samplebinaryt (matrixref<float> us, matrixref<float> P, unsigned int randomseed)
    {   // one thread is one column --highly inefficient re memory access
        const size_t t = threadIdx.x + (blockIdx.x * blockDim.x);
        if (t >= us.cols())
            return;
        unsigned long holdrand = randomseed + (unsigned int) t;     // srand()
        for (size_t i = 0; i < us.rows(); i++)
        {
            float randval = rand (holdrand) / (float) 0x7fff;       // RAND_MAX=0x7fff
            float bit = randval < P(i,t) ? 1.0f : 0.0f;
            us(i,t) = bit;
        }
    }
    void cudamatrixops::samplebinary (const cudamatrixops & P, unsigned int randomseed)
    {
        assert (rows() == P.rows() && cols() == P.cols());
        samplebinaryt<<<dim3((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, P, randomseed);
        checklaunch ("samplebinary");
    }

    // addtoallcolumns
    __global__ void addtoallcolumnsi (matrixref<float> us, const matrixref<float> other)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // fetch the vector
        float otheri = other(i,0);
        // add to 'this'
        for (size_t j = 0; j < us.cols(); j++)
            us(i,j) += otheri;
    }
    void cudamatrixops::addtoallcolumns (const cudamatrixops & other)
    {
        assert (rows() == other.rows() && other.cols() == 1);
        // Note: We could be WAY more efficient by parallelizing into blocks. Later.
        addtoallcolumnsi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, other);
        checklaunch ("addtoallcolumns");
    }

    // llstats
    // Compute LL stats for pre-training per node over a range of frames (will be summed up to a single number on CPU side).
    __global__ void llstatsi (const matrixref<float> v, const matrixref<float> v1, matrixref<float> logllsums, bool gaussian)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= v.rows())
            return;
        float logllsumi = 0.0f;
        // add to 'this'
        for (size_t t = 0; t < v.cols(); t++)
        {
            float logll;
            if (gaussian)
            {
                float diff = v(i,t) - v1(i,t);
                logll = -0.5f * diff * diff;         // note that we assume unit variance
                // We normalize against the 'perfect reconstruction' hypothesis (diff == 0)
                // thus the Gaussian normalization factor (1/sqrt (2.0 * M_PI)) cancels out.
            }
            else
            {
                float Pv = v(i,t);     // prob of v being 1
                if (Pv < 0.000001f) Pv = 0.000001f;   // to be sure (not observed)
                if (Pv > 0.999999f) Pv = 0.999999f;
                float Pv1 = v1(i,t);   // prob of v1 being 1
                if (Pv1 < 0.000001f) Pv1 = 0.000001f;   // we do see 1.0
                if (Pv1 > 0.999999f) Pv1 = 0.999999f;
                logll = Pv * log (Pv1) + (1 - Pv) * log (1 - Pv1);
                // normalize against perfect reconstruction hypothesis for better readability
                logll -= Pv * log (Pv) + (1 - Pv) * log (1 - Pv);
            }
            logllsumi += logll;
        }
        logllsums(i,0) = logllsumi;
    }
    void cudamatrixops::llstats (const cudamatrixops & v1, cudamatrixops & logllsums, bool gaussian) const
    {
        assert (rows() == v1.rows() && cols() == v1.cols());
        assert (logllsums.rows() == rows() && logllsums.cols() == 1);
        llstatsi<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, v1, logllsums, gaussian);
        checklaunch ("llstats");
    }

    // this = softmax(this)
    __device__ void softmaxjsum (size_t j, matrixref<float> us, float & rsum, float & rcolmax) // thread = one per column
    {
        // find max (to bring exp() into comfortable range)
        float colmax = 0.0f;
        for (size_t i = 0 ; i < us.rows(); i++)
        {
            float usij = us(i,j);
            if (usij > colmax)
                colmax = usij;
        }
        // sum
        // we divide by exp (colmax), which will cancel out when we normalize below
#if 1
        // It was observed that accuracy here makes a notable difference compared to using a 'double' sum.
        // We compute 32 parallel sums, which should increase our accuracy by 5 bits.
        // Turns out this is actually worse--slower convergence since less noisy??
        // TODO: This takes 32 x 32 = 4 kb of registers--likely to be in RAM and slow. We could smartly parallelize those 32 across 32 sub-threads.
        float sums[32] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (size_t i = 0 ; i < us.rows(); i++)
        {
            float usexp = expf (us(i,j)-colmax);
            us(i,j) = usexp;
            sums[i % 32] += usexp;
        }
        // aggregate the 32 parallel sums
        float sum = 0.0f;
        for (size_t k = 0; k < 32; k++)
            sum += sums[k];
#else
        float sum = 0.0f;
        for (size_t i = 0 ; i < us.rows(); i++)
        {
            float usexp = expf (us(i,j)-colmax);
            us(i,j) = usexp;
            sum += usexp;
        }
#endif
        rsum = sum;
        rcolmax = colmax;
    }
    __global__ void softmaxj (matrixref<float> us) // thread = one per column
    {
        const size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        if (j >= us.cols())
            return;
        // compute the exp of value - colmax
        float sum, colmax;
        softmaxjsum (j, us, sum, colmax);
        // normalize
        for (size_t i = 0 ; i < us.rows(); i++)
            us(i,j) /= sum;
    }

    // performs softmax for one frame t; one thread out of 32 with relative index s0
    // Returns the log of the sum of the original exp(.) terms (without colmax adjustment).
    __device__ float softmaxs0t (size_t s0, size_t sstepincrement, size_t t, matrixref<float> us)
    {
        // #cols blocks, 32 threads in a block
        // Each block processes one column. Each thread processes 1/32 of the components. The code below runs parallel for 32 interleaved stripes of one column.
        // TODO: too many blocks? #cols = #frames, up to 1024 in practical terms.
        //       If too much then split it into a 2D structure. I put the tests for 't' overflow alread in (they are not needed at present).
        const size_t numsteps = (us.rows() + sstepincrement -1) / sstepincrement;

        // colmax computation
        // We subtract the column max before taking the exp(.) for numerical stability. This is critical.

        // initialize
        __shared__ float colmaxs[32];   // 32=sstepincrement
        colmaxs[s0] = -1e30f;
        __syncthreads();

        // colmax
        // We run 32 of these in parallel, and loop over stripes of height 32.
        if (t < us.cols()) for (size_t step = 0; step < numsteps; step++)
        {
            const size_t s = step * sstepincrement + s0;        // s0 = thread index
            if (s < us.rows())
            {
                float us_st = us(s,t);                          // coalesced mem access w.r.t. s
                if (us_st > colmaxs[s0])
                    colmaxs[s0] = us_st;
            }
        }
        __syncthreads();

        // now aggregate the colmaxs into one
        // We do this on all 32 threads redundantly, giving the same result in each. MP-local mem, so this is quick.
        float colmax = colmaxs[0];
        for (size_t s = 1; s < 32; s++)
            if (colmaxs[s] > colmax)
                colmax = colmaxs[s];

        // summation

        // initialize
        __shared__ float sums[32];   // 32=sstepincrement
        sums[s0] = 0.0f;
        __syncthreads();

        // perform summation
        if (t < us.cols()) for (size_t step = 0; step < numsteps; step++)
        {
            const size_t s = step * sstepincrement + s0;        // s0 = thread index
            if (s < us.rows())
            {
                float us_st = us(s,t);                          // coalesced mem access w.r.t. s
                float usexp = expf (us_st-colmax);
                us(s,t) = usexp;                                // update with exp value (coalesced write)
                sums[s0] += usexp;                              // accumulate
            }
        }
        __syncthreads();

        // now aggregate the sums into one
        // We do this on all 32 threads redundantly, giving the same result in each. MP-local mem, so this is quick.
        float sum = sums[0];
        for (size_t s = 1; s < 32; s++)
            sum += sums[s];

        // perform the normalization
        if (t < us.cols()) for (size_t step = 0; step < numsteps; step++)
        {
            const size_t s = step * sstepincrement + s0;        // s0 = thread index
            if (s < us.rows())
                us(s,t) /= sum;
        }

        // return it
        // All 32 threads return the same value here (we will only use one).
        const float logsumj = logf (sum) + colmax;
        return logsumj;
    }

    // efficient
    // TODO: clean out all the inefficient versions.
    __global__ void softmaxt (matrixref<float> us) // thread = one per column
    {
        const size_t t = blockIdx.x;

        const size_t sstepincrement = blockDim.x;       // 32
        const size_t s0 = threadIdx.x;                  // 0..sstepincrement-1

        // compute all softmax values
        softmaxs0t (s0, sstepincrement, t, us);
    }

    void cudamatrixops::softmax()
    {
#if 1
        // Each multi-proc runs one column. Each thread processes 32 interleaved rows.
        dim3 blockdim = dim3 ((unsigned int)cols());  // width = #cols
        softmaxt<<<blockdim/*blocks*/, 32/*threads per block*/, 0, GetCurrentStream()>>> (*this);
#else
        softmaxj<<<dim3 ((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this);
#endif
        checklaunch ("softmax");
    }
#if 1
    // softmax is broken up into two devices, each holding half of the output vector
    //  - step 1: softmax for this part of a vector, and return the sum for exchange with the other device
    //            The sum is exchanged as a log due to numerical issues.
    //     - a: determine colmax and compute exponent
    //     - b: normalization
    //     - c: final data conversion: copy sum over as logsum, using colmax
    //  - step 2: update the sum from the information from the second device
    // The sums that are exchanged are stored as columns, one column per device, rows (!) are frames.
    // This is unusual, but needed since our dev-to-dev copy function can only correctly handle full-height stripes.

    // inefficient
    __global__ void stripedsoftmaxstep1j (matrixref<float> us, matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (used as buffer)*/)
    {
        // cols/32 blocks, 32 threads in a block
        const size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        if (j >= us.cols())
            return;
        // compute the exp of value - colmax
        float sum, colmax;
        softmaxjsum (j, us, sum, colmax);
        partialsumvectors(j,0) = sum;
        partialsumvectors(j,1) = colmax;
    }

    // efficient
    // TODO: clean out all the inefficient versions.
    __global__ void stripedsoftmaxt (matrixref<float> us, matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (not touched here)*/)
    {
        const size_t t = blockIdx.x;

        const size_t sstepincrement = blockDim.x;       // 32
        const size_t s0 = threadIdx.x;                  // 0..sstepincrement-1

        // compute all softmax values and return the denominator (as a log to avoid numerics nastiness)
        const float logsumj = softmaxs0t (s0, sstepincrement, t, us);

        // remember it for our caller, so we can later reassemble multiple stripes
        if (s0 == 0 && t < us.cols())               // all threads have the same values, we write back only one
            partialsumvectors(t,0) = logsumj;       // partialsumvectors(.,0) is our own sum
    }

    // inefficient version
    __global__ void stripedsoftmaxstep1jb (matrixref<float> us, const matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (used as buffer)*/)
    {
        const size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        if (j >= us.cols())
            return;
        const float sum = partialsumvectors(j,0);
        // normalize
        for (size_t i = 0 ; i < us.rows(); i++)
            us(i,j) /= sum;
    }
    // efficient (?) version
    __global__ void stripedsoftmaxstep1ib (matrixref<float> us, const matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (used as buffer)*/)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < us.cols(); t++)
        {
            const float sum = partialsumvectors(t,0);
            us(i,t) /= sum;
        }
    }
    __global__ void stripedsoftmaxstep1jc (matrixref<float> us, matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (used as buffer)*/)
    {
        const size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        if (j >= us.cols())
            return;
        const float sum = partialsumvectors(j,0);
        const float colmax = partialsumvectors(j,1);
        const float logsumj = logf (sum) + colmax;
        partialsumvectors(j,0) = logsumj;     // partialsumvectors(.,0) is our own sum
    }

    // CPU wrapper
    void cudamatrixops::stripedsoftmaxstep1 (cudamatrixops & partialsumvectors)
    {
#if 1
        // Each multi-proc runs one column. Each thread processes 32 interleaved rows.
        dim3 blockdim = dim3 ((unsigned int)cols());  // width = #cols
        stripedsoftmaxt<<<blockdim/*blocks*/, 32/*threads per block*/, 0, GetCurrentStream()>>> (*this, partialsumvectors);
#else
        // cols/32 blocks, 32 threads in a block
        stripedsoftmaxstep1j<<<dim3 ((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, partialsumvectors);
        //stripedsoftmaxstep1jb<<<dim3 ((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, partialsumvectors);
        // normalize by 'sum' (in partialsumvectors)
        stripedsoftmaxstep1ib<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, partialsumvectors);
        // copy back log sum in partialsumvectors (this is a trival kernel)
        stripedsoftmaxstep1jc<<<dim3 ((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, partialsumvectors);
#endif
        checklaunch ("stripedsoftmaxstep1");
    }

    // ineffcient one
    __global__ void stripedsoftmaxstep2j (matrixref<float> us, matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (not used in this step)*/)
    {
        const size_t j = threadIdx.x + (blockIdx.x * blockDim.x);
        if (j >= us.cols())
            return;
        // update the normalization using the sum from the other device
        //  - multiply again with the sum over the part
        //  - divide instead by the sum of the two part sums
        // partialsumvectors(.,1) is the other device's
        //   * sum0 / (sum0+sum1)
        // = * exp (logsum0) / (exp (logsum0) + exp (logsum1))
        // = * 1 / (exp (logsum0-logsum0) + exp (logsum1 - logsum0))
        // = * 1 / (1 + exp (logsum1 - logsum0))
        const float rescale = 1.0f / (1.0f + (expf (partialsumvectors(j,1) - partialsumvectors(j,0))));
        for (size_t i = 0 ; i < us.rows(); i++)
            us(i,j) = us(i,j) * rescale;
    }
    // effcient one
    __global__ void stripedsoftmaxstep2i (matrixref<float> us, matrixref<float> partialsumvectors/*col 0 = us, col 1 = other (not used in this step)*/)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        //  update the normalization using the sum from the other devices(more than 2)[v-xiexie]
        //  calculate a scale and multiply it on the local 'softmax' result to get the correct softmax.
        //   * sum0 / (sum0 + sum1 + ... sumN)
        // = * exp (logsum0) / (exp(logsum0) + exp (logsum1) + ... exp (logsumN))
        // = * 1 / (1 + exp(logsum1-logsum0) + ... exp(logsumN - logsum0))
        // compute for all columns
        for (size_t t = 0; t < us.cols(); t++)
        {
#if 1       // for more than 2 cuda devices in top layer. [v-xieche]
            float sum = 1.0;
            // skip itself. [v-xieche]
            for (size_t devid = 1; devid < partialsumvectors.cols(); devid ++)
                sum += expf (partialsumvectors(t,devid) - partialsumvectors(t,0)); 
            const float rescale = 1.0f / sum;
#else       // for 2 cuda devices in top layer.
            const float rescale = 1.0f / (1.0f + (expf (partialsumvectors(t,1) - partialsumvectors(t,0))) );
#endif
            us(i,t) = us(i,t) * rescale;
        }
    }
    void cudamatrixops::stripedsoftmaxstep2 (cudamatrixops & partialsumvectors)
    {
        // assert (2 == partialsumvectors.cols());
        //stripedsoftmaxstep2j<<<dim3 ((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, partialsumvectors);
        stripedsoftmaxstep2i<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, partialsumvectors);
        checklaunch ("stripedsoftmaxstep2");
    }
#endif

    // sethessianvectorsignal
    // TODO make this more efficient (more threads)
    __global__ void sethessianvectorsignali(matrixref<float> us, const matrixref<float> Pu, const matrixref<float> forwardStatistics)
    {
        const size_t j = blockIdx.x;
        if (j >= us.cols())
            return;
        float dotproduct = 0.0f;
        // compute inner product
        for (size_t i = 0; i < us.rows(); i++)
            dotproduct += Pu(i,j) * forwardStatistics(i,j);
        for (size_t i = 0; i < us.rows(); i++)
            us(i,j) = (forwardStatistics(i,j) - dotproduct) * Pu(i,j);
    }


    void cudamatrixops::sethessianvectorsignal (const cudamatrixops & Pu, const cudamatrixops & forwardStatistics)
    {
        sethessianvectorsignali<<<dim3 ((unsigned int) cols()), 1, 0, GetCurrentStream()>>> (*this, Pu, forwardStatistics);
        checklaunch ("sethessianvectorsignal");
    }

    // TODO make this more efficient (more threads)!
    __global__ void setdiagonalpreconditioneri(matrixref<float> us, matrixref<float> gradientsquared, float nobservations, float lambda, float alpha)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            float gradientsquaredij = gradientsquared(i,j);
            us(i,j) = std::pow(gradientsquaredij / nobservations + lambda, alpha);
        }
    }

    void cudamatrixops::setdiagonalpreconditioner(const cudamatrixops &gradient, float nobservations, float lambda, float alpha)
    {
        setdiagonalpreconditioneri<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, gradient, nobservations, lambda, alpha);
    }

    // TODO make this more efficient !
    __global__ void elementwisedivisioni(matrixref<float> us, const matrixref<float> a, const matrixref<float> b)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            us(i,j) = a(i,j) / b(i,j);
        }
    }

    void cudamatrixops::elementwisedivision(const cudamatrixops &a, const cudamatrixops &b)
    {
        elementwisedivisioni<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, a, b);
    }

    // TODO make this more efficient !
    __global__ void elementwisesquarei(matrixref<float> us, const matrixref<float> a)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            us(i,j) = a(i,j) * a(i,j);
        }
    }

    void cudamatrixops::elementwisesquare(const cudamatrixops &a)
    {
        assert(rows() == a.rows());
        assert(cols() == a.cols());
        elementwisesquarei<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, a);
    }


    __global__ void weighteddoti(matrixref<float> us, const matrixref<float> weightingmatrix, const matrixref<float> a, size_t nrows, size_t ncols, float *result)
    {
        __shared__ float temp[32];
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        //temp[threadIdx.x] = index < dim ? a(index,0) * b(index,0) : 0.0f;
        temp[threadIdx.x] = 0.0f;
        if (i < nrows)
        {
            for (size_t j = 0; j < ncols; j++)
            {
                temp[threadIdx.x] += us(i,j) * weightingmatrix(i,j) * a(i,j);
            }
            
        }
        __syncthreads();
        if( 0 == threadIdx.x ) {
            float sum = 0.0f;
            for ( int i= 0; i < 32; i++ )
                sum += temp[i];
            atomicAdd( result , sum );
        }
    }

    float cudamatrixops::weighteddot(const cudamatrixops& weightingmatrix, const cudamatrixops& a) const
    {

        assert(weightingmatrix.rows() == a.rows());
        assert(weightingmatrix.cols() == a.cols());
        assert(rows() == a.rows());
        assert(cols() == a.cols());

        float result, *dev_result;
        //result = (float*) malloc(sizeof(float));
        cudaMalloc( (void**)&dev_result, sizeof(float) );
        cudaMemset(dev_result, 0, sizeof(float));
        weighteddoti<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, weightingmatrix, a, rows(), cols(), dev_result);
        cudaMemcpy( &result, dev_result, sizeof(float) , cudaMemcpyDeviceToHost);
        cudaFree(dev_result);
        return result;

    }

    //zhaorui drop the frames whose error for lable senone is a special value (-100.0f)
     __global__ void dropbadframei (matrixref<float> us, const matrixref<float> uids)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < us.cols(); t++)
        {
            const size_t uid = (size_t) uids(0,t);
            
            if( us(uid,t) == -100.0f )
                us(i,t) = 0.0f;
        }
     }

    // setbackpropagationerrorsignal
    // this = delta((Pu(i,t)==uids[t]) - Pu
    // This is the error of the top layer, the signal being back-propagated.
    // Pu may be a stripe (when running parallelized). i0 is the actual base coordinate.
    // zhaorui pass frame dropping thresh to error calculation funtion
    __global__ void setbackpropagationerrorsignali (matrixref<float> us, const matrixref<float> uids, const matrixref<float> Pu, 
                                                    const size_t i0, const matrixref<float> senone2keepmodelupdate, const float framedropthresh)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < us.cols(); t++)
        {
            const size_t uid = (size_t) uids(0,t);
            const float utarget_it = (i + i0 == uid) ? 1.0f : 0.0f;
            const float latpp = Pu(i,t);
            //zhaorui if the posterior of the label senone is less than "framedropthresh",  set the error to be a special value and will drop this frame later. 
            if (framedropthresh < 1.0f && i + i0 == uid && latpp < framedropthresh )          
                             //zhaorui uid: label senone
            {
                us(i,t) = -100.0f;                
            }
            else
                us(i,t) = utarget_it - latpp;
            if (senone2keepmodelupdate.cols() != 0)
            {
                if (senone2keepmodelupdate(0,uid) == 0.0f || senone2keepmodelupdate(0,i) == 0.0f)       // set to zero if it is not kept for update
                    us(i,t) = 0;
            }
#ifdef NO_SIL_UPDATE        // reset to zeros if relates to silence, this makes no change to silence [v-hansu]
            if (us.rows() == 9304)
            {
                if (uid == 7670 || uid == 7671 || uid == 7672 || i == 7670 || i == 7671 || i == 7672)
                    us(i,t) = 0;
            }
#endif
        }
    }
    // zhaorui pass frame dropping thresh to error calculation funtion
    void cudamatrixops::setbackpropagationerrorsignal (const cudamatrixops & uids, const cudamatrixops & Pu, const size_t i0, const cudamatrixops & senone2keepmodelupdate, const float framedropthresh)
    {
        assert (cols() == uids.cols() && cols() == Pu.cols());
        assert (rows() == Pu.rows() && uids.rows() == 1);
        setbackpropagationerrorsignali<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, uids, Pu, i0, senone2keepmodelupdate,framedropthresh);
        checklaunch ("setbackpropagationerrorsignalit");
        dropbadframei<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, uids); 
        checklaunch ("dropbadframei");
    }

    // setbackpropagationerrorsignalklreg
    // this = delta((Pu(i,t)==(1-alpha)*uids[t]+alpha*refPu - Pu
    // This is the error of the top layer, the signal being back-propagated.
    __global__ void setbackpropagationerrorsignaliwithklreg (matrixref<float> us, const matrixref<float> uids, const matrixref<float> Pu, const matrixref<float> refPu, const float alpha)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const float oneminusalpha=1.0f-alpha;
        for (size_t t = 0; t < us.cols(); t++)
        {
            const size_t uid = (size_t) uids(0,t);
            float utarget_it = (i == uid) ? 1.0f : 0.0f;
            if (alpha>0) utarget_it = oneminusalpha * utarget_it + alpha * refPu(i,t);
            us(i,t) = utarget_it - Pu(i,t);
        }
    }

    void cudamatrixops::setbackpropagationerrorsignalwithklreg (const cudamatrixops & uids, const cudamatrixops & Pu, const cudamatrixops & refPu, const float alpha)
    {
        assert (cols() == uids.cols() && cols() == Pu.cols());
        assert (rows() == Pu.rows() && uids.rows() == 1);
        assert (alpha>=0 && alpha<=1);
        setbackpropagationerrorsignaliwithklreg<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, uids, Pu, refPu, alpha);
        checklaunch ("setbackpropagationerrorsignalitwithklreg");
    }

    
    // setbackpropagationerrorsignalklreg
    // errorsettingmode 0 : this(s,t) = (s==uids[t]) - (1-hsmoothingweight) * Pu(s,t) - hsmoothingweight * gammas(s,t)
    // errorsettingmode 1 : this(s,t) = (1-hsmoothingweight) * ((s==uids[t]) - Pu(s,t)) + hsmoothingweight * errors(s,t)
    // errorsettingmode 2 : this(s,t) = (1-hsmoothingweight) * ((s==uids[t]) - Pu(s,t)) * Pu(s,t) + hsmoothingweight * errors(s,t)
    // zhaorui pass frame dropping thresh to error calculation funtion
    __global__ void setbackpropagationerrorsignalhsmoothingi (matrixref<float> us, const matrixref<float> uids, const matrixref<float> Pu, const matrixref<float> refmat, const float hsmoothingweight, const size_t errorsettingmode, const float framedropthresh)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < us.cols(); t++)
        {
            const size_t uid = (size_t) uids(0,t);
            const float utarget_it = (i == uid) ? 1.0f : 0.0f;
            if (errorsettingmode == 0)
            {                
                const float latpp  = refmat(i,t);
                //zhaorui if the posterior of the label senone is less than "framedropthresh",  set the error to be a special value and will drop this frame later. 
                if (framedropthresh < 1.0f && i == uid && latpp < framedropthresh )                                 
                {                    
                    us(i,t) = -100.0f;
                    /*printf("bad score %.10g\n", latpp);         
                    printf("frame %u\n", t);
                    printf("uid %u\n", uid);*/
                }
                    
                else
                    us(i,t) = utarget_it - (1 - hsmoothingweight) * Pu(i,t) - hsmoothingweight * latpp;
            }
            else if (errorsettingmode == 1)
                us(i,t) = (1 - hsmoothingweight) * (utarget_it - Pu(i,t)) + hsmoothingweight * refmat(i,t);
            else if (errorsettingmode == 2)
                us(i,t) = (1 - hsmoothingweight) * (utarget_it - Pu(i,t)) * Pu(i,t) + hsmoothingweight * refmat(i,t);
        }
    }

    // zhaorui pass frame dropping thresh to error calculation funtion
    void cudamatrixops::setbackpropagationerrorsignalhsmoothing (const cudamatrixops & uids, const cudamatrixops & Pu, const cudamatrixops & refmat, const float hsmoothingweight, const size_t errorsettingmode, const float framedropthresh)
    {
        assert (cols() == uids.cols() && cols() == Pu.cols());
        assert (rows() == Pu.rows() && uids.rows() == 1);
        assert (hsmoothingweight>=0 && hsmoothingweight<=1);
        assert (errorsettingmode <= 2); // we only support two modes now
        setbackpropagationerrorsignalhsmoothingi<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, uids, Pu, refmat, hsmoothingweight, errorsettingmode,framedropthresh);
        checklaunch ("setbackpropagationerrorsignalhsmoothingi");
        dropbadframei<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, uids);
        checklaunch ("dropbadframei");
    }

    // posteriorstats
    // special function for backprop: multiply error vector by derivative of sigmoid function.
    // We leverage that the derivative can be computed from values of the sigmoid function in 'simg' cheaply.
    __global__ void mulbydsigmi (matrixref<float> eh, const matrixref<float> sigm)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= eh.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < eh.cols(); t++)
            //eh(i,t) *= 1e-5 + sigm(i,t) * (1.0f - sigm(i,t));
            eh(i,t) *= sigm(i,t) * (1.0f - sigm(i,t));
    }
    void cudamatrixops::mulbydsigm (const cudamatrixops & sigm)
    {
        assert (cols() == sigm.cols() && cols() == sigm.cols());
        mulbydsigmi<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, sigm);
        checklaunch ("mulbydsigm");
    }
     // mulbydlru()
    // special function for backprop: multiply error vector by derivative of LRU function
    // which amounts to setting values to zero which had a non-positive z,
    // which in turn can be tested by testing the output value lru(z) for > 0.
    __global__ void mulbydlrui (matrixref<float> eh, const matrixref<float> lruvals)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= eh.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < eh.cols(); t++)
            if (lruvals(i,t) <= 0.0f)  // err = eh .* (h > 0) in place
                eh(i,t) = 0.0f;
    }
    void cudamatrixops::mulbydlru (const cudamatrixops & lruvals)
    {
        assert (cols() == lruvals.cols() && cols() == lruvals.cols());
        mulbydlrui<<<dim3 ((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, lruvals);
        checklaunch ("mulbydlru");
    }
    // special function for counting statistics of backpropagation (for progress tracking)
    // 'this' is row vector of ground truth indices.
    __global__ void posteriorstatst (const matrixref<float> uids, const matrixref<float> Pu, matrixref<float> /*out*/logpps, matrixref<float> /*out*/pps, matrixref<float> /*out*/fcors, bool nosoftmax)
    {
        const size_t t = threadIdx.x + (blockIdx.x * blockDim.x);   // parallel over frames --a small number, does not matter
        if (t >= uids.cols())
            return;
        // get index
        const size_t clsid = (size_t) uids(0,t);
        // get posterior probability for that index
        const float pp = Pu(clsid,t);
        // save it to result vectors
        pps(0,t) = nosoftmax ? 0.0f : pp;                           // nosoftmax: we don't have the pps; return 0
        logpps(0,t) = nosoftmax ? pp : logf (max (pp, 0.000001f));  // (avoid underflow if prob has been rounded to 0)
        fcors(0,t) = pp;                                            // save as initial value for fcors
    }
    __global__ void posteriorstatsi (const matrixref<float> uids, const matrixref<float> Pu, matrixref<float> /*out*/fcors, bool nosoftmax)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= Pu.rows())
            return;
        // compute for all columns
        const float zero = nosoftmax ? -1e30f : 0.0f;
        for (size_t t = 0; t < Pu.cols(); t++)
        {
            // When there is a probability equal or larger than the clsid's then clear fcors[t] to 0.
            // We do this in place. Note that this leads to all sorts of nasty race conditions.
            // Once we found a higher probability, fcors[t] gets set to 0.
            // That means that all subsequent tests will be positive.
            // This still works since we are only setting fcors[t] to 0, no other value. I.e. order does not matter.
            // (To accomodate for 'nosoftmax' mode, '0' above means 'zero' which is -1e30 for nosoftmax mode.)
            const size_t clsid = (size_t) uids(0,t);
            const float ppclst = fcors(0,t);    // either == Pu(clsid,t) or 0 (or log Pu in case of nosoftmax)
            const float ppit = Pu(i,t);
            if (i != clsid && ppit >= ppclst)   // a competing state is at least as good -> frame error, indicated by fcors[t] == 0
                if (ppclst != zero)             // (save the duplicate memory access)
                    fcors(0,t) = zero;
        }
    }
    void cudamatrixops::posteriorstats (const cudamatrixops & Pu, cudamatrixops & logpps, cudamatrixops & pps, cudamatrixops & fcors, bool nosoftmax) const
    {
        assert (cols() == Pu.cols());
        assert (rows() == 1);
        assert (cols() == logpps.cols() && cols() == pps.cols() && cols() == fcors.cols());
        assert (logpps.rows() == 1 && pps.rows() == 1 && fcors.rows() == 1);
        // step 1: copy out pp, logpp, and also copy pp to fcors
        posteriorstatst<<<dim3 ((((unsigned int)Pu.cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, Pu, logpps, pps, fcors, nosoftmax);        
        checklaunch ("posteriorstatst");
        // step 2: set pp[t] = 0 for every Pu(i,t) >= Pu(clsid,t)
        posteriorstatsi<<<dim3 ((((unsigned int)Pu.rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, Pu, fcors, nosoftmax);
        checklaunch ("posteriorstatsi");
    }


    __global__ void stripedposteriorstatst (const matrixref<float> uids, const matrixref<float> Pu, matrixref<float> /*out*/logpps, matrixref<float> /*out*/pps, matrixref<float> /*out*/fcors, size_t i0)
    {
        const size_t t = threadIdx.x + (blockIdx.x * blockDim.x);   // parallel over frames --a small number, does not matter
        if (t >= uids.cols())
            return;
        // get index
        const size_t clsid = (size_t) uids(0,t);
        // get posterior probability for that index
        if (clsid >= i0 && clsid < i0 + Pu.rows())      // range lies in [i0, i0+Pu.rows()]
        {
            const float pp = Pu(clsid - i0,t);
            // save it to result vectors
            pps(0,t) = pp;
            logpps(0,t) = logf (max (pp, 0.000001f));   // (avoid underflow if prob has been rounded to 0)
            fcors(0,t) = pp;             // save as initial value for fcors
        }
        else
        {
            pps(0,t) = 0.0f;
            logpps(0,t) = 0.0f;
            fcors(0,t) = 0.0f;
        }
    }
    __global__ void stripedposteriorstatsi (const matrixref<float> uids, const matrixref<float> Pu, matrixref<float> /*out*/fcors, size_t i0)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);      // consider bias when set the row index.[v-xieche]
        if (i >= Pu.rows())
            return;
        // compute for all columns
        for (size_t t = 0; t < Pu.cols(); t++)
        {
            // When there is a probability equal or larger than the clsid's then clear fcors[t] to 0.
            // We do this in place. Note that this leads to all sorts of nasty race conditions.
            // Once we found a higher probability, fcors[t] gets set to 0.
            // That means that all subsequent tests will be positive.
            // This still works since we are only setting fcors[t] to 0, no other value. I.e. order does not matter.
            const size_t clsid = (size_t) uids(0,t);
            const float ppclst = fcors(0,t);    // either == Pu(clsid,t) or 0
            const float ppit = Pu(i,t);
            if (i + i0 != clsid && ppit >= ppclst)   // a competing state is at least as good -> frame error, indicated by fcors[t] == 0
                if (ppclst != 0.0f)             // (save the duplicate memory access)
                    fcors(0,t) = 0.0f;
        }
    }

    void cudamatrixops::stripedposteriorstats (const cudamatrixops & Pu, cudamatrixops & logpps, cudamatrixops & pps, cudamatrixops & fcors, size_t i0) const
    {
        assert (rows() == 1);
        assert (logpps.cols() == pps.cols() && pps.cols() == fcors.cols());
        stripedposteriorstatst<<<dim3 ((((unsigned int)Pu.cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, Pu, logpps, pps, fcors, i0);
        checklaunch ("stripedposteriorstatst");
        stripedposteriorstatsi<<<dim3 ((((unsigned int)Pu.rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, Pu, fcors, i0);
        checklaunch ("stripedposteriorstatsi");
    }

    __global__ void gemsi (matrixref<float> us, float thisscale, matrixref<float> other, float otherweight)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            // this *= thisscale
            float usij;
            if (thisscale != 0.0f)   // if 0 we won't touch the input, so this can be used to set the variable
                usij = us(i,j) * thisscale;
            else
                usij = 0.0f;
            // this += other * otherweight
            if (otherweight != 0.0f)
                usij += other(i,j) * otherweight;
            us(i,j) = usij;
        }
    }
    // this <- this * thisscale + other * otherweight
    void cudamatrixops::gems (float thisscale, const cudamatrixops & other, float otherweight)
    {
        assert (cols() == other.cols() && rows() == other.rows());
        gemsi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, thisscale, other, otherweight);
        checklaunch ("gems");
    }

    __global__ void setto0ifabsbelowi (matrixref<float> us, float threshold)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            float usij;
            usij = us(i,j);
            usij *= (usij >= threshold || usij <= -threshold);
            us(i,j) = usij;
        }
    }
    // set to 0 all elements in 'this' that are below a threshold (abs value)
    void cudamatrixops::setto0ifabsbelow (float threshold)
    {
        setto0ifabsbelowi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, threshold);
        checklaunch ("setto0ifabsbelow");
    }

    __global__ void setto0ifabsbelowi2 (matrixref<float> us, const matrixref<float> ref, float threshold)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            float refij;
            refij = ref(i,j);
            us(i,j) *= (refij >= threshold || refij <= -threshold);
        }
    }
    // set to 0 all elements in 'this' for those in ref that are below a threshold (abs value)
    void cudamatrixops::setto0ifabsbelow2 (cudamatrixops &  ref, float threshold)
    {
        setto0ifabsbelowi2<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, ref, threshold);
        checklaunch ("setto0ifabsbelow2");
    }

    __global__ void setto0ifabsabovei2 (matrixref<float> us, const matrixref<float> ref, float threshold)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            float refij;
            refij = ref(i,j);
            us(i,j) *= (refij <= threshold && refij >= -threshold);
        }
    }
    // set to 0 all elements in 'this' for those in ref that are below a threshold (abs value)
    void cudamatrixops::setto0ifabsabove2 (cudamatrixops &  ref, float threshold)
    {
        setto0ifabsabovei2<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, ref, threshold);
        checklaunch ("setto0ifabsabove2");
    }
    __global__ void setto0ifbelowi (matrixref<float> us, float threshold)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            float usij;
            usij = us(i,j);
            if (usij < threshold)
                usij = 0.0f;
            us(i,j) = usij;
        }
    }
    // set to 0 all elements in 'this' that are below a threshold (e.g. for LRUs)
    void cudamatrixops::setto0ifbelow (float threshold)
    {
        setto0ifbelowi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, threshold);
        checklaunch ("setto0ifbelowi");
    }
    __global__ void setvaluei (matrixref<float> us, float value)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // set all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
            us(i,j) = value;
    }
    void cudamatrixops::setvalue (float value)
    {
        setvaluei<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, value);
        checklaunch ("setzero");
    }
        __global__ void sumblockconvi (matrixref<float> us, const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i>=filterSize*nPrevKernel)
            return;
    
        for(size_t j = 0; j < nKernel; j ++) // assign W as the average of the blocks.
        {
            for(size_t k = 1; k < poolSize; k ++)
                us(i, j ) += us(k*nPrevKernel + i, k*nKernel + j);
        }

    }
    __global__ void avgblockconvi (matrixref<float> us, const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i>=filterSize*nPrevKernel)
            return;
    
        for(size_t j = 0; j < nKernel; j ++) // assign W as the average of the blocks.
        {
                us(i, j) = us(i, j)/poolSize;
                for(size_t k = 1; k < poolSize; k ++)
                    us(k*nPrevKernel + i, k*nKernel + j ) = us(i, j);
        }

    }

    __global__ void sumblockconvbiasi (matrixref<float> us, const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i>=nKernel)
            return;

        for(size_t k = 1; k < poolSize; k ++)
                us(i , 0) += us(k*nKernel + i , 0);

    }
    __global__ void avgblockconvbiasi (matrixref<float> us, const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i>=nKernel)
            return;

        us(i, 0) = us(i, 0)/poolSize;
        for(size_t k = 1; k < poolSize; k ++)
                us(k*nKernel + i , 0) = us(i , 0);

    }

     __global__ void patchasblockconvi (matrixref<float> us, const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // set all columns
        
        const size_t m = us.cols();
        /* implementation 1; wrong because updated us won't be kept
        for (size_t j = 0; j < m; j++)
        {
            size_t poolIdx = j / nKernel;
            if ( i < poolIdx*nPrevKernel || i >= (poolIdx+filterSize)*nPrevKernel )
                us(i, j ) = 0;
            else{
                size_t kernelIdx = j % nKernel;
                if (poolIdx > 0 ){
                    size_t weightIdx = (i - poolIdx*nPrevKernel) % (filterSize*nPrevKernel);
                    us( weightIdx , kernelIdx ) += us( i , j);
                }
            }

        }
        */
        for (size_t j = 0; j < m; j++)
        {
            size_t poolIdx = j / nKernel;
            if ( i < poolIdx*nPrevKernel || i >= (poolIdx+filterSize)*nPrevKernel )
                us(i, j ) = 0;
            
        }

    }

    void cudamatrixops::patchasblockconv (const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        if (this->cols()>1){
            patchasblockconvi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            checklaunch ("patchasblockconv");
            sumblockconvi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            checklaunch ("sumblockconv");
        }
        else{
            sumblockconvbiasi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            checklaunch ("sumblockconvbias");
        }
        //averaged across all blocks
        if (this->cols()>1){
            avgblockconvi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            checklaunch ("avgblockconv");
            //distblockconvi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            //checklaunch ("distblockconv");
        }
        else{
            avgblockconvbiasi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            checklaunch ("avgblockconvbias");
            //distblockconvbiasi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
            //checklaunch ("distblockconvbias");
        }
        //averaged across all blocks
    }

    __global__ void accumulatesqri (matrixref<float> us, const matrixref<float> other, const size_t mbframes, const float keepweight)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // perform the op
        const size_t m = us.cols();
        const float mbframesf = (float) mbframes;
        for (size_t j = 0; j < m; j++)
            us(i,j) = keepweight * us(i,j) + (1.0f - keepweight) * other(i,j) * other(i,j) / mbframesf;      // square sum
    }
    // this += (other / N) .^ 2 * N   --for AdaGrad
    void cudamatrixops::accumulatesqr (const cudamatrixops & other, size_t mbframes, float keepweight)
    {
        accumulatesqri<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, other, mbframes, keepweight);
        checklaunch ("accumulatesqr");
    }

    __global__ void adadenomi (matrixref<float> us, const matrixref<float> sqracc, const float numframes, const size_t mbframes)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // perform the op
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
            us(i,j) = sqrt (sqracc(i,j) / numframes) * mbframes; // mul with mbframes to get it into original range; for diagnostics (will cancel out anyway)
    }
    // this = AdaGrad denom from sqracc = sqrt (sqracc(i,j) / numframes)
    void cudamatrixops::adadenom (const cudamatrixops & sqracc, float numframes, size_t mbframes)
    {
        adadenomi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, sqracc, numframes, mbframes);
        checklaunch ("adadenom");
    }

    __global__ void adagradientfromdenomi (matrixref<float> us, const matrixref<float> gradient, const float actualavdenom, const float targetavdenom)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // perform the op
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            const float denomij = us(i,j);      // (this was saved in 'us')
            float weight;
            if (denomij == 0.0f)                            // special case: denom = 0 (zero denominator) -> use average
                weight = 1.0f;
            else
            {
                weight = actualavdenom / denomij;           // we weight the gradient with avdenom / denomij
                if (weight > 10.0f)                         // clip the weight somewhat  --I saw outliers up to 100k+
                    weight = 10.0f;
                else if (weight < 0.01f)
                    weight = 0.01f;
            }
            weight *= targetavdenom / actualavdenom;        // and scale to the assumed targetavdenom
            us(i,j) = gradient(i,j) * weight; // (note: in-place update; used to store denomij here)
        }
    }
    // this = AdaGrad denom from sqracc = sqrt (sqracc(i,j) / numframes)
    void cudamatrixops::adagradientfromdenom (const cudamatrixops & gradient, float actualavdenom, float targetavdenom)
    {
        adagradientfromdenomi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, gradient, actualavdenom, targetavdenom);
        checklaunch ("adagradientfromdenom");
    }

    // dropout() --randomly set X% of values in this matrix to 0
    __global__ void dropouti (matrixref<float> us, float factor, unsigned int randomseed)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        unsigned long holdrand = randomseed + (unsigned int) i;     // srand()
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            float randval = rand (holdrand) / (float) 0x7fff;       // RAND_MAX=0x7fff
            if (randval < factor)
                us(i,j) = 0.0f;
        }
    }
    void cudamatrixops::dropout (float factor, unsigned int randomseed)
    {
        dropouti<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, factor, randomseed);
        checklaunch ("dropout");
    }

    // scale() --multiply a matrix in place with a factor
    __global__ void scalei (matrixref<float> us, float factor) 
    {   // one thread is one column --highly inefficient re memory access
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
            us(i,j) *= factor;
    }
    void cudamatrixops::scale (float factor)
    {
        scalei<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, factor);
        checklaunch ("scale");
    }

    __global__ void patchasblockdiagonali (matrixref<float> us, size_t diagblocks, bool averageblocks, size_t firstcol)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // compute for all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            size_t realj = j + firstcol;    // 'this' is a column stripe of a matrix. 'realj' is the actual column in the matrix
            // TODO: add code here
            realj += 0; // (avoid a compiler warning for now;; remove this)
            us(i,j) += 0.0f;    // example code how to access us(i,j) (remove this)
        }
    }
    // set all elements to 0 that are off block-diagonal; and average the diagonal blocks if 'averageblocks'
    // THIS IS BROKEN for multi-devices--we have no access to the other blocks!
    void cudamatrixops::patchasblockdiagonal (size_t diagblocks, bool averageblocks, size_t firstcol)
    {
        if (firstcol != 0)
            throw std::runtime_error ("patchasblockdiagonal: does not support multiple devices as of yet--meh!");
        patchasblockdiagonali<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, diagblocks, averageblocks, firstcol);
        checklaunch ("patchasblockdiagonal");
    }

    // ======================================================================
    // matrixaccumulator functions
    // ======================================================================

    __global__ void reseti (matrixref<double> us)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // set all columns
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
            us(i,j) = 0.0;
    }
    void cudamatrixaccumulatorops::reset()
    {
        reseti<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this);
        checklaunch ("reset");
    }

    __global__ void accumulatei (matrixref<double> us, double thisscale, const matrixref<float> other, const double otherweight)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // perform the op
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
        {
            double usij = thisscale != 0.0 ? us(i,j) : 0.0;
            usij += other(i,j);
            us(i,j) = usij;
        }
    }
    // this = thisscale * this + otherweight * other
    void cudamatrixaccumulatorops::accumulate (float thisscale, const cudamatrixops & other, float otherweight)
    {
        accumulatei<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, thisscale, other, otherweight);
        checklaunch ("accumulate");
    }

    __global__ void tomatrixi (const matrixref<double> us, matrixref<float> to)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= us.rows())
            return;
        // perform the op
        const size_t m = us.cols();
        for (size_t j = 0; j < m; j++)
            to(i,j) = (float) us(i,j);
    }
    // to = (float) this
    void cudamatrixaccumulatorops::tomatrix (cudamatrixops & to) const
    {
        tomatrixi<<<dim3((((unsigned int)rows())+31)/32), 32, 0, GetCurrentStream()>>> (*this, to);
        checklaunch ("tomatrix");
    }
    // compute euclidean norm in columnwise fashion (and optionally if maxcolnorm>0 renormalize it so we get column wise scales)
    __global__ void colwisenrm2i (const matrixref<float> us, matrixref<float> norms, const float maxcolnorm)
    {
        const size_t j =  threadIdx.x + (blockIdx.x * blockDim.x); 

        if (j >= us.cols())
            return;
        size_t rows = us.rows();
        float nsum=0.0;
        for (size_t i=0; i < rows; i++) {
            nsum += us(i,j)*us(i,j);
        }
        float norm = sqrt(nsum);
        // when maxcolnorm>0 column scaling factors are returned rather than the norms itself
        if (maxcolnorm>0.0 && norm>maxcolnorm) {
            norm = maxcolnorm / (norm + 1e-7);
        } else if (maxcolnorm>0.0)  {
            norm = 1.0f; //norm OK, do not scale
        } else { } //return norm, not the scaling factor

        norms(j,0) = norm;
    }
    void cudamatrixops::colwisenrm2 (cudamatrixops & norms, const float maxcolnorm) const
    {
        assert (cols() == norms.rows());
        colwisenrm2i<<<dim3((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, norms, maxcolnorm);
        checklaunch ("colwisenrm2");
    }


    // column wise scaling -- multiply each column by a number
    __global__ void scalecolwisei (matrixref<float> us, const matrixref<float> factors) 
    {   
        const size_t j =  threadIdx.x + (blockIdx.x * blockDim.x); 
        if (j >= us.cols())
            return;
        float factor = factors(j, 0);
        if (factor == 1.0f) 
            return;
        const size_t m = us.rows();
        for (size_t i = 0; i < m; i++) {
            us(i,j) *= factor;
        }
    }
    void cudamatrixops::scalecolwise (const cudamatrixops & factors)
    {
        assert (cols()==factors.rows());
        scalecolwisei<<<dim3((((unsigned int)cols())+31)/32), 32, 0, GetCurrentStream()>>> (*this, factors);
        checklaunch ("scalecolwise");
    }

#if 0
    struct tiling
    {
        dim3 b, t; // block and thread dimensions
        // set up the tiling for a given matrix
#define BLOCKSIZE 16        // currently we assume they are all the same, in order to do coalesced loading
#define HEIGHT BLOCKSIZE    // height of block we compute
#define WIDTH BLOCKSIZE     // width of block we compute
#define DEPTH BLOCKSIZE     // number of elements in dot products we compute
        tiling (size_t numrows, size_t numcols) : t ((unsigned int) HEIGHT, (unsigned int)WIDTH), b ((unsigned int) (numrows + HEIGHT -1) / HEIGHT, (unsigned int) (numcols + WIDTH -1) / WIDTH) {}
    };

    // 'assert'-like function to make kernel launch fail if its condition is violated
    __device__ float ensure (bool cond)
    {
        __shared__ float s;
        if (!cond)
            s = *(float*)-1;
        return s;
    }
    struct matrixpatch
    {
        // add one element to avoid bank conflicts (columns are no longer in the same bank)
        // difference: from 15 to 53 GFlops!
        float data[BLOCKSIZE][BLOCKSIZE+1];
        __device__ float &       operator() (size_t i, size_t j)       { /*ensure (i < BLOCKSIZE && j < BLOCKSIZE);*/ return data[i][j]; }
        __device__ const float & operator() (size_t i, size_t j) const { /*ensure (i < BLOCKSIZE && j < BLOCKSIZE);*/ return data[i][j]; }
    };

    __device__ float getval (size_t i, size_t j, const matrixref<float> & src, bool istransposed)
    {
        if (istransposed && j < src.rows() && i < src.cols())
        {
            //ensure (j < src.rows() && i < src.cols());
            return src(j,i); // coalesced w.r.t. j which comes from threadIdx.x
        }
        else if (!istransposed && i < src.rows() && j < src.cols())
        {
            //ensure (i < src.rows() && j < src.cols());
            return src(i,j);
        }
        else
            return 0.0f;
    }
    // executed in parallel on all coordinates (one thread each) within the bounding box, which is also the bounding box of the thread block
    // If the matrix is transposed, it will be flipped. dst(i,j) will contain the untransposed matrix.
    // TODO: we can actually fetch 4 times as much with float4 reads. Ideally 32 float4's (512 bytes) in a row for the dot product. Later.
    __device__ void getpatch (matrixpatch & dst, size_t i0, size_t j0, const matrixref<float> & src, bool istransposed)
    {
        const size_t i = i0 + (istransposed ? threadIdx.y : threadIdx.x); // assumption: thread block is square of dim BLOCKSIZE
        const size_t j = j0 + (istransposed ? threadIdx.x : threadIdx.y);
        float val = getval (i, j, src, istransposed);
        //ensure (threadIdx.x < BLOCKSIZE && threadIdx.x < BLOCKSIZE);
        //ensure (i-i0 < BLOCKSIZE && j-j0 < BLOCKSIZE);
        dst(i-i0,j-j0) = val;   // no bank conflict if transposed although i is constant because of BLOCKSIZE+1 in dimension of data[][]
    }
    // actual operations (-> matrixbase)
    // this = this * thisscale + A * B * ABweight, where A is stored as its transpose if 'Aistransposed'
    // TODO: should we also add an optional vector to each column? We will need that.
    // TODO: we may also need striping. Later.
    __global__ void gemmij (float thisscale, const matrixref<float> A, bool Aistransposed, const matrixref<float> B, float ABweight, matrixref<float> C)
    {
        //ensure (blockDim.x == BLOCKSIZE);
        //ensure (blockDim.y == BLOCKSIZE);
        // matrix dimensions
        const size_t rows = C.rows();   // == A.rows()
        const size_t cols = C.cols();   // == B.cols()
        const size_t depth = A.cols();  // == B.rows()
        // target patch
        const size_t i0 = blockIdx.x * BLOCKSIZE;
        const size_t i1 = min (i0 + BLOCKSIZE, C.rows());
        const size_t j0 = blockIdx.y * BLOCKSIZE;
        const size_t j1 = min (j0 + BLOCKSIZE, C.cols());
        const size_t irel = threadIdx.x;
        const size_t jrel = threadIdx.y;
        const size_t i = irel + i0;
        const size_t j = jrel + j0;
        float Cij = 0.0f;      // we got BLOCKSIZE^2 of these in a register
        // we are computing rectangle [i0,i1) x [j0,j1), and this thread computes C(i,j)
        // That means we are using rows [i0,i1) of A and columns [j0,j1) of B.
        for (size_t k0 = 0; k0 < depth; k0 += BLOCKSIZE)
        {
            const size_t k1 = min (k0 + BLOCKSIZE, depth);
            // get patches
            __shared__ matrixpatch Apatch, Bpatch;
            getpatch (Apatch, i0, k0, A, Aistransposed);    // A[i0..i1-1][k0..k1-1]
            //__syncthreads();
            // TODO: this causes access conflicts; better transpose one of the patches
            getpatch (Bpatch, k0, j0, B, false);            // B[k0..k1-1][j0..j1-1]
            __syncthreads();
            // perform matrix product
            // Each shared-memory location is used BLOCKSIZE=16 times.
            // This computes out-of-bounds elements C(i,j), but their values are 0, and it happens rarely, so save the check.
            // TODO: avoid -i0 etc. -> irel, jrel, krel (should save registers)
            const size_t k1mk0 = k1 - k0;
#pragma unroll BLOCKSIZE
            for (size_t krel = 0; krel < k1mk0; krel++)
                Cij += Apatch(irel,krel) * Bpatch(krel,jrel);
            __syncthreads();    // needed to avoid next iteration to overwrite Apatch/Bpatch
        }
        // apply weighting factor
        Cij *= ABweight;
        // add original value if requested (coalesced read)
        if (thisscale != 0.0f)
        {
            if (i < rows && j < cols)
                Cij += C(i,j) * thisscale;
            __syncthreads();    // (needed?)
        }
        // store it (coalesced write)
        if (j < cols)
        {
            if (i < rows)
                C(i,j) = Cij;
            else if (i < C.getcolstride())  // padding values --keep them 0
                C(i,j) = 0.0f;
        }
    }
    void cudamatrixops::gemm (float thisscale, const cudamatrixops & A, bool Aistransposed, const cudamatrixops & B, float ABweight)
    {
        assert (rows() == A.rows() && cols() == B.cols() && A.cols() == B.rows());
        tiling dims (rows(), cols());
        gemmij<<<dims.b,dims.t, 0, GetCurrentStream()>>> (thisscale, A, Aistransposed, B, ABweight, *this);
    }

    // for testing: add one matrix to the other
    __global__ void operatorplus (matrixref<float> A, matrixref<float> B)
    {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < A.rows() && j < A.cols())
            A(i,j) += B(i,j);
    }
    void cudamatrixops::operator+= (const cudamatrixops & other)
    {
        assert (rows() == other.rows() && cols() == other.cols());
        tiling dims (rows(), cols());
        operatorplus<<<dims.b,dims.t, 0, GetCurrentStream()>>> (*this, other);
    }
#endif
};};
#pragma pop_macro ("atomicAdd")