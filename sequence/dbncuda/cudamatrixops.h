// cudamatrixops.h -- cudamatrixbase class, which contains the actual math ops (device-local)
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudamatrixops.h $
// 
// 48    8/19/13 9:40p Ruizhao
// add function to remove "bad" frames in sequential DNN training
//
// 47    1/09/13 3:29p V-hansu
// add setbackpropagationerrorsignalhsmoothing() and related kernel to
// prepare for cuda based hsmoothing
// 
// 46    12/07/12 5:16a Adame
// convolution/maxpool support
// 
// 45    11/27/12 4:10p V-hansu
// add senone2keepmodelupdate to setbackpropagationerrorsignal()
// 
// 44    11/04/12 7:52a Fseide
// new class matrixaccumulator
// 
// 43    10/29/12 3:46p T-simonw
// add dot product and nrm methods (CUBLAS)
// add weighteddot method
// add elementwisedivision  and elementwisesquare methods
// add special function for hessian-vector product computation:
// sethessianvectorsignal 
// add special function for preconditioned conjugate gradient:
// setdiagonalpreconditioner
// 
// 42    10/16/12 11:25a Fseide
// two new methods dropout() and scale(), for implementing Hinton's
// drop-out method
// 
// 41    10/11/12 7:41p V-hansu
// (add a space)
// 
// 40    10/10/12 9:59a Dongyu
// added support to train models that shares the same hidden layers but
// use different senone sets from different langauges. This allows us to
// train universal ASR with separate senonoes or use models trained using
// multiple languages to adapt to new langauges.
// 
// 39    9/26/12 9:43p V-hansu
// change setzero() to setvalue()
// 
// 38    9/24/12 3:26p Fseide
// adadenom() no longer takes numsummands, as it has become obsolete
// 
// 37    9/24/12 3:03p Fseide
// updated adagradientdenom() to now take two avdenoms, the actual (for
// clipping) and the target (for scaling)
// 
// 36    9/21/12 3:33p Fseide
// implemented nosoftmax option for posteriorstats()
// 
// 35    9/18/12 11:16a Fseide
// new method adagradientfromdenom()
// 
// 34    9/18/12 11:06a Fseide
// implemented asum() and adadenom()
// 
// 33    9/18/12 10:06a Fseide
// accumulatesqr() now implements the IIR filter, and has a new argument
// 'keepweight' for that
// 
// 32    9/17/12 6:18p Fseide
// implemented accumulatesqr()
// 
// 31    9/16/12 4:35p Fseide
// new function setzero()
// 
// 30    6/08/12 9:32p V-xieche
// delete code related to delayupdate.
// 
// 29    4/05/12 9:51p V-xieche
// add code for accumulate prior and posteriorstats in striped toplayer
// pipeline training. not finished yet.
// 
// 28    4/01/12 7:13a V-xieche
// disabled _p2p copying
// 
// 27    4/01/12 8:47p Fseide
// changed assignmatrix_ua() to assignmatrix_p2p() using explicit
// contexts;
// new methods for that: getdevicecontext()
// 
// 26    4/01/12 4:47p Fseide
// added code for peer-to-peer access without UA, but not enabled yet
// 
// 25    4/01/12 2:05p Fseide
// seterrorsignal now takes an offset parameter so that it can work for
// vertical stripes
// 
// 24    4/01/12 11:23a V-xieche
// add code for striped softmax computation in 2 gpu.
// 
// 23    3/31/12 8:24p Fseide
// new method assignmatrix_ua()
// 
// 22    2/25/12 5:24p V-xieche
// Add helpler functions for coping date in CUDA device
// 
// 21    11/28/11 5:55p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 20    11/23/11 1:14p Dongyu
// add KhatriRaoProducti
// 
// 19    11/04/11 14:58 Fseide
// added new argument 'otherweight' to addrowsum() to allow unscaled
// gradients w.r.t. momentum
// 
// 18    10/28/11 14:52 Fseide
// cleaned up confusing and inconsistent alpha and beta parameters in
// gemm-like functions, now calling them 'thisscale' and 'otherweight' to
// make it crystal-clear
// 
// 17    10/25/11 5:16p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 16    10/06/11 5:16p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 15    6/21/11 13:40 Fseide
// added frame for new function patchasblockdiagonal(), but inner loop not
// implemented yet
// 
// 14    3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 13    2/26/11 4:50p Fseide
// new method softmax()
// 
// 12    2/24/11 9:50p Fseide
// new methods assign() and fetch(), to allow for non-contiguous transfers
// 
// 11    2/10/11 1:53p Fseide
// new method posteriorstats() (although it does not work correctly yet)
// 
// 10    2/10/11 11:32a Fseide
// new method mulbydsigm()
// 
// 9     2/10/11 11:17a Fseide
// new method setbackpropagationerrorsignal()
// 
// 8     2/07/11 9:52p Fseide
// llstats() implemented
// 
// 7     2/07/11 7:08p Fseide
// new method addtoallcolumns()
// 
// 6     2/07/11 6:52p Fseide
// implemented samplebinary()
// 
// 5     2/07/11 6:28p Fseide
// added rowsum() and sigmoid()
// 
// 4     2/02/11 8:22a Fseide
// gemm() now allows B to be transposed as well
// 
// 3     2/01/11 4:53p Fseide
// addcol() removed
// 
// 2     2/01/11 3:42p Fseide
// added gems() and addcol()
// 
// 1     1/31/11 3:31p Fseide
// created

#pragma once

#include "cudabasetypes.h"
#include <cuda.h>
namespace msra { namespace cuda {

// implements actual matrix operations on matrixref
class cudamatrixops : public matrixref<float>
{
public:
    // memory transfer
    void cudamatrixops::assign (size_t i0, size_t i1, size_t j0, size_t j1, const float * pi0j0, size_t colstride);
    void cudamatrixops::fetch (size_t i0, size_t i1, size_t j0, size_t j1, float * pi0j0, size_t colstride) const;
    //void fetchtodevice(float * p_dst, size_t memsize);
#ifdef  _WIN64
    //static void assignmatrix_p2p (cudamatrixops & dst, CUcontext dstContext, const cudamatrixops & src, CUcontext srcContext);
    static void assignmatrix_ua (cudamatrixops & dst, const cudamatrixops & src);
#endif

    // CUBLAS-based functions
    void gemm (float thisscale, const cudamatrixops & A, bool Aistransposed, const cudamatrixops & B, bool Bistransposed, float ABweight);
    void gems (float thisscale, const cudamatrixops & other, float otherweight);

    float dot(const cudamatrixops & b) const; 
    float nrm2() const; 
    void colwisenrm2(cudamatrixops & norms, float maxcolnorm) const;
    // self-implemented CUDA kernels
    void setto0ifabsbelow (float threshold);
    void setto0ifabsbelow2 (cudamatrixops &  ref, float threshold);
    void setto0ifabsabove2 (cudamatrixops &  ref, float threshold);
    void patchasblockdiagonal (size_t diagblocks, bool averageblocks, size_t firstcol);
    void patchasblockconv (size_t nPrevBand, size_t nPrevKernel, size_t nKernel, size_t poolSize, size_t filterSize);
    void addrowsum (float thisscale, const cudamatrixops & othercols, float otherweight);
    void addrowsumpool (float thisscale, const cudamatrixops & othercols, float otherweight, size_t poolSize, size_t bands, size_t kernels);

    // convolution model
    void reorder (cudamatrixops & to, size_t minibatchSize, size_t kernels, size_t bands, bool input) const;
    void convolutionForward(cudamatrixops & out, const cudamatrixops & weight, const cudamatrixops & bias, size_t minibatchSize, size_t kernels, size_t bands, size_t newKernels, size_t poolingBandShift, size_t poolSize, size_t filterSize);
    void computeCnnDeltaW(const cudamatrixops & deltaM, const cudamatrixops & vM, cudamatrixops & deltatM, cudamatrixops & vtM, float thisscale, float vhscale, size_t nPrevBands, size_t nPrevKernels, size_t poolingBandShift, size_t nKernels, size_t batchSize, size_t poolSize, size_t filterSize);
    void maxpoolForward(cudamatrixops & out, cudamatrixops & maxIndex, size_t poolSize, size_t bands, size_t kernels, size_t minibatchsize);
    void submaxpoolForward(cudamatrixops & out, cudamatrixops & maxIndex, size_t poolSize, size_t subpoolSize, size_t bands, size_t kernels, size_t minibatchsize);
    void maxpoolBack(cudamatrixops & out, const cudamatrixops & maxIndex, size_t poolSize, size_t bands, size_t kernels, size_t minibatchsize);
    void submaxpoolBack(cudamatrixops & out, const cudamatrixops & maxIndex, size_t poolSize, size_t subpoolSize, size_t bands, size_t kernels, size_t minibatchsize);
    void transpose(cudamatrixops & out) const;
    void dump(char* name) const;
    void sigmoid();
    void samplebinary (const cudamatrixops & P, unsigned int randomseed);
    void addtoallcolumns (const cudamatrixops & other);
    void llstats (const cudamatrixops & v1, cudamatrixops & logllsums, bool gaussian) const;
    void softmax();
#if 1  // add for striped softmax function[v-xieche]
    void stripedsoftmaxstep1 (cudamatrixops &partialsumvectors);
    void stripedsoftmaxstep2 (cudamatrixops &partialsumvectors);
#endif
    void sethessianvectorsignal (const cudamatrixops & Pu, const cudamatrixops & forwardStatistics);
    void setdiagonalpreconditioner(const cudamatrixops &gradientsquared, float nobservations, float lambda, float alpha);
    void elementwisedivision(const cudamatrixops &a, const cudamatrixops &b);
    void elementwisesquare(const cudamatrixops &a);
    float weighteddot(const cudamatrixops &weightingmatrix, const cudamatrixops &a) const;
    // zhaorui pass frame dropping thresh to error calculation funtion
    void setbackpropagationerrorsignal (const cudamatrixops & uids, const cudamatrixops & Pu, const size_t i0, const cudamatrixops & senone2keepmodelupdate, const float framedropthresh);
    void setbackpropagationerrorsignalwithklreg (const cudamatrixops & uids, const cudamatrixops & Pu, const cudamatrixops & refPu, const float alpha);
    // zhaorui pass frame dropping thresh to error calculation funtion
    void setbackpropagationerrorsignalhsmoothing (const cudamatrixops & uids, const cudamatrixops & Pu, const cudamatrixops & refmat, const float hsmoothingweight, const size_t errorsettingmode, const float framedropthresh);
    void mulbydsigm (const cudamatrixops & sigm);
    void mulbydlru (const cudamatrixops & lruvals);
    void posteriorstats (const cudamatrixops & Pu, cudamatrixops & logpps, cudamatrixops & pps, cudamatrixops & fcors, bool nosoftmax) const;
    void stripedposteriorstats (const cudamatrixops & Pu, cudamatrixops & logpps, cudamatrixops & pps, cudamatrixops & fcors, size_t i0) const;
    void KhatriRaoProduct(const cudamatrixops & m1, const cudamatrixops & m2);
    void reshapecolumnproduct (const cudamatrixops & eh, const cudamatrixops & h, const bool isehtransposed);
    void setvalue (float value);
    void accumulatesqr (const cudamatrixops & other, size_t mbframes, float keepweight);
    void adadenom (const cudamatrixops & sqracc, float numframes, size_t mbframes);
    float asum() const;
    void adagradientfromdenom (const cudamatrixops & gradient, float actualavdenom, float targetavdenom);
    void addweighted (const cudamatrixops & other, float weight);
    void dropout (float factor, unsigned int randomseed);
    void scale (float factor);
    void scalecolwise (const cudamatrixops & factors);
    void setto0ifbelow (float threshold);
};

// implements actual matrixaccumulatpr operations on matrixref
class cudamatrixaccumulatorops : public matrixref<double>
{
public:
    void reset();
    void accumulate (float thisscale, const cudamatrixops & other, float otherweight);
    void tomatrix (cudamatrixops & to) const;
};

};};
