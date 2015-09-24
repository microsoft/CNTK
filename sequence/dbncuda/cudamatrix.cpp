// cudamatrix.cpp -- matrix with CUDA execution; no CUDA calls or #includes inside this source file
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudamatrix.cpp $
// 
// 98    8/19/13 9:18p Ruizhao
// add function to remove "bad" frames in sequential DNN training
//
// 97    3/13/13 6:00 Fseide
// fixed signed/unsigned error
// 
// 96    1/09/13 3:29p V-hansu
// add setbackpropagationerrorsignalhsmoothing() and related kernel to
// prepare for cuda based hsmoothing
// 
// 95    12/07/12 5:15a Adame
// convolution/maxpool support
// 
// 94    11/27/12 4:10p V-hansu
// add senone2keepmodelupdate to setbackpropagationerrorsignal()
// 
// 93    11/04/12 7:52a Fseide
// new class matrixaccumulator
// 
// 92    11/02/12 4:31p T-simonw
// code formatting and documentation
// 
// 91    10/29/12 3:50p T-simonw
// add dot product, nrm2, elementwise operations, weighteddot, and special
// purpose method (sethessianvectorsignal)
// 
// 90    10/16/12 11:25a Fseide
// two new methods dropout() and scale(), for implementing Hinton's
// drop-out method
// 
// 89    10/11/12 3:45p V-hansu
// (add a space after the setvalue function)
// 
// 88    10/10/12 9:57a Dongyu
// added support to train models that shares the same hidden layers but
// use different senone sets from different langauges. This allows us to
// train universal ASR with separate senonoes or use models trained using
// multiple languages to adapt to new langauges.
// 
// 87    9/27/12 12:28a V-hansu
// change setzero into setvalue
// 
// 86    9/24/12 3:25p Fseide
// adadenom() no longer takes numsummands, as it has become obsolete
// 
// 85    9/24/12 3:03p Fseide
// updated adagradientdenom() to now take two avdenoms, the actual (for
// clipping) and the target (for scaling)
// 
// 84    9/21/12 3:33p Fseide
// implemented nosoftmax option for posteriorstats()
// 
// 83    9/18/12 11:16a Fseide
// new method adagradientfromdenom()
// 
// 82    9/18/12 11:06a Fseide
// implemented asum() and adadenom();
// allocate() now always clears the memory to 0, so we can use asum()...
// 
// 81    9/18/12 10:07a Fseide
// ccumulatesqr() now implements the IIR filter, and has a new argument
// 'keepweight' for that
// 
// 80    9/17/12 6:18p Fseide
// implemented accumulatesqr()
// 
// 79    9/16/12 4:40p Fseide
// allocate() now clears padding values by temporarily increasing the
// #rows to include the padding values (previous implementation relied on
// setzero() to include the padding values, which is incorrect in the case
// of a patch)
// 
// 78    9/16/12 4:35p Fseide
// new method setzero();
// allocate() now resets the padding values to 0, so we can hack vector
// operations over it
// 
// 77    9/07/12 9:28a Fseide
// (fixed a compiler warning)
// 
// 76    9/05/12 10:36p V-hansu
// add another extern function tomatrixref, change previous to
// tomatrixrefconst
// 
// 75    9/04/12 10:27p V-hansu
// add extern tomatrixref
// 
// 74    9/04/12 5:09p Fseide
// readonlymatriximpl completed (but not yet tested)
// 
// 73    9/04/12 2:42p Fseide
// began to write textureref, but that won't work since CUDA requires a
// global definition per texture :( so we need some wrapper class around
// that
// 
// 72    9/04/12 2:03p Fseide
// 
// 71    9/04/12 1:54p Fseide
// new class readonlymatrix, which is accessed through the texture
// machinery and its special caching structure
// 
// 70    9/04/12 1:43p Fseide
// renamed floatmatrix to matriximpl, for consistency with vectorbaseimpl;
// factored out the deviceid into a base class 'objectondevice' shared
// between vectorbaseimpl and matriximpl
// 
// 69    8/28/12 2:49p Fseide
// exported the two operator|| for use in cudalattice.cpp --but really
// this would belong into a shared header!
// 
// 68    8/25/12 9:15p V-hansu
// (change some indentation)
// 
// 67    7/17/12 5:31p Adame
// Update for no-sync framework
// async copy fixes
// 
// 66    6/26/12 12:16p V-xieche
// undef a macro ASYNCCOPY when copy data from GPU to CPU. Defining it
// will cause undeterminative.
// 
// 65    6/24/12 9:29p V-xieche
// switch code into a work point(an old version).
// 
// 61    6/08/12 11:24a Dongyu
// include <float.h>to use _isnan 
// 
// 60    6/08/12 9:31p V-xieche
// delete code related to DELAYUPDATE and modify the assign function.
// 
// 59    6/08/12 8:35p V-xieche
// comment the sync function for Asynccopy. And add a flag in assign
// function to select using async copy or sync copy.
// 
// 58    6/06/12 5:11p Adame
// Copy Sync update
// 
// 57    4/11/12 10:21a V-xieche
// add stripedposteriorstats function.
// 
// 56    4/01/12 7:13a V-xieche
// disabled _p2p copying
// 
// 55    4/01/12 8:47p Fseide
// changed assignmatrix_ua() to assignmatrix_p2p() using explicit
// contexts;
// new methods for that: getdevicecontext()
// 
// 54    4/01/12 2:05p Fseide
// seterrorsignal now takes an offset parameter so that it can work for
// vertical stripes
// 
// 53    4/01/12 2:00p V-xieche
// code for striped seterror signal
// 
// 52    4/01/12 11:24a V-xieche
// add code for striped softmax computation in 2 gpu.
// 
// 51    3/31/12 8:50p Fseide
// disabled broken assignmatrix_ua function
// 
// 50    3/31/12 8:32p Fseide
// assign (other matrix) now supports x64 unified addressing --to be
// tested!
// 
// 49    3/31/12 7:57p Fseide
// (added #if _WIN64 for dev-to-dev copy, but no actual code inside yet)
// 
// 48    3/31/12 7:27p Fseide
// first implementation of assign (from other matrix)
// 
// 47    3/31/12 19:16 Fseide
// new method assign() from another matrix
// 
// 46    2/26/12 6:59p V-xieche
// Add codes for coping date between CUDA device.
// 
// 45    2/25/12 5:24p V-xieche
// Add helpler function for coping date in CUDA device
// 
// 44    12/09/11 2:03p Dongyu
// added test code for GPU version of KhatriRaoProduct and
// reshapecolumnproduct
// 
// 43    12/06/11 5:43p Dongyu
// fixed bugs in reshapecolumnproduct
// 
// 42    11/28/11 5:55p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 41    11/23/11 1:41p Dongyu
// change KhatriRaoProduct input variables to const matrix
// 
// 40    11/23/11 1:20p Dongyu
// add reshape and KhatriRaoProduct
// 
// 39    11/04/11 14:58 Fseide
// added new argument 'otherweight' to addrowsum() to allow unscaled
// gradients w.r.t. momentum
// 
// 38    10/25/11 5:16p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 37    10/10/11 9:26p Dongyu
// fixed the abstract class instantiation problem caused by inconsistent
// declaration in the base and derived class for matrix.
// 
// 36    10/06/11 5:15p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 35    6/21/11 13:40 Fseide
// added frame for new function patchasblockdiagonal(), but inner loop not
// implemented yet
// 
// 34    3/03/11 8:15a Dongyu
// added weight sparseness support in training.
// 
// 33    2/26/11 4:50p Fseide
// new method softmax()
// 
// 32    2/25/11 5:55p Fseide
// new method synchronize();
// assign(0 and fetch() now take a parameter to run sync or async
// 
// 31    2/24/11 9:52p Fseide
// assign() and fetch() now based on CUBLAS to allow for non-contiguous
// transfers
// 
// 30    2/11/11 7:37p Fseide
// forgot ondevice in destructor, causing free() to fail
// 
// 29    2/11/11 3:47p Fseide
// setdevice() now fails if memory has already been allocated, since we
// cannot migrate CUDA matrices across devices
// 
// 28    2/11/11 3:31p Fseide
// new class ondevice to manage CUDA contexts;
// ondevice class used in all matriximpl functions;
// all exported functions call lazyinit()
// 
// 27    2/10/11 1:53p Fseide
// new method posteriorstats() (although it does not work correctly yet)
// 
// 26    2/10/11 11:35a Fseide
// added a join() to gemm() for accurate timing measurement
// 
// 25    2/10/11 11:32a Fseide
// new method mulbydsigm()
// 
// 24    2/10/11 11:17a Fseide
// new method setbackpropagationerrorsignal()
// 
// 23    2/10/11 10:03a Fseide
// gems() now performs a join() to allow for accurate measurement of
// runtime
// 
// 22    2/08/11 5:29p Fseide
// (fixed a bounds check)
// 
// 21    2/08/11 2:18p Fseide
// (cosmetics)
// 
// 20    2/07/11 9:52p Fseide
// llstats() implemented
// 
// 19    2/07/11 7:08p Fseide
// new method addtoallcolumns()
// 
// 18    2/07/11 6:52p Fseide
// implemented samplebinary()
// 
// 17    2/07/11 6:28p Fseide
// added rowsum() and sigmoid()
// 
// 16    2/07/11 5:21p Fseide
// added a check to allocate() to not allocate if we are a reference
// --really need to factor this out to a different class
// 
// 15    2/05/11 8:58p Fseide
// matriximpl now remembers whether to free its memory or not (i.e. when
// it is a patch)
// 
// 14    2/05/11 8:55p Fseide
// new method patch()
// 
// 13    2/02/11 8:22a Fseide
// gemm() now allows B to be transposed as well
// 
// 12    2/01/11 4:55p Fseide
// addcol() removed
// 
// 11    2/01/11 3:47p Fseide
// updated to new matrix functions;
// moved out all cublas calls to new cudamatrixops.cpp
// 
// 10    2/01/11 10:47a Fseide
// switched to cublas--nice! We can abandon our own kernel & stuff
// 
// 9     1/31/11 10:12p Fseide
// (added test code for weighting factors)
// 
// 8     1/31/11 9:57p Fseide
// (added a comment)
// 
// 7     1/31/11 9:35p Fseide
// tested for transposed A
// 
// 6     1/31/11 9:06p Fseide
// (refined time measurement)
// 
// 5     1/31/11 8:36p Fseide
// refined test code
// 
// 4     1/31/11 7:18p Fseide
// added test code for matrix product
// 
// 3     1/31/11 4:55p Fseide
// matriximpl no longer uses cuda::msra::vector
// 
// 2     1/31/11 3:32p Fseide
// initial implementation of matriximpl
// 
// 1     1/31/11 12:00p Fseide
// created

#define DLLEXPORT
#include "cudamatrix.h"             // this exports the class
#include "cudamatrixops.h"          // base class that contains CUDA computations (device-local)
#include "cudalib.h"                // generic CUDA helpers
#include "cudadevice.h"
#include <math.h>
#include <memory>                   // for auto_ptr
#include <assert.h>
#include <float.h>

namespace msra { namespace cuda {

#ifndef NOCUDA
size_t getnumdevices() { lazyinit(); return numdevices(); }

// allows to write cudaFunction() || "error"   (CUDA runtime)
void operator|| (cudaError_t rc, const char * msg)
{
    if (rc != cudaSuccess)
    {
        char buf[1000];
        sprintf_s (buf, "%s: %s (cuda error %d)", msg, cudaGetErrorString (rc), rc);
        throw std::runtime_error (buf);
    }
}

// allows to write cuFunction() || "error"  (CUDA API)
void operator|| (CUresult rc, const char * msg)
{
    if (rc != CUDA_SUCCESS)
    {
        char buf[1000];
        sprintf_s (buf, "%s: cuda API error %d", msg, rc);
        throw std::runtime_error (buf);
    }
}
#endif

// ---------------------------------------------------------------------------
// matriximpl -- simple matrix class, accessed through the msra::cuda::matrix interface
// TODO: split off a 'stripe' class and get rid of 'weownthememory' flag
// ---------------------------------------------------------------------------

template<class MTYPE>
static void checkmovebounds (const MTYPE * m, size_t i0, size_t i1, size_t j0, size_t j1, size_t colstride)
{
    if (i0 > i1 || j0 > j1)
        throw std::logic_error ("assign: rectangle inverted??");
    if (i0 >= m->rows() || j0 >= m->cols())
        throw std::logic_error ("assign: rectangle out of bounds");
}

class matriximpl : public /*interface*/matrix, public cudamatrixops, public objectondevice
{
    const bool weownthememory;          // if true then free memory (false means we are a patch)
public:
    // allocation and moving data (-> matrix interface)
    matriximpl() : weownthememory (true) { setdevice (0); }
    ~matriximpl() { if (weownthememory) { ondevice no (deviceid); free (p); } }
    void setdevice (size_t deviceid)            // just remembers it here
    {
        if (deviceid == getdevice())            // (allow repeated setting as long as the device does not change)
            return;
        if (p.get() != NULL)                    // cannot migrate across devices
            throw std::logic_error ("setdevice: device cannot be changed once matrix is allocated");
        objectondevice::setdevice (deviceid);
    }
    size_t getdevice() const {return deviceid;} 
    void allocate (size_t n, size_t m)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        // TODO: need to go through the insane thread mechanism
        if (n == numrows && m == numcols)
            return;                             // no resize needed
        if (!weownthememory)
            throw std::logic_error ("allocate: attempted to re-allocate a CUDA matrix reference");
        const size_t newcolstride = (n + 3) & ~3;     // pad to multiples of four floats (required SSE alignment)
        const size_t totalelem = newcolstride * m;
        cuda_ptr<float> pnew = totalelem > 0 ? malloc<float> (totalelem) : NULL;
        p.swap (pnew);
        free (pnew);    // pnew is now the old p
        numrows = n; numcols = m;
        colstride = newcolstride;
        // clear the padding elements, so that we can treat the thing as a vector
        // (if we don't clear it, then the padding elements may contain NaN elements)
        // It is also possible that this matrix is used as a stripe; so there are more 0 values than just the padding.
        //if (colstride != n)
        //{
        if (colstride > 0 && numcols > 0)
        {
            numrows = colstride;    // temporarily increase to include padding values
            setvalue(0);              // (TODO: set only the extras to zero; if this ever matters)
            numrows = n;
        }
        //}
    }
private:
    // constructor for a stripe
    matriximpl (matriximpl & other, size_t i0, size_t i1, size_t j0, size_t j1) : objectondevice (other.deviceid), weownthememory (false)
    {
        assert (i0 <= i1 && j0 <= j1);
        assert (i1 <= other.rows() && j1 <= other.cols());
        numrows = i1 - i0;
        numcols = j1 - j0;
        colstride = other.colstride;
        if (numrows > 0 && numcols > 0)
            p = &other(i0,j0);
    }
public:
    // return a patch of this matrix
    // Note: We really only support vertical stripes, although this function does not enforce it.
    matrix * patch (size_t i0, size_t i1, size_t j0, size_t j1) { return new matriximpl (*this, i0, i1, j0, j1); }

    size_t rows() const throw() { return matrixref::rows(); }   // need to implement these because they are virtual
    size_t cols() const throw() { return matrixref::cols(); }
    void reshape(const size_t newrows, const size_t newcols) { matrixref::reshape(newrows, newcols);};

    // memory allocations and move
public:
    void CheckBuffer(size_t device, float* buf, int size)
    {
        ondevice no(device);
        cudaDeviceSynchronize();
        for (int i=0; i < size; i++)
        {
            if (_isnan(buf[i]))
                printf("NAN found");
        }
    }

    void CopyBuffer(matriximpl* bufTo, const matriximpl* bufFrom, int flags, cudaEvent_t waitEvent, cudaStream_t streamNext)
    {
        size_t deviceFrom = bufFrom->getdevice();
        size_t deviceTo = bufTo->getdevice();

        // if the devices are different, we need to copy 
        if (deviceFrom != deviceTo)
        {
            size_t size = bufFrom->getcolstride() * bufFrom->cols() * sizeof(float);
            size_t deviceForBuffer = (flags&copyUseDestinationBuffers)?deviceTo:deviceFrom;
            BufEventStream* bufFromES = g_devices.GetCopyFromES(deviceFrom);
            BufEventStream* bufES = g_devices.GetCopyFromES(deviceForBuffer, size);
            float* buf = (float*)bufES->Buffer();
            CopyDeviceToHost(buf, bufFrom, waitEvent, NULL);
            // CheckBuffer(deviceTo, buf, size/sizeof(float));
            CopyHostToDevice(bufTo, buf, bufFromES->EventComplete(), streamNext);
        }
        else
        {
            size_t size = bufFrom->getcolstride() * bufFrom->cols() * sizeof(float);
            BufEventStream* bufESFrom = g_devices.GetCopyFromES(deviceFrom, size);

            ondevice no(deviceFrom);

            // wait for the event to fire before processing this data
            if (waitEvent != NULL)
            {
                cudaStreamWaitEvent(bufESFrom->Stream(), waitEvent, 0) || "CopyDeviceToHost() error on StreamWait";
            }

            cudaMemcpyAsync(bufTo->p.get(), bufFrom->p.get(), size, cudaMemcpyDefault, streamNext) || "CopyBuffer() error during copy.";
        }
    }

    void CopyDeviceToHost(float* bufTo, const matriximpl* bufFrom, cudaEvent_t waitEvent, cudaStream_t streamNext)
    {
        size_t deviceFrom = bufFrom->getdevice();
        size_t size = bufFrom->getcolstride() * bufFrom->cols() * sizeof(float);
        BufEventStream* bufESFrom = g_devices.GetCopyFromES(deviceFrom, size);

        ondevice no(deviceFrom);

        // wait for the event to fire before processing this data
        if (waitEvent != NULL)
        {
            cudaStreamWaitEvent(bufESFrom->Stream(), waitEvent, 0) || "CopyDeviceToHost() error on StreamWait";
        }

        cudaMemcpyAsync(bufTo, bufFrom->p.get(), size, cudaMemcpyDeviceToHost, bufESFrom->Stream()) || "CopyDeviceToHost() error during copy";

        bufESFrom->RaiseEventAndWait(streamNext);
    }

    void CopyHostToDevice(matriximpl* bufTo, const float* bufFrom, cudaEvent_t waitEvent, cudaStream_t streamNext)
    {
        size_t deviceTo = bufTo->getdevice();
        size_t size = bufTo->getcolstride() * bufTo->cols() * sizeof(float);
        BufEventStream* bufESTo = g_devices.GetCopyToES(deviceTo, size);

        ondevice no(deviceTo);

        // wait for the event to fire before processing this data
        if (waitEvent != NULL)
        {
            cudaStreamWaitEvent(bufESTo->Stream(), waitEvent, 0)  || "CopyHostToDevice() error on StreamWait";
        }

        cudaMemcpyAsync(bufTo->p.get(), bufFrom, size, cudaMemcpyHostToDevice, bufESTo->Stream())  || "CopyHostToDevice() error during copy";;

        bufESTo->RaiseEventAndWait(streamNext);
    }

    // CopyBuffer generic version, will take and pointers and devices along with a host buffer for 
    static cudaEvent_t CopyBuffer(size_t deviceTo, float* deviceToBuffer, size_t deviceFrom, float* deviceFromBuffer, size_t sizeInBytes, float* hostBuffer=NULL, cudaStream_t streamNext=NULL)
    {
        cudaEvent_t eventReturn = NULL;

        // make sure we have a host buffer passed in somewhere
        if (hostBuffer == NULL)
        {
            if (deviceTo == deviceHost)
                hostBuffer = deviceToBuffer;
            else if (deviceFrom == deviceHost)
                hostBuffer = deviceFromBuffer;
            if (hostBuffer == NULL)
                throw std::logic_error("CopyBuffer:no host buffer provided");

            // make sure the host buffer is pinned, externally done now
            //g_devices.PinBuffer(hostBuffer, sizeInBytes);
            assert(IsPinned(hostBuffer));
        }

        // make sure that they aren't both host buffers, we don't handle that
        assert(!(deviceTo == deviceHost && deviceFrom == deviceHost));
        BufEventStream* bufESFrom = NULL;
        BufEventStream* bufESTo = (deviceTo != deviceHost)?g_devices.GetCopyToES(deviceTo):NULL;

        // do the copy from device to host
        if (deviceFrom != deviceHost)
        {
            bufESFrom = g_devices.GetCopyFromES(deviceFrom);
            ondevice no(deviceFrom);

            cudaMemcpyAsync(deviceToBuffer, deviceFromBuffer, sizeInBytes, cudaMemcpyDeviceToHost, bufESFrom->Stream()) || "CopyBuffer() error during from copy";
            eventReturn = bufESFrom->EventComplete();

            if (bufESTo != NULL)
                bufESFrom->RaiseEventAndWait(bufESTo->Stream());
            else
                bufESFrom->RaiseEventAndWait(streamNext);

        }

        // do the copy from host to device
        if (deviceTo != deviceHost)
        {
            BufEventStream* bufESTo = g_devices.GetCopyToES(deviceTo);

            ondevice no(deviceTo);
            cudaMemcpyAsync(deviceToBuffer, deviceFromBuffer, sizeInBytes, cudaMemcpyHostToDevice, bufESTo->Stream())  || "CopyBuffer() error during to copy";
            bufESTo->RaiseEventAndWait(streamNext);
            eventReturn = bufESTo->EventComplete();
        }

        // return the event in case they want to wait on it
        return eventReturn;
    }

    void assign (size_t i0, size_t i1, size_t j0, size_t j1, const float * pi0j0, size_t colstride, bool synchronize)
    {
        if (i0 == i1 || j0 == j1) return;
        checkmovebounds (this, i0, i1, j0, j1, colstride);
        ondevice no (deviceid);     // switch to desired CUDA card
#if 0  // this macro should not be enable, would casued something undeterminative, tested by v-xieche. [v-xieche]
        CopyHostToDevice (dynamic_cast<matriximpl *>(this), pi0j0, NULL, NULL); 
#else
        cudamatrixops::assign (i0, i1, j0, j1, pi0j0, colstride);
        // This join is bad for multi-GPUs. Need a special exposed join operation.
#endif
        if (synchronize)
            join();
    }

    // this assigns from a matrix that is possibly on a different device
    // Currently, we blindly copy through a CPU-side buffer.
    // However, CUDA 4.1 allows direct copies in x64 mode. This is what we should be using.
    void assign (matrix & other, float * pi0j0/*CPU buffer in case it's needed*/, size_t colstride, bool synchronize, int copyFlags)
    {
        if (rows() != other.rows() || cols() != other.cols())
            throw std::logic_error ("assign: mismatching dimensions");
        if (rows() > colstride)
            throw std::logic_error ("assign: mismatching buffer colstride");
        // use x64 unified addressing if we can
        if (copyFlags & (copyAsync | copyDirect))    // async or direct copy
        {
            const matriximpl *bufFrom = dynamic_cast<const matriximpl *>(&other);
            //BufEventStream* bufFromES = g_devices.GetComputeES(bufFrom->deviceid);

            //// we need to raise the event, pass NULL so it doesn't wait yet
            //bufFromES->RaiseEventAndWait(NULL);

            //// now copy the buffer waiting for the completion event from the 'From' stream and having the 'To' stream wait for the copy to complete
            //BufEventStream* computeES = g_devices.GetComputeES(deviceid);

            if (copyDirect & copyFlags)    // direct copy
            {
                BufEventStream* bufFromES = g_devices.GetCopyFromES(bufFrom->deviceid);
                cudaMemcpyAsync(this->p.get(), bufFrom->p.get(), bufFrom->getcolstride()*bufFrom->cols(), cudaMemcpyDefault, bufFromES->Stream()) || "assign() error during copy.";
                // we need to raise the event, pass NULL so it doesn't wait
                BufEventStream* bufToES = g_devices.GetCopyFromES(this->deviceid);
                bufToES->RaiseEventAndWait(NULL);
            }
            else    // async copy
            {
                CopyBuffer(dynamic_cast<matriximpl *>(this), bufFrom, copyFlags, NULL, NULL); //bufFromES->EventComplete(), computeES->Stream());
            }
            if (synchronize)
                this->synchronize();
        }
        else    // sync copy
        {
            other.fetch (0, other.rows(), 0, other.cols(), pi0j0, colstride, true/*synchronize--must do that for fetch since cross-device copy*/);
            assign (0, rows(), 0, cols(), pi0j0, colstride, synchronize);
        }
    }

    void fetch (size_t i0, size_t i1, size_t j0, size_t j1, float * pi0j0, size_t colstride, bool synchronize) const
    {
        if (i0 == i1 || j0 == j1) return;
        checkmovebounds (this, i0, i1, j0, j1, colstride);

        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::fetch (i0, i1, j0, j1, pi0j0, colstride);

        // This join is bad for multi-GPUs. Need a special exposed join operation.
        if (synchronize)
            join();
    }

    void fetchtodevice(float * p_dst, size_t memsize)
    {
        cudaMemcpy(p_dst, p.get(), memsize, cudaMemcpyDeviceToDevice);
    }

    void assignfromdevice(const float * p_src, size_t memsize)
    {
        cudaMemcpy(p.get(), p_src, memsize, cudaMemcpyDeviceToDevice);
    }
    // wait until previous operation is completed
    // Needed for assign() and fetch(), and for matrix ops for time measurements
    void synchronize() const
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        join();
    }

    // operations implemented with CUBLAS
    // this = this * alpha + A * B * beta, where A is stored as its transpose if 'Aistransposed'
    void gemm (float beta, const matrix & A, bool Aistransposed, const matrix & B, bool Bistransposed, float alpha)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::gemm (beta, dynamic_cast<const matriximpl &> (A), Aistransposed, dynamic_cast<const matriximpl &> (B), Bistransposed, alpha);

        // We do a join() here as this is the last op of pretraining/backpropagationstats()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }
    void gems (float beta, const matrix & other, float alpha)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::gems (beta, dynamic_cast<const matriximpl &> (other), alpha);

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }

    // returns dot product with b, uses cublas
    float dot (const matrix & b) const {
        ondevice no (deviceid);     // switch to desired CUDA card
        return cudamatrixops::dot(dynamic_cast<const matriximpl &> (b));
    }
    
    // returns euclidean norm, uses cublas
    float nrm2 () const {
        ondevice no (deviceid);     // switch to desired CUDA card
        return cudamatrixops::nrm2();
    }
    // returns columnwise euclidean norms, possibly scales when maxolnorm>0
    void colwisenrm2 (matrix & norms, float maxcolnorm) const {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::colwisenrm2(dynamic_cast<matriximpl &> (norms), maxcolnorm);
    }
    void KhatriRaoProduct(const matrix & m1, const matrix & m2)
    {
        if (this->rows() != m1.rows() * m2.rows())
            throw std::runtime_error ("cudamatrix::KhatriRaoProduct: mismatched dimensions this->rows() != m1.rows() * m2.rows()");
        else if (this->cols() != m1.cols()  || this->cols() != m2.cols())
            throw std::runtime_error ("cudamatrix::KhatriRaoProduct: mismatched dimensions this->cols() != m1.cols()  || this->cols() != m2.cols()");
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::KhatriRaoProduct (dynamic_cast<const matriximpl &> (m1), dynamic_cast<const matriximpl &> (m2));

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }

    void reshapecolumnproduct (const matrix & eh, const matrix & h, const bool isehtransposed)
    {
        if (eh.rows() != this->rows() * h.rows())
        {
            throw std::runtime_error ("cudamatrix::reshapecolumnproduct: mismatched dimensions eh.rows() != this->rows() * h.rows()");
        }
        else if (this->cols() != eh.cols()  || this->cols() != h.cols())
        {
            throw std::runtime_error ("cudamatrix::reshapecolumnproduct: mismatched dimensions this->cols() != eh.cols()  || this->cols() != h.cols()");
        }

        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::reshapecolumnproduct (dynamic_cast<const matriximpl &> (eh), dynamic_cast<const matriximpl &> (h), isehtransposed);

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }

    void setto0ifabsbelow (float threshold)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::setto0ifabsbelow (threshold);

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }
    void setto0ifbelow (float threshold)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::setto0ifbelow (threshold);
    }
    void setto0ifabsbelow2 (matrix &  ref, float threshold)
    {
        if (this->rows() != ref.rows() || this->cols() != ref.cols())
        {
            throw std::runtime_error ("cudamatrix::setto0ifabsbelow2: mismatched dimensions in this and ref");
        }

        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::setto0ifabsbelow2 (dynamic_cast<matriximpl &> (ref), threshold);

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }

    void setto0ifabsabove2 (matrix &  ref, float threshold)
    {
        if (this->rows() != ref.rows() || this->cols() != ref.cols())
        {
            throw std::runtime_error ("cudamatrix::setto0ifabsabove2: mismatched dimensions in this and ref");
        }

        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::setto0ifabsabove2 (dynamic_cast<matriximpl &> (ref), threshold);

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }

    void patchasblockdiagonal (size_t diagblocks, bool averageblocks, size_t firstcol)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::patchasblockdiagonal (diagblocks, averageblocks, firstcol);

        // We do a join() here as this is the last op of pretraining/backpropagationmodelupdate()
        // so we will get accurate timing statistics. Otherwise this is not necessary.
        //join();
    }

    // operations implemented with our own kernel
    void addrowsum (float beta, const matrix & other, float alpha)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::addrowsum (beta, dynamic_cast<const matriximpl &> (other), alpha);
        //join(); // it's our own kernel launch, so check for errors
    }

    void addrowsumpool (float beta, const matrix & other, float alpha, size_t poolSize, size_t bands, size_t kernels)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::addrowsumpool (beta, dynamic_cast<const matriximpl &> (other), alpha, poolSize, bands, kernels);
        //join(); // it's our own kernel launch, so check for errors
    }


    // reorder the matrix for convolution layer
    //void reorder (matrix & to, size_t minibatchSize, size_t kernels, size_t bands)
    void reorder (matrix & to, const convolutionParams &convParams, bool input) const
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::reorder(dynamic_cast<matriximpl &> (to), convParams.minibatchSize, convParams.prevKernels, convParams.prevBands, input);
    }

    //void convolutionForward(matrix & out, const matrix & weight, const matrix & bias, size_t minibatchSize, size_t kernels, size_t bands, size_t newKernels, size_t poolingBandShift, size_t poolSize, size_t filterSize)
    void convolutionForward(matrix & out, const matrix & weight, const matrix & bias, const convolutionParams &convParams)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::convolutionForward(dynamic_cast<matriximpl &> (out), dynamic_cast<const matriximpl &> (weight), dynamic_cast<const matriximpl &> (bias), 
            convParams.minibatchSize, convParams.prevKernels, convParams.prevBands, convParams.kernels, convParams.poolingBandShift, convParams.poolSize, convParams.filterSize);
    }
    //void computeCnnDeltaW(matrix & deltaM, matrix & vM, matrix & dwM, size_t minibatchSize, int nPrevBands, int nPrevKernels, int poolingBandShift, int nKernels, int batchSize, int poolSize, int filterSize)
    void computeCnnDeltaW(const matrix & deltaM, const matrix & vM, matrix & deltatM, matrix & vtM, float thisscale, float vhscale, const convolutionParams &convParams)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::computeCnnDeltaW(dynamic_cast<const matriximpl &> (deltaM), dynamic_cast<const matriximpl &> (vM), dynamic_cast<matriximpl &> (deltatM), dynamic_cast<matriximpl &> (vtM), thisscale, vhscale,
            convParams.prevBands, convParams.prevKernels, convParams.poolingBandShift, convParams.kernels, convParams.minibatchSize, convParams.poolSize, convParams.filterSize);
    }

    void maxpoolForward(matrix & out, matrix & maxIndex, const convolutionParams &convParams)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::maxpoolForward(dynamic_cast<matriximpl &> (out), dynamic_cast<matriximpl &> (maxIndex), convParams.poolSize, convParams.bands, convParams.kernels, convParams.minibatchSize);
    }
    void submaxpoolForward(matrix & out, matrix & maxIndex, const convolutionParams &convParams)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::submaxpoolForward(dynamic_cast<matriximpl &> (out), dynamic_cast<matriximpl &> (maxIndex), convParams.poolSize, convParams.subpoolSize, convParams.bands, convParams.kernels, convParams.minibatchSize);
    }
    void maxpoolBack(matrix & out, const matrix & maxIndex, const convolutionParams &convParams)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::maxpoolBack(dynamic_cast<matriximpl &> (out), dynamic_cast<const matriximpl &> (maxIndex), convParams.poolSize, convParams.bands, convParams.kernels, convParams.minibatchSize);
    }
    void submaxpoolBack(matrix & out, const matrix & maxIndex, const convolutionParams &convParams)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::submaxpoolBack(dynamic_cast<matriximpl &> (out), dynamic_cast<const matriximpl &> (maxIndex), convParams.poolSize, convParams.subpoolSize, convParams.bands, convParams.kernels, convParams.minibatchSize);
    }
    void dump(char *name) const
    {
#ifdef DEBUG
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::dump(name);
#endif
    }

    void sigmoid()
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::sigmoid();
        //join(); // it's our own kernel launch, so check for errors
    }

    void samplebinary (const matrix & P, unsigned int randomseed)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::samplebinary (dynamic_cast<const matriximpl &> (P), randomseed);
        //join(); // it's our own kernel launch, so check for errors
    }
    void addtoallcolumns (const matrix & other)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::addtoallcolumns (dynamic_cast<const matriximpl &> (other));
        //join(); // it's our own kernel launch, so check for errors
    }
    void llstats (const matrix & v1, matrix & logllsums, bool gaussian) const
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::llstats (dynamic_cast<const matriximpl &> (v1), dynamic_cast<matriximpl &> (logllsums), gaussian);
        //join(); // it's our own kernel launch, so check for errors
    }
    void softmax()
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::softmax();
        //join(); // it's our own kernel launch, so check for errors
    }
    //this = this + weight * other
    void addweighted (const matrix & other, float weight)
    {
        ondevice no (deviceid);     // switch to desired CUDA card

        cudamatrixops::gems (1.0f, dynamic_cast<const matriximpl &> (other), weight);
    }

#if 1  // for striped top layer, the soft max function.[v-xieche]
    void stripedsoftmaxstep1( matrix &partialsumvectors)
    {
        ondevice no (deviceid);
        cudamatrixops::stripedsoftmaxstep1 (dynamic_cast<matriximpl &> (partialsumvectors) );
    }
    void stripedsoftmaxstep2 ( matrix &partialsumvectors)
    {
        ondevice no (deviceid);
        cudamatrixops::stripedsoftmaxstep2 (dynamic_cast<matriximpl &> (partialsumvectors) );
    }
#endif
    
    // sets matrix to hessianvectorsignal 
    // (required for computing hessian vector product)
    // uses the Hessian of cross-entropy training
    // this = (diag(Pu) - Pu Pu') forwardstatistics
    void sethessianvectorsignal(const matrix & Pu, const matrix &forwardStatistics)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::sethessianvectorsignal (dynamic_cast<const matriximpl &> (Pu), dynamic_cast<const matriximpl &> (forwardStatistics));
    }

    // sets this to diagonal preconditioner based on sum of squared gradients (see Martens paper)
    void setdiagonalpreconditioner(const matrix & gradientsquared, float nobservations, float lambda, float alpha)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::setdiagonalpreconditioner(dynamic_cast<const matriximpl &> (gradientsquared), nobservations, lambda, alpha);
    }

    // this = a / b (elementwise division)
    void elementwisedivision(const matrix & a, const matrix &b)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::elementwisedivision(dynamic_cast<const matriximpl &> (a), dynamic_cast<const matriximpl &> (b));
    }

    // this = a^2 (elementwise square)
    void elementwisesquare(const matrix & a)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::elementwisesquare(dynamic_cast<const matriximpl &> (a));
    }

    // returns weighted dot product of this and a, weighted by weightingmatrix
    float weighteddot(const matrix &weightingmatrix, const matrix &a) const
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        return cudamatrixops::weighteddot(dynamic_cast<const matriximpl &> (weightingmatrix), dynamic_cast<const matriximpl &> (a));
    }
    
    // zhaorui pass frame dropping thresh to error calculation funtion
    void setbackpropagationerrorsignal (const matrix & uids, const matrix & Pu, size_t i0, const matrix & senone2keepmodelupdate, const float framedropthresh)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        // zhaorui pass frame dropping thresh to error calculation funtion
        cudamatrixops::setbackpropagationerrorsignal (dynamic_cast<const matriximpl &> (uids), dynamic_cast<const matriximpl &> (Pu), i0,
                                                      dynamic_cast<const matriximpl &> (senone2keepmodelupdate), framedropthresh);
        //join(); // it's our own kernel launch, so check for errors
    }

    void setbackpropagationerrorsignalwithklreg (const matrix & uids, const matrix & Pu, const matrix & refPu, const float alpha)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::setbackpropagationerrorsignalwithklreg (dynamic_cast<const matriximpl &> (uids), dynamic_cast<const matriximpl &> (Pu), dynamic_cast<const matriximpl &> (refPu), alpha);
        //join(); // it's our own kernel launch, so check for errors
    }

    // zhaorui pass frame dropping thresh to error calculation funtion
    void setbackpropagationerrorsignalhsmoothing (const matrix & uids, const matrix & Pu, const matrix & refmat, const float hsmoothingweight, const size_t errorsettingmode, const float framedropthresh)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        // zhaorui pass frame dropping thresh to error calculation funtion
        cudamatrixops::setbackpropagationerrorsignalhsmoothing (dynamic_cast<const matriximpl &> (uids), dynamic_cast<const matriximpl &> (Pu), dynamic_cast<const matriximpl &> (refmat), hsmoothingweight, errorsettingmode,framedropthresh);
        //join(); // it's our own kernel launch, so check for errors
    }

    void mulbydsigm (const matrix & sigm)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::mulbydsigm (dynamic_cast<const matriximpl &> (sigm));
        //join(); // it's our own kernel launch, so check for errors
    }
    void mulbydlru (const matrix & lruvals)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::mulbydlru (dynamic_cast<const matriximpl &> (lruvals));
    }
    void posteriorstats (const matrix & Pu, matrix & logpps, matrix & pps, matrix & fcors, bool nosoftmax) const
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::posteriorstats (dynamic_cast<const matriximpl &> (Pu), dynamic_cast<matriximpl &> (logpps), dynamic_cast<matriximpl &> (pps), dynamic_cast<matriximpl &> (fcors), nosoftmax);
        //join(); // it's our own kernel launch, so check for errors
    }

    void stripedposteriorstats (const matrix & Pu, matrix & logpps, matrix & pps, matrix & fcors, size_t i0) const
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::stripedposteriorstats (dynamic_cast<const matriximpl &> (Pu), dynamic_cast<matriximpl &> (logpps), dynamic_cast<matriximpl &> (pps), dynamic_cast<matriximpl &> (fcors), i0);
    }

    void setvalue (float value)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::setvalue(value);
    }
    void patchasblockconv(const size_t nPrevBand, const size_t nPrevKernel, const size_t nKernel, const size_t poolSize, const size_t filterSize)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::patchasblockconv(nPrevBand, nPrevKernel, nKernel, poolSize, filterSize);
    }
    // functions for AdaGrad
    void accumulatesqr (const matrix & other, size_t mbframes, float keepweight)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::accumulatesqr (dynamic_cast<const matriximpl &> (other), mbframes, keepweight);
    }

    void adadenom (const matrix & sqracc, float numframes, size_t mbframes)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::adadenom (dynamic_cast<const matriximpl &> (sqracc), numframes, mbframes);
    }

    float asum() const
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        return cudamatrixops::asum();
    }

    void adagradientfromdenom (const matrix & gradient, float actualavdenom, float targetavdenom)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::adagradientfromdenom (dynamic_cast<const matriximpl &> (gradient), actualavdenom, targetavdenom);
    }

    void dropout (float factor, unsigned int randomseed)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        return cudamatrixops::dropout (factor, randomseed);
    }

    void scale (float factor)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        return cudamatrixops::scale (factor);
    }
    
    void scalecolwise (const matrix & factors)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixops::scalecolwise(dynamic_cast<const matriximpl &> (factors));
    }
};

const matrixref<float> & tomatrixrefconst (const matrix & m) { return dynamic_cast<const matriximpl &> (m); }
matrixref<float> & tomatrixref (matrix & m) { return dynamic_cast<matriximpl &> (m); }

// factory method
#ifndef NOCUDA
matrix * newmatrix() { lazyinit(); return new matriximpl; }
#endif


// ---------------------------------------------------------------------------
// matrixaccumulatorimpl -- double-precision matrix class, currently specifically for accumulating things
// ---------------------------------------------------------------------------

class matrixaccumulatorimpl : public /*interface*/matrixaccumulator, protected cudamatrixaccumulatorops, public objectondevice
{
public:
    // allocation and moving data (-> matrix interface)
    matrixaccumulatorimpl() { setdevice (0); }
    ~matrixaccumulatorimpl() { ondevice no (deviceid); free (p); }
    void setdevice (size_t deviceid)            // just remembers it here
    {
        if (deviceid == getdevice())            // (allow repeated setting as long as the device does not change)
            return;
        if (p.get() != NULL)                    // cannot migrate across devices
            throw std::logic_error ("setdevice: device cannot be changed once matrix is allocated");
        objectondevice::setdevice (deviceid);
    }

    void allocate (size_t n, size_t m)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        if (n == numrows && m == numcols)
            return;                                 // no resize needed
        const size_t newcolstride = (n + 3) & ~3;   // pad the same way as single precision (so we may do vector ops)
        const size_t totalelem = newcolstride * m;
        cuda_ptr<double> pnew = totalelem > 0 ? malloc<double> (totalelem) : NULL;
        p.swap (pnew);
        free (pnew);    // pnew is now the old p
        numrows = n; numcols = m;
        colstride = newcolstride;
    }

    size_t rows() const throw() { return cudamatrixaccumulatorops::rows(); }   // need to implement these because they are virtual
    size_t cols() const throw() { return cudamatrixaccumulatorops::cols(); }
    
    // reset --reset to 0
    void reset()
    {
        ondevice no (deviceid);
        cudamatrixaccumulatorops::reset();
    }

    // accumulate
    // this = thisscale * this + otherweight * other
    void accumulate (float thisscale, const matrix & other, float otherweight)
    {
        ondevice no (deviceid);
        cudamatrixaccumulatorops::accumulate (thisscale, dynamic_cast<const matriximpl &> (other), otherweight);
    }

    // read out result into a single-precision CUDA matrix
    // to = (float) this
    void tomatrix (matrix & to) const
    {
        if (rows() != to.rows() || cols() != to.cols())
            throw std::logic_error ("fetch: mismatching dimensions");

        ondevice no (deviceid);     // switch to desired CUDA card
        cudamatrixaccumulatorops::tomatrix (dynamic_cast<matriximpl &> (to));
    }
};

// factory method
#ifndef NOCUDA
matrixaccumulator * newmatrixaccumulator() { lazyinit(); return new matrixaccumulatorimpl; }
#endif


// ---------------------------------------------------------------------------
// readonlymatriximpl -- simple read-only matrix class, accessed through the msra::cuda::readonlymatrix interface
// ---------------------------------------------------------------------------

class readonlymatriximpl : public /*interface*/readonlymatrix, protected cudaarrayref<float>, public objectondevice
{
    cudaChannelFormatDesc desc;                 // CUDA descriptor; has no destructor
public:
    // allocation and moving data (-> matrix interface)
    readonlymatriximpl() { desc = cudaCreateChannelDesc (32, 0, 0, 0, cudaChannelFormatKindFloat); }
    ~readonlymatriximpl() { ondevice no (deviceid); if (a) cudaFreeArray (a); }
    void setdevice (size_t deviceid)            // just remembers it here
    {
        if (deviceid == getdevice())            // (allow repeated setting as long as the device does not change)
            return;
        if (a != NULL)                      // cannot migrate across devices
            throw std::logic_error ("setdevice: device cannot be changed once matrix is allocated");
        objectondevice::setdevice (deviceid);
    }

    void allocate (size_t n, size_t m)
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        if (n == numrows && m == numcols)
            return;                             // no resize needed
        cudaArray * anew = NULL;                // (ugh: unwrapped pointers are evil, even if short-lived!)
        if (n > 0 && m > 0)
            cudaMallocArray (&anew, &desc, n, m)    // (note: swapping indices--n is height; consistent with tex2D() accessor)
                || "allocate: cudaMallocArray failed to allocate readonlymatrix instance";
        if (a) cudaFreeArray (a);
        a = anew;
    }

    size_t rows() const throw() { return cudaarrayref::rows(); }   // need to implement these because they are virtual
    size_t cols() const throw() { return cudaarrayref::cols(); }

    void assign (size_t i0, size_t i1, size_t j0, size_t j1, const float * pi0j0, size_t colstride, bool synchronize)
    {
        if (i0 == i1 || j0 == j1) return;
        checkmovebounds (this, i0, i1, j0, j1, colstride);
        ondevice no (deviceid);     // switch to desired CUDA card
        // copy as an array
        // We swap the dimensions -- row index maps to width, col index to height
        cudaMemcpy2DToArray (a, i0, j0, pi0j0, colstride, i1 - i0, j1 - j0, cudaMemcpyHostToDevice)
            || "allocate: cudaMemcpyToArray failed";
        // This join is bad for multi-GPUs. Need a special exposed join operation.
        if (synchronize)
            join();
    }

    // wait until previous operation is completed
    // Needed for assign().
    void synchronize() const
    {
        ondevice no (deviceid);     // switch to desired CUDA card
        join();
    }
};

// factory method
#ifndef NOCUDA
readonlymatrix * newreadonlymatrix() { lazyinit(); return new readonlymatriximpl; }
#endif


// ---------------------------------------------------------------------------
// functions for async architecture
// ---------------------------------------------------------------------------

// convert from EventContext to cudaEvent_t
cudaEvent_t EventFromContext(EventContext event, cudaStream_t* stream=NULL)
{
    cudaEvent_t cudaEvent = NULL;
    switch(event.eventId)
    {
    case eventComputeReady:
        cudaEvent = g_devices.GetComputeES(event.deviceId)->EventAvailable();
        if (stream != NULL) *stream = g_devices.GetComputeES(event.deviceId)->Stream();
        break;
    case eventComputeComplete:
        cudaEvent = g_devices.GetComputeES(event.deviceId)->EventComplete();
        if (stream != NULL) *stream = g_devices.GetComputeES(event.deviceId)->Stream();
        break;
    case eventCopyFromReady:
        cudaEvent = g_devices.GetCopyFromES(event.deviceId)->EventAvailable();
        if (stream != NULL) *stream = g_devices.GetCopyFromES(event.deviceId)->Stream();
        break;
    case eventCopyFromComplete:
        cudaEvent = g_devices.GetCopyFromES(event.deviceId)->EventComplete();
        if (stream != NULL) *stream = g_devices.GetCopyFromES(event.deviceId)->Stream();
        break;
    case eventCopyToReady:
        cudaEvent = g_devices.GetCopyToES(event.deviceId)->EventAvailable();
        if (stream != NULL) *stream = g_devices.GetCopyToES(event.deviceId)->Stream();
        break;
    case eventCopyToComplete:
        cudaEvent = g_devices.GetCopyToES(event.deviceId)->EventComplete();
        if (stream != NULL) *stream = g_devices.GetCopyToES(event.deviceId)->Stream();
        break;
    }
    return cudaEvent;
}

// convert a event into an EventContext. Currently we need to do this by searching through all the known events and checking for equality
// if we get too many we could do a binary search or something, right now sequential will work just fine.
EventContext ContextFromEvent(size_t deviceId, cudaEvent_t cudaEvent)
{
    Event eventId = eventNil;

    // now check each event in order to see which one we have, put most common first
    if (cudaEvent == g_devices.GetCopyToES(deviceId)->EventComplete())
    {
        eventId = eventCopyToComplete;
    }
    else if (cudaEvent == g_devices.GetComputeES(deviceId)->EventAvailable())
    {
        eventId = eventComputeReady;
    }
    else if (cudaEvent == g_devices.GetComputeES(deviceId)->EventComplete())
    {
        eventId = eventComputeComplete;
    }
    else if (cudaEvent == g_devices.GetCopyFromES(deviceId)->EventAvailable())
    {
        eventId = eventCopyFromReady;
    }
    else if (cudaEvent == g_devices.GetCopyFromES(deviceId)->EventComplete())
    {
        eventId = eventCopyFromComplete;
    }
    else if (cudaEvent == g_devices.GetCopyToES(deviceId)->EventAvailable())
    {
        eventId = eventCopyToReady;
    }
    EventContext eventContext(deviceId, eventId);
    return eventContext;
}

// Get the cudaStream from a OperationContext
cudaStream_t StreamFromContext(OperationContext operation)
{
    cudaStream_t stream;
    switch (operation.operationId)
    {
    case operationCompute:
        stream = g_devices.GetComputeES(operation.deviceId)->Stream();
        break;
    case operationCopyFrom:
        stream = g_devices.GetCopyFromES(operation.deviceId)->Stream();
        break;
    case operationCopyTo:
        stream = g_devices.GetCopyToES(operation.deviceId)->Stream();
        break;
    }
    return stream;
}

// fire an event, use WaitForEvent() to cause all subsequent calls in an operationContext to wait for this event
// Calling this simply puts an call in the current CUDA command queue, so it will not happen immediately.
void FireEvent(EventContext event)
{
    // get the event and stream from the context
    cudaStream_t stream;
    cudaEvent_t cudaEvent = EventFromContext(event, &stream);

    // now fire the event
    ondevice no(event.deviceId);
    cudaEventRecord(cudaEvent, stream) || "FireEvent() error during EventRecord";
}

// Causes all calls in the given operationContext to wait until the event fires 
void WaitForEvent(EventContext event, OperationContext nextOperation)
{
    // get the event from the context
    cudaEvent_t cudaEvent = EventFromContext(event);
    cudaStream_t stream = StreamFromContext(nextOperation);

    // wait for the event copy to the buffer to complete first
    cudaStreamWaitEvent (stream, cudaEvent, 0) || "WaitForEvent() error on StreamWait";
}

// Fire an event and have all subsequent activity on the operation context dealy until the event actually fires
void FireAndWaitEvent(EventContext event, OperationContext nextOperation)
{
    FireEvent(event);
    WaitForEvent(event, nextOperation);
}

void SyncDevice(size_t device)
{
    ondevice no(device);
    cudaDeviceSynchronize();
}

// Copy commands:
EventContext CopyBuffer(size_t deviceTo, float* deviceToBuffer, size_t deviceFrom, float* deviceFromBuffer, size_t sizeInBytes, float* hostBuffer)
{
    cudaEvent_t returnEvent = matriximpl::CopyBuffer(deviceTo, deviceToBuffer, deviceFrom, deviceFromBuffer, sizeInBytes, hostBuffer);
    EventContext eventContext = ContextFromEvent(deviceTo!=deviceHost?deviceTo:deviceFrom, returnEvent);
    return eventContext;
}

EventContext CopyBufferWait(EventContext eventToWaitFor, size_t deviceTo, float* deviceToBuffer, size_t deviceFrom, float* deviceFromBuffer, size_t sizeInBytes, float* hostBuffer)
{
    OperationContext operationContext(deviceFrom != deviceHost?deviceFrom:deviceTo, deviceFrom != deviceHost?operationCopyFrom:operationCopyTo);
    WaitForEvent(eventToWaitFor, operationContext);
    return CopyBuffer(deviceTo, deviceToBuffer, deviceFrom, deviceFromBuffer, sizeInBytes, hostBuffer);
}

void PinBuffer(const float *bufHost, size_t size)
{
    g_devices.PinBuffer(bufHost, size);
}

// unpin the buffer, if it wasn't pinned, do nothing
// WARNING: Unpin operations do a CPU sync
void UnpinBuffer(const float *bufHost)
{
    g_devices.UnpinBuffer(bufHost);
}

bool IsPinned(const float *bufHost)
{
    return g_devices.IsPinned(bufHost);    
}


// test code
class simplematrix
{
    std::vector<float> data;
    size_t numrows;     // rows()
    size_t numcols;     // cols()
    size_t locate (size_t i, size_t j) const { assert(i < rows() && j < cols()); return j * colstride + i; }   // matrix in column-wise storage
public:
    size_t colstride;   // height of column = rows() rounded to multiples of 4
    simplematrix (size_t n, size_t m) { resize (n, m); initrandom(); }
    void resize (size_t n, size_t m)
    {
        colstride = (n + 3) & ~3;     // pad to multiples of four floats (required SSE alignment)
        const size_t totalelem = colstride * m;
        data.resize (totalelem);
        numrows = n; numcols = m;
    }
    void initrandom()
    {
        for (size_t i = 0; i < rows(); i++) for (size_t j = 0; j < cols(); j++)
            (*this)(i,j) = rand() / (float) RAND_MAX;
    }
    size_t rows() const throw() { return numrows; }
    size_t cols() const throw() { return numcols; }
    void reshape(const size_t newrows, const size_t newcols) { assert (rows() * cols() == newrows * newcols); numrows=newrows; numcols = newcols;}
    float &       operator() (size_t i, size_t j)       { return data[locate(i,j)]; }
    const float & operator() (size_t i, size_t j) const { return data[locate(i,j)]; }
    void matprod_mm (const simplematrix & A, const simplematrix & B)
    {
        assert (rows() == A.rows() && cols() == B.cols() && A.cols() == B.rows());
        for (size_t i = 0; i < rows(); i++) for (size_t j = 0; j < cols(); j++)
        {
            float sum = 0.0;
            for (size_t k = 0; k < A.cols(); k++)
                sum += A(i,k) * B(k,j);
            (*this)(i,j) = sum;
        }
    }
    void matprod_mtm (const simplematrix & At, const simplematrix & B)
    {
        assert (rows() == At.cols() && cols() == B.cols() && At.rows() == B.rows());
        for (size_t i = 0; i < rows(); i++) for (size_t j = 0; j < cols(); j++)
        {
            float sum = 0.0;
            for (size_t k = 0; k < At.rows(); k++)
                sum += At(k,i) * B(k,j);
            (*this)(i,j) = sum;
        }
    }
    //both m1 and m2 are passed in normal form (i.e., not transposed)
    void KhatriRaoProduct(const simplematrix & m1, const simplematrix & m2)
    {
        assert(m1.cols() == m2.cols());
        assert (rows() == m1.rows() * m2.rows());

        for (size_t k=0; k<cols(); k++)
        {
            size_t jj = 0;
            for (size_t j=0; j<m2.rows(); j++)
            {
                for (size_t i=0; i<m1.rows(); i++)
                {
                    (*this)(jj++, k) = m1(i,k) * m2(j,k);
                }
            }
        }
    }

    void reshapecolumnproduct (const simplematrix & eh, const simplematrix & h, const bool isehtransposed)
    {
        if (isehtransposed)
        {
            //find nrows and ncols of the reshpaed eh
            size_t nrows = h.rows();
            size_t ncols = eh.rows() / nrows;
            assert (eh.rows() % nrows == 0);

            for (size_t t=0; t<eh.cols(); t++)
            {
                size_t k=0;
                for (size_t j=0; j<ncols; j++)   // row and col is transposed
                {
                    (*this)(j,t) = 0.0f;
                    for (size_t i=0; i<nrows; i++)
                    {
                        (*this)(j,t) += eh(k,t) * h(i,t);
                        k++;
                    }
                }
            }
        }
        else
        {
            size_t ncols = h.rows();
            size_t nrows = eh.rows() / ncols;
            assert (eh.rows() % ncols == 0);

            for (size_t t=0; t<eh.cols(); t++)
            {
                size_t k=0;
                for (size_t j=0; j<ncols; j++)
                {
                    for (size_t i=0; i<nrows; i++)
                    {
                        if (j == 0) 
                            (*this)(i,t) = eh(k,t) * h(j,t);
                        else
                            (*this)(i,t) += eh(k,t) * h(j,t);
                        k++;
                    }
                }
            }
        }
    }
    void check (const simplematrix & ref)
    {
        for (size_t i = 0; i < rows(); i++) for (size_t j = 0; j < cols(); j++)
        {
            float diff = (*this)(i,j) - ref(i,j);
            if (fabs (diff) > 1e-3)
                fprintf (stderr, "(%d,%d) mismatch %.10f vs. %.10f\n", i, j, (*this)(i,j), ref(i,j));
        }
    }
};

#include <Windows.h>
class auto_timer
{
    LARGE_INTEGER freq, start;
    auto_timer (const auto_timer &); void operator= (const auto_timer &);
public:
    auto_timer()
    {
        if (!QueryPerformanceFrequency (&freq))
            throw std::runtime_error ("auto_timer: QueryPerformanceFrequency failure");
        QueryPerformanceCounter (&start);
    }
    operator double() const     // each read gives time elapsed since start
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter (&end);
        return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    }
};

#ifndef NOCUDA
void test() // a simple test routine
{
    lazyinit();
    size_t ndev = getnumdevices();
    fprintf (stderr, "%d devices\n", ndev);

    // test case
    // make rows==depth to test transposed case
    // With CUBLAS, transposed A seems 1/3.5 slower.
#if 0
    const size_t rows = 16;
    const size_t cols = 16;
    const size_t depth = 16;
#else
    const size_t rows = 16384;
    const size_t cols = 16384;
    const size_t depth = 16384;
#endif
    //fprintf (stderr, "instantiating the matrices with random numbers\n");
    //simplematrix A(rows,depth);
    //simplematrix B(depth,cols);
    //simplematrix C(rows,cols);
    double ops = 2.0 * rows * cols * depth; // 2.0 for mul+add
    //fprintf (stderr, "total %.2f M operations\n", 1e-6 * ops);
    simplematrix A(3,4);
    simplematrix B(2,4);
    simplematrix C(6,4);

    A(0,0)=0.8147f;    A(0,1)=0.9134f;   A(0,2)= 0.2785f;    A(0,3)=0.9649f;
    A(1,0)= 0.9058f;    A(1,1)=0.6324f;    A(1,2)=0.5469f;   A(1,3)=0.1576f;
    A(2,0)=0.1270f;    A(2,1)=0.0975f;    A(2,2)=0.9575f;    A(2,3)=0.9706f;


    B(0,0)=0.9572f;    B(0,1)=0.8003f;   B(0,2)=0.4218f;    B(0,3)=0.7922f;
    B(1,0)=0.4854f;    B(1,1)=0.1419f;    B(1,2)=0.9157f;   B(1,3)=0.9595f;

    C.KhatriRaoProduct(A,B);
    printf("C.KhatriRaoProduct(A,B)=\n");
    for (int i=0;i<(int)C.rows(); i++)
    {
        for (int j=0;j<(int)C.cols(); j++)
            printf("%f  ", C(i,j));
        printf("\n");
    }

    // GPU-side KhatriRaoProduct
    std::auto_ptr<matrix> ap (newmatrix());
    matrix & a = *ap;
    std::auto_ptr<matrix> bp (newmatrix());
    matrix & b = *bp;
    std::auto_ptr<matrix> cp (newmatrix());
    matrix & c = *cp;

    a.allocate (A.rows(), A.cols());
    b.allocate (B.rows(), B.cols());
    c.allocate (C.rows(), C.cols());
    a.assign (0, A.rows(), 0, A.cols(), &A(0,0), A.colstride, false);
    b.assign (0, B.rows(), 0, B.cols(), &B(0,0), B.colstride, false);
    c.KhatriRaoProduct(a, b);
    join(); // wait until copying is complete
    c.fetch (0, C.rows(), 0, C.cols(), &C(0,0), C.colstride, true);
    printf("GPU-side  C.KhatriRaoProduct(A,B)=\n");
    for (int i=0;i<(int)C.rows(); i++)
    {
        for (int j=0;j<(int)C.cols(); j++)
            printf("%f  ", C(i,j));
        printf("\n");
    }

    simplematrix eh(6,2);
    simplematrix h(3,2);
    simplematrix newh(2,2);

    eh(0,0)=0.6557f;    eh(0,1)=0.7431f;
    eh(1,0)=0.0357f;    eh(1,1)=0.3922f;
    eh(2,0)=0.8491f;    eh(2,1)=0.6555f;
    eh(3,0)=0.9340f;    eh(3,1)=0.1712f;
    eh(4,0)=0.6787f;    eh(4,1)=0.7060f;
    eh(5,0)=0.7577f;    eh(5,1)=0.0318f;

    h(0,0)=0.2769f;    h(0,1)=0.8235f;
    h(1,0)=0.0462f;    h(1,1)=0.6948f;
    h(2,0)=0.0971f;    h(2,1)=0.3171f;

    newh.reshapecolumnproduct(eh, h, false);
    printf("newh.reshapecolumnproduct(eh, h, false)\n");
    for (int i=0;i<(int)newh.rows(); i++)
    {
        for (int j=0;j<(int)newh.cols(); j++)
            printf("%f  ", newh(i,j));
        printf("\n");
    }

    newh.reshapecolumnproduct(eh, h, true);
    printf("newh.reshapecolumnproduct(eh, h, true)\n");
    for (int i=0;i<(int)newh.rows(); i++)
    {
        for (int j=0;j<(int)newh.cols(); j++)
            printf("%f  ", newh(i,j));
        printf("\n");
    }

    // GPU-side reshapecolumnproduct
    a.allocate (eh.rows(), eh.cols());
    b.allocate (h.rows(), h.cols());
    c.allocate (newh.rows(), newh.cols());
    a.assign (0, eh.rows(), 0, eh.cols(), &eh(0,0), eh.colstride, false);
    b.assign (0, h.rows(), 0, h.cols(), &h(0,0), h.colstride, false);

    c.reshapecolumnproduct(a, b, false);
    join(); // wait until copying is complete
    c.fetch (0, newh.rows(), 0, newh.cols(), &newh(0,0), newh.colstride, true);
    printf("GPU newh.reshapecolumnproduct(eh, h, false)\n");
    for (int i=0;i<(int)newh.rows(); i++)
    {
        for (int j=0;j<(int)newh.cols(); j++)
            printf("%f  ", newh(i,j));
        printf("\n");
    }

    c.reshapecolumnproduct(a, b, true);
    join(); // wait until copying is complete
    c.fetch (0, newh.rows(), 0, newh.cols(), &newh(0,0), newh.colstride, true);
    printf("GPU newh.reshapecolumnproduct(eh, h, true)\n");
    for (int i=0;i<(int)newh.rows(); i++)
    {
        for (int j=0;j<(int)newh.cols(); j++)
            printf("%f  ", newh(i,j));
        printf("\n");
    }

    return;

    // GPU-side matrix product
    //    fprintf (stderr, "GPU-side allocation\n");
    //    std::auto_ptr<matrix> ap (newmatrix());
    //    matrix & a = *ap;
    //    std::auto_ptr<matrix> bp (newmatrix());
    //    matrix & b = *bp;
    //    std::auto_ptr<matrix> cp (newmatrix());
    //    matrix & c = *cp;
    //
    //    a.allocate (rows,depth);
    //    b.allocate (depth,cols);
    //    c.allocate (rows,cols);
    //    a.assign (0, A.rows(), 0, A.cols(), &A(0,0), A.colstride, false);
    //    b.assign (0, B.rows(), 0, B.cols(), &B(0,0), B.colstride, false);
    //
    //    fprintf (stderr, "GPU-side matrix product\n");
    //    join(); // wait until copying is complete
    //    auto_timer gputimer;
    //#if 0
    //    c.gemm (0.0f, a, true, b, false, 1.0f);
    //    c.gemm (0.5f, a, true, b, false, 0.5f);  // test factors  --very little runtime impact
    //#else
    //    c.gemm (0.0f, a, false, b, false, 1.0f);
    //#endif
    //    join(); // wait until computation is complete
    //    double gputime = gputimer;
    //    fprintf (stderr, "runtime %.4f ms, ### %.2f GFlops/sec ###\n", 1e3 * gputime, 1e-9 * ops / gputime);
    //
    //    simplematrix C1(rows,cols);
    //    c.fetch (0, C1.rows(), 0, C1.cols(), &C1(0,0), C1.colstride, true);
    //
    //    // CPU-side matrix product --note: NOT optimized at all! No SSE, no cache optimality etc.
    //    fprintf (stderr, "CPU-side matrix product\n");
    //    auto_timer cputimer;
    //#if 0
    //    C.matprod_mtm (A, B);
    //#else
    //    C.matprod_mm (A, B);
    //#endif
    //    double cputime = cputimer;
    //    fprintf (stderr, "runtime %.4f sec, %.2f MFlops/sec\n", cputime, 1e-6 * ops / cputime);
    //    fprintf (stderr, "speed-up = %.2f\n", cputime / gputime);
    //
    //    // check correctness
    //    C1.check (C);
    // test dot product

    for (int testit = 0; testit < 6; testit++)
    {
        size_t dim1 = 0;
        size_t dim2 = 0;
        switch(testit)
        {
            case 0: dim1 = 3; dim2 = 4; break;
            case 1: dim1 = 37; dim2 = 1; break;
            case 2: dim1 = 5000; dim2 = 1; break;
            case 3: dim1 = 32; dim2 = 1; break;
            case 4: dim1 = 31; dim2 = 1; break;
            case 5: dim1 = 1; dim2 = 1; break;
        }
        simplematrix Ad(dim1,dim2);
        simplematrix Bd(dim1,dim2);
        simplematrix Cd(dim1,dim2);
        float checkval = 0.0f;
        for (size_t i = 0; i < dim1; i++)
        {
            for (size_t j = 0; j < dim2; j++)
            {
                float val = i > 100 || j > 100 ? 1.0f : (float) (i + j);
                Ad(i,j) = val;
                Bd(i,j) = val;
                checkval += (float) val * val;
            }
        }

        std::auto_ptr<matrix> apd (newmatrix());
        matrix & ad = *apd;
        std::auto_ptr<matrix> bpd (newmatrix());
        matrix & bd = *bpd;

        ad.allocate (Ad.rows(), Ad.cols());
        bd.allocate (Bd.rows(), Bd.cols());


        ad.assign (0, Ad.rows(), 0, Ad.cols(), &Ad(0,0), Ad.colstride, true);
        bd.assign (0, Bd.rows(), 0, Bd.cols(), &Bd(0,0), Bd.colstride, true);

        float r = ad.dot(bd);
        printf("dot product %dx%d (CUDA)  :\t%f\n", dim1, dim2, r);
        printf("dot product %dx%d (CPU)   :\t%f\n", dim1, dim2, checkval);
        assert(r == checkval); // TODO numerical precision ?
    }
}
#endif

};};
