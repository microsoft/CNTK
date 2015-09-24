// cudamatrixops.cpp -- cudamatrixbase class; all cublas calls are encapsulated in this file
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudamatrixops.cpp $
// 
// 23    10/29/12 3:43p T-simonw
// add dot product and norm methods
// 
// 22    9/18/12 11:15a Fseide
// made asum() const
// 
// 21    9/18/12 11:06a Fseide
// implemented asum() and adadenom()
// 
// 20    7/17/12 5:32p Adame
// Update for no-sync framework
// async copy fixes
// 
// 19    6/08/12 9:32p V-xieche
// delete code related to delayupdate.
// 
// 18    6/06/12 5:12p Adame
// Copy Sync update
// 
// 17    4/01/12 7:16a V-xieche
// undid forbidding of partial-height copies
// 
// 16    4/01/12 7:13a V-xieche
// disabled _p2p copying
// 
// 15    4/01/12 8:47p Fseide
// changed assignmatrix_ua() to assignmatrix_p2p() using explicit
// contexts;
// new methods for that: getdevicecontext()
// 
// 14    4/01/12 4:47p Fseide
// added code for peer-to-peer access without UA, but not enabled yet
// 
// 13    3/31/12 9:53p Fseide
// (documented a potential bug)
// 
// 12    3/31/12 9:50p Fseide
// update to assignmatrix_ua() to use linear copy when possible
// 
// 11    3/31/12 8:24p Fseide
// new method assignmatrix_ua()
// 
// 10    2/25/12 5:24p V-xieche
// Add helpler function for coping date in CUDA device
// 
// 9     10/28/11 14:52 Fseide
// cleaned up confusing and inconsistent alpha and beta parameters in
// gemm-like functions, now calling them 'thisscale' and 'otherweight' to
// make it crystal-clear
// 
// 8     2/25/11 2:06p Fseide
// gems() no longer implemented using CUBLAS but with our own kernel to
// support non-contiguous memory
// 
// 7     2/24/11 9:50p Fseide
// new methods assign() and fetch(), to allow for non-contiguous transfers
// 
// 6     2/11/11 3:31p Fseide
// commented out unused (old) code
// 
// 5     2/02/11 8:22a Fseide
// gemm() now allows B to be transposed as well
// 
// 4     2/01/11 6:44p Fseide
// fixed an assert()
// 
// 3     2/01/11 4:53p Fseide
// gems() implemented;
// addcol() removed
// 
// 2     2/01/11 3:47p Fseide
// now clearly encapsulates all cublas calls
// 
// 1     2/01/11 3:42p Fseide
// moved out from cudamatrix.cpp, to isolate cublas operations

// move this to separate source file
#include <cublas.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cudamatrixops.h"
#include "cudalib.h"        // generic CUDA helpers
#include "cudadevice.h"
#include <string>
#include <assert.h>

#pragma comment (lib, "cublas.lib")     // link CUDA BLAS library

namespace msra { namespace cuda {

// allows to write cublasFunction() || cublasfailure ("error")
typedef std::string cublasfailure;
static void operator|| (cublasStatus rc, const cublasfailure & msg)
{
    if (rc != CUBLAS_STATUS_SUCCESS)
    {
        char buf[1000];
        sprintf_s (buf, "%s: cublas error code %d", msg.c_str(), rc);    // ... TODO: add error message
        throw std::runtime_error (buf);
    }
}
typedef std::string cudafailure;
static void operator|| (cudaError_t rc, const cudafailure & msg)
{
    if (rc != cudaSuccess)
    {
        char buf[1000];
        sprintf_s (buf, "%s: launch failure: %s (cuda error %d)", msg.c_str(), cudaGetErrorString (rc), rc);
        throw std::runtime_error (buf);
    }
}
static void operator|| (CUresult rc, const cudafailure & msg)
{
    if (rc != cudaSuccess)
    {
        char buf[1000];
        sprintf_s (buf, "%s: cuda API error %d", msg, rc);
        throw std::runtime_error (buf);
    }
}

// assign to a patch from a host-based matrix (given by address of first element and colstride)
void cudamatrixops::assign (size_t i0, size_t i1, size_t j0, size_t j1/*dst rgn*/, const float * otherpi0j0, size_t othercolstride/*src*/)
{
    assert (i1 <= rows() && j1 <= cols() && i1 >= i0 && j1 >= j0 && i1 - i0 <= othercolstride);
    if (i1 > i0 && j1 > j0)
        cublasSetMatrixAsync (int(i1 - i0), int (j1 - j0), sizeof (float), otherpi0j0, (int) othercolstride, &(*this)(i0,j0), (int) colstride, GetCurrentStream()) || cublasfailure ("cublasSetMatrix");
}

// copy a patch to a host-based matrix (given by address of first element and colstride)
void cudamatrixops::fetch (size_t i0, size_t i1, size_t j0, size_t j1/*src rgn*/, float * otherpi0j0, size_t othercolstride/*dst*/) const
{
    assert (i1 <= rows() && j1 <= cols() && i1 >= i0 && j1 >= j0 && i1 - i0 <= othercolstride);
    if (i1 > i0 && j1 > j0)
        cublasGetMatrixAsync (int (i1 - i0), int (j1 - j0), sizeof (float), &(*this)(i0,j0), (int) colstride, otherpi0j0, (int) othercolstride, GetCurrentStream()) || cublasfailure ("cublasGetMatrix");
}

#ifdef  _WIN64
// cross-device assignment
// This function can only be called in unified-addressing mode, which is only available in x64.
///*static*/ void cudamatrixops::assignmatrix_p2p (cudamatrixops & dst, CUcontext dstContext, const cudamatrixops & src, CUcontext srcContext)
/*static*/ void cudamatrixops::assignmatrix_ua (cudamatrixops & dst, const cudamatrixops & src)
{
    assert (dst.rows() == src.rows() && dst.cols() == src.cols() && dst.colstride == src.colstride);
    const size_t elsize = sizeof(float);
    // BUGBUG: we don't really know if the following is indeed full height; it could be a near-full height patch... We will ignore this for now since we know we don't do that.
    if (((dst.rows() + 3) & ~3) == dst.colstride)   // full height: use a linear copy  --TODO: test this
#if 0   // TODO: try this
        cuMemcpyPeer ((CUdeviceptr) dst.p.get(), dstContext, (CUdeviceptr) src.p.get(), srcContext, elsize * dst.colstride * dst.cols()) || cudafailure ("cuMemcpyPeer");
#else
        cudaMemcpy (dst.p.get(), src.p.get(), elsize * dst.colstride * dst.cols(), cudaMemcpyDefault) || cudafailure ("cudaMemcpy");
#endif
    else                                            // use 2D array copy  --TODO: this crashes; alignment issue?
        //throw std::runtime_error ("assignmatrix_p2p: partial-height assignments not allowed");
        cudaMemcpy2D (dst.p.get(), dst.colstride * elsize, src.p.get(), src.colstride * elsize, dst.rows() * elsize, dst.cols(), cudaMemcpyDefault) || cudafailure ("cudaMemcpy2D");
}
#endif

// this = this * thisscale + A * B * ABweight, where A is stored as its transpose if 'Aistransposed'
void cudamatrixops::gemm (float thisscale, const cudamatrixops & A, bool Aistransposed, const cudamatrixops & B, bool Bistransposed, float ABweight)
{
    cudamatrixops & C = *this;
    // matrix dimension parameters
    const size_t m = Aistransposed ? A.cols() : A.rows();
    const size_t n = Bistransposed ? B.rows() : B.cols();
    const size_t k = Aistransposed ? A.rows() : A.cols();
    assert (rows() == m && cols() == n && k == (Bistransposed ? B.cols() : B.rows()) && n == C.cols());
    // matrix storage parameters
    const size_t lda = A.getcolstride();
    const size_t ldb = B.getcolstride();
    const size_t ldc = C.getcolstride();
    // matrix pointers
    cuda_ptr<float> A00 = A.p;    // TODO: 1-based correction?
    cuda_ptr<float> B00 = B.p;
    cuda_ptr<float> C00 = C.p;
	cublasSetKernelStream(GetCurrentStream());
    // This function seems limited to 2048 components, but inconsistent. Another meh! moment!
    cublasSgemm (Aistransposed ? 't' : 'n', Bistransposed ? 't' : 'n',
        (int) m, (int) n, (int) k,
        ABweight, A00.get(), (int) lda, B00.get(), (int) ldb,
        thisscale, C00.get(), (int) ldc);
    cublasGetError() || cublasfailure ("cublasSgemm");
}

// computes dot product of two cuda vectors
// for vectors in matrix format: compute add dot-products of columns
float cudamatrixops::dot(const cudamatrixops & b) const {
    const cudamatrixops & a = *this;
    // dimension parameters
    size_t n = a.getcolstride();
    size_t m = a.cols();
    assert (a.rows() == b.rows());
    assert (a.cols() == b.cols());
    // matrix pointers
    cuda_ptr<float> A00 = a.p;
    cuda_ptr<float> B00 = b.p;
    cublasSetKernelStream(GetCurrentStream());
    float result = 0.0;
    result = cublasSdot ((int) (n*m), A00.get(), 1, B00.get(), 1);
    cublasGetError() || cublasfailure ("cublasSdot");
    return result;
    
}

float cudamatrixops::nrm2() const {
    const cudamatrixops & a = *this;
    // dimension parameters
    size_t n = a.rows();
    size_t m = a.cols();
    // matrix pointers
    cuda_ptr<float> A00 = a.p;
    cublasSetKernelStream(GetCurrentStream());
    float result = 0.0;
    result = cublasSnrm2 ((int) (n*m), A00.get(), 1);
    cublasGetError() || cublasfailure ("cublasSnrm2");
    return result;
    
}

#if 0   // replacing this by own kernel to allow for mismatching colstride/patches
// this = this * thisscale + other * otherweight
void cudamatrixops::gems (float thisscale, const cudamatrixops & other, float otherweight)
{
    assert (cols() == other.cols() && rows() == other.rows() && getcolstride() == other.getcolstride());
    // Cublas has no matrix addition; so we misuse the vector addition with fake dimensions.
    const size_t n = getcolstride() * cols();
    // this *= thisscale
    if (thisscale != 1.0f)
    {
        cublasSscal ((int) n, thisscale, p.get(), 1);    // in-place
        cublasGetError() || cublasfailure ("cublasSscal");
    }
    // this += other * otherweight
    cublasSaxpy ((int) n, otherweight, other.p.get(), 1, p.get(), 1);
    cublasGetError() || cublasfailure ("cublasSaxpy");
}
#endif

// return sum of abs values of all elements
// BUGBUG: This will not work correctly on stripes unless the padding is all zero.
float cudamatrixops::asum() const
{
    float retval = cublasSasum ((int) (cols() * getcolstride()), p.get(), 1/*incx*/);
    cublasGetError() || cublasfailure ("cublasSasum");
    return retval;
}

};};
