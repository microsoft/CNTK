// cudalattice.cpp -- lattice forward/backward functions for CUDA execution (glue code)
//
// F. Seide, V-hansu

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#define DLLEXPORT
#define __kernel_emulation__    // allow the compilation of CUDA kernels on the CPU
#include "latticefunctionskernels.h"    // for the actual inner kernels and any argument types that are not yet defined in latticestorage.h
#undef __kernel_emulation__
#include "cudalattice.h"        // this exports the class
#include "cudalatticeops.h"     // brings in the actual lattice functions/kernels
#include "cudalib.h"            // generic CUDA helpers
#include "cudadevice.h"
#include <math.h>
#include <memory>               // for auto_ptr
#include <assert.h>
#include <float.h>

namespace msra { namespace cuda {

extern void operator|| (cudaError_t rc, const char * msg);      // TODO: imported from cudamatrix.cpp --better move to cudalib.h
extern void operator|| (CUresult rc, const char * msg);

// this implements the basic operations of exported interface vectorbase<>, from which all vectors derive
// TODO: This really should not be in cudalattice, since it is more general; we need a cudavector.cpp/h
template<typename VECTORTYPE,typename OPSTYPE> class vectorbaseimpl :
    public /*interface*/VECTORTYPE,                 // user-type interface; must derive from vectorbase<VECTORBASE::elemtype>
    public OPSTYPE,                              // type of class that implements the kernels; must derive from vectorref<VECTORBASE::elemtype>
    public objectondevice                           // setdevice()
{
    typedef typename VECTORTYPE::elemtype elemtype; // (for convenience)
    size_t capacity;                                // amount of allocated storage (like capacity() vs. vectorref::n = size())
    void release() { ondevice no (deviceid); free (this->reset (NULL, 0)); }
public:
    vectorbaseimpl() : capacity (0) { }
    ~vectorbaseimpl() { release(); }
    void setdevice (size_t deviceid)            // just remembers it here
    {
        if (deviceid == getdevice()) return;    // no change
        if (size() != 0)                        // cannot migrate across devices
            throw std::logic_error ("setdevice: device cannot be changed once matrix is allocated");
        if (capacity != 0)                      // if we still hold memory, it is valid to get rid of it
            release();
        objectondevice::setdevice (deviceid);
    }
    void allocate (size_t sz)
    {
        if (sz > capacity)                                      // need to grow
        {
            ondevice no (deviceid);                             // switch to desired CUDA card
            cuda_ptr<elemtype> pnew = malloc<elemtype> (sz);    // allocate memory inside CUDA device (or throw)
            capacity = sz;                                      // if succeeded then: remember
            cuda_ptr<elemtype> p = this->reset (pnew, sz);            //  and swap the pointers and update n
            free (p);                                           //  then release the old one
        }
        else                                                    // not growing: keep same allocation
            this->reset (this->get(), sz);
    }
    size_t size() const throw() { return vectorref<elemtype>::size(); }
    void assign (const elemtype * p, size_t nelem, bool synchronize)
    {
        allocate (nelem);           // assign will resize the target appropriately
        ondevice no (deviceid);     // switch to desired CUDA card
        if (nelem > 0)
            memcpy (this->get(), 0, p, nelem);
        if (synchronize)
            join();
    }
    void fetch (elemtype * p, size_t nelem, bool synchronize) const
    {
        if (nelem != size())        // fetch() cannot resize the target; caller must do that
            throw std::logic_error ("fetch: vector size mismatch");
        ondevice no (deviceid);     // switch to desired CUDA card
        if (nelem > 0)
            memcpy (p, this->get(), 0, nelem);
        if (synchronize)
            join();
    };
};

// ---------------------------------------------------------------------------
// glue code for lattice-related classes
// The XXXvectorimpl classes must derive from vectorbaseimpl<XXXvector,XXXvectorops>.
// For classes without kernels that operate on the vector, XXXvectorimpl is not
// needed, use vectorbaseimpl<XXXvector,vectorref<XXX>> instead, where
// XXXvector is an alias for vectorbase<XXX> (but better keep that alias in cudalattice.h
// to document which vectors are implemented).
// ---------------------------------------------------------------------------

matrixref<float> tomatrixref(const Microsoft::MSR::CNTK::Matrix<float>& m)
{
    return matrixref<float>(m.BufferPointer(), m.GetNumRows(), m.GetNumCols(), m.GetNumRows());
}

class latticefunctionsimpl : public vectorbaseimpl<latticefunctions,latticefunctionsops>
{
    void edgealignment (const lrhmmdefvector & hmms, const lr3transPvector & transPs, const size_t spalignunitid, 
                        const size_t silalignunitid, const Microsoft::MSR::CNTK::Matrix<float>& logLLs, const nodeinfovector & nodes, 
                        const edgeinfowithscoresvector & edges, const aligninfovector & aligns,
                        const uintvector & alignoffsets, ushortvector & backptrstorage, const sizetvector & backptroffsets,
                        ushortvector & alignresult, floatvector & edgeacscores)         // output
    {
        ondevice no (deviceid); 

        matrixref<float> logLLsMatrixRef = tomatrixref(logLLs);
        latticefunctionsops::edgealignment (dynamic_cast<const vectorbaseimpl<lrhmmdefvector, vectorref<lrhmmdef>> &> (hmms), 
                                            dynamic_cast<const vectorbaseimpl<lr3transPvector, vectorref<lr3transP>> &> (transPs),
                                            spalignunitid, silalignunitid, logLLsMatrixRef,
                                            dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                            dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges), 
                                            dynamic_cast<const vectorbaseimpl<aligninfovector, vectorref<msra::lattices::aligninfo>> &> (aligns),
                                            dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                            dynamic_cast<vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (backptrstorage), 
                                            dynamic_cast<const vectorbaseimpl<sizetvector, vectorref<size_t>> &> (backptroffsets),
                                            dynamic_cast<vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignresult), 
                                            dynamic_cast<vectorbaseimpl<floatvector, vectorref<float>> &> (edgeacscores));
    }
    
    void forwardbackwardlattice (const size_t * batchsizeforward, const size_t * batchsizebackward, 
                                 const size_t numlaunchforward, const size_t numlaunchbackward,
                                 const size_t spalignunitid, const size_t silalignunitid,
                                 const floatvector & edgeacscores, const edgeinfowithscoresvector & edges, 
                                 const nodeinfovector & nodes, const aligninfovector & aligns, 
                                 const ushortvector & alignments, const uintvector & alignoffsets,
                                 doublevector & logpps, doublevector & logalphas, doublevector & logbetas,
                                 const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect, 
                                 const ushortvector & uids, const ushortvector & senone2classmap, doublevector & logaccalphas, 
                                 doublevector & logaccbetas, doublevector & logframescorrectedge,
                                 doublevector & logEframescorrect, doublevector & Eframescorrectbuf, double & logEframescorrecttotal, double & totalfwscore)
    {
        ondevice no (deviceid);
        latticefunctionsops::forwardbackwardlattice (batchsizeforward, batchsizebackward, numlaunchforward, numlaunchbackward,
                                                     spalignunitid, silalignunitid,
                                                     dynamic_cast<const vectorbaseimpl<floatvector, vectorref<float>> &> (edgeacscores),
                                                     dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges), 
                                                     dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                                     dynamic_cast<const vectorbaseimpl<aligninfovector, vectorref<msra::lattices::aligninfo>> &> (aligns),
                                                     dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignments), 
                                                     dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets), 
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logpps), 
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logalphas), 
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logbetas),
                                                     lmf, wp, amf, boostingfactor, returnEframescorrect,
                                                     dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (uids),
                                                     dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (senone2classmap),
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logaccalphas),
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logaccbetas),
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logframescorrectedge),
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (logEframescorrect),
                                                     dynamic_cast<vectorbaseimpl<doublevector, vectorref<double>> &> (Eframescorrectbuf),
                                                     logEframescorrecttotal, totalfwscore);
    }

    void sMBRerrorsignal (const ushortvector & alignstateids, 
                          const uintvector & alignoffsets,
                          const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                          const doublevector & logpps, const float amf, const doublevector & logEframescorrect,
                          const double logEframescorrecttotal, Microsoft::MSR::CNTK::Matrix<float>& dengammas, Microsoft::MSR::CNTK::Matrix<float>& dengammasbuf)
    {
        ondevice no (deviceid);

        matrixref<float> dengammasMatrixRef = tomatrixref(dengammas);
        matrixref<float> dengammasbufMatrixRef = tomatrixref(dengammasbuf);
        latticefunctionsops::sMBRerrorsignal (dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignstateids),
                                              dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                              dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges),
                                              dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                              dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logpps),
                                              amf,
                                              dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logEframescorrect),
                                              logEframescorrecttotal, dengammasMatrixRef, dengammasbufMatrixRef);
    }

    void mmierrorsignal (const ushortvector & alignstateids, const uintvector & alignoffsets,
                         const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                         const doublevector & logpps, Microsoft::MSR::CNTK::Matrix<float>& dengammas)
    {
        ondevice no (deviceid);

        matrixref<float> dengammasMatrixRef = tomatrixref(dengammas);
        latticefunctionsops::mmierrorsignal (dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignstateids),
                                             dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                             dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges),
                                             dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                             dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logpps),
                                             dengammasMatrixRef);
    }

    void stateposteriors (const ushortvector & alignstateids, const uintvector & alignoffsets,
                          const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                          const doublevector & logqs, Microsoft::MSR::CNTK::Matrix<float>& logacc)
    {
        ondevice no (deviceid);

        matrixref<float> logaccMatrixRef = tomatrixref(logacc);
        latticefunctionsops::stateposteriors (dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignstateids),
                                              dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                              dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges),
                                              dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                              dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logqs),
                                              logaccMatrixRef);
    }
};

latticefunctions * newlatticefunctions() { lazyinit(); return new latticefunctionsimpl(); }

// implementation of lrhmmdefvector
// Class has no vector-level member functions, so no need for an extra type
lrhmmdefvector * newlrhmmdefvector() { lazyinit(); return new vectorbaseimpl<lrhmmdefvector,vectorref<lrhmmdef>>(); }
lr3transPvector * newlr3transPvector() { lazyinit(); return new vectorbaseimpl<lr3transPvector,vectorref<lr3transP>>(); }
ushortvector * newushortvector() { lazyinit(); return new vectorbaseimpl<ushortvector,vectorref<unsigned short>>(); }
uintvector * newuintvector() { lazyinit(); return new vectorbaseimpl<uintvector,vectorref<unsigned int>>(); }
floatvector * newfloatvector() { lazyinit(); return new vectorbaseimpl<floatvector,vectorref<float>>(); }
doublevector * newdoublevector() { lazyinit(); return new vectorbaseimpl<doublevector,vectorref<double>>(); }
sizetvector * newsizetvector() { lazyinit(); return new vectorbaseimpl<sizetvector,vectorref<size_t>>(); }
nodeinfovector * newnodeinfovector() { lazyinit(); return new vectorbaseimpl<nodeinfovector,vectorref<nodeinfo>>(); }
edgeinfowithscoresvector * newedgeinfovector() { lazyinit(); return new vectorbaseimpl<edgeinfowithscoresvector,vectorref<edgeinfowithscores>>(); }
aligninfovector * newaligninfovector() { lazyinit(); return new vectorbaseimpl<aligninfovector,vectorref<aligninfo>>(); }

};};
