// cudalattice.h -- lattice forward/backward functions CUDA execution
//
// F. Seide, V-hansu

#pragma once
#include <stdexcept>                 // (for NOCUDA version only)
#include "latticestorage.h"          // basic data types for storing lattices
#define __kernel_emulation__         // allow the compilation of CUDA kernels on the CPU
#include "latticefunctionskernels.h" // for data types passed to kernel functions
#undef __kernel_emulation__

#include "Matrix.h"

namespace msra { namespace cuda {

// ---------------------------------------------------------------------------
// how to add vector types here:
//  - for types with methods that operate on the vector, follow the example of 'somedatavector'
//  - for types without extra methods, follow the example of 'ushortvector'
//  - the total set of classes that need to be defined is:
//     - XXXvector       --the interface that the calling code sees (in this header file)
//     - XXXvectorimpl   --glue for each method: switch CUDA device, then call kernel (in XXXvectorops, cudalattice.cpp)
//     - XXXvectorops    --contains the actual kernel functions (cudalatticeops.h, cudalatticeops.cu.h)
//  - comments:
//     - follow precisely this pattern
//     - for non-vector elements you need to pass by reference, you can still use this with a size-1 vector
//     - and for functions that have no 'this', use a size-0 vector
//     - note that you cannot add data members to XXXvector, as there is no mechanism to sync between GPU and CPU; only vector elements are synced
//     - remember to Get Latest in both projects to see the change, for cudalattice.h and latticestorage.h
// ---------------------------------------------------------------------------

// a vector type; use this as a basetype
// Note that this will only be instantiated for types known inside this lib, and a newvector<> function must be exported for each.
template <typename ELEMTYPE>
struct /*interface*/ vectorbase
{
    virtual ~vectorbase()
    {
    }
    virtual void allocate(size_t n) = 0;
    virtual size_t size() const throw() = 0;
    virtual void assign(const ELEMTYPE* p, size_t n, bool synchronize) = 0;
    template <class VECTOR>
    void assign(const VECTOR& v, bool synchronize)
    {
        allocate(v.size());
        if (!v.empty())
            assign(&v[0], v.size(), synchronize);
    }
    virtual void fetch(ELEMTYPE* p, size_t n, bool synchronize) const = 0;
    template <class VECTOR>
    void fetch(VECTOR& v, bool synchronize) const
    {
        v.resize(size());
        if (!v.empty())
            fetch(&v[0], v.size(), synchronize);
    }
    typedef ELEMTYPE elemtype;
};

// ---------------------------------------------------------------------------
// vectors of these with custom functions
// The XXXvector classes must derive from vectorbase<XXX>.
// ---------------------------------------------------------------------------

struct somedatavector : public vectorbase<msra::lattices::somedata>
{
    // must implement all members of vectorbase<>, and can add operations here
    virtual int somedataoperation(size_t arg) = 0;
};

typedef vectorbase<unsigned short> ushortvector;
typedef vectorbase<float> floatvector;
typedef vectorbase<double> doublevector;
typedef vectorbase<unsigned int> uintvector;
typedef vectorbase<size_t> sizetvector;
typedef vectorbase<msra::lattices::lrhmmdef> lrhmmdefvector;
typedef vectorbase<msra::lattices::lr3transP> lr3transPvector;
typedef vectorbase<msra::lattices::nodeinfo> nodeinfovector;
typedef vectorbase<msra::lattices::edgeinfowithscores> edgeinfowithscoresvector;
typedef vectorbase<msra::lattices::aligninfo> aligninfovector;

struct latticefunctions : public vectorbase<msra::lattices::empty>
{
    virtual void edgealignment(const lrhmmdefvector& hmms, const lr3transPvector& transPs, const size_t spalignunitid,
                               const size_t silalignunitid, const Microsoft::MSR::CNTK::Matrix<float>& logLLs, const nodeinfovector& nodes,
                               const edgeinfowithscoresvector& edges, const aligninfovector& aligns,
                               const uintvector& alignoffsets, ushortvector& backptrstorage, const sizetvector& backptroffsets,
                               ushortvector& alignresult, floatvector& edgeacscores) = 0; // output
    virtual void forwardbackwardlattice(const size_t* batchsizeforward, const size_t* batchsizebackward,
                                        const size_t numlaunchforward, const size_t numlaunchbackward,
                                        const size_t spalignunitid, const size_t silalignunitid,
                                        const floatvector& edgeacscores, const edgeinfowithscoresvector& edges,
                                        const nodeinfovector& nodes, const aligninfovector& aligns,
                                        const ushortvector& alignoutput, const uintvector& alignoffsets,
                                        doublevector& logpps, doublevector& logalphas, doublevector& logbetas,
                                        const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect,
                                        const ushortvector& uids, const ushortvector& senone2classmap,
                                        doublevector& logaccalphas, doublevector& logaccbetas,
                                        doublevector& logframescorrectedge, doublevector& logEframescorrect,
                                        doublevector& Eframescorrectbuf, double& logEframescorrecttotal, double& totalfwscore) = 0;
    virtual void sMBRerrorsignal(const ushortvector& alignstateids, const uintvector& alignoffsets,
                                 const edgeinfowithscoresvector& edges, const nodeinfovector& nodes,
                                 const doublevector& logpps, const float amf, const doublevector& logEframescorrect,
                                 const double logEframescorrecttotal, Microsoft::MSR::CNTK::Matrix<float>& dengammas, Microsoft::MSR::CNTK::Matrix<float>& dengammasbuf) = 0;
    virtual void mmierrorsignal(const ushortvector& alignstateids, const uintvector& alignoffsets,
                                const edgeinfowithscoresvector& edges, const nodeinfovector& nodes,
                                const doublevector& logpps, Microsoft::MSR::CNTK::Matrix<float>& dengammas) = 0;
    virtual void stateposteriors(const ushortvector& alignstateids, const uintvector& alignoffsets,
                                 const edgeinfowithscoresvector& edges, const nodeinfovector& nodes,
                                 const doublevector& logqs, Microsoft::MSR::CNTK::Matrix<float>& logacc) = 0;
};

// ---------------------------------------------------------------------------
// factor methods
// ---------------------------------------------------------------------------

somedatavector* newsomedatavector(size_t deviceid);
ushortvector* newushortvector(size_t deviceid);
uintvector* newuintvector(size_t deviceid);
floatvector* newfloatvector(size_t deviceid);
doublevector* newdoublevector(size_t deviceid);
sizetvector* newsizetvector(size_t deviceid);
latticefunctions* newlatticefunctions(size_t deviceid);
lrhmmdefvector* newlrhmmdefvector(size_t deviceid);
lr3transPvector* newlr3transPvector(size_t deviceid);
nodeinfovector* newnodeinfovector(size_t deviceid);
edgeinfowithscoresvector* newedgeinfovector(size_t deviceid);
aligninfovector* newaligninfovector(size_t deviceid);
};
};
