// cudalattice.cpp -- lattice forward/backward functions for CUDA execution (glue code)
//
// F. Seide, Aug 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudalattice.cpp $
// 
// 62    11/21/12 8:58p V-hansu
// rename state2classmap to senone2classmap
// 
// 61    11/21/12 7:10p V-hansu
// rename statetoclassmap to state2classmap
// 
// 60    11/21/12 6:04p V-hansu
// add statetoclassmap to forwardbackward() to get prepared for mpe
// approximation
// 
// 59    11/05/12 12:26a V-hansu
// undo last check in :( not got enough time to get it done
// 
// 58    11/02/12 9:44a V-hansu
// add logframescorrect to edgealignment(), prepare for moving computation
// from forwardbackward() to edgealignment()
// 
// 57    10/29/12 3:36p V-hansu
// add boosting factor to prepare for BMMI
// 
// 56    10/19/12 7:51p V-hansu
// adjust the order of interface arguments, finish some TODO
// 
// 55    10/19/12 3:55p V-hansu
// finish the mmierrorsignal, not tested yet
// 
// 54    10/17/12 5:59p V-hansu
// rename Eframescorrectotal to logEframescorrectotal, rename
// Eframescorrectbuf or Eframescorrectdiff to logEframescorrect
// 
// 53    10/17/12 3:33p V-hansu
// turn eframecorrect (eframecorrectdiff) from float vector to double
// vector
// 
// 52    10/17/12 2:03p Fseide
// new method stateposteriors()
// 
// 51    10/15/12 5:25p V-hansu
// add aligns to forwardbackwardlattice()
// 
// 50    10/14/12 8:26p V-hansu
// add silunitid and spunitid to forwardbackwardlattice
// 
// 49    9/30/12 5:32p V-hansu
// fix syntax error to make build
// 
// 48    9/30/12 5:21p V-hansu
// add backptr to edgealignment related functions
// 
// 47    9/28/12 6:08p V-hansu
// rename spalignunit to spalignunitid
// 
// 46    9/28/12 2:56p V-hansu
// rename transPindex to alignindex
// 
// 45    9/26/12 7:36p V-hansu
// add siltransPindex, rename returnEframecorrect to returnEframescorrect 
// 
// 44    9/26/12 2:29p V-hansu
// change logpps in errorsignal from float to double. remove
// Eframecorrecttotal in sMBRerrorsignal, change the location of resize
// towards logpps and Eframecorrect
// 
// 43    9/26/12 1:52p V-hansu
// rename Eframescorrect to Eframescorrectbuf, and add Eframescorrectdiff
// for difference computation.
// 
// 42    9/26/12 1:08p V-hansu
// rename combinemode to returnEframescorrect
// 
// 41    9/26/12 12:55p Fseide
// (rermoved some old demo code)
// 
// 40    9/26/12 12:53p Fseide
// errorsignal() renamed to sMBRerrorsignal()
// 
// 39    9/26/12 12:27p Fseide
// renamed logdengammaspos/neg to dengammas/dengammasbuf
// 
// 38    9/26/12 11:57a Fseide
// sMBRerrorsignal() now takes two dengammas accumulators, in prep for pos/neg
// logadd
// 
// 37    9/25/12 3:11p V-hansu
// add sizetvector and change uids from uintvector into sizetvector
// 
// 36    9/25/12 1:12p V-hansu
// add alignemts and alignmentoffsets to forwardlattce related function to
// finish the algorithm
// 
// 35    9/24/12 10:07p V-hansu
// change the interface relating to forwardbackwardlattice to get prepared
// for the combined mode fwbw, not finished yet
// 
// 34    9/21/12 3:55p V-hansu
// change the interface of latticefunctionsops::forwardbackwardlattice to
// pass in batchsizeforward and batchsizebackward by pointer
// 
// 33    9/19/12 9:33a Fseide
// renamed edgeinfo to edgeinfowithscores, in prep for V2 lattice format
// 
// 32    9/16/12 9:31p V-hansu
// add atomicLogAdd, not finished. add doublevector
// 
// 31    9/16/12 8:57p V-hansu
// add doublevector
// 
// 30    9/14/12 2:37p V-hansu
// add forwardlatticej and forwardbackwardlattice in
// latticefunctionskernels and related classes
// 
// 29    9/07/12 5:45p V-hansu
// delete previous checking line
// 
// 28    9/07/12 12:02p V-hansu
// add a print in allocate function
// 
// 27    9/06/12 8:02p Fseide
// changed __device__ to __kernel_emulation__ to signal
// latticefunctionskernels.h to define emulation versions of CUDA
// functions
// 
// 26    9/06/12 7:24p V-hansu
// add alignoffsets into interface, same as alignstateids
// 
// 25    9/05/12 10:36p V-hansu
// add function sMBRerrorsignal and codes relating to it
// 
// 24    9/05/12 8:50a V-hansu
// delete testing code
// 
// 23    9/04/12 10:26p V-hansu
// changed function edgealignment, not finished yet
// 
// 22    9/04/12 4:57p V-hansu
// add realization of something like uintvector
// 
// 21    9/04/12 4:39p V-hansu
// (change an interface)
// 
// 20    9/04/12 4:37p V-hansu
// change an interface
// 
// 19    9/04/12 1:43p Fseide
// factored out the deviceid into a base class 'objectondevice' shared
// between vectorbaseimpl and matriximpl
// 
// 18    9/04/12 1:37p Fseide
// (fixed a typo that caused a compiler warning)
// 
// 17    9/03/12 4:54p V-hansu
// change the interface of edgealignment
// 
// 16    9/01/12 7:29p V-hansu
// add implementation of nodeinfovector, edgeinfowithscoresvector, aligninfovector
// 
// 15    9/01/12 2:58p Fseide
// now uses dynamic_cast to cast from interfaces to actual types
// 
// 13    8/31/12 10:58p V-hansu
// add class latticefunctionsimpl
// 
// 12    8/28/12 6:42p Fseide
// (added more comments)
// 
// 11    8/28/12 6:28p Fseide
// split off the -ops class
// 
// 10    8/28/12 5:39p Fseide
// hack-implementation of somedataoperation() for testing CUDA vectors
// 
// 9     8/28/12 4:59p Fseide
// added ushortvector and some minor code changes
// 
// 8     8/28/12 4:06p Fseide
// rename vectorbase::resize() to allocate(), since it does not resize()
// like STL which retains the content;
// fixed ~vectorbase()
// 
// 7     8/28/12 3:58p Fseide
// bug fix: assign() and fetch() now have ondevice logic
// 
// 6     8/28/12 3:56p Fseide
// implemented vectorbase::assign() and fetch() (not tested)
// 
// 5     8/28/12 3:27p Fseide
// implemented resizing
// 
// 4     8/28/12 3:14p Fseide
// added a std::vector<> version of fetch() and assign() to vectorbase<>
// 
// 3     8/28/12 3:05p Fseide
// (fixed name of example newdatavector() -> newsomedatavector())
// 
// 2     8/28/12 2:59p Fseide
// (fixed tabs...meh)
// 
// 1     8/28/12 2:51p Fseide
// newly created cudalattice and latticestorage, as a frame for lattice
// forward/backward executed on CUDA

#define DLLEXPORT
#define __kernel_emulation__    // allow the compilation of CUDA kernels on the CPU
#include "latticefunctionskernels.h"    // for the actual inner kernels and any argument types that are not yet defined in latticestorage.h
#undef __kernel_emulation__
#include "cudamatrix.h"
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
    void release() { ondevice no (deviceid); free (reset (NULL, 0)); }
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
            cuda_ptr<elemtype> p = reset (pnew, sz);            //  and swap the pointers and update n
            free (p);                                           //  then release the old one
        }
        else                                                    // not growing: keep same allocation
            reset (get(), sz);
    }
    size_t size() const throw() { return vectorref::size(); }
    void assign (const elemtype * p, size_t nelem, bool synchronize)
    {
        allocate (nelem);           // assign will resize the target appropriately
        ondevice no (deviceid);     // switch to desired CUDA card
        if (nelem > 0)
            memcpy (get(), 0, p, nelem);
        if (synchronize)
            join();
    }
    void fetch (typename elemtype * p, size_t nelem, bool synchronize) const
    {
        if (nelem != size())        // fetch() cannot resize the target; caller must do that
            throw std::logic_error ("fetch: vector size mismatch");
        ondevice no (deviceid);     // switch to desired CUDA card
        if (nelem > 0)
            memcpy (p, get(), 0, nelem);
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

extern const matrixref<float> & tomatrixrefconst (const matrix & m);
matrixref<float> & tomatrixref (matrix & m) ;

class latticefunctionsimpl : public vectorbaseimpl<latticefunctions,latticefunctionsops>
{
    void edgealignment (const lrhmmdefvector & hmms, const lr3transPvector & transPs, const size_t spalignunitid, 
                        const size_t silalignunitid, const msra::cuda::matrix & logLLs, const nodeinfovector & nodes, 
                        const edgeinfowithscoresvector & edges, const aligninfovector & aligns,
                        const uintvector & alignoffsets, ushortvector & backptrstorage, const sizetvector & backptroffsets,
                        ushortvector & alignresult, floatvector & edgeacscores)         // output
    {
        ondevice no (deviceid); 

        latticefunctionsops::edgealignment (dynamic_cast<const vectorbaseimpl<lrhmmdefvector, vectorref<lrhmmdef>> &> (hmms), 
                                            dynamic_cast<const vectorbaseimpl<lr3transPvector, vectorref<lr3transP>> &> (transPs),
                                            spalignunitid, silalignunitid, tomatrixrefconst (logLLs), 
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
                          const double logEframescorrecttotal, msra::cuda::matrix & dengammas, msra::cuda::matrix & dengammasbuf) 
    {
        ondevice no (deviceid);

        latticefunctionsops::sMBRerrorsignal (dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignstateids),
                                              dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                              dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges),
                                              dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                              dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logpps),
                                              amf,
                                              dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logEframescorrect),
                                              logEframescorrecttotal, tomatrixref (dengammas), tomatrixref (dengammasbuf));
    }

    void mmierrorsignal (const ushortvector & alignstateids, const uintvector & alignoffsets,
                         const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                         const doublevector & logpps, msra::cuda::matrix & dengammas)
    {
        ondevice no (deviceid);

        latticefunctionsops::mmierrorsignal (dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignstateids),
                                             dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                             dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges),
                                             dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                             dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logpps),
                                             tomatrixref (dengammas));
    }

    void stateposteriors (const ushortvector & alignstateids, const uintvector & alignoffsets,
                          const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                          const doublevector & logqs, msra::cuda::matrix & logacc) 
    {
        ondevice no (deviceid);

        latticefunctionsops::stateposteriors (dynamic_cast<const vectorbaseimpl<ushortvector, vectorref<unsigned short>> &> (alignstateids),
                                              dynamic_cast<const vectorbaseimpl<uintvector, vectorref<unsigned int>> &> (alignoffsets),
                                              dynamic_cast<const vectorbaseimpl<edgeinfowithscoresvector, vectorref<msra::lattices::edgeinfowithscores>> &> (edges),
                                              dynamic_cast<const vectorbaseimpl<nodeinfovector, vectorref<msra::lattices::nodeinfo>> &> (nodes),
                                              dynamic_cast<const vectorbaseimpl<doublevector, vectorref<double>> &> (logqs),
                                              tomatrixref (logacc));
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
