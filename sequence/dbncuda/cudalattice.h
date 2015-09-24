// cudalattice.h -- lattice forward/backward functions CUDA execution
//
// F. Seide, Aug 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudalattice.h $
// 
// 56    11/27/12 4:44p V-hansu
// move vectorbase<> like ushortvector from cudamatrix.h to cudalattice.h
// 
// 55    11/27/12 3:43p V-hansu
// move vectorbase<> like ushortvector from cudalattice.h to cudamatrix.h
// 
// 54    11/21/12 8:58p V-hansu
// rename state2classmap to senone2classmap
// 
// 53    11/21/12 7:10p V-hansu
// rename statetoclassmap to state2classmap
// 
// 52    11/21/12 6:04p V-hansu
// add statetoclassmap to forwardbackward() to get prepared for mpe
// approximation
// 
// 51    11/05/12 12:22a V-hansu
// undo last checkin :( not got enough time to do so...
// 
// 50    11/02/12 9:44a V-hansu
// add logframescorrect to edgealignment(), prepare for moving computation
// from forwardbackward() to edgealignment()
// 
// 49    10/29/12 3:36p V-hansu
// add boosting factor to prepare for BMMI
// 
// 48    10/19/12 7:51p V-hansu
// adjust the order of interface arguments, finish some TODO
// 
// 47    10/19/12 3:55p V-hansu
// finish the mmierrorsignal, not tested yet
// 
// 46    10/17/12 5:59p V-hansu
// rename Eframescorrectotal to logEframescorrectotal, rename
// Eframescorrectbuf or Eframescorrectdiff to logEframescorrect
// 
// 45    10/17/12 3:33p V-hansu
// turn eframecorrect (eframecorrectdiff) from float vector to double
// vector
// 
// 44    10/17/12 2:03p Fseide
// new method stateposteriors()
// 
// 43    10/15/12 5:25p V-hansu
// add aligns to forwardbackwardlattice()
// 
// 42    10/14/12 8:26p V-hansu
// add silunitid and spunitid to forwardbackwardlattice
// 
// 41    9/30/12 5:21p V-hansu
// add backptr to edgealignment related functions
// 
// 40    9/28/12 6:08p V-hansu
// rename spalignunit to spalignunitid
// 
// 39    9/28/12 2:56p V-hansu
// remove transPindex to alignunit
// 
// 38    9/26/12 7:23p V-hansu
// pass siltransPindex into forwardbackwardalign to prepare for silence
// processing
// 
// 37    9/26/12 5:29p V-hansu
// change uids back to unsigned short vector
// 
// 36    9/26/12 2:29p V-hansu
// change logpps in errorsignal from float to double. remove
// Eframecorrecttotal in sMBRerrorsignal, change the location of resize
// towards logpps and Eframecorrect
// 
// 35    9/26/12 1:52p V-hansu
// rename Eframescorrect to Eframescorrectbuf, and add Eframescorrectdiff
// for difference computation.
// 
// 34    9/26/12 1:08p V-hansu
// rename combinemode to returnEframescorrect
// 
// 33    9/26/12 12:53p Fseide
// errorsignal() renamed to sMBRerrorsignal()
// 
// 32    9/26/12 12:27p Fseide
// renamed logdengammaspos/neg to dengammas/dengammasbuf
// 
// 31    9/26/12 11:57a Fseide
// sMBRerrorsignal() now takes two dengammas accumulators, in prep for pos/neg
// logadd
// 
// 30    9/25/12 3:11p V-hansu
// add sizetvector and change uids from uintvector into sizetvector
// 
// 29    9/25/12 1:12p V-hansu
// add alignemts and alignmentoffsets to forwardlattce related function to
// finish the algorithm
// 
// 28    9/24/12 10:07p V-hansu
// change the interface relating to forwardbackwardlattice to get prepared
// for the combined mode fwbw, not finished yet
// 
// 27    9/21/12 3:55p V-hansu
// change the interface of latticefunctionsops::forwardbackwardlattice to
// pass in batchsizeforward and batchsizebackward by pointer
// 
// 26    9/21/12 3:25p V-hansu
// change batchsizeforward and batchsizebackward into cpu side vector
// 
// 25    9/19/12 9:33a Fseide
// renamed edgeinfo to edgeinfowithscores, in prep for V2 lattice format
// 
// 24    9/16/12 9:31p V-hansu
// add atomicLogAdd, not finished. add doublevector
// 
// 23    9/16/12 8:57p V-hansu
// add doublevector
// 
// 22    9/07/12 12:18 Fseide
// fixed a typo in NOCUDA mode
// 
// 21    9/06/12 8:02p Fseide
// changed __device__ to __kernel_emulation__ to signal
// latticefunctionskernels.h to define emulation versions of CUDA
// functions
// 
// 20    9/06/12 7:24p V-hansu
// add alignoffsets into interface, same as alignstateids
// 
// 19    9/05/12 10:36p V-hansu
// add function sMBRerrorsignal and codes relating to it
// 
// 18    9/04/12 10:27p V-hansu
// change the interface of edgealignment
// 
// 17    9/04/12 4:30p V-hansu
// change the interface again
// 
// 16    9/04/12 4:11p V-hansu
// fix an interface
// 
// 15    9/03/12 5:10p V-hansu
// change the interface of edgealignment
// 
// 14    9/03/12 4:53p V-hansu
// change the interface of edgealignment
// 
// 13    9/02/12 3:20p V-hansu
// add uintvector
// 
// 12    9/02/12 2:47p V-hansu
// add floatvector
// 
// 11    9/01/12 3:00p Fseide
// moved lattice argument types to a new header
// 
// 10    8/31/12 10:39p V-hansu
// add latticefunctions and function edgealignment
// 
// 9     8/31/12 5:11p V-hansu
// extract lrhmmdef and lr3trans to latticestorage.h
// 
// 8     8/30/12 7:14p V-hansu
// moved CUDA-related lattice-related structs in here from
// latticestorage.h, since they are not related to basic storage but
// rather to algorithms running on lattices on CUDA
// 
// 7     8/28/12 7:02p Fseide
// (fixed a comment)
// 
// 6     8/28/12 6:42p Fseide
// (added more comments)
// 
// 5     8/28/12 4:59p Fseide
// added ushortvector and some minor code changes
// 
// 4     8/28/12 3:30p Fseide
// (comments added)
// 
// 3     8/28/12 3:05p Fseide
// added a how-to for creating new classes here
// 
// 2     8/28/12 2:59p Fseide
// (fixed tabs...meh)
// 
// 1     8/28/12 2:51p Fseide
// newly created cudalattice and latticestorage, as a frame for lattice
// forward/backward executed on CUDA

#pragma once
#include <stdexcept>			// (for NOCUDA version only)
#include "cudamatrix.h"			// we inherit the EXPORT #define from here
#include "latticestorage.h"		// basic data types for storing lattices
#define __kernel_emulation__            // allow the compilation of CUDA kernels on the CPU
#include "latticefunctionskernels.h"    // for data types passed to kernel functions
#undef __kernel_emulation__

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




// ---------------------------------------------------------------------------
// vectors of these with custom functions
// The XXXvector classes must derive from vectorbase<XXX>.
// ---------------------------------------------------------------------------

struct somedatavector : public vectorbase<msra::lattices::somedata>
{
    // must implement all members of vectorbase<>, and can add operations here
    virtual int somedataoperation (size_t arg) = 0;
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
    virtual void edgealignment (const lrhmmdefvector & hmms, const lr3transPvector & transPs, const size_t spalignunitid, 
                                const size_t silalignunitid, const msra::cuda::matrix & logLLs, const nodeinfovector & nodes, 
                                const edgeinfowithscoresvector & edges, const aligninfovector & aligns,
                                const uintvector & alignoffsets, ushortvector & backptrstorage, const sizetvector & backptroffsets,
                                ushortvector & alignresult, floatvector & edgeacscores) = 0;        // output
    virtual void forwardbackwardlattice (const size_t * batchsizeforward, const size_t * batchsizebackward, 
                                         const size_t numlaunchforward, const size_t numlaunchbackward,
                                         const size_t spalignunitid, const size_t silalignunitid,
                                         const floatvector & edgeacscores, const edgeinfowithscoresvector & edges, 
                                         const nodeinfovector & nodes, const aligninfovector & aligns, 
                                         const ushortvector & alignoutput, const uintvector & alignoffsets,
                                         doublevector & logpps, doublevector & logalphas, doublevector & logbetas,
                                         const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect, 
                                         const ushortvector & uids, const ushortvector & senone2classmap, 
                                         doublevector & logaccalphas, doublevector & logaccbetas,
                                         doublevector & logframescorrectedge, doublevector & logEframescorrect, 
                                         doublevector & Eframescorrectbuf, double & logEframescorrecttotal, double & totalfwscore) = 0;
    virtual void sMBRerrorsignal (const ushortvector & alignstateids, const uintvector & alignoffsets,
                                  const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                                  const doublevector & logpps, const float amf, const doublevector & logEframescorrect, 
                                  const double logEframescorrecttotal, msra::cuda::matrix & dengammas, msra::cuda::matrix & dengammasbuf) = 0;
    virtual void mmierrorsignal (const ushortvector & alignstateids, const uintvector & alignoffsets,
                                 const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                                 const doublevector & logpps, msra::cuda::matrix & dengammas) = 0;
    virtual void stateposteriors (const ushortvector & alignstateids, const uintvector & alignoffsets,
                                  const edgeinfowithscoresvector & edges, const nodeinfovector & nodes, 
                                  const doublevector & logqs, msra::cuda::matrix & logacc) = 0;
                                           
};

// ---------------------------------------------------------------------------
// factor methods
// ---------------------------------------------------------------------------

#ifndef NOCUDA
EXPORT somedatavector * newsomedatavector();
EXPORT ushortvector * newushortvector();
EXPORT uintvector * newuintvector();
EXPORT floatvector * newfloatvector();
EXPORT doublevector * newdoublevector();
EXPORT sizetvector * newsizetvector();
EXPORT latticefunctions * newlatticefunctions();
EXPORT lrhmmdefvector * newlrhmmdefvector();
EXPORT lr3transPvector * newlr3transPvector();
EXPORT nodeinfovector * newnodeinfovector();
EXPORT edgeinfowithscoresvector * newedgeinfovector();
EXPORT aligninfovector * newaligninfovector();
#else           // dummies when building without CUDA
static inline somedatavector * newsomedatavector() { throw std::runtime_error ("should not be here"); }
static inline ushortvector * newushortvector() { throw std::runtime_error ("should not be here"); }
static inline uintvector * newuintvector() { throw std::runtime_error ("should not be here"); }
static inline floatvector * newfloatvector() { throw std::runtime_error ("should not be here"); }
static inline doublevector * newdoublevector() { throw std::runtime_error ("should not be here"); }
static inline sizetvector * newsizetvector() { throw std::runtime_error ("should not be here"); }
static inline latticefunctions * newlatticefunctions() { throw std::runtime_error ("should not be here"); }
static inline lrhmmdefvector * newlrhmmdefvector() { throw std::runtime_error ("should not be here"); }
static inline lr3transPvector * newlr3transPvector() { throw std::runtime_error ("should not be here"); }
static inline nodeinfovector * newnodeinfovector() { throw std::runtime_error ("should not be here"); }
static inline edgeinfowithscoresvector * newedgeinfovector() { throw std::runtime_error ("should not be here"); }
static inline aligninfovector * newaligninfovector() { throw std::runtime_error ("should not be here"); }
#endif

};};
