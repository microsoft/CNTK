// cudalatticeops.h -- contains all actual CUDA-side lattice ops
//
// F. Seide, Aug 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudalatticeops.h $
// 
// 45    11/21/12 8:58p V-hansu
// rename state2classmap to senone2classmap
// 
// 44    11/21/12 7:10p V-hansu
// rename statetoclassmap to state2classmap
// 
// 43    11/21/12 6:04p V-hansu
// add statetoclassmap to forwardbackward() to get prepared for mpe
// approximation
// 
// 42    11/05/12 12:26a V-hansu
// undo last check in :( not got enough time to get it done
// 
// 41    11/02/12 9:44a V-hansu
// add logframescorrect to edgealignment(), prepare for moving computation
// from forwardbackward() to edgealignment()
// 
// 40    10/29/12 3:36p V-hansu
// add boosting factor to prepare for BMMI
// 
// 39    10/19/12 7:51p V-hansu
// adjust the order of interface arguments, finish some TODO
// 
// 38    10/19/12 3:55p V-hansu
// finish the mmierrorsignal, not tested yet
// 
// 37    10/17/12 5:59p V-hansu
// rename Eframescorrectotal to logEframescorrectotal, rename
// Eframescorrectbuf or Eframescorrectdiff to logEframescorrect
// 
// 36    10/17/12 3:33p V-hansu
// turn eframecorrect (eframecorrectdiff) from float vector to double
// vector
// 
// 35    10/17/12 2:03p Fseide
// new method stateposteriors()
// 
// 34    10/15/12 5:25p V-hansu
// add aligns to forwardbackwardlattice()
// 
// 33    10/14/12 8:26p V-hansu
// add silunitid and spunitid to forwardbackwardlattice
// 
// 32    9/30/12 5:38p V-hansu
// add backptr in edgealignment
// 
// 31    9/28/12 6:08p V-hansu
// rename spalignunit to spalignunitid
// 
// 30    9/28/12 4:17p V-hansu
// rename errorsignalbuf to errorsignalneg, activate log mode error
// accumulation, refactor atomicCASfloatdouble
// 
// 29    9/28/12 2:03p V-hansu
// rename dengammas to errorsignal, rename transPindex to alignunit
// 
// 28    9/26/12 7:36p V-hansu
// add siltransPindex, rename returnEframecorrect to returnEframescorrect 
// 
// 27    9/26/12 2:31p V-hansu
// change the definition from float vector to double vector
// 
// 26    9/26/12 2:29p V-hansu
// change logpps in errorsignal from float to double. remove
// Eframecorrecttotal in sMBRerrorsignal, change the location of resize
// towards logpps and Eframecorrect
// 
// 25    9/26/12 1:52p V-hansu
// rename Eframescorrect to Eframescorrectbuf, and add Eframescorrectdiff
// for difference computation.
// 
// 24    9/26/12 1:08p V-hansu
// rename combinemode to returnEframescorrect
// 
// 23    9/26/12 12:53p Fseide
// errorsignal() renamed to sMBRerrorsignal()
// 
// 22    9/26/12 12:27p Fseide
// renamed logdengammaspos/neg to dengammas/dengammasbuf
// 
// 21    9/26/12 11:57a Fseide
// sMBRerrorsignal() now takes two dengammas accumulators, in prep for pos/neg
// logadd
// 
// 20    9/25/12 3:11p V-hansu
// add sizetvector and change uids from uintvector into sizetvector
// 
// 19    9/25/12 1:12p V-hansu
// add alignemts and alignmentoffsets to forwardlattce related function to
// finish the algorithm
// 
// 18    9/24/12 10:07p V-hansu
// change the interface relating to forwardbackwardlattice to get prepared
// for the combined mode fwbw, not finished yet
// 
// 17    9/21/12 3:55p V-hansu
// change the interface of latticefunctionsops::forwardbackwardlattice to
// pass in batchsizeforward and batchsizebackward by pointer
// 
// 16    9/21/12 3:25p V-hansu
// change batchsizeforward and batchsizebackward into cpu side vector
// 
// 15    9/21/12 1:43p V-hansu
// (fix some indentation)
// 
// 14    9/19/12 9:33a Fseide
// renamed edgeinfo to edgeinfowithscores, in prep for V2 lattice format
// 
// 13    9/16/12 9:31p V-hansu
// add atomicLogAdd, not finished. add doublevector
// 
// 12    9/14/12 2:37p V-hansu
// add forwardlatticej and forwardbackwardlattice in
// latticefunctionskernels and related classes
// 
// 11    9/14/12 1:27p V-hansu
// add fowardlatticej, not tested
// 
// 10    9/06/12 7:24p V-hansu
// add alignoffsets into interface, same as alignstateids
// 
// 9     9/06/12 7:14p V-hansu
// add alignoffsets in sMBRerrorsignal
// 
// 8     9/05/12 10:36p V-hansu
// add function sMBRerrorsignal and codes relating to it
// 
// 7     9/04/12 10:25p V-hansu
// change the interface of edgealignment
// 
// 6     9/03/12 4:53p V-hansu
// change the interface of edgealignment
// 
// 5     9/01/12 3:00p Fseide
// moved lattice argument types to a new header
// 
// 4     9/01/12 2:10p Fseide
// removed dependency on cudalattice.h, as the DLL-related stuff tripped
// up the CUDA compiler (and rightly so)
// 
// 3     8/31/12 10:58p V-hansu
// add latticefunctionsops class
// 
// 2     8/28/12 6:42p Fseide
// (added more comments)
// 
// 1     8/28/12 6:27p Fseide
// created

#pragma once

#include "cudabasetypes.h"              // for vectorref<>
#include "latticestorage.h"             // for the lattice types
#include "latticefunctionskernels.h"    // for the actual inner kernels and any argument types that are not yet defined in latticestorage.h

using namespace msra::lattices;

namespace msra { namespace cuda {

// The XXXvectorops classes must derive from vectorref<XXX>.

// Find the kernels in cudalatticeops.cu.h.
class somedatavectorops : protected vectorref<somedata>
{
protected:
    int somedataoperation (size_t arg);
};

class latticefunctionsops : protected vectorref<empty>
{
protected:
    void edgealignment (const vectorref<lrhmmdef> & hmms, const vectorref<lr3transP> & transPs, const size_t spalignunitid,
                        const size_t silalignunitid, const matrixref<float> & logLLs, 
                        const vectorref<msra::lattices::nodeinfo> & nodes, const vectorref<msra::lattices::edgeinfowithscores> & edges, 
                        const vectorref<msra::lattices::aligninfo> & aligns, const vectorref<unsigned int> & alignoffsets, 
                        vectorref<unsigned short> & backptrstorage, const vectorref<size_t> & backptroffsets,
                        vectorref<unsigned short> & alignresult, vectorref<float> & edgeacscores) const;        // output

    void forwardbackwardlattice (const size_t * batchsizeforward, const size_t * batchsizebackward, 
                                 const size_t numlaunchforward, const size_t numlaunchbackward,
                                 const size_t spalignunitid, const size_t silalignunitid,
                                 const vectorref<float> & edgeacscores, const vectorref<msra::lattices::edgeinfowithscores> & edges, 
                                 const vectorref<msra::lattices::nodeinfo> & nodes,
                                 const vectorref<msra::lattices::aligninfo> & aligns, const vectorref<unsigned short> & aligments,
                                 const vectorref<unsigned int> & aligmentoffsets,
                                 vectorref<double> & logpps, vectorref<double> & logalphas, vectorref<double> & logbetas,
                                 const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect, 
                                 const vectorref<unsigned short> & uids, const vectorref<unsigned short> & senone2classmap,
                                 vectorref<double> & logaccalphas, vectorref<double> & logaccbetas,
                                 vectorref<double> & logframescorrectedge, vectorref<double> & logEframescorrect, vectorref<double> & Eframescorrectbuf, 
                                 double & logEframescorrecttotal, double & totalfwscore) const;

    void sMBRerrorsignal (const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
                          const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes, 
                          const vectorref<double> & logpps, const float amf, const vectorref<double> & logEframescorrect, const double logEframescorrecttotal, 
                          matrixref<float> & errorsignal, matrixref<float> & errorsignalneg) const;

    void mmierrorsignal (const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
                         const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes, 
                         const vectorref<double> & logpps, matrixref<float> & errorsignal) const;

    void stateposteriors (const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
                          const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes, 
                          const vectorref<double> & logqs, matrixref<float> & logacc) const;
};
};};
