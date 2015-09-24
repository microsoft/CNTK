// parallelforwardbackward.cpp -- parallelized implementation(s) of lattice forward/backward implemented --currently through CUDA
//
// F. Seide, Sep 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/parallelforwardbackward.cpp $
// 
// 136   5/13/13 10:27 Fseide
// [from Rui Zhao] changed lr3transP to a full trans matrix, to allow for
// IPE-Speech's fully ergodic silence model--note the TODOs and one BUGBUG
// in there that must be fixed
// 
// 135   4/08/13 9:02p V-hansu
// (fix last check-in)
// 
// 134   4/08/13 9:01p V-hansu
// remove PARALLEL_DEBUG
//
// 133   1/15/13 5:09p V-hansu
// (fix last checkin)
// 
// 132   1/15/13 2:59p V-hansu
// remove SUM_AND_PRODUCT
// 
// 131   12/25/12 2:08p V-hansu
// (modify a fprintf)
// 
// 130   11/21/12 9:05p V-hansu
// modify cachehset() to use interface of hmm rather than member
// 
// 129   11/21/12 8:56p V-hansu
// modify cachehset() to separate sil and sp when constructing
// senone2classmap
// 
// 128   11/21/12 8:21p V-hansu
// (remove {})
// 
// 127   11/21/12 7:49p V-hansu
// add enum mbrclassdefinition, add that to entercomputation() and
// cachehset() to specify which kind of MBR we use
// 
// 126   11/21/12 7:10p V-hansu
// rename statetoclassmap to state2classmap
// 
// 125   11/21/12 7:06p V-hansu
// get statetoclassmap done making use of hset
// 
// 124   11/21/12 6:32p V-hansu
// add statetoclassmap in parallelforwardbackward(), not finished.
// 
// 123   11/09/12 5:41p V-hansu
// modify a little w.r.t. emulateforwardbackwardlattice()
// 
// 122   11/05/12 12:22a V-hansu
// undo last checkin :( not got enough time to do so...
// 
// 121   11/02/12 9:56a V-hansu
// add logframescorrect to forwardbackwardalign() related functions to
// prepare for moving computation from forwardbackwardlattice() to
// forwardbackwardalign()
// 
// 120   10/29/12 5:44p V-hansu
// activate BMMI with silence_penalty
// 
// 119   10/29/12 4:20p V-hansu
// allocate memory for framescorrect when boost mmi is performed
// 
// 118   10/29/12 3:48p V-hansu
// add boostingfactor, not enabled by now.
// 
// 117   10/29/12 11:04a V-hansu
// activate TWO_CHANNEL
// 
// 116   10/22/12 2:51p V-hansu
// move PARALLEL_SIL to latticearchive.h and change edgehassil to local
// variable
// 
// 115   10/21/12 7:37p V-hansu
// change the interface of emulateedgealignment
// 
// 114   10/21/12 1:03p V-hansu
// add method getedgeacscores(), getedgealignments() in parallelstateimpl
// 
// 113   10/19/12 8:41p V-hansu
// put some code to if{} to save time/space for mmi
// 
// 112   10/19/12 4:07p V-hansu
// finish parallelmmierrorsignal() and emulatemmierrorsignal()
// combine allocerrormatrix and cacheandallocatematrix into
// cacheerrorsignal()
// 
// 111   10/19/12 2:53p V-hansu
// add method cacheandallocatematrix,
// factor allocerrormatrix to two part of cacheandallocatematrix
// 
// 110   10/19/12 2:42p V-hansu
// new method parallelmmierrorsignal, not finished
// 
// 109   10/19/12 2:11p V-hansu
// modify allocatevectors() to prepare for mmi
// 
// 108   10/19/12 12:21p V-hansu
// rename FORBID to TWO_CHANNEL and remove timer. deactivate
// SUM_AND_PRODUCT
// 
// 107   10/17/12 11:59p V-hansu
// get emulation compatible with cuda version (use kernel)
// 
// 106   10/17/12 8:39p V-hansu
// fix some debugging code
// 
// 105   10/17/12 8:03p V-hansu
// modify emulation code relating to smbrerrorsignal()
// 
// 104   10/17/12 7:42p V-hansu
// add a comment and change to cuda mode
// 
// 103   10/17/12 6:07p V-hansu
// rename Eframescorrectdiff to logEframescorrect, rename
// Eframescorrecttotal to logEframescorrecttotal
// 
// 102   10/17/12 4:24p V-hansu
// (modify some comments)
// 
// 101   10/17/12 4:18p V-hansu
// add error initialization to sMBRerrorsignal now
// 
// 100   10/17/12 3:50p V-hansu
// change eframecorrect from float vector to double vector
// add function weighteddotproduct
// 
// 99    10/17/12 2:57p Fseide
// bug fix for sMBR: added the calls into stateposteriors(), but commented
// out for now
// 
// 98    10/17/12 2:48p Fseide
// emulation layer for stateposteriors() added
// 
// 97    10/17/12 12:15p V-hansu
// remove the posprocessing in emulateforwardbackwardlattice in two-
// channel mode
// 
// 96    10/17/12 2:04a V-hansu
// add some debugging code and disenable FORBID_INVALID_SIL_PATHS
// 
// 95    10/16/12 5:06p V-hansu
// activate FORBID_INVALID_SIL_PATHS
// 
// 94    10/16/12 3:41p V-hansu
// (add some debugging code)
// 
// 93    10/16/12 12:45p V-hansu
// add a log printing sentence in FORBID_INVALID_SIL_PATH mode
// 
// 92    10/16/12 12:32p V-hansu
// change SILENCE_PRUNING to FORBID_INVALID_SIL_PATHS
// 
// 91    10/15/12 7:38p V-hansu
// add postprocessing for combining silence path and non-silence path
// 
// 90    10/15/12 5:36p V-hansu
// add align to forwardbackwardlattice() related functions
// 
// 89    10/14/12 10:05p V-hansu
// add silalignunitid and spalignunitid to forwardlatticej and
// backwardlatticej
// 
// 88    10/12/12 4:24p V-hansu
// (add a space)
// 
// 87    10/12/12 1:02p V-hansu
// add some debugging code
// 
// 86    10/12/12 1:19a V-hansu
// activate PARALLEL_SIL and add some code for debugging
// 
// 85    10/09/12 7:42p V-hansu
// fix initialization of member in lr3transPcpuforgpu
// 
// 84    10/08/12 10:56a V-hansu
// deactivate parallel_sil
// 
// 83    10/05/12 5:12p V-hansu
// add cpumode to parallelstate and get cpumode back to work now.
// 
// 82    10/05/12 4:07p V-hansu
// activate parallel silence processing
// 
// 81    10/04/12 9:06p V-hansu
// add copyalignments function to sync back the alignments in cuda mode
// for diagnostic
// 
// 80    9/30/12 7:06p V-hansu
// prepare for silence processing
// 
// 79    9/30/12 5:26p V-hansu
// add backptr to edgealignments and enable logPergodicskip in lr3transP
// 
// 78    9/30/12 3:44p V-hansu
// add something relating to processing sil on cuda, not finished, include
// in #if 0
// 
// 77    9/28/12 6:11p V-hansu
// rename allocgammas to allocmatrix and factor out setvalue function
// 
// 76    9/28/12 4:36p V-hansu
// rename transPindex to alignunit
// 
// 75    9/27/12 11:29p V-hansu
// cache gammas in parallelstate and add shuffle check
// 
// 74    9/27/12 12:29a V-hansu
// move memory allocation things into parallelstate
// 
// 73    9/26/12 7:33p V-hansu
// change setvaluej, add expdiffj, pull alpha/beta vectors for fwbw to
// parallestate, pass siltransPindex into fwbwalign
// 
// 72    9/26/12 2:29p V-hansu
// change logpps in errorsignal from float to double. remove
// Eframecorrecttotal in sMBRerrorsignal, change the location of resize
// towards logpps and Eframecorrect
// 
// 71    9/26/12 1:08p V-hansu
// rename combinemode to returnEframescorrect
// 
// 70    9/26/12 12:57p Fseide
// renamed errorsignalj() to sMBRerrorsignalj()
// 
// 69    9/26/12 12:53p Fseide
// errorsignal() renamed to sMBRerrorsignal()
// 
// 68    9/26/12 12:42p Fseide
// (added a comment and renamed some local variables0
// 
// 67    9/26/12 12:38p Fseide
// trainlayer() now passes a buffer to L.forwardbackward() for
// error-signal computation
// 
// 66    9/25/12 11:31p V-hansu
// pass the alignment (with silence processed) back to gpu to do
// parallelforwardbackwardlattice
// 
// 65    9/25/12 9:36p V-hansu
// get returnEframescorrect fwbw working for emulation, still debugging through
// CUDA version
// 
// 64    9/25/12 5:33p V-hansu
// remove totalfwacc in backwardlatticej
// 
// 63    9/25/12 3:34p V-hansu
// temporarily check in to make build
// 
// 62    9/25/12 1:57p V-hansu
// modify the interface of parallelforwardbackward to get prepared for
// emulation of that
// 
// 61    9/24/12 10:11p V-hansu
// modify the interface for parallelfwbw to prepare for the combined mode.
// turn off parallelforwardbackwardlattice now
// 
// 60    9/24/12 7:10p V-hansu
// change the interface of parallelforwardbackwardlattice to get ready for
// gpu combination of latticelevel fwbw
// 
// 59    9/24/12 1:08a V-hansu
// add #define SHUFFLE_FORWARD, not turned on
// 
// 58    9/22/12 9:16p V-hansu
// change the location of initialization in parallelforwardbackwardlattice
// 
// 57    9/21/12 10:13p V-hansu
// copy edgeacscores to gpu again because sil is not computed on gpu.
// change the interface of emulateforwardbackwardlattice as well.
// 
// 56    9/21/12 4:08p V-hansu
// change the interface of forwardbackwardlattice
// 
// 55    9/21/12 3:14p V-hansu
// remove some unimportant resize() in parallelforwardbackwardlattice()
// 
// 54    9/21/12 2:42p V-hansu
// fix a bug that require elements before allocate in
// parallelforwardbackwardlattice (logalphas...)
// 
// 53    9/21/12 1:42p V-hansu
// finish CUDA call of forwardbackwardlattice and add some comments
// 
// 52    9/21/12 10:28a V-hansu
// (remove some unuseful references)
// 
// 51    9/21/12 9:03a Fseide
// fixed the message that the parallelstate constructor prints
// 
// 50    9/21/12 8:10a Fseide
// new method msra::cuda::numcudadevices() inside cudamatrix.h, which
// determines the # devices but does not crash if CUDA DLL missing
// (returning 0 instead), this was factored out from
// msra::dbn::numcudadevices() so we can share it with lattice code;
// parallelstate() constructor now uses proper numcudadevices() function
// to determine whether CUDA is available (before just assumed it is,
// which was an early hack)
// 
// 49    9/19/12 9:33a Fseide
// renamed edgeinfo to edgeinfowithscores, in prep for V2 lattice format
// 
// 48    9/17/12 8:06p V-hansu
// change float into double for logalphas and logbetas in forwardbackward
// 
// 47    9/14/12 10:02p V-hansu
// (fix a bug relating to batchsizebackward)
// 
// 46    9/14/12 9:27p V-hansu
// modify parallelforwardbackwardlattice
// 
// 45    9/14/12 5:56p V-hansu
// add parallelforwardbackwardlattice, forwardlatticej and
// backwardlatticej 
// 
// 44    9/14/12 1:57p V-hansu
// finish forwardlatticej, change forwardbackwardlattice's lambda into
// normal vector, both not tested
// 
// 43    9/13/12 5:40p V-hansu
// add parallelforwardbackwardlattice, not finished
// 
// 42    9/07/12 12:36p V-hansu
// replace timers, move all timers to latticefwbw.cpp
// 
// 41    9/07/12 12:20p V-hansu
// get the comments relating to timing right
// 
// 40    9/07/12 12:14p V-hansu
// (format some comments)
// 
// 39    9/07/12 12:10p V-hansu
// (fix a bug of calling cuda state fwbw)
// 
// 38    9/07/12 11:34a Fseide
// bug fix: initial data transfer did not wait for completion before
// freeing the CPU-side vector;
// added static_asserts for CUDA HMM types;
// 
// 37    9/07/12 11:28a V-hansu
// add a static_assert into parallelstate
// 
// 36    9/07/12 11:00a V-hansu
// (format the comment again)
// 
// 35    9/07/12 10:26a V-hansu
// (format some comments)
// 
// 34    9/07/12 9:41a V-hansu
// (format some comments)
// 
// 33    9/07/12 9:24a V-hansu
// format the comments
// 
// 32    9/07/12 8:50a V-hansu
// (add some comments about timing)
// 
// 31    9/07/12 8:45a V-hansu
// finish the cuda version of sMBRerrorsignal
// 
// 30    9/06/12 11:49p V-hansu
// finish cuda sMBRerrorsignal computation, but the result are not correct
// 
// 29    9/06/12 7:26p V-hansu
// add alignoffsets into sMBRerrorsignal function, next step shall be
// implementation
// 
// 28    9/06/12 6:57p V-hansu
// modify interface of parallelsMBRerrorsignal
// 
// 27    9/06/12 12:15p V-hansu
// (fix a bug relating to hmm.size() checking)
// 
// 26    9/06/12 8:10a Fseide
// parallelstate and GPU model transfer for lattice processing now moved
// out to trainlayer(), such that models (and memory allocations) are
// actually cached across all lattices
// 
// 25    9/05/12 10:54p V-hansu
// 
// 24    9/05/12 7:07p Fseide
// (editorial)
// 
// 23    9/05/12 7:05p Fseide
// moved all parallel state to parallelstate object
// 
// 22    9/05/12 6:29p Fseide
// new class parallelstate to encapsulate and carry over state from
// parallel lattice processing
// 
// 21    9/05/12 5:31p V-hansu
// (fix some indentation)
// 
// 20    9/05/12 10:28a Fseide
// minor edits of time measurements
// 
// 19    9/05/12 10:08a V-hansu
// add TEST_TIME
// 
// 18    9/05/12 8:47a V-hansu
// fetch scores and enable cuda mode
// 
// 17    9/05/12 8:10a Fseide
// fixed compilation error due to inconsistent check-in (numframes
// parameter)
// 
// 16    9/04/12 6:30p V-hansu
// extract PRINT_ALIGNMENTS into latticefwbw.cpp
// 
// 15    9/04/12 6:18p V-hansu
// fix dim3 t and b in emulator
// 
// 14    9/04/12 3:06p V-hansu
// modify some copying codes
// 
// 13    9/03/12 10:53p V-hansu
// get through a hmm copy bug
// 
// 12    9/03/12 4:58p V-hansu
// change blockDim to gridDim and enable parallelization
// 
// 11    9/02/12 8:24p V-hansu
// add sptransPindex
// 
// 10    9/02/12 3:33p V-hansu
// change interface of edgealignmentj and add cudalogLLs
// 
// 9     9/02/12 12:38p V-hansu
// change initialization of lr3transPcpuforgpu and invoke parallelization
// mode
// 
// 8     9/02/12 11:49a V-hansu
// change the initialization of lr3transPcpuforgpu
// 
// 7     9/01/12 8:42p V-hansu
// comment out lr3transPcpuforgpu[i].loga[j][k] temp temporarily
// 
// 6     9/01/12 8:04p Fseide
// minor update due to new lr3transP structure, not compiling yet--fix
// this soon!
// 
// 5     9/01/12 7:54p Fseide
// (tidied up header dependencies)
// 
// 4     9/01/12 7:45p Fseide
// now, all infrastructure is, at least basically, in place--ready to
// actually implement edge alignment
// 
// 3     9/01/12 7:22p Fseide
// completed CUDA wrappers and emulator (but not tested, and inner CUDA
// kernel function still missing)
// 
// 2     9/01/12 5:45p V-hansu
// add edgealignmentj
// 
// 1     9/01/12 3:23p Fseide
// created parallelforwardbackward
#if 0
#endif

#include "latticearchive.h"     // we implement parts of class lattice
#include "simple_checked_arrays.h"
#include "simplesenonehmm.h"    // the model
#include "ssematrix.h"          // the matrices
#include "cudalattice.h"
#include "latticefunctionskernels.h"    // for emulation
#include <numeric>              // for debug
#include "cudalib.h"

#define TWO_CHANNEL         // [v-hansu]
using namespace msra::cuda;

namespace msra { namespace lattices {

    // emulation support
    struct dim3
    {
        size_t x, y, z;
        dim3(size_t x = 1, size_t y = 1, size_t z = 1) : x(x), y(y), z(z){}
    };

    static dim3 blockIdx, gridDim, threadIdx, blockDim;

    template<typename FUNC>
    __forceinline void emulatecuda (const dim3 & b, const dim3 & t, FUNC f)
    {
        gridDim = b;
        blockDim = t;
        for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++)
            for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++)
                for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++)
                    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++)
                        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++)
                            for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++)
                                f();
    }

    void setvaluej (std::vector<double> & thisvector, double value, size_t nelem)
    {
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        const size_t j = jinblock + blockIdx.x * tpb;
        if (j < nelem)       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::setvaluej (j, thisvector, value);
        }
    }

    void expdiffj (std::vector<double> & Eframescorrectbuf, double & logEframescorrecttotal, size_t nelem, std::vector<double> & logEframescorrect)
    {
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        const size_t j = jinblock + blockIdx.x * tpb;
        if (j < nelem)       // note: will cause issues if we ever use __synctreads()
        {
            logEframescorrect[j] = (float)(Eframescorrectbuf[j] - logEframescorrecttotal);
        }
    }
    // this must be identical to an actual CUDA kernel (except for the input data types: vectorref -> std::vector)
    void edgealignmentj (const std::vector<lrhmmdef> & hmms, const std::vector<lr3transP> & transPs, const size_t spalignunitid, const size_t silalignunitid,
                         const std::vector<msra::lattices::nodeinfo> & nodes, const std::vector<msra::lattices::edgeinfowithscores> & edges, 
                         const std::vector<msra::lattices::aligninfo> & aligns,
                         const msra::math::ssematrixbase & logLLs, const std::vector<unsigned int> & alignoffsets, 
                         std::vector<unsigned short> & backptrstorage, const std::vector<size_t> & backptroffsets,
                         std::vector<unsigned short> & alignresult, std::vector<float> & edgeacscores)
    {
        // this function identifies which element in the grid we are, and then passes to the actual function that does stuff
        // compute j; but check if it is in range; return if not
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        const size_t j = jinblock + blockIdx.x * tpb;
        if (j < edges.size())       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::edgealignmentj (j, hmms, transPs, spalignunitid, silalignunitid, logLLs, nodes, edges, aligns,
                                                                     alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
        }
    }

    void forwardlatticej (const size_t batchsize, const size_t startindex, const std::vector<float> & edgeacscores, 
                          const size_t spalignunitid, const size_t silalignunitid,
                          const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes, 
                          const std::vector<msra::lattices::aligninfo> & aligns, 
                          const std::vector<unsigned short> & alignments, const std::vector<unsigned int> & alignmentoffsets,
                          std::vector<double> & logalphas, float lmf, float wp, float amf, const float boostingfactor, 
                          const std::vector<unsigned short> & uids, const std::vector<unsigned short> & senone2classmap, const bool returnEframescorrect,
                          std::vector<double> & logframescorrectedge, std::vector<double> & logaccalphas)
    {
        const size_t shufflemode = 1;
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < batchsize)       // note: will cause issues if we ever use __synctreads() in forwardlatticej
        {
            msra::lattices::latticefunctionskernels::forwardlatticej (j + startindex, edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns, alignments, alignmentoffsets, 
                                                                      logalphas, lmf, wp, amf, boostingfactor, uids, senone2classmap, returnEframescorrect, logframescorrectedge, logaccalphas);
        }
    }

    void backwardlatticej (const size_t batchsize, const size_t startindex, const std::vector<float> & edgeacscores, 
                           const size_t spalignunitid, const size_t silalignunitid,
                           const std::vector<msra::lattices::edgeinfowithscores> & edges, 
                           const std::vector<msra::lattices::nodeinfo> & nodes, 
                           const std::vector<msra::lattices::aligninfo> & aligns, const double totalfwscore, 
                           std::vector<double> & logpps, std::vector<double> & logalphas, std::vector<double> & logbetas, 
                           float lmf, float wp, float amf, const float boostingfactor, const bool returnEframescorrect, std::vector<double> & logframescorrectedge, 
                           std::vector<double> & logaccalphas, std::vector<double> & Eframescorrectbuf, std::vector<double> & logaccbetas)
    {
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        const size_t j = jinblock + blockIdx.x * tpb;
        if (j < batchsize)       // note: will cause issues if we ever use __synctreads() in backwardlatticej
        {
            msra::lattices::latticefunctionskernels::backwardlatticej (j + startindex, edgeacscores, spalignunitid, silalignunitid,
                                                                       edges, nodes, aligns, totalfwscore, logpps, logalphas, 
                                                                       logbetas, lmf, wp, amf, boostingfactor, returnEframescorrect, logframescorrectedge, 
                                                                       logaccalphas, Eframescorrectbuf, logaccbetas);
        }
    }

    void sMBRerrorsignalj (const std::vector<unsigned short> & alignstateids, const std::vector<unsigned int> & alignoffsets,
                           const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes, 
                           const std::vector<double> & logpps, const float amf, const std::vector<double> & logEframescorrect,
                           const double logEframescorrecttotal, msra::math::ssematrixbase & errorsignal, msra::math::ssematrixbase & errorsignalneg)
    {
        const size_t shufflemode = 3;
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < edges.size())       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::sMBRerrorsignalj (j, alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal,
                                                                       errorsignal, errorsignalneg);
        }
    }

    void stateposteriorsj (const std::vector<unsigned short> & alignstateids, const std::vector<unsigned int> & alignoffsets,
                           const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes, 
                           const std::vector<double> & logqs, msra::math::ssematrixbase & logacc)
    {
        const size_t shufflemode = 3;
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < edges.size())       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::stateposteriorsj (j, alignstateids, alignoffsets, edges, nodes, logqs, logacc);
        }
    }

    void checkshuffle (std::vector<int> & checkvector1, std::vector<int> & checkvector2, size_t shufflemode)
    {
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < checkvector1.size())
        {
            checkvector1[j] = -1;
        }
        if (j < checkvector2.size())
        {
            checkvector2[j] -= 1;
        }
    }

    void errorcomputationi (msra::math::ssematrixbase & errorsignal, msra::math::ssematrixbase & errorsignalneg, float amf)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i >= errorsignal.rows())
            return;
        // clear all columns
        const size_t m = errorsignal.cols();
        for (size_t j = 0; j < m; j++)
            errorsignal(i,j) = (expf(errorsignal(i,j)) - expf(errorsignalneg(i,j)))/amf;
    }

    void weightedcomponentexpproducts (const msra::math::ssematrixbase & loggammas, const msra::math::ssematrixbase & logEframescorrect, 
                                       const double logEframecorrecttotal, const float kappa, msra::math::ssematrixbase & errorsignal)
    {
        const size_t s = threadIdx.x + (blockIdx.x * blockDim.x);
        if (s < errorsignal.rows())
        {
            msra::lattices::latticefunctionskernels::computesMBRerrorsignals (s, loggammas, logEframescorrect, logEframecorrecttotal, kappa, errorsignal);
        }
    }

    // this function behaves as its CUDA counterpart, except that it takes CPU-side std::vectors for everything
    // this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
    static void emulateedgealignment (const std::vector<lrhmmdef> & hmms, const std::vector<lr3transP> & transPs, const size_t spalignunitid, const size_t silalignunitid,
                                      const std::vector<msra::lattices::nodeinfo> & nodes, const std::vector<msra::lattices::edgeinfowithscores> & edges, 
                                      const std::vector<msra::lattices::aligninfo> & aligns,
                                      const msra::math::ssematrixbase & logLLs, const std::vector<unsigned int> & alignoffsets, 
                                      std::vector<unsigned short> & backptrstorage, const std::vector<size_t> & backptroffsets,
                                      std::vector<unsigned short> & alignresult, std::vector<float> & edgeacscores)
    {
        // TODO: This function is about determining the parallelization layout
        const size_t numedges = edges.size();
        dim3 t (32, 8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));
        emulatecuda (b, t, [&]() { edgealignmentj (hmms, transPs, spalignunitid, silalignunitid, nodes, edges, aligns, logLLs, alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores); });
    }

    static double emulateforwardbackwardlattice (const size_t * batchsizeforward, const size_t * batchsizebackward, 
                                                 const size_t numlaunchforward, const size_t numlaunchbackward,
                                                 const size_t spalignunitid, const size_t silalignunitid,
                                                 const std::vector<float> & edgeacscores,
                                                 const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes,
                                                 const std::vector<msra::lattices::aligninfo> & aligns,
                                                 const std::vector<unsigned short> & alignments, const std::vector<unsigned int> & alignoffsets,
                                                 std::vector<double> & logpps, std::vector<double> & logalphas, std::vector<double> & logbetas,
                                                 const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect, 
                                                 const std::vector<unsigned short> & uids, const std::vector<unsigned short> & senone2classmap, 
                                                 std::vector<double> & logaccalphas, std::vector<double> & logaccbetas, std::vector<double> & logframescorrectedge, 
                                                 std::vector<double> & logEframescorrect, std::vector<double> & Eframescorrectbuf, double & logEframescorrecttotal)
    {
        Eframescorrectbuf;      // TODO: remove this [v-hansu]
        dim3 t (32, 8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((logalphas.size() + tpb - 1) / tpb));
        emulatecuda (b, t, [&]() { setvaluej (logalphas, LOGZERO, logalphas.size());});
        emulatecuda (b, t, [&]() { setvaluej (logbetas, LOGZERO, logbetas.size());});
        emulatecuda (b, t, [&]() { setvaluej (logaccalphas, LOGZERO, logaccalphas.size());});
        emulatecuda (b, t, [&]() { setvaluej (logaccbetas, LOGZERO, logaccbetas.size());});

        logalphas.front() = 0;
        logbetas[nodes.size() -1] = 0;

        // forward pass

        size_t startindex = 0;
        for (size_t i = 0; i < numlaunchforward; i++)
        {
            dim3 b ((unsigned int) ((batchsizeforward[i] + tpb - 1) / tpb));
            emulatecuda (b, t, [&]() { forwardlatticej (batchsizeforward[i], startindex, edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns, alignments, alignoffsets,
                                                        logalphas, lmf, wp, amf, boostingfactor, uids, senone2classmap, returnEframescorrect, logframescorrectedge, logaccalphas);} ); 
            startindex += batchsizeforward[i];
        }
        double totalfwscore = logalphas[nodes.size() - 1];
        double totalfwacc;
        if (returnEframescorrect)
        {
            totalfwacc = logaccalphas[nodes.size() - 1];
            totalfwacc -= totalfwscore;
        }

        // backward pass
        startindex = edges.size();
        for (size_t i = 0; i < numlaunchbackward; i++)
        {
            dim3 b ((unsigned int) ((batchsizebackward[i] + tpb - 1) / tpb));
            emulatecuda (b, t, [&]() { backwardlatticej (batchsizebackward[i], startindex - batchsizebackward[i], edgeacscores, 
                                                         spalignunitid, silalignunitid, edges, nodes, aligns, 
                                                         totalfwscore, logpps, logalphas, logbetas, lmf, wp, amf, boostingfactor, 
                                                         returnEframescorrect, logframescorrectedge, logaccalphas, logEframescorrect, logaccbetas); }); 
            startindex -= batchsizebackward[i];
        }
        double totalbwscore = logbetas.front();
        if (returnEframescorrect)
            logEframescorrecttotal = logaccbetas.front() - totalbwscore;

#if 1   //check for matching
        double difffwbw = totalfwscore - totalbwscore;
        double absdifffwbw = difffwbw > 0 ? difffwbw : 0 - difffwbw;

        if (absdifffwbw / nodes.size() > 1e-4)
            fprintf (stderr, "forwardbackward: WARNING: lattice fw and bw scores %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwscore, (float) totalbwscore, (int) nodes.size(), (int) edges.size());
#endif
        return totalfwscore;
    }
    // this function behaves as its CUDA conterparts, except that it takes CPU-side std::vectors for everything
    // this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
    static void emulatesMBRerrorsignal (const std::vector<unsigned short> & alignstateids, const std::vector<unsigned int> & alignoffsets, 
                                        const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes, 
                                        const std::vector<double> & logpps, const float amf,
                                        const std::vector<double> & logEframescorrect, const double logEframescorrecttotal,
                                        msra::math::ssematrixbase & errorsignal, msra::math::ssematrixbase & errorsignalneg) 
    {

        const size_t numedges = edges.size();
        dim3 t (32, 8);
        const size_t tpb = t.x * t.y;
        foreach_coord (i, j, errorsignal)
            errorsignal(i,j) = errorsignalneg(i,j) = LOGZERO;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));
        emulatecuda (b, t, [&]() { sMBRerrorsignalj (alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg); });
        dim3 b1 ((((unsigned int) errorsignal.rows())+31)/32);
        emulatecuda (b1, 32, [&]() { errorcomputationi (errorsignal, errorsignalneg, amf); });
    }

    // this function behaves as its CUDA conterparts, except that it takes CPU-side std::vectors for everything
    // this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
    static void emulatemmierrorsignal (const std::vector<unsigned short> & alignstateids, const std::vector<unsigned int> & alignoffsets, 
                                       const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes, 
                                       const std::vector<double> & logpps, msra::math::ssematrixbase & errorsignal) 
    {
        const size_t numedges = edges.size();
        dim3 t (32, 8);
        const size_t tpb = t.x * t.y;
        foreach_coord (i, j, errorsignal)
            errorsignal(i,j) = LOGZERO;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));
        emulatecuda (b, t, [&]() { stateposteriorsj (alignstateids, alignoffsets, edges, nodes, logpps, errorsignal);});
        foreach_coord (i, j, errorsignal)
            errorsignal(i,j) = expf(errorsignal(i,j));
    }

    // this function behaves as its CUDA conterparts, except that it takes CPU-side std::vectors for everything
    // this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
    /*static*/ void emulatestateposteriors (const std::vector<unsigned short> & alignstateids, const std::vector<unsigned int> & alignoffsets, 
                                            const std::vector<msra::lattices::edgeinfowithscores> & edges, const std::vector<msra::lattices::nodeinfo> & nodes, 
                                            const std::vector<double> & logqs, msra::math::ssematrixbase & logacc) 
    {
        foreach_coord (i, j, logacc)
            logacc(i,j) = LOGZERO;
        const size_t numedges = edges.size();
        dim3 t (32, 8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));
        emulatecuda (b, t, [&]() { stateposteriorsj (alignstateids, alignoffsets, edges, nodes, logqs, logacc); });
    }

    // -----------------------------------------------------------------------
    // parallelstate (-impl) --holds variables for CUDA access
    // -----------------------------------------------------------------------

    struct parallelstateimpl
    {
        bool emulation;
        parallelstateimpl() : emulation (false),    // change this to true to switch to emulation
            // models
            lr3transPgpu (msra::cuda::newlr3transPvector()), hmmsgpu (msra::cuda::newlrhmmdefvector()), spalignunitid (SIZE_MAX), silalignunitid (SIZE_MAX),
            // current lattice, logLLs, and return values
            edgesgpu (msra::cuda::newedgeinfovector()), nodesgpu (msra::cuda::newnodeinfovector()), aligngpu (msra::cuda::newaligninfovector()),
            alignresult (msra::cuda::newushortvector()), alignoffsetsgpu (msra::cuda::newuintvector()),
            edgeacscoresgpu (msra::cuda::newfloatvector()), cudalogLLs (msra::cuda::newmatrix()), logppsgpu (msra::cuda::newdoublevector()),
            logalphasgpu (msra::cuda::newdoublevector()), logbetasgpu (msra::cuda::newdoublevector()), logaccalphasgpu (msra::cuda::newdoublevector()),
            logaccbetasgpu (msra::cuda::newdoublevector()), logframescorrectedgegpu (msra::cuda::newdoublevector()), Eframescorrectbufgpu (msra::cuda::newdoublevector()), 
            logEframescorrectgpu (msra::cuda::newdoublevector()), uidsgpu (msra::cuda::newushortvector()), senone2classmapgpu (msra::cuda::newushortvector()),
            errorsignalgpu (msra::cuda::newmatrix()), errorsignalneggpu (msra::cuda::newmatrix()), errorsignalgpustorage (msra::cuda::newmatrix()),
            errorsignalneggpustorage (msra::cuda::newmatrix()), backptrstoragegpu (msra::cuda::newushortvector()), backptroffsetsgpu (msra::cuda::newsizetvector())
        {
        }

        // models

        std::unique_ptr<lr3transPvector> lr3transPgpu;
        std::unique_ptr<lrhmmdefvector> hmmsgpu;

        /*lr3transPvector* lr3transPgpu;
        lrhmmdefvector* hmmsgpu;*/
        size_t spalignunitid;
        size_t silalignunitid;
        // models for use with emulation
        std::vector<msra::lattices::lrhmmdef> hmmscpuforgpu;
        std::vector<lr3transP> lr3transPcpuforgpu;
        std::vector<unsigned short> senone2classmapcpuforgpu;

        // this caches the models over
        // static_assert (sizeof (lr3transP) == 32, "unexpected size of lr3transP");
        // TOOD: give this function a proper name
        static_assert (sizeof (lrhmmdef) == 8, "unexpected size of lrhmmdef");
        void cachehset (const msra::asr::simplesenonehmm & hset, mbrclassdefinition mbrclassdef)
        {
            // only copy once
            // TODO: this can only be cached once --but there is no check whether a different model is passed
            if (lr3transPgpu->size() > 0)
                throw std::logic_error ("cachehset: cannot bind to multiple model sets");

            auto_timer copyhmms;
            // transPs
            lr3transPcpuforgpu.resize (hset.transPs.size());
            const std::vector<msra::asr::simplesenonehmm::transP> & transPs = hset.transPs;
            spalignunitid = hset.gethmmid ("sp");
            silalignunitid = hset.gethmmid ("sil");

            foreach_index (i, transPs)
            {
                const size_t numstates = transPs[i].getnumstates();

                lr3transPcpuforgpu[i].numstates = numstates;
                for(int m = 0; m < numstates+1; m++)
                {

                    for(int n = 0; n < numstates+1; n++)
                    {
                        lr3transPcpuforgpu[i].loga[m][n] = transPs[i](m-1, n);
                    }
                }
            }
            
            lr3transPgpu->assign (lr3transPcpuforgpu, false);

            if (mbrclassdef == monophone)
            {
                const auto & sphmm = hset.gethmm (spalignunitid);
                const auto & silhmm = hset.gethmm (silalignunitid);
                const size_t numsenones = hset.getnumsenone();
                senone2classmapcpuforgpu.resize(numsenones);
                for (size_t i = 0; i < numsenones; i++)
                {
                    if (hset.statebelongstohmm (i, sphmm) || hset.statebelongstohmm (i, silhmm))
                        senone2classmapcpuforgpu[i] = silhmm.transPindex;
                    else
                        senone2classmapcpuforgpu[i] = (unsigned short) hset.senonetransP(i);
                }
                senone2classmapgpu->assign (senone2classmapcpuforgpu, false);
            }

            // else // mbrclassdefinition:: senones has no mapping 

            // hmm defs
            hmmscpuforgpu.resize (hset.hmms.size());
            const std::vector<msra::asr::simplesenonehmm::hmm> & hmms = hset.hmms;
            foreach_index (i, hmmscpuforgpu)
            {
                hmmscpuforgpu[i].numstates = (unsigned char) hmms[i].getnumstates();
                if (hmmscpuforgpu[i].numstates != hmms[i].getnumstates())
                    throw std::logic_error("parallelforwardbackwardalign : hmms.numstates is out of range of unsigned char");

                for (size_t m = 0; m < hmmscpuforgpu[i].numstates; m++)
                {
                    hmmscpuforgpu[i].senoneids[m] = (unsigned short) hmms[i].getsenoneid(m);
                    if (hmmscpuforgpu[i].senoneids[m] != hmms[i].getsenoneid(m))
                        throw std::logic_error("parallelforwardbackwardalign : hmms.numstates is out of range of unsigned short");
                }

                hmmscpuforgpu[i].transPindex = (unsigned short) hmms[i].gettransPindex();
                if (hmmscpuforgpu[i].transPindex != hmms[i].gettransPindex())
                    throw std::logic_error("parallelforwardbackwardalign : hmms.transPindex is out of range of unsigned short");
            }
            hmmsgpu->assign (hmmscpuforgpu, true/*sync*/);  // need to sync if we free the memory right after (and we won't buy much from async)
            copyhmms.show("copyhmms");      // 246.776281 ms  --note: forgot hmmsgpu

            // if we are not emulating then we will delete our CPU-side copy to save memory
            if (!emulation)
            {
                lr3transPcpuforgpu.clear();
                senone2classmapcpuforgpu.clear();
                hmmscpuforgpu.clear();
            }
#ifdef FORBID_INVALID_SIL_PATH
            fprintf (stderr, "forbid invalid sil path\n");      // [v-hansu] might be inappropriate to print here, but it is convenient.
#endif
        }
        // check that we got a model and the right one
        void validatehset (const msra::asr::simplesenonehmm & hset)
        {
            if (hmmsgpu->size() != hset.hmms.size() || lr3transPgpu->size() != hset.transPs.size())
                throw std::logic_error ("validatehset: not bound to hset or to wrong hset");
        }

        // current lattice
        std::unique_ptr<edgeinfowithscoresvector> edgesgpu;
        std::unique_ptr<nodeinfovector> nodesgpu;
        std::unique_ptr<aligninfovector> aligngpu;
        std::unique_ptr<ushortvector> alignresult;  // concatenated alignments; edges[j]'s alignment starts at offset alignoffsets[j]
        std::unique_ptr<msra::cuda::uintvector> alignoffsetsgpu;
        std::unique_ptr<floatvector> edgeacscoresgpu;
        std::unique_ptr<msra::cuda::matrix> cudalogLLs;
        
        std::unique_ptr<doublevector> logppsgpu;
        std::unique_ptr<doublevector> logalphasgpu;
        std::unique_ptr<doublevector> logbetasgpu;
        std::unique_ptr<doublevector> logaccalphasgpu;
        std::unique_ptr<doublevector> logaccbetasgpu;

        std::unique_ptr<doublevector> logframescorrectedgegpu;
        std::unique_ptr<doublevector> Eframescorrectbufgpu;
        std::unique_ptr<doublevector> logEframescorrectgpu;

        std::unique_ptr<ushortvector> backptrstoragegpu;
        std::unique_ptr<sizetvector> backptroffsetsgpu;

        std::unique_ptr<ushortvector> uidsgpu;
        std::unique_ptr<ushortvector> senone2classmapgpu;

        std::unique_ptr<msra::cuda::matrix> errorsignalgpustorage;
        std::unique_ptr<msra::cuda::matrix> errorsignalneggpustorage;
        std::unique_ptr<msra::cuda::matrix> errorsignalgpu;
        std::unique_ptr<msra::cuda::matrix> errorsignalneggpu;

        // cache current lattice
        // This is a weird mix of const/non-const and private lattice data... :(
        template<class edgestype, class nodestype, class aligntype, class edgealignments, class backpointers>
        void setutterancedata (const edgestype & edges, const nodestype & nodes, const aligntype & align,
                               const msra::math::ssematrixbase & logLLs, std::vector<float> & edgeacscores, 
                               edgealignments & edgealignments, backpointers & backpointers)
        {
            // lattice data
            edgesgpu->assign (edges, false);
            nodesgpu->assign (nodes, false);
            aligngpu->assign (align, false);
            alignoffsetsgpu->assign (edgealignments.getalignoffsets(), false);
            backptrstoragegpu->allocate (backpointers.getbackptrstoragesize());
            backptroffsetsgpu->assign (backpointers.getbackptroffsets(), false);

#ifndef PARALLEL_SIL
            alignresult->assign (edgealignments.getalignmentsbuffer(), false);
            edgeacscoresgpu->assign (edgeacscores, false);
#else
            alignresult->allocate (edgealignments.getalignbuffersize());
            edgeacscoresgpu->allocate (edges.size());
            edgeacscores;               // reference to make compilor happy
#endif

            // LLs
            //zhaorui
            /*cudalogLLs->allocate (logLLs.rows(), logLLs.cols());
            cudalogLLs->assign(0, logLLs.rows(), 0, logLLs.cols(), &logLLs(0,0), logLLs.getcolstride(), true);  // doing this last with 'true' so we can measure time better; maybe remove later*/

        }
        //template<class ElemType>
        void allocateloglls(size_t numrowls, size_t numcols)
        {
            // LLs
            cudalogLLs->allocate(numrowls, numcols);          
        }
        void setloglls(const float * loglls, size_t numrows, size_t numcols)
        { 
            if (numcols > cudalogLLs->cols())
                cudalogLLs->allocate(numrows, numcols);
            cudalogLLs->assignfromdevice( loglls, sizeof(float)*numcols*numrows);  // doing this last with 'true' so we can measure time better; maybe remove later
        }
        void getgamma(float * loglls, size_t numrows, size_t numcols)
        {
            //cudaMemcpy(loglls, errorsignalgpu->, sizeof(float)*numrows*numcols, cudaMemcpyDeviceToDevice);
            //errorsignalgpu->fetch(0, numrows, 0, numcols, loglls, numrows, true);
            errorsignalgpu->fetchtodevice(loglls, sizeof(float)*numrows*numcols);
        }
        template<class edgealignments>
        void copyalignments (edgealignments & edgealignments)
        {
            alignresult->fetch(edgealignments.getalignmentsbuffer(), true);
        }

        // [v-hansu] allocate memory for vectors relating to forward-backward
        // allocateaccvectors implies return Eframecorrect
        template<class edgestype, class nodestype>
        void allocfwbwvectors (const edgestype & edges, const nodestype & nodes, const std::vector<unsigned short> & uids, 
                               const bool allocateframescorrect, const bool copyuids, const bool allocateaccvectors)
        {
            logppsgpu->allocate (edges.size());
#ifndef TWO_CHANNEL
            const size_t alphabetanoderatio = 1;
#else
            const size_t alphabetanoderatio = 2;
#endif
            logalphasgpu->allocate (alphabetanoderatio * nodes.size());
            logbetasgpu->allocate (alphabetanoderatio * nodes.size());

            if (allocateframescorrect)
                logframescorrectedgegpu->allocate (edges.size());

            if (copyuids)
                uidsgpu->assign (uids, true);

            if (allocateaccvectors)
            {
                logaccalphasgpu->allocate (alphabetanoderatio * nodes.size());
                logaccbetasgpu->allocate (alphabetanoderatio * nodes.size());

                Eframescorrectbufgpu->allocate (edges.size());
                logEframescorrectgpu->allocate (edges.size());

            }
        }

        // check if gpumatrixstorage supports size of cpumatrix, if not allocate. set gpumatrix to part of gpumatrixstorage
        void cacheerrorsignal (const msra::math::ssematrixbase & errorsignal, const bool cacheerrsignalneg)
        {
            if (errorsignalgpustorage->rows() != 0 && errorsignalgpustorage->rows() != errorsignal.rows())
                throw::logic_error ("gpumatrixstorage->rows() shall be fixed once allocated");
            if (errorsignalgpustorage->cols() < errorsignal.cols())
                errorsignalgpustorage->allocate (errorsignal.rows(), errorsignal.cols());
            errorsignalgpu.reset(errorsignalgpustorage->patch (0, errorsignal.rows(), 0, errorsignal.cols()));

            if (cacheerrsignalneg)
            {
                if (errorsignalneggpustorage->rows() != 0 && errorsignalneggpustorage->rows() != errorsignal.rows())
                    throw::logic_error ("gpumatrixstorage->rows() shall be fixed once allocated");
                if (errorsignalneggpustorage->cols() < errorsignal.cols())
                    errorsignalneggpustorage->allocate (errorsignal.rows(), errorsignal.cols());
                errorsignalneggpu.reset(errorsignalneggpustorage->patch (0, errorsignal.rows(), 0, errorsignal.cols()));
            }
        }

        void getedgeacscores (std::vector<float> & edgeacscores)
        {
            edgeacscores.resize (edgeacscoresgpu->size());
            edgeacscoresgpu->fetch (edgeacscores, true);
        }

        void getedgealignments (std::vector<unsigned short> & edgealignments)
        {
            edgealignments.resize (alignresult->size());
            alignresult->fetch (edgealignments, true);
        }
    };

    // helper to get number of CUDA devices in a cached fashion
    static size_t numcudadevices()
    {
        static size_t cudadevices = SIZE_MAX;    // SIZE_MAX = unknown yet
        if (cudadevices == SIZE_MAX)
            cudadevices = msra::cuda::numcudadevices();
        return cudadevices;
    }

    void lattice::parallelstate::setmode(bool pcpumode)
    {
        
        if (!pcpumode)
        {
            /*if (pimpl != NULL)
            {
                delete pimpl;
                pimpl = NULL;
            }
            fprintf (stderr, "numcudadevices: using CUDA for lattice processing\n");*/
            //initwithdeviceid(DeviceId);
            pimpl = new parallelstateimpl();
        }
        else
        {
            delete pimpl;
            pimpl = NULL;
        }
        // else we leave it at NULL
    }

    void lattice::parallelstate::release(bool pcpumode)
    {
        if (!pcpumode)
        {
           /* if (pimpl != NULL)
            {
                delete pimpl;
                pimpl = NULL;
            }*/
        }
    }
	lattice::parallelstate::parallelstate() { pimpl = nullptr; }
    lattice::parallelstate::~parallelstate() { delete pimpl; }
    void lattice::parallelstate::entercomputation (const msra::asr::simplesenonehmm & hset, const mbrclassdefinition mbrclassdef) { pimpl->cachehset (hset, mbrclassdef);  }    // pass models in (to GPU) //TODO: rethink the naming of this function
    void lattice::parallelstate::copyalignments (edgealignments & edgealignments) { pimpl->copyalignments (edgealignments); }
    const size_t lattice::parallelstate::getsilunitid () { return pimpl->silalignunitid; }
    void lattice::parallelstate::getedgeacscores (std::vector<float> & edgeacscores) { pimpl->getedgeacscores (edgeacscores); }
    void lattice::parallelstate::getedgealignments (std::vector<unsigned short> & edgealignments) { pimpl->getedgealignments (edgealignments); }
    //template<class ElemType> 
    void lattice::parallelstate::setloglls(const float * loglls, size_t numrowls, size_t numcols) { pimpl->setloglls(loglls, numrowls, numcols); }
    void lattice::parallelstate::allocateloglls(size_t numrowls, size_t numcols) { pimpl->allocateloglls(numrowls, numcols); }
    void lattice::parallelstate::getgamma(float * loglls, size_t numrows, size_t numcols) { pimpl->getgamma(loglls, numrows, numcols); }
    // -----------------------------------------------------------------------
    // parallel implementations of key processing steps
    // -----------------------------------------------------------------------

    // parallelforwardbackwardalign() -- compute the statelevel gammas or viterbi alignments
    void lattice::parallelforwardbackwardalign (parallelstate & parallelstate,
                                                const msra::asr::simplesenonehmm & hset, const msra::math::ssematrixbase & logLLs,
                                                std::vector<float> & edgeacscores, edgealignments & edgealignments, backpointers & backpointers) const
    {
        parallelstate->validatehset (hset);     // ensure the models have been correctly cached on the GPU already
        
        if (!parallelstate->emulation)
        {            
            // move lattice to GPU
            parallelstate->setutterancedata (edges, nodes, align, logLLs,   // inputs
                                             edgeacscores, edgealignments, backpointers); // inouts

            // launch the kernel
            std::unique_ptr<latticefunctions> latticefunctions (msra::cuda::newlatticefunctions());
            latticefunctions->edgealignment(*parallelstate->hmmsgpu.get(), *parallelstate->lr3transPgpu.get(), 
                                            parallelstate->spalignunitid, parallelstate->silalignunitid,
                                            *parallelstate->cudalogLLs.get(), *parallelstate->nodesgpu.get(), 
                                            *parallelstate->edgesgpu.get(), *parallelstate->aligngpu.get(),
                                            *parallelstate->alignoffsetsgpu.get(), 
                                            *parallelstate->backptrstoragegpu.get(), *parallelstate->backptroffsetsgpu.get(),
                                            *parallelstate->alignresult.get(), *parallelstate->edgeacscoresgpu.get());
        }
        else
        {
            if (parallelstate->hmmscpuforgpu.size() == 0)
                throw::logic_error("we no longer support emulation for edgealign, please copy hmmscpuforgpu and lr3transPcpuforgpu if you want");
            edgeacscores.resize (edges.size());
            emulateedgealignment (parallelstate->hmmscpuforgpu, parallelstate->lr3transPcpuforgpu, parallelstate->spalignunitid,
                                  parallelstate->silalignunitid,
                                  nodes, edges, align, logLLs, edgealignments.getalignoffsets(), 
                                  backpointers.getbackptrbuffer(), backpointers.getbackptroffsets(),
                                  edgealignments.getalignmentsbuffer(), edgeacscores);
            // emulate the GPU version, save result back to GPU
            parallelstate->alignresult->assign (edgealignments.getalignmentsbuffer(), false);
            parallelstate->edgeacscoresgpu->assign (edgeacscores, true);
        }
    }

    // parallelforwardbackwardlattice() -- compute the latticelevel logpps using forwardbackward
    double lattice::parallelforwardbackwardlattice (parallelstate & parallelstate, const std::vector<float> & edgeacscores,
                                                    const edgealignments & thisedgealignments, const float lmf, const float wp, 
                                                    const float amf, const float boostingfactor, std::vector<double> & logpps,
                                                    std::vector<double> & logalphas, std::vector<double> & logbetas, const bool returnEframescorrect, 
                                                    const_array_ref<size_t> & uids, std::vector<double> & logEframescorrect, 
                                                    std::vector<double> & Eframescorrectbuf, double & logEframescorrecttotal) const
    {                                                                           //^^ TODO: remove this
        vector<size_t> batchsizeforward;      // record the batch size that exclude the data dependency for forward
        vector<size_t> batchsizebackward;     // record the batch size that exclude the data dependency for backward

        size_t endindexforward = edges[0].E;
        size_t countbatchforward = 0;

        size_t endindexbackward = edges.back().S;
        size_t countbatchbackward = 0;
        foreach_index (j, edges)                    // compute the batch size info for kernel launches
        {
            if (edges[j].S < endindexforward)
                countbatchforward++;                // note: we don't check forward because the order of end node is assured.
            else
            {
                batchsizeforward.push_back (countbatchforward);
                countbatchforward = 1;
                endindexforward = edges[j].E;
            }
            const size_t backj = edges.size() -1 - j;
            if (edges[backj].E > endindexbackward)
            {
                countbatchbackward++;
                if (endindexbackward < edges[backj].S)
                    endindexbackward = edges[backj].S;
            }
            else
            {
                batchsizebackward.push_back (countbatchbackward);
                countbatchbackward = 1;
                endindexbackward = edges[backj].S;
            }
        }
        batchsizeforward.push_back (countbatchforward);
        batchsizebackward.push_back (countbatchbackward);

        std::vector<unsigned short> uidsuint(uids.size());       // actually we shall not do this, but as it will not take much time, let us just leave it here now.
        foreach_index(i, uidsuint) uidsuint[i] = (unsigned short)uids[i];

        double totalfwscore = 0.0f;
        if (!parallelstate->emulation)
        {

            fprintf(stderr, "parallelforwardbackwardlattice: %d launches for forward, %d launches for backward\n", batchsizeforward.size(), batchsizebackward.size());

            const bool allocateframescorrect = (returnEframescorrect || boostingfactor != 0.0f);
            const bool copyuids = (returnEframescorrect || boostingfactor != 0.0f);
            const bool allocateaccvectors = returnEframescorrect;
            parallelstate->allocfwbwvectors (edges, nodes, uidsuint, allocateframescorrect, copyuids, allocateaccvectors);

            std::unique_ptr<latticefunctions> latticefunctions (msra::cuda::newlatticefunctions());     // final CUDA call
            latticefunctions->forwardbackwardlattice (&batchsizeforward[0], &batchsizebackward[0], batchsizeforward.size(), batchsizebackward.size(),
                                                      parallelstate->spalignunitid, parallelstate->silalignunitid,
                                                      *parallelstate->edgeacscoresgpu.get(), *parallelstate->edgesgpu.get(), 
                                                      *parallelstate->nodesgpu.get(), *parallelstate->aligngpu.get(), 
                                                      *parallelstate->alignresult.get(), *parallelstate->alignoffsetsgpu.get(), 
                                                      *parallelstate->logppsgpu.get(), *parallelstate->logalphasgpu.get(), 
                                                      *parallelstate->logbetasgpu.get(), lmf, wp, amf, boostingfactor,
                                                      returnEframescorrect, *parallelstate->uidsgpu.get(), *parallelstate->senone2classmapgpu.get(), 
                                                      *parallelstate->logaccalphasgpu.get(), *parallelstate->logaccbetasgpu.get(), 
                                                      *parallelstate->logframescorrectedgegpu.get(), *parallelstate->logEframescorrectgpu.get(), 
                                                      *parallelstate->Eframescorrectbufgpu.get(), logEframescorrecttotal, totalfwscore);
        }
        else            // emulation
        {
#ifndef TWO_CHANNEL
            fprintf (stderr, "forbid invalid sil path\n");
            const size_t alphabetanoderatio = 1;
#else
            const size_t alphabetanoderatio = 2;
#endif
            logpps.resize (edges.size());
            logalphas.resize (alphabetanoderatio * nodes.size());
            logbetas.resize (alphabetanoderatio * nodes.size());

            std::vector<double> logaccalphas;
            std::vector<double> logaccbetas;
            std::vector<double> logframescorrectedge;
            logframescorrectedge.resize (edges.size());

            if (returnEframescorrect)
            {
                logaccalphas.resize (alphabetanoderatio * nodes.size());
                logaccbetas.resize (alphabetanoderatio * nodes.size());

                logEframescorrect.resize (edges.size());
                Eframescorrectbuf.resize (edges.size());
            }

            totalfwscore = emulateforwardbackwardlattice (&batchsizeforward[0], &batchsizebackward[0], 
                                                          batchsizeforward.size(), batchsizebackward.size(), 
                                                          parallelstate->spalignunitid, parallelstate->silalignunitid,
                                                          edgeacscores, edges, nodes, align,
                                                          thisedgealignments.getalignmentsbuffer(), thisedgealignments.getalignoffsets(),
                                                          logpps, logalphas, logbetas, lmf, wp, amf, boostingfactor, returnEframescorrect, uidsuint, parallelstate->senone2classmapcpuforgpu, 
                                                          logaccalphas, logaccbetas, logframescorrectedge, logEframescorrect, Eframescorrectbuf, logEframescorrecttotal);

        }
        return totalfwscore;
    }

    // ------------------------------------------------------------------------
    // parallel implementations of sMBR error updating step
    // ------------------------------------------------------------------------
    void lattice::parallelsMBRerrorsignal (parallelstate & parallelstate, const edgealignments & thisedgealignments, 
                                           const std::vector<double> & logpps, const float amf, const std::vector<double> & logEframescorrect, 
                                           const double logEframescorrecttotal, msra::math::ssematrixbase & errorsignal, msra::math::ssematrixbase & errorsignalneg) const
    {
        /*  time measurement for CUDA
            cudaerrorcopyto:        3.058592 ms 
            cudaerrorsignalsync:    0.356998 ms 
            cudaerrorcopyback:      8.983703 ms  */ 

        /*  time measurement for emulation
            emuresettozeros:    102.979899 ms
            emuerrorsignal:     336.407459 ms  */

        if (!parallelstate->emulation)
        {
            // We allocate a pos and a neg buffer.
            const bool cacheerrorsignalneg = true;
            parallelstate->cacheerrorsignal (errorsignal, cacheerrorsignalneg);

            std::unique_ptr<latticefunctions> latticefunctions (msra::cuda::newlatticefunctions());
            latticefunctions->sMBRerrorsignal (*parallelstate->alignresult.get(), *parallelstate->alignoffsetsgpu.get(), *parallelstate->edgesgpu.get(),
                                              *parallelstate->nodesgpu.get(), *parallelstate->logppsgpu.get(),amf, *parallelstate->logEframescorrectgpu.get(), 
                                              logEframescorrecttotal,
                                              *parallelstate->errorsignalgpu.get(), *parallelstate->errorsignalneggpu.get());

            parallelstate->errorsignalgpu->fetch (0, errorsignal.rows(), 0, errorsignal.cols(), &errorsignal(0, 0), errorsignal.getcolstride(), true);

        }
        else
        {
            emulatesMBRerrorsignal (thisedgealignments.getalignmentsbuffer(), thisedgealignments.getalignoffsets(), edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg);
        }
    }

    // ------------------------------------------------------------------------
    // parallel implementations of MMI error updating step
    // ------------------------------------------------------------------------
    void lattice::parallelmmierrorsignal (parallelstate & parallelstate, const edgealignments & thisedgealignments, 
                                 const std::vector<double> & logpps, msra::math::ssematrixbase & errorsignal) const
    {
        if (!parallelstate->emulation)
        {
            const bool cacheerrorsignalneg = false;     // we do not need it in mmi mode
            parallelstate->cacheerrorsignal (errorsignal, cacheerrorsignalneg);
            
            std::unique_ptr<latticefunctions> latticefunctions (msra::cuda::newlatticefunctions());
            latticefunctions->mmierrorsignal (*parallelstate->alignresult.get(), *parallelstate->alignoffsetsgpu.get(), *parallelstate->edgesgpu.get(),
                                               *parallelstate->nodesgpu.get(), *parallelstate->logppsgpu.get(), *parallelstate->errorsignalgpu.get());

            //parallelstate->errorsignalgpu->fetch (0, errorsignal.rows(), 0, errorsignal.cols(), &errorsignal(0, 0), errorsignal.getcolstride(), true);
        }
        else
        {
            emulatemmierrorsignal (thisedgealignments.getalignmentsbuffer(), thisedgealignments.getalignoffsets(), edges, nodes, logpps, errorsignal);
        }
    }
};};