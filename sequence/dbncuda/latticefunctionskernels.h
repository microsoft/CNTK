// latticefunctionskernels.cu(.h) -- kernels for lattice ops intended for use with CUDA, to be called from actual CUDA kernels.
//
// To make this compile for the CPU (emulation for testing), add this line before #including this:
// #define __device__
//
// F. Seide, Aug 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/latticefunctionskernels.h $
// 
// 181   5/13/13 10:28 Fseide
// (yups, accidentally had modified the prev code before check-in, undone)
// 
// 180   5/13/13 10:27 Fseide
// [from Rui Zhao] changed lr3transP to a full trans matrix, to allow for
// IPE-Speech's fully ergodic silence model--note the TODOs and one BUGBUG
// in there that must be fixed
// 
// 179   5/10/13 12:27 Fseide
// (added a comment)
// 
// 178   4/08/13 8:43p V-hansu
// (fix some comments)
// 
// 177   1/11/13 6:48p V-hansu
// add code for DIRECT_MODE of smbr, not activated
// 
// 176   11/21/12 8:58p V-hansu
// rename state2classmap to senone2classmap
// 
// 175   11/21/12 7:10p V-hansu
// rename statetoclassmap to state2classmap
// 
// 174   11/21/12 6:30p V-hansu
// modify isofsameclass() and remove index checking
// 
// 173   11/21/12 6:04p V-hansu
// add statetoclassmap to forwardbackward() to get prepared for mpe
// approximation
// 
// 172   11/21/12 5:49p V-hansu
// (add some comments)
// 
// 171   11/21/12 5:46p V-hansu
// add method isofsameclass()
// 
// 170   11/20/12 16:08 Fseide
// another size_t-correctness for Win32 builds
// 
// 169   11/20/12 13:59 Fseide
// fixed some size_t correctness for Win32 builds
// 
// 168   11/20/12 12:34p V-hansu
// fix the bug in issilencestate()
// 
// 167   11/20/12 12:06p V-hansu
// fix the bug w.r.t. silstate in sMBR and BMMI
// 
// 166   11/19/12 8:32p V-hansu
// exclude silence state as well if skipsilence is set to true, remove
// HOLD_SIL_SMBR
// 
// 165   11/18/12 7:54p V-hansu
// (finish last check in)
// 
// 164   11/18/12 7:54p V-hansu
// fix the bug wrt logframescorrectedgej, add SIL_HOLD_SMBR, not enabled
// 
// 163   11/10/12 3:30p V-hansu
// (modify a small bug)
// 
// 162   11/10/12 3:28p V-hansu
// add logaddseen() to support new setunseen stratege
// 
// 161   11/09/12 1:20p V-hansu
// remove #undef EXCLUDE_SILENCE_ACCURACY
// 
// 160   11/06/12 8:45p V-hansu
// add code to logadd() to prepare for separation of unseen states
// 
// 159   11/06/12 3:21p Fseide
// new function force_crash()
// 
// 158   11/05/12 12:22a V-hansu
// undo last checkin :( not got enough time to do so...
// 
// 157   11/02/12 9:44a V-hansu
// add logframescorrect to edgealignment(), prepare for moving computation
// from forwardbackward() to edgealignment()
// 
// 156   10/31/12 9:09a V-hansu
// (add some comments)
// 
// 155   10/30/12 11:30p V-hansu
// get bmmi to work and enable silence exclude in bmmi mode
// 
// 154   10/30/12 5:46p V-hansu
// add exclude_sil_accuracy for bmmi and patch bmmi for
// forbid_invalid_sil_path
// 
// 153   10/29/12 4:18p V-hansu
// add boost mmi
// 
// 152   10/29/12 3:36p V-hansu
// add boosting factor to prepare for BMMI
// 
// 151   10/22/12 9:24p V-hansu
// fix the bug relating to parallel_sil, back up pathscore2 for
// computation of pathscore0
// 
// 150   10/22/12 3:02p V-hansu
// deactivate HACK_IN_SILENCE, move PARALLEL_SIL to latticestorage.h
// 
// 149   10/21/12 1:40p V-hansu
// rename alignoutput to alignresult abd change the interface of
// edgealignmentj
// 
// 148   10/19/12 11:57a V-hansu
// fix the bound for extra path signal pass in forwardlatticej and
// backwardlatticej
// 
// 147   10/18/12 2:09p V-hansu
// reference nodes to make build
// 
// 146   10/18/12 2:08p V-hansu
// rename Eframecorrecttotal to Eframescorrecttotal
// 
// 145   10/18/12 9:58a Fseide
// further tidying of fwbw functions
// 
// 144   10/18/12 8:32a Fseide
// commented expdiff(), and templated it to a float and double version
// 
// 143   10/17/12 8:36p Fseide
// new kernel-kernel computesMBRerrorsignals()
// 
// 142   10/17/12 8:10p Fseide
// removed a redundant logadd
// 
// 141   10/17/12 6:24p V-hansu
// add expdiff() and change Eframescorrect to edgelogEframescorrect
// 
// 140   10/17/12 6:15p V-hansu
// change diff () to expdiff () in sMBRerrorsignalj
// 
// 139   10/17/12 5:59p V-hansu
// rename Eframescorrectotal to logEframescorrectotal, rename
// Eframescorrectbuf or Eframescorrectdiff to logEframescorrect
// 
// 138   10/17/12 5:02p Fseide
// carried over changes to forwardlatticej() also to backwardlatticej()
// 
// 137   10/17/12 3:49p V-hansu
// change eframecorrect from float vector to double vector
// 
// 136   10/17/12 3:33p V-hansu
// turn eframecorrect (eframecorrectdiff) from float vector to double
// vector
// 
// 135   10/17/12 2:42p V-hansu
// make build
// 
// 134   10/17/12 2:02p Fseide
// (fixed a typo in the last check-in)
// 
// 133   10/17/12 1:09p Fseide
// rewrote forwardlatticej() into a more systematic pattern --TODO: do the
// same to backwardlatticej();
// new method stateposteriorsj() in prep for fixing sMBR implementation
// 
// 132   10/17/12 12:06p V-hansu
// checked in a not working version :(
// 
// 131   10/17/12 12:05p V-hansu
// (fix an indentation)
// 
// 130   10/17/12 10:36a V-hansu
// fix a bug relating to nodes.size() -1
// 
// 129   10/17/12 10:32a V-hansu
// (change tab to space)
// 
// 128   10/17/12 2:45a V-hansu
// activate FORBID_INVALID_SIL_PATHS again and fix the boundary w.r.t
// silalphas and silbetas
// 
// 127   10/17/12 2:09a V-hansu
// fix if (isaddedsil)
// 
// 126   10/16/12 5:04p V-hansu
// enable FORBID_INVALID_SIL_PATH and fix a bug relating to it
// 
// 125   10/16/12 3:38p V-hansu
// finish the logpps, logeframecorrect and logaccbetas in "dual-channel"
// mode
// 
// 124   10/16/12 2:46p V-hansu
// change startwithsil and endwithsil to isaddedsil, judged from
// edge.unused == 1
// 
// 123   10/16/12 12:31p V-hansu
// finish the logaccalpha computation in forbid sil path mode
// 
// 122   10/15/12 7:36p V-hansu
// rename SILENCE_PRUNING to FORBID_INVALID_SIL_PATHS, get logbetas
// support sil_path pruning
// 
// 121   10/15/12 5:55p V-hansu
// change the computation of logalpha to support sil pruning
// 
// 120   10/15/12 5:21p V-hansu
// add aligns to backwardlatticej, change the location of #define
// SILENCE_PRUNING
// 
// 119   10/14/12 10:05p V-hansu
// add silalignunitid and spalignunitid to forwardlatticej and
// backwardlatticej
// 
// 118   10/14/12 10:01p V-hansu
// add spalignunitid, silalignunitid and nodes to backwardlatticej and
// forwardlatticej
// 
// 117   10/12/12 1:14a V-hansu
// activate PARALLEL_SIL and fix the bpmatrix class
// 
// 116   10/11/12 7:44p V-hansu
// remove the __host__ warning temporarily and swith back to no parallel
// sil
// 
// 115   10/09/12 8:08p V-hansu
// fix a type cast problem for senoneid
// 
// 114   10/09/12 7:43p V-hansu
// get parallel sil processing similar to previous edgealignmentj, not
// turned on
// 
// 113   10/08/12 7:52p V-hansu
// add boundary check for edgealignmentj
// 
// 112   10/08/12 10:56a V-hansu
// deactivate parallel_sil
// 
// 111   10/05/12 4:21p V-hansu
// add comments to edgealignmentsj
// 
// 110   10/05/12 1:43p V-hansu
// activate parallel sil processing, fix a bug relating to pointer.
// 
// 109   10/04/12 3:31p V-hansu
// (temporarily remove the unreferenced variable)
// 
// 108   9/30/12 7:01p V-hansu
// deactivate sil processing
// 
// 107   9/30/12 6:36p V-hansu
// activate sil processing
// 
// 106   9/30/12 5:21p V-hansu
// add a #endif to make build
// 
// 105   9/30/12 5:18p V-hansu
// add backptroffsets and backptrstorage in edgealignmentsj
// 
// 104   9/30/12 3:44p V-hansu
// add something relating to processing sil on cuda, not finished, include
// in #if 0
// 
// 103   9/28/12 6:10p V-hansu
// rename spalignunit to alignunitid
// 
// 102   9/28/12 4:34p V-hansu
// change the reference method to get rid of warning
// 
// 101   9/28/12 4:17p V-hansu
// rename errorsignalbuf to errorsignalneg, activate log mode error
// accumulation, refactor atomicCASfloatdouble
// 
// 100   9/28/12 2:35p V-hansu
// refactor functions w.r.t. double and float
// 
// 99    9/28/12 1:58p V-hansu
// rename dengammas to errorsignal, rename the parameter in shuffle(),
// activate linear mode errorsignal
// 
// 98    9/27/12 10:53p V-hansu
// add two kinds of shuffle order
// 
// 97    9/27/12 7:54p V-hansu
// add shuffle function
// 
// 96    9/27/12 4:08p V-hansu
// activate log mode in sMBRerrorsignalj
// 
// 95    9/27/12 12:19a V-hansu
// add back dengammasbuf to reference it
// 
// 94    9/27/12 12:18a V-hansu
// add back amf again to enable exp mode sMBRerrorsignal
// 
// 93    9/26/12 11:35p V-hansu
// remove amf from sMBRerrorsignalj
// 
// 92    9/26/12 10:52p V-hansu
// modify computation of logedgecorrect to include amf
// 
// 91    9/26/12 10:48p V-hansu
// add atomicLogAddfloat and related functions, add log version of
// errorsignalj
// 
// 90    9/26/12 7:23p V-hansu
// pass siltransPindex into forwardbackwardalign to prepare for silence
// processing
// 
// 89    9/26/12 5:42p V-hansu
// change uids from sizetvector back to ushortvector
// 
// 88    9/26/12 3:52p V-hansu
// add setvaluej and expdiffj function
// 
// 87    9/26/12 2:29p V-hansu
// change logpps in errorsignal from float to double. remove
// Eframecorrecttotal in sMBRerrorsignal, change the location of resize
// towards logpps and Eframecorrect
// 
// 86    9/26/12 1:52p V-hansu
// rename Eframescorrect to Eframescorrectbuf, and add Eframescorrectdiff
// for difference computation.
// 
// 85    9/26/12 1:08p V-hansu
// rename combinemode to returnEframescorrect
// 
// 84    9/26/12 12:57p Fseide
// renamed errorsignalj() to sMBRerrorsignalj()
// 
// 83    9/26/12 12:53p Fseide
// errorsignal() renamed to sMBRerrorsignal()
// 
// 82    9/26/12 12:27p Fseide
// renamed logdengammaspos/neg to dengammas/dengammasbuf
// 
// 81    9/26/12 11:57a Fseide
// sMBRerrorsignal() now takes two dengammas accumulators, in prep for pos/neg
// logadd
// 
// 80    9/25/12 5:31p V-hansu
// remove totalfwacc in backwardlatticej
// 
// 79    9/25/12 3:34p V-hansu
// temporarily check in to make build
// 
// 78    9/25/12 3:11p V-hansu
// add sizetvector and change uids from uintvector into sizetvector
// 
// 77    9/25/12 1:12p V-hansu
// add alignemts and alignmentoffsets to forwardlattce related function to
// finish the algorithm
// 
// 76    9/25/12 11:36a V-hansu
// change some atomiclogadd to logadddouble to get compiled
// 
// 75    9/24/12 10:07p V-hansu
// change the interface relating to forwardbackwardlattice to get prepared
// for the combined mode fwbw, not finished yet
// 
// 74    9/24/12 11:02a V-hansu
// remove batchsize in interface of forwadlatticej
// 
// 73    9/24/12 1:07a V-hansu
// add QUEUE_LOGADD to forwardj and backwardj
// 
// 72    9/21/12 7:31p V-hansu
// (add direct logadddouble call, not enabled)
// 
// 71    9/21/12 6:50p V-hansu
// (change tab into space)
// 
// 70    9/21/12 6:43p V-hansu
// (add some comments)
// 
// 69    9/21/12 6:30p V-hansu
// use new logadd(double) to avoid numerical issue
// 
// 68    9/21/12 3:13p V-hansu
// modify #define atomicCAS again to make it the "same" as GPU version
// 
// 67    9/21/12 2:43p V-hansu
// factor out logAdd, rewrite cpu version of atomicCAS
// 
// 66    9/21/12 1:42p V-hansu
// (remove some unnecessary brackets)
// 
// 65    9/20/12 11:53p V-hansu
// modify __double_as_longlong to change type cast to reinterpret
// 
// 64    9/20/12 12:04p V-hansu
// remove some warnings by reference arguments in #define atomicAdd
// 
// 63    9/20/12 11:38a V-hansu
// remove sm_12_atomic_functions.h and move function to
// latticefunctionskernels.h
// 
// 62    9/19/12 9:33a Fseide
// renamed edgeinfo to edgeinfowithscores, in prep for V2 lattice format
// 
// 61    9/18/12 8:28a Fseide
// commented out a build-breaking call
// 
// 60    9/17/12 8:06p V-hansu
// change float into double for logalphas and logbetas in forwardbackward
// 
// 59    9/17/12 7:07p V-hansu
// finish atomicLogAdd
// 
// 58    9/16/12 9:34p V-hansu
// add some fake code to compile
// 
// 57    9/16/12 9:32p V-hansu
// remove cuda.h
// 
// 56    9/16/12 9:31p V-hansu
// add atomicLogAdd, not finished. add doublevector
// 
// 55    9/16/12 6:38p V-hansu
// comment out unfinished code to make it compile
// 
// 54    9/16/12 5:23p V-hansu
// add atomicLogAdd function
// 
// 53    9/14/12 9:10p V-hansu
// fix a bug relating to backwardj
// 
// 52    9/14/12 5:55p V-hansu
// add backwardlatticej
// 
// 51    9/14/12 5:34p V-hansu
// add backwardlatticej
// 
// 50    9/14/12 2:37p V-hansu
// add forwardlatticej and forwardbackwardlattice in
// latticefunctionskernels and related classes
// 
// 49    9/14/12 1:57p V-hansu
// finish forwardlatticej, change forwardbackwardlattice's lambda into
// normal vector, both not tested
// 
// 48    9/14/12 1:27p V-hansu
// add fowardlatticej, not tested
// 
// 47    9/08/12 6:39p V-hansu
// change the interface of latticeforwardbackward for sMBR, now only use
// signal (numgammas)
// 
// 46    9/07/12 9:33a Fseide
// (fixed signed/unsigned/64-bit warnings for 32-bit build)
// 
// 45    9/06/12 10:19p V-hansu
// add a macro to check #if __CUDA_ARCH__ < 200
// 
// 44    9/06/12 9:29p V-hansu
// add another #define to make atomicAdd temporarily work for cuda
// 
// 43    9/06/12 8:34p V-hansu
// fix a bug relating to index of alignstateids
// 
// 42    9/06/12 8:24p V-hansu
// disable atomicAdd, there is something wrong with that
// 
// 41    9/06/12 8:20p V-hansu
// enable atomicAdd
// 
// 40    9/06/12 8:07p Fseide
// added emulation function atomicAdd()
// 
// 39    9/06/12 8:02p V-hansu
// finish sMBRerrorsignalj temporarily
// 
// 38    9/06/12 7:26p V-hansu
// add alignoffsets into sMBRerrorsignal function, next step shall be
// implementation
// 
// 37    9/06/12 7:24p V-hansu
// add alignoffsets into interface, same as alignstateids
// 
// 36    9/06/12 7:13p V-hansu
// add alignoffsets in sMBRerrorsignalj
// 
// 35    9/05/12 10:36p V-hansu
// add function sMBRerrorsignal and codes relating to it
// 
// 34    9/04/12 10:25p V-hansu
// change the interface of edgealignmentj
// 
// 33    9/04/12 5:04p Fseide
// (fixed a compiler warning that trips up CUDA)
// 
// 32    9/04/12 3:36p V-hansu
// (fix a bug)
// 
// 31    9/04/12 3:36p V-hansu
// fix a but relating to pathscore2 computation
// 
// 30    9/04/12 3:24p Fseide
// minor edits and further mem-access optimizations
// 
// 29    9/04/12 3:01p V-hansu
// reoptimize the instructions relating to pathscore computation
// 
// 28    9/04/12 2:42p V-hansu
// changed the position of some instructions
// 
// 27    9/04/12 2:27p V-hansu
// fix all bugs, next step shall be optimization
// 
// 26    9/04/12 1:59p V-hansu
// fix a bug relating to pathscore0
// 
// 25    9/04/12 1:43p V-hansu
// fix a bug relaing to pathscore0
// 
// 24    9/04/12 1:26p V-hansu
// fix a bug relaing to tee model score computing
// 
// 23    9/04/12 12:48p V-hansu
// fix a bug relating to align index
// 
// 22    9/04/12 12:25p V-hansu
// fix a bug
// 
// 21    9/04/12 11:09a V-hansu
// fix a bug relating to fwscore
// 
// 20    9/03/12 10:52p V-hansu
// get through a bug relaing to sp tee model
// 
// 19    9/03/12 10:26p V-hansu
// get through a bug
// 
// 18    9/03/12 10:14p V-hansu
// remove pathscoreold[0-3]
// 
// 17    9/03/12 9:21p V-hansu
// get through a bug
// 
// 16    9/03/12 9:20p V-hansu
// pull out t = ts initialization procedure
// 
// 15    9/03/12 9:04p V-hansu
// add pathscoreold[0-2] to record log pp
// 
// 14    9/03/12 8:43p V-hansu
// modify some code in edgealignmentj
// 
// 13    9/03/12 8:20p V-hansu
// modify edgealignmentj, not finished testing
// 
// 12    9/03/12 8:16p V-hansu
// add definition of LOGZERO
// 
// 11    9/03/12 8:13p V-hansu
// change some code in edgealignmentj and add some comments
// 
// 10    9/03/12 7:30p V-hansu
// modify a bug
// 
// 9     9/03/12 6:36p V-hansu
// add a comment
// 
// 8     9/03/12 4:56p V-hansu
// rewrite the function edgealignmentj
// 
// 7     9/02/12 8:24p V-hansu
// modify some codes in edgealignmentj
// 
// 6     9/02/12 4:45p V-hansu
// add sptransPindex into interface
// 
// 5     9/02/12 3:32p V-hansu
// change interface of edgealignmentj
// 
// 4     9/01/12 8:00p Fseide
// optimized lr3transP
// 
// 3     9/01/12 7:57p Fseide
// (added a comment)
// 
// 2     9/01/12 7:22p Fseide
// (fixed two compiler warnings)
// 
// 1     9/01/12 3:01p Fseide
#if 0
#endif

#define FORBID_INVALID_SIL_PATHS    // [v-hansu] prune path that start from sil(sp) and go into sil, only used with addsil is adopted

#pragma once

#pragma push_macro ("__device__")
#pragma push_macro ("atomicAdd")
#pragma push_macro ("atomicCAS")

#include "latticestorage.h"

namespace msra { namespace cuda { class passtextureref; }}

#ifdef __kernel_emulation__
#include "math.h"       // to get exp() and log() compiled correctly with c++
#include<stdexcept>
using namespace std;
#ifndef __device__
#define __device__
#endif
#define CUDART_MIN_DENORM_F numeric_limits<float>::denorm_min()
#define atomicAdd(address,value) (*(address)+=(value))  // don't forget to #undef (#praga pop_macro)! Otherwise CUDA might compile with this...
#define atomicCAS(address, compare, val) *address; *address = *address == compare ? val : *address;
#define __double_as_longlong(in) (*(unsigned long long int *) &in)
#define __longlong_as_double(in) (*(double *) &in)
#define __float_as_int(in) (*(int *) &in)
#define __int_as_float(in) (*(float *) &in)
#else    // TODO: remove this once we got this figured out
#include "math_constants.h"
#if __CUDA_ARCH__ < 200
//#warning Sequence training not supported on 1.x CUDA machines.
#define force_crash() (*((int*)-1)=0)               // TODO: this does not in fact seem to crash it...
#define atomicAdd(a,v) (force_crash(),*(a)=v)       // force a crash if used with 1.x devices 
#define atomicCAS(address, compare, val) (*(address) = compare + val, *((int*)-1)=0)
#define __double_as_longlong(in) (force_crash(), in)
#define __longlong_as_double(in) (force_crash(), in)
#define __float_as_int(in) (force_crash(), in)
#define __int_as_float(in) (force_crash(), in)
#endif
#endif

namespace msra { namespace lattices {

struct somedata         // example type to have a pattern to copy from
{
    size_t fortytwo;
};

struct empty {};

// Note that the code that uses this (edgealignmentj) will assume 3-state left-to-right
// except for /sil/ and /sp/ which are treated specially.
// TODO: either check when creating this whether this assumption is true, or control this through a flag in here.
struct lr3transP        // lr3 = 3-state left-to-right architecture
{
    
        static const size_t MAXSTATES = 3;
        size_t numstates;
        float loga[MAXSTATES+1][MAXSTATES+1];
        

    lr3transP ()
    {
#ifdef INITIAL_STRANGE  
        numstates = 3;
        for (size_t i = 0; i < NUMSTATES+1; i++)
            for (size_t j = 0; j < NUMSTATES+1; j++)
            {
                loga[i][j] = LOGZERO;            
            }
#endif
    }
};

struct lrhmmdef         // left-to-right HMM (no /sil/)
{
    static const size_t MAXSTATES = 3;              // we use a fixed memory allocation since it's almost always 3 anyway

    unsigned char transPindex;                      // index of monophone to find transP matrix
    unsigned char numstates;                        // number of states; code supports only either 1 or 3
    unsigned short senoneids[MAXSTATES];             // [0..numstates-1] senone indices

    size_t getsenoneid (size_t i) const { return (size_t) senoneids[i]; }
    size_t getnumstates() const { return (size_t) numstates; }
    const struct lr3transP & gettransP(const lr3transP * transPs) const { return transPs[transPindex]; }

    lrhmmdef ()
    {
#ifdef INITIAL_STRANGE
        transPindex = unsigned char (-1);
        numstates = unsigned char (-1);
        for (size_t i = 0; i < MAXSTATES; i++)
        {
            senoneids[i] = unsigned short (-1);
        }
#endif
    }
};

#if 1   // straight-forward version
#else   // CUDA hacked version
hmm gethmm (hmms, i)
{
    ushort4 u4 = *((ushort4) &hmms[i]);
    lrhmmdef hmm;
    hmm.transPindex = u4.x & 0xff;
    hmm.numstates = u4.x >> 8;
    hmm.senoneids[0] = u4.y;
    hmm.senoneids[1] = u4.z;
    hmm.senoneids[2] = u4.w;
}
#endif

#ifndef LOGZERO
#define LOGZERO -1e30f
#endif

class bpmatrixref
{
private:
    unsigned short * p;      // pointer in CUDA space of this device
    size_t numrows;     // rows()
    size_t numcols;     // cols()
    size_t colstride;   // height of column = rows() rounded to multiples of 4
    __device__ void checkbounds (size_t i, size_t j) const 
    {
        if (i >= numrows || j >= numcols)
#ifdef __kernel_emulation__
            throw::logic_error ("out of boundary!!!");
#else
            *((int*)-1)=0;
#endif
    }
    __device__ size_t locate (size_t i, size_t j) const 
    {
        checkbounds (i,j);
        return j * colstride + i; 
    }   // matrix in column-wise storage
public:
    __device__ bpmatrixref (unsigned short * address, size_t n, size_t m)
    {
        numrows = n;
        numcols = m;
        colstride = n;
        p = address;
    }

    __device__ unsigned short &       operator() (size_t i, size_t j)       { return p[locate(i,j)]; }
    __device__ const unsigned short & operator() (size_t i, size_t j) const { return p[locate(i,j)]; }
};

// this class contains all-static methods that are inner pieces of thread kernels for use with CUDA
struct latticefunctionskernels
{
    // [v-hansu]  mimic the float version of atomicCAS
    static __device__ float atomicCASfloatdouble (float *address, float compare, float val)
    {
        int * intaddress = (int *) address;
        int intcompare = __float_as_int(compare);             // __double_as_longlong : read double as unsigned long long
        int intval = __float_as_int(val);
        int result = atomicCAS (intaddress, intcompare, intval);
        return __int_as_float(result);
    }

    static __device__ double atomicCASfloatdouble (double *address, double compare, double val)
    {
        unsigned long long int * longlongintaddress = (unsigned long long int *) address;
        unsigned long long int longlongintcompare = __double_as_longlong (compare);             // __double_as_longlong : read double as unsigned long long
        unsigned long long int longlongintval = __double_as_longlong (val);
        unsigned long long int result = atomicCAS(longlongintaddress, longlongintcompare, longlongintval);
        return __longlong_as_double(result);
    }

    static __device__ void logaddratio (double & loga, double diff)
    {
        if (diff < -37.0f) return;      // log (2^-53), 52-bit mantissa -> cut of after 53th bit
        loga += log (1.0 + exp (diff));
    }
    static __device__ void logaddratio (float & loga, float diff) 
    { 
        if (diff < -17.0f) return; // log (2^-24), 23-bit mantissa -> cut of after 24th bit 
        loga += logf (1.0f + expf (diff)); 
    }

    template<typename FLOAT> static __device__ void swap (FLOAT & left, FLOAT & right)
    {    // exchange values stored at _Left and _Right
        FLOAT tmp = left;
        left = right;
        right = tmp;
    }

    // overloads for exp() for float and double, so that we can use templates
    static __device__ float expfd (float x) { return ::expf (x); }  // TODO: ain't there an overload for this?
    static __device__ double expfd (double x) { return ::exp (x); }

    // Compute the difference of two numbers, which are represented as their logs.
    // The return value is a non-log value. exp(loga) - exp(logb)
    template<typename FLOAT>
    static __device__ FLOAT expdiff (FLOAT loga, FLOAT logb)
    {
        if (logb < loga)  // logb - loga < 0 => exp(logb-loga) < 1
            return  expfd (loga) * (1 - expfd (logb - loga));
        else
            return -expfd (logb) * (1 - expfd (loga - logb));
    }

    // loga <- log (exp (loga) + exp (logb)) = log (exp (loga) * (1.0 + exp (logb - loga)) = loga + log (1.0 + exp (logb - loga))
    template<typename FLOAT> static __device__ void logadd (FLOAT & loga, FLOAT logb)
    {
        if (logb > loga)            // we add smaller to bigger
            swap (loga, logb);      // loga is bigger
        if (loga <= LOGZERO)         // both are 0
            return;
        logaddratio (loga, logb - loga);
    }

    // does the same as above but if the bigger one is too small, we assign a small value to it
    template<typename FLOAT> static __device__ void logaddseen (FLOAT & loga, FLOAT logb)
    {
        if (logb > loga)            // we add smaller to bigger
            swap (loga, logb);      // loga is bigger
        if (loga <= LOGZERO)         // both are 0
        {
            loga = logf(CUDART_MIN_DENORM_F);   // [v-hansu] we hope to separate LOGZERO (unseen states) and logf(CUDART_MIN_DENORM_F) (seen states with small prob)
            return;
        }
        logaddratio (loga, logb - loga);
    }

    //same pattern as atomicAdd(), but performing the log-add operation instead 
    template<typename FLOAT> static __device__ FLOAT atomicLogAdd (FLOAT * address, FLOAT val)
    {
        FLOAT old = *address;
        FLOAT assumed;
        FLOAT logaddresult;
        do {
            assumed = old;                                  // old is the assumed value at address
            logaddresult = assumed;                         // for next step to compute logaddresult, assumed shall be the same as before
            logaddseen (logaddresult, val);
            old = atomicCASfloatdouble (address, assumed, logaddresult);
        } while (assumed != old);                           // if old == assumed, the *address is not changed in this loop, so this is safe
        return old;
    }

    // [v-hansu] shuffling accessing order for a item in cubic(Ni, Nj, Nk) with index i, j, k according to shufflemode
    static inline __device__ size_t shuffle (size_t i, size_t Ni, size_t j, size_t Nj, size_t k, size_t Nk, size_t shufflemode)
    {
        if (shufflemode == 0)
            return i + j * Ni + k * Nj * Ni;
        else if (shufflemode == 1)  // inverse
            return k + j * Nk + i * Nj * Nk;
        else if (shufflemode == 2)  // flip i and j
            return j + i * Nj + k * Ni * Nj;
        else if (shufflemode == 3)  // flip j and k
            return i + k * Ni + j * Nk * Ni;
        else if (shufflemode == 4)
            return j + k * Nj + i * Nk * Nj;
        else
            *((int*)-1)=0;            // shall not get here, WRONG
        return 0;
    }

    template<typename doublevector>
    static __device__ void setvaluej (size_t j, doublevector & thisvector, double value)
    {
        thisvector[j] = value;
    }

    //zhaorui
    static inline __device__ float getlogtransp (lr3transP transP, int from, int to)
    {
        /*if (from < -1 || from >= transP.MAXSTATES || to > transP.MAXSTATES) 
        {
            //printf("from: %d to: %d\n", from, to);
            return LOGZERO;
        }*/
        return transP.loga[from+1][to];
    }
    template<typename lrhmmdefvector, typename lr3transPvector, typename matrix, typename nodeinfovector, typename edgeinfowithscoresvector, typename aligninfovector, typename ushortvector, typename uintvector, typename floatvector, typename sizetvector>
    static inline __device__ void edgealignmentj (size_t j, const lrhmmdefvector & hmms, const lr3transPvector & transPs, const size_t spalignunitid,
                                                  const size_t silalignunitid, const matrix & logLLs, const nodeinfovector & nodes,
                                                  const edgeinfowithscoresvector & edges, const aligninfovector & aligns, 
                                                  const uintvector & alignoffsets, ushortvector & backptrstorage, const sizetvector & backptroffsets,
                                                  ushortvector & alignresult, floatvector & edgeacscores)
    {                                            // TODO: alignresult will change to (start,end) 
        // mostly finished
        // some preparation
        size_t as = edges[j].firstalign;        // align start
        size_t ae = (j+1) < edges.size() ? (size_t) edges[j+1].firstalign : aligns.size();
        if (as == ae)       // the last empty alignment
            return;
        size_t ts = nodes[edges[j].S].t;

        float fwscore = 0.0f;       // score passed across phone boundaries
        size_t alignindex = alignoffsets[j];        // index to set (result)
#ifndef PARALLEL_SIL
        const bool isSil = (aligns[as].unit == silalignunitid || aligns[ae-1].unit == silalignunitid);
        if (isSil)  return;     // we do not support silence edge now, which is computed by cpu, may change when we support it
#endif
        // Viterbi alignment
        for (size_t k = as; k < ae; k++)
        {
            const aligninfo align = aligns[k];
            const size_t numframes = align.frames;
            const bool isSp = (align.unit == spalignunitid);
            const bool isSil = (align.unit == silalignunitid);

            const lrhmmdef hmm = hmms[align.unit];
            const lr3transP transP = transPs[hmm.transPindex];
          

            // pre-fetch senone ids into registers
            size_t senoneid0 = hmm.senoneids[0];
            size_t senoneid1 = 0;
            size_t senoneid2 = 0;
            if (!isSp)      // fetch only if needed--may save some memory cycles
            {
                senoneid1 = hmm.senoneids[1];
                senoneid2 = hmm.senoneids[2];
            }

            const size_t te = ts + numframes;               // end time of current unit

            size_t state1step0to1 = te;                     // inflection point from state 0 to 1, record in state 1
            size_t state2step0to1 = te;                     // inflection point from state 0 to 1, record in state 2
            size_t state2step1to2 = te;                     // inflection point from state 1 to 2, record in state 2

            //now we only support transition from -1 to 0 or 2 for sil
            float pathscore0 = fwscore ;                     // log pp in state 0
            float pathscore1 = LOGZERO;                     // log pp in state 1
            float pathscore2 = LOGZERO;                     // log pp in state 2
            if(isSil)
                pathscore2 = fwscore;                    
                
            // first frame
            if (ts != te)                                                              // for t = ts, initialization
            {                           
                if (isSil)                                                              //for sil, -1 to 2 and -1 to 0 is permitted
                {
                    pathscore0 += getlogtransp(transP,-1,0) + logLLs(senoneid0,ts); 
                    pathscore2 += getlogtransp(transP,-1,2) + logLLs(senoneid2,ts);      
                }
                else                                                                    //for others, only -1 to 0 is permitted
                    pathscore0 +=  logLLs(senoneid0,ts);                                // Note: no need to incorporate LLs for state [1] and [2] because the path log LLs are LOGZERO anyway
            }
            
            
            float pathscore2last = pathscore2;              // allocate last for state 2 because the order of computation below is 2->1->0, last state 2 is needed because from 2 to 0 or 1 is permitedted for sil.
            float pathscore1last = pathscore1;              // allocate last for state 1 because the order of computation below is 2->1->0, last state 1 is needed because from 1 to 0 is permitedted for sil.
            
            size_t backptroffset = backptroffsets[j];               // we make use of backptrstorage in backptroffsets[j] for viterbi of ergodic model (silence)

            bpmatrixref backptrmatrix (&backptrstorage[backptroffset], hmm.MAXSTATES, numframes);
            //subsequent frames
            for (size_t t = ts + 1; t < te; t++)
            {
                if (!isSp)
                {
                    // state [2]                    
                    pathscore2 += getlogtransp(transP,2,2);                                           // log pp from state 2 to 2
                    if (isSil)
                        backptrmatrix (2, t-ts-1) = 2;
                    const float pathscore12 = pathscore1 + getlogtransp(transP,1,2);                  // log pp from state 1 to 2
                    if (pathscore12 >= pathscore2)                                              // if state 1->2
                    {
                        pathscore2 = pathscore12;
                        state2step0to1 = state1step0to1;                                        // record the inflection point
                        state2step1to2 = t;                                                     // record the inflection point
                        if (isSil)
                            backptrmatrix (2, t-ts-1) = 1;
                    }
                    if (isSil)                                                                  // only silence have path from 0 to 2
                    {
                        const float pathscore02 = pathscore0 + getlogtransp(transP,0,2);          // log pp from state 0 to 2
                        if (pathscore02 >= pathscore2)                                          // if state 0->2
                        {
                            pathscore2 = pathscore02;
                            backptrmatrix (2, t-ts-1) = 0;
                        }
                    }

                    // state [1]
                    pathscore1 += getlogtransp(transP,1,1);                                           // log pp from state 1 to 1
                    if (isSil)
                        backptrmatrix (1, t-ts-1) = 1;
                    const float pathscore01 = pathscore0 + getlogtransp(transP,0,1);                  // log pp from state 0 to 1
                    if (pathscore01 >= pathscore1)                                              // if state 0 -> 1
                    {
                        pathscore1 = pathscore01;
                        state1step0to1 = t;                                                     // record the inflection point
                        if (isSil)
                            backptrmatrix (1, t-ts-1) = 0;
                    }
                    if (isSil)                                                                  // only silence have path from 2 to 1
                    {
                        const float pathscore21 = pathscore2last + getlogtransp(transP,2,1); 
                        if (pathscore21 >= pathscore1)                                              // if state 2 -> 1
                        {
                            pathscore1 = pathscore21;
                            backptrmatrix (1, t-ts-1) = 2;
                        }
                    }
                }
                // state [0]
                pathscore0 += getlogtransp(transP,0,0);
                if(isSil)                                                                       // only silence have path from 2 or 1 to 0
                {
                    backptrmatrix (0, t-ts-1) = 0;
                    const float pathscore20 = pathscore2last + getlogtransp(transP,2,0);      // log pp from state 2 to 0
                    if (pathscore20 >= pathscore0)
                    {
                        pathscore0 = pathscore20;
                        backptrmatrix (0, t-ts-1) = 2;
                    }
                    const float pathscore10 = pathscore1last + getlogtransp(transP,1,0);      // log pp from state 1 to 0
                    if (pathscore10 >= pathscore0)
                    {
                        pathscore0 = pathscore10;
                        backptrmatrix (0, t-ts-1) = 1;
                    }
                }

                // add log LLs
                pathscore0 += logLLs(senoneid0,t);
                if (!isSp)      // only fetch if needed, saves mem access
                {
                    pathscore1 += logLLs(senoneid1,t);
                    pathscore2 += logLLs(senoneid2,t);
                }
                pathscore1last = pathscore1;                                                    // update pathscore1last
                pathscore2last = pathscore2;                                                   // update pathscore2last 
            }

            // final 'next' transition that exits from last frame

            if (ts == te)                                                                   // if sp tee model, will not in next loop
            {                
                pathscore2 = pathscore0 + getlogtransp(transP,-1,1);                                               
            }
            else if (isSp)
            {
                pathscore2 = pathscore0 + getlogtransp(transP,0,1) ;          // sp model, from 0 to 1
                //printf(" sp, %f\n", pathscore2);
            }
            
            else if(isSil)                                                                   //for sil, the exit state can be 0 or 2.
            {
                const float pathscore03 =  pathscore0 + getlogtransp(transP,0,3);
                pathscore2 += getlogtransp(transP,2,3);
                if(pathscore03 > pathscore2)
                {
                    pathscore2 = pathscore03;
                }
            }
            else
                pathscore2 += getlogtransp(transP,2,3);

            fwscore = pathscore2;                                                           // propagate across phone boundaries
            
            // emit alignment

            if (!isSil)
            {
                state2step0to1 += alignindex - ts;                              // convert to align measure
                state2step1to2 += alignindex - ts;
                for (size_t t = alignindex; t < alignindex + numframes; t++)    // set the final alignment
                {
                    size_t senoneid;
                    if (t < state2step0to1)                                     // in state 0
                        senoneid = senoneid0;
                    else if(t < state2step1to2)                                 // in state 1
                        senoneid = senoneid1;
                    else                                                        // in state 2
                        senoneid = senoneid2;
                    alignresult[t] = (unsigned short) senoneid;
                }
            }
            else                                                                        // for silence
            {
                size_t lastpointer = 2;
                const float pathscore03 =  pathscore0 + getlogtransp(transP,0,3);
                if(pathscore03 >= pathscore2)                                                       //exit state is 0
                {
                    alignresult[alignindex + numframes - 1] = (unsigned short) senoneid0;                  
                    lastpointer = 0;
                }
                else                                                                                  //exit state is 2
                    alignresult[alignindex + numframes - 1] = (unsigned short) senoneid2;             
                
                for (size_t t = alignindex + numframes - 2; (t + 1) > alignindex; t--)  // set the final alignment
                {
                    lastpointer = backptrmatrix (lastpointer, t-alignindex);
                    size_t senoneid = (size_t) (-1);
                    if (lastpointer == 0)
                        senoneid = senoneid0;
                    else if (lastpointer == 1)
                        senoneid = senoneid1;
                    else if (lastpointer == 2)
                        senoneid = senoneid2;
                    alignresult[t] = (unsigned short) senoneid;
                }
            }
            ts = te;
            alignindex += numframes;
        }
        edgeacscores[j] = fwscore;
    }

    // compute the final error signal from gammas and state-consolidated Eframescorrect
    // in-place operation is supported (i.e. output = one of the inputs) 
    template<typename matrix>
    static inline __device__  void computesMBRerrorsignals (const size_t s, const matrix & loggammas, const matrix & logEframescorrect, 
                                                            const double logEframescorrecttotal, const float kappa, matrix & errorsignal)
    {
        const float Eframescorrecttotal = expf ((float)logEframescorrecttotal);
        const size_t T = errorsignal.cols();
        for (size_t t = 0; t < T; t++)
            errorsignal(s,t) = expf (loggammas(s,t)) * (expf (logEframescorrect(s,t)) - Eframescorrecttotal) * kappa; 
    }

    // test if a state is silence [v-hansu]
    // WARNING, this function only support models with 9304 states
    // TODO: change this later on
    static inline __device__ bool issilencestate (size_t stateid, size_t numsenones)
    {
        if (numsenones == 9304 && (stateid == 7670 || stateid == 7671 || stateid == 7672))
            return true;
        else
            return false;
    }

    // compare two states and check if they are of the same class [v-hansu]
    template<typename ushortvector>
    static inline __device__ bool isofsameclass (size_t statea, size_t stateb, ushortvector senone2classmap)
    {
        if (senone2classmap.size() == 0)        // no map provided, we just do normal comparison
            return (statea == stateb);
        else
            return senone2classmap[statea] == senone2classmap[stateb];
    }

    // Phase 1 of forwardbackward algorithm
    // returnEframescorrect means sMBR mode
    template<typename edgeinforvector, typename nodeinfovector, typename aligninfovector, typename ushortvector, typename uintvector, typename floatvector, typename doublevector>
    static inline __device__ void forwardlatticej (const size_t j, const floatvector & edgeacscores, 
                                                   const size_t /*spalignunitid --unused*/, const size_t silalignunitid, 
                                                   const edgeinforvector & edges, const nodeinfovector & nodes, const aligninfovector & aligns, 
                                                   const ushortvector & alignments, const uintvector & alignmentoffsets,
                                                   doublevector & logalphas, float lmf, float wp, float amf, const float boostingfactor, 
                                                   const ushortvector & uids, const ushortvector senone2classmap, const bool returnEframescorrect, 
                                                   doublevector & logframescorrectedge, doublevector & logaccalphas)
    {
        // edge info
        const edgeinfowithscores & e = edges[j];
        double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;      // note: edgeacscores[j] == LOGZERO if edge was pruned
		//zhaorui to deal with the abnormal score for sent start. 
        if(e.l < -200.0f)
            edgescore = (0.0 * lmf + wp + edgeacscores[j]) / amf;
        const bool boostmmi = (boostingfactor != 0.0f);
        // compute the frames-correct count for this edge
        double logframescorrectedgej = LOGZERO;
        const size_t numsenones = 9304;             // WARNING: this is a hack, please fix this once smbr or bmmi is working! [v-hansu]
        bool skipsilence = true;                    // currently we skip silence for BMMI and sMBR [v-hansu]
        if (returnEframescorrect || boostmmi)
        {
            size_t ts = nodes[e.S].t;
            size_t te = nodes[e.E].t;
            size_t framescorrect = 0;                                                   // count raw number of correct frames
            size_t startindex = alignmentoffsets[j];

            size_t as = e.firstalign;                                                   // align start
            size_t ae = (j+1) < edges.size() ? (size_t) edges[j+1].firstalign : aligns.size();
            const bool isSil = (ae == as + 1 && aligns[as].unit == silalignunitid);     // the order of this judgement shall be changed to save memory access
            if (!(isSil && skipsilence))         // we don't count silence when 1. is silence; 2, skip them
            {
                for (size_t t = ts; t < te; t++)
                {
                    if (!(skipsilence && issilencestate (alignments[t-ts+startindex], numsenones)))
                        framescorrect += isofsameclass (alignments[t-ts+startindex], uids[t], senone2classmap);  // we only count correct && non-silence state
                }
            }
            logframescorrectedgej = (framescorrect > 0) ? log ((double) framescorrect) : LOGZERO;          // remember for backward pass
            logframescorrectedge[j] = logframescorrectedgej;
        }
        if (boostmmi)
            edgescore -= boostingfactor * exp (logframescorrectedge[j]);
        
#ifdef FORBID_INVALID_SIL_PATHS
        const bool forbidinvalidsilpath = (logalphas.size() > nodes.size());    // we constrain sil to sil path if node space has been blown up
        const bool isaddedsil = forbidinvalidsilpath && (e.unused == 1);        // HACK: 'unused' indicates artificially added sil/sp edge

        // original mode
        if (!isaddedsil)
#endif
        {
            const size_t S = e.S;
            const size_t E = e.E;

            const double inscore = logalphas[S];
            const double pathscore = inscore + edgescore;

            atomicLogAdd (&logalphas[E], pathscore);

            if (returnEframescorrect)
            {
#ifdef DIRECT_MODE
                double loginaccs = logaccalphas[e.S] + edgescore;
                double logpathaccs = logalphas[e.S] + edgescore + logframescorrectedgej;
                logadd (logpathaccs, loginaccs);
                atomicLogAdd (&logaccalphas[e.E], logpathaccs);
#else
                double loginaccs = logaccalphas[S] - logalphas[S];
                logadd (loginaccs, logframescorrectedgej);
                double logpathacc = loginaccs + logalphas[S] + edgescore;
                atomicLogAdd (&logaccalphas[E], logpathacc);
#endif
            }
        }

#ifdef FORBID_INVALID_SIL_PATHS
        // silence edge or second speech edge
        if ((isaddedsil && e.E != nodes.size() -1) || (forbidinvalidsilpath && e.S != 0))
        {
            const size_t S = (size_t) (!isaddedsil ? e.S + nodes.size() : e.S);          // second speech edge comes from special 'silence state' node
            const size_t E = (size_t) (isaddedsil  ? e.E + nodes.size() : e.E);          // silence edge goes into special 'silence state' node
            // remaining lines here are 100% code dup from above, just operating on different (S, E)
            const double inscore = logalphas[S];
            const double pathscore = inscore + edgescore;
            atomicLogAdd (&logalphas[E], pathscore);

            if (returnEframescorrect)
            {
                double loginaccs = logaccalphas[S] - logalphas[S];
                logadd (loginaccs, logframescorrectedgej);
                double logpathacc = loginaccs + logalphas[S] + edgescore;
                atomicLogAdd (&logaccalphas[E], logpathacc);
            }
        }
#endif
    }

    template<typename edgeinforvector, typename nodeinfovector, typename aligninfovector, typename floatvector, typename doublevector>
    static inline __device__ void backwardlatticej (size_t j, const floatvector & edgeacscores, 
                                                    const size_t /*spalignunitid --unused*/, const size_t /*silalignunitid --unused*/, 
                                                    const edgeinforvector & edges, const nodeinfovector & nodes, 
                                                    const aligninfovector & /*aligns -- unused*/, const double totalfwscore, doublevector & logpps, 
                                                    doublevector & logalphas, doublevector & logbetas, float lmf, float wp, 
                                                    float amf, const float boostingfactor, const bool returnEframescorrect,
                                                    doublevector & logframescorrectedge, doublevector & logaccalphas, 
                                                    doublevector & logEframescorrect, doublevector & logaccbetas)
    {
        // output values
        double logpp = LOGZERO;
        double logEframescorrectj = LOGZERO;
        const bool boostmmi = (boostingfactor != 0.0f);

        // edge info
        const edgeinfowithscores & e = edges[j];
        double edgescore = (e.l * lmf + wp + edgeacscores[j]) / amf;
		//zhaorui to deal with the abnormal score for sent start. 
        if (e.l < -200.0f)
            edgescore = (0.0 * lmf + wp + edgeacscores[j]) / amf;
        if (boostmmi)
            edgescore -= boostingfactor * exp (logframescorrectedge[j]);

        // get the frames-correct count for this edge that was computed during the forward pass
        double logframescorrectedgej = (returnEframescorrect || boostmmi) ? logframescorrectedge[j] : LOGZERO;

#ifdef FORBID_INVALID_SIL_PATHS
        // original mode
        const bool forbidinvalidsilpath = (logalphas.size() > nodes.size());    // we prune sil to sil path if alphabetablowup != 1
        const bool isaddedsil = forbidinvalidsilpath && (e.unused == 1);        // HACK: 'unused' indicates artificially added sil/sp edge

        if (!isaddedsil) // original mode
#endif
        {
            const size_t S = e.S;
            const size_t E = e.E;

            // backward pass
            const double inscore = logbetas[E];
            const double pathscore = inscore + edgescore;
            atomicLogAdd (&logbetas[S], pathscore);

            // compute lattice posteriors on the fly since we are at it
            logpp = logalphas[S] + edgescore + logbetas[E] - totalfwscore;

            // similar logic for Eframescorrect
            if (returnEframescorrect)
            {
#ifdef DIRECT_MODE
                double loginaccs = logaccbetas[e.E] + edgescore;
                double logpathaccs = logbetas[e.E] + edgescore + logframescorrectedgej;
                logadd (logpathaccs, loginaccs);
                atomicLogAdd (&logaccbetas[e.S], logpathaccs);

                double logecorrect = logaccalphas[e.S] + edgescore + logbetas[e.E];
                logadd (logecorrect, logalphas[e.S] + edgescore + logframescorrectedgej + logbetas[e.E]);
                logadd (logecorrect, logalphas[e.S] + edgescore + logaccbetas[e.E]);
                logEframescorrectj = logecorrect - totalfwscore;            // be careful, this includes the denominator
#else
                // backward pass
                double loginaccs = logaccbetas[E] - logbetas[E];
                logadd (loginaccs, logframescorrectedgej);
                double logpathacc = loginaccs + logbetas[E] + edgescore;
                atomicLogAdd (&logaccbetas[S], logpathacc);

                // sum up to get final expected frames-correct count per state == per edge (since we assume hard state alignment)
                double logsum = logframescorrectedgej;                      // sum over this edge, left partial (alpha), right partial (beta)

                double edgelogaccalpha = logaccalphas[S] - logalphas[S];    // incoming partial expected frames correct
                logadd (logsum, edgelogaccalpha);

                double edgelogaccbeta = logaccbetas[E] - logbetas[E];       // partial expected frames correct from the end
                logadd (logsum, edgelogaccbeta);

                logEframescorrectj = logsum;                                // that's it
#endif
            }
        }

#ifdef FORBID_INVALID_SIL_PATHS
        double logpp2 = LOGZERO;
        double logEframescorrectj2 = LOGZERO;

        // silence edge or second speech edge
        if ((isaddedsil && e.E != nodes.size() -1) || (forbidinvalidsilpath && e.S != 0))
        {
            const size_t S = (size_t) (!isaddedsil ? e.S + nodes.size() : e.S);          // second speech edge comes from special 'silence state' node
            const size_t E = (size_t) (isaddedsil  ? e.E + nodes.size() : e.E);          // silence edge goes into special 'silence state' node
            // remaining lines here are code dup from above, with two changes: logadd2/logEframescorrectj2 instead of logadd/logEframescorrectj

            // backward pass
            const double inscore = logbetas[E];
            const double pathscore = inscore + edgescore;
            atomicLogAdd (&logbetas[S], pathscore);

            // compute lattice posteriors on the fly since we are at it
            logpp2 = logalphas[S] + edgescore + logbetas[E] - totalfwscore; // second edge (logpp2)

            // similar logic for Eframescorrect
            if (returnEframescorrect)
            {
                // backward pass
                double loginaccs = logaccbetas[E] - logbetas[E];
                logadd (loginaccs, logframescorrectedgej);
                double logpathacc = loginaccs + logbetas[E] + edgescore;
                atomicLogAdd (&logaccbetas[S], logpathacc);

                // sum up to get final expected frames-correct count per state == per edge (since we assume hard state alignment)
                double logsum = logframescorrectedgej;                      // sum over this edge, left partial (alpha), right partial (beta)

                double edgelogaccalpha = logaccalphas[S] - logalphas[S];    // incoming partial expected frames correct
                logadd (logsum, edgelogaccalpha);

                double edgelogaccbeta = logaccbetas[E] - logbetas[E];       // partial expected frames correct from the end
                logadd (logsum, edgelogaccbeta);

                logEframescorrectj2 = logsum;                               // that's it for this second edge
            }

            // sum logpp2 and logEframescorrectj2
            // Eframescorrect must be summed up in a weighted fashion, weighted by PP
            double numer = logEframescorrectj + logpp;
            logadd (numer, logEframescorrectj2 + logpp2);       // weighted sum, weighted by respective (log)pp
            logadd (logpp, logpp2);                             // (log)pp is just the sum of the two posteriors
            double denom = logpp;                               // and that is also the denominator for the weighted sum
            logEframescorrectj = numer - denom;                 // weighted sum
        }
#else
        nodes;
#endif

        // write back return values
        if (logpp > 0.0)            // clip to log 1 (may be possible due to small numeric inaccuracies, although it really shouldn't happen)
            logpp = 0.0;
        logpps[j] = logpp;
        if (returnEframescorrect)
            logEframescorrect[j] = logEframescorrectj;
    }

    template<typename ushortvector, typename uintvector, typename edgeinfowithscoresvector, typename nodeinfovector, typename doublevector, typename matrix>
    static inline __device__ void sMBRerrorsignalj (size_t j, const ushortvector & alignstateids, const uintvector & alignoffsets,
                                                    const edgeinfowithscoresvector & edges, 
                                                    const nodeinfovector & nodes, const doublevector & logpps, const float amf,
                                                    const doublevector & logEframescorrect, const double logEframescorrecttotal,
                                                    matrix & errorsignal, matrix & errorsignalneg)
    {
        size_t ts = nodes[edges[j].S].t;
        size_t te = nodes[edges[j].E].t;
        if (ts != te)
        {
#ifdef DIRECT_MODE
            float logEframescorrectj = logEframescorrect[j];
            size_t offset = alignoffsets[j];
            for (size_t t = ts; t < te; t++)
            {
                const size_t s = (size_t) alignstateids[t - ts + offset];
                atomicLogAdd (&errorsignal(s,t), logEframescorrectj);
            }
#else
            const double diff = expdiff (logEframescorrect[j], logEframescorrecttotal);
            // Note: the contribution of the states of an edge to their senones is the same for all states
            // so we compute it once and add it to all; this will not be the case without hard alignments.
#if 0       // linear mode
            const float pp = expf ((float)logpps[j]);      // edge posterior
            const float edgecorrect = (pp * diff) / amf;
            size_t offset = alignoffsets[j];
            for (size_t t = ts; t < te; t++)
            {
                const size_t s = (size_t) alignstateids[t - ts + offset];
                atomicAdd (&errorsignal(s,t), edgecorrect);
            }
            errorsignalneg(0,0) = 0;   // to reference it because we hope to support log mode as well
#else       // log mode
            double absdiff = fabs (diff);
            if (absdiff == 0.0f)
                return;
            const float logedgecorrect = (float) (logpps[j] + log (absdiff));
            size_t offset = alignoffsets[j];
            for (size_t t = ts; t < te; t++)
            {
                const size_t s = (size_t) alignstateids[t - ts + offset];
                if (diff > 0.0)
                    atomicLogAdd(&errorsignal(s,t), logedgecorrect);
                else
                    atomicLogAdd(&errorsignalneg(s,t), logedgecorrect);
            }
            absdiff = amf;        // just to reference it because we hope to support linear mode as well
#endif
#endif
        }
    }

    // accumulate a per-edge quantity into the states that the edge is aligned with
    // Use this for MMI passing the edge posteriors logpps[] as logq, or for sMBR passing logEframescorrect[].
    // j=edge index, alignment in (alignstateids, alignoffsets)
    template<typename ushortvector, typename uintvector, typename edgeinfowithscoresvector, typename nodeinfovector,typename doublevector, typename matrix>
    static inline __device__ void stateposteriorsj (size_t j, const ushortvector & alignstateids, const uintvector & alignoffsets,
                                                    const edgeinfowithscoresvector & edges,
                                                    const nodeinfovector & nodes, const doublevector & logqs /*quantity to accumulate*/,
                                                    matrix & logacc /*accumulator to accumulate into*/)
    {
        size_t ts = nodes[edges[j].S].t;
        size_t te = nodes[edges[j].E].t;
        if (ts != te)
        {
            const float logq = (float) logqs[j];                // per-edge quantity to accumulate, e.g. edge posteriors -> state posteriors
            size_t offset = alignoffsets[j];
            for (size_t t = ts; t < te; t++)
            {
                const size_t s = (size_t) alignstateids[t - ts + offset];   // get state for this (j,t)
                atomicLogAdd (&logacc(s,t), logq);
            }
       }
    }
};

};};

#pragma pop_macro ("atomicCAS")
#pragma pop_macro ("atomicAdd")
#pragma pop_macro ("__device__")
