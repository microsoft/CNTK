// cudamatrix.cu(.h) -- CUDA kernels for lattice ops. Consider this a .cu/.cpp file.
//
// F. Seide, Aug 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudalatticeops.cu.h $
// 
// 96    4/08/13 8:43p V-hansu
// (fix some comments)
// 
// 95    1/15/13 2:58p V-hansu
// remove SUM_AND_PRODUCT
// 
// 94    1/11/13 6:49p V-hansu
// add directerrorcomputationi() and code for DIRECT_MODE, not activated
// 
// 93    1/10/13 9:51p V-hansu
// (rename an argument in expdiff())
// 
// 92    1/09/13 10:05p V-hansu
// rename variable in expfi(), add dotprodi(), and add some unactivated
// code in sMBRerrorsignal()
// 
// 91    11/21/12 8:58p V-hansu
// rename state2classmap to senone2classmap
// 
// 90    11/21/12 7:10p V-hansu
// rename statetoclassmap to state2classmap
// 
// 89    11/21/12 6:04p V-hansu
// add statetoclassmap to forwardbackward() to get prepared for mpe
// approximation
// 
// 88    11/10/12 3:28p V-hansu
// modify the strategy of set unseen states
// 
// 87    11/06/12 8:44p V-hansu
// use setunseeni() to separate unseen states and seen states that have
// very low log pp
// 
// 86    11/05/12 12:26a V-hansu
// undo last check in :( not got enough time to get it done
// 
// 85    11/02/12 9:44a V-hansu
// add logframescorrect to edgealignment(), prepare for moving computation
// from forwardbackward() to edgealignment()
// 
// 84    10/29/12 3:36p V-hansu
// add boosting factor to prepare for BMMI
// 
// 83    10/22/12 10:28p V-hansu
// fix the assign accalphas LOGZERO bug for mmi mode in
// forwardbackwardlattice()
// 
// 82    10/21/12 1:41p V-hansu
// change the interface of edgealignmentj
// 
// 81    10/19/12 7:51p V-hansu
// adjust the order of interface arguments, finish some TODO
// 
// 80    10/19/12 3:55p V-hansu
// finish the mmierrorsignal, not tested yet
// 
// 79    10/18/12 9:31p V-hansu
// undef SUM_AND_PRODUCT
// 
// 78    10/18/12 9:50a Fseide
// added a missing namespace to expdiff() call
// 
// 77    10/18/12 9:29a Fseide
// added a TODO
// 
// 76    10/18/12 9:28a Fseide
// added TODOs to vectordiff
// 
// 75    10/18/12 8:30a Fseide
// removed duplicate expdiff() function, instead created a templated
// version in latticefunctionskernels.h
// 
// 74    10/17/12 8:39p Fseide
// computesMBRerrorsignals() now uses the kernel-kernel
// 
// 73    10/17/12 8:30p Fseide
// sMBRerrorsignal() tidied up
// 
// 72    10/17/12 8:21p Fseide
// removed matri setvaluej() because it should be called setvaluei() and
// that function already exists :)
// 
// 71    10/17/12 8:08p Fseide
// renamed a variable
// 
// 70    10/17/12 8:05p Fseide
// commented out weird-looking code (expdiff())
// 
// 69    10/17/12 8:02p Fseide
// renamed sMBRerrorsignal() to computesMBRerrorsignal() since CUDA
// kernels don't allow overloading it seems
// 
// 68    10/17/12 7:58p V-hansu
// add a froce data cast from double to float
// 
// 67    10/17/12 7:57p Fseide
// renamed weightedcomponentexpproducts() to sMBRerrorsignal()
// 
// 66    10/17/12 7:52p Fseide
// (and fixed a message)
// 
// 65    10/17/12 7:51p Fseide
// added a missing checklaunch() call
// 
// 64    10/17/12 7:31p V-hansu
// activate sum and product, rename weightedcomponentexpproducts
// 
// 63    10/17/12 6:52p V-hansu
// change weighteddotproducti to weightedcomponentexpproduct
// 
// 62    10/17/12 6:16p V-hansu
// change back to product and sum mode
// 
// 61    10/17/12 5:59p V-hansu
// rename Eframescorrectotal to logEframescorrectotal, rename
// Eframescorrectbuf or Eframescorrectdiff to logEframescorrect
// 
// 60    10/17/12 5:02p V-hansu
// rename weighteddotproductj to weighteddotproducti 
// 
// 59    10/17/12 4:18p V-hansu
// add error initialization to sMBRerrorsignal now
// 
// 58    10/17/12 3:34p V-hansu
// turn eframecorrect (eframecorrectdiff) from float vector to double
// get the sMBRerrorsignal function behave like formula does
// 
// 57    10/17/12 2:03p Fseide
// new method stateposteriors()
// 
// 56    10/15/12 5:25p V-hansu
// add aligns to forwardbackwardlattice()
// 
// 55    10/15/12 5:21p V-hansu
// add aligns to backwardlatticej, change the location of #define
// SILENCE_PRUNING
// 
// 54    10/14/12 10:01p V-hansu
// add spalignunitid, silalignunitid and nodes to backwardlatticej and
// forwardlatticej
// 
// 53    10/14/12 8:26p V-hansu
// add silunitid and spunitid to forwardbackwardlattice
// 
// 52    10/12/12 1:13a V-hansu
// activate shuffle in forwardbackwardlattice and errorsignal
// 
// 51    9/30/12 5:21p V-hansu
// add backptr to edgealignment related functions
// 
// 50    9/28/12 6:09p V-hansu
// incorporate setting LOGZERO in errorsignal function
// 
// 49    9/28/12 4:17p V-hansu
// rename errorsignalbuf to errorsignalneg, activate log mode error
// accumulation, refactor atomicCASfloatdouble
// 
// 48    9/28/12 2:02p V-hansu
// rename gammas to errorsignal, change sptransPindex to alignunit,
// activate linear mode in errorsignal
// 
// 47    9/27/12 11:28p V-hansu
// change the shuffle method
// 
// 46    9/27/12 12:27a V-hansu
// turn on SHUFFLE_FORWARD and add errorcomputationj
// 
// 45    9/26/12 7:35p V-hansu
// add siltransPindex, add vectordiffj
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
// 41    9/26/12 12:57p Fseide
// renamed errorsignalj() to sMBRerrorsignalj()
// 
// 40    9/26/12 12:53p Fseide
// errorsignal() renamed to sMBRerrorsignal()
// 
// 39    9/26/12 12:27p Fseide
// renamed logdengammaspos/neg to dengammas/dengammasbuf
// 
// 38    9/26/12 12:08p Fseide
// some source-code grouping, removed old dummy functions, minor cleanup
// in latticeforwardbackward()
// 
// 37    9/26/12 11:57a Fseide
// sMBRerrorsignal() now takes two dengammas accumulators, in prep for pos/neg
// logadd
// 
// 36    9/25/12 11:48p V-hansu
// add computation of Eframecorrecttotal in returnEframescorrect of
// forwardbackwardlattice
// 
// 35    9/25/12 11:29p V-hansu
// add LOGZERO to cuda::lattice namespace, change interface of setvalue to
// make it more general, add checking for framecorrect in
// forwardbackwardlattice
// 
// 34    9/25/12 5:31p V-hansu
// remove totalfwacc in backwardlatticej
// 
// 33    9/25/12 3:11p V-hansu
// add sizetvector and change uids from uintvector into sizetvector
// 
// 32    9/25/12 1:12p V-hansu
// add alignemts and alignmentoffsets to forwardlattce related function to
// finish the algorithm
// 
// 31    9/24/12 10:07p V-hansu
// change the interface relating to forwardbackwardlattice to get prepared
// for the combined mode fwbw, not finished yet
// 
// 30    9/24/12 2:42p V-hansu
// change the interface of latticefunctionskernels::forwardlatticej and
// modify SHUFFLE back to SHUFFLE_FORWARD, since it is not necessary for
// backward
// 
// 29    9/24/12 1:21a V-hansu
// change SHUFFLE_FORWARD to SHUFFLE and add it to backward as well
// 
// 28    9/24/12 1:07a V-hansu
// add auto_timer and #define SHUFFLE_FORWARD in forwardbackwardlattice
// and forwardj repectively
// 
// 27    9/23/12 4:34p V-hansu
// add timing to forwardbackwardlattice
// 
// 26    9/21/12 10:09p V-hansu
// add the initialization of logalphas and logbetas in
// parallelforwardbackwardlattice
// 
// 25    9/21/12 4:49p V-hansu
// fix a bug in backward pass, shall copy the first element of logbetas
// 
// 24    9/21/12 3:55p V-hansu
// change the interface of latticefunctionsops::forwardbackwardlattice to
// pass in batchsizeforward and batchsizebackward by pointer
// 
// 23    9/21/12 3:25p V-hansu
// change batchsizeforward and batchsizebackward into cpu side vector
// 
// 22    9/21/12 1:44p V-hansu
// modify latticefunctionsops::forwardbackwardlattice, remove a bug
// 
// 21    9/19/12 9:33a Fseide
// renamed edgeinfo to edgeinfowithscores, in prep for V2 lattice format
// 
// 20    9/17/12 8:06p V-hansu
// change float into double for logalphas and logbetas in forwardbackward
// 
// 19    9/16/12 9:31p V-hansu
// add atomicLogAdd, not finished. add doublevector
// 
// 18    9/16/12 5:23p V-hansu
// finish forwardbackwardlattice function, not tested
// 
// 17    9/14/12 9:26p V-hansu
// add backwardlatticej and finished
// 
// 16    9/14/12 5:55p V-hansu
// add backwardlatticej
// 
// 15    9/14/12 2:37p V-hansu
// add forwardlatticej and forwardbackwardlattice in
// latticefunctionskernels and related classes
// 
// 14    9/14/12 1:27p V-hansu
// add fowardlatticej, not tested
// 
// 13    9/06/12 7:24p V-hansu
// add alignoffsets into interface, same as alignstateids
// 
// 12    9/06/12 7:15p V-hansu
// add alignoffsets into interface of sMBRerrorsignal
// 
// 11    9/05/12 10:36p V-hansu
// add function sMBRerrorsignal and codes relating to it
// 
// 10    9/05/12 7:53a Fseide
// added a comment
// 
// 9     9/04/12 10:25p V-hansu
// change the interface of edgealignment
// 
// 8     9/04/12 5:41p Fseide
// added thread scheduling
// 
// 7     9/04/12 5:09p Fseide
// (added test code for textures, inactive)
// 
// 6     9/03/12 4:54p V-hansu
// change the interface of edgealignmentj and edgealignment
// 
// 5     9/03/12 4:06p V-hansu
// modify the calling of edgealignmentj
// 
// 4     9/01/12 3:01p Fseide
// lattice kernel "implemented" through a function in a new header (but as
// a hull only so far)
// 
// 3     9/01/12 2:10p Fseide
// removed dependency on cudalattice.h, as the DLL-related stuff tripped
// up the CUDA compiler (and rightly so)
// 
// 2     8/31/12 10:58p V-hansu
// add a fake latticefunctionsops::edgealignment, not finished
// 
// 1     8/28/12 6:27p Fseide
// created

#if 0   // [v-hansu] set aside codes with log
#endif

#undef DIRECT_MODE          // [v-hansu] use the direct formula for smbr mode, proven makes no difference

#include <cuda.h>
#include "cudalib.h"
#include "cudabasetypes.h"
#include "latticestorage.h"
#include "latticefunctionskernels.h"
#include "cudalatticeops.h"
#include "math.h"
#include <assert.h>
#include <stdexcept>
#include <windows.h>    // for timer

namespace msra { namespace cuda {

    // auto_timer timer; run(); double seconds = timer; // now can abandon the object
    class auto_timer
    {
        LARGE_INTEGER freq, start;
        auto_timer (const auto_timer &); void operator= (const auto_timer &);
    public:
        auto_timer()
        {
            if (!QueryPerformanceFrequency (&freq)) // count ticks per second
                throw std::runtime_error ("auto_timer: QueryPerformanceFrequency failure");
            QueryPerformanceCounter (&start);
        }
        operator double() const     // each read gives time elapsed since start, in seconds
        {
            LARGE_INTEGER end;
            QueryPerformanceCounter (&end);
            return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
        }
        void show (const std::string & msg) const
        {
            double elapsed = *this;
            fprintf (stderr, "%s: %.6f ms\n", msg.c_str(), elapsed * 1000.0/*to ms*/);
        }
    };

    // -----------------------------------------------------------------------
    // edgealignment --do alignment on a per edge level, only support normal left to right hmms and ergodic silence hmm
    // output alignresult
    // -----------------------------------------------------------------------

    __global__ void edgealignmentj (const vectorref<lrhmmdef> hmms, const vectorref<lr3transP> transPs, const size_t spalignunitid, const size_t silalignunitid,
                                    const matrixref<float> logLLs, const vectorref<msra::lattices::nodeinfo> nodes, const vectorref<msra::lattices::edgeinfowithscores> edges,
                                    const vectorref<msra::lattices::aligninfo> aligns, const vectorref<unsigned int> alignoffsets, 
                                    vectorref<unsigned short> backptrstorage, const vectorref<size_t> backptroffsets, 
                                    vectorref<unsigned short> alignresult, vectorref<float> edgeacscores)       // output
    {
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        const size_t j = jinblock + blockIdx.x * tpb;
        if (j < edges.size())       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::edgealignmentj (j, hmms, transPs, spalignunitid, silalignunitid, logLLs, nodes, edges, aligns, alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
        }
    }

    void latticefunctionsops::edgealignment (const vectorref<lrhmmdef> & hmms, const vectorref<lr3transP> & transPs, const size_t spalignunitid,
                                             const size_t silalignunitid, const matrixref<float> & logLLs, 
                                             const vectorref<msra::lattices::nodeinfo> & nodes, const vectorref<msra::lattices::edgeinfowithscores> & edges, 
                                             const vectorref<msra::lattices::aligninfo> & aligns, const vectorref<unsigned int> & alignoffsets, 
                                             vectorref<unsigned short> & backptrstorage, const vectorref<size_t> & backptroffsets,
                                             vectorref<unsigned short> & alignresult, vectorref<float> & edgeacscores) const        // output
    {
        // Layout: each thread block takes 1024 threads; and we have #edges/1024 blocks.
        // This limits us to 16 million edges. If you need more, please adjust to either use wider thread blocks or a second dimension for the grid. Don't forget to adjust the kernel as well.
        const size_t numedges = edges.size();
        dim3 t (32,8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));
        //cudaarrayref<float> logLLsarray;        // TODO: pass this in, of course
        //passtextureref texref (logLLstex, logLLsarray);    // use the same name as that global texref one, so it will match the name inside the kernel
        edgealignmentj << <b, t, 0, /*GetCurrentStream()*/ cudaStreamDefault >> > (hmms, transPs, spalignunitid, silalignunitid, logLLs, nodes, edges, aligns, alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
        checklaunch ("edgealignment");
    }

    // setvalue --helper to initialize an array to a constant value, e.g. LOGZERO
    __global__ void setvaluej (vectorref<double> arraytoset, double value, size_t nelem)
    {
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        const size_t j = jinblock + blockIdx.x * tpb;
        if (j < nelem)
        {
            msra::lattices::latticefunctionskernels::setvaluej(j, arraytoset, value);
        }
    }

    __global__ void expfi (matrixref<float> mata)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i < mata.rows())
        {
            const size_t m = mata.cols();
            for (size_t j = 0; j < m; j++)
                mata(i,j) = expf (mata(i,j));
        }
    }

    __global__ void dotprodi (matrixref<float> mata, matrixref<float> matb)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i < mata.rows())
        {
            const size_t m = mata.cols();
            for (size_t j = 0; j < m; j++)
                mata(i,j) = mata(i,j) * matb(i,j);
        }
    }

    __global__ void setunseeni (matrixref<float> errorsignal, matrixref<float> errorsignalauxbuf)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i < errorsignal.rows())
        {
            const size_t m = errorsignal.cols();
            for (size_t j = 0; j < m; j++)
                if (errorsignal(i,j) == logf(CUDART_MIN_DENORM_F) && errorsignalauxbuf(i,j) == logf(CUDART_MIN_DENORM_F))
                    errorsignalauxbuf(i,j) = LOGZERO;
        }
    }

    // errorsignal(i,j) = (exp(errorsignal(i,j)) - exp(errorsignal(i,j))) / amf
    __global__ void errorcomputationi (matrixref<float> errorsignal, matrixref<float> errorsignalauxbuf, float amf)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i < errorsignal.rows())
        {
            const size_t m = errorsignal.cols();
            for (size_t j = 0; j < m; j++)
                errorsignal(i,j) = msra::lattices::latticefunctionskernels::expdiff (errorsignal(i,j), errorsignalauxbuf(i,j)) / amf;
        }
    }

    // exp(errorsignal(i,j)) - exp(logEframescorrecttotal+errorsignalauxbuf(i,j))/amf
    __global__ void directerrorcomputationi (matrixref<float> errorsignal, matrixref<float> errorsignalauxbuf, float logEframescorrecttotal,float amf)
    {
        const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
        if (i < errorsignal.rows())
        {
            const size_t m = errorsignal.cols();
            for (size_t j = 0; j < m; j++)
                errorsignal(i,j) = msra::lattices::latticefunctionskernels::expdiff (errorsignal(i,j), logEframescorrecttotal + errorsignalauxbuf(i,j)) / amf;
        }
    }

    // compute the final error signal from gammas and state-consolidated Eframescorrect
    // in-place operation is supported (i.e. output = one of the inputs) 
    __global__ void computesMBRerrorsignals (const matrixref<float> loggammas, const matrixref<float> logEframescorrect, 
                                             const double logEframecorrecttotal, const float kappa, matrixref<float> errorsignal)
    {
        const size_t s = threadIdx.x + (blockIdx.x * blockDim.x);
        if (s < loggammas.rows())
            msra::lattices::latticefunctionskernels::computesMBRerrorsignals (s, loggammas, logEframescorrect, logEframecorrecttotal, kappa, errorsignal);
    }

    __global__ void forwardlatticej (const size_t batchsize, const size_t startindex, const vectorref<float> edgeacscores, 
                                     const size_t spalignunitid, const size_t silalignunitid,
                                     vectorref<msra::lattices::edgeinfowithscores> edges, vectorref<msra::lattices::nodeinfo> nodes, 
                                     const vectorref<msra::lattices::aligninfo> aligns, vectorref<unsigned short> alignments, 
                                     vectorref<unsigned int> alignmentoffsets, vectorref<double> logalphas, float lmf, float wp, float amf, 
                                     const float boostingfactor, const vectorref<unsigned short> uids, const vectorref<unsigned short> senone2classmap, 
                                     const bool returnEframescorrect, vectorref<double> logframescorrectedge, vectorref<double> logaccalphas)
    {
        const size_t shufflemode = 1;       // [v-hansu] this gives us about 100% speed up than shufflemode = 0 (no shuffle)
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < batchsize)       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::forwardlatticej (j + startindex, edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns, alignments, alignmentoffsets, 
                                                                      logalphas, lmf, wp, amf, boostingfactor, uids, senone2classmap, returnEframescorrect, logframescorrectedge, logaccalphas);
        }
    }

    __global__ void backwardlatticej (const size_t batchsize, const size_t startindex, const vectorref<float> edgeacscores, 
                                      const size_t spalignunitid, const size_t silalignunitid,                              
                                      vectorref<msra::lattices::edgeinfowithscores> edges, vectorref<msra::lattices::nodeinfo> nodes,
                                      vectorref<msra::lattices::aligninfo> aligns, const double totalfwscore,
                                      vectorref<double> logpps, vectorref<double> logalphas, vectorref<double> logbetas, 
                                      float lmf, float wp, float amf, const float boostingfactor, const bool returnEframescorrect, 
                                      vectorref<double> logframescorrectedge, vectorref<double> logaccalphas, 
                                      vectorref<double> logEframescorrect, vectorref<double> logaccbetas)
    {
        const size_t tpb = blockDim.x * blockDim.y;       // total #threads in a block
        const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
        size_t j = jinblock + blockIdx.x * tpb;
        if (j < batchsize)       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::backwardlatticej (j + startindex, edgeacscores, spalignunitid, silalignunitid, 
                                                                       edges, nodes, aligns, totalfwscore, logpps, logalphas, logbetas, 
                                                                       lmf, wp, amf, boostingfactor, returnEframescorrect,
                                                                       logframescorrectedge, logaccalphas, 
                                                                       logEframescorrect, logaccbetas);
        }
    }

    void latticefunctionsops::forwardbackwardlattice (const size_t * batchsizeforward, const size_t * batchsizebackward, 
                                                      const size_t numlaunchforward, const size_t numlaunchbackward,
                                                      const size_t spalignunitid, const size_t silalignunitid,
                                                      const vectorref<float> & edgeacscores,
                                                      const vectorref<msra::lattices::edgeinfowithscores> & edges, 
                                                      const vectorref<msra::lattices::nodeinfo> & nodes, 
                                                      const vectorref<msra::lattices::aligninfo> & aligns, 
                                                      const vectorref<unsigned short> & alignments,
                                                      const vectorref<unsigned int> & aligmentoffsets,
                                                      vectorref<double> & logpps, vectorref<double> & logalphas, vectorref<double> & logbetas,
                                                      const float lmf, const float wp, const float amf, const float boostingfactor, 
                                                      const bool returnEframescorrect, const vectorref<unsigned short> & uids, 
                                                      const vectorref<unsigned short> & senone2classmap, vectorref<double> & logaccalphas, 
                                                      vectorref<double> & logaccbetas, vectorref<double> & logframescorrectedge,
                                                      vectorref<double> & logEframescorrect, vectorref<double> & Eframescorrectbuf, 
                                                      double & logEframescorrecttotal, double & totalfwscore) const
    {
        // initialize log{,acc}(alhas/betas)
        dim3 t (32, 8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((logalphas.size() + tpb - 1) / tpb));

        size_t alphabetablowup = logalphas.size() / nodes.size();
        // TODO: is this really efficient? One thread per value?
        setvaluej<<<b, t, 0, GetCurrentStream()>>> (logalphas, LOGZERO, logalphas.size());
        checklaunch ("setvaluej");
        setvaluej<<<b, t, 0, GetCurrentStream()>>> (logbetas, LOGZERO, logalphas.size());
        checklaunch ("setvaluej");
        if (returnEframescorrect)
        {
            setvaluej<<<b, t, 0, GetCurrentStream()>>> (logaccalphas, LOGZERO, logalphas.size());
            checklaunch ("setvaluej");
            setvaluej<<<b, t, 0, GetCurrentStream()>>> (logaccbetas, LOGZERO, logalphas.size());
            checklaunch ("setvaluej");
        }
        // set initial tokens to probability 1 (0 in log)
        double log1 = 0.0;
        memcpy (logalphas.get(), 0, &log1, 1);
        memcpy (logbetas.get(), nodes.size()-1, &log1, 1);

        // forward pass
        size_t startindex = 0;
        for (size_t i = 0; i < numlaunchforward; i++)
        {
            dim3 b ((unsigned int) ((batchsizeforward[i] + tpb - 1) / tpb));
            forwardlatticej <<<b, t, 0, GetCurrentStream()>>> (batchsizeforward[i], startindex, edgeacscores, 
                                                               spalignunitid, silalignunitid, edges, nodes, aligns,
                                                               alignments, aligmentoffsets, logalphas, lmf, wp, amf, 
                                                               boostingfactor, uids, senone2classmap, returnEframescorrect, 
                                                               logframescorrectedge, logaccalphas);
            checklaunch ("edgealignment");
            startindex += batchsizeforward[i];
        }
        memcpy<double>(&totalfwscore, logalphas.get(), nodes.size() - 1, 1);
        double totalfwacc = 0;
        if(returnEframescorrect)
        {
            memcpy<double>(&totalfwacc, logaccalphas.get(), nodes.size() - 1, 1);
            totalfwacc -= totalfwscore;
        }

        // backward pass
        startindex = edges.size();
        for (size_t i = 0; i < numlaunchbackward; i++)
        {
            dim3 b ((unsigned int) ((batchsizebackward[i] + tpb - 1) / tpb));
            backwardlatticej <<<b, t, 0, GetCurrentStream()>>> (batchsizebackward[i], startindex - batchsizebackward[i], 
                                                                edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns,
                                                                totalfwscore, logpps, logalphas, logbetas, 
                                                                lmf, wp, amf, boostingfactor, returnEframescorrect, logframescorrectedge, 
                                                                logaccalphas, logEframescorrect, logaccbetas);
            checklaunch ("edgealignment");
            startindex -= batchsizebackward[i];
        }
        double totalbwscore = 0;
        memcpy<double>(&totalbwscore, logbetas.get(), 0, 1);
        double totalbwacc = 0;
        if (returnEframescorrect)
        {
            memcpy<double>(&totalbwacc, logaccbetas.get(), 0, 1);
            totalbwacc -= totalbwscore;
            logEframescorrecttotal = totalbwacc;
        }

#if 1   // check for matching
        double difffwbwscore = totalfwscore - totalbwscore;
        double absdifffwbwscore = difffwbwscore > 0 ? difffwbwscore : 0 - difffwbwscore;

        if (absdifffwbwscore / nodes.size() > 1e-4)
            fprintf (stderr, "forwardbackward: WARNING: lattice fw and bw scores %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwscore, (float) totalbwscore, (int) nodes.size(), (int) edges.size());

        if (returnEframescorrect)
        {
            double difffwbwacc = totalfwacc- totalbwacc;
            double absdifffwbwacc = difffwbwacc > 0 ? difffwbwacc : 0 - difffwbwacc;

            if (absdifffwbwacc / nodes.size() > 1e-4)
                fprintf (stderr, "forwardbackward: WARNING: lattice fw and bw acc %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwacc, (float) totalbwacc, (int) nodes.size(), (int) edges.size());
        }
#endif
    }

    // -----------------------------------------------------------------------
    // sMBRerrorsignal -- accumulate difference of logEframescorrect and logEframescorrecttotal into errorsignal
    // -----------------------------------------------------------------------

    __global__ void sMBRerrorsignalj (const vectorref<unsigned short> alignstateids, const vectorref<unsigned int> alignoffsets,
                                      const vectorref<msra::lattices::edgeinfowithscores> edges, const vectorref<msra::lattices::nodeinfo> nodes, 
                                      vectorref<double> logpps, const float amf, const vectorref<double> logEframescorrect, const double logEframescorrecttotal, 
                                      matrixref<float> errorsignal, matrixref<float> errorsignalneg)
    {
        const size_t shufflemode = 1;  // [v-hansu] this gives us about 100% speed up than shufflemode = 0 (no shuffle)
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < edges.size())       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::sMBRerrorsignalj (j, alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg);
        }
    }

    // -----------------------------------------------------------------------
    // stateposteriors --accumulate a per-edge quantity into the states that the edge is aligned with
    // -----------------------------------------------------------------------

    __global__ void stateposteriorsj (const vectorref<unsigned short> alignstateids, const vectorref<unsigned int> alignoffsets,
                                      const vectorref<msra::lattices::edgeinfowithscores> edges, const vectorref<msra::lattices::nodeinfo> nodes, 
                                      const vectorref<double> logqs, matrixref<float> logacc)
    {
        const size_t shufflemode = 1;  // [v-hansu] this gives us about 100% speed up than shufflemode = 0 (no shuffle)
        const size_t j = msra::lattices::latticefunctionskernels::shuffle (threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
        if (j < edges.size())       // note: will cause issues if we ever use __synctreads()
        {
            msra::lattices::latticefunctionskernels::stateposteriorsj (j, alignstateids, alignoffsets, edges, nodes, logqs, logacc);
        }
    }

    void latticefunctionsops::stateposteriors (const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
                                               const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes, 
                                               const vectorref<double> & logqs, matrixref<float> & logacc) const
    {
        // Layout: each thread block takes 1024 threads; and we have #edges/1024 blocks.
        // This limits us to 16 million edges. If you need more, please adjust to either use wider thread blocks or a second dimension for the grid. Don't forget to adjust the kernel as well.
        const size_t numedges = edges.size();
        dim3 t (32,8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));

        setvaluei <<<dim3((((unsigned int) logacc.rows())+31)/32), 32, 0, GetCurrentStream()>>> (logacc, LOGZERO);
        checklaunch ("setvaluei");

        stateposteriorsj <<<b, t, 0, GetCurrentStream()>>> (alignstateids, alignoffsets, edges, nodes, logqs, logacc);
        checklaunch ("stateposteriors");
    }

    void latticefunctionsops::sMBRerrorsignal (const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
                                               const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes, 
                                               const vectorref<double> & logpps, const float amf, const vectorref<double> & logEframescorrect, const double logEframescorrecttotal, 
                                               matrixref<float> & errorsignal, matrixref<float> & errorsignalauxbuf) const
    {
        // Layout: each thread block takes 1024 threads; and we have #edges/1024 blocks.
        // This limits us to 16 million edges. If you need more, please adjust to either use wider thread blocks or a second dimension for the grid. Don't forget to adjust the kernel as well.
        const size_t numedges = edges.size();
        dim3 t (32,8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));

#ifdef DIRECT_MODE  // compute Eframescorrect in a more direct way, proven to get same result as below
        setvaluei <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal, LOGZERO);
        checklaunch ("setvaluei");
        
        sMBRerrorsignalj <<<b, t, 0, GetCurrentStream()>>> (alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalauxbuf);
        checklaunch ("sMBRerrorsignal");        // now we get state based logEframescorrect

        matrixref<float> & loggammas = errorsignalauxbuf;
        setvaluei <<<dim3((((unsigned int) errorsignalauxbuf.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignalauxbuf, LOGZERO);
        checklaunch ("setvaluei");

        stateposteriorsj <<<b, t, 0, GetCurrentStream()>>> (alignstateids, alignoffsets, edges, nodes, logpps, loggammas);
        checklaunch ("stateposteriorsj");       // now we get state based loggammas

        directerrorcomputationi <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal, errorsignalauxbuf, logEframescorrecttotal, amf);
        checklaunch ("errorcomputationj");
#else   // this saves some computation compared with DIRECT_MODE
#if 0   // linear mode, i.e. accumulated error directly
        setvaluei <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal, 0);
        checklaunch ("setvaluei");
        setvaluei <<<dim3((((unsigned int) errorsignalauxbuf.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignalauxbuf, 0);
        checklaunch ("setvaluei");
#else   // log mode, for numerical safety
        setvaluei <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal, LOGZERO);
        checklaunch ("setvaluei");
        setvaluei <<<dim3((((unsigned int) errorsignalauxbuf.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignalauxbuf, LOGZERO);
        checklaunch ("setvaluei");
#endif
        sMBRerrorsignalj <<<b, t, 0, GetCurrentStream()>>> (alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalauxbuf);
        checklaunch ("sMBRerrorsignal");

        setunseeni <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal, errorsignalauxbuf);
        checklaunch ("setunseenj");

        errorcomputationi <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal, errorsignalauxbuf, amf);
        checklaunch ("errorcomputationj");

#endif
    }

    void latticefunctionsops::mmierrorsignal (const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
                                              const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes, 
                                              const vectorref<double> & logpps, matrixref<float> & errorsignal) const
    {
        const size_t numedges = edges.size();
        dim3 t (32,8);
        const size_t tpb = t.x * t.y;
        dim3 b ((unsigned int) ((numedges + tpb - 1) / tpb));

        matrixref<float> & loggammas = errorsignal;                     // remember--this is an alias to 'errorsignal'
        setvaluei <<<dim3((((unsigned int) loggammas.rows())+31)/32), 32, 0, GetCurrentStream()>>> (loggammas, LOGZERO);
        checklaunch ("setvaluei");
        stateposteriorsj <<<b, t, 0, GetCurrentStream()>>> (alignstateids, alignoffsets, edges, nodes, logpps, loggammas);
        checklaunch ("stateposteriorsj");

        expfi <<<dim3((((unsigned int) errorsignal.rows())+31)/32), 32, 0, GetCurrentStream()>>> (errorsignal);
        checklaunch ("expfi");
    }


};};
