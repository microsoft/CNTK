// cudamatrix.cu(.h) -- CUDA kernels for lattice ops. Consider this a .cu/.cpp file.
//
// F. Seide, V-hansu

#undef DIRECT_MODE // [v-hansu] use the direct formula for smbr mode, proven makes no difference

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cudalib.h"
#include "cudabasetypes.h"
#include "latticestorage.h"
#include "latticefunctionskernels.h"
#include "cudalatticeops.h"
#include "math.h"
#include <assert.h>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include "Windows.h" // for timer
#endif

#if __unix__
#include <sys/time.h>
#endif

namespace msra { namespace cuda {

cudaStream_t GetCurrentStream();

// auto_timer timer; run(); double seconds = timer; // now can abandon the object
#ifdef __unix__
typedef timeval LARGE_INTEGER;
#endif
class auto_timer
{
    LARGE_INTEGER freq, start;
    auto_timer(const auto_timer &);
    void operator=(const auto_timer &);

public:
    auto_timer()
    {
#ifdef _WIN32
        if (!QueryPerformanceFrequency(&freq)) // count ticks per second
            RuntimeError("auto_timer: QueryPerformanceFrequency failure");
        QueryPerformanceCounter(&start);
#endif
#ifdef __unix__
        gettimeofday(&start, NULL);
#endif
    }
    operator double() const // each read gives time elapsed since start, in seconds
    {
        LARGE_INTEGER end;
#ifdef _WIN32
        QueryPerformanceCounter(&end);
        return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
#endif
#ifdef __unix__
        gettimeofday(&end, NULL);
        return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / (1000 * 1000);
#endif
    }
    void show(const std::string &msg) const
    {
        double elapsed = *this;
        fprintf(stderr, "%s: %.6f ms\n", msg.c_str(), elapsed * 1000.0 /*to ms*/);
    }
};

// -----------------------------------------------------------------------
// edgealignment --do alignment on a per edge level, only support normal left to right hmms and ergodic silence hmm
// output alignresult
// -----------------------------------------------------------------------

__global__ void edgealignmentj(const vectorref<lrhmmdef> hmms, const vectorref<lr3transP> transPs, const size_t spalignunitid, const size_t silalignunitid,
                               const matrixref<float> logLLs, const vectorref<msra::lattices::nodeinfo> nodes, const vectorref<msra::lattices::edgeinfowithscores> edges,
                               const vectorref<msra::lattices::aligninfo> aligns, const vectorref<unsigned int> alignoffsets,
                               vectorref<unsigned short> backptrstorage, const vectorref<size_t> backptroffsets,
                               vectorref<unsigned short> alignresult, vectorref<float> edgeacscores) // output
{
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t j = jinblock + blockIdx.x * tpb;
    if (j < edges.size()) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::edgealignmentj(j, hmms, transPs, spalignunitid, silalignunitid, logLLs, nodes, edges, aligns, alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
    }
}

void latticefunctionsops::edgealignment(const vectorref<lrhmmdef> &hmms, const vectorref<lr3transP> &transPs, const size_t spalignunitid,
                                        const size_t silalignunitid, const matrixref<float> &logLLs,
                                        const vectorref<msra::lattices::nodeinfo> &nodes, const vectorref<msra::lattices::edgeinfowithscores> &edges,
                                        const vectorref<msra::lattices::aligninfo> &aligns, const vectorref<unsigned int> &alignoffsets,
                                        vectorref<unsigned short> &backptrstorage, const vectorref<size_t> &backptroffsets,
                                        vectorref<unsigned short> &alignresult, vectorref<float> &edgeacscores) const // output
{
    // Layout: each thread block takes 1024 threads; and we have #edges/1024 blocks.
    // This limits us to 16 million edges. If you need more, please adjust to either use wider thread blocks or a second dimension for the grid. Don't forget to adjust the kernel as well.
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));
    // cudaarrayref<float> logLLsarray;        // TODO: pass this in, of course
    // passtextureref texref (logLLstex, logLLsarray);    // use the same name as that global texref one, so it will match the name inside the kernel
    edgealignmentj<<<b, t, 0, /*GetCurrentStream()*/ cudaStreamDefault>>>(hmms, transPs, spalignunitid, silalignunitid, logLLs, nodes, edges, aligns, alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
    checklaunch("edgealignment");
}

// setvalue --helper to initialize an array to a constant value, e.g. LOGZERO
__global__ void setvaluej(vectorref<double> arraytoset, double value, size_t nelem)
{
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t j = jinblock + blockIdx.x * tpb;
    if (j < nelem)
    {
        msra::lattices::latticefunctionskernels::setvaluej(j, arraytoset, value);
    }
}

__global__ void expfi(matrixref<float> mata)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < mata.rows())
    {
        const size_t m = mata.cols();
        for (size_t j = 0; j < m; j++)
            mata(i, j) = expf(mata(i, j));
    }
}

__global__ void dotprodi(matrixref<float> mata, matrixref<float> matb)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < mata.rows())
    {
        const size_t m = mata.cols();
        for (size_t j = 0; j < m; j++)
            mata(i, j) = mata(i, j) * matb(i, j);
    }
}

__global__ void setunseeni(matrixref<float> errorsignal, matrixref<float> errorsignalauxbuf)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < errorsignal.rows())
    {
        const size_t m = errorsignal.cols();
        for (size_t j = 0; j < m; j++)
            if (errorsignal(i, j) == logf(CUDART_MIN_DENORM_F) && errorsignalauxbuf(i, j) == logf(CUDART_MIN_DENORM_F))
                errorsignalauxbuf(i, j) = LOGZERO;
    }
}

// errorsignal(i,j) = (exp(errorsignal(i,j)) - exp(errorsignal(i,j))) / amf
__global__ void errorcomputationi(matrixref<float> errorsignal, matrixref<float> errorsignalauxbuf, float amf)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < errorsignal.rows())
    {
        const size_t m = errorsignal.cols();
        for (size_t j = 0; j < m; j++)
            errorsignal(i, j) = msra::lattices::latticefunctionskernels::expdiff(errorsignal(i, j), errorsignalauxbuf(i, j)) / amf;
    }
}

// exp(errorsignal(i,j)) - exp(logEframescorrecttotal+errorsignalauxbuf(i,j))/amf
__global__ void directerrorcomputationi(matrixref<float> errorsignal, matrixref<float> errorsignalauxbuf, float logEframescorrecttotal, float amf)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < errorsignal.rows())
    {
        const size_t m = errorsignal.cols();
        for (size_t j = 0; j < m; j++)
            errorsignal(i, j) = msra::lattices::latticefunctionskernels::expdiff(errorsignal(i, j), logEframescorrecttotal + errorsignalauxbuf(i, j)) / amf;
    }
}

// compute the final error signal from gammas and state-consolidated Eframescorrect
// in-place operation is supported (i.e. output = one of the inputs)
__global__ void computesMBRerrorsignals(const matrixref<float> loggammas, const matrixref<float> logEframescorrect,
                                        const double logEframecorrecttotal, const float kappa, matrixref<float> errorsignal)
{
    const size_t s = threadIdx.x + (blockIdx.x * blockDim.x);
    if (s < loggammas.rows())
        msra::lattices::latticefunctionskernels::computesMBRerrorsignals(s, loggammas, logEframescorrect, logEframecorrecttotal, kappa, errorsignal);
}

__global__ void forwardlatticej(const size_t batchsize, const size_t startindex, const vectorref<float> edgeacscores,
                                const size_t spalignunitid, const size_t silalignunitid,
                                vectorref<msra::lattices::edgeinfowithscores> edges, vectorref<msra::lattices::nodeinfo> nodes,
                                const vectorref<msra::lattices::aligninfo> aligns, vectorref<unsigned short> alignments,
                                vectorref<unsigned int> alignmentoffsets, vectorref<double> logalphas, float lmf, float wp, float amf,
                                const float boostingfactor, const vectorref<unsigned short> uids, const vectorref<unsigned short> senone2classmap,
                                const bool returnEframescorrect, vectorref<double> logframescorrectedge, vectorref<double> logaccalphas)
{
    const size_t shufflemode = 1; // [v-hansu] this gives us about 100% speed up than shufflemode = 0 (no shuffle)
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < batchsize) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::forwardlatticej(j + startindex, edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns, alignments, alignmentoffsets,
                                                                 logalphas, lmf, wp, amf, boostingfactor, uids, senone2classmap, returnEframescorrect, logframescorrectedge, logaccalphas);
    }
}

__global__ void backwardlatticej(const size_t batchsize, const size_t startindex, const vectorref<float> edgeacscores,
                                 const size_t spalignunitid, const size_t silalignunitid,
                                 vectorref<msra::lattices::edgeinfowithscores> edges, vectorref<msra::lattices::nodeinfo> nodes,
                                 vectorref<msra::lattices::aligninfo> aligns, const double totalfwscore,
                                 vectorref<double> logpps, vectorref<double> logalphas, vectorref<double> logbetas,
                                 float lmf, float wp, float amf, const float boostingfactor, const bool returnEframescorrect,
                                 vectorref<double> logframescorrectedge, vectorref<double> logaccalphas,
                                 vectorref<double> logEframescorrect, vectorref<double> logaccbetas)
{
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    size_t j = jinblock + blockIdx.x * tpb;
    if (j < batchsize) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::backwardlatticej(j + startindex, edgeacscores, spalignunitid, silalignunitid,
                                                                  edges, nodes, aligns, totalfwscore, logpps, logalphas, logbetas,
                                                                  lmf, wp, amf, boostingfactor, returnEframescorrect,
                                                                  logframescorrectedge, logaccalphas,
                                                                  logEframescorrect, logaccbetas);
    }
}

void latticefunctionsops::forwardbackwardlattice(const size_t *batchsizeforward, const size_t *batchsizebackward,
                                                 const size_t numlaunchforward, const size_t numlaunchbackward,
                                                 const size_t spalignunitid, const size_t silalignunitid,
                                                 const vectorref<float> &edgeacscores,
                                                 const vectorref<msra::lattices::edgeinfowithscores> &edges,
                                                 const vectorref<msra::lattices::nodeinfo> &nodes,
                                                 const vectorref<msra::lattices::aligninfo> &aligns,
                                                 const vectorref<unsigned short> &alignments,
                                                 const vectorref<unsigned int> &aligmentoffsets,
                                                 vectorref<double> &logpps, vectorref<double> &logalphas, vectorref<double> &logbetas,
                                                 const float lmf, const float wp, const float amf, const float boostingfactor,
                                                 const bool returnEframescorrect, const vectorref<unsigned short> &uids,
                                                 const vectorref<unsigned short> &senone2classmap, vectorref<double> &logaccalphas,
                                                 vectorref<double> &logaccbetas, vectorref<double> &logframescorrectedge,
                                                 vectorref<double> &logEframescorrect, vectorref<double> & /*Eframescorrectbuf*/,
                                                 double &logEframescorrecttotal, double &totalfwscore) const
{
    // initialize log{,acc}(alhas/betas)
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((logalphas.size() + tpb - 1) / tpb));

    // TODO: is this really efficient? One thread per value?
    setvaluej<<<b, t, 0, GetCurrentStream()>>>(logalphas, LOGZERO, logalphas.size());
    checklaunch("setvaluej");
    setvaluej<<<b, t, 0, GetCurrentStream()>>>(logbetas, LOGZERO, logalphas.size());
    checklaunch("setvaluej");
    if (returnEframescorrect)
    {
        setvaluej<<<b, t, 0, GetCurrentStream()>>>(logaccalphas, LOGZERO, logalphas.size());
        checklaunch("setvaluej");
        setvaluej<<<b, t, 0, GetCurrentStream()>>>(logaccbetas, LOGZERO, logalphas.size());
        checklaunch("setvaluej");
    }
    // set initial tokens to probability 1 (0 in log)
    double log1 = 0.0;
    memcpy(logalphas.get(), 0, &log1, 1);
    memcpy(logbetas.get(), nodes.size() - 1, &log1, 1);

    // forward pass
    size_t startindex = 0;
    for (size_t i = 0; i < numlaunchforward; i++)
    {
        dim3 b2((unsigned int) ((batchsizeforward[i] + tpb - 1) / tpb));
        forwardlatticej<<<b2, t, 0, GetCurrentStream()>>>(batchsizeforward[i], startindex, edgeacscores,
                                                         spalignunitid, silalignunitid, edges, nodes, aligns,
                                                         alignments, aligmentoffsets, logalphas, lmf, wp, amf,
                                                         boostingfactor, uids, senone2classmap, returnEframescorrect,
                                                         logframescorrectedge, logaccalphas);
        checklaunch("edgealignment");
        startindex += batchsizeforward[i];
    }
    memcpy<double>(&totalfwscore, logalphas.get(), nodes.size() - 1, 1);
    double totalfwacc = 0;
    if (returnEframescorrect)
    {
        memcpy<double>(&totalfwacc, logaccalphas.get(), nodes.size() - 1, 1);
        totalfwacc -= totalfwscore;
    }

    // backward pass
    startindex = edges.size();
    for (size_t i = 0; i < numlaunchbackward; i++)
    {
        dim3 b2((unsigned int) ((batchsizebackward[i] + tpb - 1) / tpb));
        backwardlatticej<<<b2, t, 0, GetCurrentStream()>>>(batchsizebackward[i], startindex - batchsizebackward[i],
                                                          edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns,
                                                          totalfwscore, logpps, logalphas, logbetas,
                                                          lmf, wp, amf, boostingfactor, returnEframescorrect, logframescorrectedge,
                                                          logaccalphas, logEframescorrect, logaccbetas);
        checklaunch("edgealignment");
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

    double difffwbwscore = totalfwscore - totalbwscore;
    double absdifffwbwscore = difffwbwscore > 0 ? difffwbwscore : 0 - difffwbwscore;

    if (absdifffwbwscore / nodes.size() > 1e-4)
        fprintf(stderr, "forwardbackward: WARNING: lattice fw and bw scores %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwscore, (float) totalbwscore, (int) nodes.size(), (int) edges.size());

    if (returnEframescorrect)
    {
        double difffwbwacc = totalfwacc - totalbwacc;
        double absdifffwbwacc = difffwbwacc > 0 ? difffwbwacc : 0 - difffwbwacc;

        if (absdifffwbwacc / nodes.size() > 1e-4)
            fprintf(stderr, "forwardbackward: WARNING: lattice fw and bw acc %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwacc, (float) totalbwacc, (int) nodes.size(), (int) edges.size());
    }
}

// -----------------------------------------------------------------------
// sMBRerrorsignal -- accumulate difference of logEframescorrect and logEframescorrecttotal into errorsignal
// -----------------------------------------------------------------------

__global__ void sMBRerrorsignalj(const vectorref<unsigned short> alignstateids, const vectorref<unsigned int> alignoffsets,
                                 const vectorref<msra::lattices::edgeinfowithscores> edges, const vectorref<msra::lattices::nodeinfo> nodes,
                                 vectorref<double> logpps, const float amf, const vectorref<double> logEframescorrect, const double logEframescorrecttotal,
                                 matrixref<float> errorsignal, matrixref<float> errorsignalneg)
{
    const size_t shufflemode = 1; // [v-hansu] this gives us about 100% speed up than shufflemode = 0 (no shuffle)
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < edges.size()) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::sMBRerrorsignalj(j, alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg);
    }
}

// -----------------------------------------------------------------------
// stateposteriors --accumulate a per-edge quantity into the states that the edge is aligned with
// -----------------------------------------------------------------------

__global__ void stateposteriorsj(const vectorref<unsigned short> alignstateids, const vectorref<unsigned int> alignoffsets,
                                 const vectorref<msra::lattices::edgeinfowithscores> edges, const vectorref<msra::lattices::nodeinfo> nodes,
                                 const vectorref<double> logqs, matrixref<float> logacc)
{
    const size_t shufflemode = 1; // [v-hansu] this gives us about 100% speed up than shufflemode = 0 (no shuffle)
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < edges.size()) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::stateposteriorsj(j, alignstateids, alignoffsets, edges, nodes, logqs, logacc);
    }
}

__global__ void setvaluei(matrixref<float> us, float value)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= us.rows())
        return;
    // set all columns
    const size_t m = us.cols();
    for (size_t j = 0; j < m; j++)
        us(i, j) = value;
}

void latticefunctionsops::stateposteriors(const vectorref<unsigned short> &alignstateids, const vectorref<unsigned int> &alignoffsets,
                                          const vectorref<msra::lattices::edgeinfowithscores> &edges, const vectorref<msra::lattices::nodeinfo> &nodes,
                                          const vectorref<double> &logqs, matrixref<float> &logacc) const
{
    // Layout: each thread block takes 1024 threads; and we have #edges/1024 blocks.
    // This limits us to 16 million edges. If you need more, please adjust to either use wider thread blocks or a second dimension for the grid. Don't forget to adjust the kernel as well.
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));

    setvaluei<<<dim3((((unsigned int) logacc.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(logacc, LOGZERO);
    checklaunch("setvaluei");

    stateposteriorsj<<<b, t, 0, GetCurrentStream()>>>(alignstateids, alignoffsets, edges, nodes, logqs, logacc);
    checklaunch("stateposteriors");
}

void latticefunctionsops::sMBRerrorsignal(const vectorref<unsigned short> &alignstateids, const vectorref<unsigned int> &alignoffsets,
                                          const vectorref<msra::lattices::edgeinfowithscores> &edges, const vectorref<msra::lattices::nodeinfo> &nodes,
                                          const vectorref<double> &logpps, const float amf, const vectorref<double> &logEframescorrect, const double logEframescorrecttotal,
                                          matrixref<float> &errorsignal, matrixref<float> &errorsignalauxbuf) const
{
    // Layout: each thread block takes 1024 threads; and we have #edges/1024 blocks.
    // This limits us to 16 million edges. If you need more, please adjust to either use wider thread blocks or a second dimension for the grid. Don't forget to adjust the kernel as well.
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));

#ifdef DIRECT_MODE // compute Eframescorrect in a more direct way, proven to get same result as below
    setvaluei<<<dim3((((unsigned int) errorsignal.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignal, LOGZERO);
    checklaunch("setvaluei");

    sMBRerrorsignalj<<<b, t, 0, GetCurrentStream()>>>(alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalauxbuf);
    checklaunch("sMBRerrorsignal"); // now we get state based logEframescorrect

    matrixref<float> &loggammas = errorsignalauxbuf;
    setvaluei<<<dim3((((unsigned int) errorsignalauxbuf.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignalauxbuf, LOGZERO);
    checklaunch("setvaluei");

    stateposteriorsj<<<b, t, 0, GetCurrentStream()>>>(alignstateids, alignoffsets, edges, nodes, logpps, loggammas);
    checklaunch("stateposteriorsj"); // now we get state based loggammas

    directerrorcomputationi<<<dim3((((unsigned int) errorsignal.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignal, errorsignalauxbuf, logEframescorrecttotal, amf);
    checklaunch("errorcomputationj");
#else // this saves some computation compared with DIRECT_MODE
    setvaluei<<<dim3((((unsigned int) errorsignal.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignal, LOGZERO);
    checklaunch("setvaluei");
    setvaluei<<<dim3((((unsigned int) errorsignalauxbuf.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignalauxbuf, LOGZERO);
    checklaunch("setvaluei");
    sMBRerrorsignalj<<<b, t, 0, GetCurrentStream()>>>(alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalauxbuf);
    checklaunch("sMBRerrorsignal");

    setunseeni<<<dim3((((unsigned int) errorsignal.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignal, errorsignalauxbuf);
    checklaunch("setunseenj");

    errorcomputationi<<<dim3((((unsigned int) errorsignal.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignal, errorsignalauxbuf, amf);
    checklaunch("errorcomputationj");

#endif
}

void latticefunctionsops::mmierrorsignal(const vectorref<unsigned short> &alignstateids, const vectorref<unsigned int> &alignoffsets,
                                         const vectorref<msra::lattices::edgeinfowithscores> &edges, const vectorref<msra::lattices::nodeinfo> &nodes,
                                         const vectorref<double> &logpps, matrixref<float> &errorsignal) const
{
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));

    matrixref<float> &loggammas = errorsignal; // remember--this is an alias to 'errorsignal'
    setvaluei<<<dim3((((unsigned int) loggammas.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(loggammas, LOGZERO);
    checklaunch("setvaluei");
    stateposteriorsj<<<b, t, 0, GetCurrentStream()>>>(alignstateids, alignoffsets, edges, nodes, logpps, loggammas);
    checklaunch("stateposteriorsj");

    expfi<<<dim3((((unsigned int) errorsignal.rows()) + 31) / 32), 32, 0, GetCurrentStream()>>>(errorsignal);
    checklaunch("expfi");
}
};
};
