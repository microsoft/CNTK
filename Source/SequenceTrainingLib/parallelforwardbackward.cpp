// parallelforwardbackward.cpp -- parallelized implementation(s) of lattice forward/backward implemented --currently through CUDA
//
// F. Seide, V-hansu

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "BestGpu.h"        // for CPUONLY
#include "latticearchive.h" // we implement parts of class lattice
#include "simple_checked_arrays.h"
#include "simplesenonehmm.h" // the model
#include "ssematrix.h"       // the matrices
#include "cudalattice.h"
#include "latticefunctionskernels.h" // for emulation
#include "cudalatticeops.h"
#include <numeric> // for debug
#include "cudalib.h"
#include "Basics.h"

#define TWO_CHANNEL // [v-hansu]
using namespace msra::cuda;

#ifndef CPUONLY
#pragma comment(lib, "MathCUDA.lib") // built by CNTKMathCUDA project
#endif

namespace msra { namespace lattices {

// emulation support
struct dim3
{
    size_t x, y, z;
    dim3(size_t x = 1, size_t y = 1, size_t z = 1)
        : x(x), y(y), z(z)
    {
    }
};

static dim3 blockIdx, gridDim, threadIdx, blockDim;

template <typename FUNC>
__forceinline void emulatecuda(const dim3& b, const dim3& t, FUNC f)
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

void setvaluej(std::vector<double>& thisvector, double value, size_t nelem)
{
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t j = jinblock + blockIdx.x * tpb;
    if (j < nelem) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::setvaluej(j, thisvector, value);
    }
}

void expdiffj(std::vector<double>& Eframescorrectbuf, double& logEframescorrecttotal, size_t nelem, std::vector<double>& logEframescorrect)
{
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t j = jinblock + blockIdx.x * tpb;
    if (j < nelem) // note: will cause issues if we ever use __synctreads()
    {
        logEframescorrect[j] = (float) (Eframescorrectbuf[j] - logEframescorrecttotal);
    }
}
// this must be identical to an actual CUDA kernel (except for the input data types: vectorref -> std::vector)
void edgealignmentj(const std::vector<lrhmmdef>& hmms, const std::vector<lr3transP>& transPs, const size_t spalignunitid, const size_t silalignunitid,
                    const std::vector<msra::lattices::nodeinfo>& nodes, const std::vector<msra::lattices::edgeinfowithscores>& edges,
                    const std::vector<msra::lattices::aligninfo>& aligns,
                    const msra::math::ssematrixbase& logLLs, const std::vector<unsigned int>& alignoffsets,
                    std::vector<unsigned short>& backptrstorage, const std::vector<size_t>& backptroffsets,
                    std::vector<unsigned short>& alignresult, std::vector<float>& edgeacscores)
{
    // this function identifies which element in the grid we are, and then passes to the actual function that does stuff
    // compute j; but check if it is in range; return if not
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t j = jinblock + blockIdx.x * tpb;
    if (j < edges.size()) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::edgealignmentj(j, hmms, transPs, spalignunitid, silalignunitid, logLLs, nodes, edges, aligns,
                                                                alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
    }
}

void forwardlatticej(const size_t batchsize, const size_t startindex, const std::vector<float>& edgeacscores,
                     const size_t spalignunitid, const size_t silalignunitid,
                     const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                     const std::vector<msra::lattices::aligninfo>& aligns,
                     const std::vector<unsigned short>& alignments, const std::vector<unsigned int>& alignmentoffsets,
                     std::vector<double>& logalphas, float lmf, float wp, float amf, const float boostingfactor,
                     const std::vector<unsigned short>& uids, const std::vector<unsigned short>& senone2classmap, const bool returnEframescorrect,
                     std::vector<double>& logframescorrectedge, std::vector<double>& logaccalphas)
{
    const size_t shufflemode = 1;
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < batchsize) // note: will cause issues if we ever use __synctreads() in forwardlatticej
    {
        msra::lattices::latticefunctionskernels::forwardlatticej(j + startindex, edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns, alignments, alignmentoffsets,
                                                                 logalphas, lmf, wp, amf, boostingfactor, uids, senone2classmap, returnEframescorrect, logframescorrectedge, logaccalphas);
    }
}

void backwardlatticej(const size_t batchsize, const size_t startindex, const std::vector<float>& edgeacscores,
                      const size_t spalignunitid, const size_t silalignunitid,
                      const std::vector<msra::lattices::edgeinfowithscores>& edges,
                      const std::vector<msra::lattices::nodeinfo>& nodes,
                      const std::vector<msra::lattices::aligninfo>& aligns, const double totalfwscore,
                      std::vector<double>& logpps, std::vector<double>& logalphas, std::vector<double>& logbetas,
                      float lmf, float wp, float amf, const float boostingfactor, const bool returnEframescorrect, std::vector<double>& logframescorrectedge,
                      std::vector<double>& logaccalphas, std::vector<double>& Eframescorrectbuf, std::vector<double>& logaccbetas)
{
    const size_t tpb = blockDim.x * blockDim.y; // total #threads in a block
    const size_t jinblock = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t j = jinblock + blockIdx.x * tpb;
    if (j < batchsize) // note: will cause issues if we ever use __synctreads() in backwardlatticej
    {
        msra::lattices::latticefunctionskernels::backwardlatticej(j + startindex, edgeacscores, spalignunitid, silalignunitid,
                                                                  edges, nodes, aligns, totalfwscore, logpps, logalphas,
                                                                  logbetas, lmf, wp, amf, boostingfactor, returnEframescorrect, logframescorrectedge,
                                                                  logaccalphas, Eframescorrectbuf, logaccbetas);
    }
}

void sMBRerrorsignalj(const std::vector<unsigned short>& alignstateids, const std::vector<unsigned int>& alignoffsets,
                      const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                      const std::vector<double>& logpps, const float amf, const std::vector<double>& logEframescorrect,
                      const double logEframescorrecttotal, msra::math::ssematrixbase& errorsignal, msra::math::ssematrixbase& errorsignalneg)
{
    const size_t shufflemode = 3;
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < edges.size()) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::sMBRerrorsignalj(j, alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal,
                                                                  errorsignal, errorsignalneg);
    }
}

void stateposteriorsj(const std::vector<unsigned short>& alignstateids, const std::vector<unsigned int>& alignoffsets,
                      const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                      const std::vector<double>& logqs, msra::math::ssematrixbase& logacc)
{
    const size_t shufflemode = 3;
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < edges.size()) // note: will cause issues if we ever use __synctreads()
    {
        msra::lattices::latticefunctionskernels::stateposteriorsj(j, alignstateids, alignoffsets, edges, nodes, logqs, logacc);
    }
}

void checkshuffle(std::vector<int>& checkvector1, std::vector<int>& checkvector2, size_t shufflemode)
{
    const size_t j = msra::lattices::latticefunctionskernels::shuffle(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, blockIdx.x, gridDim.x, shufflemode);
    if (j < checkvector1.size())
    {
        checkvector1[j] = -1;
    }
    if (j < checkvector2.size())
    {
        checkvector2[j] -= 1;
    }
}

void errorcomputationi(msra::math::ssematrixbase& errorsignal, msra::math::ssematrixbase& errorsignalneg, float amf)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= errorsignal.rows())
        return;
    // clear all columns
    const size_t m = errorsignal.cols();
    for (size_t j = 0; j < m; j++)
        errorsignal(i, j) = (expf(errorsignal(i, j)) - expf(errorsignalneg(i, j))) / amf;
}

void weightedcomponentexpproducts(const msra::math::ssematrixbase& loggammas, const msra::math::ssematrixbase& logEframescorrect,
                                  const double logEframecorrecttotal, const float kappa, msra::math::ssematrixbase& errorsignal)
{
    const size_t s = threadIdx.x + (blockIdx.x * blockDim.x);
    if (s < errorsignal.rows())
    {
        msra::lattices::latticefunctionskernels::computesMBRerrorsignals(s, loggammas, logEframescorrect, logEframecorrecttotal, kappa, errorsignal);
    }
}

// this function behaves as its CUDA counterpart, except that it takes CPU-side std::vectors for everything
// this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
static void emulateedgealignment(const std::vector<lrhmmdef>& hmms, const std::vector<lr3transP>& transPs, const size_t spalignunitid, const size_t silalignunitid,
                                 const std::vector<msra::lattices::nodeinfo>& nodes, const std::vector<msra::lattices::edgeinfowithscores>& edges,
                                 const std::vector<msra::lattices::aligninfo>& aligns,
                                 const msra::math::ssematrixbase& logLLs, const std::vector<unsigned int>& alignoffsets,
                                 std::vector<unsigned short>& backptrstorage, const std::vector<size_t>& backptroffsets,
                                 std::vector<unsigned short>& alignresult, std::vector<float>& edgeacscores)
{
    // TODO: This function is about determining the parallelization layout
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));
    emulatecuda(b, t, [&]()
                {
                    edgealignmentj(hmms, transPs, spalignunitid, silalignunitid, nodes, edges, aligns, logLLs, alignoffsets, backptrstorage, backptroffsets, alignresult, edgeacscores);
                });
}

static double emulateforwardbackwardlattice(const size_t* batchsizeforward, const size_t* batchsizebackward,
                                            const size_t numlaunchforward, const size_t numlaunchbackward,
                                            const size_t spalignunitid, const size_t silalignunitid,
                                            const std::vector<float>& edgeacscores,
                                            const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                                            const std::vector<msra::lattices::aligninfo>& aligns,
                                            const std::vector<unsigned short>& alignments, const std::vector<unsigned int>& alignoffsets,
                                            std::vector<double>& logpps, std::vector<double>& logalphas, std::vector<double>& logbetas,
                                            const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect,
                                            const std::vector<unsigned short>& uids, const std::vector<unsigned short>& senone2classmap,
                                            std::vector<double>& logaccalphas, std::vector<double>& logaccbetas, std::vector<double>& logframescorrectedge,
                                            std::vector<double>& logEframescorrect, std::vector<double>& Eframescorrectbuf, double& logEframescorrecttotal)
{
    Eframescorrectbuf; // TODO: remove this [v-hansu]
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((logalphas.size() + tpb - 1) / tpb));
    emulatecuda(b, t, [&]()
                {
                    setvaluej(logalphas, LOGZERO, logalphas.size());
                });
    emulatecuda(b, t, [&]()
                {
                    setvaluej(logbetas, LOGZERO, logbetas.size());
                });
    emulatecuda(b, t, [&]()
                {
                    setvaluej(logaccalphas, LOGZERO, logaccalphas.size());
                });
    emulatecuda(b, t, [&]()
                {
                    setvaluej(logaccbetas, LOGZERO, logaccbetas.size());
                });

    logalphas.front() = 0;
    logbetas[nodes.size() - 1] = 0;

    // forward pass

    size_t startindex = 0;
    for (size_t i = 0; i < numlaunchforward; i++)
    {
        dim3 b((unsigned int) ((batchsizeforward[i] + tpb - 1) / tpb));
        emulatecuda(b, t, [&]()
                    {
                        forwardlatticej(batchsizeforward[i], startindex, edgeacscores, spalignunitid, silalignunitid, edges, nodes, aligns, alignments, alignoffsets,
                                        logalphas, lmf, wp, amf, boostingfactor, uids, senone2classmap, returnEframescorrect, logframescorrectedge, logaccalphas);
                    });
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
        dim3 b((unsigned int) ((batchsizebackward[i] + tpb - 1) / tpb));
        emulatecuda(b, t, [&]()
                    {
                        backwardlatticej(batchsizebackward[i], startindex - batchsizebackward[i], edgeacscores,
                                         spalignunitid, silalignunitid, edges, nodes, aligns,
                                         totalfwscore, logpps, logalphas, logbetas, lmf, wp, amf, boostingfactor,
                                         returnEframescorrect, logframescorrectedge, logaccalphas, logEframescorrect, logaccbetas);
                    });
        startindex -= batchsizebackward[i];
    }
    double totalbwscore = logbetas.front();
    if (returnEframescorrect)
        logEframescorrecttotal = logaccbetas.front() - totalbwscore;

#if 1 // check for matching
    double difffwbw = totalfwscore - totalbwscore;
    double absdifffwbw = difffwbw > 0 ? difffwbw : 0 - difffwbw;

    if (absdifffwbw / nodes.size() > 1e-4)
        fprintf(stderr, "forwardbackward: WARNING: lattice fw and bw scores %.10f vs. %.10f (%d nodes/%d edges)\n", (float) totalfwscore, (float) totalbwscore, (int) nodes.size(), (int) edges.size());
#endif
    return totalfwscore;
}
// this function behaves as its CUDA conterparts, except that it takes CPU-side std::vectors for everything
// this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
static void emulatesMBRerrorsignal(const std::vector<unsigned short>& alignstateids, const std::vector<unsigned int>& alignoffsets,
                                   const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                                   const std::vector<double>& logpps, const float amf,
                                   const std::vector<double>& logEframescorrect, const double logEframescorrecttotal,
                                   msra::math::ssematrixbase& errorsignal, msra::math::ssematrixbase& errorsignalneg)
{

    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    foreach_coord (i, j, errorsignal)
        errorsignal(i, j) = errorsignalneg(i, j) = LOGZERO;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));
    emulatecuda(b, t, [&]()
                {
                    sMBRerrorsignalj(alignstateids, alignoffsets, edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg);
                });
    dim3 b1((((unsigned int) errorsignal.rows()) + 31) / 32);
    emulatecuda(b1, 32, [&]()
                {
                    errorcomputationi(errorsignal, errorsignalneg, amf);
                });
}

// this function behaves as its CUDA conterparts, except that it takes CPU-side std::vectors for everything
// this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
static void emulatemmierrorsignal(const std::vector<unsigned short>& alignstateids, const std::vector<unsigned int>& alignoffsets,
                                  const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                                  const std::vector<double>& logpps, msra::math::ssematrixbase& errorsignal)
{
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    foreach_coord (i, j, errorsignal)
        errorsignal(i, j) = LOGZERO;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));
    emulatecuda(b, t, [&]()
                {
                    stateposteriorsj(alignstateids, alignoffsets, edges, nodes, logpps, errorsignal);
                });
    foreach_coord (i, j, errorsignal)
        errorsignal(i, j) = expf(errorsignal(i, j));
}

// this function behaves as its CUDA conterparts, except that it takes CPU-side std::vectors for everything
// this must be identical to CUDA kernel-launch function in -ops class (except for the input data types: vectorref -> std::vector)
/*static*/ void emulatestateposteriors(const std::vector<unsigned short>& alignstateids, const std::vector<unsigned int>& alignoffsets,
                                       const std::vector<msra::lattices::edgeinfowithscores>& edges, const std::vector<msra::lattices::nodeinfo>& nodes,
                                       const std::vector<double>& logqs, msra::math::ssematrixbase& logacc)
{
    foreach_coord (i, j, logacc)
        logacc(i, j) = LOGZERO;
    const size_t numedges = edges.size();
    dim3 t(32, 8);
    const size_t tpb = t.x * t.y;
    dim3 b((unsigned int) ((numedges + tpb - 1) / tpb));
    emulatecuda(b, t, [&]()
                {
                    stateposteriorsj(alignstateids, alignoffsets, edges, nodes, logqs, logacc);
                });
}

// -----------------------------------------------------------------------
// parallelstate (-impl) --holds variables for CUDA access
// -----------------------------------------------------------------------

struct parallelstateimpl
{
    bool emulation;
    size_t deviceid;
    parallelstateimpl(size_t deviceid)
        : deviceid(deviceid), emulation(false), // change this to true to switch to emulation
          // models
          lr3transPgpu(msra::cuda::newlr3transPvector(deviceid)),
          hmmsgpu(msra::cuda::newlrhmmdefvector(deviceid)),
          spalignunitid(SIZE_MAX),
          silalignunitid(SIZE_MAX),
          // current lattice, logLLs, and return values
          edgesgpu(msra::cuda::newedgeinfovector(deviceid)),
          nodesgpu(msra::cuda::newnodeinfovector(deviceid)),
          aligngpu(msra::cuda::newaligninfovector(deviceid)),
          alignresult(msra::cuda::newushortvector(deviceid)),
          alignoffsetsgpu(msra::cuda::newuintvector(deviceid)),
          edgeacscoresgpu(msra::cuda::newfloatvector(deviceid)),
          cudalogLLs(new Microsoft::MSR::CNTK::Matrix<float>((int) deviceid)),
          logppsgpu(msra::cuda::newdoublevector(deviceid)),
          logalphasgpu(msra::cuda::newdoublevector(deviceid)),
          logbetasgpu(msra::cuda::newdoublevector(deviceid)),
          logaccalphasgpu(msra::cuda::newdoublevector(deviceid)),
          logaccbetasgpu(msra::cuda::newdoublevector(deviceid)),
          logframescorrectedgegpu(msra::cuda::newdoublevector(deviceid)),
          Eframescorrectbufgpu(msra::cuda::newdoublevector(deviceid)),
          logEframescorrectgpu(msra::cuda::newdoublevector(deviceid)),
          uidsgpu(msra::cuda::newushortvector(deviceid)),
          senone2classmapgpu(msra::cuda::newushortvector(deviceid)),
          errorsignalgpu(new Microsoft::MSR::CNTK::Matrix<float>((int) deviceid)),
          errorsignalneggpu(new Microsoft::MSR::CNTK::Matrix<float>((int) deviceid)),
          errorsignalgpustorage(new Microsoft::MSR::CNTK::Matrix<float>((int) deviceid)),
          errorsignalneggpustorage(new Microsoft::MSR::CNTK::Matrix<float>((int) deviceid)),
          backptrstoragegpu(msra::cuda::newushortvector(deviceid)),
          backptroffsetsgpu(msra::cuda::newsizetvector(deviceid))
    {
    }

    size_t getdevice()
    {
        return deviceid;
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
    static_assert(sizeof(lrhmmdef) == 8, "unexpected size of lrhmmdef");
    void cachehset(const msra::asr::simplesenonehmm& hset, mbrclassdefinition mbrclassdef)
    {
        // only copy once
        // TODO: this can only be cached once --but there is no check whether a different model is passed
        if (lr3transPgpu->size() > 0)
            LogicError("cachehset: cannot bind to multiple model sets");

        // transPs
        lr3transPcpuforgpu.resize(hset.transPs.size());
        const std::vector<msra::asr::simplesenonehmm::transP>& transPs = hset.transPs;
        spalignunitid = hset.gethmmid("sp");
        silalignunitid = hset.gethmmid("sil");

        foreach_index (i, transPs)
        {
            const size_t numstates = transPs[i].getnumstates();

            lr3transPcpuforgpu[i].numstates = numstates;
            for (int m = 0; m < numstates + 1; m++)
            {

                for (int n = 0; n < numstates + 1; n++)
                {
                    lr3transPcpuforgpu[i].loga[m][n] = transPs[i](m - 1, n);
                }
            }
        }

        lr3transPgpu->assign(lr3transPcpuforgpu, false);

        if (mbrclassdef == monophone)
        {
            const auto& sphmm = hset.gethmm(spalignunitid);
            const auto& silhmm = hset.gethmm(silalignunitid);
            const size_t numsenones = hset.getnumsenone();
            senone2classmapcpuforgpu.resize(numsenones);
            for (size_t i = 0; i < numsenones; i++)
            {
                if (hset.statebelongstohmm(i, sphmm) || hset.statebelongstohmm(i, silhmm))
                    senone2classmapcpuforgpu[i] = silhmm.transPindex;
                else
                    senone2classmapcpuforgpu[i] = (unsigned short) hset.senonetransP(i);
            }
            senone2classmapgpu->assign(senone2classmapcpuforgpu, false);
        }

        // else // mbrclassdefinition:: senones has no mapping

        // hmm defs
        hmmscpuforgpu.resize(hset.hmms.size());
        const std::vector<msra::asr::simplesenonehmm::hmm>& hmms = hset.hmms;
        foreach_index (i, hmmscpuforgpu)
        {
            hmmscpuforgpu[i].numstates = (unsigned char) hmms[i].getnumstates();
            if (hmmscpuforgpu[i].numstates != hmms[i].getnumstates())
                LogicError("parallelforwardbackwardalign : hmms.numstates is out of range of unsigned char");

            for (size_t m = 0; m < hmmscpuforgpu[i].numstates; m++)
            {
                hmmscpuforgpu[i].senoneids[m] = (unsigned short) hmms[i].getsenoneid(m);
                if (hmmscpuforgpu[i].senoneids[m] != hmms[i].getsenoneid(m))
                    LogicError("parallelforwardbackwardalign : hmms.numstates is out of range of unsigned short");
            }

            hmmscpuforgpu[i].transPindex = (unsigned short) hmms[i].gettransPindex();
            if (hmmscpuforgpu[i].transPindex != hmms[i].gettransPindex())
                LogicError("parallelforwardbackwardalign : hmms.transPindex is out of range of unsigned short");
        }
        hmmsgpu->assign(hmmscpuforgpu, true /*sync*/); // need to sync if we free the memory right after (and we won't buy much from async)

        // if we are not emulating then we will delete our CPU-side copy to save memory
        if (!emulation)
        {
            lr3transPcpuforgpu.clear();
            senone2classmapcpuforgpu.clear();
            hmmscpuforgpu.clear();
        }
#ifdef FORBID_INVALID_SIL_PATH
        fprintf(stderr, "forbid invalid sil path\n"); // [v-hansu] might be inappropriate to print here, but it is convenient.
#endif
    }
    // check that we got a model and the right one
    void validatehset(const msra::asr::simplesenonehmm& hset)
    {
        if (hmmsgpu->size() != hset.hmms.size() || lr3transPgpu->size() != hset.transPs.size())
            LogicError("validatehset: not bound to hset or to wrong hset");
    }

    // current lattice
    std::unique_ptr<edgeinfowithscoresvector> edgesgpu;
    std::unique_ptr<nodeinfovector> nodesgpu;
    std::unique_ptr<aligninfovector> aligngpu;
    std::unique_ptr<ushortvector> alignresult; // concatenated alignments; edges[j]'s alignment starts at offset alignoffsets[j]
    std::unique_ptr<msra::cuda::uintvector> alignoffsetsgpu;
    std::unique_ptr<floatvector> edgeacscoresgpu;
    std::unique_ptr<Microsoft::MSR::CNTK::Matrix<float>> cudalogLLs;

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

    std::unique_ptr<Microsoft::MSR::CNTK::Matrix<float>> errorsignalgpustorage;
    std::unique_ptr<Microsoft::MSR::CNTK::Matrix<float>> errorsignalneggpustorage;
    std::unique_ptr<Microsoft::MSR::CNTK::Matrix<float>> errorsignalgpu;
    std::unique_ptr<Microsoft::MSR::CNTK::Matrix<float>> errorsignalneggpu;

    // cache current lattice
    // This is a weird mix of const/non-const and private lattice data... :(
    template <class edgestype, class nodestype, class aligntype, class edgealignments, class backpointers>
    void setutterancedata(const edgestype& edges, const nodestype& nodes, const aligntype& align,
                          const msra::math::ssematrixbase& /*logLLs*/, std::vector<float>& edgeacscores,
                          edgealignments& edgeAlignments, backpointers& backPointers)
    {
        // lattice data
        edgesgpu->assign(edges, false);
        nodesgpu->assign(nodes, false);
        aligngpu->assign(align, false);
        alignoffsetsgpu->assign(edgeAlignments.getalignoffsets(), false);
        backptrstoragegpu->allocate(backPointers.getbackptrstoragesize());
        backptroffsetsgpu->assign(backPointers.getbackptroffsets(), false);

#ifndef PARALLEL_SIL
        alignresult->assign(edgeAlignments.getalignmentsbuffer(), false);
        edgeacscoresgpu->assign(edgeacscores, false);
#else
        alignresult->allocate(edgeAlignments.getalignbuffersize());
        edgeacscoresgpu->allocate(edges.size());
        edgeacscores; // reference to make compilor happy
#endif

        // LLs
        // zhaorui
        /*cudalogLLs->allocate (logLLs.rows(), logLLs.cols());
            cudalogLLs->assign(0, logLLs.rows(), 0, logLLs.cols(), &logLLs(0,0), logLLs.getcolstride(), true);  // doing this last with 'true' so we can measure time better; maybe remove later*/
    }
    // template<class ElemType>
    void setloglls(const Microsoft::MSR::CNTK::Matrix<float>& loglls)
    {
        cudalogLLs->SetValue(loglls);
    }
    void getgamma(Microsoft::MSR::CNTK::Matrix<float>& loglls)
    {
        loglls.SetValue(*errorsignalgpu);
    }
    template <class edgealignments>
    void copyalignments(edgealignments& edgeAlignments)
    {
        alignresult->fetch(edgeAlignments.getalignmentsbuffer(), true);
    }

    // [v-hansu] allocate memory for vectors relating to forward-backward
    // allocateaccvectors implies return Eframecorrect
    template <class edgestype, class nodestype>
    void allocfwbwvectors(const edgestype& edges, const nodestype& nodes, const std::vector<unsigned short>& uids,
                          const bool allocateframescorrect, const bool copyuids, const bool allocateaccvectors)
    {
        logppsgpu->allocate(edges.size());
#ifndef TWO_CHANNEL
        const size_t alphabetanoderatio = 1;
#else
        const size_t alphabetanoderatio = 2;
#endif
        logalphasgpu->allocate(alphabetanoderatio * nodes.size());
        logbetasgpu->allocate(alphabetanoderatio * nodes.size());

        if (allocateframescorrect)
            logframescorrectedgegpu->allocate(edges.size());

        if (copyuids)
            uidsgpu->assign(uids, true);

        if (allocateaccvectors)
        {
            logaccalphasgpu->allocate(alphabetanoderatio * nodes.size());
            logaccbetasgpu->allocate(alphabetanoderatio * nodes.size());

            Eframescorrectbufgpu->allocate(edges.size());
            logEframescorrectgpu->allocate(edges.size());
        }
    }

    // check if gpumatrixstorage supports size of cpumatrix, if not allocate. set gpumatrix to part of gpumatrixstorage
    // This function checks the size of errorsignalgpustorage, and then sets errorsignalgpu to a columnslice of the
    // result, which encompases the entire matrix. Because this is a view of the underlying storage in 
    // errorsignalgpustorage, we must clear errorsignalgpu before resizing errorsignalgpustorage. After we resize,
    // we can then reset errorsignalgpu to be the result.
    void cacheerrorsignal(const msra::math::ssematrixbase& errorsignal, const bool cacheerrsignalneg)
    {
        if (errorsignalgpustorage->GetNumRows() != 0 && errorsignalgpustorage->GetNumRows() != errorsignal.rows())
            LogicError("gpumatrixstorage->rows() shall be fixed once allocated");
        if (errorsignalgpustorage->GetNumCols() < errorsignal.cols())
        {
            // Note: This is required because otherwise errorsignalgpustorage will be a view of the storage object in
            // errorsignalgpustorage, and thuse it can't resize. This is perhaps not the optimal way to do this, but
            // how else? Why do these two matrices exist? Why not just one?
            errorsignalgpu = nullptr;
            errorsignalgpustorage->Resize(errorsignal.rows(), errorsignal.cols());
        }
        errorsignalgpu = make_unique<Microsoft::MSR::CNTK::Matrix<float>>(errorsignalgpustorage->ColumnSlice(0, errorsignal.cols()));

        if (cacheerrsignalneg)
        {
            if (errorsignalneggpustorage->GetNumRows() != 0 && errorsignalneggpustorage->GetNumRows() != errorsignal.rows())
                LogicError("gpumatrixstorage->rows() shall be fixed once allocated");
            if (errorsignalneggpustorage->GetNumCols() < errorsignal.cols())
            {
                // Same as above.
                errorsignalneggpu = nullptr;
                errorsignalneggpustorage->Resize(errorsignal.rows(), errorsignal.cols());
            }
            errorsignalneggpu = make_unique<Microsoft::MSR::CNTK::Matrix<float>>(errorsignalneggpustorage->ColumnSlice(0, errorsignal.cols()));
        }
    }

    void getedgeacscores(std::vector<float>& edgeacscores)
    {
        edgeacscores.resize(edgeacscoresgpu->size());
        edgeacscoresgpu->fetch(edgeacscores, true);
    }

    void getedgealignments(std::vector<unsigned short>& edgealignments)
    {
        edgealignments.resize(alignresult->size());
        alignresult->fetch(edgealignments, true);
    }
};

void lattice::parallelstate::setdevice(size_t deviceid)
{
    bool pcpumode = (deviceid == CPUDEVICE);
    if (!pcpumode)
    {
        pimpl = new parallelstateimpl(deviceid);
    }
    else
    {
        delete pimpl;
        pimpl = NULL;
    }
    // else we leave it at NULL
}

size_t lattice::parallelstate::getdevice()
{
    return pimpl->getdevice();
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
lattice::parallelstate::parallelstate()
{
    pimpl = nullptr;
}
lattice::parallelstate::~parallelstate()
{
    delete pimpl;
}
void lattice::parallelstate::entercomputation(const msra::asr::simplesenonehmm& hset, const mbrclassdefinition mbrclassdef)
{
    pimpl->cachehset(hset, mbrclassdef);
} // pass models in (to GPU) // TODO: rethink the naming of this function
void lattice::parallelstate::copyalignments(edgealignments& edgealignments)
{
    pimpl->copyalignments(edgealignments);
}
const size_t lattice::parallelstate::getsilunitid()
{
    return pimpl->silalignunitid;
}
void lattice::parallelstate::getedgeacscores(std::vector<float>& edgeacscores)
{
    pimpl->getedgeacscores(edgeacscores);
}
void lattice::parallelstate::getedgealignments(std::vector<unsigned short>& edgealignments)
{
    pimpl->getedgealignments(edgealignments);
}
//template<class ElemType>
void lattice::parallelstate::setloglls(const Microsoft::MSR::CNTK::Matrix<float>& loglls)
{
    pimpl->setloglls(loglls);
}

// TODO: Overload to enable compilation for DoublePrecision though its currently unsupported
void lattice::parallelstate::setloglls(const Microsoft::MSR::CNTK::Matrix<double>& /*loglls*/)
{
    throw ::logic_error("Double precision not supported for sequence training");
}

void lattice::parallelstate::getgamma(Microsoft::MSR::CNTK::Matrix<float>& loglls)
{
    pimpl->getgamma(loglls);
}

// TODO: Overload to enable compilation for DoublePrecision though its currently unsupported
void lattice::parallelstate::getgamma(Microsoft::MSR::CNTK::Matrix<double>& /*loglls*/)
{
    throw ::logic_error("Double precision not supported for sequence training");
}

// -----------------------------------------------------------------------
// parallel implementations of key processing steps
// -----------------------------------------------------------------------

// parallelforwardbackwardalign() -- compute the statelevel gammas or viterbi alignments
void lattice::parallelforwardbackwardalign(parallelstate& parallelstate,
                                           const msra::asr::simplesenonehmm& hset, const msra::math::ssematrixbase& logLLs,
                                           std::vector<float>& edgeacscores, edgealignments& edgealignments, backpointers& backpointers) const
{
    parallelstate->validatehset(hset); // ensure the models have been correctly cached on the GPU already

    if (!parallelstate->emulation)
    {
        // move lattice to GPU
        parallelstate->setutterancedata(edges, nodes, align, logLLs,                 // inputs
                                        edgeacscores, edgealignments, backpointers); // inouts

        // launch the kernel
        std::unique_ptr<latticefunctions> latticefunctions(msra::cuda::newlatticefunctions(parallelstate.getdevice()));
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
            throw ::logic_error("we no longer support emulation for edgealign, please copy hmmscpuforgpu and lr3transPcpuforgpu if you want");
        edgeacscores.resize(edges.size());
        emulateedgealignment(parallelstate->hmmscpuforgpu, parallelstate->lr3transPcpuforgpu, parallelstate->spalignunitid,
                             parallelstate->silalignunitid,
                             nodes, edges, align, logLLs, edgealignments.getalignoffsets(),
                             backpointers.getbackptrbuffer(), backpointers.getbackptroffsets(),
                             edgealignments.getalignmentsbuffer(), edgeacscores);
        // emulate the GPU version, save result back to GPU
        parallelstate->alignresult->assign(edgealignments.getalignmentsbuffer(), false);
        parallelstate->edgeacscoresgpu->assign(edgeacscores, true);
    }
}

// parallelforwardbackwardlattice() -- compute the latticelevel logpps using forwardbackward
double lattice::parallelforwardbackwardlattice(parallelstate& parallelstate, const std::vector<float>& edgeacscores,
                                               const edgealignments& thisedgealignments, const float lmf, const float wp,
                                               const float amf, const float boostingfactor, std::vector<double>& logpps,
                                               std::vector<double>& logalphas, std::vector<double>& logbetas, const bool returnEframescorrect,
                                               const_array_ref<size_t>& uids, std::vector<double>& logEframescorrect,
                                               std::vector<double>& Eframescorrectbuf, double& logEframescorrecttotal) const
{                                     // ^^ TODO: remove this
    vector<size_t> batchsizeforward;  // record the batch size that exclude the data dependency for forward
    vector<size_t> batchsizebackward; // record the batch size that exclude the data dependency for backward

    size_t endindexforward = edges[0].E;
    size_t countbatchforward = 0;

    size_t endindexbackward = edges.back().S;
    size_t countbatchbackward = 0;
    foreach_index (j, edges) // compute the batch size info for kernel launches
    {
        if (edges[j].S < endindexforward)
            countbatchforward++; // note: we don't check forward because the order of end node is assured.
        else
        {
            batchsizeforward.push_back(countbatchforward);
            countbatchforward = 1;
            endindexforward = edges[j].E;
        }
        const size_t backj = edges.size() - 1 - j;
        if (edges[backj].E > endindexbackward)
        {
            countbatchbackward++;
            if (endindexbackward < edges[backj].S)
                endindexbackward = edges[backj].S;
        }
        else
        {
            batchsizebackward.push_back(countbatchbackward);
            countbatchbackward = 1;
            endindexbackward = edges[backj].S;
        }
    }
    batchsizeforward.push_back(countbatchforward);
    batchsizebackward.push_back(countbatchbackward);

    std::vector<unsigned short> uidsuint(uids.size()); // actually we shall not do this, but as it will not take much time, let us just leave it here now.
    foreach_index (i, uidsuint)
        uidsuint[i] = (unsigned short) uids[i];

    double totalfwscore = 0.0f;
    if (!parallelstate->emulation)
    {
        if (verbosity >= 2)
            fprintf(stderr, "parallelforwardbackwardlattice: %d launches for forward, %d launches for backward\n", (int) batchsizeforward.size(), (int) batchsizebackward.size());

        const bool allocateframescorrect = (returnEframescorrect || boostingfactor != 0.0f);
        const bool copyuids = (returnEframescorrect || boostingfactor != 0.0f);
        const bool allocateaccvectors = returnEframescorrect;
        parallelstate->allocfwbwvectors(edges, nodes, uidsuint, allocateframescorrect, copyuids, allocateaccvectors);

        std::unique_ptr<latticefunctions> latticefunctions(msra::cuda::newlatticefunctions(parallelstate.getdevice())); // final CUDA call
        latticefunctions->forwardbackwardlattice(&batchsizeforward[0], &batchsizebackward[0], batchsizeforward.size(), batchsizebackward.size(),
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
    else // emulation
    {
#ifndef TWO_CHANNEL
        fprintf(stderr, "forbid invalid sil path\n");
        const size_t alphabetanoderatio = 1;
#else
        const size_t alphabetanoderatio = 2;
#endif
        logpps.resize(edges.size());
        logalphas.resize(alphabetanoderatio * nodes.size());
        logbetas.resize(alphabetanoderatio * nodes.size());

        std::vector<double> logaccalphas;
        std::vector<double> logaccbetas;
        std::vector<double> logframescorrectedge;
        logframescorrectedge.resize(edges.size());

        if (returnEframescorrect)
        {
            logaccalphas.resize(alphabetanoderatio * nodes.size());
            logaccbetas.resize(alphabetanoderatio * nodes.size());

            logEframescorrect.resize(edges.size());
            Eframescorrectbuf.resize(edges.size());
        }

        totalfwscore = emulateforwardbackwardlattice(&batchsizeforward[0], &batchsizebackward[0],
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
void lattice::parallelsMBRerrorsignal(parallelstate& parallelstate, const edgealignments& thisedgealignments,
                                      const std::vector<double>& logpps, const float amf, const std::vector<double>& logEframescorrect,
                                      const double logEframescorrecttotal, msra::math::ssematrixbase& errorsignal, msra::math::ssematrixbase& errorsignalneg) const
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
        parallelstate->cacheerrorsignal(errorsignal, cacheerrorsignalneg);

        std::unique_ptr<latticefunctions> latticefunctions(msra::cuda::newlatticefunctions(parallelstate.getdevice()));
        latticefunctions->sMBRerrorsignal(*parallelstate->alignresult.get(), *parallelstate->alignoffsetsgpu.get(), *parallelstate->edgesgpu.get(),
                                          *parallelstate->nodesgpu.get(), *parallelstate->logppsgpu.get(), amf, *parallelstate->logEframescorrectgpu.get(),
                                          logEframescorrecttotal,
                                          *parallelstate->errorsignalgpu.get(), *parallelstate->errorsignalneggpu.get());

        if (errorsignal.rows() > 0 && errorsignal.cols() > 0)
        {
            parallelstate->errorsignalgpu->CopySection(errorsignal.rows(), errorsignal.cols(), &errorsignal(0, 0), errorsignal.getcolstride());
        }
    }
    else
    {
        emulatesMBRerrorsignal(thisedgealignments.getalignmentsbuffer(), thisedgealignments.getalignoffsets(), edges, nodes, logpps, amf, logEframescorrect, logEframescorrecttotal, errorsignal, errorsignalneg);
    }
}

// ------------------------------------------------------------------------
// parallel implementations of MMI error updating step
// ------------------------------------------------------------------------
void lattice::parallelmmierrorsignal(parallelstate& parallelstate, const edgealignments& thisedgealignments,
                                     const std::vector<double>& logpps, msra::math::ssematrixbase& errorsignal) const
{
    if (!parallelstate->emulation)
    {
        const bool cacheerrorsignalneg = false; // we do not need it in mmi mode
        parallelstate->cacheerrorsignal(errorsignal, cacheerrorsignalneg);

        std::unique_ptr<latticefunctions> latticefunctions(msra::cuda::newlatticefunctions(parallelstate.getdevice()));
        latticefunctions->mmierrorsignal(*parallelstate->alignresult.get(), *parallelstate->alignoffsetsgpu.get(), *parallelstate->edgesgpu.get(),
                                         *parallelstate->nodesgpu.get(), *parallelstate->logppsgpu.get(), *parallelstate->errorsignalgpu.get());

        // parallelstate->errorsignalgpu->fetch (0, errorsignal.rows(), 0, errorsignal.cols(), &errorsignal(0, 0), errorsignal.getcolstride(), true);
    }
    else
    {
        emulatemmierrorsignal(thisedgealignments.getalignmentsbuffer(), thisedgealignments.getalignoffsets(), edges, nodes, logpps, errorsignal);
    }
}
};
};
