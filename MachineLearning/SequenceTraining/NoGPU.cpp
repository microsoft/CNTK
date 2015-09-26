#include "BestGpu.h"

#ifdef CPUONLY

#include "cudalatticeops.h"

#pragma warning (disable: 4100) // unreferenced formal parameter, which is OK since all functions in here are dummies; disabling this allows to copy-paste prototypes here when we add new functions
#pragma warning (disable: 4702) // unreachable code, which we get from the NOT_IMPLEMENTED macro which is OK

namespace msra { namespace cuda {

    void latticefunctionsops::edgealignment(const vectorref<lrhmmdef> & hmms, const vectorref<lr3transP> & transPs, const size_t spalignunitid,
        const size_t silalignunitid, const matrixref<float> & logLLs,
        const vectorref<msra::lattices::nodeinfo> & nodes, const vectorref<msra::lattices::edgeinfowithscores> & edges,
        const vectorref<msra::lattices::aligninfo> & aligns, const vectorref<unsigned int> & alignoffsets,
        vectorref<unsigned short> & backptrstorage, const vectorref<size_t> & backptroffsets,
        vectorref<unsigned short> & alignresult, vectorref<float> & edgeacscores) const {}

    void latticefunctionsops::forwardbackwardlattice(const size_t * batchsizeforward, const size_t * batchsizebackward,
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
        double & logEframescorrecttotal, double & totalfwscore) const {}

    void latticefunctionsops::sMBRerrorsignal(const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
        const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes,
        const vectorref<double> & logpps, const float amf, const vectorref<double> & logEframescorrect, const double logEframescorrecttotal,
        matrixref<float> & errorsignal, matrixref<float> & errorsignalneg) const {}

    void latticefunctionsops::mmierrorsignal(const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
        const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes,
        const vectorref<double> & logpps, matrixref<float> & errorsignal) const {}

    void latticefunctionsops::stateposteriors(const vectorref<unsigned short> & alignstateids, const vectorref<unsigned int> & alignoffsets,
        const vectorref<msra::lattices::edgeinfowithscores> & edges, const vectorref<msra::lattices::nodeinfo> & nodes,
        const vectorref<double> & logqs, matrixref<float> & logacc) const {}

    template <typename ElemType>
    void FetchFromGPUMatrix(const Microsoft::MSR::CNTK::Matrix<ElemType>& gpuMatrix, msra::math::ssematrixbase& cpuMatrix) {}

    template void FetchFromGPUMatrix<float>(const Microsoft::MSR::CNTK::Matrix<float>& gpuMatrix, msra::math::ssematrixbase& cpuMatrix);
}}

#endif // CPUONLY
