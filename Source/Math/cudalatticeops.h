// cudalatticeops.h -- contains all actual CUDA-side lattice ops
//
// F. Seide, V-hansu

#pragma once

#include "cudabasetypes.h"           // for vectorref<>
#include "latticestorage.h"          // for the lattice types
#include "latticefunctionskernels.h" // for the actual inner kernels and any argument types that are not yet defined in latticestorage.h

using namespace msra::lattices;

// Forward declarations
namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType>
class Matrix;
}
}
}

namespace msra { namespace math {

class ssematrixbase;
}
}

namespace msra { namespace cuda {

// The XXXvectorops classes must derive from vectorref<XXX>.

class latticefunctionsops : protected vectorref<msra::lattices::empty>
{
protected:
    void edgealignment(const vectorref<lrhmmdef>& hmms, const vectorref<lr3transP>& transPs, const size_t spalignunitid,
                       const size_t silalignunitid, const matrixref<float>& logLLs,
                       const vectorref<msra::lattices::nodeinfo>& nodes, const vectorref<msra::lattices::edgeinfowithscores>& edges,
                       const vectorref<msra::lattices::aligninfo>& aligns, const vectorref<unsigned int>& alignoffsets,
                       vectorref<unsigned short>& backptrstorage, const vectorref<size_t>& backptroffsets,
                       vectorref<unsigned short>& alignresult, vectorref<float>& edgeacscores) const; // output

    void forwardbackwardlattice(const size_t* batchsizeforward, const size_t* batchsizebackward,
                                const size_t numlaunchforward, const size_t numlaunchbackward,
                                const size_t spalignunitid, const size_t silalignunitid,
                                const vectorref<float>& edgeacscores, const vectorref<msra::lattices::edgeinfowithscores>& edges,
                                const vectorref<msra::lattices::nodeinfo>& nodes,
                                const vectorref<msra::lattices::aligninfo>& aligns, const vectorref<unsigned short>& aligments,
                                const vectorref<unsigned int>& aligmentoffsets,
                                vectorref<double>& logpps, vectorref<double>& logalphas, vectorref<double>& logbetas,
                                const float lmf, const float wp, const float amf, const float boostingfactor, const bool returnEframescorrect,
                                const vectorref<unsigned short>& uids, const vectorref<unsigned short>& senone2classmap,
                                vectorref<double>& logaccalphas, vectorref<double>& logaccbetas,
                                vectorref<double>& logframescorrectedge, vectorref<double>& logEframescorrect, vectorref<double>& Eframescorrectbuf,
                                double& logEframescorrecttotal, double& totalfwscore) const;

    void sMBRerrorsignal(const vectorref<unsigned short>& alignstateids, const vectorref<unsigned int>& alignoffsets,
                         const vectorref<msra::lattices::edgeinfowithscores>& edges, const vectorref<msra::lattices::nodeinfo>& nodes,
                         const vectorref<double>& logpps, const float amf, const vectorref<double>& logEframescorrect, const double logEframescorrecttotal,
                         matrixref<float>& errorsignal, matrixref<float>& errorsignalneg) const;

    void mmierrorsignal(const vectorref<unsigned short>& alignstateids, const vectorref<unsigned int>& alignoffsets,
                        const vectorref<msra::lattices::edgeinfowithscores>& edges, const vectorref<msra::lattices::nodeinfo>& nodes,
                        const vectorref<double>& logpps, matrixref<float>& errorsignal) const;

    void stateposteriors(const vectorref<unsigned short>& alignstateids, const vectorref<unsigned int>& alignoffsets,
                         const vectorref<msra::lattices::edgeinfowithscores>& edges, const vectorref<msra::lattices::nodeinfo>& nodes,
                         const vectorref<double>& logqs, matrixref<float>& logacc) const;
};
};
};
