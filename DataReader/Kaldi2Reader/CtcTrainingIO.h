#pragma once

#include "kaldi.h"
#include "Matrix.h"
#include "UtteranceDerivativeComputationInterface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class deals with the CTC training in CNTK.
//TODO: this code needs to inherit from a parallel fwd/bwd propogation class:
//    -> but note that fwd/bwd on lattice would still be single threaded.
template<class ElemType>
class CtcTrainingIO :
public UtteranceDerivativeComputationInterface<ElemType>
{
private:
    wstring m_trainCriterion;
    wstring m_currentUttID;
    kaldi::RandomAccessInt32VectorReader* m_labRspecifier;        /*label sequence*/

public:
    // Ctc Constructor.
    CtcTrainingIO( const wstring& labRspecifier, const wstring& trainCriterion);

    // Ctc Destructor.
    ~CtcTrainingIO();

    virtual bool ComputeDerivative(const wstring& uttID,
                                   const Matrix<ElemType>& logLikelihoodIn,
                                   Matrix<ElemType>* derivative,
                                   ElemType* objective);

    bool HasResourceForDerivative(const wstring& uttID) const;

protected:
    void ComputeCtcLatticeForward(
          Matrix<ElemType> &alpha,
          Matrix<ElemType> &prob,
          int row,
          std::vector<size_t> &labels);

    void ComputeCtcLatticeBackward(
          Matrix<ElemType> &beta,
          Matrix<ElemType> &prob,
          int row,
          std::vector<size_t> &labels);

    void ComputeCtcError(
            Matrix<ElemType> &ctc_err,
            Matrix<ElemType> &alpha,
            Matrix<ElemType> &beta,
            Matrix<ElemType> &log_nnet_out,
            std::vector<size_t> &labels,
            float pzx);
};

}}}
