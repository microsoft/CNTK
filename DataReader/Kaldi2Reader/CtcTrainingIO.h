#pragma once

#include "kaldi.h"
#include "Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class deals with the CTC training in CNTK.
//TODO: this code needs to inherit from a parallel fwd/bwd propogation class:
//    -> but note that fwd/bwd on lattice would still be single threaded.
template<class ElemType>
class CtcTrainingIO
{
private:
    bool m_currentUttHasDeriv;
    bool m_derivRead;
    bool m_objRead;
    wstring m_trainCriterion;
    wstring m_currentUttID;
    ElemType m_objective;
    size_t m_currentUttLength;
    kaldi::TransitionModel m_transModel;
    Matrix<ElemType> m_posteriors;
    kaldi::RandomAccessInt32VectorReader* m_labRspecifier;        /*label sequence*/

public:
    // Ctc Constructor.
    CtcTrainingIO( const wstring& labRspecifier, const wstring& transModelFilename, const wstring& trainCriterion);

    // Ctc Destructor.
    ~CtcTrainingIO();

    bool HasDerivatives(const wstring& uttID);

    bool ComputeDerivatives(const wstring& uttID, const Matrix<ElemType>& outputs);

    // Gets the computed derivatives for given utterance.
    void GetDerivatives(size_t startFrame, size_t endFrame,
                        const std::wstring& uttID, Matrix<ElemType>& derivatives);

    // Gets the computed objectives for given utterance.
    void GetObjectives(size_t startFrame, size_t endFrame,
                       const std::wstring& uttID, Matrix<ElemType>& derivatives);

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
