#pragma once

#include "kaldi.h"
#include "Matrix.h"
#include "basetypes.h"
#include "UtteranceDerivativeComputationInterface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class deals with the interaction with Kaldi in order to do sequence
// in CNTK.
template <class ElemType>
class KaldiSequenceTrainingDerivative : public UtteranceDerivativeComputationInterface<ElemType>
{
private:
    bool m_oneSilenceClass;
    wstring m_trainCriterion;
    ElemType m_oldAcousticScale;
    ElemType m_acousticScale;
    ElemType m_lmScale;
    std::vector<kaldi::int32> m_silencePhones;
    kaldi::TransitionModel m_transModel;
    kaldi::RandomAccessCompactLatticeReader* m_denlatReader;
    kaldi::RandomAccessInt32VectorReader* m_aliReader;

    // Rescores the lattice with the lastest posteriors from the neural network.
    void LatticeAcousticRescore(const wstring& uttID,
                                const Matrix<ElemType>& outputs,
                                kaldi::Lattice* lat) const;

    void ConvertPosteriorToDerivative(const kaldi::Posterior& post,
                                      Matrix<ElemType>* derivative);

public:
    // Constructor.
    KaldiSequenceTrainingDerivative(const wstring& denlatRspecifier,
                                    const wstring& aliRspecifier,
                                    const wstring& transModelFilename,
                                    const wstring& silencePhoneStr,
                                    const wstring& trainCriterion,
                                    ElemType oldAcousticScale,
                                    ElemType acousticScale,
                                    ElemType lmScale,
                                    bool oneSilenceClass);

    // Destructor.
    ~KaldiSequenceTrainingDerivative();

    bool ComputeDerivative(const wstring& uttID,
                           const Matrix<ElemType>& logLikelihood,
                           Matrix<ElemType>* derivative,
                           ElemType* objective);

    bool HasResourceForDerivative(const wstring& uttID) const;
};
} } }
