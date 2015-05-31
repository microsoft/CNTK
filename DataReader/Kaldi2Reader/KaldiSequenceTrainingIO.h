#pragma once

#include "kaldi.h"
#include "Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class deals with the interaction with Kaldi in order to do sequence
// in CNTK.
template<class ElemType>
class KaldiSequenceTrainingIO
{
private:
    bool m_oneSilenceClass;
    bool m_currentUttHasDeriv;
    bool m_derivRead;
    bool m_objRead;
    wstring m_trainCriterion;
    wstring m_currentUttID;
    ElemType m_oldAcousticScale;
    ElemType m_acousticScale;
    ElemType m_lmScale;
    ElemType m_objective;
    std::vector<kaldi::int32> m_silencePhones;
    size_t m_currentUttLength;
    kaldi::TransitionModel m_transModel;
    kaldi::Posterior m_posteriors;
    kaldi::RandomAccessCompactLatticeReader* m_denlatReader;  /*denominator lattices*/
    kaldi::RandomAccessInt32VectorReader* m_aliReader;        /*alignment*/

public:
    // Constructor.
    KaldiSequenceTrainingIO(const wstring& denlatRspecifier, const wstring& aliRspecifier,
                            const wstring& transModelFilename, const wstring& silencePhoneStr,
                            const wstring& trainCriterion,
                            ElemType oldAcousticScale,
                            ElemType acousticScale,
                            ElemType lmScale,
                            bool oneSilenceClass);

    // Destructor.
    ~KaldiSequenceTrainingIO();

    bool HasDerivatives(const wstring& uttID);

    bool ComputeDerivatives(const wstring& uttID, const Matrix<ElemType>& outputs);

    // Rescores the lattice with the lastest posteriors from the neural network.
    void LatticeAcousticRescore(const std::vector<kaldi::int32>& stateTimes,
                                const Matrix<ElemType>& outputs, kaldi::Lattice* lat);

    // Gets the computed derivatives for given utterance.
    void GetDerivatives(size_t startFrame, size_t endFrame,
                        const std::wstring& uttID, Matrix<ElemType>& derivatives);

    // Gets the computed objectives for given utterance.
    void GetObjectives(size_t startFrame, size_t endFrame,
                       const std::wstring& uttID, Matrix<ElemType>& derivatives);
};

}}}
