#pragma once

#include "kaldi.h"
#include "Matrix.h"
#include "basetypes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class deals with the interaction with Kaldi in order to do sequence
// in CNTK.
template<class ElemType>
class KaldiSequenceTrainingIO
{
private:
    bool m_oneSilenceClass;
    bool m_needLikelihood;
    bool m_epochEnd;
    size_t m_numUttsPerMinibatch;
    wstring m_trainCriterion;
    ElemType m_oldAcousticScale;
    ElemType m_acousticScale;
    ElemType m_lmScale;
    std::vector<kaldi::int32> m_silencePhones;
    kaldi::TransitionModel m_transModel;
    kaldi::RandomAccessCompactLatticeReader* m_denlatReader;
    kaldi::RandomAccessInt32VectorReader* m_aliReader;

    struct UtteranceDerivativeUnit
    {
        bool hasDerivative;
        size_t uttLength;
        size_t progress;
        size_t streamID;
        Matrix<ElemType> logLikelihood;
        kaldi::Posterior posterior;
        ElemType objective;

        UtteranceDerivativeUnit() : logLikelihood(CPUDEVICE)
        {
            hasDerivative = false;
            uttLength = 0;
            progress = 0;
            streamID = 0;
        }
    };
    ElemType m_currentObj;
    int m_minCompleteMinibatchIndex;
    int m_minibatchIndex;
    std::vector<int> m_lastCompleteMinibatch;
    std::vector<std::vector<std::pair<wstring, size_t>>> m_currentUttInfo;
    unordered_map<wstring, UtteranceDerivativeUnit> m_uttPool;

    // Rescores the lattice with the lastest posteriors from the neural network.
    void LatticeAcousticRescore(
        const std::vector<kaldi::int32>& stateTimes,
        const Matrix<ElemType>& outputs, kaldi::Lattice* lat) const;

    // <uttInfoInMinibatch> is a vector of vector of the following:
    //     uttID startFrameIndexInMinibatch numFrames
    void ProcessUttInfo(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& sentenceBegin,
        const std::vector<MinibatchPackingFlag>& minibatchPackingFlag,
        std::vector<std::vector<std::pair<
            wstring, std::pair<size_t, size_t>>>>* uttInfoInMinibatch) const;

    bool ComputeDerivative(const wstring& uttID);

public:
    // Constructor.
    KaldiSequenceTrainingIO(const wstring& denlatRspecifier,
                            const wstring& aliRspecifier,
                            const wstring& transModelFilename,
                            const wstring& silencePhoneStr,
                            const wstring& trainCriterion,
                            ElemType oldAcousticScale,
                            ElemType acousticScale,
                            ElemType lmScale,
                            bool oneSilenceClass,
                            size_t numberOfuttsPerMinibatch);

    // Destructor.
    ~KaldiSequenceTrainingIO();

    bool NeedLikelihoodToComputeDerivative() const { return m_needLikelihood; }

    bool SetLikelihood(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& outputs,
        const Matrix<ElemType>& sentenceBegin,
        const std::vector<MinibatchPackingFlag>& minibatchPackingFlag);

    // Gets the computed derivatives for given utterance.
    bool GetDerivative(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& sentenceBegin,
        const std::vector<MinibatchPackingFlag>& minibatchPackingFlag,
        Matrix<ElemType>* derivativesOut);

    // Gets the computed objectives for given utterance.
    bool GetObjective(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        Matrix<ElemType>* objectivesIn);

    bool HasLatticeAndAlignment(const wstring& uttID) const;

    bool HasUtterance(const wstring& uttID) const
    {
        return (m_uttPool.find(uttID) != m_uttPool.end());
    }

    void SetEpochEnd() { m_epochEnd = true; m_needLikelihood = false; }

    void ResetEpoch();
};

}}}
