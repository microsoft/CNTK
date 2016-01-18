#pragma once

#include "Matrix.h"
#include "basetypes.h"
#include "Sequences.h"
#include "UtteranceDerivativeComputationInterface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class "gules" together the log-likelihood from different minibatches,
// and then calls <UtteranceDerivativeComputationInterface> class to compute
// the derivative for given utterance.
template <class ElemType>
class UtteranceDerivativeBuffer
{
private:
    struct UtteranceDerivativeUnit
    {
        bool hasDerivative;
        size_t uttLength;
        size_t progress;
        size_t streamID;
        Matrix<ElemType> logLikelihood;
        Matrix<ElemType> derivative;
        ElemType objective;

        UtteranceDerivativeUnit()
            : logLikelihood(CPUDEVICE), derivative(CPUDEVICE)
        {
            hasDerivative = false;
            uttLength = 0;
            progress = 0;
            streamID = 0;
        }
    };

    bool m_needLikelihood;
    bool m_epochEnd;
    size_t m_numUttsPerMinibatch;
    size_t m_dimension;
    ElemType m_currentObj;
    std::vector<bool> m_uttReady;
    std::vector<std::vector<std::pair<wstring, size_t>>> m_currentUttInfo;
    unordered_map<wstring, UtteranceDerivativeUnit> m_uttPool;
    UtteranceDerivativeComputationInterface<ElemType>* m_derivativeInterface;

    // <uttInfoInMinibatch> is a vector of vector of the following:
    //     uttID startFrameIndexInMinibatch numFrames
    void ProcessUttInfo(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const MBLayoutPtr pMBLayout,
        std::vector<std::vector<std::pair<
            wstring, std::pair<size_t, size_t>>>>* uttInfoInMinibatch) const;

    bool CompareUttInfo(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo1,
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo2);

public:
    // Constructor.
    // Does not take ownership of <derivativeInterface>.
    UtteranceDerivativeBuffer(
        size_t numberOfuttsPerMinibatch,
        UtteranceDerivativeComputationInterface<ElemType>* derivativeInterface);

    // Destructor.
    ~UtteranceDerivativeBuffer()
    {
    }

    bool NeedLikelihoodToComputeDerivative() const
    {
        return m_needLikelihood;
    }

    bool SetLikelihood(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& outputs,
        const MBLayoutPtr pMBLayout);

    // Gets the computed derivatives for given utterance.
    bool GetDerivative(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const MBLayoutPtr pMBLayout,
        Matrix<ElemType>* derivativesOut);

    // Gets the computed objectives for given utterance.
    bool GetObjective(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        Matrix<ElemType>* objectivesIn);

    bool HasResourceForDerivative(const wstring& uttID) const;

    bool HasUtterance(const wstring& uttID) const
    {
        return (m_uttPool.find(uttID) != m_uttPool.end());
    }

    void SetEpochEnd()
    {
        m_epochEnd = true;
        m_needLikelihood = false;
    }

    void ResetEpoch();
};
} } }
