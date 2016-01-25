#include "basetypes.h"
#include "htkfeatio_utils.h"
#include "UtteranceDerivativeBuffer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Constructor.
template <class ElemType>
UtteranceDerivativeBuffer<ElemType>::UtteranceDerivativeBuffer(
    size_t numberOfuttsPerMinibatch,
    UtteranceDerivativeComputationInterface<ElemType>* derivativeInterface)
{
    assert(derivativeInterface != NULL);
    m_derivativeInterface = derivativeInterface;
    m_numUttsPerMinibatch = numberOfuttsPerMinibatch;
    m_needLikelihood = true;
    m_currentObj = 0;
    m_uttReady.assign(m_numUttsPerMinibatch, false);
    m_epochEnd = false;
    m_dimension = 0;
}

template <class ElemType>
void UtteranceDerivativeBuffer<ElemType>::ProcessUttInfo(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    const MBLayoutPtr pMBLayout,
    std::vector<std::vector<std::pair<
        wstring, std::pair<size_t, size_t>>>>* uttInfoInMinibatch) const
{
    assert(uttInfoInMinibatch != NULL);
    assert(uttInfo.size() == m_numUttsPerMinibatch);
    assert(pMBLayout->GetNumParallelSequences() == m_numUttsPerMinibatch);
    uttInfoInMinibatch->clear();
    uttInfoInMinibatch->resize(uttInfo.size());

    for (size_t i = 0; i < uttInfo.size(); ++i)
    {
        size_t startFrameIndexInMinibatch = 0;
        size_t numFrames = 0;

        for (size_t j = 0; j < pMBLayout->GetNumTimeSteps(); ++j)
        {
            /*  if (pMBLayout->Is(i, j, MinibatchPackingFlags::NoLabel))
                {
                    continue;
                }*/
            FrameRange fr(pMBLayout, j);

            if (pMBLayout->IsGap(fr.Sequence(i)))
            {
                continue;
            }
            numFrames += 1;
            if (pMBLayout->IsBeyondStartOrEnd(fr.WithTimeOffset((ptrdiff_t) 1).Sequence(i)) || j == pMBLayout->GetNumTimeSteps() - 1)
            {
                size_t uttIndex = (*uttInfoInMinibatch)[i].size();
                wstring uttID = uttInfo[i][uttIndex].first;
                (*uttInfoInMinibatch)[i].push_back(
                    make_pair(uttID, make_pair(startFrameIndexInMinibatch,
                                               numFrames)));
                startFrameIndexInMinibatch = j + 1;
                numFrames = 0;
            }
        }
        assert(uttInfo[i].size() == (*uttInfoInMinibatch)[i].size());
    }
}

// Suppose we have a, b, c 3 streams, the <logLikelihoodIn> is the in the
// following format:
// 1: a11 b11 c11 a12 b12 c12...
// 2: a21 b21 c21 a22 b22 c22...
// 3: a31 b31 c31 a32 b32 c32...
template <class ElemType>
bool UtteranceDerivativeBuffer<ElemType>::SetLikelihood(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    const Matrix<ElemType>& logLikelihoodIn,
    const MBLayoutPtr pMBLayout)
{
    assert(m_needLikelihood == true);
    assert(m_epochEnd == false);

    if (m_dimension == 0)
    {
        m_dimension = logLikelihoodIn.GetNumRows();
    }
    assert(m_dimension == logLikelihoodIn.GetNumRows());

    std::vector<std::vector<
        std::pair<wstring, std::pair<size_t, size_t>>>> uttInfoInMinibatch;
    ProcessUttInfo(uttInfo, pMBLayout, &uttInfoInMinibatch);

    // Checks if we need to move data to CPU.
    Matrix<ElemType> logLikelihood(logLikelihoodIn);
    if (logLikelihood.GetDeviceId() >= 0)
    {
        logLikelihood.TransferFromDeviceToDevice(
            logLikelihood.GetDeviceId(), CPUDEVICE, true, false, false);
    }

    size_t currentMBSize = pMBLayout->GetNumTimeSteps();
    for (size_t i = 0; i < uttInfo.size(); ++i)
    {
        assert(uttInfo[i].size() == uttInfoInMinibatch[i].size());
        for (size_t j = 0; j < uttInfo[i].size(); ++j)
        {
            wstring uttID = uttInfo[i][j].first;
            if (m_uttPool.find(uttID) == m_uttPool.end())
            {
                UtteranceDerivativeUnit tmpUttUnit;
                tmpUttUnit.hasDerivative = false;
                tmpUttUnit.uttLength = uttInfo[i][j].second;
                tmpUttUnit.progress = 0;
                tmpUttUnit.streamID = i;
                tmpUttUnit.logLikelihood.Resize(logLikelihood.GetNumRows(),
                                                tmpUttUnit.uttLength);
                m_uttPool[uttID] = tmpUttUnit;
            }

            // Sets the likelihood and computes derivatives.
            assert(m_uttPool.find(uttID) != m_uttPool.end());
            if (m_uttPool[uttID].hasDerivative == false)
            {
                assert(uttID == uttInfoInMinibatch[i][j].first);
                size_t startFrame = uttInfoInMinibatch[i][j].second.first;
                size_t numFrames = uttInfoInMinibatch[i][j].second.second;
                assert(m_uttPool[uttID].progress + numFrames <= m_uttPool[uttID].uttLength);

                // Sets the likelihood.
                for (size_t k = 0; k < numFrames; ++k)
                {
                    m_uttPool[uttID].logLikelihood.SetColumn(
                        logLikelihood.ColumnSlice(
                            (startFrame + k) * m_numUttsPerMinibatch + i, 1),
                        m_uttPool[uttID].progress + k);
                }

                m_uttPool[uttID].progress += numFrames;
                if (m_uttPool[uttID].progress == m_uttPool[uttID].uttLength)
                {
                    m_derivativeInterface->ComputeDerivative(
                        uttID,
                        m_uttPool[uttID].logLikelihood,
                        &m_uttPool[uttID].derivative,
                        &m_uttPool[uttID].objective);
                    m_uttPool[uttID].hasDerivative = true;
                    m_uttPool[uttID].progress = 0;
                    m_uttReady[m_uttPool[uttID].streamID] = true;
                }
            }
        }
    }

    // Checks if we are ready to provide derivatives.
    m_needLikelihood = false;
    for (size_t i = 0; i < m_uttReady.size(); ++i)
    {
        if (m_uttReady[i] == false)
        {
            m_needLikelihood = true;
            break;
        }
    }
}

// Suppose we have a, b, c 3 streams, the <derivativesOut> should be in the
// following format:
// 1: a11 b11 c11 a12 b12 c12...
// 2: a21 b21 c21 a22 b22 c22...
// 3: a31 b31 c31 a32 b32 c32...
template <class ElemType>
bool UtteranceDerivativeBuffer<ElemType>::GetDerivative(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    const MBLayoutPtr pMBLayout,
    Matrix<ElemType>* derivativesOut)
{
    assert(derivativesOut != NULL);
    assert(m_needLikelihood == false);
    std::vector<std::vector<
        std::pair<wstring, std::pair<size_t, size_t>>>> uttInfoInMinibatch;
    ProcessUttInfo(uttInfo, pMBLayout, &uttInfoInMinibatch);

    m_currentObj = 0;
    Matrix<ElemType> derivatives(CPUDEVICE);
    derivatives.Resize(m_dimension, pMBLayout->GetNumCols());
    for (size_t i = 0; i < uttInfo.size(); ++i)
    {
        assert(uttInfo[i].size() == uttInfoInMinibatch[i].size());
        for (size_t j = 0; j < uttInfo[i].size(); ++j)
        {
            wstring uttID = uttInfo[i][j].first;

            // Checks if we have derivatives.
            if (m_uttPool.find(uttID) == m_uttPool.end() || (m_uttPool.find(uttID) != m_uttPool.end() && m_uttPool[uttID].hasDerivative == false))
            {
                RuntimeError("Derivatives are not ready for utterance:"
                             " %S\n",
                             uttID.c_str());
            }

            // Assign the derivatives.
            assert(uttID == uttInfoInMinibatch[i][j].first);
            size_t startFrame = uttInfoInMinibatch[i][j].second.first;
            size_t startFrameInUtt = m_uttPool[uttID].progress;
            size_t numFrames = uttInfoInMinibatch[i][j].second.second;
            for (size_t k = 0; k < numFrames; ++k)
            {
                derivatives.SetColumn(
                    m_uttPool[uttID].derivative.ColumnSlice(
                        startFrameInUtt + k, 1),
                    (startFrame + k) * m_numUttsPerMinibatch + i);
            }
            m_currentObj += m_uttPool[uttID].objective * numFrames / m_uttPool[uttID].uttLength;
            m_uttPool[uttID].progress += numFrames;
            assert(m_uttPool[uttID].progress <= m_uttPool[uttID].uttLength);
            if (m_uttPool[uttID].progress == m_uttPool[uttID].uttLength)
            {
                m_uttPool.erase(uttID);
            }
        }
    }

    // Checks if we need to move data to GPU.
    if (derivativesOut->GetDeviceId() >= 0)
    {
        derivatives.TransferFromDeviceToDevice(
            CPUDEVICE, derivativesOut->GetDeviceId(), true, false, false);
    }
    derivativesOut->SetValue(derivatives);

    // Keeps the utterance information so we can check next time when we
    // gives the objectives.
    m_currentUttInfo = uttInfo;

    // Checks if we need to read more loglikelihoods.
    m_needLikelihood = (m_epochEnd || m_uttPool.size() > 0) ? false : true;
    if (m_needLikelihood == true)
    {
        m_uttReady.assign(m_numUttsPerMinibatch, false);
    }
    return true;
}

template <class ElemType>
bool UtteranceDerivativeBuffer<ElemType>::GetObjective(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    Matrix<ElemType>* objectivesIn)
{
    assert(objectivesIn != NULL);

    // Checks utterance information.
    bool match = CompareUttInfo(uttInfo, m_currentUttInfo);
    if (!match)
    {
        RuntimeError("Current objective does not correspond to the"
                     " minibatch utterance information, perhaps you did not"
                     " run GetObjective() right after GetDerivative()?");
    }

    // Sets the objectives...
    objectivesIn->Resize(1, 1);
    objectivesIn->SetValue(m_currentObj);

    return true;
}

template <class ElemType>
bool UtteranceDerivativeBuffer<ElemType>::HasResourceForDerivative(
    const wstring& uttID) const
{
    return m_derivativeInterface->HasResourceForDerivative(uttID);
}

template <class ElemType>
bool UtteranceDerivativeBuffer<ElemType>::CompareUttInfo(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo1,
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo2)
{
    bool match = true;
    if (uttInfo1.size() == uttInfo2.size())
    {
        for (size_t i = 0; i < uttInfo1.size(); ++i)
        {
            if (uttInfo1[i].size() != uttInfo2[i].size())
            {
                match = false;
                break;
            }
            for (size_t j = 0; j < uttInfo1[i].size(); ++j)
            {
                if (uttInfo1[i][j].first != uttInfo2[i][j].first ||
                    uttInfo1[i][j].second != uttInfo2[i][j].second)
                {
                    match = false;
                    break;
                }
            }
        }
    }
    else
    {
        match = false;
    }
    return match;
}

template <class ElemType>
void UtteranceDerivativeBuffer<ElemType>::ResetEpoch()
{
    m_needLikelihood = true;
    m_currentObj = 0;
    m_epochEnd = false;
    m_uttPool.clear();
    m_currentUttInfo.clear();
    m_uttReady.assign(m_numUttsPerMinibatch, false);
}

template class UtteranceDerivativeBuffer<float>;
template class UtteranceDerivativeBuffer<double>;
} } }
