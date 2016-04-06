//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "ComputationNode.h"
#include "SpecialPurposeNodes.h"

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// Trace (node, say='', logFrequency=10, logFirst=10, logGradientToo=false, onlyUpToRow=100000000, onlyUpToT=100000000, format=[])
//
// Debugging aid to trace a node's value using WriteMinibatchWithFormatting().
// -----------------------------------------------------------------------

template <class ElemType>
TraceNode<ElemType>::TraceNode(const ScriptableObjects::IConfigRecordPtr configp) :
    TraceNode(configp->Get(L"deviceId"), L"<placeholder>")
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    m_message        = (const std::wstring&)configp->Get(L"say");
    m_logFirst       = configp->Get(L"logFirst");
    m_logFrequency   = configp->Get(L"logFrequency");
    m_logGradientToo = false; // configp->Get(L"logGradientToo"); not yet implemented
    m_formattingOptions = WriteFormattingOptions(*configp);
    m_onlyUpToRow    = configp->Get(L"onlyUpToRow");
    m_onlyUpToT      = configp->Get(L"onlyUpToT");
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Save(File& fstream) const /*override*/
{
    Base::Save(fstream);
    fstream << m_message;
    fstream << m_logFirst;
    fstream << m_logFrequency;
    fstream << m_logGradientToo;
    m_formattingOptions.Save(fstream);
    // BUGBUG: This serializes the pathname of the mapping file to disk. Not nice. But no better solution.
    fstream << m_onlyUpToRow;
    fstream << m_onlyUpToT;
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    Base::Load(fstream, modelVersion);
    fstream >> m_message;
    fstream >> m_logFirst;
    fstream >> m_logFrequency;
    fstream >> m_logGradientToo;
    m_formattingOptions.Load(fstream, modelVersion);
    fstream >> m_onlyUpToRow;
    fstream >> m_onlyUpToT;
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::BeginForwardProp() /*override*/
{
    Base::BeginForwardProp();
    ++m_numMBsRun;
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::ForwardProp(const FrameRange& fr) /*override*/
{
    size_t rank = DetermineElementwiseTensorRank();
    auto result =           ValueTensorFor(rank, fr);
    auto input  = Input(0)->ValueTensorFor(rank, fr);
    result.AssignCopyOf(input);
    // log the content
    if (m_numMBsRun == 1)
    {
        const auto prologue = m_formattingOptions.Processed(NodeName(), m_formattingOptions.prologue, m_numMBsRun);
        fprintf(stderr, "%s", prologue.c_str());
    }
    if (m_numMBsRun <= m_logFirst || (m_logFrequency && (m_numMBsRun-1) % m_logFrequency == 0))
    {
        char formatChar = !m_formattingOptions.isCategoryLabel ? 'f' : !m_formattingOptions.labelMappingFile.empty() ? 's' : 'u';
        auto valueFormatString = "%" + m_formattingOptions.precisionFormat + formatChar; // format string used in fprintf() for formatting the values
        const auto sequenceSeparator = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sequenceSeparator, m_numMBsRun);
        const auto sequencePrologue  = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sequencePrologue,  m_numMBsRun);
        const auto sequenceEpilogue  = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sequenceEpilogue,  m_numMBsRun);
        const auto elementSeparator  = m_formattingOptions.Processed(NodeName(), m_formattingOptions.elementSeparator,  m_numMBsRun);
        const auto sampleSeparator   = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sampleSeparator,   m_numMBsRun);

        let timeRange = fr.GetTimeRange();
        fprintf(stderr, "------- Trace["); // --- for better visual separability from actual content
        if (fr.IsAllFrames())
            fprintf(stderr, "*");
        else if (timeRange.second == timeRange.first+1)
            fprintf(stderr, "%d", (int)timeRange.first);
        else if (timeRange.second == timeRange.first + 1)
            fprintf(stderr, "%d..%d", (int)timeRange.first, (int)timeRange.second-1);
        fprintf(stderr, "] %ls --> %s\n", m_message.c_str(), Input(0)->FormatOperationPrototype("").c_str());
        Input(0)->WriteMinibatchWithFormatting(stderr, fr, m_onlyUpToRow, m_onlyUpToT, m_formattingOptions.transpose, m_formattingOptions.isCategoryLabel, m_formattingOptions.isSparse, m_labelMapping,
                                               sequenceSeparator, sequencePrologue, sequenceEpilogue, elementSeparator, sampleSeparator,
                                               valueFormatString, /*outputGradient=*/false);
    }
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr) /*override*/
{
    assert(inputIndex == 0); inputIndex;

    size_t rank = DetermineElementwiseTensorRank();
    auto sliceOutputGrad =           GradientTensorFor(rank, fr);      // propagate from this one...
    auto sliceInputGrad  = Input(0)->GradientTensorFor(rank, fr);      // ...to this one

    sliceInputGrad.AddCopyOf(sliceOutputGrad);
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Validate(bool isFinalValidationPass) // override
{
    ValidateUnaryMap(isFinalValidationPass);
    if (isFinalValidationPass)
    {
        if (m_labelMapping.empty() && (m_formattingOptions.isCategoryLabel || m_formattingOptions.isSparse) && !m_formattingOptions.labelMappingFile.empty())
            File::LoadLabelFile(m_formattingOptions.labelMappingFile, m_labelMapping);
    }
    m_numMBsRun = 0;
}

template class TraceNode<float>;
template class TraceNode<double>;

}}}
