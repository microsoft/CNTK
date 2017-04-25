//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "LinearAlgebraNodes.h"

using namespace Microsoft::MSR::CNTK;

// -----------------------------------------------------------------------
// EpochAccumulatorNode calculates mean values of all samples used in forward pass.
// -----------------------------------------------------------------------

// TODO: can this be static?
template <class ElemType>
void Microsoft::MSR::CNTK::UpdateRunningAverage(ComputationNode<ElemType>& newInput,
                                                TensorView<ElemType>& runningAverage, size_t& runningCount)
{
    FrameRange fr(newInput.GetMBLayout());
    // Set gaps to zero, since we are reducing in time.
    newInput.MaskMissingValueColumnsToZero(fr);

    size_t newSamplesCount = newInput.GetMBLayout()->GetActualNumSamples();
    size_t totalSamplesCount = runningCount + newSamplesCount;
    if (totalSamplesCount == 0)
        totalSamplesCount = 1;
    ElemType alpha =                   1.0f / totalSamplesCount;
    ElemType beta = (ElemType) runningCount / totalSamplesCount;

    size_t rank = runningAverage.GetShape().GetRank();
    auto input = newInput.ValueTensorFor(rank, fr);

    // runningMean = beta * accumulator + alpha * input.
    runningAverage.DoCopyOf(beta, input, alpha);
    runningCount += newSamplesCount;
}

template void Microsoft::MSR::CNTK::UpdateRunningAverage<float>(ComputationNode<float>& newInput,
                                                                TensorView<float>& runningAverage,
                                                                size_t& runningCount);
template void Microsoft::MSR::CNTK::UpdateRunningAverage<double>(ComputationNode<double>& newInput,
                                                                 TensorView<double>& runningAverage,
                                                                 size_t& runningCount);

template <class ElemType>
EpochAccumulatorNode<ElemType>::EpochAccumulatorNode(DEVICEID_TYPE deviceId, const wstring& name)
    : Base(deviceId, name), m_numSamples(0)
{
    m_accumulator = make_shared<Matrix<ElemType>>(deviceId);
}

template <class ElemType>
EpochAccumulatorNode<ElemType>::EpochAccumulatorNode(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp)
    : EpochAccumulatorNode(configp->Get(L"deviceId"), L"<placeholder>")
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
}

template <class ElemType>
void EpochAccumulatorNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/)
{
    LogicError("%ls operation is used for forward only.", OperationName().c_str());
}

template <class ElemType>
void EpochAccumulatorNode<ElemType>::OnEpochStart()
{
    Reset();
}

template <class ElemType>
void EpochAccumulatorNode<ElemType>::ForwardPropNonLooping()
{
    TensorView<ElemType> accumulator = EnsureAccumlator();
    UpdateRunningAverage(InputRef(0), accumulator, m_numSamples);
    CopyAccumulatorToValue();
}

// Copies internal accumulator to the output.
template <class ElemType>
void EpochAccumulatorNode<ElemType>::CopyAccumulatorToValue()
{
    // Value gets resized in UpdateFunctionValuesSize that is called in BeforeForwardProp. Resize fills matrix with NaN
    // values, so m_value matrix cannot be used as persistent storage between ForwardProp calls.
    Value().SetValue(*m_accumulator);
}

template <class ElemType>
void EpochAccumulatorNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName,
                                            const CopyNodeFlags flags) const
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = nodeP->As<EpochAccumulatorNode<ElemType>>();
        node->m_numSamples = m_numSamples;
        node->m_accumulator->SetValue(*m_accumulator);
    }
}

template <class ElemType>
void EpochAccumulatorNode<ElemType>::Validate(bool isFinalValidationPass)
{
    Base::Validate(isFinalValidationPass);
    SetDims(Input(0)->GetSampleLayout(), HasMBLayout());
}

template <class ElemType>
TensorView<ElemType> EpochAccumulatorNode<ElemType>::EnsureAccumlator()
{
    if (m_accumulator->HasNoElements())
    {
        // Accumulator has not been resized yet, allocate with necessary size.
        const size_t sampleSize = GetSampleLayout().GetNumElements();
        m_accumulator->Resize(sampleSize, 1);
        Reset();
    }
    size_t rank = DetermineElementwiseTensorRank();
    return DataTensorFor(m_accumulator, rank, FrameRange());
}

template <class ElemType>
void EpochAccumulatorNode<ElemType>::Reset()
{
    m_accumulator->SetValue(0);
    m_numSamples = 0;
}

template class EpochAccumulatorNode<float>;
template class EpochAccumulatorNode<double>;