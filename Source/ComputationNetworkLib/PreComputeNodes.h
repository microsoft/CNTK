//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "Matrix.h"

#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <string>

// this file will contain computation nodes that require several atomic computation.

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// PreComputedNodeBase
// base class for nodes requiring pre-computation
// -----------------------------------------------------------------------

template <class ElemType>
class PreComputedNodeBase : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public IPreComputeNode
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembers;
    using Base::OperationName;

public:
    PreComputedNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_hasComputed(false)
    {
        MarkValueNonSharable();
    }

    // interface through which this node is operated on are these two functions

    // check whether node has already undergone precomputation
    virtual bool /*IPreComputeNode::*/ HasComputed() const override { return m_hasComputed; }

    // call this with 'false' at start and with 'true' at end
    // This is used for resetting and updating from accumulators.
    virtual void /*IPreComputeNode::*/ MarkComputed(const bool hasComputed) override
    {
        if (!Environment().IsPreComputing())
            LogicError("MarkComputed: Network must be in preComputing mode.");
        m_hasComputed = hasComputed;
    }

    virtual bool RequiresPreCompute() const override { return true; }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_hasComputed;
        fstream << Value();
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_hasComputed;
        LoadValue(fstream);
        // Note: This loses the sample layout, but that is recovered by Validate().
    }

    virtual void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        if (printMetadata)
        {
            char str[4096];
            sprintf(str, "[%s]  ", string(GetSampleLayout()).c_str());
            fstream << string(str);
            sprintf(str, "HasComputed=%ls", HasComputed() ? L"true" : L"false");
            fstream << string(str);
        }

        PrintNodeValuesToFile(printValues, printMetadata, fstream);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && !Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation requires its input to come in minibatches of samples.", NodeName().c_str(), OperationName().c_str());

        m_pMBLayout = nullptr; // this node does not hold mini-batch data
        SetDims(Input(0)->GetSampleLayout(), false);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<PreComputedNodeBase<ElemType>>(nodeP);
            node->m_hasComputed = m_hasComputed;
        }
    }

    // this is for the special-purpose "convertdbn" command (initialize values directly from another well-trained model)
    virtual void SideLoadFromMatrix(const Matrix<ElemType>& value)
    {
        if (value.GetNumCols() != 1)
            InvalidArgument("SideLoadFromMatrix: Side-loading is only supported for column vectors.");
        m_value->SetValue(value);
        m_hasComputed = true;
        SetDims(TensorShape(value.GetNumRows()), false);
    }

public:
    bool m_hasComputed;
};

#define UsingPreComputedNodeMembers \
    UsingComputationNodeMembers;    \
    using Base::m_hasComputed;      \
    using Base::OperationName

// -----------------------------------------------------------------------
// MeanInvStdDevNodeBase (features)  -- common base class for Mean and InvStdDev
// -----------------------------------------------------------------------

template <class ElemType>
class MeanInvStdDevNodeBase : public PreComputedNodeBase<ElemType>, public NumInputs<1>
{
    typedef PreComputedNodeBase<ElemType> Base; UsingPreComputedNodeMembers;
    // static const std::wstring TypeName() { return L"MeanInvStdDev (base)"; }
public:
    // DeclareConstructorFromConfigWithNumInputs(MeanInvStdDevNodeBase);
    MeanInvStdDevNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : PreComputedNodeBase<ElemType>(deviceId, name),
          m_numSamples(SIZE_MAX)
    {
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        m_numSamples = SIZE_MAX;
    }

    // this is used by the special-purpose command "convertdbn".
    virtual void SideLoadFromMatrix(const Matrix<ElemType>& m)
    {
        Base::SideLoadFromMatrix(m);
        m_numSamples = SIZE_MAX;
    }

    virtual void /*PreComputedNodeBase::*/ MarkComputed(const bool hasComputed, size_t numSamples = 0)
    {
        Base::MarkComputed(hasComputed);
        if (!m_hasComputed) // initialize
        {
            if (IsAccumulating())
                LogicError("%ls %ls operation: MarkComputed(false) has been called while accumulating.", NodeName().c_str(), OperationName().c_str());
            m_numSamples = 0;
        }
        else // finalize
        {
            if (!IsAccumulating())
                LogicError("%ls %ls operation: MarkComputed(true) has been called without MarkComputed(false) first.", NodeName().c_str(), OperationName().c_str());
            if (m_numSamples == 0)
                LogicError("%ls %ls operation: No data accumulated during precomputation.", NodeName().c_str(), OperationName().c_str());
            m_numSamples = SIZE_MAX;
        }
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        // LogicError("Mean operation should not be involved in the gradient calculation.");
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            if (m_numSamples != SIZE_MAX)
                LogicError("%ls %ls operation: CopyTo() called while accumulating.", NodeName().c_str(), OperationName().c_str());
            auto node = dynamic_pointer_cast<MeanInvStdDevNodeBase<ElemType>>(nodeP);
            node->m_numSamples = SIZE_MAX;
        }
    }

protected:
    size_t m_numSamples; // (SIZE_MAX while outside accumulation state)
    bool IsAccumulating() const { return m_numSamples != SIZE_MAX; }
};

#define UsingMeanInvStdDevNodeBaseNodeMembers \
    ComputationNodeBoilerplate;               \
    UsingPreComputedNodeMembers;              \
    using Base::m_numSamples;                 \
    using Base::IsAccumulating

// -----------------------------------------------------------------------
// MeanNode (features)
// -----------------------------------------------------------------------

template <class ElemType>
class MeanNode : public MeanInvStdDevNodeBase<ElemType>
{
    typedef MeanInvStdDevNodeBase<ElemType> Base; UsingMeanInvStdDevNodeBaseNodeMembers;
    static const std::wstring TypeName() { return L"Mean"; }

public:
    DeclareConstructorFromConfigWithNumInputs(MeanNode);
    MeanNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    MeanNode(DEVICEID_TYPE deviceId, const wstring& name, size_t)
        : Base(deviceId, name)
    {
    }

    virtual void /*PreComputedNodeBase::*/ MarkComputed(const bool hasComputed)
    {
        Base::MarkComputed(hasComputed);
        if (!m_hasComputed) // initialize accumulation
        {
            UpdateFunctionValuesSize();
            Value().SetValue(0);
        }
        // no else branch because ForwardPropNonLooping() already leaves a valid mean in m_value
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        if (m_hasComputed)
            return; // not accumulating

        if (!IsAccumulating())
            LogicError("%ls %ls operation: MarkComputed(false) has not been called.", NodeName().c_str(), OperationName().c_str());

        size_t rank = DetermineElementwiseTensorRank();
        auto mean  =             ValueTensorFor(rank, FrameRange()); // mean is formed directly in our m_value
        auto input = InputRef(0).ValueTensorFor(rank, fr);

        UpdateRunningAverage(InputRef(0), mean, m_numSamples);
    }
};

template class MeanNode<float>;
template class MeanNode<double>;

// -----------------------------------------------------------------------
// InvStdDevNode (features)
// TODO: share stuff with MeanNode
// -----------------------------------------------------------------------

template <class ElemType>
class InvStdDevNode : public MeanInvStdDevNodeBase<ElemType>
{
    typedef MeanInvStdDevNodeBase<ElemType> Base; UsingMeanInvStdDevNodeBaseNodeMembers;
    static const std::wstring TypeName() { return L"InvStdDev"; }

public:
    DeclareConstructorFromConfigWithNumInputs(InvStdDevNode);
    InvStdDevNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_mean(make_shared<Matrix<ElemType>>(deviceId)),
          m_var (make_shared<Matrix<ElemType>>(deviceId)),
          m_temp(make_shared<Matrix<ElemType>>(deviceId)),
          m_inputTemp(make_shared<Matrix<ElemType>>(deviceId))
    {
    }

    virtual void /*PreComputedNodeBase::*/ MarkComputed(const bool hasComputed) override
    {
        Base::MarkComputed(hasComputed);

        if (!m_hasComputed) // initialize
        {
            // reset accumulators
            UpdateFunctionValuesSize();
            m_mean->Resize(Value()); // mean accumulator normalized by #samples in it
            m_var ->Resize(Value()); // likewise the variance
            m_temp->Resize(Value()); // and a temp
            m_mean->SetValue(0);  // reset the mean and var accumulators
            m_var ->SetValue(0);
            Value().SetValue(0); // and clear m_value as well: We must do this here already to avoid a NaN check to flag while this is being estimated.
        }
        else // finalize
        {
            // m_value <- 1/stddev
            ElemType sqrtFloor = 1e-10f;
            m_var->InplaceTruncateBottom(sqrtFloor); // prevent too small variance (and negative square roots due to numeric inaccuracy)
            m_var->InplaceSqrt();
            m_var->ElementInverse();
            Value().SetValue(*m_var);
        }
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        if (m_hasComputed)
            return; // not accumulating

        if (!IsAccumulating())
            LogicError("%ls %ls operation: MarkComputed(false) has not been called.", NodeName().c_str(), OperationName().c_str());

        // set gaps to zero, since we are reducing in time
        InputRef(0).MaskMissingValueColumnsToZero(fr);
        m_inputTemp->Resize(InputRef(0).Value()); // same size of input
        m_inputTemp->SetValue(0);

        size_t numNewSamples = InputRef(0).GetMBLayout()->GetActualNumSamples();
        size_t totalNumSamples = m_numSamples + numNewSamples;
        if (totalNumSamples == 0)
            totalNumSamples = 1; // 0/0=1 in this context
        ElemType alpha =                   1.0f / totalNumSamples;
        ElemType beta  = (ElemType)m_numSamples / totalNumSamples;

        size_t rank = DetermineElementwiseTensorRank();
        auto input     = InputRef(0).ValueTensorFor(        rank, fr);
        auto mean      =            DataTensorFor(m_mean, rank, FrameRange());
        auto temp      =            DataTensorFor(m_temp, rank, FrameRange());
        auto var       =            DataTensorFor(m_var,  rank, FrameRange());
        auto inputTemp = TensorView<ElemType>(m_inputTemp, InputRef(0).GetTensorSliceFor(rank, fr));

        // preserve the old mean value for the next step
        temp.AssignCopyOf(mean);

        // accumulate the mean
        mean.DoCopyOf(beta, input, alpha); // Note: This reduces over samples.

        // compute the correction term
        // var += (oldMean - newMean)^2
        temp.AddCopyOf(mean, -1.0f); // subtract new 'mean' from the old one
        var.AddSqrOf(temp);          // add the square

        // var += (input - mean)^2
        inputTemp.AssignSqrOfDifferenceOf(input, mean, 1.0f); // this doesn't reduce yet.
        MaskMissingColumnsToZero(*m_inputTemp, InputRef(0).GetMBLayout(), fr); // set gaps to zero.
        var.DoCopyOf(beta, inputTemp, alpha); // now reduce

        m_numSamples += InputRef(0).GetMBLayout()->GetActualNumSamples();
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<InvStdDevNode<ElemType>>(nodeP);
            node->m_mean->SetValue(*m_mean);
            node->m_var ->SetValue(*m_var);
            node->m_temp->SetValue(*m_temp);
        }
    }

private:
    shared_ptr<Matrix<ElemType>> m_mean;
    shared_ptr<Matrix<ElemType>> m_var;
    shared_ptr<Matrix<ElemType>> m_temp;
    shared_ptr<Matrix<ElemType>> m_inputTemp;
};

template class InvStdDevNode<float>;
template class InvStdDevNode<double>;

// -----------------------------------------------------------------------
// PerDimMeanVarDeNormalizationNode (feature, mean, invStdDev)
// Computes
//   output = feature ./ invStdDev + mean
// with parameters the same as PerDimMeanVarNormalizationNode.
// TODO: Deprecate like PerDimMeanVarNormalizationNode as soon as we have a test case. Or just delete it.
// -----------------------------------------------------------------------

template <class ElemType>
class PerDimMeanVarDeNormalizationNode : public ComputationNode<ElemType>, public NumInputs<3>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"PerDimMeanVarDeNormalization"; }

public:
    DeclareConstructorFromConfigWithNumInputs(PerDimMeanVarDeNormalizationNode);
    PerDimMeanVarDeNormalizationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&) override
    {
        InvalidArgument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage. Is any of its descendents a learnable parameter that requires gradient?");
    }

    // feature ./ invStdDev + mean
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto output    =             ValueTensorFor(rank, fr);
        auto input     = InputRef(0).ValueTensorFor(rank, fr);
        auto mean      = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        auto invStdDev = InputRef(2).ValueTensorFor(rank, fr.AllowBroadcast());

        output.AssignElementwiseQuotientOf(input, invStdDev); // output = input / invStdDev
        output.AddCopyOf(mean);                               // output += mean
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        Input(1)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());
        Input(2)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());

        if (isFinalValidationPass)
        {
            if (!Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(1)->GetSampleLayout()) || !Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(2)->GetSampleLayout()))
                InvalidArgument("PerDimMeanVarDeNormalizationNode: All inputs should have same sample layout.");
        }

        SetDims(Input(0));
    }
};

template class PerDimMeanVarDeNormalizationNode<float>;
template class PerDimMeanVarDeNormalizationNode<double>;

}}}
