//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "Matrix.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <iostream>

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
        FrameRange fr(Input(0)->GetMBLayout());
        if (m_hasComputed)
            return; // not accumulating

        if (!IsAccumulating())
            LogicError("%ls %ls operation: MarkComputed(false) has not been called.", NodeName().c_str(), OperationName().c_str());

        // set gaps to zero, since we are reducing in time
        Input(0)->MaskMissingValueColumnsToZero(fr);

        size_t numNewSamples = Input(0)->GetMBLayout()->GetActualNumSamples();
        size_t totalNumSamples = m_numSamples + numNewSamples;
        if (totalNumSamples == 0)
            totalNumSamples = 1; // 0/0=1 in this context
        ElemType alpha =                   1.0f / totalNumSamples;
        ElemType beta  = (ElemType)m_numSamples / totalNumSamples;

        size_t rank = DetermineElementwiseTensorRank();
        auto mean  =           ValueTensorFor(rank, FrameRange()); // mean is formed directly in our m_value
        auto input = Input(0)->ValueTensorFor(rank, fr);

        mean.DoCopyOf(beta, input, alpha);
        // Note: We leverage that TensorView allows "broadcasting" the output,
        // which really means a reduction.

        m_numSamples += numNewSamples;
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
          m_mean(deviceId),
          m_var(deviceId),
          m_temp(deviceId)
    {
    }

    virtual void /*PreComputedNodeBase::*/ MarkComputed(const bool hasComputed) override
    {
        Base::MarkComputed(hasComputed);

        if (!m_hasComputed) // initialize
        {
            // reset accumulators
            UpdateFunctionValuesSize();
            m_mean.Resize(Value()); // mean accumulator normalized by #samples in it
            m_var .Resize(Value()); // likewise the variance
            m_temp.Resize(Value()); // and a temp
            m_mean.SetValue(0);  // reset the mean and var accumulators
            m_var .SetValue(0);
            Value().SetValue(0); // and clear m_value as well: We must do this here already to avoid a NaN check to flag while this is being estimated.
        }
        else // finalize
        {
            // m_value <- 1/stddev
            ElemType sqrtFloor = 1e-10f;
            m_var.InplaceTruncateBottom(sqrtFloor); // prevent too small variance (and negative square roots due to numeric inaccuracy)
            m_var.InplaceSqrt();
            m_var.ElementInverse();
            Value().SetValue(m_var);
        }
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        if (m_hasComputed)
            return; // not accumulating

        if (!IsAccumulating())
            LogicError("%ls %ls operation: MarkComputed(false) has not been called.", NodeName().c_str(), OperationName().c_str());

        // set gaps to zero, since we are reducing in time
        Input(0)->MaskMissingValueColumnsToZero(fr);

        size_t numNewSamples = Input(0)->GetMBLayout()->GetActualNumSamples();
        size_t totalNumSamples = m_numSamples + numNewSamples;
        if (totalNumSamples == 0)
            totalNumSamples = 1; // 0/0=1 in this context
        ElemType alpha =                   1.0f / totalNumSamples;
        ElemType beta  = (ElemType)m_numSamples / totalNumSamples;

        size_t rank = DetermineElementwiseTensorRank();
        auto input    = Input(0)->ValueTensorFor(        rank, fr);
        auto mean     =            DataTensorFor(m_mean, rank, FrameRange());
        auto temp     =            DataTensorFor(m_temp, rank, FrameRange());
        auto var      =            DataTensorFor(m_var,  rank, FrameRange());

        // preserve the old mean value for the next step
        temp.AssignCopyOf(mean);

        // accumulate the mean
        mean.DoCopyOf(beta, input, alpha); // Note: This reduces over samples.

        // compute the correction term
        // var += (oldMean - newMean)^2
        temp.AddCopyOf(mean, -1.0f); // subtract new 'mean' from the old one
        var.AddSqrOf(temp);          // add the square

        // var += (input - mean)^2
        var.DoSqrOfDifferenceOf(beta, input, mean, alpha); // this reduces as well

        m_numSamples += Input(0)->GetMBLayout()->GetActualNumSamples();
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<InvStdDevNode<ElemType>>(nodeP);
            node->m_mean.SetValue(m_mean);
            node->m_var.SetValue(m_var);
            node->m_temp.SetValue(m_temp);
        }
    }

private:
    Matrix<ElemType> m_mean;
    Matrix<ElemType> m_var;
    Matrix<ElemType> m_temp;
};

template class InvStdDevNode<float>;
template class InvStdDevNode<double>;

// -----------------------------------------------------------------------
// PerDimMeanVarNormalizationNode (feature, mean, invStdDev)
// Computes
//   output = (feature - mean) .* invStdDev
// where mean and invStdDev are meant to be single elements while features
// is minibatch data.
// TODO: Why do we need this? Why not use Plus and ElementTimes?
// -----------------------------------------------------------------------

template <class ElemType>
class PerDimMeanVarNormalizationNode : public ComputationNode<ElemType>, public NumInputs<3>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"PerDimMeanVarNormalization";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(PerDimMeanVarNormalizationNode);
    PerDimMeanVarNormalizationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&) override
    {
        InvalidArgument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage. Is any of its descendents a learnable parameter that requires gradient?");
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto output    =           ValueTensorFor(rank, fr);
        auto input     = Input(0)->ValueTensorFor(rank, fr);
        auto mean      = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto invStdDev = Input(2)->ValueTensorFor(rank, fr.AllowBroadcast());

        output.AssignDifferenceOf(input, mean);               // output = input - mean
        output.AssignElementwiseProductOf(output, invStdDev); // output *= invStdDev
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        Input(1)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());
        Input(2)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());


#if 1
        // support for legacy models when the mean and variance vectors were stored as column vectors (N,1)
        // This code will copy the shape of Input(0) (source) to Input(1) and Input(2) (target) if:
        //   1. The source is a 3-tensor with shape 1x1xM
        //   2. The target is a vector (i.e., a 2-tensor with shape Nx1)
        //   3. Both targets have the same number of elements
        //   4. The number of elements in the target (N) is the same as the number of elements in the source (M)
        // Note: This is somewhat ugly [Jasha Droppo].

        auto dimsA = Input(0)->GetSampleLayout().GetDims();
        auto dimsB = Input(1)->GetSampleLayout().GetDims();
        auto dimsC = Input(2)->GetSampleLayout().GetDims();

        if (
            // Test condition 1.
            (dimsA.size() == 3 && dimsA[0] == 1 && dimsA[1] == 1) &&
            // Test condition 2.
            (dimsB.size() == 2 && dimsB[1] == 1) &&
            (dimsC.size() == 2 && dimsC[1] == 1) &&
            // Test condition 3. and condition 4.
            (dimsB[0] == dimsC[0] && dimsB[0] == dimsA[2])
            )
        {
            // for error messages
            string dimsBstring = string(Input(1)->GetSampleLayout());
            string dimsCstring = string(Input(2)->GetSampleLayout());

            // reshape Input(1)
            Input(1)->SetDims(TensorShape(dimsA), false);
            fprintf(stderr, "\n%ls %ls operation: For legacy compatibility, the sample layout of second input (%ls %ls operation) was patched to [%s] (from [%s])\n",
                NodeName().c_str(), OperationName().c_str(), Input(1)->NodeName().c_str(), Input(1)->OperationName().c_str(), string(Input(1)->GetSampleLayout()).c_str(), dimsBstring.c_str());

            // reshape Input(2)
            Input(2)->SetDims(TensorShape(dimsA), false);
            fprintf(stderr, "\n%ls %ls operation: For legacy compatibility, the sample layout of third input (%ls %ls operation) was patched to [%s] (from [%s])\n",
                NodeName().c_str(), OperationName().c_str(), Input(2)->NodeName().c_str(), Input(2)->OperationName().c_str(), string(Input(2)->GetSampleLayout()).c_str(), dimsCstring.c_str());
        }

#endif

        if (isFinalValidationPass)
        {
            if (!Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(1)->GetSampleLayout()) || !Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(2)->GetSampleLayout()))
                InvalidArgument("PerDimMeanVarNormalizationNode: All inputs should have same sample layout.");
        }

        SetDims(Input(0));
    }
};

template class PerDimMeanVarNormalizationNode<float>;
template class PerDimMeanVarNormalizationNode<double>;

// -----------------------------------------------------------------------
// PerDimMeanVarDeNormalizationNode (feature, mean, invStdDev)
// Computes
//   output = feature ./ invStdDev + mean
// with parameters the same as PerDimMeanVarNormalizationNode.
// TODO: Why do we need this? Why not use Plus and ElementDividedBy?
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
        auto output    =           ValueTensorFor(rank, fr);
        auto input     = Input(0)->ValueTensorFor(rank, fr);
        auto mean      = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto invStdDev = Input(2)->ValueTensorFor(rank, fr.AllowBroadcast());

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
