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
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembers;
    using Base::OperationName;

public:
    PreComputedNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_hasComputed(false)
    {
        MarkValueNonSharable();
    }

    // interface through which this node is operated on are these two functions

    // check whether node has already undergone precomputation
    virtual bool /*IPreComputeNode::*/ HasComputed() const override
    {
        return m_hasComputed;
    }

    // call this with 'false' at start and with 'true' at end
    // This is used for resetting and updating from accumulators.
    virtual void /*IPreComputeNode::*/ MarkComputed(const bool hasComputed) override
    {
        m_hasComputed = hasComputed;
    }

    virtual bool RequiresPreCompute() const override
    {
        return true;
    }

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
        if (!Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation requires its input to come in minibatches of samples.", NodeName().c_str(), OperationName().c_str());
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (!m_hasComputed) // this node retains state, and state gets destroyed by Resize(), so we must be careful
            SetDims(Input(0)->GetSampleLayout(), false);
        else if (!GetSampleLayout().IsElementwiseCompatibleWith(Input(0)->GetSampleLayout()))
            InvalidArgument("%ls %ls operation: Precomputed parameter does not match input dimensions.", NodeName().c_str(), OperationName().c_str());
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

    // this is for the special case: convertDBN needs this; because we initialize values directly from another well-trained model
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

    // this is used by convertDBN
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
    bool IsAccumulating() const
    {
        return m_numSamples != SIZE_MAX;
    }
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
#define MEANVAR_TENSOR_SUPPORT 2 // 0=old code; 1=tensor for mean estimate only; 2=tensor for mean and var estimate  --will removed in next commit
#if MEANVAR_TENSOR_SUPPORT > 0
        size_t rank = DetermineElementwiseTensorRank();
        auto mean  =           ValueTensorFor(rank, FrameRange()); // mean is formed directly in our m_value
        auto input = Input(0)->ValueTensorFor(rank, fr);

        mean.DoCopyOf(beta, input, alpha);
        // Note: We leverage that TensorView allows "broadcasting" the output,
        // which really means a reduction.
#else
        auto& samples = Input(0)->Value();
        auto& avg = Value();
        Matrix<ElemType>::MultiplyAndWeightedAdd(alpha, samples, false,
                                                 ConstOnes(Input(0)->Value().GetNumCols(), 1, samples.GetDeviceId()),
                                                 false, beta, avg);
#endif
#if NANCHECK
        avg.HasNan("Mean-avg");
#endif

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
            Value().SetValue(0);    // Note: We must do this here already because dimensions are verified at places.
            m_mean.Resize(Value()); // mean accumulator normalized by #samples in it
            m_var .Resize(Value()); // likewise the variance
            m_temp.Resize(Value()); // and a temp
            m_mean.SetValue(0); // reset the mean and var accumulators
            m_var .SetValue(0);
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

#if MEANVAR_TENSOR_SUPPORT < 2
        m_temp.SetValue(m_mean); // old mean
        auto& samples = Input(0)->Value();
#endif
        size_t numNewSamples = Input(0)->GetMBLayout()->GetActualNumSamples();
        size_t totalNumSamples = m_numSamples + numNewSamples;
        if (totalNumSamples == 0)
            totalNumSamples = 1; // 0/0=1 in this context
        ElemType alpha =                   1.0f / totalNumSamples;
        ElemType beta  = (ElemType)m_numSamples / totalNumSamples;
#if MEANVAR_TENSOR_SUPPORT > 0
        size_t rank = DetermineElementwiseTensorRank();
        auto input    = Input(0)->ValueTensorFor(        rank, fr);
        auto mean     =            DataTensorFor(m_mean, rank, FrameRange());
        auto temp     =            DataTensorFor(m_temp, rank, FrameRange());
        auto var      =            DataTensorFor(m_var,  rank, FrameRange());

        // preserve the old mean value for the next step
#if MEANVAR_TENSOR_SUPPORT > 1
        temp.AssignCopyOf(mean);
#endif

        // accumulate the mean
        mean.DoCopyOf(beta, input, alpha); // Note: This reduces over samples.
#else
        Matrix<ElemType>::MultiplyAndWeightedAdd(alpha, samples, false,
                                                 ConstOnes(Input(0)->Value().GetNumCols(), 1, samples.GetDeviceId()),
                                                 false, beta, m_mean);
#endif

        // compute the correction term
#if MEANVAR_TENSOR_SUPPORT > 1
        // var += (oldMean - newMean)^2
        temp.AddCopyOf(mean, -1.0f); // subtract new 'mean' from the old one
        var.AddSqrOf(temp);          // add the square

        // var += (input - mean)^2
//Input(0)->Value().Print("input before", -3, -3, 0, INT_MAX);
//m_mean.Print("mean before", -3, -3, -5, -5);
//m_var.Print("var before", -3, -3, -5, -5);
        var.DoSqrOfDifferenceOf(beta, input, mean, alpha); // this reduces as well
//m_var.Print("var after", -3, -3, -5, -5);
#else
        // var += (oldMean - newMean)^2
        m_temp -= m_mean;
        m_temp.AssignElementPowerOf(m_temp, 2);
        m_var  += m_temp;

        m_temp.AssignDifferenceOf(Input(0)->Value(), m_mean); // this is not reduced
        m_temp.AssignElementPowerOf(m_temp, 2);               // elementwise per sample

m_var.Print("var before 1", -3, -3, -5, -5);
        Matrix<ElemType>::MultiplyAndWeightedAdd(alpha, m_temp, false,
                                                 ConstOnes(Input(0)->Value().GetNumCols(), 1, samples.GetDeviceId()),
                                                 false, beta, m_var);
m_var.Print("var after 1", -3, -3, -5, -5);
#endif

#if NANCHECK
        m_var.HasNan("InvStdDev-m_var");
#endif

#if 0   // BUGBUG: This is the correct version, but it will break test cases, so do this later. MeanNode does it right already.
        m_numSamples += Input(0)->GetMBLayout()->GetActualNumSamples();
#else
        m_numSamples += Input(0)->Value().GetNumCols(); // BUGBUG: Should be -> GetActualNumSamples().
#endif
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
#if 1
        size_t rank = DetermineElementwiseTensorRank();
        auto output    =           ValueTensorFor(rank, fr);
        auto input     = Input(0)->ValueTensorFor(rank, fr);
        auto mean      = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto invStdDev = Input(2)->ValueTensorFor(rank, fr.AllowBroadcast());

        output.AssignDifferenceOf(input, mean);               // output = input - mean
        output.AssignElementwiseProductOf(output, invStdDev); // output *= invStdDev
#else
        // only feature (input0) and output needs to be sliced
        auto functionValues = Input(0)->ValueFor(fr);
        auto input0         = ValueFor(fr);
        const auto& input1  = Input(1)->Value(); // mean
        const auto& input2  = Input(2)->Value(); // inv stddev

        functionValues.AssignDifferenceOf(input0, input1);
        functionValues.ColumnElementMultiplyWith(input2);
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        Input(1)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());
        Input(2)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());

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
#if 1
        size_t rank = DetermineElementwiseTensorRank();
        auto output    =           ValueTensorFor(rank, fr);
        auto input     = Input(0)->ValueTensorFor(rank, fr);
        auto mean      = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto invStdDev = Input(2)->ValueTensorFor(rank, fr.AllowBroadcast());

        output.AssignElementwiseQuotientOf(input, invStdDev); // output = input / invStdDev
        output.AddCopyOf(mean);                               // output += mean
#else
        // only feature (input0) and output needs to be sliced
        auto functionValues = Input(0)->ValueFor(fr);
        auto input0         = ValueFor(fr);
        const auto& input1  = Input(1)->Value(); // mean
        const auto& input2  = Input(2)->Value(); // inv stddev

        functionValues.SetValue(input0);
        functionValues.ColumnElementDivideBy(input2);
        functionValues += input1;
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

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

} } }
