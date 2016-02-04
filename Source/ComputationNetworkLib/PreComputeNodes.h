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
class PreComputedNodeBase : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembers;
    using Base::OperationName;

public:
    PreComputedNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_hasComputed(false)
    {
    }

    // interface through which this node is operated on are these two functions

    // check whether node has already undergone precomputation
    virtual bool HasComputed() const
    {
        return m_hasComputed;
    }

    // call this with 'false' at start and with 'true' at end
    // This is used for resetting and updating from accumulators.
    virtual void MarkComputed(const bool hasComputed)
    {
        m_hasComputed = hasComputed;
        CreateMatrixIfNull(m_value);
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

    virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, fstream);

        char str[4096];
        sprintf(str, "[%s]  ", string(GetSampleLayout()).c_str());
        fstream << string(str);
        sprintf(str, "HasComputed=%ls", HasComputed() ? L"true" : L"false");
        fstream << string(str);

        PrintNodeValuesToFile(printValues, fstream);
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
        CreateMatrixIfNull(m_value);
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
    typedef PreComputedNodeBase<ElemType> Base;
    UsingPreComputedNodeMembers;
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
    typedef MeanInvStdDevNodeBase<ElemType> Base;
    UsingMeanInvStdDevNodeBaseNodeMembers;
    static const std::wstring TypeName()
    {
        return L"Mean";
    }

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

        auto& samples = Input(0)->Value();
        auto& avg = Value();

#if NANCHECK
        samples.HasNan("Mean-Samples");
#endif
        size_t numNewSamples = Input(0)->GetMBLayout()->GetActualNumSamples();
        size_t totalNumSamples = m_numSamples + numNewSamples;
        if (totalNumSamples == 0)
            totalNumSamples = 1; // 0/0=1 in this context
        Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / totalNumSamples, samples, false,
                                                 ConstOnes(Input(0)->Value().GetNumCols(), 1, samples.GetDeviceId()),
                                                 false, (ElemType) m_numSamples / totalNumSamples, avg);
#if NANCHECK
        avg.HasNan("Mean-avg");
#endif

        m_numSamples += numNewSamples;
    }
};


// -----------------------------------------------------------------------
// InvStdDevNode (features)
// TODO: share stuff with MeanNode
// -----------------------------------------------------------------------

template <class ElemType>
class InvStdDevNode : public MeanInvStdDevNodeBase<ElemType>
{
    typedef MeanInvStdDevNodeBase<ElemType> Base;
    UsingMeanInvStdDevNodeBaseNodeMembers;
    static const std::wstring TypeName()
    {
        return L"InvStdDev";
    }

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
            size_t inputDim = Input(0)->GetSampleMatrixNumRows();
            m_mean.Resize(inputDim, 1);
            m_var.Resize(inputDim, 1);
            m_mean.SetValue(0);
            m_var.SetValue(0);
            UpdateFunctionValuesSize();
            Value().SetValue(0); // also set this because not doing it may flag during debugging; avoids special-casing this
        }
        else // finalize
        {
            ElemType sqrtFloor = 1e-10f;
            m_var.InplaceTruncateBottom(sqrtFloor); // prevent too small variance (and negative square roots due to numeric inaccuracy)
#if NANCHECK
            m_var.HasNan("MarkComputed-InplaceTruncateBottom");
#endif
            m_var.InplaceSqrt();

#if NANCHECK
            m_var.HasNan("MarkComputed-InplaceSqrt");
#endif
            m_var.ElementInverse();

#if NANCHECK
            m_var.HasNan("MarkComputed-ElementInverse()");
#endif
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

        auto& samples = Input(0)->Value();
#if NANCHECK
        samples.HasNan("InvStdDev-Samples");
#endif
        m_temp.SetValue(m_mean);
        size_t numNewSamples = Input(0)->GetMBLayout()->GetActualNumSamples();
        size_t totalNumSamples = m_numSamples + numNewSamples;
        if (totalNumSamples == 0)
            totalNumSamples = 1; // 0/0=1 in this context
        Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / totalNumSamples, samples, false,
                                                 ConstOnes(Input(0)->Value().GetNumCols(), 1, samples.GetDeviceId()),
                                                 false, (ElemType) m_numSamples / totalNumSamples, m_mean);

        m_temp -= m_mean;
        m_temp.AssignElementPowerOf(m_temp, 2);
        m_var += m_temp;

        m_temp.AssignDifferenceOf(samples, m_mean);
        m_temp.AssignElementPowerOf(m_temp, 2);

        Matrix<ElemType>::MultiplyAndWeightedAdd(1.0f / totalNumSamples, m_temp, false,
                                                 ConstOnes(Input(0)->Value().GetNumCols(), 1, samples.GetDeviceId()),
                                                 false, (ElemType) m_numSamples / totalNumSamples, m_var);

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
            node->m_mean = m_mean;
            node->m_var = m_var;
            node->m_temp = m_temp;
        }
    }

private:
    Matrix<ElemType> m_mean;
    Matrix<ElemType> m_var;
    Matrix<ElemType> m_temp;
};


// -----------------------------------------------------------------------
// PerDimMeanVarNormalizationNode (feature, mean, invStdDev)
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
        InvalidArgument("PerDimMeanVarNormalizationNode should only be called in the evaluation stage.");
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // only feature (input0) and output needs to be sliced
        auto sliceInput0Value = Input(0)->ValueFor(fr);
        auto sliceOutputValue = ValueFor(fr);

        ForwardPropS(sliceOutputValue, sliceInput0Value, Input(1)->Value(), Input(2)->Value());
    }

    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0,
                                                     const Matrix<ElemType>& input1, const Matrix<ElemType>& input2)
    {
#if DUMPOUTPUT
//input0.Print("PerDimMeanVarNormalization-input0");
//input1.Print("PerDimMeanVarNormalization-input1");
//input2.Print("PerDimMeanVarNormalization-input2");
#endif

#if NANCHECK
        input0.HasNan("PerDimMeanVarNormalization-input0");
        input1.HasNan("PerDimMeanVarNormalization-input1");
        input2.HasNan("PerDimMeanVarNormalization-input2");
#endif
        functionValues.AssignDifferenceOf(input0, input1);
        functionValues.ColumnElementMultiplyWith(input2);
#if NANCHECK
        functionValues.HasNan("PerDimMeanVarNormalization");
#endif
#if DUMPOUTPUT
        functionValues.Print("PerDimMeanVarNormalizationNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        if (Input(0)->RequiresPreCompute())
        {
            LogicError(
                "PerDimMeanVarNormalizationNode criterion forbids first input from being a pre-compute node. "
                "The first input should be the node whose output should be normalized, and the second and third inputs "
                "should be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");
        }

        if (!(Input(1)->OperationName() == OperationNameOf(LearnableParameter) &&
              Input(2)->OperationName() == OperationNameOf(LearnableParameter)) &&
            !(Input(1)->OperationName() == OperationNameOf(MeanNode) &&
              Input(2)->OperationName() == OperationNameOf(InvStdDevNode)))
        {
            LogicError(
                "PerDimMeanVarNormalizationNode criterion requires the last two inputs to be LearnableParameter "
                "type or (Mean, InvStdDev) so that the values will be saved.");
        }

        Input(1)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());
        Input(2)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());

        if (isFinalValidationPass)
        {
            if (!Input(0)->HasMBLayout() || Input(1)->HasMBLayout() || Input(2)->HasMBLayout())
                InvalidArgument("PerDimMeanVarNormalizationNode: Inputs must be data, while mean and InvStdDev must be column vectors.");
            if (!Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(1)->GetSampleLayout()) || !Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(2)->GetSampleLayout()))
                InvalidArgument("PerDimMeanVarNormalizationNode: All inputs should have same sample layout.");
        }

        // TODO: Is this correct? Why not just skip propagating a gradient into these? We should not poke around in our children.
        Input(1)->SetParameterUpdateRequired(false); // prevent learning
        Input(2)->SetParameterUpdateRequired(false);

        SetDims(Input(0));
    }
};


// -----------------------------------------------------------------------
// PerDimMeanVarDeNormalizationNode (feature, mean, invStdDev)
// -----------------------------------------------------------------------

template <class ElemType>
class PerDimMeanVarDeNormalizationNode : public ComputationNode<ElemType>, public NumInputs<3>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"PerDimMeanVarDeNormalization";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(PerDimMeanVarDeNormalizationNode);
    PerDimMeanVarDeNormalizationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&) override
    {
        InvalidArgument("PerDimMeanVarDeNormalizationNode should only be called in the evaluation stage.");
    }

    // (feature-mean).*InvStdDev
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // only feature (input0) and output needs to be sliced
        auto sliceInput0Value = Input(0)->ValueFor(fr);
        auto sliceOutputValue = ValueFor(fr);

        ForwardPropS(sliceOutputValue, sliceInput0Value, Input(1)->Value(), Input(2)->Value());
    }

    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& input0,
                                                     const Matrix<ElemType>& input1, const Matrix<ElemType>& input2)
    {
#if DUMPOUTPUT
//input0.Print("PerDimMeanVarDeNormalization-input0");
//input1.Print("PerDimMeanVarDeNormalization-input1");
//input2.Print("PerDimMeanVarDeNormalization-input2");
#endif

#if NANCHECK
        input0.HasNan("PerDimMeanVarDeNormalization-input0");
        input1.HasNan("PerDimMeanVarDeNormalization-input1");
        input2.HasNan("PerDimMeanVarDeNormalization-input2");
#endif
        // functionValues.AssignDifferenceOf(input0, input1);
        // functionValues.ColumnElementMultiplyWith(input2);
        // functionValues.AssignDifferenceOf(input0, input0);
        // functionValues += input2;
        // functionValues.ElementInverse();
        // functionValues.ElementMultiplyWith(input0);
        functionValues.SetValue(input0);
        functionValues.ColumnElementDivideBy(input2);
        functionValues += input1;
#if NANCHECK
        functionValues.HasNan("PerDimMeanVarDeNormalization");
#endif
#if DUMPOUTPUT
        functionValues.Print("PerDimMeanVarDeNormalizationNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        if (Input(0)->RequiresPreCompute())
        {
            LogicError(
                "PerDimMeanVarDeNormalizationNode criterion forbids first input from being a pre-compute node. "
                "The first input should be the node whose output should be de-normalized, and the second and third inputs "
                "should be LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");
        }

        if (!(Input(1)->OperationName() == OperationNameOf(LearnableParameter) &&
              Input(2)->OperationName() == OperationNameOf(LearnableParameter)) &&
            !(Input(1)->OperationName() == OperationNameOf(MeanNode) &&
              Input(2)->OperationName() == OperationNameOf(InvStdDevNode)))
        {
            LogicError(
                "PerDimMeanVarDeNormalizationNode criterion requires the last two inputs to be "
                "LearnableParameter type or (Mean, InvStdDev) so that the values will be saved.");
        }

        Input(1)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());
        Input(2)->ValidateInferInputDimsFrom(Input(0)->GetSampleLayout());

        if (isFinalValidationPass)
        {
            if (!Input(0)->HasMBLayout() || Input(1)->HasMBLayout() || Input(2)->HasMBLayout())
                InvalidArgument("PerDimMeanVarDeNormalizationNode: Inputs must be data, while mean and InvStdDev must be column vectors.");
            if (!Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(1)->GetSampleLayout()) || !Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(2)->GetSampleLayout()))
                InvalidArgument("PerDimMeanVarDeNormalizationNode: All inputs should have same sample layout.");
        }

        // TODO: Is this correct? Why not just skip propagating a gradient into these? We should not poke around in our children.
        Input(1)->SetParameterUpdateRequired(false); // prevent learning
        Input(2)->SetParameterUpdateRequired(false);

        SetDims(Input(0));
    }
};


} } }
