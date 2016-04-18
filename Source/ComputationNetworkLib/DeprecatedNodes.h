//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// SumColumnElements (input)
// Sums up all elements in each sample (column) of the input. Every sample
// will be reduced to a scalar. This is equivalent to multiplying with a row of ones.
// This is deprecated, in favor of ReduceElements().
// -----------------------------------------------------------------------

template <class ElemType>
class SumColumnElementsNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"SumColumnElements"; }

public:
    DeclareConstructorFromConfigWithNumInputs(SumColumnElementsNode);
    SumColumnElementsNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        auto sliceInputValue  = Input(0)->ValueFor(fr);
        auto sliceOutputValue =           ValueFor(fr); // row vector

        Matrix<ElemType>::VectorSum(sliceInputValue, sliceOutputValue, true);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        auto sliceInputGrad  = Input(0)->GradientFor(fr);
        auto sliceOutputGrad =           GradientFor(fr);

        sliceInputGrad += sliceOutputGrad; // here the assumption is that sliceOutputGrad is a row vector
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        SetDims(TensorShape(1), Input(0)->HasMBLayout()); // each column is reduced to a scalar
    }
};

template class SumColumnElementsNode<float>;
template class SumColumnElementsNode<double>;

}}}
