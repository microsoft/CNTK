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

// -----------------------------------------------------------------------
// (deprecated) PerDimMeanVarNormalizationNode (feature, mean, invStdDev)
// Computes
//   output = (feature - mean) .* invStdDev
// where mean and invStdDev are meant to be single elements while features
// is minibatch data.
// Deprecated since it can be trivially expressed in BrainScript.
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

}}}
