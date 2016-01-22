//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"
#include "TensorView.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>
#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef ENABLE_TENSORVIEW

// -----------------------------------------------------------------------
// UnaryElementWiseWithOpCodeNodeBase (input) -- base for elementwise unary op
// where forward // and backward are single ElementWiseOperator opcodes and
// only inputs (but not // function values) are used.
// -----------------------------------------------------------------------

template <class ElemType, ElementWiseOperator opForward, ElementWiseOperator opBackward, bool gradientFromOutput>
class UnaryElementWiseWithOpCodeNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    UnaryElementWiseWithOpCodeNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        //static int c = 0; if (c++ == 0) { fprintf(stderr, "#NLop%d#\n", (int)opForward); }

        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input = Input(0)->ValueTensorFor(rank, fr);
        result.DoUnaryOpOf(0, input, 1, opForward);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        assert(inputIndex == 0);
        inputIndex;

        // get the args
        size_t rank = DetermineElementwiseTensorRank();
        auto sliceOutputGrad = GradientTensorFor(rank, fr);               // propagate from this one...
        auto sliceInputGrad = Input(0)->GradientTensorFor(rank, fr);      // ...to this one
        auto sliceValue = gradientFromOutput ? ValueTensorFor(rank, fr) : // using input or output value
                              Input(0)->ValueTensorFor(rank, fr);
        // If gradient can be compute from output rather than input, then that's better for mem sharing (and faster in most cases).
        // Not possible for Cos().
        sliceInputGrad.DoBinaryOpOf(1, sliceOutputGrad, sliceValue, 1, opBackward);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return gradientFromOutput;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return !gradientFromOutput;
    }
};

#define UnaryElementWiseWithOpCodeNodeBaseMembers UsingComputationNodeMembersBoilerplate;

// -----------------------------------------------------------------------
// SigmoidNode (input)
// TanhNode (input)
// RectifiedLinearNode (input)
// LogNode (input)
// ExpNode (input)
// CosineNode (input)
// These are all implemented by single-opcode functions and can thus be declared by a macro.
// -----------------------------------------------------------------------

#pragma push_macro("DeclareUnaryTensorOp")
#define DeclareUnaryElementWiseWithOpCodeNode(Name, Forward, Backward, gradientFromOutput)                                \
    template <class ElemType>                                                                                             \
    class Name##Node : public UnaryElementWiseWithOpCodeNodeBase<ElemType, op##Forward, op##Backward, gradientFromOutput> \
    {                                                                                                                     \
        typedef UnaryElementWiseWithOpCodeNodeBase<ElemType, op##Forward, op##Backward, gradientFromOutput> Base;         \
        UnaryElementWiseWithOpCodeNodeBaseMembers;                                                                        \
        static const std::wstring TypeName()                                                                              \
        {                                                                                                                 \
            return L## #Name;                                                                                             \
        }                                                                                                                 \
                                                                                                                          \
    public:                                                                                                               \
        DeclareConstructorFromConfigWithNumInputs(Name##Node);                                                            \
        Name##Node(DEVICEID_TYPE deviceId, const wstring& Name)                                                           \
            : Base(deviceId, Name)                                                                                        \
        {                                                                                                                 \
        }                                                                                                                 \
    }

//                                    Name             Forward and      Backward opcodes
DeclareUnaryElementWiseWithOpCodeNode(Sigmoid, Sigmoid, ElementwiseProductWithSigmoidDerivativeFromOutput, true);
DeclareUnaryElementWiseWithOpCodeNode(Tanh, Tanh, ElementwiseProductWithTanhDerivativeFromOutput, true);
DeclareUnaryElementWiseWithOpCodeNode(RectifiedLinear, LinearRectifier, ElementwiseProductWithLinearRectifierDerivativeFromOutput, true);
DeclareUnaryElementWiseWithOpCodeNode(Log, Log, ElementwiseProductWithLogDerivativeFromOutput, true);
DeclareUnaryElementWiseWithOpCodeNode(Exp, Exp, ElementwiseProduct, true);
DeclareUnaryElementWiseWithOpCodeNode(Cosine, Cosine, ElementwiseProductWithCosDerivative, false);

#pragma pop_macro("DeclareUnaryTensorOp")
#endif

// -----------------------------------------------------------------------
// SoftmaxNodeBase (input) -- shared base of Softmax and LogSoftmax
// -----------------------------------------------------------------------

// shared base for all element-wise non-linearities
// What this adds over a ComputationNode<ElemType> is a member m_gradientTemp for temp use by derived classes.
// TODO: This was used more broadly, but no longer, so we may be able to simplify the signatures of the virtual functions.
template <class ElemType>
class SoftmaxNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    //virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
    DeclareConstructorFromConfigWithNumInputs(SoftmaxNodeBase);
    SoftmaxNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        assert(inputIndex == 0);
        inputIndex;

        // get the args
        // Some do not consume input and/or output values. Don't touch those, pass dummies instead, since memshare may have taken them away already.
        auto sliceOutputGrad = GradientFor(fr);          // propagate from this one...
        auto sliceInputGrad = Input(0)->GradientFor(fr); // ...to this one
        auto sliceInputValue = InputUsedInComputingInputNodesGradients(0) ? Input(0)->ValueFor(fr) : Matrix<ElemType>();
        auto sliceOutputValue = OutputUsedInComputingInputNodesGradients() ? ValueFor(fr) : Matrix<ElemType>();

        // do the actual operation
        BackpropToV(*m_gradientTemp, sliceInputValue, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
    }

    // derived class implement the actual non-linear operation
    virtual void BackpropToV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues) = 0;

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        auto values = ValueFor(fr);
        ForwardPropV(values, Input(0)->ValueFor(fr));
    }

    // derived class implement the actual non-linear operation
    virtual void ForwardPropV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) = 0;

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<SoftmaxNodeBase<ElemType>>(nodeP);
            *node->m_gradientTemp = *m_gradientTemp;
        }
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_gradientTemp, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_gradientTemp, matrixPool);
    }

protected:
    shared_ptr<Matrix<ElemType>> m_gradientTemp;
};

#define UsingSoftmaxNodeBaseMembers         \
    UsingComputationNodeMembersBoilerplate; \
    using Base::m_gradientTemp

// -----------------------------------------------------------------------
// SoftmaxNode (input) -- soft-max over input vector(s)
// -----------------------------------------------------------------------

//we assume it's  column-wise by default
//the derivative will increase the Matrix<ElemType> size to the power of column size and should not be used.
template <class ElemType>
class SoftmaxNode : public SoftmaxNodeBase<ElemType>
{
    typedef SoftmaxNodeBase<ElemType> Base;
    UsingSoftmaxNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"Softmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(SoftmaxNode);
    SoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The plus node does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    /*virtual*/ void BackpropToV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
    {
        Matrix<ElemType>& diff = *m_diff;
        gradient.AssignInnerProductOf(gradientValues, functionValues, true);
        diff.AssignDifferenceOf(gradientValues, gradient);

        inputGradientValues.AddElementProductOf(diff, functionValues);
    }

    /*virtual*/ void ForwardPropV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) override
    {
        functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
        functionValues.InplaceExp();
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<SoftmaxNode<ElemType>>(nodeP);
            *node->m_diff = *m_diff;
        }
    }
    //request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_diff, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_diff, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_diff;
};

template class SoftmaxNode<float>;
template class SoftmaxNode<double>;

// -----------------------------------------------------------------------
// LogSoftmaxNode (input) -- log of soft-max over input vector(s)
// -----------------------------------------------------------------------

template <class ElemType>
class LogSoftmaxNode : public SoftmaxNodeBase<ElemType>
{
    typedef SoftmaxNodeBase<ElemType> Base;
    UsingSoftmaxNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"LogSoftmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(LogSoftmaxNode);
    LogSoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The plus node does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    /*virtual*/ void BackpropToV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
    {
        Matrix<ElemType>& softmax = *m_softmax;
        softmax.AssignExpOf(functionValues);
        Matrix<ElemType>::VectorSum(gradientValues, gradient, true);
        softmax.RowElementMultiplyWith(gradient);
        Matrix<ElemType>::AddScaledDifference(1.0, gradientValues, softmax, inputGradientValues);
    }

    /*virtual*/ void ForwardPropV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) override
    {
        functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LogSoftmaxNode<ElemType>>(nodeP);
            *node->m_softmax = *m_softmax;
        }
    }
    //request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_softmax, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_softmax, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_softmax;
};

template class LogSoftmaxNode<float>;
template class LogSoftmaxNode<double>;

// -----------------------------------------------------------------------
// GMMLogLikelihoodNode (unnormedPrior, means, logStdDevs, features) -- GMM log LL over input vector(s)
// calculates the log likelihood of a feature given parameters of a Gaussian mixture model (GMM) with shared diagonal variance
//  - unnormedPrior: mix weights, #rows = #mixture components
//  - means: means, all mix means concatenated  (i.e. dim = feature dim x prior dim)
//  - logStdDevs: std deviations, pooled across mix (i.e. same dim as features)
// UnnormedPrior, means, and logStdDevs can be either a single column or one per sample, e.g.
// when parameters are computed by other nodes.
// -----------------------------------------------------------------------

template <class ElemType>
class GMMLogLikelihoodNode : public ComputationNode<ElemType>, public NumInputs<4>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"GMMLogLikelihood";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(GMMLogLikelihoodNode);
    GMMLogLikelihoodNode(DEVICEID_TYPE deviceId, const wstring& name)
        : ComputationNode<ElemType>(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // get the right slice
        const size_t colsPrior = Input(0)->GetSampleMatrixNumCols();

        Matrix<ElemType> sliceGradientValue = DataFor(*m_gradient, fr);
        Matrix<ElemType> slicePosterior = DataFor(*m_posterior, fr);

        switch (inputIndex)
        {
        case 0:
        {
            if (colsPrior == 1)
                BackpropToUnnormedPrior(Input(0)->Gradient(), sliceGradientValue, *m_prior, slicePosterior, *m_temp);
            else
            {
                Matrix<ElemType> sliceUnnormedPriorGradient = Input(0)->GradientFor(fr);
                Matrix<ElemType> slicePrior = DataFor(*m_prior, fr); // TODO: use the right MBLayout, then we won't need the special case
                BackpropToUnnormedPrior(sliceUnnormedPriorGradient, sliceGradientValue, slicePrior, slicePosterior, *m_temp);
            }
        }
        break;
        case 1:
        {
            Matrix<ElemType> sliceNormedDeviationVectors = DataFor(*m_normedDeviationVectors, fr);
            if (colsPrior == 1)
                BackpropToMean(Input(1)->Gradient(), sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
            else
            {
                Matrix<ElemType> sliceMeanGradient = Input(1)->GradientFor(fr);
                BackpropToMean(sliceMeanGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
            }
        }
        break;
        case 2:
        {
            Matrix<ElemType> sliceNormedDeviation = DataFor(*m_normedDeviation, fr);
            if (colsPrior == 1)
                BackpropToLogStddev(Input(2)->Gradient(), sliceGradientValue, sliceNormedDeviation, slicePosterior, *m_temp);
            else
            {
                Matrix<ElemType> sliceLotStddevGradient = Input(2)->GradientFor(fr);
                BackpropToLogStddev(sliceLotStddevGradient, sliceGradientValue, sliceNormedDeviation, slicePosterior, *m_temp);
            }
        }
        break;
        case 3:
        {
            Matrix<ElemType> sliceNormedDeviationVectors = DataFor(*m_normedDeviationVectors, fr);
            Matrix<ElemType> sliceFeatureGradient = Input(3)->GradientFor(fr);
            BackpropToFeature(sliceFeatureGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
        }
        break;
        default:
            InvalidArgument("GMMLogLikelihoodNode criterion only takes four inputs.");
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The GMMLogLikelihoodNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The GMMLogLikelihoodNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    void BackpropToUnnormedPrior(Matrix<ElemType>& unnormedPriorGradientValues, const Matrix<ElemType>& gradientValues,
                                 const Matrix<ElemType>& prior, const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        temp.AssignDifferenceOf(posterior, prior);
        temp.RowElementMultiplyWith(gradientValues);
        if (prior.GetNumCols() == posterior.GetNumCols())
            unnormedPriorGradientValues += temp;
        else if (prior.GetNumCols() == 1)
            Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(posterior.GetNumCols(), 1, unnormedPriorGradientValues.GetDeviceId()), false, unnormedPriorGradientValues);
        else
            RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
    }

    void BackpropToMean(Matrix<ElemType>& meanGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
                        Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        size_t numComponent = posterior.GetNumRows();
        size_t numSamples = posterior.GetNumCols();
        size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

        temp.SetValue(normedDeviationVectors); //recall normedDeviationVectors <-- (x-u_c)/(stddev^2)
        temp.Reshape(featureSize, numSamples * numComponent);

        posterior.Reshape(1, numSamples * numComponent);
        temp.RowElementMultiplyWith(posterior); //temp <-- posterior * (x-u_c)/(stddev^2)

        posterior.Reshape(numComponent, numSamples);          //reshape back
        temp.Reshape(featureSize * numComponent, numSamples); //reshape back

        temp.RowElementMultiplyWith(gradientValues);

        if (numSamples == meanGradientValues.GetNumCols())
            meanGradientValues += temp;
        else if (meanGradientValues.GetNumCols() == 1)
            Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, meanGradientValues.GetDeviceId()), false, meanGradientValues);
        else
            RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
    }

    void BackpropToLogStddev(Matrix<ElemType>& logStddevGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviation,
                             const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        size_t numComponent = posterior.GetNumRows();
        size_t numSamples = posterior.GetNumCols();

        temp.AssignDifferenceOf(normedDeviation, (ElemType) numComponent);
        temp.ElementMultiplyWith(posterior);
        temp.RowElementMultiplyWith(gradientValues);
        if (logStddevGradientValues.GetNumCols() == numSamples)
            logStddevGradientValues += temp;
        else if (logStddevGradientValues.GetNumCols() == 1)
            Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, logStddevGradientValues.GetDeviceId()), false, logStddevGradientValues);
        else
            RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
    }

    void BackpropToFeature(Matrix<ElemType>& featureGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
                           Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        size_t numComponent = posterior.GetNumRows();
        size_t numSamples = posterior.GetNumCols();
        size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

        temp.SetValue(normedDeviationVectors);
        temp *= -1;
        temp.Reshape(featureSize, numSamples * numComponent);
        posterior.Reshape(1, numSamples * numComponent);
        temp.RowElementMultiplyWith(posterior);

        posterior.Reshape(numComponent, numSamples);
        temp.Reshape(featureSize * numComponent, numSamples);
        temp.RowElementMultiplyWith(gradientValues);

        for (int i = 0; i < numComponent; i++)
            featureGradientValues.AddWithRowSliceValuesOf(temp, i * featureSize, featureSize);
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        size_t numCols = Input(3)->GetSampleMatrixNumCols();
        size_t numComponents = Input(0)->GetSampleMatrixNumRows();
        size_t colsPrior = Input(0)->GetSampleMatrixNumCols(); // may be 1
        size_t featureSize = Input(3)->GetSampleMatrixNumRows();

        m_prior->Resize(numComponents, colsPrior);
        m_stddev->Resize(numComponents, colsPrior);
        m_normedDeviation->Resize(numComponents, numCols);
        m_normedDeviationVectors->Resize(numComponents * featureSize, numCols);
        m_posterior->Resize(numComponents, numCols);
    }

    // input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t colsPrior = Input(0)->GetSampleMatrixNumCols();
        size_t numSamples = Input(3)->GetSampleMatrixNumCols();

        // get the right slice
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        Matrix<ElemType> sliceFeature = Input(3)->ValueFor(fr);
        Matrix<ElemType> sliceNormedDeviation = DataFor(*m_normedDeviation, fr);
        Matrix<ElemType> sliceNormedDeviationVectors = DataFor(*m_normedDeviationVectors, fr);
        Matrix<ElemType> slicePosterior = DataFor(*m_posterior, fr);

        if (colsPrior == 1)
        {
            ForwardPropS(sliceOutputValue, Input(0)->Value(), Input(1)->Value(), Input(2)->Value(), sliceFeature,
                         *m_prior, *m_stddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, *m_temp);
        }
        else if (colsPrior == numSamples)
        {
            Matrix<ElemType> sliceUnnormedPrior = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceMean = Input(1)->ValueFor(fr);
            Matrix<ElemType> sliceLogstddev = Input(2)->ValueFor(fr);

            Matrix<ElemType> slicePrior = DataFor(*m_prior, fr);
            Matrix<ElemType> sliceStddev = DataFor(*m_stddev, fr);

            ForwardPropS(sliceOutputValue, sliceUnnormedPrior, sliceMean, sliceLogstddev, sliceFeature,
                         slicePrior, sliceStddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, *m_temp);
        }
        else //should not reach the code since validation should fail already
            RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
    }

    //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
    //If we want to speed up we need to replace following code with a several specialized GPU functions
    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& unnormedPrior, const Matrix<ElemType>& mean, Matrix<ElemType>& logstddev,
                                                     const Matrix<ElemType>& feature, Matrix<ElemType>& prior, Matrix<ElemType>& stddev, Matrix<ElemType>& normedDeviationVectors,
                                                     Matrix<ElemType>& normedDeviation, Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        int numComponent = unnormedPrior.GetNumRows();
        size_t numSamples = feature.GetNumCols();
        size_t featureDim = feature.GetNumRows();

        //compute prior which is softmax of unnormedPrior
        prior.AssignLogSoftmaxOf(unnormedPrior, true); //log prior

        prior.InplaceExp();

        //compute stddev
        stddev.AssignExpOf(logstddev);

#if DUMPOUTPUT
        unnormedPrior.Print("unnormedPrior", 0, min(5, unnormedPrior.GetNumRows() - 1), 0, min(10, unnormedPrior.GetNumCols() - 1));
        mean.Print("mean", 0, min(5, mean.GetNumRows() - 1), 0, min(10, mean.GetNumCols() - 1));
        logstddev.Print("logstddev", 0, min(5, logstddev.GetNumRows() - 1), 0, min(10, logstddev.GetNumCols() - 1));

        prior.Print("prior", 0, min(5, prior.GetNumRows() - 1), 0, min(10, prior.GetNumCols() - 1));
        stddev.Print("stddev", 0, min(5, stddev.GetNumRows() - 1), 0, min(10, stddev.GetNumCols() - 1));
#endif

        //compute normedDeviation <-- ||x-u_c||^2/(stddev^2)
        normedDeviationVectors.AssignRepeatOf(feature, numComponent, 1);
        normedDeviationVectors -= mean;                                        //each column of the mean has multiple mean components
        normedDeviationVectors.Reshape(featureDim, numSamples * numComponent); //now each column is feature-mean_i

        normedDeviation.AssignVectorNorm2Of(normedDeviationVectors, true);
        normedDeviation ^= 2;
        temp.AssignRepeatOf(stddev, 1, numSamples / stddev.GetNumCols()); //stddev.GetNumCols() is either 1 or =numSamples
        temp.Reshape(1, temp.GetNumElements());                           //one stddev value for each component for each sample
        temp ^= 2;
        normedDeviation.ElementDivideBy(temp); //normedDeviation and temp have same dim (1, numSamples* numComponent)

        //compute  normedDeviationVectors <-- (x-u_c)/(stddev^2)
        normedDeviationVectors.RowElementDivideBy(temp);                       //divide twice
        normedDeviationVectors.Reshape(featureDim * numComponent, numSamples); //reshape back

        //compute per-component likelihood
        posterior.AssignProductOf(-0.5f, normedDeviation); //posterior  <-- -||x-u_c||^2/(stddev^2)/2 and in (1, numSamples* numComponent) dim
        temp.InplaceLog();
        temp *= ((ElemType) numComponent / 2.0f);                   //temp <-- stddev^c and in (1, numSamples* numComponent) dim
        posterior -= temp;                                          // posterior  <-- exp[-||x-u_c||^2/(stddev^2)/2]/(stddev^c)
        posterior -= (ElemType)(numComponent / 2.0f * log(TWO_PI)); //likelihood for each component and sample is now computed and stored in posterior
        posterior.InplaceExp();                                     //posterior  <-- exp(-||x-u_c||^2/(stddev^2)/2)

        normedDeviation.Reshape(numComponent, numSamples); //reshape back
        posterior.Reshape(numComponent, numSamples);       //reshape back

        //compute posterior <-- prior_i * likelihood_i
        if (unnormedPrior.GetNumCols() == numSamples) //each sample has different prior
            posterior.ElementMultiplyWith(prior);
        else //all samples share the same prior
            posterior.ColumnElementMultiplyWith(prior);

        //compute GMM log-likelihood
        Matrix<ElemType>::Multiply(ConstOnes(1, numComponent, posterior.GetDeviceId()), false, posterior, false, functionValues); //functionValues <-- total likelihood
        posterior.RowElementDivideBy(functionValues);                                                                             //posterior <-- per-comp likelihood / total likelihood
        functionValues.InplaceLog();                                                                                              //log likelihood

#if DUMPOUTPUT
        temp.Print("temp", 0, min(5, temp.GetNumRows() - 1), 0, min(10, temp.GetNumCols() - 1));
        normedDeviation.Print("normedDeviation", 0, min(5, normedDeviation.GetNumRows() - 1), 0, min(10, normedDeviation.GetNumCols() - 1));

        posterior.Print("posterior", 0, min(5, posterior.GetNumRows() - 1), 0, min(10, posterior.GetNumCols() - 1));
        functionValues.Print("functionValues", 0, min(5, functionValues.GetNumRows() - 1), 0, min(10, functionValues.GetNumCols() - 1));

        functionValues.Print("GMMLogLikelihoodNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        size_t rows[4];
        for (int i = 0; i < 4; i++)
            rows[i] = Input(i)->GetSampleMatrixNumRows();

        if (isFinalValidationPass)
        {
            if (!Input(3)->HasMBLayout())
                InvalidArgument("GMMLogLikelihoodNode: Features must be a minibatch.");
            if (Input(0)->GetMBLayout() != Input(1)->GetMBLayout() || Input(0)->GetMBLayout() != Input(2)->GetMBLayout())
                InvalidArgument("GMMLogLikelihoodNode: First three arguments must have the same MBLayout (which may be none).");

            if (rows[0] != rows[2])
                LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input) should have same dimension as logStddev (third input), i.e., all dimensions in each Gaussian component share the same stddev.");

            if (rows[1] != rows[0] * rows[3])
                LogicError("GMMLogLikelihoodNode: the number of rows in mean (second input) should equal rows(unnormedPrior(first input) * rows(feature(fourth input)).");
        }

        SetDims(TensorShape(1), true);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<GMMLogLikelihoodNode<ElemType>>(nodeP);
            *node->m_prior = *m_prior;
            *node->m_normedDeviation = *m_normedDeviation;
            *node->m_normedDeviationVectors = *m_normedDeviationVectors;
            *node->m_stddev = *m_stddev;
            *node->m_posterior = *m_posterior;
        }
    }

    //request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_prior, matrixPool);
        RequestMatrixFromPool(m_normedDeviation, matrixPool);
        RequestMatrixFromPool(m_normedDeviationVectors, matrixPool);
        RequestMatrixFromPool(m_stddev, matrixPool);
        RequestMatrixFromPool(m_posterior, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_prior, matrixPool);
        ReleaseMatrixToPool(m_normedDeviation, matrixPool);
        ReleaseMatrixToPool(m_normedDeviationVectors, matrixPool);
        ReleaseMatrixToPool(m_stddev, matrixPool);
        ReleaseMatrixToPool(m_posterior, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
    }

protected:
    shared_ptr<Matrix<ElemType>> m_prior;
    shared_ptr<Matrix<ElemType>> m_normedDeviation;
    shared_ptr<Matrix<ElemType>> m_normedDeviationVectors;
    shared_ptr<Matrix<ElemType>> m_stddev;
    shared_ptr<Matrix<ElemType>> m_posterior;
    shared_ptr<Matrix<ElemType>> m_temp;
};

template class GMMLogLikelihoodNode<float>;
template class GMMLogLikelihoodNode<double>;

// -----------------------------------------------------------------------
// DropoutNode (input) -- perform drop-out
// Output is scaled such that no post-scaling is necessary.
// -----------------------------------------------------------------------

template <class ElemType>
class DropoutNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Dropout";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(DropoutNode);
    DropoutNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_dropoutRate(0)
    {
        m_randomSeed = (unsigned long) CreateUniqId();
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        if (m_dropoutRate > 0)
            sliceInput0Grad.AddElementProductOf(sliceOutputGrad, DataFor(*m_maskOfDropout, fr));
        else
            sliceInput0Grad += sliceOutputGrad;
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The DropoutNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The DropoutNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();
        // resize temporaries to their proper size
        if (m_dropoutRate > 0)
            m_maskOfDropout->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        if (m_dropoutRate > 0)
        {
            // determine drop-out mask for this minibatch
            auto sliceMask = DataFor(*m_maskOfDropout, fr);
            sliceMask.SetUniformRandomMask((ElemType) m_dropoutRate, (ElemType)(1.0 / (1.0 - m_dropoutRate)) /*pre-scaled*/, m_randomSeed);
            m_randomSeed += 1073807359; // 1073807359 is a very large prime number to avoid collision with other dropout nodes
            // apply dropout mask
            sliceOutputValue.AssignElementProductOf(sliceMask, sliceInput0Value);
        }
        else
        {
            sliceOutputValue.SetValue(sliceInput0Value);
        }
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    // special methods for this node type which ComputationNetwork knows about and calls to pass parameters
    void SetDropoutRate(const double val)
    {
        if (val < 0 || val >= 1)
            LogicError("DropoutRate must be >= 0 and < 1.");
        m_dropoutRate = val;
    }

    void SetRandomSeed(const unsigned long val)
    {
        m_randomSeed = (unsigned long) val;
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(nodeP);
            node->m_dropoutRate = m_dropoutRate;
            node->m_randomSeed = m_randomSeed;
            node->m_maskOfDropout = m_maskOfDropout;
        }
    }
    //request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maskOfDropout, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_maskOfDropout, matrixPool);
    }

private:
    double m_dropoutRate;
    unsigned long m_randomSeed;

    shared_ptr<Matrix<ElemType>> m_maskOfDropout;
};

template class DropoutNode<float>;
template class DropoutNode<double>;

// -----------------------------------------------------------------------
// Hardmax(prediction)
// -----------------------------------------------------------------------
// the result is a 1 of n coding in which the (r, c) = 1 if row r has max value in column c
// this node is not differentiable and so cannot be used in the backpropagation
// TODO: make function value sparse?
template <class ElemType>
class HardmaxNode : public SoftmaxNodeBase /*ComputationNode*/<ElemType>
{
    typedef SoftmaxNodeBase<ElemType> Base;
    UsingSoftmaxNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"Hardmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(HardmaxNode);
    HardmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    /*virtual*/ void BackpropToV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues) override
    {
        gradient;
        inputFunctionValues;
        inputGradientValues;
        gradientValues;
        LogicError("Hardmax is not differentiable and is used for evaluation only.");
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    /*virtual*/ void ForwardPropV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) override
    {
        //TODO: temp solution, we need to write a math function specifically for this
        functionValues.AssignHardmaxOf(inputFunctionValues, true);
    }
};

template class HardmaxNode<float>;
template class HardmaxNode<double>;
} } }
