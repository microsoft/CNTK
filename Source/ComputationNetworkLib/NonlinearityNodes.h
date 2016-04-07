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

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// UnaryElementWiseWithOpCodeNodeBase (input) -- base for elementwise unary op
// where forward // and backward are single ElementWiseOperator opcodes and
// only inputs (but not // function values) are used.
// -----------------------------------------------------------------------

enum GradientOperationType 
{
    UnaryGradient,
    BinaryWithInputGradient,
    BinaryWithOutputGradient
};

template <class ElemType, ElementWiseOperator opForward, ElementWiseOperator opBackward, GradientOperationType opType>
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

        // we expect a constant conditional expression here -- suppress the warning that leads to an error
        // TODO: alternative: assign to a non-const variable and test that.
#pragma warning( push )
#pragma warning( disable : 4127 )
        if (opType == UnaryGradient) 
        {
            sliceInputGrad.DoUnaryOpOf(1, sliceOutputGrad, 1, opBackward);
        }
        else 
        {
            // If gradient can be compute from output rather than input, then that's better for mem sharing (and faster in most cases).
            // Not possible for Cos().
            auto sliceValue = (opType == BinaryWithOutputGradient) ? ValueTensorFor(rank, fr) : // using input or output value
                Input(0)->ValueTensorFor(rank, fr);
            sliceInputGrad.DoBinaryOpOf(1, sliceOutputGrad, sliceValue, 1, opBackward);
        }
#pragma warning( pop )
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return opType == BinaryWithOutputGradient;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return opType == BinaryWithInputGradient;
    }
};

#define UnaryElementWiseWithOpCodeNodeBaseMembers UsingComputationNodeMembersBoilerplate;

// -----------------------------------------------------------------------
// Pass (Input)
// SigmoidNode (input)
// TanhNode (input)
// RectifiedLinearNode (input)
// LogNode (input)
// ExpNode (input)
// CosineNode (input)
// SinNode (input)
// Abs(input)
// Negate (input)
// Sqrt (input)
// Reciprocal (input)
// These are all implemented by single-opcode functions and can thus be declared by a macro.
// -----------------------------------------------------------------------

#pragma push_macro("DeclareUnaryElementWiseWithOpCodeNode")
#define DeclareUnaryElementWiseWithOpCodeNode(Name, Forward, Backward, opType)                                                               \
    template <class ElemType>                                                                                                                \
    class Name##Node : public UnaryElementWiseWithOpCodeNodeBase<ElemType, op##Forward, op##Backward, opType>                                \
    {                                                                                                                                        \
        typedef UnaryElementWiseWithOpCodeNodeBase<ElemType, op##Forward, op##Backward, opType> Base;                                        \
        UnaryElementWiseWithOpCodeNodeBaseMembers;                                                                                           \
        static const std::wstring TypeName()                                                                                                 \
        {                                                                                                                                    \
            return L## #Name;                                                                                                                \
        }                                                                                                                                    \
                                                                                                                                             \
    public:                                                                                                                                  \
        DeclareConstructorFromConfigWithNumInputs(Name##Node);                                                                               \
        Name##Node(DEVICEID_TYPE deviceId, const wstring& Name) :                                                                            \
            Base(deviceId, Name)                                                                                                             \
        {                                                                                                                                    \
        }                                                                                                                                    \
    }

//                                    Name             Forward and      Backward opcodes                                           Gradient optype
DeclareUnaryElementWiseWithOpCodeNode(Pass,            Copy,            Copy,                                                      UnaryGradient);
DeclareUnaryElementWiseWithOpCodeNode(Sigmoid,         Sigmoid,         ElementwiseProductWithSigmoidDerivativeFromOutput,         BinaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Tanh,            Tanh,            ElementwiseProductWithTanhDerivativeFromOutput,            BinaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(RectifiedLinear, LinearRectifier, ElementwiseProductWithLinearRectifierDerivativeFromOutput, BinaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Log,             Log,             ElementwiseProductWithLogDerivativeFromOutput,             BinaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Exp,             Exp,             ElementwiseProduct,                                        BinaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Cosine,          Cosine,          ElementwiseProductWithCosDerivative,                       BinaryWithInputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Sin,             Sin,             ElementwiseProductWithSinDerivative,                       BinaryWithInputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Abs,             Abs,             ElementwiseProductWithAbsDerivative,                       BinaryWithInputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Negate,          Negate,          Negate,                                                    UnaryGradient);
DeclareUnaryElementWiseWithOpCodeNode(Sqrt,            Sqrt,            ElementwiseProductWithSqrtDerivative,                      BinaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Reciprocal,      Reciprocal,      ElementwiseProductWithReciprocalDerivative,                BinaryWithOutputGradient);

#pragma pop_macro("DeclareUnaryElementWiseWithOpCodeNode")

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
    // virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
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
        auto sliceInputValue = InputUsedInComputingInputNodesGradients(0) ? Input(0)->ValueFor(fr) : Matrix<ElemType>(sliceInputGrad.GetDeviceId());
        auto sliceOutputValue = OutputUsedInComputingInputNodesGradients() ? ValueFor(fr) : Matrix<ElemType>(sliceInputGrad.GetDeviceId());

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
            node->m_gradientTemp->SetValue(*m_gradientTemp);
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

    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

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
            node->m_diff->SetValue(*m_diff);
        }
    }
    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_diff, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
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

    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

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
            node->m_softmax->SetValue(*m_softmax);
        }
    }
    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_softmax, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
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
        gradient; inputFunctionValues; inputGradientValues; gradientValues;
        // Hardmax cannot back-propagate a gradient.
        // We must not forbid this function to be called, though, since Hardmax may be running
        // as part of a recurrent decoding loop. Sequence-to-sequence models run the Hardmax
        // node inside the training without back-propagating into them.
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    /*virtual*/ void ForwardPropV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) override
    {
        functionValues.AssignHardmaxOf(inputFunctionValues, true);
    }
};

template class HardmaxNode<float>;
template class HardmaxNode<double>;

}}}
