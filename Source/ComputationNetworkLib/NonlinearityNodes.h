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
    noGradient,
    unaryGradient,
    binaryWithInputGradient,
    binaryWithOutputGradient
};

template <class ElemType, ElementWiseOperator opForward, ElementWiseOperator opBackward, GradientOperationType opType>
class UnaryElementWiseWithOpCodeNodeBase : public ComputationNode<ElemType>, public NumInputs<1>, public IdentityTransformerNode
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
        auto result =             ValueTensorFor(rank, fr);
        auto input  = InputRef(0).ValueTensorFor(rank, fr);
        result.DoUnaryOpOf(0, input, 1, opForward, opSum);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        assert(inputIndex == 0), inputIndex;

        // get the args
        size_t rank = DetermineElementwiseTensorRank();
        auto sliceOutputGrad =             GradientTensorFor(rank, fr); // propagate from this one...
        auto sliceInputGrad  = InputRef(0).GradientTensorFor(rank, fr); // ...to this one

        GradientOperationType opTypeHolder = opType;  // preventing pragma warning C4127

        if (opTypeHolder == noGradient)
        {
            // Do nothing
        }
        else if (opTypeHolder == unaryGradient)
        {
            sliceInputGrad.DoUnaryOpOf(Input(inputIndex)->ParentOverwritesGradient() ? 0.0f : 1.0f, sliceOutputGrad, 1, opBackward, opSum);
        }
        else 
        {
            // If gradient can be compute from output rather than input, then that's better for mem sharing (and faster in most cases).
            // Not possible for Cos().
            auto sliceValue = (opType == binaryWithOutputGradient) ? ValueTensorFor(rank, fr) : // using input or output value
                InputRef(0).ValueTensorFor(rank, fr);
            sliceInputGrad.DoBinaryOpOf(Input(inputIndex)->ParentOverwritesGradient() ? 0.0f : 1.0f, sliceOutputGrad, sliceValue, 1, opBackward, opSum);
        }
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return opType == binaryWithOutputGradient;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return opType == binaryWithInputGradient;
    }

    virtual bool ImplementsGradientOverwriteOptimization() const override { return (opType != noGradient); }
};

#define UnaryElementWiseWithOpCodeNodeBaseMembers UsingComputationNodeMembersBoilerplate;

// -----------------------------------------------------------------------
// Pass (Input)
// SigmoidNode (input)
// TanhNode (input)
// RectifiedLinearNode (input)
// LogNode (input)
// ExpNode (input)
// FloorNode (input)
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
DeclareUnaryElementWiseWithOpCodeNode(Abs,             Abs,             ElementwiseProductWithAbsDerivative,                       binaryWithInputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Cosine,          Cosine,          ElementwiseProductWithCosDerivative,                       binaryWithInputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Exp,             Exp,             ElementwiseProduct,                                        binaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Floor,           Floor,           None,                                                      noGradient);
DeclareUnaryElementWiseWithOpCodeNode(Log,             Log,             ElementwiseProductWithLogDerivativeFromOutput,             binaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Negate,          Negate,          Negate,                                                    unaryGradient);
DeclareUnaryElementWiseWithOpCodeNode(Pass,            Copy,            Copy,                                                      unaryGradient);
DeclareUnaryElementWiseWithOpCodeNode(LabelsToGraph,   Copy,            Copy,                                                      unaryGradient);
DeclareUnaryElementWiseWithOpCodeNode(Reciprocal,      Reciprocal,      ElementwiseProductWithReciprocalDerivative,                binaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(RectifiedLinear, LinearRectifier, ElementwiseProductWithLinearRectifierDerivativeFromOutput, binaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Sigmoid,         Sigmoid,         ElementwiseProductWithSigmoidDerivativeFromOutput,         binaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Sin,             Sin,             ElementwiseProductWithSinDerivative,                       binaryWithInputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Sqrt,            Sqrt,            ElementwiseProductWithSqrtDerivative,                      binaryWithOutputGradient);
DeclareUnaryElementWiseWithOpCodeNode(Tanh,            Tanh,            ElementwiseProductWithTanhDerivativeFromOutput,            binaryWithOutputGradient);

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
        auto sliceInputGrad = InputRef(0).GradientFor(fr); // ...to this one
        auto sliceInputValue = InputUsedInComputingInputNodesGradients(0) ? InputRef(0).ValueFor(fr) : Matrix<ElemType>(sliceInputGrad.GetDeviceId());
        auto sliceOutputValue = OutputUsedInComputingInputNodesGradients() ? ValueFor(fr) : Matrix<ElemType>(sliceInputGrad.GetDeviceId());

        // do the actual operation
        BackpropToV(*m_gradientTemp, sliceInputValue, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
    }

    // derived class implement the actual non-linear operation
    virtual void BackpropToV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues) = 0;

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // move the target matrix to the target device, since below it is accessed as slices which cannot move
        // TODO: once this gets reimplemented using TensorView, then this is no longer needed.
        InputRef(0).Value().TransferToDeviceIfNotThere(Value().GetDeviceId(), /*isBeingMoved=*/ false);

        auto values = ValueFor(fr);
        ForwardPropV(values, InputRef(0).ValueFor(fr));
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

// -----------------------------------------------------------------------
// If (flag, ifValue, elseValue)
// -----------------------------------------------------------------------
// Similar to C's ternary operator "flag ? ifValue : elseValue". If first input is !=0 return second input, else third
template <class ElemType>
class IfNode : public ComputationNode<ElemType>, public NumInputs<3>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;

    static const std::wstring TypeName()
    {
        return L"If";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(IfNode);
    IfNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*IComputationNode::*/ BeginForwardProp() override // called before first iteration step of ForwardProp()
    {
        Base::BeginForwardProp();
        // we switch result to dense as a work-around because ColumnSlice doesn't support all the sparse formats
        // TODO: This is a stopgap. Is this the right thing to do? It changes the matrix type in-place.
        Value().SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateNaryZip(isFinalValidationPass, /* allow broadcast */ true, /* num Inputs */ 3);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result =           ValueTensorFor(rank, fr);
        auto input0 = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input2 = InputRef(2).ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignCondOf(input0, input1, input2);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex == 0) // derivative of the first input (the flag) is always 0 => no action.
            return;

        size_t rank = DetermineElementwiseTensorRank();
        auto gradient      =                    GradientTensorFor(rank, fr);
        auto input0        = InputRef(0).            ValueTensorFor(rank, fr.AllowBroadcast());
        auto inputGradient = InputRef(inputIndex).GradientTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (InputRef(inputIndex).ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);

        if (inputIndex == 1)
        {
            inputGradient.AddCopyIfOf(input0, gradient);
        }
        else if (inputIndex == 2)
        {
            inputGradient.AddCopyIfNotOf(input0, gradient);
        }
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex)  const override { return childIndex == 0; }
    virtual bool OutputUsedInComputingInputNodesGradients()  const override { return false; }
};

template class IfNode<float>;
template class IfNode<double>;

// -----------------------------------------------------------------------
// ClipNode (minValue, maxValue, tensor)
// -----------------------------------------------------------------------
// This node clips the values in a tensor elements-wise to ensure they are within minValue <= x >= maxValue
// The gradient (per element) is (ge(x, minValue) AND le(x, maxValue)), or in other words, 1 if the value has
// not been clipped, and 0 if the value has been clipped.

template <class ElemType>
class ClipNode : public ComputationNode<ElemType>, public NumInputs<3>
{
    typedef ComputationNode<ElemType> Base;    
    UsingComputationNodeMembersBoilerplate;

    static const std::wstring TypeName()
    {
        return L"Clip";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(ClipNode);
    ClipNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result =             ValueTensorFor(rank, fr);
        auto input0 = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input2 = InputRef(2).ValueTensorFor(rank, fr.AllowBroadcast());

        result.AssignClipOf(input0, input1, input2);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // there is only a gradient for the input tensor that is to be clipped
        if (inputIndex == 2)
        {
            size_t rank = DetermineElementwiseTensorRank();
            auto gradient =                           GradientTensorFor(rank, fr);
            auto inputGradient = InputRef(inputIndex).GradientTensorFor(rank, fr.AllowBroadcast());
            auto input =         InputRef(inputIndex).ValueTensorFor(rank, fr.AllowBroadcast());
            auto output =                             ValueTensorFor(rank, fr.AllowBroadcast());

            inputGradient.AddCopyIfEqualOf(input, output, gradient);
        }        
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateNaryZip(isFinalValidationPass, /* allow broadcast */ true, /* num Inputs */ 3);
    }
};

template class ClipNode<float>;
template class ClipNode<double>;


// -----------------------------------------------------------------------
// CompareNode(a,b)
// -----------------------------------------------------------------------
// Template parameters compType (-1, 0, 1) and polarity (0, 1) are used selecting one of the six basic comparison operations. 
// Note: parametrizing the 6 comparison operations with the the two parameters 'compType' an 'polarity' is motivated by:
//
// comp(a, b, compType, polarity) <==> sign(a-b) == compType, if polarity == 0
//                                     sign(a-b) != compType, if polarity == 1
template <class ElemType, int compType, int polarity>
class ComparisonNode : public BinaryElementWiseNode<ElemType>
{
private:
    // Index corresponds to different comparison operations. 
    const static int index = 1 + compType + 3 * polarity;

    // The operations are indexed in the same order they appear in enum ElementWiseOperator: "Less", "Equal", "Greater", "GreaterEqual", "NotEqual", "LessEqual".
    // This ordering is checked below:
    static_assert(1 == ElementWiseOperator::opEqual         - ElementWiseOperator::opLess, "ElementWiseOperator::opEqual has wrong value relative to ElementWiseOperator::opLess");
    static_assert(2 == ElementWiseOperator::opGreater       - ElementWiseOperator::opLess, "ElementWiseOperator::opGreater has wrong value relative to ElementWiseOperator::opLess");
    static_assert(3 == ElementWiseOperator::opGreaterEqual  - ElementWiseOperator::opLess, "ElementWiseOperator::opGreaterEqual has wrong value relative to ElementWiseOperator::opLess");
    static_assert(4 == ElementWiseOperator::opNotEqual      - ElementWiseOperator::opLess, "ElementWiseOperator::opNotEqual has wrong value relative to ElementWiseOperator::opLess");
    static_assert(5 == ElementWiseOperator::opLessEqual     - ElementWiseOperator::opLess, "ElementWiseOperator::opLessEqual has wrong value relative to ElementWiseOperator::opLess");

public:
    typedef BinaryElementWiseNode<ElemType> Base; UsingBinaryElementwiseNodeBaseMembers;

    static const std::wstring TypeName()
    {
    const wchar_t* names[] = { L"Less", L"Equal", L"Greater", L"GreaterEqual", L"NotEqual", L"LessEqual" };
        return names[index];
    }

    DeclareConstructorFromConfigWithNumInputs(ComparisonNode);
    ComparisonNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex)  const override { return childIndex == 0; }
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result =             ValueTensorFor(rank, fr);
        auto input0 = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());

        result.DoBinaryOpOf(0, input0, input1, 1.0f, static_cast<ElementWiseOperator> (ElementWiseOperator::opLess + index), ElementWiseOperator::opSum);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // Function is piecewise constant --> gradient = 0
    }
};

// Define macro that defines and instantiates different comparison nodes.
// Unfortuanately the C++ 11 type alias syntax doesn't work for mpic++ so we use this more ugly way.
#define DefineComparisonNode(ClassName, compType, polarity)             \
template <class ElemType>                                               \
class ClassName : public ComparisonNode<ElemType, compType, polarity>   \
{                                                                       \
    typedef ComparisonNode<ElemType, compType, polarity> Base;          \
    UsingComputationNodeMembersBoilerplate;                             \
                                                                        \
public:                                                                 \
    static const std::wstring TypeName() { return Base::TypeName(); }   \
    DeclareConstructorFromConfigWithNumInputs(ClassName);               \
    ClassName(DEVICEID_TYPE deviceId, const wstring& name)              \
            : Base(deviceId, name)                                      \
    {                                                                   \
    }                                                                   \
};                                                                      \
                                                                        \
template class ClassName<float>;                                        \
template class ClassName<double>;

DefineComparisonNode(LessNode,         -1, 0)
DefineComparisonNode(EqualNode,         0, 0)
DefineComparisonNode(GreaterNode,       1, 0)
DefineComparisonNode(GreaterEqualNode, -1, 1)
DefineComparisonNode(NotEqualNode,      0, 1)
DefineComparisonNode(LessEqualNode,     1, 1)
}}}
