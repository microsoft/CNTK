//
// <copyright file="LinearAlgebraNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ConvolutionalNodes.h"
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
// PlusNode (summand1, summand2)
// -----------------------------------------------------------------------

template <class ElemType>
class PlusNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base;
    UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"Plus";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(PlusNode);
    PlusNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        //static int c = 0; if (c++ == 0) { fprintf(stderr, "#PLUSBP#\n"); }
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient = GradientTensorFor(rank, fr);
        auto inputGradient = Input(inputIndex)->GradientTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->GetNumCols() < GetNumCols())
            MaskMissingGradientColumnsToZero(fr);

        inputGradient.AddCopyOf(gradient);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        //static int c = 0; if (c++ == 0) { fprintf(stderr, "#PLUS#\n"); }
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input0 = Input(0)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignSumOf(input0, input1);
    }
};

template class PlusNode<float>;
template class PlusNode<double>;

// -----------------------------------------------------------------------
// MinusNode (minuend, subtrahend)
// -----------------------------------------------------------------------

template <class ElemType>
class MinusNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base;
    UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"Minus";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(MinusNode);
    MinusNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        ElemType sign = inputIndex == 0 ? 1.0f : -1.0f;
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient = GradientTensorFor(rank, fr);
        auto inputGradient = Input(inputIndex)->GradientTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->GetNumCols() < GetNumCols())
            MaskMissingGradientColumnsToZero(fr);

        inputGradient.AddCopyOf(gradient, sign);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        //static int c = 0; if (c++ == 0) { fprintf(stderr,"#MINUS#\n"); }
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input0 = Input(0)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignDifferenceOf(input0, input1);
    }
};

template class MinusNode<float>;
template class MinusNode<double>;

#endif // ENABLE_TENSORVIEW

// -----------------------------------------------------------------------
// NegateNode (input)
// computes the negative of its input
// -----------------------------------------------------------------------

template <class ElemType>
class NegateNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Negate";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(NegateNode);
    NegateNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        Input(0)->GradientFor(fr) -= GradientFor(fr);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The NegateNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The NegateNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        ValueFor(fr).AssignDifferenceOf(0, Input(0)->ValueFor(fr));
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }
};

template class NegateNode<float>;
template class NegateNode<double>;

// -----------------------------------------------------------------------
// TimesNode (A, B)
// right operand and output can have MB layout, while left operand cannot
// -----------------------------------------------------------------------

template <class ElemType>
class TimesNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Times";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(TimesNode);
    TimesNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex == 0) // left derivative
        {
            // this potentially computes inner products over time, so we use the Masked- variants
            Matrix<ElemType> sliceOutputGrad = MaskedGradientFor(fr);
            Matrix<ElemType> sliceInput1Value = Input(1)->MaskedValueFor(fr);

            // currently we only support one combination when the input is sparse.
            if (sliceInput1Value.GetMatrixType() == SPARSE && Input(0)->Gradient().GetMatrixType() == DENSE && sliceOutputGrad.GetMatrixType() == DENSE)
                Input(0)->Gradient().SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

            Matrix<ElemType>::MultiplyAndAdd(sliceOutputGrad, false, sliceInput1Value, true, Input(0)->Gradient());
        }
        else // right derivative
        {
            Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

            Matrix<ElemType>::MultiplyAndAdd(Input(0)->Value(), true, sliceOutputGrad, false, sliceInput1Grad);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The TimesNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rows0 = Input(0)->GetNumRows(), cols1 = Input(1)->GetNumCols();
        VerifyDims(rows0, cols1);

        // right operand and output can have MB layout, while left operand cannot
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
#if DUMPOUTPUT
        Input(0)->Value().Print("TimesNode - Input0");
#endif
        sliceOutputValue.AssignProductOf(Input(0)->Value(), false, sliceInput1Value, false);
#if NANCHECK
        sliceOutputValue.HasNan("Times");
#endif
#if DUMPOUTPUT
        sliceOutputValue.Print("TimesNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation requires the first factor to not be minibatch data (must not have an MBLayout).", NodeName().c_str(), OperationName().c_str());
        InferMBLayoutFromInputsForStandardCase();

        //support automatic dimension inference for learnable parameters
        size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
        size_t rows1 = Input(1)->GetNumRows(), cols1 = Input(1)->GetNumCols();

        if (isFinalValidationPass && (rows0 == 0 || (cols1 == 0 && !Input(1)->GetMBLayout())))
            RuntimeError("Times operation: Input(0)->GetNumRows() and Input(1)->GetNumCols() should not be 0 since it cannot be automatically inferred");

        // limited automatic dimension inference for *children*, useful for CNN since it can be hard to know the size of each input parameter without deep knowledge how CNN is implemented (padding, stride)
        // TODO: ^^ There must be a better solution. Maybe MBLayout as well?
        // TODO: use dynamic_pointer_cast
        // infer cols0 as rows1
        if (cols0 == 0 && !Input(0)->GetMBLayout() && rows1 != 0 && isFinalValidationPass)
            ValidateInferInputDims(0, rows0, rows1);

        // infer rows1 as cols0
        if (cols0 != 0 && rows1 == 0)
            ValidateInferInputDims(1, cols0, cols1);

        if (isFinalValidationPass && Input(1)->GetNumRows() != Input(0)->GetNumCols())
            LogicError("The inner matrix dimension in the %ls %ls operation does not match (%d vs. %d).", NodeName().c_str(), OperationName().c_str(), (int) Input(1)->GetNumRows(), (int) Input(0)->GetNumCols());

        // TODO: With tensors, inner dimensions must match.
        // after multiplication the structure is lost
        SetDims(TensorShape(rows0), cols1);
    }

    virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool) override
    {
        // this is a special handling case. We need to allocate sparse matrix directly instead of from pool.
        if (m_inputs[0]->NeedGradient() && Input(1)->Value().GetMatrixType() == SPARSE)
        {
            Input(0)->CreateGradientMatrixIfNull();
            Input(0)->Gradient().SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);
        }

        // we need to call base allocation at end since we will need to allocate special ones first
        // so that the default allocator will not allocate it again.
        Base::AllocateGradientMatricesForInputs(matrixPool);
    }
};

template class TimesNode<float>;
template class TimesNode<double>;

// -----------------------------------------------------------------------
// TransposeTimesNode (A', B)
// right operand and output can have MB layout, while left operand cannot
// TODO: merge with TimesNode?
// -----------------------------------------------------------------------

template <class ElemType>
class TransposeTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"TransposeTimes";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(TransposeTimesNode);
    TransposeTimesNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex == 0) //left derivative
        {
            // this potentially computes inner products over time, so we use the Masked- variants
            Matrix<ElemType> sliceOutputGrad = MaskedGradientFor(fr);
            Matrix<ElemType> sliceInput1Value = Input(1)->MaskedValueFor(fr);

            BackpropToLeft(sliceInput1Value, Input(0)->Gradient(), sliceOutputGrad);
        }
        else //right derivative
        {
            Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

            BackpropToRight(Input(0)->Value(), sliceInput1Grad, sliceOutputGrad);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The TransposeTimesNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    /*TODO: merge with call site*/ void BackpropToLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
    {
#if DUMPOUTPUT
        gradientValues.Print("Gradient-in");
        inputGradientValues.Print("child Gradient-in/out");
        inputFunctionValues.Print("child Function values");
#endif
        //currently we only support one combination when the input is sparse.
        if (inputFunctionValues.GetMatrixType() == SPARSE && inputGradientValues.GetMatrixType() == DENSE && gradientValues.GetMatrixType() == DENSE)
            inputGradientValues.SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

        Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, true, inputGradientValues);

#if DUMPOUTPUT
        inputGradientValues.Print("child Gradient-out");
#endif
    }

    /*TODO: merge with call site*/ void BackpropToRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
    {
#if DUMPOUTPUT
        gradientValues.Print("Gradient-in");
        inputGradientValues.Print("child Gradient-in/out");
        inputFunctionValues.Print("child Function values");
#endif
        Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, false, inputGradientValues);

#if DUMPOUTPUT
        inputGradientValues.Print("child Gradient-out");
#endif
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        sliceOutputValue.AssignProductOf(Input(0)->Value(), true, sliceInput1Value, false);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        //support automatic dimension inference for learnable parameters
        size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
        size_t rows1 = Input(1)->GetNumRows(), cols1 = Input(1)->GetNumCols();

        if (isFinalValidationPass && (rows0 == 0 || (!Input(1)->HasMBLayout() && cols1 == 0)))
            RuntimeError("TransposeTimes operation: Input(0)->GetNumRows() and Input(1)->GetNumCols() should not be 0 since it cannot be automatically inferred");

        if (cols0 == 0 && rows1 != 0 && isFinalValidationPass)
            ValidateInferInputDims(0, rows0, rows1);

        if (cols0 != 0 && rows1 == 0)
            ValidateInferInputDims(1, cols0, cols1);

        //cols0 and rows1 may have been changed so don't use them in the following check
        if (isFinalValidationPass && Input(1)->GetNumRows() != Input(0)->GetNumRows())
            LogicError("The Matrix dimension in the TransposeTimes operation does not match.");

        // TODO: What should the tensor story be?
        //after multiplication the structure is lost
        SetDims(TensorShape(cols0), cols1);
    }
};

template class TransposeTimesNode<float>;
template class TransposeTimesNode<double>;

#ifdef ENABLE_TENSORVIEW

// -----------------------------------------------------------------------
// ElementTimesNode (factor1, factor2)
//
// This allows broadcasting, and can thus also scale with a row, a column, or a scalar.
// -----------------------------------------------------------------------

template <class ElemType>
class ElementTimesNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base;
    UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"ElementTimes";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(ElementTimesNode);
    ElementTimesNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient = GradientTensorFor(rank, fr);
        auto inputGradient = Input(inputIndex)->GradientTensorFor(rank, fr.AllowBroadcast());
        auto otherInputValue = Input(1 - inputIndex)->ValueTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->GetNumCols() < GetNumCols())
            MaskMissingGradientColumnsToZero(fr);
        if (Input(inputIndex)->GetNumCols() < Input(1 - inputIndex)->GetNumCols())
            Input(1 - inputIndex)->MaskMissingValueColumnsToZero(fr);

        inputGradient.AddElementwiseProductOf(gradient, otherInputValue);
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        return true;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        //static int c = 0; if (c++ == 0) { fprintf(stderr,"#ETIMES#\n"); }
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input0 = Input(0)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignElementwiseProductOf(input0, input1);
    }
};

template class ElementTimesNode<float>;
template class ElementTimesNode<double>;

#endif // ENABLE_TENSORVIEW

// -----------------------------------------------------------------------
// DiagTimesNode (vector representing the diagonal of a square matrix, data)
// -----------------------------------------------------------------------

template <class ElemType>
class DiagTimesNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"DiagTimes";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(DiagTimesNode);
    DiagTimesNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex == 0) // left derivative
        {
            Matrix<ElemType> sliceOutputGrad = MaskedGradientFor(fr); // use Masked- version since this is reducing over frames
            Matrix<ElemType> sliceInput1Value = Input(1)->MaskedValueFor(fr);
            m_innerproduct->AssignInnerProductOf(sliceOutputGrad, sliceInput1Value, false);
            Input(0)->Gradient() += *m_innerproduct;
        }
        else // right derivative
        {
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);
            Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);
            m_rightGradient->SetValue(sliceOutputGrad);
            m_rightGradient->ColumnElementMultiplyWith(Input(0)->Value());
            sliceInput1Grad += *m_rightGradient;
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The DiagTimesNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        sliceOutputValue.SetValue(sliceInput1Value);
        sliceOutputValue.ColumnElementMultiplyWith(Input(0)->Value());
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        //if dimension not specified we assume two operands' dimensions should match
        if (Input(0)->GetNumRows() == 0 && Input(1)->GetNumRows() != 0)
            ValidateInferInputDims(0, Input(1)->GetNumRows(), 1);

        if (Input(0)->GetNumRows() != 0 && Input(1)->GetNumRows() == 0)
            ValidateInferInputDims(1, Input(0)->GetNumRows(), Input(1)->GetNumCols());

        if (isFinalValidationPass)
        {
            if (Input(1)->GetNumRows() != Input(0)->GetNumRows())
                LogicError("The Matrix dimension in the DiagTimes operation does not match.");

            if (Input(0)->GetNumCols() != 1)
                LogicError("The first matrix should be a vector representing the diagonal of a square matrix in the DiagTimes operation.");
        }

        // TODO: Should Input(0) have a specific tensor structure? E.g. match Input(1)?
        SetDims(Input(1));
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<DiagTimesNode<ElemType>>(nodeP);
            *node->m_innerproduct = *m_innerproduct;
            *node->m_rightGradient = *m_rightGradient;
        }
    }
    //request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_innerproduct, matrixPool);
        RequestMatrixFromPool(m_rightGradient, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_innerproduct, matrixPool);
        ReleaseMatrixToPool(m_rightGradient, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_innerproduct;
    shared_ptr<Matrix<ElemType>> m_rightGradient;
};

template class DiagTimesNode<float>;
template class DiagTimesNode<double>;

// -----------------------------------------------------------------------
// SumElementsNode (input)
// sums up all elements in the input into a single scalar
// -----------------------------------------------------------------------

template <class ElemType>
class SumElementsNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SumElements";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(SumElementsNode);
    SumElementsNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        Input(0)->GradientFor(fr) += Gradient(); // here the assumption is that gradientValues are 1x1 matrix
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The SumElementsNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The SumElementsNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Value().AssignSumOfElements(Input(0)->MaskedValueFor(fr)); // since we are reducing over frames, we must first mask gaps in input to zero
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data
        SetDims(TensorShape(1), 1);
    }
};

template class SumElementsNode<float>;
template class SumElementsNode<double>;

// -----------------------------------------------------------------------
// SumColumnElementsNode (input)
// sums up each column of the input
// TODO: This should be deprecated, in favor of a reduce node.
// TODO: Implement this with the tensor library.
// -----------------------------------------------------------------------

template <class ElemType>
class SumColumnElementsNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SumColumnElements";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(SumColumnElementsNode);
    SumColumnElementsNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInputGrad = Input(0)->GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        sliceInputGrad += sliceOutputGrad; // here the assumption is that gradientValues is a row vector
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The SumColumnElementsNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The SumColumnElementsNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInputValue = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        //ForwardPropS(sliceOutputValue, sliceInputValue);
        Matrix<ElemType>::VectorSum(sliceInputValue, sliceOutputValue, true);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        SetDims(TensorShape(1), Input(0)->GetNumCols()); // each column is reduced to a scalar
    }
};

template class SumColumnElementsNode<float>;
template class SumColumnElementsNode<double>;

// -----------------------------------------------------------------------
// TransposeNode (input matrix)
// TODO: extend towards tensor transpose (swap 2 dimensions)
// -----------------------------------------------------------------------

template <class ElemType>
class TransposeNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Transpose";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(TransposeNode);
    TransposeNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        Matrix<ElemType>& inputGradientValues = Input(0)->Gradient();
        const Matrix<ElemType>& gradientValues = Gradient();
#if DUMPOUTPUT
        gradientValues.Print("Gradient-in");
        inputGradientValues.Print("child Gradient-in/out");
        inputFunctionValues.Print("child Function values");
#endif
        const Matrix<ElemType>& ones = ConstOnes(inputGradientValues.GetNumRows(), inputGradientValues.GetNumRows(), inputGradientValues.GetDeviceId());
        Matrix<ElemType>::MultiplyAndAdd(ones, false, gradientValues, true, inputGradientValues);
#if DUMPOUTPUT
        inputGradientValues.Print("child Gradient-out");
#endif
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
#if DUMPOUTPUT
        return true;
#else
        // The TransposeNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
#endif
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The TransposeNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
#if DUMPOUTPUT
        Input(0)->Value().Print("TransposeNode- Input0");
#endif
        Value().AssignTransposeOf(Input(0)->Value());
#if NANCHECK
        Value().HasNan("Transpose");
#endif
#if DUMPOUTPUT
        Value().Print("TransposeNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation cannot operate on minibatch data (which have a layout)", NodeName().c_str(), OperationName().c_str());
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (Input(0)->HasSampleLayout()) // must be a plain matrix without tensor substructure
            InvalidArgument("%ls %ls operation cannot operate on input tensors", NodeName().c_str(), OperationName().c_str());
        size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
        SetDims(TensorShape(cols0), rows0);
    }
};

template class TransposeNode<float>;
template class TransposeNode<double>;

// -----------------------------------------------------------------------
// DiagonalNode -- extract diagonal elements of a square matrix
// -----------------------------------------------------------------------

template <class ElemType>
class DiagonalNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Diagonal";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(DiagonalNode);
    DiagonalNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<DiagonalNode<ElemType>>(nodeP);
        }
    }

    virtual void PrintSelfBeforeValidation(bool allowNulls = false) const
    {
        fprintf(stderr, "\nValidating --> %ls = %ls", NodeName().c_str(), OperationName().c_str());

        if (!IsLeaf())
        {
            fprintf(stderr, "(");
            for (size_t i = 0; i < GetNumInputs(); i++)
            {
                ComputationNodePtr child = Input(i);
                if (i > 0)
                    fprintf(stderr, ", ");

                if (child == nullptr)
                {
                    if (allowNulls)
                    {
                        fprintf(stderr, "NULL");
                        continue;
                    }
                    RuntimeError("One of the children is missing.");
                }

                fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->Value().GetNumRows(), child->Value().GetNumCols());
            }

            fprintf(stderr, ")");
        }
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr;

        if (isFinalValidationPass && Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation cannot operate on minibatch data (which have a layout)", NodeName().c_str(), OperationName().c_str());

        size_t dim = Input(0)->GetNumCols();
        if (isFinalValidationPass && dim != Input(0)->GetNumRows())
            InvalidArgument("%ls %ls operation requires a square matrix as its input.", NodeName().c_str(), OperationName().c_str());

        if (Input(0)->HasSampleLayout())
            fprintf(stderr, "WARNING: Diagonal operation cannot inherit image size information from its child. Image size info is lost.\n");

        SetDims(TensorShape(1), dim);
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        Input(0)->Value().AssignDiagonalValuesTo(Value());
#if NANCHECK
        Value().HasNan("Diagonal");
#endif
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        Matrix<ElemType>& inputGradientValues = Input(0)->Gradient();
        const Matrix<ElemType>& gradientValues = Gradient();

        // BUGBUG: This should use the memshare mechanism
        Matrix<ElemType> diag(gradientValues.GetNumRows(), gradientValues.GetNumCols(), gradientValues.GetDeviceId());
        diag = gradientValues;
        diag.Resize(gradientValues.GetNumCols(), 1);

        inputGradientValues.SetValue(0);
        // BUGBUG: Must *add* to gradient!
        inputGradientValues.SetDiagonalValue(diag);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The DiagonalNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        // The DiagonalNode does not require any of it's input's values for computing
        // the gradients of its input nodes
        UNREFERENCED_PARAMETER(childIndex);
        return false;
    }
};

template class DiagonalNode<float>;
template class DiagonalNode<double>;

// -----------------------------------------------------------------------
// CosDistanceNode (left, right)
// column-wise cos distance
// TODO: Would it be useful to allow one of the two to be a single column?
// -----------------------------------------------------------------------

//The first matrix should be a vector regpresting the diagonal of a square matrix in the DiagTimes operation
template <class ElemType>
class CosDistanceNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CosDistance";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CosDistanceNode);
    CosDistanceNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
        // temp, rightTerm, leftTerm - temporary matrices
        if (inputIndex == 0) //left derivative
            m_temp->AssignElementProductOf(*m_invNorm0, *m_invNorm0);
        else //right derivative
            m_temp->AssignElementProductOf(*m_invNorm1, *m_invNorm1);

        m_temp->ElementMultiplyWith(ValueFor(fr));
        m_rightTerm->SetValue(Input(inputIndex)->ValueFor(fr));
        m_rightTerm->RowElementMultiplyWith(*m_temp);

        m_temp->AssignElementProductOf(*m_invNorm0, *m_invNorm1);
        m_leftTerm->SetValue(Input(1 - inputIndex)->ValueFor(fr));
        m_leftTerm->RowElementMultiplyWith(*m_temp);

        *m_leftTerm -= *m_rightTerm;
        m_leftTerm->RowElementMultiplyWith(GradientFor(fr));
        Input(inputIndex)->GradientFor(fr) += *m_leftTerm;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_invNorm0->AssignVectorNorm2Of(sliceInput0Value, true);
        m_invNorm0->AssignElementInverseOf(*m_invNorm0);

        m_invNorm1->AssignVectorNorm2Of(sliceInput1Value, true);
        m_invNorm1->AssignElementInverseOf(*m_invNorm1);

        sliceOutputValue.AssignInnerProductOf(sliceInput0Value, sliceInput1Value, true);
        sliceOutputValue.ElementMultiplyWith(*m_invNorm0);
        sliceOutputValue.ElementMultiplyWith(*m_invNorm1);
        // TODO: This formulation above allows to use the tensor lib for this, with automatic broadcasting.
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        ValidateInferBinaryInputDims();

#if 0
            if (isFinalValidationPass && (Input(1)->GetNumRows() != Input(0)->GetNumRows() || (HasMBLayout() && (Input(1)->GetNumCols() != Input(0)->GetNumCols()))))
                LogicError("%ls %ls operation: The input dimensions do not match.", NodeName().c_str(), OperationName().c_str());
#endif

        // TODO: We could do something more interesting with tensors.
        //       E.g. apply a cos distance of a whole set of data with a single reference.
        SetDims(TensorShape(1), Input(1)->GetNumCols());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CosDistanceNode<ElemType>>(nodeP);
            *node->m_invNorm0 = *m_invNorm0;
            *node->m_invNorm1 = *m_invNorm1;
            *node->m_leftTerm = *m_leftTerm;
            *node->m_rightTerm = *m_rightTerm;
            *node->m_temp = *m_temp;
        }
    }
    //request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_invNorm0, matrixPool);
        RequestMatrixFromPool(m_invNorm1, matrixPool);
    }

    //request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_leftTerm, matrixPool);
        RequestMatrixFromPool(m_rightTerm, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_invNorm0, matrixPool);
        ReleaseMatrixToPool(m_invNorm1, matrixPool);
        ReleaseMatrixToPool(m_leftTerm, matrixPool);
        ReleaseMatrixToPool(m_rightTerm, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
    }

private:
    // invNorm nodes tranfer data between ForwardProp and BackpropTo
    shared_ptr<Matrix<ElemType>> m_invNorm0;
    shared_ptr<Matrix<ElemType>> m_invNorm1;
    // the rest are temporaries, values don't need to be maintained
    shared_ptr<Matrix<ElemType>> m_leftTerm;
    shared_ptr<Matrix<ElemType>> m_rightTerm;
    shared_ptr<Matrix<ElemType>> m_temp;
};

template class CosDistanceNode<float>;
template class CosDistanceNode<double>;

// -----------------------------------------------------------------------
// KhatriRaoProductNode (left, right)
// -----------------------------------------------------------------------

template <class ElemType>
class KhatriRaoProductNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"KhatriRaoProduct";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(KhatriRaoProductNode);
    KhatriRaoProductNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        if (inputIndex == 0) //left derivative
        {
            Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
            Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);

            sliceInput0Grad.AddColumnReshapeProductOf(sliceOutputGrad, sliceInput1Value, false);
        }
        else //right derivative
        {
            Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);

            sliceInput1Grad.AddColumnReshapeProductOf(sliceOutputGrad, sliceInput0Value, true);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The KhatriRaoProductNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        ValueFor(fr).AssignKhatriRaoProductOf(Input(0)->ValueFor(fr), Input(1)->ValueFor(fr));
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        //support automatic dimension inference for learnable parameters
        size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
        size_t rows1 = Input(1)->GetNumRows(), cols1 = Input(1)->GetNumCols();

        if (cols0 == 0 && cols1 != 0)
            ValidateInferInputDims(0, rows0, cols1);

        if (cols0 != 0 && cols1 == 0)
            ValidateInferInputDims(1, rows1, cols0);

        if (isFinalValidationPass && !HasMBLayout() && Input(1)->GetNumCols() != Input(0)->GetNumCols())
            LogicError("The Matrices should have same number of columns.");

        // after KhatriRaoProduct the structure is lost
        // TODO: ^^ Is that correctWhat is the correct sample layout?
        SetDims(TensorShape(rows0 * rows1), Input(0)->GetNumCols());
    }
};

template class KhatriRaoProductNode<float>;
template class KhatriRaoProductNode<double>;

// -----------------------------------------------------------------------
// CosDistanceWithNegativeSamplesNode (left, right, shift, neg)
//
// Left and right forms pairs of positive samples. They are symmetric but usually
// the search key is used as the left. For example, Left is search query and right is document.
// The negative samples are formed on the fly by shifting the right side.
// The 'shift' indicates how many samples in the right node you should shift to form each negative sample pair.
// It is often choose to be one. 'Neg' indicates how many negative samples you want to generate.
// -----------------------------------------------------------------------

template <class ElemType>
class CosDistanceWithNegativeSamplesNode : public ComputationNode<ElemType>, public NumInputs<4>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CosDistanceWithNegativeSamples";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CosDistanceWithNegativeSamplesNode);
    CosDistanceWithNegativeSamplesNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    void BackpropToMap(const size_t inputIndex)
    {
        if (inputIndex > 1)
            InvalidArgument("CosDistanceWithNegativeSamples operation only takes grdients on the first two inputs.");

        BackpropToS(inputIndex, *m_invNorm0, *m_invNorm1, Value(), *m_temp, *m_rightTerm, *m_leftTerm, *m_invNormSquare, Input(0)->Value(), Input(1)->Value(), Input(2)->Value(), Input(3)->Value(), Input(inputIndex)->Gradient(), Gradient());
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (fr.IsAllFrames())
        {
            BackpropToMap(inputIndex);
            return;
        } // TODO: remove these one by one
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        Matrix<ElemType> sliceInputGrad = Input(inputIndex)->GradientFor(fr);
        Matrix<ElemType> sliceThisGrad = GradientFor(fr);

        BackpropToS(inputIndex, *m_invNorm0, *m_invNorm1, sliceOutputValue, *m_temp, *m_rightTerm, *m_leftTerm, *m_invNormSquare, sliceInput0Value, sliceInput1Value, Input(2)->Value(), Input(3)->Value(), sliceInputGrad, sliceThisGrad);
    }

    // functionValues, invNorm0, invNorm1 - output from the EvaluateNode() method
    // temp, rightTerm, leftTerm - temporary matrices
    // in0, in1, in2, in3 - input functionValues from other nodes
    // inputGradientValues(x) - gradients to update, where x matches inputIndex
    /*TODO: merge with call site*/ void BackpropToS(const size_t inputIndex, const Matrix<ElemType>& invNorm0, const Matrix<ElemType>& invNorm1, const Matrix<ElemType>& functionValues,
                                                    Matrix<ElemType>& temp, Matrix<ElemType>& rightTerm, Matrix<ElemType>& leftTerm, Matrix<ElemType>& invNormSquare, // the temporary variables
                                                    const Matrix<ElemType>& in0, const Matrix<ElemType>& in1, const Matrix<ElemType>& in2, const Matrix<ElemType>& in3,
                                                    Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& thisGradientValues)
    {
        size_t shift = (size_t) in2.Get00Element();
        size_t negNumber = (size_t) in3.Get00Element();
        size_t numCols = in0.GetNumCols(); // used in computing right child's graident

        if (inputIndex == 0) // left derivative
        {
            invNormSquare.AssignElementProductOf(invNorm0, invNorm0);

            for (long m = 0; m < negNumber + 1; m++)
            {
                temp.GetARowByIndex(functionValues, m); // set this matrx to be the m-th row in functionValues
                temp.ElementMultiplyWith(invNormSquare);

                Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in0, rightTerm, 0, true);

                if (m == 0)
                {
                    temp.AssignElementProductOf(invNorm0, invNorm1);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in1, leftTerm, 0, true);
                }
                else
                {
                    size_t currshift = m + shift - 1; // for current line, how much should we shift

                    temp.AssignElementProductOfWithShift(invNorm0, invNorm1, currshift); // this is a row vector

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in1, leftTerm, currshift, true);
                }

                leftTerm = leftTerm - rightTerm;

                temp.GetARowByIndex(thisGradientValues, m);

                Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, 0, true);

                inputGradientValues += rightTerm;
            }
        }
        else // right part
        {
            invNormSquare.AssignElementProductOf(invNorm1, invNorm1); //this matrix should be save and unchanged. It should not be changed

            for (long m = 0; m < negNumber + 1; m++)
            {
                temp.GetARowByIndex(functionValues, m); // set this matrx to be the m-th row in functionValues

                if (m == 0) // this is the first line. computation should be symmetric
                {
                    // the following is for the right part
                    temp.ElementMultiplyWith(invNormSquare);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in1, rightTerm, 0, true);

                    // the following is for the left part
                    temp.AssignElementProductOf(invNorm0, invNorm1);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in0, leftTerm, 0, true);

                    leftTerm = leftTerm - rightTerm;

                    temp.GetARowByIndex(thisGradientValues, m);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, 0, true);

                    inputGradientValues += rightTerm;
                }
                else // this requires shift
                {
                    size_t currshift = (m + shift - 1) % numCols;
                    size_t reverseshift = numCols - currshift;

                    leftTerm.AssignElementProductOfWithShift(invNormSquare, temp, reverseshift); //use leftTerm as a temp variable here

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(leftTerm, in1, rightTerm, 0, true);

                    temp.AssignElementProductOfWithShift(invNorm1, invNorm0, reverseshift);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, in0, leftTerm, reverseshift, true);

                    leftTerm = leftTerm - rightTerm;

                    temp.GetARowByIndex(thisGradientValues, m);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, reverseshift, false);

                    inputGradientValues += rightTerm;
                }
            }
        }
    }

    void ForwardPropMap() // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
    {
        ForwardPropS(*m_invNorm0, *m_invNorm1, Value(), Input(0)->Value(), Input(1)->Value(), Input(2)->Value(), Input(3)->Value(), *m_leftTerm, *m_rightTerm);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        //if (fr.IsAllFrames()) { ForwardPropMap(); return; }
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        ForwardPropS(*m_invNorm0, *m_invNorm1, sliceOutputValue, sliceInput0Value, sliceInput1Value, Input(2)->Value(), Input(3)->Value(), *m_leftTerm, *m_rightTerm);
    }

    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType>& invNorm0, Matrix<ElemType>& invNorm1, Matrix<ElemType>& functionValues, Matrix<ElemType>& in0, Matrix<ElemType>& in1, Matrix<ElemType>& in2, Matrix<ElemType>& in3, Matrix<ElemType>& leftTermTemp, Matrix<ElemType>& rightTermTemp)
    {
        invNorm0.AssignVectorNorm2Of(in0, true); // seems to modify input (in0)
        invNorm0.AssignElementInverseOf(invNorm0);

        invNorm1.AssignVectorNorm2Of(in1, true); // seems to modify the input (in1)
        invNorm1.AssignElementInverseOf(invNorm1);

        size_t shift = (size_t) in2.Get00Element();
        size_t negNumber = (size_t) in3.Get00Element();

        // mutiply invNorm0 and invNorm1 with shift and neg.
        // The result is a matrix of (numberneg+1, invNorm0.Cols)
        leftTermTemp.AssignElementProductOfWithShiftNeg(invNorm0, invNorm1, shift, negNumber);

        // compute the right values
        // Again, the ouput is a matrix of (negNumber+1, invNorm0.cols)
        rightTermTemp.AssignInnerProductOfWithShiftNeg(in0, in1, true, shift, negNumber);

        // compute the evaluation result matrix by multiply these two matrices, element by element
        // we get a (negNumber+1, n) matrix
        functionValues.AssignElementProductOf(leftTermTemp, rightTermTemp);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        //if dimension is missing make the two operatants to have same size
        // TODO: use a for loop?? Or don't we have a function for this?
        size_t index = 0;
        {
            size_t rows = Input(index)->GetNumRows() == 0 ? Input(1 - index)->GetNumRows() : Input(index)->GetNumRows();
            size_t cols = Input(index)->GetNumCols() == 0 ? Input(1 - index)->GetNumCols() : Input(index)->GetNumCols();
            ValidateInferInputDims(index, rows, cols);
        }

        index = 1;
        {
            size_t rows = Input(index)->GetNumRows() == 0 ? Input(1 - index)->GetNumRows() : Input(index)->GetNumRows();
            size_t cols = Input(index)->GetNumCols() == 0 ? Input(1 - index)->GetNumCols() : Input(index)->GetNumCols();
            ValidateInferInputDims(index, rows, cols);
        }

        if (isFinalValidationPass &&
            (Input(1)->GetNumRows() != Input(0)->GetNumRows() ||
             (!Input(1)->GetMBLayout() && Input(1)->GetNumCols() != Input(0)->GetNumCols())))
        {
            LogicError("The Matrix dimension in the %ls %ls operation does not match.", NodeName().c_str(), OperationName().c_str());
        }

        // input(2) is shift, input(3) is the #neg
        size_t negNumber = (size_t) Input(3)->Get00Element();

        // TODO: This calls for a tensor representation!
        SetDims(TensorShape(negNumber + 1), Input(1)->GetNumCols());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CosDistanceWithNegativeSamplesNode<ElemType>>(nodeP);
            *node->m_invNorm0 = *m_invNorm0;
            *node->m_invNorm1 = *m_invNorm1;
            *node->m_invNormSquare = *m_invNormSquare;
            *node->m_leftTerm = *m_leftTerm;
            *node->m_rightTerm = *m_rightTerm;
            *node->m_temp = *m_temp;
        }
    }
    //request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_invNorm0, matrixPool);
        RequestMatrixFromPool(m_invNorm1, matrixPool);
        RequestMatrixFromPool(m_leftTerm, matrixPool);
        RequestMatrixFromPool(m_rightTerm, matrixPool);
    }

    //request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_invNormSquare, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_invNorm0, matrixPool);
        ReleaseMatrixToPool(m_invNorm1, matrixPool);
        ReleaseMatrixToPool(m_leftTerm, matrixPool);
        ReleaseMatrixToPool(m_rightTerm, matrixPool);
        ReleaseMatrixToPool(m_invNormSquare, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
    }

private:
    // invNorm nodes tranfer data between ForwardProp and BackpropTo
    shared_ptr<Matrix<ElemType>> m_invNorm0;
    shared_ptr<Matrix<ElemType>> m_invNorm1;
    shared_ptr<Matrix<ElemType>> m_leftTerm;
    shared_ptr<Matrix<ElemType>> m_rightTerm;
    // the rest are temporaries, values don't need to be maintained
    shared_ptr<Matrix<ElemType>> m_invNormSquare;
    shared_ptr<Matrix<ElemType>> m_temp;
};

template class CosDistanceWithNegativeSamplesNode<float>;
template class CosDistanceWithNegativeSamplesNode<double>;
} } }
