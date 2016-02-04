//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
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
#include <utility>
#include <assert.h>

namespace Microsoft { namespace MSR { namespace CNTK {

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
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient = GradientTensorFor(rank, fr);
        auto inputGradient = Input(inputIndex)->GradientTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);

        inputGradient.AddCopyOf(gradient);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // static int c = 0; if (c++ == 0) { fprintf(stderr, "#PLUS#\n"); }
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input0 = Input(0)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignSumOf(input0, input1);
    }
};


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
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);

        inputGradient.AddCopyOf(gradient, sign);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input0 = Input(0)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignDifferenceOf(input0, input1);
    }
};


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


// -----------------------------------------------------------------------
// TimesNodeBase (A, B)
// shared code of TimesNode and TransposeTimesNode (which transposes A)
// right operand and output can have MB layout, while left operand cannot
// -----------------------------------------------------------------------

template <class ElemType, bool m_transpose>
class TimesNodeBase : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    TimesNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex == 0) // left derivative
        {
            // this potentially computes inner products over time, so we use the Masked- variants
            auto sliceOutputGrad = MaskedGradientFor(fr);
            auto sliceInput1Value = Input(1)->MaskedValueFor(fr);
            auto& input0Grad = Input(0)->GradientAsMatrix();

            // currently we only support one combination when the input is sparse.
            if (sliceInput1Value.GetMatrixType() == SPARSE && Input(0)->Gradient().GetMatrixType() == DENSE && sliceOutputGrad.GetMatrixType() == DENSE)
                Input(0)->Gradient().SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

            bool transpose = m_transpose; // (assigning to a non-const variable avoids a compiler warning C4127: conditional expression is constant)
            if (!transpose)
                Matrix<ElemType>::MultiplyAndAdd(sliceOutputGrad, false, sliceInput1Value, true, input0Grad);
            else
                Matrix<ElemType>::MultiplyAndAdd(sliceInput1Value, false, sliceOutputGrad, true, input0Grad);
        }
        else // right derivative
        {
            auto sliceInput1Grad = Input(1)->GradientFor(fr);
            auto sliceOutputGrad = GradientFor(fr);

            Matrix<ElemType>::MultiplyAndAdd(Input(0)->ValueAsMatrix(), !m_transpose, sliceOutputGrad, false, sliceInput1Grad);
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
        // right operand and output can have MB layout, while left operand cannot
        auto sliceInput1Value = Input(1)->ValueFor(fr);
        auto sliceOutputValue = ValueFor(fr);
#if DUMPOUTPUT
        Input(0)->ValueAsMatrix().Print("TimesNode - Input0");
#endif
        // BUGBUG: This uses correct Matrix dimensions when multiplying with a non-minibatch only by luck. To be fixed when we allow to apply TimesNode to a subset of tensor dimensions.
        sliceOutputValue.AssignProductOf(Input(0)->ValueAsMatrix(), m_transpose, sliceInput1Value, false);
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
            InvalidArgument("%ls Times operation requires the first factor to not be minibatch data (must not have an MBLayout).", NodeName().c_str());
        InferMBLayoutFromInputsForStandardCase();

        // support automatic dimension inference for learnable parameters
        size_t rows0 = Input(0)->GetAsMatrixNumRows(), cols0 = Input(0)->GetAsMatrixNumCols();
        bool transpose = m_transpose; // (assigning to a non-const variable avoids a compiler warning C4127: conditional expression is constant)
        if (transpose)
            std::swap(rows0, cols0);
        size_t rows1 = Input(1)->HasMBLayout() ? Input(1)->GetSampleMatrixNumRows() : Input(1)->GetAsMatrixNumRows();

        // limited automatic dimension inference for *children*, useful for CNN since it can be hard to know the size of each input parameter without deep knowledge how CNN is implemented (padding, stride)
        // infer cols0 as rows1
        Input(0)->ValidateInferInputDimsFrom(m_transpose ? TensorShape(rows1, rows0) : TensorShape(rows0, rows1));

        // TODO: With tensors, inner dimensions must match.
        // after multiplication the tensor structure is lost
        if (Input(1)->HasMBLayout())
        {
            // infer rows1 as cols0
            Input(1)->ValidateInferInputDimsFrom(TensorShape(cols0));
            SetDims(TensorShape(rows0), true);
        }
        else // multiplying two straight matrices
        {
            size_t cols1 = Input(1)->GetAsMatrixNumCols();
            // infer rows1 as cols0
            Input(1)->ValidateInferInputDimsFrom(TensorShape(cols0, cols1));
            SetDims(TensorShape(rows0, cols1), false);
        }

        // update after inference
        cols0 = m_transpose ? Input(0)->GetAsMatrixNumRows() : Input(0)->GetAsMatrixNumCols();
        rows1 = Input(1)->HasMBLayout() ? Input(1)->GetSampleMatrixNumRows() : Input(1)->GetAsMatrixNumRows();
        if (isFinalValidationPass && cols0 != rows1)
            InvalidArgument("The inner matrix dimension in the %ls Times operation does not match (%d vs. %d).", NodeName().c_str(), (int) rows1, (int) cols0);
    }

    virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool) override
    {
        // this is a special handling case. We need to allocate sparse matrix directly instead of from pool.
        if (Input(0)->NeedGradient() && Input(1)->Value().GetMatrixType() == SPARSE)
        {
            Input(0)->CreateGradientMatrixIfNull();
            Input(0)->Gradient().SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);
        }

        // we need to call base allocation at end since we will need to allocate special ones first
        // so that the default allocator will not allocate it again.
        Base::AllocateGradientMatricesForInputs(matrixPool);
    }
};

// -----------------------------------------------------------------------
// TimesNode (A, B)
// right operand and output can have MB layout, while left operand cannot
// -----------------------------------------------------------------------

template <class ElemType>
class TimesNode : public TimesNodeBase<ElemType, false>
{
    typedef TimesNodeBase<ElemType, false> Base;
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
};


// -----------------------------------------------------------------------
// TransposeTimesNode (A', B)
// right operand and output can have MB layout, while left operand cannot
// -----------------------------------------------------------------------

template <class ElemType>
class TransposeTimesNode : public TimesNodeBase<ElemType, true>
{
    typedef TimesNodeBase<ElemType, true> Base;
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
};


// -----------------------------------------------------------------------
// ElementTimesNode (factor1, factor2)
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
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);
        if (Input(inputIndex)->ReducesInTimeWrt(Input(1 - inputIndex)))
            Input(1 - inputIndex)->MaskMissingValueColumnsToZero(fr);

        inputGradient.AddElementwiseProductOf(gradient, otherInputValue);
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
    {
        return true;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto input0 = Input(0)->ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = Input(1)->ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignElementwiseProductOf(input0, input1);
    }
};


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
            Input(0)->GradientAsMatrix() += *m_innerproduct;
        }
        else // right derivative
        {
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);
            Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);
            m_rightGradient->SetValue(sliceOutputGrad);
            m_rightGradient->ColumnElementMultiplyWith(Input(0)->ValueAsMatrix());
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
        sliceOutputValue.ColumnElementMultiplyWith(Input(0)->ValueAsMatrix());
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        size_t rows0 = Input(0)->GetAsMatrixNumRows();
        size_t rows1 = Input(1)->HasMBLayout() ? Input(1)->GetSampleMatrixNumRows() : Input(1)->GetAsMatrixNumRows();

        // if dimension not specified we assume two operands' dimensions should match
        Input(0)->ValidateInferInputDimsFrom(TensorShape(rows1));

        if (Input(1)->HasMBLayout())
        {
            // infer rows1 as rows0
            Input(1)->ValidateInferInputDimsFrom(TensorShape(rows0));
            SetDims(TensorShape(rows0), true);
        }
        else // multiplying two straight matrices
        {
            size_t cols1 = Input(1)->GetAsMatrixNumCols();
            // infer rows1 as rows0
            Input(1)->ValidateInferInputDimsFrom(TensorShape(rows0, cols1));
            SetDims(TensorShape(rows0, cols1), false);
        }

        // update after inference
        rows0 = Input(0)->GetAsMatrixNumRows();
        rows1 = Input(1)->HasMBLayout() ? Input(1)->GetSampleMatrixNumRows() : Input(1)->GetAsMatrixNumRows();
        if (isFinalValidationPass && rows0 != rows1)
            InvalidArgument("The inner matrix dimension in the %ls %ls operation does not match (%d vs. %d).", NodeName().c_str(), OperationName().c_str(), (int) rows1, (int) rows0);
        size_t cols0 = Input(0)->GetAsMatrixNumCols();
        if (isFinalValidationPass && cols0 != 1)
            InvalidArgument("The first matrix should be a column vector representing the diagonal of a square matrix in the DiagTimes operation.");

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
    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_innerproduct, matrixPool);
        RequestMatrixFromPool(m_rightGradient, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
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
        SetDims(TensorShape(1), false);
    }
};


// -----------------------------------------------------------------------
// SumColumnElementsNode (input)
// sums up all elements in each column of the input, reducing each column to a scalar
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
        auto sliceInputGrad = Input(0)->GradientFor(fr);
        auto sliceOutputGrad = GradientFor(fr);

        sliceInputGrad += sliceOutputGrad; // here the assumption is that sliceOutputGrad is a row vector
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
        auto sliceInputValue = Input(0)->ValueFor(fr);
        auto sliceOutputValue = ValueFor(fr); // row vector

        Matrix<ElemType>::VectorSum(sliceInputValue, sliceOutputValue, true);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        SetDims(TensorShape(1), Input(0)->HasMBLayout()); // each column is reduced to a scalar
    }
};


#ifdef COMING_SOON  // known bug in backprop; generalize to tensor

// -----------------------------------------------------------------------
// TransposeNode (input matrix)
// TODO: extend towards tensor transpose (swap 2 dimensions, incl. time)
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
        auto& inputGradientValues = Input(0)->GradientAsMatrix();
        auto& gradientValues = GradientAsMatrix();
#if DUMPOUTPUT
        gradientValues.Print("Gradient-in");
        inputGradientValues.Print("child Gradient-in/out");
        inputFunctionValues.Print("child Function values");
#endif
        const Matrix<ElemType>& ones = ConstOnes(inputGradientValues.GetNumRows(), inputGradientValues.GetNumRows(), inputGradientValues.GetDeviceId());
        // BUGBUG: This should be ^^ Identity(). This will be fixed once we switch to the more generic tensor Transpose operation, which can handle this easily.
        Matrix<ElemType>::MultiplyAndAdd(ones, false, gradientValues, true, inputGradientValues);
#if DUMPOUTPUT
        inputGradientValues.Print("child Gradient-out");
#endif
        InvalidArgument("TransposeNode::BackpropTo() has a known bug. it is not functional.");
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
        Input(0)->ValueAsMatrix().Print("TransposeNode- Input0");
#endif
        ValueAsMatrix().AssignTransposeOf(Input(0)->ValueAsMatrix());
#if NANCHECK
        Value().HasNan("Transpose");
#endif
#if DUMPOUTPUT
        ValueAsMatrix().Print("TransposeNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation cannot operate on minibatch data (which have a layout)", NodeName().c_str(), OperationName().c_str());
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        size_t rows0 = Input(0)->GetAsMatrixNumRows(), cols0 = Input(0)->GetAsMatrixNumCols();
        SetDims(TensorShape(cols0, rows0), false);
    }
};


#endif

// -----------------------------------------------------------------------
// CosDistanceNode (left, right)
// column-wise cos distance
// TODO: Would it be useful to allow one of the two to be a single column?
// -----------------------------------------------------------------------

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
        if (inputIndex == 0) // left derivative
            m_temp->AssignElementProductOf(*m_invNorm0, *m_invNorm0);
        else // right derivative
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

        // TODO: We could do something more interesting with tensors.
        //       E.g. apply a cos distance of a whole set of data with a single reference.
        SetDims(TensorShape(1), Input(1)->HasMBLayout());
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
    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_invNorm0, matrixPool);
        RequestMatrixFromPool(m_invNorm1, matrixPool);
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_leftTerm, matrixPool);
        RequestMatrixFromPool(m_rightTerm, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
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


// -----------------------------------------------------------------------
// KhatriRaoProductNode (left, right)
// compute an outer product of column vectors (for each sample)
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

        if (inputIndex == 0) // left derivative
        {
            Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
            Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);

            sliceInput0Grad.AddColumnReshapeProductOf(sliceOutputGrad, sliceInput1Value, false);
        }
        else // right derivative
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

        size_t rows0 = Input(0)->GetSampleMatrixNumRows();
        size_t rows1 = Input(1)->GetSampleMatrixNumRows();

        // after KhatriRaoProduct the structure is lost
        // TODO: ^^ Is that correct? Should we use a tensor here, TensorShape(rows0, rows1)?
        SetDims(TensorShape(rows0 * rows1), HasMBLayout());
    }
};


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

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
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
            invNormSquare.AssignElementProductOf(invNorm1, invNorm1); // this matrix should be save and unchanged. It should not be changed

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

                    leftTerm.AssignElementProductOfWithShift(invNormSquare, temp, reverseshift); // use leftTerm as a temp variable here

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

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
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

        ValidateInferBinaryInputDims();

        if (isFinalValidationPass &&
            (Input(0)->GetSampleMatrixNumRows() != Input(1)->GetSampleMatrixNumRows() || Input(0)->GetMBLayout() != Input(1)->GetMBLayout()))
        {
            LogicError("The tensor dimension in the %ls %ls operation does not match.", NodeName().c_str(), OperationName().c_str());
        }

        // input(2) is shift, input(3) is the #neg
        size_t negNumber = (size_t) Input(3)->Get00Element();

        // TODO: This calls for a tensor representation!
        SetDims(TensorShape(negNumber + 1), HasMBLayout());
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
    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_invNorm0, matrixPool);
        RequestMatrixFromPool(m_invNorm1, matrixPool);
        RequestMatrixFromPool(m_leftTerm, matrixPool);
        RequestMatrixFromPool(m_rightTerm, matrixPool);
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_invNormSquare, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
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

} } }
