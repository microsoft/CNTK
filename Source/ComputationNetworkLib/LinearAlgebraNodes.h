//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Constants.h"
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
#include <set>
#include "Quantizers.h"
#include "InputAndParamNodes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// PlusNode (summand1, summand2)
// -----------------------------------------------------------------------

template <class ElemType>
class PlusNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base; UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName() { return L"Plus"; }

public:
    DeclareConstructorFromConfigWithNumInputs(PlusNode);
    PlusNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result =             ValueTensorFor(rank, fr);
        auto input0 = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignSumOf(input0, input1);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient      =                    GradientTensorFor(rank, fr);
        auto inputGradient = Input(inputIndex)->GradientTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);

        if (Input(inputIndex)->ParentOverwritesGradient())
            inputGradient.AssignCopyOf(gradient);
        else
            inputGradient.AddCopyOf(gradient);
    }

    virtual bool ImplementsGradientOverwriteOptimization() const override { return true; }
};

template class PlusNode<float>;
template class PlusNode<double>;

// -----------------------------------------------------------------------
// LogPlusNode (summand1, summand2)
// Computes ln(exp(summand1) + exp(summand2)) in an overflow safe way.
// Useful e.g. for computing softmax over sequence.
// -----------------------------------------------------------------------

template <class ElemType>
class LogPlusNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base; UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName() { return L"LogPlus"; }

public:
    DeclareConstructorFromConfigWithNumInputs(LogPlusNode);
    LogPlusNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result =             ValueTensorFor(rank, fr);
        auto input0 = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignLogSumOf(input0, input1);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient      =                    GradientTensorFor(rank, fr);
        auto inputGradient = InputRef(inputIndex).GradientTensorFor(rank, fr.AllowBroadcast());
        auto input0        = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1        = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());        

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);
        if (Input(inputIndex)->ReducesInTimeWrt(Input(1 - inputIndex)))
            Input(1 - inputIndex)->MaskMissingValueColumnsToZero(fr);

        if (inputIndex == 0)
        {
            // d/dx (ln( exp(x) + (exp(y)) = exp(x) / (exp(x) + exp(y)) = 1 / (1 + exp(y-x)) = sigmoid(x-y)
            inputGradient.AddElementwiseProductWithLogSumDerivativeOf(gradient, input1, input0);
        }
        else
        {
            // d/dy (ln( exp(x) + (exp(y)) = exp(y) / (exp(x) + exp(y)) = 1 / (1 + exp(x-y)) = sigmoid(y-x)
            inputGradient.AddElementwiseProductWithLogSumDerivativeOf(gradient, input0, input1);
        }
    }
};

template class LogPlusNode<float>;
template class LogPlusNode<double>;


// -----------------------------------------------------------------------
// PowNode (base, exponent)
// Computes base ** exponent.
// -----------------------------------------------------------------------

template <class ElemType>
class PowNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base; UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName() { return L"Pow"; }

public:
    DeclareConstructorFromConfigWithNumInputs(PowNode);
    PowNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result = ValueTensorFor(rank, fr);
        auto base   = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto expo   = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignPowOf(base, expo);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient = GradientTensorFor(rank, fr);
        auto inputGradient = InputRef(inputIndex).GradientTensorFor(rank, fr.AllowBroadcast());
        auto base = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);
        if (Input(inputIndex)->ReducesInTimeWrt(Input(1 - inputIndex)))
            Input(1 - inputIndex)->MaskMissingValueColumnsToZero(fr);


        if (inputIndex == 0)
        {
            auto exponent = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
            // d/dx x**y = y * x**(y-1)
            inputGradient.AddElementwiseProductWithPowBaseDerivativeOf(gradient, base, exponent);
        }
        else
        {
            auto result = ValueTensorFor(rank, fr);
            // d/dy x**y = ln(x) * x**y
            inputGradient.AddElementwiseProductWithPowExponentDerivativeOf(gradient, result, base);
        }
    }
};

template class PowNode<float>;
template class PowNode<double>;


// -----------------------------------------------------------------------
// MinusNode (minuend, subtrahend)
// -----------------------------------------------------------------------

template <class ElemType>
class MinusNode : public BinaryElementWiseNode<ElemType>
{
    typedef BinaryElementWiseNode<ElemType> Base; UsingBinaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName() { return L"Minus"; }

public:
    DeclareConstructorFromConfigWithNumInputs(MinusNode);
    MinusNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto result =             ValueTensorFor(rank, fr);
        auto input0 = InputRef(0).ValueTensorFor(rank, fr.AllowBroadcast());
        auto input1 = InputRef(1).ValueTensorFor(rank, fr.AllowBroadcast());
        result.AssignDifferenceOf(input0, input1);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto gradient      =                    GradientTensorFor(rank, fr);
        auto inputGradient = Input(inputIndex)->GradientTensorFor(rank, fr.AllowBroadcast());

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);

        ElemType sign = inputIndex == 0 ? 1.0f : -1.0f;
        inputGradient.AddCopyOf(gradient, sign);
    }
};

template class MinusNode<float>;
template class MinusNode<double>;

// -----------------------------------------------------------------------
// ElementTimesNode (factor1, factor2)
// This allows broadcasting, and can thus also scale with a row, a column, or a scalar,
// as well as mutliplying with a diagonal matrix (if represented as a column vector).
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

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        ForwardPropImpl(*this, fr, true/*allowBroadcast*/);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        BackpropToImpl(*this, inputIndex, fr, true/*allowBroadcast*/);
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return true; }

    template <typename classType>
    static void ForwardPropImpl(classType& c, const FrameRange& fr, bool allowBroadcast)
    {
        size_t rank = c.DetermineElementwiseTensorRank();
        auto result =             c.ValueTensorFor(rank, fr);
        auto input0 = c.InputRef(0).ValueTensorFor(rank, allowBroadcast ? fr.AllowBroadcast() : fr);
        auto input1 = c.InputRef(1).ValueTensorFor(rank, allowBroadcast ? fr.AllowBroadcast() : fr);
        result.AssignElementwiseProductOf(input0, input1);
    }

    template <typename classType>
    static void BackpropToImpl(classType& c, const size_t inputIndex, const FrameRange& fr, bool allowBroadcast)
    {
        size_t rank = c.DetermineElementwiseTensorRank();
        auto gradient        =                        c.GradientTensorFor(rank, fr);
        auto inputGradient   = c.Input(    inputIndex)->GradientTensorFor(rank, allowBroadcast ? fr.AllowBroadcast() : fr);
        auto otherInputValue = c.Input(1 - inputIndex)->ValueTensorFor   (rank, allowBroadcast ? fr.AllowBroadcast() : fr);

        // if reduction then mask the respective input(s) (zero out the gaps)
        if (c.Input(inputIndex)->ReducesInTimeWrt(c.shared_from_this()))
            c.MaskMissingGradientColumnsToZero(fr);
        if (c.Input(inputIndex)->ReducesInTimeWrt(c.Input(1 - inputIndex)))
            c.Input(1 - inputIndex)->MaskMissingValueColumnsToZero(fr);

        if (c.Input(inputIndex)->ParentOverwritesGradient())
            inputGradient.AssignElementwiseProductOf(gradient, otherInputValue);
        else
            inputGradient.AddElementwiseProductOf(gradient, otherInputValue);
    }
};

template class ElementTimesNode<float>;
template class ElementTimesNode<double>;

// -----------------------------------------------------------------------
// TimesNodeBase (A, B, outputRank=1)
// shared code of TimesNode and TransposeTimesNode (which transposes A)
// The common case, W * v with weights W and minibatch data v is efficiently
// implemented as a per-minibatch BLAS GEMM call.
// If A is minibatch data, then this operation is currently not efficient.
// TODO: Implement this with TensorView::DoElementwiseProductOf() and stride magic
// TODO: Transpose flags for all matrices, inputs and outputs?
// TODO: allow outputRank < 0 meaning to denote "all but", from right
// -----------------------------------------------------------------------

template <class ElemType, bool m_transpose>
class TimesNodeBase : public ComputationNode<ElemType>, public NumInputs<2>
{
    friend class ElementTimesNode<ElemType>;

    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers; using Base::OperationName;                                                                                                                           \

public:
    enum : int
    {
        NoInferredInputRank = -1,                        // the default, do not infer left operand input rank from right operand
        ReduceSequenceAxisWithoutInferredInputRank = -2, // reduce sequence axis. Currently only support cases like (m x k) x (k) -> (m) for sequences
    };

public:
    TimesNodeBase(DEVICEID_TYPE deviceId, const wstring& name, size_t outputRank = 1, int inferInputRankToMap = NoInferredInputRank)
        : Base(deviceId, name), m_outputRank(outputRank), m_inferInputRankToMap(inferInputRankToMap), m_beingUnrolled(false)
    {
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<TimesNodeBase<ElemType, m_transpose>>(nodeP);
            node->m_outputRank          = m_outputRank;
            node->m_inferInputRankToMap = m_inferInputRankToMap;
        }
    }

    void Save(File& fstream) const
    {
        Base::Save(fstream);
        fstream << m_outputRank;
        fstream << m_inferInputRankToMap;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        if (modelVersion >= CNTK_MODEL_VERSION_3)
            fstream >> m_outputRank;
        else
            m_outputRank = 1;
        if (modelVersion >= CNTK_MODEL_VERSION_12)
            fstream >> m_inferInputRankToMap;
        else
            m_inferInputRankToMap = NoInferredInputRank;
    }

protected:
    // if the left argument of the matrix product (A) has a time axis, it can only be applied sample by sample
    // where each sample is treated as a separate matrix object (as a consequence, it then also applies to B and the result as well)
    TensorView<ElemType> OneSampleTensorFor(int inputIndex/*-1 for output*/, bool gradient/*instead of value*/, const FrameRange& fr)
    {
        auto input = inputIndex < 0 ? this : Input(inputIndex).get();
        auto data = gradient ? input->GradientPtr() : input->ValuePtr();
        size_t rank = input->GetSampleLayout().GetRank();
        if (inputIndex == 0 && m_transpose && rank == 1) // transposing a 1D tensor implies it is really a 2D tensor. Note that m_transpose applies to left operand only.
            rank = 2;
        if (!InputRef(0).HasMBLayout()) // left input is no MB data: run normally
            return input->DataTensorFor(data, rank, fr);
        auto tensorShape = input->GetOneSampleTensorSliceFor(rank, fr);
        return TensorView<ElemType>(data, tensorShape);
    }

private:
    // Check if TimesNodeBase could be simplified to ElementTimes to avoid unroll when:
    // 1. input0: is rank-1 and transposed, or is rank-2 with Dim(0)==1
    // 2. input1: is rank-1
    // 3. output: rank-1 and reduced to a scalar (Dim(0)==1)
    // 4. m_transpose (becomes Matrix::InnerProduct), or both input being dense
    bool IsReduceableDotProduct(const FrameRange& fr, bool& hasSparse)
    {
        const auto& shape0   = InputRef(0).GetSampleLayout();
        const auto& shape1   = InputRef(1).GetSampleLayout();
        const auto& shapeOut =             GetSampleLayout();

        bool input0Sparse = (InputRef(0).Value().GetMatrixType() != DENSE);
        bool input1Sparse = (InputRef(1).Value().GetMatrixType() != DENSE);

        bool input0_ok =
            (shape0.GetRank() == 1 && m_transpose) ||
            (shape0.GetRank() == 2 && shape0.GetDim(0) == 1);

        bool input1_ok =
            (shape1.GetRank() == 1);

        bool outputScalar =
            (shapeOut.GetRank() == 1) &&
            (shapeOut.GetDim(0) == 1);

        bool notBothSparse = !(input0Sparse && input1Sparse);

        hasSparse = (input0Sparse || input1Sparse);
        
        return input0_ok && input1_ok && outputScalar && notBothSparse && (m_transpose || !hasSparse);
    }

    void RequestReduceSequenceAxisMatricesIfNeeded(MatrixPool& matrixPool)
    {
        if (!ReduceSequenceAxis()) return;

        for (int i = 0; i < NumInputs; i++)
        {
            RequestMatrixFromPool(m_tempScatterIndices[i], matrixPool, InputRef(i).GetMBLayout()->GetNumCols(), true);
            const auto& packedData = InputRef(i).Value();
            if (packedData.GetMatrixType() == DENSE)
                RequestMatrixFromPool(m_tempUnpackedValue[i], matrixPool, InputRef(i).GetSampleLayout().GetNumElements(), true);
            else
                m_tempUnpackedValue[i] = std::make_shared<Matrix<ElemType>>(packedData.GetNumRows(), packedData.GetNumCols(), packedData.GetDeviceId(), packedData.GetMatrixType(), packedData.GetFormat());
        }
    }

    void ReleaseReduceSequenceAxisMatricesIfNeeded(MatrixPool& matrixPool)
    {
        if (!ReduceSequenceAxis()) return;

        for (int i = 0; i < NumInputs; i++)
        {
            ReleaseMatrixToPool(m_tempScatterIndices[i], matrixPool);
            if (InputRef(i).Value().GetMatrixType() == DENSE)
                ReleaseMatrixToPool(m_tempUnpackedValue[i], matrixPool);
            else
                m_tempUnpackedValue[i].reset();
        }
    }

    void ForwardProp_ReduceSequenceAxis()
    {
        // Input are stored as (m * k) x b*(batch axis) x s*(sequence axis) and k x b* x s*
        // We unpack them to (m * k) x s* x b* and k x s* x b*
        // Then perform b* matrix multiplies to get m x b* with both k and s* being resolved
        auto inputMBLayout = InputRef(0).GetMBLayout();
        auto numSequences = inputMBLayout->GetNumSequences(); // b*
        auto maxNumTimeSteps = inputMBLayout->GetNumTimeSteps(); // s*
        size_t m = InputRef(0).GetSampleLayout()[0];
        size_t k = InputRef(1).GetSampleLayout()[0];

        if (InputRef(1).Value().GetMatrixType() == SPARSE)
            LogicError("Right operand cannot be sparse in times reduce sequence axis");

        GetMBLayout()->InitAsFrameMode(numSequences);
        UpdateFunctionValuesSize();

        TensorView<ElemType> unpackedInput[NumInputs];
        for (int i = 0; i < NumInputs; i++)
        {
            ElemType gapPadValue = 0;
            unpackedInput[i] = ComputationNode<ElemType>::Unpack(
                InputRef(i).GetSampleLayout(),
                InputRef(i).Value(),
                inputMBLayout,
                m_tempUnpackedValue[i],
                m_tempScatterIndices[i],
                std::shared_ptr<Matrix<char>>(nullptr),
                /*batchMajor=*/ false,
                &gapPadValue);
        }

        // note the unpacked input is not the normal MBLayout (batchMajor) so do ColumnSlice directly
        const Matrix<ElemType>& mat0 = unpackedInput[0].GetSOB();
        const Matrix<ElemType>& mat1 = unpackedInput[1].GetSOB();

        // unroll in the batch axis, we may use batched GEMM in future
        for (int s = 0; s < numSequences; s++)
        {
            Matrix<ElemType> mat0Slice = mat0.ColumnSlice(s * maxNumTimeSteps, maxNumTimeSteps); // (m * k) x s*
            mat0Slice.Reshape(m, k * maxNumTimeSteps); // m x (k * s*)
            Matrix<ElemType> mat1Slice = mat1.ColumnSlice(s * maxNumTimeSteps, maxNumTimeSteps); // k x s*
            mat1Slice.Reshape(k * maxNumTimeSteps, 1); // (k * s*) x 1

            Matrix<ElemType> value = Value().ColumnSlice(s, 1);
            Matrix<ElemType>::Multiply(mat0Slice, false, mat1Slice, false, value);
        }
    }

    void BackpropTo_ReduceSequenceAxis(size_t inputIndex)
    {
        auto input0MBLayout = InputRef(0).GetMBLayout();
        auto numSequences = input0MBLayout->GetNumSequences(); // b*
        auto maxNumTimeSteps = input0MBLayout->GetNumTimeSteps(); // s*
        size_t m = InputRef(0).GetSampleLayout()[0];
        size_t k = InputRef(1).GetSampleLayout()[0];

        TensorView<ElemType> unpackedInput[NumInputs];
        bool unpacked[NumInputs];
        for (int i = 0; i < NumInputs; i++)
        {
            ElemType gapPadValue = 0;
            unpackedInput[i] = ComputationNode<ElemType>::Unpack(
                InputRef(i).GetSampleLayout(),
                InputRef(i).Value(),
                input0MBLayout, // the same for both operands
                m_tempUnpackedValue[i],
                m_tempScatterIndices[i],
                std::shared_ptr<Matrix<char>>(nullptr),
                /*batchMajor=*/ false,
                &gapPadValue);

            unpacked[i] = ((input0MBLayout->GetNumTimeSteps() > 1) && (input0MBLayout->GetNumSequences() > 1));
        }

        const auto& unpackedInputValue = unpackedInput[1 - inputIndex].GetSOB();

        ElemType beta = InputRef(inputIndex).ParentOverwritesGradient() ? (ElemType)0 : (ElemType)1;

        // note the unpacked input is not the normal MBLayout (batchMajor), so do ColumnSlice directly
        if (inputIndex == 0)
        {
            Matrix<ElemType> tempGradientUnpacked(m * k, maxNumTimeSteps * numSequences, InputRef(inputIndex).GetDeviceId());
            Matrix<ElemType>& inputGradientUnpacked = unpacked[inputIndex] ? tempGradientUnpacked : InputRef(inputIndex).Gradient();

            for (int s = 0; s < numSequences; s++)
            {
                Matrix<ElemType> inputGradientSlice = inputGradientUnpacked.ColumnSlice(s * maxNumTimeSteps, maxNumTimeSteps); // (m * k) x s*
                inputGradientSlice.Reshape(m, k * maxNumTimeSteps); // m x (k * s*)
                Matrix<ElemType> inputValueSlice = unpackedInputValue.ColumnSlice(s * maxNumTimeSteps, maxNumTimeSteps); // k x s*
                inputValueSlice.Reshape(k * maxNumTimeSteps, 1); // (k * s*) x 1
                Matrix<ElemType> gradientSlice = Gradient().ColumnSlice(s, 1); // m x 1
                Matrix<ElemType>::MultiplyAndWeightedAdd(1, gradientSlice, false, inputValueSlice, true, unpacked[inputIndex] ? 0 : beta, inputGradientSlice);
            }

            if (unpacked[inputIndex])
                InputRef(inputIndex).Gradient().DoGatherColumnsOf(beta, *m_tempScatterIndices[inputIndex], inputGradientUnpacked, (ElemType)1);
        }
        else
        {
            Matrix<ElemType> tempGradientUnpacked(k, maxNumTimeSteps * numSequences, InputRef(inputIndex).GetDeviceId());
            Matrix<ElemType>& inputGradientUnpacked = unpacked[inputIndex] ? tempGradientUnpacked : InputRef(inputIndex).Gradient();

            for (int s = 0; s < numSequences; s++)
            {
                Matrix<ElemType> inputGradientSlice = inputGradientUnpacked.ColumnSlice(s * maxNumTimeSteps, maxNumTimeSteps); // k x s*
                inputGradientSlice.Reshape(k * maxNumTimeSteps, 1); // (k * s*) x 1
                Matrix<ElemType> inputValueSlice = unpackedInputValue.ColumnSlice(s * maxNumTimeSteps, maxNumTimeSteps); // (m * k) x s*
                inputValueSlice.Reshape(m, k * maxNumTimeSteps); // m x (k * s*)
                Matrix<ElemType> gradientSlice = Gradient().ColumnSlice(s, 1); // m x 1
                Matrix<ElemType>::MultiplyAndWeightedAdd(1, inputValueSlice, true, gradientSlice, false, unpacked[inputIndex] ? 0 : beta, inputGradientSlice);
            }
            
            if (unpacked[inputIndex])
                InputRef(inputIndex).Gradient().DoGatherColumnsOf(beta, *m_tempScatterIndices[inputIndex], inputGradientUnpacked, (ElemType)1);
        }
    }

public:
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // If argument A is minibatch data, then this must be performed frame-by-frame, sequence-by-sequence, one GEMM call each.
        // This will be inefficient. We hope this will be the baseline of a future, more efficient TensorView-based implementation.
        auto inputMBLayout = InputRef(0).GetMBLayout();
        if (!fr.IsOneColumnWrt(inputMBLayout))
        {
            if (ReduceSequenceAxis())
            {
                // only works in PAR mode
                if (!fr.IsAllFrames())
                    RuntimeError("%ls %ls operation can perform sequence axis reduction only for all frames.", NodeName().c_str(), OperationName().c_str());

                if (inputMBLayout->HasSequenceBeyondBegin() || inputMBLayout->HasSequenceBeyondEnd())
                    RuntimeError("%ls %ls operation cannot perform sequence axis reduction for truncated sequence.", NodeName().c_str(), OperationName().c_str());

                ForwardProp_ReduceSequenceAxis();
                return;
            }

            // speed up using ElementTimes or InnerProduct to avoid unroll if possible
            bool hasSparse;
            if (IsReduceableDotProduct(fr, hasSparse))
            {
                // for sparse transposed, use InnerProduct
                if (hasSparse)
                {
                    Matrix<ElemType> value  =             ValueFor(fr);
                    Matrix<ElemType> input0 = InputRef(0).ValueFor(fr);
                    Matrix<ElemType> input1 = InputRef(1).ValueFor(fr);
                    if (input0.GetMatrixType() == SPARSE)
                        Matrix<ElemType>::InnerProduct(input0, input1, value, true/*isColWise*/);
                    else
                        Matrix<ElemType>::InnerProduct(input1, input0, value, true/*isColWise*/);
                    // TODO: better move this special-casing into TensorView::AssignElementwiseProductOf()
                }
                else
                {
                    ElementTimesNode<ElemType>::ForwardPropImpl(*this, fr, false/*allowBroadcast*/);
                }
                return;
            }

            // recursively call ourselves for each individual time and sequence

            // note this is not performant, warn user about the slow path being used
            if (Base::HasEnvironmentPtr() && Base::Environment().traceLevel > 0)
                std::call_once(m_unrollWarningOnceFlag, [this]{ fprintf(stderr, "WARNING: %ls %ls operation: being unrolled, execution may be slow\n", NodeName().c_str(), OperationName().c_str()); });

            auto timeRange     = fr.GetTimeRange();
            auto sequenceRange = fr.GetSequenceRange();
            m_beingUnrolled = true;
            for (auto t = timeRange.first; t < timeRange.second; t++)
                for (auto s = sequenceRange.first; s < sequenceRange.second; s++)
                    ForwardProp(fr.WithTimeStep(t).Sequence(s));
            m_beingUnrolled = false;
            return;
        }

        // TensorView::DoMatrixProductOf() will reduce each tensor object into a 2D tensor (or fail if it cannot)
        // and recreate actual Matrix objects (in case of sparse, they must be identical to the original tensor storage object).
        // Transposition is applied after flattening into 2D, but only allowed if the input sample is 2D anyway.
        auto input0 = OneSampleTensorFor(0,  /*gradient=*/false, fr.AllowBroadcast());
        auto input1 = OneSampleTensorFor(1,  /*gradient=*/false, fr.AllowBroadcast());
        auto output = OneSampleTensorFor(-1, /*gradient=*/false, fr);
        output.AssignMatrixProductOf(false/*transC*/, input0, m_transpose/*transA*/, input1, false/*transB*/, 1.0f, this->m_pQuantizedMultiplier);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // special treatment if A is minibatch data; see Forward() for comment
        if (!fr.IsOneColumnWrt(InputRef(0).GetMBLayout()))
        {
            if (ReduceSequenceAxis())
            {
                // only works in PAR mode
                if (!fr.IsAllFrames())
                    RuntimeError("%ls %ls operation can perform sequence axis reduction only for all frames.", NodeName().c_str(), OperationName().c_str());

                BackpropTo_ReduceSequenceAxis(inputIndex);
                return;
            }
            
            // speed up using ElementTimes or InnerProduct to avoid unroll if possible
            bool hasSparse;
            if (IsReduceableDotProduct(fr, hasSparse))
            {
                if (hasSparse)
                {
                    Matrix<ElemType> gradient = GradientFor(fr);
                    Matrix<ElemType> inputValue = InputRef(1 - inputIndex).ValueFor(fr);
                    Matrix<ElemType> inputGradient = InputRef(inputIndex).GradientFor(fr);
                    Matrix<ElemType>::ColumnwiseScaleAndWeightedAdd(
                        (ElemType)1.0, inputValue, gradient,
                        Input(inputIndex)->ParentOverwritesGradient() ? (ElemType)0.0 : (ElemType)1.0,
                        inputGradient);
                    // TODO: better move this special-casing into TensorView::AssignElementwiseProductOf()
                    // Note: We do not need to mask gaps here, since this code branch operates sample by sample (no reduction over samples).
                }
                else
                {
                    ElementTimesNode<ElemType>::BackpropToImpl(*this, inputIndex, fr, false/*allowBroadcast*/);
                }
                return;
            }

            auto timeRange     = fr.GetTimeRange();
            auto sequenceRange = fr.GetSequenceRange();
            // when unroll, parent overwrite gradient should be ignored
            m_beingUnrolled = true;
            if (Input(inputIndex)->ParentOverwritesGradient())
            {
                Input(inputIndex)->Gradient().SetValue(0);
            }
            for (auto t = timeRange.first; t < timeRange.second; t++) // step left to right to allow to build a sparse matrix
                for (auto s = sequenceRange.first; s < sequenceRange.second; s++)
                    BackpropTo(inputIndex, fr.WithTimeStep(t).Sequence(s));
            m_beingUnrolled = false;
            return;
        }

        // this potentially computes inner products over time, so we must mask gaps to 0
        if (Input(inputIndex)->ReducesInTimeWrt(shared_from_this()))
            MaskMissingGradientColumnsToZero(fr);
        if (Input(inputIndex)->ReducesInTimeWrt(Input(1 - inputIndex)))
            Input(1 - inputIndex)->MaskMissingValueColumnsToZero(fr);

        bool overwriteInputGradient = (Input(inputIndex)->ParentOverwritesGradient() && !m_beingUnrolled);

        if (inputIndex == 0) // left derivative
        {
            if (InputRef(1).Value().GetMatrixType() == SPARSE &&
                InputRef(0).GetPreferredGradientMatrixType() == UNDETERMINED &&
                Gradient().GetMatrixType() == DENSE)
            {
                // Special case for DENSE * SPARSE -> DENSE, which leads to a SPARSE gradient for input0 (common for embedding)
                // We need a sparse matrix for the gradient. We allocate a new one instead of switching the type in place
                // since switching in place may affect other nodes who share this matrix due to memory sharing
                //
                // Note this normally only happen once (per minibatch) for embedding cases where all calculations are dense * sparse
                // This is needed in addition to AllocateGradientMatricesForInputs as some sparse inputs that are not input_variable
                // cannot be determined as sparse when AllocateGradientMatricesForInputs happens.

                auto& currentInput0GradientMatrixRef = InputRef(0).Gradient();
                auto newInput0SparseGradientMatrix =
                    std::make_shared<Matrix<ElemType>>(
                        currentInput0GradientMatrixRef.GetNumRows(),
                        currentInput0GradientMatrixRef.GetNumCols(),
                        currentInput0GradientMatrixRef.GetPreferredDeviceId(),
                        SPARSE,
                        MatrixFormat::matrixFormatSparseBlockCol);

                // no need not set sparse with value from dense as its just cleared to zero when PreferredGradientMatrixType == UNDETERMINED

                InputRef(0).GradientPtrRef() = newInput0SparseGradientMatrix;
                InputRef(0).SetPreferredGradientMatrixType(SPARSE);
            }
            else if (InputRef(1).Value().GetMatrixType() == DENSE &&
                     InputRef(0).GetPreferredGradientMatrixType() != DENSE)
            {
                // for dense * dense, if we found previously gradient accumulated with sparse for input0, switch to dense
                // this is a rare case and the performance is not optimized
                if (InputRef(0).GetPreferredGradientMatrixType() == SPARSE)
                    InputRef(0).Gradient().SwitchToMatrixType(DENSE, matrixFormatDense, !overwriteInputGradient);

                InputRef(0).SetPreferredGradientMatrixType(DENSE);
            }

            auto input0Gradient = OneSampleTensorFor(0,  /*gradient=*/true,  fr.AllowBroadcast());
            auto input1         = OneSampleTensorFor(1,  /*gradient=*/false, fr.AllowBroadcast());
            auto outputGradient = OneSampleTensorFor(-1, /*gradient=*/true,  fr);
            if (overwriteInputGradient)
                input0Gradient.AssignMatrixProductOf(m_transpose/*transC*/, outputGradient, false/*transA*/, input1, true/*transB*/);
            else
                input0Gradient.AddMatrixProductOf(m_transpose/*transC*/, outputGradient, false/*transA*/, input1, true/*transB*/);
        }
        else if (inputIndex == 1) // right derivative
        {
            auto input0         = OneSampleTensorFor(0,  /*gradient=*/false, fr.AllowBroadcast());
            auto input1Gradient = OneSampleTensorFor(1,  /*gradient=*/true,  fr.AllowBroadcast());
            auto outputGradient = OneSampleTensorFor(-1, /*gradient=*/true, fr);

            if (InputRef(1).Gradient().GetMatrixType() == SPARSE)
            {
                // we only support dense * sparse to have sparse gradient for input0, so if input1 has sparse gradient, switch to dense
                // this is a rare case and the performance is not optimized
                InputRef(1).Gradient().SwitchToMatrixType(DENSE, matrixFormatDense, !overwriteInputGradient);
            }
            InputRef(1).SetPreferredGradientMatrixType(DENSE);

            if (overwriteInputGradient)
                input1Gradient.AssignMatrixProductOf(false/*transC*/, input0, !m_transpose/*transA*/, outputGradient, false/*transB*/);
            else
                input1Gradient.AddMatrixProductOf(false/*transC*/, input0, !m_transpose/*transA*/, outputGradient, false/*transB*/);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    // but both *inputs* are used, so we don't overload the InputUsed-() function which defaults to 'true'

    virtual bool ImplementsGradientOverwriteOptimization() const override { return true; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        if (ReduceSequenceAxis())
        {
            // generate MBLayout without sequence axis
            if (!Input(0)->HasMBLayout() || !Input(1)->HasMBLayout() || *(Input(0)->GetMBLayout()) != *(Input(1)->GetMBLayout()))
                InvalidArgument("%ls %ls operation can perform sequence axis reduction only on matching dynamic axes for both operands (which have the same layouts).", NodeName().c_str(), OperationName().c_str());

            const auto& input0SampleLayout = InputRef(0).GetSampleLayout();
            const auto& input1SampleLayout = InputRef(1).GetSampleLayout();

            if ((input0SampleLayout.GetRank() > 1 && input0SampleLayout[1] != input1SampleLayout[0]) ||
                (input1SampleLayout.GetRank() > 1 && input1SampleLayout[1] != 1) ||
                (input0SampleLayout.GetRank() <= 1 && input1SampleLayout[0] != 1) ||
                m_transpose)
                InvalidArgument("%ls %ls operation can perform sequence axis reduction only in forms of (m x k) * (k x 1), without transpose.", NodeName().c_str(), OperationName().c_str());

            if (m_pMBLayout == nullptr)
            {
                m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
                m_pMBLayout->SetUniqueAxisName(ComputationNodeBase::DefaultNoSequenceAxisName);
            }
        }
        else
        {
            InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);
        }

        bool transpose = m_transpose; // (assigning to a non-const variable avoids a compiler warning C4127: conditional expression is constant)

        // get tensor shapes
        auto dimsA = Input(0)->GetSampleLayout().GetDims();
        auto dimsB = Input(1)->GetSampleLayout().GetDims();
        string dimsAstring = string(Input(0)->GetSampleLayout()); // for error messages
        string dimsBstring = string(Input(1)->GetSampleLayout());

        // validate and infer
        if (isFinalValidationPass || (dimsA.size() > 0 && dimsB.size() > 0)) // only if we got at least some input dimensions to work with or need to wrap up
        {
            // if transposing then only support actual matrices or column vectors
            if (transpose)
            {
                if (dimsA.size() == 1) // column vector transposed becomes a 2D tensor
                    dimsA.push_back(1);
                else if (dimsA.size() != 2)
                    InvalidArgument("%ls %ls operation: Transposition requires a 2D tensor (matrix) or a 1D tensor (column vector), instead of a [%s].", NodeName().c_str(), OperationName().c_str(), dimsAstring.c_str());
                else if (m_outputRank != 1)
                    InvalidArgument("%ls %ls operation: The outputRank (%d) must be 1 when transposing.", NodeName().c_str(), OperationName().c_str(), (int)m_outputRank);
                // swap them temporarily, to get transposition out of the way for validation
                std::swap(dimsA[0], dimsA[1]);
            }

            if (m_outputRank > dimsA.size()) // note: it may be equal in case of dyadic product uv'
                InvalidArgument("%ls %ls operation: outputRank %d exceeds left argument's shape [%s].", NodeName().c_str(), OperationName().c_str(), (int)m_outputRank, dimsAstring.c_str());
            auto numReductionDims = dimsA.size() - m_outputRank;  // we reduce over the remaining dims; this is their number. Can be 0 in case of dyadic product uv'
            if (numReductionDims > dimsB.size())
                InvalidArgument("%ls %ls operation: right argument shape [%s] has too few dimensions for outputRank %d.", NodeName().c_str(), OperationName().c_str(), dimsBstring.c_str(), (int)m_outputRank);

#if 1       // support for legacy models when only the matrix dimensions had to match
            // Note: This is non-ambiguous w.r.t. valid new configurations because this condition would otherwise just be considered an error.
            //       But it will fail to discover trailing reduction dimensions that are 1. We assume that no such legacy models exist.
            // Note: This is very ugly [Wayne Xiong]. I agree [fseide].
            if (dimsA.size() == 2 && !transpose && m_outputRank == 1 && dimsA[1] != dimsB[0] && dimsB[0] != 0)
            {
                // search whether we can interpret dimsA[1] as the flattening of the first dimensions
                size_t dim = 1;
                for (size_t k = 0; k < dimsB.size(); k++)
                {
                    dim *= dimsB[k];
                    if (dim == dimsA[1])
                    {
                        // OK, we have an explanation: Patch dimsA and back-patch the sample layout of Input(0)
                        numReductionDims = k + 1;
                        dimsA.resize(m_outputRank + numReductionDims);
                        for (size_t kk = 0; kk < numReductionDims; kk++)
                            dimsA[m_outputRank + kk] = dimsB[kk];
                        Input(0)->SetDims(TensorShape(dimsA), false);
                        fprintf(stderr, "\n%ls %ls operation: For legacy compatibility, the sample layout of left input (%ls %ls operation) was patched to [%s] (from [%s])\n",
                                NodeName().c_str(), OperationName().c_str(), Input(0)->NodeName().c_str(), Input(0)->OperationName().c_str(), string(Input(0)->GetSampleLayout()).c_str(), dimsAstring.c_str());
                        dimsAstring = string(Input(0)->GetSampleLayout()); // for error messages
                        break; // we will continue with this patched up model from here on
                    }
                }
            }
#endif

            // validate or automatically infer dimension inference for learnable parameters
            for (size_t k = 0; k < m_outputRank; k++) // outputRank dimensions cannot be inferred
                if (dimsA[k] == 0)
                    InvalidArgument("%ls %ls operation: The outputRank (%d) dimensions in left argument's shape [%s] must not be 0.", NodeName().c_str(), OperationName().c_str(), (int)m_outputRank, dimsAstring.c_str());

            // infer rank of dimsA
            // For purpose of dimension inference, Times() accepts an optional parameter inferInputRankToMap (default -1=unspecified).
            // The last 'inferInputRankToMap' axes are considered those that the matrix product should keep (Times()
            // is applied one by one, like a "map" operation) rather than reducing over.
            // Specifically, inferInputRankToMap=0 means to reduce over all input axes, e.g. for an image input that
            // should be flattened.
            // Examples:
            //  [I x Inferred] * [J x K],                    inferInputRankToMap=n/a --> Inferred  := J, result is [I x K]
            //  [I x Inferred] * [W x H x C],                inferInputRankToMap=n/a --> Inferred  := W, result is [I x H x C] (not desired)
            //  [I x Inferred x Inferred] * [W x H x C],     inferInputRankToMap=n/a --> Inf x Inf := [W x H], result is [I x C]
            //  [I x Inferred] * [W x H x C],                inferInputRankToMap=0   --> Inferred  := W x H x C, result is [I] (desired)
            //  [I x Inferred] * [W x H x C x R],            inferInputRankToMap=1   --> Inferred  := W x H x C, result is [I x R] (desired)
            // If W's shape is too short, it will be padded with 0 (i.e. inferred in a subsequent step).
            // (the second check below (dimsA.back() == 0) is required to infer dimensions correctly for fixed input tensors where a new dimension is added,
            // e.g. when adding an ROI dimension to a pretrained weights tensor of a dense layer after ROI pooling)
            if (m_inferInputRankToMap >= 0 && dimsA.back() == 0) // if given, we pad if needed
            {
                if ((size_t)m_inferInputRankToMap >= dimsB.size() && isFinalValidationPass) // at least one axis must be left to reduce over
                    InvalidArgument("%ls %ls operation: 'inferInputRankToMap' argument %d must be less than rank of second operand [%s].", NodeName().c_str(), OperationName().c_str(), m_inferInputRankToMap, dimsBstring.c_str());
                assert(dimsA.size() == m_outputRank + numReductionDims);
                while (numReductionDims + (size_t)m_inferInputRankToMap < dimsB.size())
                {
                    dimsA.push_back(0);
                    numReductionDims++;
                }
            }

            // fill in the missing ones
            // We fill in dimensions given as 0. The tensor rank is not inferred here (that is done above).
            for (size_t k = m_outputRank; k < dimsA.size(); k++)
            {
                auto& dimA = dimsA[k];
                auto& dimB = dimsB[k - m_outputRank];
                if (isFinalValidationPass && dimB == 0)
                    InvalidArgument("%ls %ls operation: Right [%s] operand must not have zero dimensions.", NodeName().c_str(), OperationName().c_str(), dimsBstring.c_str());
                else if (dimA == 0)
                    dimA = dimB; // infer dimension
                else if (isFinalValidationPass && dimA != dimB)
                    InvalidArgument("%ls %ls operation: Left [%s] and right [%s] operands' shapes are not compatible.", NodeName().c_str(), OperationName().c_str(), dimsAstring.c_str(), dimsBstring.c_str());
            }

            // now determine result dimensions
            auto dimsC = dimsA;
            dimsC.resize(m_outputRank);    // output dims
            for (size_t k = numReductionDims; k < dimsB.size(); k++)
                dimsC.push_back(dimsB[k]); // input dims
            SetDims(TensorShape(dimsC), HasMBLayout());

            // update dimensions of A
            if (transpose)
                std::swap(dimsA[0], dimsA[1]);
            // update if LearnableParameter
            Input(0)->ValidateInferInputDimsFrom(TensorShape(dimsA));
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseReduceSequenceAxisMatricesIfNeeded(matrixPool);
    }

    size_t OutputRank() const { return m_outputRank; }
    int InferInputRankToMap() const { return m_inferInputRankToMap; }

protected: 
    shared_ptr<QuantizedMultiplier<ElemType>> m_pQuantizedMultiplier;

private:
    size_t m_outputRank;
    int m_inferInputRankToMap;  // -1 (not specified) or says how to expand shape of W, to keep this many mapping dims
    bool m_beingUnrolled;
    std::once_flag m_unrollWarningOnceFlag;

    bool ReduceSequenceAxis() const { return m_inferInputRankToMap == ReduceSequenceAxisWithoutInferredInputRank; }

    static const int NumInputs = 2;
    shared_ptr<Matrix<ElemType>> m_tempScatterIndices[NumInputs];
    shared_ptr<Matrix<ElemType>> m_tempUnpackedValue[NumInputs];
};

// -----------------------------------------------------------------------
// TimesNode (A, B, outputRank=1) -- matrix product
// Right operand and output can have MB layout, while left operand cannot.
// This is generalized to tensors, in that B's leading dimension(s) must match
// the trailing dimension(s) of A, and those dimensions get flattened into a single
// dimension, after which the regular matrix product is applied.
// Leading dimension(s) of A and trailing ones of B remain unchanged.
// For example:
//  [I x J x K] * [J x K x L x *] = [I x L x *]
// How many dimensions must match is controlled by an optional parameter
// 'outputRank', where all but the first 'outputRank' dimensions of A must match.
// 'outputRank' defaults to 1, which is used in the above example.
// Example for outputRank = 2:
//  [I x J x K] * [K x L x M x *] = [I x J x L x M x *]
// -----------------------------------------------------------------------

template <class ElemType>
class TimesNode : public TimesNodeBase<ElemType, false>
{
    typedef TimesNodeBase<ElemType, false> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"Times"; }

public:
    TimesNode(DEVICEID_TYPE deviceId, const wstring& name, size_t outputRank = 1, int inferInputRankToMap = Base::NoInferredInputRank)
        : Base(deviceId, name, outputRank, inferInputRankToMap)
    {
    }
    TimesNode(const ScriptableObjects::IConfigRecordPtr configp)
        : TimesNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"outputRank"), configp->Get(L"inferInputRankToMap"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }
};

template class TimesNode<float>;
template class TimesNode<double>;

// -----------------------------------------------------------------------
// TransposeTimesNode (A', B)
// Right operand and output can have MB layout, while left operand cannot.
// This differs from TimesNode in that A is transposed, where A must be a
// rank-1 or rank-2 tensor.
// A common use of transposition is trace(X'X) where X is a matrix of samples.
// This can be more efficiently implemented as ReduceSum (ElementTimes (X, X))
// -----------------------------------------------------------------------

template <class ElemType>
class TransposeTimesNode : public TimesNodeBase<ElemType, true>
{
    typedef TimesNodeBase<ElemType, true> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"TransposeTimes"; }

public:
    DeclareConstructorFromConfigWithNumInputs(TransposeTimesNode);
    TransposeTimesNode(DEVICEID_TYPE deviceId, const wstring& name, size_t outputRank = 1)
        : Base(deviceId, name, outputRank, Base::NoInferredInputRank)
    {
        if (outputRank != 1)
            LogicError("TransposeTimes does not yet support outputRank other than 1");
    }
};

template class TransposeTimesNode<float>;
template class TransposeTimesNode<double>;

// Fixed-point matrix product. This scales inputs to 16bit signed integers by Symmetric quantizers, performs
// integer multiplication using SSE/AVX2, and transforms the results back.
// Only dense untransposed matrix multiplication will be quantized. If at least one matrix is sparse then it will fall back to un-quantized default evaluation
// Currently it works for CPU only. On GPU logicError will be thrown.
// One way to include this node to the network is with the Edit command:
// ...
// node => if node.name == 'LSTMoutput1.output' then QuantizedTimes(node.inputs[0], node.inputs[1], bitShiftA=1, bitShiftB=2) else node,
// ...
// bitShift(A|B) - bit shift parameters of quantizers for matrices A and B, see the quantizers for more details. Decreases the maximum range of quantziation by 2^bitShift to prevent integer overflow during BLAS routines.
// bitShift=0 doesn't change the range; higher bitShift will decrease precision of quantization, but will make BLAS routines less prone to overflow.
// Other parameters - refer to the base multiplication class
template <class ElemType>
class QuantizedTimesNode : public TimesNodeBase<ElemType, false>
{
    typedef TimesNodeBase<ElemType, false> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"QuantizedTimes";
    }

private:
    // Quantizer bit shift for matrices A and B
    size_t m_bitShiftA; 
    size_t m_bitShiftB; 

public:
    QuantizedTimesNode(DEVICEID_TYPE deviceId, const wstring& name, size_t bitShiftA = 1, size_t bitShiftB = 1, size_t outputRank = 1, int inferInputRankToMap = Base::NoInferredInputRank)
        : Base(deviceId, name, outputRank, inferInputRankToMap), m_bitShiftA(bitShiftA), m_bitShiftB(bitShiftB)
    {
        // TODO support multiplication on GPUs as well.
        if (deviceId != CPUDEVICE)
            LogicError("Quantized operation is supposed to be used on CPU device only.");

        shared_ptr<SymmetricQuantizer<ElemType, short>> pQA(new SymmetricQuantizer<ElemType, short>(m_bitShiftA));
        shared_ptr<SymmetricQuantizer<ElemType, short>> qQB(new SymmetricQuantizer<ElemType, short>(m_bitShiftB));
        this->m_pQuantizedMultiplier = shared_ptr<QuantizedMultiplier<ElemType>>(new QuantizedMultiplier<ElemType>(pQA, qQB));
    }

    QuantizedTimesNode(const ScriptableObjects::IConfigRecordPtr configp)
        : QuantizedTimesNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"bitShiftA"), configp->Get(L"bitShiftB"), configp->Get(L"outputRank"), configp->Get(L"inferInputRankToMap"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<QuantizedTimesNode<ElemType>>(nodeP);
            node->m_bitShiftA = m_bitShiftA;
            node->m_bitShiftB = m_bitShiftB;
        }
    }

    void Save(File& fstream) const
    {
        Base::Save(fstream);
        fstream << m_bitShiftA;
        fstream << m_bitShiftB;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_bitShiftA;
        fstream >> m_bitShiftB;
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        if (dynamic_pointer_cast<LearnableParameter<ElemType>>(Input(0)))
            this->m_pQuantizedMultiplier->SetIsAConstant(true);
        if (dynamic_pointer_cast<LearnableParameter<ElemType>>(Input(1)))
            this->m_pQuantizedMultiplier->SetIsBConstant(true);

        Base::ForwardProp(fr);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& /*fr*/) override
    {
        // This operation is intended only for inference
        NOT_IMPLEMENTED;
    }
};

template class QuantizedTimesNode<float>;
template class QuantizedTimesNode<double>;

// -----------------------------------------------------------------------
// SumElementsNode (input)
// Sums up all elements in the input across all samples into a single scalar.
// When applied to minibatch data, this will sum across all sequences in the
// minibatch, like a training-criterion node. This is one of the few operations
// that cross the boundary between input sequences.
// Note that SGD itself aggregates over samples in a criterion node.
// So the only proper use of this node is for multi-task learning, where
// different nodes have different numbers of samples (sequence lenth).
// -----------------------------------------------------------------------

template <class ElemType>
class SumElementsNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
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

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        // TODO: change to TensorView and AssignCopyOf() with reduction
        Value().AssignSumOfElements(InputRef(0).MaskedValueFor(fr)); // since we are reducing over frames, we must first mask gaps in input to zero
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        InputRef(0).GradientFor(fr) += Gradient(); // here the assumption is that gradientValues are 1x1 matrix
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryReduce(isFinalValidationPass);
    }
};

template class SumElementsNode<float>;
template class SumElementsNode<double>;

// -----------------------------------------------------------------------
// TransposeDimensions (input, axis1, axis2)
//  - swaps index dimensions axis1 and axis2. The values are 1-based; 1 stands for the leading dimension.
//  - new dimensions can be created; e.g. a column vector can be transposed into a row vector, which is a [1 x N] tensor
//  - transposing into the time dimension is currently not supported
//  - internally implemented with tensor lib by shuffling dimensions with their strides
//  - input may be minibatch data or not
// Transpose (input) = TransposeDimensions (input, 1, 2)
// -----------------------------------------------------------------------

template <class ElemType>
class TransposeDimensionsNode : public ComputationNode /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"TransposeDimensions"; }

public:
    TransposeDimensionsNode(DEVICEID_TYPE deviceId, const wstring& name, int axis1 = 1, int axis2 = 2)
        : Base(deviceId, name), m_axis1(axis1), m_axis2(axis2)
    {
    }
    TransposeDimensionsNode(const ScriptableObjects::IConfigRecordPtr configp)
        : TransposeDimensionsNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"axis1"), configp->Get(L"axis2"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const
    {
        Base::Save(fstream);
        fstream << m_axis1 << m_axis2;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        if (modelVersion >= CNTK_MODEL_VERSION_3)
            fstream >> m_axis1 >> m_axis2;
        else
            m_axis1 = 1, m_axis2 = 2; // default
    }

    int Axis1() const { return m_axis1; }
    int Axis2() const { return m_axis2; }

private:
    // compute the transposed tensor shape (in-place)
    void TransposeShape(TensorShape& shape) const
    {
        assert(m_axis1 > 0 && m_axis2 > 0);
        size_t i = m_axis1 - 1;
        size_t j = m_axis2 - 1;
        shape.SwapDimsInPlace(i, j);
    }

    // get the transposed input shape
    // Using this shape yields the same tensor but index positions are swapped.
    TensorShape GetTransposedTensorSliceFor(size_t rank, const FrameRange& fr)
    {
        auto shape = InputRef(0).GetTensorSliceFor(rank, fr);
        TransposeShape(shape);
        return shape;
    }

public:
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto output =                                  ValueTensorFor(                         rank, fr);
        auto input  = TensorView<ElemType>(InputRef(0).ValuePtr(), GetTransposedTensorSliceFor(rank, fr));
        output.AssignCopyOf(input);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto outputGradient =                                  GradientTensorFor(                         rank, fr);
        auto inputGradient  = TensorView<ElemType>(InputRef(0).GradientPtr(), GetTransposedTensorSliceFor(rank, fr));
        inputGradient.AddCopyOf(outputGradient);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        assert(m_inputs.size() == 1);
        ComputationNodeBase::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        // input shape
        auto shape = Input(0)->GetSampleLayout();
        // validate indices
        if (m_axis1 < 1 || m_axis2 < 1)
            InvalidArgument("%ls %ls operation: Indices to transpose must be >= 1.", NodeName().c_str(), OperationName().c_str());
        size_t i = m_axis1 - 1;
        size_t j = m_axis2 - 1;
        if (i >= shape.GetRank() && j >= shape.GetRank())
            InvalidArgument("%ls %ls operation: At least one index must refer to an existing index.", NodeName().c_str(), OperationName().c_str());
        // pad
        // Permutation is allowed to create new dimensions, specifically to be able to transpose a [N] column vector into a [1 x N] row vector.
        // One can also use SplitDimensions() for this, but this seems a natural thing to do.
        size_t maxij = std::max(i, j);
        if (maxij >= shape.GetRank())
            shape.PadRankInPlace(maxij + 1);
        // apply the permutation
        TransposeShape(shape);
        // drop the strides, since output is dense (swapped strides will be used with the input in ForwardProp())
        SetDims(TensorShape(shape.GetDims()), HasMBLayout());
    }

private:
    int m_axis1, m_axis2; // the two dimensions (axes, 1-based) to swap
};

template class TransposeDimensionsNode<float>;
template class TransposeDimensionsNode<double>;

// -----------------------------------------------------------------------
// CosDistanceNode (left, right)
// column-wise cos distance
// TODO: Would it be useful to allow one of the two to be a single column?
// TODO: Allow to reduce only over a single dimension, or a subset.
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
        int rank = DetermineElementwiseTensorRank();
        auto tempView = TensorView<ElemType>(m_temp, ValueTensorFor(rank, fr).GetShape());
        tempView.AssignElementwiseProductOf(ValueTensorFor(rank, fr), tempView);
        tempView.AssignElementwiseProductOf(GradientTensorFor(rank, fr), tempView);
        auto gradientView = Input(inputIndex)->GradientTensorFor(rank, fr);
        gradientView.AddElementwiseProductOf(tempView, Input(inputIndex)->ValueTensorFor(rank, fr), -1);

        m_temp->AssignElementProductOf(*m_invNorm0, *m_invNorm1);
        tempView.AssignElementwiseProductOf(GradientTensorFor(rank, fr), tempView);
        gradientView.AddElementwiseProductOf(tempView, Input(1 - inputIndex)->ValueTensorFor(rank, fr));
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
        Matrix<ElemType> sliceInput1Value = InputRef(1).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_invNorm0->AssignVectorNorm2Of(sliceInput0Value, true);
        m_invNorm0->AssignElementInverseOf(*m_invNorm0);

        m_invNorm1->AssignVectorNorm2Of(sliceInput1Value, true);
        m_invNorm1->AssignElementInverseOf(*m_invNorm1);

        sliceOutputValue.AssignInnerProductOf(sliceInput0Value, sliceInput1Value, true);
        sliceOutputValue.ElementMultiplyWith(*m_invNorm0);
        sliceOutputValue.ElementMultiplyWith(*m_invNorm1);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

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
            node->m_invNorm0->SetValue(*m_invNorm0);
            node->m_invNorm1->SetValue(*m_invNorm1);
            node->m_temp->SetValue(*m_temp);
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_invNorm0, matrixPool, 1, true, true);
        RequestMatrixFromPool(m_invNorm1, matrixPool, 1, true, true);
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool, 1, true, true);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_invNorm0, matrixPool);
        ReleaseMatrixToPool(m_invNorm1, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
    }

private:
    // invNorm nodes tranfer data between ForwardProp and BackpropTo
    shared_ptr<Matrix<ElemType>> m_invNorm0;
    shared_ptr<Matrix<ElemType>> m_invNorm1;
    // the rest are temporaries, values don't need to be maintained
    shared_ptr<Matrix<ElemType>> m_temp;
};

template class CosDistanceNode<float>;
template class CosDistanceNode<double>;

// -----------------------------------------------------------------------
// KhatriRaoProductNode (left, right)
// Compute an outer product of column vectors (for each sample).
// TODO: This is a special kind of tensor product, and calls for a tensor representation.
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
            Matrix<ElemType> sliceInput0Grad = InputRef(0).GradientFor(fr);
            Matrix<ElemType> sliceInput1Value = InputRef(1).ValueFor(fr);

            sliceInput0Grad.AddColumnReshapeProductOf(sliceOutputGrad, sliceInput1Value, false);
        }
        else // right derivative
        {
            Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
            Matrix<ElemType> sliceInput1Grad = InputRef(1).GradientFor(fr);

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
        ValueFor(fr).AssignKhatriRaoProductOf(InputRef(0).ValueFor(fr), InputRef(1).ValueFor(fr));
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        size_t rows0 = Input(0)->GetSampleMatrixNumRows();
        size_t rows1 = Input(1)->GetSampleMatrixNumRows();

        // after KhatriRaoProduct the structure is lost
        // TODO: ^^ Is that correct? Should we use a tensor here, TensorShape(rows0, rows1)?
        SetDims(TensorShape(rows0 * rows1), HasMBLayout());
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

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
        Matrix<ElemType> sliceInput1Value = InputRef(1).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        Matrix<ElemType> sliceInputGrad = Input(inputIndex)->GradientFor(fr);
        Matrix<ElemType> sliceThisGrad = GradientFor(fr);

        BackpropToS(inputIndex, *m_invNorm0, *m_invNorm1, sliceOutputValue, *m_temp, *m_rightTerm, *m_leftTerm, *m_invNormSquare, sliceInput0Value, sliceInput1Value, InputRef(2).Value(), InputRef(3).Value(), sliceInputGrad, sliceThisGrad);
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

                leftTerm -= rightTerm;

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

                    leftTerm -= rightTerm;

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

                    leftTerm -= rightTerm;

                    temp.GetARowByIndex(thisGradientValues, m);

                    Matrix<ElemType>::ConductRowElementMultiplyWithShift(temp, leftTerm, rightTerm, reverseshift, false);

                    inputGradientValues += rightTerm;
                }
            }
        }
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
        Matrix<ElemType> sliceInput1Value = InputRef(1).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        ForwardPropS(*m_invNorm0, *m_invNorm1, sliceOutputValue, sliceInput0Value, sliceInput1Value, InputRef(2).Value(), InputRef(3).Value(), *m_leftTerm, *m_rightTerm);
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
        // Again, the output is a matrix of (negNumber+1, invNorm0.cols)
        rightTermTemp.AssignInnerProductOfWithShiftNeg(in0, in1, true, shift, negNumber);

        // compute the evaluation result matrix by multiply these two matrices, element by element
        // we get a (negNumber+1, n) matrix
        functionValues.AssignElementProductOf(leftTermTemp, rightTermTemp);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        ValidateInferBinaryInputDims();

        if (isFinalValidationPass &&
            (Input(0)->GetSampleMatrixNumRows() != Input(1)->GetSampleMatrixNumRows() || Input(0)->GetMBLayout() != Input(1)->GetMBLayout()))
        {
            LogicError("The tensor dimension in the %ls %ls operation does not match.", NodeName().c_str(), OperationName().c_str());
        }

        auto input3AsLearnableParameterNode = Input(3)->template As<LearnableParameter<ElemType>>();
        if (isFinalValidationPass && (!input3AsLearnableParameterNode || input3AsLearnableParameterNode->GetLearningRateMultiplier() != 0) || (Input(3)->GetSampleLayout().GetNumElements() != 1))
            LogicError("%ls %ls operation expects a constant scalar for Input(3) which corresponds to number of negative samples.", NodeName().c_str(), OperationName().c_str());

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
            node->m_invNorm0->SetValue(*m_invNorm0);
            node->m_invNorm1->SetValue(*m_invNorm1);
            node->m_invNormSquare->SetValue(*m_invNormSquare);
            node->m_leftTerm->SetValue(*m_leftTerm);
            node->m_rightTerm->SetValue(*m_rightTerm);
            node->m_temp->SetValue(*m_temp);
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

template class CosDistanceWithNegativeSamplesNode<float>;
template class CosDistanceWithNegativeSamplesNode<double>;

template <class ElemType>
void UpdateRunningAverage(ComputationNode<ElemType>& newInput, TensorView<ElemType>& runningAverage,
                          size_t& runningCount);

class MPIWrapper;
struct DistGradHeader;

template <typename ElemType>
void AggregateAccumulatorValuesAndUpdateEvaluation(
    shared_ptr<ComputationNetwork> net,
    set<shared_ptr<ComputationNodeBase>> evalNodesWhichAccumulateResult,
    shared_ptr<DistGradHeader> gradHeader,
    shared_ptr<MPIWrapper> mpi,
    size_t packThresholdSizeInBytes = (size_t)DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES);

// -----------------------------------------------------------------------
// EpochAccumulatorNode calculates mean values of all samples used in forward pass.
// During training, mean sample value is calculated in each epoch. Value of the node will contain mean sample value of
// its input node values since the beginning of epoch.
// This node is useful for creating "per class" metrics like average class recall or mean intersection over union (mean
// IOU) which is standard metric in semantic labeling.
// For mean IOU, we calculate ratio true_positives / (true_positives + false_negatives + false_positives) for all target
// classes and then get mean of those values. true_positives, false_negatives, false_positives should be calculated over
// the whole data set. Here we cannot calculate mean IOU per sample and then average the result. Instead, we use
// EpochAccumulatorNode to store those values over the whole data set.
// -----------------------------------------------------------------------
template <class ElemType>
class EpochAccumulatorNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"EpochAccumulator"; }

public:
    EpochAccumulatorNode(DEVICEID_TYPE deviceId, const wstring& name);

    EpochAccumulatorNode(const ScriptableObjects::IConfigRecordPtr configp);

    virtual void BackpropToNonLooping(size_t inputIndex) override;

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }

    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void OnEpochStart() override;

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override;

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName,
                        const CopyNodeFlags flags) const override;

    virtual void Validate(bool isFinalValidationPass);

    // Returns tensor view for the accumulator matrix. If accumulator matrix memory is not allocated
    // accumulator matrix will be resized (memory will be allocated).
    TensorView<ElemType> EnsureAccumlator();

protected:

    friend void AggregateAccumulatorValuesAndUpdateEvaluation<ElemType>(
        shared_ptr<ComputationNetwork> net,
        set<shared_ptr<ComputationNodeBase>> evalNodesWhichAccumulateResult,
        shared_ptr<DistGradHeader> gradHeader,
        shared_ptr<MPIWrapper> mpi,
        size_t packThresholdSize);

    void Reset();

    size_t GetNumberOfSamples() const { return m_numSamples; }
    void SetNumberOfSamples(size_t samples) { m_numSamples = samples; }
    shared_ptr<Matrix<ElemType>> GetAccumulator() { return m_accumulator; }
    // Copies internal accumulator to the output.
    void CopyAccumulatorToValue();

    shared_ptr<Matrix<ElemType>> m_accumulator;
    size_t m_numSamples;
};

}}}
