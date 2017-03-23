//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReshapingNodes.h -- collection of nodes that reshape or sub-sample matrices leading to layout changes
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "Sequences.h"

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
// Reshape(x, tensorShape, beginAxis=0, endAxis=0) -- reinterpret input samples as having different tensor dimensions
//  - just replaces metadata m_sampleLayout, does not change data values
//  - one dimension may be specified as 0 and will be inferred
//  - optional beginAxis/endAxis denote to only replace a sub-range of dims, for implementing ReshapeDimension() and FlattenRank()
//
// Derived operations:
//
// ReshapeDimension(x, dim, tensorShape) = Reshape(x, tensorShape, beginAxis=dim, endAxis=dim+1)
//  - reinterprets one dimension as multiple, where the number of elements remains the same
//  - one of the new dimensions may be specified as 0 and will be inferred
//
// FlattenDimensions(x, dim, num) = Reshape(x, 0, beginAxis=dim, endAxis=dim+num)
//  - replace two or more consecutive dims by a single dim with the same number of elements
//
// SplitDimension(x, dim, N) = ReshapeDimension(x, dim, 0:N)
//  - splits a dimension into a new tensor dimension, injecting them into a new dimension
//  - note: to split into multiple outputs (like tf.split()), use a BrainScript loop with Slice().
// -----------------------------------------------------------------------

template <class ElemType>
class ReshapeNode : public UnaryElementWiseNode<ElemType>
{
    typedef UnaryElementWiseNode<ElemType> Base; UsingUnaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName() { return L"Reshape"; }

public:
    ReshapeNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& replacementSampleLayout = TensorShape(), int beginAxis = 1, int endAxis = 0)
        : Base(deviceId, name),
          m_replacementSampleLayout(replacementSampleLayout),
          m_beginDimParameter(beginAxis),
          m_endDimParameter(endAxis)
    {
    }
    ReshapeNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ReshapeNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"shape"), configp->Get(L"beginAxis"), configp->Get(L"endAxis"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ReshapeNode<ElemType>>(nodeP);
            node->m_beginDimParameter = m_beginDimParameter;
            node->m_endDimParameter   = m_endDimParameter;
            node->m_replacementSampleLayout = m_replacementSampleLayout;
        }
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_beginDimParameter << m_endDimParameter;
        m_replacementSampleLayout.Save(fstream);
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_beginDimParameter >> m_endDimParameter;
        m_replacementSampleLayout.Load(fstream);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        // BUGBUG: For inputs without MBLayout, the sample layout should include the column dimension, but it does not currently. Needs to be fleshed out.
        const auto& inputSampleLayout = Input(0)->GetSampleLayout();
        const auto& inputDims = inputSampleLayout.GetDims();

        auto replacementDims = m_replacementSampleLayout.GetDims();

        size_t beginAxis = m_beginDimParameter > 0 ? m_beginDimParameter - 1 : 0;
        size_t endAxis = m_endDimParameter > 0 ? m_endDimParameter - 1 : inputDims.size();
        if (!isFinalValidationPass) // non-final: be tolerant, no errors
        {
            if (endAxis > inputDims.size())
                endAxis = inputDims.size();
            if (beginAxis > endAxis)
                beginAxis = endAxis;
        }

        // TODO: We should allow to reduce to a 0-length tensor if the dimension is 0

        // if a dimension is specified as zero then infer it, otherwise verify that total #elements matches
        size_t inputElements = 1; // get #elements in range to be replaced
        for (size_t k = beginAxis; k < endAxis; k++)
            inputElements *= inputDims[k];
        size_t targetElements = 1; // check/infer #elements to replace with
        size_t zeroIndex = SIZE_MAX;
        for (size_t k = 0; k < replacementDims.size(); k++)
        {
            if (replacementDims[k] != 0)
                targetElements *= replacementDims[k];
            else if (zeroIndex == SIZE_MAX)
                zeroIndex = k;
            else
                InvalidArgument("%ls %ls operation: More than one dimension was specified as zero in the replacement (sub-)dimensions [%s]", NodeName().c_str(), OperationName().c_str(), string(m_replacementSampleLayout).c_str());
        }
        if (zeroIndex != SIZE_MAX)
            replacementDims[zeroIndex] = inputElements / targetElements; // infer the number (ignore errors at this point)

        // assemble actual full dimension vector
        SmallVector<size_t> dims;
        dims.append(inputDims.begin(), inputDims.begin() + beginAxis);
        dims.append(replacementDims.begin(), replacementDims.end());
        dims.append(inputDims.begin() + endAxis, inputDims.end());
        auto sampleLayout = TensorShape(dims);

        // validate total dimension
        if (isFinalValidationPass && inputSampleLayout.GetNumElements() != sampleLayout.GetNumElements())
        {
            auto subShape = TensorShape(std::vector<size_t>(inputDims.begin() + beginAxis, inputDims.begin() + endAxis));
            InvalidArgument("%ls %ls operation: Input (sub-)dimensions [%s] incompatible with desired (sub-)dimensions [%s]. Number of elements %s.",
                            NodeName().c_str(), OperationName().c_str(),
                            string(subShape).c_str(), string(m_replacementSampleLayout).c_str(),
                            zeroIndex == SIZE_MAX ? "must be the same" : "is not an integer multiple of the non-0 dimensions");
        }

        // that's it
        SetDims(sampleLayout, HasMBLayout());
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        auto result     =             ValueFor(fr);
        auto inputValue = InputRef(0).ValueFor(fr);
        result.AssignValuesOf(inputValue.Reshaped(result.GetNumRows(), result.GetNumCols()));
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto gradient      =                      GradientFor(fr);
        auto inputGradient = InputRef(inputIndex).GradientFor(fr);
        inputGradient += gradient.Reshaped(inputGradient.GetNumRows(), inputGradient.GetNumCols());
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

private:
    TensorShape m_replacementSampleLayout; // user-specified dimensions to replace dimensions [beginAxis, endAxis]
    int m_beginDimParameter;               // 1-based index range as specified
    int m_endDimParameter;
};

template class ReshapeNode<float>;
template class ReshapeNode<double>;

// -----------------------------------------------------------------------
// ReduceElements (op, axis=, input)
// Reduces (e.g. sums up) all elements in each sample (column) of the input.
// The optional axis can be 0 (meaning all elements) or a specific axis.
// Allowed operations:
//  - "Sum"
//  - "LogSum"
//  - "Mean"
//  - "Max"
//  - "Min"
//  - "All"       --not implemented yet
//  - "Any"       --not implemented yet
// TODO:
//  - move to a different header, since it's not really Reshaping
//  - consider to change to pass in a set of axes instead of only one
// -----------------------------------------------------------------------

template <class ElemType>
class ReduceElementsNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ReduceElements"; }

    void ValidateOp();
public:
    ReduceElementsNode(DEVICEID_TYPE deviceId, const wstring& name, const std::wstring& operation = std::wstring(), int axis = CNTKInternalIdxValueForAllStaticAxes) :
        Base(deviceId, name), m_operation(operation), m_axis(axis), m_reductionOp((ElementWiseOperator)-1/*invalid*/), m_scale(0/*invalid*/)
    {
        if (!m_operation.empty()) // verify validity already here out of courtesy (would otherwise be caught in Validate())
            ValidateOp();
    }

    ReduceElementsNode(const ScriptableObjects::IConfigRecordPtr configp) :
        ReduceElementsNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"reductionOp"), configp->Get(L"axis"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void /*ComputationNodeBase::*/ CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override;
    virtual void /*ComputationNodeBase::*/ Load(File& fstream, size_t modelVersion) override;
    virtual void /*ComputationNodeBase::*/ Save(File& fstream) const override;
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override;
    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override;
    virtual bool /*ComputationNodeBase::*/ OutputUsedInComputingInputNodesGradients() const override;
    virtual bool /*ComputationNodeBase::*/ InputUsedInComputingInputNodesGradients(size_t childIndex) const override;
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempScatterIndices, matrixPool);
        RequestMatrixFromPool(m_tempUnpackedData, matrixPool);
    }

    void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_tempScatterIndices, matrixPool);
        ReleaseMatrixToPool(m_tempUnpackedData, matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_tempGatherIndices, matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempGatherIndices, matrixPool);
    }

    std::wstring ReductionOpName() const { return m_operation; }
    int ReductionAxis() const { return m_axis; }

    static const int  CNTKInternalIdxValueForAllStaticAxes = 0;
    static const int  CNTKInternalIdxValueForAllAxes = -1;
    static const int  CNTKInternalIdxValueForSequenceAxis = -2;

private:
    bool IsMean() const { return (m_operation == L"Mean"); }
    bool ReduceAllStaticAxes() const { return m_axis == CNTKInternalIdxValueForAllStaticAxes; }
    bool ReduceAllAxes() const { return m_axis == CNTKInternalIdxValueForAllAxes; }
    bool ReduceSequenceAxis() const { return m_axis == CNTKInternalIdxValueForSequenceAxis; }

private:
    // operation attributes
    int m_axis;
    std::wstring m_operation;          // the operation as a string, e.g. "Sum", see ValidateOp()

    // things cached during validation
    ElementWiseOperator m_reductionOp; // the reduction operation mapped to our internal opCode
    ElemType m_scale;                  // 1 or, for Mean, 1/number of elements we are reducing over

    shared_ptr<Matrix<ElemType>> m_tempGatherIndices;
    shared_ptr<Matrix<ElemType>> m_tempScatterIndices;
    shared_ptr<Matrix<ElemType>> m_tempUnpackedData;
};

// -----------------------------------------------------------------------
// ReconcileDynamicAxis (dataInput, layoutInput)
// This node copies data from 'dataInput' while it propagates the minibatch-layout information from 'layoutInput'.
// It does perform a runtime check to enforce that the layout of 'dataInput' is compatible to that of 'layoutInput'.
// 'dataInput' can also be without MBLayout, so that we can implement OnesLike().
// If 'dataInput' does not have a MBLayout, it broadcasts the dataInput along the MBLayout of the 'layoutInput'.
// This node is meant to be used from BrainScript macros that bracket expand/reduce pairs of nodes. It is not meant to really be used directly.
// -----------------------------------------------------------------------

template <class ElemType>
class ReconcileDynamicAxisNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ReconcileDynamicAxis"; }

public:
    DeclareConstructorFromConfigWithNumInputs(ReconcileDynamicAxisNode);
    ReconcileDynamicAxisNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        // enforce compatibility of 'dataInput' with 'layoutInput'
        // TODO: how to deal with boundary flags?

        m_layoutsMatch = InputRef(0).HasMBLayout() && *m_pMBLayout == *InputRef(0).GetMBLayout(); // this does a deep value-level comparison

        if (InputRef(0).HasMBLayout() && !m_layoutsMatch &&                                       // input is a mismatching data input --only allowed case is broadcast_as()
            ((InputRef(0).GetMBLayout()->GetNumTimeSteps() != 1) ||                               // not broadcast_as()
             (InputRef(0).GetMBLayout()->GetNumSequences() != m_pMBLayout->GetNumSequences())))   // different batch??
        {
            InvalidArgument("%ls %ls operation discovered that %ls %ls operation produced an MB layout that is incompatible with that of %ls %ls.",
                            NodeName().c_str(), OperationName().c_str(),
                            InputRef(0).NodeName().c_str(), InputRef(0).OperationName().c_str(),
                            InputRef(1).NodeName().c_str(), InputRef(1).OperationName().c_str());
        }

        if (!InputRef(0).HasMBLayout() || m_layoutsMatch)   // no shuffle-case: everything matches or non-data that can use tensor broadcast
        {
            // copy the data from 'dataInput'
            size_t rank = GetSampleLayout().GetRank();
            auto result =             ValueTensorFor(rank, fr);
            auto input0 = InputRef(0).ValueTensorFor(rank, InputRef(0).HasMBLayout() ? fr.WithLayout(InputRef(0).GetMBLayout()) : fr.AllowBroadcast());
            // If data input has a layout (which is known to match), then replace the pointer here ^^ to avoid another runtime check.
            // If it has no layout, then set the broadcast-allowed flag, which will accept any layout to be passed in.
            result.AssignCopyOf(input0);
            // TODO: Once we do in-place, the above must include a copy-to-self check (either here or inside the tensor lib).
        }
        else // Broadcasting along the sequence case: must reshuffle
        {
            auto result = ValueFor(fr);
            ComputationNode<ElemType>::BroadcastToPacked(InputRef(0).Value(), InputRef(0).GetMBLayout(), /*beta =*/ 0, result, fr, m_tempGatherIndices);
        }
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex == 0)
        {
            size_t rank = GetSampleLayout().GetRank();

            // if reduction then mask the respective input(s) (zero out the gaps)
            if (InputRef(inputIndex).ReducesInTimeWrt(shared_from_this()))
                MaskMissingGradientColumnsToZero(fr);

            TensorView<ElemType> gradient;
            TensorView<ElemType> inputGradient;
            if (!InputRef(0).GetMBLayout() || m_layoutsMatch)
            {
                gradient      =                      GradientTensorFor(rank, fr);
                inputGradient = InputRef(inputIndex).GradientTensorFor(rank, InputRef(inputIndex).HasMBLayout() ? fr.WithLayout(InputRef(inputIndex).GetMBLayout()) : fr.AllowBroadcast());
            }
            else
            {
                // Broadcasting along the sequence
                if (!fr.IsAllFrames())
                    InvalidArgument("%ls %ls operation does not support broadcasting the left operand to the right operand's dynamic axis, inside a recurrent loop.", NodeName().c_str(), OperationName().c_str());

                gradient = ComputationNode<ElemType>::Unpack(GetSampleLayout(), GradientFor(fr), m_pMBLayout, m_tempUnpackedData, m_tempScatterIndices, /*batchMajor=*/ true, /*maskGaps=*/ true);
                inputGradient = Input(inputIndex)->GradientTensorFor(rank, FrameRange(InputRef(inputIndex).GetMBLayout(), 0));
            }

            if (InputRef(inputIndex).ParentOverwritesGradient())
                inputGradient.AssignCopyOf(gradient);
            else
                inputGradient.AddCopyOf(gradient);

            // TODO: Once we do in-place, the above must include a copy-to-self check (pay special attention to adding vs. copying).
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && !InputRef(1).HasMBLayout())
            RuntimeError("%ls %ls operation requires that the second argument has a dynamic axis.", NodeName().c_str(), OperationName().c_str());
        m_pMBLayout = InputRef(1).GetMBLayout(); // output layout is that of 'layoutInput'

        SetDims(Input(0)->GetSampleLayout(), HasMBLayout());
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempGatherIndices, matrixPool);
    }

    void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_tempGatherIndices, matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_tempScatterIndices, matrixPool);
        RequestMatrixFromPool(m_tempUnpackedData, matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempScatterIndices, matrixPool);
        ReleaseMatrixToPool(m_tempUnpackedData, matrixPool);
    }

private:
    bool m_layoutsMatch;
    shared_ptr<Matrix<ElemType>> m_tempGatherIndices;
    shared_ptr<Matrix<ElemType>> m_tempScatterIndices;
    shared_ptr<Matrix<ElemType>> m_tempUnpackedData;
};

template class ReconcileDynamicAxisNode<float>;
template class ReconcileDynamicAxisNode<double>;

// -----------------------------------------------------------------------
// SliceNode (input)
// This node extracts a slice of the first tensor dimension (row).
// This does not support slicing the time axis. That has to be done in BrainScript using Gather.
// -----------------------------------------------------------------------

template <class ElemType>
class SliceNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"Slice"; }

public:
    SliceNode(DEVICEID_TYPE deviceId, const wstring& name, std::vector<int> beginIndex = { 0 }, std::vector<int> endIndex = { 0 }, std::vector<int> axis = { 1 })
        : Base(deviceId, name), m_beginIndex(beginIndex), m_endIndex(endIndex), m_axis(axis)
    {
        if (m_beginIndex.size() != m_endIndex.size() || m_beginIndex.size() != m_axis.size())
            InvalidArgument("%ls %ls operation: invalid size of beginIndex (%d), endIndx (%d) and axis (%d). They must agree.", NodeName().c_str(), OperationName().c_str(), (int)m_beginIndex.size(), (int)m_endIndex.size(), (int)m_axis.size());
    }

    SliceNode(const ScriptableObjects::IConfigRecordPtr configp)
        : SliceNode(configp->Get(L"deviceId"), L"<placeholder>", { configp->Get(L"beginIndex") }, { configp->Get(L"endIndex") }, { configp->Get(L"axis") })
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }
    //SliceNode(const ScriptableObjects::IConfigRecordPtr configp)
    //    : Base(configp->Get(L"deviceId"), L"<placeholder>")
    //{
    //    m_beginIndex.clear(); m_beginIndex.push_back((int)configp->Get(L"beginIndex"));
    //    m_endIndex.clear(); m_endIndex.push_back((int)configp->Get(L"endIndex"));
    //    m_axis.clear(); m_axis.push_back((int)configp->Get(L"axis"));
    //    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    //}

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        auto node = dynamic_pointer_cast<SliceNode<ElemType>>(nodeP);
        node->m_beginIndex = m_beginIndex;
        node->m_endIndex   = m_endIndex;
        node->m_axis       = m_axis;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        int num = 1, axis = 1;  // axis = 1 to emulate old RowSliceNode 
        ptrdiff_t beginIndex, height;
        if (modelVersion >= CNTK_MODEL_VERSION_22)
            fstream >> num; 
        if (num < 1)
            InvalidArgument("Slice node number of axes (%d) invalid, must be >=1", num); 

        m_beginIndex.clear(); 
        m_endIndex.clear();
        m_axis.clear(); 
        for (int i = 0; i < num; i++)
        {
            fstream >> beginIndex >> height; // legacy format stored (end-begin)
            m_beginIndex.push_back((int)beginIndex);
            m_endIndex.push_back((int)(beginIndex + height));
            if (modelVersion >= CNTK_MODEL_VERSION_3)
                fstream >> axis;
            m_axis.push_back(axis); 
        }
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        int num = (int)m_axis.size(); 
        fstream << num; 
        for (auto i = 0; i < num; i++)
        {
            fstream << (ptrdiff_t)m_beginIndex[i] << (ptrdiff_t)(m_endIndex[i] - m_beginIndex[i]); // legacy file format stores (end-begin), we keep it that way
            fstream << m_axis[i];
        }
    }

    // these implement numpy-style negative bound values to index from the end
    std::vector<int> BeginIndex() const { return m_beginIndex; }
    size_t BeginIndex(int idx) const 
    {
        if (idx >= (int)m_axis.size())
            InvalidArgument("Slice BeginIndex call with invalid index (%d) >= axis size (%d)", idx, (int)m_axis.size()); 
        return m_beginIndex[idx] >= 0 ? (size_t)m_beginIndex[idx] : (size_t)(m_beginIndex[idx] + InputRef(0).GetSampleLayout()[m_axis[idx] - 1]); 
    }
    std::vector<int> EndIndex() const { return m_endIndex; }
    size_t EndIndex(int idx)   const 
    {
        if (idx >= (int)m_axis.size())
            InvalidArgument("Slice EndIndex call with invalid index (%d) >= axis size (%d)", idx, (int)m_axis.size());
        return m_endIndex[idx]   >  0 ? (size_t)m_endIndex[idx] : (size_t)(m_endIndex[idx] + InputRef(0).GetSampleLayout()[m_axis[idx] - 1]); 
    }
    std::vector<int> Axis() const { return m_axis; }
    int Axis(int idx) const 
    { 
        if (idx >= (int)m_axis.size())
            InvalidArgument("Slice Axis call with invalid index (%d) >= axis size (%d)", idx, (int)m_axis.size());
        return m_axis[idx]; 
    }

private:

    // determine the tensor shape that represents slice of the input that we are taking
    TensorShape GetInputSlice(size_t rank, const FrameRange & fr) const
    {
        auto inputSlice = InputRef(0).GetTensorSliceFor(rank, fr);    // input must be narrowed down
        for (int i = 0; i < (int)m_axis.size(); i++)  
            inputSlice.NarrowTo(Axis(i)-1, BeginIndex(i), EndIndex(i));
        return inputSlice;
    }

public:

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        auto output = ValueTensorFor(rank, fr);
        let   input = TensorView<ElemType>(InputRef(0).ValuePtr(), GetInputSlice(rank, fr.AllowBroadcast()));
        output.AssignCopyOf(input);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        let outputGrad = GradientTensorFor(rank, fr);
        auto inputGrad = TensorView<ElemType>(InputRef(0).GradientPtr(), GetInputSlice(rank, fr.AllowBroadcast()));
        inputGrad.AddCopyOf(outputGrad);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        auto sampleLayout = Input(0)->GetSampleLayout();
        for (int i = 0; i < (int)m_axis.size(); i++)
        {
            if (m_axis[i] < 1 || (isFinalValidationPass && m_axis[i] > sampleLayout.GetRank()))
                RuntimeError("%ls %ls operation: axis parameter %d (%d) must be in range 1..rank of input ([%s]).", NodeName().c_str(), OperationName().c_str(), i, m_axis[i], string(sampleLayout).c_str());

            if (isFinalValidationPass && (sampleLayout[m_axis[i] - 1] < EndIndex(i) || EndIndex(i) < BeginIndex(i) || BeginIndex(i) < 0))
                RuntimeError("%ls %ls operation: Index range [%d,%d), interpreted as [%d,%d), is invalid for input ([%s]).", NodeName().c_str(), OperationName().c_str(), m_beginIndex[i], m_endIndex[i], (int)BeginIndex(i), (int)EndIndex(i), string(sampleLayout).c_str());

            // propagate as much as we can
            if (isFinalValidationPass || (m_axis[i] - 1 < sampleLayout.GetRank() && 0 <= BeginIndex(i) && BeginIndex(i) <= EndIndex(i) && EndIndex(i) <= sampleLayout[m_axis[i] - 1])) // (the second condition guards against failing an out-of-bounds error if not isFinalValidationPass)
                sampleLayout.NarrowTo(m_axis[i] - 1, BeginIndex(i), EndIndex(i));
        }
        SetDims(TensorShape(sampleLayout.GetDims()), HasMBLayout());
    }

private:
    std::vector<int> m_beginIndex, m_endIndex; // 'int' because negative indices are allowed, to index from end Python-style
    std::vector<int> m_axis;                   // note: axes are 1-based
};

template class SliceNode<float>;
template class SliceNode<double>;

// -----------------------------------------------------------------------
// CropNode
//
// Extracts portion of inputNode1 (input to be cropped) that corresponds to
// inputNode2 (input that defines crop dimensions).

// Cropping offsets can be given directly (offsetX, offsetY parameters in BS/NDL).
// These offsets must be given in absolute values (pixels).
//
// Alternatively, offsets can be calculated automatically using network graph
// and node transforms. The offsets are computed by traversing the network graph
// and finding common ancestor of crop node inputs. Once ancestor is found affine
// transform is computed along the paths from first and second input to common
// ancestor. Complete transform from one input to other it finally calculated
// composing these two transforms. Translate components of final transform define
// crop offsets.
// Automatic crop calculation uses concept of equivalence nodes. Equivalence nodes
// are sort of virtual common ancestors. For example two inputs to network may be
// equivalent in spatial sense (for example input and target in case of pixelwise
// semantic labeling) but they are separate leaf nodes which cannot be common
// ancestors for inputs to crop node. However, they can be declared as equivalent
// using equivalence nodes option (when traversing from one crop input and other
// once we reach two equivalence nodes we will consider that path between two
// crop inputs is closed over them).
//
// Usage (Both NDL and BS):
//  CropNode(input1, input2, offsetX, offsetY) or
//  CropNode(input1, input2) or
//  CropNode(input1, input2, eqNode1, eqNode2) or
// where:
//  input1 - computation node to be cropped at given/calculated offsets with width and height taken from input2
//  input2 - computation node that defines cropping shape (width and height to be used when cropping input1)
//  offsetX - manually given absolute offset in pixels along x axis (must be used with type="manual")
//  offsetY - manually given absolute offset in pixels along y axis (must be used with type="manual")
//  eqNode1 - first equivalence node
//  eqNode2 - second equivalence node
// -----------------------------------------------------------------------

template <class ElemType>
class CropNode : public ComputationNode<ElemType>, public TransformerNode
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;

    static const std::wstring TypeName() { return L"Crop"; }

public:
    CropNode(DEVICEID_TYPE deviceId, const std::wstring& name);

    CropNode(size_t offsetX, size_t offsetY, DEVICEID_TYPE deviceId, const std::wstring& name);

    CropNode(const ScriptableObjects::IConfigRecordPtr configp);

    void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& /*fr*/) override;

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& /*fr*/) override;

    void Save(File& fstream) const override;

    void Load(File& fstream, size_t modelVersion) override;

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override;

private:
    using ComputationNodeBase::GetInputs;
    using TransformerNode::m_transforms;

    // Declaration of matrix getting method to unify accessing values and gradients.
    typedef MatrixBasePtr(ComputationNode<ElemType>::*MatrixGetter)() const;

    // Helper structure to store input/output views which define parts of input and output we work with.
    struct CroppedIOViews
    {
        CroppedIOViews(CropNode* cropNode, MatrixGetter matrixGetter, TensorShape inputShapeCropped, TensorShape ouputShape) :
            // Input view is derived from first input.
            inputViewCropped((cropNode->Input(0).get()->*matrixGetter)(), inputShapeCropped),
            // Output view corresponds to single output.
            outputView((cropNode->*matrixGetter)(), ouputShape)
        {}

        TensorView<ElemType> inputViewCropped;
        TensorView<ElemType> outputView;
    };

    // Creates input and output views (TensorViews that define parts of input and output we work with). MatrixGetter is
    // the pointer to method that returns appropriate matrix (values in forward or gradients in backward). Using
    // MatrixGetter we can reuse code without copy-pasting.
    CroppedIOViews CreateIOViews(MatrixGetter matrixGetter);

    // Performs offsets computation if necessary.
    void ComputeCropOffsets();

    virtual void /*TransformerNode::*/ComputeTransforms() override;

    virtual bool /*TransformerNode::*/SupportsTransformOnInput(size_t inputIndex) override;

protected:
    // Offset along x axis. We need to store offsets as floats for precision if one crop node affects computation of other.
    double m_xOffset;
    // Offset along y axis.
    double m_yOffset;
};

// -----------------------------------------------------------------------
// RowStack (input0, input1, ...)
// stacks multiple inputs on top of each other
// The inputs will be spliced w.r.t. their first tensor dimension (the "row" dimension).
// TODO: This is very close to the planned SpliceNode (just make m_spliceDim actually configurable) except for splicing along time.
// -----------------------------------------------------------------------

template <class ElemType>
class RowStackNode : public ComputationNode<ElemType> // note: not deriving from NumInputs<> like most other nodes, because this one takes a variable number of inputs
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"RowStack"; }

public:
    RowStackNode(DEVICEID_TYPE deviceId, const wstring& name, int spliceDim = 1/*TODO: complete this*/)
        : Base(deviceId, name), m_spliceDim(spliceDim)
    {
    }

    RowStackNode(const ScriptableObjects::IConfigRecordPtr configp)
        : RowStackNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"axis"))
    {
        AttachInputsFromConfig(configp);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeInputLinks)
        {
            auto node = dynamic_pointer_cast<RowStackNode<ElemType>>(nodeP);
            node->m_firstIndices = m_firstIndices;
            node->m_spliceDim = m_spliceDim; 
        }
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        if (modelVersion >= CNTK_MODEL_VERSION_3)
            fstream >> m_spliceDim;
        else
            m_spliceDim = 1;
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_spliceDim;
    }

private:
    // changes the result slice (which includes all stacked inputs) to the stripe that matches where one of the inputs goes
    TensorShape NarrowToStripe(const TensorShape & resultSlice, size_t inputIndex)
    {
        auto resultSubSlice = resultSlice;
        assert(m_spliceDim > 0);
        size_t index = (size_t)m_spliceDim - 1;
        resultSubSlice.NarrowTo(index, m_firstIndices[inputIndex], m_firstIndices[inputIndex + 1]);
        return resultSubSlice;
    }

public:
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        let outputSlice = GetTensorSliceFor(rank, fr); // tensor slice that represents the entire output for FrameRange

        for (size_t inputIndex = 0; inputIndex < GetNumInputs(); inputIndex++)
        {
            let input = InputRef(inputIndex).ValueTensorFor(rank, fr.AllowBroadcast());
            let outputSubSlice = NarrowToStripe(outputSlice, inputIndex);
            auto output = TensorView<ElemType>(ValuePtr(), outputSubSlice);
            output.AssignCopyOf(input);
        }
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        size_t rank = DetermineElementwiseTensorRank();
        let outputSlice = GetTensorSliceFor(rank, fr); // tensor slice that represents the entire output for FrameRange

        auto inputGrad = InputRef(inputIndex).GradientTensorFor(rank, fr.AllowBroadcast());
        let outputSubSlice = NarrowToStripe(outputSlice, inputIndex);
        let outputGrad = TensorView<ElemType>(GradientPtr(), outputSubSlice);
        inputGrad.AddCopyOf(outputGrad);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        // we must fuse all tensor shapes
        // All dimensions but the last must be the same. (In a future version, we should be able to stack along any given dimension.)

        // determine maximum rank (we can stack tensors with lower rank, which will have their dimensions paded to max automatically)
        assert(m_spliceDim > 0);
        size_t index = (size_t)m_spliceDim - 1;
        size_t maxRank = index + 1; // spliceDim may exceed all of them, which will create a new dimension, e.g. stacking column vectors into a matrix
        for (int i = 0; i < GetNumInputs(); i++)
            if (maxRank < Input(i)->GetSampleLayout().GetRank())
                maxRank = Input(i)->GetSampleLayout().GetRank();

        // the following loop does multiple things:
        //  - count total dimension along index, and form associated m_firstIndices[] array
        //  - verify all other dimension's compatibility (we allow broadcasting)
        auto dims = Input(0)->GetSampleLayout().PadRank(maxRank).GetDims(); // dimensions padded to max rank; start with dims of first input
        dims[index] = 0;                                                    // this dimension is created, while all others are verified for consistency
        m_firstIndices.assign(1, 0);                                        // accumulative splice dimension; start with 0
        for (int i = 0; i < GetNumInputs(); i++)
        {
            // check/fuse dims and accumulate the spliced dimension
            let & shape = Input(i)->GetSampleLayout();
            for (size_t k = 0; k < maxRank; k++)
            {
                size_t dim = shape.GetDimPadded(k);
                if (k == index)
                {
                    // accumulate the spliced dimension
                    dims[index] += dim;
                    m_firstIndices.push_back(dims[index]);    // and remember it
                }
                else
                {
                    // check/fuse dimensions
                    if (isFinalValidationPass && dim != dims[k] && dim != 1 && dims[k] != 1)
                        InvalidArgument("%ls %ls operation: Conflicting dimension %d between %ls %ls operation (%d) and other(s) (%d)",
                                        NodeName().c_str(), OperationName().c_str(), (int)k, Input(i)->NodeName().c_str(), Input(i)->OperationName().c_str(), (int)dim, (int)dims[k]);
                    if (dims[k] == 1)   // broadcast
                        dims[k] = dim;
                }
            }
        }

        SetDims(TensorShape(dims), HasMBLayout());
    }

    int GetSpliceDim() const { return m_spliceDim; }

private:
    std::vector<size_t> m_firstIndices; // start row number in the stacked matrix of each input (child) (cumsum of matrix heights); plus one final entry that equals the total dimension
    int m_spliceDim;                    // tensor dimension according to which to stack (1-based)
};

template class RowStackNode<float>;
template class RowStackNode<double>;

// -----------------------------------------------------------------------
// RowRepeatNode (input) -- duplicate row(s) of a matrix multiple times
// -----------------------------------------------------------------------

template <class ElemType>
class RowRepeatNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"RowRepeat"; }

public:
    RowRepeatNode(DEVICEID_TYPE deviceId, const wstring& name, size_t numRepeats = 1)
        : Base(deviceId, name),
          m_numRepeat(numRepeats)
    {
    }
    RowRepeatNode(const ScriptableObjects::IConfigRecordPtr configp)
        : RowRepeatNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"numRepeats"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<RowRepeatNode<ElemType>>(nodeP);
            node->m_numRepeat = m_numRepeat;
        }
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_numRepeat;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_numRepeat;
    }

    virtual std::string FormatOperationPrototype(const std::string& extraArgs) const override
    {
        return Base::FormatOperationPrototype(extraArgs + msra::strfun::strprintf(", numRepeats=%lu", m_numRepeat));
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        // the trailing dimension gets multiplied
        // TODO: Or should we add an additional dimension?
        SmallVector<size_t> dims = GetInputSampleLayout(0).GetDims();
        dims.back() *= m_numRepeat;

        SetDims(TensorShape(dims), HasMBLayout());
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        ValueFor(fr).AssignRepeatOf(InputRef(0).ValueFor(fr), m_numRepeat, 1);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        InputRef(0).GradientFor(fr).AddToRowRepeatValuesOf(GradientFor(fr), m_numRepeat);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

private:
    size_t m_numRepeat;
};

template class RowRepeatNode<float>;
template class RowRepeatNode<double>;

// -----------------------------------------------------------------------
// WhereNode(cond) -- extract indices of a sequence repeated according to
// an indicator vector. Kinds of indicator values:
//  - 0 and 1: frames with 0 are deleted
//  - int > 0: each frame be repeated this many times
//  - float > 0: each frame will on average be repeated this many times
//               by rounding and carrying over the error
// Example for float:
//  - weights all 1.5
//  - sequence A B C D E F G H I J
//  - ->       A A B C C D E E F G G H I I J
//  - algorithm:
//     - keep an accumulated count of actual and desired #frames at any point
//     - append as many frames as missing (rounded up)
// As this implies a runtime-value dependent reduction in dimension, it can
// only be applied to time sequences, and not other tensor dimensions.
// The result will have a different MBLayout reflecting the shortened result sequences.
// -----------------------------------------------------------------------

/* Notes on Where(), PackedIndex(), and Gather-/ScatterPacked():
This is one of the few nodes that creates new MBLayouts inside this system.
This node is meant to operate jointly with PackedIndexNode.
The difference between Index and PackedIndex is that Index is in human-readable
form referring to indices WITHIN a sequence (since NDL and BS only talk about individual
sequences and never expose anything cross-sequence, except for aggregates like CE or BN.
PackedIndex maps that to the internal lookup table that has strides resolved etc.
The reason that PackedIndex is separate from Gather/ScatterPacked is that the GPU has no
access to the STL-heavy MBLayout. So PackedIndex applies the relevant information from
the MBLayout into a GPU object that then drives the memory-copy operations in Gather()
and Scatter().
*/

template <class ElemType>
class WhereNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"Where"; }

    static const std::wstring DefaultWhereNodeDynamicAxisName() { return L"WhereNodeAxis"; }
public:
    DeclareConstructorFromConfigWithNumInputs(WhereNode);
    WhereNode(DEVICEID_TYPE deviceId, const wstring& name, const wstring& dynamicAxisName = DefaultWhereNodeDynamicAxisName()) :
        Base(deviceId, name), m_dynamicAxisName(dynamicAxisName)
    {
        MarkValueNonSharable();
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override;
    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t /*inputIndex*/) override;
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }
    virtual void Validate(bool isFinalValidationPass) override;

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        if (modelVersion >= CNTK_MODEL_VERSION_11)
            fstream >> m_dynamicAxisName;
        else
            m_dynamicAxisName = DefaultWhereNodeDynamicAxisName();
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_dynamicAxisName;
    }

    std::wstring DynamicAxisName() const { return m_dynamicAxisName; }

private:
    // buffers for creating the result sequences (kept as object state to avoid memory allocations)
    std::vector<std::vector<size_t>>   m_indexSequenceBuffer; // [sequenceIndex][t] for creating the result sequences
    std::vector<size_t>               m_rowAllocationsBuffer; // [row] for determining new MBLayout packing
    std::vector<std::pair<size_t, size_t>> m_placementBuffer; // [sequenceIndex] assigned location for a sequence
    std::wstring m_dynamicAxisName;
};

// -----------------------------------------------------------------------
// PackedIndexNode(targetObject, indexSequence) -- convert sequence indices
// to internal packed column indices w.r.t. targetObject.
// Intended use is
//  - Gather  (cond, x) = GatherPacked  (PackedIndex (x, Where (xCond)), x)
//  - Scatter (cond, y) = ScatterPacked (yCond, PackedIndex (y, Where (yCond)), y)
// This maps sequence-specific time indices t to GetColumnIndex(seq,t),
// as input for subsequent GatherPacked() or ScatterPacked() operations.
// -----------------------------------------------------------------------

template <class ElemType>
class PackedIndexNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"PackedIndex"; }

    // our inputs
    static const size_t SOURCEDATA = 0;
    static const size_t INDEXDATA  = 1;

public:
    DeclareConstructorFromConfigWithNumInputs(PackedIndexNode);
    PackedIndexNode(DEVICEID_TYPE deviceId, const wstring& name) :
        Base(deviceId, name)
    {
        MarkValueNonSharable();
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override;
    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t /*inputIndex*/) override;
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }
    virtual void Validate(bool isFinalValidationPass) override;
};

// -----------------------------------------------------------------------
// GatherPackedNode(packedIndex, sourceData) -- gather operation
// Copies subset of samples pointed to by packedIndex from sourceData.
// Sequence lengths are equal to those from packedIndex.
// PackedIndex must have been created with PackedIndex() node, and is
// otherwise opaque to users.
// -----------------------------------------------------------------------

template <class ElemType>
class GatherPackedNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"GatherPacked"; }

    // our inputs
    static const size_t INDEXDATA = 0;
    static const size_t SOURCEDATA = 1;

public:
    DeclareConstructorFromConfigWithNumInputs(GatherPackedNode);
    GatherPackedNode(DEVICEID_TYPE deviceId, const wstring& name) :
        Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override;
    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override;
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override { return childIndex == INDEXDATA; }
    virtual void Validate(bool isFinalValidationPass) override;
};

// -----------------------------------------------------------------------
// ScatterPackedNode(layoutData, packedIndex, sourceData) -- scatter operation
// Copies sourceData to sample positions pointed to by packedIndex.
// The first arg, 'layoutData', is used only to determine sequence lengths,
// and should be the same that was used to Where().
// PackedIndex must have been created with PackedIndex() node, and is
// otherwise opaque to users.
// -----------------------------------------------------------------------

template <class ElemType>
class ScatterPackedNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ScatterPacked"; }

    // our inputs
    static const size_t LAYOUTDATA = 0;
    static const size_t INDEXDATA  = 1;
    static const size_t SOURCEDATA = 2;

public:
    DeclareConstructorFromConfigWithNumInputs(ScatterPackedNode);
    ScatterPackedNode(DEVICEID_TYPE deviceId, const wstring& name) :
        Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override;
    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override;
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override { return childIndex == INDEXDATA; }
    virtual void Validate(bool isFinalValidationPass) override;
};

// -----------------------------------------------------------------------
// DiagonalNode -- extract diagonal elements of a square matrix into a row vector
// -----------------------------------------------------------------------

template <class ElemType>
class DiagonalNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"Diagonal"; }

public:
    DeclareConstructorFromConfigWithNumInputs(DiagonalNode);
    DiagonalNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        InputRef(0).ValueAsMatrix().AssignDiagonalValuesTo(ValueAsMatrix()); // TODO: use tensor lib; this is a stride operation
#if NANCHECK
        Value().HasNan("Diagonal");
#endif
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t /*inputIndex*/) override
    {
#if 1
        NOT_IMPLEMENTED;
#else
        // The Implementation below is currently broken
        auto& inputGradientValues = InputRef(0).GradientAsMatrix();
        auto& gradientValues = GradientAsMatrix();

        // BUGBUG: This should use the memshare mechanism.
        // TODO: use tensor lib, then this will be easy, no memsharing needed
        Matrix<ElemType> diag = gradientValues.DeepClone();
        // BUGBUG: Resize does not preserve data - should be a reinterpret operation
        diag.Resize(gradientValues.GetNumCols(), 1);

        inputGradientValues.SetValue(0);
        // BUGBUG: Must *add* to gradient!
        inputGradientValues.SetDiagonalValue(diag);
#endif
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr;

        if (isFinalValidationPass && Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation cannot operate on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());

        size_t dim = Input(0)->GetAsMatrixNumCols();
        if (isFinalValidationPass && dim != Input(0)->GetAsMatrixNumRows())
            InvalidArgument("%ls %ls operation requires a square matrix as its input.", NodeName().c_str(), OperationName().c_str());

        if (Input(0)->HasSampleLayout())
            fprintf(stderr, "WARNING: Diagonal operation cannot inherit image size information from its child. Image size info is lost.\n");

        SetDims(TensorShape(1, dim), false);
    }
};

template class DiagonalNode<float>;
template class DiagonalNode<double>;

// -----------------------------------------------------------------------
// ReinterpretNodeBase (input) -- base class for nodes that reinterpret
// -----------------------------------------------------------------------

template <class ElemType>
class ReinterpretNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    // DeclareConstructorFromConfigWithNumInputs(ReinterpretNodeBase);
    ReinterpretNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    // stack K consecutive frames into a single frame that is K times taller
    // FrameRange and MBLayout refer to the 'to' (reduced) timeline.
    // BUGBUG: THIS IS UNTESTED!!
    static void Stack(const FrameRange& fr, const shared_ptr<MBLayout>& pMBLayout, /*const*/ Matrix<ElemType>& from, Matrix<ElemType>& to, size_t K, bool addTo)
    {
        // example
        //  input: T=2, D=2, K=3, S=2 (abcdef and uvwxyz)
        //   abc def
        //   ABC DEF
        //
        //   uvw xyz
        //   UVW XYZ
        //  target:
        //   a d
        //   A D
        //   b e
        //   B E
        //   c f
        //   C F
        //
        //   u x
        //   U X
        //   v y
        //   V Y
        //   w z
        //   W Z
        // underlying matrix storage is actually this:
        //  input:
        //   aubvcw dxeyfz
        //   AUBVCW DXEYFZ
        //  target:
        //   abcuvw defxyz
        //   ABCUVW DEFXYZ

        // I.e. this operation swaps index dimensions of a tensor:
        //   The input is a tensor of the form (D,       S, M, K, T).
        //   The output is of the form         (D, K, M, S,       T).
        //     K = stacking factor
        //     T = target steps
        //     S = #sequences
        //     D = featDim
        //     M = 1, thrown in for generality of underlying Matrix function

        // We operate on the 'to' layout, fr refers to result, not the input.
        // The input layout is different, but reshaping the input to output dimensions will allow us to pull out the right values anyway.
        auto from0 = from.Reshaped(to.GetNumRows(), to.GetNumCols()); // we operate on 'to' layout
        auto fromSlice0 = DataWithMBLayoutFor(from0, fr, pMBLayout);
        auto toSlice0 = DataWithMBLayoutFor(to, fr, pMBLayout);
        // now we got views on the right ranges of values, but with weird dimensions

        // reshape them into a unified view with D being the row dimension, and (S,M,K,T) the column dimension
        size_t D = from.GetNumRows();
        size_t SMKT = from.GetNumCols();
        auto fromSlice = fromSlice0.Reshaped(D, SMKT);
        auto toSlice = toSlice0.Reshaped(D, SMKT);

        // now to the shuffle dance
        size_t S = pMBLayout->GetNumParallelSequences();
        size_t T = pMBLayout->GetNumTimeSteps();
        size_t M = 1;
        Matrix<ElemType>::TensorShuffleScaleAndAdd(addTo ? 1.0f : 0, fromSlice, D, S, M, K, T, 1.0f, toSlice, toSlice);
    }

    // split frames of D*K elements into K consecutive frames of dimension D.
    // FrameRange and MBLayout refer to the 'from' (reduced) timeline.
    // This function is the inverse of Stack(). See comments there and exchange from and to.
    static void Unstack(const FrameRange& fr, const shared_ptr<MBLayout>& pMBLayout, /*const*/ Matrix<ElemType>& from, Matrix<ElemType>& to, size_t K, bool addTo)
    {
        auto fromSlice0 = DataWithMBLayoutFor(from, fr, pMBLayout);
        auto to0 = to.Reshaped(from.GetNumRows(), from.GetNumCols());
        auto toSlice0 = DataWithMBLayoutFor(to0, fr, pMBLayout);

        size_t D = to.GetNumRows();
        size_t SMKT = to.GetNumCols();
        auto fromSlice = fromSlice0.Reshaped(D, SMKT);
        auto toSlice = toSlice0.Reshaped(D, SMKT);

        size_t S = pMBLayout->GetNumParallelSequences();
        size_t T = pMBLayout->GetNumTimeSteps();
        size_t M = 1;
        Matrix<ElemType>::TensorShuffleScaleAndAdd(addTo ? 1.0f : 0, fromSlice, D, K, M, S, T, 1.0f, toSlice, toSlice);
    }
};

#define UsingReinterpretNodeBaseMembers UsingComputationNodeMembersBoilerplate

// TODO: This ReshapeNode should no longer be used. Its function will be taken over by Transpose and the Reshape that follows this one below.

// -----------------------------------------------------------------------
// LegacyReshapeNode (input) -- reinterpret input matrix as having different dimensions
// where the new row dimension is given, and the column dimension is inferred.
// Also optionally associate a different TensorShape with the data.
//
// DEPRECATED, do not use anymore.
//
// If input has no layout, then this reshapes the input matrix
// from (rows x cols) to (newRows x (cols / newRows * rows)).
//
// If input has a layout, then it adds or removes a nested time dimension.
//  - If newRows > rows, then we remove a time dimension by stacking all frames from the dimension into one:
//       (rows x (newRows/rows nested time steps) x T time steps)
//    -> (newRows x T time steps).
//  - If newRows < rows, then we add a time dimension, going
//       (rows x T time steps)
//    -> (newRows x (rows/newRows nested time steps) x T time steps).
//    which requires the nested time sequence to have the correct number of steps.
// E.g. going from rows=20 to newRows=40 assumes a nested time sequence of 2 steps, which are grouped into one step, with the two vectors stacked.
// Multiple parallel sequences are treated independently.
// TODO: This definition is poor; we should use a different node name, and specify the factor directly.
//       We may hide that in BrainScript, but better use different node types.
//       E.g. ReinterpretRowStackAsSequence and ReinterpretSequenceAsRowStack.
// BUGBUG: This is not actually implemented yet. Instead, it goes from 1 to K steps or from K to 1 step. This is temporary/experimental, until the plumbing for nesting is there.
//
// Thirdly, LegacyReshapeNode can also be used to update only the TensorShape. In that case, the MBLayout is kept as is.
//
// Note: The new row dimension must be a straight multiple or divisor of the current row dimension.
// To reshape to a non-multiple go to row dim 1 first.
//
// Unlike most other nodes, this node has intimate inside knowlegde of MBLayouts and frameRanges.
// TODO: Changing the TensorShape does not seem to belong here.
// -----------------------------------------------------------------------

template <class ElemType>
class LegacyReshapeNode : public ReinterpretNodeBase<ElemType>
{
    typedef ReinterpretNodeBase<ElemType> Base;
    UsingReinterpretNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"LegacyReshape";
    }

public:
    LegacyReshapeNode(DEVICEID_TYPE deviceId, const wstring& name, size_t numRows = 0, const TensorShape& imageLayout = TensorShape())
        : Base(deviceId, name),
          m_numTargetRows(numRows),
          m_targetImageLayout(imageLayout)
    {
    }
    LegacyReshapeNode(const ScriptableObjects::IConfigRecordPtr configp)
        : LegacyReshapeNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"numRows"), ImageDimensions::AsTensorShape(configp->Get(L"imageWidth"), configp->Get(L"imageHeight"), configp->Get(L"imageChannels"), ImageLayoutKindFrom(configp->Get(L"imageLayout"))/*legacy*/))
    {
        // BUGBUG: We should not operate on image layouts here, but on a proper tensor layout.
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LegacyReshapeNode<ElemType>>(nodeP);
            node->m_numTargetRows = m_numTargetRows;
            node->m_targetImageLayout = m_targetImageLayout;
        }
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_numTargetRows;
        m_targetImageLayout.Load(fstream, /*acceptLegacyFormat=*/true);
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_numTargetRows;
        m_targetImageLayout.Save(fstream);
    }

    virtual std::string /*IComputationNode::*/ FormatOperationPrototype(const std::string& extraArgs) const override
    {
        return Base::FormatOperationPrototype(extraArgs + msra::strfun::strprintf(", NumOfRows=%lu, imageWidth=%lu, imageHeight=%lu, imageChannels=%lu)", m_numTargetRows, m_targetImageLayout[1], m_targetImageLayout[2], m_targetImageLayout[0]));
    }

    // TODO: Clarify/resolve the semantic overlap between BeginForwardProp() and UpdateFunctionMBSize().
    virtual void /*IComputationNode::*/ BeginForwardProp() override
    {
        // create the derived layout
        if (m_pMBLayout && factor() != 1)
        {
            // BUGBUG: This assumes that the layout is complete at this point in time (RecurrentNodeBase makes the same assumption).
            //         This assumption is correct at present, but will becomes invalid once we go sequence-to-sequence.
            if (weStack())
            {
                // going from many samples to one: layout entry will get no flags
                if (InputRef(0).GetMBLayout()->GetNumTimeSteps() * InputRef(0).GetSampleMatrixNumRows() / m_numTargetRows != 1)
                    LogicError("LegacyReshapeNode::BeginForwardProp() faking to remove a nested time dimension only works when going back to a single frame per sequence.");
                // we are in frame mode now
                m_pMBLayout->InitAsFrameMode(InputRef(0).GetNumParallelSequences());
            }
            else
            {
                // going from one sample to many: layout will get SentenceStart/SentenceEnd flags for the sequence we expand into
                if (InputRef(0).GetMBLayout()->GetNumTimeSteps() != 1)
                    LogicError("LegacyReshapeNode::BeginForwardProp() faking to add a nested time dimension only works when coming from a single frame per sequence.");
                m_pMBLayout->Init(InputRef(0).GetNumParallelSequences(), InputRef(0).GetMBLayout()->GetNumTimeSteps() * InputRef(0).GetSampleMatrixNumRows() / m_numTargetRows);
                for (size_t s = 0; s < m_pMBLayout->GetNumParallelSequences(); s++)
                    m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, s, 0, GetMBLayout()->GetNumTimeSteps());
                // BUGBUG: In the future, NEW_SEQUENCE_ID will be incorrect here; need an iterator over sequences in there.
            }
        }
        // Call this at the end because this will resize Value(), but that requires the updated MBLayout. TODO: Clarify the sequence of events. Should we update the MBLayout in UpdateFunctionMBSize()?
        Base::BeginForwardProp();
    }

    // notes:
    //  - input and output have different time base and different layouts (unless the canonical case of factor() == 1)
    //  - fr refers to *functionValues*, not the inputs
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t rows = InputRef(0).Value().GetNumRows(), cols = InputRef(0).Value().GetNumCols();
        size_t newCols = cols * rows / m_numTargetRows;
        assert(newCols * m_numTargetRows == cols * rows); // follows from above check
        Value().VerifySize(m_numTargetRows, newCols);

        // no layout case: this is indeed just a reshape. Same for canonical case
        // (We still need to copy the values since there is currently no way to point to an input function value while reshaping at the same time.)
        if (!m_pMBLayout || factor() == 1)
        {
            Value().Reshaped(newCols * m_numTargetRows, 1).AssignValuesOf(InputRef(0).Value().Reshaped(cols * rows, 1)); // copy the values as one long vector
        }
        // layout case: reshape semantics happens across parallel seqeunces, i.e. requiring data shuffling
        else
        {
            // TODO: It does not make sense to run LegacyReshapeNode frame-by-frame inside a loop, because it changes the time base.
            //       However, in the future, we should be able to run inside an outer loop.
            if (!fr.IsAllFrames())
                InvalidArgument("%ls %ls operation cannot be run from inside a loop since it changes the time base.", NodeName().c_str(), OperationName().c_str());
            if (weStack())
                Base::Stack(fr, m_pMBLayout, InputRef(0).Value(), Value(), factor(), false /*addTo*/);
            else
                Base::Unstack(fr.WithLayout(InputRef(0).GetMBLayout()), InputRef(0).GetMBLayout(), InputRef(0).Value(), Value(), factor(), false /*addTo*/);
        }
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        size_t rows = InputRef(0).Value().GetNumRows(), cols = InputRef(0).Value().GetNumCols();
        size_t newCols = cols * rows / m_numTargetRows;

        // no layout case: this is indeed just a reshape. Same for canonical case
        if (!m_pMBLayout || factor() == 1)
        {
            InputRef(0).Gradient().Reshaped(cols * rows, 1) += Gradient().Reshaped(newCols * m_numTargetRows, 1); // treat the values as one long vector
        }
        // layout case: reshape semantics happens across parallel seqeunces, i.e. requiring data shuffling
        else
        {
            if (weStack())
                Base::Unstack(fr, m_pMBLayout, Gradient(), InputRef(0).Gradient(), factor(), true /*addTo*/);
            else
                Base::Stack(fr.WithLayout(InputRef(0).GetMBLayout()), InputRef(0).GetMBLayout(), Gradient(), InputRef(0).Gradient(), factor(), true /*addTo*/);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (factor() == 1) // canonical case: keeps the MBLayout(e.g. only changing the TensorShape)
            m_pMBLayout = Input(0)->GetMBLayout();
        else if (Input(0)->HasMBLayout())
        {
            if (!m_pMBLayout)
            {
                m_pMBLayout = make_shared<MBLayout>(); // mini-batch data: this generates a new layout
                m_pMBLayout->SetUniqueAxisName(NodeName());
            }
        }
        else
            assert(!m_pMBLayout); // reshaping non-mini-batch data

        size_t newCols = 1; // dummy
        if (!m_pMBLayout)
        {
            size_t rows = Input(0)->GetAsMatrixNumRows(), cols = Input(0)->GetAsMatrixNumCols();
            newCols = cols * rows / m_numTargetRows;
            if (isFinalValidationPass)
            {
                if ((m_numTargetRows > rows && m_numTargetRows % rows != 0) || // grouping columns
                    (m_numTargetRows < rows && rows % m_numTargetRows != 0))   // splitting columns
                    InvalidArgument("%ls %ls operation: output row dimension %d is not an integer multiple or divisor of input dimension %d", NodeName().c_str(), OperationName().c_str(), (int) m_numTargetRows, (int) rows);
                if (rows * cols != m_numTargetRows * newCols)
                    LogicError("%ls %ls operation: unexpected dimension mismatch", NodeName().c_str(), OperationName().c_str());
            }
        }

        // patch up m_targetImageLayout, which was originally a construction parameter
        InferTargetSampleLayout();

        // setting any dimension to 0 means lose the tensor, flatten to vector
        if (m_targetImageLayout.GetNumElements() == 0)
        {
            if (Input(0)->HasSampleLayout())
                fprintf(stderr, "WARNING: Reshape operation cannot inherit image size information from its child. Image size info is lost.\n");
            // TODO: We need to decide what reshaping means in presence of a tensor.
            if (HasMBLayout())
                SetDims(TensorShape(m_numTargetRows), true);
            else
                SetDims(TensorShape(m_numTargetRows, newCols), false);
        }
        else
        {
            if (m_numTargetRows != m_targetImageLayout.GetNumElements())
                LogicError("LegacyReshapeNode: InferTargetSampleLayout() computed a sample layout [%s] that mismatches m_numTargetRows %d.", string(m_targetImageLayout).c_str(), (int) m_numTargetRows);
            SetDims(m_targetImageLayout, HasMBLayout());
        }
    }

private:
    size_t m_numTargetRows;
    bool weStack() const
    {
        return m_numTargetRows > Input(0)->GetSampleMatrixNumRows();
    } // do we stack (multiple frames into one)
    size_t factor() const
    {
        return m_numTargetRows > Input(0)->GetSampleMatrixNumRows() ? m_numTargetRows / Input(0)->GetSampleMatrixNumRows() : Input(0)->GetSampleMatrixNumRows() / m_numTargetRows;
    } // factor by which we stack or unstack
    TensorShape m_targetImageLayout;

    // This infers dimensions in m_targetImageLayout.
    // Users are allowed to provide 2 (out of 3) image dimensions.
    // One missing dimension can be inferred. If two dimensions are
    // unspecified it throws a runtime error.
    void InferTargetSampleLayout()
    {
        // BUGBUG: Below is the result of refactoring and only works for rank-3 tensors. Generalize.
        if (m_targetImageLayout[1] > 0)
        {
            if (m_targetImageLayout[2] > 0)
            {
                if (m_targetImageLayout[0] > 0)
                {
                    if (m_targetImageLayout.GetNumElements() != m_numTargetRows)
                        RuntimeError("Image dimensions do not match row size.");
                }
                else
                {
                    if (m_numTargetRows % (m_targetImageLayout[1] * m_targetImageLayout[2]) > 0)
                        RuntimeError("Image row size is not a multiple of specified image dimensions.");
                    else
                        m_targetImageLayout = TensorShape(m_numTargetRows / (m_targetImageLayout[1] * m_targetImageLayout[2]), m_targetImageLayout[1], m_targetImageLayout[2]);
                }
            }
            else
            {
                if (m_targetImageLayout[0] > 0)
                {
                    if (m_numTargetRows % (m_targetImageLayout[1] * m_targetImageLayout[0]) > 0)
                        RuntimeError("Image row size is not a multiple of specified image dimensions.");
                    else
                        m_targetImageLayout = TensorShape(m_targetImageLayout[0], m_targetImageLayout[1], m_numTargetRows / (m_targetImageLayout[1] * m_targetImageLayout[0]));
                }
                else
                {
                    RuntimeError("At least two image dimensions must be specified.");
                }
            }
        }
        else
        {
            if (m_targetImageLayout[2] > 0)
            {
                if (m_targetImageLayout[0] > 0)
                {
                    if (m_numTargetRows % (m_targetImageLayout[2] * m_targetImageLayout[0]) > 0)
                        RuntimeError("Image row size is not a multiple of specified image dimensions.");
                    else
                        m_targetImageLayout = TensorShape(m_targetImageLayout[0], m_numTargetRows / (m_targetImageLayout[2] * m_targetImageLayout[0]), m_targetImageLayout[2]);
                }
                else
                    RuntimeError("At least two image dimensions must be specified.");
            }
            else if (m_targetImageLayout[0] > 0)
                RuntimeError("At least two image dimensions must be specified.");
            else
                m_targetImageLayout = TensorShape(1, m_numTargetRows, 1);
        }
    }
};

template class LegacyReshapeNode<float>;
template class LegacyReshapeNode<double>;

/*

notes on tensor operations
==========================

reshaping
---------

 - on dimension index 'axis' and 'tensorShape'
    - 'tensorShape': a vector of dimensions, e.g. 640:480:3:30 could describe a 1-second RGB video of VGA dimensions at 30 fps
    - 'axis' specifies a specific tensor index
       - axis > 0 is a regular sample-tensor axis. E.g. for a matrix, axis=1 would be the row dimension, and axis=2 in the above example has dimension 480.
       - axis < 0 denote time indices (recurrent loops). Rank=-1 is the innermost time index.
          - some operation may have the same BS surface form but may be implemented differently for axis < 0
       - axis = 0 denotes the index of the parallel sequence
          - Since all operations logically operate on a single sequence, i.e. parallel sequences generally cannot be indexed by the user.
          - Exceptions: training criteria, BatchNormalization, ...WithNegativeSamples (we should not need this)
       - I don't like that 'axis' refers to the index of the dimension as well as the number of elements in that dimension. Axis (numpy)?

 - Reshaping:   --these are all implemented in C++ by ReshapeNode
    - Reshape(x, tensorShape, beginAxis=0, endAxis=0)
        - just replaces metadata m_sampleLayout
        - one dimension may be specified as 0 and will be inferred
        - optional beginAxis/endAxis denote to only replace a sub-range of dims, primarly for implementing ReshapeDimension() and FlattenRank()
    - ReshapeDimension(x, axis, tensorShape) = Reshape(x, tensorShape, beginAxis=axis, endAxis=axis+1)
       - reinterprets one dimension as multiple, where the number of elements remains the same
       - one of the new dimensions may be specified as 0 and will be inferred
    - FlattenDimensions(x, axis, num) = Reshape(x, 0, beginAxis=axis, endAxis=axis+1)
       - replace two or more consecutive dims by a single axis with the same number of elements
    - SplitDimension(x, axis, N) = ReshapeDimension(x, axis, 0:N)
       - splits a dimension into a new tensor dimension, injecting them into a new dimension
       - note: to split into multiple outputs (like tf.split()), use a BrainScript loop with Slice().
 - Slicing   --all implemented in C++ by SliceNode
    - Slice(x, axis, beginIndex, endIndex, stride=1, phase=0)
       - reduces a axis to index range [beginIndex,endIndex)
       - negative bounds specify "from end" (end=0 means end if stride>0, and begin=0 means end if stride<0)
       - also applies to time, e.g.:
          - pick last frame of a sequence (for s2s): Slice(x, -1, -1, 0)    // first -1 is axis and means the time index
          - trim first and last 3 frames of a sequence: Slice(x, -1, 3, -3) // 3 means begin at frame 3, -3 means end is 3rd frame from the end
          - this will update MBLayout
          - but is implemented completely differently using Gather()
       - the optional stride and phase parameters are for implementing downsampling (stride>1) and reversing (begin=-1, stride=-1)
          - not implemented
       - multiple slice operations can be combined by concatenating the spec vector, e.g. Slice(x, dim1:dim2, begin1:begin2, end1:end2)
          - not implemented; could be implemented in BS
       - today's RowSlice(begin, num, x) = Slice(x, 1, begin, begin + num)
       - like torch.narrow()
       - can implement TF unpack() and Torch split() as a BrainScript loop with multiple Slice() operations
       - internally implemented by tensor lib opCopy with manipulated m_strides/m_offset
    - Select(x, axis, index) = FlattenDimensions(Slice(x, axis, index, index+1), index > 1 ? index-1 : index, index > 1 ? index : index+1)
       - narrow axis to a single index, then drop the axis. Result will have one axis less.
       - like torch.select()
       - can implement squeezing a axis-1 axis: Select(x, axis:0)
    - Squeeze(x, axis) = Select(x, axis, 0)
 - Splicing:   --all implemented in C++ by SpliceNode
    - Splice(inputs, axis)
       - splice multiple inputs inputs[0]:inputs[1]:... along given axis (=RowStack for vectors)
       - inputs must have identical dimensions except for:
          - the specified axis
          - broadcasting dimensions (e.g. used to implement Pad())
       - one can splice in time
          - e.g. prepend a vector to a time sequence
          - this will create a new MBLayout
          - this will be implemented separately on BrainScript level
       - like tf.concat()
    - Pack(inputs, axis) = ReshapeDimension(Splice(inputs, axis), axis, (0:Length(inputs)) )
       - like splice but inserts new axis of dimension Length(inputs)
       - inputs must have identical dimensions for all dims (except for broadcasting)
       - axis can be a time dimension; then a new inner-most time dimension will be inserted
       - like tf.pack()
    - Pad(x, axis, howManyBefore, howManyAfter, with=0) = Splice(Constant(with, tensorShape=1*(axis-1):howManyBefore),  x,  Constant(with, tensorShape=1*(axis-1):howManyAfter), axis)
       - inverse of slice, pad with a constant value
       - dimensions specified relative, can pad at start and end
       - in time: pad neighbor frames
    - Repeat(x, axis, numRepeats) = Splice(x*numRepeats, axis)
       - generalizes CNTK RowRepeat(x, numRepeats) = Repeat(x, 1, numRepeats)
       - to repeat multiple, specify vectors, e.g. Repeat(x, dim1:dim2, numRepeats1:numRepeats2)
       - like tf.tile() and Matlab's repmat()
 - Transposition
    - TransposeDimensions (input, dim1, dim2)
       - swaps index dimensions dim1 and dim2. The values are 1-based; 1 stands for the leading dimension.
       - new dimensions can be created; e.g. a column vector can be transposed into a row vector, which is a [1 x N] tensor
       - transposing into the time dimension is currently not supported
       - internally implemented with tensor lib by shuffling dimensions with their strides
       - input may be minibatch data or not
       - like torch.transpose()
    - Transpose (input) = TransposeDimensions (input, 1, 2)
 - Re-indexing:   --implemented by ReindexRankNode and SliceNode
    - ReindexDimension(x, axis, indexVector)
       - splice x[..., indexVector[0], ...], x[..., indexVector[1], ...], etc. with indexVector[.] at given axis
       - indexVector must be invertible if it is intended to backpropagate through this node
    - DownsampleDimension(x, axis, n, phase=0) = Slice(x, axis, 0, 0, stride=n)
       - select every n-th element, starting with index 'phase'
       - time dims allowed. Phase is then a modulus w.r.t. where a sequence is inside the minibatch (may require a ReconcileLayout() before to match layouts)
       - TODO: use a bool vector for the time dimensions --> Gather()
    - ReverseDimension(x, axis) = Slice(x, axis, -1, 0, stride=-1)
       - reverses the direction of a axis
       - when applied to time dims, this creates a new layout (which is also flipped)
       - could be implemented with Gather() as well!

 - misc.:
    - note: much would look more natural if we had OO syntax, e.g. x.Slice(axis, begin, end).FlattenDimensions(...)
      Could be done by exposing all methods on ComputationNode... not currently feasible with BrainScript, but e.g. with Python bindings
    - torch.unfold (axis, size, step)
       - create a convolution matrix (stride magic)
    - CyclicallyPermuteRank(x, axis, step)
       - rotates indices
       - also applies to time dimensions
    - duplicate elements
    - Gather
       - from Torch and TF
    - TF also has:
       - 'gather': reindexing
       - 'dynamic_partition', 'dynamic_stitch'
    - Torch:
       - expand (axis, range): broadcasts dimension 'axis' as a new dimension with 'range'. Not needed I think.
       - repeatTensor: like tile but with weird reshaping
       - squeeze: removes all singleton dimensions, or a specific one. We can remove a specific one with Select().
    - TODO:
       - give names to dimensions?
       - do we want to allow time offsets in layouts?

reductions
----------

 - these are/will be implemented as a node for samples, and as recurrences for sequences
 - ReduceSum
    - sum over all elements of a dimension, or over time
 - ReduceMax, ReduceMin
    - max
    - can use MaxPooling?
 - ReduceMean
    - av
    - can use AvPooling?
 - ArgMax, ArgMin
    - we already have that somewhere, for evaluation
 - All, Any
    - logical test
 - TF also has:
    - reduce_prod
    - segment_sum etc.; we use sequences
    - listdiff
    - where: indices of 'true' values  -> 2D tensor of coordinates (unlike our Where)
    - unique (1D only)
    - edit_distance
    - invert_permutation: invert a permutation index vector
    - top_k

convolutions
------------

 - convolution
    - convolution with filter
    - max pool (=convolution with weights 1 and max reduction)
    - av pool (=convolution with uniform filter)
 - also in time: by specifying more filter dimensions [TODO]
    - tricky bit: boundaries; may need expansion or reduction of sequences

element-wise operations
-----------------------

 - PlusNode, MinusNode, ElementTimes
 - with broadcasting, these implement:
    - PlusNode with bias, PlusNode for images
    - 1-x
    - ScaleNode, RowElementTimes, ColumnElementTimes
 - elementwise nonlinearities as usual  [TODO: complete them]
 - logical ops (can be done by comparison ops actually)
 - Clamp
    - bounds are passed as 'Const'
 - TF: in_top_k
 - Torch performs these ops (e.g. add) as vector, without broadcasting
    - e.g. max reduces, while cmax does not. Our solution is better... really? How to specify reduce?

gradient operations
-------------------

 - TF: are nodes, e.g. clip_by_value
    - input should be parameters as well, so they can be computed
 - need a node to stop gradient propagation?
 - can we use nodes to specify things like AdaGrad and momentum?

debugging
---------

 - node that prints activations
 - node that prints mean/var of gradients

other
-----

 - per-node learning rate: can specify additional parameter for each node? Maybe fold with updateLearnableParameter?
 - give dimensions a name?
 - can we interleave variable-length ones? Concat into a single dimensions, using strides?

 */

}}}
