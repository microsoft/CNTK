//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReshapingNodes.cpp -- collection of nodes that reshape or sub-sample matrices leading to layout changes
//

#include "Basics.h"
#include "ReshapingNodes.h"
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
// ReduceElements (op, axis=, input)
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const /*override*/
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<ReduceElementsNode<ElemType>>(nodeP);
        node->m_axis        = m_axis;
        node->m_operation   = m_operation;
        node->m_reductionOp = m_reductionOp;
        node->m_scale       = m_scale;
    }
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    Base::Load(fstream, modelVersion);
    fstream >> m_axis >> m_operation;
    ValidateOp();
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::Save(File& fstream) const /*override*/
{
    Base::Save(fstream);
    fstream << m_axis << m_operation; // note: we serialize the string and not the opcode, since opcodes may change
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::ForwardProp(const FrameRange& fr) /*override*/
{
    // get the args
    size_t rank = DetermineElementwiseTensorRank();
    auto result =             ValueTensorFor(rank, fr);
    auto input  = InputRef(0).ValueTensorFor(rank, fr);

    // the actual operation is a Copy with reduction, where the magic is in the reduction op
    // For "Mean", m_scale is 1/#elements, and 1 otherwise.
    result.DoUnaryOpOf(0, input, m_scale, ElementWiseOperator::opCopy, m_reductionOp);
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr) /*override*/
{
    assert(inputIndex == 0), inputIndex;

    // get the args
    size_t rank = DetermineElementwiseTensorRank();
    auto sliceOutputGrad =             GradientTensorFor(rank, fr); // propagate from this one...
    auto sliceInputGrad  = InputRef(0).GradientTensorFor(rank, fr); // ...to this one

    // gradients are not as simple as passing an op-code, unfortunately
    switch (m_reductionOp)
    {
    case ElementWiseOperator::opSum:
        // "Sum":  broadcast the gradient
        // "Mean": same as "Sum" with scaling by 1/#dims
        sliceInputGrad.AddCopyOf(sliceOutputGrad, m_scale);
        break;

    case ElementWiseOperator::opLogSum:
        {
            auto input = InputRef(inputIndex).ValueTensorFor(rank, fr);
            auto output = ValueTensorFor(rank, fr.AllowBroadcast());
            // Let: f(x, y, z) = log(exp x + exp y + exp z)
            // For the derivative we get:
            // df / dx = exp(x)/exp(f)
            //         = exp(x � f)
            sliceInputGrad.AddElementwiseProductWithExpOfDiffOf(sliceOutputGrad, input, output);
    }
        break;

    case ElementWiseOperator::opMin:
    case ElementWiseOperator::opMax:
        auto input = InputRef(inputIndex).ValueTensorFor(rank, fr);
        auto output = ValueTensorFor(rank, fr.AllowBroadcast());

        // POTENTIAL PROBLEM:
        // For ReduceMin/Max there are combinations of input values where the gradient is not defined because the function has an edge at these points.
        // E.g. for ReduceMin this is the case when the minimum input value is attained by several inputs at the same time.
        // In these cases there is no correct gradient.The question is if this could lead to any problems.
        // Let's look at two scenarios where this might happen:
        //
        // * Scenario 1: The input comes from a layer of nodes like e.g. ReLU and some of them might operate in the regime where they clip to a constant value.
        //   In this case it's not a problem that the input gradient is kind of bad as the derivative of the concerning input nodes will be zero anyway.
        //
        // * Scenario 2: The input data is directly coming from training data. Here bad gradients don't matter as we wouldn't wan't to propagate gradients to the training data.
        //
        // So as we don't have a better solution yet and it probably doesn't have impact let's stay with the current solution.
        // Also note that for Clip , Min, Max and ReLU we have the same kind of problem.
        sliceInputGrad.AddCopyIfEqualOf(input, output, sliceOutputGrad);
        break;

        // more coming
    }
}

template <class ElemType>
/*virtual*/ bool ReduceElementsNode<ElemType>::OutputUsedInComputingInputNodesGradients() const /*override*/
{
    switch (m_reductionOp)
    {
    case ElementWiseOperator::opSum:    return false;
    case ElementWiseOperator::opLogSum: return true;
    case ElementWiseOperator::opMin:    return true;
    case ElementWiseOperator::opMax:    return true;
    }
    LogicError("Should not get here.");
}

template <class ElemType>
/*virtual*/ bool ReduceElementsNode<ElemType>::InputUsedInComputingInputNodesGradients(size_t inputIndex) const /*override*/
{
    switch (m_reductionOp)
    {
    case ElementWiseOperator::opSum:    return false;
    case ElementWiseOperator::opLogSum: return true;
    case ElementWiseOperator::opMin:    return true;
    case ElementWiseOperator::opMax:    return true;
    }
    LogicError("Should not get here.");
}

// map the operation specified as a string to an ElementWiseOperator value.
template <class ElemType>
void ReduceElementsNode<ElemType>::ValidateOp()
{
#if 1 // legacy with initial experiments, delete this soon
    if (m_operation == L"Plus") m_reductionOp = ElementWiseOperator::opSum;
    else
#endif
    if      (m_operation == L"Sum")    m_reductionOp = ElementWiseOperator::opSum;
    else if (m_operation == L"Mean")   m_reductionOp = ElementWiseOperator::opSum;
    else if (m_operation == L"LogSum") m_reductionOp = ElementWiseOperator::opLogSum;
    else if (m_operation == L"Min")    m_reductionOp = ElementWiseOperator::opMin;
    else if (m_operation == L"Max")    m_reductionOp = ElementWiseOperator::opMax;

    // more here
    else InvalidArgument("%ls was given an invalid operation code '%ls'. Allowed are: 'Sum', 'Max', 'Min'.", NodeDescription().c_str(), m_operation.c_str());
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    Base::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    // validate the opcode (in case we got instantiated empty and never updated)
    ValidateOp();

    let shape = Input(0)->GetSampleLayout();
    auto dims = shape.GetDims();
    size_t reducedDim = 0; // (init to keep compiler happy)
    if (m_axis == 0)
    {
        reducedDim = shape.GetNumElements();
        dims = { 1 };                       // entire sample is reduced to a scalar
    }
    else if (m_axis - 1 >= 0 && m_axis - 1 < dims.size())
    {
        reducedDim = dims[m_axis - 1];
        dims[m_axis - 1] = 1;               // one axis is reduced to a scalar
    }
    else if (isFinalValidationPass)
        InvalidArgument("The shape of %ls [%s] has no axis %d", NodeDescription().c_str(), string(shape).c_str(), m_axis);

    // for "Mean", we must divide by #elements
    if (isFinalValidationPass && m_operation == L"Mean")
        m_scale = (ElemType)(1.0 / reducedDim);
    else
        m_scale = (ElemType)1;

    SetDims(TensorShape(dims), Input(0)->HasMBLayout());
}

template class ReduceElementsNode<float>;
template class ReduceElementsNode<double>;

// -----------------------------------------------------------------------
// Where(bitVector) -- extract indices of non-0 values in a sequence
// -----------------------------------------------------------------------

// wrapper class to pass MBLayout sequence vector to PackSequences()
struct SequenceLengthVector
{
    typedef vector<vector<size_t>> SequenceVector;
    typedef MBLayout::SequenceInfo SequenceInfo;
    const SequenceVector& m_sequenceVector;        // vector of sequences (to get sequence length)
    const vector<SequenceInfo>& m_sequenceInfo;    // original sequence info (for seqId)
    SequenceLengthVector(const vector<SequenceInfo>& sequenceInfo, const SequenceVector& sequenceVector) : m_sequenceInfo(sequenceInfo), m_sequenceVector(sequenceVector) { }
    size_t size() const { return m_sequenceInfo.size(); }
    MBLayout::SequenceInfo operator[](size_t i) const // return a descriptor of the new sequence
    {
        SequenceInfo seq;
        seq.seqId = m_sequenceInfo[i].seqId;
        seq.s = i;
        seq.tBegin = 0;
        seq.tEnd = m_sequenceVector[i].size();
        return seq;
    }
    void operator=(const SequenceLengthVector&) = delete;
};

// TODO: Where should the MBLayout be created--in BeginForwardProp() or ForwardProp()?
//       BeginForwardProp() should generally have no access to the actual values,
//       while ForwardProp() might be too late. We may have to define the semantics here.
// BUGBUG: This is the first node with value-dependent MBLayout. It resizes Value(), which we otherwise always do before.
template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    // gather all sequences
    let& inMBLayout = InputRef(0).GetMBLayout();
    let& input = InputRef(0).Value();
    let& sequences = inMBLayout->GetAllSequences();
    auto& indexSequences = m_indexSequenceBuffer;
    if (indexSequences.size() < sequences.size())
        indexSequences.resize(sequences.size());
    for (size_t i = 0; i < sequences.size(); i++)
    {
        let& seq = sequences[i];
        if (seq.seqId == GAP_SEQUENCE_ID)
            continue;
        auto& indexSequence = indexSequences[i];
        indexSequence.clear();
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
            if (input(0, inMBLayout->GetColumnIndex(seq, t))) // this is the condition check that this node performs; the meat
                indexSequence.push_back(t);
        // Note: The above accesses m_value directly on the CPU, putting it into BOTH state, possibly for other consumers as well.
    }
    input.CollapseDataLocation(); // BUGBUG: Move back, since BOTH state is broken at present.
    // create a new MBLayout
    let& outMBLayout = GetMBLayout();
    outMBLayout->InitAsPackedSequences(SequenceLengthVector(sequences, indexSequences), /*temp*/m_placementBuffer, /*temp*/m_rowAllocationsBuffer);
    // copy to output
    vector<ElemType> buf(outMBLayout->GetNumCols(), numeric_limits<ElemType>::quiet_NaN()); // STL cannot easily avoid initializing, so we might as well init with NaN for gaps
    let size = min(sequences.size(), outMBLayout->GetAllSequences().size()); // no non-gap sequence has an index beyond this
    for (size_t i = 0; i < size; i++)
    {
        let& seq = outMBLayout->GetAllSequences()[i];
        if (seq.seqId == GAP_SEQUENCE_ID) // gaps will keep the NaN
            continue;
        let& indexSequence = indexSequences[i];
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
            buf[outMBLayout->GetColumnIndex(seq, t)] = (ElemType)indexSequence[t];
    }
    // there may be dangling gaps at the end. Take the opportunity to verify this.
    for (size_t i = size; i < sequences.size(); i++)
        assert(sequences[i].seqId == GAP_SEQUENCE_ID);
    for (size_t i = size; i < outMBLayout->GetAllSequences().size(); i++)
        assert(outMBLayout->GetAllSequences()[i].seqId == GAP_SEQUENCE_ID);
    // the result will be kept in CPUDEVICE, since most likely we will access it again in PackedIndexNode
    Value().TransferToDeviceIfNotThere(CPUDEVICE, /*isBeingMoved=*/ true, /*emptyTransfer=*/ true, /*updatePreferredDevice=*/ true);
    Value().SetValue(1, outMBLayout->GetNumCols(), CPUDEVICE, buf.data(), MatrixFormat::matrixFormatColMajor);
}

template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/) /*override*/
{
    // we cannot backprop through a condition
    return;
}

template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);
    // we generate its own MBLayout
    if (isFinalValidationPass && !Input(0)->HasMBLayout())
        InvalidArgument("%ls %ls operation can only operate on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());
    if (!m_pMBLayout)
    {
        m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
        m_pMBLayout->SetUniqueAxisName(m_dynamicAxisName);
    }
    // we map scalars to scalars
    if (isFinalValidationPass && Input(0)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls %ls operation can only operate on scalar input.", NodeName().c_str(), OperationName().c_str());
    SetDims(TensorShape(1), true);
}

template class WhereNode<float>;
template class WhereNode<double>;

// -----------------------------------------------------------------------
// PackedIndexNode(targetObject, indexSequence) -- map sequence
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void PackedIndexNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    let& sourceMBLayout = InputRef(SOURCEDATA).GetMBLayout(); // only used for index conversion
    let& indexMBLayout  = InputRef(INDEXDATA).GetMBLayout();
    let&  index  = InputRef(INDEXDATA).Value(); // per-seq index values that are to be mapped
    auto& result =                   Value(); // packed index values as mapped to sourceData's layout
    // loop over sourceSequences
    // Input matrix contains time indices for each sequence that refer to frames inside that sequence.
    // We replace every per-sequence index by the resolved column index w.r.t. the same MBLayout.
    let& sourceSequences = sourceMBLayout->GetAllSequences();
    for (size_t i = 0; i < sourceSequences.size(); i++)
    {
        let& sourceSeq = sourceSequences[i];
        if (sourceSeq.seqId == GAP_SEQUENCE_ID)
            continue;
        let& indexSeq = indexMBLayout->FindSequence(sourceSeq.seqId);          // find corresponding entry in indexMBLayout
        for (size_t tIndex = 0; tIndex < indexSeq.GetNumTimeSteps(); tIndex++) // map all index values in index sequence
        {
            let jIndex  = indexMBLayout->GetColumnIndex(indexSeq, tIndex);    // map time index to actual location in the matrix storage object
            let tSource = (size_t)index(0, jIndex);                           // the new time location (relative to source sequence)
            let jSource = sourceMBLayout->GetColumnIndex(sourceSeq, tSource); // map new time index as well. This performs a range check.
            result(0, jIndex) = (ElemType)jSource;
        }
    }
    // Note: maybe this is no longer needed, now that we do the same inside UpdateFunctionValueSize() for all nodes.
    result.CollapseDataLocation(); // BUGBUG: Move back, since BOTH state is broken at present.
}

template <class ElemType>
/*virtual*/ void PackedIndexNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/) /*override*/
{
    // we cannot backprop through a condition
    // Can we?
    return;
}

template <class ElemType>
/*virtual*/ void PackedIndexNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);

    // inherit both MBLayout and sample dimension (scalar) from indexData
    // Because we map (per-seq) index sequence to (packed) index sequence. Target is only for index calculation.
    m_pMBLayout = Input(INDEXDATA)->GetMBLayout();
    if (isFinalValidationPass && (!Input(INDEXDATA)->HasMBLayout() || !Input(SOURCEDATA)->HasMBLayout()))
        LogicError("%ls %ls operation requires both inputs to be minibatch data (must have MBLayouts).", NodeName().c_str(), OperationName().c_str());

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls %ls operation requires the second argument (indexData) to be a scalar sequence.", NodeName().c_str(), OperationName().c_str());

    SetDims(Input(INDEXDATA)->GetSampleLayout(), HasMBLayout());
}

template class PackedIndexNode<float>;
template class PackedIndexNode<double>;

// -----------------------------------------------------------------------
// GatherPackedNode(packedIndex, sourceData) -- gather operation
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    InputRef(INDEXDATA).MaskMissingValueColumnsTo(FrameRange(InputRef(INDEXDATA).GetMBLayout()), -1); // indicates an invalid column to Gather/Scatter
    let&  index  = InputRef(INDEXDATA) .Value(); // column indices to copy from
    let&  source = InputRef(SOURCEDATA).Value(); // source data to copy
    auto& output =                      Value(); // output goes here
    output.DoGatherColumnsOf(/*beta=*/0, index, source, /*alpha=*/1);
}

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::BackpropToNonLooping(size_t inputIndex) /*override*/
{
    if (inputIndex == SOURCEDATA)
    {
        let&  index          = InputRef(INDEXDATA) .Value();    // column indices to copy from
        auto& sourceGradient = InputRef(SOURCEDATA).Gradient(); // source to propagate the gradient intpu
        auto& outputGradient =                      Gradient(); // output gradient to propagate
        sourceGradient.DoScatterColumnsOf(/*beta=*/1, index, outputGradient, /*alpha=*/1);
    }
}

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);

    // inherit MBLayout from indexData
    m_pMBLayout = Input(INDEXDATA)->GetMBLayout();
    if (isFinalValidationPass && (!Input(INDEXDATA)->HasMBLayout()))
        LogicError("%ls requires first argument (index data) to have a time dimension.", NodeDescription().c_str());

    bool sourceHasTimeDimension = Input(SOURCEDATA)->HasMBLayout();

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls requires the first argument (index data) to be a scalar time sequence.", NodeDescription().c_str());

    // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
    if (sourceHasTimeDimension)
        SetDims(Input(SOURCEDATA)->GetSampleLayout(), HasMBLayout());
    else
    {
        SmallVector<size_t> layout = { 1 }; // Scalar
        if (Input(SOURCEDATA)->GetSampleLayout().GetRank() > 1)
        {
            auto srcLayout = Input(SOURCEDATA)->GetSampleLayout().GetDims();
            layout.assign(srcLayout.begin(), srcLayout.end() - 1);
        }
        SetDims(TensorShape(layout), HasMBLayout());
    }
}

template class GatherPackedNode<float>;
template class GatherPackedNode<double>;

// -----------------------------------------------------------------------
// ScatterPackedNode(layoutData, packedIndex, sourceData) -- scatter operation
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    if (*InputRef(INDEXDATA).GetMBLayout() != *InputRef(SOURCEDATA).GetMBLayout())
        InvalidArgument("%ls %ls operation requires the minibatch layout of index and source data to be the same.", NodeName().c_str(), OperationName().c_str());
    InputRef(INDEXDATA).MaskMissingValueColumnsTo(FrameRange(InputRef(INDEXDATA).GetMBLayout()), -1); // indicates an invalid column to Gather/Scatter
    let&  index  = InputRef(INDEXDATA) .Value(); // column indices to copy from
    let&  source = InputRef(SOURCEDATA).Value(); // source data to copy
    auto& output =                      Value(); // output goes here
    output.DoScatterColumnsOf(/*beta=*/0, index, source, /*alpha=*/1);
}

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::BackpropToNonLooping(size_t inputIndex) /*override*/
{
    if (inputIndex == SOURCEDATA)
    {
        let&  index          = InputRef(INDEXDATA).Value();     // column indices to copy from
        auto& sourceGradient = Input(SOURCEDATA)->Gradient(); // source to propagate the gradient input
        auto& outputGradient =                      Gradient(); // output gradient to propagate
        sourceGradient.DoGatherColumnsOf(/*beta=*/1, index, outputGradient, /*alpha=*/1);
    }
}

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);

    // inherit MBLayout from layoutData (that's the only thing we use it for)
    m_pMBLayout = Input(LAYOUTDATA)->GetMBLayout();
    if (isFinalValidationPass && (!Input(LAYOUTDATA)->HasMBLayout() || !Input(INDEXDATA)->HasMBLayout() || !Input(SOURCEDATA)->HasMBLayout()))
        LogicError("%ls %ls operation requires all inputs to be minibatch data (must have MBLayouts).", NodeName().c_str(), OperationName().c_str());

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls %ls operation requires the second argument (indexData) to be a scalar sequence.", NodeName().c_str(), OperationName().c_str());

    // TODO: We also know that indexData and sourceData must have the same MBLayout. But that is checked at runtime.

    // inherit tensor dimension from sourceData
    SetDims(Input(SOURCEDATA)->GetSampleLayout(), HasMBLayout());
}

template class ScatterPackedNode<float>;
template class ScatterPackedNode<double>;

}}}
