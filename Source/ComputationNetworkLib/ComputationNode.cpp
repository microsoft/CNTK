//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "ComputationNetworkBuilder.h" // TODO: We should only pull in NewComputationNodeFromConfig(). Nodes should not know about network at large.
#include "TensorShape.h"

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// -----------------------------------------------------------------------
// subroutines for evaluation
// -----------------------------------------------------------------------

template<class ElemType>
void ComputationNode<ElemType>::Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) /*override*/
{
    // Normally our gradient matrix was created as an input of another node.
    // This does not happen though in the special case of a node inside a loop
    // that no consumer outside depends on. Those might get topologically sorted
    // after nodes that propagate outside of the loop, and thus, in the last
    // time step of the sequence, have not yet received a gradient from a parent
    // and thus may not have had their gradient matrices allocated.
#if 1 // keep enabled once this works
#if 0 // log the cases where this is needed
    if (m_needsGradient && !m_gradientInitialized)
    {
        static size_t c = 0;
        if (c++ < 100)
            fprintf(stderr, "%ls %ls operation: Initializing gradient out of line.\n", NodeName().c_str(), OperationName().c_str());
    }
#endif
    if (m_needsGradient)
        LazyZeroGradient(); // set gradient to 0 if this is the first time
#endif

    if (fr.IsAllFrames() && IsPartOfLoop() && childrenInThisLoop)
        LogicError("%ls %ls operation: Backprop called with whole-batch FrameRange on node that participates in a loop", NodeName().c_str(), OperationName().c_str());

    for (size_t i = 0; i < m_inputs.size(); i++)
    {
        ComputationNodePtr child = Input(i);
        if (child->m_needsGradient &&
            ((childrenInThisLoop  && child->IsPartOfLoop() == IsPartOfLoop()) ||
             (childrenInOuterLoop && child->IsPartOfLoop() != IsPartOfLoop()) ))
        {
            // fprintf(stderr, "Backprop: %ls %ls operation -> child %d %ls %ls\n", NodeName().c_str(), OperationName().c_str(), (int)i, child->NodeName().c_str(), child->OperationName().c_str());
            if (!m_needsGradient)
                LogicError("%ls %ls operation has m_needsGradient set to false but children require it.", NodeName().c_str(), OperationName().c_str());
#if DUMPOUTPUT
            fprintf(stderr, "Backprop%d_%ls\n", i, NodeName().c_str());
#endif
            child->LazyZeroGradient(); // set gradient to 0 if this is the first time

            // If we propagate from a loop to a node that is outside the loop, we are not efficient.
            // This case is handled by SEQTraversalFlowControlNode::Backprop().
            // The check below is to verify that.
            if (IsPartOfLoop() && !child->IsPartOfLoop() && !fr.IsAllFrames())
            {
                LogicError("Backprop: Inefficiency: %ls %ls operation in loop propagates gradient to non-loop %ls %ls\n",
                           NodeName().c_str(), OperationName().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
            }

            // fprintf(stderr, "BackpropTo %d %d %ls %ls\n", (int)fr.timeIdxInSeq, (int)i, NodeName().c_str(), OperationName().c_str());
            BackpropTo(i, fr); // this computes partial wrt to the child and sums the gradient value in the child

            //child->DebugLogMinibatch(/*gradient*/true);
        }
#ifdef DISPLAY_DEBUG
        else
            fprintf(stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, child->OperationName().c_str(), child->NodeName().c_str());
#endif
    }
}

template<class ElemType>
/*static*/ TensorView<ElemType> ComputationNode<ElemType>::Unpack(const TensorShape& sampleShape,
                                                                  const Matrix<ElemType>& packedData,
                                                                  const MBLayoutPtr& layout,
                                                                  const std::shared_ptr<Matrix<ElemType>>& unpackedDataStorage,
                                                                  const std::shared_ptr<Matrix<ElemType>>& tempIndicesStorage,
                                                                  bool batchMajor,
                                                                  bool maskGaps)
{
    size_t maxNumTimeSteps = 1;
    size_t numSequences = 1;
    TensorShape unpackedShape = sampleShape;
    if (layout != nullptr)
    {
        maxNumTimeSteps = layout->GetNumTimeSteps();
        numSequences = layout->GetNumSequences();
        size_t i = unpackedShape.GetRank();
        unpackedShape = unpackedShape.AppendInPlace(i++, batchMajor ? numSequences : maxNumTimeSteps);
        unpackedShape = unpackedShape.AppendInPlace(i++, batchMajor ? maxNumTimeSteps : numSequences);
    }

    std::shared_ptr<Matrix<ElemType>> unpackedData;
    if ((maxNumTimeSteps == 1) || (numSequences == 1) || (batchMajor && (layout->GetNumParallelSequences() == layout->GetNumSequences())))
    {
        unpackedData = std::make_shared<Matrix<ElemType>>(packedData.AsReference());

        if (maskGaps && layout && layout->HasGaps())
        {
            MaskMissingColumnsTo<ElemType>(*unpackedData, layout, FrameRange(layout), 0);
        }
    }
    else
    {
        unpackedData = unpackedDataStorage;
        if (!unpackedData)
            unpackedData = std::make_shared<Matrix<ElemType>>(packedData.GetNumRows(), maxNumTimeSteps * numSequences, packedData.GetDeviceId(), packedData.GetMatrixType(), packedData.GetFormat());
        else
            unpackedData->Resize(packedData.GetNumRows(), maxNumTimeSteps * numSequences);

        if (maskGaps)
            unpackedData->SetValue(0.0f);

        size_t i = 0;
        auto& layoutSequences = layout->GetAllSequences();
        int numLayoutSequences = (int)layoutSequences.size();
        std::vector<ElemType> scatterIndicesVector(layout->GetNumCols(), -1);

        for (int layoutSequenceIdx = 0; layoutSequenceIdx < numLayoutSequences; ++layoutSequenceIdx)
        {
            auto sequenceInfo = layoutSequences[layoutSequenceIdx];
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
            {
                size_t targetParallelStreamIdx = sequenceInfo.s;
                auto currentSequenceBeginIdx = std::max<ptrdiff_t>(0, sequenceInfo.tBegin);
                auto currentSequenceEndIdx = std::min(maxNumTimeSteps, sequenceInfo.tEnd);
                size_t currentSequenceLength = (currentSequenceEndIdx - currentSequenceBeginIdx);

                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    auto targetIdx = (ElemType)(batchMajor ? ((j * numSequences) + i) : ((i * maxNumTimeSteps) + j));
                    scatterIndicesVector[((currentSequenceBeginIdx + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx] = targetIdx;
                }

                i++;
            }
        }

        auto scatterIdxMatrix = tempIndicesStorage;
        if (!scatterIdxMatrix)
            scatterIdxMatrix = std::make_shared<Matrix<ElemType>>(1, layout->GetNumCols(), scatterIndicesVector.data(), packedData.GetDeviceId());
        else
            scatterIdxMatrix->SetValue(1, layout->GetNumCols(), packedData.GetDeviceId(), scatterIndicesVector.data());

        unpackedData->DoScatterColumnsOf(0, *scatterIdxMatrix, packedData, 1);
    }

    return TensorView<ElemType>(unpackedData, unpackedShape);
}

template<class ElemType>
/*static*/ void ComputationNode<ElemType>::BroadcastToPacked(const Matrix<ElemType>& dataToBroadcast,
                                                             const MBLayoutPtr& inputLayout,
                                                             ElemType beta,
                                                             Matrix<ElemType>& broadcastTo,
                                                             const FrameRange& targetFrameRange,
                                                             const std::shared_ptr<Matrix<ElemType>>& tempIndicesStorage)
{
    auto targetLayout = targetFrameRange.m_pMBLayout;
    
    // Generate the gather indices
    std::vector<ElemType> gatherIndicesVector(broadcastTo.GetNumCols(), -1);
    auto& layoutSequences = targetLayout->GetAllSequences();
    int numLayoutSequences = (int)layoutSequences.size();

    // 2-way thread parallelism is sufficient for the memory bound
    // operation of just setting the values of an array.
    const unsigned NUM_THREADS = 2;
    UNUSED(NUM_THREADS); // in case OMP is turned off.
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int layoutSequenceIdx = 0; layoutSequenceIdx < numLayoutSequences; ++layoutSequenceIdx)
    {
        auto sequenceInfo = layoutSequences[layoutSequenceIdx];

        if ((sequenceInfo.seqId != GAP_SEQUENCE_ID) && 
            (targetFrameRange.IsAllFrames() || ((sequenceInfo.tBegin <= (ptrdiff_t)(targetFrameRange.timeIdxInSeq + targetFrameRange.m_timeOffset)) && (sequenceInfo.tEnd > (targetFrameRange.timeIdxInSeq + targetFrameRange.m_timeOffset)))))
        {
            auto srcSequenceInfo = inputLayout->FindSequence(sequenceInfo.seqId);
            auto gatherFromIndex = inputLayout->GetColumnIndex(srcSequenceInfo, 0);
            std::vector<size_t> currentSequenceColumnIndices;
            if (targetFrameRange.IsAllFrames())
                currentSequenceColumnIndices = targetLayout->GetColumnIndices(sequenceInfo);
            else
                currentSequenceColumnIndices.push_back(sequenceInfo.s);

            for (auto i : currentSequenceColumnIndices)
                gatherIndicesVector[i] = (ElemType)gatherFromIndex;
        }
    }

    auto gatherIdxMatrix = tempIndicesStorage;
    if (!gatherIdxMatrix)
        gatherIdxMatrix = std::make_shared<Matrix<ElemType>>(1, broadcastTo.GetNumCols(), gatherIndicesVector.data(), broadcastTo.GetDeviceId());
    else
        gatherIdxMatrix->SetValue(1, broadcastTo.GetNumCols(), broadcastTo.GetDeviceId(), gatherIndicesVector.data());

    broadcastTo.DoGatherColumnsOf(beta, *gatherIdxMatrix, dataToBroadcast, 1);
}

/*static*/ const std::wstring ComputationNodeBase::DefaultDynamicAxisName = L"*";
/*static*/ const std::wstring ComputationNodeBase::DefaultNoSequenceAxisName = L"__noSequenceAxis";


// -----------------------------------------------------------------------
// subroutines for Validate() implementations
// -----------------------------------------------------------------------

// compare two MBLayouts, and alert if they are different
void ComputationNodeBase::ValidateMBLayout(const ComputationNodeBasePtr which, const ComputationNodeBasePtr vsWhich) const
{
    if (!which->HasMBLayout() || !vsWhich->HasMBLayout() || which->GetMBLayout() == vsWhich->GetMBLayout())
        return;
    // MBLayouts are inconsistent
#if 0
    // can't have that
    RuntimeError("%ls: Dynamic axes mismatch between %ls and %ls. If this is by design, use ReconcileDynamicAxis().",
                 NodeDescription().c_str(), which->NodeDescription().c_str(), vsWhich->NodeDescription());
#else
    // We will let this slip with a reminder, assuming that this will be caught at runtime.
    // By allowing this, users will not need ReconcileDynamicAxis() for reductions over a sequence like BS.Sequences.Last().
    if (GetEnvironmentPtr() && (Environment().traceLevel > 0))
    {
        fprintf(stderr, "WARNING: %ls: Dynamic axes mismatch between %ls and %ls. If they are incompatible, this will fail later.\n",
                NodeDescription().c_str(), which->NodeDescription().c_str(), vsWhich->NodeDescription().c_str());
    }
#endif
}

// helper function to infer the MBLayout for this node from inputs, for the *standard case*
// the standard case is:
//  - all inputs must share the same layout (e.g. adding two minibatches)
//  - with the exception of NULL layouts (e.g. TimesNode)
//  - all layouts may be NULL (e.g. W' = W * Exp(Stabilizer))
//  - if there are more than one different layouts involved, this function will fail
void ComputationNodeBase::InferMBLayoutFromInputsForStandardCase(bool isFinalValidationPass)
{
    ComputationNodeBasePtr firstInputWithMBLayout;
    for (auto input : m_inputs)
    {
        if (!input) // node not set yet (DelayedValueNodeBase seems to allow this)--BUGBUG: Then this function won't operate correctly.
            ;
        else if (!input->m_pMBLayout) // NULL layout (typical for parameter nodes)
            ;
        else if (!firstInputWithMBLayout) // first input with layout: remember this child
            firstInputWithMBLayout = input;
        else if (isFinalValidationPass) // got a layout--compare whether it is the same
            ValidateMBLayout(firstInputWithMBLayout, input);
    }
    // all are consistent: install it
    LinkToMBLayout(firstInputWithMBLayout ? firstInputWithMBLayout->m_pMBLayout : nullptr);
}

// single input that maps its input element-wise (e.g. Sigmoid)
void ComputationNodeBase::ValidateUnaryMap(bool isFinalValidationPass)
{
    assert(m_inputs.size() == 1);
    ComputationNodeBase::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);
    SetDims(Input(0));
}

// binary zip operation, e.g. Plus
// If allowBroadcast then one can be a sub-dimension of the other (if layout then only for rows, otherwise for cols, too).
// This also helpfully resizes the children if not yet sized.
void ComputationNodeBase::ValidateBinaryZip(bool isFinalValidationPass, bool allowBroadcast)
{
    assert(m_inputs.size() == 2);
    ComputationNodeBase::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    ValidateInferBinaryInputDims();

    if (isFinalValidationPass)
        ValidateMBLayout(Input(0), Input(1));

    // result has tensor shape with dimensions being the max over both
    let shape0 = GetInputSampleLayout(0);
    let shape1 = GetInputSampleLayout(1);
    SmallVector<size_t> dims = shape0.GetDims();
    if (shape1.GetRank() > dims.size())
        dims.resize(shape1.GetRank(), 1); // pad with ones

    // If rank of [0] is higher than we only need to take max over rank [1].
    // If rank of [1] is higher then we have padded to equal lentgh.
    for (size_t k = 0; k < shape1.GetRank(); k++)
    {
        size_t dim1 = shape1[k];
        // BUGBUG: We must consider the allowBroadcast flag here.
        if (dims[k] <= 1 && dim1 != 0)                     // is [0] broadcasting (1) or unspecified (0)?
            dims[k] = dim1;                                // then use dimension we broadcast to
        else if (dim1 <= 1 && dims[k] != 0)                // if [1] is broadcasting or unspecified
            ;                                              // then dims is already correct
        else if (isFinalValidationPass && dim1 != dims[k]) // no broadcasting or unspecified: they must match
            InvalidArgument("%ls: Input dimensions [%s] and [%s] are not compatible.",
                            NodeDescription().c_str(), string(shape0).c_str(), string(shape1).c_str());
    }

    SetDims(TensorShape(dims), HasMBLayout());
}

// N-nary zip operation, e.g. for TernaryZip for clip()
// If allowBroadcast then one can be a sub-dimension of the other (if layout then only for rows, otherwise for cols, too).
// This also helpfully resizes the children if not yet sized.
void ComputationNodeBase::ValidateNaryZip(bool isFinalValidationPass, bool allowBroadcast, size_t numInputs)
{
    assert(m_inputs.size() == numInputs);
    ComputationNodeBase::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    ValidateInferNaryInputDims(numInputs);

    // check minibatch layout consistency for all possible pairs (n choose 2)
    if (isFinalValidationPass)
        for (size_t i = 0; i < numInputs; i++)
            for (size_t j = i + 1; j < numInputs; j++)
                ValidateMBLayout(Input(i), Input(j));

    // result has tensor shape with dimensions being the max over all inputs
    let shape0 = GetInputSampleLayout(0);

    // dims is max over all inputs
    size_t maxRank = shape0.GetRank();    
    for (size_t i = 1; i < numInputs; i++)
    {
        let shape = GetInputSampleLayout(i);
        if (shape.GetRank() > maxRank)
            maxRank = shape.GetRank();
    }        
    SmallVector<size_t> dims = shape0.GetDims();
    dims.resize(maxRank, 1); // pad with 1

    // first check for invalid dimensions
    for (size_t k = 0; k < maxRank; k++)
    {
        size_t maxDim = 0;
        TensorShape maxShape = shape0; // arbitrary; this is just used for the error message
        for (size_t i = 0; i < numInputs; i++)
        {
            let currentShape = GetInputSampleLayout(i);
            size_t currentRank = currentShape.GetRank();
            // make sure that the rank of this input is bigger than the current index (otherwise, these are implied singleton dimensions that do not need to be checked)
            if (currentRank > k)
            {
                size_t currentDim = currentShape[k];
                if (currentDim > 1 && maxDim != currentDim && maxDim > 1) // 1=broadcasting, 0=not known yet, meant to be inferred
                {
                    InvalidArgument("%ls: Input dimensions [%s] and [%s] are not compatible.",
                        NodeDescription().c_str(), string(maxShape).c_str(), string(currentShape).c_str());
                }
                else if (currentDim > maxDim)
                {
                    maxDim = currentDim;
                    maxShape = currentShape;
                }
            }
        }
    }

    // now set up the right dims
    for (size_t k = 0; k < maxRank; k++)
    {
        for (size_t i = 0; i < numInputs; i++)
        {
            let shape = GetInputSampleLayout(i);

            if (shape.GetRank() > k)
            {
                size_t dim = shape[k];
                if (dims[k] <= 1 && dim != 0)
                    dims[k] = dim;
            }
        }
    }

    SetDims(TensorShape(dims), HasMBLayout());
}

// unary reduce-to-(1,1) operation, e.g. MatrixL1RegNode
void ComputationNodeBase::ValidateUnaryReduce(bool isFinalValidationPass)
{
    assert(m_inputs.size() == 1);
    ComputationNodeBase::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr; // this node does not hold mini-batch data
    SetDims(TensorShape(1), false);
}

// binary reduce-to-(1,1) operation, e.g. CrossEntropyWithSoftmaxNode
// Currently only called by criterion nodes.
// This function also infers child LearnableParameters. In case you wonder why this is needed for criterion nodes, there are edge cases, e.g. a
// learnable parameter being regularized by a criterion node, where the learnable parameter is fed both into that criterion node and other places.
void ComputationNodeBase::ValidateBinaryReduce(bool isFinalValidationPass)
{
    ComputationNodeBase::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr; // this node does not hold mini-batch data
    ValidateInferBinaryInputDims();

    if (isFinalValidationPass)
    {
        if (!(Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(1)->GetSampleLayout())))
        {
            string s1 = Input(0)->GetSampleLayout();
            string s2 = Input(1)->GetSampleLayout();
            // BUGBUG: Allow broadcasting?
            LogicError("%ls: The tensor dimensions in the inputs do not match. %s != %s", NodeDescription().c_str(), s1.c_str(), s2.c_str());
        }
        else if (!(Input(0)->HasMBLayout()))
            LogicError("%ls: Expected MBLayout in Input 0.", NodeDescription().c_str());
        else if (!(Input(1)->HasMBLayout()))
            LogicError("%ls: Expected MBLayout in Input 1.", NodeDescription().c_str());
        // Shape of the MBLayouts is checked at runtime.
    }
    SetDims(TensorShape(1), false);
}

// helper function for validation
// In complex cases of convolution, dimensions are quite difficult for a user to know/derive.
// This is a feature that allows a node to help resizing its input node to the expected value
// iff that input must be a learnable parameter.
void ComputationNodeBase::ValidateInferBinaryInputDims()
{
    // limited inference of children dimensions
    // if dimension not specified we assume two operands' dimensions should be the same
    // NOTE: The assert is set to check if >= 2 since this is called from nodes which have more than two children.
    //      The number of children is formally verified elsewhere, so this will not break consistency.
    assert(m_inputs.size() >= 2);
    for (size_t index = 0; index < 2; index++)
    {
        auto in    = Input(    index);
        auto other = Input(1 - index);
        // borrow any unset dimension on one input from the other input
        in->ValidateInferInputDimsFrom(other->GetSampleLayout());
    }
}

// as above but for N-ary cases
void ComputationNodeBase::ValidateInferNaryInputDims(size_t numInputs)
{
    // limited inference of children dimensions
    // if dimension not specified we assume two operands' dimensions should be the same
    // NOTE: The assert is set to check if >= numInputs since this is called from nodes which have more than 'nInputs' children.
    //      The number of children is formally verified elsewhere, so this will not break consistency.
    assert(m_inputs.size() >= numInputs);
    for (size_t index = 0; index < numInputs; index++)
    {
        const auto& in = Input(index);
        
        for (size_t indexOther = 0; indexOther < numInputs; indexOther++)
        {
            if (indexOther != index) 
            {
                const auto& other = Input(indexOther);
                // borrow any unset dimension on one input from the other input
                in->ValidateInferInputDimsFrom(other->GetSampleLayout());
            }
        }
    }
}

// in case of an error, we just back out, and leave it to outside code to detect errors
template <class ElemType>
void ComputationNode<ElemType>::ValidateInferInputDimsFrom(const TensorShape& otherShape)
{
    // we can only infer learnable parameters at this point
    auto node = dynamic_cast<LearnableParameter<ElemType>*>(this);
    if (node)
        node->InferInputDimsFrom(otherShape);
}

// -----------------------------------------------------------------------
// tensor helpers
// -----------------------------------------------------------------------

// determine the sample tensor dimension to use for operations based on output and all inputs
// 'Sample tensor' means we only consider single samples. If we have an MBLayout, that is the sample layout of a single matrix column.
// TODO: Turn rank into a member variable, and call this method once in validation (currently called for every single ForwardProp/BackpropTo()).
size_t ComputationNodeBase::DetermineElementwiseTensorRank() const
{
    // determine largest tensor dimension amongst the sample shapes of output and the selected inputs
    size_t maxRank = GetSampleLayout().GetRank();
    for (size_t i = 0; i < GetNumInputs(); i++)
    {
        size_t rank = Input(i)->GetSampleLayout().GetRank();
        if (maxRank < rank)
            maxRank = rank;
    }
    return maxRank;
}

// form the actual tensor that describes the full object
TensorShape ComputationNodeBase::GetTensorShape(size_t rank) const
{
    // If we have an MB layout then add the necessary sequence and time axes. If we have none, then absorb the column dimension.
    TensorShape tensorShape = GetSampleLayout(); // TODO: Do we need to expect this tensor to have arbitrary strides? In case it came out of a Slice, Reshape, or Transpose op in-place?
    if (HasMBLayout())
    {
        size_t i = (rank != SIZE_MAX) ? rank : tensorShape.GetRank();
        tensorShape.AppendInPlace(i++, GetMBLayout()->GetNumParallelSequences());
        tensorShape.AppendInPlace(i++, GetMBLayout()->GetNumTimeSteps());
    }
    return tensorShape;
}

// get tensor shape of the slice referenced by a given FrameRange
// Important: This shape does carry offset and stride; it's not just dimensions.
TensorShape ComputationNodeBase::GetTensorSliceFor(size_t rank, const FrameRange& fr) const
{
    // form the actual tensor that describes the full object
    // Note: This may have strides.
    auto tensorShape = GetTensorShape(rank);

    // determine the slice dimensions described by the FrameRange
    // Note: These are dimensions without strides.
    let slice = TensorSliceWithMBLayoutFor(tensorShape.GetDims(), fr, GetMBLayout());

    // narrow the tensor
    // Note: Strides are honored correctly.
    tensorShape.NarrowTo(slice);

    return tensorShape;
}

// same as GetTensorSliceFor() except that 'fr' refers to a single column, and result will not have seq/time axes
// This is needed by TimesNode when the left argument has to be broken up into individual matrices/GEMM calls.
// To enable its first argument to have an MBLayout, it needs to un-pad if we have an MBLayout but only refer to a single sequence and time step.
TensorShape ComputationNodeBase::GetOneSampleTensorSliceFor(size_t rank, const FrameRange& fr) const
{
    TensorShape result = GetTensorSliceFor(rank, fr);
    // undo the adding of (seq, time) axes that was done by GetTensorShape()
    if (!fr.IsOneColumnWrt(GetMBLayout()))
        LogicError("GetOneSampleTensorSliceFor: Requires 'fr' to refer to a single sample.");
    if (HasMBLayout())
        result.TrimRankInPlace(rank); // Note: This function will verify once again that the extra dimensions have been reduced to [1 x 1]
    return result;
}

// -----------------------------------------------------------------------
// others
// -----------------------------------------------------------------------

/*virtual*/ string ComputationNodeBase::FormatOperationPrototype(const string& extraArgs) const
{
    string prototype;
    prototype += msra::strfun::strprintf("%ls = %ls", NodeName().c_str(), OperationName().c_str());

    // arguments of operation
    if (IsLeaf())
        prototype += "()";
    else
    {
        prototype += " (";
        for (size_t i = 0; i < GetNumInputs(); i++)
        {
            const auto& child = m_inputs[i];
            if (i > 0)
                prototype += ", ";

            if (child)
                prototype += msra::strfun::strprintf("%ls", child->NodeName().c_str());
            else
                prototype += "NULL";
        }
        prototype += extraArgs;
        prototype += ")";
    }

    // type (tensor dimensions) of operation
    prototype += " : ";

    if (!IsLeaf())
    {
        //prototype += "(";
        for (size_t i = 0; i < GetNumInputs(); i++)
        {
            const auto& child = m_inputs[i];
            if (i > 0)
                prototype += ", ";

            if (child == nullptr)
            {
                prototype += "NULL";
                continue;
            }
            prototype += child->ShapeDescription().c_str();
        }
        prototype += extraArgs;
        //prototype += ")";
    }

    prototype += msra::strfun::strprintf(" -> %s", ShapeDescription().c_str());

    return prototype;
}

const std::string ComputationNodeBase::ShapeDescription() const
{
    return msra::strfun::strprintf("[%s%s%ls]",
        string(m_sampleLayout).c_str(),
        HasMBLayout() ? " x " : "",
        HasMBLayout() ? GetMBLayout()->GetAxisName() : L"");
}

template <class ElemType>
/*virtual*/ void ComputationNode<ElemType>::BeginForwardProp()
{
    Base::BeginForwardProp();

    // update the actual m_value allocation
    if (!IsLeaf() && !RequiresPreCompute()) // TODO: guard this through overrides instead
        UpdateFunctionValuesSize();

    // give nodes a chance to update their internal state that may also have to match MB size
    UpdateFunctionMBSize();

    // and make sure dimensions are what we expect
    VerifyDataSize(Value());
}

template <class ElemType>
/*virtual*/ void ComputationNode<ElemType>::EndForwardProp()
{
    Base::EndForwardProp();

    if (HasEnvironmentPtr() && Environment().trackGapNans)
    {
        MaskMissingValueColumnsToZero(FrameRange(m_pMBLayout)); // HasNaN() operates on a whole matrix, so first flatten all gaps to 0
        if (Value().HasNan("EndForwardProp"))
        {
            LogicError("%ls %ls operation unexpectedly produced NaN values.", NodeName().c_str(), OperationName().c_str());
        }
        InvalidateMissingValueColumns(FrameRange(m_pMBLayout)); // blast NaNs into columns that are gaps in a packed layout
    }

    // tracing
    Trace();
}

template <class ElemType>
/*virtual*/ void ComputationNode<ElemType>::BeginBackprop()
{
    Base::BeginBackprop();

    if (NeedsGradient())
    {
        // Verify that the shapes of the output/input Value matrices that the gradient backprop for this node needs
        // are intact and have not been erroneously reshaped due to incorrect memory sharing
        auto VerifyValueShape = [](const ComputationNode<ElemType>& node) {
            size_t rows, cols;
            node.DetermineDataSize(rows, cols);

            auto& valueMatrix = node.Value();
            if ((valueMatrix.GetNumRows() != rows) || (valueMatrix.GetNumCols() != cols))
            {
                LogicError("%ls %ls operation found to have incorrect Value() matrix shape %lu x %lu during backprop; expected shape is %lu x %lu. "
                    "This may be due to incorrect memory sharing.",
                    node.NodeName().c_str(), node.OperationName().c_str(), valueMatrix.GetNumRows(), valueMatrix.GetNumCols(), rows, cols);
            }
        };

        if (IsOutputNeededDuringBackprop())
            VerifyValueShape(*this);

        for (size_t i = 0; i < m_inputs.size(); i++)
        {
            if (InputUsedInComputingInputNodesGradients(i))
                VerifyValueShape(InputRef(i));
        }
    }
}

template <class ElemType>
/*virtual*/ void ComputationNode<ElemType>::EndBackprop()
{
    Base::EndBackprop();

    if (HasEnvironmentPtr() && Environment().trackGapNans)
    {
        for (size_t i = 0; i < m_inputs.size(); i++)
        {
            ComputationNodePtr child = Input(i);
            if (child->m_needsGradient)
            {
                child->MaskMissingGradientColumnsToZero(FrameRange(child->GetMBLayout())); // HasNaN() operates on a whole matrix, so first flatten all gaps to 0
                if (child->Gradient().HasNan("EndBackprop"))
                {
                    LogicError("%ls %ls operation unexpectedly produced NaN gradients.", child->NodeName().c_str(), child->OperationName().c_str());
                }
            }
        }
    }
}

template <class ElemType>
/*virtual*/ void ComputationNode<ElemType>::DumpNodeInfo(const bool /*printValues*/, const bool printMetadata, File& fstream) const
{
    if (printMetadata)
    {
        fstream << L"\n" + NodeName() + L"=" + OperationName();

        if (!IsLeaf())
        {
            fstream << wstring(L"(");
            for (size_t i = 0; i < GetNumInputs(); i++)
            {
                if (i > 0)
                    fstream << wstring(L",");
                fstream << (Input(i) ? Input(i)->NodeName() : L"NULL");
            }
            fstream << wstring(L")");
        }
    }
}

// write out the content of a node in formatted/readable form
// 'transpose' means print one row per sample (non-transposed is one column per sample).
// 'isSparse' will print all non-zero values as one row (non-transposed, which makes sense for one-hot) or column (transposed).
template <class ElemType>
void ComputationNode<ElemType>::WriteMinibatchWithFormatting(FILE* f,
                                                             const FrameRange& fr,
                                                             size_t onlyUpToRow, size_t onlyUpToT, bool transpose, bool isCategoryLabel, bool isSparse,
                                                             const vector<string>& labelMapping, const string& sequenceSeparator, 
                                                             const string& sequencePrologue, const string& sequenceEpilogue,
                                                             const string& elementSeparator, const string& sampleSeparator,
                                                             string valueFormatString,
                                                             bool outputGradient,
                                                             bool onlyShowAbsSumForDense,
                                                             std::function<std::string(size_t)> getKeyById) const
{
    // get minibatch matrix -> matData, matRows, matStride
    const Matrix<ElemType>& outputValues = outputGradient ? Gradient() : Value();
    let matRows   = outputValues.GetNumRows();
    let matStride = matRows; // how to get from one column to the next
    unique_ptr<ElemType[]> matDataPtr(outputValues.CopyToArray());
    ElemType* matData = matDataPtr.get();
    let sampleLayout = GetSampleLayout(); // this is currently only used for sparse; dense tensors are linearized

    // process all sequences one by one
    MBLayoutPtr pMBLayout = GetMBLayout();
    if (!pMBLayout) // no MBLayout: We are printing aggregates (or LearnableParameters?)
    {
        pMBLayout = make_shared<MBLayout>();
        pMBLayout->Init(1, outputValues.GetNumCols()); // treat this as if we have one single sequence consisting of the columns
        pMBLayout->AddSequence(0, 0, 0, outputValues.GetNumCols());
    }
    let& sequences = pMBLayout->GetAllSequences();
    let  width     = pMBLayout->GetNumTimeSteps();

    TensorShape tensorShape = GetSampleLayout();
    stringstream str;
    let dims = tensorShape.GetDims();
    for (auto dim : dims)
        str << dim << ' ';
    let shape = str.str(); // BUGBUG: change to string(tensorShape) to make sure we always use the same format

    bool sequencePrologueHasShape = sequencePrologue.find("%x") != sequencePrologue.npos;
    bool sampleSeparatorHasShape  = sampleSeparator.find("%x")  != sampleSeparator.npos;
    bool sequencePrologueHasSeqId = sequencePrologue.find("%d") != sequencePrologue.npos;
    bool sampleSeparatorHasSeqId  = sampleSeparator.find("%d")  != sampleSeparator.npos;

    for (size_t s = 0; s < sequences.size(); s++)
    {
        const auto& seqInfo = sequences[s];
        if (seqInfo.seqId == GAP_SEQUENCE_ID) // nothing in gaps to print
            continue;
        let tBegin = seqInfo.tBegin >= 0     ? seqInfo.tBegin : 0;
        let tEnd   = seqInfo.tEnd   <= width ? seqInfo.tEnd   : width;
        // [tBegin,tEnd) is where the sequence resides.
        // fr is also referencing where a sequence resides.

        // narrow to FrameRange if needed
        auto t0 = fr.IsAllFrames() ? tBegin : fr.m_timeOffset + (ptrdiff_t)fr.timeIdxInSeq;
        auto t1 = fr.IsAllFrames() ? tEnd   : fr.m_timeOffset + (ptrdiff_t)fr.timeIdxInSeq + (ptrdiff_t)fr.m_timeRange;
        if (t0 < tBegin)
            t0 = tBegin;
        if (t1 > tEnd)
            t1 = tEnd;
        // [t0,t1) is the range we want to print
        if (t0 > (ptrdiff_t)t1)
            continue; // skip this sequence

        // get sequence matrix -> seqData, seqRows, seqCols, seqStride
        let  seqData   = matData + pMBLayout->GetColumnIndex(seqInfo, t0 - tBegin) * matStride;
        auto seqRows   = matRows;
        let  seqCols   = t1 - t0;
        let  seqStride = pMBLayout->GetNumParallelSequences() * matStride;

        auto seqProl = sequencePrologue;
        auto sampleSep = sampleSeparator;

        if (sequencePrologueHasShape || sampleSeparatorHasShape)
        {
            auto sh = msra::strfun::_strprintf<char>("%s%ld", shape.c_str(), (unsigned long long)seqInfo.GetNumTimeSteps());
            if (sequencePrologueHasShape)
                seqProl = msra::strfun::ReplaceAll<std::string>(seqProl, "%x", sh);
            if (sampleSeparatorHasShape)
                sampleSep = msra::strfun::ReplaceAll<std::string>(sampleSep, "%x", sh);
        }

        if (sequencePrologueHasSeqId || sampleSeparatorHasSeqId)
        {
            auto sh = msra::strfun::_strprintf<char>("%ld", (unsigned long long)seqInfo.seqId);
            if (sequencePrologueHasSeqId)
                seqProl = msra::strfun::ReplaceAll<std::string>(seqProl, "%d", sh);
            if (sampleSeparatorHasSeqId)
                sampleSep = msra::strfun::ReplaceAll<std::string>(sampleSep, "%d", sh);
        }

        if (s > 0)
            fprintfOrDie(f, "%s", sequenceSeparator.c_str());
        if (getKeyById)
            fprintfOrDie(f, "%s ", getKeyById(seqInfo.seqId).c_str());
        fprintfOrDie(f, "%s", seqProl.c_str());

        // output it according to our format specification
        auto formatChar = valueFormatString.back();
        if (isCategoryLabel) // if is category then find the max value and output its index (possibly mapped to a string)
        {
            if (formatChar == 's') // verify label dimension
            {
                if (outputValues.GetNumRows() != labelMapping.size() &&
                    sampleLayout[0] != labelMapping.size()) // if we match the first dim then use that
                {
                    static size_t warnings = 0;
                    if (warnings++ < 5)
                        fprintf(stderr, "write: Row dimension %d does not match number of entries %d in labelMappingFile, not using mapping\n", (int)seqRows, (int)labelMapping.size());
                    valueFormatString.back() = 'u'; // this is a fallback
                    formatChar = valueFormatString.back();
                }
            }
            // update the matrix in-place from one-hot (or max) to index
            // find the max in each column
            for (size_t j = 0; j < seqCols; j++) // loop over all time steps of the sequence
            {
                double maxLoc = -1;
                double maxVal = 0;
                for (size_t i = 0; i < seqRows; i++) // loop over rows
                {
                    let val = seqData[i + j * seqStride];
                    if (maxLoc < 0 || val >= maxVal)
                    {
                        maxLoc = (double)i;
                        maxVal = val;
                    }
                }
                seqData[0 + j * seqStride] = (ElemType)maxLoc; // overwrite first element in-place
            }
            seqRows = 1; // ignore remaining dimensions
        }
        // function to print a value
        auto print = [&](double dval)
        {
            if (formatChar == 'f') // print as real number
            {
                if (dval == 0) dval = fabs(dval);    // clear the sign of a negative 0, which are produced inconsistently between CPU and GPU
                fprintfOrDie(f, valueFormatString.c_str(), dval);
            }
            else if (formatChar == 'u') // print category as integer index
            {
                fprintfOrDie(f, valueFormatString.c_str(), (unsigned int)dval);
            }
            else if (formatChar == 's') // print category as a label string
            {
                size_t uval = (size_t)dval;
                if (!labelMapping.empty())
                    uval %= labelMapping.size();
                assert(uval < labelMapping.size());
                const char * sval = labelMapping[uval].c_str();
                fprintfOrDie(f, valueFormatString.c_str(), sval);
            }
        };
        // bounds for printing
        let iend    = transpose ?     seqRows : seqCols;     // true dimension of the data to print
        let jend    = transpose ?     seqCols : seqRows;
        let istop   = transpose ? onlyUpToRow : onlyUpToT;   // we stop at these dimensions (for debugging, one often needs only the first few values of those huge matrices)
        let jstop   = transpose ?   onlyUpToT : onlyUpToRow;
        let istride = transpose ?           1 : seqStride;
        let jstride = transpose ?   seqStride : 1;
        if (isSparse)
        {
            // sparse linearizes the entire matrix into a single vector, and prints that one with coordinates
            // TODO: This can be done more nicely. We should keep the block structure.
            size_t numPrinted = 0;
            for (size_t i = 0; i < iend; i++) // loop over elements --we just flatten them all out
            {
                for (size_t j = 0; j < jend; j++) // loop over rows
                {
                    double dval = seqData[i * istride + j * jstride];
                    if (dval == 0) // only print non-0 values
                        continue;
                    if (numPrinted++ > 0)
                        fprintfOrDie(f, "%s", transpose ? sampleSeparator.c_str() : elementSeparator.c_str());
                    if (dval != 1.0 || formatChar != 'f') // hack: we assume that we are either one-hot or never precisely hitting 1.0
                        print(dval);
                    size_t row = transpose ? i : j;
                    size_t col = transpose ? j : i;
                    for (size_t k = 0; k < sampleLayout.size(); k++)
                    {
                        fprintfOrDie(f, "%c%d", k == 0 ? '[' : ',', row % sampleLayout[k]);
                        if (sampleLayout[k] == labelMapping.size()) // annotate index with label if dimensions match (which may misfire once in a while)
                            fprintfOrDie(f, "=%s", labelMapping[row % sampleLayout[k]].c_str());
                        row /= sampleLayout[k];
                    }
                    if (seqInfo.GetNumTimeSteps() > 1)
                        fprintfOrDie(f, ";%d", col);
                    fprintfOrDie(f, "]");
                }
            }
        }
        else
        {
            if (onlyShowAbsSumForDense)
            {
                // the concise version to make matrix comparision easier
                double absSum = 0;
                
                #pragma omp parallel for reduction(+:absSum)
                for (int i = 0; i < (int)iend; i++) // loop over output rows
                {
                    double absSumLocal = 0;
                    for (size_t j = 0; j < jend; j++) // loop over elements
                    {
                        absSumLocal += abs(seqData[i * istride + j * jstride]);
                    }
                    absSum += absSumLocal;
                }
                fprintfOrDie(f, "absSum: %f", absSum);
            }
            else
            {
                for (size_t j = 0; j < jend; j++) // loop over output rows     --BUGBUG: row index is 'i'!! Rename these!!
                {
                    if (j > 0)
                        fprintfOrDie(f, "%s", sampleSep.c_str());
                    if (j == jstop && jstop < jend - 1) // if jstop == jend-1 we may as well just print the value instead of '...'
                    {
                        fprintfOrDie(f, "...+%d", (int)(jend - jstop)); // 'nuff said
                        break;
                    }
                    // inject sample tensor index if we are printing row-wise and it's a tensor
                    if (!transpose && sampleLayout.size() > 1 && !isCategoryLabel) // each row is a different sample dimension
                    {
                        for (size_t k = 0; k < sampleLayout.size(); k++)
                            fprintfOrDie(f, "%c%d", k == 0 ? '[' : ',', (int)((j / sampleLayout.GetStrides()[k])) % sampleLayout[k]);
                        fprintfOrDie(f, "]\t");
                    }
                    // print a row of values
                    for (size_t i = 0; i < iend; i++) // loop over elements
                    {
                        if (i > 0)
                            fprintfOrDie(f, "%s", elementSeparator.c_str());
                        if (i == istop && istop < iend - 1)
                        {
                            fprintfOrDie(f, "...+%d", (int)(iend - istop));
                            break;
                        }
                        double dval = seqData[i * istride + j * jstride];
                        print(dval);
                    }
                }
            }
        }
        fprintfOrDie(f, "%s", sequenceEpilogue.c_str());
    } // end loop over sequences
    fflushOrDie(f);
}

/*static*/ string WriteFormattingOptions::Processed(const wstring& nodeName, string fragment, size_t minibatchId)
{
    fragment = msra::strfun::ReplaceAll<string>(fragment, "\\n", "\n");
    fragment = msra::strfun::ReplaceAll<string>(fragment, "\\r", "\r");
    fragment = msra::strfun::ReplaceAll<string>(fragment, "\\t", "\t");
    fragment = msra::strfun::ReplaceAll<string>(fragment, "\\s", " "); // Config might strip spaces.
    if (fragment.find("%s") != fragment.npos)
        fragment = msra::strfun::ReplaceAll<string>(fragment, "%s", msra::strfun::utf8(nodeName));
    if (fragment.find("%n") != fragment.npos)
        fragment = msra::strfun::ReplaceAll<string>(fragment, "%n", msra::strfun::_strprintf<char>("%ld", minibatchId).c_str());
    // %d: sequenceId
    return fragment;
}

template <class ConfigRecordType>
WriteFormattingOptions::WriteFormattingOptions(const ConfigRecordType& config) :
    WriteFormattingOptions()
{
    // gather additional formatting options
    if (config.Exists(L"format"))
    {
        const ConfigRecordType& formatConfig(config(L"format", ConfigRecordType::Record()));
        if (formatConfig.ExistsCurrent(L"type")) // do not inherit 'type' from outer block
        {
            wstring type = formatConfig(L"type");
            if      (type == L"real")     ; // default
            else if (type == L"category") isCategoryLabel = true;
            else if (type == L"sparse")   isSparse = true;
            else                         InvalidArgument("write: type must be 'real', 'category', or 'sparse'");
            labelMappingFile = (wstring)formatConfig(L"labelMappingFile", L"");
        }
        transpose = formatConfig(L"transpose", transpose);
        prologue  = formatConfig(L"prologue",  prologue);
        epilogue  = formatConfig(L"epilogue",  epilogue);
        sequenceSeparator = msra::strfun::utf8(formatConfig(L"sequenceSeparator", (wstring)msra::strfun::utf16(sequenceSeparator)));
        sequencePrologue  = msra::strfun::utf8(formatConfig(L"sequencePrologue",  (wstring)msra::strfun::utf16(sequencePrologue)));
        sequenceEpilogue  = msra::strfun::utf8(formatConfig(L"sequenceEpilogue",  (wstring)msra::strfun::utf16(sequenceEpilogue)));
        elementSeparator  = msra::strfun::utf8(formatConfig(L"elementSeparator",  (wstring)msra::strfun::utf16(elementSeparator)));
        sampleSeparator   = msra::strfun::utf8(formatConfig(L"sampleSeparator",   (wstring)msra::strfun::utf16(sampleSeparator)));
        precisionFormat   = msra::strfun::utf8(formatConfig(L"precisionFormat",   (wstring)msra::strfun::utf16(precisionFormat)));
        // TODO: change those strings into wstrings to avoid this conversion mess
    }
}

void WriteFormattingOptions::Save(File& fstream) const
{
    fstream << isCategoryLabel;
    fstream << labelMappingFile;
    fstream << isSparse;
    fstream << transpose;
    fstream << prologue;
    fstream << epilogue;
    fstream << sequenceSeparator;
    fstream << sequencePrologue;
    fstream << sequenceEpilogue;
    fstream << elementSeparator;
    fstream << sampleSeparator;
    fstream << precisionFormat;
}

void WriteFormattingOptions::Load(File& fstream, size_t modelVersion)
{
    fstream >> isCategoryLabel;
    fstream >> labelMappingFile;
    fstream >> isSparse;
    fstream >> transpose;
    fstream >> prologue;
    fstream >> epilogue;
    fstream >> sequenceSeparator;
    fstream >> sequencePrologue;
    fstream >> sequenceEpilogue;
    fstream >> elementSeparator;
    fstream >> sampleSeparator;
    fstream >> precisionFormat;
}

template WriteFormattingOptions::WriteFormattingOptions(const ConfigParameters&);
template WriteFormattingOptions::WriteFormattingOptions(const ScriptableObjects::IConfigRecord&);

// -----------------------------------------------------------------------
// static variables
// -----------------------------------------------------------------------

atomic_ullong TimeStamp::s_timeStampCounter = ATOMIC_VAR_INIT(0);

template <> map<size_t, map<size_t, shared_ptr<SingleMatrix>>> ComputationNode<float>::s_constOnes{};
template <> map<size_t, map<size_t, shared_ptr<DoubleMatrix>>> ComputationNode<double>::s_constOnes{};

// -----------------------------------------------------------------------
// instantiate the core class templates
// -----------------------------------------------------------------------

template class ComputationNode<float>;
template class ComputationNode<double>;

}}}

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

using namespace Microsoft::MSR::CNTK;

// -----------------------------------------------------------------------
// register ComputationNode with the ScriptableObject system
// -----------------------------------------------------------------------

template <>
shared_ptr<Object> MakeRuntimeObject<ComputationNodeBase>(const IConfigRecordPtr configp)
{
    let node = NewComputationNodeFromConfig(configp);
    // temporarily disabling this, as it caused a test to fail:
    //if (!node->Is<IRecurrentNode>())
    //    node->Validate(/*isFinalValidationPass*/false); // do an initial validation, so that we have access to dimensions
    return node;
}

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNodeBase> registerComputationNode(L"ComputationNode");

// -----------------------------------------------------------------------
// register a boxed version of TensorShape with the ScriptableObject system
// -----------------------------------------------------------------------

// e.g.
// new TensorShape [ dims = 13:42 ]
class BoxedTensorShape : public BoxOf<TensorShape>
{
public:
    BoxedTensorShape(const IConfigRecordPtr configp) :
        BoxOf<TensorShape>(TensorShape(ConfigArray::FlattenedVectorFrom<size_t>(configp->Get(L"dims"))))
    {
    }
};

template <typename E>
class BoxedVector : public BoxOf<vector<E>>
{
public:
    BoxedVector(const IConfigRecordPtr configp) :
        BoxOf<vector<E>>(ConfigArray::FlattenedVectorFrom<E>(configp->Get(L"items")))
    {
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedTensorShape>    registerTensorShape(L"TensorShape");
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedVector<int>>    registerIntVector  (L"IntVector");
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedVector<size_t>> registerSizeVector (L"SizeVector");
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedVector<bool>>   registerBoolVector (L"BoolVector");

}}}
