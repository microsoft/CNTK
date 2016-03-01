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
    //if (m_needsGradient)
    //    LazyZeroGradient(); // set gradient to 0 if this is the first time

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
        }
#ifdef DISPLAY_DEBUG
        else
            fprintf(stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, child->OperationName().c_str(), child->NodeName().c_str());
#endif
    }
}

// -----------------------------------------------------------------------
// subroutines for Validate() implementations
// -----------------------------------------------------------------------

// helper function to infer the MBLayout for this node from inputs, for the *standard case*
// the standard case is:
//  - all inputs must share the same layout (e.g. adding two minibatches)
//  - with the exception of NULL layouts (e.g. TimesNode)
//  - all layouts may be NULL (e.g. W' = W * Exp(Stabilizer))
//  - if there are more than one different layouts involved, this function will fail
void ComputationNodeBase::InferMBLayoutFromInputsForStandardCase()
{
    MBLayoutPtr pMBLayout; // start with NULL layout
    for (auto child : m_inputs)
    {
        if (!child) // node not set yet (DelayedValueNodeBase seems to allow this)--BUGBUG: Then this function won't operate correctly.
            ;
        else if (!child->m_pMBLayout) // NULL layout (typical for parameter nodes)
            ;
        else if (!pMBLayout) // first non-NULL layout: just copy it
            pMBLayout = child->m_pMBLayout;
        else if (pMBLayout != child->m_pMBLayout) // got a layout--compare whether it is the same
            RuntimeError("%ls: InferMBLayoutFromInputsForStandardCase: Expected minibatch layouts to be the same between all children. Child '%ls' (%ls) uses a different layout than previously checked children and might get out of sync during runtime. If this is by design, use ReconcileMBLayout() to forward layouts between nodes.",
                         NodeDescription().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
    }
    // all are consistent: install it
    LinkToMBLayout(pMBLayout);
}

// single input that maps its input element-wise (e.g. Sigmoid)
void ComputationNodeBase::ValidateUnaryMap(bool isFinalValidationPass)
{
    assert(m_inputs.size() == 1);
    ComputationNodeBase::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase();
    SetDims(Input(0));
}

// binary zip operation, e.g. Plus
// If allowBroadcast then one can be a sub-dimension of the other (if layout then only for rows, otherwise for cols, too).
// This also helpfully resizes the children if not yet sized.
void ComputationNodeBase::ValidateBinaryZip(bool isFinalValidationPass, bool allowBroadcast)
{
    assert(m_inputs.size() == 2);
    ComputationNodeBase::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase();

    ValidateInferBinaryInputDims();

    if (isFinalValidationPass &&
        Input(0)->GetMBLayout() != Input(1)->GetMBLayout() && Input(0)->HasMBLayout() && Input(1)->HasMBLayout())
    {
        LogicError("%ls: Minibatch layouts are not the same between arguments and might get out of sync during runtime. If this is by design, use ReconcileMBLayout() to forward layouts between nodes.", NodeDescription().c_str());
    }

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
        if (dims[k] == 1)                                  // is [0] broadcasting?
            dims[k] = dim1;                                // then use dimension we broadcast to
        else if (dim1 == 1)                                // if [1] is broadcasting
            ;                                              // dims is already correct
        else if (isFinalValidationPass && dim1 != dims[k]) // no broadcasting: they must match
            InvalidArgument("%ls: Input dimensions [%s] and [%s] are not compatible.",
                            NodeDescription().c_str(), string(shape0).c_str(), string(shape1).c_str());
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
            std::string s1 = Input(0)->GetSampleLayout();
            std::string s2 = Input(1)->GetSampleLayout();
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
        auto in = Input(index);
        auto other = Input(1 - index);
        // borrow any unset dimension on one input from the other input
        in->ValidateInferInputDimsFrom(other->GetSampleLayout());
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
    // If we have an MB layout then add the necessary dimensions. If we have none, then absorb the column dimension.
    TensorShape tensorShape = GetSampleLayout(); // TODO: Can this tensor have arbitrary strides? In case it came out of a Slice, Reshape, or Transpose op in-place
    if (HasMBLayout())
    {
        size_t i = rank;
        tensorShape.AppendInPlace(i++, GetMBLayout()->GetNumParallelSequences());
        tensorShape.AppendInPlace(i++, GetMBLayout()->GetNumTimeSteps());
    }
    return tensorShape;
}

// get tensor shape of the slice referenced by a given FrameRange
TensorShape ComputationNodeBase::GetTensorSliceFor(size_t rank, const FrameRange& fr) const
{
    // form the actual tensor that describes the full object
    // Note: This may have strides.
    auto tensorShape = GetTensorShape(rank);

    // determine the slice dimensions described by the FrameRange
    // Note: These are dimensions without strides.
    auto slice = TensorSliceWithMBLayoutFor(tensorShape.GetDims(), fr, GetMBLayout());

    // narrow the tensor
    // Note: Strides are honored correctly.
    tensorShape.NarrowTo(slice);
    return tensorShape;
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

            const char* mbSizeMark = child->m_pMBLayout ? " x *" : "";
#if 0
            if (child->m_sampleLayout.GetRank() == 3 && (child->m_sampleLayout[1] != 1 || child->m_sampleLayout[0] != 1)) // looks like an image: use WHC notation
                prototype += msra::strfun::strprintf("%ls[%s%s {W=%lu, H=%lu, C=%lu}]", child->NodeName().c_str(), string(child->m_sampleLayout).c_str(), mbSizeMark,
                child->m_sampleLayout[1], child->m_sampleLayout[2], child->m_sampleLayout[0]);
            // BUGBUG: This ^^ will print based on the old legacy layout, and we have no way of knowing here whether that is correct.
            else
#endif
                prototype += msra::strfun::strprintf("[%s%s]", string(child->m_sampleLayout).c_str(), mbSizeMark);
        }
        prototype += extraArgs;
        //prototype += ")";
    }

    prototype += msra::strfun::strprintf(" -> [%s%s]", string(GetSampleLayout()).c_str(), HasMBLayout() ? " x *" : "");

    return prototype;
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
template <class ElemType>
void ComputationNode<ElemType>::WriteMinibatchWithFormatting(FILE* f, size_t onlyUpToRow, size_t onlyUpToT, bool transpose, bool isCategoryLabel, 
                                                             const std::vector<std::string>& labelMapping, const string& sequenceSeparator, 
                                                             const string& sequencePrologue, const string& sequenceEpilogue,
                                                             const string& elementSeparator, const string& sampleSeparator,
                                                             const string& valueFormatString,
                                                             bool outputGradient) const
{
    // get minibatch matrix -> matData, matRows, matStride
    const Matrix<ElemType>& outputValues = outputGradient ? Gradient() : Value();
    let matRows   = outputValues.GetNumRows();
    let matStride = matRows; // how to get from one column to the next
    std::unique_ptr<ElemType[]> matDataPtr(outputValues.CopyToArray());
    ElemType* matData = matDataPtr.get();

    // process all sequences one by one
    MBLayoutPtr pMBLayout = GetMBLayout();
    if (!pMBLayout) // no MBLayout: We are printing aggregates (or LearnableParameters?)
    {
        pMBLayout = make_shared<MBLayout>();
        pMBLayout->InitAsFrameMode(1); // treat this as if we have one single sample
        // TODO: This can be done more efficiently, if ever needed.
    }
    let& sequences = pMBLayout->GetAllSequences();
    let  width     = pMBLayout->GetNumTimeSteps();
    for (size_t s = 0; s < sequences.size(); s++)
    {
        const auto& seqInfo = sequences[s];
        if (seqInfo.seqId == GAP_SEQUENCE_ID) // nothing in gaps to print
            continue;
        let tBegin = seqInfo.tBegin >= 0     ? seqInfo.tBegin : 0;
        let tEnd   = seqInfo.tEnd   <= width ? seqInfo.tEnd   : width;

        // get sequence matrix -> seqData, seqRows, seqCols, seqStride
        let  seqData   = matData + pMBLayout->GetColumnIndex(seqInfo, 0) * matStride;
        auto seqRows   = matRows;
        let  seqCols   = tEnd - tBegin;
        let  seqStride = pMBLayout->GetNumParallelSequences() * matStride;

        if (s > 0)
            fprintfOrDie(f, "%s", sequenceSeparator.c_str());
        fprintfOrDie(f, "%s", sequencePrologue.c_str());

        // output it according to our format specification
        let formatChar = valueFormatString.back();
        if (isCategoryLabel) // if is category then find the max value and output its index (possibly mapped to a string)
        {
            if (formatChar == 's') // verify label dimension
            {
                if (outputValues.GetNumRows() != labelMapping.size())
                    InvalidArgument("write: Row dimension %d does not match number of entries %d in labelMappingFile", (int)seqRows, (int)labelMapping.size());
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
        // bounds for printing
        let iend    = transpose ?     seqRows : seqCols;         // true dimension of the data to print
        let jend    = transpose ?     seqCols : seqRows;
        let istop   = transpose ? onlyUpToRow : onlyUpToT; // we stop at these dimensions (for debugging, one often needs only the first few values of those huge matrices)
        let jstop   = transpose ?   onlyUpToT : onlyUpToRow;
        let istride = transpose ?           1 : seqStride;
        let jstride = transpose ?   seqStride : 1;
        for (size_t j = 0; j < jend; j++)
        {
            if (j > 0)
                fprintfOrDie(f, "%s", sampleSeparator.c_str());
            if (j == jstop)
            {
                fprintf(f, "...+%d", (int)(jend - jstop)); // 'nuff said
                break;
            }
            for (size_t i = 0; i < iend; i++)
            {
                if (i > 0)
                    fprintfOrDie(f, "%s", elementSeparator.c_str());
                if (i == istop)
                {
                    fprintf(f, "...+%d", (int)(iend - istop));
                    break;
                }
                double dval = seqData[i * istride + j * jstride];
                if (formatChar == 'f') // print as real number
                {
                    fprintfOrDie(f, valueFormatString.c_str(), dval);
                }
                else if (formatChar == 'u') // print category as integer index
                {
                    fprintfOrDie(f, valueFormatString.c_str(), (unsigned int)dval);
                }
                else if (formatChar == 's') // print category as a label string
                {
                    size_t uval = (size_t)dval;
                    assert(uval < labelMapping.size());
                    const char * sval = labelMapping[uval].c_str();
                    fprintfOrDie(f, valueFormatString.c_str(), sval);
                }
            }
        }
        fprintfOrDie(f, "%s", sequenceEpilogue.c_str());
    } // end loop over sequences
    fflushOrDie(f);
}

// -----------------------------------------------------------------------
// instantiate the core class templates
// -----------------------------------------------------------------------

typedef Matrix<float> FloatMatrix;
typedef Matrix<double> DoubleMatrix;

atomic_ullong TimeStamp::s_timeStampCounter = ATOMIC_VAR_INIT(0);

template <>
std::map<size_t, std::map<size_t, FloatMatrix*>> ComputationNode<float>::s_constOnes{};
template <>
std::map<size_t, std::map<size_t, DoubleMatrix*>> ComputationNode<double>::s_constOnes{};

template class ComputationNode<float>;
template class ComputationNode<double>;

template class LearnableParameter<float>;
template class LearnableParameter<double>;

}}}

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

using namespace Microsoft::MSR::CNTK;

// -----------------------------------------------------------------------
// register ComputationNode with the ScriptableObject system
// -----------------------------------------------------------------------

template <>
shared_ptr<Object> MakeRuntimeObject<ComputationNodeBase>(const IConfigRecordPtr configp)
{
    return NewComputationNodeFromConfig(configp);
}

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNodeBase> registerComputationNode(L"ComputationNode");

// -----------------------------------------------------------------------
// register a boxed version of TensorShape with the ScriptableObject system
// -----------------------------------------------------------------------

// create a vector from config
// Nested vectors are flattened, which is useful e.g. for building TensorShapes.
template <typename E>
static vector<E> VectorFromConfig(const ConfigValuePtr& valp)
{
    if (valp.Is<vector<E>>())
        return valp.AsRef<vector<E>>(); // UNTESTED
    else if (valp.Is<ConfigArray>())
        return valp.AsRef<ConfigArray>().AsVector<E>([&](const wstring& msg) { valp.Fail(msg); }, /*flatten=*/true);
    else
        return std::vector<E>(1, (E)valp); // single element
}

// create a vector from config
template <typename E>
static vector<E> VectorFromConfig(const IConfigRecord& config, const wstring& paramName)
{
    const auto& valp = config[paramName];
    return VectorFromConfig<E>(valp);
}

// e.g.
// new TensorShape [ dims = 13:42 ]
class BoxedTensorShape : public BoxOf<TensorShape>
{
public:
    BoxedTensorShape(const IConfigRecordPtr configp)
        : BoxOf<TensorShape>(TensorShape(VectorFromConfig<size_t>(*configp, L"dims")))
    {
    }
};

template <typename E>
class BoxedVector : public BoxOf<vector<E>>
{
public:
    BoxedVector(const IConfigRecordPtr configp)
        : BoxOf<vector<E>>(VectorFromConfig<E>(*configp, L"items"))
    {
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedTensorShape> registerTensorShape(L"TensorShape");
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedVector<int>> registerIntVector(L"IntVector");
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedVector<size_t>> registerSizeVector(L"SizeVector");
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<BoxedVector<bool>> registerBoolVector(L"BoolVector");

}}}
