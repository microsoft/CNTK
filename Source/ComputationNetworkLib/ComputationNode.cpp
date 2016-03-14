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
            RuntimeError("InferMBLayoutFromInputsForStandardCase: Found inconsistent layout in %ls %ls operation, mismatch detected for child %ls %ls.",
                         NodeName().c_str(), OperationName().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
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
        LogicError("MB layouts in the %ls %ls operation do not match.", NodeName().c_str(), OperationName().c_str());
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
            InvalidArgument("%ls %ls operation: Input dimensions [%s] and [%s] are not compatible.",
                            NodeName().c_str(), OperationName().c_str(), string(shape0).c_str(), string(shape1).c_str());
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
    if (isFinalValidationPass &&
        !(Input(0)->GetSampleLayout().IsElementwiseCompatibleWith(Input(1)->GetSampleLayout()) && // TODO: Do we need broadcasting for these cases?
          (Input(0)->GetMBLayout() == Input(1)->GetMBLayout() || !Input(0)->HasMBLayout() || !Input(1)->HasMBLayout())))
        LogicError("The Matrix dimensions or MB layout in the %ls %ls operation do not match.", NodeName().c_str(), OperationName().c_str());
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

}}}
