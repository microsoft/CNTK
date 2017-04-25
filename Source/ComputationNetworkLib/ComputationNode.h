//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "TensorView.h"
#include "ScriptableObjects.h"
#include "Sequences.h"
#include "TensorShape.h"
#include "MatrixPool.h"
#include "ComputationEnvironment.h"
#include "Globals.h"

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

#define DEFAULT_HIDDEN_ACTIVATION 0.1

#pragma warning(disable : 4267) // conversion from size_t to int or other types

// version number to control how to read and write
#define CNTK_MODEL_VERSION_1 1
#define CNTK_MODEL_VERSION_2 2
#define CNTK_MODEL_VERSION_3 3
#define CNTK_MODEL_VERSION_4 4   // PastValue
#define CNTK_MODEL_VERSION_5 5   // ND convolution and pooling
#define CNTK_MODEL_VERSION_6 6   // batch-norm blending
#define CNTK_MODEL_VERSION_7 7   // ElemType tag in model file
#define CNTK_MODEL_VERSION_8 8   // DynamicAxis for inputs
#define CNTK_MODEL_VERSION_9 9   // transpose flag in ConvolutionNode to support deconvolution
#define CNTK_MODEL_VERSION_10 10 // learning-rate multiplier for input nodes
#define CNTK_MODEL_VERSION_11 11 // dynamic axis name for where nodes
#define CNTK_MODEL_VERSION_12 12 // Times() m_inputRank to support parameter-rank inference
#define CNTK_MODEL_VERSION_13 13 // batch norm: switch running inverse std deviation -> variance, MB count -> samplesSeen; CuDNN v5
#define CNTK_MODEL_VERSION_14 14 // axis parameter in OptimizedRNNStackNode
#define CNTK_MODEL_VERSION_15 15 // add new nodes: LambdaRankNode and NDCG1Eval
#define CNTK_MODEL_VERSION_16 16 // save/load rng state for Dropout and RandomSample nodes.
#define CNTK_MODEL_VERSION_17 17 // use 8 bytes for rng seeds on both platforms
#define CNTK_MODEL_VERSION_18 18 // reserving 18 for dilated convolution, write out one more TensorShape 
#define CNTK_MODEL_VERSION_19 19 // batch norm: flag whether running mean count is 0
#define CNTK_MODEL_VERSION_20 20 // adding output shape to convolution node
#define CNTK_MODEL_VERSION_21 21 // pooling: add a ceilOutDim to decide whether ceil or floor while computing the output size
#define CNTK_MODEL_VERSION_22 22 // Slice and pad accepts multiple axes 
#define CNTK_MODEL_VERSION_23 23 // pooling: add include pad func for average pooling
#define CNTK_MODEL_VERSION_24 24 // ReduceElements: add keepDimensions
#define CURRENT_CNTK_MODEL_VERSION CNTK_MODEL_VERSION_24


// helper mode for debugging
// If TRACK_GAP_NANS is defined then initialize layout gaps to NaN and do NaN checks. Also do detailed logging of node computations.
// #define TRACK_GAP_NANS
// TODO: Make this a trace option, e.g. enabled by the ComputeEnvironment.

namespace Microsoft { namespace MSR { namespace CNTK {

enum CopyNodeFlags // flags to be passed to the CopyTo() function
{
    copyNodeValue          = 1, // copy everything except for the input links
    copyNodeInputLinks     = 2, // copy over input links
    copyNodeAll            = 3, // copy everything
    copyNodeAcrossNetworks = 4  // allow a cross network child copy
};

#pragma region base computation class

// =======================================================================
// IComputationNode -- set of methods that are to be implemented (or optionally overridable) by node implementations.
// =======================================================================

class ComputationNodeBase;
struct /*interface*/ IComputationNode
{
    typedef shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

    // --- these must be implemented by each node

    virtual ComputationNodeBase* NewThis(DEVICEID_TYPE deviceId, const wstring& name) const = 0;
    // TODO: OperationName calls static TypeName which does not match the actual type names in that the 'Node' is missing.
    virtual const std::wstring OperationName() const = 0;
#define OperationNameOf(T) (T<float>::TypeName()) // convenience macro

    virtual void UpdateFunctionMBSize() = 0; // recalculate our column dimensions from MBLayout. Override to update temps.

    virtual void BeginForwardProp() = 0;             // called beforefirst iteration step of ForwardProp()
    virtual void ForwardProp(const FrameRange&) = 0; // forward prop for one minibatch
    virtual void EndForwardProp() = 0;               // called after last iteration step of ForwardProp()

    virtual void PostForwardAndBackProp() {} // Optional: Post forward and backprop prop for one minibatch, this will be called in a second 
                                             //           looping on the graph, after the backward pass finish. Or after forward pass in inference
                                             //           mode.

    virtual void BeginBackprop() = 0;                                        // called before first iteration step of ComputeGradient()
    virtual void BackpropTo(const size_t inputIndex, const FrameRange&) = 0; // backprop gradient into one of the inputs
    virtual void EndBackprop() = 0;                                          // called after last iteration step of ComputeGradient()

    // --- this is meant to be overridden by ControlFlowNodes

    virtual void Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) = 0;

    // --- optional overrides that add functionality

    // Any override must call Base version as well.
    // Default implementations are in ComputationNodeBase or ComputationNode<ElemType>.

    virtual void Validate(bool isFinalValidationPass) = 0; // main base validation function
    virtual void Save(File& fstream) const = 0;
    virtual void Load(File& /*fstream*/, size_t /*modelVersion*/) = 0;
    virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const = 0;

    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) = 0; // request matrices needed to do node function value evaluation
    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) = 0;  // release temp matrices that are only used by forward computation. Don't release matrices that need to be used in the gradient computation
    virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool) = 0;
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) = 0; // request matrices that are needed for gradient computation
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) = 0;  // release gradient and temp matrices that no longer needed after all the children's gradients are computed.

    // --- optional overrides that describe a feature or property of the node

    virtual bool RequiresPreCompute() const = 0; // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.

    // --- optional overrides for more informative logging

    virtual std::string FormatOperationPrototype(const std::string& extraArgs) const = 0; // format the operation into a "prototype" (listing dimensions and parameters)
    virtual void DumpNodeInfo(const bool /*printValues*/, const bool /*printMetadata*/, File& fstream) const = 0;

protected:
    virtual ~IComputationNode()
    {
    }
};

// =======================================================================
//  Interface for stateful node (e.g., DelayNodeBase) and definition of state
//  This interface allows to Export and Import state from elsewhere, e.g. for sub-minibatching.
// =======================================================================

class INodeState : public std::enable_shared_from_this<INodeState> { public: virtual ~INodeState() { } };

struct /*interface*/ IStatefulNode
{
    typedef std::shared_ptr<INodeState> NodeStatePtr;
    virtual NodeStatePtr ExportState() = 0;
    virtual void ImportState(const NodeStatePtr& state) = 0;
};
typedef IStatefulNode::NodeStatePtr NodeStatePtr;

// =======================================================================
// ComputationNetworkOwnedNodeState -- class to collect ComputationNode members that are really owned by ComputationNetwork
// These members are only to be set, changed, and read by ComputationNetwork code.
// =======================================================================

class ComputationNetwork;
struct ComputationNetworkOwnedNodeState
{
    friend class ComputationNetwork;

    ComputationNetworkOwnedNodeState()
        : m_needsGradient(false), m_needsDynamicValidation(false), m_valueSharable(true), m_parentOverwritesGradient(false)
    {
        PurgeStateForFormingRecurrentLoops();
        m_isPartOfLoop = false;
    }

    void CopyTo(ComputationNetworkOwnedNodeState& other) const
    {
        other.m_isPartOfLoop                  = m_isPartOfLoop;
        other.m_needsGradient                 = m_needsGradient;
        other.m_needsDynamicValidation        = m_needsDynamicValidation;
        other.m_valueSharable                 = m_valueSharable;
        other.m_traceNodeValueReal            = m_traceNodeValueReal;
        other.m_traceNodeValueAsCategoryLabel = m_traceNodeValueAsCategoryLabel;
        other.m_traceNodeValueSparse          = m_traceNodeValueSparse;
        other.m_traceNodeValueUpToDim         = m_traceNodeValueUpToDim;
        other.m_traceNodeValueUpToT           = m_traceNodeValueUpToT;
        other.m_parentOverwritesGradient = m_parentOverwritesGradient;
    }

    bool IsPartOfLoop() const { return m_isPartOfLoop; }

    void MarkParentOverwritesGradient() { m_parentOverwritesGradient = true; }
    bool ParentOverwritesGradient() const { return m_parentOverwritesGradient; }

    virtual void MarkValueNonSharable() { m_valueSharable = false; }
    virtual void MarkValueSharable() { m_valueSharable = true; }
    bool IsValueSharable() const { return m_valueSharable; }

    // tracing flags
    // Enable to print the value of the function-value matrix in somewhat readable format.
    // These are public since you are meant to set these flags manually in the debugger or temporarily poke into them from code as needed.
    bool m_traceNodeValueReal = false;
    bool m_traceNodeValueAsCategoryLabel = false;
    bool m_traceNodeValueSparse = false;
    size_t m_traceNodeValueUpToDim = 3; // 3 should be enough to see simple patterns such as all values are identical or out of range
    size_t m_traceNodeValueUpToT = 8;   // 8 time steps fit comfortably into a normal-sized console
    void EnableNodeTracing(bool asReal, bool asCategoryLabel, bool asSparse) { m_traceNodeValueReal = asReal; m_traceNodeValueAsCategoryLabel = asCategoryLabel; m_traceNodeValueSparse = asSparse; }

    virtual bool ImplementsGradientOverwriteOptimization() const { return false; }

protected:                // TODO: should be fully encapsulated here
    bool m_needsGradient; // true if this node or any children need a gradient to be computed (for own consumption or propagation to somewhere in the child tree)
    bool m_needsDynamicValidation;

    bool m_valueSharable; // a flag is needed for memory share.
                          // If it is false (e.g., LearnableParameters/InputValue and those nodes are solely induced by LearnableParameters),
                          // it will never be released to memory pool

    bool m_parentOverwritesGradient; // flag indicating whether the parent of this node overwrites the gradient of this node instead of accumulating to it

private:
    bool m_isPartOfLoop; // true if this loop is part of a recurrent loop

protected:
    // owned by FormRecurrentLoops() and stuff it calls, only used from inside there (FormRecurrentLoops() calls PurgeStateForFormingRecurrentLoops() at its end to make that super-clear)
    void PurgeStateForFormingRecurrentLoops()
    {
        m_loopId = -1;
        m_visitedOrder = -1;
        m_numNonDelayedParentsInLoop = 0;
        m_visited = false;
        m_index = -1;
        m_minIndex = -1;
        m_inStack = false;
    }

    int m_loopId;       // index into m_allSEQNodes array, for use by reordering operation only
    int m_visitedOrder; // remembers order in which nodes were visited by EnumerateNodes(), but gets updated
    bool m_visited;     // note: also used by ValidateSubNetwork()
    int m_numNonDelayedParentsInLoop;
    // only used inside DetermineSCCs():
    int m_index;    // index denoting order in which nodes were visited in DetermineSCCs()
    int m_minIndex; // min of m_index over all nodes within a single loop
    bool m_inStack;
};

// =======================================================================
// TimeStamp -- helper class to manage a "time stamp" (unique value) of a computation result to avoid recomputation
// =======================================================================

class TimeStamp
{
public:
    TimeStamp()
    {
        ResetEvalTimeStamp();
    }
    void CopyTo(TimeStamp& other) const
    {
        other.m_evalTimeStamp = m_evalTimeStamp;
    }
    void ResetEvalTimeStamp()
    {
        m_evalTimeStamp = s_timeStampCounter;
    }
    void SetEvalTimeStampOutdatedWrtAll()
    {
        m_evalTimeStamp = 0;
    }
    uint64_t GetEvalTimeStamp() const
    {
        return m_evalTimeStamp;
    }

    // create a new unique time stamp
    void BumpEvalTimeStamp()
    {
        m_evalTimeStamp = CreateUniqId();
    }

    bool IsOlderThan(const TimeStamp& other) const
    {
        return GetEvalTimeStamp() < other.GetEvalTimeStamp();
    }

    uint64_t CreateUniqId() const
    {
        return ++s_timeStampCounter;
    }

private:
    static atomic_ullong s_timeStampCounter;
    uint64_t m_evalTimeStamp; // this is used to reduce unnecessary recomputation when a different node in the model is reevaluated
};

// =======================================================================
// ComputationNodeBase -- abstract base class for all computation nodes
// =======================================================================

class ComputationNodeBase : public IComputationNode,
                            public /*protected*/ ComputationNetworkOwnedNodeState, // TODO: figure the 'protected' business out, somehow the 'friend' thing does not work
                            public TimeStamp,                                      // for time-stamp management
                            public ScriptableObjects::ComputationNodeObject,
                            public ScriptableObjects::WithTags,
                            public ScriptableObjects::HasName,
                            public ScriptableObjects::HasToString,
                            public ScriptableObjects::CustomConfigRecord, // make members accessible as BS expressions
                            public std::enable_shared_from_this<ComputationNodeBase>
{
    // note: enable_shared_from_this<> allows to create a shared_ptr from a raw pointer to this that is correctly aware of all other shared_ptrs (same ref count)
public:
    typedef shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

    // -----------------------------------------------------------------------
    // constructors, copying, (de-)serialization
    // -----------------------------------------------------------------------

    ComputationNodeBase(DEVICEID_TYPE deviceId, const wstring& name) :
        m_deviceId(deviceId), m_outputNeededDuringBackprop(true), m_learningRateMultiplier(0),
        m_gradientInitialized(false), m_nodeName(name == L"" ? CreateUniqNodeName() : name), m_isValueSparse(false)
    {
        // TODO: should m_learningRateMultiplier be set to 0? Or should every node have a way to add its own say on the learning rate for all its inputs?
        // we store a unique numeric number for every node that is constructed, as a debugging aid
        static size_t uniqueNumericId = 0;
        m_uniqueNumericId = uniqueNumericId++;
    }
    virtual ~ComputationNodeBase()
    {
    }

    virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const
    {
        if (OperationName() != node->OperationName())
            RuntimeError("Cannot copy from one node type to another node type");
        if (flags & CopyNodeFlags::copyNodeInputLinks)
        {
            node->m_inputs = m_inputs;
        }
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            node->m_deviceId = m_deviceId;
            node->m_learningRateMultiplier = m_learningRateMultiplier;
            node->m_nodeName = newName;

            node->m_sampleLayout = m_sampleLayout;

            ComputationNetworkOwnedNodeState::CopyTo(*node);
            TimeStamp::CopyTo(*node);
        }

        if (flags & CopyNodeFlags::copyNodeAll)
        {
            for (auto tag : this->GetTags())
            {
                node->SetTag(tag);
            }
        }

        node->ClearConfigMemberCache();
    }

    virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName = L"", const CopyNodeFlags flags = CopyNodeFlags::copyNodeAll) const = 0;   // (called on here implemented by ComputationNode<ElemType>

    virtual void Load(File& /*fstream*/, size_t /*modelVersion*/)
    {
        // it is assumed that OperationName and NodeName have already been consumed
        // base class has nothing else to load
    }

    virtual void Save(File& /*fstream*/) const
    {
        // it is assumed that OperationName and NodeName have already been saved
        // base class has nothing else to save
    }

    std::wstring CreateUniqNodeName() const
    {
#ifdef USE_GUID_AS_NAME
        UUID uuid;
        ZeroMemory(&uuid, sizeof(UUID));
        std::wstring name;

        UuidCreate(&uuid);
        WCHAR* szUuid = nullptr;
        if (UuidToStringW(&uuid, (RPC_WSTR*) &szUuid) != RPC_S_OK)
            RuntimeError("Failed to craete unique node name.");
        else
        {
            name = szUuid;
            RpcStringFreeW((RPC_WSTR*) &szUuid);
        }
#else
        uint64_t id = CreateUniqId();
        std::wstring base = L"AutoName";
        std::wstringstream sstm;
        sstm << base.c_str() << id;
        std::wstring name = sstm.str();
//msra::strfun::wstrprintf name(L"%s%d", L"AutoName", id);
#endif

        return name;
    }

    // -----------------------------------------------------------------------
    // dimensions of the value held by this node
    // -----------------------------------------------------------------------

    // The value of a node is a tensor in one of two variants:
    //
    //  - single matrix, vector, tensor
    //     - m_sampleLayout contains the shape. Accessed through GetSampleLayout().
    //     - m_pMBLayout is null
    //  - minibatch data
    //     - consists of many samples which are all tensors of m_sampleLayout
    //     - adds two additional tensor dimensions, time step and parallel sequence
    //       These change for each minibatch and are unknown during validation.
    //     - m_sampleLayout is the tensor shape of the samples
    //     - m_pMBLayout defines the number of time steps and parallel sequences (="tensor shape" of the minibatch)
    //       Accessed through GetMBLayout(); test for through HasMBLayout().
    //
    // The values can be accessed in three ways:
    //
    //  - as a tensor
    //     - GetTensorShape() forms the joint tensor that incorporates both m_sampleLayout and, if present, m_pMBLayout
    //        - Elementwise tensor operations operate on these.
    //        - If no MBLayout is present in one of multiple elementwise operands, it will be interpreted as a one-sample minibatch that broadcasts to all samples.
    //     - learnable parameters hold tensors that are not minibatches
    //  - as a sample matrix
    //     - many nodes do not care about the specific sample-tensor dimensions
    //     - but may care about selecting a single time step out of a minibatch
    //     - minibatch: each matrix column contains a sample tensor flattened, with one column per time step and parallel sequence
    //     - tensor: one column containing the sample tensor flattened
    //     - GetSampleMatrixNumRows(), GetSampleMatrixNumCols()
    //  - as a Matrix reference
    //     - actual object is a 2D tensor without MB Layout
    //     - ValueAsMatrix(), GradientAsMatrix() returns tensor as a 2D Matrix object
    //     - nodes that do this are: TimesNode, DiagTimesNode, ConvolutionNode, NoiseContrastiveEstimationNode, ClassBasedCrossEntropyWithSoftmaxNode, TransposeDimensionsNode, DiagonalNode
    //
    // How values are stored:
    //
    //  - minibatch: Matrix of columns, where each column is a sample
    //  - tensor: Matrix where column dimension contains all but the first dimension
    //     - This only matters for sparse matrices, which cannot easily be Reshaped().
    //       For those, we keep the underlying storage identical to the semantic meaning.

    // accessor to sample layout
    const TensorShape& GetSampleLayout() const { return m_sampleLayout; }
    bool HasSampleLayout() const { return m_sampleLayout.GetRank() != 1; } // does it have a layout that is not just a vector?
    const TensorShape& GetInputSampleLayout(const size_t index) const { return m_inputs[index]->GetSampleLayout(); }

    // interpretation as sample matrix (each column is a sample, individual sample tensor dimensions do not matter for the operation)
    size_t GetSampleMatrixNumRows() const
    {
        return m_sampleLayout.GetNumElements();
    }
    size_t GetSampleMatrixNumCols() const
    {
        if (HasMBLayout())
            return GetMBLayout()->GetNumCols();
        else
            return 1; // no layout: treat as 1-sample minibatch that is meant to broadcast
    }
    // determine if we are the output of an op over 'other', whether that would be a reduction, so that we need to mask
    bool ReducesInTimeWrt(const ComputationNodeBasePtr& other) const
    {
        return GetSampleMatrixNumCols() < other->GetSampleMatrixNumCols();
    }

    // interpretation as a Matrix reference
private:
    void CheckTensorIsMatrix() const
    {
        if (HasMBLayout())
            LogicError("%ls: Minibatch data cannot be interpreted as a single 2D tensor.", NodeDescription().c_str());

        bool notFlattenableTo2D = false;
        for (size_t i = 2; i < m_sampleLayout.GetRank(); ++i)
        {
            if (!m_sampleLayout.CanFlatten(i))
            {
                notFlattenableTo2D = true;
                break;
            }
        }

        if (m_sampleLayout.GetRank() < 1 || ((m_sampleLayout.GetRank() > 2) && notFlattenableTo2D)) // note: scalars are not stored as tensors of rank 0, but rather as 1-dim vectors. TODO: clean this up some day
            LogicError("%ls: Sample [%s] is not a column vector or matrix (1D or 2D tensor).", NodeDescription().c_str(), string(m_sampleLayout).c_str());
    }
public:
    size_t GetAsMatrixNumRows() const
    {
        CheckTensorIsMatrix();
        return m_sampleLayout[0];
    }
    size_t GetAsMatrixNumCols() const
    {
        CheckTensorIsMatrix();
        auto flattenedLayout = m_sampleLayout;
        if (flattenedLayout.GetRank() > 2)
            flattenedLayout.FlattenTo2DInPlace(1, "GetAsMatrixNumCols()");

        return flattenedLayout.GetRank() > 1 ? flattenedLayout[1] : 1; // a column vector is also a Matrix
    }

    // setting/updating the dimensions of the node
    // The MBLayout must be set first, and 'isMinibatch' will be checked against it.
    void SetDims(const TensorShape& sampleLayout, bool isMinibatch)
    {
        if (HasMBLayout() != isMinibatch)
            LogicError("%ls: SetDims: MBLayout must be set first, before calling this function.", NodeDescription().c_str());
        m_sampleLayout = sampleLayout;
    }
    // copy dimensions (rows, cols, sample layout) from another node
    void SetDims(const ComputationNodeBasePtr& node)
    {
        SetDims(node->GetSampleLayout(), node->HasMBLayout());
    }

    // the following two are only for legacy testing code; don't use this
    void SetDims1(size_t rows, size_t cols) { SetDims(TensorShape(rows, cols), false); }
    size_t GetNumCols1() const { return GetSampleMatrixNumCols(); } // dummy

    // checking the dimensions of the node
    virtual void NotifyFunctionValuesMBSizeModified() = 0;
    void VerifyDims(const TensorShape& shape, bool isMinibatch)
    {
        if (m_sampleLayout.GetDims() != shape.GetDims() || HasMBLayout() != isMinibatch)
        {
            LogicError("%ls: VerifyDims: Expected a %s of [%s], but it is a %s of [%s]",
                       NodeDescription().c_str(),
                       isMinibatch ? "minibatch" : "tensor", string(shape).c_str(),
                       HasMBLayout() ? "minibatch" : "tensor", string(m_sampleLayout).c_str());
        }
    }
    virtual void VerifyDims(ComputationNodeBasePtr node)
    {
        VerifyDims(node->GetSampleLayout(), node->HasMBLayout());
    }

    // MBLayout (minibatch structure)
    void LinkToMBLayout(MBLayoutPtr pMBLayout)
    {
        m_pMBLayout = pMBLayout;
    }
    const MBLayoutPtr& GetMBLayout() const { return m_pMBLayout; }
    bool HasMBLayout() const { return !!m_pMBLayout; }

    // for logging: get the string fragment for displaying the dimension
    std::wstring GetMBLayoutAxisString() const
    {
        if (!HasMBLayout())
            return L"";
        const wstring& axisName = GetMBLayout()->GetAxisName();
        if (axisName.empty())
            return L" x *";
        else
            return L" x " + axisName;
    }

protected: public: // ...the following should be protected, but nodes inquire about their children, requiring public access

    size_t GetNumParallelSequences() const
    {
#if 1
        if (!m_pMBLayout) // TODO: temporary workaround to Check_t() calls which call this. TODO: Delete the first arg from Check_t() after memshare merge.
            return SIZE_MAX;
#endif
        return m_pMBLayout->GetNumParallelSequences();
    }

    // get our current number of time steps for this node
    // This inquires the MB layout.
    size_t GetNumTimeSteps() const
    {
        if (!m_pMBLayout)
            LogicError("%ls: GetNumTimeSteps: invalid to call on a node without MB layout", NodeDescription().c_str()); // since it has no notion of time
        return m_pMBLayout->GetNumTimeSteps();
    }

public:

    // forming the actual tensor that describes the full object
    TensorShape GetTensorShape(size_t rank) const;

protected:

    size_t DetermineElementwiseTensorRank() const;                          // determine tensor rank when considering all inputs with padding

public:

    TensorShape GetTensorSliceFor(size_t rank, const FrameRange& fr) const; // form tensor shape of the slice referenced by FrameRange. Public since nodes may call it for their inputs.
    TensorShape GetOneSampleTensorSliceFor(size_t rank, const FrameRange& fr) const; // same but 'fr' refers to a single column, and result will not have seq/time axes

    // -----------------------------------------------------------------------
    // inputs
    // -----------------------------------------------------------------------

    // access an input
    const ComputationNodeBasePtr& Input(size_t index) const { return m_inputs[index]; }

    // access all inputs (use this for range-based for loops)
    const std::vector<ComputationNodeBasePtr>& GetInputs() const { return m_inputs; }
    const size_t GetNumInputs() const { return m_inputs.size(); }
    bool IsLeaf() const { return GetNumInputs() == 0; }

    // attaching/detaching inputs
    virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs) = 0;

    // Note: This function is used to break cyclic references.
    virtual void DetachInputs()
    {
        ClearConfigMemberCache(); // cached config may also hold pointers, must clear, too
        m_inputs.clear();
    }

    virtual void SetInput(const size_t childIndex, const ComputationNodeBasePtr& node) = 0;

    // helper for the factory function for ComputationNodes
    static vector<ComputationNodeBasePtr> GetInputsFromConfig(const ScriptableObjects::IConfigRecordPtr configp)
    {
        return GetInputsFromConfig(configp, L"inputs");
    }

    static vector<ComputationNodeBasePtr> GetInputsFromConfig(const ScriptableObjects::IConfigRecordPtr configp, const std::wstring& property)
    {
        vector<ComputationNodeBasePtr> inputs;
        const auto* inputsArg = configp->Find(property);
        if (inputsArg)
        {
            if (inputsArg->Is<ComputationNodeBase>()) // single arg
                inputs.push_back(*inputsArg);
            else // a whole vector
            {
                ScriptableObjects::ConfigArrayPtr inputsArray = *inputsArg;
                const auto range = inputsArray->GetIndexBeginEnd();
                for (int i = range.first; i < range.second; i++) // pull them. This will resolve all of them.
                    inputs.push_back(inputsArray->At(i, [](const wstring&) { LogicError("GetInputs: out of bounds index while iterating??"); }));
            }
        }
        return inputs;
    }

    // -----------------------------------------------------------------------
    // accessors
    // -----------------------------------------------------------------------

    DEVICEID_TYPE GetDeviceId() const { return m_deviceId; }

    // helper to access to element(0,0) without having to type-cast
    virtual double Get00Element() const = 0;
    virtual MatrixBasePtr ValuePtr() const = 0; // for use in readers that pass the agnostic object around

    // TODO: two sets of functions, choose one
    const std::wstring& NodeName() const { return m_nodeName; }
    std::wstring GetName() const { return m_nodeName; }
    void SetNodeName(const std::wstring& nodeName) { m_nodeName = nodeName; }
    /*HasName::*/ void SetName(const std::wstring& newName) override // also for use by ExperimentalNetworkBuilder
    {
        ClearConfigMemberCache();
        m_nodeName = newName;
        //fprintf(stderr, "Node --> %ls : %ls\n", NodeName().c_str(), OperationName().c_str()), fflush(stderr);
    }

    bool NeedsGradient() const { return m_needsGradient; }

    void MarkNeedsDynamicValidation() { m_needsDynamicValidation = true; }
    virtual bool NeedsDynamicValidation() const { return m_needsDynamicValidation; }

    void SetLearningRateMultiplier(float f) 
    { 
        if (f < 0)
            InvalidArgument("%ls: LearningRateMultiplier should be non-negative. You are tring to set it to %f.", NodeDescription().c_str(), f);
        m_learningRateMultiplier = f; 
    }
    float GetLearningRateMultiplier() const { return m_learningRateMultiplier; }
    bool IsParameterUpdateRequired() const { return m_learningRateMultiplier > 0; }

    // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
    virtual bool /*IComputationNode::*/ RequiresPreCompute() const { return false; }

    const ComputationEnvironment& Environment() const
    {
        if (!m_environment)
            LogicError("Environment: No environment has been set.");
        return *m_environment;
    }

    bool HasEnvironmentPtr() const { return m_environment.get() != nullptr; }
    ComputationEnvironmentPtr GetEnvironmentPtr() const { return m_environment; }
    void SetEnvironment(ComputationEnvironmentPtr environment) { m_environment = environment; }

    virtual std::set<std::pair<const MatrixBase*, std::wstring>> GetMatrixInfo() const = 0; // to be defined by <ElemType> version

    // -----------------------------------------------------------------------
    // validation
    // -----------------------------------------------------------------------

    // This is overridden by every node. This base class just checks for unconnected and empty inputs. Overrides must call their base version first.
    virtual void Validate(bool isFinalValidationPass) // main base validation function
    {
        // check for NULL pointers
        for (size_t i = 0; i < m_inputs.size(); i++)
        {
            if (!m_inputs[i])
                RuntimeError("%ls: Validate: Input [%d] is empty (NULL, not connected).", NodeDescription().c_str(), (int)i);
        }
        // check for empty inputs
        if (isFinalValidationPass)
        {
            for (const auto& child : m_inputs)
                if (child->GetSampleMatrixNumRows() == 0)
                    RuntimeError("%ls: input %ls %ls has 0 elements.", NodeDescription().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
        }

        // By default the only case when the Value of a node is sparse 
        // is when the node has a single input with sparse Value
        if ((GetNumInputs() == 1) && m_inputs[0]->IsValueSparse())
            m_isValueSparse = true;
    }

protected:

    // helper functions for common cases
    void ValidateUnaryMap(bool isFinalValidationPass);
    void ValidateUnaryReduce(bool isFinalValidationPass, bool keepDimensions = false);
    void ValidateInferBinaryInputDims();
    void ValidateInferNaryInputDims(size_t numInputs);    
    void ValidateBinaryZip(bool isFinalValidationPass, bool allowBroadcast);
    void ValidateBinaryReduce(bool isFinalValidationPass);    
    void ValidateNaryZip(bool isFinalValidationPass, bool allowBroadcast, size_t numInputs);
    void ValidateMBLayout(const ComputationNodeBasePtr which, const ComputationNodeBasePtr vsWhich) const;
    void InferMBLayoutFromInputsForStandardCase(bool isFinalValidationPass);
    virtual void ValidateInferInputDimsFrom(const TensorShape&) = 0;    // (implemented by ComputationNode<ElemType>)

public:

    virtual void OnEpochStart() {}

    // -----------------------------------------------------------------------
    // forward prop, backprop
    // -----------------------------------------------------------------------

    virtual void /*IComputationNode::*/ BeginForwardProp() override // called before first iteration step of ForwardProp()
    {
#ifdef TRACK_GAP_NANS
        fprintf(stderr, "BeginForwardProp: %ls %ls operation [%s]\n", NodeName().c_str(), OperationName().c_str(), std::string(GetTensorShape(DetermineElementwiseTensorRank())).c_str());
#endif
    }
    virtual void /*IComputationNode::*/ EndForwardProp() override // called after last iteration step of ForwardProp()
    {
#ifdef TRACK_GAP_NANS
        fprintf(stderr, "EndForwardProp: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
    }
    virtual void /*IComputationNode::*/ BeginBackprop() override // called before first iteration step of ComputeGradient()
    {
#ifdef TRACK_GAP_NANS
        fprintf(stderr, "BeginBackprop: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
    }
    virtual void /*IComputationNode::*/ EndBackprop() override // called after last iteration step of ComputeGradient()
    {
#ifdef TRACK_GAP_NANS
        fprintf(stderr, "EndBackprop: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
    }

    // check whether a node is out of date w.r.t. its children, for lazy evaluation
    // If this returns true, node must be evaluated to update m_value.
    // This is virtual because it is overridden by traversal nodes, which would check all their nodes' inputs.
    virtual bool IsOutOfDateWrtInputs() const
    {
        for (const auto & input : GetInputs())
            if (!input->IsOlderThan(*this))
                return true;
        // Note: This ^^ must also return true when time stamps are the same, for an unknown reason (possibly an initialization condition). We should track this down some day.
        return false;
    }

    // reset gradients of a node's inputs
    // This really only clears the lazy-init flags (LazyZeroGradient() actually clears the values lazily).
    void /*ComputationNodeBase::*/ ZeroGradientsOfInputs()
    {
        for (size_t i = 0; i < m_inputs.size(); i++)
            Input(i)->m_gradientInitialized = false;
    }

    // -----------------------------------------------------------------------
    // masking
    // -----------------------------------------------------------------------

    // overridden by <ElemType> variant only
    virtual void MaskMissingValueColumnsToZero(const FrameRange&) = 0;
    virtual void MaskMissingGradientColumnsToZero(const FrameRange&) = 0;
    virtual void InvalidateMissingValueColumns(const FrameRange&) = 0;
    virtual void InvalidateMissingGradientColumns(const FrameRange&) = 0;

    // -----------------------------------------------------------------------
    // memory sharing
    // -----------------------------------------------------------------------

    // Is the output value of the computation node needed for computing
    // gradients of any of the input nodes
    // Base-class version makes conservative assumption that it is. Override if not.
    virtual bool OutputUsedInComputingInputNodesGradients() const { return true; }

    // Is the output value of the specified  input node needed for computing
    // gradients of any of the input nodes
    // Base-class version makes conservative assumption that it is. Override if not.
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const { return true; }

    void SetOutputNeededDuringBackprop(bool f) { m_outputNeededDuringBackprop = f; }
    bool IsOutputNeededDuringBackprop() const 
    { 
        return !Globals::ShouldEnableShareNodeValueMatrices() || m_outputNeededDuringBackprop; 
    }

    // -----------------------------------------------------------------------
    // helpers for network traversal
    // -----------------------------------------------------------------------

    // determine enumeration order for everything needed to evaluate this node (and its children)
    // This creates a list such that children are evaluated before their parents.
    // If !forForwardProp then the order will be reversed, suitable for backprop.
    // TODO: This should be a method of ComputationNetwork, not ComputationNode.
    static std::list<ComputationNodeBasePtr> EnumerateNodes(const std::vector<ComputationNodeBasePtr>& allRoots)
    {
        std::list<ComputationNodeBasePtr> nodes;
        std::unordered_set<ComputationNodeBasePtr> visited;

        for (const auto& root : allRoots)
            root->EnumerateNodesRec(visited, nodes); // call into the recursive portion of this function below

        return nodes;
    }

    // and a version that does it for only one root 'this'
    std::list<ComputationNodeBasePtr> EnumerateNodes() /*const*/
    {
        return EnumerateNodes(std::vector<ComputationNodeBasePtr>{shared_from_this()});
    }

    static const std::wstring DefaultDynamicAxisName;
    static const std::wstring DefaultNoSequenceAxisName;

private:
    // Recursive part of EnumerateNodes().
    void EnumerateNodesRec(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result) /*const*/ // const not working due to shared_from_this()
    {
        if (visited.find(shared_from_this()) == visited.end()) // do not include a node twice
        {
            visited.insert(shared_from_this()); // have visited tagged here to avoid infinite loop over children, children's children, etc

            // children first for function evaluation
            for (int i = 0; i < m_inputs.size(); i++)
                if (m_inputs[i])
                    m_inputs[i]->EnumerateNodesRec(visited, result);

            // now that all children are in list before us, put ourselves
            result.push_back(shared_from_this());
        }
    }

public:
    // -----------------------------------------------------------------------
    // scripting integration
    // -----------------------------------------------------------------------

    // pretend to be a ConfigRecord
    void /*CustomConfigRecord::*/ LazyCreateConfigMember(const std::wstring& id) const override;
    std::vector<std::wstring> /*IConfigRecord::*/ GetMemberIds() const override;

    // -----------------------------------------------------------------------
    // miscellaneous
    // -----------------------------------------------------------------------

    // casting helpers
    template <typename N>
    N* As()
    {
        auto p = dynamic_cast<N*>(this);
        if (!p)
            LogicError("%ls: Attempted to type-cast node to %s, which is not possible.", NodeDescription().c_str(), typeid(N).name());
        return p;
    }
    template <typename N>
    bool Is()
    {
        return dynamic_cast<N*>(this) != nullptr;
    }

    virtual bool UnitTest() { return true; }

    // implemented by ComputationNode<ElemType>
    // for debugging purpose
    virtual void PrintSelf(bool printMatrices = false) const = 0;

    // called in validation loop right before Validate()
    virtual std::string /*IComputationNode::*/ FormatOperationPrototype(const std::string& extraArgs) const;

    // helper for topology plot: enumerate arcs that can be reached starting from the current node's children
    typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;
    // TODO: This should be a method of ComputationNetwork, not ComputationNode.
    void EnumerateArcs(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationArc>& arcs);

    // Helper for generating error messages and the like
    const std::wstring NodeDescription() const
    { 
        return std::wstring(L"Node '") + NodeName().c_str() + L"' (" + OperationName().c_str() + L" operation)"; 
    };

    // Helper that returns [a x b x c], including dynamic axes.
    const std::string ShapeDescription() const;

    bool IsValueSparse() const { return m_isValueSparse; }

    // debugging helper
    size_t m_uniqueNumericId; // (a unique handle for debugging)
protected:

    // -----------------------------------------------------------------------
    // data members
    // -----------------------------------------------------------------------

    // administrative
    DEVICEID_TYPE m_deviceId; // CPU=-1, >=0 GPU
    std::wstring m_nodeName;

    // inputs
    std::vector<ComputationNodeBasePtr> m_inputs;

    bool m_isValueSparse;

    // dimensions and layout
    // Data is stored as a Matrix object, but often it is interpreted as a tensor.
    // For nodes that carry data (samples), each sample is a column of the matrix, which is interpreted as
    // a tensor (n-dimensional array) described by m_sampleLayout. The MBLayout describes the meaning
    // of the column index.
    // For nodes that do not carry data, the last tensor index of m_sampleLayout is the number of columns.
    TensorShape m_sampleLayout; // sample layout
    MBLayoutPtr m_pMBLayout;

    // environment information
    // This structure is shared with the ComputationNetwork that this node lives in
    ComputationEnvironmentPtr m_environment;

    // flags related to gradient propagation
    float m_learningRateMultiplier;    // update parameters? Only used for LearnableParameters.    --TODO: Should we make this a member of LearnableParameters actually? And require a type cast? Currently it is read out for all leaves.
    bool m_gradientInitialized;        // indicates whether the gradient matrix has been resized and initialized to 0
    bool m_outputNeededDuringBackprop; // indicates whether the output value of the node is needed during backprop
};
typedef ComputationNodeBase::ComputationNodeBasePtr ComputationNodeBasePtr;

// =======================================================================
// NumInputs -- little helper interface to allow derived Node classes to 
// specify how many inputs they expect
// =======================================================================

struct INumInputs { virtual size_t GetExpectedNumInputs() const = 0; };
template <size_t m_numInputs>
struct NumInputs : public INumInputs // e.g. derive from NumInputs<2>
{
    size_t GetExpectedNumInputs() const override final
    {
        return m_numInputs;
    }
};

// =======================================================================
// AxisTransform -- Defines transformation along one axis. Currently, just
// scale and translation are supported.
// =======================================================================

struct AxisTransform
{
public:
    bool operator==(const AxisTransform& other) const
    {
        return (scale == other.scale) && (translate == other.translate);
    }

    bool operator!=(const AxisTransform& other) const
    {
        return !operator==(other);
    }

    // Scale along the axis (by default identity transform -> 1 scale).
    double scale = 1.0;
    // Translation along the axis (by default identity transform -> 0 translate).
    double translate = 0.0;
};

// =======================================================================
// SpaceTransform -- Combines several axis transforms into space transform.
// =======================================================================

struct SpaceTransform
{
public:
    SpaceTransform() {}

    // Returns all axis transforms.
    std::vector<AxisTransform>* GetTransform()
    {
        return &m_axisTransforms;
    }

    bool operator==(const SpaceTransform& other) const
    {
        CheckCompatibility(other);
        for (size_t i = 0; i < m_axisTransforms.size(); i++)
        {
            if (m_axisTransforms[i] != other.m_axisTransforms[i])
                return false;
        }
        return true;
    }

    bool operator!=(const SpaceTransform& other) const
    {
        return !operator==(other);
    }

    // Returns identity transform with given number of dimensions.
    static SpaceTransform Identity(int dimensions)
    {
        SpaceTransform result;
        result.m_axisTransforms.resize(dimensions);
        return result;
    }

    // Returns composition of this transform with given one (without modifying this one).
    SpaceTransform Compose(const SpaceTransform& other) const
    {
        CheckCompatibility(other);
        SpaceTransform result = SpaceTransform::Identity(m_axisTransforms.size());
        for (size_t ia = 0; ia < m_axisTransforms.size(); ia++)
        {
            result.m_axisTransforms[ia].scale     = m_axisTransforms[ia].scale * other.m_axisTransforms[ia].scale;
            result.m_axisTransforms[ia].translate = m_axisTransforms[ia].scale * other.m_axisTransforms[ia].translate + m_axisTransforms[ia].translate;
        }
        return result;
    }

    // Returns inverse of this transform without modifying it.
    SpaceTransform Inverse() const
    {
        SpaceTransform result = SpaceTransform::Identity(m_axisTransforms.size());
        for (size_t ia = 0; ia < m_axisTransforms.size(); ia++)
        {
            result.m_axisTransforms[ia].scale = 1 / m_axisTransforms[ia].scale;
            result.m_axisTransforms[ia].translate = -m_axisTransforms[ia].translate / m_axisTransforms[ia].scale;
        }
        return result;
    }

    // Check if this transform is compatible with given one.
    void CheckCompatibility(const SpaceTransform& other) const
    {
        // Transforms are compatible if they have same number of axis transforms.
        if (m_axisTransforms.size() != other.m_axisTransforms.size())
        {
            RuntimeError("Incompatible space transforms.");
        }
    }

    std::vector<AxisTransform> m_axisTransforms;
};

// =======================================================================
// TransformerNode -- Base class for all nodes that implement input-output
// transformation. Using individual node transformations one can calculate cumulative
// transformation between two nodes and establish spatial matching of its inputs or
// outputs. Node needs to provide its type and template argument (we use recurring
// template pattern to access number of inputs of the derived object).
// Note: This interface assumes that node also inherits from NumInputs<> class.
// =======================================================================

struct TransformerNode
{
public:
    TransformerNode() {}

    virtual ~TransformerNode() {}

    // Derived class needs to return if it supports transform computation between input at given index and output.
    virtual bool SupportsTransformOnInput(size_t index) = 0;

    // Derived class needs to compute transforms for all axes for all supported input-output paths (
    // (see SupportsTransformOnInput above) on this call.
    virtual void ComputeTransforms() = 0;

    // Derived classes need to inform us regarding number of inputs they have using this call before first
    // GetTransformForInput call.
    void SetNumberOfInputs(size_t inputsCount)
    {
        // Allocate appropriate number of transforms. Here transforms will be set to identity, node needs to compute
        // them during ComputeTransforms.
        m_transforms.resize(inputsCount);
    }

    // Handles transform accessing for all derive classes. Derived objects still need to
    // implement rest of ITransformerNode interface.
    const SpaceTransform& GetTransformForInput(size_t inputIndex)
    {
        if (m_transforms.empty())
            LogicError("No transforms present on GetTransformForInput call. Maybe SetNumberOfInputs has not been called?");

        // Check that we are within range.
        if (inputIndex >= m_transforms.size())
            RuntimeError("Invalid transform index in TransformerNode.");

        // Verify that derived object supports transform on given input.
        if (!SupportsTransformOnInput(inputIndex))
            RuntimeError("Space transform requested on unsupported input");

        // All good, ask derived object to compute transforms.
        ComputeTransforms();
        // Return transform for requested input.
        return m_transforms[inputIndex];
    }

protected:
    // Transforms for all node inputs.
    std::vector<SpaceTransform> m_transforms;
};

// =======================================================================
// IdentityTransformerNode -- Helper class for nodes that have identity
// transform for all inputs.
// =======================================================================

struct IdentityTransformerNode : public TransformerNode
{
private:
    using TransformerNode::m_transforms;

    // Set all transforms to identity.
    virtual void ComputeTransforms() override
    {
        if (m_transforms[0].m_axisTransforms.empty())
        {
            for (size_t it = 0; it < m_transforms.size(); it++)
            {
                m_transforms[it].m_axisTransforms.resize(2);
            }
        }
    }

    // Support transforms for all inputs.
    virtual bool SupportsTransformOnInput(size_t /*index*/) override { return true; }
};

// =======================================================================
// IdentityTransformerNodeOnOneInput -- Helper class for nodes that support
// identity transform for one input (defined with template argument).
// =======================================================================

template <size_t supportedInputIndex>
struct IdentityTransformerNodeOnOneInput : public TransformerNode
{
private:
    using TransformerNode::m_transforms;

    virtual void ComputeTransforms() override
    {
        if (m_transforms[supportedInputIndex].m_axisTransforms.empty())
        {
            // m_axisTransforms defaults to identity.
            m_transforms[supportedInputIndex].m_axisTransforms.resize(2);
        }
    }

    // Support transforms just one input.
    virtual bool SupportsTransformOnInput(size_t inputIndex) override
    {
        return (inputIndex == supportedInputIndex);
    }
};

// =======================================================================
// Nodes that can take a dynamic axis need to implement this.
// =======================================================================
struct ITakesDynamicAxis
{
    virtual const std::wstring GetRequestedDynamicAxis() const = 0;
};

// =======================================================================
// Nodes that have multiple outputs must derive from this.
// =======================================================================
template <typename ElemType>
struct MultiOutputNode
{
public:
    MultiOutputNode(size_t numOutputs)
        : m_numOutputs(numOutputs)
    {
        m_outputsShape.resize(m_numOutputs);
        m_outputsHasNewMBLayout.resize(m_numOutputs);
        m_outputsMBLayout.resize(m_numOutputs);
        m_outputsIsValueSparse.resize(m_numOutputs, false);
        m_outputsValue.resize(m_numOutputs);
        m_outputsGradient.resize(m_numOutputs);
    }

    size_t m_numOutputs;
    std::vector<TensorShape> m_outputsShape;
    std::vector<bool> m_outputsHasNewMBLayout;
    std::vector<std::shared_ptr<MBLayout>> m_outputsMBLayout;
    std::vector<bool> m_outputsIsValueSparse;
    std::vector<std::shared_ptr<Matrix<ElemType>>> m_outputsValue;
    std::vector<std::shared_ptr<Matrix<ElemType>>> m_outputsGradient;
};

// =======================================================================
// ComputationNode -- abstract base class for computation nodes, deriving
// from CompuationNodeBase, parameterized by float vs. double
// =======================================================================

template <class ElemType>
class ComputationNode : public ComputationNodeBase // abstract class that cannot be instantiated
{
    typedef ComputationNodeBase Base;

protected:

    // std containers such as list and map does not support class reference so we need to use pointer
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

public:

    using ComputationNodeBase::AttachInputs; // import the convenience functions that take 1..6 parameters
    using ComputationNodeBase::SetDims;
    typedef ElemType OurElemType;

    // -----------------------------------------------------------------------
    // construction, copying, (de-)serialization
    // -----------------------------------------------------------------------

    // public constructor
    // Note: use the New<> helper function that is declared next, which gives you the convenience of returning a shared_ptr
    ComputationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : ComputationNodeBase(deviceId, name)
    {
    }

    // recover a shared_ptr from ourselves if given a naked pointer
    ComputationNodePtr shared_from_this()
    {
        return dynamic_pointer_cast<ComputationNode<ElemType>>(ComputationNodeBase::shared_from_this());
    }

    // recover a ComputationNodePtr (which is a shared_ptr) from a naked pointer to our base type (ComputationNodeBase) stored as a void* (old NDL parser does that)
    static ComputationNodePtr FromVoidPtr(void* vp)
    {
        auto p = dynamic_cast<ComputationNode<ElemType>*>((ComputationNodeBase*)vp); // TODO: check that all void* casts really come from ComputationNodeBasePtr; or add a method ToVoidPtr(). Or get rid of the void*?!
        return p ? p->shared_from_this() : nullptr;
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = DownCast(nodeP);
            if (m_value)
            {
                node->CreateValueMatrixIfNull();
                node->m_value->SetValue(*m_value);
            }
            else
                node->m_value = nullptr;
            if (m_gradient)
            {
                node->CreateGradientMatrixIfNull();
                node->m_gradient->SetValue(*m_gradient);
            }
            else
                node->m_gradient = nullptr;
        }
    }

    // duplicate a node
    // Create a copy of a ComputationNode object. Inputs will be shared. Values (and gradients if applicable) are copied.
    ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        const std::wstring& name = (newName == L"") ? NodeName() : newName;
        ComputationNodeBasePtr node(NewThis(m_deviceId, name));  // NewThis() is a virtual function that creates a new node of the actual type of 'this'
        CopyTo(node, name, flags);
        return node;
    }

    // creation from configuration
    // Nodes with NumInputs<> should say DeclareConstructorFromConfigWithNumInputs(ClassName), and nodes without DeclareConstructorFromConfig(ClassName).
    // The macro will forward to the regular constructor of the node (which may do more than just calling the base constructor), and then attach the inputs from config.
#define DeclareConstructorFromConfigWithNumInputs(C)                     \
    C(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp) \
        : C(configp->Get(L"deviceId"), L"<placeholder>")                 \
    {                                                                    \
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());   \
    }
#define DeclareConstructorFromConfig(C)                                  \
    C(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp) \
        : C(configp->Get(L"deviceId"), L"<placeholder>")                 \
    {                                                                    \
        AttachInputsFromConfig(configp);                                 \
    }

    // helper to load m_value from a stream
    // This function updates the dimensions to a 2D matrix.
    // If a different tensor layout is associated with this, it must be implanted afterwards.
    // Nodes that call this never have an MB layout.
    void LoadValue(File& fstream)
    {
        CreateMatrixIfNull(m_value);
        fstream >> Value();
        // above reads dimensions, so we must update our own dimensions
        SetDims(TensorShape(Value().GetNumRows(), Value().GetNumCols()), false);
    }

    // reader updated m_functionValue and MBLayout--ensure our internal state is consistent
    virtual void NotifyFunctionValuesMBSizeModified() override final
    {
        if (!HasMBLayout())
            LogicError("NotifyFunctionValuesMBSizeModified: Must only be called on nodes with MBLayout.");
        if (GetSampleMatrixNumRows() != Value().GetNumRows())
            LogicError("NotifyFunctionValuesMBSizeModified: %ls %ls operation had its row dimension %d changed by the reader to %d.", NodeName().c_str(), OperationName().c_str(), (int) GetSampleMatrixNumRows(), (int) Value().GetNumRows());
        if (GetMBLayout()->GetNumCols() != Value().GetNumCols())
            LogicError("NotifyFunctionValuesMBSizeModified: %ls %ls operation had its col dimension %d changed by the reader to %d, but different from MBLayout.", NodeName().c_str(), OperationName().c_str(), (int) GetMBLayout()->GetNumCols(), (int) Value().GetNumCols());
    }

    // -----------------------------------------------------------------------
    // inputs
    // -----------------------------------------------------------------------

    // AttachInputs() -- attach the inputs of a node
    // This verifies the number of inputs. For that, nodes with fixed number of inputs derive from NumInputs<N>.
    // This function discovers this through RTTI and performs a runtime check. Nodes should not have additional checks in their implementation (save the code).
    // Note: Nodes with variable number of inputs will not derive from NumInputs<>, but instead check their inputs in Validate().
    void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs)
    {
#ifdef _DEBUG
        wstring name = NodeName();
        name; // (for easier debugging)
#endif
        ClearConfigMemberCache();
        const auto* pNumInputs = dynamic_cast<INumInputs*>(this); // if this class also derives from NumInputs<N> then N is the expected number of inputs
        if (pNumInputs && pNumInputs->GetExpectedNumInputs() != inputs.size())
            RuntimeError("%ls operation '%ls' expects %d inputs (given: %d)", OperationName().c_str(), NodeName().c_str(), (int) pNumInputs->GetExpectedNumInputs(), (int) inputs.size());
        m_inputs.resize(inputs.size());
        for (size_t i = 0; i < m_inputs.size(); i++)
            if (inputs[i])
                m_inputs[i] = DownCast(inputs[i]); // (DownCast() checks the type; the assignment then downcasts it again)
            else
                m_inputs[i] = nullptr; // during network creation, nullptrs are possible

        // If this object implements also TransformerNode interface we need to notify it about number of inputs.
        if (Is<TransformerNode>())
        {
            auto transformerNode = As<TransformerNode>();
            transformerNode->SetNumberOfInputs(m_inputs.size());
        }
    }

protected:

    // AttachInputs() from config
    void AttachInputsFromConfig(const ScriptableObjects::IConfigRecordPtr configp, size_t expectedNumInputs = SIZE_MAX)
    {
        const auto inputs = GetInputsFromConfig(configp);
        if (expectedNumInputs != SIZE_MAX)
        {
            if (inputs.size() != expectedNumInputs)
            {
                // print an error. For that, find at least one argument
                auto* val = configp->Find(L"inputs");
                if (!val) // if there is no 'inputs' then get the first item of this config record for a Fail() function
                {
                    auto members2 = configp->GetMemberIds();
                    if (members2.size() > 0)
                        val = configp->Find(members2.front());
                }
                if (val)
                    val->Fail(msra::strfun::wstrprintf(L"Expected %d inputs, but %d were given.", (int) expectedNumInputs, (int) inputs.size()));
                else
                    InvalidArgument("Expected %d inputs, but %d were given.", (int) expectedNumInputs, (int) inputs.size());
            }
        }
        AttachInputs(inputs);
    }

    // up-cast to make life easier
    static ComputationNodePtr DownCast(ComputationNodeBasePtr inode)
    {
        ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(inode);
        if (!node)
            InvalidArgument("an ComputationNodeBasePtr of mismatching precision was passed");
        return node;
    }

    inline ComputationNodePtr Input(const size_t inputIndex) const
    {
        if (inputIndex >= m_inputs.size())
            LogicError("Inputs: inputIndex %d is out of range for %ls %ls operation.", (int) inputIndex, NodeName().c_str(), OperationName().c_str());
        return DownCast(m_inputs[inputIndex]);
    }

    // Fast downcast without runtime type check of dynamic_pointer_cast.
    // Meant to be used in Forward and BackPropTo, assuming that Validate() has already used Input() which validated the correct types.
    inline ComputationNode<ElemType>& InputRef(const size_t inputIndex) const
    {
        return static_cast<ComputationNode<ElemType>&>(*m_inputs[inputIndex].get());
    }

    void /*ComputationNodeBase::*/ SetInput(const size_t childIndex, const ComputationNodeBasePtr& inode) override
    {
        ClearConfigMemberCache();

        const ComputationNodePtr node = DownCast(inode);

        // require first nodes specified before the second to avoid null nodes condition.
        if (childIndex > m_inputs.size())
            InvalidArgument("SetInput: You must specify the input for children with index less than this one first.");

        // expand the inputs to exist up to the desired index
        while (childIndex >= m_inputs.size())
            m_inputs.push_back(nullptr);

        // set the input value
        m_inputs[childIndex] = node;
    }

public:

    // -----------------------------------------------------------------------
    // validation
    // -----------------------------------------------------------------------

    void ValidateInferInputDimsFrom(const TensorShape& otherShape) override final;

    // -----------------------------------------------------------------------
    // masking
    // -----------------------------------------------------------------------

    static void MaskMissingColumnsToZero(Matrix<ElemType>& matrixToBeMasked, const MBLayoutPtr& pMBLayout, const FrameRange& fr)
    {
        // fprintf(stderr, "masking column range %d\n", (int)fr.timeIdxInSeq);
        MaskMissingColumnsTo(matrixToBeMasked, pMBLayout, fr, (ElemType) 0);
    }

    void /*ComputationNodeBase::*/ MaskMissingValueColumnsToZero(const FrameRange& fr) override final
    {
        // fprintf(stderr, "%ls %ls m_value ", NodeName().c_str(), OperationName().c_str());
        MaskMissingColumnsToZero(*m_value, m_pMBLayout, fr);
    }
    void /*ComputationNodeBase::*/ MaskMissingGradientColumnsToZero(const FrameRange& fr) override final
    {
        // fprintf(stderr, "%ls %ls m_gradient ", NodeName().c_str(), OperationName().c_str());
        MaskMissingColumnsToZero(*m_gradient, m_pMBLayout, fr);
    }

    // for index vectors: Invalid entries must be set to -1.
    void MaskMissingValueColumnsTo(const FrameRange& fr, ElemType val)
    {
        MaskMissingColumnsTo(*m_value, m_pMBLayout, fr, val);
    }

    // for debugging, set the gaps to NaN instead (to track whether it bubbles up somewhere)
    void InvalidateMissingValueColumns(const FrameRange& fr) override final
    {
        if (m_value->GetMatrixType() != SPARSE) // Sparse matrices can only be masked with 0s
            MaskMissingColumnsTo(*m_value, m_pMBLayout, fr, Matrix<ElemType>::MakeNan(__LINE__));
    }
    void InvalidateMissingGradientColumns(const FrameRange& fr) override final
    {
        if (m_gradient->GetMatrixType() != SPARSE) // Sparse matrices can only be masked with 0s
            MaskMissingColumnsTo(*m_gradient, m_pMBLayout, fr, Matrix<ElemType>::MakeNan(__LINE__));
    }

    static TensorView<ElemType> Unpack(const TensorShape& sampleShape,
                                       const Matrix<ElemType>& packedData,                                       
                                       const MBLayoutPtr& layout,
                                       const std::shared_ptr<Matrix<ElemType>>& unpackedDataStorage,
                                       const std::shared_ptr<Matrix<ElemType>>& tempIndicesStorage,
                                       const std::shared_ptr<Matrix<char>>& tempMaskStorage,
                                       bool batchMajor,
                                       const ElemType* gapPadValue);

    static TensorView<ElemType> Unpack(const TensorShape& sampleShape,
                                       const Matrix<ElemType>& packedData,
                                       const MBLayoutPtr& layout,
                                       bool batchMajor,
                                       const ElemType* gapPadValue)
    {
        auto nullSharedPtr = std::shared_ptr<Matrix<ElemType>>(nullptr);
        return Unpack(sampleShape, packedData, layout, nullSharedPtr, nullSharedPtr, std::shared_ptr<Matrix<char>>(nullptr), batchMajor, gapPadValue);
    }

    static void BroadcastToPacked(const Matrix<ElemType>& dataToBroadcast,
                                  const MBLayoutPtr& inputLayout,
                                  ElemType beta,
                                  Matrix<ElemType>& broadcastTo,
                                  const FrameRange& targetFrameRange,
                                  const std::shared_ptr<Matrix<ElemType>>& tempIndicesStorage);

    // -----------------------------------------------------------------------
    // accessors for value and gradient
    // -----------------------------------------------------------------------

    const Matrix<ElemType>& Value() const { return *m_value; }
    Matrix<ElemType>&       Value()       { return *m_value; }

    MatrixBasePtr ValuePtr() const override final { return m_value; }    // readers want this as a shared_ptr straight
    std::shared_ptr<Matrix<ElemType>>& ValuePtrRef() { return m_value; }

    // Note: We cannot return a const& since returning m_value as a MatrixBasePtr is a type cast that generates a temporary. Interesting.

    const Matrix<ElemType>& Gradient() const { return *m_gradient; }
    Matrix<ElemType>&       Gradient()       { return *m_gradient; }

    MatrixBasePtr GradientPtr() const { return m_gradient; }
    std::shared_ptr<Matrix<ElemType>>& GradientPtrRef() { return m_gradient; }
    // TODO: This is only used for testing whether a gradient has been allocated. Maybe reduce to bool HasGradient()?

    MatrixType GetPreferredGradientMatrixType() { return m_preferredGradientMatrixType; }
    void SetPreferredGradientMatrixType(MatrixType requestType) { m_preferredGradientMatrixType = requestType; }

private:

    template<class E>
    void RethrowAs(const std::exception & e, const std::string & what) const
    {
        const auto * pe = dynamic_cast<const ExceptionWithCallStack<E> *>(&e);
        if (pe)
            throw ExceptionWithCallStack<E>(what, pe->CallStack());
        else if (dynamic_cast<const E *>(&e))
            throw E(what);
    }

    // rethrow an exception with added node-name information
    // Use this for exceptions we may get e.g. from the Matrix library, such as VerifySize().
    __declspec_noreturn
    void Rethrow(const std::exception & e) const
    {
        string what = msra::strfun::strprintf("%ls: %s", NodeDescription().c_str(), e.what());
        RethrowAs<std::runtime_error>   (e, what);
        RethrowAs<std::logic_error>     (e, what);
        RethrowAs<std::invalid_argument>(e, what);
        //RethrowAs<std::bad_alloc>       (e, what); // can't throw with message
        //RethrowAs<std::exception>       (e, what); // ditto
        throw e;
    }

    // map a tensor to a matrix
    // The leading dimension maps to rows, the rest to columns, for compat with sparse matrix lib.
    Matrix<ElemType>& TensorAsMatrix(Matrix<ElemType>& data)
    {
        size_t numRows = GetAsMatrixNumRows();
        size_t numCols = GetAsMatrixNumCols();
        // We only get here if the tensor indeed describes an 1D or 2D object. In that case, just verify the dimensions.
        try
        {
        data.VerifySize(numRows, numCols);
        }
        catch (const std::exception& e)
        {
            Rethrow(e);
        }
        return data;
    }

public:

    Matrix<ElemType>& ValueAsMatrix() { return TensorAsMatrix(*m_value); }
    Matrix<ElemType>& GradientAsMatrix() { return TensorAsMatrix(*m_gradient); }

    // function to access any input and output, value and gradient, whole batch or single frame
    // Note: This returns a reference into 'data' in the form of a column slice, i.e. a small matrix object that just points into 'data'.
    Matrix<ElemType> DataFor(Matrix<ElemType>& data, const FrameRange& fr /*select frame or entire batch*/)
    {
        try
        {
            return DataWithMBLayoutFor(data, fr, m_pMBLayout);
        }
        catch (const std::exception& e) // catch the error and rethrow it with the node name attached
        {
            Rethrow(e);
        }
    }
#if 0
    Matrix<ElemType> DataFor(const Matrix<ElemType>& data, const FrameRange& fr /*select frame or entire batch*/) const
    {
        return const_cast<ComputationNode<ElemType>*>(this)->DataFor(const_cast<Matrix<ElemType>&>(data), fr);
    }
#endif

    Matrix<ElemType> ValueFor   (const FrameRange& fr /*select frame or entire batch*/)       { return DataFor(Value(),    fr); }
    Matrix<ElemType> GradientFor(const FrameRange& fr /*select frame or entire batch*/)       { return DataFor(Gradient(), fr); }
#if 0 // causes grief with gcc
    Matrix<ElemType> ValueFor   (const FrameRange& fr /*select frame or entire batch*/) const { return DataFor(Value(),    fr); }
    Matrix<ElemType> GradientFor(const FrameRange& fr /*select frame or entire batch*/) const { return DataFor(Gradient(), fr); }
#endif
    // use the following two versions if you assume the inputs may contain gaps that must be set to zero because you want to reduce over frames with a BLAS operation
    Matrix<ElemType> MaskedValueFor(const FrameRange& fr /*select frame or entire batch*/)
    {
        MaskMissingValueColumnsToZero(fr);
        return ValueFor(fr);
    }
    Matrix<ElemType> MaskedGradientFor(const FrameRange& fr /*select frame or entire batch*/)
    {
        MaskMissingGradientColumnsToZero(fr);
        return GradientFor(fr);
    }
    // tensor version of the above functions
    TensorView<ElemType> DataTensorFor(const MatrixBasePtr& data, size_t rank, const FrameRange& fr) const
    {
        try
        {
            return TensorView<ElemType>(data, GetTensorSliceFor(rank, fr));
        }
        catch (const std::exception& e) // catch the error and rethrow it with the node name attached
        {
            Rethrow(e);
        }
    }
    TensorView<ElemType> ValueTensorFor(size_t rank, const FrameRange& fr)
    {
        return DataTensorFor(ValuePtr(), rank, fr);
    }
    TensorView<ElemType> GradientTensorFor(size_t rank, const FrameRange& fr)
    {
        return DataTensorFor(GradientPtr(), rank, fr);
    }

    // TODO: Are all these meant to read out a scalar? Then rename and verify dimensions.
    virtual double Get00Element() const override final { return Value().Get00Element(); }

    // -----------------------------------------------------------------------
    // dimensions and allocation
    // -----------------------------------------------------------------------

    // update temporary variables of a node to match MBLayout
    virtual void UpdateFunctionMBSize() override
    {
    }

protected:

    // determine the size that we should set our Matrix storage to
    void DetermineDataSize(size_t& rows, size_t& cols) const
    {
        if (m_isValueSparse && HasMBLayout())
        {
            const auto& shape = GetSampleLayout();
            size_t rank = shape.GetRank();
            rows = rank > 0 ? shape[0] : 1;

            // TODO: TensorShape should have a method to 
            // easily compute size of subshapes
            cols = 1;
            for (size_t k = 1; k < rank; k++)   // all dimensions except leading one
                cols *= shape[k];

            cols *= GetMBLayout()->GetNumCols();
        }
        else
        {
            if (HasMBLayout())
            {
                rows = GetSampleMatrixNumRows();
                cols = GetSampleMatrixNumCols();
            }
            else
            {
                const auto& shape = GetSampleLayout();
                size_t rank = shape.GetRank();
                rows = rank > 0 ? shape[0] : 1;
                cols = 1;
                for (size_t k = 1; k < rank; k++)   // all dimensions except leading one
                    cols *= shape[k];
            }
        }
    }

protected:

    // set the size of the underlying Matrix object to match node dimensions
    void UpdateDataSize(Matrix<ElemType>& m)
    {
        size_t rows, cols;
        DetermineDataSize(rows, cols);
        m.Resize(rows, cols);
    }
    // and verify the condition that UpdateDataSize() creates (used for sanity checking after loading parameters)
    void VerifyDataSize(Matrix<ElemType>& m)
    {
        size_t rows, cols;
        DetermineDataSize(rows, cols);
        try
        {
            m.VerifySize(rows, cols);
        }
        catch (const std::exception& e)
        {
            Rethrow(e);
        }
    }

public:
    // update the actual matrix allocation for m_value based on the node dimension
    void UpdateFunctionValuesSize()
    {
        UpdateDataSize(Value());
        Value().CollapseDataLocation();
    }

    // -----------------------------------------------------------------------
    // forward propagation, backpropagation
    // -----------------------------------------------------------------------

    // this is called before a node's ForwardProp() function is called (in loops: for the first time)
    // This is where we
    //  - update the node dimension based on actual MB size
    //  - (re-)allocate the m_value matrix, which may be shared across nodes and thus have changed dimensions
    virtual void /*IComputationNode::*/ BeginForwardProp() override; // called before first iteration step of ForwardProp()

    virtual void /*IComputationNode::*/ EndForwardProp() override;

    virtual void /*IComputationNode::*/BeginBackprop() override;

    virtual void /*IComputationNode::*/ EndBackprop() override;

    // this is the entry point from Network; while it will call virtual BackpropTo() into the actual node implementation
    // TODO: move to -Base (or -Network?)
    void Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) override;

    // lazy resetting of gradient
    // This performs the actual zeroing out.
    void LazyZeroGradient()
    {
        if (!m_needsGradient)
            LogicError("%ls %ls operation: LazyZeroGradient() called although this node needs no gradient.", NodeName().c_str(), OperationName().c_str());

        if (m_gradientInitialized)
            return;

        ResetGradient(0);
    }

    // resize and reset this node's gradient to a given value (normally 0, 1 for root)
    void ResetGradient(ElemType val)
    {
        UpdateDataSize(Gradient());

        // No need to zero initialize the gradient if the node's parent is going to overwrite it anyways
        if ((val != 0) || !ParentOverwritesGradient())
            Gradient().SetValue(val);

        m_gradientInitialized = true;
    }

    // Assign the given matrix's value to this node's gradient. The matrix sizes must match.
    void AssignGradient(const Matrix<ElemType>& val)
    {
        UpdateDataSize(Gradient());

        // The specified value matrix's dimensions must match the gradient matrix dimensions
        if ((val.GetNumRows() != Gradient().GetNumRows()) || (val.GetNumCols() != Gradient().GetNumCols()))
            LogicError("%ls %ls operation: The value matrix specified for ResetGradient() does not match the dimensions of the gradient matrix.", NodeName().c_str(), OperationName().c_str());

        Gradient().AssignValuesOf(val);

        m_gradientInitialized = true;
    }

    // -----------------------------------------------------------------------
    // memory sharing
    // -----------------------------------------------------------------------

    // helper function for formatting memory sharing information
    // TODO: customize this function for all nodes that uses temp internal matrices.
    virtual std::set<std::pair<const MatrixBase*, std::wstring>> GetMatrixInfo() const override
    {
        std::set<std::pair<const MatrixBase*, std::wstring>> matrixInfo;
        matrixInfo.insert    (make_pair(ValuePtr().get(),    NodeName() + L" : " + msra::strfun::utf16(ShapeDescription())));
        if (GradientPtr())
            matrixInfo.insert(make_pair(GradientPtr().get(), NodeName() + L" : " + msra::strfun::utf16(ShapeDescription()) + L" (gradient)"));
        return matrixInfo;
    }

    // request matrices needed to do node function value evaluation
    // for memory pool utilization optimizaiton, the requested pointer is not immediately useable until the entire network has gone through all requests 
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        size_t matrixSize = m_sampleLayout.GetNumElements();
        if (IsValueSharable() && !m_isValueSparse)
            RequestMatrixFromPool(m_value, matrixPool, matrixSize, HasMBLayout());
        else
            CreateMatrixIfNull(m_value);

        auto multiOutputNode = dynamic_cast<MultiOutputNode<ElemType>*>(this);
        if (multiOutputNode)
        {
            for (size_t i = 1; i < multiOutputNode->m_numOutputs; ++i)
            {
                if (!multiOutputNode->m_outputsIsValueSparse[i])
                    RequestMatrixFromPool(multiOutputNode->m_outputsValue[i], matrixPool, multiOutputNode->m_outputsShape[i].GetNumElements(), multiOutputNode->m_outputsMBLayout[i] != nullptr);
                else
                    CreateMatrixIfNull(multiOutputNode->m_outputsValue[i]);
            }
        }
    }

    // release temp matrices that are only used by forward computation
    // don't release matrices that need to be used in the gradient computation
    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        if (!IsOutputNeededDuringBackprop() && !m_isValueSparse && IsValueSharable())
            ReleaseMatrixToPool(m_value, matrixPool);
    }

    virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool) override
    {
        for (int i = 0; i < m_inputs.size(); i++)
        {
            if (m_inputs[i]->NeedsGradient())
                m_inputs[i]->RequestMatricesBeforeBackprop(matrixPool);
        }
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        size_t matrixSize = m_sampleLayout.GetNumElements();
        RequestMatrixFromPool(m_gradient, matrixPool, matrixSize, HasMBLayout());

        auto multiOutputNode = dynamic_cast<MultiOutputNode<ElemType>*>(this);
        if (multiOutputNode)
        {
            for (size_t i = 1; i < multiOutputNode->m_numOutputs; ++i)
                RequestMatrixFromPool(multiOutputNode->m_outputsGradient[i], matrixPool, multiOutputNode->m_outputsShape[i].GetNumElements(), multiOutputNode->m_outputsMBLayout[i] != nullptr);
        }
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        if (!IsLeaf() && !RequiresPreCompute())
        {
            if (m_gradient != nullptr && m_gradient->GetMatrixType() != SPARSE) // since we don't have a sparse pool yet
                ReleaseMatrixToPool(m_gradient, matrixPool);

            // Release the Value matrix only if the output value is needed during backprop
            // since in the case it isn't used, we release it during forward prop itself
            if (IsOutputNeededDuringBackprop() && !m_isValueSparse && IsValueSharable())
                ReleaseMatrixToPool(m_value, matrixPool);

            auto multiOutputNode = dynamic_cast<MultiOutputNode<ElemType>*>(this);
            if (multiOutputNode)
            {
                for (size_t i = 1; i < multiOutputNode->m_numOutputs; ++i)
                    ReleaseMatrixToPool(multiOutputNode->m_outputsGradient[i], matrixPool);

                for (size_t i = 1; i < multiOutputNode->m_numOutputs; ++i)
                {
                    if (!multiOutputNode->m_outputsIsValueSparse[i])
                        ReleaseMatrixToPool(multiOutputNode->m_outputsValue[i], matrixPool);
                }
            }
        }
    }

    void CreateValueMatrixIfNull()
    {
        CreateMatrixIfNull(m_value);
    }

    void CreateGradientMatrixIfNull()
    {
        CreateMatrixIfNull(m_gradient);
    }

    void MarkValueNonSharable() override
    {
        m_valueSharable = false;
        CreateMatrixIfNull(m_value);
    }

protected:

    // this function is used to create matrices for those needed before matrix pool is available
    // e.g., for model parameters and input nodes you will need to resize the functions based on NDL
    // and before matrix pool is available
    void CreateMatrixIfNull(shared_ptr<Matrix<ElemType>>& matrixPtr)
    {
        if (!matrixPtr)
            matrixPtr = make_shared<Matrix<ElemType>>(m_deviceId);
    }

    // matrixSize is per sample size, if unknown or hard to estimate, set matrixSize = 0
    // if the matrix's size will scale with minibatch size, set mbScale = true 
    // if workspace flag is true, the memory request will be treated specially. We assume workspace memory will share their own pointers 
    // this is currently a workaround for workspace memory for convolutions
    void RequestMatrixFromPool(shared_ptr<Matrix<ElemType>>& matrixPtr, MatrixPool& matrixPool, size_t matrixSize=0, bool mbScale=false, bool isWorkSpace=false)
    {
        if (matrixPtr == nullptr)
        {
            matrixPool.RequestAllocate<ElemType>(m_deviceId, &matrixPtr, matrixSize, mbScale, isWorkSpace);
        }
    }

    void ReleaseMatrixToPool(shared_ptr<Matrix<ElemType>>& matrixPtr, MatrixPool& matrixPool)
    {
        assert(matrixPtr != nullptr);
        matrixPool.RequestRelease<ElemType>(&matrixPtr);
    }

public:
    // -----------------------------------------------------------------------
    // miscellaneous
    // -----------------------------------------------------------------------

    virtual void DumpNodeInfo(const bool /*printValues*/, const bool /*printMetadata*/, File& fstream) const;

    // helper for SimpleOutWriter, living in here to be able to use in debugging
    void WriteMinibatchWithFormatting(FILE* f, const FrameRange& fr, size_t onlyUpToRow, size_t onlyUpToT, bool transpose, bool isCategoryLabel, bool isSparse,
                                      const std::vector<std::string>& labelMapping, const std::string& sequenceSeparator, 
                                      const std::string& sequencePrologue, const std::string& sequenceEpilogue, const std::string& elementSeparator,
                                      const std::string& sampleSeparator, std::string valueFormatString,
                                      bool outputGradient = false, bool onlyShowAbsSumForDense = false,
                                      std::function<std::string(size_t)> getKeyById = std::function<std::string(size_t)>()) const;

    // simple helper to log the content of a minibatch
    void DebugLogMinibatch(bool outputGradient = false) const
    {
        fprintf(stderr, "<<<<<<\n"); // some prologue and epilogue so that we can use diff -c1 to see the node name
        fprintf(stderr, "<<<<<<\n");
        fprintf(stderr, "DebugLogMinibatch: <<<<< %ls%s >>>>>\n", NodeName().c_str(), outputGradient ? " (gradient)" : "");
        WriteMinibatchWithFormatting(stderr, FrameRange(), 8, 10, false/*transpose*/, /*isCategoryLabel=*/false, /*isSparse=*/false, std::vector<std::string>(),
            ""/*sequenceSeparator*/, "  "/*sequencePrologue*/, "\n"/*sequenceEpilogue*/, " "/*elementSeparator*/, "\n  "/*sampleSeparator*/,
            "%.8f"/*valueFormatString*/, outputGradient);
        fprintf(stderr, ">>>>>>\n");
        fprintf(stderr, ">>>>>>\n");
    }

    void Trace()
    {
        //DebugLogMinibatch();
#if 0
        static const std::set<std::wstring> toLog{
            L"labelSentenceStartEmbedded",
            L"delayedDecoderFeedback.h.x",
            L"delayedDecoderFeedback.h.flags",
            L"delayedDecoderFeedback.h.out.thenVal.h.indexSequence.h.indexSequence.h",
            L"delayedDecoderFeedback.h.out.thenVal.h.indexSequence.h",
            L"delayedDecoderFeedback.h.out.thenVal.h",
            L"delayedDecoderFeedback.h.out.PlusArgs[0]",
            L"delayedDecoderFeedback.h.out.PlusArgs[1].ElementTimesArgs[0]",
            L"delayedDecoderFeedback.h.out.elseVal",
            L"delayedDecoderFeedback.h.out.PlusArgs[1]",
            L"delayedDecoderFeedback.h.out",
            L"delayedDecoderFeedback"
        };
        if (toLog.find(NodeName()) != toLog.end())
            DebugLogMinibatch();
        if (NodeName() == L"delayedDecoderFeedback.h.out")
        {
            static int i = 0;
            if (++i == 2)
                exit(1);
        }
#endif
        if (m_traceNodeValueReal || m_traceNodeValueAsCategoryLabel || m_traceNodeValueSparse)
        {
            fprintf(stderr, "Trace --> %s\n", FormatOperationPrototype("").c_str());
            if (m_traceNodeValueReal)
                WriteMinibatchWithFormatting(stderr, FrameRange(), m_traceNodeValueUpToDim, m_traceNodeValueUpToT, false/*transpose*/, /*isCategoryLabel=*/false, /*isSparse=*/false, std::vector<std::string>(),
                                             ""/*sequenceSeparator*/, "  "/*sequencePrologue*/, "\n"/*sequenceEpilogue*/, " "/*elementSeparator*/, "\n  "/*sampleSeparator*/,
                                             "%13.10f"/*valueFormatString*/);
            if (m_traceNodeValueAsCategoryLabel)
                WriteMinibatchWithFormatting(stderr, FrameRange(), m_traceNodeValueUpToDim, m_traceNodeValueUpToT, false/*transpose*/, /*isCategoryLabel=*/true,  /*isSparse=*/false, std::vector<std::string>(),
                                             ""/*sequenceSeparator*/, "  "/*sequencePrologue*/, "\n"/*sequenceEpilogue*/, " "/*elementSeparator*/, "\n  "/*sampleSeparator*/,
                                             "%13.10f"/*valueFormatString*/);
            if (m_traceNodeValueSparse)
                WriteMinibatchWithFormatting(stderr, FrameRange(), SIZE_MAX,                SIZE_MAX,              false/*transpose*/, /*isCategoryLabel=*/false, /*isSparse=*/true, std::vector<std::string>(),
                                         ""/*sequenceSeparator*/, "  "/*sequencePrologue*/, "\n"/*sequenceEpilogue*/, " "/*elementSeparator*/, "\n  "/*sampleSeparator*/,
                                         "%13.10f"/*valueFormatString*/);
        }
    }

protected:
    // print node values
    // This is used for dumping model parameters, not minibatch data.
    void PrintNodeValuesToFile(const bool printValues, const bool printMetadata, File& fstream) const
    {
        if (printValues)
        { 
            if (printMetadata)
            {
                fstream << wstring(L"\n");
            }
            const Matrix<ElemType>& m = Value();
            for (size_t i = 0; i < m.GetNumRows(); i++)
            {
                for (size_t j = 0; j < m.GetNumCols(); j++)
                {
                    fstream << m(i, j);
                }
                fstream << wstring(L"\n");
            }
            if (printMetadata)
            {
                fstream << wstring(L"####################################################################");
            }
        }
    }

public:

    // TODO: similar to DumpInfo; used by ExperimentalNetworkBuilder test implementation
    /*HasToString::*/ wstring ToString() const override
    {
        // we format it like "name : type rows x cols ( args )"
        wstring result = NodeName() + L" : " + OperationName();
        result.append(msra::strfun::wstrprintf(L" [%s%ls]", string(GetSampleLayout()).c_str(), GetMBLayoutAxisString().c_str()));
        if (m_inputs.empty())
            result.append(L" ()");
        else
        {
            wstring args;
            bool first = true;
            for (auto& child : m_inputs)
            {
                if (first)
                    first = false;
                else
                    args.append(L"\n");
                args.append(/*TidyName*/ (child->NodeName()));
            }
            result += L" " + NestString(args, L'(', true, ')');
        }
        return result;
    }

    // for debugging purposes
    void /*ComputationNodeBase::*/ PrintSelf(bool printMatrices = false) const
    {
        fprintf(stderr, "\n%ls[%s%ls] = %ls", NodeName().c_str(), string(GetSampleLayout()).c_str(), GetMBLayoutAxisString().c_str(), OperationName().c_str());

        if (!IsLeaf())
        {
            fprintf(stderr, "(");
            for (size_t i = 0; i < GetNumInputs(); i++)
            {
                if (i > 0)
                    fprintf(stderr, ", ");
                fprintf(stderr, "%ls[%s%ls] = %ls", m_inputs[i] ? m_inputs[i]->NodeName().c_str() : L"NULL", string(m_inputs[i]->GetSampleLayout()).c_str(), m_inputs[i]->GetMBLayoutAxisString().c_str(), OperationName().c_str());
            }
            fprintf(stderr, ")");
        }

        if (printMatrices)
        {
            fprintf(stderr, "\n    $$$$ Function Values\n");
            Value().Print("FunctionValue");

            fprintf(stderr, "\n    $$$$ Gradient Values\n");
            Gradient().Print("GradientValue");
        }
    }

    // NOTE: we should reimplement this to be thread-safe and use a larger than requested initialized memory block
    // we can then just wrap that memory block in a matrix of the correct dimensions since it will be const no one can change it
    // should only need one memory block per device
    // Thread-safety could be achieved by changing this to a shared_ptr.
    // When using the TensorView interface, one could instead just use a 1x1 matrix with a view that broadcasts its columns (stride 0).
    static const Matrix<ElemType>& ConstOnes(const size_t rows, const size_t cols, const DEVICEID_TYPE deviceId)
    {
        if (s_constOnes.find(rows) == s_constOnes.end() ||
            s_constOnes[rows].find(cols) == s_constOnes[rows].end()) // not found
        {
            shared_ptr<Matrix<ElemType>> matrix = make_shared<Matrix<ElemType>>(rows, cols, (DEVICEID_TYPE) deviceId);
            matrix->SetValue(1);
            s_constOnes[rows][cols] = matrix;
        }

        shared_ptr<Matrix<ElemType>> m = s_constOnes[rows][cols];
        m->TransferFromDeviceToDevice(m->GetDeviceId(), deviceId);

        return *m;
    }

    // -----------------------------------------------------------------------
    // data members
    // -----------------------------------------------------------------------

protected:

    shared_ptr<Matrix<ElemType>> m_value, m_gradient;

    static std::map<size_t, std::map<size_t, shared_ptr<Matrix<ElemType>>>> s_constOnes;

    MatrixType m_preferredGradientMatrixType = UNDETERMINED;
};

// convenience wrapper for ComputationNode::New()
template <class C, class... _Types>
inline shared_ptr<C> New(_Types&&... _Args)
{
    return make_shared<C>(forward<_Types>(_Args)...);
}

// helper class for parsing parameters for WriteMinibatchWithFormatting() below
// pass this to WriteOutput() (to file-path, below) to specify how the output should be formatted
struct WriteFormattingOptions
{
    // How to interpret the data:
    bool isCategoryLabel = false;  // true: find max value in column and output the index instead of the entire vector
    std::wstring labelMappingFile; // optional dictionary for pretty-printing category labels
    bool isSparse = false;
    bool transpose = true;         // true: one line per sample, each sample (column vector) forms one line; false: one column per sample
    // The following strings are interspersed with the data:
    // overall
    std::string prologue; // print this at the start (e.g. a global header or opening bracket)
    std::string epilogue; // and this at the end
    // sequences
    std::string sequenceSeparator; // print this between sequences (i.e. before all sequences but the first)
    std::string sequencePrologue;  // print this before each sequence (after sequenceSeparator)
    std::string sequenceEpilogue;  // and this after each sequence
    // elements
    std::string elementSeparator;  // print this between elements on a row
    std::string sampleSeparator;   // and this between rows
    // Optional printf precision parameter:
    std::string precisionFormat;        // printf precision, e.g. ".2" to get a "%.2f"

    WriteFormattingOptions() : // TODO: replace by initializers?
        isCategoryLabel(false), transpose(true), sequenceEpilogue("\n"), elementSeparator(" "), sampleSeparator("\n")
    { }

    template <class ConfigRecordType>
    WriteFormattingOptions(const ConfigRecordType& config);

    void Save(File& fstream) const;
    void Load(File& fstream, size_t modelVersion);

    // Process -- replace newlines and all %s by the given string
    static std::string Processed(const std::wstring& nodeName, std::string fragment, size_t minibatchId);
};

// =======================================================================
// ComputationNodeNonLooping -- abstract base class for computation nodes that do not implement eval/partial for individual frames
// Such as CRFNode, SequenceDecoderNode, and training criteria.
// =======================================================================

// This will provide default implementations for those two functions that will fail at runtime with a meaningful error.
// TODO: Most of these are reduce nodes that output a single number, no MBLayout. Maybe abstract those out further
template <class ElemType>
class ComputationNodeNonLooping : public ComputationNode<ElemType>
{
    typedef ComputationNode<ElemType> Base;

public:
    ComputationNodeNonLooping(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    // these two implement the ComputationNode<> interface
    void ForwardProp(const FrameRange& fr) override final
    {
        if (fr.IsAllFrames())
            ForwardPropNonLooping();
        else
            LogicError("%ls: %s node should never be in a loop.", Base::NodeDescription().c_str(), typeid(*this).name());
    }
    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override final
    {
        if (fr.IsAllFrames())
            BackpropToNonLooping(inputIndex);
        else
            LogicError("%ls: %s node should never be in a loop.", Base::NodeDescription().c_str(), typeid(*this).name());
    }

    // non-looping node types instead implement these functions
    virtual void ForwardPropNonLooping() = 0;
    virtual void BackpropToNonLooping(size_t inputIndex) = 0;
};

// =======================================================================
// FlowControlNode -- special wrapper node for use by ComputationNetwork only
// =======================================================================

class FlowControlNode : public ComputationNodeBase
{
    typedef ComputationNodeBase Base;

public:
    FlowControlNode()
        : ComputationNodeBase(DEVICEID_NOTYETDETERMINED /*we don't own matrices*/, L"" /*name: we don't care*/)
    {
    }

#pragma warning(disable : 4100)
    // these are meant to be implemented by ComputationNode<ElemType> but should never be called on traversal nodes
    // TODO: There are too many of these. This indicates improper class hierarchies.
    virtual ComputationNodeBase* NewThis(DEVICEID_TYPE deviceId, const wstring& name) const override { NOT_IMPLEMENTED; }
    virtual void Validate(bool isFinalValidationPass) override { NOT_IMPLEMENTED; }
    virtual void Save(File& fstream) const override { NOT_IMPLEMENTED; }
    virtual void Load(File& /*fstream*/, size_t /*modelVersion*/) override { NOT_IMPLEMENTED; }
    virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const override { NOT_IMPLEMENTED; }
    virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const override { NOT_IMPLEMENTED; }
    virtual double Get00Element() const override { NOT_IMPLEMENTED; }
    virtual MatrixBasePtr ValuePtr() const override { NOT_IMPLEMENTED; }
    virtual void UpdateFunctionMBSize() override { NOT_IMPLEMENTED; }
    virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs) override { NOT_IMPLEMENTED; }
    virtual void PrintSelf(bool) const override { NOT_IMPLEMENTED; }
    virtual void ValidateInferInputDimsFrom(const TensorShape&) override { NOT_IMPLEMENTED; }
    virtual void SetInput(const size_t, const Microsoft::MSR::CNTK::ComputationNodeBase::ComputationNodeBasePtr&) override { NOT_IMPLEMENTED; }
    virtual void MaskMissingValueColumnsToZero(const Microsoft::MSR::CNTK::FrameRange&) override { NOT_IMPLEMENTED; }
    virtual void MaskMissingGradientColumnsToZero(const Microsoft::MSR::CNTK::FrameRange&) override { NOT_IMPLEMENTED; }
    virtual void InvalidateMissingValueColumns(const Microsoft::MSR::CNTK::FrameRange&) override { NOT_IMPLEMENTED; }
    virtual void InvalidateMissingGradientColumns(const Microsoft::MSR::CNTK::FrameRange&) override { NOT_IMPLEMENTED; }
    virtual void NotifyFunctionValuesMBSizeModified(void) override { NOT_IMPLEMENTED; }
    virtual std::wstring ToString(void) const override { NOT_IMPLEMENTED; }
    // these are meant to be called during computation, so provide dummy implementations
    virtual bool RequiresPreCompute() const override { return false; } // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
    virtual std::string FormatOperationPrototype(const std::string& extraArgs) const override { return ""; }
    virtual void DumpNodeInfo(const bool /*printValues*/, const bool /*printMetadata*/, File& fstream) const override {}
    virtual std::set<std::pair<const MatrixBase*, std::wstring>> GetMatrixInfo() const override { NOT_IMPLEMENTED; }

protected: public:                                     // needed in ComputationNetwork::FindInRecurrentLoops(), which really should be part of SEQTraversalFlowControlNode
    std::vector<ComputationNodeBasePtr> m_nestedNodes; // nodes tucked away in this node, in evaluation order
};

// =======================================================================
// ILateAttachingNode -- helper wrapper class for ComputationNodes that must
// AttachInputs() late due to circular references
// =======================================================================

// Instantiate with LateAttachingNode<node type>(lambda, args for node constructor).
// To resolve, call AttachInputs()
// TODO: This is a bit indirect. Can it be done more nicely?
struct ILateAttachingNode { virtual void LateAttachInputs() = 0; };

// =======================================================================
// IRecurrentNode -- interface implemented by ComputationNodes that can be recurrent
// =======================================================================

struct IRecurrentNode { virtual int GetRecurrenceSteppingDirection() const = 0; };

// =======================================================================
// IFreezable -- nodes that have parameters that can be frozen
// e.g. if a trained model is to be used as a fixed feature extractor for another
// =======================================================================

struct IFreezable { virtual void FreezeParameters() { } };

// =======================================================================
// PreComputedNodeBase -- interface implemented by ComputationNodes that precompute
// TODO: We can use this interface in more places.
// =======================================================================

struct IPreComputeNode
{
    // check whether node has already undergone precomputation
    virtual bool HasComputed() const = 0;
    // call this with 'false' at start and with 'true' at end
    // This is used for resetting and updating from accumulators.
    virtual void MarkComputed(const bool hasComputed) = 0;
};

// =======================================================================
// helper macro to ease access to base members in presence of C++ two-phase name lookup
// =======================================================================

// Add 'typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;' at the start of each derived class
// (some derived classes define a similar macro; there please modify the typedef for Base accordingly.)
// This macro imports, one by one, every member of ComputationNode into the name space of the derived class.
// Without this, one would have to use the name prefix, or alternatively this->, in front of all base member,
// because the standard does not allow the compiler to do that for you (as MSVC still kindly does).
// If you add new members to ComputationNode, please also add them here.
// This macro expects 'Base' to be the name of the base class. Please also use 'Base' outside this macro to make it less likely to accidentally call the wrong base class members.
// Note: Whoever invented that C++ insanity called two-phase name lookup shall rot in hell, for the crime of causing infinite pain on unsuspecting programmers. [fseide]
#define UsingComputationNodeMembers /*without OperationName; needed to support inconsistent pattern of InputValue--TODO: This comment it out of date. */ \
    \
protected:                                                                                                                                               \
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;                                                                                    \
    using Base::BackpropTo;                                                                                                                              \
    using Base::ConstOnes;                                                                                                                               \
    using Base::CopyTo;                                                                                                                                  \
    using Base::CreateMatrixIfNull;                                                                                                                      \
    using Base::CreateUniqId;                                                                                                                            \
    using Base::CreateUniqNodeName;                                                                                                                      \
    using Base::DataFor;                                                                                                                                 \
    using Base::DataTensorFor;                                                                                                                           \
    using Base::DetachInputs;                                                                                                                            \
    using Base::DetermineElementwiseTensorRank;                                                                                                          \
    using Base::DumpNodeInfo;                                                                                                                            \
    using Base::EnumerateNodes;                                                                                                                          \
    using Base::Environment;                                                                                                                             \
    using Base::ForwardProp;                                                                                                                             \
    using Base::GetAsMatrixNumCols;                                                                                                                      \
    using Base::GetAsMatrixNumRows;                                                                                                                      \
    using Base::GetDeviceId;                                                                                                                             \
    using Base::GetEnvironmentPtr;                                                                                                                       \
    using Base::GetInputSampleLayout;                                                                                                                    \
    using Base::GetInputsFromConfig;                                                                                                                     \
    using Base::GetMBLayout;                                                                                                                             \
    using Base::GetMBLayoutAxisString;                                                                                                                   \
    using Base::GetNumInputs;                                                                                                                            \
    using Base::GetNumParallelSequences;                                                                                                                 \
    using Base::GetNumTimeSteps;                                                                                                                         \
    using Base::GetSampleLayout;                                                                                                                         \
    using Base::GetSampleMatrixNumCols;                                                                                                                  \
    using Base::GetSampleMatrixNumRows;                                                                                                                  \
    using Base::GetTensorShape;                                                                                                                          \
    using Base::GetTensorSliceFor;                                                                                                                       \
    using Base::Gradient;                                                                                                                                \
    using Base::GradientAsMatrix;                                                                                                                        \
    using Base::GradientFor;                                                                                                                             \
    using Base::GradientPtr;                                                                                                                             \
    using Base::GradientTensorFor;                                                                                                                       \
    using Base::HasMBLayout;                                                                                                                             \
    using Base::InferMBLayoutFromInputsForStandardCase;                                                                                                  \
    using Base::Input;                                                                                                                                   \
    using Base::InputRef;                                                                                                                                \
    using Base::InputUsedInComputingInputNodesGradients;                                                                                                 \
    using Base::InvalidateMissingGradientColumns;                                                                                                        \
    using Base::InvalidateMissingValueColumns;                                                                                                           \
    using Base::IsLeaf;                                                                                                                                  \
    using Base::IsOutOfDateWrtInputs;                                                                                                                    \
    using Base::IsPartOfLoop;                                                                                                                            \
    using Base::LinkToMBLayout;                                                                                                                          \
    using Base::Load;                                                                                                                                    \
    using Base::LoadValue;                                                                                                                               \
    using Base::MaskMissingColumnsToZero;                                                                                                                \
    using Base::MaskMissingGradientColumnsToZero;                                                                                                        \
    using Base::MaskMissingValueColumnsToZero;                                                                                                           \
    using Base::MaskedGradientFor;                                                                                                                       \
    using Base::MaskedValueFor;                                                                                                                          \
    using Base::MarkValueNonSharable;                                                                                                                    \
    using Base::NodeDescription;                                                                                                                         \
    using Base::OutputUsedInComputingInputNodesGradients;                                                                                                \
    using Base::PrintNodeValuesToFile;                                                                                                                   \
    using Base::FormatOperationPrototype;                                                                                                               \
    using Base::ReleaseMatricesAfterBackprop;                                                                                                            \
    using Base::ReleaseMatricesAfterForwardProp;                                                                                                         \
    using Base::ReleaseMatrixToPool;                                                                                                                     \
    using Base::RequestMatricesBeforeBackprop;                                                                                                           \
    using Base::RequestMatricesBeforeForwardProp;                                                                                                        \
    using Base::RequestMatrixFromPool;                                                                                                                   \
    using Base::Save;                                                                                                                                    \
    using Base::SetDims1;                                                                                                                                \
    using Base::SetDims;                                                                                                                                 \
    using Base::SetInput;                                                                                                                                \
    using Base::SetLearningRateMultiplier;                                                                                                               \
    using Base::UpdateFunctionMBSize;                                                                                                                    \
    using Base::UpdateFunctionValuesSize;                                                                                                                \
    using Base::Validate;                                                                                                                                \
    using Base::ValidateBinaryReduce;                                                                                                                    \
    using Base::ValidateBinaryZip;                                                                                                                       \
    using Base::ValidateNaryZip;                                                                                                                         \
    using Base::ValidateInferBinaryInputDims;                                                                                                            \
    using Base::ValidateInferNaryInputDims;                                                                                                              \
    using Base::ValidateInferInputDimsFrom;                                                                                                              \
    using Base::ValidateUnaryMap;                                                                                                                        \
    using Base::ValidateUnaryReduce;                                                                                                                     \
    using Base::ValueFor;                                                                                                                                \
    using Base::ValuePtr;                                                                                                                                \
    using Base::ValueTensorFor;                                                                                                                          \
    using Base::VerifyDataSize;                                                                                                                          \
    using Base::VerifyDims;                                                                                                                              \
    using Base::WriteMinibatchWithFormatting;                                                                                                            \
    using Base::ZeroGradientsOfInputs;                                                                                                                   \
    using Base::m_deviceId;                                                                                                                              \
    using Base::m_gradient;                                                                                                                              \
    using Base::m_inputs;                                                                                                                                \
    using Base::m_nodeName;                                                                                                                              \
    using Base::m_pMBLayout;                                                                                                                             \
    using Base::m_learningRateMultiplier;                                                                                                                \
    using Base::m_sampleLayout;                                                                                                                          \
    using Base::m_value;                                                                                                                                 \
    using Base::m_valueSharable;                                                                                                                         \
    using Base::shared_from_this;                                                                                                                        \
    \
public:                                                                                                                                                  \
    using Base::AttachInputs;                                                                                                                            \
    using Base::AttachInputsFromConfig;                                                                                                                  \
    using Base::CreateGradientMatrixIfNull;                                                                                                              \
    using Base::NodeName;                                                                                                                                \
    using Base::RequiresPreCompute;                                                                                                                      \
    using Base::ValueAsMatrix;                                                                                                                           \
    using Base::Value;

#define ComputationNodeBoilerplate                                                                                \
protected: /* some boilerplate goes here */                                                                       \
    virtual const std::wstring OperationName() const override { return TypeName(); }                              \
    virtual ComputationNodeBase* NewThis(DEVICEID_TYPE deviceId, const wstring& name) const override              \
    {                                                                                                             \
        const ComputationNodeBase* p = new typename std::remove_reference<decltype(*this)>::type(deviceId, name); \
        return const_cast<ComputationNodeBase*>(p);                                                               \
    }

#define UsingComputationNodeMembersBoilerplate \
    ComputationNodeBoilerplate;                \
    UsingComputationNodeMembers

// =======================================================================
// a few standard base classes for N-nary operations
// =======================================================================

// -----------------------------------------------------------------------
// UnaryElementWiseNode (operand)
//
// unary elementwise operations that are implemented with the tensor lib
//
// Derived clases only need to override ForwardProp() and BackpropTo().
// -----------------------------------------------------------------------

template <class ElemType>
class UnaryElementWiseNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    UnaryElementWiseNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }
};

#define UsingUnaryElementwiseNodeBaseMembers UsingComputationNodeMembersBoilerplate;

// -----------------------------------------------------------------------
// BinaryElementWiseNode (operand1, operand2)
//
// binary elementwise operations that are implemented with the tensor lib
//
// Derived clases only need to override ForwardProp() and BackpropTo().
// -----------------------------------------------------------------------

template <class ElemType>
class BinaryElementWiseNode : public ComputationNode<ElemType>, public NumInputs<2>, public IdentityTransformerNode
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    BinaryElementWiseNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

#if DUMPOUTPUT
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return true; }
#else
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
#endif
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*IComputationNode::*/ BeginForwardProp() override // called before first iteration step of ForwardProp()
    {
        Base::BeginForwardProp();
        // we switch result to dense as a work-around because ColumnSlice doesn't support all the sparse formats
        // TODO: This is a stopgap. Is this the right thing to do? It changes the matrix type in-place.
        Value().SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryZip(isFinalValidationPass, true /*allowBroadcast*/);
    }
};

#define UsingBinaryElementwiseNodeBaseMembers UsingComputationNodeMembersBoilerplate;

#pragma endregion base computation class

}}}
