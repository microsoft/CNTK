//
// <copyright file="ComputationNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "TensorView.h"
#include "ScriptableObjects.h"
#include "Sequences.h"
#include "DataTensor.h"
#include "MatrixPool.h"

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

 #define ENABLE_TENSORVIEW   // flip this switch once the tensor lib is confirmed to be working

//#define RNN_DEBUG 1
#define DEFAULT_HIDDEN_ACTIVATION 0.1

#pragma warning (disable: 4267)

// version number to control how to read and write 
#define CNTK_MODEL_VERSION_1 1
#define CNTK_MODEL_VERSION_2 2
#define CURRENT_CNTK_MODEL_VERSION 2

extern bool g_shareNodeValueMatrices;

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P)          (P)
#endif

// helper mode for debugging
// If TRACK_GAP_NANS is defined then initialize layout gaps to NaN and do NaN checks. Also do detailed logging of node computations.
// #define TRACK_GAP_NANS

namespace Microsoft { namespace MSR { namespace CNTK {

    enum CopyNodeFlags  // flags to be passed to the CopyTo() function
    {
        copyNodeNull                 = 0,   // invalid value
        copyNodeValue                = 1,   // copy everything but the children links
        copyNodeChildren             = 2,   // only copy over children links
        copyNodeAll                  = 3,   // copy everything
        copyNodeChildrenCrossNetwork = 4,   // allow a cross network child copy
    };

#pragma region base computation class

    // =======================================================================
    // IComputationNode -- set of methods that are to be implemented (or optionally overridable) by node implementations.
    // =======================================================================

    class ComputationNodeBase;
    struct/*interface*/ IComputationNode
    {
        typedef shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

        // --- these must be implemented by each node

        virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        // TODO: OperationName calls static TypeName which does not match the actual type names in that the 'Node' is missing.
        virtual const std::wstring OperationName() const = 0;
#define OperationNameOf(T) (T<float>::TypeName())               // convenience macro

        virtual void UpdateFunctionMBSize() = 0;                // recalculate our column dimensions from MBLayout. Override to update temps.

        virtual void BeginForwardProp() = 0;                    // called beforefirst iteration step of ForwardProp()
        virtual void ForwardProp(const FrameRange &) = 0;       // forward prop for one minibatch
        virtual void EndForwardProp() = 0;                      // called after last iteration step of ForwardProp()

        virtual void BeginBackprop() = 0;                       // called before first iteration step of ComputeGradient()
        virtual void BackpropTo(const size_t inputIndex, const FrameRange &) = 0;   // backprop gradient into one of the inputs
        virtual void EndBackprop() = 0;                         // called after last iteration step of ComputeGradient()

        // --- these are meant to be overridden by ControlFlowNodes

        virtual void Backprop(const FrameRange & fr, bool childrenInThisLoop, bool childrenInOuterLoop) = 0;

        // --- optional overrides that add functionality

        // Any override must call Base version as well.
        // Default implementations are in ComputationNodeBase or ComputationNode<ElemType>.

        virtual void Validate(bool isFinalValidationPass) = 0;          // main base validation function
        virtual void InferImageDimsFromInputs() = 0;
        virtual void Save(File& fstream) const = 0;
        virtual void Load(File& /*fstream*/, size_t /*modelVersion*/) = 0;
        virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const = 0;

        virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) = 0;  // request matrices needed to do node function value evaluation
        virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) = 0;   // release temp matrices that are only used by forward computation. Don't release matrices that need to be used in the gradient computation
        virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool) = 0;
        virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) = 0;     // request matrices that are needed for gradient computation
        virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) = 0;      // release gradient and temp matrices that no longer needed after all the children's gradients are computed.

        // --- optional overrides that describe a feature or property of the node

        virtual bool RequiresPreCompute() const = 0;                    // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.

        // --- optional overrides for more informative logging

        virtual void PrintSelfBeforeValidation() const = 0;             // called in validation loop right before Validate()
        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const = 0;
    protected:
        virtual ~IComputationNode() { }
    };

    // =======================================================================
    //  This provide a interface for stateful node (e.g., DelayNodeBase) and definition of state
    //  This interface allows to Export and Import state from elsewhere 
    //  It is needed when doing sub-minibatch implementation 
    // =======================================================================

    class INodeState: public std::enable_shared_from_this<INodeState>
    {
    public:
        virtual ~INodeState() {} 
    };

    struct /*interface*/ IStateFulNode
    {
        typedef std::shared_ptr<INodeState> NodeStatePtr;
        virtual NodeStatePtr ExportState() = 0;
        virtual void ImportState(const NodeStatePtr& pImportedState) = 0;
    };

    // =======================================================================
    // ComputationNetworkOwnedNodeState -- class to collect ComputationNode members that are really owned by ComputationNetwork
    // These members are only to be set, changed, and read by ComputationNetwork code.
    // =======================================================================

    class ComputationNetwork;
    struct ComputationNetworkOwnedNodeState
    {
        friend class ComputationNetwork;

        ComputationNetworkOwnedNodeState() :
            m_needsGradient(false)
        {
            PurgeStateForFormingRecurrentLoops();
            m_isPartOfLoop = false;
        }

        void CopyTo(ComputationNetworkOwnedNodeState & other) const
        {
            // TODO: is that really all we copy? (this is a result of refactoring, so it seems yes indeed). Should we at least ClearCache()?
            other.m_isPartOfLoop = m_isPartOfLoop;
            other.m_needsGradient = m_needsGradient;
        }

        bool IsPartOfLoop() const { return m_isPartOfLoop; }

    protected:  // TODO: should be fully encapsulated here

        bool m_needsGradient;   // true if this node or any children need a gradient to be computed (for own consumption or propagation to somewhere in the child tree)

    private:

        bool m_isPartOfLoop;        // true if this loop is part of a recurrent loop

    protected:

        // owned by FormRecurrentLoops() and stuff it calls, only used from inside there (FormRecurrentLoops() calls PurgeStateForFormingRecurrentLoops() at its end to make that super-clear)
        void PurgeStateForFormingRecurrentLoops()
        {
            m_loopId = -1;
            m_visitedOrder = -1;
            m_indexInLoop = 0;
            m_visited = false;
            m_index = -1;
            m_minIndex = -1;
            m_inStack = false;
        }

        int m_loopId;           // index into m_allSEQNodes array, for use by reordering operation only
        int m_visitedOrder;     // remembers order in which nodes were visited by EnumerateNodes(), but gets updated
        bool m_visited;         // note: also used by ValidateSubNetwork()
        int m_indexInLoop;
        // only used inside DetermineSCCs():
        int m_index;            // index denoting order in which nodes were visited in DetermineSCCs()
        int m_minIndex;         // min of m_index over all nodes within a single loop
        bool m_inStack;
    };

    // =======================================================================
    // TimeStamp -- helper class to manage a "time stamp" (unique value) of a computation result to avoid recomputation
    // =======================================================================

    class TimeStamp
    {
    public:
        TimeStamp() { ResetEvalTimeStamp(); }
        void CopyTo(TimeStamp & other) const { other.m_evalTimeStamp = m_evalTimeStamp; }
        void ResetEvalTimeStamp() { m_evalTimeStamp = s_timeStampCounter; }
        int64_t GetEvalTimeStamp() const { return m_evalTimeStamp; }

        // create a new unique time stamp
        void BumpEvalTimeStamp() { m_evalTimeStamp = CreateUniqId(); }

        // the difference is taken to take into account numeric overflow (which really should never happen for a 64-bit integer... but hey, it's free!)
        bool IsOlderThan(const TimeStamp & other) const
        {
            // BUGBUG: For some reason, we must test equality as well, although that does not indicate being older.
            return GetEvalTimeStamp() - other.GetEvalTimeStamp() /*<*/ <= 0;
        }

        int64_t CreateUniqId() const
        {
            return /*1 +*/ atomic_fetch_add(&s_timeStampCounter, (unsigned long long int) 1);
        }

    private:
        static atomic_ullong s_timeStampCounter;
        int64_t m_evalTimeStamp; //this is used to reduce unnecessary recomputation when a different node in the model is reevaluated
    };

    // =======================================================================
    // ComputationNodeBase -- abstract base class for all computation nodes
    // =======================================================================

    class ComputationNodeBase :
        public IComputationNode,
        public/*protected*/ ComputationNetworkOwnedNodeState,   // TODO: figure the 'protected' business out, somehow the 'friend' thing does not work
        public TimeStamp,                                       // for time-stamp management
        public ScriptableObjects::ComputationNodeObject,
        public ScriptableObjects::WithTag, public ScriptableObjects::HasName, public ScriptableObjects::HasToString,
        public std::enable_shared_from_this<ComputationNodeBase>
    {
        // note: enable_shared_from_this<> allows to create a shared_ptr from a raw pointer to this that is correctly aware of all other shared_ptrs (same ref count)
    public:
        typedef shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

        ComputationNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            m_deviceId(deviceId), m_outputNeededDuringBackprop(true),
            m_parameterUpdateRequired(false), m_gradientInitialized(false),
            m_nodeName(name == L"" ? CreateUniqNodeName() : name),
            m_numRows(0), m_numCols(0)
        { }
        virtual ~ComputationNodeBase(){}

        virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            if (OperationName() != node->OperationName())
                RuntimeError("Cannot copy from one node type to another node type");
            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                node->m_inputs = m_inputs;
            }
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_deviceId = m_deviceId;
                node->m_parameterUpdateRequired = m_parameterUpdateRequired;
                node->m_nodeName = newName;

                node->m_inputSampleLayout = m_inputSampleLayout;
                node->m_sampleLayout = m_sampleLayout;

                ComputationNetworkOwnedNodeState::CopyTo(*node);
                TimeStamp::CopyTo(*node);
            }
        }

        virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) = 0;

        // TODO: make sure this does not get implemented in any of the base classes
        DEVICEID_TYPE GetDeviceId() const { return m_deviceId; }    // TODO: remove, only used from copy constructor which will go away

        virtual void Save(File& fstream) const
        {
            fstream << OperationName() << NodeName();
        }

        virtual void Load(File& /*fstream*/, size_t /*modelVersion*/)
        {
            // it is assumed that OperationName and NodeName have already been consumed--some asymmetry between Save and Load
            // base class has nothing to load
        }

        // dimensions

        size_t GetNumRows() const { return m_numRows; }
        size_t GetNumCols() const { return m_numCols; }
        pair<size_t, size_t> GetDims() { return make_pair(GetNumRows(), GetNumCols()); }
        // TODO: add an overload SetDims(TensorShape, cols)
        // Currently called from:
        //  - Validate()   --intended
        //  - LearnableParameterNode (init, load)
        //  - InputValue (init, load)
        //  - DelayedValueNodeBase (Init())
        // only changes col dim:
        //  - ResizeAllFeatureNodes()
        // use a different name for these:
        //  - ReshapeNode::UpdateFunctionMBSize()    --??
        //  - various unit tests
        //  - ComputationNetwork::FixupInputMinibatchSize()
        //  - TimeReverseNode (first step--deprecate and/or move to UpdateMB... function)
        //  - StrideTimesNode
        //  - PairNetworkNode
        //  - LSTMNode
        //  - MultiNetworks-
        void SetDims(size_t rows, size_t cols)
        {
            m_numRows = rows;
            m_numCols = cols;
            // actual memory allocation happens elsewhere
        }
        void SetDims(ComputationNodeBasePtr node) { SetDims(node->GetNumRows(), node->GetNumCols()); }
        void SetDims(const TensorShape & sampleLayout, size_t cols)
        {
            m_sampleLayout = sampleLayout;
            m_numRows = m_sampleLayout.GetNumElements();
            m_numCols = cols;
        }
        virtual void NotifyFunctionValuesMBSizeModified() { } // someone outside changed our m_value--update our internal state, e.g. m_numRows, m_numCols
        void VerifyDims(size_t rows, size_t cols)
        {
            if (rows != GetNumRows() || cols != GetNumCols())
            {
                LogicError("VerifyDims: %ls %ls operation expected size %d x %d, but it is %d x %d",
                           NodeName().c_str(), OperationName().c_str(),
                           (int)rows, (int)cols, (int)GetNumRows(), (int)GetNumCols());
            }
        }
        virtual void VerifyDims(ComputationNodeBasePtr node) { VerifyDims(node->GetNumRows(), node->GetNumCols()); }
        virtual void VerifyDimsMatch() const = 0;       // verify that m_value dimensions match ours

        const TensorShape & GetSampleLayout() const { return m_sampleLayout; }
    protected:
        // TODO: There are temporarily two confusing functions; either unify them, or name them better:
        //  - GetSampleLayout() just reads out m_sampleLayout, which is the layout of matrix coluns
        //  - GetSampleShape() makes up a sample layout in case of a bad m_sampleLayout, and includes columns in case of no MBLayout
        TensorShape GetSampleShape() const;             // TODO: Once numRows is consistent with m_sampleLayout, this will go away
        size_t DetermineElementwiseTensorRank() const;
    public:
        TensorShape GetTensorShape(size_t dims, const FrameRange & fr) const;

        // access to element(0,0) without having to type-cast
        virtual double Get00Element() const = 0;

        // validation
        // This is overridden by every node. This base class just checks for unconnected and empty inputs.
        virtual void Validate(bool isFinalValidationPass)           // main base validation function
        {
            // check for NULL pointers
            for (size_t i = 0; i < m_inputs.size(); i++)
            {
                if (!m_inputs[i])
                    RuntimeError("Validate: Input [%d] of %ls node '%ls' is empty (NULL, not connected).", (int)i, OperationName().c_str(), NodeName().c_str());
            }
            // check for empty inputs
            if (isFinalValidationPass)
            {
                for (const auto & child : m_inputs)
                {
                    if (child->GetNumRows() == 0 || (!child->HasMBLayout() && child->GetNumCols() == 0))
                        RuntimeError("%ls %ls operation: input %ls %ls has 0 elements.",
                                     NodeName().c_str(), OperationName().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
                }
            }
        }
        // helper functions for common cases
    protected:
        void ValidateUnaryMap(bool isFinalValidationPass);
        void ValidateUnaryReduce(bool isFinalValidationPass);
        void ValidateInferBinaryInputDims();
        void ValidateBinaryZip(bool isFinalValidationPass, bool allowMultiples);
        void ValidateBinaryReduce(bool isFinalValidationPass);
    public:

        virtual bool UnitTest() { return true; }

        virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs) = 0;
        // convenience versions that take individual arguments
        void AttachInputs(const ComputationNodeBasePtr& singleInput) { AttachInputs(std::vector<ComputationNodeBasePtr> { singleInput } ); }
        void AttachInputs(const ComputationNodeBasePtr& leftInput, const ComputationNodeBasePtr& rightInput) { AttachInputs(std::vector<ComputationNodeBasePtr> { leftInput, rightInput } ); }
        void AttachInputs(const ComputationNodeBasePtr& leftInput, const ComputationNodeBasePtr& middleInput, const ComputationNodeBasePtr& rightInput) { AttachInputs(std::vector<ComputationNodeBasePtr> { leftInput, middleInput, rightInput } ); }
        void AttachInputs(const ComputationNodeBasePtr& firstInput, const ComputationNodeBasePtr& secondInput, const ComputationNodeBasePtr &thirdInput, const ComputationNodeBasePtr& fourthInput) { AttachInputs(std::vector<ComputationNodeBasePtr> { firstInput, secondInput, thirdInput, fourthInput } ); }
        void AttachInputs(const ComputationNodeBasePtr& firstInput, const ComputationNodeBasePtr& secondInput, const ComputationNodeBasePtr &thirdInput, const ComputationNodeBasePtr& fourthInput, const ComputationNodeBasePtr& fifthInput) { AttachInputs(std::vector<ComputationNodeBasePtr> { firstInput, secondInput, thirdInput, fourthInput, fifthInput } ); }
        void AttachInputs(const ComputationNodeBasePtr& firstInput, const ComputationNodeBasePtr& secondInput, const ComputationNodeBasePtr &thirdInput, const ComputationNodeBasePtr& fourthInput, const ComputationNodeBasePtr& fifthInput, const ComputationNodeBasePtr& sixthInput) { AttachInputs(std::vector<ComputationNodeBasePtr> { firstInput, secondInput, thirdInput, fourthInput, fifthInput, sixthInput } ); }

        virtual void DetachInputs() { m_inputs.clear(); }

        // helper for the factory function for ComputationNodes
        static vector<ComputationNodeBasePtr> GetInputsFromConfig(const ScriptableObjects::IConfigRecordPtr configp)
        {
            vector<ComputationNodeBasePtr> inputs;
            const auto * inputsArg = configp->Find(L"inputs");
            if (inputsArg)
            {
                if (inputsArg->Is<ComputationNodeBase>())                // single arg
                    inputs.push_back(*inputsArg);
                else                                                    // a whole vector
                {
                    ScriptableObjects::ConfigArrayPtr inputsArray = *inputsArg;
                    const auto range = inputsArray->GetIndexRange();
                    for (int i = range.first; i <= range.second; i++)   // pull them. This will resolve all of them.
                        inputs.push_back(inputsArray->At(i, [](const wstring &){ LogicError("GetInputs: out of bounds index while iterating??"); }));
                }
            }
            return inputs;
        }

        const std::vector<ComputationNodeBasePtr> & GetInputs() const { return m_inputs; }
        ComputationNodeBasePtr Input(size_t index) const { return m_inputs[index]; } // TODO: delete this; change to m_inputs

        //return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual bool /*IComputationNode::*/RequiresPreCompute() const { return false; }

        // casting helpers
        template<typename N>
        N * As()
        {
            auto p = dynamic_cast<N*>(this);
            if (!p)
                LogicError("Attempted to type-cast node %ls %ls to %s, which is not possible.", NodeName().c_str(), OperationName().c_str(), typeid(N).name());
            return p;
        }
        template<typename N> bool Is() { return dynamic_cast<N*>(this) != nullptr; }

        /*HasName::*/void SetName(const std::wstring & newName) // also for use by ExperimentalNetworkBuilder
        {
            m_nodeName = newName;
            fprintf(stderr, "Node --> %ls = %ls\n", NodeName().c_str(), OperationName().c_str()), fflush(stderr);
        }

        void LinkToMBLayout(MBLayoutPtr pMBLayout) { m_pMBLayout = pMBLayout; }
        MBLayoutPtr GetMBLayout() { return m_pMBLayout; }
        bool HasMBLayout() const { return !!m_pMBLayout; }

        std::wstring GetName() const { return m_nodeName; }

        // temporary function that is called to verify stuff is called as I think it is. Delete if this does not fire for a while.
        void VerifyNumParallelSequences(size_t bsz)
        {
            if (bsz != m_pMBLayout->GetNumParallelSequences())
                LogicError("VerifyNumParallelSequences: value inconsistent with MB layout");
        }

    protected:
    public:     // ...the following should be protected, but nodes inquire about their children, requiring public access

        size_t GetNumParallelSequences() const
        {
#if 1
            if (!m_pMBLayout)       // TODO: temporary workaround to Check_t() calls which call this. TODO: Delete the first arg from Check_t() after memshare merge.
                return SIZE_MAX;
#endif
            return m_pMBLayout->GetNumParallelSequences();
        }

        // get our current number of time steps for this node
        // This inquires the MB layout.
        size_t GetNumTimeSteps() const
        {
            if (!m_pMBLayout)
                LogicError("GetNumTimeSteps: invalid to call on a node without MB layout"); // since it has no notion of time
            return m_pMBLayout->GetNumTimeSteps();
        }
    public:

        // implemented by ComputationNode<ElemType>
        // for debugging purpose
        virtual void PrintSelf(bool printMatrices = false) const = 0;

        // called in validation loop right before Validate()
        virtual void /*IComputationNode::*/PrintSelfBeforeValidation() const
        {
            fprintf(stderr, "\nValidating --> %ls = %ls", NodeName().c_str(), OperationName().c_str());

            if (!IsLeaf())
            {
                fprintf(stderr, "(");
                for (size_t i = 0; i<GetNumInputs(); i++)
                {
                    const auto & child = m_inputs[i];
                    if (i > 0)
                        fprintf(stderr, ", ");

                    if (child == nullptr)
                    {
                        fprintf(stderr, "NULL");
                        continue;
                    }

                    const char * mbSizeMark = child->m_pMBLayout ? "MBSize " : "";
                    if (child->m_sampleLayout.GetRank() == 3 && (child->m_sampleLayout.GetWidth() != 1 || child->m_sampleLayout.GetNumChannels() != 1))  // looks like an image: use WHC notation
                        fprintf(stderr, "%ls[%lu {W=%lu, H=%lu, C=%lu}, %s%lu]", child->NodeName().c_str(), child->GetNumRows(),
                                child->m_sampleLayout.GetWidth(), child->m_sampleLayout.GetHeight(), child->m_sampleLayout.GetNumChannels(), mbSizeMark, child->GetNumCols());
                    else if (child->m_sampleLayout.GetRank() > 1)           // tensor: output the tensor dimensions   --TODO: there will be no numRows in the future, only the tensor
                        fprintf(stderr, "%ls[%lu [%s], %s%lu]", child->NodeName().c_str(), child->GetNumRows(), string(child->m_sampleLayout).c_str(), mbSizeMark, child->GetNumCols());
                    else
                        fprintf(stderr, "%ls[%lu, %s%lu]", child->NodeName().c_str(), child->GetNumRows(), mbSizeMark, child->GetNumCols());
                }
                fprintf(stderr, ")");
            }
#if 0
            else
            {
                if (m_pMBLayout)
                    fprintf(stderr, "[%lu, MBSize]", GetNumRows());
                else
                    fprintf(stderr, "[%lu, %lu]", GetNumRows(), GetNumCols());
            }
#endif
        }

        const std::wstring& NodeName() const { return m_nodeName; }
        void SetNodeName(const std::wstring & nodeName) { m_nodeName = nodeName; }

        bool IsLeaf() const { return GetNumInputs() == 0; }
        bool& NeedGradient() { return m_needsGradient; }
        const bool& NeedGradient() const { return m_needsGradient; }

        void SetParameterUpdateRequired(bool f) { m_parameterUpdateRequired = f; }
        bool IsParameterUpdateRequired() const { return m_parameterUpdateRequired; }

        void SetOutputNeededDuringBackprop(bool f) { m_outputNeededDuringBackprop = f; }
        bool IsOutputNeededDuringBackprop() const 
        {
            return !g_shareNodeValueMatrices || m_outputNeededDuringBackprop;
        }

        virtual void /*IComputationNode::*/InferImageDimsFromInputs()
        {
            if (!IsLeaf())
                InferImageDimsFromInput(0); //copy from child 0 by default.
        }

        virtual void ValidateInferInputDims(size_t i, size_t rows, size_t cols) = 0;

        // TODO: Remove this.
        // used from:
        //  - Plus/Minus/ElementTimesNode --> replace by max dim over inputs. Make this standard behavior for all binary element-wise ops.
        bool IsInputAnImage(const size_t index) const
        {
            return m_inputs[index]->m_sampleLayout.IsInputAnImage();
        }

        const TensorShape & GetImageLayout() const { return m_sampleLayout; }

        pair<TensorShape, TensorShape> GetImageLayouts() const { return make_pair(m_inputSampleLayout, m_sampleLayout); }   // helper for Validate()

        const size_t GetNumInputs() const { return m_inputs.size(); }

        virtual void SetInput(const size_t childIndex, const ComputationNodeBasePtr& node) = 0;

        // masking
        // overridden by <ElemType> variant only
        virtual void MaskMissingValueColumnsToZero(const FrameRange &) = 0;
        virtual void MaskMissingGradientColumnsToZero(const FrameRange &) = 0;
        virtual void InvalidateMissingValueColumns(const FrameRange &) = 0;
        virtual void InvalidateMissingGradientColumns(const FrameRange &) = 0;

        virtual void ZeroGradientsOfInputs() = 0;

        virtual void /*IComputationNode::*/BeginForwardProp() override             // called before first iteration step of ForwardProp()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "BeginForwardProp: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }
        virtual void /*IComputationNode::*/EndForwardProp() override               // called after last iteration step of ForwardProp()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "EndForwardProp: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }
        // TODO: the following two are not really utilized yet other than printing trace information
        virtual void /*IComputationNode::*/BeginBackprop() override             // called before first iteration step of ComputeGradient()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "BeginBackprop: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }
        virtual void /*IComputationNode::*/EndBackprop() override               // called after last iteration step of ComputeGradient()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "EndBackprop: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }

        // Is the output value of the computation node needed for computing 
        // gradients of any of the input nodes
        // Base-class version makes conservative assumption that it is. Override if not.
        virtual bool OutputUsedInComputingInputNodesGradients() const
        {
            return true;
        }

        // Is the output value of the specified  input node needed for computing
        // gradients of any of the input nodes
        // Base-class version makes conservative assumption that it is. Override if not.
        virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const
        {
            UNREFERENCED_PARAMETER(childIndex);
            return true;
        }

    protected:

        void InferImageDimsFromInput(const size_t index, const bool outputSameAsInput = true)
        {
            if (index >= GetNumInputs())
                InvalidArgument("InferImageDimsFromInput: output index");

            const auto & child = m_inputs[index];
            if (child != nullptr)
                m_inputSampleLayout = child->m_sampleLayout;
            if (outputSameAsInput)
                m_sampleLayout = m_inputSampleLayout;
        }

        void InferMBLayoutFromInputsForStandardCase();

    public:

        bool IsEqualTo(const ComputationNodeBasePtr& other) const //this will be used to determine whehter two nodes are the same
        {
            if (OperationName() != other->OperationName() || m_inputs.size() != other->m_inputs.size())
                return false;

            if (NodeName() == other->NodeName())  //assume names are unique in the system
                return true;

            if (IsLeaf() && other->IsLeaf())  //since names are not equal otherwise will return above
                return false;

            for (size_t i=0; i<m_inputs.size(); i++)
                if (!(m_inputs[i] == other->m_inputs[i]))
                    return false;

            return true;
        }

        // determine enumeration order for everything needed to evaluate this node (and its children)
        // This creates a list such that children are evaluated before their parents.
        // If !forForwardProp then the order will be reversed, suitable for backprop.
        // The 'recurrent' version is only called from FormRecurrentLoops().
        // TODO: This should be a method of ComputationNetwork, not ComputationNode.
        static std::list<ComputationNodeBasePtr> EnumerateNodes(const std::vector<ComputationNodeBasePtr> & allRoots, bool skipPairNetwork = false/*legacy*/)
        {
            std::list<ComputationNodeBasePtr> nodes;
            std::unordered_set<ComputationNodeBasePtr> visited;

            for (const auto & root : allRoots)
                root->EnumerateNodesRec(visited, nodes, skipPairNetwork);  // call into the recursive portion of this function below

            return nodes;
        }

        // and a version that does it for only one root 'this'
        std::list<ComputationNodeBasePtr> EnumerateNodes(bool skipPairNetwork) /*const*/ { return EnumerateNodes(std::vector<ComputationNodeBasePtr> { shared_from_this() }, skipPairNetwork); }

    private:
        // Recursive part of EnumerateNodes().
        void EnumerateNodesRec(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result, bool skipPairNetwork) /*const*/ // const not working due to shared_from_this()
        {
            if (visited.find(shared_from_this()) == visited.end())      // do not include a node twice
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                // children first for function evaluation
                if (OperationName() != L"PairNetwork" || !skipPairNetwork)    // (don't step through network-pair boundary if called from FormRecurrentLoops())
                {
                    for (int i = 0; i < m_inputs.size(); i++)
                    {
                        if (m_inputs[i])
                            m_inputs[i]->EnumerateNodesRec(visited, result, skipPairNetwork);
                    }
                }

                // now that all children are in list before us, put ourselves
                result.push_back(shared_from_this());
            }
        }
    public:

        // check whether a node is up-to-date w.r.t. its children, for lazy evaluation
        // If this returns false, node must be evaluated to update m_value.
        // BUGBUG: The function name is incorrect. It also returns 'true' if a child has the same time stamp (not older).
        // This is virtual because it is overridden by traversal nodes.
        virtual bool IsOutputOlderThanInputs() const
        {
            // TODO: use range-based for
            for (size_t i = 0; i < GetNumInputs(); i++)
            {
                if (IsOlderThan(*m_inputs[i]))
                    return true;
            }

            return false;
        }

        typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;
        // [1/13/2015 erw] add to enumerate all the edges 
        // enumerate arcs that can be reached starting from the current node's children
        // [in/out] visited record already visited nodes 
        // TODO: This should be a method of ComputationNetwork, not ComputationNode.
        void EnumerateArcs(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationArc>& arcs)
        {
            std::list<ComputationNodeBasePtr>	tovisit;

            if (visited.find(shared_from_this()) == visited.end()) // only do when this node has not been visited before
            {
                tovisit.push_back(shared_from_this());

                while (!tovisit.empty())
                {
                    ComputationNodeBasePtr curNode = tovisit.front();
                    tovisit.pop_front();

                    if (visited.find(curNode) == visited.end())
                    {
                        for (size_t i = 0; i < curNode->m_inputs.size(); i++)
                        {
                            arcs.push_back(ComputationArc(curNode, curNode->m_inputs[i]));

                            if (visited.find(curNode->m_inputs[i]) == visited.end()) // this children has not been visited before 
                                tovisit.push_front(curNode->m_inputs[i]);		// going to visit each of the children
                        }
                        visited.insert(curNode);
                    }
                }
            }
        }

        std::wstring CreateUniqNodeName() const
        {
#ifdef USE_GUID_AS_NAME
            UUID uuid;
            ZeroMemory(&uuid, sizeof(UUID));
            std::wstring name;

            UuidCreate(&uuid);
            WCHAR* szUuid = nullptr;
            if (UuidToStringW(&uuid, (RPC_WSTR*)&szUuid) != RPC_S_OK)
                RuntimeError("Failed to craete unique node name.");
            else
            {
                name = szUuid;
                RpcStringFreeW((RPC_WSTR*)&szUuid);
            }
#else
            int64_t id = CreateUniqId();
            std::wstring base = L"AutoName";
            std::wstringstream sstm;
            sstm << base.c_str() << id;
            std::wstring name = sstm.str();
            //msra::strfun::wstrprintf name(L"%s%d", L"AutoName", id);
#endif

            return name;
        }

    protected:

        DEVICEID_TYPE m_deviceId;   // CPU=-1, >=0 GPU
        std::wstring m_nodeName;

        // inputs
        std::vector<ComputationNodeBasePtr> m_inputs;

        // dimensions and layout
        // Data is stored as a matrix, but often it is interpreted as a more complex structure.
        // If the matrix is minibatch data (inputs, activations, labels), then matrix columns are samples.
        // Note that the actual matrix storage does not always exist.
        size_t m_numRows, m_numCols;        // matrix dimension of function values and gradients
        TensorShape m_sampleLayout;    // and the output
        MBLayoutPtr m_pMBLayout;

        TensorShape m_inputSampleLayout;     // how to interpret each column in the input as an image
        // TODO: Why is the input layout not just the layout of the input node?

        // flags related to gradient propagation
        bool m_parameterUpdateRequired;     // update parameters? Only used for LearnableParameters.    --TODO: Should we make this a member of LearnableParameters actually? And require a type cast? Currently it is read out for all leaves.
        bool m_gradientInitialized;         // indicates whether the gradient matrix has been resized and initialized to 0
        bool m_outputNeededDuringBackprop;  // indicates whether the output value of the node is needed during backprop
    };
    typedef ComputationNodeBase::ComputationNodeBasePtr ComputationNodeBasePtr;

    // =======================================================================
    // ComputationNode -- abstract base class for computation nodes, deriving from CompuationNodeBase, parameterized by float vs. double
    // =======================================================================

    // little helper class to allow derived Node classes to specify how many inputs they expect
    struct INumInputs { virtual size_t GetExpectedNumInputs() const = 0; };
    template<size_t m_numInputs> struct NumInputs : public INumInputs { size_t GetExpectedNumInputs() const override final { return m_numInputs; } };  // e.g. derive from NumInputs<2>

    template<class ElemType>
    class ComputationNode : public ComputationNodeBase // abstract class that cannot be instantiated
    {
        typedef ComputationNodeBase Base;
    protected:
        //std containers such as list and map does not support class reference so we need to use pointer
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    public:
        using ComputationNodeBase::AttachInputs;    // import the convenience functions that take 1..6 parameters
        using ComputationNodeBase::SetDims;
        typedef ElemType OurElemType;

        // public constructor
        // Note: use the New<> helper function that is declared next, which gives you the convenience of returning a shared_ptr
        ComputationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNodeBase(deviceId, name)
        { }

        // creation from configuration
        // Nodes with NumInputs<> should say DeclareConstructorFromConfigWithNumInputs(ClassName), and nodes without DeclareConstructorFromConfig(ClassName).
        // The macro will forward to the regular constructor of the node (which may do more than just calling the base constructor), and then attach the inputs from config.
#define DeclareConstructorFromConfig(C)              C(const ScriptableObjects::IConfigRecordPtr configp) : C(configp->Get(L"deviceId"), L"<placeholder>") { AttachInputs(configp); }
#define DeclareConstructorFromConfigWithNumInputs(C) C(const ScriptableObjects::IConfigRecordPtr configp) : C(configp->Get(L"deviceId"), L"<placeholder>") { AttachInputs(configp, this->GetExpectedNumInputs()); }

#ifdef DISPLAY_DEBUG
        virtual ~ComputationNode()
        {
            fprintf (stderr, "Called Destructor NodeName: %s\n", (msra::strfun::utf8 (NodeName())).c_str()), fflush(stderr);
        }
#endif

        // helper to load m_value from a stream
        // Since the dimensions are read as well, this function also updates m_numRows/m_numCols.
        void LoadValue(File& fstream)
        {
            CreateMatrixIfNull(m_value);
            fstream >> Value();
            // above reads dimensions, so we must update our own m_numRows/m_numCols
            m_numRows = Value().GetNumRows();
            m_numCols = Value().GetNumCols();
        }

        // reader updated m_functionValue--update our internal state, i.e. m_numCols
        // This is meant for the case when a new minibatch was read. Hence, theonly change that is allowed if for column dimension.
        virtual void NotifyFunctionValuesMBSizeModified() override final
        {
            if (m_numRows != Value().GetNumRows())
                LogicError("NotifyFunctionValuesMBSizeModified: %ls %ls operation had its row dimension %d changed by the reader to %d.", NodeName().c_str(), OperationName().c_str(), (int)m_numRows, (int)Value().GetNumRows());
            m_numCols = Value().GetNumCols();
        }
        virtual double Get00Element() const override final { return Value().Get00Element(); }

        // recover a shared_ptr from ourselves if given a naked pointer
        ComputationNodePtr shared_from_this()
        {
            return dynamic_pointer_cast<ComputationNode<ElemType>>(ComputationNodeBase::shared_from_this());
        }

        // recover a ComputationNodePtr (which is a shared_ptr) from a naked pointer to our base type (ComputationNodeBase) stored as a void* (old NDL parser does that)
        static ComputationNodePtr FromVoidPtr(void * vp)
        {
            auto p = dynamic_cast<ComputationNode<ElemType>*>((ComputationNodeBase*)vp);  // TODO: check that all void* casts really come from ComputationNodeBasePtr; or add a method ToVoidPtr(). Or get rid of the void*?!
            return p->shared_from_this();
        }

        // AttachInputs() -- attach the inputs of a node
        // This verifies the number of inputs. For that, nodes with fixed number of inputs derive from NumInputs<N>.
        // This function discovers this through RTTI and performs a runtime check. Nodes should not have additional checks in their implementation (save the code).
        // Note: Nodes with variable number of inputs will not derive from NumInputs<>, but instead check their inputs in Validate().
        void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs)
        {
#ifdef _DEBUG
            wstring name = NodeName(); name;    // (for easier debugging)
#endif
            const auto * pNumInputs = dynamic_cast<INumInputs*>(this);    // if this class also derives from NumInputs<N> then N is the expected number of inputs
            if (pNumInputs && pNumInputs->GetExpectedNumInputs() != inputs.size())
                RuntimeError("%ls operation '%ls' expects %d inputs (given: %d)", OperationName().c_str(), NodeName().c_str(), (int)pNumInputs->GetExpectedNumInputs(), (int)inputs.size());
            m_inputs.resize(inputs.size());
            for (size_t i = 0; i < m_inputs.size(); i++)
                if (inputs[i])
                    m_inputs[i] = UpCast(inputs[i]);          // (UpCast() checks the type; the assignment then downcasts it again)
                else
                    m_inputs[i] = nullptr;                    // during network creation, nullpts are possible
        }

    protected:
        // AttachInputs() from config
        void AttachInputs(const ScriptableObjects::IConfigRecordPtr configp, size_t expectedNumInputs = SIZE_MAX)
        {
            const auto inputs = GetInputsFromConfig(configp);
            if (expectedNumInputs != SIZE_MAX)
            {
                if (inputs.size() != expectedNumInputs)
                {
                    // print an error. For that, find at least one argument
                    auto * val = configp->Find(L"inputs");
                    if (!val)   // if there is no 'inputs' then get the first item of this config record for a Fail() function
                    {
                        auto members = configp->GetMemberIds();
                        if (members.size() > 0)
                            val = configp->Find(members.front());
                    }
                    if (val)
                        val->Fail(msra::strfun::wstrprintf(L"Expected %d inputs, but %d were given.", (int)expectedNumInputs, (int)inputs.size()));
                    else
                        InvalidArgument("Expected %d inputs, but %d were given.", (int)expectedNumInputs, (int)inputs.size());
                }
            }
            AttachInputs(inputs);
        }
    public:

        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
        {
            RequestMatrixFromPool(m_value, matrixPool);
        }

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
        {
            if (!IsOutputNeededDuringBackprop() && (m_value->GetMatrixType() != SPARSE))
                ReleaseMatrixToPool(m_value, matrixPool);
        }

        virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool) override
        {
            for (int i = 0; i < m_inputs.size(); i++)
            {
                if (m_inputs[i]->NeedGradient())
                    m_inputs[i]->RequestMatricesBeforeBackprop(matrixPool);
            }
        }

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
        {
            RequestMatrixFromPool(m_gradient, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
        {
            if (!IsLeaf() && !RequiresPreCompute())
            {
                if (m_gradient != nullptr && m_gradient->GetMatrixType() != SPARSE)  //since we don't have a sparse pool yet
                    ReleaseMatrixToPool(m_gradient, matrixPool);

                // Release the Value matrix only if the output value is needed during backprop
                // since in the case it isn't used, we release it during forward prop itself
                if (IsOutputNeededDuringBackprop() && m_value->GetMatrixType() != SPARSE)
                    ReleaseMatrixToPool(m_value, matrixPool);
            }
        }

        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const;

        // TODO: similar to DumpInfo; used by ExperimentalNetworkBuilder test implementation
        /*HasToString::*/ wstring ToString() const
        {
            // we format it like "name : type rows x cols ( args )"
            wstring result = /*TidyName*/(NodeName()) + L" : " + OperationName();
            result.append(msra::strfun::wstrprintf(L" %d x %d", (int)GetNumRows(), (int)GetNumCols()));
            if (m_inputs.empty()) result.append(L" ()");
            else
            {
                wstring args;
                bool first = true;
                for (auto & child : m_inputs)
                {
                    if (first)
                        first = false;
                    else
                        args.append(L"\n");
                    args.append(/*TidyName*/(child->NodeName()));
                }
                result += L" " + NestString(args, L'(', true, ')');
            }
            return result;
        }

        // update size (#columns) of node to match MBLayout
        // This must be called right before ForwardProp() the first time for a given minibatch.
        // Currently overridden by
        //  - InputValue, which verifies instead of resizing (since Resize() is specified to be destructive, it should not call it).
        //  - LearnableParameters
        //  - GMMLogLikelihoodNode (which allocates some internal temp memory).
        // Note: This only updates the dimensions but does not actually allocate anything.
        // The actual allocation happens later, in BeginForwardProp().
        // TODO: How is this function different from BeginForwardProp()?  --> answer: it will be called from there some day
        virtual void UpdateFunctionMBSize() override
        {
            if (m_pMBLayout)               // if no layout, this node contains parameters independent of MB size, don't resize
                SetDims(GetNumRows(), m_pMBLayout->GetNumCols());
        }
        virtual void VerifyDimsMatch() const override final
        {
            if (!m_value)
                return;
            auto f_numRows = m_value->GetNumRows();    // variables for easy inspection in debugger
            auto f_numCols = m_value->GetNumCols();
            if (f_numRows != m_numRows || f_numCols != m_numCols)
                LogicError("UpdateFunctionMBSize: m_value out of sync with m_numRows/m_numCols");

#ifdef SHOW_MATRIX_TYPE
            fprintf(stderr, "MatrixType %ls: %ls(%ls  %ls)\n",
                NodeName().c_str(),
                OperationName().c_str(),
                Value().GetMatrixType() == MatrixType::DENSE ? L"Dense" : L"Sparse",
                Value().GetCurrentMatrixLocation() == GPU ? L"GPU" :
                Value().GetCurrentMatrixLocation() == CPU ? L"CPU" : L"BOTH");
#endif        
        }

        void ValidateInferInputDims(size_t i, size_t rows, size_t cols) override final;

    public:
        static void MaskMissingColumnsToZero(Matrix<ElemType>& matrixToBeMasked, const MBLayoutPtr & pMBLayout, const FrameRange & fr)
        {
            //fprintf(stderr, "masking column range %d\n", (int)fr.timeIdxInSeq);
            MaskMissingColumnsTo(matrixToBeMasked, pMBLayout, fr, (ElemType)0);
        }

        void /*ComputationNodeBase::*/MaskMissingValueColumnsToZero(const FrameRange & fr) override final
        {
            //fprintf(stderr, "%ls %ls m_value ", NodeName().c_str(), OperationName().c_str());
            MaskMissingColumnsToZero(*m_value, m_pMBLayout, fr);
        }
        void /*ComputationNodeBase::*/MaskMissingGradientColumnsToZero(const FrameRange & fr) override final
        {
            //fprintf(stderr, "%ls %ls m_gradient ", NodeName().c_str(), OperationName().c_str());
            MaskMissingColumnsToZero(*m_gradient, m_pMBLayout, fr);
        }

        // for debugging, set the gaps to NaN instead (to track whether it bubbles up somewhere)
        void InvalidateMissingValueColumns(const FrameRange & fr) override final
        {
            //fprintf(stderr, "invalidating %ls %ls m_value column range %d\n", NodeName().c_str(), OperationName().c_str(), (int)fr.timeIdxInSeq);
            MaskMissingColumnsTo(*m_value, m_pMBLayout, fr, Matrix<ElemType>::MakeNan(__LINE__));
        }
        void InvalidateMissingGradientColumns(const FrameRange & fr) override final
        {
            //fprintf(stderr, "invalidating %ls %ls m_gradient column range %d\n", NodeName().c_str(), OperationName().c_str(), (int)fr.timeIdxInSeq);
            MaskMissingColumnsTo(*m_gradient, m_pMBLayout, fr, Matrix<ElemType>::MakeNan(__LINE__));
        }

        // for debugging purposes
        void /*ComputationNodeBase::*/PrintSelf(bool printMatrices = false) const
        {
            fprintf(stderr, "\n%ls[%lu, %lu] = %ls", NodeName().c_str(), GetNumRows(), GetNumCols(), OperationName().c_str());           

            if (!IsLeaf())
            {
                fprintf(stderr, "(");           
                for (size_t i=0; i<GetNumInputs(); i++)
                {
                    if (i > 0)
                        fprintf(stderr, ", ");           
                    fprintf(stderr, "%ls[%lu, %lu]", m_inputs[i] ? m_inputs[i]->NodeName().c_str():L"NULL", m_inputs[i]->GetNumRows(), m_inputs[i]->GetNumCols());
                }
                fprintf(stderr, ")");           
            }

            if (printMatrices)
            {
                fprintf (stderr, "\n    $$$$ Function Values\n");
                Value().Print("FunctionValue");

                fprintf (stderr, "\n    $$$$ Gradient Values\n");
                Gradient().Print("GradientValue");
            }
        }

        // up-cast to make life easier
        static ComputationNodePtr UpCast(ComputationNodeBasePtr inode)
        {
            ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(inode);
            if (!node)
                InvalidArgument("an ComputationNodeBasePtr of mismatching precision was passed");
            return node;
        }

        inline ComputationNodePtr Input(const size_t inputIndex) const
        {
            if (inputIndex >= m_inputs.size())
                LogicError("Inputs: inputIndex %d is out of range for %ls %ls operation.", (int)inputIndex, NodeName().c_str(), OperationName().c_str());
            return UpCast(m_inputs[inputIndex]);
        }

        void /*ComputationNodeBase::*/SetInput(const size_t childIndex, const ComputationNodeBasePtr& inode) override
        {
            const ComputationNodePtr node = UpCast(inode);

            //require first nodes specified before the second to avoid null nodes condition.
            if (childIndex > m_inputs.size())
                InvalidArgument("SetInput: You must specify the input for children with index less than this one first.");

            // expand the inputs to exist up to the desired index
            while (childIndex >= m_inputs.size())
                m_inputs.push_back(nullptr);

            // set the input value
            m_inputs[childIndex] = node;
        }

        const Matrix<ElemType>& Value() const    { return *m_value; }
        Matrix<ElemType>& Value()                { return *m_value; }

        const Matrix<ElemType>& Gradient() const { return *m_gradient; }
        Matrix<ElemType>& Gradient()             { return *m_gradient; }

    protected:
        std::vector<TensorView<ElemType>> GetTensorsForwardBinary(const FrameRange & fr);

    public:
        // Function to return the number of columns for whole batch or single frame
        size_t GetNumColsFor(const FrameRange & fr/*select frame or entire batch*/)
        {
            try
            {
                return ColumnRangeWithMBLayoutFor(GetNumCols(), fr, m_pMBLayout).second;
            }
            catch (const logic_error & e)   // catch the error and rethrow it with the node name attached
            {
                LogicError("%s, for %ls %ls operation.", e.what(), NodeName().c_str(), OperationName().c_str());
            }
        }

        // function to access any input and output, value and gradient, whole batch or single frame
        // Note: This returns a reference into 'data' in the form of a column slice, i.e. a small matrix object that just points into 'data'.
        Matrix<ElemType> DataFor(Matrix<ElemType> & data, const FrameRange & fr/*select frame or entire batch*/)
        {
            try
            {
                return DataWithMBLayoutFor(data, fr, m_pMBLayout);
            }
            catch (const logic_error & e)   // catch the error and rethrow it with the node name attached
            {
                LogicError("%s, for %ls %ls operation.", e.what(), NodeName().c_str(), OperationName().c_str());
            }
        }

        Matrix<ElemType> ValueFor(const FrameRange & fr/*select frame or entire batch*/)
        {
            return DataFor(Value(), fr);
        }
        Matrix<ElemType> GradientFor(const FrameRange & fr/*select frame or entire batch*/)
        {
            return DataFor(Gradient(), fr);
        }
        // use the following two versions if you assume the inputs may contain gaps that must be set to zero because you want to reduce over frames with a BLAS operation
        Matrix<ElemType> MaskedValueFor(const FrameRange & fr/*select frame or entire batch*/)
        {
            MaskMissingValueColumnsToZero(fr);
            return ValueFor(fr);
        }
        Matrix<ElemType> MaskedGradientFor(const FrameRange & fr/*select frame or entire batch*/)
        {
            MaskMissingGradientColumnsToZero(fr);
            return GradientFor(fr);
        }
        // tensor variants
        TensorView<ElemType> DataTensorFor(Matrix<ElemType> & data, size_t rank, const FrameRange & fr)
        {
            return TensorView<ElemType>(DataFor(data, fr), GetTensorShape(rank, fr));
        }
        TensorView<ElemType> ValueTensorFor(size_t rank, const FrameRange & fr)
        {
            return TensorView<ElemType>(ValueFor(fr), GetTensorShape(rank, fr));
        }
        TensorView<ElemType> GradientTensorFor(size_t rank, const FrameRange & fr)
        {
            return TensorView<ElemType>(GradientFor(fr), GetTensorShape(rank, fr));
        }

        // update the actual matrix allocation for m_value based on the node dimension
        void UpdateFunctionValuesSize()
        {
            Value().Resize(m_numRows, m_numCols);
        }

        // this is called before a node's ForwardProp() function is called (in loops: for the first time)
        // This is where we
        //  - update the node dimension based on actual MB size
        //  - (re-)allocate the m_value matrix, which may be shared across nodes and thus have changed dimensions
        virtual void /*IComputationNode::*/BeginForwardProp() override             // called before first iteration step of ForwardProp()
        {
            Base::BeginForwardProp();

            // update dimensions based on MB size
            UpdateFunctionMBSize();

            // update the actual m_value allocation
            if (!IsLeaf() && !RequiresPreCompute())     // TODO: guard this through overrides instead
                UpdateFunctionValuesSize();

            // and make sure dimensions are what we expect
            VerifyDimsMatch();
        }

#ifdef _DEBUG
        // NaN checks
        virtual void /*IComputationNode::*/EndForwardProp() override
        {
            Base::EndForwardProp();
#ifdef TRACK_GAP_NANS
            MaskMissingValueColumnsToZero(FrameRange(m_pMBLayout));       // HasNaN() operates on a whole matrix, so first flatten all gaps to 0
            if (Value().HasNan("EndForwardProp"))
                LogicError("%ls %ls operation unexpectedly produced NaN values.", NodeName().c_str(), OperationName().c_str());
#endif
            InvalidateMissingValueColumns(FrameRange(m_pMBLayout));        // blast NaNs into columns that are gaps in a packed layout
        }
#endif

#if 0   // (keep it around in case we need to add stuff in the future)
        virtual void /*IComputationNode::*/BeginBackprop() override
        {
            Base::BeginBackprop();
        }
#endif

#ifdef _DEBUG
        virtual void /*IComputationNode::*/EndBackprop() override
        {
            Base::EndBackprop();
#ifdef TRACK_GAP_NANS
            for (size_t i = 0; i < m_inputs.size(); i++)
            {
                ComputationNodePtr child = Input(i);
                if (child->m_needsGradient)
                {
                    child->MaskMissingGradientColumnsToZero(FrameRange(child->GetMBLayout()));       // HasNaN() operates on a whole matrix, so first flatten all gaps to 0
                    if (child->Gradient().HasNan("EndBackprop"))
                        LogicError("%ls %ls operation unexpectedly produced NaN gradients.", child->NodeName().c_str(), child->OperationName().c_str());
                }
            }
#endif
        }
#endif

        // this is the entry point from Network; while it will call virtual BackpropTo() into the actual node implementation
        // TODO: move to -Base (or -Network?)
        void Backprop(const FrameRange & fr, bool childrenInThisLoop, bool childrenInOuterLoop) override
        {
            if (fr.IsAllFrames() && IsPartOfLoop() && childrenInThisLoop)
                LogicError("%ls %ls operation: Backprop called with whole-batch FrameRange on node that participates in a loop", NodeName().c_str(), OperationName().c_str());

            for (size_t i = 0; i < m_inputs.size(); i++)
            {
                ComputationNodePtr child = Input(i);
                if (child->m_needsGradient &&
                    (childrenInThisLoop  && child->IsPartOfLoop() == IsPartOfLoop() ||
                     childrenInOuterLoop && child->IsPartOfLoop() != IsPartOfLoop()
                    ))
                {
                    //fprintf(stderr, "Backprop: %ls %ls operation -> child %d %ls %ls\n", NodeName().c_str(), OperationName().c_str(), (int)i, child->NodeName().c_str(), child->OperationName().c_str());
                    if (!m_needsGradient)
                        LogicError("%ls %ls operation has m_needsGradient set to false but children require it.", NodeName().c_str(), OperationName().c_str());
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %ls(%ls)\n", i, child->OperationName().c_str(), child->NodeName().c_str());
#endif
#if DUMPOUTPUT
                    fprintf(stderr, "Backprop%d_%ls\n", i, NodeName().c_str());
#endif
                    child->LazyZeroGradient();              // set gradient to 0 if this is the first time

                    // If we propagate from a loop to a node that is outside the loop, we are not efficient.
                    // This case is handled by SEQTraversalFlowControlNode::Backprop().
                    // The check below is to verify that.
                    if (IsPartOfLoop() && !child->IsPartOfLoop() && !fr.IsAllFrames())
                    {
                        LogicError("Backprop: Inefficiency: %ls %ls operation in loop propagates gradient to non-loop %ls %ls\n",
                                   NodeName().c_str(), OperationName().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
                    }

                    //fprintf(stderr, "BackpropTo %d %d %ls %ls\n", (int)fr.timeIdxInSeq, (int)i, NodeName().c_str(), OperationName().c_str());
                    BackpropTo(i, fr);     // this computes partial wrt to the child and sums the gradient value in the child
                }
#ifdef DISPLAY_DEBUG
                else fprintf (stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, child->OperationName().c_str(), child->NodeName().c_str());
#endif
            }
        }

        // TODO: why of the inputs, and not the node itself?
        void /*ComputationNodeBase::*/ZeroGradientsOfInputs() override   // clears the lazy-init flags (LazyZeroGradient() actually clears the values lazily)
        {
            for (size_t i = 0; i < m_inputs.size(); i++)
                Input(i)->m_gradientInitialized = false;
        }

        // lazy resetting of gradient
        void LazyZeroGradient()
        {
            if (!m_needsGradient)
                LogicError("%ls %ls operation: LazyZeroGradient() called although this node needs no gradient.", NodeName().c_str(), OperationName().c_str());

            if (m_gradientInitialized)
                return;

            Gradient().Resize(GetNumRows(), GetNumCols());
            Gradient().SetValue(0);

            m_gradientInitialized = true;
        }

        // NOTE: we should reimplement this to be thread-safe and use a larger than requested initialized memory block
        // we can then just wrap that memory block in a matrix of the correct dimensions since it will be const no one can change it
        // should only need one memory block per device
        // Thread-safety could be achieved by changing this to a shared_ptr.
        // When using the TensorView interface, one could instead just use a 1x1 matrix with a view that broadcasts its columns (stride 0).
        static const Matrix<ElemType>& ConstOnes(const size_t rows, const size_t cols, const DEVICEID_TYPE deviceId)
        {
            if (s_constOnes.find(rows) == s_constOnes.end() ||
                s_constOnes[rows].find(cols) == s_constOnes[rows].end()) //not found
            {
                Matrix<ElemType>* matrix = new Matrix<ElemType>(rows, cols, (DEVICEID_TYPE)deviceId);
                matrix->SetValue(1);
                s_constOnes[rows][cols] = matrix;
            }

            Matrix<ElemType>* m = s_constOnes[rows][cols];
            m->TransferFromDeviceToDevice(m->GetDeviceId(), deviceId);

            return *m;
        }

        void CreateGradientMatrixIfNull()
        {
            CreateMatrixIfNull(m_gradient);
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

        void RequestMatrixFromPool(shared_ptr<Matrix<ElemType>>& matrixPtr, MatrixPool& matrixPool)
        {
            if (matrixPtr == nullptr)
            {
                matrixPtr = matrixPool.Request<ElemType>(m_deviceId);
            }
        }

        void ReleaseMatrixToPool(shared_ptr<Matrix<ElemType>>& matrixPtr, MatrixPool& matrixPool)
        {
            assert(matrixPtr != nullptr);
            matrixPool.Release<ElemType>(matrixPtr);
        }

        //to be called by derived classed if that class needs to print node values
        void PrintNodeValuesToFile(const bool printValues, File& fstream) const
        {
            if (printValues)
            {
                fstream << wstring(L"\n");
                const Matrix<ElemType>&  m = Value();
                for (size_t i=0; i < m.GetNumRows(); i++)
                {
                    for (size_t j=0; j < m.GetNumCols(); j++)
                    {
                        fstream << m(i,j);
                    }
                    fstream << wstring(L"\n");
                }
                fstream << wstring(L"####################################################################");
            }
        }

    public:
        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = UpCast(nodeP);
                *node->m_value = *m_value;
                if (m_gradient)
                    *node->m_gradient = *m_gradient;
                else
                    node->m_gradient = nullptr;
            }
        }

        // duplicate a node
        ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags)
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;
            ComputationNodeBasePtr node(NewThis(m_deviceId, name)); // NewThis() is a virtual function that creates a new node of the actual type of 'this'
            node->CopyTo(shared_from_this(), newName, flags);       // note: shared_from_this() is the base class, but CopyTo() up-casts it as needed
            return node;
        }

        // these are used to export hidden state activations
        virtual bool GetHistory(Matrix<ElemType>&, bool) { return false; }
        virtual void SetHistory(const Matrix<ElemType>&) { }

        /// these two are used to pass gradients from future minibatch
        virtual void GetErrorsToPreviousMinibatch(Matrix<ElemType>&) {}
        virtual void SetErrorsFromFutureMinibatch(Matrix<ElemType>&) {}

    protected:

        shared_ptr<Matrix<ElemType>> m_value, m_gradient;

        static std::map<size_t, std::map<size_t, Matrix<ElemType>*>> s_constOnes;
    };

    // convenience wrapper for ComputationNode::New()
    template<class C, class... _Types> inline shared_ptr<C> New(_Types&&... _Args)
    {
        return make_shared<C>(forward<_Types>(_Args)...);
        //return ComputationNode<typename C::OurElemType>::template New<C>(forward<_Types>(_Args)...);
    }

    // =======================================================================
    // ComputationNodeNonLooping -- abstract base class for computation nodes that do not implement eval/partial for individual frames
    // Such as CRFNode, LSTMNode, ParallelNode, SequenceDecoderNode, TimeReverseNode (BatchModeNode), and TransposeNode.
    // =======================================================================

    // This will provide default implementations for those two functions that will fail at runtime with a meaningful error.
    // TODO: Most of these are reduce nodes that output a single number, no MBLayout. Maybe abstract those out further
    template<class ElemType>
    class ComputationNodeNonLooping : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base;
    public:
        ComputationNodeNonLooping(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        // these two implement the ComputationNode<> interface
        void ForwardProp(const FrameRange & fr) override final
        {
            if (fr.IsAllFrames())
                ForwardPropNonLooping();
            else
                LogicError("%s node should never be in a loop.", typeid(*this).name());
        }
        void BackpropTo(const size_t inputIndex, const FrameRange & fr) override final
        {
            if (fr.IsAllFrames())
                BackpropToNonLooping(inputIndex);
            else
                LogicError("%s node should never be in a loop.", typeid(*this).name());
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
        FlowControlNode() : ComputationNodeBase(DEVICEID_NOTYETDETERMINED/*we don't own matrices*/, L""/*name: we don't care*/) { }

#pragma warning (disable: 4100)
        // these are meant to be implemented by ComputationNode<ElemType> but should never be called on traversal nodes
        // TODO: There are too many of these. This indicates improper class hierarchies.
        virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) override { NOT_IMPLEMENTED; }
        virtual void Validate(bool isFinalValidationPass) override { NOT_IMPLEMENTED; }          // main base validation function
        virtual void InferImageDimsFromInputs() override { NOT_IMPLEMENTED; }
        virtual void Save(File& fstream) const override { NOT_IMPLEMENTED; }
        virtual void Load(File& /*fstream*/, size_t /*modelVersion*/) override { NOT_IMPLEMENTED; }
        virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const override { NOT_IMPLEMENTED; }
        virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) override { NOT_IMPLEMENTED; }
        virtual double Get00Element() const override     { NOT_IMPLEMENTED; }
        virtual void UpdateFunctionMBSize() override     { NOT_IMPLEMENTED; }
        virtual void VerifyDimsMatch() const override    { NOT_IMPLEMENTED; }
        virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs) override { NOT_IMPLEMENTED; }
        virtual void PrintSelf(bool) const override { NOT_IMPLEMENTED; }
        virtual void ValidateInferInputDims(size_t,size_t,size_t) override { NOT_IMPLEMENTED; }
        virtual void SetInput(const size_t,const Microsoft::MSR::CNTK::ComputationNodeBase::ComputationNodeBasePtr &) override { NOT_IMPLEMENTED; }
        virtual void ZeroGradientsOfInputs(void) override { NOT_IMPLEMENTED; }
        virtual void MaskMissingValueColumnsToZero(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual void MaskMissingGradientColumnsToZero(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual void InvalidateMissingValueColumns(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual void InvalidateMissingGradientColumns(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual std::wstring ToString(void) const override { NOT_IMPLEMENTED; }
        // these are meant to be called during computation, so provide dummy implementations
        virtual bool RequiresPreCompute() const override { return false; }                    // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual void PrintSelfBeforeValidation() const override { }
        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const override { }
    protected:
    public: // needed in ComputationNetwork::FindInRecurrentLoops(), which really should be part of SEQTraversalFlowControlNode
        std::vector<ComputationNodeBasePtr> m_nestedNodes;                  // nodes tucked away in this node, in evaluation order
    };

    // =======================================================================
    // ILateAttachingNode -- helper wrapper class for ComputationNodes that must AttachInputs() late due to circular references
    // =======================================================================

    // Instantiate with LateAttachingNode<node type>(lambda, args for node constructor).
    // To resolve, call AttachInputs()
    // TODO: This is a bit indirect. Can it be done more nicely?
    struct ILateAttachingNode { virtual void LateAttachInputs() = 0; };
    template<class N>
    class LateAttachingNode : public N, public ILateAttachingNode
    {
        typedef typename N::OurElemType ElemType;
        function<void(ComputationNode<ElemType>*)> attachInputs;
    public:
        // constructor
        template<class... _Types>
        LateAttachingNode(DEVICEID_TYPE deviceId, const wstring & name, const function<void(ComputationNode<ElemType>*)> & attachInputs, _Types&&... _Args) : attachInputs(attachInputs), N(deviceId, name, forward<_Types>(_Args)...) {}
        // the one member that does the work
        void /*ILateAttachingNode::*/LateAttachInputs()
        {
            attachInputs(dynamic_cast<N*>(this));
            attachInputs = [](ComputationNode<ElemType>*){ LogicError("LateAttachingNode::AttachInputs: must only be called once"); };
        }
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
#define UsingComputationNodeMembers /*without OperationName; needed to support inconsistent pattern of InputValue--TODO: This comment it out of date. */    \
protected: \
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr; \
    using Base::m_deviceId; using Base::SetDims; using Base::GetNumRows; using Base::GetNumCols; using Base::UpdateFunctionValuesSize; using Base::LoadValue; \
    using Base::m_pMBLayout; using Base::GetNumTimeSteps; using Base::GetNumParallelSequences; \
    using Base::MaskMissingColumnsToZero; using Base::MaskMissingValueColumnsToZero; using Base::MaskMissingGradientColumnsToZero; using Base::InvalidateMissingValueColumns; using Base::InvalidateMissingGradientColumns; \
    using Base::DataFor; using Base::ValueFor; using Base::Gradient; using Base::GradientFor; \
    using Base::MaskedValueFor; using Base::MaskedGradientFor; using Base::DataTensorFor; using Base::ValueTensorFor; using Base::GradientTensorFor; \
    using Base::ForwardProp; using Base::BackpropTo; \
    using Base::m_inputs; using Base::m_value; using Base::m_gradient; \
    using Base::m_inputSampleLayout; using Base::m_sampleLayout; \
    using Base::m_parameterUpdateRequired; using Base::m_nodeName; \
    using Base::CreateMatrixIfNull; using Base::RequestMatrixFromPool; using Base::ReleaseMatrixToPool; \
    using Base::CreateUniqId; \
    using Base::GetNumInputs; using Base::ZeroGradientsOfInputs; using Base::VerifyDims; \
    using Base::ConstOnes; \
    using Base::GetTensorsForwardBinary; using Base::DetermineElementwiseTensorRank; \
    using Base::GetImageLayout; using Base::InferImageDimsFromInput; using Base::InferImageDimsFromInputs; using Base::InferMBLayoutFromInputsForStandardCase; \
    using Base::CopyTo; using Base::CreateUniqNodeName; using Base::DetachInputs; using Base::GetInputsFromConfig; \
    using Base::DumpNodeInfo; using Base::EnumerateNodes; \
    using Base::HasMBLayout; using Base::GetMBLayout; using Base::LinkToMBLayout; \
    using Base::Input; using Base::SetInput; \
    using Base::IsInputAnImage; using Base::IsEqualTo; using Base::IsOutputOlderThanInputs; using Base::IsLeaf; using Base::SetParameterUpdateRequired; \
    using Base::Load; \
    using Base::PrintNodeValuesToFile; using Base::PrintSelfBeforeValidation; \
    using Base::Save; using Base::UpdateFunctionMBSize; \
    using Base::RequestMatricesBeforeForwardProp; using Base::ReleaseMatricesAfterForwardProp; \
    using Base::RequestMatricesBeforeBackprop; using Base::ReleaseMatricesAfterBackprop; \
    using Base::InputUsedInComputingInputNodesGradients; using Base::OutputUsedInComputingInputNodesGradients; \
    using Base::Validate; using Base::ValidateUnaryMap; using Base::ValidateBinaryZip; using Base::ValidateUnaryReduce; using Base::ValidateBinaryReduce; using Base::ValidateInferBinaryInputDims; using Base::ValidateInferInputDims; \
public: \
    using Base::RequiresPreCompute; \
    using Base::AttachInputs; using Base::CreateGradientMatrixIfNull; using Base::NodeName; \
    using Base::Value; using Base::GetTensorShape;

#define ComputationNodeBoilerplate \
protected:    /* some boilerplate goes here */ \
    virtual const std::wstring OperationName() const override { return TypeName(); } \
    virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) override { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }

#define UsingComputationNodeMembersBoilerplate \
    ComputationNodeBoilerplate; UsingComputationNodeMembers

    // =======================================================================
    // a few standard base classes for N-nary operations
    // =======================================================================

    // -----------------------------------------------------------------------
    // BinaryElementWiseNode (operand1, operand2)
    //
    // binary elementwise operations that are implemented with the tensor lib
    //
    // Derived clases only need to override ForwardProp() and BackpropTo().
    // -----------------------------------------------------------------------

    template<class ElemType>
    class BinaryElementWiseNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        BinaryElementWiseNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual bool OutputUsedInComputingInputNodesGradients() const override
        {
#if DUMPOUTPUT
            return true;
#else
            // By default, the BinaryElementWiseNode does not require its output value for computing
            // the gradients of its input nodes
            return false;
#endif
        }

        virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
        {
            // By default, the BinaryElementWiseNode does not require any of it's input's values for computing
            // the gradients of its input nodes
            UNREFERENCED_PARAMETER(childIndex);
            return false;
        }

        virtual void /*IComputationNode::*/BeginForwardProp() override             // called before first iteration step of ForwardProp()
        {
            Base::BeginForwardProp();
            // we switch result to dense as a work-around because ColumnSlice doesn't support all the sparse formats
            // TODO: This is a stopgap. Is this the right thing to do? It changes the matrix type in-place.
            Value().SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateBinaryZip(isFinalValidationPass, true/*allowMultiples*/);
        }

        virtual void InferImageDimsFromInputs()
        {
            // TODO: change to infer as maximum of the two
            if (IsInputAnImage(0))
                InferImageDimsFromInput(0);
            else
                InferImageDimsFromInput(1);
        }
    };

#define UsingBinaryElementwiseNodeBaseMembers UsingComputationNodeMembersBoilerplate;

#pragma endregion base computation class

}}}
