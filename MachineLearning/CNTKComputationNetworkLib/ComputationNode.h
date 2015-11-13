//
// <copyright file="ComputationNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ScriptableObjects.h"
#include "Sequences.h"
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

//#define RNN_DEBUG 1
#define DEFAULT_HIDDEN_ACTIVATION 0.1

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED \
{   \
    fprintf(stderr, "Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__); \
    LogicError("Not Implemented"); \
}
#endif

#pragma warning (disable: 4267)

//version number to control how to read and write 
#define CNTK_MODEL_VERSION_1 1
#define CNTK_MODEL_VERSION_2 2
#define CURRENT_CNTK_MODEL_VERSION 2

// #define TRACK_GAP_NANS  // if defined then initialize layout gaps to NaN and do NaN checks

namespace Microsoft { namespace MSR { namespace CNTK {

    enum CopyNodeFlags
    {
        copyNodeNull = 0,               // invalid value
        copyNodeValue=1,                // copy everything but the children links
        copyNodeChildren=2,             // only copy over children links
        copyNodeAll=3,                  // copy everything
        copyNodeChildrenCrossNetwork=4, // allow a cross network child copy
    };

    // describes inner layout of feature vector that is an image
    // TODO: This will grow into a more general tensor mechanism.
    // TODO: SaveToFile() and LoadFromFile() currently use individual elements; provide an overload for the entire object.
    struct ImageLayout
    {
        size_t width, height, channels;
        // BUGBUG: This initialization is not correct. This must match GetNumRows(). We probably cannot have all three members here.
        // Idea: We could construct this thing with a ref to the enclosing ComputationNode, and replace 'width' by an expression.
        ImageLayout() : width(1), height(1), channels(1) { }
        ImageLayout(size_t width, size_t height, size_t channels) : width(width), height(height), channels(channels) { }
        //void Set(size_t width, size_t height, size_t channels) { this->width = width; this->height = height; this->channels = channels; }
        void Invalidate() { width = SIZE_MAX; height = SIZE_MAX; channels = SIZE_MAX; } // TODO: clean up the valid/invalid situation (this is currently done inconsistently)
        size_t GetNumElements() const { return width * height * channels; }
        bool operator==(const ImageLayout & other) const { return width == other.width && height == other.height &&channels == other.channels; }
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
#define OperationNameOf(T) (T<float>::TypeName())    // we are templated, but for this the type param matters not. So we just pick one, and hide that fact.

        virtual void UpdateFunctionMBSize() = 0;                // recalculate our column dimension from MBLayout

        virtual void OnEvaluateBeginIteration() = 0;
        virtual void EvaluateThisNode(const FrameRange &) = 0;  // forward prop for one minibatch
        virtual void OnEvaluateEndIteration() = 0;              // called after last iteration step of EvaluateThisNode()

        virtual void OnComputeGradientBeginIteration() = 0;     // called before first iteration step of ComputeGradient()
        virtual void ComputeInputPartial(const size_t inputIndex, const FrameRange &) = 0;
        virtual void OnComputeGradientEndIteration() = 0;       // called after last iteration step of ComputeGradient()

        // TODO: this one does not quite fit here --they are implemented in Base and overridden by the ControlFlowNodes
        // functions that are called from Network, but not necessarily overridden by the node implementations themselves
        virtual void ComputeGradientForChildren(const FrameRange & frameRange, bool childrenInThisLoop, bool childrenInOuterLoop) = 0;

        // --- optional overrides that add functionality

        // Any override must call Base version as well.
        // Default implementations are in ComputationNodeBase or ComputationNode<ElemType>.

        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool) = 0;         //request matrices needed to do node function value evaluation
        virtual void ReleaseMatricesAfterEval(MatrixPool& matrixPool) = 0;          //release temp matrices that are only used by forward computation. Don't release matrices that need to be used in the gradient computation
        virtual void AllocateGradientMatricesForChildren(MatrixPool& matrixPool) = 0;
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool) = 0; //request matrices that are needed for gradient computation
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool) = 0;  //release gradient and temp matrices that no longer needed after all the children's gradients are computed.

        virtual void Validate(bool isFinalValidationPass) = 0;          // main base validation function
        virtual void InferImageDimsFromInputs() = 0;
        virtual void SaveToFile(File& fstream) const = 0;
        virtual void LoadFromFile(File& /*fstream*/, size_t /*modelVersion*/) = 0;
        // TODO: is this always just called with deviceId == m_deviceId?   TODO: Where is this actually EVERY called?? I don't see it!
        virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const = 0;

        // --- optional overrides that describe a feature or property of the node

        virtual bool RequiresPreCompute() const = 0;                    // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() = 0; // // indicates whether special handling is needed.The standard handleing will be just mask the function values after the evalaution and mask the gradient before gradiant computation for the children. this is not valid for all criterion nodes whose result is a scalar.

        // --- optional overrides for more informative logging

        virtual void PrintSelfBeforeValidation() const = 0;             // called in validation loop right before Validate()
        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const = 0;
    protected:
        virtual ~IComputationNode() { }
    };

    // =======================================================================
    // ComputationNetworkOwnedNodeState -- class to collect ComputationNode members that are really owned by ComputationNetwork
    // These members are only to be set, changed, and read by ComputationNetwork code.
    // TODO: We could go much further and move all network-level evaluation routines into here as well.
    //       I won't do it now as it will create a massive diff that would make merging of other ongoing changes much harder.
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

        static bool ByVisitedOrder(const ComputationNetworkOwnedNodeState * lhs, const ComputationNetworkOwnedNodeState * rhs)  // sorting predicate
        {
            return lhs->m_visitedOrder < rhs->m_visitedOrder;
        }

        bool IsPartOfLoop() const { return m_isPartOfLoop; }

    private:

        bool m_isPartOfLoop;        // true if this loop is part of a recurrent loop

    protected:  // TODO: should be fully encapsulated here
        bool m_needsGradient;   // true if this node or any children need a gradient to be computed (for own consumption or propagation to somewhere in the child tree)

    protected:
        // owned by FormRecurrentLoops() and stuff it calls, only used from inside there (FormRecurrentLoops() calls PurgeStateForFormingRecurrentLoops() at its end to make that super-clear)
        void PurgeStateForFormingRecurrentLoops()
        {
            m_loopId = -1;
            m_visitedOrder = -1;
            m_indexInLoop = 0;
            m_visited = false;
            m_index = -1;
            m_lowLink = -1;
            m_inStack = false;
        }

        int m_loopId;           // index into recurrent info array (TODO: verify this)
        int m_visitedOrder;     // remembers order in which nodes were visited by EnumerateNodes(), but gets updated
        bool m_visited;         // note: also used by ValidateSubNetwork()
        int m_indexInLoop;
        // only used inside DetermineSCCs():
        int m_index;            // index denoting order in which nodes were visited in DetermineSCCs()
        int m_lowLink;          // min of m_index over all nodes within a single loop
        bool m_inStack;
    };

    // =======================================================================
    // TimeStamp -- helper class to manage a time stamp
    // =======================================================================

    class TimeStamp
    {
    public:
        TimeStamp() { ResetEvalTimeStamp(); }
        void CopyTo(TimeStamp & other) const { other.m_evalTimeStamp = m_evalTimeStamp; }
        void ResetEvalTimeStamp() { m_evalTimeStamp = s_timeStampCounter; }
        int64_t GetEvalTimeStamp() const { return m_evalTimeStamp; }

        // create a new unique time stamp
        void UpdateEvalTimeStamp() { m_evalTimeStamp = CreateUniqId(); }

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
    // TODO: decide the name. This does contain actual members such as the node name, so it's not really a pure interface.
    // =======================================================================

    class ComputationNodeBase :
        public IComputationNode,
        public/*protected*/ ComputationNetworkOwnedNodeState,   // TODO: figure this out, somehow the 'friend' thing does not work
        public TimeStamp,                                       // for time-stamp management
        public ScriptableObjects::ComputationNodeObject,
        public ScriptableObjects::WithTag, public ScriptableObjects::HasName, public ScriptableObjects::HasToString,
        public std::enable_shared_from_this<ComputationNodeBase>
    {
        // note: enable_shared_from_this<> allows to create a shared_ptr from a raw pointer to this that is correctly aware of all other shared_ptrs (same ref count)
    public:
        typedef shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

        ComputationNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            m_deviceId(deviceId),
            m_parameterUpdateRequired(false), m_gradientInitialized(false),
            m_nodeName(name == L"" ? CreateUniqNodeName() : name),
            m_numRows(0), m_numCols(0)
        {
        }
        virtual ~ComputationNodeBase(){}

        virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            if (OperationName() != node->OperationName())
                RuntimeError("Cannot copy from one node type to another node type");
            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                node->m_children = m_children;
            }
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                node->m_deviceId = m_deviceId;
                node->m_parameterUpdateRequired = m_parameterUpdateRequired;
                node->m_nodeName = newName;

                node->m_inputImageLayout = m_inputImageLayout;
                node->m_outputImageLayout = m_outputImageLayout;

                ComputationNetworkOwnedNodeState::CopyTo(*node);
                TimeStamp::CopyTo(*node);
            }
        }

        virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) = 0;

        // TODO: make sure this does not get implemented in any of the base classes
        DEVICEID_TYPE GetDeviceId() const { return m_deviceId; }    // TODO: remove, only used from copy constructor which will go away

        virtual void SaveToFile(File& fstream) const
        {
            fstream << OperationName() << NodeName();
        }

        virtual void LoadFromFile(File& /*fstream*/, size_t /*modelVersion*/)
        {
            // it is assumed that OperationName and NodeName have already been consumed--some asymmetry between Save and Load
            // base class has nothing to load
        }

        // float/double-independent access to the m_functionValues for a few specific use cases
        // TODO: Not nice. This would go away if we abstracted out the matrix type as well from float/double.
        size_t GetNumRows() const { return m_numRows; }
        size_t GetNumCols() const { return m_numCols; }
        pair<size_t, size_t> GetDims() { return make_pair(GetNumRows(), GetNumCols()); }
        virtual // for now virtual as this still updates m_functionValues
        void SetDims(size_t rows, size_t cols)
        {
            m_numRows = rows;
            m_numCols = cols;
            // actual memory allocation happens elsewhere
            // NOTE: current ComputationNode<> overrides this in order to still do actual memory allocation like before
        }
        void SetDims(ComputationNodeBasePtr node) { SetDims(node->GetNumRows(), node->GetNumCols()); }
        virtual void NotifyFunctionValuesModified() { } // someone outside changed our m_functionValues--update our internal state, e.g. m_numRows, m_numCols
        void VerifyDims(size_t rows, size_t cols)
        {
            if (rows != GetNumRows() || cols != GetNumCols())
                LogicError("VerifyDims: expected m_functionValues size %d x %d, but it is %d x %d",
                           (int)rows, (int)cols, (int)GetNumRows(), (int)GetNumCols());
        }
        virtual void VerifyDims(ComputationNodeBasePtr node) { VerifyDims(node->GetNumRows(), node->GetNumCols()); }
        virtual double Get00Element() const = 0;

        // validation
        virtual void Validate(bool isFinalValidationPass)           // main base validation function
        {
            // check for NULL pointers
            for (size_t i = 0; i < m_children.size(); i++)
            {
                if (!m_children[i])
                    RuntimeError("Validate: Input [%d] of %ls node '%ls' is empty (NULL, not connected).", (int)i, OperationName().c_str(), NodeName().c_str());
            }
            // check for empty inputs
            if (isFinalValidationPass)
            {
                for (const auto & child : m_children)
                {
                    if (child->GetNumRows() == 0 || (!child->HasMBLayout() && child->GetNumCols() == 0))
                        RuntimeError("%ls %ls operation: input %ls %ls has 0 elements.",
                                     NodeName().c_str(), OperationName().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
                }
            }
        }
        // helper functions for common cases
    private:
        // determine number of columns from a child and/or layout
        size_t DetermineNumCols(const ComputationNodeBasePtr & child) const
        {
            size_t childCols = child->GetNumCols();     // this is what the child says
            if (!m_pMBLayout)                           // no layout: copy from child
                return childCols;
            size_t cols = m_pMBLayout->GetNumCols();    // layout: get it from there, but validate against child
            if (childCols != cols)
                RuntimeError("%ls %ls operation: Mismatch in number of columns", OperationName().c_str(), NodeName().c_str());
            return cols;
        }
    protected:
        void ValidateUnaryMap(bool isFinalValidationPass);
        void ValidateUnaryReduce(bool isFinalValidationPass);
        void ValidateInferBinaryChildrenDims();
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

        virtual void DetachInputs() { m_children.clear(); }

        const std::vector<ComputationNodeBasePtr> & GetChildren() const { return m_children; }
        ComputationNodeBasePtr Inputs(size_t index) const { return m_children[index]; } // TODO: delete this; change to m_children

        //return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual bool /*IComputationNode::*/RequiresPreCompute() const { return false; }

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
    public: // the following should be protected, but nodes inquire about their children, requiring public access
        // This is used at 284 places inside nodes, most of the time as
        // ...Slice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences()), m_pMBLayout)
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
                //return GetNumCols();
#if 0       // can't check here; this is sometimes inquired as part of the process of setting the right #cols
            if (m_pMBLayout->GetNumTimeSteps() * m_pMBLayout->GetNumParallelSequences() != GetNumCols())
            {
                // TODO: remove this fprintf() once it no longer triggers
                fprintf(stderr, "GetNumTimeSteps: inconsistency between layout and actual number of columns for node '%ls', seq=%d x T=%d vs. cols=%d\n",
                        NodeName().c_str(), (int)m_pMBLayout->GetNumParallelSequences(), (int)m_pMBLayout->GetNumTimeSteps(), (int)GetNumCols());
                LogicError("GetNumTimeSteps: inconsistency between layout and actual number of columns for node '%ls', seq=%d x T=%d vs. cols=%d",
                           NodeName().c_str(), (int)m_pMBLayout->GetNumParallelSequences(), (int)m_pMBLayout->GetNumTimeSteps(), (int)GetNumCols());
            }
            // TODO: ^^ much of this should go away, as in the future, the layout will always correctly know the #samples
#endif
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
                for (size_t i = 0; i<ChildrenSize(); i++)
                {
                    const auto & child = m_children[i];
                    if (i > 0)
                        fprintf(stderr, ", ");

                    if (child == nullptr)
                    {
                        fprintf(stderr, "NULL");
                        continue;
                    }

                    const char * mbSizeMark = child->m_pMBLayout ? "MBSize " : "";
                    if (IsChildAnImage(i))  //image
                        fprintf(stderr, "%ls[%lu {W=%lu, H=%lu, C=%lu}, %s%lu]", child->NodeName().c_str(), child->GetNumRows(),
                                child->m_outputImageLayout.width, child->m_outputImageLayout.height, child->m_outputImageLayout.channels, mbSizeMark, child->GetNumCols());
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

        bool IsLeaf() const { return ChildrenSize() == 0; }
        bool& NeedGradient() { return m_needsGradient; }
        const bool& NeedGradient() const { return m_needsGradient; }

        void SetParameterUpdateRequired(bool f) { m_parameterUpdateRequired = f; }
        bool IsParameterUpdateRequired() const { return m_parameterUpdateRequired; }

        virtual void /*IComputationNode::*/InferImageDimsFromInputs()
        {
            if (!IsLeaf())
                InferImageDimsFromInput(0); //copy from child 0 by default.
        }

        virtual void ValidateInferChildDims(size_t i, size_t rows, size_t cols) = 0;

        bool IsChildAnImage(const size_t index) const
        {
            return m_children[index]->m_outputImageLayout.width != 1 || m_children[index]->m_outputImageLayout.channels != 1;
        }

        pair<ImageLayout, ImageLayout> GetImageLayouts() const { return make_pair(m_inputImageLayout, m_outputImageLayout); }

        const size_t ChildrenSize() const { return m_children.size(); }     // TODO: rename to NumChildren() or NumInputs(); and inside here where we use m_children, use m_children.size() as well

        virtual void SetInput(const size_t childIndex, const ComputationNodeBasePtr& node) = 0;

        virtual void /*IComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange &) = 0;    // (redeclaring, as compiler gets confused otherwise--will go away with ComputeInputPartial(t))
        //void ComputeInputPartialMap(const size_t inputIndex)   // TODO: this will be replaced by FrameRange version
        //{
        //    ComputeInputPartial(inputIndex, FrameRange(/*whole batch*/));      // nodes that do not implement this will know to understand SIZE_MAX as full batch
        //}
        //virtual void ComputeGradientForChildren(const FrameRange & frameRange) = 0;
        virtual void ClearGradientForChildren() = 0;

        // masking
        // overridden by <ElemType> variant only
        virtual void MaskMissingValuesColumnsToZero(const FrameRange &) = 0;
        virtual void MaskMissingGradientColumnsToZero(const FrameRange &) = 0;
        virtual void InvalidateMissingValuesColumns(const FrameRange &) = 0;
        virtual void InvalidateMissingGradientColumns(const FrameRange &) = 0;

        // indicates whether special handling is needed.The standard handleing will be just mask the function values after the evalaution and mask the gradient before gradiant computation for the children. this is not valid for all criterion nodes whose result is a scalar.
        // overridden to return true by training/eval criteria (and the soon-to-be-deprecated PairNetworkNode, LSTMNode)
        // The need for this seems an artifact of the old inconsistent layout architecture. In the future, this can probably just go away.
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return false; }

        virtual void /*IComputationNode::*/OnEvaluateBeginIteration() override             // called before first iteration step of EvaluateThisNode()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "OnEvaluateBeginIteration: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }
        virtual void /*IComputationNode::*/OnEvaluateEndIteration() override               // called after last iteration step of EvaluateThisNode()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "OnEvaluateEndIteration: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }
        // TODO: the following two are not really utilized yet other than printing trace information
        virtual void /*IComputationNode::*/OnComputeGradientBeginIteration() override             // called before first iteration step of ComputeGradient()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "OnComputeGradientBeginIteration: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }
        virtual void /*IComputationNode::*/OnComputeGradientEndIteration() override               // called after last iteration step of ComputeGradient()
        {
#ifdef TRACK_GAP_NANS
            fprintf(stderr, "OnComputeGradientEndIteration: %ls %ls operation\n", NodeName().c_str(), OperationName().c_str());
#endif
        }

    protected:

        void InferImageDimsFromInput(const size_t index, const bool outputSameAsInput = true)
        {
            if (index >= ChildrenSize())
                InvalidArgument("InferImageDimsFromInput: output index");

            const auto & child = m_children[index];
            if (child != nullptr)
                m_inputImageLayout = child->m_outputImageLayout;
            if (outputSameAsInput)
                m_outputImageLayout = m_inputImageLayout;
        }

        void InferMBLayoutFromInputsForStandardCase();

    public:

        static bool ByVisitedOrder(const ComputationNodeBasePtr& lhs, const ComputationNodeBasePtr& rhs)    // sorting predicate
        {
            return ComputationNetworkOwnedNodeState::ByVisitedOrder(lhs.get(), rhs.get());
        }

        bool IsEqualTo(const ComputationNodeBasePtr& other) const //this will be used to determine whehter two nodes are the same
        {
            if (OperationName() != other->OperationName() || m_children.size() != other->m_children.size())
                return false;

            if (NodeName() == other->NodeName())  //assume names are unique in the system
                return true;

            if (IsLeaf() && other->IsLeaf())  //since names are not equal otherwise will return above
                return false;

            for (size_t i=0; i<m_children.size(); i++)
                if (!(m_children[i] == other->m_children[i]))
                    return false;

            return true;
        }

        // determine enumeration order for everything needed to evaluate this node (and its children)
        // This creates a list such that children are evaluated before their parents.
        // If !forForwardProp then the order will be reversed, suitable for backprop.
        // The 'recurrent' version is only called from FormRecurrentLoops().
        // TODO: This should be a method of ComputationNetwork, not ComputationNode.
        std::list<ComputationNodeBasePtr> EnumerateNodes(bool forForwardProp/*else get order for backprop*/, bool skipPairNetwork)
        {
            std::list<ComputationNodeBasePtr> nodes;
            std::unordered_set<ComputationNodeBasePtr> visited;

            // get forward computation order
            EnumerateNodesR(visited, nodes, skipPairNetwork);  // call into the recursive portion of this function below

            // if caller wants order for backprop then reverse it
            if (!forForwardProp)
                nodes.reverse();            // and go backwards

            return nodes;
        }
    private:
        // Recursive part of EnumerateNodes().
        void EnumerateNodesR(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result, bool skipPairNetwork)
        {
            if (visited.find(shared_from_this()) == visited.end())      // do not include a node twice
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                // children first for function evaluation
                if (OperationName() != L"PairNetwork" || !skipPairNetwork)    // (don't step through network-pair boundary if called from FormRecurrentLoops())
                {
                    for (int i = 0; i < m_children.size(); i++)
                    {
                        if (m_children[i])
                            m_children[i]->EnumerateNodesR(visited, result, skipPairNetwork);
                    }
                }

#if 0
                // propagate m_needsGradient flags upwards from leaves
                // TODO: This belongs into Validate().
                if (!IsLeaf())
                    m_needsGradient = ChildrenNeedGradient();
#endif

                // now that all children are in list before us, put ourselves
                result.push_back(shared_from_this());

#if 0           // this does not work, since m_visitedOrder gets cleared out, while the list survives in a cache
                if (setVisitedOrder)    // FormRecurrentNodes() would like this variable to be set as well
                    m_visitedOrder = result.size();
#endif
            }
        }
    public:

#if 0
        bool ChildrenNeedGradient()  const //this is only valid when called in the forward computation order.
        {
            for (int i = 0; i<m_children.size(); i++)
            {
                if (!m_children[i])
                    continue;
                if (m_children[i]->m_needsGradient)
                    return true;
            }
            return false;
        }
#endif

        // check whether a node is up-to-date w.r.t. its children, for lazy evaluation
        // If this returns false, node must be evaluated to update m_functionValues.
        // BUGBUG: The function name is incorrect. It also returns 'true' if a child has the same time stamp (not older).
        // This is virtual because it is overridden by traversal nodes.
        virtual bool IsFuncValueOlderThanInputs() const
        {
            for (size_t i = 0; i<ChildrenSize(); i++)
            {
#if 1
                if (IsOlderThan(*m_children[i]))
                    return true;
#else
                //the second condition is used when the time stamp change from positive to negative
                if (m_children[i]->GetEvalTimeStamp() >= GetEvalTimeStamp() || m_children[i]->GetEvalTimeStamp() + 1e10 < GetEvalTimeStamp())
                    return true;
#endif
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
                        for (size_t i = 0; i < curNode->m_children.size(); i++)
                        {
                            arcs.push_back(ComputationArc(curNode, curNode->m_children[i]));

                            if (visited.find(curNode->m_children[i]) == visited.end()) // this children has not been visited before 
                                tovisit.push_front(curNode->m_children[i]);		// going to visit each of the children
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
        // data members
        //std::vector<ComputationNodeBasePtr> m_parents; //m_parents are dynamically determined based on the root node you want to compute
        DEVICEID_TYPE m_deviceId;   // CPU=-1, >=0 GPU
        std::wstring m_nodeName;

        // inputs
        std::vector<ComputationNodeBasePtr> m_children;

        // dimensions and layout
        // Data is stored as a matrix, but often it is interpreted as a more complex structure.
        // If the matrix is minibatch data (inputs, activations, labels), then matrix columns are samples.
        // Note that the actual matrix storage does not always exist.
        size_t m_numRows, m_numCols;        // matrix dimension of function values and gradients
        ImageLayout m_inputImageLayout;     // how to interpret each column in the input as an image
        ImageLayout m_outputImageLayout;    // and the output
        MBLayoutPtr m_pMBLayout;

        // flags related to gradient propagation
        bool m_parameterUpdateRequired;     // update parameters? Only used for LearnableParameters.    --TODO: Should we make this a member of LearnableParameters actually? And require a type cast? Currently it is read out for all leaves.
        bool m_gradientInitialized;         // indicates whether the gradient matrix has been resized and initialized to 0
    };
    typedef ComputationNodeBase::ComputationNodeBasePtr ComputationNodeBasePtr;

    // =======================================================================
    // ComputationNode -- abstract base class for computation nodes, deriving from CompuationNodeBase, parameterized by float vs. double
    // =======================================================================

    // little helper class to allow derived Node classes to specify how many inputs they expect
    struct INumInputs { virtual size_t GetExpectedNumInputs() const = 0; };
    template<size_t m_numInputs> struct NumInputs : public INumInputs { size_t GetExpectedNumInputs() const override { return m_numInputs; } };  // e.g. derive from NumInputs<2>

    template<class ElemType>
    class ComputationNode : public ComputationNodeBase // abstract class that cannot be instantiated
    {
        typedef ComputationNodeBase Base;
    protected:
        //std containers such as list and map does not support class reference so we need to use pointer
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
        ComputationNode() { }
    public:
        using ComputationNodeBase::AttachInputs;    // import the convenience functions that take 1..6 parameters
        using ComputationNodeBase::SetDims;
        typedef ElemType OurElemType;
    protected:
        // TODO: this should be protected and only accessible to the New method; maybe just move it in here?
        //        TODO: after memshare, this requirement has gone away. We can construct directly again.
        // TODO: Once we switch to VS 2015, we shall use inheriting constructors, i.e. we can delete all those redundant constructor forwards in each ComputationNode derivate
        // TODO: verify that we initialize all members (e.g. m_parameterUpdateRequired was missing before)
        ComputationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNodeBase(deviceId, name)
        {
        }
    public:
        // public constructor
        // You must construct ComputationNode derivates with this function. The real C++ constructor itself is hidden,
        // as we need to call a virtual function after construction. This function does that.
        // TODO: Actually, that is no longer necessary. We can get rid of this again. Or keep it just as a convenience function that returns a shared_ptr.
        template<class C, class... _Types> static inline shared_ptr<C> New(DEVICEID_TYPE deviceId, const wstring & name, _Types&&... _Args)
        {
            return make_shared<C>(deviceId, name, forward<_Types>(_Args)...);     // creates objects, esp. assigns deviceId to matrices, but otherwise does nothing
        }

        virtual ~ComputationNode()
        {
#ifdef DISPLAY_DEBUG
            fprintf (stderr, "Called Destructor NodeName: %s\n", (msra::strfun::utf8 (NodeName())).c_str()), fflush(stderr);
#endif
        }

        // our own output dimensions
        virtual void SetDims(size_t rows, size_t cols) override final
        {
            Base::SetDims(rows, cols);
            // TODO: in the future we will NOT resize here, but in a different function
            //       Then this function will no longer be virtual.
            UpdateSize();
            // This ^^ recovers the previous behavior of this function, keeping it compatible at this time, for testing.
        }
        // update m_functionValues to match the dimensions given in m_numRows, m_numCols
        void UpdateSize()
        {
            FunctionValues().Resize(m_numRows, m_numCols);
        }
        // someone outside changed our m_functionValues--update our internal state, e.g. m_numRows, m_numCols
        // Yes, it is bad design that this is possible.
        virtual void NotifyFunctionValuesModified() override final
        {
            m_numRows = FunctionValues().GetNumRows();
            m_numCols = FunctionValues().GetNumCols();
        }
        virtual double Get00Element() const override final { return FunctionValues().Get00Element(); }

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
            wstring name = NodeName(); name;
            const auto * pNumInputs = dynamic_cast<INumInputs*>(this);    // if this class also derives from NumInputs<N> then N is the expected number of inputs
            if (pNumInputs && pNumInputs->GetExpectedNumInputs() != inputs.size())
                RuntimeError("%ls operation '%ls' expects %d inputs (given: %d)", OperationName().c_str(), NodeName().c_str(), (int)pNumInputs->GetExpectedNumInputs(), (int)inputs.size());
            m_children.resize(inputs.size());
            for (size_t i = 0; i < m_children.size(); i++)
                if (inputs[i])
                    m_children[i] = UpCast(inputs[i]);          // (UpCast() checks the type; the assignment then downcasts it again)
                else
                    m_children[i] = nullptr;                    // during network creation, nullpts are possible
        }

        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            RequestMatrixFromPool(m_functionValues, matrixPool);
        }

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterEval(MatrixPool& /*matrixPool*/)
        {
        }

        virtual void AllocateGradientMatricesForChildren(MatrixPool& matrixPool) override
        {
            for (int i = 0; i < m_children.size(); i++)
            {
                if (m_children[i]->NeedGradient())
                    m_children[i]->RequestMatricesBeforeGradientComp(matrixPool);
            }
        }

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            RequestMatrixFromPool(m_gradientValues, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            if (!IsLeaf() && !RequiresPreCompute())
            {
                if (m_gradientValues != nullptr && m_gradientValues->GetMatrixType() != SPARSE)  //since we don't have a sparse pool yet
                    ReleaseMatrixToPool(m_gradientValues, matrixPool);

                ReleaseMatrixToPool(m_functionValues, matrixPool);
            }
        }
        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const;

        // TODO: similar to DumpInfo; used by ExperimentalNetworkBuilder test implementation
        /*HasToString::*/ wstring ToString() const
        {
            // we format it like "name : type rows x cols ( args )"
            wstring result = /*TidyName*/(NodeName()) + L" : " + OperationName();
            result.append(msra::strfun::wstrprintf(L" %d x %d", (int)m_functionValues->GetNumRows(), (int)m_functionValues->GetNumCols()));
            if (m_children.empty()) result.append(L" ()");
            else
            {
                wstring args;
                bool first = true;
                for (auto & child : m_children)
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

        // update size (#columns) of m_{function,gradient}Values to match MBLayout
        // This must be called right before EvaluateThisNode() the first time for a given minibatch.
        // Currently overridden by
        //  - InputValue, which verifies instead of resizing (since Resize() is specified to be destructive, it should not call it).
        //  - LearnableParameters
        //  - GMMLogLikelihoodNode (which allocates some internal temp memory).
        // Important: Unless overridden, this function is destructive. Nodes cannot carry over minibatch-size dependent state across minibatches through m_functionValues because of this.
        // TODO: How is this function different from OnEvaluateBeginIteration()?
        // TODO: This should be part of the split of Resize() and UpdateSize().
        virtual void UpdateFunctionMBSize() override
        {
            if (m_pMBLayout)               // if no layout, this node contains parameters independent of MB size, don't resize
                m_functionValues->ResizeColumns(m_pMBLayout->GetNumCols());
        }

        void ValidateInferChildDims(size_t i, size_t rows, size_t cols) override final;

#if 0   // (this function cannot be used currently since sentenceBegin is not a Matrix<ElemType> anymore; only affects LSTMNode which is no longer used)
        static void WINAPI SetToInitStateValueForResetSeg(const Matrix<ElemType>& sentenceBegin,
                                                          size_t nStream, ElemType initStateValue, Matrix<ElemType>& newprevstate)
        {
            Matrix<ElemType> colSeg(sentenceBegin.GetDeviceId());
            colSeg.Resize(nStream, nStream);
            size_t nStateRow = newprevstate.GetNumRows();

            assert(nStream == sentenceBegin.GetNumRows());

            /// only set state to init state value for segmentation = 0, and -1
            /// e.g., -1 0 1 -> 0 0 1 -> 0 0 -1 -> 1 1 0 

            Matrix<ElemType> colPos(sentenceBegin.GetDeviceId());
            colPos.SetValue(sentenceBegin); /// -1 0 1
            colPos.InplaceTruncateBottom(((int) MinibatchPackingFlags::SequenceStart));
            Matrix<ElemType>::Scale((ElemType)-1.0, colPos);
            colPos += ((int) MinibatchPackingFlags::None);
            // BUGBUG: ^^ What is this? colPos is a matrix, None is a flag; and it is 0
            colSeg.SetDiagonalValue(colPos);
            Matrix<ElemType> ones(sentenceBegin.GetDeviceId());
            ones.Resize(nStateRow, nStream);
            ones.SetValue((ElemType)1);
            /// add default state value if it is for reset
            Matrix<ElemType>::MultiplyAndWeightedAdd(initStateValue, ones, false, colSeg, false, 1.0, newprevstate);  /// += [0 initStateValue 0 ]
        }
#endif

    public:
        static bool MaskMissingColumnsToZero(Matrix<ElemType>& matrixToBeMasked, const MBLayoutPtr & pMBLayout, const FrameRange & frameRange)
        {
            //fprintf(stderr, "masking column range %d\n", (int)frameRange.timeIdxInSeq);
            return MaskMissingColumnsTo(matrixToBeMasked, pMBLayout, frameRange, (ElemType)0);
        }

        void /*ComputationNodeBase::*/MaskMissingValuesColumnsToZero(const FrameRange & frameRange) override final
        {
            //fprintf(stderr, "%ls %ls m_functionValues ", NodeName().c_str(), OperationName().c_str());
            MaskMissingColumnsToZero(*m_functionValues, m_pMBLayout, frameRange);
        }
        void /*ComputationNodeBase::*/MaskMissingGradientColumnsToZero(const FrameRange & frameRange) override final
        {
            //fprintf(stderr, "%ls %ls m_gradientValues ", NodeName().c_str(), OperationName().c_str());
            MaskMissingColumnsToZero(*m_gradientValues, m_pMBLayout, frameRange);
        }

        // for debugging, set the gaps to NaN instead (to track whether it bubbles up somewhere)
        void InvalidateMissingValuesColumns(const FrameRange & frameRange) override final
        {
            //fprintf(stderr, "invalidating %ls %ls m_functionValues column range %d\n", NodeName().c_str(), OperationName().c_str(), (int)frameRange.timeIdxInSeq);
            MaskMissingColumnsTo(*m_functionValues, m_pMBLayout, frameRange, Matrix<ElemType>::MakeNan(__LINE__));
        }
        void InvalidateMissingGradientColumns(const FrameRange & frameRange) override final
        {
            //fprintf(stderr, "invalidating %ls %ls m_gradientValues column range %d\n", NodeName().c_str(), OperationName().c_str(), (int)frameRange.timeIdxInSeq);
            MaskMissingColumnsTo(*m_gradientValues, m_pMBLayout, frameRange, Matrix<ElemType>::MakeNan(__LINE__));
        }

        // for debugging purposes
        void /*ComputationNodeBase::*/PrintSelf(bool printMatrices = false) const
        {
            fprintf(stderr, "\n%ls[%lu, %lu] = %ls", NodeName().c_str(), GetNumRows(), GetNumCols(), OperationName().c_str());           

            if (!IsLeaf())
            {
                fprintf(stderr, "(");           
                for (size_t i=0; i<ChildrenSize(); i++)
                {
                    if (i > 0)
                        fprintf(stderr, ", ");           
                    fprintf(stderr, "%ls[%lu, %lu]", m_children[i] ? m_children[i]->NodeName().c_str():L"NULL", m_children[i]->GetNumRows(), m_children[i]->GetNumCols());
                }
                fprintf(stderr, ")");           
            }

            if (printMatrices)
            {
                fprintf (stderr, "\n    $$$$ Function Values\n");
                FunctionValues().Print("FunctionValue");

                fprintf (stderr, "\n    $$$$ Gradient Values\n");
                GradientValues().Print("GradientValue");
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

        inline ComputationNodePtr Inputs(const size_t childIndex) const       // TODO: rename to Input
        {
#ifdef _DEBUG // profile shows this is range check very expensive in release mode, skip it  
            if (childIndex >= m_children.size())
                LogicError("Inputs: childIndex is out of range.");
#endif
            return UpCast(m_children[childIndex]);
        }

        void /*ComputationNodeBase::*/SetInput(const size_t childIndex, const ComputationNodeBasePtr& inode) override
        {
            const ComputationNodePtr node = UpCast(inode);

            //require first nodes specified before the second to avoid null nodes condition.
            if (childIndex > m_children.size())
                InvalidArgument("SetInput: You must specify the input for children with index less than this one first.");

            // expand the inputs to exist up to the desired index
            while (childIndex >= m_children.size())
                m_children.push_back(nullptr);

            // set the input value
            m_children[childIndex] = node;
        }

        const Matrix<ElemType>& FunctionValues() const { return *m_functionValues; }
        Matrix<ElemType>& FunctionValues() { return *m_functionValues; }

        const Matrix<ElemType>& GradientValues() const { return *m_gradientValues; }
        Matrix<ElemType>& GradientValues() { return *m_gradientValues; }

        // function to access any input and output, value and gradient, whole batch or single frame
        // Note: This returns a reference into 'data' in the form of a column slice, i.e. a small matrix object that just points into 'data'.
        Matrix<ElemType> DataSlice(Matrix<ElemType> & data, const FrameRange & frameRange/*select frame or entire batch*/)
        {
            try
            {
                return DataSliceWithMBLayout(data, frameRange, m_pMBLayout);
            }
            catch (const logic_error & e)   // catch the error and rethrow it with the node name attached
            {
                LogicError("%s, for %ls %ls operation.", e.what(), NodeName().c_str(), OperationName().c_str());
            }
        }
        Matrix<ElemType> ValueSliceToDense(const FrameRange & frameRange/*select frame or entire batch*/, bool keepValuesOnSwitch)
        {
            FunctionValues().SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, keepValuesOnSwitch);
            return ValueSlice(frameRange);
        }
        Matrix<ElemType> ValueSlice(const FrameRange & frameRange/*select frame or entire batch*/)
        {
            return DataSlice(FunctionValues(), frameRange);
        }
        Matrix<ElemType> GradientSlice(const FrameRange & frameRange/*select frame or entire batch*/)
        {
            return DataSlice(GradientValues(), frameRange);
        }
        // use the following two versions if you assume the inputs may contain gaps that must be set to zero because you want to reduce over frames with a BLAS operation
        Matrix<ElemType> MaskedValueSlice(const FrameRange & frameRange/*select frame or entire batch*/)
        {
            MaskMissingValuesColumnsToZero(frameRange);
            return ValueSlice(frameRange);
        }
        Matrix<ElemType> MaskedGradientSlice(const FrameRange & frameRange/*select frame or entire batch*/)
        {
            MaskMissingGradientColumnsToZero(frameRange);
            return GradientSlice(frameRange);
        }

        //virtual void /*IComputationNode::*/OnEvaluateBeginIteration() override             // called before first iteration step of EvaluateThisNode()
        //{
        //    if (!IsLeaf())
        //        UpdateSize();
        //}

#ifdef _DEBUG
        // NaN checks
        virtual void /*IComputationNode::*/OnEvaluateEndIteration() override
        {
            Base::OnEvaluateEndIteration();
#ifdef TRACK_GAP_NANS
            MaskMissingValuesColumnsToZero(FrameRange(m_pMBLayout));       // HasNaN() operates on a whole matrix, so first flatten all gaps to 0
            if (FunctionValues().HasNan("OnEvaluateEndIteration"))
                LogicError("%ls %ls operation unexpectedly produced NaN values.", NodeName().c_str(), OperationName().c_str());
#endif
            InvalidateMissingValuesColumns(FrameRange(m_pMBLayout));        // blast NaNs into columns that are gaps in a packed layout
        }
#endif

        virtual void /*IComputationNode::*/OnComputeGradientBeginIteration() override
        {
            Base::OnComputeGradientBeginIteration();

#if 0       // BUGBUG: This does not work, for unknown reasons. For now, we must keep the LazyZeroGradient() call inside ComputeGradientForChildren
            // allocate gradients for ourselves and also our children that we propagate into
            if (m_needsGradient)
            {
                LazyZeroGradient();          // set gradient to 0 if this is the first time
                for (size_t i = 0; i < m_children.size(); i++)
                {
                    ComputationNodePtr child = Inputs(i);
                    if (child->m_needsGradient)
                        child->LazyZeroGradient();          // set gradient to 0 if this is the first time
                }
            }
#endif

#if 0       // TODO: If you get a NaN failure, feel free to put this back in
            // many gradients are reduction operations
            // They touch both in-flowing gradients and function values, so we must set both to 0.
            // BUGBUG: This masks a bug: Nodes should do that by themselves, like in EvaluateThisNode(), but they currently don't.
            if (m_needsGradient)
            {
                MaskMissingValuesColumnsToZero(FrameRange(m_pMBLayout));
                if (m_gradientInitialized)
                    MaskMissingGradientColumnsToZero(FrameRange(m_pMBLayout));
            }
            bool anyChildNeedsGradient = false;
            for (size_t i = 0; i < m_children.size(); i++)
                anyChildNeedsGradient |= Inputs(i)->m_needsGradient;
            if (anyChildNeedsGradient)
                for (size_t i = 0; i < m_children.size(); i++)
                    Inputs(i)->MaskMissingValuesColumnsToZero(FrameRange(Inputs(i)->GetMBLayout()));
#endif
        }

#ifdef _DEBUG
        virtual void /*IComputationNode::*/OnComputeGradientEndIteration() override
        {
            Base::OnComputeGradientEndIteration();
#ifdef TRACK_GAP_NANS
            for (size_t i = 0; i < m_children.size(); i++)
            {
                ComputationNodePtr child = Inputs(i);
                if (child->m_needsGradient)
                {
                    child->MaskMissingGradientColumnsToZero(FrameRange(child->GetMBLayout()));       // HasNaN() operates on a whole matrix, so first flatten all gaps to 0
                    if (child->GradientValues().HasNan("OnComputeGradientEndIteration"))
                        LogicError("%ls %ls operation unexpectedly produced NaN gradients.", child->NodeName().c_str(), child->OperationName().c_str());
                }
            }
#endif
        }
#endif

        // this is the entry point from Network; while it will call virtual ComputeInputPartial() into the actual node implementation
        // TODO: move to -Base (or -Network?)
        void ComputeGradientForChildren(const FrameRange & frameRange, bool childrenInThisLoop, bool childrenInOuterLoop) override
        {
            if (frameRange.IsAllFrames() && IsPartOfLoop() && childrenInThisLoop)
                LogicError("%ls %ls operation: ComputeGradientForChildren called with whole-batch FrameRange on node that participates in a loop", NodeName().c_str(), OperationName().c_str());

            for (size_t i = 0; i < m_children.size(); i++)
            {
                ComputationNodePtr child = Inputs(i);
                if (child->m_needsGradient &&
                    (childrenInThisLoop  && child->IsPartOfLoop() == IsPartOfLoop() ||
                     childrenInOuterLoop && child->IsPartOfLoop() != IsPartOfLoop()
                    ))
                {
                    //fprintf(stderr, "ComputeGradientForChildren: %ls %ls operation -> child %d %ls %ls\n", NodeName().c_str(), OperationName().c_str(), (int)i, child->NodeName().c_str(), child->OperationName().c_str());
                    if (!m_needsGradient)
                        LogicError("%ls %ls operation has m_needsGradient set to false but children require it.", NodeName().c_str(), OperationName().c_str());
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %ls(%ls)\n", i, child->OperationName().c_str(), child->NodeName().c_str());
#endif
#if DUMPOUTPUT
                    fprintf(stderr, "Backprop%d_%ls\n", i, NodeName().c_str());
#endif
                    child->LazyZeroGradient();              // set gradient to 0 if this is the first time

#if 1
                    // TODO: There is an inefficiency here which we should fix.
                    if (IsPartOfLoop() && !child->IsPartOfLoop() && !frameRange.IsAllFrames())
                    {
                        static int warnings = 0;
                        if (warnings++ < 20)
                            fprintf (stderr, "ComputeGradientForChildren: Inefficiency: %ls %ls operation in loop propagates gradient to non-loop %ls %ls\n",
                            NodeName().c_str(), OperationName().c_str(), child->NodeName().c_str(), child->OperationName().c_str());
                    }
#endif

                    //fprintf(stderr, "ComputeInputPartial %d %d %ls %ls\n", (int)frameRange.timeIdxInSeq, (int)i, NodeName().c_str(), OperationName().c_str());
                    ComputeInputPartial(i, frameRange);     // this computes partial wrt to the child and sums the gradient value in the child
                }
#ifdef DISPLAY_DEBUG
                else fprintf (stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, child->OperationName().c_str(), child->NodeName().c_str());
#endif
            }
        }

        void /*ComputationNodeBase::*/ClearGradientForChildren() override   // TODO: bad naming--this just clears the lazy flags, whereas LazyZeroGradient() actually clears the values
        {
            for (size_t i = 0; i < m_children.size(); i++)
                Inputs(i)->m_gradientInitialized = false;
        }

        // lazy resetting of gradient
        // TODO: We can inline this once the Resize etc. below has been reduced to a single Matrix call
        void LazyZeroGradient()
        {
            if (!m_needsGradient)
                LogicError("%ls %ls operation: LazyZeroGradient() called although this node needs no gradient.", NodeName().c_str(), OperationName().c_str());

            if (m_gradientInitialized)
                return;

            // TODO: we should move this pattern to class Matrix. We should not be concerned here with the storage format of the gradient.
            GradientValues().Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
            if (GradientValues().GetMatrixType() == DENSE)
                GradientValues().SetValue(0);
            else
                GradientValues().Reset();

            m_gradientInitialized = true;
        }

        // NOTE: we should reimplement this to be thread-safe and use a larger than requested initialized memory block
        // we can then just wrap that memory block in a matrix of the correct dimensions since it will be const no one can change it
        // should only need one memory block per device
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

    protected:

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

        //this function is used to create matrices for those needed before matrix pool is available
        //e.g., for model parameters and input nodes you will need to resize the functions based on NDL
        //and before matrix pool is available
        void CreateMatrixIfNull(shared_ptr<Matrix<ElemType>>& matrixPtr)
        {
            if (matrixPtr == nullptr)
            {
                matrixPtr = make_shared<Matrix<ElemType>>(m_deviceId);
            }
        }
        //to be called by derived classed if that class needs to print node values
        void PrintNodeValuesToFile(const bool printValues, File& fstream) const
        {
            if (printValues)
            {
                fstream << wstring(L"\n");
                const Matrix<ElemType>&  m = FunctionValues();
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
                *node->m_functionValues = *m_functionValues;
                if (m_gradientValues)
                    *node->m_gradientValues = *m_gradientValues;
                else
                    node->m_gradientValues = nullptr;
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

        shared_ptr<Matrix<ElemType>> m_functionValues, m_gradientValues;

        static std::map<size_t, std::map<size_t, Matrix<ElemType>*>> s_constOnes;
    };

    // convenience wrapper for ComputationNode::New()
    template<class C, class... _Types> inline shared_ptr<C> New(DEVICEID_TYPE deviceId, const wstring & name, _Types&&... _Args)
    {
        return ComputationNode<typename C::OurElemType>::template New<C>(deviceId, name, forward<_Types>(_Args)...);
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
        //virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        ComputationNodeNonLooping(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        // these two implement the ComputationNode<> interface
        void EvaluateThisNode(const FrameRange & frameRange) override final
        {
            if (frameRange.IsAllFrames())
                EvaluateThisNodeNonLooping();
            else
                LogicError("%s node should never be in a loop.", typeid(*this).name());
        }
        void ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override final
        {
            if (frameRange.IsAllFrames())
                ComputeInputPartialNonLooping(inputIndex);
            else
                LogicError("%s node should never be in a loop.", typeid(*this).name());
        }

        // non-looping node types instead implement these functions
        virtual void EvaluateThisNodeNonLooping() = 0;
        virtual void ComputeInputPartialNonLooping(size_t inputIndex) = 0;
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
        virtual void SaveToFile(File& fstream) const override { NOT_IMPLEMENTED; }
        virtual void LoadFromFile(File& /*fstream*/, size_t /*modelVersion*/) override { NOT_IMPLEMENTED; }
        virtual void CopyTo(ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const override { NOT_IMPLEMENTED; }
        virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) override { NOT_IMPLEMENTED; }
        //virtual void SetDims(size_t rows, size_t cols) override { NOT_IMPLEMENTED; }
        virtual double Get00Element() const override { NOT_IMPLEMENTED; }
        virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs) override { NOT_IMPLEMENTED; }
        virtual void PrintSelf(bool) const override { NOT_IMPLEMENTED; }
        virtual void ValidateInferChildDims(size_t,size_t,size_t) override { NOT_IMPLEMENTED; }
        virtual void SetInput(const size_t,const Microsoft::MSR::CNTK::ComputationNodeBase::ComputationNodeBasePtr &) override { NOT_IMPLEMENTED; }
        virtual void ClearGradientForChildren(void) override { NOT_IMPLEMENTED; }
        virtual void MaskMissingValuesColumnsToZero(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual void MaskMissingGradientColumnsToZero(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual void InvalidateMissingValuesColumns(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual void InvalidateMissingGradientColumns(const Microsoft::MSR::CNTK::FrameRange &) override { NOT_IMPLEMENTED; }
        virtual std::wstring ToString(void) const override { NOT_IMPLEMENTED; }
        // these are meant to be called during computation, so provide dummy implementations
        virtual bool RequiresPreCompute() const override { return false; }                    // return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() override { return true; }
        virtual void PrintSelfBeforeValidation() const override { }
        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const override { }
    protected:
    public: // needed in ComputationNetwork::FindInRecurrentLoops(), which really should be part of RecurrentFlowControlNode
        std::vector<ComputationNodeBasePtr> m_nestedNodes;                  // nodes tucked away in this node, in evaluation order
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
    // BUGBUG: some should be protected, not public
    // Note: Whoever invented that insanity called two-phase name lookup shall rot in hell, for the crime of causing infinite pain. [fseide]
#define UsingComputationNodeMembers /*without OperationName; needed to support inconsistent pattern of InputValue */    \
protected: \
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;  /*TODO: can we just use 'using?' */ \
    using Base::SetDims; using Base::GetNumRows; using Base::GetNumCols; \
    using Base::m_pMBLayout; using Base::GetNumTimeSteps; using Base::GetNumParallelSequences; \
    using Base::MaskMissingColumnsToZero; using Base::MaskMissingValuesColumnsToZero; using Base::MaskMissingGradientColumnsToZero; using Base::InvalidateMissingValuesColumns; using Base::InvalidateMissingGradientColumns; \
    using Base::DataSlice; using Base::ValueSlice; using Base::GradientSlice; using Base::MaskedValueSlice; using Base::MaskedGradientSlice; \
    using Base::EvaluateThisNode; using Base::ComputeInputPartial; \
    using Base::m_children; using Base::m_deviceId; using Base::m_functionValues; using Base::m_gradientValues; \
    using Base::m_inputImageLayout; using Base::m_outputImageLayout; \
    using Base::m_parameterUpdateRequired; using Base::m_nodeName; \
    using Base::CreateMatrixIfNull; using Base::RequestMatrixFromPool; using Base::ReleaseMatrixToPool; \
public: \
    using Base::CreateUniqId; \
    using Base::AttachInputs; using Base::ChildrenSize; using Base::ClearGradientForChildren; using Base::VerifyDims; \
    using Base::ConstOnes; \
    using Base::InferImageDimsFromInput; using Base::InferImageDimsFromInputs; using Base::InferMBLayoutFromInputsForStandardCase; \
    using Base::CopyTo; using Base::CreateUniqNodeName; using Base::DetachInputs; \
    using Base::DumpNodeInfo; using Base::EnumerateNodes; \
    using Base::HasMBLayout; using Base::GetMBLayout; using Base::LinkToMBLayout; \
    using Base::FunctionValues; using Base::GradientValues; \
    using Base::Inputs; using Base::SetInput; \
    using Base::IsChildAnImage; using Base::IsEqualTo; using Base::IsFuncValueOlderThanInputs; using Base::IsLeaf; \
    using Base::LoadFromFile; using Base::NodeName; \
    using Base::PrintNodeValuesToFile; using Base::PrintSelfBeforeValidation; \
    using Base::RequiresPreCompute; \
    using Base::SaveToFile; using Base::UpdateFunctionMBSize; \
    using Base::RequestMatricesBeforeEval; using Base::ReleaseMatricesAfterEval; \
    using Base::RequestMatricesBeforeGradientComp; using Base::ReleaseMatricesAfterGradientComp; \
    using Base::Validate; using Base::ValidateUnaryMap; using Base::ValidateBinaryZip; using Base::ValidateUnaryReduce; using Base::ValidateBinaryReduce; using Base::ValidateInferBinaryChildrenDims; using Base::ValidateInferChildDims

#define ComputationNodeBoilerplate \
protected:    /* some boilerplate goes here */ \
    virtual const std::wstring OperationName() const override { return TypeName(); } \
    virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) override { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }

#define UsingComputationNodeMembersBoilerplate \
    ComputationNodeBoilerplate; UsingComputationNodeMembers

#pragma endregion base computation class

}}}
