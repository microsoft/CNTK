//
// <copyright file="ComputationNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ScriptableObjects.h"

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

namespace Microsoft { namespace MSR { namespace CNTK {

    enum CopyNodeFlags
    {
        copyNodeNull = 0, // invalid value
        copyNodeValue=1, // copy everything but the children links
        copyNodeChildren=2, // only copy over children links
        copyNodeAll=3, // copy everything
        copyNodeChildrenCrossNetwork=4, // allow a cross network child copy
    };

#pragma region base computation class

    // =======================================================================
    // ComputationNodeBase -- abstract base class for all computation nodes
    // TODO: decide the name. This does contain actual members such as the node name, so it's not really a pure interface.
    // =======================================================================

    class ComputationNodeBase :
        public ScriptableObjects::ComputationNodeObject,
        public ScriptableObjects::WithTag, public ScriptableObjects::HasName, public ScriptableObjects::HasToString,
        public std::enable_shared_from_this<ComputationNodeBase>
    {
    public:
        typedef shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

        ComputationNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            m_deviceId(deviceId),
            m_needGradient(false),
            m_loopId(-1),
            m_samplesInRecurrentStep(1),
            m_visitedOrder(-1),
            m_index(-1),
            m_lowLink(-1),
            m_indexInLoop(0),
            m_visited(false),
            m_inStack(false),
            m_maskMissingColumnsToZero(false),
            m_nodeName(name == L"" ? CreateUniqNodeName() : name)
        {
        }
        virtual ~ComputationNodeBase(){}
        virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;

        virtual void CopyTo(const ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const = 0;
        virtual ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) = 0;

        // TODO: OperationName calls static TypeName which does not match the actual type names in that the 'Node' is missing.
        virtual const std::wstring OperationName() const = 0;
#define OperationNameOf(T) (T<float>::TypeName())    // we are templated, but for this the type param matters not. So we just pick one, and hide that fact.

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
        virtual size_t GetNumRows() const = 0;
        virtual size_t GetNumCols() const = 0;
        virtual void Resize(size_t rows, size_t cols) = 0;
        virtual double Get00Element() const = 0;

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            ComputeInputPartial(inputIndex, FrameRange(/*whole batch*/));      // nodes that do not implement this will know to understand SIZE_MAX as full batch
        }
        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &) = 0;

        virtual void EvaluateThisNode()
        {
            EvaluateThisNode(FrameRange(/*whole batch*/));      // nodes that do not implement this will know to understand SIZE_MAX as full batch
        }
        // evaluate only N frames at time index timeIdxInSeq
        // Normally, N is 1 or it spans the entire minibatch.
        virtual void EvaluateThisNode(const FrameRange &) = 0;
        // evaluate a node--this calls EvaluateThisNode() and MaskMissingColumnsToZero() if needed
        // this is the main entry point for Network; while EvaluateThisNode() is the virtual call into specific node implementation
        virtual void EvaluateThisNodeGivenInputs() = 0;
        virtual void EvaluateThisNodeGivenInputs(const size_t timeIdxInSeq) = 0; // TODO: change to FrameRange as well

        virtual void /*ComputationNodeBase::*/Validate() { }
        virtual bool UnitTest() { return true; }

        virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs, size_t numExpected = SIZE_MAX) = 0;
        virtual void AttachInputs(const ComputationNodeBasePtr /*singleInput*/) = 0;
        virtual void AttachInputs(const ComputationNodeBasePtr /*leftInput*/, const ComputationNodeBasePtr /*rightInput*/) = 0;
        virtual void AttachInputs(const ComputationNodeBasePtr /*leftInput*/, const ComputationNodeBasePtr /*middleInput*/, const ComputationNodeBasePtr /*rightInput*/) = 0;
        virtual void AttachInputs(const ComputationNodeBasePtr /*firstInput*/, const ComputationNodeBasePtr /*secondInput*/, const ComputationNodeBasePtr /*thirdInput*/, const ComputationNodeBasePtr /*fourthInput*/) = 0;
        virtual void AttachInputs(const ComputationNodeBasePtr /*firstInput*/, const ComputationNodeBasePtr /*secondInput*/, const ComputationNodeBasePtr /*thirdInput*/, const ComputationNodeBasePtr /*fourthInput*/, const ComputationNodeBasePtr /*fifthInput*/) = 0;
        virtual void AttachInputs(const ComputationNodeBasePtr /*firstInput*/, const ComputationNodeBasePtr /*secondInput*/, const ComputationNodeBasePtr /*thirdInput*/, const ComputationNodeBasePtr /*fourthInput*/, const ComputationNodeBasePtr /*fifthInput*/, const ComputationNodeBasePtr /* sixthInput */) = 0;

        virtual void DetachInputs() { m_children.clear(); }

        const std::vector<ComputationNodeBasePtr> & GetChildren() const { return m_children; }

        // TODO: is this always just called with deviceId == m_deviceId?
        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId) = 0;

        //return true if the node's value should be computed before the normal training. e.g., mean and invStd of input features.
        virtual bool RequiresPreCompute() const { return false; }

        // return true if the node's value should be computed in batch mode only, e.g., time-reverse node
        virtual bool RequiresBatchMode() const { return false; }

        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const = 0;

        /*HasName::*/void SetName(const std::wstring & newName) // also for use by ExperimentalNetworkBuilder
        {
            m_nodeName = newName;
            fprintf(stderr, "Node --> %ls = %ls\n", NodeName().c_str(), OperationName().c_str()), fflush(stderr);
        }

        virtual void SetFunctionAndGradientSize(const int numSamples) = 0;

        virtual void SetMBLayout(MBLayoutPtr pMBLayout)
        {
            assert(pMBLayout->GetNumTimeSteps() == pMBLayout->GetSize());  // TODO: move this check into MBLayout
            m_pMBLayout = pMBLayout;
        }

        void ClearCache()
        {
            m_loopId = -1;
            m_visitedOrder = -1;
            m_index = -1;
            m_lowLink = -1;
            m_indexInLoop = 0;
            m_visited = false;
            m_inStack = false;
        }

        void SetLoopId(const int id) { m_loopId = id; }
        int GetLoopId() const { return m_loopId; }

        void SetVisitedOrder(const int id) { m_visitedOrder = id; }
        size_t GetVisitedOrder() const { return m_visitedOrder; }

        void SetIndex(const size_t ind) { m_index = ind; }
        size_t GetIndex() const { return m_index; }

        void SetLowLink(const size_t lowlink) { m_lowLink = lowlink; }
        size_t GetLowLink() const { return m_lowLink; }

        void SetVisited(const bool visited) { m_visited = visited; }
        bool IsVisisted() const { return m_visited; }

        void SetInStack(const bool instack) { m_inStack = instack; }
        bool IsInStack() const { return m_inStack; }

        void SetIndexInLoop(const size_t index) { m_indexInLoop = index; }
        size_t GetIndexInLoop() const { return m_indexInLoop; }

        std::wstring GetName() const { return m_nodeName; }

        // temporary function that is called to verify stuff is called as I think it is. Delete if this does not fire for a while.
        void VerifyNumParallelSequences(size_t bsz)
        {
            //m_samplesInRecurrentStep = bsz;
            if (bsz != m_pMBLayout->GetNumParallelSequences())
                LogicError("VerifyNumParallelSequences: value inconsistent with MB layout");
        }

        // This is used at 284 places inside nodes, most of the time as
        // FrameSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t() * GetNumParallelSequences(), GetNumParallelSequences()), m_pMBLayout)
        size_t GetNumParallelSequences() const
        {
            //return m_samplesInRecurrentStep;
            return m_pMBLayout->GetNumParallelSequences();
        }

        // indicates whether special handling is needed.The standard handleing will be just mask the function values after the evalaution and mask the gradient before gradiant computation for the children. this is not valid for all criterion nodes whose result is a scalar.
        // overridden to return true by training/eval criteria (and the soon-to-be-deprecated PairNode, LSTMNode)
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return false; }

        int64_t UpdateEvalTimeStamp()
        {
            m_evalTimeStamp = atomic_fetch_add(&s_timeStampCounter, (unsigned long long int) 1);    // TODO: does this really need to be atomic? We are not multi-threaded
            return m_evalTimeStamp;
        }

        void ResetEvalTimeStamp()
        {
            m_evalTimeStamp = s_timeStampCounter;
        }

        // implemented by ComputationNode<ElemType>
        // for debugging purpose
        virtual void PrintSelf(bool printMatrices = false) const = 0;

        // called in validation loop right before Validate()
        virtual void PrintSelfBeforeValidation(bool allowNulls = false) const
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
                        if (allowNulls)
                        {
                            fprintf(stderr, "NULL");
                            continue;
                        }
                        throw runtime_error("One of the children is missing.");
                    }


                    if (IsChildAnImage(i))  //image
                        fprintf(stderr, "%ls[%lu {W=%lu, H=%lu, C=%lu}, %lu]", child->NodeName().c_str(), child->GetNumRows(),
                        child->m_outputWidth, child->m_outputHeight, child->m_outputChannels, child->GetNumCols());
                    else
                        fprintf(stderr, "%ls[%lu, %lu]", child->NodeName().c_str(), child->GetNumRows(), child->GetNumCols());
                }
                fprintf(stderr, ")");
            }
        }

        const std::wstring& NodeName() const { return m_nodeName; }
        std::wstring& NodeName() { return m_nodeName; }

        bool IsLeaf() const { return ChildrenSize() == 0; }
        bool& NeedGradient() { return m_needGradient; }
        const bool& NeedGradient() const { return m_needGradient; }

        void SetMaskMissingColumnsToZero() { m_maskMissingColumnsToZero = true; }
        bool NeedToMaskMissingColumnsToZero() const { return m_maskMissingColumnsToZero; }

        void InitRecurrentNode()    // this initialization says that this node is not inside a loop
        {
            SetLoop(false);
        }

        bool HasLoop() const { return m_hasloop; }
        void SetLoop(bool hasLoop) { m_hasloop = hasLoop; }

        virtual ComputationNodeBasePtr FindChildInASet(const std::list<ComputationNodeBasePtr>& loop) const
        {
            for (int i = 0; i < this->m_children.size(); i++)
            if (std::find(loop.begin(), loop.end(), this->m_children[i]) != loop.end())
                return this->m_children[i];
            return nullptr;
        }

        virtual void InferImageDimsFromInputs()
        {
            if (!IsLeaf())
                InferImageDimsFromInput(0); //copy from child 0 by default.
        }

        bool IsChildAnImage(const size_t index) const
        {
            if (index > ChildrenSize())
                throw invalid_argument("IsChildAnImage: out of index.");

            return (m_children[index]->m_outputWidth != 1 || m_children[index]->m_outputChannels != 1);
        }

        const size_t ChildrenSize() const { return m_children.size(); }

        virtual void SetInput(const size_t childIndex, const ComputationNodeBasePtr node) = 0;

        virtual void ComputeGradientForChildren() = 0;

        virtual void ComputeGradientForChildren(const size_t timeIdxInSeq) = 0; // TODO: don't we need a FrameRange here, too?

        // TODO: some evaluation method to be abstracted, but types don't match

    protected:

        void InferImageDimsFromInput(const size_t index, const bool outputSameAsInput = true)
        {
            if (index >= ChildrenSize())
                throw invalid_argument("InferImageDimsFromInput: output index");

            const auto & child = m_children[index];
            if (child != nullptr)
            {
                m_inputWidth = child->m_outputWidth;
                m_inputHeight = child->m_outputHeight;
                m_inputChannels = child->m_outputChannels;
            }

            if (outputSameAsInput)
            {
                m_outputWidth = m_inputWidth;
                m_outputHeight = m_inputHeight;
                m_outputChannels = m_inputChannels;
            }
        }

    public:

        static bool IsSmaller(const ComputationNodeBasePtr lhs, const ComputationNodeBasePtr rhs)
        {
            return lhs->m_visitedOrder < rhs->m_visitedOrder;
        }

        bool IsEqualTo(const ComputationNodeBasePtr other) const //this will be used to determine whehter two nodes are the same
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
        // Side-effects (unbeknownst to the name of the function):
        //  - m_needGradient flags, are propagated up from children
        //  - m_visitedOrder (only if 'recurrent' flag is set; otherwise leave untouched)
        std::list<ComputationNodeBasePtr> EnumerateNodes(bool forForwardProp/*else get order for backprop*/, bool recurrent)
        {
            std::list<ComputationNodeBasePtr> nodes;
            std::unordered_set<ComputationNodeBasePtr> visited;

            // get forward computation order
            EnumerateNodesR(visited, nodes, recurrent);  // call into the recursive portion of this function below

            // if caller wants order for backprop then reverse it
            if (!forForwardProp)
            {
                assert(!recurrent);     // TODO: not sure if required, but currently only called this way

                // TODO: comment why can't directly reverse(); what's wrong with EnumerateNodes()' result?
                nodes.sort(IsSmaller);  // sort nodes by m_visitedOrder   --TODO: why? What about nodes with visitedOrder -1? Will they stay the same? Comment please!!!
                nodes.reverse();        // and go backwards
            }

            return nodes;
        }
    private:
        // Recursive part of EnumerateNodes().
        void EnumerateNodesR(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result, bool recurrent)
        {
            if (visited.find(shared_from_this()) == visited.end())      // do not include a node twice
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                // children first for function evaluation
                if (OperationName() != L"PairNetwork" || !recurrent)    // (don't step through network-pair boundary if recurrent)
                {
                    for (int i = 0; i < m_children.size(); i++)
                    {
                        if (m_children[i])
                            m_children[i]->EnumerateNodesR(visited, result, recurrent);
                    }
                }

                // propagate needGradient flags upwards from leaves
                if (!IsLeaf())
                    m_needGradient = ChildrenNeedGradient();  //only nodes that require gradient calculation is included in gradient calculation

                // now that all children are in list before us, put ourselves
                result.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 

                if (recurrent)
                    m_visitedOrder = result.size();
            }
        }
    public:

        std::list<ComputationNodeBasePtr> ReshuffleNodes(std::map<int, std::list<ComputationNodeBasePtr>> recurrentResult)
        {
            std::list<ComputationNodeBasePtr> noRecurrentResult;
            std::unordered_set<ComputationNodeBasePtr> visited;

            ReshuffleNodesForEvalWithRecurrentLoops(visited, recurrentResult, noRecurrentResult);

            return noRecurrentResult;
        }

#if 0
        std::list<ComputationNodeBasePtr> EnumerateNodes(const bool forwardComputation, bool recurrent)
        {
            if (forwardComputation)
            {
                std::list<ComputationNodeBasePtr> result;
                std::unordered_set<ComputationNodeBasePtr> visited;
                EnumerateNodesForEval(visited, result, recurrent);
                return result;
            }
            else
                return EnumerateNodesForGradient();
        }
#endif

    protected:

        bool ChildrenNeedGradient()  const //this is only valid when called in the forward computation order.
        {
            for (int i = 0; i<m_children.size(); i++)
            {
                if (m_children[i] == nullptr)
                    continue;
                if (m_children[i]->NeedGradient())
                    return true;
            }
            return false;
        }

        // TODO: what does this do?
        // As a side effect, it also propagates m_needGradient to intermediate nodes
        void ReshuffleNodesForEvalWithRecurrentLoops(std::unordered_set<ComputationNodeBasePtr>& visited, std::map<int, std::list<ComputationNodeBasePtr>>& recurrentResult,
                                                     std::list<ComputationNodeBasePtr>& noRecurrentResult)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i = 0; i<m_children.size(); i++)
                    m_children[i]->ReshuffleNodesForEvalWithRecurrentLoops(visited, recurrentResult, noRecurrentResult);

                //children first for function evaluation
                if (!IsLeaf())
                    m_needGradient = ChildrenNeedGradient();  //only nodes that require gradient calculation is included in gradient calculation

                if (GetLoopId() >= 0)
                    recurrentResult[GetLoopId()].push_back(shared_from_this());
                else
                    noRecurrentResult.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
            }
        }

#if 0
        // create list such that children are evaluated before their parents
        // Unbeknownst to the name of the function, it also updates the m_needGradient flags (set if children are set).
        // TODO: when is this called vs. the other?
        virtual void EnumerateNodesForEval(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result, bool recurrent)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                // first put children into list, before putting ourselves
                for (int i = 0; i<m_children.size(); i++)
                    m_children[i]->EnumerateNodesForEval(visited, result, recurrent);

                // propagate needGradient flags upwards from leaves
                if (!IsLeaf())
                    m_needGradient = ChildrenNeedGradient();  //only nodes that require gradient calculation is included in gradient calculation

                // now that all children are in list before us, put ourselves
                result.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
            }
        }
#endif

    public:

        // check whether a node is up-to-date w.r.t. its children, for lazy evaluation
        // If this returns false, node must be evaluated to update m_functionValues.
        bool IsFuncValueOlderThanInputs() const
        {
            for (size_t i = 0; i<ChildrenSize(); i++)
            {
                //the second condition is used when the time stamp change from positive to negative
                if (m_children[i]->m_evalTimeStamp >= m_evalTimeStamp || m_children[i]->m_evalTimeStamp + 1e10 < m_evalTimeStamp)
                    return true;
            }

            return false;
        }

        virtual void ClearGradientForChildren(const int /*iActMiniBatchSize*/) = 0;

        typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;
        // [1/13/2015 erw] add to enumerate all the edges 
        // enumerate arcs that can be reached starting from the current node's children
        // [in/out] visited record already visited nodes 
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
            int64_t id = atomic_fetch_add(&s_timeStampCounter, (unsigned long long int) 1);
            std::wstring base = L"AutoName";
            std::wstringstream sstm;
            sstm << base.c_str() << id;
            std::wstring name = sstm.str();
            //msra::strfun::wstrprintf name(L"%s%d", L"AutoName", id);
#endif

            return name;
        }

        // TODO: These 4 functions will be completed after refactoring.
        //request matrices needed to do node function value evaluation
        virtual void RequestEvalMatrices(MatrixPool& matrixPool)
        {
            matrixPool;
        }

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterEval(MatrixPool& matrixPool)
        {
            matrixPool;
        }

        //request matrices that are needed for gradient computation
        virtual void RequestGradientMatrices(MatrixPool& matrixPool, const int numParents)
        {
            matrixPool; numParents;
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseGradientMatrices(MatrixPool& matrixPool)
        {
            matrixPool;
        }

    protected:
        // data members
        std::vector<ComputationNodeBasePtr> m_children;

        DEVICEID_TYPE m_deviceId; //CPU=-1, >=0 GPU
        bool m_needGradient;  //only used for leaf, i.e., learnable parameters, etc.
        bool m_maskMissingColumnsToZero;  // indicates whether the results of operation should be masked to handle the cases that the utterances have different lengths when grouped together as a minibatch.
        // ^^ This decides whether the node gets passed the full layout with flags or only the one without flags
        //    and this is only ever tested in MaskMissingColumnsToZero(), of which two versions exist, one in ComputationNode and one in ClassBasedCrossEntropyWithSoftmaxNode
        // Pertinent reduction operations (criterion nodes and gradient computation) always perform masking.
        // Hence, this flag is only needed for special use cases where regular matrix ops are used for a 'reduce' operation.
        size_t m_inputWidth, m_inputHeight, m_inputChannels;  //how to interpret each column in the input as an image
        size_t m_outputWidth, m_outputHeight, m_outputChannels;  //how to interpret each column in the output as an image

        std::wstring m_nodeName;

        static atomic_ullong s_timeStampCounter;
        int64_t m_evalTimeStamp; //this is used to reduce unnecessary recomputation when a different node in the model is reevaluated

        int     m_loopId;
        size_t  m_samplesInRecurrentStep;

        /// the order in reverse graph. 
        int m_visitedOrder;
        int m_index;
        int m_lowLink;          // TODO: comment this, as it is not obvious
        bool m_visited;
        bool m_inStack;
        int m_indexInLoop;
        MBLayoutPtr m_pMBLayout;

    private:
        // for loop nodes
        bool m_hasloop;
    };
    typedef ComputationNodeBase::ComputationNodeBasePtr ComputationNodeBasePtr;

    // =======================================================================
    // ComputationNode -- abstract base class for computation nodes parameterized by float vs. double
    // =======================================================================

    template<class ElemType>
    class ComputationNode : public ComputationNodeBase //Abstract Class that cannot be instantiated
    {
        // note: enable_shared_from_this<> allows to create a shared_ptr from a raw pointer to this that is correctly aware of all other shared_ptrs (same ref count)
    protected:
        //std containers such as list and map does not support class reference so we need to use pointer
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
        ComputationNode() { }
    public:
        typedef ElemType OurElemType;
    protected:
        // TODO: this should be protected and only accessible to the New method; maybe just move it in here?
        // TODO: Once we switch to VS 2015, we shall use inheriting constructors, i.e. we can delete all those redundant constructor forwards in each ComputationNode derivate
        // TODO: verify that we initialize all members (e.g. m_needGradient was missing before)
        ComputationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNodeBase(deviceId, name),
            m_functionValues(deviceId),
            m_gradientValues(deviceId)
        {
            InitRecurrentNode();
            ResetEvalTimeStamp();   // bring it into defined state
            // This constructor does not call MoveMatricesToDevice(), but that is needed for full initialization.
            // Only call this constructor through the New() factory below, which will ensure this.
        }
    public:
        // public constructor
        // You must construct ComputationNode derivates with this function. The real C++ constructor itself is hidden,
        // as we need to call a virtual function after construction. This function does that.
        template<class C, class... _Types> static inline shared_ptr<C> New(DEVICEID_TYPE deviceId, const wstring & name, _Types&&... _Args)
        {
            auto p = make_shared<C>(deviceId, name, forward<_Types>(_Args)...);     // creates objects, esp. assigns deviceId to matrices, but otherwise does nothing
            p->MoveMatricesToDevice(deviceId);                                      // this is a virtual call, i.e. it will handle extra matrices an object might own
            return p;
        }

        virtual ~ComputationNode()
        {
#ifdef DISPLAY_DEBUG
            fprintf (stderr, "Called Destructor NodeName: %s\n", (msra::strfun::utf8 (NodeName())).c_str()), fflush(stderr);
#endif
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId);

        // our own output dimensions
        /*implement*/size_t GetNumRows() const { return FunctionValues().GetNumRows(); }
        /*implement*/size_t GetNumCols() const { return FunctionValues().GetNumCols(); }
        /*implement*/void Resize(size_t rows, size_t cols) { FunctionValues().Resize(rows, cols); }
        /*implement*/double Get00Element() const { return FunctionValues().Get00Element(); }

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

        // these take ComputationNodePtr, not ComputationNodeBasePtr, as these are being overloaded by nodes
        virtual void AttachInputs(const ComputationNodePtr /*singleInput*/) 
        {
            LogicError("This operation does not support single input.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*leftInput*/, const ComputationNodePtr /*rightInput*/) 
        {
            LogicError("This operation does not support two inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*leftInput*/, const ComputationNodePtr /*middleInput*/, const ComputationNodePtr /*rightInput*/) 
        {
            LogicError("This operation does not support three inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*firstInput*/, const ComputationNodePtr /*secondInput*/, const ComputationNodePtr /*thirdInput*/, const ComputationNodePtr /*fourthInput*/)
        {
            LogicError("This operation does not support four inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*firstInput*/, const ComputationNodePtr /*secondInput*/, const ComputationNodePtr /*thirdInput*/, 
                                  const ComputationNodePtr /*fourthInput*/, const ComputationNodePtr /*fifthInput*/)
        {
            LogicError("This operation does not support five inputs.");
        }

        virtual void AttachInputs(const ComputationNodePtr /*firstInput*/, const ComputationNodePtr /*secondInput*/, const ComputationNodePtr /*thirdInput*/,
                                  const ComputationNodePtr /*fourthInput*/, const ComputationNodePtr /*fifthInput*/, const ComputationNodePtr /* sixthInput */)
        {
            LogicError("This operation does not support six inputs.");
        }

        virtual void AttachInputs(const ComputationNodeBasePtr singleInput) { AttachInputs(UpCast(singleInput)); }
        virtual void AttachInputs(const ComputationNodeBasePtr leftInput, const ComputationNodeBasePtr rightInput) { AttachInputs(UpCast(leftInput), UpCast(rightInput)); }
        virtual void AttachInputs(const ComputationNodeBasePtr leftInput, const ComputationNodeBasePtr middleInput, const ComputationNodeBasePtr rightInput) { AttachInputs(UpCast(leftInput), UpCast(middleInput), UpCast(rightInput)); }
        virtual void AttachInputs(const ComputationNodeBasePtr firstInput, const ComputationNodeBasePtr secondInput, const ComputationNodeBasePtr thirdInput, const ComputationNodeBasePtr fourthInput) { AttachInputs(UpCast(firstInput), UpCast(secondInput), UpCast(thirdInput), UpCast(fourthInput)); }
        virtual void AttachInputs(const ComputationNodeBasePtr firstInput, const ComputationNodeBasePtr secondInput, const ComputationNodeBasePtr thirdInput, const ComputationNodeBasePtr fourthInput, const ComputationNodeBasePtr fifthInput) { AttachInputs(UpCast(firstInput), UpCast(secondInput), UpCast(thirdInput), UpCast(fourthInput), UpCast(fifthInput)); }
        virtual void AttachInputs(const ComputationNodeBasePtr firstInput, const ComputationNodeBasePtr secondInput, const ComputationNodeBasePtr thirdInput, const ComputationNodeBasePtr fourthInput, const ComputationNodeBasePtr fifthInput, const ComputationNodeBasePtr sixthInput) { AttachInputs(UpCast(firstInput), UpCast(secondInput), UpCast(thirdInput), UpCast(fourthInput), UpCast(fifthInput), UpCast(sixthInput)); }
        virtual void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs, size_t numExpected = SIZE_MAX)
        {
            if (numExpected != SIZE_MAX && numExpected != inputs.size())
                RuntimeError(msra::strfun::strprintf("AttachInputs: unexpected number of arguments: %d, expected: %d", (int)inputs.size(), (int)numExpected));
            m_children.resize(inputs.size());
            for (size_t i = 0; i < m_children.size(); i++)
                m_children[i] = UpCast(inputs[i]);      // (this checks the type)
        }

        virtual void DumpNodeInfo(const bool /*printValues*/, File& fstream) const;

        // TODO: similar to DumpInfo; used by ExperimentalNetworkBuilder test implementation
        /*HasToString::*/ wstring ToString() const
        {
            // we format it like "name : type rows x cols ( args )"
            wstring result = /*TidyName*/(NodeName()) + L" : " + OperationName();
            result.append(msra::strfun::wstrprintf(L" %d x %d", (int)m_functionValues.GetNumRows(), (int)m_functionValues.GetNumCols()));
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

        virtual void SetFunctionAndGradientSize(const int numSamples) 
        {
            size_t numRows = m_functionValues.GetNumRows();
            if (numRows > 0 && numSamples > 0)
            {
                m_functionValues.Resize(numRows, numSamples); 
                m_gradientValues.Resize(numRows, numSamples); 
            }
        }

        /*implement*/void EvaluateThisNodeGivenInputs()
        {
            EvaluateThisNode();     // this is a call to the virtual function that implements the actual operation

            if (NeedToMaskMissingColumnsToZero() && !NodeDoesItsOwnCustomizedMissingColumnsMasking())       // this means the node does it by itself; if not, we do it for the node
                MaskMissingColumnsToZero(m_functionValues);
        }

        // TODO: use a FrameRange arg, then unify with above
        // TODO: do we even need this extra function? Should Node know about this masking business, or is that the job of Network?
        // TODO: rename this to make it more clear what this function does
        /*implement*/void EvaluateThisNodeGivenInputs(const size_t timeIdxInSeq) // TODO: change to FrameRange as well
        {
            EvaluateThisNode(FrameRange(timeIdxInSeq, GetNumParallelSequences()));

            if (NeedToMaskMissingColumnsToZero() && !NodeDoesItsOwnCustomizedMissingColumnsMasking())
                MaskMissingColumnsToZero(m_functionValues, timeIdxInSeq);
        }

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

        /**
        reset to error signals to 0 for any elements without labels
        */
        // This sets MB columns to 0 that have the NoLabel or NoFeature flag set.
        // This happens as a result of packing multiple sequences for parallel processing--there will be some gaps, which are flagged by these flags.
        // Nodes that operate in 'map' style (input(j) -> output(j) independently) can ignore this; it will be garbage-in-garbage-out.
        // However, nodes that 'reduce' minibatches (e.g. computing the sum of all frames across all sequences) must deal with the garbage.
        // This function sets those to 0, assuming that now they can be reduced without affecting the result.
        // This function can operate on the whole range or on a selected single frame and/or a single sequence.
        // It is indirectly guarded by the m_maskMissingColumnsToZero flag, which, if false, will install a layout with IsAllNone() to be true. TODO: we better always install the same layout, and instead test m_maskMissingColumnsToZero here.
        // Note that existing 'reduce' style operations--the criterion nodes and gradient computation--already call this.
        bool MaskMissingColumnsToZero(Matrix<ElemType>& matrixToBeMasked, size_t timeIdxInSeq = SIZE_MAX, size_t seqIndex = SIZE_MAX) const
        {
            bool foundLabelOrFeatureMissing = false; /// set to true if either nolabel or feature missing is processed

            if (!m_pMBLayout->IsAllNone())
            {
                size_t nT = m_pMBLayout->GetNumTimeSteps();
                size_t nS = m_pMBLayout->GetNumParallelSequences();

                if (matrixToBeMasked.GetNumCols() != nT * nS)
                    LogicError("MaskMissingColumnsToZero: m_pMBLayout->m_minibatchPackingFlags should have one element for each timestep of all streams. Check feature reader. ");

                size_t startT = (timeIdxInSeq == SIZE_MAX) ?  0 : timeIdxInSeq;
                size_t endT   = (timeIdxInSeq == SIZE_MAX) ? nT : timeIdxInSeq + 1;

                size_t startS = (seqIndex == SIZE_MAX) ?  0 : seqIndex;
                size_t endS   = (seqIndex == SIZE_MAX) ? nS : seqIndex + 1;

                for (size_t t = startT; t < endT; t++)
                {
                    if (m_pMBLayout->Is(t, MinibatchPackingFlags::NoLabel | MinibatchPackingFlags::NoFeature))
                    {
                        for (size_t id = startS; id < endS; id++)
                            if (m_pMBLayout->Is(id, t, MinibatchPackingFlags::NoLabel | MinibatchPackingFlags::NoFeature))
                                matrixToBeMasked.ColumnSlice(t * nS  +  id, 1).SetValue(0);
                        foundLabelOrFeatureMissing = true;
                    }
                }
            }

            return foundLabelOrFeatureMissing;
        }

        /*
        virtual size_t GetNumSamplesWithLabel(const size_t numAllSamples)
        {
            if (m_mbLayout.m_sentenceBoundaryFlags != nullptr &&
                m_mbLayout.m_minibatchPackingFlags != nullptr &&
                !m_mbLayout.m_sentenceBoundaryFlags->IsEmpty() &&
                !m_mbLayout.m_minibatchPackingFlags->size() == 0)
            {
                size_t numTimeSteps = m_mbLayout.m_sentenceBoundaryFlags->GetNumCols();
                size_t numSequences = m_mbLayout.m_sentenceBoundaryFlags->GetNumRows();

                if (m_mbLayout.m_minibatchPackingFlags->size() != numTimeSteps)
                {
                    LogicError("GetNumSamplesWithLabel(): m_mbLayout.m_minibatchPackingFlags should have one element for each timestep of all streams.Check feature reader. ");
                }

                size_t numSamplesWithoutLabel = 0;

                for (size_t j = 0; j < numTimeSteps; j++)
                {
                    if (m_pMBLayout->m_minibatchPackingFlags[j] & MinibatchPackingFlags::NoLabel)
                    {
                        for (int i = 0; i < numSequences; i++)
                        {
                            if ((int)m_pMBLayout->m_sentenceBoundaryFlags(i, j) & ((int) MinibatchPackingFlags::NoLabel))
                            {
                                numSamplesWithoutLabel++;
                            }
                        }
                    }
                }

                return numTimeSteps*numSequences - numSamplesWithoutLabel;
            }
            else
            {
                return numAllSamples;
            }
        }
        */

        // for debugging purpose
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
#ifdef DEBUG // profile shows this is range check very expensive in release mode, skip it  
            if (childIndex >= m_children.size())
                InvalidArgument ("childIndex is out of range.");
#endif
            return UpCast(m_children[childIndex]);
        }

        /*implement*/void SetInput(const size_t childIndex, const ComputationNodeBasePtr inode)
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

        // these are overridden by DropoutNode, ReshapeNode, and RowRepeatNode to optimize for the trivial case that those don't do anything
        // TODO: lots of nodes read out m_functionValues directly--was that a bug or intentional? They have now been changed to ValueSlice(), i.e. would pick it up
        virtual const Matrix<ElemType>& FunctionValues() const { return m_functionValues; }
        virtual Matrix<ElemType>& FunctionValues() { return m_functionValues; }

        const Matrix<ElemType>& GradientValues() const { return m_gradientValues; }
        Matrix<ElemType>& GradientValues() { return m_gradientValues; }

        // function to access any input and output, value and gradient, whole batch or single frame
        // Note: This returns an object, not a reference. That object is a column slice, i.e. a small object that just points into another object.
        // TODO: remove FrameRange::samplesInRecurrentStep from FrameRange, as it belongs into pMBLayout. Hence this function that binds both together.
        // Note: This is not used anywhere yet, only a sketch how we may further abstract timing.
        Matrix<ElemType> DataSlice(Matrix<ElemType> & data,
                                   const FrameRange & frameRange/*select frame or entire batch*/)
        {
            auto sequence = SIZE_MAX;
            if (frameRange.IsAllFrames())
            {
                if (sequence == SIZE_MAX)
                    return data.ColumnSlice(0, data.GetNumCols());
                else
                    LogicError("DataSlice: sequence index only supported when accessing individual frame"); // (not needed; doable but more involved, requiring a reshape)
            }
            else
            {
                size_t numParallelSequences = m_pMBLayout->GetNumParallelSequences();
                if (numParallelSequences != frameRange.samplesInRecurrentStep)
                    LogicError("DataSlice: inconsistent samplesInRecurrentStep");   // TODO: this will go away when we remove this memebr from FrameRange
                size_t startColumn = frameRange.t() * numParallelSequences;
                if (sequence == SIZE_MAX)
                    return data.ColumnSlice(startColumn, numParallelSequences);
                else
                    return data.ColumnSlice(startColumn + sequence, 1);
            }
        }
        enum ValueOrGradient { VALUE, GRADIENT };
        Matrix<ElemType> DataSlice(ValueOrGradient valueOrGradient/*as it says*/,
            const FrameRange & frameRange/*select frame or entire batch*/)
        {
            Matrix<ElemType> & data = (valueOrGradient == VALUE) ? FunctionValues() : GradientValues();
            return DataSlice(data, frameRange);
        }
        Matrix<ElemType> ValueSlice(const FrameRange & frameRange/*select frame or entire batch*/)
        {
            return DataSlice(FunctionValues(), frameRange);
        }
        Matrix<ElemType> GradientSlice(const FrameRange & frameRange/*select frame or entire batch*/)
        {
            return DataSlice(GradientValues(), frameRange);
        }

        // this is the entry point from Network; while it will call virtual ComputeInputPartial() into the actual node implementation
        /*implement*/void ComputeGradientForChildren()
        {
            // batch is done only for feed-forward nodes
            if (HasLoop()) 
                return;

            for (size_t i=0; i<m_children.size(); i++)
            {
                if (NeedToMaskMissingColumnsToZero() && !NodeDoesItsOwnCustomizedMissingColumnsMasking())
                    MaskMissingColumnsToZero(m_gradientValues);

                ComputationNodePtr child = Inputs(i);
                if (child->NeedGradient())
                {
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %s(%s)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
#if DUMPOUTPUT
                    fprintf(stderr,"Backprop%d_%ls\n",i,NodeName().c_str());
#endif
                    ComputeInputPartial(i); //this computes partial wrt to the child and sums the gradient value in the child
                }
#ifdef DISPLAY_DEBUG
                else fprintf (stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
            }
        }
        
        // TODO: use a FrameRange here as well, then unify with above
        /*implement*/void ComputeGradientForChildren(const size_t timeIdxInSeq)
        {
            for (size_t i=0; i<m_children.size(); i++)
            {
                if (NeedToMaskMissingColumnsToZero() && !NodeDoesItsOwnCustomizedMissingColumnsMasking())
                    MaskMissingColumnsToZero(m_gradientValues, timeIdxInSeq);

                ComputationNodePtr child = Inputs(i);
                if (child->NeedGradient())
                {
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %s(%s)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
                    ComputeInputPartial(i, FrameRange(timeIdxInSeq, GetNumParallelSequences())); //this computes partial wrt to the child and sums the gradient value in the child
                }
#ifdef DISPLAY_DEBUG
                else fprintf (stderr, "    [%lu]: %s(%s) (no gradient needed so don't compute for)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
            }
        }

        /*implement*/void ClearGradientForChildren(const int /*iActMiniBatchSize*/)
        {
            for (size_t i=0; i<m_children.size(); i++)
            {
                ComputationNodePtr child = Inputs(i);
                if (child->NeedGradient())
                {
                    if(child->GradientValues().GetMatrixType() == DENSE) 
                    {
                        child->GradientValues().Resize(child->FunctionValues().GetNumRows(), child->FunctionValues().GetNumCols());
                        child->GradientValues().SetValue(0); 
                    }
                    else
                    {
                        child->GradientValues().Reset();
                    }
                }
            }
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
                matrix->SetValue(ElemType(1.000));
                s_constOnes[rows][cols] = matrix;
            }

            Matrix<ElemType>* m = s_constOnes[rows][cols];
            m->TransferFromDeviceToDevice(m->GetDeviceId(), deviceId);

            return *m;
        }

    protected:

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
        /*implement*/void CopyTo(const ComputationNodeBasePtr node, const std::wstring& newName, const CopyNodeFlags flags) const
        {
            CopyTo(UpCast(node), newName, flags);
        }
        virtual void CopyTo(const ComputationNodePtr node, const std::wstring& newName, const CopyNodeFlags flags) const
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
                node->m_needGradient = m_needGradient;
                node->m_nodeName = newName;
                node->m_evalTimeStamp = m_evalTimeStamp;

                //node->m_hasloop = m_hasloop;
                node->SetLoop(HasLoop());

                node->m_inputWidth = m_inputWidth;
                node->m_inputHeight = m_inputHeight;
                node->m_inputChannels = m_inputChannels;

                node->m_outputWidth = m_outputWidth;
                node->m_outputHeight = m_outputHeight;
                node->m_outputChannels = m_outputChannels;

                node->m_functionValues = m_functionValues; 
                node->m_gradientValues = m_gradientValues;

                node->m_maskMissingColumnsToZero = m_maskMissingColumnsToZero;
            }
        }

        // duplicate a node
        ComputationNodeBasePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags)
        {
            const std::wstring& name = (newName == L"") ? NodeName() : newName;
            ComputationNodeBasePtr node(NewThis(m_deviceId, name));    // NewThis() is a virtual function that creates a new node of the actual type of 'this'
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

        Matrix<ElemType> m_functionValues, m_gradientValues;

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
    template<class ElemType>
    class ComputationNodeNonLooping : public ComputationNode<ElemType>
    {
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        ComputationNodeNonLooping(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        virtual void ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange &)
        {
            LogicError("%s node should never be in a loop.", typeid(*this).name());
        }
        virtual void EvaluateThisNode(const FrameRange &)
        {
            LogicError("%s node should never be in a loop.", typeid(*this).name());
        }
        // classes that derive from this must implement the non-range version
        virtual void ComputeInputPartial(const size_t inputIndex) = 0;
        virtual void EvaluateThisNode() = 0;
    };

    // helper macro to ease access to base members in presence of C++ two-phase name lookup
    // Add 'typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;' at the start of each derived class
    // (some derived classes define a similar macro; there please modify the typedef for Base accordingly.)
    // This macro imports, one by one, every member of ComputationNode into the name space of the derived class.
    // Without this, one would have to use the name prefix, or alternatively this->, in front of all base member,
    // because the standard does not allow the compiler to do that for you (as MSVC still kindly does).
    // If you add new members to ComputationNode, please also add them here.
    // This macro expects 'Base' to be the name of the base class. Please also use 'Base' outside this macro to make it less likely to accidentally call the wrong base class members.
    // BUGBUG: some should be protected, not public
    // Note: Whoever invented that insanity called two-phase name lookup shall rot in hell, for the crime of causing infinite pain. [fseide]
#define UsingComputationNodeMembers    \
protected:  \
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;  \
        /* TODO: move NewThis() here  */ \
public: \
    using Base::AttachInputs; using Base::ChildrenNeedGradient; using Base::ChildrenSize; using Base::ClearGradientForChildren; \
    using Base::ComputeGradientForChildren; using Base::ComputeInputPartial; using Base::ConstOnes; using Base::InferImageDimsFromInput; \
    using Base::InferImageDimsFromInputs; using Base::CopyTo; using Base::CreateUniqNodeName; using Base::DetachInputs; \
    using Base::DumpNodeInfo; using Base::EnumerateNodes; \
    using Base::EvaluateThisNode; using Base::FindChildInASet; using Base::FunctionValues; \
    using Base::GradientValues; using Base::HasLoop; using Base::InitRecurrentNode; using Base::Inputs; \
    using Base::IsChildAnImage; using Base::IsEqualTo; using Base::IsFuncValueOlderThanInputs; using Base::IsLeaf; using Base::IsSmaller; \
    using Base::LoadFromFile; using Base::MoveMatricesToDevice; using Base::NeedGradient; using Base::NodeName; \
    using Base::OperationName; using Base::PrintNodeValuesToFile; using Base::PrintSelfBeforeValidation; \
    using Base::RequiresPreCompute; using Base::ReshuffleNodes; using Base::ReshuffleNodesForEvalWithRecurrentLoops; \
    using Base::SaveToFile; using Base::SetFunctionAndGradientSize; using Base::SetInput; using Base::Validate; \
protected:  \
    using Base::m_loopId; using Base::m_samplesInRecurrentStep; \
    using Base::m_visitedOrder; using Base::m_index; using Base::m_lowLink; using Base::m_visited; using Base::m_inStack; \
    using Base::m_indexInLoop; \
    using Base::m_pMBLayout; \
    using Base::m_maskMissingColumnsToZero; using Base::NodeDoesItsOwnCustomizedMissingColumnsMasking; using Base::GetNumParallelSequences; \
    using Base::DataSlice; using Base::ValueSlice; using Base::GradientSlice; using Base::SetMaskMissingColumnsToZero; \
    using Base::m_children; using Base::m_deviceId; using Base::m_evalTimeStamp; using Base::m_functionValues; using Base::m_gradientValues; \
    using Base::m_inputChannels; using Base::m_inputHeight; using Base::m_inputWidth; using Base::m_needGradient; using Base::m_nodeName; \
    using Base::m_outputChannels; using Base::m_outputHeight; using Base::m_outputWidth; using Base::s_constOnes; using Base::s_timeStampCounter; \
    using Base::shared_from_this; \
public:

#pragma endregion base computation class

}}}
