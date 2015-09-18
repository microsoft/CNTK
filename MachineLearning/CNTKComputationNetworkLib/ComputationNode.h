//
// <copyright file="ComputationNode.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "BrainScriptObjects.h"

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

    class ComputationNodeBase : public BS::ComputationNodeObject, public BS::WithTag, public BS::HasName, public BS::HasToString, public std::enable_shared_from_this<ComputationNodeBase>
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
            m_lowlink(-1),
            m_indexInLoop(0),
            m_visited(false),
            m_inStack(false),
            m_minibatchPackingFlag(nullptr),
            m_sentenceSeg(nullptr),
            m_reqMultiSeqHandling(false),
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

        const size_t ParentSize() const { return m_parents.size(); }
        void ClearParents() { m_parents.clear(); }
        void AddParent(ComputationNodeBasePtr pNode) { m_parents.push_back(pNode); }
        inline ComputationNodeBasePtr Parent(const size_t parentIndex) const
        {
#ifdef DEBUG // profile shows this is range check very expensive in release mode, skip it  
            if (parentIndex >= m_parents.size())
                InvalidArgument("parentIndex is out of range.");
#endif
            return m_parents[parentIndex];
        }

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

        virtual void ResetBound(Matrix<float> * seg, vector<MinibatchPackingFlag> *minibatchPackingFlag)
        {
            assert(seg->GetNumCols() == minibatchPackingFlag->size());
            m_sentenceSeg = seg;
            m_minibatchPackingFlag = minibatchPackingFlag;
        }

        void SetLoopId(const int id)
        {
            m_loopId = id;
        }
        void SetVisitedOrder(const int id)
        {
            m_visitedOrder = id;
        }
        void SetIndex(const size_t ind)
        {
            m_index = ind;
        }

        void Setlowlink(const size_t lowlink)
        {
            m_lowlink = lowlink;
        }

        void SetVisited(const bool visited)
        {
            m_visited = visited;
        }

        void SetInStack(const bool instack)
        {
            m_inStack = instack;
        }

        void SetIndexInLoop(const size_t index)
        {
            m_indexInLoop = index;
        }

        void clearCache()
        {
            m_loopId = -1;
            m_visitedOrder = -1;
            m_index = -1;
            m_lowlink = -1;
            m_indexInLoop = 0;
            m_visited = false;
            m_inStack = false;
        }

        size_t GetIndex() const
        {
            return m_index;
        }

        size_t GetVisitedOrder() const
        {
            return m_visitedOrder;
        }

        size_t Getlowlink() const
        {
            return m_lowlink;
        }

        size_t GetIndexInLoop() const
        {
            return m_indexInLoop;
        }

        std::wstring GetName() const
        {
            return m_nodeName;
        }

        bool isVisisted() const
        {
            return m_visited;
        }

        bool isInStack() const
        {
            return m_inStack;
        }
        int LoopId() const
        {
            return m_loopId;
        }

        // TODO: these two will disappear once the information is correctly held in a FrameRange record
        // This is called at 3 places; two are directly before ComputeGradientForChildren().
        void SetNbrSlicesInEachRecurrentIteration(size_t bsz)
        {
            m_samplesInRecurrentStep = bsz;
        }

        // Note: only used in one place, SimpleEvaluator.h PreComputeActivityAtTime().
        // The member is, however, read out at 284 places inside nodes,
        // most of the time as
        // FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep)
        // This expression will be turned into a function call to right here, so that we compute this only at one place
        // and can also handle the full-minibatch case.
        // Let us try to get this member out of this class altogether; it belongs elsewhere.
        size_t GetNbrSlicesInEachRecurrentIteration() const
        {
            return m_samplesInRecurrentStep;
        }

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

        void SetReqMultiSeqHandlingTo(const bool v) { m_reqMultiSeqHandling = v; }
        bool ReqMultiSeqHandling() const { return m_reqMultiSeqHandling; }

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

        virtual void ComputeGradientForChildren(const size_t timeIdxInSeq) = 0;

        virtual void ComputeGradient(const size_t timeIdxInSeq = -1) = 0;

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

        std::list<ComputationNodeBasePtr> EnumerateNodes(const bool forwardComputation, std::vector<ComputationNodeBasePtr>& rootOfLoop)
        {
            std::list<ComputationNodeBasePtr> result;

            if (forwardComputation)
            {
                std::unordered_set<ComputationNodeBasePtr> visited;
                EnumerateNodesForEval(visited, result, rootOfLoop, false);
            }
            else
            {
                result = EnumerateNodesForGradient();
            }

            return result;
        }

        std::list<ComputationNodeBasePtr> ReshuffleNodes(std::map<int, std::list<ComputationNodeBasePtr>> recurrentResult)
        {
            std::list<ComputationNodeBasePtr> noRecurrentResult;
            std::unordered_set<ComputationNodeBasePtr> visited;

            ReshuffleNodesForEvalWithRecurrentLoops(visited, recurrentResult, noRecurrentResult);

            return noRecurrentResult;
        }

        std::list<ComputationNodeBasePtr> EnumerateNodes(const bool forwardComputation)
        {
            std::list<ComputationNodeBasePtr> result;

            if (forwardComputation)
            {
                std::unordered_set<ComputationNodeBasePtr> visited;
                EnumerateNodesForEval(visited, result);
            }
            else
            {
                result = EnumerateNodesForGradient();
            }

            return result;
        }

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
        
        // TODO: why virtual?
        virtual void EnumerateNodesForEval(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result,
                                           std::vector<ComputationNodeBasePtr>& sourceRecurrentNodePtr, const bool isFromPastOrFutureValueNode)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i = 0; i<m_children.size(); i++)
                {
                    if (m_children[i] == nullptr)
                        continue;
                    m_children[i]->EnumerateNodesForEval(visited, result, sourceRecurrentNodePtr,
                                                         this->OperationName() == L"PastValue" || this->OperationName() == L"FutureValue");
                }

                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }

                result.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                this->m_visitedOrder = result.size();
            }
            else
            {
                if (!IsLeaf() && isFromPastOrFutureValueNode)
                    sourceRecurrentNodePtr.push_back(shared_from_this());
            }
        }

        void ReshuffleNodesForEvalWithRecurrentLoops(std::unordered_set<ComputationNodeBasePtr>& visited, std::map<int, std::list<ComputationNodeBasePtr>>& recurrentResult,
                                                     std::list<ComputationNodeBasePtr>& noRecurrentResult)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i = 0; i<m_children.size(); i++)
                {
                    m_children[i]->ReshuffleNodesForEvalWithRecurrentLoops(visited, recurrentResult, noRecurrentResult);
                }

                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }

                if (LoopId() >= 0)
                {
                    recurrentResult[LoopId()].push_back(shared_from_this());
                }
                else
                {
                    noRecurrentResult.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
                }
            }
        }

        virtual void EnumerateNodesForEval(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationNodeBasePtr>& result)
        {
            if (visited.find(shared_from_this()) == visited.end())  //not visited
            {
                visited.insert(shared_from_this());   // have visited tagged here to avoid infinite loop over children, children's children, etc

                for (int i = 0; i<m_children.size(); i++)
                {
                    m_children[i]->EnumerateNodesForEval(visited, result);
                }

                //children first for function evaluation
                if (!IsLeaf())
                {
                    if (ChildrenNeedGradient())  //only nodes that require gradient calculation is included in gradient calculation
                        m_needGradient = true;
                    else
                        m_needGradient = false;
                }

                result.push_back(shared_from_this());  //we put this in the list even if it's leaf since we need to use it to determine learnable params 
            }
        }

    public:

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
        virtual void ClearGradient(const bool clearExistingGradientValue) = 0;

        typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;
        //  [1/13/2015 erw] add to enumerate all the edges 
        void EnumerateArcs(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationArc>& arcs)
            //  enumerate arcs that can be reached starting from the current node's children
            //  [in/out] visited record already visited nodes 
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
                            {
                                tovisit.push_front(curNode->m_children[i]);		// going to visit each of the children
                            }
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

        std::list<ComputationNodeBasePtr> EnumerateNodesForGradient()
        {
            std::list<ComputationNodeBasePtr>  nodes = this->EnumerateNodes(true);  //get forward computation order first

            nodes.sort(IsSmaller);
            nodes.reverse();

            return nodes;
        }

        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool) = 0;

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterEval(MatrixPool& matrixPool) = 0;

        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool) = 0;

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool) = 0;

    protected:
        // data members
        std::vector<ComputationNodeBasePtr> m_children;
        std::vector<ComputationNodeBasePtr> m_parents; //m_parents are dynamically determined based on the root node you want to compute
        std::vector<bool> m_childrenGradientComputed; //used to indicate which child's gradient has been computed.

        DEVICEID_TYPE m_deviceId; //CPU=-1, >=0 GPU
        bool m_needGradient;  //only used for leaf, i.e., learnable parameters, etc.
        bool m_reqMultiSeqHandling;  // indicates whether the results of operation should be masked to handle the cases that the utterances have different lengths when grouped together as a minibatch.
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
        int m_lowlink;
        bool m_visited;
        bool m_inStack;
        int m_indexInLoop;
        Matrix<float> * m_sentenceSeg;  // TODO: this should be not a float but some integer type
        /// conditionally point to either a pointer to that provided by network, or point to 
        /// an indiviaul sentence boundary info, which happens if timeStep > 1 is required for PastValue node
        vector<MinibatchPackingFlag> * m_minibatchPackingFlag;

    private:
        // for loop nodes
        bool m_hasloop;
    };
    typedef ComputationNodeBase::ComputationNodeBasePtr ComputationNodeBasePtr;

    // =======================================================================
    // ComputationNode -- abstract base class for computation nodes parameterized by float vs. double
    // =======================================================================

    // TODO: number of inputs should be a template parameter! SIZE_MAX for those that take variable numvber

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
            ComputationNodeBase(deviceId, name), m_functionValues(nullptr), m_gradientValues(nullptr)
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

            //disable this line. Instead we should make sure matrices are allocated at the right device
            //p->MoveMatricesToDevice(deviceId);                                      // this is a virtual call, i.e. it will handle extra matrices an object might own
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
                ReleaseMatrixToPool(m_gradientValues, matrixPool);
                //ReleaseMatrixToPool(m_functionValues, matrixPool);
            }
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

        //making them virtual so that nodes that only copy values from it's children (e.g., dropout) can be efficient in evaluation
        virtual const Matrix<ElemType>& FunctionValues() const { return *m_functionValues; }
        virtual Matrix<ElemType>& FunctionValues() { return *m_functionValues; }

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

        virtual void SetFunctionAndGradientSize(const int numSamples)
        {
            size_t numRows = m_functionValues->GetNumRows();
            if (numRows > 0 && numSamples > 0)
            {
                m_functionValues->Resize(numRows, numSamples);
                m_gradientValues->Resize(numRows, numSamples);
            }
        }

        /*implement*/ void EvaluateThisNodeGivenInputs()
        {
            EvaluateThisNode();

            if (!UseCustomizedMultiSeqHandling())
                MaskToZeroWhenLabelAndFeatureMissing(FunctionValues());
        }

        /*implement*/void EvaluateThisNodeGivenInputs(const size_t timeIdxInSeq) // TODO: change to FrameRange as well
        {
            EvaluateThisNode(FrameRange(timeIdxInSeq, m_samplesInRecurrentStep));

            if (!UseCustomizedMultiSeqHandling())
                MaskToZeroWhenLabelAndFeatureMissing(FunctionValues(), timeIdxInSeq);
        }

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
            colPos.InplaceTruncateBottom(SEQUENCE_START);
            Matrix<ElemType>::Scale((ElemType)-1.0, colPos);
            colPos += SEQUENCE_MIDDLE;
            colSeg.SetDiagonalValue(colPos);
            Matrix<ElemType> ones(sentenceBegin.GetDeviceId());
            ones.Resize(nStateRow, nStream);
            ones.SetValue((ElemType)1);
            /// add default state value if it is for reset
            Matrix<ElemType>::MultiplyAndWeightedAdd(initStateValue, ones, false, colSeg, false, 1.0, newprevstate);  /// += [0 initStateValue 0 ]
        }

        /**
        reset to error signals to 0 for any elements without labele
        */
        bool MaskToZeroWhenLabelAndFeatureMissing(Matrix<ElemType>& matrixToBeMasked, const size_t timeIdxInSeq = (size_t)-1)
        {
            bool processedExistsNoLabelorFeatureMissing = false; /// set to true if either nolabel or feature missing is processed 

            if (m_sentenceSeg != nullptr &&
                m_minibatchPackingFlag != nullptr &&
                !m_sentenceSeg->IsEmpty() &&
                !m_minibatchPackingFlag->size() == 0)
            {
                size_t nT = matrixToBeMasked.GetNumCols();
                size_t nS = m_sentenceSeg->GetNumRows();

                if (m_minibatchPackingFlag->size() != nT / nS)
                    LogicError("MaskToZeroWhenLabelAndFeatureMissing: m_minibatchPackingFlag should have one element for each timestep of all streams. Check feature reader. ");

                //Matrix<ElemType> colSeg(m_sentenceSeg->GetDeviceId());

                size_t startT = (timeIdxInSeq == (size_t)-1) ? 0 : timeIdxInSeq * nS;
                size_t endT = (timeIdxInSeq == (size_t)-1) ? nT : timeIdxInSeq * nS + nS;
                for (size_t utt_t = startT; utt_t < endT; utt_t += nS)
                {
                    size_t j = utt_t / nS;

                    if ((*m_minibatchPackingFlag)[j] & MinibatchPackingFlag::NoLabel)
                    {
                        const auto & colSeg = m_sentenceSeg->ColumnSlice(j, 1);
                        for (int i = 0; i < nS; i++)
                        if ((int)colSeg(i, 0) & NO_LABEL)
                            matrixToBeMasked.ColumnSlice(utt_t + i, 1).SetValue(0);
                        processedExistsNoLabelorFeatureMissing = true;
                    }
                }
            }

            return processedExistsNoLabelorFeatureMissing;
        }

        /*
        virtual size_t GetNumSamplesWithLabel(const size_t numAllSamples)
        {
        if (m_sentenceSeg != nullptr &&
        m_minibatchPackingFlag != nullptr &&
        !m_sentenceSeg->IsEmpty() &&
        !m_minibatchPackingFlag->size() == 0)
        {
        size_t numTimeSteps = m_sentenceSeg->GetNumCols();
        size_t numSequences = m_sentenceSeg->GetNumRows();

        if (m_minibatchPackingFlag->size() != numTimeSteps)
        {
        LogicError("GetNumSamplesWithLabel(): m_minibatchPackingFlag should have one element for each timestep of all streams.Check feature reader. ");
        }

        size_t numSamplesWithoutLabel = 0;

        for (size_t j = 0; j < numTimeSteps; j++)
        {
        if ((*m_minibatchPackingFlag)[j] & MinibatchPackingFlag::NoLabel)
        {
        for (int i = 0; i < numSequences; i++)
        {
        if ((int)(*m_sentenceSeg)(i, j) & NO_LABEL)
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
                for (size_t i = 0; i<ChildrenSize(); i++)
                {
                    if (i > 0)
                        fprintf(stderr, ", ");
                    fprintf(stderr, "%ls[%lu, %lu]", m_children[i] ? m_children[i]->NodeName().c_str() : L"NULL", m_children[i]->GetNumRows(), m_children[i]->GetNumCols());
                }
                fprintf(stderr, ")");
            }

            if (printMatrices)
            {
                fprintf(stderr, "\n    $$$$ Function Values\n");
                FunctionValues().Print("FunctionValue");

                fprintf(stderr, "\n    $$$$ Gradient Values\n");
                GradientValues().Print("GradientValue");
            }
        }

        const Matrix<ElemType>& GradientValues() const { return *m_gradientValues; }
        Matrix<ElemType>& GradientValues() { return *m_gradientValues; }

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

        virtual void SetInput(const size_t childIndex, const ComputationNodeBasePtr inode)
        {
            const ComputationNodePtr node = UpCast(inode);

            //require first nodes specified before the second to avoid null nodes condition.
            if (childIndex > m_children.size())
                InvalidArgument("SetInput: You must specify the input for children with index less than this one first.");

            // expand the inputs to exist up to the desired index
            while (childIndex >= m_children.size())
            {
                m_children.push_back(NULL);
            }

            // set the input value
            m_children[childIndex] = node;
        }

        //if existing gradient values need to be cleared it needs to be done separately
        //this is to make it consistent with the per sample version to avoid setting samples to 0 one by one
        //if timeIdxInSeq = -1 we treat it as batch
        virtual void ComputeGradient(const size_t timeIdxInSeq = -1)
        {
            //no need to compute gradient at all if NeedGradient is false
            if (!NeedGradient())
            {
                return;
            }

            if (ParentSize() == 0)  // root node
            {
                assert(timeIdxInSeq == (size_t)-1);  //should not be used in a loop
                GradientValues().Resize(1, 1);
                GradientValues().SetValue(1);
            }
            else
            {
                //call each parent's ComputeInputPartial function
                for (int i = 0; i < ParentSize(); i++)
                {
                    ComputationNodePtr pNode = UpCast(m_parents[i]);
                    bool inLoop = !(timeIdxInSeq == (size_t)-1);

                    if (!(pNode->UseCustomizedMultiSeqHandling()))
                        pNode->MaskToZeroWhenLabelAndFeatureMissing(pNode->GradientValues());

                    int j = pNode->GetChildIdForGradientComputation(shared_from_this(), inLoop);

                    if (!inLoop)
                    {
                        pNode->ComputeInputPartial(j);
                    }
                    else
                    {
                        pNode->ComputeInputPartial(j, FrameRange(timeIdxInSeq, m_samplesInRecurrentStep));
                    }
                }
            }
        }


        virtual void ComputeGradientForChildren()
        {
            // batch is done only for feed-forward nodes
            if (HasLoop()) 
                return;

            for (size_t i=0; i<m_children.size(); i++)
            {
                if (!UseCustomizedMultiSeqHandling())
                    MaskToZeroWhenLabelAndFeatureMissing(GradientValues());

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

        virtual void ComputeGradientForChildren(const size_t timeIdxInSeq)
        {
            for (size_t i=0; i<m_children.size(); i++)
            {
                if (!UseCustomizedMultiSeqHandling())
                    MaskToZeroWhenLabelAndFeatureMissing(GradientValues(), timeIdxInSeq);

                ComputationNodePtr child = Inputs(i);
                if (child->NeedGradient())
                {
#ifdef DISPLAY_DEBUG
                    fprintf (stderr, "    [%lu]: %s(%s)\n", i, 
                        (msra::strfun::utf8 (child->OperationName())).c_str(),
                        (msra::strfun::utf8 (child->NodeName())).c_str());
#endif              
                    ComputeInputPartial(i, FrameRange(timeIdxInSeq, m_samplesInRecurrentStep)); //this computes partial wrt to the child and sums the gradient value in the child
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
                child->ClearGradient(true);
            }
        }

        virtual void ClearGradient(const bool clearExistingGradientValue)
        {
            if (NeedGradient())
            {
                ClearChildGradientComputationFlag();

                if (clearExistingGradientValue)
                {
                    GradientValues().Resize(FunctionValues().GetNumRows(), FunctionValues().GetNumCols());
                    if (GradientValues().GetMatrixType() == DENSE)
                    {
                        GradientValues().SetValue(0);
                    }
                    else
                    {
                        GradientValues().Reset();
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

        //m_childrenGradientComputed is a flag to indicate which child has been computed. 
        //this is needed because the same child can be the child of node multiple times, e.g, Y=X*X
        void ClearChildGradientComputationFlag()
        {
            m_childrenGradientComputed.resize(ChildrenSize());

            for (int i = 0; i < ChildrenSize(); i++)
                m_childrenGradientComputed[i] = false;
        }

        //decide which child to compute based on the child pointer passed in and the flag which a child has been computed
        int GetChildIdForGradientComputation(ComputationNodeBasePtr n, const bool inLoop)
        {
            for (int i = 0; i < ChildrenSize(); i++)
            {
                if (m_children[i] == n && (inLoop || !m_childrenGradientComputed[i]))
                {
                    m_childrenGradientComputed[i] = true;
                    return i;
                }
            }

            LogicError("GetChildIdForGradientComputation: cannot find a matched child node that has not been computed yet.\n");
            return -1; //should not go here
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

                node->m_reqMultiSeqHandling = m_reqMultiSeqHandling;
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

        // indicatess whether special handling is needed.The standard handleing will be just mask the function values after the evalaution and mask the gradient before gradiant computation for the children. this is not valid for all criterion nodes whose result is a scalar.
        virtual bool UseCustomizedMultiSeqHandling() { return false; }

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
    using Base::DumpNodeInfo; using Base::EnumerateNodes; using Base::EnumerateNodesForEval; \
    using Base::EnumerateNodesForGradient; using Base::EvaluateThisNode; using Base::FindChildInASet; using Base::FunctionValues; \
    using Base::GradientValues; using Base::HasLoop; using Base::InitRecurrentNode; using Base::Inputs; \
    using Base::IsChildAnImage; using Base::IsEqualTo; using Base::IsFuncValueOlderThanInputs; using Base::IsLeaf; using Base::IsSmaller; \
    using Base::LoadFromFile; using Base::MoveMatricesToDevice; using Base::NeedGradient; using Base::NodeName; \
    using Base::OperationName; using Base::PrintNodeValuesToFile; using Base::PrintSelfBeforeValidation; \
    using Base::RequiresPreCompute; using Base::ReshuffleNodes; using Base::ReshuffleNodesForEvalWithRecurrentLoops; \
    using Base::SaveToFile; using Base::SetFunctionAndGradientSize; using Base::SetInput; using Base::Validate; \
protected:  \
    using Base::m_loopId; using Base::m_samplesInRecurrentStep; \
    using Base::m_visitedOrder; using Base::m_index; using Base::m_lowlink; using Base::m_visited; using Base::m_inStack; \
    using Base::m_indexInLoop; \
    using Base::m_sentenceSeg; using Base::m_minibatchPackingFlag; \
    using Base::m_reqMultiSeqHandling; using Base::UseCustomizedMultiSeqHandling; \
    using Base::m_children; using Base::m_deviceId; using Base::m_evalTimeStamp; using Base::m_functionValues; using Base::m_gradientValues; \
    using Base::m_inputChannels; using Base::m_inputHeight; using Base::m_inputWidth; using Base::m_needGradient; using Base::m_nodeName; \
    using Base::m_outputChannels; using Base::m_outputHeight; using Base::m_outputWidth; using Base::s_constOnes; using Base::s_timeStampCounter; \
    using Base::shared_from_this; \
public:

#pragma endregion base computation class

}}}
