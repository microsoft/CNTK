#pragma warning (disable: 4702) // this function is flagged but unclear why
//
// <copyright file="ComputationNetwork.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// TODOs:
//  - eliminate Network::SetActualMiniBatchSizeFromFeatures() entirely, it should already be covered by Node::UpdateFunctionMBSize() which is called from inside the Eval loop
//  - everywhere complete folding EvaluateThisNodeS() into EvaluateThisNode(FrameRange()), same for partial; also make the same change for ComputePartial()
//  - need Matrix::RowSlice() (problem: currently has no 'lead' dimension separate from numRows)
//  - revise constructors, merge by means of default parameters
//  - add a runtime check that nodes deriving from ComputeNodeNonLooping may never participate in a loop
//  - ClassbasedCrossEntropyWithSoftmax::EvaluateThisNodeS()'s calls to MaskMissingColumnsToZero() are likely wrong.
//    Need to understand what this does, then implement it correctly w.r.t. MBLayout.
//  - CRFNode::ComputeInputPartial() fails for >1 parallel seqeunce due to DataSlice() not being able to return whole sequences
//  - BUGBUG: All frameRange.Check() expressions are wrong that operate on children, since the wrong layout pointer is passed in (for most nodes it is identical to the correct one though).
//    This will get fixed as we remove these Check() expressions when absorbing EvaluateThisNodeS().
//  - implement reading of MB Layout in Binary, DSSM, LivbSVM, and SparsePCReader
//  - BUGBUG (in the future): Once we have > 1 layout in the system, all nodes must compare their actual layouts upon Evaluate().
//    Example: TimeReverse must create a new layout. A second TimeReverse ideally would revert back, but can't know. Hence, all consumers of layouts must compare upon Evaluate().
//  - more informative CUDA errors (show original error, machine name, and card id; maybe even a time stamp), to better be able to handle hardware issues
//  - automatic inference of time window w.r.t. delay nodes (and related nodes such as a temporal pooling)
//  - have overrides of RuntimeError etc. in ComputationNode, which prepend the error string with the node name and operation

// The basic idea of this implementation is learned from Brian Guenter <bguenter@microsoft.com>

#include "File.h"
#include "Matrix.h"
#include "commandArgUtil.h" // for nocase_compare

#include "ComputationNode.h"
#include "ScriptableObjects.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <regex>
#include <chrono>


namespace Microsoft { namespace MSR { namespace CNTK {

class ComputationNetwork : public ScriptableObjects::Object, public ScriptableObjects::HasToString, public ScriptableObjects::IConfigRecord
{
protected:
    // recurrent loops in CNTK are like little local ComputationNetworks, but stored in a completely separate set of structures
    // This structure stores that little sub-network.
    struct RecurrentInfo
    {
        std::vector<ComputationNodeBasePtr> m_recurrentNodes;               // all nodes involved in thisloop
        std::vector<ComputationNodeBasePtr> m_recurrentNodesForForward;
        ComputationNodeBasePtr m_sourceNode;
        int m_loopId;                                                       // the loop id (index in xxx array)
        bool m_completedGradient;
        bool m_completedEvaluate;
        bool m_loopClosed;
        bool m_isForwardLoop;                                               // true if left to right (t=0..T-1), false otherwise (t=T-1..0)

        void Reset()
        {
            m_completedGradient = false;
            m_completedEvaluate = false;
            m_loopClosed = false;
        }

#if 0
        // TODO: why is this not a copy constructor or assignment operator?
        void Copy(const RecurrentInfo & src)
        {
            m_recurrentNodes = src.m_recurrentNodes;
            m_recurrentNodesForForward = src.m_recurrentNodesForForward;
            m_sourceNode = src.m_sourceNode;
            m_loopId = src.m_loopId;
            m_completedGradient = src.m_completedGradient;
            m_completedEvaluate = src.m_completedEvaluate;
            m_loopClosed = src.m_loopClosed;
            // m_isForwardLoop??
        }
#endif
    };

public:

    // TODO: sort methods into functional groups; some methods are at random places

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    ComputationNetwork(DEVICEID_TYPE deviceId = AUTOPLACEMATRIX) :
        m_randomSeedOffset(0),
        m_actualMBSize(0),
        m_deviceId(deviceId), m_pMBLayout(make_shared<MBLayout>())
    {
        SetDeviceId(deviceId);
    }

    virtual ~ComputationNetwork()
    {
        ClearNet();
    }

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void ClearNet();

    // -----------------------------------------------------------------------
    // diagnostics
    // -----------------------------------------------------------------------

    //if node name is not found, dump all nodes
    //otherwise dump just that node
    void DumpNodeInfoToFile(const std::wstring & nodeName, const bool printValues, const std::wstring outputFile)
    {
        if (NodeNameExist(nodeName))
        {
            ValidateNetwork(true); //some internal values in the nodes are computed during validation

            File fstream(outputFile,
                         FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

            const ComputationNodeBasePtr& nodePtr = GetNodeFromName(nodeName);
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
        else  //node name is not found, dump all nodes
        {
            fprintf(stderr, "Warning: node name %ls does not exist in the network. dumping all nodes.\n",
                    nodeName.c_str());
            DumpAllNodesToFile(printValues, outputFile);
        }
    }

    //dump all nodes in the network to file
    void DumpAllNodesToFile(const bool printValues,
                            const std::wstring outputFile,
                            const bool validateBeforeDump = true)
    {
        if (validateBeforeDump) 
        {
            //some internal values in the nodes are computed during validation
            ValidateNetwork();
        }

        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = nodeIter->second;
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
    }

    void DumpNodeInfoToFile(const vector<ComputationNodeBasePtr>& nodes,
                            const bool printValues,
                            const std::wstring outputFile)
    {
        ValidateNetwork(); //some internal values in the nodes are computed during validation

        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = *nodeIter;
            nodePtr->DumpNodeInfo(printValues, fstream);
        }
    }

    // -----------------------------------------------------------------------
    // topological plot [erw]
    // TODO: Can this be a separate class? Can it be moved to a CPP?
    // -----------------------------------------------------------------------

private:
    wstring FormSpecialNodes(wstring style, std::vector<ComputationNodeBasePtr>& specialNodes);
    typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;
public:
    void DescribeNetworkUsingDot(std::list<ComputationArc>& arcs, std::wstring outFile);
    void PlotNetworkTopology(const std::wstring outputFile); //  [1/13/2015 erw] plot network topology using dot language

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void SetDeviceId(const DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
    {
        if (m_deviceId == AUTOPLACEMATRIX)
            m_deviceId = Matrix<float>::GetBestGPUDeviceId();
        else
            m_deviceId = deviceId;
        m_deviceId = EnforceOneGPUOnly(m_deviceId);      // see EnforceOneGPUOnly() for comment on what this is
    }

    DEVICEID_TYPE GetDeviceId() { return m_deviceId; }

    unsigned long GetRandomSeedOffset() { return m_randomSeedOffset; }
    void SetRandomSeedOffset(unsigned long value) { m_randomSeedOffset = value; }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // determine the actual MB size from the feature nodes
    // This returns max number of columns over the feature nodes.
    // Note that if we have multiple slices, MB size != #frames.
    size_t DetermineActualMBSizeFromFeatures() const
    {
        size_t actualMBSize = 0;

        const auto & featureNodes = this->FeatureNodes();   // TODO: a getter; should be called GetFeatureNodes()
        for (auto nodeIter = featureNodes.begin(); nodeIter != featureNodes.end(); nodeIter++)
            actualMBSize = max(actualMBSize, (*nodeIter)->GetNumCols());

        return actualMBSize;
    }

    // a helper function for some places that like to hack the features directly
    // This is for a few places (FindBestPath stuff) that don't follow the normal pattern but instead called the old SetFeaturesMiniBatchSize() function with a value of their choosing.
    // This is now changed in that they must actually resize the features, and then the system takes it from here.
    // UNTESTED stopgap. Most likely places that are never used.
    void ResizeAllFeatureNodes(size_t cols)
    {
        auto & featureNodes = this->FeatureNodes();
        for (auto nodeIter = featureNodes.begin(); nodeIter != featureNodes.end(); nodeIter++)
            (*nodeIter)->Resize((*nodeIter)->GetNumRows(), cols);
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    void SaveToFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary) const;
private:
    void SaveToFileImpl(const std::wstring& fileName, const FileOptions fileFormat) const;
public:

    void LoadPersistableParametersFromFile(const std::wstring& fileName, const bool requireValidation = true,
                                           const FileOptions fileFormat = FileOptions::fileOptionsBinary);
    // design BUGBUG: binary files do not know whether they are float or double.
    // TODO: modify file format to know this; then eliminate the <ElemType> dependency (and in some future, allow nodes to be different)
    template<class ElemType>
    void LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary,
                      const bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr);

#pragma region Network Modification

    void SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr& rootNode = nullptr);

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // TODO: describe what this function does
    //this is a temp solution since some nodes such as plus can be just aggregate of two scalar values 
    //in which case the packing info is not available (and not meaningful) for them
    // TODO: Does this belong into MBLayout?
    size_t GetNumSamplesWithLabel(const size_t numAllSamples)
    {
        if (m_pMBLayout && !m_pMBLayout->IsAllNone())
        {
            size_t numTimeSteps = m_pMBLayout->GetNumTimeSteps();
            size_t numSequences = m_pMBLayout->GetNumParallelSequences();

            size_t numSamplesWithoutLabel = 0;

            for (size_t t = 0; t < numTimeSteps; t++)
            {
                if (m_pMBLayout->Is(t, MinibatchPackingFlags::NoLabel))
                {
                    for (int id = 0; id < numSequences; id++)
                    {
                        if (m_pMBLayout->Is(id, t, MinibatchPackingFlags::NoLabel))
                            numSamplesWithoutLabel++;
                    }
                }
            }

            return numTimeSteps*numSequences - numSamplesWithoutLabel;
        }
        else
            return numAllSamples;
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    // non-static version needed because it accesses m_randomSeedOffset
    // Excessively used by SimpleNetworkBuilder, but always after CreateLearnableParameter(), so we should really absorb it there
    template<class ElemType>
    void InitLearnableParameters(const ComputationNodeBasePtr& node,
                                 const bool uniformInit,
                                 const unsigned long randomSeed,
                                 const ElemType initValueScale,
                                 bool initOnCPUOnly = false);

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void DeleteNode(const std::wstring & nodeName)
    {
        //so that deleted node will not be referenced
        ClearCaches();

        ComputationNodeBasePtr nodeToDelete = GetNodeFromName(nodeName);

        //first delete links, if this node is involved, the whole connection will be removed
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (size_t i = 0; i < node->ChildrenSize(); i++)
            {
                ComputationNodeBasePtr child = node->GetChildren()[i];

                //nodeToDelete is a child
                if (child == nodeToDelete)
                {
                    // this used to call DetatchInputs(), but it's better for MEL to retain other inputs
                    node->SetInput(i, nullptr);
                    break;
                }
            }
        }

        //nodeToDelete is a parent
        nodeToDelete->DetachInputs();       // deref all its inputs; if we don't do that, we might end up with a mem leak due to a circular reference

        // unlink from all node-group sets
        for (auto groupIter : GetAllNodeGroups())
        {
            auto search = std::find(groupIter->begin(), groupIter->end(), nodeToDelete);
            if (search != groupIter->end())
                groupIter->erase(search);
        }

        // ? how to deal with m_recurrentInfo, when we delete a node.

        //delete the node itself
        m_nameToNodeMap.erase(nodeName);    // this will deref the node and possibly deallocate it
    }

    // RenameNode - Rename a node to another name
    // nodeNameOrig - original node name
    // nodeNameNew - new node name
    void RenameNode(const std::wstring& nodeNameOrig, const std::wstring& nodeNameNew)
    {
        //so that renamed node will not be referenced
        ClearCaches();

        ComputationNodeBasePtr nodeToRename = GetNodeFromName(nodeNameOrig);

        auto iter = m_nameToNodeMap.find(nodeNameNew);
        if (iter != m_nameToNodeMap.end()) //found
            RuntimeError("RenameNode: Target name already exists.");

        //rename the node and update the mapping table
        nodeToRename->SetNodeName(nodeNameNew);
        m_nameToNodeMap.erase(nodeNameOrig);
        m_nameToNodeMap[nodeNameNew] = nodeToRename;
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    template<typename N>
    static shared_ptr<N> AsNodePtr(const ComputationNodeBasePtr&  inode)
    {
        return dynamic_pointer_cast<N>(inode);
    }
    template<typename N>
    static bool IsNodePtr(const ComputationNodeBasePtr& inode)
    {
        return AsNodePtr<N>(inode) != nullptr;
    }

    // TODO: comment what this function does. Seems to either initialize LearnableParameters or precompute nodes.
    ComputationNodeBasePtr SetNodeValue(const std::wstring & nodeName, const double value);

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    ComputationNodeBasePtr CopyNode(const ComputationNetwork & fromNet,
                                const std::wstring fromName,
                                std::wstring toName = L"",
                                const CopyNodeFlags flags = CopyNodeFlags::copyNodeAll)
    {
        if (toName == L"") {
            toName = fromName;
        }

        ComputationNodeBasePtr pFromNode = fromNet.GetNodeFromName(fromName);
        ComputationNodeBasePtr pToNode;

        // don't allow cross network child copy unless caller explicity handles children fixup
        if ((flags & CopyNodeFlags::copyNodeChildren) &&
            this != &fromNet && !(flags & CopyNodeFlags::copyNodeChildrenCrossNetwork))
        {
            LogicError("CopyNode: Copying node children across network is invalid.");
        }

        if (!NodeNameExist(toName))
        {
            pToNode = pFromNode->Duplicate(toName, flags);
            AddNodeToNet(pToNode);
        }
        else
        {
            //node already exists

            pToNode = GetNodeFromName(toName);

            //same node. no copy needed
            if (pFromNode == pToNode)
                LogicError("CopyNode: You are copying the node to the same network with same node name.");
            else
                pFromNode->CopyTo(pToNode, toName, flags);  // blast it over the existing node
        }
        return pToNode;
    }

    //only copy a complete independent tree
    //when node name exists
    void CopySubTree(const ComputationNetwork & fromNet,
                     const std::wstring fromName, std::wstring toNamePrefix = L"",
                     const CopyNodeFlags flags = copyNodeAll)
    {
        if (!(flags & CopyNodeFlags::copyNodeValue))
            LogicError("CopySubTree: you cannot copy a tree without copying the node values.");

        ComputationNodeBasePtr fromRoot = fromNet.GetNodeFromName(fromName);

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(fromRoot, false);
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr fromNode = *nodeIter;
            wstring fromNodeName = fromNode->NodeName();
            wstring toNodeName = toNamePrefix + fromNodeName;

            ComputationNodeBasePtr toNode = CopyNode(fromNet, fromNodeName,
                                                  toNodeName,
                                                  CopyNodeFlags::copyNodeValue);

            if (flags & CopyNodeFlags::copyNodeChildren)
            {
                //copy the children structure but use the new nodes generated
                for (int i = 0; i < fromNode->ChildrenSize(); i++)
                    toNode->SetInput(i, GetNodeFromName(toNamePrefix + fromNode->GetChildren()[i]->NodeName()));
            }
        }
    }

    //you can only copy inputs from nodes in the same network
    void CopyInputs(const std::wstring fromName, std::wstring toName)
    {
        CopyNode(*this, fromName, toName, CopyNodeFlags::copyNodeChildren);
    }

#pragma endregion Network Modification

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    bool NodeNameExist(const std::wstring& name) const
    {
        auto iter = m_nameToNodeMap.find(name);
        return (iter != m_nameToNodeMap.end());
    }

    ComputationNodeBasePtr GetNodeFromName(const std::wstring& name, ComputationNetwork* anotherNetwork = nullptr, bool bPanic = true) const
    {
        auto iter = m_nameToNodeMap.find(name);
        if (iter != m_nameToNodeMap.end())
        {
            //found
            return iter->second;
        }

        if (anotherNetwork != nullptr)
            return anotherNetwork->GetNodeFromName(name);

        if (bPanic)
            RuntimeError("GetNodeFromName: Node name %s does not exist.", name.c_str());
        else
            return nullptr;
    }

    // GetNodesFromName - Get all the nodes from a name that may match a wildcard '*' pattern
    //   only patterns with a single '*' at the beginning, in the middle, or at the end are accepted
    // name - node name (with possible wildcard)
    // returns: vector of nodes that match the pattern, may return an empty vector for no match
    std::vector<ComputationNodeBasePtr> GetNodesFromName(const std::wstring& name) const
    {
        std::vector<ComputationNodeBasePtr> nodes;
        size_t found = name.find_first_of(L'*');
        if (found == std::wstring::npos)
        {
            if (NodeNameExist(name))
                nodes.push_back(GetNodeFromName(name));
            }
        else
        {
            std::wstring head = name.substr(0, found);
            std::wstring tail = name.substr(found + 1);
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                const wstring& nodeName = nodeIter->first;

                // if it matches on both ends (we only support A*B patterns it's a match
                bool headMatch = head.empty() || nodeName.find(head) == 0;
                bool tailMatch = tail.empty() || nodeName.rfind(tail) == nodeName.size() - tail.size();
                if (headMatch && tailMatch)
                    nodes.push_back(nodeIter->second);
                }
            }
        return nodes;
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // find if node is part of a recurrent loop; and return the loop id
    // If found then return a pointer to the list of nodes of this loop.
    // TODO: This should just return &m_recurrentInfo of the matching loop, or nullptr if no match. If needed, m_recurrentInfo knows its loop id.
    RecurrentInfo * FindInRecurrentLoops(const ComputationNodeBasePtr& node)
    {
        // look in all recurrent loops of the network
        for (auto & iter : m_recurrentInfo)
            if (std::find(iter.m_recurrentNodes.begin(), iter.m_recurrentNodes.end(), node) != iter.m_recurrentNodes.end())
                return &iter;
        return nullptr;  // not part of a recurrent loop
    }

    bool IsFuncValueOlderThanInputs(const std::vector<ComputationNodeBasePtr>& recurrentNodes);

    bool IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr);

    //void SetRequestNodesMultiSeqHandling();

    bool IsNodeReqMultiSeqHandling(const ComputationNodeBasePtr & node) const;

    // GetMaxMBSize - Get the maximum minibatch size that will be seen in a training run
    // returns the result from PropagateActualMiniBatchSize(). Note DetermineActualMBSizeFromFeatures() also exists but returns a value derived from the inputs dimensions
    size_t GetMaxMBSize() { return m_actualMBSize; }

#if 0
    // always called in this pattern:
    evalnet->SetActualMiniBatchSizeFromFeatures();
    dataReader->CopyMBLayoutTo(evalnet->GetMBLayoutPtr());
    evalnet->VerifyActualNumParallelSequences(dataReader->GetNumParallelSequences());
    // well... most of the time. Not in TrainOneEpoch().
    void SetActualNumParallelSequencesInEachRecurentIteration(const size_t aSize)
    {
        m_nbrSlicesInEachRecurrentIteration() = aSize;   // TODO: this has to go
    }
#endif
    size_t GetNumParallelSequences() const
    {
        return m_pMBLayout->GetNumParallelSequences();
    }
    // temporary function: Call this after CopyMBLayoutTo(evalnet->GetMBLayoutPtr()) to ensure everything is consistent as expected
    // It is actually called after every CopyMBLayoutTo() in the entire system (except for multi-reader CopyMBLayoutTo() itself).
    // Remove this function after a few weeks of not firing.
    void VerifyActualNumParallelSequences(const size_t aSize)
    {
        if (GetNumParallelSequences() != aSize)
            LogicError("VerifyActualNumParallelSequences: mismatching MB size in MBLayout");
    }

    // propagate the features' MB size to all nodes of the network
    // TODO: This function should go. Resizing is now part of Validate() and EvaluateThisNode().
    size_t SetActualMiniBatchSizeFromFeatures()
    {
        m_actualMBSize = DetermineActualMBSizeFromFeatures();

        // assume that all nodes in recurrent loops need to be reset to aSize minibatch size, so need to reset the following
        for (int i = 0; i < m_recurrentInfo.size(); i++)
        {
            m_recurrentInfo[i].m_completedEvaluate = false;
            m_recurrentInfo[i].m_completedGradient = false;
        }

        // resize function values and gradients of everything in m_recurrentInfo
        for (int i = 0; i < m_recurrentInfo.size(); i++)
            for (auto & nodeIter : m_recurrentInfo[i].m_recurrentNodes)
                nodeIter->UpdateFunctionMBSize(m_actualMBSize);

        return m_actualMBSize;
    }

    // MAIN ENTRY POINT for evaluating one minibatch (forward prop)
    // TODO: pass a set of nodes instead of only one
    // TODO: rename to ForwardProp()? To make it very clear?
    // TODO: big stuff like this should be in the CPP
    // This calls EvaluateThisNode() on all nodes in order of data flow through the network.
    // By default, the network is applied concurrently on all frames in a minibatch in parallel (a "map" operation)
    // Recurrent loops deviate:
    //  - a recurrent loop is the loop of nodes that make up computation for one time step (e.g. Times -> Plus -> Sigmoid -> Delay)
    //  - these must be executed frame by frame rather than as a map
    //  - such a loop is treated as if they were a little nested network; this is done inside here
    //  - these little nested networks are defined in m_recurrentInfo[]
    void Evaluate(const ComputationNodeBasePtr & rootNode)
    {
        // caller must call BuildAndValidateSubNetwork() before
        // TODO: Some places are hard to fix, e.g. encoder-decoder best-path functions. Those may be broken; this message will tell you.
        if (!BuiltAndValidatedSubNetwork(rootNode))
            LogicError("Evaluate for node %ls %ls: BuildAndValidateSubNetwork() has not been called on this node.");

        // determines order of evaluation, such that children get evaluated before their parent nodes
        std::list<ComputationNodeBasePtr>& allNodes = GetEvalOrder(rootNode, false);

#ifdef DISPLAY_DEBUG
        for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedEvaluate = false;

        // (left-over from refactoring: now we only verify that stuff is consistent)
        for (const auto & node : allNodes)
            if (node->GetMBLayout())
                node->VerifyNumParallelSequences(GetNumParallelSequences());

        // traverse all nodes in the pre-determined evaluation order
        for (auto & node : allNodes)
        {
            // --- first, evaluate all recurrence that hangs off this

            RecurrentInfo * recInfo = FindInRecurrentLoops(node);   // check if this node participates in a recurrent loop

            if (recInfo && IsFuncValueOlderThanInputs(recInfo->m_recurrentNodesForForward) && !recInfo->m_completedEvaluate)
            {
                // node participates in a recurrent loop: process the loop frame by frame
                const auto & recurrentNodes = recInfo->m_recurrentNodesForForward;

                // get layout associated with this loop
                auto pMBLayout = recurrentNodes[0]->GetMBLayout();

                // tell all that loop is about to commence
                for (auto & node2 : recurrentNodes)
                {
                    if (!pMBLayout || node2->GetMBLayout() != pMBLayout)  // take the opportunity to check that layout is shared by all nodes in the loop
                        LogicError("Evaluate: all nodes inside a recurrent loop must have a layout that is identical; mismatch found for nodes '%ls' vs. '%ls'",
                                   node2->NodeName().c_str(), recurrentNodes[0]->NodeName().c_str());
                    node2->UpdateFunctionMBSize(); // TODO: for sequence-to-sequence models we will need to be able to grow this step by step since size is unknown upfront
                    node2->OnEvaluateBeginIteration();
                }

                //since we share memory we need to resize function value matrices correctly
                for (auto & node2 : recurrentNodes)
                {
                    node2->UpdateFunctionMBSize();
                    node2->Validate(true);
                }

                // for every time step run through all nodes in this particular loop (treat the loop like a little ComputationNetwork)
                FrameRangeIteration range(pMBLayout, recInfo->m_isForwardLoop ? -1 : +1);
                for (auto t = range.begin(); t != range.end(); t++)
                {
                    for (auto & node2 : recurrentNodes)
                    {
                        node2->EvaluateThisNode(t);
                        if (IsNodeReqMultiSeqHandling(node2))
                            node2->MaskMissingValuesColumnsToZero(t.t());  // TODO: This should take a FrameRange as well
                        node2->UpdateEvalTimeStamp();
                    }
                } 

                // tell all that loop is done  --e.g. PastValueNode will capture its state for BPTT processing
                for (auto & nodeIter : recurrentNodes)
                    nodeIter->OnEvaluateEndIteration();

                recInfo->m_completedEvaluate = true;
            }

            // --- second, do the whole batch (unless it's already done)

            else if (!recInfo && node->IsFuncValueOlderThanInputs())
            {
#ifdef DISPLAY_DEBUG
                fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 (node->NodeName())).c_str());
#endif
#if DUMPOUTPUT
                fprintf(stderr,"Forward_%ls\n",node->NodeName().c_str());
#endif
                // evaluate the node for all frames concurrently (map)
                // we manage time stamp here so that derived classes don't need to worry about it
                node->UpdateFunctionMBSize();
                if (!node->IsLeaf() && !node->RequiresPreCompute())
                    node->Validate(true);
                node->OnEvaluateBeginIteration();
                node->EvaluateThisNode(FrameRange());
                if (IsNodeReqMultiSeqHandling(node))
                    node->MaskMissingValuesColumnsToZero();
                node->OnEvaluateEndIteration();
                node->UpdateEvalTimeStamp();
            }
            else
                node->OnEvaluateEndIteration();  // HACK to enforce NaN check
        }
    }
    template<class NODESET>
    void Evaluate(const NODESET & nodes)
    {
        for (auto & node : nodes)
            Evaluate(node);
    }

    // MAIN ENTRY POINT for evaluation followed by gradient computation (forward prop then back prop)
    // TODO: pass a set of nodes instead of only one?
    // TODO: remove Evaluate() from here, instead call it at call site, and in here merely check whether everything is computed already
    template<class ElemType>
    void ComputeGradient(const ComputationNodeBasePtr rootNode,
        bool bResetToOne = true,                                    // true if reset the gradient of rootnode to 1.0
        const Matrix<ElemType>* rootGradientInitValue = nullptr,    // if given then this is the starting gradient from the top
        bool bClearGradient = true,                                 // if false then gradients are not cleared  --TODO: When does that happen?
        bool resetTimeStampAfterComputation = false
        )
    {
        // run forward pass first
        // The actual call pattern is
        //  - Evaluate() for eval nodes
        //  - ComputeGradient() for the training criterion
        // I.e. we must call Evaluate() inside here as well, but it will typically only evaluate the training criterion bits because the eval nodes already require most of the network to be computed.
        Evaluate(rootNode);

        // TODO: comment what the purpose/condition of this is
        if (bClearGradient)
            ClearGradientForAllNodes(rootNode);

        // run backprop pass
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        // TODO: do a runtime check for float vs. double. Also use the Is/AsPtr macros
        // The normal case is with the top root with a scalar gradient value of 1.0. This assumes a single and closure network. 
        // Allowing to not initialize to 1 allows network to be open to accept gradients from somewhere.
        // TODO: aren't these two mechanisms mutually exclusive?
        if (bResetToOne)
        {
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().Resize(1, 1);   // TODO: make this a function of ComputationNode; but first need to get rid of Matrix<ElemType> here, or make it a local template parameter
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(1);
        }
        if (rootGradientInitValue != nullptr)   // user-specified gradient to start with
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(*rootGradientInitValue);

        // process nodes in pre-determined order
        for (auto & node : allNodes)
        {
#ifdef DISPLAY_DEBUG
            fprintf(stderr, "Compute Gradient For Node: %ls(%ls) Against Children\n", node->OperationName().c_str(), node->NodeName().c_str());
#endif
            node->OnComputeGradientBeginIteration();  // (currently this only logs)

            // --- first, perform recurrent loops if this node participates in one

            RecurrentInfo * recInfo = FindInRecurrentLoops(node);
            if (recInfo)
            {
                if (recInfo->m_completedGradient == false)
                {
                    const auto & recurrentNodes = recInfo->m_recurrentNodesForForward;
                    auto pMBLayout = recurrentNodes[0]->GetMBLayout();
                    FrameRangeIteration range(pMBLayout, recInfo->m_isForwardLoop ? -1 : +1);
                    for (auto t = range.rbegin(); t != range.rend(); t++)   // note: reverse iteration
                    {
                        for (auto nodeIter2 = recurrentNodes.rbegin(); nodeIter2 != recurrentNodes.rend(); ++nodeIter2)
                        {
                            (*nodeIter2)->VerifyNumParallelSequences(GetNumParallelSequences());
                            if (IsNodeReqMultiSeqHandling(*nodeIter2))
                                (*nodeIter2)->MaskMissingGradientColumnsToZero(t.t());   // TODO: should accept a FrameRange as well
                            (*nodeIter2)->ComputeGradientForChildren(t);
                        }
                    }
                    recInfo->m_completedGradient = true;
                }
            }

            // --- second, do whole-batch operation if not recurrent

            else
            {
                if (IsNodeReqMultiSeqHandling(node))
                {
                    // batch is done only for feed-forward nodes
                    if (node->HasLoop()) // (this test was moved out from MaskMissingGradientColumnsToZero(void), it is likely unnecessary)
                        LogicError("Evaluate: Applying whole-MB operation to node that participates in a loop. This is likely wrong.");
                    node->MaskMissingGradientColumnsToZero();
                }
                node->ComputeGradientForChildren(FrameRange());
            }

            node->OnComputeGradientEndIteration();  // (currently this only logs)
        }

        //since we now allow sharing of the matrix for function value and gradient value. the function values are now destroyed
        //after gradient computation and need to be recomputed. This is indicated by the timestamp updated using this function
        //resetTimeStampAfterComputation is by default false because ComputeGradient in normal case is followed by new batch of input
        if (resetTimeStampAfterComputation)
            ResetEvalTimeStamp();
    }


    //for debugging purpose
    void PrintComputationTree(const ComputationNodeBasePtr& rootNode,
                              const bool forwardCompute,
                              const bool printMatrices = false)
    {
        std::list<ComputationNodeBasePtr> nodes;
        if (forwardCompute)
        {
            fprintf(stderr, "\n\nPrinting Forward Computation Node Order ... \n");
            nodes = GetEvalOrder(rootNode, false);
        }
        else
        {
            fprintf(stderr, "\n\nPrinting Gradient Computation Node Order ... \n");
            nodes = GetGradientCalcOrder(rootNode);
        }

        if (nodes.size() == 0)
        {
            fprintf(stderr, "\n$$$$ EMPTY !!!!!\n");
            return;
        }

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = (*nodeIter);
            node->PrintSelf(printMatrices);
        }
    }

    // a few more helpers
    static void UpdateEvalTimeStamps(const std::vector<ComputationNodeBasePtr> & nodes);
    template<class ElemType> // TODO: dropoutRate change to double
    static void SetDropoutRate(ComputationNetwork& net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed);
    template<class ElemType>
    static void SetSeqParam(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, double hsmoothingWeight, double frameDropThresh, const bool doreferencealign);
    static void SetMaxTempMemSizeForCNN(ComputationNetwork& net, const ComputationNodeBasePtr& criterionNode, const size_t maxTempMemSizeInSamples);

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void RenameNode(ComputationNodeBasePtr node, const std::wstring& newNodeName)
    {
        // TODO: check if new name exists
        m_nameToNodeMap.erase(node->NodeName());
        node->SetNodeName(newNodeName);
        AddNodeToNet(node);
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // called by model editing operations, such as DeleteNode(); and by RebuildNetwork()
    void ClearCaches()
    {
        m_built.clear();
        m_inputs.clear();
        m_learnableParameters.clear();
        ClearCalcOrderCaches();
    }

    // called by TrainOrAdaptModel() for refNet, and from PerformSVDDecomposition()
    // TODO: Is this function really needed?
    void RebuildNetwork(const ComputationNodeBasePtr& rootNode)
    {
        ClearCaches();
        BuildAndValidateSubNetwork(rootNode);
    }

    // -----------------------------------------------------------------------
    // node-group access
    // -----------------------------------------------------------------------

    std::list<ComputationNodeBasePtr>& InputNodes(const ComputationNodeBasePtr& rootNode, bool bNoBuild = false)
    {
        if (bNoBuild == false)
            BuildAndValidateSubNetwork(rootNode);
        return m_inputs[rootNode];
    }

    std::list<ComputationNodeBasePtr>& LearnableNodes(const ComputationNodeBasePtr& rootNode)
    {
        BuildAndValidateSubNetwork(rootNode);
        return m_learnableParameters[rootNode];
    }

    inline       std::vector<ComputationNodeBasePtr> & FeatureNodes()        { return m_features; }
    inline const std::vector<ComputationNodeBasePtr> & FeatureNodes() const  { return m_features; }
    inline       std::vector<ComputationNodeBasePtr> & LabelNodes()          { return m_labels; }
    inline       std::vector<ComputationNodeBasePtr> & FinalCriterionNodes() { return m_finalCriteria; }

    inline std::vector<ComputationNodeBasePtr> CriterionNodesFrom(const wstring & criterionNodeName)
    {
        ComputationNodeBasePtr node = GetNodeFromName(criterionNodeName);
        ValidateSubNetwork(node);
        if (node->GetNumRows() != 1 || node->GetNumCols() != 1)
            InvalidArgument("the criterionNodeName specified in the config file is not a valid training or eval criterion node.");
        // TODO: test this, then remove this comment
        return std::vector<ComputationNodeBasePtr> { node };
    }

    inline std::vector<ComputationNodeBasePtr> & RequestNodesMultiSeqHandling() { return m_requestNodesMultiSeqHandling; }  // user-specified list 'NodesReqMultiSeqHandling' (NDL and MEL create/modify this list)
    inline std::vector<ComputationNodeBasePtr> & EvaluationNodes()              { return m_evalNodes; }
    inline std::vector<ComputationNodeBasePtr> & OutputNodes()                  { return m_outputNodes; }
    inline std::vector<ComputationNodeBasePtr> & PairNodes()                    { return m_pairNodes; }

    inline std::vector<RecurrentInfo> & RecurrentNodes() { return m_recurrentInfo; }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    size_t GetTotalNumberOfNodes() const { return m_nameToNodeMap.size(); }

    // TODO: could be a dup
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare> & GetNameToNodeMap()    // specially for ExperimentalNetworkBuilder; don't use this otherwise
    {
        return m_nameToNodeMap;
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    void ResetEvalTimeStamp()
    {
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            nodeIter->second->ResetEvalTimeStamp();
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    //change the node associated with nodeName to newNode; used in the KL-reg based adaptation to reduce feature copy
    //need to update all the mappings as well childrens
    void ChangeNode(wstring nodeName, ComputationNodeBasePtr newNode)
    {
        ComputationNodeBasePtr oldNode = GetNodeFromName(nodeName);
        if (oldNode->OperationName() != newNode->OperationName())
            InvalidArgument("newNode must have the same type as the old node.");

        //change children
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (int i = 0; i < node->ChildrenSize(); i++)
                if (node->GetChildren()[i] == oldNode)
                    node->SetInput(i, newNode);
        }

        //change name map
        m_nameToNodeMap[nodeName] = newNode;
        for (int i = 0; i < oldNode->ChildrenSize(); i++)
            newNode->SetInput(i, oldNode->GetChildren()[i]);

        //change other maps
        for (auto groupIter : GetAllNodeGroups())
        {
            auto & group = *groupIter;
            for (int i = 0; i < group.size(); i++)
                if (group[i] == oldNode)
                    group[i] = newNode;
        }
    }

    // replace the old node with the current node, assuming the old node is a leaf node
    // need to update those nodes who use oldNode as their child
    void ReplaceLeafNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
    {
        ComputationNodeBasePtr oldNode = GetNodeFromName(oldNodeName);

        // change the input of those nodes whose child is oldNode
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (int i = 0; i < node->ChildrenSize(); i++)
                if (node->GetChildren()[i] == oldNode)
                    node->SetInput(i, newNode);
        }
        m_nameToNodeMap[newNode->GetName()] = newNode;

        // now the old node becomes a orphan node , remove it
        DeleteNode(oldNodeName);
        //RemoveOrphanNode(oldNode);
    }

    void ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
    {
        // Checks if the node is a criterion node.
        int index = -1;
        for (int i = 0; i < m_finalCriteria.size(); ++i)
        {
            if (m_finalCriteria[i]->NodeName() == oldNodeName)
            {
                index = i;
                break;
            }
        }
        if (index == -1)
            RuntimeError("ReplaceFinalCriterionNode: the node to be replaced is not a criterion node.");

        // Replaces children.
        for (int i = 0; i < newNode->ChildrenSize(); ++i)
        {
            if (m_nameToNodeMap.find(newNode->GetChildren()[i]->NodeName()) == m_nameToNodeMap.end())
                RuntimeError("Child node does not exist.");
            newNode->SetInput(i, m_nameToNodeMap[newNode->GetChildren()[i]->NodeName()]);
        }

        // Addes it to criterion node list.
        m_finalCriteria[index] = newNode;
        m_nameToNodeMap[newNode->NodeName()] = newNode;
    }

    void AddFeatureNode(ComputationNodeBasePtr featureNode)
    {
        wstring nodeName = featureNode->NodeName();
        if (NodeNameExist(nodeName))
            RuntimeError("AddFeatureNode: feature node already exists.");
        m_nameToNodeMap[nodeName] = featureNode;
        m_features.push_back(featureNode);
    }

    // We only remove the node, not delete it.
    void RemoveFeatureNode(ComputationNodeBasePtr featureNode)
    {
        wstring nodeName = featureNode->NodeName();
        if (!NodeNameExist(nodeName))
            RuntimeError("RemoveFeatureNode: feature node does not exist.");

        ClearCaches();

        // Removes links.
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); ++nodeIter)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            for (size_t i = 0; i < node->ChildrenSize(); ++i)
            {
                ComputationNodeBasePtr child = node->GetChildren()[i];
                if (child == featureNode)
                {
                    node->SetInput(i,NULL);
                    break;
                }
            }
        }

        // Removes from feature list.
        auto search = std::find(m_features.begin(), m_features.end(), featureNode);
        if (search != m_features.end())
            m_features.erase(search);

        m_nameToNodeMap.erase(nodeName);
    }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    std::vector<ComputationNodeBasePtr> GetAllNodes() const
    {
        std::vector<ComputationNodeBasePtr> nodes;
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            nodes.push_back(node);
        }
        return nodes;
    }

    std::list<ComputationNodeBasePtr> GetNodesWithType(const wstring typeName, const ComputationNodeBasePtr& rootNode = nullptr)
    {
        std::list<ComputationNodeBasePtr> nodesWithType;

        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->OperationName() == typeName)
                    nodesWithType.push_back(node);
            }
        }
        else
        {
            //for calculating a specific node
            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if (node->OperationName() == typeName)
                    nodesWithType.push_back(node);
            }
        }

        return nodesWithType;
    }

private:
    template<class N> void GetNodesRequiringX(std::list<ComputationNodeBasePtr> & nodesRequirePreComputation, const ComputationNodeBasePtr& rootNode, bool checkComputed);
public:
    //return list of nodes that require precomputation and not precomputed yet.
    std::list<ComputationNodeBasePtr> GetNodesRequiringPreComputation(const ComputationNodeBasePtr& rootNode = nullptr, bool checkComputed = true);
    //return list of nodes that require precomputation and not precomputed yet.
    std::list<ComputationNodeBasePtr> GetNodesRequiringBatchMode(const ComputationNodeBasePtr& rootNode = nullptr, bool checkComputed = true);

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

public: // yak--used by NDLUtil. Will go away someday.
    // ValidateNetwork() - Validate the entire network
    // This calls ValidateNetowrk(Node) for all output nodes.
    // This is used after loading or for dumping the network.
    void ValidateNetwork(bool allowFragment = false, const bool bAllowNoCriterion = false)
    {
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");

        // TODO: allocation does not belong here. This is called e.g. after loading. Memory should be allocated only when actually evaluating.
        AllocateAllEvalMatrices(EvaluationNodes(), OutputNodes(), FinalCriterionNodes());

        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (ComputationNodeBasePtr & node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
#ifdef _DEBUG
                PrintComputationTree(node, false);
#endif
                SetActualMiniBatchSizeFromFeatures();
                ValidateSubNetwork(node);
            }
        }
        else if (bAllowNoCriterion == true)
        {
            // do nothing
        }
        else if (!allowFragment)
            RuntimeError("No Criterion nodes specified");

        // now output nodes
        if (OutputNodes().size() > 0)
        {
            for (ComputationNodeBasePtr node : OutputNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateSubNetwork(node);
            }
        }
        else if (!allowFragment)
            RuntimeError("No Output nodes specified");

        // now evaluation nodes
        if (EvaluationNodes().size() > 0)
        {
            for (ComputationNodeBasePtr node : EvaluationNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateSubNetwork(node);
            }
        }
    }
private:
    void ValidateNodes(list<ComputationNodeBasePtr> nodes, bool isFinalValidationPass, size_t & todo);
    void ValidateSubNetwork(const ComputationNodeBasePtr& rootNode);
public:
    // prepares the network for computation
    void BuildAndValidateSubNetwork(const ComputationNodeBasePtr rootNode);
    // and for a set of nodes
    void StartEvaluateMinibatchLoop(const ComputationNodeBasePtr & rootNode)  // (ugly name; meant to be unique so we can rename if needed)
    {
        BuildAndValidateSubNetwork(rootNode);
    }
    template<class NODESET>
    void StartEvaluateMinibatchLoop(const NODESET & nodes)  // (ugly name; meant to be unique so we can rename if needed)
    {
        for (auto & node : nodes)
            StartEvaluateMinibatchLoop(node);
    }
    template<class NODESET>
    void StartEvaluateMinibatchLoop(const NODESET & nodes1, const NODESET & nodes2) // often needed for two sets (training & evaluation criteria)
    {
        StartEvaluateMinibatchLoop(nodes1);
        StartEvaluateMinibatchLoop(nodes2);
    }
    bool BuiltAndValidatedSubNetwork(const ComputationNodeBasePtr & rootNode);

    //this function will need to be called before actual validation and execution to 
    //predetermine how to share matrices to reduce memory usage.
    //TODO: find a simple topological order and allocateEvalMatrices on that order directly
    //without passing in eval, out, and train nodes.
    void AllocateAllEvalMatrices(std::vector<ComputationNodeBasePtr>& evalRootNodes, 
                                 std::vector<ComputationNodeBasePtr>& outValueRootNodes,
                                 std::vector<ComputationNodeBasePtr>& trainRootNodes)
    {
        //allocate memory for forward computation
        fprintf(stderr, "\n\nAllocating matrices for forward computing\n");
        for (int i = 0; i < evalRootNodes.size(); i++)
            AllocateEvalMatrices(evalRootNodes[i]);

        for (int i = 0; i < outValueRootNodes.size(); i++)
            AllocateEvalMatrices(outValueRootNodes[i]);

        for (int i = 0; i < trainRootNodes.size(); i++)
            AllocateEvalMatrices(trainRootNodes[i]);
    }

    void AllocateEvalMatrices(ComputationNodeBasePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        std::list<ComputationNodeBasePtr>& allNodes = GetEvalOrder(rootNode, false);

        //determine parent size
        std::map<ComputationNodeBasePtr, int> parentCount;
        for (auto &n : allNodes)
        {
            for (int i = 0; i < n->ChildrenSize(); i++)
            {
                ComputationNodeBasePtr pNode = n->GetChildren()[i];
                parentCount[pNode]++;
            }
        }

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedEvaluate = false;

        for (auto &nodeIter : allNodes)
        {
            if (nodeIter->HasLoop())
            {
                RecurrentInfo* recInfo = FindInRecurrentLoops(nodeIter);
                assert(recInfo != nullptr);
                if (recInfo->m_completedEvaluate == false)
                {
                    const auto & recurrentNodes = recInfo->m_recurrentNodesForForward;
                    for (auto &nodeLoopIter : recurrentNodes)
                    {
                        nodeLoopIter->RequestMatricesBeforeEval(m_matrixPool);
                    }

                    recInfo->m_completedEvaluate = true;

                    for (auto &nodeLoopIter : recurrentNodes)
                    {
                        ReleaseMatricesAfterEvalForChildren(nodeLoopIter, parentCount);
                    }
                }
            }
            else
            {
                nodeIter->RequestMatricesBeforeEval(m_matrixPool);
                //we only release matrices for the children since the root node's informatioin will be used and should not be shared
                //with others
                ReleaseMatricesAfterEvalForChildren(nodeIter, parentCount);
            }
        }
    }

    void ReleaseMatricesAfterEvalForChildren(ComputationNodeBasePtr n, std::map<ComputationNodeBasePtr, int>& parentCount)
    {
        for (int i = 0; i < n->ChildrenSize(); i++)
        {
            ComputationNodeBasePtr pNode = n->GetChildren()[i];
            parentCount[pNode]--;
            if (parentCount[pNode] == 0)
                pNode->ReleaseMatricesAfterEval(m_matrixPool);
        }
    }

    void AllocateGradientMatrices(ComputationNodeBasePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        //PopulateParents(rootNode);

        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        //determine children size
        //std::map<ComputationNodeBasePtr, int> childrenCount;
        //for (auto &nodeIter : allNodes)
        //{
        //    childrenCount[nodeIter] = nodeIter->ChildrenSize();
        //}

        //now, simulate the gradient computation order to determine how to allocate matrices
        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;

        //we need to call it here since we always compute gradients for children and root node is not children of other node
        rootNode->RequestMatricesBeforeGradientComp(m_matrixPool);

        for (auto &n : allNodes)
        {
            if (n->HasLoop())
            {
                std::vector<ComputationNodeBasePtr> recurrentNodes;
                RecurrentInfo * recInfo = FindInRecurrentLoops(n);
                if (recInfo && recInfo->m_completedGradient == false)
                {
                    const auto & recurrentNodes = recInfo->m_recurrentNodesForForward;
                    //loops are computed sample by sample so we have to allocate them all 
                    for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                    {
                        AllocateGradientMatricesForChildren(*nodeIter);
                    }
                    
                    recInfo->m_completedGradient = true;

                    for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                    {
                        (*nodeIter)->ReleaseMatricesAfterGradientComp(m_matrixPool);
                    }
                }
            }
            else
            {
                AllocateGradientMatricesForChildren(n);
                if (n != rootNode)  //root node's informatioin will be used and should not be shared with others, also it's small (1x1)
                    n->ReleaseMatricesAfterGradientComp(m_matrixPool);
            }
        }
    }

    //void ReleaseMatricesAfterGradientCompForParents(ComputationNodeBasePtr n, std::map<ComputationNodeBasePtr, int>& childrenCount)
    //{
    //    for (int i = 0; i < n->ParentSize(); i++)
    //    {
    //        ComputationNodeBasePtr pNode = n->Parent(i);
    //        childrenCount[pNode] --;
    //        if (childrenCount[pNode] == 0)
    //            pNode->ReleaseMatricesAfterGradientComp(m_matrixPool);
    //    }
    //}
  

    void AllocateGradientMatricesForChildren(ComputationNodeBasePtr parentNode)
    {
        std::vector<ComputationNodeBasePtr> children = parentNode->GetChildren();
        for (int i = 0; i < children.size(); i++)
        {
            if (children[i]->NeedGradient())
                children[i]->RequestMatricesBeforeGradientComp(m_matrixPool);
        }
    }

    /**
    call unit test of each node
    this adds a verification of the correctness of node operations.
    */
    bool UnitTest(bool allowFragment = false)
    {
        vector<wstring> vErrors;
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");
        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (auto & node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                this->SetActualMiniBatchSizeFromFeatures();
                if (!UnitTest(node))
                    vErrors.push_back(node->NodeName().c_str());
            }
        }
        else if (!allowFragment)
            RuntimeError("No Criterion nodes specified");
        // now output nodes
        if (OutputNodes().size() > 0)
        {
            for (auto & node : OutputNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
        else if (!allowFragment)
            RuntimeError("No Output nodes specified");
        // now evaluation nodes
        if (EvaluationNodes().size() > 0)
        {
            for (auto & node : EvaluationNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
        return vErrors.empty();
    }

    bool UnitTest(const ComputationNodeBasePtr& rootNode)
    {
        fprintf(stderr, "\n\n Unit test node %ls \n", rootNode->NodeName().c_str());

        std::list<ComputationNodeBasePtr>&  nodes = GetEvalOrder(rootNode, false);

        for (auto & nodeIter : nodes)
            if (!nodeIter->UnitTest())
                return false;

        fprintf(stderr, "\n\n");

        return true;
    }

    // -----------------------------------------------------------------------
    // specialized operations
    // -----------------------------------------------------------------------

    // TODO: lift this into config language, move underlying code to math lib

    //========================================
    // This function performs SVD decomposition for different groups of learnable  parameters
    // we perform SVD decomposition such that
    //  A \approx B*C, where rank(B)=rank(C)=r < rank(A)
    // After SVD decomposition, the node A will become an intermediate node whose children are B,C ;
    // B and C are two learnable parameters
    //========================================
    // BUGBUG: this only currently works for one ElemType, not both
    template<class ElemType>
    void PerformSVDecomposition(const map<wstring, float>& SVDConfig);

public:
    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // TODO: make these templated on <ElemType> locally
    template<class ElemType>
    void GetHistory(map<wstring, Matrix<ElemType>>& history, bool bLastTime = false)
    {
        //put all node info first
        Matrix<ElemType> hist;
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            shared_ptr<ComputationNode<ElemType>> nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(nodeIter->second);
            if (nodePtr && nodePtr->GetHistory(hist, bLastTime))
                history[nodeIter->first] = hist;
        }
    };

    // only called from FindBestPath() and FindbestPathWithVariableLength()
    template<class ElemType>
    void SetHistory(map<wstring, Matrix<ElemType>>& history)
    {
        //put all node info first
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            shared_ptr<ComputationNode<ElemType>> nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(nodeIter->second);
            if (nodePtr && history.find(nodeIter->first) != history.end())
                nodePtr->SetHistory(history[nodeIter->first]);
        }
    };

    // note: this is called to write into our existing MBLayout instance
    // TODO: This is broken. Instead, we should pass this from the reader, or better, do batching inside here.
    //       The problem is that we cannot post-process. E.g. is the layout guaranteed to reflect the minibatch size, in the case of no recurrence??
    const MBLayoutPtr & GetMBLayoutPtr() { return m_pMBLayout; }
protected:
    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    // Copy constructor, should never be called.
#pragma warning (push)
#pragma warning (disable: 4702) // this function is flagged but unclear why
    ComputationNetwork(const ComputationNetwork& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork(const ComputationNetwork& deepCopyFrom)' should never be called.");
    }
#pragma warning (pop)

    // Assignment operator, should never be called.
    ComputationNetwork& operator=(const ComputationNetwork& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork& operator=(const ComputationNetwork& deepCopyFrom)' should never be called.");
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // The methods below determine evaluation order, which is tricky in presence of recurrent loops.
    // TODO: Can this be moved to a separate class, or at least a separate CPP?

    void ClearCalcOrderCaches();
    void MergeRecurrentLoops(const ComputationNodeBasePtr& /*rootNode*/);
    // get the strong connected component from the graph
    void getStrongSCC(const ComputationNodeBasePtr& rootNode);    // TODO: method names start uppercase
    void strongSCC(ComputationNodeBasePtr cur, std::list<ComputationNodeBasePtr>& sccStack, size_t& index, size_t& loopId);     // TODO: method names start uppercase
    void getLoopForwordOrder(std::unordered_set<ComputationNodeBasePtr>& visited, std::unordered_set<ComputationNodeBasePtr>& recStack, std::list<ComputationNodeBasePtr>& nodesStack, ComputationNodeBasePtr cur);   // TODO: method name
    //must be called before ValidateSubNetwork
    void FormRecurrentLoops(const ComputationNodeBasePtr& rootNode);
    void DetermineLoopTypes();
    void ReorderLoops(std::list<ComputationNodeBasePtr>& nodes, const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/, const std::list<ComputationNodeBasePtr> & /*noRecurrentNodes*/);
    void CollectInputAndLearnableParameters(const ComputationNodeBasePtr& rootNode);

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

public:

    // TODO: move these close to where they are used

    // add a node to m_nameToNodeMap[], which is our node holder
    // Duplicate node names are rejected.
    ComputationNodeBasePtr AddNodeToNet(const ComputationNodeBasePtr& nodePtr)
    {
        //found
        // TODO: use .insert() and test result.second == false means not inserted since already exists
        if (m_nameToNodeMap.find(nodePtr->NodeName()) != m_nameToNodeMap.end())
            RuntimeError("Duplicated computation node name.");

        m_nameToNodeMap[nodePtr->NodeName()] = nodePtr;
        return nodePtr; // allows e.g. return AddNodeToNet(New...);
    }
    // TODO: not very nice--need to fix way more outside to get this right
    template<class N>
    shared_ptr<N> AddNodeToNetWithElemType(const shared_ptr<N> nodePtr)
    {
        return dynamic_pointer_cast<N>(AddNodeToNet(nodePtr));
    }

    template<class N, class... _Types>
    shared_ptr<N> AddNodeToNetAndAttachInputs(const shared_ptr<N> nodePtr, _Types&&... _Args)
    {
        nodePtr->AttachInputs(std::forward<_Types>(_Args)...);
        return AddNodeToNetWithElemType(nodePtr);
        //return nodePtr; // allows e.g. return AddNodeToNetAndAttachInputs(New..., inputs);
    }

public:

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    //void ClearGradientForAllNodes(const ComputationNodeBasePtr& rootNode)
    //{
    //    std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

    //    for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
    //        (*nodeIter)->ClearGradientForChildren((int)m_actualMBSize);

    //    //for (auto nodeIter = m_recurrentInfo.begin(); nodeIter != m_recurrentInfo.end(); nodeIter++)
    //    //    (*nodeIter).m_completedGradient = false;

    //    for (int i = 0; i < m_recurrentInfo.size(); i++)
    //        m_recurrentInfo[i].m_completedGradient = false;
    //}

    void ClearGradientForAllNodes(const ComputationNodeBasePtr& rootNode)
    {
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (auto &node : allNodes)
            node->ClearGradientForChildren();

//            node->MarkGradientInitialized(false);

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;
    }

    // determine the required order in which nodes must be computed in order to compute 'rootNode'
    // recurrent == true is only used when called from FormRecurrentLoops()
    std::list<ComputationNodeBasePtr>& GetEvalOrder(const ComputationNodeBasePtr& rootNode, bool recurrent)
    {
        return GetCalcOrder(rootNode, m_cacheEvalOrders, true/*means for forward prop*/, recurrent);
    }

    // determine the required order in which nodes must be computed in order to compute the gradient of 'rootNode'
    // Basically returns the reverse of GetEvalOrder(), with some special consideration to loops.
    std::list<ComputationNodeBasePtr>& GetGradientCalcOrder(const ComputationNodeBasePtr& rootNode)
    {
        return GetCalcOrder(rootNode, m_cacheGradientCalcOrders, false/*means for backprop*/, false/*recurrent*/);
    }

private:

    //this will determine the parents for each node. Parents info will be used by the gradient computation
    //void PopulateParents(const ComputationNodeBasePtr& rootNode)
    //{
    //    std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);

    //    for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
    //    {
    //        ComputationNodeBasePtr n = (*nodeIter);

    //        //clear parents
    //        n->ClearParents(); 

    //        //add it to children's parents. children's parent collection has alraedy been cleared
    //        std::vector<ComputationNodeBasePtr> children = n->GetChildren();
    //        for (int i = 0; i < children.size(); i++)
    //            children[i]->AddParent(n);
    //    }
    //}


    static std::list<ComputationNodeBasePtr>& GetCalcOrder(const ComputationNodeBasePtr rootNode,
                                                           std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>>& orderMap,
                                                           const bool forwardCompute, bool recurrent)
    {
        if (!rootNode)
            LogicError("rootNode is NULL.");
        if (orderMap.find(rootNode) == orderMap.end())
            orderMap[rootNode] = rootNode->EnumerateNodes(forwardCompute, recurrent);
        return orderMap[rootNode];
    }

public:

    // FixupInputMinibatchSize - go through all the inputs and make sure they have a consistent minibatch size (after creation)
    void FixupInputMinibatchSize();

    // -----------------------------------------------------------------------
    // BS integration
    // -----------------------------------------------------------------------

    // create a somewhat readable representation, aimed at diagnostics/debugging
    wstring /*HasToString::*/ToString() const
    {
        wstring args;
        for (auto & iter : m_nameToNodeMap)
        {
            const auto node = iter.second;
            if (!args.empty())
                args.append(L"\n");
            args.append(node->ToString());
        }
        return TypeId<decltype(*this)>() + L" " + NestString(args, L'[', true, ']');
    }

    // pretending to be a ConfigRecord. TODO: implement this when we actually need it (when we get to MEL)
    const ScriptableObjects::ConfigValuePtr & /*IConfigRecord::*/operator[](const wstring & id) const   // e.g. confRec[L"message"]
    {
        id; RuntimeError("unknown class parameter");    // (for now)
    }
    const ScriptableObjects::ConfigValuePtr * /*IConfigRecord::*/Find(const wstring & id) const         // returns nullptr if not found
    {
        id; return nullptr; // (for now)
    }
    vector<wstring> /*IConfigRecord::*/GetMemberIds() const
    {
        return vector<wstring>();
    }

protected:

    // -----------------------------------------------------------------------
    // data members
    // -----------------------------------------------------------------------

    // TODO: move basic accessors in here?

    DEVICEID_TYPE m_deviceId;           // TODO: is this shared by all nodes?
    unsigned long m_randomSeedOffset;

    // node groups
    std::vector<ComputationNodeBasePtr> m_features;
    std::vector<ComputationNodeBasePtr> m_labels;
    std::vector<ComputationNodeBasePtr> m_finalCriteria;
    std::vector<ComputationNodeBasePtr> m_evalNodes;
    std::vector<ComputationNodeBasePtr> m_outputNodes;
    std::vector<ComputationNodeBasePtr> m_pairNodes; /// nodes for the children network to pair
    std::vector<ComputationNodeBasePtr> m_requestNodesMultiSeqHandling;
    vector<std::vector<ComputationNodeBasePtr>*> GetAllNodeGroups()    // get all groups to allow to iterate over all of them ...continue
    {
        return vector<std::vector<ComputationNodeBasePtr>*> { &m_features, &m_labels, &m_finalCriteria, &m_evalNodes, &m_outputNodes, &m_pairNodes, &m_requestNodesMultiSeqHandling };
    }

    std::vector<RecurrentInfo> m_recurrentInfo;     // [loopId] each entry is one recurrent loop (local little network that implements a recurrence)

    // used for sentence boundary information passed from reader to reset RNN state 
    // specify how the minibatch is packed for each sample
    MBLayoutPtr m_pMBLayout;    // note that this must be installed before doing anything that needs it (default leaves a nullptr)
    //MBLayoutPtr m_pMBNoLayout;  // this alternative one is passed when no layout is available/should be used

    size_t m_actualMBSize;      // current MB size in columns --note: this is not #frames, if we have multiple parallel sequences, cf. MBLayout

    // main node holder
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare> m_nameToNodeMap;   // [name] -> node; this is the main container that holds this networks' nodes

private:    // TODO: make all private that can be made private
    // cache for evaluation ordering:
    std::unordered_set<ComputationNodeBasePtr> m_built;   // [node] flag: BuildAndValidateSubNetwork() has been called

    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_cacheEvalOrders;
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_cacheGradientCalcOrders;

    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_inputs;                 // [out node] -> all input nodes feeding into out node
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_learnableParameters;    // [out node] -> all parameter nodes feeding into out node

    // pool for matrices that can be shared across nodes
    // TODO: does this apply to anything else besides temporary node-internal intermediate results? What, for example?
    MatrixPool m_matrixPool;
};

}}}
