#pragma warning (disable: 4702) // this function is flagged but unclear why
//
// <copyright file="ComputationNetwork.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// TODOs:
//  - fully eliminate EvaluateThisNode(void), replace by EvaluateThisNode(FrameRange(void)), which will often elinminate special-casing
//  - eliminate Network::SetActualMiniBatchSizeFromFeatures() entirely, instead change Node::SetActualMiniBatchSizeFromFeatures()
//    such that it infers its own MB size, and call it during the Evaluate() loop (maybe directly from EvaluateThisNode(), i.e. eliminate it as well or replace the virtual by a helper function)
//    For moving into EvaluateThisNode directly,
//    PRO:
//     - doing it for every call allows to dynamically grow MB and layout frame-by-frame, allowing sophisticated models and ways
//     - more code gone from individual nodes
//    CON:
//     - if moved out from Validate(), errors are not caught during validation (early failure is valuable).
//       Although Validate() can check dimensions other than MB size, while MB-size consistency itself is ensured through layout sharing (and the Matrix functions as a last resort)
//     - inside a loop
//        - it has to check for every frame, although only the first frame really will make a difference
//        - it's weird if it is called for a singe frame to update the size for the whole minibatch
//  - elinminate individual AttachInput() versions with 1..6 arguments (change the outer ones to just call AttachInputs(vector))
//    and implement the dimension check by a using a template parameter to ComputationNode<ElemType>.
//    This will take out and standardize much code from Validate() (will often just be { InferMBLayoutFromInputsForStandardCase(); InferImageDimsFromInputs(); }
//    i.e. Validate() could be completely removed in many cases.
//  - fold EvaluateThisNodeS() into EvaluateThisNode(), same for partial
//  - move NewThis() and OperationName() into the boilerplate macro UsingComputationNodeMembers
//  - also eliminate EvaluateThisNodeGivenInputs()--that stuff is at least as much responsibility of Network
//    We can also replace NeedToMaskMissingColumnsToZero() to a check against the node set itself inside Network --one more Node member gone (and a weird, unneccessary one at that)
//  - finally, revise constructors, merge by means of default parameters
// => many ComputationNode implementations only have two functions EvaluateThisNode(FrameRange) and ComputePartial(), as it was meant to be!

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
    // TODO: comment: what info is stored here? For what?
    struct RecurrentInfo
    {
        std::vector<ComputationNodeBasePtr> m_recurrentNodes;
        std::vector<ComputationNodeBasePtr> m_recurrentNodesForForward;
        ComputationNodeBasePtr m_sourceNode;
        int m_loopId;
        bool m_completedGradient;
        bool m_completedEvaluate;
        bool m_loopClosed;
        bool m_isForwardLoop; 

        void Reset()
        {
            m_completedGradient = false;
            m_completedEvaluate = false;
            m_loopClosed = false;
        }

        // TODO: why is this not a copy constructor or assignment operator?
        void Copy(const RecurrentInfo& src)
        {
            m_recurrentNodes = src.m_recurrentNodes;
            m_recurrentNodesForForward = src.m_recurrentNodesForForward;
            m_sourceNode = src.m_sourceNode;
            m_loopId = src.m_loopId;
            m_completedGradient = src.m_completedGradient;
            m_completedEvaluate = src.m_completedEvaluate;
            m_loopClosed = src.m_loopClosed;
        }
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
    // evaluation
    // -----------------------------------------------------------------------

    static bool IsSmaller(const ComputationNodeBasePtr lhs, const ComputationNodeBasePtr rhs)
    {
        return lhs->GetVisitedOrder() < rhs->GetVisitedOrder();
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

            const ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
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

    void SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr rootNode = nullptr);

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
    void InitLearnableParameters(const ComputationNodeBasePtr node,
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
        nodeToRename->NodeName() = nodeNameNew;
        m_nameToNodeMap.erase(nodeNameOrig);
        m_nameToNodeMap[nodeNameNew] = nodeToRename;
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    template<typename N>
    static shared_ptr<N> AsNodePtr(const ComputationNodeBasePtr & inode)
    {
        return dynamic_pointer_cast<N>(inode);
    }
    template<typename N>
    static bool IsNodePtr(const ComputationNodeBasePtr & inode)
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
    RecurrentInfo * FindInRecurrentLoops(const ComputationNodeBasePtr node)
    {
        // look in all recurrent loops of the network
        for (auto & iter : m_recurrentInfo)
            if (std::find(iter.m_recurrentNodes.begin(), iter.m_recurrentNodes.end(), node) != iter.m_recurrentNodes.end())
                return &iter;
        return nullptr;  // not part of a recurrent loop
    }

    bool IsFuncValueOlderThanInputs(const std::vector<ComputationNodeBasePtr>& recurrentNodes);

    bool IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr);

    void SetRequestNodesMultiSeqHandling();

    // MAIN ENTRY POINT for evaluating one minibatch (forward prop)
    // TODO: pass a set of nodes instead of only one
    // TODO: rename to ForwardProp()? To make it very clear?
    void Evaluate(const ComputationNodeBasePtr rootNode)
    {
        // We have a matching layout structure that matches pMBLayout in number of sequences while not having any flags set.
        // This is used for nodes that do not need recurrent processing, but can be done in batch.
        // TODO: Does it harm if we have flags, for those that can be done in batch? I.e. why don't we just always provide flags?
        //m_pMBNoLayout->Resize(m_pMBLayout->GetNumParallelSequences(), 0);   // TODO: this is not nice, but we currently have no trigger to detect changes in layout

        // prepare to compute with the subnetwork that this rootNode depends on, including
        //  - auto-detecting recurrent loops
        //  - calling Validate() on all nodes, which, a.o, resizes the matrices
        //  - ...more stuff
        BuildAndValidateSubNetwork(rootNode);

        // determines order of evaluation, such that children get evaluated before their parent nodes
        std::list<ComputationNodeBasePtr>& allNodes = GetEvalOrder(rootNode, false);

#ifdef DISPLAY_DEBUG
        for (auto nodeIter=allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedEvaluate = false;

        // (left-over from refactoring: now we only verify that stuff is consistent)
        for (auto node : allNodes)
            if (node->GetMBLayout())
                node->VerifyNumParallelSequences(GetNumParallelSequences());

        // traverse all nodes in the pre-determined evaluation order
        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
            // --- first, evaluate all recurrence that hangs off this

            RecurrentInfo * recInfo = FindInRecurrentLoops(*nodeIter);   // check if this node participates in a recurrent loop

            if (recInfo && IsFuncValueOlderThanInputs(recInfo->m_recurrentNodesForForward) && !recInfo->m_completedEvaluate)
            {
                const auto & recurrentNodes = recInfo->m_recurrentNodesForForward;
                // node participates in a recurrent loop: process the loop frame by frame
                // TODO: move this out of the loop, always do it; gives nodes change to update internal cached layout
                //       ^^ not that simple, since some should not be scaled, such as parameters and criterion nodes
                for (auto & nodeIter : recurrentNodes)
                    nodeIter->SetFunctionAndGradientMBSize(m_actualMBSize);

                const size_t T = m_actualMBSize / GetNumParallelSequences();

                // for every time step run through all nodes in this particular loop
                if (recInfo->m_isForwardLoop)
                {
                    for (size_t t = 0; t < T; t ++)
                    {
                        for (auto nodeIter = recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                        {
                            (*nodeIter)->EvaluateThisNodeGivenInputs(t);
                            (*nodeIter)->UpdateEvalTimeStamp();
                        }
                    } 
                }
                else
                {
                    for (size_t t = T - 1; t --> 0; )
                    {
                        for (auto nodeIter = recurrentNodes.begin(); nodeIter != recurrentNodes.end(); nodeIter++)
                        {
                            (*nodeIter)->EvaluateThisNodeGivenInputs(t);
                            (*nodeIter)->UpdateEvalTimeStamp();
                        }
                    }
                }
    
                recInfo->m_completedEvaluate = true;
            }

            // --- second, do the whole batch (unless it's already done)

            else if (!recInfo && (*nodeIter)->IsFuncValueOlderThanInputs())
            {
#ifdef DISPLAY_DEBUG
                fprintf (stderr, "Evaluate Node: %s\n",(msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif
#if DUMPOUTPUT
                fprintf(stderr,"Forward_%ls\n",(*nodeIter)->NodeName().c_str());
#endif
                // we manage time stamp here so that derived classes don't need to worry about it
                // TODO: is this the whole-batch evaluation?
                (*nodeIter)->EvaluateThisNodeGivenInputs(); 
                (*nodeIter)->UpdateEvalTimeStamp();
            }
        }
    }

    // propagate the features' MB size to all nodes of the network
    // TODO: only resizes nodes participating in recurrent loops ... but can't just resize all since that would also hit, for example, parameter or criterion nodes
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
                nodeIter->SetFunctionAndGradientMBSize(m_actualMBSize);

        return m_actualMBSize;
    }

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


    // MAIN ENTRY POINT for evaluation followed by gradient computation (forward prop then back prop)
    // TODO: pass a set of nodes instead of only one
    template<class ElemType>
    void ComputeGradient(const ComputationNodeBasePtr rootNode, 
                         bool bResetToOne = true,  /// true if reset the gradient of rootnode to 1.0
                         const Matrix<ElemType>* rootGradientInitValue = nullptr,
                         bool bClearGradient = true,
                         bool resetTimeStampAfterComputation = false
                    )
    {
        if (bResetToOne && (rootNode->GetNumRows() != 1 || rootNode->GetNumCols() != 1))
            RuntimeError("ComputeGradient: The root of the Gradient computation must evaluate to R1 value.");

        //run forward pass first
        // TODO: feels out of place; can't we stick for ForwardProp()/BackwardProp()?
        Evaluate(rootNode);

        // TODO: comment what the purpose of this is
        if (bClearGradient)
            ClearGradientForAllNodes(rootNode);

        //run backward pass
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        // TODO: do a runtime check for float vs. double. Also use the Is/AsPtr macros
        if (bResetToOne)
        {
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().Resize(1, 1);   // TODO: make this a function of ComputationNode; but first need to get rid of Matrix<ElemType> here, or make it a local template parameter
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(1);
        }

        if (rootGradientInitValue != nullptr)
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(*rootGradientInitValue);

        // process nodes in pre-determined order
        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
        {
#ifdef DISPLAY_DEBUG
            fprintf(stderr, "Compute Gradient For Node: %s(%s) Against Children\n",
                        (msra::strfun::utf8 ((*nodeIter)->OperationName())).c_str(),
                        (msra::strfun::utf8 ((*nodeIter)->NodeName())).c_str());
#endif
            // --- first, perform recurrent loops if this node participates in one

            RecurrentInfo * recInfo = FindInRecurrentLoops(*nodeIter);
            if (recInfo)
            {
                if (recInfo->m_completedGradient == false)
                {
                    const auto & recurrentNodes = recInfo->m_recurrentNodesForForward;
                    size_t T = m_actualMBSize / GetNumParallelSequences();
                    if (recInfo->m_isForwardLoop)
                    {
                        for (size_t t = T; t--> 0;)
                        {
                            for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                            {
                                (*nodeIter)->VerifyNumParallelSequences(GetNumParallelSequences());
                                (*nodeIter)->ComputeGradientForChildren(t);
                            }
                        }
                    }
                    else
                    {
                        for (size_t t = 0; t < T; t++)
                        {
                            for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                            {
                                (*nodeIter)->VerifyNumParallelSequences(GetNumParallelSequences());
                                (*nodeIter)->ComputeGradientForChildren(t);
                            }
                        }
                    }

                    recInfo->m_completedGradient = true;
                }
            }

            // --- second, do whole-batch operation if not recurrent

            (*nodeIter)->ComputeGradientForChildren();
        }

        //since we now allow sharing of the matrix for function value and gradient value. the function values are now destroyed
        //after gradient computation and need to be recomputed. This is indicated by the timestamp updated using this function
        //resetTimeStampAfterComputation is by default false because ComputeGradient in normal case is followed by new batch of input
        if (resetTimeStampAfterComputation)
            ResetEvalTimeStamp();
    }

    //for debugging purpose
    void PrintComputationTree(const ComputationNodeBasePtr rootNode,
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
    static void SetDropoutRate(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed);
    static void SetMaxTempMemSizeForCNN(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const size_t maxTempMemSizeInSamples);

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    void RenameNode(const ComputationNodeBasePtr node, const std::wstring newNodeName)
    {
        // TODO: check if new name exists
        m_nameToNodeMap.erase(node->NodeName());
        node->NodeName() = newNodeName;
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
    void RebuildNetwork(const ComputationNodeBasePtr rootNode)
    {
        ClearCaches();
        BuildAndValidateSubNetwork(rootNode);
    }

    // -----------------------------------------------------------------------
    // node-group access
    // -----------------------------------------------------------------------

    std::list<ComputationNodeBasePtr> & InputNodes(const ComputationNodeBasePtr rootNode, bool bNoBuild = false)
    {
        if (bNoBuild == false)
            BuildAndValidateSubNetwork(rootNode);
        return m_inputs[rootNode];
    }

    std::list<ComputationNodeBasePtr> & LearnableNodes(const ComputationNodeBasePtr rootNode)
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

    std::list<ComputationNodeBasePtr> GetNodesWithType(const wstring typeName, const ComputationNodeBasePtr rootNode = nullptr)
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
    template<class N> void GetNodesRequiringX(std::list<ComputationNodeBasePtr> & nodesRequirePreComputation, const ComputationNodeBasePtr rootNode, bool checkComputed);
public:
    //return list of nodes that require precomputation and not precomputed yet.
    std::list<ComputationNodeBasePtr> GetNodesRequiringPreComputation(const ComputationNodeBasePtr rootNode = nullptr, bool checkComputed = true);
    //return list of nodes that require precomputation and not precomputed yet.
    std::list<ComputationNodeBasePtr> GetNodesRequiringBatchMode(const ComputationNodeBasePtr rootNode = nullptr, bool checkComputed = true);

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

private:

public: // yak--used by NDLUtil. Will go away someday.
    // ValidateNetwork() - Validate the entire network
    // This calls ValidateNetowrk(Node) for all output nodes.
    // This is used after loading or for dumping the network.
    void ValidateNetwork(bool allowFragment = false, const bool bAllowNoCriterion = false)
    {
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");

        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (ComputationNodeBasePtr & node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                PrintComputationTree(node, false);
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
    void ValidateSubNetwork(const ComputationNodeBasePtr rootNode);
public:
    // prepares the network for computation
    void BuildAndValidateSubNetwork(const ComputationNodeBasePtr rootNode);

    //this function will need to be called before actual validation and execution to 
    //predetermine how to share matrices to reduce memory usage.
    //evalRootNodes do not need gradient computation
    //trainRootNodes need gradient computation
    void AllocateMatrices(std::vector<ComputationNodeBasePtr>& evalRootNodes, std::vector<ComputationNodeBasePtr>& trainRootNodes)
    {
        //allocate memory for forward computation
        fprintf(stderr, "\n\nAllocate matrices for forward computing\n");
        for (int i = 0; i < evalRootNodes.size(); i++)
            AllocateEvalMatrices(evalRootNodes[i]);

        for (int i = 0; i < trainRootNodes.size(); i++)
            AllocateEvalMatrices(trainRootNodes[i]);

        //allocate memory for backward computation
        //we intentionally separate it from above loop to make sure forward computing gets the right matrices
        for (int i = 0; i < trainRootNodes.size(); i++)
            AllocateGradientMatrices(trainRootNodes[i]);
    }

    void AllocateEvalMatrices(ComputationNodeBasePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);

        for (auto & nodeIter : nodes)
        {
            nodeIter->RequestEvalMatrices(m_matrixPool);
            nodeIter->ReleaseMatricesAfterEval(m_matrixPool);
        }
    }

    void AllocateGradientMatrices(ComputationNodeBasePtr rootNode)
    {
        //first, compute the number of parents for each node
        std::map<ComputationNodeBasePtr, int> numParents;

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);

        for (auto & nodeIter : nodes)
        {
            std::vector<ComputationNodeBasePtr> children = nodeIter->GetChildren();
            for (int i = 0; i < children.size(); i++)
                numParents[children[i]] ++;
        }

        //now, simulate the gradient computation order to determine how to allocate matrices
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;

        for (auto & nodeIter : allNodes)
        {
            RecurrentInfo * recInfo = FindInRecurrentLoops(nodeIter);
            if (recInfo && !recInfo->m_completedGradient)
            {
                for (auto & nodeIterInLoop : recInfo->m_recurrentNodesForForward)
                    AllocateGradientMatricesForChildren(nodeIterInLoop, numParents);
                recInfo->m_completedGradient = true;
            }
            else
                AllocateGradientMatricesForChildren(nodeIter, numParents);

            nodeIter->ReleaseGradientMatrices(m_matrixPool);
        }
    }

    void AllocateGradientMatricesForChildren(ComputationNodeBasePtr parentNode, std::map<ComputationNodeBasePtr, int>& numParents)
    {
        std::vector<ComputationNodeBasePtr> children = parentNode->GetChildren();
        for (int i = 0; i < children.size(); i++)
            children[i]->RequestGradientMatrices(m_matrixPool, numParents[children[i]]);
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

    bool UnitTest(const ComputationNodeBasePtr rootNode)
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
    void MergeRecurrentLoops(const ComputationNodeBasePtr /*rootNode*/);
    // get the strong connected component from the graph
    void getStrongSCC(const ComputationNodeBasePtr rootNode);    // TODO: method names start uppercase
    void strongSCC(ComputationNodeBasePtr cur, std::list<ComputationNodeBasePtr>& sccStack, size_t& index, size_t& loopId);     // TODO: method names start uppercase
    void getLoopForwordOrder(std::unordered_set<ComputationNodeBasePtr>& visited, std::unordered_set<ComputationNodeBasePtr>& recStack, std::list<ComputationNodeBasePtr>& nodesStack, ComputationNodeBasePtr cur);   // TODO: method name
    //must be called before ValidateSubNetwork
    void FormRecurrentLoops(const ComputationNodeBasePtr rootNode);
    void DetermineLoopTypes();
    void ReorderLoops(std::list<ComputationNodeBasePtr>& nodes, const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/, const std::list<ComputationNodeBasePtr> & /*noRecurrentNodes*/);
    void CollectInputAndLearnableParameters(const ComputationNodeBasePtr rootNode);

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

public:

    // TODO: move these close to where they are used

    // add a node to m_nameToNodeMap[], which is our node holder
    // Duplicate node names are rejected.
    ComputationNodeBasePtr AddNodeToNet(const ComputationNodeBasePtr nodePtr)
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

    void ClearGradientForAllNodes(const ComputationNodeBasePtr rootNode)
    {
        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); nodeIter++)
            (*nodeIter)->ClearGradientForChildren((int)m_actualMBSize);

        //for (auto nodeIter = m_recurrentInfo.begin(); nodeIter != m_recurrentInfo.end(); nodeIter++)
        //    (*nodeIter).m_completedGradient = false;

        for (int i = 0; i < m_recurrentInfo.size(); i++)
            m_recurrentInfo[i].m_completedGradient = false;
    }

    // determine the required order in which nodes must be computed in order to compute 'rootNode'
    // recurrent == true is only used when called from FormRecurrentLoops()
    std::list<ComputationNodeBasePtr>& GetEvalOrder(const ComputationNodeBasePtr rootNode, bool recurrent)
    {
        return GetCalcOrder(rootNode, m_cacheEvalOrders, true/*means for forward prop*/, recurrent);
    }

    // determine the required order in which nodes must be computed in order to compute the gradient of 'rootNode'
    // Basically returns the reverse of GetEvalOrder(), with some special consideration to loops.
    std::list<ComputationNodeBasePtr>& GetGradientCalcOrder(const ComputationNodeBasePtr rootNode)
    {
        return GetCalcOrder(rootNode, m_cacheGradientCalcOrders, false/*means for backprop*/, false/*recurrent*/);
    }

private:

    // determine computation order to evalute a node (forward prop) or a node's gradient (backprop)
    // This is the common part of GetEvalOrder() and GetGradientCalcOrder().
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

    std::vector<RecurrentInfo> m_recurrentInfo;     // [index--TODO: comment what this is indexed with]

    // used for sentence boundary information passed from reader to reset RNN state 
    // specify how the minibatch is packed for each sample
    MBLayoutPtr m_pMBLayout;    // note that this must be installed before doing anything that needs it (default leaves a nullptr)
    MBLayoutPtr m_pMBNoLayout;  // this alternative one is passed when no layout is available/should be used

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
