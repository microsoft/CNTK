//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"
#include "File.h"
#include "Matrix.h"
#include "Config.h"

#include "ComputationNode.h"
#include "ScriptableObjects.h"
#include "ComputationEnvironment.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <regex>
#include <chrono>
#include <unordered_map>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

// ===========================================================================
// ComputationNetwork -- computation graph and operations
// ===========================================================================

class ComputationNetwork :
    public ScriptableObjects::Object,
    public ScriptableObjects::HasToString,
    public ScriptableObjects::CustomConfigRecord
{
public:
    typedef shared_ptr<ComputationNetwork> ComputationNetworkPtr;

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    ComputationNetwork() :
        m_randomSeedOffset(0),
        m_isCompiled(false),
        m_areMatricesAllocated(false),
        m_pMBLayoutOfNetwork(make_shared<MBLayout>(1, 0, L"*")),
        m_environment(make_shared<ComputationEnvironment>())
    {
        //m_pMBLayoutOfNetwork->SetAxisName(L"T");
    }

    ComputationNetwork(DEVICEID_TYPE deviceId) :
        ComputationNetwork()
    {
        SetDeviceId(deviceId);
    }
    ComputationNetwork(const ScriptableObjects::IConfigRecordPtr configp); // construct from config

    virtual ~ComputationNetwork()
    {
        ClearNetwork(); // This will explicitly remove all nodes. This is needed to break circular references in loops.
    }

    void ClearNetwork();
    void InvalidateCompiledNetwork();

    void SetDeviceId(DEVICEID_TYPE deviceId)
    {
        m_deviceId = deviceId;
    }

    DEVICEID_TYPE GetDeviceId() const { return m_deviceId; }

protected:
    void ConstructFromRoots(DEVICEID_TYPE deviceId, std::deque<ComputationNodeBasePtr>&& roots, const map<ComputationNodeBasePtr, ComputationNodeBasePtr>& replacements);
    void ProcessSpecialNodes(const ScriptableObjects::IConfigRecord& config, std::deque<ComputationNodeBasePtr>& roots);

public:
    // -----------------------------------------------------------------------
    // (de-)serialization
    // -----------------------------------------------------------------------

    template <class ElemType>
    void ReadPersistableParameters(File& fstream, bool create);
    // reload node content only, e.g. used by SGD::Train() when going back to an older model that had better training objective
    template <class ElemType>
    void RereadPersistableParameters(const std::wstring& fileName)
    {
        File fstream(fileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);
        ReadPersistableParameters<ElemType>(fstream, false);
    }
    // design BUGBUG: binary files do not know whether they are float or double.
    // TODO: modify file format to know this; then eliminate the <ElemType> dependency (and in some future, allow nodes to be different)
    template <class ElemType> void Read(const std::wstring& fileName);
    template <class ElemType> void Load(const std::wstring& fileName)
    {
        Read<ElemType>(fileName);
        // perform all further post-processing, caching, etc.
        CompileNetwork();
    }

    // static helper to instantiate a network from a file
    template <class ElemType>
    static ComputationNetworkPtr CreateFromFile(DEVICEID_TYPE deviceId, const std::wstring& fileName)
    {
        auto net = make_shared<ComputationNetwork>(deviceId);
        net->Load<ElemType>(fileName);
        return net;
    }

    void Save(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary) const;
    void SaveEdited(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary);

private:

    void SaveToFileImpl(const std::wstring& fileName, const FileOptions fileFormat) const;

public:

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // main entry point for forward prop
    void ForwardProp(const ComputationNodeBasePtr rootNode);

    // main entry point for backprop
    void Backprop(const ComputationNodeBasePtr rootNode);

    template <class NODESET> // version that takes multiple nodes
    void ForwardProp(const NODESET& nodes)
    {
        for (auto& node : nodes)
            ForwardProp(node);
    }

    static void BumpEvalTimeStamp(const std::vector<ComputationNodeBasePtr>& nodes);
    void ResetEvalTimeStamps();

    // and for a set of nodes
    void StartEvaluateMinibatchLoop(const ComputationNodeBasePtr& rootNode) // (ugly name; meant to be unique so we can rename if needed)
    {
        VerifyIsCompiled("StartEvaluateMinibatchLoop");
        ResetEvalTimeStamps(); // invalidate all m_value fields  --TODO: redundant (called over again for every root node). Make this private and only call for sets of nodes.
    }
    template <class NODESET>
    void StartEvaluateMinibatchLoop(const NODESET& nodes) // (ugly name; meant to be unique so we can rename if needed)
    {
        for (auto& node : nodes)
            StartEvaluateMinibatchLoop(node);
    }
    template <class NODESET>
    void StartEvaluateMinibatchLoop(const NODESET& nodes1, const NODESET& nodes2) // often needed for two sets (training & evaluation criteria)
    {
        StartEvaluateMinibatchLoop(nodes1);
        StartEvaluateMinibatchLoop(nodes2);
    }

    // -----------------------------------------------------------------------
    // evaluation: preparation
    // -----------------------------------------------------------------------

    void CompileNetwork(); // call this after creation, Load(), and any modification

private:
    void ValidateNetwork();
    size_t ValidateNodes(list<ComputationNodeBasePtr> nodes, bool isFirstPass, bool isFinalValidationPass);
    bool ValidateNode(ComputationNodeBasePtr node, bool isFinalValidationPass) const;
    void MarkValueNonSharableNodes();
    void ChangeNodeInputs(ComputationNodeBasePtr fromNode, ComputationNodeBasePtr toNode);

private:
    void DetermineSetOfAllRoots();
    void CollectInputAndLearnableParameters(const ComputationNodeBasePtr& rootNode);
    void CollectInputAndLearnableParametersRec(const ComputationNodeBasePtr& node, set<ComputationNodeBasePtr>& visited, list<ComputationNodeBasePtr>& inputs, list<ComputationNodeBasePtr>& learnableParameters);
    void ResetMBLayouts();
    bool IsCompiled() const { return m_isCompiled; }
    bool AreMatricesAllocated() const { return m_areMatricesAllocated; }
    void VerifyIsCompiled(const char* where) const;
public:
    void AllocateAllMatrices(const std::vector<ComputationNodeBasePtr>& evalRootNodes, const std::vector<ComputationNodeBasePtr>& outValueRootNodes, ComputationNodeBasePtr trainRootNode);

private:
    void PrintMemorySharingStructure(const std::vector<ComputationNodeBasePtr>& nodes);
    void ReleaseMatricesAfterEvalForChildren(ComputationNodeBasePtr n, std::unordered_map<ComputationNodeBasePtr, int>& parentCount);
    void AllocateGradientMatricesForInputs(ComputationNodeBasePtr parentNode);

public:
    // -----------------------------------------------------------------------
    // evaluation: execution plan and network recurrent-loop analysis
    // -----------------------------------------------------------------------

    void FormNestedNetwork(const ComputationNodeBasePtr& rootNode);
    ComputationNodeBasePtr GetNestedNetwork(const ComputationNodeBasePtr& rootNode);

    // The methods below determine evaluation order, which is tricky in presence of recurrent loops.
    // TODO: Can this be moved to a separate class?
private:
    // This is part of the FormRecurrentLoops() process, and only called from there.
    void FormRecurrentLoops(const ComputationNodeBasePtr& rootNode);
    void DetermineSCCs(const ComputationNodeBasePtr& rootNode);
    void DetermineSCCsR(ComputationNodeBasePtr cur, std::list<ComputationNodeBasePtr>& sccStack, size_t& index, size_t& loopId);
    void DetermineLoopForwardOrderR(std::unordered_set<ComputationNodeBasePtr>& visited, std::unordered_set<ComputationNodeBasePtr>& recStack, std::list<ComputationNodeBasePtr>& nodesStack, ComputationNodeBasePtr cur);
    void GatherLoopNodesR(const ComputationNodeBasePtr& rootNode, std::unordered_set<ComputationNodeBasePtr>& visited, std::map<int, std::list<ComputationNodeBasePtr>>& recurrentResult, std::list<ComputationNodeBasePtr>& noRecurrentResult);
    void ReorderLoops(std::list<ComputationNodeBasePtr>& nodes, const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/, const std::list<ComputationNodeBasePtr>& /*noRecurrentNodes*/);

public:
    // -----------------------------------------------------------------------
    // evaluation: traversal
    // These three functions create and cache traversal orders of the network.
    // -----------------------------------------------------------------------

    // determine the required order in which nodes must be computed in order to compute 'rootNode'
    // If passed nullptr, this will traverse the entire net.
    // If passed non-null, it will take the global traveral in ITS order and sub-filter against root's dependents.
    void FormEvalOrder(const ComputationNodeBasePtr& rootNode)
    {
        if (m_evalOrders.find(rootNode) != m_evalOrders.end())
        {
            if (rootNode)
                fprintf(stderr, "FormEvalOrder: WARNING: Was called twice for %ls %ls operation.\n", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());
            else
                fprintf(stderr, "FormEvalOrder: WARNING: Was called twice.\n");
        }

        std::list<ComputationNodeBasePtr> evalOrder;
        if (!rootNode) // this creates the global one
        {
            evalOrder = ComputationNodeBase::EnumerateNodes(m_allRoots);
        }
        else // this creates a subset of the global eval order of all nodes that rootNode depends on
        {
            auto rawTraversalForRoot = ComputationNodeBase::EnumerateNodes({ rootNode }); // traverse to find the set (we ignore the order)
            set<ComputationNodeBasePtr> rawSet(rawTraversalForRoot.begin(), rawTraversalForRoot.end());
            for (const auto& node : GetEvalOrder(nullptr)) // iterate over global one and pull out everything that is included in the set for rootNode
            {
                if (rawSet.find(node) != rawSet.end())
                    evalOrder.push_back(node);
            }
        }
        m_evalOrders[rootNode] = evalOrder;
    }

    // replace an existing eval order with an updated one
    // This is meant to be used by FormRecurrentLoops().  TODO: Hopefully this can be not done anymore some day.
    void UpdateEvalOrder(const ComputationNodeBasePtr& rootNode, std::list<ComputationNodeBasePtr>& nodes)
    {
        GetEvalOrder(rootNode); // verify that there is already an entry for rootNode
        m_evalOrders[rootNode] = nodes;
    }

    bool EvalOrderExists(const ComputationNodeBasePtr& rootNode) const
    {
        return m_evalOrders.find(rootNode) != m_evalOrders.end();
    }

    // get depth-first traversal order
    // TODO: This is currently not immutable because it gets patched w.r.t. recurrent loops. Ideally we don't patch. Need to review and verify that it is sufficient.
    const std::list<ComputationNodeBasePtr>& GetEvalOrder(const ComputationNodeBasePtr& rootNode) const
    {
        auto iter = m_evalOrders.find(rootNode);
        if (iter == m_evalOrders.end())
        {
            LogicError("GetEvalOrder: Called without prior call to FormEvalOrder() for %ls %ls operation", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());
        }
        return iter->second;
    }

    // same as GetEvalOrder() where ordering is irrelevant
    const std::list<ComputationNodeBasePtr>& GetAllNodesForRoot(const ComputationNodeBasePtr& rootNode) const
    {
        return GetEvalOrder(rootNode);
    }

protected:
    class SEQTraversalFlowControlNode;

private:
    static std::shared_ptr<SEQTraversalFlowControlNode> FindInRecurrentLoops(const std::vector<std::shared_ptr<SEQTraversalFlowControlNode>>& recurrentInfo, const ComputationNodeBasePtr& node);

public:
    // -----------------------------------------------------------------------
    // MBLayouts
    // -----------------------------------------------------------------------

    // Note: this is also used to copy MBLayouts into our existing MBLayout instance, which is a somewhat questionable design.
    // BUGBUG (Issue #95): This function will conflict once we have multiple input layouts in the network.
    const MBLayoutPtr& GetMBLayoutPtrOfNetwork() { return m_pMBLayoutOfNetwork; }

    // determine the actual MB size from the feature nodes
    // This returns max number of columns over the feature nodes.
    // Note that if we have multiple slices, MB size != #frames.
    // BUGBUG: This will break once we have inconsistent layouts.
    // BUGBUG: The number computed here is completely off (it the layout has gaps
    // they will also be counted towards the actualMBSize)
    size_t DetermineActualMBSizeFromFeatures() const
    {
        size_t actualMBSize = 0;

        const auto& featureNodes = FeatureNodes(); // TODO: a getter; should be called GetFeatureNodes()
        for (auto& nodeIter : featureNodes)
            actualMBSize = max(actualMBSize, nodeIter->GetMBLayout()->GetNumCols());

        return actualMBSize;
    }

    // When external code (readers, namely) updates InputValue's m_value,
    // calling this function is required to make sure that any internal state gets updated correctly.
    // Only a change to the column dimension i sallowed
    void NotifyInputNodesFunctionValuesMBSizeModified()
    {
        for (auto& nodeIter : FeatureNodes())
            nodeIter->NotifyFunctionValuesMBSizeModified();
        for (auto& nodeIter : LabelNodes())
            nodeIter->NotifyFunctionValuesMBSizeModified();
    }

    // this counts the actual number of frames in a minibatch (not counting gaps in parallel sequences)
    // TODO: Instead of passing numAllSamples in here, we should determine it from the inputs in case of no layout. Or simply forbid this case.
    // BUGBUG (Issue #95): With variable-length sequences, this can no longer be a network method.
    size_t GetNumSamplesWithLabelOfNetwork(const size_t numAllSamples) const
    {
        if (m_pMBLayoutOfNetwork)
            return m_pMBLayoutOfNetwork->GetActualNumSamples();
        else
            return numAllSamples; // TODO: Return the actual number of samples, by inquiring our own input nodes; then eliminate the numAllSamples parameter.
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    // this function is only for use by NDL (deprecated)
    void InitLearnableParameters(const ComputationNodeBasePtr& node,
                                 const wchar_t* initString, // "uniform"|"gaussian"|"fixedValue"
                                 double initValue,          //  scale   | scale    | value
                                 unsigned long randomSeed = 0,
                                 bool initOnCPUOnly = false) const;
    // non-static version needed because it accesses m_randomSeedOffset
    // Legacy version that is for random only.
    void RandomInitLearnableParameters(const ComputationNodeBasePtr& node, const bool uniformInit, const unsigned long randomSeed, const double initValueScale, bool initOnCPUOnly = false) const;

    template <class ElemType>
    void InitLearnableParametersWithBilinearFill(const ComputationNodeBasePtr& node, size_t kernelWidth, size_t kernelHeight);

    template <typename N>
    static shared_ptr<N> AsNodePtr(const ComputationNodeBasePtr& inode)
    {
        return dynamic_pointer_cast<N>(inode);
    }
    template <typename N>
    static bool IsNodePtr(const ComputationNodeBasePtr& inode)
    {
        return AsNodePtr<N>(inode) != nullptr;
    }

    // -----------------------------------------------------------------------
    // network editing
    // -----------------------------------------------------------------------

    ComputationNodeBasePtr CopyNode(const ComputationNetwork& fromNet, const std::wstring fromName, std::wstring toName, const CopyNodeFlags flags);
    void CopySubTree(const ComputationNetwork& fromNet, const std::wstring fromName, std::wstring toNamePrefix, const CopyNodeFlags flags);
    void CopyInputs(const std::wstring fromName, std::wstring toName);
    void RenameNode(const std::wstring& nodeNameOrig, const std::wstring& nodeNameNew);
    void RenameNode(ComputationNodeBasePtr node, const std::wstring& newNodeName);
    void DeleteNode(const std::wstring& nodeName);
    void ReplaceNode(wstring nodeName, ComputationNodeBasePtr newNode);
    void InsertNode(wstring nodeName, ComputationNodeBasePtr newNode, const std::set<std::wstring>& newNodeTags);
    void ReplaceLeafNode(wstring oldNodeName, ComputationNodeBasePtr newNode);
    void ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodeBasePtr newNode);
    void AddFeatureNode(ComputationNodeBasePtr featureNode);
    //ComputationNodeBasePtr RemoveFeatureNode(ComputationNodeBasePtr featureNode);
    void SetLearnableNodesBelowLearningRateMultiplier(const float learningRateMultiplier, const ComputationNodeBasePtr& rootNode = nullptr);

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    bool NodeNameExists(const std::wstring& name) const
    {
        auto iter = m_nameToNodeMap.find(name);
        return (iter != m_nameToNodeMap.end());
    }

    ComputationNodeBasePtr GetNodeFromName(const std::wstring& name) const
    {
        auto iter = m_nameToNodeMap.find(name);
        if (iter == m_nameToNodeMap.end())
            RuntimeError("GetNodeFromName: Network has no node named '%ls'.", name.c_str());
        return iter->second;
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
            if (NodeNameExists(name))
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
    // environment properties
    // -----------------------------------------------------------------------

    ComputationEnvironment& Environment() const { return *m_environment; }

    // -----------------------------------------------------------------------
    // functions to pass on specific SGD options to nodes
    // -----------------------------------------------------------------------

    // TODO: Why are all these static, but then take a network as the first argument? --> make them class members
    template <class ElemType>
    static void SetDropoutRate(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double& prevDropoutRate);

    template <class ElemType>
    static void SetIRngUserSeed(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, size_t randSeedBase);
    
    template <class ElemType>
    static void SetBatchNormalizationTimeConstants(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, 
                                                   double normalizationTimeConstant, double& prevNormalizationTimeConstant,
                                                   double blendTimeConstant, double& prevBlendTimeConstant);

    template <class ElemType>
    static void SetSeqParam(ComputationNetworkPtr net,
                            const ComputationNodeBasePtr criterionNode,
                            const double& hsmoothingWeight, // TODO: Why are all these passed by reference?
                            const double& frameDropThresh,
                            const bool& doreferencealign,
                            const double& amf = 14.0f,
                            const double& lmf = 14.0f,
                            const double& wp = 0.0f,
                            const double& bMMIfactor = 0.0f,
                            const bool& sMBR = false);
    template<class ElemType>
    static void SetCTCParam(ComputationNetworkPtr net, 
                            const ComputationNodeBasePtr criterionNode, 
                            const ComputationNodeBasePtr evaluationNode, 
                            const size_t& blanknum = 1, const int &delayConstraint=-1);

    static void SetMaxTempMemSizeForCNN(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const size_t maxTempMemSizeInSamples);

    // -----------------------------------------------------------------------
    // node-group access
    // -----------------------------------------------------------------------

    // these two groups are determined from the network to be executed
    // They depend on the root node that is being evaluated.
    const std::list<ComputationNodeBasePtr>& InputNodes(const ComputationNodeBasePtr& rootNode /*, bool bNoBuild = false*/)
    {
        auto iter = m_inputValues.find(rootNode);
        if (iter == m_inputValues.end())
            LogicError("InputNodes() called for root %ls %ls operation for the set of inputs has not (yet?) been determined.", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());
        return iter->second;
    }

    const std::list<ComputationNodeBasePtr>& LearnableParameterNodes(const ComputationNodeBasePtr& rootNode)
    {
        auto iter = m_learnableParameters.find(rootNode);
        if (iter == m_learnableParameters.end())
            LogicError("LearnableParameterNodes() called for root %ls %ls operation for which the set of learnable parameters has not (yet?) been determined.", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());
        return iter->second;
    }

    inline const std::vector<ComputationNodeBasePtr>& CriterionNodesFrom(const wstring& criterionNodeName)
    {
        ComputationNodeBasePtr node = GetNodeFromName(criterionNodeName);
        if (node->HasMBLayout() || node->GetSampleLayout().GetNumElements() != 1)
            InvalidArgument("%ls %ls operation is not a valid training or eval criterion node.", node->NodeName().c_str(), node->OperationName().c_str());
        m_namedCriterionNodes[criterionNodeName] = std::vector<ComputationNodeBasePtr>{node};
        return m_namedCriterionNodes[criterionNodeName];
    }

    std::vector<ComputationNodeBasePtr> OutputNodesByName(const std::vector<std::wstring>& outputNodeNames) 
    {
        std::vector<ComputationNodeBasePtr> outputNodes;

        if (outputNodeNames.size() == 0)
        {
            if (OutputNodes().size() == 0)
                RuntimeError("There is no default output node specified in the network.");

            outputNodes = OutputNodes();
        }
        else
        {
            for (int i = 0; i < outputNodeNames.size(); i++)
                outputNodes.push_back(GetNodeFromName(outputNodeNames[i]));
        }

        return outputNodes;
    }

    // Collect all input nodes that outputNodes depend on.
    std::vector<ComputationNodeBasePtr> InputNodesForOutputs(const std::vector<std::wstring>& outputNodeNames)
    {
        // use set to remove duplicated items
        auto outputNodes = OutputNodesByName(outputNodeNames);

        std::set<ComputationNodeBasePtr> inputNodesMap;
        for (auto& onode : outputNodes)
        {
            for (auto& inode : InputNodes(onode))
                inputNodesMap.insert(inode);
        }

        std::vector<ComputationNodeBasePtr> inputNodes;
        for (auto& inode : inputNodesMap)
            inputNodes.push_back(inode);

        return inputNodes;
    }


    const std::vector<ComputationNodeBasePtr>& RootNodes()           const { return m_allRoots; }

    // these are specified as such by the user
    const std::vector<ComputationNodeBasePtr>& FeatureNodes()        const { return m_featureNodes   ; }
    const std::vector<ComputationNodeBasePtr>& LabelNodes()          const { return m_labelNodes     ; }
    const std::vector<ComputationNodeBasePtr>& FinalCriterionNodes() const { return m_criterionNodes ; }
    const std::vector<ComputationNodeBasePtr>& EvaluationNodes()     const { return m_evaluationNodes; }
    const std::vector<ComputationNodeBasePtr>& OutputNodes()         const { return m_outputNodes    ; }

private:
    // determine the node-group array by the group tag
    std::vector<ComputationNodeBasePtr>& GetNodeGroup(const std::wstring& groupTag)
    {
        if      (groupTag == L"feature"   ) return m_featureNodes;
        else if (groupTag == L"label"     ) return m_labelNodes;
        else if (groupTag == L"criterion" ) return m_criterionNodes;
        else if (groupTag == L"evaluation") return m_evaluationNodes;
        else if (groupTag == L"output"    ) return m_outputNodes;
        else InvalidArgument("Invalid group tag '%ls', must be one of 'feature', 'label', 'criterion', 'evaluation', 'output'.", groupTag.c_str());
    }

public:
    // add a node to a node group
    void AddToNodeGroup(const std::wstring& groupTag, const ComputationNodeBasePtr& node)
    {
        // determine the node group by its group tag string
        auto& nodeGroup = GetNodeGroup(groupTag);
        // if node is already in the list then we are done
        if (node->HasTag(groupTag))
        {
            for (const auto& groupNode : nodeGroup) // TODO: is there an STL algorithm?
                if (groupNode == node)
                    return;
            // we get here if the node has the tag but is not in the node group yet
        }
        // verify and update the node's tag
        node->SetTag(groupTag);
        // add to the node group
        nodeGroup.push_back(node);
    }

    // remove a node from its node group
    // Returns true if the node was there.
    bool RemoveFromNodeGroup(const std::wstring& groupTag, const ComputationNodeBasePtr& node)
    {
        bool wasActuallySet = node->ClearTag(groupTag);
        if (!wasActuallySet) // if node was not member of the group, we are done
            return false;
        auto& nodeGroup = GetNodeGroup(groupTag);
        for (auto iter = nodeGroup.begin(); iter != nodeGroup.end(); iter++)
        {
            if (*iter == node)
            {
                nodeGroup.erase(iter);
                return true;
            }
        }
        LogicError("RemoveFromNodeGroup: %ls %ls operation not found in its node group '%ls'.", node->NodeName().c_str(), node->OperationName().c_str(), groupTag.c_str());
    }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    size_t GetTotalNumberOfNodes() const
    {
        return m_nameToNodeMap.size();
    }

    std::vector<ComputationNodeBasePtr> GetAllNodes() const
    {
        std::vector<ComputationNodeBasePtr> nodes;
        for (const auto& iter : m_nameToNodeMap)
            nodes.push_back(iter.second);
        return nodes;
    }

    // determine parent map (this is needed in some editing steps)
    // Returns a map[node] -> set of parent nodes.
    std::map<ComputationNodeBasePtr, std::set<ComputationNodeBasePtr>> CreateParentsMap() const
    {
        std::map<ComputationNodeBasePtr, std::set<ComputationNodeBasePtr>> parents; // use a set because a node may have the same input multiple times, e.g. to compute x^2 as x.*x
        for (const auto& iter : m_nameToNodeMap)
        {
            const auto& node = iter.second;
            parents[node]; // make sure there is an entry for every parent
            for (const auto& child : node->GetInputs())
                parents[child].insert(node);
        }
        return parents;
    }

    // Return set of immediate output (parent) nodes for given input (child) node
    // TODO: there should be a map from output nodes to inputs, so that this operation doesn't take square time
    std::vector<ComputationNodeBasePtr> GetParentNodes(const std::wstring& inputNodeName)
    {
        std::set<ComputationNodeBasePtr> outputNodes;
        for (const auto& iter : m_nameToNodeMap)
        {
            const auto& node = iter.second;

            //Iterate over inputs of this node
            for (const auto& inputNode : node->GetInputs())
            {
                if (inputNode->GetName() == inputNodeName)
                {
                    outputNodes.insert(node);
                }
            }
        }

        return std::vector<ComputationNodeBasePtr>(outputNodes.begin(), outputNodes.end());
    }

    std::list<ComputationNodeBasePtr> GetNodesWhere(std::function<bool(const ComputationNodeBasePtr&)>& predicate, const ComputationNodeBasePtr& rootNode = nullptr) const
    {
        std::list<ComputationNodeBasePtr> filteredNodes;

        // find nodes from all available nodes
        // TODO: This distinction should not be necessary anymore. Calling GetEvalOrder(nullptr) will have the same effect.
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (predicate(node))
                    filteredNodes.push_back(node);
            }
        }
        else
        {
            // for calculating a specific node
            for (const auto& node : GetEvalOrder(rootNode)) // TODO: verify that no use of this requires the actual eval order, then change to GetAllNodesForRoot()
            {
                if (predicate(node))
                    filteredNodes.push_back(node);
            }
        }

        return filteredNodes;
    }

    std::list<ComputationNodeBasePtr> GetNodesWithType(const wstring typeName, const ComputationNodeBasePtr& rootNode = nullptr) const
    {
        std::function<bool(const ComputationNodeBasePtr&)> predicate = [typeName](const ComputationNodeBasePtr& node) { return node->OperationName() == typeName; };
        return GetNodesWhere(predicate, rootNode);
    }

    // Get the eval nodes with names
    // if evalNodeNames are not specified, return all the default evalnodes and training criterion nodes.
    std::vector<ComputationNodeBasePtr> GetEvalNodesWithName(const std::vector<wstring> evalNodeNames)
    {
        // determine nodes to evaluate
        std::vector<ComputationNodeBasePtr> evalNodes;

        set<ComputationNodeBasePtr> criteriaLogged; // (keeps track ot duplicates to avoid we don't double-log critera)
        if (evalNodeNames.size() == 0)
        {
            fprintf(stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
            if (EvaluationNodes().empty() && FinalCriterionNodes().empty())
                InvalidArgument("There is no default evaluation node or training criterion specified in the network.");

            for (const auto& node : EvaluationNodes())
                if (criteriaLogged.insert(node).second)
                    evalNodes.push_back(node);

            for (const auto& node : FinalCriterionNodes())
                if (criteriaLogged.insert(node).second)
                    evalNodes.push_back(node);
        }
        else
        {
            for (int i = 0; i < evalNodeNames.size(); i++)
            {
                const auto& node = GetNodeFromName(evalNodeNames[i]);
                if (!criteriaLogged.insert(node).second)
                    continue;
                if (node->GetSampleLayout().GetNumElements() != 1)
                    InvalidArgument("Criterion nodes to evaluate must have dimension 1x1.");
                evalNodes.push_back(node);
            }
        }

        return evalNodes;
    }

public:
    // return list of nodes that require precomputation and not precomputed yet
    std::list<ComputationNodeBasePtr> GetNodesRequiringPreComputation(const ComputationNodeBasePtr& rootNode = nullptr, bool checkComputed = true);

    // -----------------------------------------------------------------------
    // unit testing
    // -----------------------------------------------------------------------

    bool UnitTest(bool allowFragment = false);
    bool UnitTest(const ComputationNodeBasePtr& rootNode);

    // -----------------------------------------------------------------------
    // specialized operations
    // -----------------------------------------------------------------------

    template <class ElemType>
    void PerformSVDecomposition(const map<wstring, float>& SVDConfig, size_t AlignedSize);

    template <class ElemType>
    void SaveToDbnFile(ComputationNetworkPtr net, const std::wstring& fileName) const;

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

protected:

    // Copy constructor, should never be called.
#pragma warning(push)
#pragma warning(disable : 4702) // this function is flagged but unclear why
    ComputationNetwork(const ComputationNetwork& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork(const ComputationNetwork& deepCopyFrom)' should never be called.");
    }
#pragma warning(pop)

    // Assignment operator, should never be called.
    ComputationNetwork& operator=(const ComputationNetwork& /*deepCopyFrom*/)
    {
        // TODO: can we just define it as private without implementation?
        LogicError("'ComputationNetwork& operator=(const ComputationNetwork& deepCopyFrom)' should never be called.");
    }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

public:
    // TODO: move these to ComputationNetworkBuilder.cpp

    // add a node to m_nameToNodeMap[], which is our node holder
    // This only adds the node to the network's node set, without considering linkage.
    // Duplicate node names are rejected.
    ComputationNodeBasePtr AddNodeToNet(const ComputationNodeBasePtr& node)
    {
        auto result = m_nameToNodeMap.insert(make_pair(node->NodeName(), node));
        if (!result.second)
            RuntimeError("AddNodeToNet: Duplicated name for %ls %ls operation.", node->NodeName().c_str(), node->OperationName().c_str());
        node->SetEnvironment(m_environment);
        return node; // allows e.g. return AddNodeToNet(New...);
    }
    // TODO: not very nice--need to fix way more outside to get this right
    template <class N>
    shared_ptr<N> AddNodeToNetWithElemType(const shared_ptr<N> node)
    {
        return dynamic_pointer_cast<N>(AddNodeToNet(node));
    }

    template <class N>
    shared_ptr<N> AddNodeToNetAndAttachInputs(const shared_ptr<N> nodePtr, const std::vector<ComputationNodeBasePtr>& inputs)
    {
        nodePtr->AttachInputs(inputs);
        return AddNodeToNetWithElemType(nodePtr);
        // return nodePtr; // allows e.g. return AddNodeToNetAndAttachInputs(New..., inputs);
    }

    // add a node to the network unless it's already there
    // Returns false if the node was already there.
    // If the network already contains a different node with the same name,
    //  - then the function will fail
    //  - unless 'makeUniqueName=true', in which case it will patch the node's name to a unique name. 
    bool AddNodeToNetIfNotYet(const ComputationNodeBasePtr& node, bool makeUniqueName = false)
    {
        auto result = m_nameToNodeMap.insert(make_pair(node->NodeName(), node));
        // if there's already one under this name, it better be node
        // unless user requested 'makeUniqueName', then we will modify the name
        while (!result.second/*if already there*/ && result.first->second != node)
        {
            if (!makeUniqueName || node->NodeName().find_first_of(L".[]") == wstring::npos)
                RuntimeError("AddNodeToNetIfNotYet: Duplicated name for %ls %ls operation (%d vs. %d).", node->NodeName().c_str(), node->OperationName().c_str(), (int)node->m_uniqueNumericId, (int)result.first->second->m_uniqueNumericId);
            node->SetName(L"_" + node->NodeName());
            result = m_nameToNodeMap.insert(make_pair(node->NodeName(), node));
        }
        node->SetEnvironment(m_environment); // (note: redundant if already part of the network)
        return result.second;
    }

    // remove a node from the network's node set
    // This does NOT update any links referencing it, or node groups.
    // TODO: We should verify that indeed this node is not referenced by other nodes or node groups,
    //       nor that this node references any node inside the network.
    ComputationNodeBasePtr RemoveNodeFromNet(const ComputationNodeBasePtr& node)
    {
        node->SetEnvironment(nullptr);
        m_nameToNodeMap.erase(node->NodeName());
        return node;
    }
public:
    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // zeroes out all gradients except the root itself (since its gradient is set from outside rather than propagated down)
    // (Note that inside the nodes this only really sets a flag to do it later when needed, but that's not our concern.)
    void ZeroInputGradients(const ComputationNodeBasePtr& rootNode)
    {
        for (auto& node : GetAllNodesForRoot(rootNode))
            node->ZeroGradientsOfInputs();
    }

private:
    bool IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr);
    void PrintComputationTree(const ComputationNodeBasePtr& rootNode, const bool forwardCompute, const bool printMatrices = false);

public:
    // -----------------------------------------------------------------------
    // diagnostics
    // -----------------------------------------------------------------------

    void SetTraceLevel(int traceLevel)
    {
        m_environment->traceLevel = traceLevel;
    }
    int TraceLevel() const { return m_environment->traceLevel; }

    // call EnableNodeTracing() on the given nodes for real, category, and sparse printing
    void EnableNodeTracing(const std::vector<std::wstring>& traceNodeNamesReal,
                           const std::vector<std::wstring>& traceNodeNamesCategory,
                           const std::vector<std::wstring>& traceNodeNamesSparse)
    {
        for (const auto& name : traceNodeNamesReal)
            if (NodeNameExists(name))
                GetNodeFromName(name)->EnableNodeTracing(/*asReal=*/true,  /*asCategoryLabel=*/false, /*asSparse=*/false);
            else
                fprintf(stderr, "EnableNodeTracing: No node named '%ls'; skipping\n", name.c_str());
        for (const auto& name : traceNodeNamesCategory)
            if (NodeNameExists(name))
                GetNodeFromName(name)->EnableNodeTracing(/*asReal=*/false, /*asCategoryLabel=*/true,  /*asSparse=*/false);
            else
                fprintf(stderr, "EnableNodeTracing: No node named '%ls'; skipping\n", name.c_str());
        for (const auto& name : traceNodeNamesSparse)
            if (NodeNameExists(name))
                GetNodeFromName(name)->EnableNodeTracing(/*asReal=*/false, /*asCategoryLabel=*/false, /*asSparse=*/true);
            else
                fprintf(stderr, "EnableNodeTracing: No node named '%ls'; skipping\n", name.c_str());
    }

    // if node name is not found, dump all nodes
    // otherwise dump just that node
    // This function is called from MEL, i.e. must be prepared to operate on an uncompiled network (only m_nameToNodeMap is valid).
    void DumpNodeInfoToFile(const std::wstring& nodeName, const bool printValues, const bool printMetadata, const std::wstring outputFile, const std::wstring& nodeNameInRegEx = L"")
    {
        if (nodeNameInRegEx.empty())
        {
            if (NodeNameExists(nodeName))
            {
                File fstream(outputFile,
                             FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

                const ComputationNodeBasePtr& nodePtr = GetNodeFromName(nodeName);
                nodePtr->DumpNodeInfo(printValues, printMetadata, fstream);
            }
            else // node name is not found, dump all nodes
            {
                fprintf(stderr, "Warning: node name '%ls' does not exist in the network. dumping all nodes instead.\n",
                        nodeName.c_str());
                DumpAllNodesToFile(printValues, printMetadata, outputFile);
            }
        }
        else
        {
            std::wregex NameRegEx(nodeNameInRegEx);
            std::vector<ComputationNodeBasePtr> NodeList;
            std::vector<wstring> NameList;
            for (auto m : m_nameToNodeMap)
            {
                if (regex_match(m.first, NameRegEx))
                {
                    NodeList.push_back(m.second);
                    NameList.push_back(m.first);
                }
            }
            fprintf(stderr, "DumpNodeInfo: %d nodes matching RegEx(%ls): \n", (int) NameList.size(), nodeNameInRegEx.c_str());
            for (auto x : NameList)
            {
                fprintf(stderr, "\t%ls\n", x.c_str());
            }
            fprintf(stderr, "DumpNodeInfo: dumping node info (%s printing values) to %ls\n", printValues ? "with" : "without", outputFile.c_str());
            DumpNodeInfoToFile(NodeList, printValues, printMetadata, outputFile);
        }
    }

    // dump all nodes in the network to file
    void DumpAllNodesToFile(const bool printValues,
                            const bool printMetadata,
                            const std::wstring outputFile)
    {
        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = nodeIter->second;
            nodePtr->DumpNodeInfo(printValues, printMetadata, fstream);
        }
    }

    // this one is called from MEL and from DumpNodeInfoToFile() above
    void DumpNodeInfoToFile(const vector<ComputationNodeBasePtr>& nodes,
                            const bool printValues,
                            const bool printMetadata,
                            const std::wstring outputFile)
    {
        File fstream(outputFile,
                     FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = *nodeIter;
            nodePtr->DumpNodeInfo(printValues, printMetadata, fstream);
        }
    }

    // -----------------------------------------------------------------------
    // topological plot [1/13/2015 erw] plot network topology using dot language
    // -----------------------------------------------------------------------

private:
    wstring FormSpecialNodes(wstring style, std::vector<ComputationNodeBasePtr>& specialNodes);
    typedef std::pair<ComputationNodeBasePtr, ComputationNodeBasePtr> ComputationArc;

public:
    void DescribeNetworkUsingDot(std::list<ComputationArc>& arcs, std::wstring outFile);
    void PlotNetworkTopology(const std::wstring& outputFile);

    // -----------------------------------------------------------------------
    // scripting integration
    // -----------------------------------------------------------------------

    // pretend to be a ConfigRecord
    void /*CustomConfigRecord::*/ LazyCreateConfigMember(const wstring& id) const override;
    vector<wstring> /*IConfigRecord::*/ GetMemberIds() const override;

    // create a somewhat readable representation, aimed at diagnostics/debugging
    wstring /*HasToString::*/ ToString() const
    {
        wstring args;
        for (auto& iter : m_nameToNodeMap)
        {
            const auto node = iter.second;
            if (!args.empty())
                args.append(L"\n");
            args.append(node->ToString());
        }
        return TypeId<decltype(*this)>() + L" " + NestString(args, L'[', true, ']');
    }

protected:
    // FlowControlNodes for internal use by this class:

    // -----------------------------------------------------------------------
    // SEQTraversalFlowControlNode -- FlowControlNode to traverse a (sub-)network time step by time step
    //
    // This is to implement recurrent loops. All nodes inside a loop are listed
    // inside this node. This node's ForwardProp() function will execute
    // them inside a loop over all time steps of the recurrence.
    // For every time step, the entire chain of nodes is called, with the time index
    // passed as a FrameRange object.
    // -----------------------------------------------------------------------

    class SEQTraversalFlowControlNode : public FlowControlNode
    {
    public: // m_nestedNodes needed public by ComputationNetwork::FindInRecurrentLoops(), which really should be part of SEQTraversalFlowControlNode
        typedef FlowControlNode Base;
        using Base::m_nestedNodes;

    public:
        virtual const std::wstring OperationName() const override
        {
            return L"SEQTraversalFlowControlNode";
        }
        virtual void BeginForwardProp() override;
        virtual void ForwardProp(const FrameRange&) override;
        virtual void EndForwardProp() override;
        virtual void BeginBackprop() override;
        virtual void BackpropTo(const size_t inputIndex, const FrameRange&) override
        {
            NOT_IMPLEMENTED;
        }
        virtual void EndBackprop() override;
        virtual void Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) override;
        virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool);
        virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool);
        virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool);
        virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool);
        virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool);
        virtual bool IsOutOfDateWrtInputs() const override;

    public:
        ComputationNodeBasePtr m_sourceNode; // one of the nodes of the loop   --TODO: What is the special meaning of this node? It seems to always be a delay node.
        int m_loopId;                        // unique loop id, index in m_allSEQNodes array
        int m_steppingDirection;             // +1 if left to right (t=0..T-1), -1 if rightt to left (t=T-1..0)

        SEQTraversalFlowControlNode(int loopId, ComputationNodeBasePtr cur)
            : m_loopId(loopId),
              m_sourceNode(cur)
        {
            SetNodeName(L"Loop_" + m_sourceNode->NodeName());
        }
    };

    // -----------------------------------------------------------------------
    // PARTraversalFlowControlNode -- FlowControlNode that traverses a (sub-)network
    //
    // This node contains a list of nodes in a (sub-)network. This node's
    // ForwardProp() method will execute all those nodes once in PAR mode,
    // that is, by passing a FrameRange object that represents to operate
    // on all frames in the node simultaneously.
    //
    // The outermost network level is also represented by this node for execution.
    // -----------------------------------------------------------------------

    class PARTraversalFlowControlNode : public FlowControlNode
    {
        typedef FlowControlNode Base;
        using Base::m_nestedNodes;

    public:
        virtual const std::wstring OperationName() const override
        {
            return L"PARTraversalFlowControlNode";
        }
        virtual void BeginForwardProp() override
        {
        }
        virtual void ForwardProp(const FrameRange&) override;
        virtual void EndForwardProp() override
        {
        }
        virtual void BeginBackprop() override
        {
        }
        virtual void BackpropTo(const size_t inputIndex, const FrameRange&) override
        {
            NOT_IMPLEMENTED;
        } // ugh, call Backprop() instead
        virtual void EndBackprop() override
        {
        }
        virtual void Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) override;
        virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool);
        virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool);
        virtual void AllocateGradientMatricesForInputs(MatrixPool& matrixPool);
        virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool);
        virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool);

    public:
        // this special constructor constructs the top-level network node
        // There is currently no other constructor for inner nested PAR-traversed sub-networks, but there will be.
        PARTraversalFlowControlNode(const std::vector<shared_ptr<SEQTraversalFlowControlNode>>& recurrentInfo, const std::list<ComputationNodeBasePtr>& allNodes);
        // Base::m_nestedNodes contains all top-level nodes, in evaluation order
    };

public:
    // -----------------------------------------------------------------------
    // data members
    // -----------------------------------------------------------------------

    unsigned long GetRandomSeedOffset() const
    {
        return m_randomSeedOffset;
    }
    void SetRandomSeedOffset(unsigned long value)
    {
        m_randomSeedOffset = value;
    }

private:
    DEVICEID_TYPE m_deviceId; // TODO: is this shared by all nodes?
    unsigned long m_randomSeedOffset;

    // main node holder
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare> m_nameToNodeMap; // [name] -> node; this is the main container that holds this networks' nodes

    // node groups
    // These are specified by the user by means of tags or explicitly listing the node groups.
    // TODO: Are these meant to be disjoint?
    std::vector<ComputationNodeBasePtr> m_featureNodes;    // tag="feature"
    std::vector<ComputationNodeBasePtr> m_labelNodes;      // tag="label"
    std::vector<ComputationNodeBasePtr> m_criterionNodes;  // tag="criterion"
    std::vector<ComputationNodeBasePtr> m_evaluationNodes; // tag="evaluation"
    std::vector<ComputationNodeBasePtr> m_outputNodes;     // tag="output"
    vector<std::vector<ComputationNodeBasePtr>*> GetAllNodeGroups() // get all groups to allow to iterate over all of them ...continue
    {
        return vector<std::vector<ComputationNodeBasePtr>*>{&m_featureNodes, &m_labelNodes, &m_criterionNodes, &m_evaluationNodes, &m_outputNodes};
    }

    // used for sentence boundary information passed from reader to reset RNN state
    // specify how the minibatch is packed for each sample
    // BUGBUG (Issue #95): With variable-length inconsistent layouts, this can no longer be a network property.
    MBLayoutPtr m_pMBLayoutOfNetwork; // note that this must be installed before doing anything that needs it (default leaves a nullptr)

    // environment information that nodes may want to inquire, e.g. to know whether we are training
    ComputationEnvironmentPtr m_environment;

    std::map<std::wstring, std::vector<ComputationNodeBasePtr>> m_namedCriterionNodes;

private:
    // -----------------------------------------------------------------------
    // the following members are all result of post-processing by CompileNetwork()
    // -----------------------------------------------------------------------

    // list of all roots in this network
    // A root is a node that can run as a target of ForwardProp(). See DetermineSetOfAllRoots().
    std::vector<ComputationNodeBasePtr> m_allRoots;

    std::vector<std::shared_ptr<SEQTraversalFlowControlNode>> m_allSEQNodes; // [loopId] cached set of SEQTraversalFlowControlNodes to allow sharing and idempotence of FormRecurrentLoops()

    // cache for evaluation ordering:
    bool m_isCompiled; // CompileNetwork has been called
    bool m_areMatricesAllocated; // AllocateAllMatrices has been called

    // cached network iterations
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_evalOrders; // [out node] flat depth-first traversal starting from out node
    std::map<const ComputationNodeBasePtr, ComputationNodeBasePtr> m_nestedNetworks;        // [out node] network rewritten as recursive traveral, potentially optimized; execution plan

    // cached quick-access list for inputs and parameters
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_inputValues;         // [out node] -> all input nodes feeding into out node
    std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>> m_learnableParameters; // [out node] -> all parameter nodes feeding into out node

private:
    // pool for matrices that can be shared across nodes
    // TODO: does this apply to anything else besides temporary node-internal intermediate results? What, for example?
    MatrixPool m_matrixPool;
};
typedef ComputationNetwork::ComputationNetworkPtr ComputationNetworkPtr;

// helper that returns 'float' or 'double' depending on ElemType
template <typename ElemType> static inline const wchar_t* ElemTypeName();
template <> /*static*/ inline const wchar_t* ElemTypeName<float>()  { return L"float"; }
template <> /*static*/ inline const wchar_t* ElemTypeName<double>() { return L"double"; }

// The following emits the class and enables the BaseMatrix<double> to be available (used by EvalDll)
// The corresponding Matrix<float> is emitted in the SetDeviceId function above.
template class Matrix<double>;

// TODOs:
//  - automatic inference of time window w.r.t. delay nodes (and related nodes such as a temporal pooling)
//  - have overrides of RuntimeError etc. in ComputationNode, which prepend the error string with the node name and operation

} } }
