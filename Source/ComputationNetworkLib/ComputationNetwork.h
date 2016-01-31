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
#include <unordered_map>

namespace Microsoft { namespace MSR { namespace CNTK {

// ===========================================================================
// ComputationNetwork -- computation graph and operations
// ===========================================================================

class ComputationNetwork : public ScriptableObjects::Object, public ScriptableObjects::HasToString, public ScriptableObjects::IConfigRecord
{
public:
    typedef shared_ptr<ComputationNetwork> ComputationNetworkPtr;

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    ComputationNetwork()
        : m_randomSeedOffset(0),
          m_isCompiled(false),
          m_pMBLayout(make_shared<MBLayout>())
    {
    }
    ComputationNetwork(DEVICEID_TYPE deviceId)
        : ComputationNetwork()
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
        if (deviceId == AUTOPLACEMATRIX)
            deviceId = Matrix<float>::GetBestGPUDeviceId();
        m_deviceId = deviceId;
        m_deviceId = EnforceOneGPUOnly(m_deviceId); // see EnforceOneGPUOnly() for comment on what this is
    }

    DEVICEID_TYPE GetDeviceId() const { return m_deviceId; }

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
    template <class ElemType>
    void Read(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary,
              const bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr);
    template <class ElemType>
    void Load(const std::wstring& fileName, const FileOptions fileFormat = FileOptions::fileOptionsBinary,
              const bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr)
    {
        Read<ElemType>(fileName, fileFormat, bAllowNoCriterionNode, anotherNetwork);
        // perform all further post-processing, caching, etc.
        CompileNetwork();
    }

    // static helper to instantiate a network from a file
    template <class ElemType>
    static ComputationNetworkPtr CreateFromFile(DEVICEID_TYPE deviceId, const std::wstring& fileName,
                                                const FileOptions fileFormat = FileOptions::fileOptionsBinary,
                                                const bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr)
    {
        auto net = make_shared<ComputationNetwork>(deviceId);
        net->Load<ElemType>(fileName, FileOptions::fileOptionsBinary, bAllowNoCriterionNode, anotherNetwork);
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

    // void ValidateNetwork(bool allowFragment = false, const bool bAllowNoCriterion = false);
    // prepares the network for computation
    // void BuildAndValidateSubNetwork(const ComputationNodeBasePtr rootNode);
private:
    void ValidateNodes(list<ComputationNodeBasePtr> nodes, bool isFinalValidationPass, size_t& todo);
    void ValidateSubNetwork(const ComputationNodeBasePtr& rootNode);
    void MarkValueNonSharableNodes();

private:
    void DetermineSetOfAllRoots();
    void CollectInputAndLearnableParameters(const ComputationNodeBasePtr& rootNode);
    bool IsCompiled() const
    {
        return m_isCompiled;
    }
    void VerifyIsCompiled(const char* where) const;
    // bool BuiltAndValidatedSubNetwork(const ComputationNodeBasePtr & rootNode);
public:
    void AllocateAllMatrices(const std::vector<ComputationNodeBasePtr>& evalRootNodes, const std::vector<ComputationNodeBasePtr>& outValueRootNodes, ComputationNodeBasePtr trainRootNode);

private:
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
    void DetermineLoopForwardOrder(std::unordered_set<ComputationNodeBasePtr>& visited, std::unordered_set<ComputationNodeBasePtr>& recStack, std::list<ComputationNodeBasePtr>& nodesStack, ComputationNodeBasePtr cur);
    void GatherLoopNodesR(const ComputationNodeBasePtr& rootNode, std::unordered_set<ComputationNodeBasePtr>& visited, std::map<int, std::list<ComputationNodeBasePtr>>& recurrentResult, std::list<ComputationNodeBasePtr>& noRecurrentResult);
    void ReorderLoops(std::list<ComputationNodeBasePtr>& nodes, const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/, const std::list<ComputationNodeBasePtr>& /*noRecurrentNodes*/);

public:
    // -----------------------------------------------------------------------
    // evaluation: traversal
    // These three functions create and cache traversal orders of the network.
    // -----------------------------------------------------------------------

    // determine the required order in which nodes must be computed in order to compute 'rootNode'
    // skipPairNetwork == true is only used when called from FormRecurrentLoops()
    void FormEvalOrder(const ComputationNodeBasePtr& rootNode)
    {
        if (m_evalOrders.find(rootNode) != m_evalOrders.end())
        {
            if (rootNode)
                fprintf(stderr, "FormEvalOrder: WARNING: Was called twice for %ls %ls operation.\n", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());
            else
                fprintf(stderr, "FormEvalOrder: WARNING: Was called twice.\n");
        }

        if (rootNode)
            m_evalOrders[rootNode] = rootNode->EnumerateNodes();
        else
            m_evalOrders[rootNode] = ComputationNodeBase::EnumerateNodes(m_allRoots);
    }

    // replace an existing eval order with an updated one
    // This is meant to be used by FormRecurrentLoops().  TODO: Hopefully this can be not done anymore some day.
    void UpdateEvalOrder(const ComputationNodeBasePtr& rootNode, std::list<ComputationNodeBasePtr>& nodes)
    {
        GetEvalOrder(rootNode); // verify that there is already an entry for rootNode
        m_evalOrders[rootNode] = nodes;
    }

    std::list<ComputationNodeBasePtr>& GetEvalOrder(const ComputationNodeBasePtr& rootNode)
    {
        if (m_evalOrders.find(rootNode) == m_evalOrders.end())
            LogicError("GetEvalOrder: Called without prior call to FormEvalOrder() for %ls %ls operation", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());

        return m_evalOrders[rootNode];
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
    const MBLayoutPtr& GetMBLayoutPtr() { return m_pMBLayout; }
    size_t GetNumParallelSequences() const { return m_pMBLayout->GetNumParallelSequences(); }

    // determine the actual MB size from the feature nodes
    // This returns max number of columns over the feature nodes.
    // Note that if we have multiple slices, MB size != #frames.
    // BUGBUG: This will break once we have inconsistent layouts.
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
    size_t GetNumSamplesWithLabel(const size_t numAllSamples) const
    {
        if (m_pMBLayout)
            return m_pMBLayout->GetActualNumSamples();
        else
            return numAllSamples; // TODO: Return the actual number of samples, by inquiring our own input nodes; then eliminate the numAllSamples parameter.
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    // non-static version needed because it accesses m_randomSeedOffset
    // Excessively used by SimpleNetworkBuilder, but always after CreateLearnableParameter(), so we should really absorb it there
    template <class ElemType>
    void InitLearnableParameters(const ComputationNodeBasePtr& node,
                                 const bool uniformInit,
                                 const unsigned long randomSeed,
                                 const ElemType initValueScale,
                                 bool initOnCPUOnly = false);

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
    void ChangeNode(wstring nodeName, ComputationNodeBasePtr newNode);
    void ReplaceLeafNode(wstring oldNodeName, ComputationNodeBasePtr newNode);
    void ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodeBasePtr newNode);
    void AddFeatureNode(ComputationNodeBasePtr featureNode);
    void RemoveFeatureNode(ComputationNodeBasePtr featureNode);
    void SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr& rootNode = nullptr);
    void SetBatchNormlizationNodesBelowEvalMode(const bool evalMode, const ComputationNodeBasePtr& rootNode = nullptr);

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    bool NodeNameExists(const std::wstring& name) const
    {
        auto iter = m_nameToNodeMap.find(name);
        return (iter != m_nameToNodeMap.end());
    }

    ComputationNodeBasePtr GetNodeFromName(const std::wstring& name, ComputationNetwork* anotherNetwork = nullptr, bool bPanic = true) const
    {
        auto iter = m_nameToNodeMap.find(name);
        if (iter != m_nameToNodeMap.end())
        {
            // found
            return iter->second;
        }

        if (anotherNetwork != nullptr)
            return anotherNetwork->GetNodeFromName(name);

        if (bPanic)
            RuntimeError("GetNodeFromName: Node name %ls does not exist.", name.c_str());
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
    // functions to pass on specific SGD options to nodes
    // -----------------------------------------------------------------------

    template <class ElemType>
    static void SetDropoutRate(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double& prevDropoutRate, unsigned long& dropOutSeed);

    template <class ElemType>
    static void SetSeqParam(ComputationNetworkPtr net,
                            const ComputationNodeBasePtr criterionNode,
                            const double& hsmoothingWeight,
                            const double& frameDropThresh,
                            const bool& doreferencealign,
                            const double& amf = 14.0f,
                            const double& lmf = 14.0f,
                            const double& wp = 0.0f,
                            const double& bMMIfactor = 0.0f,
                            const bool& sMBR = false);
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

    // these are specified as such by the user
    inline std::vector<ComputationNodeBasePtr>& FeatureNodes()
    {
        return m_features;
    }
    inline const std::vector<ComputationNodeBasePtr>& FeatureNodes() const
    {
        return m_features;
    }
    inline std::vector<ComputationNodeBasePtr>& LabelNodes()
    {
        return m_labels;
    }
    inline std::vector<ComputationNodeBasePtr>& FinalCriterionNodes()
    {
        return m_finalCriteria;
    }

    inline std::vector<ComputationNodeBasePtr> CriterionNodesFrom(const wstring& criterionNodeName)
    {
        ComputationNodeBasePtr node = GetNodeFromName(criterionNodeName);
        ValidateSubNetwork(node);
        if (node->HasMBLayout() || node->GetSampleLayout().GetNumElements() != 1)
            InvalidArgument("%ls %ls operation is not a valid training or eval criterion node.", node->NodeName().c_str(), node->OperationName().c_str());
        return std::vector<ComputationNodeBasePtr>{node};
    }

    inline std::vector<ComputationNodeBasePtr>& EvaluationNodes()
    {
        return m_evalNodes;
    }
    inline std::vector<ComputationNodeBasePtr>& OutputNodes()
    {
        return m_outputNodes;
    }
    inline std::vector<ComputationNodeBasePtr>& PairNodes()
    {
        return m_pairNodes;
    }

    // -----------------------------------------------------------------------
    // node access
    // -----------------------------------------------------------------------

    size_t GetTotalNumberOfNodes() const
    {
        return m_nameToNodeMap.size();
    }

    // TODO: could be a dup
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare>& GetNameToNodeMap() // specially for ExperimentalNetworkBuilder; don't use this otherwise
    {
        return m_nameToNodeMap;
    }

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

        // find nodes from all available nodes
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
            // for calculating a specific node
            const std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);
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
    template <class N>
    void GetNodesRequiringX(std::list<ComputationNodeBasePtr>& nodesRequirePreComputation, const ComputationNodeBasePtr& rootNode, bool checkComputed);

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
    // TODO: move these close to where they are used

    // add a node to m_nameToNodeMap[], which is our node holder
    // Duplicate node names are rejected.
    ComputationNodeBasePtr AddNodeToNet(const ComputationNodeBasePtr& nodePtr)
    {
        // found
        // TODO: use .insert() and test result.second == false means not inserted since already exists
        if (m_nameToNodeMap.find(nodePtr->NodeName()) != m_nameToNodeMap.end())
            RuntimeError("Duplicated computation node name.");

        m_nameToNodeMap[nodePtr->NodeName()] = nodePtr;
        return nodePtr; // allows e.g. return AddNodeToNet(New...);
    }
    // TODO: not very nice--need to fix way more outside to get this right
    template <class N>
    shared_ptr<N> AddNodeToNetWithElemType(const shared_ptr<N> nodePtr)
    {
        return dynamic_pointer_cast<N>(AddNodeToNet(nodePtr));
    }

    template <class N, class... _Types>
    shared_ptr<N> AddNodeToNetAndAttachInputs(const shared_ptr<N> nodePtr, _Types&&... _Args)
    {
        nodePtr->AttachInputs(std::forward<_Types>(_Args)...);
        return AddNodeToNetWithElemType(nodePtr);
        // return nodePtr; // allows e.g. return AddNodeToNetAndAttachInputs(New..., inputs);
    }

public:
    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // zeroes out all gradients except the root itself
    // TODO: why not the root?
    // (Note that inside the nodes this only really sets a flag to do it later when needed, but that's not our concern.)
    void ZeroGradients(const ComputationNodeBasePtr& rootNode)
    {
        for (auto& node : GetEvalOrder(rootNode)) // note: any order will do
            node->ZeroGradientsOfInputs();
    }

private:
    bool IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr);
    void PrintComputationTree(const ComputationNodeBasePtr& rootNode, const bool forwardCompute, const bool printMatrices = false);

public:
    // -----------------------------------------------------------------------
    // diagnostics
    // -----------------------------------------------------------------------

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
                fprintf(stderr, "Warning: node name %ls does not exist in the network. dumping all nodes.\n",
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
    void PlotNetworkTopology(const std::wstring outputFile);

    // -----------------------------------------------------------------------
    // scripting integration
    // -----------------------------------------------------------------------

    // pretend to be a ConfigRecord
    const ScriptableObjects::ConfigValuePtr& /*IConfigRecord::*/ operator[](const wstring& id) const override; // e.g. confRec[L"message"]
    const ScriptableObjects::ConfigValuePtr* /*IConfigRecord::*/ Find(const wstring& id) const override;       // returns nullptr if not found
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
        virtual bool IsOutputOlderThanInputs() const override;

    public:
        // std::vector<ComputationNodeBasePtr> m_nestedNodes;               // all nodes involved in this loop, in evaluation order
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

    unsigned long GetRandomSeedOffset()
    {
        return m_randomSeedOffset;
    }
    void SetRandomSeedOffset(unsigned long value)
    {
        m_randomSeedOffset = value;
    }

protected:
    DEVICEID_TYPE m_deviceId; // TODO: is this shared by all nodes?
    unsigned long m_randomSeedOffset;

    // main node holder
    std::map<const std::wstring, ComputationNodeBasePtr, nocase_compare> m_nameToNodeMap; // [name] -> node; this is the main container that holds this networks' nodes

    // node groups
    // These are specified by the user by means of tags or explicitly listing the node groups.
    std::vector<ComputationNodeBasePtr> m_features;
    std::vector<ComputationNodeBasePtr> m_labels;
    std::vector<ComputationNodeBasePtr> m_finalCriteria;
    std::vector<ComputationNodeBasePtr> m_evalNodes;
    std::vector<ComputationNodeBasePtr> m_outputNodes;
    std::vector<ComputationNodeBasePtr> m_pairNodes;                // nodes for the children network to pair
    vector<std::vector<ComputationNodeBasePtr>*> GetAllNodeGroups() // get all groups to allow to iterate over all of them ...continue
    {
        return vector<std::vector<ComputationNodeBasePtr>*>{&m_features, &m_labels, &m_finalCriteria, &m_evalNodes, &m_outputNodes, &m_pairNodes};
    }

    // used for sentence boundary information passed from reader to reset RNN state
    // specify how the minibatch is packed for each sample
    // TODO: This will change once we allow for multiple inconsistent layouts.
    MBLayoutPtr m_pMBLayout; // note that this must be installed before doing anything that needs it (default leaves a nullptr)

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

// The following emits the class and enables the BaseMatrix<double> to be available (used by EvalDll)
// The corresponding Matrix<float> is emitted in the SetDeviceId function above.
template class Matrix<double>;

// TODOs:
//  - automatic inference of time window w.r.t. delay nodes (and related nodes such as a temporal pooling)
//  - have overrides of RuntimeError etc. in ComputationNode, which prepend the error string with the node name and operation
//  - code prettification:
//     - sort all node implementations' methods into the same order; esp, ForwardProp() comes before partial
//     - sort important nodes first; move unused/experimental nodes into source files named accordingly

} } }
