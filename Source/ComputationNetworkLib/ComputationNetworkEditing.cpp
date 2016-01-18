//
// <copyright file="ComputationNetworkEditing.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "InputAndParamNodes.h"
#include "ConvolutionalNodes.h"
#include <string>
#include <vector>
#include <list>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

// This source file contains files related to model editing with MEL. Future BrainScript editing will not modify nodes in-place.

// -----------------------------------------------------------------------
// network editing
// -----------------------------------------------------------------------

ComputationNodeBasePtr ComputationNetwork::CopyNode(const ComputationNetwork& fromNet,
                                                    const std::wstring fromName,
                                                    std::wstring toName,
                                                    const CopyNodeFlags flags)
{
    InvalidateCompiledNetwork();

    if (toName == L"")
        toName = fromName;

    ComputationNodeBasePtr pFromNode = fromNet.GetNodeFromName(fromName);
    ComputationNodeBasePtr pToNode;

    // don't allow cross network child copy unless caller explicity handles children fixup
    if ((flags & CopyNodeFlags::copyNodeChildren) &&
        this != &fromNet && !(flags & CopyNodeFlags::copyNodeChildrenCrossNetwork))
    {
        LogicError("CopyNode: Copying node children across network is invalid.");
    }

    if (!NodeNameExists(toName))
    {
        pToNode = pFromNode->Duplicate(toName, flags);
        AddNodeToNet(pToNode);
    }
    else
    {
        // node already exists
        pToNode = GetNodeFromName(toName);

        // same node. no copy needed
        if (pFromNode == pToNode)
            LogicError("CopyNode: You are copying the node to the same network with same node name.");
        else
            pFromNode->CopyTo(pToNode, toName, flags); // blast it over the existing node
    }
    return pToNode;
}

// only copy a complete independent tree
// when node name exists
void ComputationNetwork::CopySubTree(const ComputationNetwork& fromNet,
                                     const std::wstring fromName, std::wstring toNamePrefix,
                                     const CopyNodeFlags flags)
{
    InvalidateCompiledNetwork();

    if (!(flags & CopyNodeFlags::copyNodeValue))
        LogicError("CopySubTree: you cannot copy a tree without copying the node values.");

    ComputationNodeBasePtr fromRoot = fromNet.GetNodeFromName(fromName);

    for (const auto& fromNode : GetEvalOrder(fromRoot))
    {
        wstring fromNodeName = fromNode->NodeName();
        wstring toNodeName = toNamePrefix + fromNodeName;

        ComputationNodeBasePtr toNode = CopyNode(fromNet, fromNodeName,
                                                 toNodeName,
                                                 CopyNodeFlags::copyNodeValue);

        if (flags & CopyNodeFlags::copyNodeChildren)
        {
            // copy the children structure but use the new nodes generated
            for (int i = 0; i < fromNode->GetNumInputs(); i++)
                toNode->SetInput(i, GetNodeFromName(toNamePrefix + fromNode->GetInputs()[i]->NodeName()));
        }
    }
}

// you can only copy inputs from nodes in the same network
void ComputationNetwork::CopyInputs(const std::wstring fromName, std::wstring toName)
{
    CopyNode(*this, fromName, toName, CopyNodeFlags::copyNodeChildren);
}

// RenameNode - Rename a node to another name
// nodeNameOrig - original node name
// nodeNameNew - new node name
void ComputationNetwork::RenameNode(const std::wstring& nodeNameOrig, const std::wstring& nodeNameNew)
{
    InvalidateCompiledNetwork();

    ComputationNodeBasePtr nodeToRename = GetNodeFromName(nodeNameOrig);

    auto iter = m_nameToNodeMap.find(nodeNameNew);
    if (iter != m_nameToNodeMap.end()) //found
        RuntimeError("RenameNode: Target name already exists.");

    //rename the node and update the mapping table
    nodeToRename->SetNodeName(nodeNameNew);
    m_nameToNodeMap.erase(nodeNameOrig);
    m_nameToNodeMap[nodeNameNew] = nodeToRename;
}

void ComputationNetwork::RenameNode(ComputationNodeBasePtr node, const std::wstring& newNodeName)
{
    // TODO: check if new name exists
    m_nameToNodeMap.erase(node->NodeName());
    node->SetNodeName(newNodeName);
    AddNodeToNet(node);
}

void ComputationNetwork::DeleteNode(const std::wstring& nodeName)
{
    InvalidateCompiledNetwork();

    ComputationNodeBasePtr nodeToDelete = GetNodeFromName(nodeName);

    //first delete links, if this node is involved, the whole connection will be removed
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (size_t i = 0; i < node->GetNumInputs(); i++)
        {
            ComputationNodeBasePtr child = node->GetInputs()[i];

            //nodeToDelete is a child
            if (child == nodeToDelete)
            {
                // this used to call DetatchInputs(), but it's better for MEL to retain other inputs
                node->SetInput(i, nullptr);
                break;
            }
        }
    }

    // nodeToDelete is a parent
    nodeToDelete->DetachInputs(); // deref all its inputs; if we don't do that, we might end up with a mem leak due to a circular reference

    // unlink from all node-group sets
    for (auto groupIter : GetAllNodeGroups())
    {
        auto search = std::find(groupIter->begin(), groupIter->end(), nodeToDelete);
        if (search != groupIter->end())
            groupIter->erase(search);
    }

    // Note: the necessary update of m_allSEQNodes is hanlded by the InvalidateCompiledNetwork() call above

    //delete the node itself
    m_nameToNodeMap.erase(nodeName); // this will deref the node and possibly deallocate it
}

// change the node associated with nodeName to newNode; used in the KL-reg based adaptation to reduce feature copy
// need to update all the mappings as well childrens
void ComputationNetwork::ChangeNode(wstring nodeName, ComputationNodeBasePtr newNode)
{
    InvalidateCompiledNetwork();

    ComputationNodeBasePtr oldNode = GetNodeFromName(nodeName);
    if (oldNode->OperationName() != newNode->OperationName())
        InvalidArgument("newNode must have the same type as the old node.");

    //change children
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (int i = 0; i < node->GetNumInputs(); i++)
            if (node->GetInputs()[i] == oldNode)
                node->SetInput(i, newNode);
    }

    //change name map
    m_nameToNodeMap[nodeName] = newNode;
    for (int i = 0; i < oldNode->GetNumInputs(); i++)
        newNode->SetInput(i, oldNode->GetInputs()[i]);

    //change other maps
    for (auto groupIter : GetAllNodeGroups())
    {
        auto& group = *groupIter;
        for (int i = 0; i < group.size(); i++)
            if (group[i] == oldNode)
                group[i] = newNode;
    }
}

// replace the old node with the current node, assuming the old node is a leaf node
// need to update those nodes who use oldNode as their child
void ComputationNetwork::ReplaceLeafNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
{
    InvalidateCompiledNetwork();

    ComputationNodeBasePtr oldNode = GetNodeFromName(oldNodeName);

    // change the input of those nodes whose child is oldNode
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (int i = 0; i < node->GetNumInputs(); i++)
            if (node->GetInputs()[i] == oldNode)
                node->SetInput(i, newNode);
    }
    m_nameToNodeMap[newNode->GetName()] = newNode;

    // now the old node becomes a orphan node , remove it
    DeleteNode(oldNodeName);
    //RemoveOrphanNode(oldNode);
}

void ComputationNetwork::ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
{
    InvalidateCompiledNetwork();

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
    for (int i = 0; i < newNode->GetNumInputs(); ++i)
    {
        if (m_nameToNodeMap.find(newNode->GetInputs()[i]->NodeName()) == m_nameToNodeMap.end())
            RuntimeError("Child node does not exist.");
        newNode->SetInput(i, m_nameToNodeMap[newNode->GetInputs()[i]->NodeName()]);
    }

    // Addes it to criterion node list.
    m_finalCriteria[index] = newNode;
    m_nameToNodeMap[newNode->NodeName()] = newNode;
}

void ComputationNetwork::AddFeatureNode(ComputationNodeBasePtr featureNode)
{
    InvalidateCompiledNetwork();

    wstring nodeName = featureNode->NodeName();
    if (NodeNameExists(nodeName))
        RuntimeError("AddFeatureNode: feature node already exists.");
    m_nameToNodeMap[nodeName] = featureNode;
    m_features.push_back(featureNode);
}

// We only remove the node, not delete it.
void ComputationNetwork::RemoveFeatureNode(ComputationNodeBasePtr featureNode)
{
    InvalidateCompiledNetwork();

    wstring nodeName = featureNode->NodeName();
    if (!NodeNameExists(nodeName))
        RuntimeError("RemoveFeatureNode: feature node does not exist.");

    // Removes links.
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); ++nodeIter)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (size_t i = 0; i < node->GetNumInputs(); ++i)
        {
            ComputationNodeBasePtr child = node->GetInputs()[i];
            if (child == featureNode)
            {
                node->SetInput(i, NULL);
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

// sets m_parameterUpdateRequired in all LearnableParameters feeding into the passed rootNode
// Called from MEL  --TODO: correct?
void ComputationNetwork::SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr& rootNode)
{
    // find nodes from all available nodes
    if (rootNode == nullptr)
    {
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            if (node->OperationName() == OperationNameOf(LearnableParameter))
                node->SetParameterUpdateRequired(needGradient);
        }
    }
    else
    {
        // for calculating a specific node
        for (const auto& node : GetEvalOrder(rootNode))
        {
            if (node->OperationName() == OperationNameOf(LearnableParameter))
                node->SetParameterUpdateRequired(needGradient);
        }
    }
}

void ComputationNetwork::SetBatchNormlizationNodesBelowEvalMode(const bool evalMode, const ComputationNodeBasePtr& rootNode /* = nullptr */)
{
    vector<ComputationNodeBasePtr> nodes;
    if (rootNode == nullptr)
    {
        for (auto pair : m_nameToNodeMap)
        {
            nodes.push_back(pair.second);
        }
    }
    else
    {
        auto allnodes = rootNode->EnumerateNodes(true);
        for (auto node : allnodes)
            nodes.push_back(node);
    }

    for (auto& node : nodes)
    {
        if (node->OperationName() == OperationNameOf(BatchNormalizationNode))
        {
            auto pNode = dynamic_pointer_cast<BatchNormalizationNode<float>>(node);
            if (!pNode)
            {
                auto pNode2 = dynamic_pointer_cast<BatchNormalizationNode<double>>(node);
                if (!pNode2)
                {
                    RuntimeError("Invalid node type: node name=%ls. We assume either BatchNormalizationNode<float> or BatchNormalizationNode<double>\n", node->NodeName().c_str());
                }
            }
            else
            {
                pNode->SetEvalMode(evalMode);
            }
        }
    }
}
} } }
