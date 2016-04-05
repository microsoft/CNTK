//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "InputAndParamNodes.h"
#include "TrainingNodes.h"
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
    if ((flags & CopyNodeFlags::copyNodeInputLinks) &&
        this != &fromNet && !(flags & CopyNodeFlags::copyNodeAcrossNetworks))
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

    for (const auto& fromNode : GetEvalOrder(fromRoot)) // BUGBUG: This probably will fail because the precomputed eval orders are invalid at this point.
    {
        wstring fromNodeName = fromNode->NodeName();
        wstring toNodeName = toNamePrefix + fromNodeName;

        ComputationNodeBasePtr toNode = CopyNode(fromNet, fromNodeName,
                                                 toNodeName,
                                                 CopyNodeFlags::copyNodeValue);

        if (flags & CopyNodeFlags::copyNodeInputLinks)
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
    CopyNode(*this, fromName, toName, CopyNodeFlags::copyNodeInputLinks);
}

// RenameNode - Rename a node to another name
// nodeNameOrig - original node name
// nodeNameNew - new node name
void ComputationNetwork::RenameNode(const std::wstring& nodeNameOrig, const std::wstring& newNodeName)
{
    RenameNode(GetNodeFromName(nodeNameOrig), newNodeName);
}

void ComputationNetwork::RenameNode(ComputationNodeBasePtr node, const std::wstring& newNodeName)
{
    // make sure the new name is not already used
    auto iter = m_nameToNodeMap.find(newNodeName);
    if (iter != m_nameToNodeMap.end()) // found
        RuntimeError("RenameNode: Target name already exists.");

    InvalidateCompiledNetwork();

    RemoveNodeFromNet(node);        // take it out remporarily
    node->SetNodeName(newNodeName); // change the name
    AddNodeToNet(node);             // and put it back
}

// deletes a node from the network including setting all input links to it to null, and removing it from the node groups
void ComputationNetwork::DeleteNode(const std::wstring& nodeName)
{
    InvalidateCompiledNetwork();

    ComputationNodeBasePtr nodeToDelete = GetNodeFromName(nodeName);

    // first delete links, if this node is involved, the whole connection will be removed
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (size_t i = 0; i < node->GetNumInputs(); i++)
        {
            ComputationNodeBasePtr child = node->GetInputs()[i];

            // nodeToDelete is a child
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

    // delete the node itself
    RemoveNodeFromNet(nodeToDelete);
}

// replace a named node by newNode of the same type under the same name, including moving over all network links
// This is used in the KL-reg based adaptation to reduce feature copy
// need to update all the mappings as well childrens.
void ComputationNetwork::ChangeNode(wstring nodeName, ComputationNodeBasePtr newNode)
{
    ComputationNodeBasePtr oldNode = GetNodeFromName(nodeName);

    if (newNode->NodeName() != nodeName) // TODO: This was not tested for earlier; I hope no code depends on this.
        InvalidArgument("ChangeNode: newNode must have the same name as the old node.");
    if (oldNode->OperationName() != newNode->OperationName())
        InvalidArgument("ChangeNode: newNode must have the same type as the old node.");

    InvalidateCompiledNetwork();

    // change all nodes to have old node as input to point to the new node instead
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (int i = 0; i < node->GetNumInputs(); i++)
            if (node->GetInputs()[i] == oldNode)
                node->SetInput(i, newNode);
    }

    // change all inputs of this new node to share the old one's inputs
    for (int i = 0; i < oldNode->GetNumInputs(); i++)
    {
        newNode->SetInput(i, oldNode->GetInputs()[i]); // TODO: use AttachInput()?
        //oldNode->SetInput(i, nullptr); // BUGBUG: old node should no longer point into the network
    }

    // replace the node in the network
    RemoveNodeFromNet(oldNode);
    AddNodeToNet(newNode);

    // also update node groups
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
// TODO: Can this be called with a node that's already part of the network? This is currently allowed, but should it?
// BUGBUG: Seems ChangeNode() also updates node groups. Why doesn't this function?
// BUGBUG: What if newNode is the one referenced by oldNodeName?
// BUGBUG: Or what if an unrelated node of the same name exists?
void ComputationNetwork::ReplaceLeafNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
{
    InvalidateCompiledNetwork();

    ComputationNodeBasePtr oldNode = GetNodeFromName(oldNodeName);

    // relink the input of those nodes whose child is oldNode to point to the new one instead
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr node = nodeIter->second;
        for (int i = 0; i < node->GetNumInputs(); i++)
            if (node->GetInputs()[i] == oldNode)
                node->SetInput(i, newNode);
    }

    // add the new, remove the old
    AddNodeToNetIfNotYet(newNode);
    DeleteNode(oldNodeName); // TODO: can this just be RemoveNodeFromNet()?
}

// add a new criterion node and at the same time orphan the previous one (it won't be removed)
// The newNode can have the same name and come with pre-connected inputs, which will be used to connect to existing nodes of the same name.
// BUGBUG: Can this operate on both new and existing nodes?
void ComputationNetwork::ReplaceFinalCriterionNode(wstring oldNodeName, ComputationNodeBasePtr newNode)
{
    InvalidateCompiledNetwork();

    // remove old criterion node
    // BUGBUG: The old node is not removed from the network. Seems strangely inconsistent.
    bool wasThere = RemoveFromNodeGroup(L"criterion", GetNodeFromName(oldNodeName));
    if (!wasThere)
        RuntimeError("ReplaceFinalCriterionNode: The node to be replaced is not a criterion node.");

    // replace children
    // This looks for nodes in the network that have the same name as its current inputs, and then relinks its inputs to those.
    // I.e. this allows to move a node from network to another and reconnect by the names if its inputs.
    for (int i = 0; i < newNode->GetNumInputs(); ++i)
    {
        if (m_nameToNodeMap.find(newNode->GetInputs()[i]->NodeName()) == m_nameToNodeMap.end())
            RuntimeError("Child node %ls is not part of the network.", newNode->GetInputs()[i]->NodeName().c_str());
        newNode->SetInput(i, m_nameToNodeMap[newNode->GetInputs()[i]->NodeName()]);
    }

    // add it to the network
    AddNodeToNetIfNotYet(newNode);

    // add new node to criterion node group
    AddToNodeGroup(L"criterion", newNode);
}

void ComputationNetwork::AddFeatureNode(ComputationNodeBasePtr featureNode)
{
    InvalidateCompiledNetwork();

    AddNodeToNet(featureNode);
    AddToNodeGroup(L"feature", featureNode);
}

#if 0 // unused--delete
// We only remove the node from the net, not destruct it.
ComputationNodeBasePtr ComputationNetwork::RemoveFeatureNode(ComputationNodeBasePtr featureNode)
{
    InvalidateCompiledNetwork();

    wstring nodeName = featureNode->NodeName();
    if (!NodeNameExists(nodeName))
        RuntimeError("RemoveFeatureNode: feature node does not exist.");

    // remove links to this node
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
    auto search = std::find(m_featureNodes.begin(), m_featureNodes.end(), featureNode);
    if (search != m_featureNodes.end())
        m_featureNodes.erase(search);

    // note: we don't bother resetting the tag since the node is gone
    return RemoveNodeFromNet(featureNode);
}
#endif

// sets m_learningRateMultiplier in all LearnableParameters feeding into the passed rootNode
// Called from MEL
void ComputationNetwork::SetLearnableNodesBelowLearningRateMultiplier(const float learningRateMultiplier, const ComputationNodeBasePtr& rootNode)
{
    // find nodes from all available nodes
    if (rootNode == nullptr)
    {
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = nodeIter->second;
            if (node->OperationName() == OperationNameOf(LearnableParameter))
                node->SetLearningRateMultiplier(learningRateMultiplier);
        }
    }
    else
    {
        // for calculating a specific node
        for (const auto& node : GetAllNodesForRoot(rootNode))
        {
            if (node->OperationName() == OperationNameOf(LearnableParameter))
                node->SetLearningRateMultiplier(learningRateMultiplier);
        }
    }
}

}}}
