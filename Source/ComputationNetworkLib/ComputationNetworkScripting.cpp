//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ScriptableObjects.h"

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "RecurrentNodes.h"
#include "NonlinearityNodes.h"
#include "LinearAlgebraNodes.h"
#include "ReshapingNodes.h"

#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"

#include <memory>
#include <deque>
#include <set>
#include <string>

#ifndef let
#define let const auto
#endif

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace Microsoft::MSR::ScriptableObjects;

// ===================================================================
// construction from config
// ===================================================================

// construct a ComputationNetwork from a ConfigRecord
ComputationNetwork::ComputationNetwork(const IConfigRecordPtr configp) :
    ComputationNetwork()
{
    let& config = *configp;

    DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int) config[L"deviceId"];

    deque<ComputationNodeBasePtr> workList;
    // flatten the set of all nodes
    // we collect all root ComputationNodes from the config record, and then expand into all their children by work-list processing
    // TODO: This currently only supports nodes of the same ElemType. We could allow conversion operators.
    for (let& id : config.GetMemberIds())
    {
        let& value = config[id];
        if (value.Is<ComputationNodeBase>())
            workList.push_back((const ComputationNodeBasePtr&) value);
    }

    // TODO: process "outputNodes" etc. arrays: Sync to node Tags, and make them all roots.

    // construct from roots
    ConstructFromRoots(deviceId, move(workList), map<ComputationNodeBasePtr, ComputationNodeBasePtr>()/*no mapping*/);
}

// construct a network from a list of roots (passed in 'workList')
// This will add to m_nameToNodeMap[] all roots and all nodes reachable from those roots.
// If 'replacements' is given, all root pointers as well as all input pointers of reachable nodes will be mapped. This is needed for model editing.
void ComputationNetwork::ConstructFromRoots(DEVICEID_TYPE deviceId, deque<ComputationNodeBasePtr>&& workList, const map<ComputationNodeBasePtr, ComputationNodeBasePtr>& replacements)
{
    SetDeviceId(deviceId);
    assert(this->GetTotalNumberOfNodes() == 0);

    // replace if requested
    // This happens for model editing.
    // workList operates on mapped nodes.
    size_t numRelinked = 0;
    for (auto& nodeRef : workList)
    {
        let iter = replacements.find(nodeRef);
        if (iter != replacements.end())
        {
            assert(nodeRef->GetEnvironmentPtr()); // must be in some network if mapped
            nodeRef = iter->second; // nodeRef is a reference, so this patches the workList in-place
            numRelinked++;
        }
    }

    // process work list
    // Also call LateAttachInputs() where needed.
    while (!workList.empty())
    {
        let node = workList.front();
        workList.pop_front();

        // add to set
        let wasAdded = AddNodeToNetIfNotYet(node, /*makeUniqueName=*/ true);
        if (!wasAdded) // node already there (above will fail if there is a different node with the same name)
            continue;

        // If node derives from ILateAttachingNode() then it has unresolved inputs. Resolve them now.
        // This may generate a whole new load of nodes, including nodes which in turn have late init.
        // Note: In case of editing, we may be adding a new node that references nodes from the old
        // network that must be mapped because their inputs have changed. Hence, it is important to
        // to the mapping *after* late attaching.
        if (node->GetNumInputs() == 0) // (if this function is called during model editing, we may already have our inputs)
        {
            let lateAttachingNode = dynamic_pointer_cast<ILateAttachingNode>(node);
            if (lateAttachingNode)
                lateAttachingNode->LateAttachInputs();
        }

        // add it to the respective node groups based on the tags
        for (auto tag : node->GetTags())
        {
#if 1       // we keep this for a while (we already verified that our samples no longer use this)
            // map legacy names
            if      (tag == L"criteria") tag = L"criterion";
            else if (tag == L"eval"    ) tag = L"evaluation";
#endif
            AddToNodeGroup(tag, node); // tag may be empty, or may have been set by array parameters
        }

        // traverse children: append them to the end of the work list
        // In case of model editing, map inputs.
        for (size_t i = 0; i < node->GetNumInputs(); i++)
        {
            auto input = node->Input(i);

            // replace input if needed
            let iter = replacements.find(input);
            if (iter != replacements.end())
            {
                assert(input->GetEnvironmentPtr()); // must be in some network if mapped
                input = iter->second;
                numRelinked++;
                node->SetInput(i, input);
            }

            workList.push_back(input); // (we could check whether c is in 'nodes' already here to optimize, but this way it is cleaner)
        }
    }
    if (numRelinked > 0)
        fprintf(stderr, "ConstructFromRoots: %d references were remapped.", (int)numRelinked);

    // perform all necessary post-processing
    CompileNetwork();
}

// ===================================================================
// behave like a config
// This allows to access nodes inside a network as if it was an IConfigRecord.
// This is meant to be used by whatever we will replace MEL.
// TODO: Is there more than nodes that we want to return? Node groups? deviceId?
// ===================================================================

// not in the cache yet: create it (or not if no such member)
void /*CustomConfigRecord::*/ ComputationNetwork::LazyCreateConfigMember(const wstring& id) const /*override*/
{
    let iter = m_nameToNodeMap.find(id);
    if (iter == m_nameToNodeMap.end())
        return; // no such node
    const ComputationNodeBasePtr& node = iter->second;
    // TODO: What is the expressionPath?
    let& nodeName = node->NodeName();   // failFn lambda below holds a copy of the name for the error message. Let's not hold an unneccessary shared_ptr to the node, risking cycles & stuff.
    auto valuep = ConfigValuePtr(node, [nodeName](const std::wstring &) { LogicError("ComputationNetwork: Failed to retrieve node '%ls'.", nodeName.c_str()); }, node->NodeName());
    InsertConfigMember(id, move(valuep));
}

vector<wstring> /*IConfigRecord::*/ ComputationNetwork::GetMemberIds() const
{
    vector<wstring> nodeNames;
    for (let& iter : m_nameToNodeMap)
    {
        const ComputationNodeBasePtr& node = iter.second;
        const wstring& nodeName = node->NodeName();
        if (nodeName.find_first_of(L".[$")) // only expose the top-level names
            continue;
        nodeNames.push_back(nodeName);
    }
    return nodeNames;
}

// ===================================================================
// ComputationNetworkFromFile
// scripting wrapper to construct ComputationNetwork from file (aka 'Load')
// ===================================================================

template<class ElemType>
class ComputationNetworkFromFile : public ComputationNetwork
{
public:
    ComputationNetworkFromFile(const IConfigRecordPtr configp) :
        ComputationNetwork()
    {
        let& config = *configp;

        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        SetDeviceId(deviceId);

        wstring pathName = config[L"pathName"];
        fprintf(stderr, "Load: Loading model file: %ls", pathName.c_str());
        Load<ElemType>(pathName); // note that for CNTK_MODEL_VERSION_5 and above, 'ElemType' is ignored
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<ComputationNetworkFromFile<float>, ComputationNetworkFromFile<double>> registerComputationNetworkFromFile(L"ComputationNetworkFromFile");

// ===================================================================
// ComputationNetworkWithEdits
// scripting wrapper to construct by modifying an input network (aka 'Edit')
// ===================================================================

class ComputationNetworkWithEdits : public ComputationNetwork
{
    // helper to execute a BS function that maps a CompuationNode to a ComputationNode
    // The function may return:
    //  - its input --> no edit was made
    //  - an different existing node --> all nodes that use this input should use the returned node instead
    //  - a newly created node or sub-graph --> this node should replace the old one
    // In the latter two cases, the returned node may have inputs that are totally different
    // from the original node's.
    ComputationNodeBasePtr CallEditFunction(ComputationNodeBasePtr node, const ConfigLambda& editFn)
    {
        // wrap the argument in a ConfigValuePtr
        const wstring& nodeName = node->NodeName();
        const wstring& expressionName = nodeName;   // TODO: think this through
        auto valuep = ConfigValuePtr(static_pointer_cast<Object>(node), [nodeName](const std::wstring &) { LogicError("CallEditFunction: Failed to retrieve node '%ls'.", nodeName.c_str()); }, expressionName);
        vector<ConfigValuePtr> args{ valuep };
        // execute the lambda (this executes a function that is BS)
        ConfigValuePtr result = editFn.Apply(move(args), ConfigLambda::NamedParams(), expressionName);
        // cast the result back
        return result.AsPtr<ComputationNodeBase>();
    }

public:
    // constructor
    // This constructs a new model from an existing one by:
    //  - iterating over all nodes
    //  - trying a sequence of edit functions until one made an edit
    //    This is like pattern matching: The first edit function that matches will return an updated node.
    //  - assemble a new network that consists of the old network with edits applied
    // Note that the old model is not edited in-place; instead a new copy is made that shares
    // unchanged nodes with the original one.
    ComputationNetworkWithEdits(const IConfigRecordPtr configp) :
        ComputationNetwork()
    {
        // get config parameters
        let& config = *configp;
        let& net = config[L"inputModel"].AsRef<ComputationNetwork>();
        let editFunctions = ScriptableObjects::ConfigArray::FlattenedVectorFrom<ConfigLambda>(config[L"editFunctions"]);
        let additionalRoots = ScriptableObjects::ConfigArray::FlattenedVectorFrom<ComputationNodeBasePtr>(config[L"additionalRoots"]);

        // gather all the edits
        // This runs the edit functions over all nodes.
        map<ComputationNodeBasePtr, ComputationNodeBasePtr> replacements; // [orig, replacement] all return values from the Edit-function calls
        let allNodes = net.GetAllNodes();
        for (let& node : allNodes) // iterate over all nodes
        {
            for (let& editFn : editFunctions) // try all edit functions until one matched
            {
                let newNode = CallEditFunction(node, editFn);
                if (newNode != node) // true if the edit function provided a replacement (an "edit")
                {
                    replacements[node] = newNode; // remember the replaceent
                    break;                        // we only apply the first edit function & stop
                }
            }
        }
        fprintf(stderr, "Edit: %d nodes were edited.\n", (int)replacements.size());
#ifdef _DEBUG
        for (let& replacement : replacements)
            fprintf(stderr, "\t%ls = %ls() --> %ls = %ls()\n", replacement.first->NodeName().c_str(), replacement.first->OperationName().c_str(), replacement.second->NodeName().c_str(), replacement.second->OperationName().c_str());
#endif

        // also 'edit' all nodes that have updated *inputs*
        // All nodes that take inputs that have been edited must have their inputs updated.
        // Since we do not update the model in-place, we must also create replacements for these.
        // That is achieved by recursively including all parents of edits into the set of edits.
        let parents = net.CreateParentsMap();
        deque<ComputationNodeBasePtr> workList; // work list for recursion
        for (let& replacement : replacements)
            workList.push_back(replacement.first);
        while (!workList.empty())
        {
            let node = workList.front();
            workList.pop_front();
            // loop over the node's parents
            for (let& parent : parents.find(node)->second)
            {
                // "edit" (clone) the parent if not yet
                if (replacements.find(parent) != replacements.end())
                    continue; // already a known replacement
                // we must "edit" the parent since it depends on a replaced input
                replacements[parent] = parent->Duplicate();
                // and put this parent into the workList, so that we will gets its parent in turn, etc.
                workList.push_back(parent);
#if 0 //def _DEBUG
                fprintf(stderr, "\t%ls = %ls() --> relink %ls\n", parent->NodeName().c_str(), parent->OperationName().c_str(), replacements[parent]->NodeName().c_str());
#endif
            }
        }
        fprintf(stderr, "Edit: %d out of %d nodes were either edited or need to be relinked.\n", (int)replacements.size(), (int)net.GetTotalNumberOfNodes());
        // Now the keys of replacements[] define the set of all nodes that must be relinked.

        // replacements may point to nodes that are replacements themselves
        // This really can only happen if a replacement itself is an old node.
        for (auto& iter : replacements)
            while (replacements.find(iter.second) != replacements.end())
                iter.second = replacements.find(iter.second)->second;

        // Now we have three kinds of nodes:
        //  - unmodified nodes that will be shared with the old network
        //  - modified nodes (user edits and their parents)
        //  - original nodes that are no longer referenced
        // The new network will be constructed to have the same roots as the original.

        // determine all roots
        deque<ComputationNodeBasePtr> roots;
        // start with the original network
        for (let& node : allNodes)
            if (parents.find(node)->second.empty()) // no parents: it's a root
                roots.push_back(node);
        // also add new roots
        for (let& node : additionalRoots)
            roots.push_back(node);
        fprintf(stderr, "Edit: %d roots to construct the network from.\n", (int)roots.size());
#ifdef _DEBUG
        for (let& node : roots)
            fprintf(stderr, "\t%ls = %ls()\n", node->NodeName().c_str(), node->OperationName().c_str());
#endif
        // The new network is now defined by roots.

        // now construct the new network
        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        ConstructFromRoots(deviceId, move(roots), replacements);
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNetworkWithEdits> registerComputationNetworkWithEdits(L"ComputationNetworkWithEdits");

}}}
