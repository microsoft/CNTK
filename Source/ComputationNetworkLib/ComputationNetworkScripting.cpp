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
#include "ConvolutionalNodes.h"
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
    SetDeviceId(deviceId);

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
    // process work list
    // Also call FinalizeInit where we must.
    while (!workList.empty())
    {
        let node = workList.front();
        workList.pop_front();

        // add to set
        let wasAdded = AddNodeToNetIfNotYet(node);
        if (!wasAdded) // node already there (above will fail if there is a different node with the same name)
            continue;

        // If node derives from ILateAttachingNode() then it has unresolved inputs. Resolve them now.
        // This may generate a whole new load of nodes, including nodes which in turn have late init.
        let lateAttachingNode = dynamic_pointer_cast<ILateAttachingNode>(node);
        if (lateAttachingNode)
            lateAttachingNode->LateAttachInputs();

        // add it to the respective node group based on the tag
        auto tag = node->GetTag();
#if 0   // TODO: reenable this for back compat after we verified that at least none of our Jenkins tests use these anymore
        // legacy names
        if      (tag == L"criteria") tag = L"criterion";
        else if (tag == L"eval"    ) tag = L"evaluation";
#endif
        AddToNodeGroup(tag, node); // tag may be empty, or may have been set by array parameters

        // traverse children: append them to the end of the work list
        let& children = node->GetInputs();
        for (auto& child : children)
            workList.push_back(child); // (we could check whether c is in 'nodes' already here to optimize, but this way it is cleaner)
    }
    // TODO: process "outputNodes" etc. arrays

    // perform all necessary post-processing
    CompileNetwork();
}

// ===================================================================
// behave like a config
// This allows to access nodes inside a network as if it was an IConfigRecord.
// This is meant to be used by whatever we will replace MEL.
// TODO: Is there more than nodes that we want to return? Node groups? deviceId?
// ===================================================================

const ScriptableObjects::ConfigValuePtr& /*IConfigRecord::*/ ComputationNetwork::operator[](const wstring& id) const // e.g. confRec[L"message"]
{
    let* valuep = Find(id);
    if (!valuep)
        RuntimeError("Network does not contain a node called '%ls'", id.c_str());
    return *valuep;
}

#if 0 // unused, remove
static void RecoverTagFromNodeGroup(const ComputationNodeBasePtr& node, const std::vector<ComputationNodeBasePtr>& nodeList, const std::wstring& tag)
{
    // search nodeList
    for (auto& listNode : nodeList)
    {
        if (listNode == node)
        {
            // found it: set the tag
            let nodeWithTag = dynamic_pointer_cast<WithTag>(node);
            if (nodeWithTag)
            {
                let currentTag = nodeWithTag->GetTag();
                if (!currentTag.empty() && currentTag != tag)
                    RuntimeError("%ls %ls operation is in two node groups ('%ls' and '%ls'), which is unsupported.", node->NodeName().c_str(), node->OperationName().c_str(), currentTag.c_str(), tag.c_str());
                nodeWithTag->SetTag(tag);
            }
            else LogicError("RecoverTagFromNodeGroup: Unexpected type.");
            return;
        }
    }
}
#endif

const ScriptableObjects::ConfigValuePtr* /*IConfigRecord::*/ ComputationNetwork::Find(const wstring& id) const // returns nullptr if not found
{
    let iter = m_nameToNodeMap.find(id);
    if (iter == m_nameToNodeMap.end())
        return nullptr; // no such node
    const ComputationNodeBasePtr& node = iter->second;
    // TODO: What is the expressionPath?
    // We have a small problem: We want to return a ComputationNodeBasePtr, but must return the *address* of a ConfigValuePtr.
    // Hence, we will create this ConfigValuePtr upon first access and hold it in a map, and furtheron return that map entry.
    let& mapIter = m_nodesAsConfigValues.find(node);
    if (mapIter != m_nodesAsConfigValues.end())
        return &mapIter->second;
    // not in the cache yet: create it
    auto nodeName = node->NodeName();   // failFn lambda below holds a copy of the name for the error message. Let's not hold an unneccessary shared_ptr to the node, risking cycles & stuff.
    auto valuep = ConfigValuePtr(static_pointer_cast<Object>(node), [nodeName](const std::wstring &) { LogicError("ComputationNetwork: Failed to retrieve node '%ls'.", nodeName.c_str()); }, node->NodeName());
    let res = m_nodesAsConfigValues.insert(make_pair(node, move(valuep)));
    assert(&res.first->second == &m_nodesAsConfigValues.find(node)->second);
    assert(res.second);        // this says whether it has been inserted. It better be.
    return &res.first->second; // this is the cached ConfigValuePtr
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
// scripting wrapper to construct ComputationNetwork from file (aka 'load')
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
        Load<ElemType>(pathName); // note that for CNTK_MODEL_VERSION_5 and above, 'ElemType' is ignored
    }
};

ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<ComputationNetworkFromFile<float>, ComputationNetworkFromFile<double>> registerComputationNetworkFromFile(L"ComputationNetworkFromFile");

}}}
