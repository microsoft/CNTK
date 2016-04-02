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
ComputationNetwork::ComputationNetwork(const IConfigRecordPtr configp)
    : ComputationNetwork()
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
        let nodeWithTag = dynamic_pointer_cast<WithTag>(node);
        if (nodeWithTag)
        {
            wstring tag = nodeWithTag->GetTag();
            if (tag == L"feature")
                FeatureNodes().push_back(node);
            else if (tag == L"label")
                LabelNodes().push_back(node);
            else if (tag == L"criterion" || tag == L"criteria")
                FinalCriterionNodes().push_back(node); // 'criteria' is wrong (plural); we keep it for compat
            else if (!_wcsnicmp(tag.c_str(), L"eval", 4))
                EvaluationNodes().push_back(node); // eval*
            else if (tag == L"output")
                OutputNodes().push_back(node);
            else if (!tag.empty())
                RuntimeError("ComputationNetwork: unknown tag '%ls'", tag.c_str());
            // TODO: are there nodes without tag? Where do they go?
        }

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
// TODO: implement this
// ===================================================================

const ScriptableObjects::ConfigValuePtr& /*IConfigRecord::*/ ComputationNetwork::operator[](const wstring& id) const // e.g. confRec[L"message"]
{
    id;
    RuntimeError("unknown class parameter"); // (for now)
}
const ScriptableObjects::ConfigValuePtr* /*IConfigRecord::*/ ComputationNetwork::Find(const wstring& id) const // returns nullptr if not found
{
    id;
    return nullptr; // (for now)
}
vector<wstring> /*IConfigRecord::*/ ComputationNetwork::GetMemberIds() const
{
    return vector<wstring>();
}

}}}
