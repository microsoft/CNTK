//
// <copyright file="ComputationNetworkScipting.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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

    // -------------------------------------------------------------------
    // construction from config
    // -------------------------------------------------------------------

    // construct a ComputationNetwork from a ConfigRecord
    ComputationNetwork::ComputationNetwork(const IConfigRecordPtr configp) :
        ComputationNetwork()
    {
        let & config = *configp;

        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        SetDeviceId(deviceId);

        deque<ComputationNodeBasePtr> workList;
        // flatten the set of all nodes
        // we collect all root ComputationNodes from the config record, and then expand into all their children by work-list processing
        // TODO: This currently only supports nodes of the same ElemType. We could allow conversion operators.
        for (let & id : config.GetMemberIds())
        {
            let & value = config[id];
            if (value.Is<ComputationNodeBase>())
                workList.push_back((const ComputationNodeBasePtr&)value);
        }
        // process work list
        // Also call FinalizeInit where we must.
        while (!workList.empty())
        {
            let node = workList.front();
            workList.pop_front();

            // add to set
            let res = m_nameToNodeMap.insert(make_pair(node->NodeName(), node));
            if (!res.second)        // not inserted: we already got this one
                if (res.first->second == node)
                    continue;       // the same
                else                // oops, a different node with the same name
                    LogicError("ComputationNetwork: multiple nodes with the same NodeName() '%ls'", node->NodeName().c_str());

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
                if (tag == L"feature")                              FeatureNodes().push_back(node);
                else if (tag == L"label")                           LabelNodes().push_back(node);
                else if (tag == L"criterion" || tag == L"criteria") FinalCriterionNodes().push_back(node); // 'criteria' is wrong (plural); we keep it for compat
                else if (!_wcsnicmp(tag.c_str(), L"eval", 4))       EvaluationNodes().push_back(node);     // eval*
                else if (tag == L"output")                          OutputNodes().push_back(node);
#if 0           // deprecated
                else if (tag == L"pair")                            PairNodes().push_back(node);           // TODO: I made this up; the original code in SynchronousExecutionEngine did not have this
#endif
                else if (!tag.empty())
                    RuntimeError("ComputationNetwork: unknown tag '%ls'", tag.c_str());
                // TODO: are there nodes without tag? Where do they go?
            }

            // traverse children: append them to the end of the work list
            let & children = node->GetChildren();
            for (auto & child : children)
                workList.push_back(child);  // (we could check whether c is in 'nodes' already here to optimize, but this way it is cleaner)
        }

        ValidateNetwork();
#if 1
        wstring args = ToString();
        fprintf(stderr, "%ls\n", args.c_str());
#endif
        // these post-processing steps are done by the other network builders, but I don't know why they are necessary
        FixupInputMinibatchSize();         // make sure dimensions are set up correctly
        ResetEvalTimeStamp();              // (should not really be needed)
    }

}}}
