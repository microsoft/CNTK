//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "RecurrentNodes.h"
#include <string>
#include <set>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// network recurrent-loop analysis
// -----------------------------------------------------------------------

// Helper functions
static int DetermineLoopDirection(const std::vector<ComputationNodeBasePtr>& nestedNodes);
static int GetRecurrenceSteppingDirection(const ComputationNodeBasePtr& node);

//
// The method below determine evaluation order, which is tricky in the presence of recurrent loops.
// It is the main entry for network recurrent-loop analysis.
// This function analysis the networks for recurrent loops present in the computation graph.
// It sets/updates:
//  - m_allSEQNodes
//  - ComputationNode::m_isPartOfLoop (exposed to outside as IsPartOfLoop())
//  - the cached m_evalOrders[root], reordered to make nodes belonging to the same loop consecutive. TODO: Try not to do that.
// It is often called before ValidateNetwork() on the roots and is called from inside ValidateNetwork() as well.
// Note: This function does not cache anything. BuildAndValidateSubNetwork() caches, but others don't.
//
void ComputationNetwork::FormRecurrentLoops()
{
    ExecutionGraph graph(m_allRoots);
    auto strongComponents = StrongComponents(graph);

    // In order not to change the existing behavior/naming for BrainScript,
    // let's remember the 'source' node of each strong component.
    std::vector<ComputationNodeBasePtr> componentRootNodes;
    componentRootNodes.reserve(strongComponents.size());
    for (const auto& c : strongComponents)
        componentRootNodes.push_back(c.Nodes().back());

    // Sort nodes inside the strong components in the evaluation order.
    std::function<bool(const ComputationNodeBasePtr&)> delay
        = [this](const ComputationNodeBasePtr& n) { return GetRecurrenceSteppingDirection(n) != 0; };
    EvaluationSort(graph, delay, strongComponents);

    // Update m_allSEQNodes accordingly.
    for (size_t i = 0; i < strongComponents.size(); ++i)
    {
        const auto& c = strongComponents[i];
        SEQTraversalFlowControlNode flowControlNode(i, componentRootNodes[i]);
        flowControlNode.m_nestedNodes = c.Nodes(); // TODO: make these two part of the constructor
        for (auto node : flowControlNode.m_nestedNodes)
            node->m_isPartOfLoop = true; // this is the only flag in ComputationNode that escapes FormRecurrentLoops()!
        flowControlNode.m_steppingDirection = DetermineLoopDirection(flowControlNode.m_nestedNodes);
        m_allSEQNodes.push_back(make_shared<SEQTraversalFlowControlNode>(std::move(flowControlNode)));
    }

    // Peform global sort on all nodes honoring inner strong component sorting.
    auto sortedNodes = GlobalEvaluationSort(graph, strongComponents);

    // Update global eval order in m_evalOrder.
    // TODO: Get rid of this after-the-fact patch.
    UpdateEvalOrder(nullptr, std::list<ComputationNodeBasePtr>(sortedNodes.begin(), sortedNodes.end()));

    // log the loops
    if (TraceLevel() > 0)
    {
        for (auto& iter : m_allSEQNodes)
        {
            fprintf(stderr, "\nLoop[%d] --> %ls -> %d nodes\n", (int)iter->m_loopId, iter->NodeName().c_str(), (int)iter->m_nestedNodes.size());
            size_t n = 0;
            for (auto itr = iter->m_nestedNodes.begin(); itr != iter->m_nestedNodes.end(); itr++)
            {
                if (n++ % 3 == 0)
                    fprintf(stderr, "\n");
                fprintf(stderr, "\t%ls", (*itr)->NodeName().c_str());
            }
            fprintf(stderr, "\n");
        }
    }
}

// checks whether a node is recurrent, and which direction
static int GetRecurrenceSteppingDirection(const ComputationNodeBasePtr& node)
{
    if (node->Is<IRecurrentNode>())
        return node->As<IRecurrentNode>()->GetRecurrenceSteppingDirection();
    else
        return 0;
}

// set m_steppingDirection for all loops
// TODO: Move this up to where it is used (in a separate commit since git cannot track moving and changing at the same time).
// BUGBUG: Need to extend to multi-dimensional loop directions. Use a vector<int>.
static int DetermineLoopDirection(const std::vector<ComputationNodeBasePtr>& nestedNodes)
{
    int steppingDirection = 0;

    for (auto& node : nestedNodes)
    {
        int dir = GetRecurrenceSteppingDirection(node);
        if (dir == 0) // not a recurrent node
            continue;
        if (steppingDirection == 0)
            steppingDirection = dir;
        else if (steppingDirection != dir)
            InvalidArgument("It is not allowed to have multiple different stepping directions in the same loop (loop connected to %ls %ls operation).",
                            nestedNodes.front()->NodeName().c_str(), nestedNodes.front()->OperationName().c_str());
    }

    if (steppingDirection == 0)
        LogicError("There is no recurrent node in the loop connected to %ls %ls operation.",
                   nestedNodes.front()->NodeName().c_str(), nestedNodes.front()->OperationName().c_str());
    // BUGBUG: Multiple recurrence dimensions not yet supported beyond this point.
    return steppingDirection;
}

}}}
