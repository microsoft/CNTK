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

// The methods below determine evaluation order, which is tricky in presence of recurrent loops.
// TODO: Can this be moved to a separate class?

static int GetRecurrenceSteppingDirection(const ComputationNodeBasePtr&);

// FormRecurrentLoops() -- MAIN ENTRY POINT for network recurrent-loop analysis. All other functions in this CPP are called only from this one.
// This function analysis the networks for recurrent loops present in the computation of 'rootNode.'
// This sets/updates:
//  - m_allSEQNodes
//  - ComputationNode::m_isPartOfLoop and m_loopId
//  - the cached m_evalOrders[root], reordered to make nodes belonging to the same loop consecutive. TODO: Try not to do that.
// Is often called before ValidateNetwork() on a root; will be called from inside ValidateNetwork() as well.
// This function is called for multiple nodes, e.g. eval and training criterion. I.e. it must be able to add to a previous result. E.g. it does not clear the m_visited flags at start.
// Note: This function is not lazy, i.e. not cached. BuildAndValidateSubNetwork() caches, but others don't.
// TODO: In the future, this function will not take a rootNode parameter anymore.
void ComputationNetwork::FormRecurrentLoops(const ComputationNodeBasePtr& rootNode /*or nullptr for all*/)
{
    // get the depth-first traversal order
    // Note: This is only used for resetting the state and resetting m_visitedOrder. I think we only need the set, not the order.
    const list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);

    // initialize the node state owned by us
    for (auto& node : nodes)
        node->PurgeStateForFormingRecurrentLoops();

    // determine the strongly connected cliques -> m_allSEQNodes[]
    DetermineSCCs(rootNode);
    // now we have formed all loops, with all nodes assigned to a loop or none

    // recover m_visitedOrder in original depth-first traversal order
    size_t i = 0;
    for (auto& node : nodes)
        node->m_visitedOrder = i++; // (Note: There is some redundancy between m_index and m_visitedOrder; won't fix since I rather intend to remove the whole reordering.)

    // update m_visitedOrder of all nodes that participate in a loop
    // All nodes that participate in a loop get the same m_visitedOrder value (their max).
    for (auto& iter : m_allSEQNodes)
    {
        size_t max_visitedOrderInLoop = 0;
        // TODO: I am sure there is an STL algorithm for this.
        for (auto itr : iter->m_nestedNodes)
            if (max_visitedOrderInLoop < itr->m_visitedOrder)
                max_visitedOrderInLoop = itr->m_visitedOrder;
        for (auto itr : iter->m_nestedNodes)
            itr->m_visitedOrder = max_visitedOrderInLoop;
    }

    // for reordering operation that will follow next, implant m_loopId in all nodes in all loops
    for (auto& iter : m_allSEQNodes)
    {
        for (auto& node : iter->m_nestedNodes)
        {
            node->m_isPartOfLoop = true; // this is the only flag in ComputationNode that escapes FormRecurrentLoops()!
            // TODO: ^^ We should instead remember a pointer to our loop sentinel
            node->m_loopId = iter->m_loopId;
        }
    }

    for (auto& iter : m_allSEQNodes)
    {
        list<ComputationNodeBasePtr> result;
        unordered_set<ComputationNodeBasePtr> visited;
        unordered_set<ComputationNodeBasePtr> recStack;

        // set m_indexInLoop for all nodes except recurrent nodes in all loops
        // This value is only used in the block right after this.
        for (size_t j = 0; j < iter->m_nestedNodes.size(); j++)
        {
            const auto& node = iter->m_nestedNodes[j];
            for (size_t i = 0; i < node->GetNumInputs(); i++)
            {
                if (node->Input(i)->m_loopId == node->m_loopId && GetRecurrenceSteppingDirection(node) == 0/*not a Delay node*/)
                {
                    // assert(node->Input(i)->m_indexInLoop == 0);                    // No. It seems this variable really counts the number of parents.

                    // BUGBUG: this is bumping up the m_indexInLoop, but I don't think it is initialized anywhere other than PurgeStateForFormingRecurrentLoops(). i-1?
                    node->Input(i)->m_indexInLoop++;
                }
            }
        }

        for (size_t i = 0; i < iter->m_nestedNodes.size(); i++)
        {
            ComputationNodeBasePtr node = iter->m_nestedNodes[i];
            if (visited.find(node) == visited.end() && node->m_indexInLoop == 0)
                DetermineLoopForwardOrder(visited, recStack, result, node);
        }

        // update m_nestedNodes with 'result'
        iter->m_nestedNodes.assign(result.begin(), result.end());
    }

    if (m_allSEQNodes.size() > 0)
    {
        unordered_set<ComputationNodeBasePtr> visited;

        // get set of all nodes in and outside loops hanging off rootNode
        map<int, list<ComputationNodeBasePtr>> recurrentNodes;
        list<ComputationNodeBasePtr> noRecurrentNodes;
#if 1 // will soon no longer be allowed
        if (rootNode)
            GatherLoopNodesR(rootNode, visited, recurrentNodes, noRecurrentNodes);
        else
#endif
        {
            for (const auto& rootNode : m_allRoots)
                GatherLoopNodesR(rootNode, visited, recurrentNodes, noRecurrentNodes);
        }

        auto reorderedNodes = nodes;

        // first sort by the updated m_visitedOrder, which is identical for all nodes in a loop
        reorderedNodes.sort([](const ComputationNodeBasePtr& lhs, const ComputationNodeBasePtr& rhs)
                            {
                                return lhs->m_visitedOrder < rhs->m_visitedOrder;
                            });

        ReorderLoops(reorderedNodes, recurrentNodes, noRecurrentNodes); // group nodes in loops together

        UpdateEvalOrder(rootNode, reorderedNodes);
    }

    // log the loops
    for (auto& iter : m_allSEQNodes)
    {
        fprintf(stderr, "\nLoop[%d] --> %ls -> %d nodes\n", (int) iter->m_loopId, iter->NodeName().c_str(), (int) iter->m_nestedNodes.size());
        size_t n = 0;
        for (auto itr = iter->m_nestedNodes.begin(); itr != iter->m_nestedNodes.end(); itr++)
        {
            if (n++ % 3 == 0)
                fprintf(stderr, "\n");
            fprintf(stderr, "\t%ls", (*itr)->NodeName().c_str());
        }
        fprintf(stderr, "\n");
    }

#if 1
    // now turn this into a nested network, ready for evaluation
    if (rootNode)
        FormNestedNetwork(rootNode);
#endif
}

// checks whether a node is recurrent, and which direction
static int GetRecurrenceSteppingDirection(const ComputationNodeBasePtr& node)
{
    if (node->Is<IRecurrentNode>())
        return node->As<IRecurrentNode>()->GetRecurrenceSteppingDirection();
    else
        return 0;
}

static int DetermineLoopDirection(const std::vector<ComputationNodeBasePtr>& nestedNodes);
// get the strongly connected components from the graph
// This sets index, lowLink, m_visited, and m_inStack.
void ComputationNetwork::DetermineSCCs(const ComputationNodeBasePtr& rootNode)
{
    list<ComputationNodeBasePtr> sccStack;
    size_t index = 0;
    size_t loopId = 0; // BUGBUG: I think this is currently buggy in an edge case, and not needed (use m_allSEQNodes.size() instead).
#if 1
    if (rootNode)
    {
        if (!rootNode->m_visited)
            DetermineSCCsR(rootNode, sccStack, index, loopId);
        return;
    }
#endif
    // traverse all root nodes (as if they were all children of a master root)
    for (auto& rootNode : m_allRoots)
        if (!rootNode->m_visited)
            DetermineSCCsR(rootNode, sccStack, index, loopId);
}

// (recursive part of DetermineSCCs())
void ComputationNetwork::DetermineSCCsR(ComputationNodeBasePtr cur,
                                        list<ComputationNodeBasePtr>& sccStack,
                                        size_t& index, size_t& loopId)
{
    assert(!cur->m_visited);

    // set the index (in order of visitation)
    cur->m_index = index;    // TODO: can this be used as m_visitedOrder?
    cur->m_minIndex = index; // also set m_minIndex
    index++;

    cur->m_visited = true;
    sccStack.push_back(cur);
    cur->m_inStack = true;

    // set m_minIndex to min over m_lowLinks of children
    for (int i = 0; i < cur->GetNumInputs(); i++)
    {
        if (!cur->Input(i)->m_visited)
        {
            DetermineSCCsR(cur->Input(i), sccStack, index, loopId);
            cur->m_minIndex = min(cur->m_minIndex, cur->Input(i)->m_minIndex);
        }
        else if (cur->Input(i)->m_inStack)
        {
            cur->m_minIndex = min(cur->m_minIndex, cur->Input(i)->m_minIndex);
        }
    }

    // if we closed a loop then create an entry in m_allSEQNodes
    if (cur->m_minIndex == cur->m_index) // m_minIndex is still equal to m_index, as we set it at the start of this function: we closed a loop
    {
        // gather the list of all nodes in this loop
        vector<ComputationNodeBasePtr> nestedNodes;
        // TODO: build array first in a local array. Only if succeeds, then construct the node off it.
        SEQTraversalFlowControlNode rInfo(m_allSEQNodes.size() /*loopId*/, cur);
        for (;;)
        {
            ComputationNodeBasePtr w = sccStack.back();
            sccStack.pop_back();
            w->m_inStack = false;
            nestedNodes.push_back(w);
            if (w == cur) // hit our starting point: done
                break;
        }
        // insert loop into m_allSEQNodes
        if (nestedNodes.size() > 1) // non-looped nodes are detected here as loops of size 1 --skip those
        {
            // only add to the array if the loop is not already there
            // We end up producing the same loop multiple times because:
            //  - FormRecurrentLoops() is called multiple times from different roots
            //  - depth-first traversal might have led us to enter a loop multiple times?
            // TODO: Check whether this edge case of idempotence is done correctly:
            //  - a recurrent loop with two delay nodes
            //  - two root nodes
            //  - the first root takes the first delay node's value, the second root that of the second delay node
            //    I.e. the depth-first tree traversals enter the loop at two different places (m_sourceNode).
            //  -> Are these two loops detected as identical? (determined by m_minIndex, but m_index depends on traversal from each root, so maybe not)
            bool bFound = false; // find a dup  --TODO: check whether there is an STL algorithm for this
            for (const auto& iter2 : m_allSEQNodes)
            {
                if (iter2->m_sourceNode == cur)
                {
                    bFound = true;
                    break;
                }
            }
            if (!bFound)
            {
#if 1
                if (loopId != m_allSEQNodes.size())
                    LogicError("DetermineSCCsR(): inconsistent loopId (%d) vs. m_allSEQNodes.size() (%d)", (int) loopId, (int) m_allSEQNodes.size());
                SEQTraversalFlowControlNode rInfo(m_allSEQNodes.size(), cur);
#else
                assert(loopId == m_allSEQNodes.size()); // BUGBUG: Only true if all loops are shared among roots. Fix: use m_allSEQNodes.size() instead
                SEQTraversalFlowControlNode rInfo(loopId, cur);
#endif
                // TODO: can we prove that 'cur' == nestedNodes.front()? If so, we won't need to store it separately.
                rInfo.m_nestedNodes = move(nestedNodes); // TODO: make these two part of the constructor
                rInfo.m_steppingDirection = DetermineLoopDirection(rInfo.m_nestedNodes);
                m_allSEQNodes.push_back(make_shared<SEQTraversalFlowControlNode>(move(rInfo)));
                loopId++; // and count it  TODO: may be removed
            }
        }
    }
}

// recovers the processing order within a recurrent loop
// TODO: Once we only use the nested network for recurrent traversal, this will be no longer necessary.
void ComputationNetwork::DetermineLoopForwardOrder(unordered_set<ComputationNodeBasePtr>& visited,
                                                   unordered_set<ComputationNodeBasePtr>& recStack,
                                                   list<ComputationNodeBasePtr>& nodesStack,
                                                   ComputationNodeBasePtr cur)
{
    if (visited.find(cur) == visited.end())
    {
        visited.insert(cur);
        recStack.insert(cur);

        if (GetRecurrenceSteppingDirection(cur) == 0) // recurrence stops at delay nodes
        {
            for (size_t i = 0; i < cur->GetNumInputs(); i++)
                if (cur->Input(i)->m_loopId == cur->m_loopId)
                    DetermineLoopForwardOrder(visited, recStack, nodesStack, cur->Input(i));
        }
        recStack.erase(cur);
        nodesStack.push_back(cur);
    }
    else if (recStack.find(cur) != recStack.end())
        LogicError("%ls %ls operation is part of an infinite loop that cannot be unrolled.", cur->NodeName().c_str(), cur->OperationName().c_str());
}

// traverse sub-graph feeding this node (which is a top-level node at start, e.g. training criterion) and list
//  - all nodes that participate in a loop -> recurrentResult[loopId][]
//  - all nodes that don't                 -> noRecurrentResult[]
// in order of traversal (depth-first).
// This is part of the FormRecurrentLoops() process, and only called from there from one place.
void ComputationNetwork::GatherLoopNodesR(const ComputationNodeBasePtr& node, unordered_set<ComputationNodeBasePtr>& visited,
                                          map<int, list<ComputationNodeBasePtr>>& recurrentResult,
                                          list<ComputationNodeBasePtr>& noRecurrentResult)
{
    if (visited.find(node) != visited.end())
        return; // do each node only once
    visited.insert(node);

    for (int i = 0; i < node->GetNumInputs(); i++)
        GatherLoopNodesR(node->Input(i), visited, recurrentResult, noRecurrentResult);

    if (node->m_loopId >= 0)
        recurrentResult[node->m_loopId].push_back(node);
    else
        noRecurrentResult.push_back(node);
}

// takes a list of nodes and modifies it such that all nodes of the same loop are consecutive
//  - 'nodes' is in some traversal order
//  - that order is preserved for all nodes outside loops
//  - each node that belongs to a loop is replaced by all nodes of that loop in loop order
// Called only from FormRecurrentLoops().
void ComputationNetwork::ReorderLoops(list<ComputationNodeBasePtr>& nodes,
                                      const map<int, list<ComputationNodeBasePtr>>& /*recurrentNodes*/,
                                      const list<ComputationNodeBasePtr>& /*noRecurrentNodes*/)
{
    list<ComputationNodeBasePtr> newList;

    list<ComputationNodeBasePtr> vTmp;
    list<ComputationNodeBasePtr> vRecurrentTmp;
    vector<bool> accessed(m_allSEQNodes.size(), false);
    for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
    {
        const shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(m_allSEQNodes, *nodeIter);
        if (recInfo)
        {
            int iId = recInfo->m_loopId;
            if (!accessed[iId])
            {
                newList.insert(newList.end(), recInfo->m_nestedNodes.begin(), recInfo->m_nestedNodes.end());
                accessed[iId] = true;
            }
        }
        else
        {
            newList.push_back(*nodeIter);
        }
    }

    if (vRecurrentTmp.size() > 0)
    {
        newList.insert(newList.end(), vRecurrentTmp.begin(), vRecurrentTmp.end());
        vRecurrentTmp.clear();
    }

    if (vTmp.size() > 0)
    {
        newList.insert(newList.end(), vTmp.begin(), vTmp.end());
        vTmp.clear();
    }

    nodes = newList;
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

} } }
