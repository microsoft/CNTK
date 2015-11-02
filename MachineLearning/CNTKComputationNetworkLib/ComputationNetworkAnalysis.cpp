//
// <copyright file="ComputationNetworkAnalysis.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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
    // TODO: Can this be moved to a separate class, or at least a separate CPP?

    // MAIN ENTRY POINT for network recurrent-loop analysis. All other functions below are called from this one.

    // forms the recurrent loop that 'rootNode' participates in
    // TODO: This function is not lazy, i.e. not cached. BuildAndValidateSubNetwork() caches, but others don't. Not sure why/how that's OK--won't we reassign loop ids?
    // This sets/updates:
    //  - m_recurrentInfo
    //  - ComputationNode::m_isPartOfLoop and m_loopId
    // Is often called before ValidateNetwork() on a root; will be called from inside ValidateNetwork() as well.
    // This function is called for multiple nodes, e.g. eval and training criterion. I.e. it must be able to add to a previous result. E.g. it does not clear the m_visited flags at start. This seems brittle.
    // BUGBUG: m_visited is also used by ValidateSubNetwork(). Hence, it may be in unexpected state when calling into this multiple times.
    // BUGBUG: This currently does not handle nested loops. To handle that:
    //  - loops are isolated by a ReconcileMBLayout--loop determination should see right through it, and then include everything inside
    //  - ...? Need to figure this out.
    void ComputationNetwork::FormRecurrentLoops(const ComputationNodeBasePtr& rootNode)
    {
        // determine the strongly connected cliques -> m_recurrentInfo[]
        DetermineSCCs(rootNode);

        list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, true/*set m_visitedOrder*/);

        // purge identical loops (i.e. loops that have the same source node)
        // TODO: Is this for the case that we call this function multiple times, or do the nodes of a loop generate multiple entries? Comment this.
        UniqRecurrentLoops();

        // now we have formed all loops, with all nodes assigned to a loop or none

        // update m_visitedOrder of all nodes
        // This was originally set by EnumerateNodes(), which gets called from GetEvalOrder().
        // All nodes that participate in a loop get the same m_visitedOrder value.
        for (auto & iter : m_recurrentInfo)
        {
            size_t max_visitedOrderInLoop = 0;
            for (auto itr = iter.m_recurrentNodes.begin(); itr != iter.m_recurrentNodes.end(); itr++)
                if (max_visitedOrderInLoop < (*itr)->m_visitedOrder)
                    max_visitedOrderInLoop = (*itr)->m_visitedOrder;
            for (auto itr = iter.m_recurrentNodes.begin(); itr != iter.m_recurrentNodes.end(); itr++)
                (*itr)->m_visitedOrder = max_visitedOrderInLoop;
        }

        // implant m_loopId in all nodes in all loops
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
            // it is done in the mergerecurrentloops function, but just keep the code       --TODO: why?? Why not rather verify the order?
            sort(iter->m_recurrentNodes.begin(),
                 iter->m_recurrentNodes.end(),
                 iter->m_recurrentNodes[0]->ByVisitedOrder);
 
            for (auto & node : iter->m_recurrentNodes)
            {
                node->m_isPartOfLoop = true;        // this is the only flag in ComputationNode that escapes FormRecurrentLoops()!
                node->m_loopId = iter->m_loopId;
            }
        }

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R   --TODO: is this comment correct?
            iter->m_recurrentNodesForForward.clear();

            list<ComputationNodeBasePtr> result;
            unordered_set<ComputationNodeBasePtr> visited;
            unordered_set<ComputationNodeBasePtr> recStack;

            // set m_indexInLoop for all nodes except Past/FutureValueNodes in all loops
            // This value is only used in the block right after this.
            for (size_t j = 0; j < iter->m_recurrentNodes.size(); j++)
            {
                ComputationNodeBasePtr node = iter->m_recurrentNodes[j];
                for (size_t i = 0; i < node->ChildrenSize(); i++)
                {
                    if (node->Inputs(i)->m_loopId == node->m_loopId && 
                        node->OperationName() != OperationNameOf(PastValueNode) &&
                        node->OperationName() != OperationNameOf(FutureValueNode))     // TODO: test for type RecurrentNode instead?
                    {
                        node->Inputs(i)->m_indexInLoop = node->Inputs(i)->m_indexInLoop + 1;
                    }
                }
            }

            for (size_t i = 0; i < iter->m_recurrentNodes.size(); i++)
            {
                ComputationNodeBasePtr node = iter->m_recurrentNodes[i];
                if (visited.find(node) == visited.end() && node->m_indexInLoop == 0)
                    DetermineLoopForwardOrder(visited, recStack, result, node);
            }

            // TODO: this loop seems to just reverse the list
            //       m_recurrentNodesForForward = reverse(result)
            for (size_t i = 0; i < iter->m_recurrentNodes.size(); i++)
            {
                iter->m_recurrentNodesForForward.push_back(result.front());
                result.pop_front();
            }

            iter->m_recurrentNodes = iter->m_recurrentNodesForForward;  // TODO: are they ever different?
        }

        if (m_recurrentInfo.size() > 0)
        {
            unordered_set<ComputationNodeBasePtr> visited;

            // get set of all nodes in and outside loops hanging off rootNode
            map<int, list<ComputationNodeBasePtr>> recurrentNodes;
            list<ComputationNodeBasePtr> noRecurrentNodes;
            GatherLoopNodesR(rootNode, visited, recurrentNodes, noRecurrentNodes);

            nodes.sort(ComputationNodeBase::ByVisitedOrder);        // sorts by m_visitedOrder

            ReorderLoops(nodes, recurrentNodes, noRecurrentNodes);  // group nodes in loops together

            m_cacheEvalOrders[rootNode] = nodes;
            list<ComputationNodeBasePtr> nodesForGrad = nodes;
            nodesForGrad.reverse();
            m_cacheGradientCalcOrders[rootNode] = nodesForGrad;

#ifdef DISPLAY_DEBUG
            fprintf(stderr, "Reordered nodes\n");
            for (auto itr = nodes.begin(); itr != nodes.end(); itr++)
            {
                fprintf (stderr, "%ls\n", (*itr)->NodeName().c_str() );
            }
#endif
        }
        
        DetermineLoopDirections();

        // done: clear up after ourselves
        // TODO: don't we better do that at the start as well?
        for (auto & node : nodes)
            node->PurgeStateForFormingRecurrentLoops();

        // log the loops
        for (auto & iter : m_recurrentInfo)
        {
            fprintf(stderr, "\nLoop[%d] --> %ls -> %d nodes\n", (int)iter.m_loopId, iter.m_sourceNode->NodeName().c_str(), (int)iter.m_recurrentNodes.size());
            size_t n = 0;
            for (auto itr = iter.m_recurrentNodes.begin(); itr != iter.m_recurrentNodes.end(); itr++)
            {
                if (n++ % 3 == 0)
                    fprintf(stderr, "\n");
                fprintf(stderr, "\t%ls", (*itr)->NodeName().c_str());
            }
            fprintf(stderr, "\n");
        }
    }

    // get the strongly connected components from the graph
    // This sets index, lowLink, m_visited, and m_inStack.
    void ComputationNetwork::DetermineSCCs(const ComputationNodeBasePtr& rootNode)
    {
        // notice that this graph including graphs from a parent networks if two or more networks are connected via PairNetworkNode
        list<ComputationNodeBasePtr> sccStack;
        size_t index = 0;
        size_t loopId = 0;
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
        cur->m_index = index;
        cur->m_lowLink = index; // also set m_lowLink
        index++;

        cur->m_visited = true;
        sccStack.push_back(cur);
        cur->m_inStack = true;

        if (cur->OperationName() != L"PairNetwork")     // PairNetwork is the connection from another network, so ignore its children (they are part of the other network)
        {
            // set m_lowLink to min over m_lowLinks of children
            for (int i = 0; i < cur->ChildrenSize(); i++)
            {
                if (!cur->Inputs(i)->m_visited)
                {
                    DetermineSCCsR(cur->Inputs(i), sccStack, index, loopId);
                    cur->m_lowLink = min(cur->m_lowLink, cur->Inputs(i)->m_lowLink);
                }
                else if (cur->Inputs(i)->m_inStack)
                {
                    cur->m_lowLink = min(cur->m_lowLink, cur->Inputs(i)->m_lowLink);
                }
            }
        }

        // if we closed a loop then create an entry in m_recurrentInfo
        if (cur->m_lowLink == cur->m_index)   // m_lowLink is still equal to m_index, as we set it at the start of this function: we closed a loop
        {
            RecurrentInfo rInfo;
            rInfo.m_loopId = loopId;
            rInfo.m_sourceNode = cur;               // source node is the node with low link equal to index
            for (;;)
            {
                ComputationNodeBasePtr w = sccStack.back();
                sccStack.pop_back();
                w->m_inStack = false;
                rInfo.m_recurrentNodes.push_back(w);
                if (w == cur)                       // hit our starting point: done
                    break;
            }
            if (rInfo.m_recurrentNodes.size() > 1)  // non-looped nodes are detected here as loops of size 1 --skip those
            {
                loopId++;
                rInfo.ResetInfoState();             // init for use      --TODO: does this have to be a method?
                m_recurrentInfo.push_back(rInfo);
            }
        }
    }

    // purge identical loops (i.e. loops that have the same source node)
    void ComputationNetwork::UniqRecurrentLoops()
    {
        if (m_recurrentInfo.size() <= 1)
            return;

        // uniq the m_recurrentInfo array w.r.t. m_sourceNode
        vector<RecurrentInfo> m_recurrentInfoTmp;
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)    // enumerate all loops
        {
            bool bFound = false;    // find a dup  --TODO: check whether there is an STL algorithm for this
            for (auto iter2 = m_recurrentInfoTmp.begin(); iter2 != m_recurrentInfoTmp.end(); iter2++)
            {
                if ((*iter2).m_sourceNode == (*iter).m_sourceNode)
                {
                    bFound = true;
                    break;
                }
            }
            if (!bFound)
                m_recurrentInfoTmp.push_back(*iter);
        }
        m_recurrentInfo = move(m_recurrentInfoTmp);
    }

    void ComputationNetwork::DetermineLoopForwardOrder(unordered_set<ComputationNodeBasePtr>& visited,
                                                       unordered_set<ComputationNodeBasePtr>& recStack,
                                                       list<ComputationNodeBasePtr>& nodesStack,
                                                       ComputationNodeBasePtr cur)
    {
        if (visited.find(cur) == visited.end())
        {
            visited.insert(cur);
            recStack.insert(cur);

            if (cur->OperationName() != OperationNameOf(PastValueNode) &&   // recurrence stops at delays
                cur->OperationName() != OperationNameOf(FutureValueNode))
            {
                for (size_t i = 0; i < cur->ChildrenSize(); i++)
                    if (cur->Inputs(i)->m_loopId == cur->m_loopId)
                        DetermineLoopForwardOrder(visited, recStack, nodesStack, cur->Inputs(i));
            }
            recStack.erase(cur);
            nodesStack.push_back(cur);
        }
        else
        {
            if (!(recStack.find(cur) == recStack.end()))
                LogicError("%ls %ls operation is part of an infinite loop that cannot be unrolled.", cur->NodeName().c_str(), cur->OperationName().c_str());
        }
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
            return;                                 // do each node only once
        visited.insert(node);

        for (int i = 0; i < node->ChildrenSize(); i++)
            GatherLoopNodesR(node->Inputs(i), visited, recurrentResult, noRecurrentResult);

#if 0
        //children first for function evaluation
        // TODO: This seems not necessary here. Why does this get set here?
        if (!IsLeaf())
            m_needsGradient = ChildrenNeedGradient();  //only nodes that require gradient calculation is included in gradient calculation
#endif

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
    // TODO: This could be a good place to insert sentinel nodes for nesting?
    void ComputationNetwork::ReorderLoops(list<ComputationNodeBasePtr>& nodes,
                                          const map<int, list<ComputationNodeBasePtr>>& /*recurrentNodes*/,
                                          const list<ComputationNodeBasePtr> & /*noRecurrentNodes*/)
    {
        list<ComputationNodeBasePtr> newList;

        list<ComputationNodeBasePtr> vTmp;
        list<ComputationNodeBasePtr> vRecurrentTmp;
        vector<bool> accessed(m_recurrentInfo.size(), false);
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            const RecurrentInfo * recInfo = FindInRecurrentLoops(*nodeIter);
            if (recInfo)
            {
                int iId = recInfo->m_loopId;
                if (!accessed[iId])
                {
                    newList.insert(newList.end(),
                                   m_recurrentInfo[iId].m_recurrentNodes.begin(),
                                   m_recurrentInfo[iId].m_recurrentNodes.end());
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
    void ComputationNetwork::DetermineLoopDirections()
    {
        for (auto & rInfo : m_recurrentInfo)
        {
            bool hasPastValueNode = false;
            bool hasFutureValueNode = false;

            if (rInfo.m_recurrentNodes.size() > 0)
            {
                for (size_t j = 0; j < rInfo.m_recurrentNodes.size(); j++)
                {
                    ComputationNodeBasePtr node = rInfo.m_recurrentNodes[j];

                    if (node->OperationName() == OperationNameOf(PastValueNode))
                        hasPastValueNode = true;
                    else if (node->OperationName() == OperationNameOf(FutureValueNode))
                        hasFutureValueNode = true;
                }

                if (hasPastValueNode && hasFutureValueNode)
                    InvalidArgument("It is not allowed to have both PastValue and FutureValue nodes in the same loop. How is that going to work anyway.");
                else if (!hasPastValueNode && !hasFutureValueNode)
                    LogicError("There is neither PastValue nor FutureValue nodes in the loop.");
                else if (hasPastValueNode)
                    rInfo.m_steppingDirection = +1;
                else
                    rInfo.m_steppingDirection = -1;
            }
        }
    }

}}}
