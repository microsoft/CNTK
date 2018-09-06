//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <list>
#include <set>
#include <map>
#include <functional>
#include <memory>

//
// Header only algorithms for working with execution graphs.
// The functionality is used by the computational network and the code gen evaluation engine.
// Currently this is refactoring of existing legacy code.
// In the future we should consider using Boost::Graph instead, but this will require more testing
// in order not to break current behavior/baselines.
//
namespace CNTK
{
    //
    // Interface for a directed graph.
    // The graph can be traversed starting from the graph roots (usually nodes with no successors)
    // and using the predecessor information.
    //
    // TNode is a type that represents a graph vertex. It should be:
    //     - container friendly (set,vector, map):
    //         copyable
    //         define less operator
    //     - provide ToString function (used in erroneous situations for exception messages).
    //
    template<class TNode>
    class DirectedGraph
    {
    public:
        //
        // A list of predecessors for a given node.
        //
        virtual std::vector<TNode> Predecessors(const TNode& node) const = 0;

        //
        // A list of root nodes used as starting points for graph traversal.
        // Usually these are leafs, but can also be some inner nodes.
        //
        virtual const std::vector<TNode>& Roots() const = 0;

        virtual ~DirectedGraph() {}
    };

    //
    // Forward declaration of the main algorithms that are used for defining 
    // execution order of a computational network.
    // For the actual implementation please see the end of this file.
    //

    //
    // Returns a list of nodes reachable from 'startNodes' in the post-order.
    // Firstly it visits all predecessors of a starting node, then the node itself.
    // Starting nodes are evaluated in order and all nodes are visited exactly once.
    //
    template<class TNode>
    inline std::list<TNode> PostOrderTraversal(const DirectedGraph<TNode>& graph, const std::vector<TNode>& startNodes);

    //
    // Class representing a strongly connected component.
    //
    template<class TNode>
    struct StrongComponent final
    {
        StrongComponent(const std::vector<TNode>&& nodes) :
            m_nodes(std::move(nodes))
        {}

        //
        // Returns a list of nested nodes.
        //
        const std::vector<TNode>& Nodes() const
        {
            return m_nodes;
        }

        //
        // Updates the order of nested nodes.
        //
        void UpdateNodeOrder(std::vector<TNode>&& nodes)
        {
            assert(std::set<TNode>(m_nodes.begin(), m_nodes.end()) == std::set<TNode>(nodes.begin(), nodes.end()));
            m_nodes = std::move(nodes);
        }

        //
        // Checks if the node belongs to the component.
        //
        bool Contains(const TNode& node) const
        {
            return std::find(m_nodes.begin(), m_nodes.end(), node) != m_nodes.end();
        }

    private:
        std::vector<TNode> m_nodes;
    };

    //
    // Returns a list of strongly connected components in the graph.
    //
    template<class TNode>
    std::vector<StrongComponent<TNode>> StrongComponents(const DirectedGraph<TNode>& graph);

    //
    // Sorts nodes inside strong components for evaluation.
    // The order is defined as follows:
    //  - take a connected component
    //  - find all its nodes that feed only into delay nodes, these nodes become new roots
    //  - perform the topological sort starting at these roots and breaking at delay nodes
    //  - update the component with the reordered list of sorted nodes
    //
    template<class TNode>
    void EvaluationSort(const DirectedGraph<TNode>& graph, std::function<bool(const TNode&)> delay, std::vector<StrongComponent<TNode>>& strongComponents);

    //
    // Sorts all nodes of the graph in the evaluation order given by the root nodes.
    // Strongly connected componentes should be already sorted using EvaluationSort function.
    //
    template<class TNode>
    std::vector<TNode> GlobalEvaluationSort(const DirectedGraph<TNode>& graph, const std::vector<StrongComponent<TNode>>& strongComponents);

    //
    // Actual implementation of the above functions.
    //
    namespace Internal
    {
        // Functions from this namespace should not be used directly.

        //
        // Function performs post-order traversal of the graph and returns
        // collected nodes.
        //
        template<class TNode>
        static void PostOrderTraversalImpl(const DirectedGraph<TNode>& graph, const TNode& node, std::set<TNode>& visited, std::list<TNode>& result)
        {
            if (visited.find(node) != visited.end())
                return;

            visited.insert(node);
            for (const auto& p : graph.Predecessors(node))
                PostOrderTraversalImpl(graph, p, visited, result);
            result.push_back(node);
        }

        //
        // Helper struct used in StrongComponents function.
        // Contains additional information needed for Tarjan algorithm for
        // performing strong component search.
        // Same as in wikipedia,
        // please see https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        //
        struct StrongComponentNodeState final
        {
            bool m_visited{ false };   // flag indicating whether the node was visited
            int m_index{ -1 };         // index denoting order in which nodes were visited
            int m_minIndex{ -1 };      // min of m_index over all nodes within a single component
            bool m_inStack{ false };   // flag indicating whether the node is still on the stack
        };

        //
        // Recursive implementation of the Tarjan algorithm for finding all stronly connected
        // components.
        //
        template<class TNode>
        void StrongComponentsImpl(
            const DirectedGraph<TNode>& graph,
            const TNode& node,
            std::stack<TNode>& nodeStack,
            int& index,
            std::map<TNode, Internal::StrongComponentNodeState>& state,
            std::vector<StrongComponent<TNode>>& strongComponents)
        {
            assert(!state[node].m_visited);

            // set the index (in order of visitation)
            // Each node is assigned a unique integer m_index, which numbers the nodes consecutively in the order in which they are discovered.
            state[node].m_index = index;
            state[node].m_minIndex = index;
            index++;

            state[node].m_visited = true;

            // The nodes are placed on the stack in the order in which they are visited.
            // When the depth-first search recursively explores a node 'node' and its descendants,
            // those nodes are not all necessarily popped from the stack when this recursive call returns.
            // The crucial invariant property is that a node remains on the stack after exploration if and only if it has a path to some node earlier on the stack.
            // At the end of the call that explores 'node' and its descendants, we know whether 'node' itself has a path to any node earlier on the stack.
            // If so, the call returns, leaving 'node' on the stack to preserve the stack invariant.
            // If not, then 'node' must be the root of its strongly connected component, which consists of 'node' together with any later nodes on the stack
            // (such nodes all have paths back to 'node' but not to any earlier node,
            // because if they had paths to earlier nodes then 'node' would also have paths to earlier nodes which is false).
            // This entire component is then popped from the stack and returned, again preserving the invariant. [Wikipedia]
            nodeStack.push(node);
            state[node].m_inStack = true;

            // set m_minIndex to min over m_minIndex of children
            // m_minIndex (lowlink in Tarjan's notation) represents (roughly speaking) the smallest index of any node known to be reachable from 'node', including 'node' itself. [Wikipedia]
            for (const auto& predecessor : graph.Predecessors(node))
            {
                if (!state[predecessor].m_visited)
                {
                    // predecessor w has not yet been visited; recurse on it
                    StrongComponentsImpl(graph, predecessor, nodeStack, index, state, strongComponents);
                    state[node].m_minIndex = std::min(state[node].m_minIndex, state[predecessor].m_minIndex);
                }
                else if (state[predecessor].m_inStack)
                {
                    // successor w is in stack S and hence in the current SCC
                    // NOTE! This line is actually different from the BS algorithm
                    state[node].m_minIndex = std::min(state[node].m_minIndex, state[predecessor].m_index);
                }
            }

            // if 'node' is a root node, then we closed a loop.
            // 'node' must be left on the stack if m_minIndex < m_index,
            // whereas it must be removed as the root of a strongly connected component if m_minIndex == m_index.
            // m_minIndex is computed during the depth-first search from 'node' (above), as this finds the nodes that are reachable from 'node'. [Wikipedia]
            assert(state[node].m_minIndex <= state[node].m_index);
            if (state[node].m_minIndex == state[node].m_index) // m_minIndex is still equal to m_index, as we set it at the start of this function: we closed a loop
            {
                // gather the list of all nodes in this loop
                std::vector<TNode> nestedNodes;

                for (;;)
                {
                    TNode current = nodeStack.top();
                    nodeStack.pop();

                    state[current].m_inStack = false;
                    nestedNodes.push_back(current);

                    if (current == node) // hit our starting point: done
                        break;
                }

                // not a real loop. In degenerate situation it could be that the delay
                // feeds directly into itself though, but then its still just returns the same value
                // so can be evaluated in a topological sort order.
                if (nestedNodes.size() <= 1)
                    return;

                strongComponents.emplace_back(std::move(nestedNodes));
            }
        }

        //
        // Helper function for EvaluationSort of nodes inside connected components.
        // Creates the processing order within a recurrent loop.
        // Re-traverses the set of nodes between 'node' and the first delay node on each sub-graph.
        //
        template<class TNode>
        void LoopEvaluationSort(std::set<TNode>& visited,
            std::set<TNode>& nodesOnThePathFromRoot,
            std::vector<TNode>& result,
            TNode node,
            const DirectedGraph<TNode>& graph,
            const StrongComponent<TNode>& component,
            std::function<bool(const TNode&)> delay)
        {
            if (visited.find(node) != visited.end())
            {
                // Check if we have a loop without a delay node.
                if (nodesOnThePathFromRoot.find(node) != nodesOnThePathFromRoot.end())
                    LogicError("Node %ls is part of an infinite loop that cannot be unrolled.", ToString(node).c_str());
                return;
            }

            visited.insert(node);
            nodesOnThePathFromRoot.insert(node);

            // Recurse if not a delay, stop when see a recurrence.
            if (!delay(node))
            {
                for (const auto& p : graph.Predecessors(node))
                {
                    if (component.Contains(p))
                        LoopEvaluationSort(visited, nodesOnThePathFromRoot, result, p, graph, component, delay);
                }
            }

            nodesOnThePathFromRoot.erase(node);
            result.push_back(node);
        }
    }

    //
    // Returns a list of nodes reachable from 'startNodes' in the post-order traversal.
    // For more information please see the forward declaration at the beginning of the file.
    //
    template<class TNode>
    inline std::list<TNode> PostOrderTraversal(const DirectedGraph<TNode>& graph, const std::vector<TNode>& startNodes)
    {
        std::list<TNode> result;
        std::set<TNode> visited;
        for (const auto& node : startNodes)
            Internal::PostOrderTraversalImpl(graph, node, visited, result);
        return result;
    }

    //
    // Returns a list of strongly connected components using Tarjan algorithm.
    //
    template<class TNode>
    std::vector<StrongComponent<TNode>> StrongComponents(const DirectedGraph<TNode>& graph)
    {
        std::map<TNode, Internal::StrongComponentNodeState> state;
        std::vector<StrongComponent<TNode>> result;
        std::stack<TNode> nodeStack;
        int index = 0;
        for (auto& root : graph.Roots())
        {
            if (state[root].m_visited)
                continue;
            StrongComponentsImpl(graph, root, nodeStack, index, state, result);
        }
        return result;
    }

    //
    // Sorts nodes inside strongly connected components according to their evaluation order,
    // breaking loops at the delay nodes.
    //
    // Used algorithm goes as follows:
    //  - take a connected component
    //  - find all its nodes that feed only into delay nodes, these nodes become new roots
    //  - perform the topological sort starting at these roots and breaking at delay nodes
    //  - update the component with the reordered list of sorted nodes
    //
    template<class TNode>
    inline void EvaluationSort(const DirectedGraph<TNode>& graph, std::function<bool(const TNode&)> delay, std::vector<StrongComponent<TNode>>& strongComponents)
    {
        for (auto& component : strongComponents)
        {
            // Get all nodes that only have a delay child, these
            // will become new roots for evaluation.
            const auto& nestedNodes = component.Nodes();
            std::set<TNode> newRoots(nestedNodes.begin(), nestedNodes.end());
            for (const auto& node : nestedNodes)
            {
                if (delay(node))
                    continue;

                for (const auto& predecessor : graph.Predecessors(node))
                {
                    if (component.Contains(predecessor))
                        newRoots.erase(predecessor);
                }
            }

            // Perform the topological sort stopping at delay nodes
            // to break the loops.
            std::vector<TNode> reordered;
            reordered.reserve(component.Nodes().size());

            std::set<TNode> visited;
            for (const auto& root : newRoots)
            {
                if (visited.find(root) != visited.end())
                    continue;

                std::set<TNode> checkInfinity;
                Internal::LoopEvaluationSort(visited, checkInfinity, reordered, root, graph, component, delay);
            }

            // Update the component.
            component.UpdateNodeOrder(std::move(reordered));
        }
    }

    //
    // Sorts all nodes of the graph in the evaluation order given by the root nodes.
    // Strongly connected components should be already sorted using EvaluationSort function.
    //
    template<class TNode>
    inline std::vector<TNode> GlobalEvaluationSort(const DirectedGraph<TNode>& graph, const std::vector<StrongComponent<TNode>>& strongComponents)
    {
        auto nodes = PostOrderTraversal(graph, graph.Roots());
        if (strongComponents.empty())
            return std::vector<TNode>(nodes.begin(), nodes.end());

        // Now we need to collect all strong components and the rest of the nodes
        // in the global evaluation order.

        // Prepare additional structure that contains the number of nodes per
        // component.
        std::map<decltype(strongComponents.begin()), size_t> componentToNodeCount;
        for (auto i = strongComponents.begin(); i != strongComponents.end(); ++i)
            componentToNodeCount.insert(std::make_pair(i, i->Nodes().size()));

        // Strong components should already be sorted in a proper evaluation order.
        // The whole strong component gets evaluated on its last node position in the global
        // topological order list('nodes').
        std::vector<TNode> result;
        result.reserve(nodes.size());
        for (const auto& node : nodes)
        {
            auto component = std::find_if(strongComponents.begin(), strongComponents.end(),
                [&node](const StrongComponent<TNode>& c) { return c.Contains(node); });
            if (component == strongComponents.end())
            {
                result.push_back(node);
            }
            else
            {
                // Check if the last node of the component in the global topological
                // sort order. If that is the case, insert all nodes of the component.
                assert(componentToNodeCount[component] > 0);
                if (--componentToNodeCount[component] == 0)
                    result.insert(result.end(), component->Nodes().begin(), component->Nodes().end());
            }
        }
        return result;
    }
}