//
// <copyright file="ComputationNetworkEvaluation.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "RecurrentNodes.h"
#include <string>
#include <vector>
#include <list>
#include <set>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    // This source file contains methods related to evaluation (forward prop, backprop), network validation, and matrix memory allocation (memory sharing).

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // MAIN ENTRY POINT for evaluating one minibatch (forward prop)
    // TODO: pass a set of nodes instead of only one
    // TODO: rename to ForwardProp()? To make it very clear?
    // This calls EvaluateThisNode() on all nodes in order of data flow through the network.
    // By default, the network is applied concurrently on all frames in a minibatch in parallel (PAR mode, a "map" operation)
    // Recurrent loops deviate:
    //  - a recurrent loop is the loop of nodes that make up computation for one time step (e.g. Times -> Plus -> Sigmoid -> Delay)
    //  - these must be executed frame by frame rather than as a map
    //  - such a loop is treated as if they were a little nested network; this is done inside SEQTraversalFlowControlNodes
    //  - these little nested networks are defined in m_recurrentInfo[]
    void ComputationNetwork::Evaluate(const ComputationNodeBasePtr & rootNode)
    {
        // caller must call BuildAndValidateSubNetwork() before
        // TODO: Some places are hard to fix, e.g. encoder-decoder best-path functions. Those may be broken; this message will tell you.
        if (!BuiltAndValidatedSubNetwork(rootNode))
            LogicError("Evaluate for node %ls %ls: BuildAndValidateSubNetwork() has not been called on this node.", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());

        // traverse all nodes in the pre-determined evaluation order
        GetOuterLoopNode(rootNode)->EvaluateThisNode(FrameRange(nullptr));
    }

    // MAIN ENTRY POINT for evaluation followed by gradient computation (forward prop then back prop)
    // TODO: pass a set of nodes instead of only one?
    // TODO: remove Evaluate() from here, instead call it at call site, and in here merely check whether everything is computed already
    template<class ElemType>
    void ComputationNetwork::ComputeGradient(const ComputationNodeBasePtr rootNode,         // training criterion to compute the gradients for
                                             bool bResetToOne,                              // true if reset the gradient of rootnode to 1.0  --This is the default.
                                             const Matrix<ElemType>* rootGradientInitValue, // if given then this is the starting gradient from the top
                                             bool bClearGradient,                           // if false then gradients are not cleared  --TODO: When does that happen?
                                             bool resetTimeStampAfterComputation)
    {
        // run forward pass first for criterion node
        // The actual call pattern is
        //  - Evaluate() for eval nodes
        //  - ComputeGradient() for the training criterion
        // I.e. we must call Evaluate() inside here as well, but it will typically only evaluate the training criterion bits because the eval nodes already require most of the network to be computed.
        Evaluate(rootNode);

        // TODO: comment what the purpose/condition of this is
        if (bClearGradient)
            ClearGradientForAllNodes(rootNode);     // reset m_completedGradient, which is meant to make sure each gradient is computed only once. Only used for recurrence, actually.

        // TODO: do a runtime check for float vs. double. Also use the Is/AsPtr macros
        // The normal case is with the top root with a scalar gradient value of 1.0. This assumes a single and closure network. 
        // Allowing to not initialize to 1 allows network to be open to accept gradients from somewhere.
        // TODO: aren't these two mechanisms mutually exclusive?
        if (bResetToOne)
        {
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().Resize(1, 1);   // TODO: make this a function of ComputationNode; but first need to get rid of Matrix<ElemType> here, or make it a local template parameter
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(1);    // TODO: is there not a single SetValue() call that also takes dimensions?
        }

        if (rootGradientInitValue != nullptr)   // user-specified gradient to start with
            dynamic_pointer_cast<ComputationNode<ElemType>>(rootNode)->GradientValues().SetValue(*rootGradientInitValue);

        GetOuterLoopNode(rootNode)->ComputeGradientForChildren(FrameRange(nullptr), true, true);

        // Since we allow sharing of the matrix for function value and gradient value. the function values are destroyed
        // after gradient computation and need to be recomputed. This is indicated by the timestamp updated using this function
        // resetTimeStampAfterComputation is by default false because ComputeGradient in normal case is followed by new batch of input
        if (resetTimeStampAfterComputation)
            ResetEvalTimeStamp();
    }

    template void ComputationNetwork::ComputeGradient<float>(const ComputationNodeBasePtr rootNode, bool bResetToOne, const Matrix<float>* rootGradientInitValue, bool bClearGradient, bool resetTimeStampAfterComputation);
    template void ComputationNetwork::ComputeGradient<double>(const ComputationNodeBasePtr rootNode, bool bResetToOne, const Matrix<double>* rootGradientInitValue, bool bClearGradient, bool resetTimeStampAfterComputation);

    // -----------------------------------------------------------------------
    // PARTraversalFlowControlNode methods -- implements PAR traversal
    //
    // This implements an outer loop over non-recurrent nodes, where each node can be
    // executed in PAR mode; that is, all samples are independent and allow for
    // concurrent computation in bulk CUDA launches.
    // -----------------------------------------------------------------------

    ComputationNetwork::PARTraversalFlowControlNode::PARTraversalFlowControlNode(/*const*/ std::vector<shared_ptr<SEQTraversalFlowControlNode>> & recurrentInfo, const std::list<ComputationNodeBasePtr> & allNodes/*must be in eval order*/)
    {
        // traverse the network in evaluation order and create a new list that replaces all recurrence by a SEQTraversalFlowControlNode
        set<shared_ptr<IComputationNode>> loopsSeen;  // for consistency check only
        for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end(); )
        {
            shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(recurrentInfo, *nodeIter);   // check if this node participates in a recurrent loop
            if (recInfo)            // node is part of a SEQ loop: gather all of them. The nodes must be consecutive in 'allNodes'
            {
                // instead of the node itself, include the sentinel SEQTraversalFlowControlNode in our list
                m_nestedNodes.push_back(recInfo);
                // and verify that we only encountered the loop once (all nodes should have been consecutive)
                if (!loopsSeen.insert(recInfo).second)
                    LogicError("PARTraversalFlowControlNode: members of loop %ls are not consecutive in node list.", recInfo->NodeName().c_str());
                // consume all nodes that are part of the same loop (they are all consecutive)
                while (nodeIter != allNodes.end() && (*nodeIter)->IsPartOfLoop() && FindInRecurrentLoops(recurrentInfo, *nodeIter) == recInfo)
                    nodeIter++;
            }
            else                    // regular top-level node (non-looping, PAR)
            {
                m_nestedNodes.push_back(*nodeIter);
                nodeIter++;         // and consume this node
            }
        }
    }
    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::EvaluateThisNode(const FrameRange & frameRange) /*override*/
    {
        for (auto & node : m_nestedNodes)
        {
            if (node->IsFuncValueOlderThanInputs())
            {
                auto recInfo = dynamic_pointer_cast<SEQTraversalFlowControlNode>(node);
                if (recInfo)
                    assert(recInfo->m_sourceNode->GetMBLayout() == node->GetMBLayout());

                node->OnEvaluateBeginIteration();
                node->EvaluateThisNode(frameRange.WithLayout(node->GetMBLayout()));
                node->OnEvaluateEndIteration();

                node->UpdateEvalTimeStamp();
            }
#ifdef _DEBUG
            else if (node)
                node->OnEvaluateEndIteration();  // HACK: performs NaN check, but does nothing else
#endif
        }
    }

    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::ComputeGradientForChildren(const FrameRange & frameRange, bool childrenInThisLoop, bool childrenInOuterLoop) /*override*/
    {
        childrenInThisLoop, childrenInOuterLoop;    // TODO: think through what these mean when coming from PAR mode
        // process nodes in pre-determined order
        for (auto pnode = m_nestedNodes.rbegin(); pnode != m_nestedNodes.rend(); pnode++)   // iterate backwards over evaluation order
        {
            auto & node = *pnode;

            node->OnComputeGradientBeginIteration();
            node->ComputeGradientForChildren(frameRange.WithLayout(node->GetMBLayout()), true/*childrenInThisLoop*/, true/*childrenInOuterLoop*/);
            node->OnComputeGradientEndIteration();
        }
    }
    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::RequestMatricesBeforeEval(MatrixPool& matrixPool) /*override*/ { }
    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::ReleaseMatricesAfterEval(MatrixPool& matrixPool) /*override*/ { }
    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::AllocateGradientMatricesForChildren(MatrixPool& matrixPool) /*override*/ { }
    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::RequestMatricesBeforeGradientComp(MatrixPool& matrixPool) /*override*/ { }
    /*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool) /*override*/ { }

    // -----------------------------------------------------------------------
    // SEQTraversalFlowControlNode methods -- implements SEQ traversal (loop unrolling)
    //
    // While PAR mode processes all samples in the MB independently, and thus in
    // PARallel, SEQ mode is to honor sequential dependencies. As such, it
    // unrolls the loop over time steps and runs the network once per time step.
    // -----------------------------------------------------------------------

    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::OnEvaluateBeginIteration() /*override*/
    {
        // take the opportunity to check that layout is shared by all nodes in the loop
        // TODO: we should do this in a constructor.
        for (auto & node : m_nestedNodes)
        {
            if (node->GetMBLayout() != GetMBLayout())
                LogicError("Evaluate: all nodes inside a recurrent loop must have a layout that is identical; mismatch found for nodes '%ls' vs. '%ls'",
                            node->NodeName().c_str(), m_nestedNodes[0]->NodeName().c_str());
        }

        // tell all that loop is about to commence
        for (auto & node : m_nestedNodes)
            node->OnEvaluateBeginIteration();
    }

    // evaluation of a SEQTraversalFlowControlNode FlowControlNode
    // This evaluates all nodes in this FlowControlNode in SEQ mode: process the loop frame by frame in a nested loop.
    // This is where the time axis changes.
    // TODO: Once we do nested loops, then the FrameRange argument to this will refer to the outer loop.
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::EvaluateThisNode(const FrameRange &) /*override*/
    {
        // get layout associated with this loop
        // All nodes share the same layout.
        assert(GetMBLayout() == m_nestedNodes[0]->GetMBLayout());

        // for every time step run through all nodes in this particular loop (treat the loop like a little ComputationNetwork)
        // Note: Currently, this is limited to linear-time loops. But nothing stops the iteration below to, e.g., be a 2D iteration over an image
        // if we implement an according FrameRangeIteration.
        FrameRangeIteration range(GetMBLayout(), m_steppingDirection);
        for (auto t = range.begin(); t != range.end(); t++)
        {
            for (auto & node : m_nestedNodes)
            {
                node->EvaluateThisNode(t);
                node->UpdateEvalTimeStamp();
            }
        } 
    }

    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::OnEvaluateEndIteration() /*override*/
    {
        // tell all that loop is done  --e.g. PastValueNode will capture its state for BPTT processing
        for (auto & node : m_nestedNodes)
            node->OnEvaluateEndIteration();
    }

    // called before first iteration step of ComputeGradient()
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::OnComputeGradientBeginIteration() /*override*/
    {
        for (auto & node2 : m_nestedNodes)
            node2->OnComputeGradientBeginIteration();
    }

    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::ComputeGradientForChildren(const FrameRange &, bool childrenInThisLoop, bool childrenInOuterLoop) /*override*/
    {
        childrenInThisLoop, childrenInOuterLoop;    // TODO: think through what these mean when coming from PAR mode
        const auto & recurrentNodes = m_nestedNodes;       // BUGBUG: -ForForward?? Does this mean we can remove non-ForForward?
        auto pMBLayout = recurrentNodes[0]->GetMBLayout();
        FrameRangeIteration range(pMBLayout, m_steppingDirection);
        for (auto t = range.rbegin(); t != range.rend(); t++)   // note: reverse iteration
        {
            for (auto nodeIter2 = recurrentNodes.rbegin(); nodeIter2 != recurrentNodes.rend(); ++nodeIter2)
            {
                auto & node2 = *nodeIter2;
                node2->ComputeGradientForChildren(t, true/*childrenInThisLoop*/, false/*childrenInOuterLoop*/);
                // The above flags tell ComputeGradientForChildren() to skip back-propagation from inside a node into
                // a node that is outside the loop, which is done later in OnComputeGradientEndIteration() in PAR mode.
            }
        }
    }

    // called after last iteration step of ComputeGradient()
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::OnComputeGradientEndIteration() /*override*/
    {
        // The following loop handles the case that a node inside the loop back-propagates a gradient into a node outside of the loop.
        // For efficiency, we perform this outside the loop in PAR mode. E.g., in one LSTM speech setup, we measured 12..14% overall speed-up.
        for (auto nodeIter2 = m_nestedNodes.rbegin(); nodeIter2 != m_nestedNodes.rend(); ++nodeIter2)
        {
            auto & node2 = *nodeIter2;
            node2->ComputeGradientForChildren(FrameRange(m_nestedNodes[0]->GetMBLayout()), false/*childrenInThisLoop*/, true/*childrenInOuterLoop*/);
        }

        // tell all nodes we are done for this iteraTion
        for (auto & node2 : m_nestedNodes)
            node2->OnComputeGradientEndIteration();
    }

    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::RequestMatricesBeforeEval(MatrixPool& matrixPool) /*override*/
    {
        for (auto & nodeLoopIter : m_nestedNodes)
            nodeLoopIter->RequestMatricesBeforeEval(matrixPool);
    }
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::ReleaseMatricesAfterEval(MatrixPool& matrixPool) /*override*/ { }
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::AllocateGradientMatricesForChildren(MatrixPool& matrixPool) /*override*/
    {
        // TODO: should we deallocate in opposite order?
        for (auto nodeIter = m_nestedNodes.rbegin(); nodeIter != m_nestedNodes.rend(); ++nodeIter)
        {
            (*nodeIter)->AllocateGradientMatricesForChildren(matrixPool);
        }
    }
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::RequestMatricesBeforeGradientComp(MatrixPool& matrixPool) /*override*/ { }
    /*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool) /*override*/
    {
        for (auto nodeIter = m_nestedNodes.rbegin(); nodeIter != m_nestedNodes.rend(); ++nodeIter)
        {
            if ((*nodeIter)->NeedGradient())
                (*nodeIter)->ReleaseMatricesAfterGradientComp(matrixPool);
        }
    }

    // find if node is part of a recurrent loop; and return the loop id
    // If found then return a pointer to the list of nodes of this loop.
    /*static*/ shared_ptr<ComputationNetwork::SEQTraversalFlowControlNode> ComputationNetwork::FindInRecurrentLoops(/*const*/ std::vector<std::shared_ptr<SEQTraversalFlowControlNode>> & recurrentInfo, const ComputationNodeBasePtr& node)
    {
        // look in all recurrent loops of the network
        // TODO: Check for IsPartOfLoop(). Also why not store the loop id in the node for direct lookup?
        for (auto & iter : recurrentInfo)
            if (std::find(iter->m_nestedNodes.begin(), iter->m_nestedNodes.end(), node) != iter->m_nestedNodes.end())  // TODO: should this loop need to be a method of SEQTraversalFlowControlNode?
                return iter;
        return nullptr;  // not part of a recurrent loop
    }

    // check if any of the nodes in the recurrence IsFuncValueOlderThanInputs(), with exception of delay nodes for which this check would fail and can be skipped
    // TODO: Would it be sufficient to check against our own time stamp, so that we can use a unified time-stamping mechanism? Then we'd not need this special check for delayed nodes; just check all inputs against our own time stamp.
    bool ComputationNetwork::SEQTraversalFlowControlNode::IsFuncValueOlderThanInputs() const
    {
        for (auto & ptr : m_nestedNodes)
        {
            if (ptr->IsFuncValueOlderThanInputs() &&
                ptr->OperationName() != OperationNameOf(PastValueNode) &&
                ptr->OperationName() != OperationNameOf(FutureValueNode))
            {
                return true;
            }
        }
        return false;
    }

    // TODO: do this on PARTraversalFlowControlNode
    void ComputationNetwork::ResetEvalTimeStamp()
    {
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            nodeIter->second->ResetEvalTimeStamp();
    }

    /*static*/void ComputationNetwork::UpdateEvalTimeStamps(const vector<ComputationNodeBasePtr> & nodes)
    {
        for (size_t i = 0; i<nodes.size(); i++)
            nodes[i]->UpdateEvalTimeStamp();
    }

    // for debugging
    void ComputationNetwork::PrintComputationTree(const ComputationNodeBasePtr& rootNode,
                                                  const bool forwardCompute,
                                                  const bool printMatrices)
    {
        std::list<ComputationNodeBasePtr> nodes;
        if (forwardCompute)
        {
            fprintf(stderr, "\n\nPrinting Forward Computation Node Order ... \n");
            nodes = GetEvalOrder(rootNode, false);
        }
        else
        {
            fprintf(stderr, "\n\nPrinting Gradient Computation Node Order ... \n");
            nodes = GetGradientCalcOrder(rootNode);
        }

        if (nodes.size() == 0)
        {
            fprintf(stderr, "\n$$$$ EMPTY !!!!!\n");
            return;
        }

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            ComputationNodeBasePtr node = (*nodeIter);
            node->PrintSelf(printMatrices);
        }
    }

    // -----------------------------------------------------------------------
    // validation
    // -----------------------------------------------------------------------

    // ValidateNetwork() - Validate the entire network
    // This calls ValidateNetowrk(Node) for all output nodes.
    // This is used after loading or for dumping the network.
    void ComputationNetwork::ValidateNetwork(bool allowFragment, const bool bAllowNoCriterion)
    {
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");

#if 1   // If it is not done here, it will causea crash. But it really only belongs into StartEvaluationMinibatchLoop()
        // TODO: allocation does not belong here. This is called e.g. after loading. Memory should be allocated only when actually evaluating.
        // TODO: move into StartEvaluateMinibatchLoop(), but that is called for output nodes individually--can the process handle that?
        AllocateAllEvalMatrices(EvaluationNodes(), OutputNodes(), FinalCriterionNodes());
#endif
        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (ComputationNodeBasePtr & node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
#ifdef _DEBUG
                PrintComputationTree(node, false);
#endif
                //SetActualMiniBatchSizeFromFeatures();
                ValidateSubNetwork(node);
            }
        }
        else if (bAllowNoCriterion == true)
        {
            // do nothing
        }
        else if (!allowFragment)
            RuntimeError("No Criterion nodes specified");

        // now output nodes
        if (OutputNodes().size() > 0)
        {
            for (ComputationNodeBasePtr node : OutputNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateSubNetwork(node);
            }
        }
        else if (!allowFragment)
            RuntimeError("No Output nodes specified");

        // now evaluation nodes
        if (EvaluationNodes().size() > 0)
        {
            for (ComputationNodeBasePtr node : EvaluationNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                ValidateSubNetwork(node);
            }
        }
    }

    // validate sub-network needed to evalute a specific output node
    // This calls Validate() on every node in evaluation order (allowing to propagate things forwards through the net).
    // This is called lazily but once only per node until next ClearCache().
    // This also sets up MBLayout links.
    // TODO: I can't see a clear pattern when ClearCache() is called. E.g. at the start of each epoch? Or never in normal operation (init only at construction)?
    // Note: under some circumstances, one must call FormRecurrentNodes() on this node before calling this. TODO: Not clear which ones.
    // TODO: ^^ is this really needed? Can we just call it inside?
    void ComputationNetwork::ValidateSubNetwork(const ComputationNodeBasePtr& rootNode)
    {
        // set up MBLayout links of inputs (all others get propagated upwards through Validate())
        // TODO: Once we support mismatching layouts, this will be more involved. For now, everything shares the one layout that the Network knows about.
        for (auto node : InputNodes(rootNode))
        {
            node->LinkToMBLayout(m_pMBLayout);
            // handle the special case of being validated before reading a minibatch
            // In that case, the layout is empty. We set up a dummy layout to match the first InputValue.
            // TODO: This is a stop-gap. We need a better-controlled way of when what gets validated.
            if (m_pMBLayout->GetNumCols() == 0)
                m_pMBLayout->Init(1, node->GetNumCols(), false);
        }

        // we call all nodes' Validate() in order to validate, that is, set up MBLayout and FunctionValues dimension
        // A problem is that recurrent loops may require partial validation.
        // Nodes validated on partial input (i.e. some children not yet validated) will be revisited.
        const auto & nodes = GetEvalOrder(rootNode, false);

        for (auto & node : nodes)
        {
            node->m_visited = false;
            node->m_needsGradient = node->IsParameterUpdateRequired();  // these get propagated upwards in the following
        }

        // loop and validate until we are done
        // steps:
        //  - validate (not final)          // not final means no dimension checks
        //    Keep going through the list until all nodes have been validated and all inputs have been validated as well.
        //  - validate (final)              // final means consistency checks
        //    Fail if any change during this stage.
        size_t pass = 0;
        size_t toValidate = nodes.size();
        while (toValidate > 0)
        {
            pass++;
            fprintf(stderr, "\n\nValidating for node %ls. %d nodes to process in pass %d.\n", rootNode->NodeName().c_str(), (int)toValidate, (int)pass);
            ValidateNodes(nodes, false/*isFinalValidationPass*/, toValidate);
        }
        fprintf(stderr, "\n\nValidating for node %ls, final verification.\n", rootNode->NodeName().c_str());
        ValidateNodes(nodes, true/*isFinalValidationPass*/, toValidate);
        if (toValidate != 0)
            LogicError("ValidateSubNetwork: ValidateNodes(true) unexpectedly returned with work left to do.");

        // propagate some info to SEQTraversalFlowControlNode
        // TODO: In the future we should validate not on the flat list but the PARTraversalFlowControlNode structure. Then this will be unnecessary.
        for (auto & recInfo : m_recurrentInfo)
        {
            auto & node = recInfo->m_sourceNode;
            recInfo->m_needsGradient = node->m_needsGradient;
            recInfo->LinkToMBLayout(node->GetMBLayout());
        }

        for (auto & node : nodes)
        {
            // nodes must output non-zero dimensional data, otherwise assume user error
            if (node->GetNumRows() == 0 && (node->GetMBLayout() || node->GetNumCols() == 0))
                RuntimeError("%ls operation has 0 elements", node->NodeName().c_str());
        }
        fprintf(stderr, "\n\n");

        // logging the non-default-layout nodes
        vector<ComputationNodeBasePtr> nonDefaultNodes;
        for (auto node : nodes)
        {
            if (!(node->GetMBLayout() == m_pMBLayout))
                nonDefaultNodes.push_back(node);
        }
        if (!nonDefaultNodes.empty())
        {
            fprintf(stderr, "%d out of %d nodes do not share the minibatch layout with the input data.\n\n", (int)nonDefaultNodes.size(), (int)nodes.size());
            //for (auto node : nonDefaultNodes)
            //    fprintf(stderr, "    %ls\n", node->NodeName().c_str());
            //fprintf(stderr, "\n\n");
        }
    }

    void ComputationNetwork::ValidateNodes(list<ComputationNodeBasePtr> nodes, bool isFinalValidationPass, size_t & todo)
    {
        todo = 0;           // returns how many nodes are to be redone
        for (auto & node : nodes)
        {
            const auto & children = node->GetChildren();
            const bool isLeaf = node->IsLeaf();
            // only validate a node if it has at least one child
            bool hasVisitedChild = false;
            bool allChildrenVisited = true;
            for (auto & child : children)
            {
                hasVisitedChild |= child->m_visited;    // if not a single visited child then no point in validating
                allChildrenVisited &= child->m_visited;
            }
            // if there is not at least one visited child
            bool valid = false;
            if (hasVisitedChild || isLeaf)
            {
                // got at least one child: it makes sense to call Validate()
                // keep state
                MBLayoutPtr oldMBLayoutPtr = node->GetMBLayout();
                auto dim = node->GetDims();
                vector<pair<size_t, size_t>> childDims;
                for (auto & child : children)
                    childDims.push_back(child->GetDims());
                auto imageLayouts = node->GetImageLayouts();
                // We do call validate(final) as many times as needed, since stuff may have changed underneath.
                node->PrintSelfBeforeValidation();
                node->Validate(isFinalValidationPass/*final*/);      // all nodes have been visited: do verification instead of just inference
                fprintf(stderr, " -> [%lu, %s%lu]", node->GetNumRows(), node->HasMBLayout() ? "MBSize " : "", node->GetNumCols());
                node->m_visited = true;
                // also take the opportunity to propagate m_needsGradient
                auto needsGradient = node->m_needsGradient;
                for (auto & child : children)       // TODO: do we need a check that this is stable if isFinalValidationPass?
                    node->m_needsGradient |= child->m_needsGradient;
                // check state --node will be valid if all nodes have been visited and node has not been updated
                bool unchanged = true;
                unchanged &= (oldMBLayoutPtr == node->GetMBLayout());
                unchanged &= (dim == node->GetDims());
                vector<pair<size_t, size_t>> newChildDims;
                for (auto & child : children)
                    newChildDims.push_back(child->GetDims());
                unchanged &= (childDims == newChildDims);
                unchanged &= (imageLayouts == node->GetImageLayouts());
                unchanged &= (needsGradient == node->m_needsGradient);
                if (isFinalValidationPass && !unchanged)
                    LogicError("ValidateSubNetwork: %ls %ls operation changed during final validation.", node->NodeName().c_str(), node->OperationName().c_str());
                if (isFinalValidationPass && !allChildrenVisited)
                    LogicError("ValidateSubNetwork: %ls %ls operation in final validation although not all children were visited?", node->NodeName().c_str(), node->OperationName().c_str());
                // if all children valid then 
                valid = (allChildrenVisited && unchanged) || isLeaf;
            }
            // count those that we need to redo
            if (!valid)
                todo++;
        }
    }

    // prepare to compute with the subnetwork that this rootNode depends on, including
    //  - auto-detecting recurrent loops
    //  - collect input and learnable nodes
    //  - calling Validate() on all nodes lazily, which sizes all matrices (column dimensions get updated to MB size)
    // Done lazily, called for every minibatch's invocation of EvaluateNode(), but memoizing which nodes were done already.
    // BUGBUG? Lazy triggers on the root node. I.e. for two different root nodes (training, eval), it validates twice.
    void ComputationNetwork::BuildAndValidateSubNetwork(const ComputationNodeBasePtr rootNode)
    {
        bool inserted = m_built.insert(rootNode).second;  // remember we built it
        if (!inserted)
            return;                                             // already done

        // detect recurrent loops for this root node
        // TODO: not nice--why not always call this in ValidateSubNetwork() only?
        FormRecurrentLoops(rootNode);

        // for the m_inputValues and m_learnableParameters sets for this rootNode
        CollectInputAndLearnableParameters(rootNode);

        // validate the rootNode and all nodes it depends on, in evaluation order
        ValidateSubNetwork(rootNode);
    }

    // tests whether BuildAndValidateSubNetwork() was called
    bool ComputationNetwork::BuiltAndValidatedSubNetwork(const ComputationNodeBasePtr & rootNode)
    {
        return m_built.find(rootNode) != m_built.end();
    }

    // -----------------------------------------------------------------------
    // memory allocation
    // -----------------------------------------------------------------------

    // this function will need to be called before actual validation and execution to 
    // predetermine how to share matrices to reduce memory usage.
    // TODO: find a simple topological order and allocateEvalMatrices on that order directly
    // without passing in eval, out, and train nodes.
    void ComputationNetwork::AllocateAllEvalMatrices(std::vector<ComputationNodeBasePtr>& evalRootNodes,
                                                     std::vector<ComputationNodeBasePtr>& outValueRootNodes,
                                                     std::vector<ComputationNodeBasePtr>& trainRootNodes)
    {
        //allocate memory for forward computation
        fprintf(stderr, "\n\nAllocating matrices for forward propagation.\n");
        for (int i = 0; i < evalRootNodes.size(); i++)
            AllocateEvalMatrices(evalRootNodes[i]);
        for (int i = 0; i < outValueRootNodes.size(); i++)
            AllocateEvalMatrices(outValueRootNodes[i]);
        for (int i = 0; i < trainRootNodes.size(); i++)
            AllocateEvalMatrices(trainRootNodes[i]);

    }

    // TODO: use the same loop mechanism as Evaluate()
    void ComputationNetwork::AllocateEvalMatrices(ComputationNodeBasePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        list<ComputationNodeBasePtr>& allNodes = GetEvalOrder(rootNode, false);

        //determine parent size
        map<ComputationNodeBasePtr, int> parentCount;
        for (auto &n : allNodes)
        {
            for (int i = 0; i < n->ChildrenSize(); i++)
            {
                ComputationNodeBasePtr pNode = n->GetChildren()[i];
                parentCount[pNode]++;
            }
        }

        set<ComputationNodeBasePtr> completedEvaluate;

        for (auto &nodeIter : allNodes)
        {
            if (nodeIter->IsPartOfLoop())
            {
                // TODO: use GetOuterLoopNode() here to avoid completedEvaluate[] check
                shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(m_recurrentInfo, nodeIter);
                assert(recInfo != nullptr);
                if (completedEvaluate.insert(recInfo).second)
                {
#if 1
                    recInfo->RequestMatricesBeforeEval(m_matrixPool);
#else
                    for (auto &nodeLoopIter : recInfo->m_nestedNodes)
                    {
                        nodeLoopIter->RequestMatricesBeforeEval(m_matrixPool);
                    }
#endif

                    for (auto &nodeLoopIter : recInfo->m_nestedNodes)
                    {
                        ReleaseMatricesAfterEvalForChildren(nodeLoopIter, parentCount);
                    }
                }
            }
            else
            {
                nodeIter->RequestMatricesBeforeEval(m_matrixPool);
                //we only release matrices for the children since the root node's informatioin will be used and should not be shared
                //with others
                ReleaseMatricesAfterEvalForChildren(nodeIter, parentCount);
            }
        }
    }

    void ComputationNetwork::ReleaseMatricesAfterEvalForChildren(ComputationNodeBasePtr n, std::map<ComputationNodeBasePtr, int>& parentCount)
    {
        for (int i = 0; i < n->ChildrenSize(); i++)
        {
            ComputationNodeBasePtr pNode = n->GetChildren()[i];
            parentCount[pNode]--;
            if (parentCount[pNode] == 0)
                pNode->ReleaseMatricesAfterEval(m_matrixPool);
        }
    }

    void ComputationNetwork::AllocateGradientMatrices(ComputationNodeBasePtr rootNode)
    {
        FormRecurrentLoops(rootNode);

        std::list<ComputationNodeBasePtr>& allNodes = GetGradientCalcOrder(rootNode);

        //now, simulate the gradient computation order to determine how to allocate matrices
        set<ComputationNodeBasePtr> completedGradient;

        //we need to call it here since we always compute gradients for children and root node is not children of other node
        rootNode->RequestMatricesBeforeGradientComp(m_matrixPool);

        for (auto &n : allNodes)
        {
            if (n->IsPartOfLoop())
            {
                std::vector<ComputationNodeBasePtr> recurrentNodes;
                shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(m_recurrentInfo, n);
                if (completedGradient.insert(recInfo).second)
                {
                    // SEQ mode: allocate all in loop first, then deallocate again
#if 1               // TODO: next step: use PARTraversalFlowControlNode::AllocateGradientMatricesForChildren() and ReleaseMatricesAfterGradientComp()...
                    // BUGBUG: naw, ^^ would not work! Wrong order! Need to rethink this. Need to make AllocateEvalMatrices() and AllocateGradientMatrices() the virtual functions.
                    recInfo->AllocateGradientMatricesForChildren(m_matrixPool);
                    //loops are computed sample by sample so we have to allocate them all 
                    recInfo->ReleaseMatricesAfterGradientComp(m_matrixPool);
#else
                    const auto & recurrentNodes = recInfo->m_nestedNodes;
                    //loops are computed sample by sample so we have to allocate them all 
                    for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                    {
                        (*nodeIter)->AllocateGradientMatricesForChildren(m_matrixPool);
                    }
                    recInfo->m_completedGradient = true;
                    for (auto nodeIter = recurrentNodes.rbegin(); nodeIter != recurrentNodes.rend(); ++nodeIter)
                    {
                        if ((*nodeIter)->NeedGradient())
                        {
                            (*nodeIter)->ReleaseMatricesAfterGradientComp(m_matrixPool);
                        }
                    }
#endif
                }
            }
            else
            {
                // PAR mode: we can allocate and immediately deallocate one by one
                n->AllocateGradientMatricesForChildren(m_matrixPool);
                if ((n != rootNode) && n->NeedGradient())  //root node's information will be used and should not be shared with others, also it's small (1x1)
                    n->ReleaseMatricesAfterGradientComp(m_matrixPool);
            }
        }
    }

#if 0
    void ComputationNetwork::AllocateGradientMatricesForChildren(ComputationNodeBasePtr parentNode)
    {
        std::vector<ComputationNodeBasePtr> children = parentNode->GetChildren();
        for (int i = 0; i < children.size(); i++)
        {
            if (children[i]->NeedGradient())
                children[i]->RequestMatricesBeforeGradientComp(m_matrixPool);
        }
    }
#endif

}}}
