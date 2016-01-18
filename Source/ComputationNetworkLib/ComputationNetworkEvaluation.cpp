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
#include "InputAndParamNodes.h"
#include <string>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <map>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

// This source file contains methods related to evaluation (forward prop, backprop), network validation, and matrix memory allocation (memory sharing).

// -----------------------------------------------------------------------
// forward and backward propagation
// -----------------------------------------------------------------------

// MAIN ENTRY POINT for evaluating one minibatch (forward prop)
// This calls ForwardProp() on all nodes in order of data flow through the network.
// By default, the network is applied concurrently on all frames in a minibatch in parallel (PAR mode, a "map" operation)
// Recurrent loops must be treated differently:
//  - a recurrent loop is the loop of nodes that make up computation for one time step (e.g. Times -> Plus -> Sigmoid -> Delay)
//  - these must be executed frame by frame (SEQuential) rather than as a map
//  - such a loop is treated as if they were a little nested network; this is done inside SEQTraversalFlowControlNodes
//  - these little nested networks are defined in the execution network in the form of nested sentinel nodes of type SEQTraversalFlowControlNode
void ComputationNetwork::ForwardProp(const ComputationNodeBasePtr rootNode)
{
    VerifyIsCompiled("ForwardProp");

    // traverse all nodes in the pre-determined evaluation order
    GetNestedNetwork(rootNode)->ForwardProp(FrameRange(nullptr));
}

// set the gradient matrix of a node to an 1x1 matrix containing 1.0
// Returns false if the node is not a ComputationNode<ElemType>.
template <class ElemType>
static bool SetGradientToScalarOne(ComputationNodeBasePtr nodep)
{
    auto node = dynamic_pointer_cast<ComputationNode<ElemType>>(nodep);
    bool hasMatchingType = (node != nullptr);
    if (hasMatchingType)
    {
        node->VerifyDims(1, 1);
        node->Gradient().Resize(1, 1);
        node->Gradient().SetValue((ElemType) 1.0);
    }
    return hasMatchingType;
}

// MAIN ENTRY POINT for evaluation followed by gradient computation (forward prop then back prop)
// The typical calling pattern is:
//  - ForwardProp() for eval nodes
//  - ForwardProp() for the training criterion (which will reuse computation results from the previous step)
//  - Backprop() for the training criterion
void ComputationNetwork::Backprop(const ComputationNodeBasePtr rootNode) // training criterion to compute the gradients for
{
    // reset all gradients to zero (actually, internally, this is lazy, but we don't care here)
    ZeroGradients(rootNode);

    // initialize root gradient with a scalar value of 1.0
    if (!SetGradientToScalarOne<float>(rootNode) && !SetGradientToScalarOne<double>(rootNode))
        LogicError("Backprop: Training criterion is neither ComputationNode<float> nor ComputationNode<double>.");

    // backpropagate through the network
    GetNestedNetwork(rootNode)->Backprop(FrameRange(nullptr), true, true);
}

void ComputationNetwork::FormNestedNetwork(const ComputationNodeBasePtr& rootNode)
{
    if (m_nestedNetworks.find(rootNode) != m_nestedNetworks.end())
        fprintf(stderr, "FormNestedNetwork: WARNING: Was called twice for %ls %ls operation\n", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());

    m_nestedNetworks[rootNode] = make_shared<PARTraversalFlowControlNode>(m_allSEQNodes, GetEvalOrder(rootNode));
}

ComputationNodeBasePtr ComputationNetwork::GetNestedNetwork(const ComputationNodeBasePtr& rootNode)
{
    if (m_nestedNetworks.find(rootNode) == m_nestedNetworks.end())
        LogicError("GetNestedNetwork: Called without prior call to FormNestedNetwork() for %ls %ls operation", rootNode->NodeName().c_str(), rootNode->OperationName().c_str());
    return m_nestedNetworks[rootNode];
}

// -----------------------------------------------------------------------
// PARTraversalFlowControlNode methods -- implements PAR traversal
//
// This implements an outer loop over non-recurrent nodes, where each node can be
// executed in PAR mode; that is, all samples are independent and allow for
// concurrent computation in bulk CUDA launches.
// -----------------------------------------------------------------------

ComputationNetwork::PARTraversalFlowControlNode::PARTraversalFlowControlNode(const std::vector<shared_ptr<SEQTraversalFlowControlNode>>& recurrentInfo, const std::list<ComputationNodeBasePtr>& allNodes /*must be in eval order*/)
{
    // traverse the network in evaluation order and create a new list that replaces all recurrence by a SEQTraversalFlowControlNode
    set<shared_ptr<IComputationNode>> loopsSeen; // for consistency check only
    for (auto nodeIter = allNodes.begin(); nodeIter != allNodes.end();)
    {
        shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(recurrentInfo, *nodeIter); // check if this node participates in a recurrent loop
        if (recInfo)                                                                                      // node is part of a SEQ loop: gather all of them. The nodes must be consecutive in 'allNodes'
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
        else // regular top-level node (non-looping, PAR)
        {
            m_nestedNodes.push_back(*nodeIter);
            nodeIter++; // and consume this node
        }
    }
}
/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::ForwardProp(const FrameRange& fr) /*override*/
{
    for (auto& node : m_nestedNodes)
    {
        if (node->IsOutputOlderThanInputs())
        {
            auto recInfo = dynamic_pointer_cast<SEQTraversalFlowControlNode>(node);
            if (recInfo)
                assert(recInfo->m_sourceNode->GetMBLayout() == node->GetMBLayout());

            node->BeginForwardProp();
            node->ForwardProp(fr.WithLayout(node->GetMBLayout()));
            node->EndForwardProp();

            node->BumpEvalTimeStamp();
        }
    }
}

/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) /*override*/
{
    childrenInThisLoop, childrenInOuterLoop; // TODO: think through what these mean when coming from PAR mode
    // process nodes in pre-determined order
    for (auto pnode = m_nestedNodes.rbegin(); pnode != m_nestedNodes.rend(); pnode++) // iterate backwards over evaluation order
    {
        auto& node = *pnode;

        node->BeginBackprop();
        node->Backprop(fr.WithLayout(node->GetMBLayout()), true /*childrenInThisLoop*/, true /*childrenInOuterLoop*/);
        node->EndBackprop();
    }
}
/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) /*override*/
{
}
/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) /*override*/
{
}
/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::AllocateGradientMatricesForInputs(MatrixPool& matrixPool) /*override*/
{
}
/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::RequestMatricesBeforeBackprop(MatrixPool& matrixPool) /*override*/
{
}
/*virtual*/ void ComputationNetwork::PARTraversalFlowControlNode::ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) /*override*/
{
}

// -----------------------------------------------------------------------
// SEQTraversalFlowControlNode methods -- implements SEQ traversal (loop unrolling)
//
// While PAR mode processes all samples in the MB independently, and thus in
// PARallel, SEQ mode is to honor sequential dependencies. As such, it
// unrolls the loop over time steps and runs the network once per time step.
// -----------------------------------------------------------------------

/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::BeginForwardProp() /*override*/
{
    // take the opportunity to check that layout is shared by all nodes in the loop
    // TODO: we should do this in a constructor.
    for (auto& node : m_nestedNodes)
    {
        if (node->GetMBLayout() != GetMBLayout())
            LogicError("Evaluate: all nodes inside a recurrent loop must have a layout that is identical; mismatch found for nodes '%ls' vs. '%ls'",
                       node->NodeName().c_str(), m_nestedNodes[0]->NodeName().c_str());
    }

    // tell all that loop is about to commence
    for (auto& node : m_nestedNodes)
        node->BeginForwardProp();
}

// evaluation of a SEQTraversalFlowControlNode FlowControlNode
// This evaluates all nodes in this FlowControlNode in SEQ mode: process the loop frame by frame in a nested loop.
// This is where the time axis changes.
// TODO: Once we do nested loops, then the FrameRange argument to this will refer to the outer loop.
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::ForwardProp(const FrameRange&) /*override*/
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
        for (auto& node : m_nestedNodes)
        {
            node->ForwardProp(t);
            node->BumpEvalTimeStamp();
        }
    }
}

/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::EndForwardProp() /*override*/
{
    // tell all that loop is done  --e.g. PastValueNode will capture its state for BPTT processing
    for (auto& node : m_nestedNodes)
        node->EndForwardProp();
}

// called before first iteration step of ComputeGradient()
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::BeginBackprop() /*override*/
{
    for (auto& node2 : m_nestedNodes)
        node2->BeginBackprop();
}

/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::Backprop(const FrameRange&, bool childrenInThisLoop, bool childrenInOuterLoop) /*override*/
{
    childrenInThisLoop, childrenInOuterLoop;    // TODO: think through what these mean when coming from PAR mode
    const auto& recurrentNodes = m_nestedNodes; // BUGBUG: -ForForward?? Does this mean we can remove non-ForForward?
    auto pMBLayout = recurrentNodes[0]->GetMBLayout();
    FrameRangeIteration range(pMBLayout, m_steppingDirection);
    for (auto t = range.rbegin(); t != range.rend(); t++) // note: reverse iteration
    {
        for (auto nodeIter2 = recurrentNodes.rbegin(); nodeIter2 != recurrentNodes.rend(); ++nodeIter2)
        {
            auto& node2 = *nodeIter2;
            node2->Backprop(t, true /*childrenInThisLoop*/, false /*childrenInOuterLoop*/);
            // The above flags tell Backprop() to skip back-propagation from inside a node into
            // a node that is outside the loop, which is done later in EndBackprop() in PAR mode.
        }
    }
}

// called after last iteration step of ComputeGradient()
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::EndBackprop() /*override*/
{
    // The following loop handles the case that a node inside the loop back-propagates a gradient into a node outside of the loop.
    // For efficiency, we perform this outside the loop in PAR mode. E.g., in one LSTM speech setup, we measured 12..14% overall speed-up.
    for (auto nodeIter2 = m_nestedNodes.rbegin(); nodeIter2 != m_nestedNodes.rend(); ++nodeIter2)
    {
        auto& node2 = *nodeIter2;
        node2->Backprop(FrameRange(m_nestedNodes[0]->GetMBLayout()), false /*childrenInThisLoop*/, true /*childrenInOuterLoop*/);
    }

    // tell all nodes we are done for this iteraTion
    for (auto& node2 : m_nestedNodes)
        node2->EndBackprop();
}

/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) /*override*/
{
    for (auto& nodeLoopIter : m_nestedNodes)
        nodeLoopIter->RequestMatricesBeforeForwardProp(matrixPool);
}
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) /*override*/
{
}
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::AllocateGradientMatricesForInputs(MatrixPool& matrixPool) /*override*/
{
    // TODO: should we deallocate in opposite order?
    for (auto nodeIter = m_nestedNodes.rbegin(); nodeIter != m_nestedNodes.rend(); ++nodeIter)
    {
        (*nodeIter)->AllocateGradientMatricesForInputs(matrixPool);
    }
}
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::RequestMatricesBeforeBackprop(MatrixPool& matrixPool) /*override*/
{
}
/*virtual*/ void ComputationNetwork::SEQTraversalFlowControlNode::ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) /*override*/
{
    for (auto nodeIter = m_nestedNodes.rbegin(); nodeIter != m_nestedNodes.rend(); ++nodeIter)
    {
        if ((*nodeIter)->NeedGradient())
            (*nodeIter)->ReleaseMatricesAfterBackprop(matrixPool);
    }
}

// find if node is part of a recurrent loop; and return the loop id
// If found then return a pointer to the list of nodes of this loop.
/*static*/ shared_ptr<ComputationNetwork::SEQTraversalFlowControlNode> ComputationNetwork::FindInRecurrentLoops(const std::vector<std::shared_ptr<SEQTraversalFlowControlNode>>& recurrentInfo, const ComputationNodeBasePtr& node)
{
    // look in all recurrent loops of the network
    // TODO: Check for IsPartOfLoop(). Also why not store the loop id in the node for direct lookup?
    for (auto& iter : recurrentInfo)
        if (std::find(iter->m_nestedNodes.begin(), iter->m_nestedNodes.end(), node) != iter->m_nestedNodes.end()) // TODO: should this loop need to be a method of SEQTraversalFlowControlNode?
            return iter;
    return nullptr; // not part of a recurrent loop
}

// check if any of the nodes in the recurrence IsOutputOlderThanInputs(), with exception of delay nodes for which this check would fail and can be skipped
// TODO: Would it be sufficient to check against our own time stamp, so that we can use a unified time-stamping mechanism? Then we'd not need this special check for delayed nodes; just check all inputs against our own time stamp.
bool ComputationNetwork::SEQTraversalFlowControlNode::IsOutputOlderThanInputs() const
{
    for (auto& ptr : m_nestedNodes)
    {
        if (ptr->IsOutputOlderThanInputs() &&
            ptr->OperationName() != OperationNameOf(PastValueNode) &&
            ptr->OperationName() != OperationNameOf(FutureValueNode))
        {
            return true;
        }
    }
    return false;
}

// TODO: do this on PARTraversalFlowControlNode
void ComputationNetwork::ResetEvalTimeStamps()
{
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        nodeIter->second->ResetEvalTimeStamp();
}

/*static*/ void ComputationNetwork::BumpEvalTimeStamp(const vector<ComputationNodeBasePtr>& nodes)
{
    for (size_t i = 0; i < nodes.size(); i++)
        nodes[i]->BumpEvalTimeStamp();
}

// for debugging
void ComputationNetwork::PrintComputationTree(const ComputationNodeBasePtr& rootNode,
                                              const bool forwardCompute,
                                              const bool printMatrices)
{
    auto nodes = GetEvalOrder(rootNode); // note: don't take a reference, since we reverse() below
    if (forwardCompute)
    {
        fprintf(stderr, "\n\nPrinting forward-computation node order ... \n");
    }
    else
    {
        fprintf(stderr, "\n\nPrinting gradient-computation node order ... \n");
        nodes.reverse();
    }

    if (nodes.size() == 0)
        fprintf(stderr, "\n(empty)\n");
    else
        for (const auto& node : nodes)
            node->PrintSelf(printMatrices);
}

// -----------------------------------------------------------------------
// preparation of network
// -----------------------------------------------------------------------

// called by model editing operations, such as DeleteNode(); and by RebuildNetwork()
// These invalidates any post-processed structures. If they are accessed, we will fail.
void ComputationNetwork::InvalidateCompiledNetwork()
{
    m_isCompiled = false;
    m_allSEQNodes.clear();
    m_evalOrders.clear();
    m_nestedNetworks.clear();
    m_inputValues.clear();
    m_learnableParameters.clear();
}

// verify that network has undergone CompileNetwork()
void ComputationNetwork::VerifyIsCompiled(const char* where) const
{
    if (!IsCompiled())
        LogicError("%s: A compiled network was expected.", where);
}

// CompileNetwork() -- bring network into executable state
// Call this after creation, load, and any modification.
// This method sets up all members that are cleared in InvalidateCompiledNetwork();
// TODO: This should be the only entry point, subsuming all other Validate, Build, etc. functions.
// TODO: Related functions today do lots of stuff lazily. There are redundant calls. That will all be removed.
// TODO: This is in a somewhat partial state in that we now have a global eval order (keyed by a nullptr), but don't use it yet.
void ComputationNetwork::CompileNetwork()
{
    fprintf(stderr, "\nPost-processing network...\n");

    // all steps below have to be repeated for all root nodes (=nodes without parents and PreComputeNodes)
    DetermineSetOfAllRoots();

    fprintf(stderr, "\n%d roots:\n", (int) m_allRoots.size());
    for (const auto& root : m_allRoots)
        fprintf(stderr, "\t%ls = %ls\n", root->NodeName().c_str(), root->OperationName().c_str());

    // Note: Steps below are loops over root nodes. We will gradually push those loops through to the functions,
    //       to reduce redundant operation on shared portions of the network.

    // STEP: Create a depth-first tree-traversal order through original graph for every root.
    // This is used wherever a nested structure is not relevant.
    FormEvalOrder(nullptr); // form the global one
    for (auto& node : m_allRoots)
        FormEvalOrder(node);

    // STEP: form the m_inputValues and m_learnableParameters sets for this rootNode
    for (const auto& root : m_allRoots)
        CollectInputAndLearnableParameters(root);

    // STEP: Discover nested loops.
    FormRecurrentLoops(nullptr); // form the global one
    for (auto& node : m_allRoots)
        FormRecurrentLoops(node);

    // STEP: Form nested structure of PAR and SEQ traversal nodes.
    for (auto& node : m_allRoots)
        FormNestedNetwork(node);

    // STEP: Infer node dimensions.
    // This leverages the nested structure.  TODO: ... one day
    for (auto& node : m_allRoots)
        ValidateSubNetwork(node);

    // STEP: Optimize the network.
    // :)

    // STEP: Some final details.
    FixupInputMinibatchSize(); // post-fix MB sizes in InputValues(). Will not be needed with next-gen reader.
    ResetEvalTimeStamps();     // invalidate all m_value fields. Really belongs into StartEvaluateMinibatchLoop()

    fprintf(stderr, "\nPost-processing network complete.\n");
    m_isCompiled = true;
}

// determine the set of all root nodes
// Roots are nodes that ForwardProp() may be called for.
//  - training criterion, eval criteria
//  - outputs
//  - PreComputeNodes
// Result is stored in m_allRoots.
// BUGBUG: In the current implementation, outputs that are also inputs to others must be specified explicitly e.g. by a tag.
void ComputationNetwork::DetermineSetOfAllRoots()
{
    // start with all non-referenced nodes
    set<ComputationNodeBasePtr> allNodes, referencedNodes;
    for (const auto& iter : m_nameToNodeMap)
    {
        auto node = iter.second;
        allNodes.insert(node);
        for (size_t i = 0; i < node->GetNumInputs(); i++)
        {
            auto input = node->Input(i);
            if (!input) // this may be the result of an incorrect MEL operation
            {
                InvalidArgument("DetermineSetOfAllRoots: Input %d of %ls %ls operation if not connected, network is malformed.",
                                (int) i, node->NodeName().c_str(), node->OperationName().c_str());
            }
            referencedNodes.insert(input);
        }
    }
    set<ComputationNodeBasePtr> unreferencedNodes;
    set_difference(allNodes.begin(), allNodes.end(), referencedNodes.begin(), referencedNodes.end(), inserter(unreferencedNodes, unreferencedNodes.end()));

    // add in all explicitly specified nodes.
    // TODO: This is not ideal. We will also need on-demand compilation, to allow any node to be used as an output after the fact.
    set<ComputationNodeBasePtr> allKnownRoots;
    for (const auto& node : FinalCriterionNodes())
        allKnownRoots.insert(node);
    for (const auto& node : EvaluationNodes())
        allKnownRoots.insert(node);
    for (const auto& node : OutputNodes())
        allKnownRoots.insert(node);
    for (const auto& iter : m_nameToNodeMap) // PreComputeNodes
    {
        auto node = iter.second;
        if (node->RequiresPreCompute())
            allKnownRoots.insert(node);
    }

    // set m_allRoots to include both non-referenced nodes and also all explicitly specified roots
    m_allRoots.clear();
    set_union(unreferencedNodes.begin(), unreferencedNodes.end(), allKnownRoots.begin(), allKnownRoots.end(), inserter(m_allRoots, m_allRoots.end()));
}

// -----------------------------------------------------------------------
// validation
// -----------------------------------------------------------------------

#if 0
    // ValidateNetwork() - Validate the entire network
    // This calls ValidateNetowrk(Node) for all output nodes.
    // This is used after loading or for dumping the network.
    void ComputationNetwork::ValidateNetwork(bool allowFragment, const bool bAllowNoCriterion)
    {
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");

#if 1 // If it is not done here, it will causea crash. But it really only belongs into StartEvaluationMinibatchLoop()
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
#endif

// validate sub-network needed to evalute a specific output node
// This calls Validate() on every node in evaluation order (allowing to propagate things forwards through the net).
// This is called lazily but once only per node until next ClearCache().
// This also sets up MBLayout links.
void ComputationNetwork::ValidateSubNetwork(const ComputationNodeBasePtr& rootNode)
{
    // reset to a well-defined MBLayout (any meaningful layout should do here)
    // Note that Validate is never called during operation. Any actual computation will lead to MBLayout to be set.
    m_pMBLayout->Init(1, 0);

    // set up MBLayout links of inputs (all others get propagated upwards through Validate())
    // TODO: Once we support mismatching layouts, this will be more involved. For now, everything shares the one layout that the Network knows about.
    for (auto node : InputNodes(rootNode))
        node->LinkToMBLayout(m_pMBLayout);

    // we call all nodes' Validate() in order to validate, that is, set up MBLayout and FunctionValues dimension
    // A problem is that recurrent loops may require partial validation.
    // Nodes validated on partial input (i.e. some children not yet validated) will be revisited.
    const auto& nodes = GetEvalOrder(rootNode);

    for (auto& node : nodes)
    {
        node->m_visited = false;
        node->m_needsGradient = node->IsParameterUpdateRequired(); // these get propagated upwards in the following
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
        fprintf(stderr, "\n\nValidating for node %ls. %d nodes to process in pass %d.\n", rootNode->NodeName().c_str(), (int) toValidate, (int) pass);
        ValidateNodes(nodes, false /*isFinalValidationPass*/, toValidate);
    }
    fprintf(stderr, "\n\nValidating for node %ls, final verification.\n", rootNode->NodeName().c_str());
    ValidateNodes(nodes, true /*isFinalValidationPass*/, toValidate);
    if (toValidate != 0)
        LogicError("ValidateSubNetwork: ValidateNodes(true) unexpectedly returned with work left to do.");

    // propagate some info to SEQTraversalFlowControlNode
    // TODO: In the future we should validate not on the flat list but the PARTraversalFlowControlNode structure. Then this will be unnecessary.
    for (auto& recInfo : m_allSEQNodes)
    {
        auto& node = recInfo->m_sourceNode;
        recInfo->m_needsGradient = node->m_needsGradient;
        recInfo->LinkToMBLayout(node->GetMBLayout());
    }

    for (auto& node : nodes)
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
        fprintf(stderr, "%d out of %d nodes do not share the minibatch layout with the input data.\n", (int) nonDefaultNodes.size(), (int) nodes.size());
        //for (auto node : nonDefaultNodes)
        //    fprintf(stderr, "    %ls\n", node->NodeName().c_str());
        //fprintf(stderr, "\n\n");
    }
}

void ComputationNetwork::ValidateNodes(list<ComputationNodeBasePtr> nodes, bool isFinalValidationPass, size_t& todo)
{
    todo = 0; // returns how many nodes are to be redone
    for (auto& node : nodes)
    {
        const auto& children = node->GetInputs();
        const bool isLeaf = node->IsLeaf();
        // only validate a node if it has at least one child
        bool hasVisitedChild = false;
        bool allChildrenVisited = true;
        for (auto& child : children)
        {
            hasVisitedChild |= child->m_visited; // if not a single visited child then no point in validating
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
            for (auto& child : children)
                childDims.push_back(child->GetDims());
            auto sampleLayout = node->GetSampleLayout();
            // We do call validate(final) as many times as needed, since stuff may have changed underneath.
            node->PrintSelfBeforeValidation();
            node->Validate(isFinalValidationPass /*final*/); // all nodes have been visited: do verification instead of just inference
            fprintf(stderr, " -> [%lu [%s], %s%lu]", node->GetNumRows(), string(node->GetSampleLayout()).c_str(), node->HasMBLayout() ? "MBSize " : "", node->GetNumCols());
            node->m_visited = true;
            // also take the opportunity to propagate m_needsGradient
            auto needsGradient = node->m_needsGradient;
            for (auto& child : children) // TODO: do we need a check that this is stable if isFinalValidationPass?
                node->m_needsGradient |= child->m_needsGradient;
            // check state --node will be valid if all nodes have been visited and node has not been updated
            bool unchanged = true;
            unchanged &= (oldMBLayoutPtr == node->GetMBLayout());
            unchanged &= (dim == node->GetDims());
            vector<pair<size_t, size_t>> newChildDims;
            for (auto& child : children)
                newChildDims.push_back(child->GetDims());
            unchanged &= (childDims == newChildDims);
            unchanged &= (sampleLayout == node->GetSampleLayout());
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

#if 0
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
#endif

// -----------------------------------------------------------------------
// memory allocation
// -----------------------------------------------------------------------
// mark nodes that are purely induced by parameters as non-sharable and create space for value if null
void ComputationNetwork::MarkValueNonSharableNodes()
{
    const auto& nodes = GetEvalOrder(nullptr);
    std::map<wstring, bool> allLeafDescendentsAreParameters;
    std::list<ComputationNodeBasePtr> allLearnableParameters = GetNodesWithType(OperationNameOf(LearnableParameter));
    // note that: we cannot use m_learnableParameters because we need all parameters node, regardless whether it requires update or not

    for (auto& node : nodes)
    {
        auto children = node->GetInputs();
        wstring myname = node->NodeName();
        bool allParameters = true;

        if (children.size()) // we don't do the check for leaf node, cause all the possible leaf nodes (input/parameters/precompute node) are marked as non-sharable already
        {
            for (auto child : children)
            {
                wstring ChildName = child->NodeName();
                if (allLeafDescendentsAreParameters.find(ChildName) == allLeafDescendentsAreParameters.end())
                {
                    // not found, means it is a leaf node (we are at eval order )
                    assert(child->IsLeaf() || child->IsPartOfLoop());
                    if (std::find(allLearnableParameters.begin(), allLearnableParameters.end(), child) != allLearnableParameters.end())
                    {
                        allLeafDescendentsAreParameters[ChildName] = true;
                    }
                    else
                    {
                        allParameters = false;
                        allLeafDescendentsAreParameters[ChildName] = false;
                        break;
                    }
                }
                else
                {
                    if (allLeafDescendentsAreParameters[ChildName] == false)
                    {
                        allParameters = false;
                        break;
                    }
                }
            }
            allLeafDescendentsAreParameters[myname] = allParameters;
            if (allParameters)
            {
                node->MarkValueNonSharable();
            }
            else
            {
                node->MarkValueSharable();
            }
        }
    }
}

// this function will need to be called before actual validation and execution to
// predetermine how to share matrices to reduce memory usage.
// TODO: find a simple topological order and allocateEvalMatrices on that order directly
// without passing in eval, out, and train nodes.
void ComputationNetwork::AllocateAllMatrices(const std::vector<ComputationNodeBasePtr>& evalRootNodes,
                                             const std::vector<ComputationNodeBasePtr>& outValueRootNodes,
                                             ComputationNodeBasePtr trainRootNode)
{
    // Allocate memory for forward/backward computation
    fprintf(stderr, "\n\nAllocating matrices for forward and/or backward propagation.\n");

    VerifyIsCompiled("AllocateAllMatrices");

    // Due to special topology, if a node is solely induced by parameters, its function value should not be shared
    MarkValueNonSharableNodes();

    bool performingBackPropagation = (trainRootNode != nullptr);

    // Create a composite Eval order with the specified nodes as roots
    std::vector<ComputationNodeBasePtr> forwardPropRoots;
    forwardPropRoots.insert(forwardPropRoots.end(), evalRootNodes.begin(), evalRootNodes.end());
    forwardPropRoots.insert(forwardPropRoots.end(), outValueRootNodes.begin(), outValueRootNodes.end());
    if (trainRootNode != nullptr)
        forwardPropRoots.push_back(trainRootNode);

    // For each node determine parents and whether the output of the
    // node is needed during back propagation
    std::unordered_map<ComputationNodeBasePtr, bool> outputValueNeededDuringBackProp;
    std::unordered_map<ComputationNodeBasePtr, std::unordered_set<ComputationNodeBasePtr>> parentsMap;
    for (auto& rootNode : forwardPropRoots)
    {
        list<ComputationNodeBasePtr>& currentRootEvalNodes = GetEvalOrder(rootNode);
        for (auto& currentNode : currentRootEvalNodes)
        {
            for (int i = 0; i < currentNode->GetNumInputs(); i++)
            {
                ComputationNodeBasePtr pNode = currentNode->GetInputs()[i];
                parentsMap[pNode].insert(currentNode);

                if (performingBackPropagation)
                {
                    if (outputValueNeededDuringBackProp.find(pNode) == outputValueNeededDuringBackProp.end())
                        outputValueNeededDuringBackProp[pNode] = pNode->OutputUsedInComputingInputNodesGradients();

                    outputValueNeededDuringBackProp[pNode] |= currentNode->InputUsedInComputingInputNodesGradients(i);
                }
                else
                {
                    outputValueNeededDuringBackProp[pNode] = false;
                }
            }
        }
    }

    std::unordered_map<ComputationNodeBasePtr, int> parentCount;
    for (auto& keyValue : parentsMap)
    {
        parentCount[keyValue.first] = keyValue.second.size();
    }

    // Construct the composite forward prop eval order by enumerating the
    // nodes corresponding to each of our roots and then arranging them in the
    // relative order that they appear in the global evaluation order
    const std::list<ComputationNodeBasePtr>& allNodesEvalOrder = GetEvalOrder(nullptr);
    std::list<ComputationNodeBasePtr> nodesForForwardPropRoots = ComputationNodeBase::EnumerateNodes(forwardPropRoots);
    std::vector<ComputationNodeBasePtr> compositeForwardPropEvalOrder;
    for (auto& node : allNodesEvalOrder)
    {
        if (std::find(nodesForForwardPropRoots.cbegin(), nodesForForwardPropRoots.cend(), node) != nodesForForwardPropRoots.cend())
        {
            compositeForwardPropEvalOrder.push_back(node);
        }
    }

    set<ComputationNodeBasePtr> completedEvaluate;
    for (auto& nodeIter : compositeForwardPropEvalOrder)
    {
        nodeIter->SetOutputNeededDuringBackprop(outputValueNeededDuringBackProp[nodeIter]);

        if (nodeIter->IsPartOfLoop())
        {
            // TODO: use FormNestedNetwork() here to avoid completedEvaluate[] check
            shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(m_allSEQNodes, nodeIter);
            assert(recInfo != nullptr);
            if (completedEvaluate.insert(recInfo).second)
            {
                recInfo->RequestMatricesBeforeForwardProp(m_matrixPool);

                for (auto& nodeLoopIter : recInfo->m_nestedNodes)
                {
                    ReleaseMatricesAfterEvalForChildren(nodeLoopIter, parentCount);
                }
            }
        }
        else
        {
            nodeIter->RequestMatricesBeforeForwardProp(m_matrixPool);
            //we only release matrices for the children since the root node's informatioin will be used and should not be shared
            //with others
            ReleaseMatricesAfterEvalForChildren(nodeIter, parentCount);
        }
    }

    if (trainRootNode != nullptr)
    {
        std::list<ComputationNodeBasePtr>& backPropNodes = GetEvalOrder(trainRootNode);

        //now, simulate the gradient computation order to determine how to allocate matrices
        set<ComputationNodeBasePtr> completedGradient;

        // we need to call it here since we always compute gradients for children and root node is not children of other node
        trainRootNode->RequestMatricesBeforeBackprop(m_matrixPool);

        for (auto iter = backPropNodes.rbegin(); iter != backPropNodes.rend(); iter++) // for gradient computation, traverse in reverse order
        {
            auto n = *iter;
            if (n->IsPartOfLoop())
            {
                std::vector<ComputationNodeBasePtr> recurrentNodes;
                shared_ptr<SEQTraversalFlowControlNode> recInfo = FindInRecurrentLoops(m_allSEQNodes, n);
                if (completedGradient.insert(recInfo).second)
                {
                    // SEQ mode: allocate all in loop first, then deallocate again
                    // TODO: next step: use PARTraversalFlowControlNode::AllocateGradientMatricesForInputs() and ReleaseMatricesAfterBackprop()...
                    // BUGBUG: naw, ^^ would not work! Wrong order! Need to rethink this. Need to make AllocateEvalMatrices() and AllocateGradientMatrices() the virtual functions.
                    recInfo->AllocateGradientMatricesForInputs(m_matrixPool);
                    // Loops are computed sample by sample so we have to allocate them all
                    recInfo->ReleaseMatricesAfterBackprop(m_matrixPool);
                }
            }
            else
            {
                // PAR mode: we can allocate and immediately deallocate one by one
                n->AllocateGradientMatricesForInputs(m_matrixPool);
                // Root node's information will be used and should not be shared with others, also it's small (1x1)
                if ((n != trainRootNode) && n->NeedGradient())
                    n->ReleaseMatricesAfterBackprop(m_matrixPool);
            }
        }
    }
}

void ComputationNetwork::ReleaseMatricesAfterEvalForChildren(ComputationNodeBasePtr n, std::unordered_map<ComputationNodeBasePtr, int>& parentCount)
{
    for (int i = 0; i < n->GetNumInputs(); i++)
    {
        ComputationNodeBasePtr pNode = n->GetInputs()[i];
        parentCount[pNode]--;
        if (parentCount[pNode] == 0)
            pNode->ReleaseMatricesAfterForwardProp(m_matrixPool);
    }
}
} } }
