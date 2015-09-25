//
// <copyright file="ComputationNetwork.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "TrainingCriterionNodes.h"
#include "Basics.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"  // used for load & save
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "TrainingCriterionNodes.h"
#include "CompositeComputationNodes.h"
#include "EvaluationCriterionNodes.h"
#include "MPIWrapper.h"
#include <string>
#include <fstream>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    void ComputationNetwork::ClearNet()
    {
        for (auto groupIter : GetAllNodeGroups())
            (groupIter)->clear();

        m_recurrentInfo.clear();

        m_built.clear();

        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();

        m_inputs.clear();
        m_learnableParameters.clear();

        //for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        //{
        //    delete nodeIter->second;
        //}
        m_nameToNodeMap.clear();    // will also deref and likely deallocate all nodes we hold in here
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    void ComputationNetwork::SaveToFile(const std::wstring& fileName, const FileOptions fileFormat) const
    {
        // In case of parallel training only the main node should we saving the model to prevent
        // the parallel training nodes from colliding to write the same file
        if ((g_mpi == nullptr) || g_mpi->IsMainNode())
        {
            // Saving into temporary file and then renaming it to the requested fileName
            // This is a standard trick to avoid havign corrupted model files if process dies during writing
            wstring tmpFileName = fileName + L".tmp";
            SaveToFileImpl(tmpFileName, fileFormat);
            renameOrDie(tmpFileName, fileName);
        }
    }

    // TODO: how does the file distinguish float vs double nodes?
    void ComputationNetwork::SaveToFileImpl(const std::wstring& fileName, const FileOptions fileFormat) const
    {
        File fstream(fileName, fileFormat | FileOptions::fileOptionsWrite);
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCN");

        //model version
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BVersion");
        fstream << (size_t) CURRENT_CNTK_MODEL_VERSION;
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EVersion");

        fstream << (size_t) m_nameToNodeMap.size();

        //put all node info first
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = nodeIter->second;
            nodePtr->SaveToFile(fstream);
        }

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        //put relationship
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
        for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
        {
            ComputationNodeBasePtr nodePtr = nodeIter->second;
            fstream << nodePtr->NodeName() << nodePtr->ChildrenSize();
            for (size_t i = 0; i < nodePtr->ChildrenSize(); i++)
            {
                if (nodePtr->GetChildren()[i] == nullptr)
                    fprintf(stderr, "Warning: node %ls 's child is null, please check your ndl/mel file.\n", nodePtr->NodeName().c_str());
                else
                    fstream << nodePtr->GetChildren()[i]->NodeName();
            }
        }
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERelation");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes");
        fstream << m_features.size();
        for (size_t i = 0; i < m_features.size(); i++)
            fstream << m_features[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
        fstream << m_labels.size();
        for (size_t i = 0; i < m_labels.size(); i++)
            fstream << m_labels[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes");
        fstream << m_finalCriteria.size();
        for (size_t i = 0; i < m_finalCriteria.size(); i++)
            fstream << m_finalCriteria[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECriteriaNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling");
        fstream << m_requestNodesMultiSeqHandling.size();
        for (size_t i = 0; i<m_requestNodesMultiSeqHandling.size(); i++)
            fstream << m_requestNodesMultiSeqHandling[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodesReqMultiSeqHandling");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes");
        fstream << m_evalNodes.size();
        for (size_t i = 0; i < m_evalNodes.size(); i++)
            fstream << m_evalNodes[i]->NodeName();
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
        fstream << m_outputNodes.size();
        for (size_t i = 0; i < m_outputNodes.size(); i++)
        {
            fstream << m_outputNodes[i]->NodeName();
        }
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

        if (m_pairNodes.size() > 0)
        {
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BPairNodes");

            fstream << m_pairNodes.size();
            for (size_t i = 0; i < m_pairNodes.size(); i++)
                fstream << m_pairNodes[i]->NodeName();
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EPairNodes");
        }

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECN");
       
        fstream.Flush();
    }

    void ComputationNetwork::LoadPersistableParametersFromFile(const std::wstring& fileName, const bool requireValidation,
                                                               const FileOptions fileFormat)
    {
        File fstream(fileName, fileFormat | FileOptions::fileOptionsRead);

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCN");

        //model version
        size_t modelVersion = CNTK_MODEL_VERSION_1; //if version info is not there it is version 1
        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
        {
            fstream >> modelVersion;
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
        }

        size_t numNodes;
        fstream >> numNodes;

        //get all node info first
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
        for (size_t i = 0; i < numNodes; i++)
        {
            std::wstring opName, nodeName;
            fstream >> opName >> nodeName;
            ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
            // TODO: don't we have a load constructor? Then when to call which? Document the calling sequence
            nodePtr->LoadFromFile(fstream, modelVersion);
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        SetActualMiniBatchSizeFromFeatures();

        if (requireValidation)
            ValidateNetwork();
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    ComputationNodeBasePtr ComputationNetwork::SetNodeValue(const std::wstring & nodeName, const double value)
    {
        ComputationNodeBasePtr pNode = GetNodeFromName(nodeName);

        // TODO: this is a bit ugly, but does SetNodeValue() really belong here?
        if (IsNodePtr<LearnableParameter<float>>(pNode))
            AsNodePtr<LearnableParameter<float>>(pNode)->FunctionValues().SetValue((float)value);
        else if (IsNodePtr<LearnableParameter<double>>(pNode))
            AsNodePtr<LearnableParameter<double>>(pNode)->FunctionValues().SetValue((double)value);
        else if (pNode->RequiresPreCompute())
        {
            if (IsNodePtr<PreComputedNode<float>>(pNode))
            {
                auto preComputedNode = AsNodePtr<PreComputedNode<float>>(pNode);
                preComputedNode->FunctionValues().SetValue((float)value);    // TODO: comment: is this an expensive operation?
                preComputedNode->MarkComputed(true);
            }
            else
            {
                auto preComputedNode = AsNodePtr<PreComputedNode<double>>(pNode);
                preComputedNode->FunctionValues().SetValue((double)value);    // TODO: comment: is this an expensive operation?
                preComputedNode->MarkComputed(true);
            }
        }
        else
            LogicError("Only values of learnable parameters and precomputed nodes can be set.");

        return pNode;
    }

    void ComputationNetwork::SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr rootNode)
    {
        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->OperationName() == OperationNameOf(LearnableParameter))
                    node->NeedGradient() = needGradient;
            }
        }
        else
        {
            //for calculating a specific node
            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if (node->OperationName() == OperationNameOf(LearnableParameter))
                    node->NeedGradient() = needGradient;
            }
        }
    }

    // non-static version needed because it accesses m_randomSeedOffset
    // Excessively used by SimpleNetworkBuilder, but always after CreateLearnableParameter(), so we should really absorb it there
    template<class ElemType> void ComputationNetwork::InitLearnableParameters(const ComputationNodeBasePtr node, const bool uniformInit, const unsigned long randomSeed, const ElemType initValueScale, bool initOnCPUOnly)
    {
        auto learnableParameterNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(node);
        learnableParameterNode->InitRandom(uniformInit, randomSeed + GetRandomSeedOffset(), initValueScale, initOnCPUOnly);
    }

    // FixupInputMinibatchSize - go through all the inputs and make sure they have a consistent minibatch size (after creation)
    void ComputationNetwork::FixupInputMinibatchSize()
    {
        std::list<ComputationNodeBasePtr> inputs = GetNodesWithType(OperationNameOf(InputValue));
        int minibatchMax = 0;
        bool minibatchDifferent = false; // flag to see if all the values are already the same
        for (ComputationNodeBasePtr node : inputs)
        {
            size_t cols = node->GetNumCols();
            if (cols != minibatchMax)
            {
                if (minibatchMax != 0)
                    minibatchDifferent = true;
                if (minibatchMax < cols)
                    minibatchMax = cols;
            }
        }
        if (minibatchDifferent)
        {
            for (ComputationNodeBasePtr node : inputs)
            {
                size_t cols = node->GetNumCols();
                if (cols != minibatchMax)
                    node->Resize(node->GetNumRows(), minibatchMax);
            }
        }
    }

    // -----------------------------------------------------------------------
    // evaluation
    // -----------------------------------------------------------------------

    // validate sub-network needed to evalute a specific output node
    // This calls Validate() on every node in evaluation order (allowing to propagate things forwards through the net).
    // This is called lazily but once only per node until next ClearCache().
    // This also sets up MBLayout links.
    // TODO: I can't see a clear pattern when ClearCache() is called. E.g. at the start of each epoch? Or never in normal operation (init only at construction)?
    // Note: under some circumstances, one must call FormRecurrentNodes() on this node before calling this. TODO: Not clear which ones.
    // TODO: ^^ is this really needed? Can we just call it inside?
    void ComputationNetwork::ValidateSubNetwork(const ComputationNodeBasePtr rootNode)
    {
        fprintf(stderr, "\n\nValidating node %ls \n", rootNode->NodeName().c_str());

        // set up MBLayout links of inputs (all others get propagated upwards through Validate())
        // TODO: Once we support mismatching layouts, this will be more involved. For now, everything shares the one layout that the Network knows about.
        for (auto node : InputNodes(rootNode))
            node->LinkToMBLayout(m_pMBLayout);

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);

        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            auto node = *nodeIter;
            node->PrintSelfBeforeValidation();
            node->Validate();
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
            fprintf(stderr, "\n\n%d out of %d nodes do not share the minibatch layout with the input data.\n", (int)nonDefaultNodes.size(), (int)nodes.size());
            //for (auto node : nonDefaultNodes)
            //    fprintf(stderr, "    %ls\n", node->NodeName().c_str());
            //fprintf(stderr, "\n\n");
        }
    }

    // prepares the network for computation
    // Done lazily, called for every minibatch's invocation of EvaluateNode(), but memoizing which nodes were done already.
    // BUGBUG? Lazy triggers on the root node. I.e. for two different root nodes (training, eval), it validates twice.
    void ComputationNetwork::BuildAndValidateSubNetwork(const ComputationNodeBasePtr rootNode)
    {
        const auto inserted = m_built.insert(rootNode).second;  // remember we built it
        if (!inserted)
            return;                                             // already done

        // detect recurrent loops for this root node (more loops will be detected inside ValidateSubNetwork())
        // TODO: not nice--why not always call this in ValidateSubNetwork() only?
        FormRecurrentLoops(rootNode);

        // for the m_inputs and m_learnableParameters sets for this rootNode
        CollectInputAndLearnableParameters(rootNode);

        // validate the rootNode and all nodes it depends on, in evaluation order
        ValidateSubNetwork(rootNode);

        //
        SetRequestNodesMultiSeqHandling();
    }

    bool ComputationNetwork::IsFuncValueOlderThanInputs(const std::vector<ComputationNodeBasePtr>& recurrentNodes)
    {
        for (auto ptr = recurrentNodes.begin(); ptr != recurrentNodes.end(); ptr++)
        {
            if ((*ptr)->IsFuncValueOlderThanInputs() && 
                (*ptr)->OperationName() != OperationNameOf(PastValueNode) &&
                (*ptr)->OperationName() != OperationNameOf(FutureValueNode))
            {
                return true;
            }
        }
        return false;
    }

    // note: all of these have NodeDoesItsOwnCustomizedMissingColumnsMasking() returning true
    bool ComputationNetwork::IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr)
    {
        if (nodePtr->OperationName() == OperationNameOf(SquareErrorNode) ||
            nodePtr->OperationName() == OperationNameOf(CrossEntropyWithSoftmaxNode) ||
			nodePtr->OperationName() == OperationNameOf(SequenceWithSoftmaxNode) ||
            nodePtr->OperationName() == OperationNameOf(CrossEntropyNode) ||
            nodePtr->OperationName() == OperationNameOf(ClassBasedCrossEntropyWithSoftmaxNode) ||
            nodePtr->OperationName() == OperationNameOf(ErrorPredictionNode) ||               
            nodePtr->OperationName() == OperationNameOf(CRFNode) ||
            nodePtr->OperationName() == OperationNameOf(DummyCriterionNode))
            return true;

        return false;
    }

    // transfer user-specified request for masking to the indivudal nodes
    // This is only needed if users explicitly perform reduce-like operations.
    // It makes no sense for some nodes, so we skip those.
    void ComputationNetwork::SetRequestNodesMultiSeqHandling()
    {
        for (auto & node : m_requestNodesMultiSeqHandling)  // this set is defined in NDL; here we propagate that into the actual nodes' flags, except for a few where it makes no sense (avoid user error)
        {
            //SumElements node will generate a scalar value and so it should never require special handling
            //TransposeNode will change the size of columns and so it should also not included for special handling
            //their child node should instead
#if 0
            if (node->OperationName() != OperationNameOf(SumElementsNode) &&
                node->OperationName() != OperationNameOf(TransposeNode) &&
                node->OperationName() != OperationNameOf(MeanNode) &&
                node->OperationName() != OperationNameOf(InvStdDevNode) 
                )
                node->SetMaskMissingColumnsToZero();
#else
            if (node->OperationName() == OperationNameOf(SumElementsNode) ||
                node->OperationName() == OperationNameOf(TransposeNode) ||
                node->OperationName() == OperationNameOf(MeanNode) ||
                node->OperationName() == OperationNameOf(InvStdDevNode))
            {
                RuntimeError("SetRequestNodesMultiSeqHandling: NodesReqMultiSeqHandling cannot be used with operation '%ls'\nIn the past, CNTK silently fixed this; now please change your NDL instead", node->OperationName().c_str());
            }
            node->SetMaskMissingColumnsToZero();
#endif
        }

        // if a typical criterion node is used as the training criterion node we assume it requires multiseq handling 
        // this is for backward compatibility
        // All of these have NodeDoesItsOwnCustomizedMissingColumnsMasking() return true, i.e. they will not have MaskMissingColumnsToZero() auto-called from Network.
        // Hence, instead of setting the flag, we just ensure that this is true.
        for (auto & node : m_finalCriteria)
            if (IsTypicalCriterionNode(node))
                //node->SetMaskMissingColumnsToZero();
                if (!node->NodeDoesItsOwnCustomizedMissingColumnsMasking())
                    LogicError("criterion %ls's NodeDoesItsOwnCustomizedMissingColumnsMasking() function must return true", node->OperationName().c_str());

        for (auto & node : m_evalNodes)
            if (IsTypicalCriterionNode(node))
                //node->SetMaskMissingColumnsToZero();
                if (!node->NodeDoesItsOwnCustomizedMissingColumnsMasking())
                    LogicError("criterion %ls's NodeDoesItsOwnCustomizedMissingColumnsMasking() function must return true", node->OperationName().c_str());
    }

    template<class N> void ComputationNetwork::GetNodesRequiringX(std::list<ComputationNodeBasePtr> & nodesRequirePreComputation, const ComputationNodeBasePtr rootNode, bool checkComputed)
    {
        if (rootNode == nullptr)        // find nodes from all available nodes
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->RequiresPreCompute()) // TODO: why not check directly for the type with a dynamic_cast?
                {
                    auto preComputedNode = static_pointer_cast<N>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                        nodesRequirePreComputation.push_back(node);
                }
            }
        }
        else                            // or for calculating a specific node
        {
            const auto & nodes = GetEvalOrder(rootNode, false);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = *nodeIter;
                if (node->RequiresPreCompute()) // TODO: why not check directly for the type with a dynamic_cast?
                {
                    auto preComputedNode = static_pointer_cast<N>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                        nodesRequirePreComputation.push_back(node);
                }
            }
        }
    }

    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has a grammar error, fix
    std::list<ComputationNodeBasePtr> ComputationNetwork::GetNodesRequiringPreComputation(const ComputationNodeBasePtr rootNode, bool checkComputed)
    {
        std::list<ComputationNodeBasePtr> nodesRequirePreComputation;
        GetNodesRequiringX<PreComputedNode<float>>(nodesRequirePreComputation, rootNode, checkComputed);
        GetNodesRequiringX<PreComputedNode<double>>(nodesRequirePreComputation, rootNode, checkComputed);
        return nodesRequirePreComputation;
    }

    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has grammar error, fix
    std::list<ComputationNodeBasePtr> ComputationNetwork::GetNodesRequiringBatchMode(const ComputationNodeBasePtr rootNode, bool checkComputed)
    {
        std::list<ComputationNodeBasePtr> nodesRequirePreComputation;
        GetNodesRequiringX<BatchModeNode<float>>(nodesRequirePreComputation, rootNode, checkComputed);
        GetNodesRequiringX<BatchModeNode<double>>(nodesRequirePreComputation, rootNode, checkComputed);
        return nodesRequirePreComputation;
    }

    // The methods below determine evaluation order, which is tricky in presence of recurrent loops.
    // TODO: Can this be moved to a separate class, or at least a separate CPP?

    void ComputationNetwork::ClearCalcOrderCaches()
    {
        for (auto & it : m_cacheEvalOrders)
            for (auto & iter2 : m_cacheEvalOrders[it.first])
                iter2->ClearCache();
        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();
    }

    void ComputationNetwork::MergeRecurrentLoops(const ComputationNodeBasePtr /*rootNode*/)
    {
        /// merge loops if they have the same source node
        std::vector<RecurrentInfo> m_recurrentInfoTmp;
        if (m_recurrentInfo.size() <= 1)
            return;

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            if (m_recurrentInfoTmp.size() == 0)
            {
                RecurrentInfo rInfo;
                rInfo.Copy(*iter);
                m_recurrentInfoTmp.push_back(rInfo);
            }
            else
            {
                bool bFound = false;
                for (auto iter2 = m_recurrentInfoTmp.begin(); iter2 != m_recurrentInfoTmp.end(); iter2++)
                {
                    if ((*iter2).m_sourceNode == (*iter).m_sourceNode)
                    {
                        bFound = true;
                        break;
                    }
                }

                if (bFound == false)
                {
                    RecurrentInfo rInfo;
                                rInfo.Copy(*iter);
                    m_recurrentInfoTmp.push_back(rInfo);
                }
                else
                    continue;
            }
        }

        // no need to sort the vector of recurrent loops, because they are pushed and later used as FIFO
        m_recurrentInfo.clear();
        for (auto iter = m_recurrentInfoTmp.begin(); iter != m_recurrentInfoTmp.end(); iter++)
            m_recurrentInfo.push_back(*iter);

        // for debug purposes
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            fprintf(stderr, " nodes in the recurrent loops : \n");
            for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                fprintf(stderr, "%ls\t", (*itr)->NodeName().c_str());
        }
    }

    // get the strong connected component from the graph
    // This sets index, lowLink, m_visited, m_inStack
    void ComputationNetwork::getStrongSCC(const ComputationNodeBasePtr rootNode)    // TODO: method names start uppercase
    {
        /// notice that this graph including graphs from a parent networks if two or more networks are connected via PairNode
        std::unordered_set<ComputationNodeBasePtr> visited;
        std::list<ComputationNodeBasePtr> sccStack;
        size_t index = 0;
        size_t loopId = 0;
        if (rootNode->IsVisisted() == false)
            strongSCC(rootNode, sccStack, index, loopId);
    }

    // (called only from getStrongSCC())
    void ComputationNetwork::strongSCC(ComputationNodeBasePtr cur,      // TODO: method names start uppercase
                                       std::list<ComputationNodeBasePtr>& sccStack,
                                       size_t& index, size_t& loopId)
    {
        cur->SetIndex(index);
        cur->SetLowLink(index);
        index++;

        cur->SetVisited(true);
        sccStack.push_back(cur);
        cur->SetInStack(true);

        if (cur->OperationName() != L"PairNetwork")
        {
            // pairnetwork is the socket from other network, so ignore its children, which are in the other networks
            for (int i = 0; i < cur->ChildrenSize(); i++)
            {
                if (cur->GetChildren()[i]->IsVisisted() == false)
                {
                    strongSCC(cur->GetChildren()[i], sccStack, index, loopId);
                    cur->SetLowLink(min(cur->GetLowLink(), cur->GetChildren()[i]->GetLowLink()));
                }
                else if (cur->GetChildren()[i]->IsInStack())
                {
                    cur->SetLowLink(min(cur->GetLowLink(), cur->GetChildren()[i]->GetLowLink()));
                }
            }
        }

        if (cur->GetLowLink() == cur->GetIndex())   // something special has happened   --TODO: comment what that was!!
        {
            RecurrentInfo rInfo;
            rInfo.m_loopId = loopId;
            rInfo.m_sourceNode = cur;
            size_t sccSize = 0;
            for (;;)
            {
                ComputationNodeBasePtr w = sccStack.back();
                sccStack.pop_back();
                w->SetInStack(false);
                rInfo.m_recurrentNodes.push_back(w);
                sccSize++;
                if (w == cur)
                    break;
            }
            rInfo.Reset();
            if (sccSize > 1)
            {
                loopId++;
                m_recurrentInfo.push_back(rInfo);
            }
        }
    }

    void ComputationNetwork::getLoopForwordOrder(std::unordered_set<ComputationNodeBasePtr>& visited,   // TODO: method name
                                                 std::unordered_set<ComputationNodeBasePtr>& recStack,
                                                 std::list<ComputationNodeBasePtr>& nodesStack,
                                                 ComputationNodeBasePtr cur)
    {
        if (visited.find(cur) == visited.end())
        {
            visited.insert(cur);
            recStack.insert(cur);

            if (cur->OperationName() != OperationNameOf(PastValueNode) && 
                cur->OperationName() != OperationNameOf(FutureValueNode))
            {
                for (size_t i = 0; i < cur->ChildrenSize(); i++)
                    if (cur->GetChildren()[i]->GetLoopId() == cur->GetLoopId())
                        getLoopForwordOrder(visited, recStack, nodesStack, cur->GetChildren()[i]);
            }
            recStack.erase(cur);
            nodesStack.push_back(cur);
        }
        else
        {
            if (!(recStack.find(cur) == recStack.end()))
                LogicError("There is infinite Loop which cannot be unrolled!!");
        }
    }

    // forms the recurrent loop that 'rootNode' participates in
    // TODO: This function is not lazy, i.e. not cached. BuildAndValidateSubNetwork() caches, but others don't. Not sure why/how that's OK--won't we reassign loop ids?
    // This sets/updates:
    //  - 
    // Is often called before ValidateNetwork() on a root; will be called from inside ValidateNetwork() as well.
    void ComputationNetwork::FormRecurrentLoops(const ComputationNodeBasePtr rootNode)
    {
        // ...?
        getStrongSCC(rootNode);

        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, true/*recurrent*/);

        // ??
        MergeRecurrentLoops(rootNode);

        /// debug purpose  --TODO: <-- this comment seems incorrect; SetVisitedOrder() is basis of IsSmaller()
        // ... where does m_recurrentInfo get set?
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            fprintf(stderr, " nodes in the recurrent loops : \n");
            size_t max_visitedOrderInLoop = 0;
            for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
            {
                fprintf(stderr, "%ls\t", (*itr)->NodeName().c_str());
                if (max_visitedOrderInLoop < (*itr)->GetVisitedOrder())
                    max_visitedOrderInLoop = (*itr)->GetVisitedOrder();
            }
            for (auto itr = (*iter).m_recurrentNodes.begin(); itr != (*iter).m_recurrentNodes.end(); itr++)
                (*itr)->SetVisitedOrder(max_visitedOrderInLoop);
        }

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
            if ((*iter).m_recurrentNodes.size() > 1)
            {
                /// it is done in the mergerecurrentloops function, but just keep the code
                std::sort((*iter).m_recurrentNodes.begin(),
                          (*iter).m_recurrentNodes.end(),
                          (*iter).m_recurrentNodes[0]->IsSmaller);

                for (auto nodeRecIter = (*iter).m_recurrentNodes.begin(); nodeRecIter != (*iter).m_recurrentNodes.end(); nodeRecIter++)
                {
                    (*nodeRecIter)->SetLoop(true);
                    (*nodeRecIter)->SetLoopId((*iter).m_loopId);
                }
            }
        }

        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            // sort the recurrent nodes in their ascending name, which is the same as visiting nodes in G^R
            (*iter).m_recurrentNodesForForward.clear();
            if ((*iter).m_recurrentNodes.size() > 1)
            {
                std::list<ComputationNodeBasePtr> result;
                std::unordered_set<ComputationNodeBasePtr> visited;
                std::unordered_set<ComputationNodeBasePtr> recStack;

                for (size_t j = 0; j < (*iter).m_recurrentNodes.size(); j++)
                {
                    ComputationNodeBasePtr nodeRecIter = (*iter).m_recurrentNodes[j];
                    for (size_t i = 0; i < nodeRecIter->ChildrenSize(); i++)
                    {
                        if (nodeRecIter->GetChildren()[i]->GetLoopId() == nodeRecIter->GetLoopId() && 
                            nodeRecIter->OperationName() != OperationNameOf(PastValueNode) &&
                            nodeRecIter->OperationName() != OperationNameOf(FutureValueNode))     // TODO: test for type RecurrentNode instead?
                        {
                            nodeRecIter->GetChildren()[i]->SetIndexInLoop(nodeRecIter->GetChildren()[i]->GetIndexInLoop() + 1);
                        }
                    }
                }

                //for (auto nodeRecIter = startNodes.begin(); nodeRecIter != startNodes.end(); nodeRecIter++)

                for (size_t i = 0; i < (*iter).m_recurrentNodes.size(); i++)
                {
                    ComputationNodeBasePtr nodeRecIter = (*iter).m_recurrentNodes[i];
                    if (visited.find(nodeRecIter) == visited.end() && nodeRecIter->GetIndexInLoop() == 0)
                        getLoopForwordOrder(visited, recStack, result, nodeRecIter);
                }

                for (size_t i = 0; i < (*iter).m_recurrentNodes.size(); i++)
                {
                    (*iter).m_recurrentNodesForForward.push_back(result.front());
                    result.pop_front();
                }

                (*iter).m_recurrentNodes = (*iter).m_recurrentNodesForForward;
            }
        }

        if (m_recurrentInfo.size() > 0)
        {
            std::map<int, std::list<ComputationNodeBasePtr>> recurrentNodes;
            std::list<ComputationNodeBasePtr> noRecurrentNodes;

            noRecurrentNodes = rootNode->ReshuffleNodes(recurrentNodes);

            nodes.sort(IsSmaller);

            ReorderLoops(nodes, recurrentNodes, noRecurrentNodes);

            m_cacheEvalOrders[rootNode] = nodes;
            std::list<ComputationNodeBasePtr> nodesForGrad = nodes;
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
        
        DetermineLoopTypes();
        
        for (auto & iter : nodes)
            iter->ClearCache();
    }

    void ComputationNetwork::DetermineLoopTypes()
    {
        for (auto iter = m_recurrentInfo.begin(); iter != m_recurrentInfo.end(); iter++)
        {
            bool hasPastValueNode = false;
            bool hasFutureValueNode = false;

            RecurrentInfo* recurrentInfo = &(*iter);

            if (recurrentInfo->m_recurrentNodes.size() > 0)
            {
                for (size_t j = 0; j < recurrentInfo->m_recurrentNodes.size(); j++)
                {
                    ComputationNodeBasePtr nodeRecIter = recurrentInfo->m_recurrentNodes[j];

                    if (nodeRecIter->OperationName() == OperationNameOf(PastValueNode))
                    {
                        hasPastValueNode = true;
                    }
                    else if (nodeRecIter->OperationName() == OperationNameOf(FutureValueNode))
                    {
                        hasFutureValueNode = true;
                    }
                }

                if (hasPastValueNode && hasFutureValueNode)
                {
                    RuntimeError("It is not allowed to have both PastValue and FutureValue nodes in the same loop.");
                }
                else if (!hasPastValueNode && !hasFutureValueNode)
                {
                    RuntimeError("There is neither PastValue nor FutureValue nodes in the loop.");
                }
                else if (hasPastValueNode)
                {
                    recurrentInfo->m_isForwardLoop = true;
                }
                else
                {
                    recurrentInfo->m_isForwardLoop = false;
                }
            }
        }
    }

    void ComputationNetwork::ReorderLoops(std::list<ComputationNodeBasePtr>& nodes,
                                                    const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/,
                                                    const std::list<ComputationNodeBasePtr> & /*noRecurrentNodes*/)
    {
        std::list<ComputationNodeBasePtr> newList;

        std::list<ComputationNodeBasePtr> vTmp;
        std::list<ComputationNodeBasePtr> vRecurrentTmp;
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

    // lazily reate the m_inputs[] and m_learnableParameters lists
    // The only other side effect is to call GetEvalOrder(), which will cache the evaluation order for the given root node.
    void ComputationNetwork::CollectInputAndLearnableParameters(const ComputationNodeBasePtr rootNode)
    {
        //not found
        if (m_inputs.find(rootNode) == m_inputs.end())
        {
            std::list<ComputationNodeBasePtr> inputs;

            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = *nodeIter;
                if (node->OperationName() == OperationNameOf(InputValue) /*L"InputValue"*/ ||
                    node->OperationName() == InputValue<float>::SparseTypeName())
                {
                    inputs.push_back(node);
                }
            }
            m_inputs[rootNode] = inputs;
        }

        //not found
        if (m_learnableParameters.find(rootNode) == m_learnableParameters.end())
        {
            std::list<std::wstring> learnableParameterNames;
            std::list<ComputationNodeBasePtr> learnableParameters;

            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);

            // instead of collecting the nodes themselves, collect the names (they will be sorted below)
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = *nodeIter;
                if ((node->OperationName() == OperationNameOf(LearnableParameter) && node->NeedGradient()) ||
                    (node->OperationName() == OperationNameOf(SparseLearnableParameter) && node->NeedGradient()))
                {
                    learnableParameterNames.push_back(node->NodeName());
                }
            }

            // we need to sort it so that we get consistent order when load it from saved file
            learnableParameterNames.sort();

            // now collect the actual nodes in the sort order of their node names
            for (auto nodeNameIter = learnableParameterNames.begin(); nodeNameIter != learnableParameterNames.end(); nodeNameIter++)
            {
                learnableParameters.push_back(GetNodeFromName((*nodeNameIter)));
            }

            m_learnableParameters[rootNode] = learnableParameters;
        }
    }

    /*static*/void ComputationNetwork::UpdateEvalTimeStamps(const std::vector<ComputationNodeBasePtr> & nodes)
    {
        for (size_t i = 0; i<nodes.size(); i++)
            nodes[i]->UpdateEvalTimeStamp();
    }

    template<class ElemType>
    /*static*/void ComputationNetwork::SetDropoutRate(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed)
    {
        if (dropoutRate != prevDropoutRate)
        {
            fprintf(stderr, "Switching dropout rate to %.8g.\n", dropoutRate);
            std::list<ComputationNodeBasePtr> dropoutNodes = net.GetNodesWithType(OperationNameOf(DropoutNode), criterionNode);
            if (dropoutNodes.size() == 0 && dropoutRate > 0)
                fprintf(stderr, "WARNING: there is no dropout node.\n");
            else for (auto nodeIter = dropoutNodes.begin(); nodeIter != dropoutNodes.end(); nodeIter++)
            {
                auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(*nodeIter);
                node->SetDropoutRate(dropoutRate);
                node->SetRandomSeed(dropOutSeed++);
            }

            prevDropoutRate = dropoutRate;
        }
    }

	//set sequence training parameters, e.g. smoothing weight, frame drop threshhold
	template<class ElemType>
	void ComputationNetwork::SetSeqParam(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const ElemType hsmoothingWeight, const ElemType frameDropThresh, const bool doreferencealign)
	{

		fprintf(stderr, "set Hsmoothing weight %.8g and frame drop thresh %.8g\n", hsmoothingWeight, frameDropThresh);
		std::list<ComputationNodeBasePtr> seqNodes = net.GetNodesWithType(OperationNameOf(SequenceWithSoftmaxNode), criterionNode);
		if (seqNodes.size() == 0)
		{
			fprintf(stderr, "WARNING: there is no sequence node.\n");
		}
		else
		{
			for (auto nodeIter = seqNodes.begin(); nodeIter != seqNodes.end(); nodeIter++)
			{				
				auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(*nodeIter);
				node->SetSmoothWeight(hsmoothingWeight);
				node->SetFrameDropThresh(frameDropThresh);
				node->SetRefrencealign(doreferencealign);
			}
		}
	}
    /*static*/void ComputationNetwork::SetMaxTempMemSizeForCNN(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const size_t maxTempMemSizeInSamples)
    {
        fprintf(stderr, "Set Max Temp Mem Size For Convolution Nodes to %lu samples.\n", maxTempMemSizeInSamples);
        std::list<ComputationNodeBasePtr> convolutionNodes = net.GetNodesWithType(OperationNameOf(ConvolutionNode), criterionNode);
        if (convolutionNodes.size() == 0 && maxTempMemSizeInSamples != 0)
        {
            fprintf(stderr, "WARNING: there is no convolution node.\n");
        }
        else
        {
            for (auto nodeIter = convolutionNodes.begin(); nodeIter != convolutionNodes.end(); nodeIter++)
            {
                auto node = dynamic_pointer_cast<ConvolutionNode<float>>(*nodeIter);
                node->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
            }
        }
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    template<class ElemType> void ComputationNetwork::LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork* anotherNetwork)
    {
        ClearNet();

        File fstream(fileName, fileFormat | FileOptions::fileOptionsRead);

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCN");

        //model version
        size_t modelVersion = CNTK_MODEL_VERSION_1; //if version info is not there it is version 1
        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
        {
            fstream >> modelVersion;
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
        }

        size_t numNodes;
        fstream >> numNodes;

        //get all node info first
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
        for (size_t i = 0; i < numNodes; i++)
        {
            std::wstring opName, nodeName;
            fstream >> opName >> nodeName;

            auto newNode = ComputationNetworkBuilder<ElemType>::NewNode(opName, m_deviceId, nodeName);
            if (!newNode)
            {
                fprintf(stderr, "Unknown ComputationNode type %ls (node name %ls)\n", opName.c_str(), nodeName.c_str());
                InvalidArgument("Invalid node type.");
            }
            newNode->LoadFromFile(fstream, modelVersion);
            AddNodeToNet(newNode);
        }
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        //put relationship
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
        for (size_t i = 0; i < numNodes; i++)
        {
            std::wstring nodeName;
            size_t numChildren;
            fstream >> nodeName >> numChildren;
            if (numChildren > 0)
            {
                std::vector<std::wstring> childrenNames;
                childrenNames.resize(numChildren);
                for (size_t j = 0; j < numChildren; j++)
                {
                    fstream >> childrenNames[j];
                }

                // TODO: how does the file distinguish float from double?
                ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
                std::vector<ComputationNodeBasePtr> childrenNodes;
                childrenNodes.resize(numChildren);
                for (int j = 0; j < numChildren; j++)
                    childrenNodes[j] = GetNodeFromName(childrenNames[j], anotherNetwork);

                if (nodePtr->OperationName() == OperationNameOf(RowStackNode)) {
                    //allow for variable input nodes
                    nodePtr->AttachInputs(childrenNodes);
                }
                else
                {
                    //fixed input nodes
                    switch (numChildren)
                    {
                        case 1:
                            nodePtr->AttachInputs(childrenNodes[0]);
                            break;

                        case 2:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1]);
                            break;
                        case 3:
                            nodePtr->AttachInputs(childrenNodes[0],childrenNodes[1],
                                                  childrenNodes[2]);
                            break;
                        case 4:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1],
                                                  childrenNodes[2], childrenNodes[3]);
                            break;
                        case 5:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2],
                                                  childrenNodes[3], childrenNodes[4]);
                            break;
                        case 6:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2],
                                                  childrenNodes[3], childrenNodes[4], childrenNodes[5]);
                            break;

                        default:
                            LogicError("Invalid number of children.");
                    }
                }
            }
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERelation");

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");
        {
            std::wstring nodeName;
            size_t num;

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes");
            fstream >> num;

            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_features.push_back(GetNodeFromName(nodeName));
            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_labels.push_back(GetNodeFromName(nodeName));
            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_finalCriteria.push_back(GetNodeFromName(nodeName));
            }

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECriteriaNodes");

            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling"))
            {
                fstream >> num;
                for (size_t i = 0; i<num; i++)
                {
                    fstream >> nodeName;
                    m_requestNodesMultiSeqHandling.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodesReqMultiSeqHandling");
            }

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_evalNodes.push_back(GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                m_outputNodes.push_back(GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BPairNodes"))
            {
                fstream >> num;
                for (size_t i = 0; i < num; i++)
                {
                    fstream >> nodeName;
                    m_pairNodes.push_back(GetNodeFromName(nodeName));
                }
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EPairNodes");
            }
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECN");

        //some internal values in the nodes are computed during validation
        ValidateNetwork(false, bAllowNoCriterionNode);
    }

    // -----------------------------------------------------------------------
    // topological plot [erw]
    // -----------------------------------------------------------------------

    class DotGraphConfigure
    {
    public:
        wstring m_LearnableParameterStyle;
        wstring m_featuresStyle;
        wstring m_CriteriaStyle;
        wstring m_nodesReqMultiSeqHandlingStyle;
        wstring m_labelsStyle;
        wstring m_normalNodeStyle;
        wstring m_PrecomputingNodeStyle;
        wstring m_pastValueNodeStyle;
        wstring m_futureValueNodeStyle;

        DotGraphConfigure()
        {
            m_LearnableParameterStyle = L"node [ shape = box     , color = gray , style = \"filled, rounded\"  ]; ";
            m_featuresStyle = L"node [ shape = ellipse , color = red  , fillcolor = white ]; ";
            m_CriteriaStyle = L"node [ shape = doublecircle , color =  red , fillcolor = white  ]; ";
            m_nodesReqMultiSeqHandlingStyle = L"node [ shape = doublecircle , color =  brown , fillcolor = white  ]; ";
            m_normalNodeStyle = L"node [ shape = ellipse, color = blue, fillcolor = white, style = solid ]; ";
            m_PrecomputingNodeStyle = L"node [ shape = box    , color = black, style = \"dashed, filled\",  fillcolor= limegreen ] ;";
            m_labelsStyle = L"node [ shape = diamond, color = brown, style = bold ] ;  ";
            m_pastValueNodeStyle = L"node [ shape = box3d  , color = lightgray, style = \"filled\" , fillcolor = white ] ";
            m_futureValueNodeStyle = L"node [ shape = box3d  , color = red, style = \"filled\" , fillcolor = white ] ";
        }
    };

    wstring ComputationNetwork::FormSpecialNodes(wstring style, std::vector<ComputationNodeBasePtr>& specialNodes)
    {
        if (specialNodes.empty())
            return L"";

        wstring str = style;

        for (const auto & x : specialNodes)
            str = str + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        return str + L"; \n";
    }

    void ComputationNetwork::DescribeNetworkUsingDot(std::list<ComputationArc>& arcs,
                                                     std::wstring outFile)
    {
        DotGraphConfigure dotcfg;

        File fstream(outFile,FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        // get precompute node
        std::vector<ComputationNodeBasePtr> PreComputedNodes;
        std::vector<ComputationNodeBasePtr> allnodes = GetAllNodes();
        for (const auto & n : allnodes)
        {
            if (n->RequiresPreCompute())
                PreComputedNodes.push_back(n);
        }

        // get PastValue node
        std::vector<ComputationNodeBasePtr> pastValueNodes;
        for (const auto & n : allnodes)
        {
            if (n->OperationName() == OperationNameOf(PastValueNode) || n->OperationName() == L"Delay")
                pastValueNodes.push_back(n);
        }

        // get FuturetValue node
        std::vector<ComputationNodeBasePtr> futureValueNodes;
        for (const auto & n : allnodes)
        {
            if (n->OperationName() == OperationNameOf(FutureValueNode))
                futureValueNodes.push_back(n);
        }
        // get learnableParameters
        std::vector<ComputationNodeBasePtr> learnableParameters;
        for (const auto & n : allnodes)
        {
            if (n->OperationName() == OperationNameOf(LearnableParameter))
                learnableParameters.push_back(n);
        }

        fstream << "strict digraph {\n";
        fstream << "rankdir = BT ;  \n";

        //////////////////////////////////////////////////////////////////////////
        //	special nodes
        //////////////////////////////////////////////////////////////////////////
        fstream << L"// special nodes \n";

        // learnable parameters:
        fstream << FormSpecialNodes(dotcfg.m_LearnableParameterStyle, learnableParameters);
        // features
        fstream << FormSpecialNodes(dotcfg.m_featuresStyle, m_features);
        // labels
        fstream << FormSpecialNodes(dotcfg.m_labelsStyle, m_labels);
        // critera
        fstream << FormSpecialNodes(dotcfg.m_CriteriaStyle, m_finalCriteria);
        // nodes that requires multi sequence handling 
        fstream << FormSpecialNodes(dotcfg.m_nodesReqMultiSeqHandlingStyle, m_requestNodesMultiSeqHandling);            
        // pre-compute nodes
        fstream << FormSpecialNodes(dotcfg.m_PrecomputingNodeStyle, PreComputedNodes);
        // PastValue nodes
        fstream << FormSpecialNodes(dotcfg.m_pastValueNodeStyle, pastValueNodes);
        // FutureValue nodes
        fstream << FormSpecialNodes(dotcfg.m_futureValueNodeStyle, futureValueNodes);
        // normal nodes
        fstream << dotcfg.m_normalNodeStyle << L"\n";

        //////////////////////////////////////////////////////////////////////////
        //	add labels for each node
        //////////////////////////////////////////////////////////////////////////
        fstream << L"\n// add labels and operation name\n";
        wstring line;
        for (const auto & x : allnodes)
        {
            line.clear();
            size_t nrows = x->GetNumRows();
            size_t ncols = x->GetNumCols();
            line = msra::strfun::wstrprintf(L" \"%ls\" [ label = \"%ls [%d,%d]\\n%ls\" ] ;\n",
                                            x->GetName().c_str(), x->GetName().c_str(), nrows, ncols,
                                            x->OperationName().c_str());
            fstream << line;
        }

        //////////////////////////////////////////////////////////////////////////
        //	sub-graph
        //////////////////////////////////////////////////////////////////////////
        // subgraph source
        fstream << L"subgraph {\n";
        fstream << L"\t\t rank=source ; ";
        line.clear();
        for (const auto & x : m_features)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        fstream << line << L"\n}\n";

        // subgraph eval/output/criteria
        fstream << L"subgraph {\n";
        fstream << L"\t\t rank=sink ; ";
        line.clear();
        for (const auto & x : m_finalCriteria)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (const auto & x : m_requestNodesMultiSeqHandling)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (const auto & x : m_outputNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (const auto & x : m_pairNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (const auto & x : m_evalNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());

        fstream << line << L"\n}\n";

        //////////////////////////////////////////////////////////////////////////
        //	specify arc connections
        //////////////////////////////////////////////////////////////////////////
        for (auto x = arcs.begin(); x != arcs.end(); x++)
        {
            ComputationNodeBasePtr src = (*x).first;
            ComputationNodeBasePtr des = (*x).second;

            std::wstring srcname = src->GetName();
            std::wstring desname = des->GetName();

            if (des->OperationName() == OperationNameOf(PastValueNode) || des->OperationName() == L"Delay")
            {
                // special treament for arc with PastValue node as the children
                // create a dummy node
                ComputationNodeBasePtr pastValueNode = des;
                wstring dummyName = des->GetName() + L".dummy";
                wstring out = msra::strfun::wstrprintf(L"node [ shape = box3d  , color = lightgray, style = \"filled\" , label = \"%ls\" ] ; \"%ls\"\n",
                                                       (pastValueNode->GetName() + L"\\n(PastValue)").c_str(),
                                                       dummyName.c_str());
                line = out;
                line += msra::strfun::wstrprintf(L"\"%ls\" -> \"%ls\" ; \n", dummyName.c_str(), srcname.c_str());
            }
            else if (des->OperationName() == OperationNameOf(FutureValueNode))
            {
                // special treament for arc with FutureValue node as the children
                // create a dummy node
                ComputationNodeBasePtr futureValueNode = des;
                wstring dummyName = des->GetName() + L".dummy";
                wstring out = msra::strfun::wstrprintf(L"node [ shape = box3d  , color = red, style = \"filled\" , label = \"%ls\" ] ; \"%ls\"\n",
                    (futureValueNode->GetName() + L"\\n(FutureValue)").c_str(),
                    dummyName.c_str());
                line = out;
                line += msra::strfun::wstrprintf(L"\"%ls\" -> \"%ls\" ; \n", dummyName.c_str(), srcname.c_str());
            }
            else
            {
                line = msra::strfun::wstrprintf(L"\"%ls\" -> \"%ls\" ; \n", desname.c_str(), srcname.c_str());
            }

            fstream << line;
        }
        fstream << L"\n}\n";
    }

    void ComputationNetwork::PlotNetworkTopology(const std::wstring outputFile) //  [1/13/2015 erw] plot network topology using dot language
    {
        BuildAndValidateSubNetwork(m_evalNodes[0]);

        //////////////////////////////////////////////////////////////////////////
        //	step 1.		get all the arcs in the network
        //////////////////////////////////////////////////////////////////////////
        std::unordered_set<ComputationNodeBasePtr> visited;
        std::list<ComputationArc> arcs;

        for (auto groupIter : GetAllNodeGroups())
        {
            // note: this will also loop over m_features and m_labels, which will do nothing since they have no inputs
            // TODO: test whether that is true
            const auto & group = *groupIter;
            for (size_t i = 0; i < group.size(); i++)
                group[i]->EnumerateArcs(visited, arcs);
        }

        //////////////////////////////////////////////////////////////////////////
        //	step 2.		output dot description
        //////////////////////////////////////////////////////////////////////////
        DescribeNetworkUsingDot(arcs, outputFile);
    }

    // -----------------------------------------------------------------------
    // specialized operations
    // -----------------------------------------------------------------------

    // This function performs SVD decomposition for different groups of learnable  parameters
    template<class ElemType> void ComputationNetwork::PerformSVDecomposition(const map<wstring, float>& SVDConfig)
    {
        vector<pair<vector<wstring>, float>> nodeGroups;
        wregex NameFilter;

        for (const auto & e : SVDConfig)
        {
            wstring regexStr = e.first;
            float keepRatio = e.second;
            vector<wstring> NamesInGroup;

            NameFilter.assign(regexStr);

            for (auto n = m_nameToNodeMap.begin(); n != m_nameToNodeMap.end();  n++)
            {
                if (!regexStr.empty() && !regex_match(n->first, NameFilter))
                {
                    // if regexStr is not empty and the the node node does not match with the regexStr
                    continue;
                }

                shared_ptr<ComputationNode<ElemType>> ptr = dynamic_pointer_cast<LearnableParameter<ElemType>>(n->second);
                if (!ptr)
                    continue;

                Matrix<ElemType> W = ptr->FunctionValues();
                if (W.GetNumCols() == 1 || W.GetNumRows() == 1)
                    continue;

                // still here ?
                NamesInGroup.push_back(n->first);
            }
            nodeGroups.push_back(make_pair(NamesInGroup, keepRatio));
        }

        size_t groupID = 0;
        for (auto& group : nodeGroups)
        {
            float keepratio = group.second;
            fprintf(stderr,
                    "--------------------------------------------------------------------------------------------\n");
            fprintf(stderr,
                    "ParameterSVD: start to process group %d with KeepRatio=%.2f\n",
                    (int) groupID++, keepratio);
            fprintf(stderr,
                    "--------------------------------------------------------------------------------------------\n");

            for (const auto & name : group.first)
            {
                if (m_nameToNodeMap.find(name) == m_nameToNodeMap.end())
                {
                    // could be deleted in the previous groups
                    continue;
                }

                shared_ptr<ComputationNode<ElemType>> pNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(m_nameToNodeMap[name]);
                //========================================
                // Step 1. do SVD decomposition
                //========================================
                Matrix<ElemType> A = pNode->FunctionValues();

                // it is a vector, no need to do it
                if (A.GetNumCols() == 1 || A.GetNumRows() == 1)
                    continue;

                size_t m = A.GetNumRows();
                size_t n = A.GetNumCols();

                Matrix<ElemType> S(-1), U(-1), VT(-1), W(-1);
                std::chrono::time_point < std::chrono::system_clock > stTime = std::chrono::system_clock::now();
                Matrix<ElemType>::SVD(A, S, U, VT, W);
                std::chrono::time_point < std::chrono::system_clock > enTime = std::chrono::system_clock::now();

                // A \in R^{mXn}
                // U \in R^{mXm}
                // VT \in R^{nXn}
                // S \in R^{min(m,n),1}
                // S is in descending order
                //
                ElemType totalenergy = 0.0f;
                for (size_t i = 0; i < S.GetNumRows(); i++)
                    totalenergy += S(i, 0);
                ElemType keepenergy = totalenergy * keepratio;
                ElemType runenergy = 0.0f;

                size_t r = 0;
                for (size_t indx = 0; indx < S.GetNumRows(); indx++)
                {
                    runenergy += S(indx, 0);
                    if (runenergy > keepenergy)
                    {
                        r = indx + 1;
                        break;
                    }
                }

                r = (r + 7) & (~7); //  to keep the number of rows/cols of resultant matrix a multipier of 8
                //  which can be helpful at runtime

                std::chrono::duration<double> elapsedtime = enTime - stTime;
                fprintf(stderr,
                        "Performing SVD for a %5d-by-%-5d matrix (node name: %-20ls) ---  computation time %5.2f secs ;  keep %4.1f%% energy ===> keep %5d svd values (reduce to %4.1f%% parameters) \n",
                        (int) m, (int) n, name.c_str(), elapsedtime.count(),
                        keepratio * 100, (int) r,
                        ((m + n) * r + 0.0f) / m / n * 100);

                // redU in R^ {mXr}
                Matrix<ElemType> redU = U.ColumnSlice(0, r);
                Matrix<ElemType> redVT(-1);

                // redVT in R^{rXn}
                redVT.Resize(r, n);
                redVT.AssignRowSliceValuesOf(VT, 0, r);

                Matrix<ElemType> redS(r, (size_t) 1);
                for (size_t i = 0; i < r; i++)
                {
                    ElemType sqrtsigma = (ElemType) sqrt((double) S(i, 0));
                    redS(i, 0) = sqrtsigma;
                }

                redU.RowElementMultiplyWith(redS.Transpose());
                redVT.ColumnElementMultiplyWith(redS);

                //========================================
                // Step 2. create two new Parameter nodes and one Times node
                //========================================
                wstring leftChildName = name + L"-U";
                wstring rightChildName = name + L"-V";
                shared_ptr<ComputationNode<ElemType>> pLeft =  AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(m_deviceId, leftChildName,  m, r));
                shared_ptr<ComputationNode<ElemType>> pRight = AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(m_deviceId, rightChildName, r, n));

                pLeft->FunctionValues() = redU;
                pRight->FunctionValues() = redVT;

                shared_ptr<ComputationNode<ElemType>> pTimes = AddNodeToNetAndAttachInputs(New<TimesNode<ElemType>>(m_deviceId, name + L"-SVD"), pLeft, pRight);

                //========================================
                // Step 3. remove old node
                //========================================
                ReplaceLeafNode(name, pTimes);
            }
        }
        RebuildNetwork(m_finalCriteria[0]);
    }

    template void ComputationNetwork::InitLearnableParameters<float>(const ComputationNodeBasePtr node, const bool uniformInit, const unsigned long randomSeed, const float initValueScale, bool initOnCPUOnly);
    template void ComputationNetwork::LoadFromFile<float>(const std::wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork* anotherNetwork);
    template void ComputationNetwork::PerformSVDecomposition<float>(const map<wstring, float>& SVDConfig);
    template /*static*/void ComputationNetwork::SetDropoutRate<float>(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed);
    template void ComputationNetwork::SetSeqParam<float>(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const   float hsmoothingWeight, const float frameDropThresh, const bool doreferencealign);

    template void ComputationNetwork::InitLearnableParameters<double>(const ComputationNodeBasePtr node, const bool uniformInit, const unsigned long randomSeed, const double initValueScale, bool initOnCPUOnly);
    template void ComputationNetwork::LoadFromFile<double>(const std::wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork* anotherNetwork);
    template void ComputationNetwork::PerformSVDecomposition<double>(const map<wstring, float>& SVDConfig);
    template /*static*/void ComputationNetwork::SetDropoutRate<double>(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed);
    template void ComputationNetwork::SetSeqParam<double>(ComputationNetwork& net, const ComputationNodeBasePtr criterionNode, const   double hsmoothingWeight, const double frameDropThresh, const bool doreferencealign);
}}}
