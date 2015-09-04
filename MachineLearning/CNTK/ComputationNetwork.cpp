//
// <copyright file="ComputationNetwork.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"  // used for load & save
//#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
//#include "NonlinearityNodes.h"
//#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
//#include "DecoderNode.h"
#include "TrainingCriterionNodes.h"
#include "CompositeComputationNodes.h"
#include "EvaluationCriterionNodes.h"
#include <string>
#include <fstream>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    template<typename ElemType>
    void ComputationNetwork<ElemType>::ClearNet()
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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::SaveToFile(const std::wstring& fileName, const FileOptions fileFormat) const
    {
       // Saving into temporary file and then renaming it to the requested fileName
       // This is a standard trick to avoid havign corrupted model files if process dies during writing
       wstring tmpFileName = fileName + L".tmp";
       SaveToFileImpl(tmpFileName, fileFormat);
       renameOrDie(tmpFileName, fileName);
    }

    // TODO: how does the file distinguish float vs double nodes?
    template<typename ElemType>
    void ComputationNetwork<ElemType>::SaveToFileImpl(const std::wstring& fileName, const FileOptions fileFormat) const
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
        fstream << m_nodesReqMultiSeqHandling.size();
        for (size_t i = 0; i<m_nodesReqMultiSeqHandling.size(); i++)
            fstream << m_nodesReqMultiSeqHandling[i]->NodeName();
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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::LoadPersistableParametersFromFile(const std::wstring& fileName, const bool requireValidation = true,
                                           const FileOptions fileFormat = FileOptions::fileOptionsBinary)
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

        size_t actualMBSize = GetActualMBSize();
        SetActualMiniBatchSize(actualMBSize);

        if (requireValidation)
        {
            ValidateNetwork();
        }
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    template<typename ElemType>
    ComputationNodeBasePtr ComputationNetwork<ElemType>::SetNodeValue(const std::wstring & nodeName, const double value)
    {
        ComputationNodeBasePtr pNode = GetNodeFromName(nodeName);

        // TODO: this is a bit ugly, but does SetNodeValue() really belong here?
        if (IsNodePtr<LearnableParameter<float>>(pNode))
            AsNodePtr<LearnableParameter<float>>(pNode)->FunctionValues().SetValue((float)value);
        else if (IsNodePtr<LearnableParameter<double>>(pNode))
            AsNodePtr<LearnableParameter<double>>(pNode)->FunctionValues().SetValue((double)value);
        else if (pNode->RequirePreCompute())
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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::SetLearnableNodesBelowNeedGradient(const bool needGradient, const ComputationNodeBasePtr rootNode = nullptr)
    {
        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->OperationName() == LearnableParameter<float>::TypeName())
                    node->NeedGradient() = needGradient;
            }
        }
        else
        {
            //for calculating a specific node
            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if (node->OperationName() == LearnableParameter<float>::TypeName())
                    node->NeedGradient() = needGradient;
            }
        }
    }

    // non-static version needed because it accesses m_randomSeedOffset
    // Excessively used by SimpleNetworkBuilder, but always after CreateLearnableParameter(), so we should really absorb it there
    template<typename ElemType>
    void ComputationNetwork<ElemType>::InitLearnableParameters(const ComputationNodeBasePtr node,
                                                               const bool uniformInit, const unsigned long randomSeed, const ElemType initValueScale,
                                                               bool initOnCPUOnly = false)
    {
        auto learnableParameterNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(node);
        learnableParameterNode->InitRandom(uniformInit, randomSeed + GetRandomSeedOffset(), initValueScale, initOnCPUOnly);
    }

    // FixupInputMinibatchSize - go through all the inputs and make sure they have a consistent minibatch size (after creation)
    template<typename ElemType>
    void ComputationNetwork<ElemType>::FixupInputMinibatchSize()
    {
        std::list<ComputationNodeBasePtr> inputs = GetNodesWithType(InputValue<ElemType>::TypeName());
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

    template<typename ElemType>
    bool ComputationNetwork<ElemType>::IsFuncValueOlderThanInputs(const std::vector<ComputationNodeBasePtr>& recurrentNodes)
    {
        for (auto ptr = recurrentNodes.begin(); ptr != recurrentNodes.end(); ptr++)
        {
            if ((*ptr)->IsFuncValueOlderThanInputs() && 
                (*ptr)->OperationName() != PastValueNode<ElemType>::TypeName() &&
                (*ptr)->OperationName() != FutureValueNode<ElemType>::TypeName())
            {
                return true;
            }
        }
        return false;
    }

    template<typename ElemType>
    bool ComputationNetwork<ElemType>::IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr)
    {
        if (nodePtr->OperationName() == SquareErrorNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == CrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == CrossEntropyNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == ErrorPredictionNode<ElemType>::TypeName() ||               
            nodePtr->OperationName() == CRFNode<ElemType>::TypeName() ||
            nodePtr->OperationName() == DummyCriterionNode<ElemType>::TypeName())
            return true;

        return false;
    }

    template<typename ElemType>
    void ComputationNetwork<ElemType>::SetNodesReqMultiSeqHandling()
    {
        for (auto node : m_nodesReqMultiSeqHandling)
        {
            //SumElements node will generate a scalar value and so it should never require special handling
            //TransposeNode will change the size of columns and so it should also not included for special handling
            //their child node should instead
            if (node->OperationName() != SumElementsNode<ElemType>::TypeName() &&
                node->OperationName() != TransposeNode<ElemType>::TypeName() &&
                node->OperationName() != MeanNode<ElemType>::TypeName() &&
                node->OperationName() != InvStdDevNode<ElemType>::TypeName() 
                )
                node->SetReqMultiSeqHandlingTo(true);
        }

        //if a typical criterion node is used as the training criterion node we assume it requires multiseq handling 
        //this is for backward compatibility
        for (auto node : m_finalCriteria)
            if (IsTypicalCriterionNode(node))
                node->SetReqMultiSeqHandlingTo(true);

        for (auto node : m_evalNodes)
            if (IsTypicalCriterionNode(node))
                node->SetReqMultiSeqHandlingTo(true);
    }


    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has a grammar error, fix
    template<typename ElemType>
    std::list<ComputationNodeBasePtr> ComputationNetwork<ElemType>::GetNodesRequirePreComputation(const ComputationNodeBasePtr rootNode = nullptr, bool checkComputed = true)
    {
        std::list<ComputationNodeBasePtr> nodesRequirePreComputation;

        //find nodes from all available nodes
        if (rootNode == nullptr)
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->RequirePreCompute())
                {
                    auto preComputedNode = static_pointer_cast<PreComputedNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                    {
                        nodesRequirePreComputation.push_back(node);
                    }
                }
            }
        }
        else //for calculating a specific node
        {
            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = *nodeIter;
                if (node->RequirePreCompute())
                {
                    auto preComputedNode = static_pointer_cast<PreComputedNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                    {
                        nodesRequirePreComputation.push_back(node);
                    }
                }
            }
        }

        return nodesRequirePreComputation;
    }

    //return list of nodes that require precomputation and not precomputed yet.
    // TODO: name has grammar error, fix
    template<typename ElemType>
    std::list<ComputationNodeBasePtr> ComputationNetwork<ElemType>::GetNodesRequireBatchMode(const ComputationNodeBasePtr rootNode = nullptr, bool checkComputed = true)
    {
        std::list<ComputationNodeBasePtr> nodesRequirePreComputation;

        if (rootNode == nullptr) //find nodes from all available nodes
        {
            for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = nodeIter->second;
                if (node->RequireBatchMode())
                {
                    auto preComputedNode = static_pointer_cast<BatchModeNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                        nodesRequirePreComputation.push_back(node);
                }
            }
        }
        else //for calculating a specific node
        {
            std::list<ComputationNodeBasePtr>&  nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if (node->RequireBatchMode())
                {
                    auto preComputedNode = static_pointer_cast<BatchModeNode<ElemType>>(node);
                    if (!checkComputed || !preComputedNode->HasComputed())
                        nodesRequirePreComputation.push_back(node);
                }
            }
        }

        return nodesRequirePreComputation;
    }

    // The methods below determine evaluation order, which is tricky in presence of recurrent loops.
    // TODO: Can this be moved to a separate class, or at least a separate CPP?

    template<typename ElemType>
    void ComputationNetwork<ElemType>::ClearCalcOrderCaches()
    {
        for (typename std::map<const ComputationNodeBasePtr, std::list<ComputationNodeBasePtr>>::iterator it = m_cacheEvalOrders.begin(); it != m_cacheEvalOrders.end(); ++it)
            for (auto iter2 = m_cacheEvalOrders[it->first].begin(); iter2 != m_cacheEvalOrders[it->first].end(); iter2++)
                (*iter2)->clearCache();
        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();
    }

    template<typename ElemType>
    void ComputationNetwork<ElemType>::MergeRecurrentLoops(const ComputationNodeBasePtr /*rootNode*/)
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
    template<typename ElemType>
    void ComputationNetwork<ElemType>::getStrongSCC(const ComputationNodeBasePtr rootNode)    // TODO: method names start uppercase
    {
                    /// notice that this graph including graphs from a parent networks if two or more networks are connected via pairnetwork node
        std::unordered_set<ComputationNodeBasePtr> visited;
        std::list<ComputationNodeBasePtr> sccStack;
        size_t index = 0;
        size_t loopId = 0;
        if (rootNode->isVisisted() == false)
            strongSCC(rootNode, sccStack, index, loopId);
    }

    template<typename ElemType>
    void ComputationNetwork<ElemType>::strongSCC(ComputationNodeBasePtr cur,      // TODO: method names start uppercase
                                                 std::list<ComputationNodeBasePtr>& sccStack,
                                                 size_t& index, size_t& loopId)
    {
        cur->SetIndex(index);
        cur->Setlowlink(index);
        index++;

        cur->SetVisited(true);
        sccStack.push_back(cur);
        cur->SetInStack(true);

        if (cur->OperationName() != L"PairNetwork")
        {
            // pairnetwork is the socket from other network, so ignore its children, which are in the other networks
            for (int i = 0; i < cur->ChildrenSize(); i++)
            {
                if (cur->GetChildren()[i]->isVisisted() == false)
                {
                    strongSCC(cur->GetChildren()[i], sccStack, index, loopId);
                    cur->Setlowlink(min(cur->Getlowlink(), cur->GetChildren()[i]->Getlowlink()));
                }
                else if (cur->GetChildren()[i]->isInStack())
                {
                    cur->Setlowlink(min(cur->Getlowlink(), cur->GetChildren()[i]->Getlowlink()));
                }
            }
        }

        if (cur->Getlowlink() == cur->GetIndex())
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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::getLoopForwordOrder(std::unordered_set<ComputationNodeBasePtr>& visited,   // TODO: method name
                                                           std::unordered_set<ComputationNodeBasePtr>& recStack,
                                                           std::list<ComputationNodeBasePtr>& nodesStack,
                                                           ComputationNodeBasePtr cur)
    {
        if (visited.find(cur) == visited.end())
        {
            visited.insert(cur);
            recStack.insert(cur);

            if (cur->OperationName() != PastValueNode<ElemType>::TypeName() && 
                cur->OperationName() != FutureValueNode<ElemType>::TypeName())
            {
                for (size_t i = 0; i < cur->ChildrenSize(); i++)
                    if (cur->GetChildren()[i]->LoopId() == cur->LoopId())
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
            
    //must be called before ValidateNetwork
    template<typename ElemType>
    void ComputationNetwork<ElemType>::FormRecurrentLoops(const ComputationNodeBasePtr rootNode)
    {
        std::vector<ComputationNodeBasePtr> sourceLoopNodes;

        getStrongSCC(rootNode);
        std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, sourceLoopNodes);
        std::list<ComputationNodeBasePtr> nodesForGrad;

        MergeRecurrentLoops(rootNode);

        /// debug purpose
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
                        if (nodeRecIter->GetChildren()[i]->LoopId() == nodeRecIter->LoopId() && 
                            nodeRecIter->OperationName() != PastValueNode<ElemType>::TypeName() &&
                            nodeRecIter->OperationName() != FutureValueNode<ElemType>::TypeName())
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
            nodesForGrad = nodes;
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
        
        for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
            (*iter)->clearCache();
    }

    template<typename ElemType>
    void ComputationNetwork<ElemType>::DetermineLoopTypes()
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

                    if (nodeRecIter->OperationName() == PastValueNode<ElemType>::TypeName())
                    {
                        hasPastValueNode = true;
                    }
                    else if (nodeRecIter->OperationName() == FutureValueNode<ElemType>::TypeName())
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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::ReorderLoops(std::list<ComputationNodeBasePtr>& nodes,
                                                    const std::map<int, std::list<ComputationNodeBasePtr>>& /*recurrentNodes*/,
                                                    const std::list<ComputationNodeBasePtr> & /*noRecurrentNodes*/)
    {
        std::list<ComputationNodeBasePtr> newList;

        std::list<ComputationNodeBasePtr> vTmp;
        std::list<ComputationNodeBasePtr> vRecurrentTmp;
        //int  prevId = -1;
        vector<bool> accessed;
        accessed.assign(m_recurrentInfo.size(), false);
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            int iId = FindInRecurrentLoop(*nodeIter);
            if (iId >= 0)
            {

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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::CollectInputAndLeanableParameters(const ComputationNodeBasePtr rootNode)
    {
        //not found
        if (m_inputs.find(rootNode) == m_inputs.end())
        {
            std::list<ComputationNodeBasePtr> inputs;

            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end();
                    nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if (node->OperationName() == InputValue<ElemType>::TypeName() /*L"InputValue"*/ ||
                    node->OperationName() == InputValue<ElemType>::SparseTypeName())
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

            std::list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode);
            ;
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = (*nodeIter);
                if ((node->OperationName() == LearnableParameter<ElemType>::TypeName() && node->NeedGradient()) ||
                    (node->OperationName() == SparseLearnableParameter<ElemType>::TypeName() && node->NeedGradient()))
                {
                    learnableParameterNames.push_back(node->NodeName());
                }
            }

            //we need to sort it so that we get consistent order when load it from saved file
            learnableParameterNames.sort();
            for (auto nodeNameIter = learnableParameterNames.begin(); nodeNameIter != learnableParameterNames.end(); nodeNameIter++)
            {
                learnableParameters.push_back(GetNodeFromName((*nodeNameIter)));
            }

            m_learnableParameters[rootNode] = learnableParameters;
        }
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    template<typename ElemType>
    void ComputationNetwork<ElemType>::LoadFromFile(const std::wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork<ElemType>* anotherNetwork)
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

                if (nodePtr->OperationName() == RowStackNode<float>::TypeName()) {
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
                    m_nodesReqMultiSeqHandling.push_back(GetNodeFromName(nodeName));
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

    template<typename ElemType>
    wstring ComputationNetwork<ElemType>::FormSpecialNodes(wstring style, std::vector<ComputationNodeBasePtr>& specialNodes)
    {
        if (specialNodes.empty())
            return L"";

        wstring str = style;

        for (auto x : specialNodes)
            str = str + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        return str + L"; \n";
    }

    template<typename ElemType>
    void ComputationNetwork<ElemType>::DescribeNetworkUsingDot(std::list<ComputationArc>& arcs,
                                                               std::wstring outFile)
    {
        DotGraphConfigure dotcfg;

        File fstream(outFile,FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        // get precompute node
        std::vector<ComputationNodeBasePtr> PreComputedNodes;
        std::vector<ComputationNodeBasePtr> allnodes = GetAllNodes();
        for (auto n : allnodes)
        {
            if (n->RequirePreCompute())
            {
                PreComputedNodes.push_back(n);
            }
        }

        // get PastValue node
        std::vector<ComputationNodeBasePtr> pastValueNodes;
        for (auto n : allnodes)
        {
            if (n->OperationName() == PastValueNode<ElemType>::TypeName() || 
                n->OperationName() == L"Delay")
            {
                pastValueNodes.push_back(n);
            }
        }

        // get FuturetValue node
        std::vector<ComputationNodeBasePtr> futureValueNodes;
        for (auto n : allnodes)
        {
            if (n->OperationName() == FutureValueNode<ElemType>::TypeName())
            {
                futureValueNodes.push_back(n);
            }
        }
        // get learnableParameters
        std::vector<ComputationNodeBasePtr> learnableParameters;
        for (auto n : allnodes)
        {
            if (n->OperationName() == LearnableParameter<ElemType>::TypeName())
            {
                learnableParameters.push_back(n);
            }
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
        fstream << FormSpecialNodes(dotcfg.m_nodesReqMultiSeqHandlingStyle, m_nodesReqMultiSeqHandling);            
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
        for (auto x : allnodes)
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
        for (auto x : m_features)
        {
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        }
        fstream << line << L"\n}\n";

        // subgraph eval/output/criteria
        fstream << L"subgraph {\n";
        fstream << L"\t\t rank=sink ; ";
        line.clear();
        for (auto x : m_finalCriteria)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_nodesReqMultiSeqHandling)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_outputNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_pairNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (auto x : m_evalNodes)
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

            if (des->OperationName() == PastValueNode<ElemType>::TypeName() || des->OperationName() == L"Delay")
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
            else if (des->OperationName() == FutureValueNode<ElemType>::TypeName())
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

    template<typename ElemType>
    void ComputationNetwork<ElemType>::PlotNetworkTopology(const std::wstring outputFile) //  [1/13/2015 erw] plot network topology using dot language
    {
        BuildAndValidateNetwork(m_evalNodes[0]);

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
    template<typename ElemType>
    void ComputationNetwork<ElemType>::PerformSVDecomposition(const map<wstring, float>& SVDConfig)
    {
        vector<pair<vector<wstring>, float>> nodeGroups;
        wregex NameFilter;

        for (auto e : SVDConfig)
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

                ComputationNodePtr ptr = dynamic_pointer_cast<LearnableParameter<ElemType>>(n->second);
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

            for (auto name : group.first)
            {
                if (m_nameToNodeMap.find(name) == m_nameToNodeMap.end())
                {
                    // could be deleted in the previous groups
                    continue;
                }

                ComputationNodePtr pNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(m_nameToNodeMap[name]);
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
                ComputationNodePtr pLeft =  AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(m_deviceId, leftChildName,  m, r));
                ComputationNodePtr pRight = AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(m_deviceId, rightChildName, r, n));

                pLeft->FunctionValues() = redU;
                pRight->FunctionValues() = redVT;

                ComputationNodePtr pTimes = AddNodeToNetAndAttachInputs(New<TimesNode<ElemType>>(m_deviceId, name + L"-SVD"), pLeft, pRight);

                //========================================
                // Step 3. remove old node
                //========================================
                ReplaceLeafNode(name, pTimes);
            }
        }
        RebuildNetwork(m_finalCriteria[0]);
    }

    template class ComputationNetwork<float>;
    template class ComputationNetwork<double>;

}}}
