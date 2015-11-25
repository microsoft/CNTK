//
// <copyright file="ComputationNetwork.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"  // used for load & save
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "ReshapingNodes.h"
#include "TrainingCriterionNodes.h"
#include "CompositeComputationNodes.h"
#include "EvaluationCriterionNodes.h"
#include "EsotericNodes.h"
#include "MPIWrapper.h"                 // TODO: does not belong here
#include <string>
#include <vector>
#include <list>
#include <set>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // MatrixPool methods
    // -----------------------------------------------------------------------

    template<>
    vector<shared_ptr<Matrix<float>>>& MatrixPool::GetReleasedMatrices<float>()
    {
        return m_releasedFloatMatrices;
    }

    template<>
    vector<shared_ptr<Matrix<double>>>& MatrixPool::GetReleasedMatrices<double>()
    {
        return m_releasedDoubleMatrices;
    }

    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    // TODO: why is this needed? Why is this not just construction?
    void ComputationNetwork::ClearNet()
    {
        for (auto groupIter : GetAllNodeGroups())
            (groupIter)->clear();

        m_recurrentInfo.clear();

        m_built.clear();

        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();
        m_cachedOuterLoopNodes.clear();

        m_inputs.clear();
        m_learnableParameters.clear();

        m_nameToNodeMap.clear();    // will also deref and likely deallocate all nodes we hold in here
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    void ComputationNetwork::SaveToFile(const wstring& fileName, const FileOptions fileFormat) const
    {
        // In case of parallel training only the main node should we saving the model to prevent
        // the parallel training nodes from colliding to write the same file
        // TODO: This does not belong here.
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
    void ComputationNetwork::SaveToFileImpl(const wstring& fileName, const FileOptions fileFormat) const
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
                if (!nodePtr->Inputs(i))
                    fprintf(stderr, "Warning: node %ls 's child is null, please check your ndl/mel file.\n", nodePtr->NodeName().c_str());
                else
                    fstream << nodePtr->Inputs(i)->NodeName();
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

    void ComputationNetwork::LoadPersistableParametersFromFile(const wstring& fileName, const bool requireValidation,
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
            wstring opName, nodeName;
            fstream >> opName >> nodeName;
            ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
            // TODO: don't we have a load constructor? Then when to call which? Document the calling sequence
            nodePtr->LoadFromFile(fstream, modelVersion);
        }

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

        //SetActualMiniBatchSizeFromFeatures();   // TODO: this should go

        if (requireValidation)
        {
            // validation needs some layout to work with
            m_pMBLayout->Init(1, 0, false);
            ValidateNetwork();
        }
    }

    template<class ElemType> void ComputationNetwork::LoadFromFile(const wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork* anotherNetwork)
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
            wstring opName, nodeName;
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
            wstring nodeName;
            size_t numChildren;
            fstream >> nodeName >> numChildren;
            if (numChildren > 0)
            {
                vector<wstring> childrenNames;
                childrenNames.resize(numChildren);
                for (size_t j = 0; j < numChildren; j++)
                {
                    fstream >> childrenNames[j];
                }

                // TODO: how does the file distinguish float from double?
                ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
                vector<ComputationNodeBasePtr> childrenNodes;
                childrenNodes.resize(numChildren);
                for (int j = 0; j < numChildren; j++)
                    childrenNodes[j] = GetNodeFromName(childrenNames[j], anotherNetwork);

                if (nodePtr->OperationName() == OperationNameOf(RowStackNode))
                {
                    //allow for variable input nodes
                    nodePtr->AttachInputs(childrenNodes);
                }
                else
                {
                    //fixed input nodes
                    // TODO: don't we have a variable-length AttachInputs() now?
                    switch (numChildren)
                    {
                        case 1:
                            nodePtr->AttachInputs(childrenNodes[0]);
                            break;
                        case 2:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1]);
                            break;
                        case 3:
                            nodePtr->AttachInputs(childrenNodes[0],childrenNodes[1], childrenNodes[2]);
                            break;
                        case 4:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2], childrenNodes[3]);
                            break;
                        case 5:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2], childrenNodes[3], childrenNodes[4]);
                            break;
                        case 6:
                            nodePtr->AttachInputs(childrenNodes[0], childrenNodes[1], childrenNodes[2], childrenNodes[3], childrenNodes[4], childrenNodes[5]);
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
            wstring nodeName;
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

            // TODO: this section is defunct
            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling"))
            {
                fprintf(stderr, "WARNING: Ignoring defunct 'BNodesReqMultiSeqHandling' section in input file.\n");
                fstream >> num;
                for (size_t i = 0; i<num; i++)
                    fstream >> nodeName;
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

        // some internal values in the nodes are computed during validation
        ValidateNetwork(false, bAllowNoCriterionNode);
        ResetEvalTimeStamp();
    }

    // -----------------------------------------------------------------------
    // node construction
    // -----------------------------------------------------------------------

    ComputationNodeBasePtr ComputationNetwork::SetNodeValue(const wstring & nodeName, const double value)
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
                preComputedNode->FunctionValues().SetValue((float)value);
                preComputedNode->MarkComputed(true);
            }
            else
            {
                auto preComputedNode = AsNodePtr<PreComputedNode<double>>(pNode);
                preComputedNode->FunctionValues().SetValue((double)value);
                preComputedNode->MarkComputed(true);
            }
        }
        else
            LogicError("Only values of learnable parameters and precomputed nodes can be set.");

        return pNode;
    }


    // non-static version needed because it accesses m_randomSeedOffset
    // Excessively used by SimpleNetworkBuilder, but always after CreateLearnableParameter(), so we should really absorb it there
    template<class ElemType> void ComputationNetwork::InitLearnableParameters(const ComputationNodeBasePtr& node, const bool uniformInit, const unsigned long randomSeed, const ElemType initValueScale, bool initOnCPUOnly)
    {
        auto learnableParameterNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(node);
        learnableParameterNode->InitRandom(uniformInit, randomSeed + GetRandomSeedOffset(), initValueScale, initOnCPUOnly);
    }

    // FixupInputMinibatchSize - go through all the inputs and make sure they have a consistent minibatch size (after creation)
    void ComputationNetwork::FixupInputMinibatchSize()
    {
        list<ComputationNodeBasePtr> inputs = GetNodesWithType(OperationNameOf(InputValue));
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
                node->SetDims(node->GetNumRows(), minibatchMax);
        }
    }

    bool ComputationNetwork::IsTypicalCriterionNode(ComputationNodeBasePtr nodePtr)
    {
        // TODO: just use return!
        if (nodePtr->OperationName() == OperationNameOf(SquareErrorNode) ||
			nodePtr->OperationName() == OperationNameOf(LogisticNode) ||
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

    template<class N> void ComputationNetwork::GetNodesRequiringX(list<ComputationNodeBasePtr>& nodesRequiringX, const ComputationNodeBasePtr& rootNode, bool checkComputed)
    {
        if (!rootNode)              // find nodes from all available nodes
        {
            for (const auto & nodep : m_nameToNodeMap)
            {
                auto node = dynamic_pointer_cast<N>(nodep.second);
                if (node)
                {
                    assert(node->RequiresPreCompute());
                    if (!checkComputed || !node->HasComputed())
                        nodesRequiringX.push_back(node);
                }
            }
        }
        else                            // or for calculating a specific node
        {
            for (const auto & nodei : GetEvalOrder(rootNode, false))
            {
                auto node = dynamic_pointer_cast<N>(nodei);
                if (node)
                {
                    assert(node->RequiresPreCompute());
                    if (!checkComputed || !node->HasComputed())
                        nodesRequiringX.push_back(node);
                }
            }
        }
        nodesRequiringX.unique();
    }

    //return list of nodes that require precomputation and not precomputed yet.
    list<ComputationNodeBasePtr> ComputationNetwork::GetNodesRequiringPreComputation(const ComputationNodeBasePtr& rootNode, bool checkComputed)
    {
        list<ComputationNodeBasePtr> nodesRequiringX;
        GetNodesRequiringX<PreComputedNode<float>>(nodesRequiringX, rootNode, checkComputed);
        GetNodesRequiringX<PreComputedNode<double>>(nodesRequiringX, rootNode, checkComputed);
        return nodesRequiringX;
    }

    //return list of nodes that require batch mode and not precomputed yet.
    list<ComputationNodeBasePtr> ComputationNetwork::GetNodesRequiringBatchMode(const ComputationNodeBasePtr& rootNode, bool checkComputed)
    {
        list<ComputationNodeBasePtr> nodesRequiringX;
        GetNodesRequiringX<BatchModeNode<float>>(nodesRequiringX, rootNode, checkComputed);
        GetNodesRequiringX<BatchModeNode<double>>(nodesRequiringX, rootNode, checkComputed);
        return nodesRequiringX;
    }

    // this is called from ClearCache() only, which in turn is called by model editing operations, such as DeleteNode(), and by RebuildNetwork()
    // Basically, it invalidates all post-processing, reducing the network to the graph.
    void ComputationNetwork::ClearCalcOrderCaches()
    {
        for (auto & it : m_cacheEvalOrders)
            for (auto & iter2 : m_cacheEvalOrders[it.first])
                iter2->PurgeStateForFormingRecurrentLoops();
        // TODO: ^^ Why is this done? This looks like an error (this function was called ClearCache() before, so maybe someone threw this call in for good measure)

        // clear network Iterations cache
        m_cacheEvalOrders.clear();
        m_cacheGradientCalcOrders.clear();
        m_cachedOuterLoopNodes.clear();
    }

    // lazily reate the m_inputs[] and m_learnableParameters lists
    // The only other side effect is to call GetEvalOrder(), which will cache the evaluation order for the given root node.
    void ComputationNetwork::CollectInputAndLearnableParameters(const ComputationNodeBasePtr& rootNode)
    {
        //not found
        if (m_inputs.find(rootNode) == m_inputs.end())
        {
            list<ComputationNodeBasePtr> inputs;

            list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = *nodeIter;
                if (node->OperationName() == OperationNameOf(InputValue) /*L"InputValue"*/ ||
                    node->OperationName() == OperationNameOf(SparseInputValue) /*L"SparseInputValue"*/)
                {
                    inputs.push_back(node);
                }
            }
            m_inputs[rootNode] = inputs;
        }

        //not found
        if (m_learnableParameters.find(rootNode) == m_learnableParameters.end())
        {
            list<wstring> learnableParameterNames;
            list<ComputationNodeBasePtr> learnableParameters;

            list<ComputationNodeBasePtr>& nodes = GetEvalOrder(rootNode, false);

            // instead of collecting the nodes themselves, collect the names (they will be sorted below)
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                ComputationNodeBasePtr node = *nodeIter;
                if ((node->OperationName() == OperationNameOf(LearnableParameter) && node->IsParameterUpdateRequired()) ||
                    (node->OperationName() == OperationNameOf(SparseLearnableParameter) && node->IsParameterUpdateRequired()))
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

    template<class ElemType>
    /*static*/void ComputationNetwork::SetDropoutRate(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed)
    {
        if (dropoutRate != prevDropoutRate)
        {
            fprintf(stderr, "Switching dropout rate to %.8g.\n", dropoutRate);
            list<ComputationNodeBasePtr> dropoutNodes = net->GetNodesWithType(OperationNameOf(DropoutNode), criterionNode);
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
    void ComputationNetwork::SetSeqParam(ComputationNetworkPtr net, const ComputationNodeBasePtr criterionNode, double hsmoothingWeight, double frameDropThresh, const bool doreferencealign)
    {
        fprintf(stderr, "Setting Hsmoothing weight to %.8g and frame-dropping threshhold to %.8g\n", hsmoothingWeight, frameDropThresh);
        list<ComputationNodeBasePtr> seqNodes = net->GetNodesWithType(OperationNameOf(SequenceWithSoftmaxNode), criterionNode);
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
                node->SetReferenceAlign(doreferencealign);
            }
        }
    }

    /*static*/void ComputationNetwork::SetMaxTempMemSizeForCNN(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const size_t maxTempMemSizeInSamples)
    {
        fprintf(stderr, "Set Max Temp Mem Size For Convolution Nodes to %lu samples.\n", maxTempMemSizeInSamples);
        list<ComputationNodeBasePtr> convolutionNodes = net->GetNodesWithType(OperationNameOf(ConvolutionNode), criterionNode);
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
    // unit test
    // -----------------------------------------------------------------------

    /**
    call unit test of each node
    this adds a verification of the correctness of node operations.
    */
    bool ComputationNetwork::UnitTest(bool allowFragment)
    {
        vector<wstring> vErrors;
        // currently only validates nodes, we should validate everything we can
        if (FeatureNodes().size() == 0 && !allowFragment)
            RuntimeError("No Feature nodes specified");
        // first give criteria nodes as root node
        if (FinalCriterionNodes().size() > 0)
        {
            for (auto & node : FinalCriterionNodes())
            {
                if (!allowFragment)
                    FormRecurrentLoops(node);
                //this->SetActualMiniBatchSizeFromFeatures();
                if (!UnitTest(node))
                    vErrors.push_back(node->NodeName().c_str());
            }
        }
        else if (!allowFragment)
            RuntimeError("No Criterion nodes specified");
        // now output nodes
        if (OutputNodes().size() > 0)
        {
            for (auto & node : OutputNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
        else if (!allowFragment)
            RuntimeError("No Output nodes specified");
        // now evaluation nodes
        if (EvaluationNodes().size() > 0)
        {
            for (auto & node : EvaluationNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
        return vErrors.empty();
    }

    bool ComputationNetwork::UnitTest(const ComputationNodeBasePtr& rootNode)
    {
        fprintf(stderr, "\n\n Unit test node %ls \n", rootNode->NodeName().c_str());

        std::list<ComputationNodeBasePtr>&  nodes = GetEvalOrder(rootNode, false);

        for (auto & nodeIter : nodes)
            if (!nodeIter->UnitTest())
                return false;

        fprintf(stderr, "\n\n");

        return true;
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

    wstring ComputationNetwork::FormSpecialNodes(wstring style, vector<ComputationNodeBasePtr>& specialNodes)
    {
        if (specialNodes.empty())
            return L"";

        wstring str = style;

        for (const auto & x : specialNodes)
            str = str + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        return str + L"; \n";
    }

    void ComputationNetwork::DescribeNetworkUsingDot(list<ComputationArc>& arcs,
                                                     wstring outFile)
    {
        DotGraphConfigure dotcfg;

        File fstream(outFile,FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

        // get precompute node
        vector<ComputationNodeBasePtr> PreComputedNodes;
        vector<ComputationNodeBasePtr> allnodes = GetAllNodes();
        for (const auto & n : allnodes)
        {
            if (n->RequiresPreCompute())
                PreComputedNodes.push_back(n);
        }

        // get PastValue node
        vector<ComputationNodeBasePtr> pastValueNodes;
        for (const auto & n : allnodes)
        {
            if (n->OperationName() == OperationNameOf(PastValueNode) || n->OperationName() == L"Delay")
                pastValueNodes.push_back(n);
        }

        // get FuturetValue node
        vector<ComputationNodeBasePtr> futureValueNodes;
        for (const auto & n : allnodes)
        {
            if (n->OperationName() == OperationNameOf(FutureValueNode))
                futureValueNodes.push_back(n);
        }
        // get learnableParameters
        vector<ComputationNodeBasePtr> learnableParameters;
        for (const auto & n : allnodes)
        {
            if (n->OperationName() == OperationNameOf(LearnableParameter))
                learnableParameters.push_back(n);
        }

        fstream << "strict digraph {\n";
        fstream << "rankdir = BT ;  \n";

        //////////////////////////////////////////////////////////////////////////
        //    special nodes
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
        // pre-compute nodes
        fstream << FormSpecialNodes(dotcfg.m_PrecomputingNodeStyle, PreComputedNodes);
        // PastValue nodes
        fstream << FormSpecialNodes(dotcfg.m_pastValueNodeStyle, pastValueNodes);
        // FutureValue nodes
        fstream << FormSpecialNodes(dotcfg.m_futureValueNodeStyle, futureValueNodes);
        // normal nodes
        fstream << dotcfg.m_normalNodeStyle << L"\n";

        //////////////////////////////////////////////////////////////////////////
        //    add labels for each node
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
        //    sub-graph
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
        for (const auto & x : m_outputNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (const auto & x : m_pairNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
        for (const auto & x : m_evalNodes)
            line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());

        fstream << line << L"\n}\n";

        //////////////////////////////////////////////////////////////////////////
        //    specify arc connections
        //////////////////////////////////////////////////////////////////////////
        for (auto x = arcs.begin(); x != arcs.end(); x++)
        {
            ComputationNodeBasePtr src = (*x).first;
            ComputationNodeBasePtr des = (*x).second;

            wstring srcname = src->GetName();
            wstring desname = des->GetName();

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

    void ComputationNetwork::PlotNetworkTopology(const wstring outputFile) //  [1/13/2015 erw] plot network topology using dot language
    {
        BuildAndValidateSubNetwork(m_evalNodes[0]);

        //////////////////////////////////////////////////////////////////////////
        //    step 1.        get all the arcs in the network
        //////////////////////////////////////////////////////////////////////////
        unordered_set<ComputationNodeBasePtr> visited;
        list<ComputationArc> arcs;

        for (auto groupIter : GetAllNodeGroups())
        {
            // note: this will also loop over m_features and m_labels, which will do nothing since they have no inputs
            // TODO: test whether that is true
            const auto & group = *groupIter;
            for (size_t i = 0; i < group.size(); i++)
                group[i]->EnumerateArcs(visited, arcs);
        }

        //////////////////////////////////////////////////////////////////////////
        //    step 2.        output dot description
        //////////////////////////////////////////////////////////////////////////
        DescribeNetworkUsingDot(arcs, outputFile);
    }

    // -----------------------------------------------------------------------
    // specialized operations
    // -----------------------------------------------------------------------

    // TODO: lift this into config language, move underlying code to math lib

    //========================================
    // This function performs SVD decomposition for different groups of learnable  parameters
    // we perform SVD decomposition such that
    //  A \approx B*C, where rank(B)=rank(C)=r < rank(A)
    // After SVD decomposition, the node A will become an intermediate node whose children are B,C ;
    // B and C are two learnable parameters
    //========================================
    // BUGBUG: this only currently works for one ElemType, not both
    template<class ElemType> void ComputationNetwork::PerformSVDecomposition(const map<wstring, float>& SVDConfig, size_t AlignedSize)
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
                chrono::time_point < chrono::system_clock > stTime = chrono::system_clock::now();
                Matrix<ElemType>::SVD(A, S, U, VT, W);
                chrono::time_point < chrono::system_clock > enTime = chrono::system_clock::now();

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

                r = r > S.GetNumRows() ? S.GetNumRows() : r;

                if (r%AlignedSize != 0)
                {
                    r -= r%AlignedSize;
                    r = r + AlignedSize > S.GetNumRows() ? S.GetNumRows() : r + AlignedSize;
                }
                // r = (r + 7) & (~7); //  to keep the number of rows/cols of resultant matrix a multipier of 8
                //  which can be helpful at runtime

                chrono::duration<double> elapsedtime = enTime - stTime;
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

    template void ComputationNetwork::InitLearnableParameters<float>(const ComputationNodeBasePtr& node, const bool uniformInit, const unsigned long randomSeed, const float initValueScale, bool initOnCPUOnly);
    template void ComputationNetwork::LoadFromFile<float>(const wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork* anotherNetwork);
    template void ComputationNetwork::PerformSVDecomposition<float>(const map<wstring, float>& SVDConfig, size_t alignedsize);
    template /*static*/void ComputationNetwork::SetDropoutRate<float>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed);
    template void ComputationNetwork::SetSeqParam<float>(ComputationNetworkPtr net, const ComputationNodeBasePtr criterionNode, double hsmoothingWeight, double frameDropThresh, const bool doreferencealign);

    template void ComputationNetwork::InitLearnableParameters<double>(const ComputationNodeBasePtr& node, const bool uniformInit, const unsigned long randomSeed, const double initValueScale, bool initOnCPUOnly);
    template void ComputationNetwork::LoadFromFile<double>(const wstring& fileName, const FileOptions fileFormat, const bool bAllowNoCriterionNode, ComputationNetwork* anotherNetwork);
    template void ComputationNetwork::PerformSVDecomposition<double>(const map<wstring, float>& SVDConfig, size_t alignedsize);
    template /*static*/void ComputationNetwork::SetDropoutRate<double>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double & prevDropoutRate, unsigned long & dropOutSeed);
    template void ComputationNetwork::SetSeqParam<double>(ComputationNetworkPtr net, const ComputationNodeBasePtr criterionNode, double hsmoothingWeight, double frameDropThresh, const bool doreferencealign);

    // register ComputationNetwork with the ScriptableObject system
    ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNetwork> registerComputationNetwork(L"ComputationNetwork");
}}}
