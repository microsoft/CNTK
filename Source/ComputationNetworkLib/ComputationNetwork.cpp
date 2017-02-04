//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h" // used for load & save
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RNNNodes.h"
#include "RecurrentNodes.h"
#include "ReshapingNodes.h"
#include "TrainingNodes.h"
#include "PreComputeNodes.h"
#include "EvaluationNodes.h"
#include "SpecialPurposeNodes.h"
#include "DeprecatedNodes.h" // (for SaveToDbnFile(), which is also deprecated)
#include "MPIWrapper.h" // TODO: does not belong here
#include <string>
#include <vector>
#include <stack>
#include <list>
#include <set>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// MatrixPool methods
// -----------------------------------------------------------------------

template <>
vector<shared_ptr<Matrix<float>>>& MatrixPool::GetReleasedMatrices<float>()
{
    return m_releasedFloatMatrices;
}

template <>
vector<shared_ptr<Matrix<double>>>& MatrixPool::GetReleasedMatrices<double>()
{
    return m_releasedDoubleMatrices;
}

// -----------------------------------------------------------------------
// construction
// -----------------------------------------------------------------------

// clear the object to empty state; this is used in the destructor, and also when loading
// This is necessary to make sure we don't leave nodes hanging due to recurrent cyclic references.
void ComputationNetwork::ClearNetwork()
{
    // release all references to nodes
    InvalidateCompiledNetwork();

    for (auto groupIter : GetAllNodeGroups())
        groupIter->clear();

    // break cycles
    // Note: During editing, new networks may be constructed by "sharing" portions of other networks. Nodes cannot, however, be shared during evaluation.
    // I.e. the networks that are shared from can no longer be evaluated, but they must still be releasable.
    // Hence we must only break cycles for nodes that have not been taken over into another network. We know from their m_environment pointer.
    // (The correct way to do this is to use weak pointers, but that's too intrusive a change for now.)
    for (auto& iter : m_nameToNodeMap)
    {
        auto& node = iter.second;
        if (node->GetEnvironmentPtr() != m_environment)
            continue; // was taken over by another network
        node->SetEnvironment(nullptr);
        node->DetachInputs();
    }

    m_nameToNodeMap.clear();

    m_pMBLayoutOfNetwork->Init(1, 0);
}

// -----------------------------------------------------------------------
// serialization
// -----------------------------------------------------------------------

// after after editing--network is possibly not validated/compiled
void ComputationNetwork::SaveEdited(const wstring& fileName, const FileOptions fileFormat)
{
    if (!IsCompiled())
        CompileNetwork();
    Save(fileName, fileFormat);
}

void ComputationNetwork::Save(const wstring& fileName, const FileOptions fileFormat) const
{
    VerifyIsCompiled("Save");
    // Saving into temporary file and then renaming it to the requested fileName
    // This is a standard trick to avoid havign corrupted model files if process dies during writing
    wstring tmpFileName = fileName + L".tmp";
    SaveToFileImpl(tmpFileName, fileFormat);
    renameOrDie(tmpFileName, fileName);
}

// TODO: how does the file distinguish float vs double nodes?
void ComputationNetwork::SaveToFileImpl(const wstring& fileName, const FileOptions fileFormat) const
{
    File fstream(fileName, fileFormat | FileOptions::fileOptionsWrite);
    // Buffer writes in memory then flush to filesystem, which reduces number of small writes
    fstream.Setvbuf();
    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCN");

    // model version
    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BVersion");
    fstream << (size_t) CURRENT_CNTK_MODEL_VERSION;
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EVersion");

    fstream << (size_t) m_nameToNodeMap.size();

    // put all node info first
    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr nodePtr = nodeIter->second;
        // type
#if CURRENT_CNTK_MODEL_VERSION >= CNTK_MODEL_VERSION_7
        wstring precision;
        if (nodePtr->Is<ComputationNode<float>>())
            precision = ElemTypeName<float>();
        else if (nodePtr->Is<ComputationNode<double>>())
            precision = ElemTypeName<double>();
        else LogicError("Unexpected node type.");
        fstream << precision;
#endif
        fstream << nodePtr->OperationName();
        // name
        fstream << nodePtr->NodeName();
        // content
        nodePtr->Save(fstream);
    }

    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ENodeList");

    // put relationship
    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRelation");
    for (auto nodeIter = m_nameToNodeMap.begin(); nodeIter != m_nameToNodeMap.end(); nodeIter++)
    {
        ComputationNodeBasePtr nodePtr = nodeIter->second;
        fstream << nodePtr->NodeName() << nodePtr->GetNumInputs();
        for (size_t i = 0; i < nodePtr->GetNumInputs(); i++)
        {
            if (!nodePtr->Input(i))
                fprintf(stderr, "Warning: node %ls 's child is null, please check your ndl/mel file.\n", nodePtr->NodeName().c_str());
            else
                fstream << nodePtr->Input(i)->NodeName();
        }
    }
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERelation");

    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");

    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes");
    fstream << m_featureNodes.size();
    for (size_t i = 0; i < m_featureNodes.size(); i++)
        fstream << m_featureNodes[i]->NodeName();
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");

    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes");
    fstream << m_labelNodes.size();
    for (size_t i = 0; i < m_labelNodes.size(); i++)
        fstream << m_labelNodes[i]->NodeName();
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCriterionNodes");
    fstream << m_criterionNodes.size();
    for (size_t i = 0; i < m_criterionNodes.size(); i++)
        fstream << m_criterionNodes[i]->NodeName();
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECriterionNodes");

    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes");
    fstream << m_evaluationNodes.size();
    for (size_t i = 0; i < m_evaluationNodes.size(); i++)
        fstream << m_evaluationNodes[i]->NodeName();
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");

    fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes");
    fstream << m_outputNodes.size();
    for (size_t i = 0; i < m_outputNodes.size(); i++)
        fstream << m_outputNodes[i]->NodeName();
    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");

    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

    fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECN");

    fstream.Flush();
}


size_t ComputationNetwork::GetModelVersion(File& fstream) 
{
    fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCN");

    // model version
    size_t modelVersion = CNTK_MODEL_VERSION_1; // if version info is not there it is version 1
    if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
    {
        fstream >> modelVersion;
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
    }
    if (modelVersion > CURRENT_CNTK_MODEL_VERSION)
        InvalidArgument("Read: The model file has a newer format version (%d) than this CNTK version can handle (%d).", (int)modelVersion, (int)CURRENT_CNTK_MODEL_VERSION);
    
    return modelVersion;
}

// load the section of nodes that contain persistable parameters
// This is also used for reloading a model without recreating it, e.g. during training.
// TODO: Why not just reload it? Because SGD::Train() holds pointers to the parameters directly? That should be fixed.
template <class ElemType> // ElemType is the default for models prior to CNTK_MODEL_VERSION_7; after that, it is serialized, and ElemType is ignored
void ComputationNetwork::ReadPersistableParameters(size_t modelVersion, File& fstream, bool create)
{
    size_t numNodes;
    fstream >> numNodes;

    // get all node info first
    fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BNodeList");
    for (size_t i = 0; i < numNodes; i++)
    {
        wstring precision;
        if (modelVersion >= CNTK_MODEL_VERSION_7)
            fstream >> precision; // "float" or "double"; default is "" meaning <ElemType> as passed in from outside

        wstring opName, nodeName;
        fstream >> opName >> nodeName;

        ComputationNodeBasePtr node;
        if (!create) // reloading existing
            node = GetNodeFromName(nodeName);
        else if (precision == L"float")
            node = ComputationNetworkBuilder<float>::NewNode(opName, m_deviceId, nodeName);
        else if (precision == L"double")
            node = ComputationNetworkBuilder<double>::NewNode(opName, m_deviceId, nodeName);
        else if (precision == L"") // old file format: default to <ElemType>
            node = ComputationNetworkBuilder<ElemType>::NewNode(opName, m_deviceId, nodeName);
        else
            RuntimeError("Read: Unexpected precision tag '%ls'", precision.c_str());

        node->Load(fstream, modelVersion);

        if (create) // loaded from scratch
            AddNodeToNet(node);
        else                      // reloaded existing
        {
            let old = node->GetSampleLayout();
            let changed = ValidateNode(node, /*isFinalValidationPass=*/true);
            if (changed)
            {
                let upd = node->GetSampleLayout();
                fprintf(stderr, "ValidateSubNetwork: %ls %ls operation changed, from [%s] to [%s].", node->NodeName().c_str(), node->OperationName().c_str(),
                    string(old).c_str(), string(upd).c_str());
                //LogicError("ValidateSubNetwork: %ls %ls operation changed during reload or re-validation.", node->NodeName().c_str(), node->OperationName().c_str());
            }
        }
    }

    fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodeList");
}

// deserialize the model
// This does not post-process the model (CompileNetwork()). Use Load() instead.
template <class ElemType> // for ReadPersistableParameters()
void ComputationNetwork::Read(const wstring& fileName)
{
    ClearNetwork();

    File fstream(fileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);

    auto modelVersion = GetModelVersion(fstream);

    ReadPersistableParameters<ElemType>(modelVersion, fstream, true);

    size_t numNodes = m_nameToNodeMap.size();

    // get relationship
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
                fstream >> childrenNames[j];

            // TODO: how does the file distinguish float from double?
            ComputationNodeBasePtr nodePtr = GetNodeFromName(nodeName);
            vector<ComputationNodeBasePtr> childrenNodes;
            childrenNodes.resize(numChildren);
            for (int j = 0; j < numChildren; j++)
                childrenNodes[j] = GetNodeFromName(childrenNames[j]);

            if (modelVersion < CNTK_MODEL_VERSION_19 && nodePtr->OperationName() == OperationNameOf(BatchNormalizationNode)) 
            {
                ComputationNodeBasePtr runSampleCount = New<LearnableParameter<ElemType>>(m_deviceId, nodeName + L".run_sample_count", TensorShape(1));
                runSampleCount->SetLearningRateMultiplier(0);
                AddNodeToNet(runSampleCount);
                InitLearnableParameters(runSampleCount, L"fixedValue", 0);
                childrenNodes.push_back(runSampleCount);
            }

            nodePtr->AttachInputs(childrenNodes);
        }
    }
    fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERelation");

    fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BRootNodes");
    {
        wstring nodeName;
        size_t num;

        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BFeatureNodes"))
        {
            fstream >> num;

            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                AddToNodeGroup(L"feature", GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EFeatureNodes");
        }

        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BLabelNodes"))
        {
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                AddToNodeGroup(L"label", GetNodeFromName(nodeName));
            }
        }
        // BUGBUG: Should this be inside the block?
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ELabelNodes");

        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BCriterionNodes") ||
            fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BCriteriaNodes" /*legacy*/))
        {
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                AddToNodeGroup(L"criterion", GetNodeFromName(nodeName));
            }

            if (!fstream.TryGetMarker(FileMarker::fileMarkerEndSection, L"ECriteriaNodes" /*legacy*/))
            {
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECriterionNodes"); // check legacy first so err msg will use new name
            }
        }

        // this section is for back compat only, skip over
        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BNodesReqMultiSeqHandling"))
        {
            fprintf(stderr, "WARNING: Ignoring defunct 'BNodesReqMultiSeqHandling' section in input file.\n");
            fstream >> num;
            for (size_t i = 0; i < num; i++)
                fstream >> nodeName; // dummy
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ENodesReqMultiSeqHandling");
        }

        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BEvalNodes"))
        {
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                AddToNodeGroup(L"evaluation", GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EEvalNodes");
        }

        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BOutputNodes"))
        {
            fstream >> num;
            for (size_t i = 0; i < num; i++)
            {
                fstream >> nodeName;
                AddToNodeGroup(L"output", GetNodeFromName(nodeName));
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EOutputNodes");
        }

        // this section is for back compat only, skip over
        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BPairNodes"))
        {
            fstream >> num;
            if (num > 0)
                RuntimeError("Read: PairNodes are no longer supported");
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EPairNodes");
        }
    }
    fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ERootNodes");

    fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECN");
}

// -----------------------------------------------------------------------
// node construction
// -----------------------------------------------------------------------

// helper of InitLearnableParameters()
// Note: This should really be done through an interface without <ElemType> that LearnableParameter would derive from.
// However, this is only for NDL (which is deprecated), so I rather not pollute the code with more interfaces just for a deprecated cause.
template<class ElemType>
static bool TryPostInitParameters(const ComputationNodeBasePtr& node, const wchar_t* initString, double initValue, unsigned long randomSeed, bool initOnCPUOnly)
{
    auto learnableParameterNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(node);
    if (!learnableParameterNode)
        return false;
    learnableParameterNode->PostInitParameters(initString, (ElemType) initValue, randomSeed, initOnCPUOnly);
    return true;
}

// non-static version needed because it accesses m_randomSeedOffset
void ComputationNetwork::InitLearnableParameters(const ComputationNodeBasePtr& node,
                                                 const wchar_t* initString, // "uniform"|"gaussian"|"fixedValue"
                                                 double initValue,        //  scale   | scale    | value
                                                 unsigned long randomSeed /*= 0*/,
                                                 bool initOnCPUOnly /*= false*/) const
{
    randomSeed += GetRandomSeedOffset();
    if (TryPostInitParameters<float> (node, initString, initValue, randomSeed, initOnCPUOnly) ||
        TryPostInitParameters<double>(node, initString, initValue, randomSeed, initOnCPUOnly))
        return;
    LogicError("InitLearnableParameters: Input node is not a LearnableParameter<float or double>");
}

// non-static version needed because it accesses m_randomSeedOffset
// Legacy version that is for random only.
void ComputationNetwork::RandomInitLearnableParameters(const ComputationNodeBasePtr& node, const bool uniformInit, const unsigned long randomSeed, const double initValueScale, bool initOnCPUOnly) const
{
    InitLearnableParameters(node, uniformInit ? L"uniform" : L"gaussian", initValueScale, randomSeed, initOnCPUOnly);
}

template <class ElemType>
void ComputationNetwork::InitLearnableParametersWithBilinearFill(const ComputationNodeBasePtr& node, size_t kernelWidth, size_t kernelHeight)
{
    auto learnableParameterNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(node);
    learnableParameterNode->InitBilinear(kernelWidth, kernelHeight);
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
        nodePtr->OperationName() == OperationNameOf(ClassificationErrorNode) ||
        nodePtr->OperationName() == OperationNameOf(EditDistanceErrorNode) ||
#ifdef COMING_SOON
        nodePtr->OperationName() == OperationNameOf(CRFNode) ||
#endif
        nodePtr->OperationName() == OperationNameOf(DummyCriterionNode))
        return true;

    return false;
}

// return list of nodes that require precomputation and not precomputed yet
list<ComputationNodeBasePtr> ComputationNetwork::GetNodesRequiringPreComputation(const ComputationNodeBasePtr& rootNode, bool checkComputed)
{
    list<ComputationNodeBasePtr> nodes;
    for (const auto& node : GetEvalOrder(rootNode)) // TODO: verify that order does not matter here, then replace by GetAllNodesForRoot()
    {
        auto pcnode = dynamic_pointer_cast<IPreComputeNode>(node);
        if (pcnode)
        {
            assert(node->RequiresPreCompute());
            if (!checkComputed || !pcnode->HasComputed())
                nodes.push_back(node);
        }
    }
    return nodes;
}

// create the m_inputValues[] and m_learnableParameters[] lists
// This enumerates all leaves reachable from rootNode.
// Leaves are:
//  - inputs
//  - learnable parameters
// It does not traverse disabled ones, i.e.
//  - inputs that are only reachable through PrecomputeNodes that have completed computation
//  - learnable parameters that are constants
void ComputationNetwork::CollectInputAndLearnableParameters(const ComputationNodeBasePtr& rootNode)
{
    assert(m_inputValues.find(rootNode) == m_inputValues.end()); // this function must only be called once
    assert(m_learnableParameters.find(rootNode) == m_learnableParameters.end());

    // gather the lists
    set<ComputationNodeBasePtr> visited;
    list<ComputationNodeBasePtr> inputs, learnableParameters;
    if (rootNode)
        CollectInputAndLearnableParametersRec(rootNode, visited, inputs, learnableParameters);
    else
        for (const auto& root : m_allRoots)
            CollectInputAndLearnableParametersRec(root, visited, inputs, learnableParameters);

    // sort learnable parameters by name so that we get consistent order when load it from saved file
    learnableParameters.sort([](const ComputationNodeBasePtr& a, const ComputationNodeBasePtr& b)
    {
        return a->NodeName() < b->NodeName();
    });

    m_inputValues[rootNode] = move(inputs);
    m_learnableParameters[rootNode] = move(learnableParameters);
}

void ComputationNetwork::CollectInputAndLearnableParametersRec(const ComputationNodeBasePtr& node, set<ComputationNodeBasePtr>& visited, list<ComputationNodeBasePtr>& inputs, list<ComputationNodeBasePtr>& learnableParameters)
{
    if (visited.find(node) != visited.end())    // allready got this one
        return;
    else if (node->OperationName() == OperationNameOf(InputValue) || node->OperationName() == OperationNameOf(SparseInputValue))
        inputs.push_back(node);
    else if (node->OperationName() == OperationNameOf(LearnableParameter) && node->IsParameterUpdateRequired())
        learnableParameters.push_back(node);
    else
    {
        // PreComputeNodes that are already done should not be traversed
        auto pcnode = dynamic_pointer_cast<IPreComputeNode>(node);
        if (pcnode && pcnode->HasComputed())
            return;
        // recurse
        visited.insert(node);
        for (const auto & input : node->GetInputs())
            CollectInputAndLearnableParametersRec(input, visited, inputs, learnableParameters);
    }
}

template <class ElemType>
/*static*/ void ComputationNetwork::SetDropoutRate(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double& prevDropoutRate)
{
    list<ComputationNodeBasePtr> dropoutNodes = net->GetNodesWithType(OperationNameOf(DropoutNode), criterionNode);
    if (dropoutRate != prevDropoutRate)
    {
        fprintf(stderr, "Setting dropout rate to %.8g.\n", dropoutRate);
        // TODO: Change this to use an interface that is independent of <ElemType>.
        if (dropoutNodes.size() == 0 && dropoutRate > 0)
            fprintf(stderr, "WARNING: Attempting to set dropout rate, but there is no dropout node in the network.\n");
    }

    for (auto& nodeIter : dropoutNodes)
    {
        auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(nodeIter);
        if (dropoutRate != prevDropoutRate)
            node->SetDropoutRate(dropoutRate);
    }

    prevDropoutRate = dropoutRate;
}

template <class ElemType>
/* static */ void ComputationNetwork::SetIRngUserSeed(ComputationNetworkPtr net, const ComputationNodeBasePtr& node, size_t randSeedBase)
{
    // Predicate checking if the node is derived from IRngUser
    function<bool(const ComputationNodeBasePtr&)> nodeIsIRngUser = [](const ComputationNodeBasePtr& p) { return dynamic_pointer_cast<IRngUser>(p) != nullptr; };

    list<ComputationNodeBasePtr> rngUserNodes = net->GetNodesWhere(nodeIsIRngUser, node);

    // Each IRngUser gets a distinct seed. This seed is computed as follows:
    // seed = (((parallelWorkerIdx * maxEpochs) + currentEpochNum) /*i.e. randSeedBase*/ * rngUserNodes.size()) + dropoutNodeIdx.
    size_t randSeed = randSeedBase * rngUserNodes.size();
    for (auto& nodeIter : rngUserNodes)
    {
        auto rngUser = dynamic_pointer_cast<IRngUser>(nodeIter);
        rngUser->SetRngState(randSeed);
        randSeed++;
    }
}

template <class ElemType>
/*static*/ void ComputationNetwork::SetBatchNormalizationTimeConstants(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode,
                                                                       double normalizationTimeConstant, double& prevNormalizationTimeConstant,
                                                                       double blendTimeConstant, double& prevBlendTimeConstant)
{
    if (normalizationTimeConstant != prevNormalizationTimeConstant || blendTimeConstant != prevBlendTimeConstant)
    {
        if (normalizationTimeConstant != prevNormalizationTimeConstant)
            fprintf(stderr, "Setting batch normalization time constant to %.8g.\n", normalizationTimeConstant);
        if (blendTimeConstant != prevBlendTimeConstant)
            fprintf(stderr, "Setting batch normalization blend time constant to %.8g.\n", blendTimeConstant);
        // TODO: Change this to use an interface that is independent of <ElemType>.
        auto batchNormalizationNodes = net->GetNodesWithType(OperationNameOf(BatchNormalizationNode), criterionNode);
        if (batchNormalizationNodes.size() == 0)
            fprintf(stderr, "WARNING: there is no batch normalization node.\n");
        else
        { 
            for (auto& nodeIter : batchNormalizationNodes)
            {
                auto node = dynamic_pointer_cast<BatchNormalizationNode<ElemType>>(nodeIter);
                node->SetNormalizationTimeConstants(normalizationTimeConstant, prevNormalizationTimeConstant,
                                                    blendTimeConstant, prevBlendTimeConstant);
            }
        }

        prevNormalizationTimeConstant = normalizationTimeConstant;
        prevBlendTimeConstant = blendTimeConstant;
    }
}

//set sequence training parameters, e.g. smoothing weight, frame drop threshhold
template <class ElemType>
void ComputationNetwork::SetSeqParam(ComputationNetworkPtr net,
                                     const ComputationNodeBasePtr criterionNode,
                                     const double& hsmoothingWeight,
                                     const double& frameDropThresh,
                                     const bool& doreferencealign,
                                     const double& amf /*= 14.0f*/,
                                     const double& lmf /*= 14.0f*/,
                                     const double& wp /*= 0.0f*/,
                                     const double& bMMIfactor /*= 0.0f*/,
                                     const bool& sMBR /*= false*/
                                     )
{
    fprintf(stderr, "Setting Hsmoothing weight to %.8g and frame-dropping threshhold to %.8g\n", hsmoothingWeight, frameDropThresh);
    fprintf(stderr, "Setting SeqGammar-related parameters: amf=%.2f, lmf=%.2f, wp=%.2f, bMMIFactor=%.2f, usesMBR=%s\n",
            amf, lmf, wp, bMMIfactor, sMBR ? "true" : "false");
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
            node->SetGammarCalculationParam(amf, lmf, wp, bMMIfactor, sMBR);
        }
    }
}

/*static*/ void ComputationNetwork::SetMaxTempMemSizeForCNN(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const size_t maxTempMemSizeInSamples)
{
    if (maxTempMemSizeInSamples > 0)
        fprintf(stderr, "Setting max temp memory size for Convolution operations to %lu samples.\n", (unsigned long)maxTempMemSizeInSamples);
    list<ComputationNodeBasePtr> convolutionNodes = net->GetNodesWithType(OperationNameOf(ConvolutionNode), criterionNode);
    if (convolutionNodes.size() == 0 && maxTempMemSizeInSamples != 0)
    {
        fprintf(stderr, "WARNING: No Convolution operation found.\n");
    }
    else
    {
        for (auto nodeIter = convolutionNodes.begin(); nodeIter != convolutionNodes.end(); nodeIter++)
        {
            auto nodef = dynamic_pointer_cast<ConvolutionNode<float>>(*nodeIter);
            if (nodef)
                nodef->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
            auto noded = dynamic_pointer_cast<ConvolutionNode<double>>(*nodeIter);
            if (noded)
                noded->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
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
        for (auto& node : FinalCriterionNodes())
        {
            //if (!allowFragment)
            //    FormRecurrentLoops(node);
            // this->SetActualMiniBatchSizeFromFeatures();
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
        }
    }
    else if (!allowFragment)
        RuntimeError("No Criterion nodes specified");
    // now output nodes
    if (OutputNodes().size() > 0)
    {
        for (auto& node : OutputNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
    }
    else if (!allowFragment)
        RuntimeError("No Output nodes specified");
    // now evaluation nodes
    if (EvaluationNodes().size() > 0)
    {
        for (auto& node : EvaluationNodes())
            if (!UnitTest(node))
                vErrors.push_back(node->NodeName().c_str());
    }
    return vErrors.empty();
}

bool ComputationNetwork::UnitTest(const ComputationNodeBasePtr& rootNode)
{
    fprintf(stderr, "\n\n Unit test node %ls \n", rootNode->NodeName().c_str());

    for (const auto& node : GetAllNodesForRoot(rootNode))
        if (!node->UnitTest())
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

    for (const auto& x : specialNodes)
        str = str + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
    return str + L"; \n";
}

void ComputationNetwork::DescribeNetworkUsingDot(list<ComputationArc>& arcs,
                                                 wstring outFile)
{
    DotGraphConfigure dotcfg;

    File fstream(outFile, FileOptions::fileOptionsText | FileOptions::fileOptionsWrite);

    vector<ComputationNodeBasePtr> preComputedNodes;
    vector<ComputationNodeBasePtr> pastValueNodes;
    vector<ComputationNodeBasePtr> futureValueNodes;
    vector<ComputationNodeBasePtr> learnableParameters;
    vector<ComputationNodeBasePtr> allnodes = GetAllNodes();
    for (const auto& n : allnodes)
    {
        if (n->RequiresPreCompute())
            preComputedNodes.push_back(n);

        const auto operationName = n->OperationName();
        if (operationName == OperationNameOf(PastValueNode) || operationName == L"Delay"/*legacy*/) 
            pastValueNodes.push_back(n);
        else if (operationName == OperationNameOf(FutureValueNode))
            futureValueNodes.push_back(n);
        else if (operationName == OperationNameOf(LearnableParameter))
            learnableParameters.push_back(n);
    }

    fstream << "strict digraph {\n";
    fstream << "rankdir = BT ;  \n";

    // ////////////////////////////////////////////////////////////////////////
    //    special nodes
    // ////////////////////////////////////////////////////////////////////////
    fstream << L"// special nodes \n";

    // learnable parameters:
    fstream << FormSpecialNodes(dotcfg.m_LearnableParameterStyle, learnableParameters);
    // features
    fstream << FormSpecialNodes(dotcfg.m_featuresStyle, m_featureNodes);
    // labels
    fstream << FormSpecialNodes(dotcfg.m_labelsStyle, m_labelNodes);
    // critera
    fstream << FormSpecialNodes(dotcfg.m_CriteriaStyle, m_criterionNodes);
    // pre-compute nodes
    fstream << FormSpecialNodes(dotcfg.m_PrecomputingNodeStyle, preComputedNodes);
    // PastValue nodes
    fstream << FormSpecialNodes(dotcfg.m_pastValueNodeStyle, pastValueNodes);
    // FutureValue nodes
    fstream << FormSpecialNodes(dotcfg.m_futureValueNodeStyle, futureValueNodes);
    // normal nodes
    fstream << dotcfg.m_normalNodeStyle << L"\n";

    // ////////////////////////////////////////////////////////////////////////
    //    add labels for each node
    // ////////////////////////////////////////////////////////////////////////
    fstream << L"\n// add labels and operation name\n";
    wstring line;
    for (const auto& x : allnodes)
    {
        line.clear();
        line = msra::strfun::wstrprintf(L" \"%ls\" [ label = \"%ls [%ls%ls]\\n%ls\" ] ;\n",
                                        x->GetName().c_str(), x->GetName().c_str(), wstring(x->GetSampleLayout()).c_str(), x->HasMBLayout() ? L" x *" : L"",
                                        x->OperationName().c_str());
        fstream << line;
    }

    // ////////////////////////////////////////////////////////////////////////
    //    sub-graph
    // ////////////////////////////////////////////////////////////////////////
    // subgraph source
    fstream << L"subgraph {\n";
    fstream << L"\t\t rank=source ; ";
    line.clear();
    for (const auto& x : m_featureNodes)
        line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
    fstream << line << L"\n}\n";

    // subgraph eval/output/criteria
    fstream << L"subgraph {\n";
    fstream << L"\t\t rank=sink ; ";
    line.clear();
    for (const auto& x : m_criterionNodes)
        line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
    for (const auto& x : m_outputNodes)
        line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());
    for (const auto& x : m_evaluationNodes)
        line = line + msra::strfun::wstrprintf(L"\"%ls\" ", x->GetName().c_str());

    fstream << line << L"\n}\n";

    // ////////////////////////////////////////////////////////////////////////
    //    specify arc connections
    // ////////////////////////////////////////////////////////////////////////
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

void ComputationNetwork::PlotNetworkTopology(const wstring& outputFile) 
{
    VerifyIsCompiled("PlotNetworkTopology");
    // ValidateNetwork(false, true);

    // ////////////////////////////////////////////////////////////////////////
    //    step 1.        get all the arcs in the network
    // ////////////////////////////////////////////////////////////////////////
    unordered_set<ComputationNodeBasePtr> visited;
    list<ComputationArc> arcs;

    for (auto groupIter : GetAllNodeGroups())
    {
        // note: this will also loop over m_featureNodes and m_labelNodes, which will do nothing since they have no inputs
        // TODO: test whether that is true
        const auto& group = *groupIter;
        for (size_t i = 0; i < group.size(); i++)
            group[i]->EnumerateArcs(visited, arcs);
    }

    // ////////////////////////////////////////////////////////////////////////
    //    step 2.        output dot description
    // ////////////////////////////////////////////////////////////////////////
    DescribeNetworkUsingDot(arcs, outputFile);
}

// enumerate all arcs that can be reached starting from the current node's children
// [in/out] visited record already visited nodes
void ComputationNodeBase::EnumerateArcs(std::unordered_set<ComputationNodeBasePtr>& visited, std::list<ComputationArc>& arcs)
{
    std::list<ComputationNodeBasePtr> tovisit;

    if (visited.find(shared_from_this()) == visited.end()) // only do when this node has not been visited before
    {
        tovisit.push_back(shared_from_this());

        while (!tovisit.empty())
        {
            ComputationNodeBasePtr curNode = tovisit.front();
            tovisit.pop_front();

            if (visited.find(curNode) == visited.end())
            {
                for (size_t i = 0; i < curNode->m_inputs.size(); i++)
                {
                    arcs.push_back(ComputationArc(curNode, curNode->m_inputs[i]));

                    if (visited.find(curNode->m_inputs[i]) == visited.end()) // this children has not been visited before
                        tovisit.push_front(curNode->m_inputs[i]);            // going to visit each of the children
                }
                visited.insert(curNode);
            }
        }
    }
}

// -----------------------------------------------------------------------
// specialized operations
// -----------------------------------------------------------------------

// TODO: Lift this into config language, move underlying code to math lib. This should be a model-editing operation.

// ========================================
// This function performs SVD decomposition for different groups of learnable  parameters
// we perform SVD decomposition such that
//  A \approx B*C, where rank(B)=rank(C)=r < rank(A)
// After SVD decomposition, the node A will become an intermediate node whose children are B,C ;
// B and C are two learnable parameters
// ========================================
// BUGBUG: this only currently works for one ElemType, not both
template <class ElemType>
void ComputationNetwork::PerformSVDecomposition(const map<wstring, float>& SVDConfig, size_t alignedSize)
{
    vector<pair<vector<wstring>, float>> nodeGroups;
    wregex nameFilter;

    for (const auto& e : SVDConfig)
    {
        wstring regexStr = e.first;
        if (regexStr.empty())
            continue;

        float keepRatio = e.second;
        vector<wstring> namesInGroup;

        nameFilter.assign(regexStr);

        for (auto n = m_nameToNodeMap.begin(); n != m_nameToNodeMap.end(); n++)
        {
            if (!regex_match(n->first, nameFilter))
            {
                // if regexStr is not empty and the the node does not match with the regexStr
                continue;
            }

            shared_ptr<ComputationNode<ElemType>> ptr = dynamic_pointer_cast<LearnableParameter<ElemType>>(n->second);
            if (!ptr)
                continue;

            if (ptr->Value().GetNumCols() == 1 || ptr->Value().GetNumRows() == 1)
                continue;

            // still here ?
            namesInGroup.push_back(n->first);
        }
        nodeGroups.push_back(make_pair(namesInGroup, keepRatio));
    }

    size_t groupID = 0;
    for (auto& group : nodeGroups)
    {
        float keepRatio = group.second;
        fprintf(stderr,
                "--------------------------------------------------------------------------------------------\n");
        fprintf(stderr,
                "ParameterSVD: start to process group %d with KeepRatio=%.2f\n",
                (int) groupID++, keepRatio);
        fprintf(stderr,
                "--------------------------------------------------------------------------------------------\n");

        for (const auto& name : group.first)
        {
            if (m_nameToNodeMap.find(name) == m_nameToNodeMap.end())
            {
                // could be deleted in the previous groups
                continue;
            }

            shared_ptr<ComputationNode<ElemType>> pNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(m_nameToNodeMap[name]);

            // Step 1. do SVD decomposition
            Matrix<ElemType> A = pNode->ValueAsMatrix().DeepClone();

            // it is a vector, no need to do it
            if (A.GetNumCols() == 1 || A.GetNumRows() == 1)
                continue;

            size_t m = A.GetNumRows();
            size_t n = A.GetNumCols();

            Matrix<ElemType> S(-1), U(-1), VT(-1), W(-1);
            chrono::time_point<chrono::system_clock> stTime = chrono::system_clock::now();
            Matrix<ElemType>::SVD(A, S, U, VT, W);
            chrono::time_point<chrono::system_clock> enTime = chrono::system_clock::now();

            // A \in R^{mXn}
            // U \in R^{mXm}
            // VT \in R^{nXn}
            // S \in R^{min(m,n),1}
            // S is in descending order

            ElemType totalEnergy = 0.0f;
            for (size_t i = 0; i < S.GetNumRows(); i++)
                totalEnergy += S(i, 0);
            ElemType keepEnergy = totalEnergy * keepRatio;
            ElemType runEnergy = 0.0f;

            size_t r = 0;
            for (size_t indx = 0; indx < S.GetNumRows(); indx++)
            {
                runEnergy += S(indx, 0);
                if (runEnergy > keepEnergy)
                {
                    r = indx + 1;
                    break;
                }
            }

            r = r > S.GetNumRows() ? S.GetNumRows() : r;

            if (r % alignedSize != 0)
            {
                r -= r % alignedSize;
                r = r + alignedSize > S.GetNumRows() ? S.GetNumRows() : r + alignedSize;
            }
            // r = (r + 7) & (~7); //  to keep the number of rows/cols of resultant matrix a multipier of 8
            //  which can be helpful at runtime

            chrono::duration<double> elapsedtime = enTime - stTime;
            fprintf(stderr,
                    "Performing SVD for a %5d-by-%-5d matrix (node name: %-20ls) ---  computation time %5.2f secs ;  keep %4.1f%% energy ===> keep %5d svd values (reduce to %4.1f%% parameters) \n",
                    (int) m, (int) n, name.c_str(), elapsedtime.count(),
                    keepRatio * 100, (int) r,
                    ((m + n) * r + 0.0f) / m / n * 100);

            // redU in R^ {mXr}
            Matrix<ElemType> redU = U.ColumnSlice(0, r);
            Matrix<ElemType> redVT(-1);

            // redVT in R^{rXn}
            redVT.Resize(r, n);
            redVT.AssignRowSliceValuesOf(VT, 0, r);

            Matrix<ElemType> redS(r, (size_t)1, A.GetDeviceId());
            for (size_t i = 0; i < r; i++)
            {
                ElemType sqrtSigma = (ElemType) sqrt((double) S(i, 0));
                redS(i, 0) = sqrtSigma;
            }

            redU.RowElementMultiplyWith(redS.Transpose());
            redVT.ColumnElementMultiplyWith(redS);

            // Step 2. create two new Parameter nodes and one Times node
            wstring leftChildName = name + L"_U";
            wstring rightChildName = name + L"_V";
            shared_ptr<ComputationNode<ElemType>> pLeft = AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(m_deviceId, leftChildName, m, r));
            shared_ptr<ComputationNode<ElemType>> pRight = AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(m_deviceId, rightChildName, r, n));
            InitLearnableParameters(pLeft,  L"fixedValue", 0); // follow the protocol; otherwise deferred initialization will overwrite the SVD values in validation
            InitLearnableParameters(pRight, L"fixedValue", 0);

            // TODO: We should be able to move instead of copy but it currently isn't straightforward
            // due to redU and redVT being slices
            pLeft->ValueAsMatrix()  = redU.DeepClone();
            pRight->ValueAsMatrix() = redVT.DeepClone();

            // Step 3. Change the network hierachy to include the SVD nodes
            auto parentNodes = GetParentNodes(name);

            for (auto& pParentNode : parentNodes)
            {
                // Change the hierarchy of the network if the node is immediately used in a product
                auto pParentTimesNode = dynamic_pointer_cast<TimesNode<ElemType>>(pParentNode);
                if (pParentTimesNode)
                {
                    // Change the hierarchy to ensure multiplication order
                    // U*(V*X)
                    shared_ptr<ComputationNode<ElemType>> pTimes = New<TimesNode<ElemType>>(m_deviceId, name + L"_SVD");
                    pTimes->AttachInputs({ pLeft, pParentNode });
                    
                    InsertNode(pParentNode->GetName(), pTimes, pParentNode->GetTags());
                    ReplaceLeafNode(name, pRight);
                }
                else
                {
                    // Default multiplication order
                    shared_ptr<ComputationNode<ElemType>> pTimes = AddNodeToNetAndAttachInputs(New<TimesNode<ElemType>>(m_deviceId, name + L"_SVD"), { pLeft, pRight });

                    ReplaceLeafNode(name, pTimes);
                }
            }
        }
    }

    // redo necessary post-processing
    CompileNetwork();
}

// Helper class to form a logical DBN layer while exporting the network (used by SaveToDbnFile)
class DbnLayer
{
public:
    DbnLayer() : Node(nullptr), Bias(nullptr), Sigmoided(false) {}
    ComputationNodeBasePtr Node;
    ComputationNodeBasePtr Bias;
    bool Sigmoided;
    ~DbnLayer() {};
};

// Save network in the format of the Microsoft-internal legacy "DBN.exe" tool (this function is not useful outside of Microsoft).
template <class ElemType>
void ComputationNetwork::SaveToDbnFile(ComputationNetworkPtr net, const std::wstring& fileName) const 
{
    // Helper methods
    auto VerifyTypeAll = [](const std::vector<ComputationNodeBasePtr>& nodes, const std::wstring& typeValue) -> bool
    {
        return std::find_if(nodes.begin(), nodes.end(), [&typeValue](ComputationNodeBasePtr node)->bool { return node->OperationName() != typeValue; }) == nodes.end();
    };
    auto GetNodeConsumers = [&net](const ComputationNodeBasePtr node) -> std::vector<ComputationNodeBasePtr>
    {
        std::vector<ComputationNodeBasePtr> consumers;
        for (auto& item : net->GetAllNodes())
        {
            for (auto& input : item->GetInputs())
            {
                if (input == node)
                {
                    consumers.push_back(item);
                    break;
                }
            }
        }

        return consumers;
    };
    auto GetFirstDifferentNode = [](const std::vector<ComputationNodeBasePtr>& list, const ComputationNodeBasePtr node) -> ComputationNodeBasePtr
    {
        auto foundNode = std::find_if(list.begin(), list.end(), [&node](ComputationNodeBasePtr item)->bool { return item != node; });
        return foundNode == list.end() ? nullptr : *foundNode;
    };
    auto GetFirstNodeWithDifferentType = [](const std::vector<ComputationNodeBasePtr>& list, const std::wstring& type) -> ComputationNodeBasePtr
    {
        auto foundNode = std::find_if(list.begin(), list.end(), [&type](ComputationNodeBasePtr item)->bool { return item->OperationName() != type; });
        return foundNode == list.end() ? nullptr : *foundNode;
    };
    auto WhereNode = [](const std::vector<ComputationNodeBasePtr>& nodes, const function<bool(ComputationNodeBasePtr)>& predicate) -> std::vector<ComputationNodeBasePtr>
    {
        std::vector<ComputationNodeBasePtr> results;

        for (auto& node : nodes)
        {
            if (predicate(node))
            {
                results.push_back(node);
            }
        }

        return results;
    };
    auto GetNodesWithType = [](const std::vector<ComputationNodeBasePtr>& list, const std::wstring& type) -> std::vector<ComputationNodeBasePtr>
    {
        std::vector<ComputationNodeBasePtr> results;

        for (auto& node : list)
        {
            if (node->OperationName() == type)
            {
                results.push_back(node);
            }
        }

        return results;
    };
    auto GetAllPriorNodes = [](ComputationNodeBasePtr node)->bool
    {
        std::wstring lowerName = node->GetName();
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

        return node->OperationName() == OperationNameOf(LearnableParameter) && (lowerName.find(L"prior") != wstring::npos);
    };
    auto FindReplicationContext = [](std::vector<ElemType>& arr)->int
    {
        for (int i = 25; i >= 1; i--)
        {
            int ctx = i * 2 + 1;
            if (arr.size() % ctx != 0)
                continue;

            int baseLen = arr.size() / ctx;
            bool matched = true;

            for (int k = 1; k < ctx && matched; k++)
            {
                for (int j = 0; j < baseLen; j++)
                {
                    if (arr[j] != arr[k * baseLen + j])
                    {
                        matched = false;
                        break;
                    }
                }
            }

            if (matched)
                return ctx;
        }

        return 1;
    };

    // Get output node
    std::list<ComputationNodeBasePtr> outputNodes = net->GetNodesWithType(OperationNameOf(ClassificationErrorNode));
    ComputationNodeBasePtr outputNode = GetFirstNodeWithDifferentType(outputNodes.front()->GetInputs(), OperationNameOf(InputValue));

    if (outputNode == nullptr)
    {
        RuntimeError("Cannot find output node");
    }

    std::list<ComputationNodeBasePtr> orderList;
    std::stack<ComputationNodeBasePtr> nodeStack;

    nodeStack.push(outputNode);

    while (nodeStack.size() > 0)
    {
        auto node = nodeStack.top();
        nodeStack.pop();
        auto nodeInputs = node->GetInputs();
        for (auto& input : nodeInputs)
        {
            bool cyclic = false;
            for (auto& item : orderList)
            {
                if (item == input)
                {
                    Warning("Cyclic dependency on node '%ls'\n", item->GetName().c_str());
                    cyclic = true;
                }
            }

            if (!cyclic)
                nodeStack.push(input);
        }
        orderList.push_back(node);
    }

    orderList.reverse();

    // All multiplication nodes that multiply a symbolic variable
    std::list<ComputationNodeBasePtr> multNodes;
    typedef shared_ptr<DbnLayer> DbnLayerPtr;
    std::list<DbnLayerPtr> dbnLayers;

    for (auto& item : orderList)
    {
        if (item->OperationName() == OperationNameOf(TimesNode) && !VerifyTypeAll(item->GetInputs(), OperationNameOf(LearnableParameter)))
        {
            multNodes.push_back(item);
        }
    }

    for (auto& node : multNodes)
    {
        std::vector<ComputationNodeBasePtr> consumers = GetNodeConsumers(node);
        if (consumers.size() == 1)
        {
            bool sigmoided = false;
            std::wstring layerId(node->GetName());

            ComputationNodeBasePtr firstConsumer = consumers.front();

            if (firstConsumer->OperationName() != OperationNameOf(PlusNode))
            {
                RuntimeError("Expected a plus node to consume the times node.");
            }

            ComputationNodeBasePtr bias = GetFirstDifferentNode(firstConsumer->GetInputs(), node);

            auto consumer2 = GetNodeConsumers(consumers.front()).front();
            if (consumer2->OperationName() == L"Sigmoid")
            {
                sigmoided = true;
                layerId = consumer2->GetName();
            }
            else
            {
                layerId = firstConsumer->GetName();
            }

            // If one of its inputs was itself a multiplication node, then split it out
            // into dbn-style.  
            std::vector<ComputationNodeBasePtr> aggTimes = GetNodesWithType(node->GetInputs(), OperationNameOf(TimesNode));
            if (aggTimes.size() > 0)
            {
                ComputationNodeBasePtr multNode = aggTimes.front();
                DbnLayerPtr l1 = make_shared<DbnLayer>();
                DbnLayerPtr l2 = make_shared<DbnLayer>();

                auto firstInput = multNode->GetInputs()[0];
                auto secondInput = multNode->GetInputs()[1];
                l2->Bias = bias;
                l2->Node = firstInput;

                l1->Bias = nullptr;
                l1->Node = secondInput;

                l1->Sigmoided = false;
                l2->Sigmoided = sigmoided;

                dbnLayers.push_back(l1);
                dbnLayers.push_back(l2);
            }
            else
            {
                auto paramNode = GetNodesWithType(node->GetInputs(), OperationNameOf(LearnableParameter)).front();
                DbnLayerPtr l1 = make_shared<DbnLayer>();
                l1->Bias = bias;
                l1->Node = paramNode;
                l1->Sigmoided = sigmoided;

                dbnLayers.push_back(l1);
            }
        }
    }

    // Write the layers to the output 
    // DBN wants std not invstd, so need to invert each element
    std::vector<ComputationNodeBasePtr> normalizationNodes = GetNodesWithType(net->GetAllNodes(), OperationNameOf(PerDimMeanVarNormalizationNode));
    if (normalizationNodes.size() == 0)
    {
        RuntimeError("Model does not contain at least one node with the '%ls' operation.", OperationNameOf(PerDimMeanVarNormalizationNode).c_str());
    }

    ComputationNodeBasePtr meanNode = normalizationNodes.front()->GetInputs()[1];
    ComputationNodeBasePtr stdNode = normalizationNodes.front()->GetInputs()[2];
    
    Matrix<ElemType> meanNodeMatrix = meanNode->As<ComputationNode<ElemType>>()->Value().DeepClone();
    Matrix<ElemType> invStdNodeMatrix(std::move(stdNode->As<ComputationNode<ElemType>>()->Value().DeepClone().ElementInverse()));
    std::vector<ElemType> arr(invStdNodeMatrix.GetNumElements());
    ElemType* refArr = &arr[0];
    size_t arrSize = arr.size();
    invStdNodeMatrix.CopyToArray(refArr, arrSize);

    int ctx = FindReplicationContext(arr);
    std::vector<ComputationNodeBasePtr> priorNodes = WhereNode(net->GetAllNodes(), GetAllPriorNodes);
    if (priorNodes.size() != 1)
    {
        Warning("Could not reliably determine the prior node!");
    }

    // =================
    // Write to the file
    // =================
    File fstream(fileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsWrite);

    // local helper functions for writing stuff in DBN.exe-expected format
    auto PutTag = [&fstream](const char * tag) { while (*tag) fstream << *tag++; };
    auto PutString = [&fstream](const char * string) { fstream.WriteString(string, 0); };
    auto PutInt = [&fstream](int val) { fstream << val; };

    // write a DBN matrix object, optionally applying a function
    auto PutMatrixConverted = [&](const Matrix<ElemType> * m, size_t maxelem, const char * name, float(*f)(float))
    {
        PutTag("BMAT");
        PutString(name);
        size_t numRows = m->GetNumRows();
        size_t numCols = m->GetNumCols();

        if (maxelem == SIZE_MAX)
        {
            PutInt(numRows);
            PutInt(numCols);
        }
        else    // this allows to shorten a vector, as we need for mean/invstd
        {
            PutInt(maxelem);
            PutInt(1);
        }

        // this code transposes the matrix on the fly, and outputs at most maxelem floating point numbers to the stream
        size_t k = 0;
        for (size_t j = 0; j < numCols && k < maxelem; j++)
            for (size_t i = 0; i < numRows && k < maxelem; i++, k++)
                fstream << f((float)(*m)(i, j));

        PutTag("EMAT");
    };
    auto PutMatrix = [&](const Matrix<ElemType> * m, const char * name) { PutMatrixConverted(m, SIZE_MAX, name, [](float v) { return v; }); };

    // write out the data
    // Dump DBN header
    PutString("DBN");
    PutTag("BDBN");
    PutInt(0);                                                                              // a version number
    PutInt(static_cast<int>(dbnLayers.size()));                                             // number of layers

    // Dump feature norm
    PutMatrixConverted(&meanNodeMatrix, meanNodeMatrix.GetNumRows() / ctx, "gmean", [](float v) { return v; });
    PutMatrixConverted(&invStdNodeMatrix, invStdNodeMatrix.GetNumRows() / ctx, "gstddev", [](float v) { return v; });

    PutTag("BNET");
    auto lastOne = dbnLayers.end();
    --lastOne;
    for (auto ii = dbnLayers.begin(), e = dbnLayers.end(); ii != e; ++ii)
    {
        DbnLayerPtr& layer = *ii;

        if (ii == dbnLayers.begin())
        {
            PutString("rbmgaussbernoulli");
        }
        else if (ii == lastOne)
        {
            PutString("perceptron");
        }
        else if (layer->Sigmoided)
        {
            PutString("rbmbernoullibernoulli");
        }
        else
        {
            PutString("rbmisalinearbernoulli");
        }

        // Write out the main weight matrix
        auto weight = (layer->Node->As<ComputationNode<ElemType>>()->Value().DeepClone());
        auto transpose = weight.Transpose();
        PutMatrix(&transpose, "W");

        // Write out biasing vector
        // Is mandatory, so pack with zeroes if not given
        auto rows = layer->Node->GetAsMatrixNumRows();
        if (layer->Bias == nullptr)
        {
            auto zeros = Matrix<ElemType>::Zeros(rows, 1, CPUDEVICE);
            PutMatrixConverted(&zeros, rows, "a", [](float v) { return v; });
        }
        else
        {
            PutMatrixConverted(&(layer->Bias->As<ComputationNode<ElemType>>()->Value()), rows, "a", [](float v) { return v; });
        }

        // Some sort of legacy vector that is useless
        auto zeros = Matrix<ElemType>::Zeros(0, 0, CPUDEVICE);
        PutMatrix(&(zeros), "b");
    }

    // Dump the priors
    PutTag("ENET");
    if (priorNodes.size() > 0)
    {
        PutMatrix(&(priorNodes.front()->As<ComputationNode<ElemType>>()->Value()), "Pu");
    }
    else
    {
        Warning("No priority node(s) found!");
    }
    PutTag("EDBN");
}

template void ComputationNetwork::InitLearnableParametersWithBilinearFill<float>(const ComputationNodeBasePtr& node, size_t kernelWidth, size_t kernelHeight);
template void ComputationNetwork::Read<float>(const wstring& fileName);
template void ComputationNetwork::ReadPersistableParameters<float>(size_t modelVersion, File& fstream, bool create);
template void ComputationNetwork::PerformSVDecomposition<float>(const map<wstring, float>& SVDConfig, size_t alignedsize);
template /*static*/ void ComputationNetwork::SetDropoutRate<float>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double& prevDropoutRate);
template /*static*/ void ComputationNetwork::SetIRngUserSeed<float>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, size_t randSeedBase);
template /*static*/ void ComputationNetwork::SetBatchNormalizationTimeConstants<float>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double normalizationTimeConstant, double& prevNormalizationTimeConstant, double blendTimeConstant, double& prevBlendTimeConstant);
template void ComputationNetwork::SetSeqParam<float>(ComputationNetworkPtr net, const ComputationNodeBasePtr criterionNode, const double& hsmoothingWeight, const double& frameDropThresh, const bool& doreferencealign,
                                                     const double& amf, const double& lmf, const double& wp, const double& bMMIfactor, const bool& sMBR);
template void ComputationNetwork::SaveToDbnFile<float>(ComputationNetworkPtr net, const std::wstring& fileName) const;

template void ComputationNetwork::InitLearnableParametersWithBilinearFill<double>(const ComputationNodeBasePtr& node, size_t kernelWidth, size_t kernelHeight);
template void ComputationNetwork::Read<double>(const wstring& fileName);
template void ComputationNetwork::ReadPersistableParameters<double>(size_t modelVersion, File& fstream, bool create);
template void ComputationNetwork::PerformSVDecomposition<double>(const map<wstring, float>& SVDConfig, size_t alignedsize);
template /*static*/ void ComputationNetwork::SetDropoutRate<double>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double dropoutRate, double& prevDropoutRate);
template /*static*/ void ComputationNetwork::SetIRngUserSeed<double>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, size_t randSeedBase);
template /*static*/ void ComputationNetwork::SetBatchNormalizationTimeConstants<double>(ComputationNetworkPtr net, const ComputationNodeBasePtr& criterionNode, const double normalizationTimeConstant, double& prevNormalizationTimeConstant, double blendTimeConstant, double& prevBlendTimeConstant);
template void ComputationNetwork::SetSeqParam<double>(ComputationNetworkPtr net, const ComputationNodeBasePtr criterionNode, const double& hsmoothingWeight, const double& frameDropThresh, const bool& doreferencealign,
                                                      const double& amf, const double& lmf, const double& wp, const double& bMMIfactor, const bool& sMBR);
template void ComputationNetwork::SaveToDbnFile<double>(ComputationNetworkPtr net, const std::wstring& fileName) const;

// register ComputationNetwork with the ScriptableObject system
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNetwork> registerComputationNetwork(L"ComputationNetwork");

}}}
