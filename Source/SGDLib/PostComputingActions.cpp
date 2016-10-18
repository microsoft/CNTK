//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// PostStat.cpp -- CNTK post statistics related actions
//

#include "PostComputingActions.h"

#include "TrainingNodes.h"
#include "ProgressTracing.h"
#include "DataReaderHelpers.h"
#include "SimpleDistGradAggregator.h"

#include <vector>

namespace Microsoft { namespace MSR{ namespace CNTK {

template <class ElemType>
void PostComputingActions<ElemType>::BatchNormalizationStatistics(IDataReader * dataReader, const vector<wstring>& evalNodeNames, 
    const wstring newModelPath, const size_t mbSize, const int iters)
{
    // since the mean and variance of bn will be modified in statistics,
    // training mode will make it work. And there is no back prop, other parameters
    // are fixed during computing.
    ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::training);

    // bn nodes need to be computed from bottom to top with evaluating order
    let evalNodes = m_net->GetEvalNodesWithName(evalNodeNames);

    // find all the  BN nodes by evalOrder
    std::vector<ComputationNodeBasePtr> bnNodes;
    std::set<ComputationNodeBasePtr> bnNodesLogged; // (avoid double record of batch normalization nodes)
    for (auto& evalNode : evalNodes)
    {
        for (auto& node : m_net->GetEvalOrder(evalNode))
        {
            let bnNode = dynamic_pointer_cast<BatchNormalizationNode<ElemType>>(node);
            if (bnNode)
            {
                if (bnNodesLogged.insert(node).second)
                {
                    // reset the statistics states of bn nodes
                    bnNode->ResetStatisticsState();
                    bnNode->SetNormalizationTimeConstants(-1, bnNode->NormalizationTimeConstant(),
                        0, bnNode->BlendTimeConstant());
                    bnNodes.push_back(node);
                    // add BN nodes into the evaluation group, then they will be added into root nodes when
                    // the network re-compile
                    m_net->AddToNodeGroup(L"evaluation", bnNode);
                }
            }
        }
    }

    // re-compile the network to add bn nodes as rootNodes. 
    m_net->CompileNetwork();

    // allocate memory for all bnNodes evalOrder
    m_net->AllocateAllMatrices(bnNodes, std::vector<ComputationNodeBasePtr>(), nullptr);

    // prepare features
    auto& featureNodes = m_net->FeatureNodes();

    StreamMinibatchInputs inputMatrices;
    for (auto& node : featureNodes)
        inputMatrices.AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());

    bool useParallelTrain = (m_mpi != nullptr);
    bool useDistributedMBReading = useParallelTrain && m_enableDistributedMBReading && dataReader->SupportsDistributedMBRead();
    size_t totalEpochSize = bnNodes.size() * mbSize * iters;

    m_net->StartEvaluateMinibatchLoop(bnNodes);

    if (useDistributedMBReading)
        dataReader->StartDistributedMinibatchLoop(mbSize, 0, m_mpi->CurrentNodeRank(), m_mpi->NumNodesInUse(), inputMatrices.GetStreamDescriptions(), totalEpochSize);
    else
        dataReader->StartMinibatchLoop(mbSize, 0, inputMatrices.GetStreamDescriptions(), totalEpochSize);

    for (auto& node : bnNodes)
    {
        let bnNode = static_pointer_cast<BatchNormalizationNode<ElemType>>(node);
        size_t actualMBSize = 0;

        LOGPRINTF(stderr, "Estimating Statistics --> %ls\n", bnNode->GetName().c_str());


        // for every single bn node, the statistics is the average of mean and variance for several times in forward prop
        // the forward prop is from the feature to the current bn node
        for (int iter = 0; iter < iters; iter++)
        {
            // during the bn stat, dataRead must be ensured
            bool wasDataRead = DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(*dataReader, m_net,
                nullptr, useDistributedMBReading, useParallelTrain, inputMatrices, actualMBSize, m_mpi);

            if (!wasDataRead) LogicError("DataRead Failure in batch normalization statistics");

            ComputationNetwork::BumpEvalTimeStamp(featureNodes);

            // forward prop till reaching the current bn node
            m_net->ForwardProp(node);
        }

        // after finished statistics, the mean and variance of the bn node should be freezd.
        bnNode->FreezeParameters();

        // Sync during or after all iters of a BN node are equivalent
        if (useParallelTrain)
        {
            if (m_gradHeader == nullptr)
            {
                m_gradHeader.reset(DistGradHeader::Create(evalNodes.size()), [](DistGradHeader* ptr)
                {
                    DistGradHeader::Destroy(ptr);
                });
            }

            // push the statistics results of mean and variance of bn nodes into mpi updating vector
            std::vector<Matrix<ElemType>*> learnParamsValues(2, nullptr);

            SimpleDistGradAggregator<ElemType> distGradAgg(m_mpi, false /*useAsyncAggregation*/, 0 /*syncStatsTrace*/);

            auto runMeanParameterPtr = node->Input(3);
            auto runStdParameterPtr  = node->Input(4);

            shared_ptr<ComputationNode<ElemType>> runMeanNode = static_pointer_cast<ComputationNode<ElemType>>(runMeanParameterPtr);
            shared_ptr<ComputationNode<ElemType>> runStdNode  = static_pointer_cast<ComputationNode<ElemType>>(runStdParameterPtr);

            learnParamsValues[0] = &(runMeanNode->Value());
            learnParamsValues[1] = &(runStdNode->Value());

            m_gradHeader->numSamples = actualMBSize ? 1 : actualMBSize;
            distGradAgg.AggregateGradients(learnParamsValues, m_gradHeader.get(), 0);

            // get the average mean and variance across all the workers
            for (auto& parameter : learnParamsValues)
            {
                (*parameter) /= (ElemType)m_mpi->NumNodesInUse();
            }
        }
    }

    dataReader->DataEnd();

    // remove all the added BN nodes from evaluation group
    for (auto& bnNode : bnNodes)
    {
        m_net->RemoveFromNodeGroup(L"evaluation", bnNode);
    }

    // save model
    if (!useParallelTrain || m_mpi->CurrentNodeRank() == m_mpi->MainNodeRank())
        m_net->Save(newModelPath);

    return;
}

template class PostComputingActions<float>;
template class PostComputingActions<double>;

}}}
