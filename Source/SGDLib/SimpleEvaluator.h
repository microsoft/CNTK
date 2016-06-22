//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "DataReaderHelpers.h"
#include "TrainingNodes.h" // TODO: we should move the functions that depend on these to the .cpp
#include "ProgressTracing.h"
#include "DistGradHeader.h"
#include "IDistGradAggregator.h"
#include "SimpleDistGradAggregator.h"
#include "Criterion.h"

#include <vector>
#include <string>
#include <set>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class IDistGradAggregator;

// TODO: get rid of dependency on ElemType
template <class ElemType>
class SimpleEvaluator
{
public:
    SimpleEvaluator(ComputationNetworkPtr net, const MPIWrapperPtr& mpi, bool enableDistributedMBReading = false, const size_t numMBsToShowResult = 100, const size_t firstMBsToShowResult = 0, const int traceLevel = 0, const size_t maxSamplesInRAM = SIZE_MAX,
                    const size_t numSubminiBatches = 1) :
        m_net(net), 
        m_numMBsToShowResult(numMBsToShowResult), 
        m_firstMBsToShowResult(firstMBsToShowResult),
        m_traceLevel(traceLevel),
        m_maxSamplesInRAM(maxSamplesInRAM), 
        m_numSubminiBatches(numSubminiBatches), 
        m_mpi(mpi), 
        m_distGradAgg(nullptr),
        m_gradHeader(nullptr),
        m_enableDistributedMBReading(enableDistributedMBReading)
    {
    }

    // returns evaluation node values per sample determined by evalNodeNames (which can include both training and eval criterion nodes)
    vector<EpochCriterion> Evaluate(IDataReader* dataReader, const vector<wstring>& evalNodeNames, const size_t mbSize, const size_t testSize = requestDataSize)
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::inferring);

        // determine nodes to evaluate
        std::vector<ComputationNodeBasePtr> evalNodes;

        set<ComputationNodeBasePtr> criteriaLogged; // (keeps track ot duplicates to avoid we don't double-log critera)
        if (evalNodeNames.size() == 0)
        {
            fprintf(stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
            if (m_net->EvaluationNodes().empty() && m_net->FinalCriterionNodes().empty())
                InvalidArgument("There is no default evaluation node or training criterion specified in the network.");

            for (const auto& node : m_net->EvaluationNodes())
                if (criteriaLogged.insert(node).second)
                    evalNodes.push_back(node);

            for (const auto& node : m_net->FinalCriterionNodes())
                if (criteriaLogged.insert(node).second)
                    evalNodes.push_back(node);
        }
        else
        {
            for (int i = 0; i < evalNodeNames.size(); i++)
            {
                const auto& node = m_net->GetNodeFromName(evalNodeNames[i]);
                if (!criteriaLogged.insert(node).second)
                    continue;
                if (node->GetSampleLayout().GetNumElements() != 1)
                    InvalidArgument("Criterion nodes to evaluate must have dimension 1x1.");
                evalNodes.push_back(node);
            }
        }

        // initialize eval results
        std::vector<EpochCriterion> evalResults(evalNodes.size(), EpochCriterion(0));

        // allocate memory for forward computation
        m_net->AllocateAllMatrices(evalNodes, {}, nullptr);

        // prepare features and labels
        auto& featureNodes = m_net->FeatureNodes();
        auto& labelNodes = m_net->LabelNodes();

        StreamMinibatchInputs inputMatrices;
        for (auto& node : featureNodes)
            inputMatrices.AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());
        for (auto& node : labelNodes)
            inputMatrices.AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());

        // evaluate through minibatches
        size_t totalEpochSamples = 0;
        size_t numMBsRun = 0;
        size_t numSamplesLastLogged = 0;
        size_t numMBsRunLastLogged = 0; // MBs run before this display

        std::vector<EpochCriterion> evalResultsLastLogged(evalResults.size(), EpochCriterion(0));

        bool useParallelTrain = (m_mpi != nullptr);
        bool useDistributedMBReading = useParallelTrain && m_enableDistributedMBReading && dataReader->SupportsDistributedMBRead();
        if (useDistributedMBReading)
            dataReader->StartDistributedMinibatchLoop(mbSize, 0, m_mpi->CurrentNodeRank(), m_mpi->NumNodesInUse(), testSize);
        else
        dataReader->StartMinibatchLoop(mbSize, 0, testSize);

        m_net->StartEvaluateMinibatchLoop(evalNodes);

        std::vector<Matrix<ElemType>*> learnParamsGradients;
        DataReaderHelpers::SubminibatchDispatcher<ElemType> smbDispatcher;
        size_t numSubminibatchesNeeded = DataReaderHelpers::GetNumSubminibatchesNeeded<ElemType>(dataReader, m_maxSamplesInRAM, m_numSubminiBatches, mbSize);

        // Passing in two empty node lists so the dispatcher can work for the evalNodes.
        std::list<ComputationNodeBasePtr> learnableNodes;
        std::vector<ComputationNodeBasePtr> criterionNodes;
        if (numSubminibatchesNeeded > 1)
            smbDispatcher.Init(m_net, learnableNodes, criterionNodes, evalNodes);

        CriterionAccumulator<ElemType> localEpochEvalErrors(evalNodes.size(), m_net->GetDeviceId());

        const size_t numIterationsBeforePrintingProgress = 100;
        size_t numItersSinceLastPrintOfProgress = 0;
        bool noMoreSamplesToProcess = false;
        for (;;)
        {
            size_t actualMBSize = 0;
            bool wasDataRead = DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(*dataReader, m_net, nullptr, useDistributedMBReading, useParallelTrain, inputMatrices, actualMBSize, m_mpi);
            // in case of distributed reading, we do a few more loops until all ranks have completed
            // end of epoch
            if (!wasDataRead && (!useDistributedMBReading || noMoreSamplesToProcess)) 
                break;

            // Note: If !wasDataRead then the data that GetMinibatchIntoNetwork() was supposed to full in are undefined.
            // Must not touch them.
            if (!wasDataRead)
                actualMBSize = 0; // (undefined if !wasDataRead)

            if (actualMBSize > 0)
        {

            size_t actualNumSubminibatches = numSubminibatchesNeeded <= 1 ? 1 : smbDispatcher.GetMinibatchIntoCache(*dataReader, *m_net, inputMatrices, numSubminibatchesNeeded);
            for (size_t ismb = 0; ismb < actualNumSubminibatches; ismb++)
            {
                if (actualNumSubminibatches > 1)
                {
                    smbDispatcher.GetSubMinibatchToNet(ismb); // get sub-minibatch from full-size one
                }

                ComputationNetwork::BumpEvalTimeStamp(featureNodes);
                ComputationNetwork::BumpEvalTimeStamp(labelNodes);

                m_net->ForwardProp(evalNodes);

                // house-keeping for sub-minibatching
                if (actualNumSubminibatches > 1)
                    smbDispatcher.DoneWithCurrentSubMinibatch(ismb); // page state out
            } // end sub-minibatch loop

            if (actualNumSubminibatches > 1)
                smbDispatcher.DoneWithCurrentMinibatch();
            } // if (actualMBSize > 0)

            // BUGBUG (Issue #95): Once we have multiple layouts, this must be done on a per-node basis.
            size_t numSamplesWithLabel = wasDataRead ? m_net->GetNumSamplesWithLabelOfNetwork(actualMBSize) : 0;
            size_t aggregateNumSamplesWithLabel = numSamplesWithLabel;
            if (useParallelTrain)
            {
                if (m_gradHeader == nullptr)
                {
                    m_gradHeader.reset(DistGradHeader::Create(evalNodes.size()), [](DistGradHeader* ptr) {
                        DistGradHeader::Destroy(ptr);
                    });
                    m_distGradAgg = make_shared<SimpleDistGradAggregator<ElemType>>(m_mpi, false, m_traceLevel);
                }

                m_gradHeader->numEvalNode = evalNodes.size();
                m_gradHeader->numSamples = actualMBSize;
                m_gradHeader->numSamplesWithLabel = numSamplesWithLabel;
                m_gradHeader->criterion = 0.0; // (not used here)
                for (size_t i = 0; i < evalNodes.size(); i++)
                    m_gradHeader->evalErrors[i] = localEpochEvalErrors.Assign(evalNodes, i, numSamplesWithLabel).GetCriterion(i);

                // TODO: We are reusing the aggregation logic inside SimpleDistGradAggregator, which has a heavy dependency
                // on the gradient matrix. At some point we should refactor the aggregator class to be able to only calculating
                // eval results and then remove this hack.
                if (learnParamsGradients.size() == 0)
                {
                    Matrix<ElemType>* matrix = new Matrix<ElemType>((DEVICEID_TYPE)m_net->GetDeviceId());
                    learnParamsGradients.push_back(matrix);
                }

                // Using SimpleDistAggregator for eval results only. At some point we should rename the class to be just
                // IDistAggregator and SimpleDistAggregator.
                bool samplesProcessed = m_distGradAgg->AggregateGradients(learnParamsGradients, m_gradHeader.get(), 0);
                noMoreSamplesToProcess = !samplesProcessed;

                aggregateNumSamplesWithLabel = m_gradHeader->numSamplesWithLabel;
                for (size_t i = 0; i < evalResults.size(); i++)
                    evalResults[i] += m_gradHeader->evalErrors[i];
            }
            else
            {
                if (actualMBSize != 0)
                {
                for (int i = 0; i < evalNodes.size(); i++)
                    evalResults[i] += localEpochEvalErrors.Assign(evalNodes, i, numSamplesWithLabel).GetCriterion(i);
                }
            }

            totalEpochSamples += aggregateNumSamplesWithLabel;
            numMBsRun++;

            if (m_traceLevel > 0)
            {
                numSamplesLastLogged += aggregateNumSamplesWithLabel;

                if (numMBsRun <= m_firstMBsToShowResult || (m_numMBsToShowResult && (numMBsRun % m_numMBsToShowResult == 0)))
                {
                    DisplayEvalStatistics(numMBsRunLastLogged + 1, numMBsRun, numSamplesLastLogged, evalNodes, evalResults, evalResultsLastLogged);

                    for (int i = 0; i < evalResults.size(); i++)
                        evalResultsLastLogged[i] = evalResults[i];
                    numSamplesLastLogged = 0;
                    numMBsRunLastLogged = numMBsRun;
                }
            }

            numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);

            // call DataEnd to check if end of sentence is reached
            // datareader will do its necessary/specific process for sentence ending
            dataReader->DataEnd();
        }

        // show last batch of results
        if (m_traceLevel > 0 && numSamplesLastLogged > 0)
        {
            DisplayEvalStatistics(numMBsRunLastLogged + 1, numMBsRun, numSamplesLastLogged, evalNodes, evalResults, evalResultsLastLogged);
        }

        // final statistics
        for (int i = 0; i < evalResultsLastLogged.size(); i++)
            evalResultsLastLogged[i] = EpochCriterion(0); // clear this since statistics display will subtract the previous value

        DisplayEvalStatistics(1, numMBsRun, totalEpochSamples, evalNodes, evalResults, evalResultsLastLogged, true, /*isFinal=*/true);

        return evalResults;
    }

protected:
    void DisplayEvalStatistics(const size_t startMBNum, const size_t endMBNum, const size_t numSamplesLastLogged,
                               const vector<ComputationNodeBasePtr>& evalNodes,
                               const EpochCriterion evalResults, const EpochCriterion evalResultsLastLogged, bool displayConvertedValue = false)
    {
        DisplayEvalStatistics(startMBNum, endMBNum, numSamplesLastLogged, evalNodes, { evalResults }, { evalResultsLastLogged }, displayConvertedValue);
    }

    void DisplayEvalStatistics(const size_t startMBNum, const size_t endMBNum, const size_t numSamplesLastLogged, const vector<ComputationNodeBasePtr>& evalNodes,
                               const vector<EpochCriterion>& evalResults, const vector<EpochCriterion>& evalResultsLastLogged, bool displayConvertedValue = false, bool isFinal = false)
    {
        LOGPRINTF(stderr, "%sMinibatch[%lu-%lu]: ", isFinal ? "Final Results: " : "", startMBNum, endMBNum);

        for (size_t i = 0; i < evalResults.size(); i++)
        {
            EpochCriterion criterionSinceLastLogged = evalResults[i] - evalResultsLastLogged[i];
            criterionSinceLastLogged.LogCriterion(evalNodes[i]->NodeName(), /*addSemicolon=*/false);

            if (displayConvertedValue)
            {
                // display Perplexity as well for crossEntropy values
                if (evalNodes[i]->OperationName() == OperationNameOf(CrossEntropyWithSoftmaxNode) ||
                    evalNodes[i]->OperationName() == OperationNameOf(CrossEntropyNode) ||
                    evalNodes[i]->OperationName() == OperationNameOf(ClassBasedCrossEntropyWithSoftmaxNode) ||
                    evalNodes[i]->OperationName() == OperationNameOf(NoiseContrastiveEstimationNode))
                    fprintf(stderr, "; perplexity = %.8f", std::exp(criterionSinceLastLogged.Average()));
            }

            if (i + 1 < evalResults.size())
                fprintf(stderr, "; ");
        }

        fprintf(stderr, "\n");
    }

protected:
    ComputationNetworkPtr m_net;
    size_t m_numMBsToShowResult;
    size_t m_firstMBsToShowResult;
    size_t m_maxSamplesInRAM;
    size_t m_numSubminiBatches;
    MPIWrapperPtr m_mpi;
    bool m_enableDistributedMBReading;

    std::shared_ptr<IDistGradAggregator<ElemType>> m_distGradAgg;
    std::shared_ptr<struct DistGradHeader> m_gradHeader;
    int m_traceLevel;
    void operator=(const SimpleEvaluator&); // (not assignable)
};

}}}
