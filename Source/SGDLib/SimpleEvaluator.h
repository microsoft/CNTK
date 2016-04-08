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
    SimpleEvaluator(ComputationNetworkPtr net, const MPIWrapperPtr& mpi, const size_t numMBsToShowResult = 100, const int traceLevel = 0, const size_t maxSamplesInRAM = SIZE_MAX,
                    const size_t numSubminiBatches = 1)
        : m_net(net), 
          m_numMBsToShowResult(numMBsToShowResult), 
          m_traceLevel(traceLevel),
          m_maxSamplesInRAM(maxSamplesInRAM), 
          m_numSubminiBatches(numSubminiBatches), 
          m_mpi(mpi), 
          m_distGradAgg(nullptr),
          m_gradHeader(nullptr)
    {
    }

    // returns evaluation node values per sample determined by evalNodeNames (which can include both training and eval criterion nodes)
    vector<double> Evaluate(IDataReader* dataReader, const vector<wstring>& evalNodeNames, const size_t mbSize, const size_t testSize = requestDataSize)
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
        std::vector<double> evalResults;
        for (int i = 0; i < evalNodes.size(); i++)
            evalResults.push_back((double) 0);

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
        size_t actualMBSize = 0;
        size_t numSamplesLastMBs = 0;
        size_t lastMBsRun = 0; // MBs run before this display

        std::vector<double> evalResultsLastMBs;
        for (int i = 0; i < evalResults.size(); i++)
            evalResultsLastMBs.push_back((ElemType) 0);

        //TODO: we should add support for distributed reading
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

        const size_t numIterationsBeforePrintingProgress = 100;
        size_t numItersSinceLastPrintOfProgress = 0;
        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(*dataReader, m_net, nullptr, dataReader->SupportsDistributedMBRead(), m_mpi != nullptr, inputMatrices, actualMBSize, m_mpi))
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

            // BUGBUG (Issue #95): Once we have multiple layouts, this must be done on a per-node basis.
            size_t numSamplesWithLabel = m_net->GetNumSamplesWithLabelOfNetwork(actualMBSize);
            size_t aggregateNumSamplesWithLabel = numSamplesWithLabel;
            if (m_mpi != nullptr)
            {
                if (m_gradHeader == nullptr)
                {
                    m_gradHeader = DistGradHeader::Create(evalNodes.size());
                    m_distGradAgg = make_shared<SimpleDistGradAggregator<ElemType>>(m_mpi, false, m_traceLevel);
                }

                m_gradHeader->numEvalNode = evalNodes.size();
                m_gradHeader->numSamples = actualMBSize;
                m_gradHeader->numSamplesWithLabel = numSamplesWithLabel;
                m_gradHeader->criterion = 0.0;
                for (size_t i = 0; i < evalNodes.size(); i++)
                    m_gradHeader->evalErrors[i] = evalNodes[i]->Get00Element();

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
                m_distGradAgg->AggregateGradients(learnParamsGradients, m_gradHeader, 0);
                aggregateNumSamplesWithLabel = m_gradHeader->numSamplesWithLabel;
                for (size_t i = 0; i < evalResults.size(); i++)
                    evalResults[i] += m_gradHeader->evalErrors[i];
            }
            else
            {
                for (int i = 0; i < evalNodes.size(); i++)
                {
                    evalResults[i] += (double)evalNodes[i]->Get00Element(); // criterionNode should be a scalar
                }
            }

            totalEpochSamples += aggregateNumSamplesWithLabel;
            numMBsRun++;

            if (m_traceLevel > 0)
            {
                numSamplesLastMBs += aggregateNumSamplesWithLabel;

                if (numMBsRun % m_numMBsToShowResult == 0)
                {
                    DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, evalNodes, evalResults, evalResultsLastMBs);

                    for (int i = 0; i < evalResults.size(); i++)
                    {
                        evalResultsLastMBs[i] = evalResults[i];
                    }
                    numSamplesLastMBs = 0;
                    lastMBsRun = numMBsRun;
                }
            }


            numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);

            // call DataEnd to check if end of sentence is reached
            // datareader will do its necessary/specific process for sentence ending
            dataReader->DataEnd();
        }

        // show last batch of results
        if (m_traceLevel > 0 && numSamplesLastMBs > 0)
        {
            DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, evalNodes, evalResults, evalResultsLastMBs);
        }

        // final statistics
        for (int i = 0; i < evalResultsLastMBs.size(); i++)
            evalResultsLastMBs[i] = 0; // clear this since statistics display will subtract the previous value

        fprintf(stderr, "Final Results: ");
        DisplayEvalStatistics(1, numMBsRun, totalEpochSamples, evalNodes, evalResults, evalResultsLastMBs, true);

        for (int i = 0; i < evalResults.size(); i++)
        {
            evalResults[i] /= totalEpochSamples;
        }

        return evalResults;
    }

protected:
    void DisplayEvalStatistics(const size_t startMBNum, const size_t endMBNum, const size_t numSamplesLastMBs,
                               const vector<ComputationNodeBasePtr>& evalNodes,
                               const double evalResults, const double evalResultsLastMBs, bool displayConvertedValue = false)
    {
        vector<double> evaR;
        evaR.push_back(evalResults);
        vector<double> evaLast;
        evaLast.push_back(evalResultsLastMBs);

        DisplayEvalStatistics(startMBNum, endMBNum, numSamplesLastMBs, evalNodes, evaR, evaLast, displayConvertedValue);
    }

    void DisplayEvalStatistics(const size_t startMBNum, const size_t endMBNum, const size_t numSamplesLastMBs, const vector<ComputationNodeBasePtr>& evalNodes,
                               const vector<double>& evalResults, const vector<double>& evalResultsLastMBs, bool displayConvertedValue = false)
    {
        fprintf(stderr, "Minibatch[%lu-%lu]: SamplesSeen = %lu    ", startMBNum, endMBNum, numSamplesLastMBs);

        for (size_t i = 0; i < evalResults.size(); i++)
        {
            double eresult = (evalResults[i] - evalResultsLastMBs[i]) / numSamplesLastMBs;
            fprintf(stderr, "%ls: %ls/Sample = %.8g    ", evalNodes[i]->NodeName().c_str(), evalNodes[i]->OperationName().c_str(), eresult);

            if (displayConvertedValue)
            {
                // display Perplexity as well for crossEntropy values
                if (evalNodes[i]->OperationName() == OperationNameOf(CrossEntropyWithSoftmaxNode) ||
                    evalNodes[i]->OperationName() == OperationNameOf(CrossEntropyNode) ||
                    evalNodes[i]->OperationName() == OperationNameOf(ClassBasedCrossEntropyWithSoftmaxNode) ||
                    evalNodes[i]->OperationName() == OperationNameOf(NoiseContrastiveEstimationNode))
                    fprintf(stderr, "Perplexity = %.8g    ", std::exp(eresult));
            }
        }

        fprintf(stderr, "\n");
    }

protected:
    ComputationNetworkPtr m_net;
    size_t m_numMBsToShowResult;
    size_t m_maxSamplesInRAM;
    size_t m_numSubminiBatches;
    MPIWrapperPtr m_mpi;

    shared_ptr<IDistGradAggregator<ElemType>> m_distGradAgg;
    struct DistGradHeader* m_gradHeader;
    int m_traceLevel;
    void operator=(const SimpleEvaluator&); // (not assignable)
};
} } }
