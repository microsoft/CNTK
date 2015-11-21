//p
//
// <copyright file="SimpleEvaluator.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
//#include "Helpers.h"    // for foreach_column() macro
//#include "fileutil.h"
#include "DataReader.h"
//#include "DataWriter.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "DataReaderHelpers.h"
#include "TrainingCriterionNodes.h" // TODO: we should move the functions that depend on these to the .cpp
//#include "CompositeComputationNodes.h"
#include <vector>
#include <string>
//#include <fstream>
//#include <queue>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    // TODO: get rid of dependency on ElemType
    template<class ElemType>
    class SimpleEvaluator
    {
    protected:

    public:

        SimpleEvaluator(ComputationNetworkPtr net, const size_t numMBsToShowResult = 100, const int traceLevel = 0)
            : m_net(net), m_numMBsToShowResult(numMBsToShowResult), m_traceLevel(traceLevel)
        {
        }

        // returns evaluation node values per sample determined by evalNodeNames (which can include both training and eval criterion nodes)
        vector<double> Evaluate(IDataReader<ElemType>* dataReader, const vector<wstring>& evalNodeNames, const size_t mbSize, const size_t testSize = requestDataSize)
        {
            //specify evaluation nodes
            std::vector<ComputationNodeBasePtr> evalNodes;

            if (evalNodeNames.size() == 0)
            {
                fprintf(stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
                if (m_net->EvaluationNodes().size() == 0 && m_net->FinalCriterionNodes().size() == 0)
                    LogicError("There is no default evalnodes or training criterion node specified in the network.");

                for (int i = 0; i < m_net->EvaluationNodes().size(); i++)
                    evalNodes.push_back(m_net->EvaluationNodes()[i]);

                for (int i = 0; i < m_net->FinalCriterionNodes().size(); i++)
                    evalNodes.push_back(m_net->FinalCriterionNodes()[i]);
            }
            else
            {
                for (int i = 0; i < evalNodeNames.size(); i++)
                {
                    const auto & node = m_net->GetNodeFromName(evalNodeNames[i]);
                    m_net->BuildAndValidateSubNetwork(node);
                    if (node->GetNumRows() != 1 || node->GetNumCols() != 1)
                        LogicError("The nodes passed to SimpleEvaluator::Evaluate function must be either eval or training criterion nodes (which evalues to 1x1 value).");
                    evalNodes.push_back(node);
                }
            }

            //initialize eval results
            std::vector<double> evalResults;
            for (int i = 0; i < evalNodes.size(); i++)
                evalResults.push_back((double)0);

            //prepare features and labels
            auto & featureNodes = m_net->FeatureNodes();
            auto & labelNodes = m_net->LabelNodes();

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i = 0; i < featureNodes.size(); i++)
                inputMatrices[featureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(featureNodes[i])->FunctionValues();
            for (size_t i = 0; i < labelNodes.size(); i++)
                inputMatrices[labelNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(labelNodes[i])->FunctionValues();

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t numMBsRun = 0;
            size_t actualMBSize = 0;
            size_t numSamplesLastMBs = 0;
            size_t lastMBsRun = 0; //MBs run before this display

            std::vector<double> evalResultsLastMBs;
            for (int i = 0; i < evalResults.size(); i++)
                evalResultsLastMBs.push_back((ElemType)0);

            dataReader->StartMinibatchLoop(mbSize, 0, testSize);
            m_net->StartEvaluateMinibatchLoop(evalNodes);

            while (DataReaderHelpers::GetMinibatchIntoNetwork(*dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
            {
                ComputationNetwork::UpdateEvalTimeStamps(featureNodes);
                ComputationNetwork::UpdateEvalTimeStamps(labelNodes);

                //for now since we share the same label masking flag we call this on one node only
                //Later, when we apply different labels on different nodes
                //we need to add code to call this function multiple times, one for each criteria node
                size_t numSamplesWithLabel = m_net->GetNumSamplesWithLabel(actualMBSize);
                for (int i = 0; i < evalNodes.size(); i++)
                {
                    m_net->Evaluate(evalNodes[i]);
                    evalResults[i] += (double)evalNodes[i]->Get00Element(); //criterionNode should be a scalar
                }

                totalEpochSamples += numSamplesWithLabel;
                numMBsRun++;

                if (m_traceLevel > 0)
                {
                    numSamplesLastMBs += numSamplesWithLabel;

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

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                dataReader->DataEnd(endDataSentence);
            }

            // show last batch of results
            if (m_traceLevel > 0 && numSamplesLastMBs > 0)
            {
                DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, evalNodes, evalResults, evalResultsLastMBs);
            }

            //final statistics
            for (int i = 0; i < evalResultsLastMBs.size(); i++)
            {
                evalResultsLastMBs[i] = 0;
            }

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
                                    const vector<double> & evalResults, const vector<double> & evalResultsLastMBs, bool displayConvertedValue = false)
        {
            fprintf(stderr, "Minibatch[%lu-%lu]: Samples Seen = %lu    ", startMBNum, endMBNum, numSamplesLastMBs);

            for (size_t i = 0; i < evalResults.size(); i++)
            {
                double eresult = (evalResults[i] - evalResultsLastMBs[i]) / numSamplesLastMBs;
                fprintf(stderr, "%ls: %ls/Sample = %.8g    ", evalNodes[i]->NodeName().c_str(), evalNodes[i]->OperationName().c_str(), eresult);

                if (displayConvertedValue)
                {
                    //display Perplexity as well for crossEntropy values
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
        int m_traceLevel;
        void operator=(const SimpleEvaluator&); // (not assignable)
    };

}}}
