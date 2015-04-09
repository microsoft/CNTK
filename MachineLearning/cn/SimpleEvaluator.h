//
// <copyright file="SimpleEvaluator.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include "ComputationNetworkHelper.h"
#include "DataReader.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "basetypes.h"
#include "fileutil.h"
#include "commandArgUtil.h"
#include <fstream>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class SimpleEvaluator : ComputationNetworkHelper<ElemType>
    {
        typedef ComputationNetworkHelper<ElemType> B;
        using B::UpdateEvalTimeStamps;
    protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;
        typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

    public:

        SimpleEvaluator(ComputationNetwork<ElemType>& net,  const size_t numMBsToShowResult=100, const int traceLevel=0) 
            : m_net(net), m_numMBsToShowResult(numMBsToShowResult), m_traceLevel(traceLevel)
        {
        }

        //returns evaluation node values per sample determined by evalNodeNames (which can include both training and eval criterion nodes)
        vector<ElemType> Evaluate(IDataReader<ElemType>& dataReader, const vector<wstring>& evalNodeNames, const size_t mbSize,  const size_t testSize=requestDataSize)
        {
            //specify evaluation nodes
            std::vector<ComputationNodePtr> evalNodes;

            if (evalNodeNames.size() == 0)
            {
                fprintf (stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
                if (m_net.EvaluationNodes().size() == 0 && m_net.FinalCriterionNodes().size() == 0)
                    throw std::logic_error("There is no default evalnodes or training criterion node specified in the network.");
            
                for (int i=0; i< m_net.EvaluationNodes().size(); i++)
                    evalNodes.push_back(m_net.EvaluationNodes()[i]);

                for (int i=0; i< m_net.FinalCriterionNodes().size(); i++)
                    evalNodes.push_back(m_net.FinalCriterionNodes()[i]);
            }
            else
            {
                for (int i=0; i<evalNodeNames.size(); i++)
                {
                    ComputationNodePtr node = m_net.GetNodeFromName(evalNodeNames[i]);
                    m_net.BuildAndValidateNetwork(node);
                    if (!node->FunctionValues().GetNumElements() == 1)
                    {
                        throw std::logic_error("The nodes passed to SimpleEvaluator::Evaluate function must be either eval or training criterion nodes (which evalues to 1x1 value).");
                    }
                    evalNodes.push_back(node);
                }
            }

            //initialize eval results
            std::vector<ElemType> evalResults;
            for (int i=0; i< evalNodes.size(); i++)
            {
                evalResults.push_back((ElemType)0);
            }

            //prepare features and labels
            std::vector<ComputationNodePtr> & FeatureNodes = m_net.FeatureNodes();
            std::vector<ComputationNodePtr> & labelNodes = m_net.LabelNodes();

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<FeatureNodes.size(); i++)
            {
                inputMatrices[FeatureNodes[i]->NodeName()] = &FeatureNodes[i]->FunctionValues();
            }
            for (size_t i=0; i<labelNodes.size(); i++)
            {
                inputMatrices[labelNodes[i]->NodeName()] = &labelNodes[i]->FunctionValues();                
            }

            //evaluate through minibatches
            size_t totalEpochSamples = 0;            
            size_t numMBsRun = 0;
            size_t actualMBSize = 0;
            size_t numSamplesLastMBs = 0;
            size_t lastMBsRun = 0; //MBs run before this display

            std::vector<ElemType> evalResultsLastMBs;
            for (int i=0; i< evalResults.size(); i++)
                evalResultsLastMBs.push_back((ElemType)0);

            dataReader.StartMinibatchLoop(mbSize, 0, testSize);

            while (dataReader.GetMinibatch(inputMatrices))
            {
                UpdateEvalTimeStamps(FeatureNodes);
                UpdateEvalTimeStamps(labelNodes);

                actualMBSize = m_net.GetActualMBSize();
                m_net.SetActualMiniBatchSize(actualMBSize);
                m_net.SetActualNbrSlicesInEachRecIter(dataReader.NumberSlicesInEachRecurrentIter());
                dataReader.SetSentenceSegBatch(m_net.m_sentenceSeg); 

                for (int i=0; i<evalNodes.size(); i++)
                {
                    m_net.Evaluate(evalNodes[i]);
                    evalResults[i] += evalNodes[i]->FunctionValues().Get00Element(); //criterionNode should be a scalar
                }

                totalEpochSamples += actualMBSize;
                numMBsRun++;

                if (m_traceLevel > 0)
                {
                    numSamplesLastMBs += actualMBSize; 

                if (numMBsRun % m_numMBsToShowResult == 0)
                {
                        DisplayEvalStatistics(lastMBsRun+1, numMBsRun, numSamplesLastMBs, evalNodes, evalResults, evalResultsLastMBs);

                        for (int i=0; i<evalResults.size(); i++)
                        {
                            evalResultsLastMBs[i] = evalResults[i];
                        }
                        numSamplesLastMBs = 0; 
                        lastMBsRun = numMBsRun;
                    }
                }

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                dataReader.DataEnd(endDataSentence); 
            }

            // show last batch of results
            if (m_traceLevel > 0 && numSamplesLastMBs > 0)
            {
                  DisplayEvalStatistics(lastMBsRun+1, numMBsRun, numSamplesLastMBs, evalNodes, evalResults, evalResultsLastMBs);
            }
            
            //final statistics
            for (int i=0; i<evalResultsLastMBs.size(); i++)
            {
                evalResultsLastMBs[i] = 0;
            }

            fprintf(stderr,"Final Results: ");
            DisplayEvalStatistics(1, numMBsRun, totalEpochSamples, evalNodes, evalResults, evalResultsLastMBs);
            
            for (int i=0; i<evalResults.size(); i++)
            {
                evalResults[i] /= totalEpochSamples;
            }

            return evalResults;
        }        

        //returns error rate
        ElemType EvaluateUnroll(IDataReader<ElemType>& dataReader, const size_t mbSize, ElemType &evalSetCrossEntropy, const wchar_t* output = nullptr, const size_t testSize = requestDataSize)
        {

            std::vector<ComputationNodePtr> FeatureNodes = m_net.FeatureNodes();
            std::vector<ComputationNodePtr> labelNodes = m_net.LabelNodes();
            std::vector<ComputationNodePtr> criterionNodes = m_net.FinalCriterionNodes();
            std::vector<ComputationNodePtr> evaluationNodes = m_net.EvaluationNodes();
            
            if (criterionNodes.size()==0)
            {
                throw std::runtime_error("No CrossEntropyWithSoftmax node found\n");
            }
            if (evaluationNodes.size()==0)
            {
                throw std::runtime_error("No Evaluation node found\n");
            }

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<FeatureNodes.size(); i++)
            {
                inputMatrices[FeatureNodes[i]->NodeName()] = &FeatureNodes[i]->FunctionValues();
            }
            for (size_t i=0; i<labelNodes.size(); i++)
            {
                inputMatrices[labelNodes[i]->NodeName()] = &labelNodes[i]->FunctionValues();                
            }
            inputMatrices[L"numberobs"] = new Matrix<ElemType>(1,1, m_net.GetDeviceID()); 

            dataReader.StartMinibatchLoop(mbSize, 0, testSize);

            ElemType epochEvalError = 0;
            ElemType epochCrossEntropy = 0;
            size_t totalEpochSamples = 0;
            ElemType prevEpochEvalError = 0;
            ElemType prevEpochCrossEntropy = 0;
            size_t prevTotalEpochSamples = 0;
            size_t prevStart = 1;
            size_t numSamples = 0;
            ElemType crossEntropy = 0;
            ElemType evalError = 0;
            
            ofstream outputStream;
            if (output)
            {
#ifdef _MSC_VER
                outputStream.open(output);
#else
                outputStream.open(charpath(output));    // GCC does not implement wide-char pathnames here
#endif
            }

            size_t numMBsRun = 0;
            size_t actualMBSize = 0;
            while (dataReader.GetMinibatch(inputMatrices))
            {
                size_t nbrSamples = (size_t)(*inputMatrices[L"numberobs"])(0, 0);
                actualMBSize = nbrSamples;

                for (int npos = 0; npos < nbrSamples ; npos++)
                {
                    FeatureNodes[npos]->UpdateEvalTimeStamp();
                    labelNodes[npos]->UpdateEvalTimeStamp();

                    m_net.Evaluate(criterionNodes[npos]); //use only the first criterion. Is there any possibility to use more?

                    m_net.Evaluate(evaluationNodes[npos]);

                    ElemType mbCrossEntropy = criterionNodes[npos]->FunctionValues().Get00Element(); // criterionNode should be a scalar
                    epochCrossEntropy += mbCrossEntropy;

                    ElemType mbEvalError = evaluationNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar

                    epochEvalError += mbEvalError;
                }

                totalEpochSamples += actualMBSize;

                if (outputStream.is_open())
                {
                    //TODO: add support to dump multiple outputs
                    ComputationNodePtr outputNode = m_net.OutputNodes()[0];
                    foreach_column(j, outputNode->FunctionValues())
                    {
                        foreach_row(i,outputNode->FunctionValues())
                        {
                            outputStream<<outputNode->FunctionValues()(i,j)<<" ";
                        }
                        outputStream<<endl;
                    }
                }

                numMBsRun++;
                if (numMBsRun % m_numMBsToShowResult == 0)
                {
                    numSamples = (totalEpochSamples - prevTotalEpochSamples);
                    crossEntropy = epochCrossEntropy - prevEpochCrossEntropy;
                    evalError = epochEvalError - prevEpochEvalError;

                    fprintf(stderr, "Minibatch[%lu-%lu]: Samples Evaluated = %lu    EvalErr Per Sample = %.8g    Loss Per Sample = %.8g\n",
                            prevStart, numMBsRun, numSamples, evalError / numSamples, crossEntropy / numSamples);

                    prevTotalEpochSamples = totalEpochSamples;
                    prevEpochCrossEntropy = epochCrossEntropy;
                    prevEpochEvalError = epochEvalError;
                    prevStart = numMBsRun + 1;
                }

            }

            // show final grouping of output
            numSamples = totalEpochSamples - prevTotalEpochSamples;
            if (numSamples > 0)
            {
                crossEntropy = epochCrossEntropy - prevEpochCrossEntropy;
                evalError = epochEvalError - prevEpochEvalError;
                fprintf(stderr, "Minibatch[%lu-%lu]: Samples Evaluated = %lu    EvalErr Per Sample = %.8g    Loss Per Sample = %.8g\n",
                    prevStart, numMBsRun, numSamples, evalError / numSamples, crossEntropy / numSamples);
            }

            //final statistics
            epochEvalError /= (ElemType)totalEpochSamples;
            epochCrossEntropy /= (ElemType)totalEpochSamples;
            fprintf(stderr, "Overall: Samples Evaluated = %lu   EvalErr Per Sample = %.8g   Loss Per Sample = %.8g\n", totalEpochSamples, epochEvalError, epochCrossEntropy);
            if (outputStream.is_open())
            {
                outputStream.close();
            }
            evalSetCrossEntropy = epochCrossEntropy;
            return epochEvalError;
        }

    protected:
        void DisplayEvalStatistics(const size_t startMBNum, const size_t endMBNum, const size_t numSamplesLastMBs, const vector<ComputationNodePtr>& evalNodes, 
            const vector<ElemType> & evalResults, const vector<ElemType> & evalResultsLastMBs)
        {
            fprintf(stderr,"Minibatch[%lu-%lu]: Samples Seen = %lu    ", startMBNum, endMBNum, numSamplesLastMBs);

            for (size_t i=0; i<evalResults.size(); i++)
            {
                fprintf(stderr, "%ls/Sample = %.8g    ", evalNodes[i]->NodeName().c_str(), (evalResults[i]-evalResultsLastMBs[i])/numSamplesLastMBs);
            }

            fprintf(stderr, "\n");
        }

    protected: 
        ComputationNetwork<ElemType>& m_net;
        size_t m_numMBsToShowResult;
        int m_traceLevel;
        void operator=(const SimpleEvaluator&); // (not assignable)

    public:
        /// for encoder-decoder RNN
        list<pair<wstring, wstring>> m_lst_pair_encoder_decode_node_names;
        list<pair<ComputationNodePtr, ComputationNodePtr>> m_lst_pair_encoder_decoder_nodes;

        void SetEncoderDecoderNodePairs(std::list<pair<ComputationNodePtr, ComputationNodePtr>>& lst_pair_encoder_decoder_nodes)
        {
            m_lst_pair_encoder_decoder_nodes.clear();
            for (std::list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = lst_pair_encoder_decoder_nodes.begin(); iter != lst_pair_encoder_decoder_nodes.end(); iter++)
                m_lst_pair_encoder_decoder_nodes.push_back(*iter);
        }

        /// this evaluates encoder network and decoder network
        vector<ElemType> EvaluateEncoderDecoderWithHiddenStates(
            ComputationNetwork<ElemType>& encoderNet,
            ComputationNetwork<ElemType>& decoderNet,
            IDataReader<ElemType>& encoderDataReader,
            IDataReader<ElemType>& decoderDataReader,
            const vector<wstring>& encoderEvalNodeNames,
            const vector<wstring>& decoderEvalNodeNames,
            const size_t mbSize,
            const size_t testSize = requestDataSize)
        {
            //specify evaluation nodes
            std::vector<ComputationNodePtr> encoderEvalNodes;
            std::vector<ComputationNodePtr> decoderEvalNodes;

            if (encoderEvalNodeNames.size() == 0)
            {
                fprintf(stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
                if (encoderNet.EvaluationNodes().size() == 0)
                    throw std::logic_error("There is no default evalnodes criterion node specified in the network.");

                for (int i = 0; i < encoderNet.EvaluationNodes().size(); i++)
                    encoderEvalNodes.push_back(encoderNet.EvaluationNodes()[i]);
            }
            else
            {
                for (int i = 0; i < encoderEvalNodeNames.size(); i++)
                {
                    ComputationNodePtr node = encoderNet.GetNodeFromName(encoderEvalNodeNames[i]);
                    encoderNet.BuildAndValidateNetwork(node);
                    if (!node->FunctionValues().GetNumElements() == 1)
                    {
                        throw std::logic_error("The nodes passed to SimpleEvaluator::Evaluate function must be either eval or training criterion nodes (which evalues to 1x1 value).");
                    }
                    encoderEvalNodes.push_back(node);
                }
            }

            if (decoderEvalNodeNames.size() == 0)
            {
                fprintf(stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
                if (decoderNet.EvaluationNodes().size() == 0)
                    throw std::logic_error("There is no default evalnodes criterion node specified in the network.");
                if (decoderNet.FinalCriterionNodes().size() == 0)
                    throw std::logic_error("There is no default criterion criterion node specified in the network.");

                for (int i = 0; i < decoderNet.EvaluationNodes().size(); i++)
                    decoderEvalNodes.push_back(encoderNet.EvaluationNodes()[i]);

                for (int i = 0; i < decoderNet.FinalCriterionNodes().size(); i++)
                    decoderEvalNodes.push_back(decoderNet.FinalCriterionNodes()[i]);
            }
            else
            {
                for (int i = 0; i < decoderEvalNodeNames.size(); i++)
                {
                    ComputationNodePtr node = decoderNet.GetNodeFromName(decoderEvalNodeNames[i]);
                    decoderNet.BuildAndValidateNetwork(node);
                    if (!node->FunctionValues().GetNumElements() == 1)
                    {
                        throw std::logic_error("The nodes passed to SimpleEvaluator::Evaluate function must be either eval or training criterion nodes (which evalues to 1x1 value).");
                    }
                    decoderEvalNodes.push_back(node);
                }
            }

            if (m_lst_pair_encoder_decoder_nodes.size() == 0)
                throw runtime_error("TrainOneEpochEncoderDecoderWithHiddenStates: no encoder and decoder node pairs");

            //initialize eval results
            std::vector<ElemType> evalResults;
            for (int i = 0; i < decoderEvalNodes.size(); i++)
            {
                evalResults.push_back((ElemType)0);
            }

            //prepare features and labels
            std::vector<ComputationNodePtr> & encoderFeatureNodes = encoderNet.FeatureNodes();

            std::vector<ComputationNodePtr> & decoderFeatureNodes = decoderNet.FeatureNodes();
            std::vector<ComputationNodePtr> & decoderLabelNodes = decoderNet.LabelNodes();

            std::map<std::wstring, Matrix<ElemType>*> encoderInputMatrices;
            for (size_t i = 0; i < encoderFeatureNodes.size(); i++)
            {
                encoderInputMatrices[encoderFeatureNodes[i]->NodeName()] = &encoderFeatureNodes[i]->FunctionValues();
            }

            std::map<std::wstring, Matrix<ElemType>*> decoderInputMatrices;
            for (size_t i = 0; i < decoderFeatureNodes.size(); i++)
            {
                decoderInputMatrices[decoderFeatureNodes[i]->NodeName()] = &decoderFeatureNodes[i]->FunctionValues();
            }
            for (size_t i = 0; i < decoderLabelNodes.size(); i++)
            {
                decoderInputMatrices[decoderLabelNodes[i]->NodeName()] = &decoderLabelNodes[i]->FunctionValues();
            }

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t numMBsRun = 0;
            size_t actualMBSize = 0;
            size_t numSamplesLastMBs = 0;
            size_t lastMBsRun = 0; //MBs run before this display

            std::vector<ElemType> evalResultsLastMBs;
            for (int i = 0; i < evalResults.size(); i++)
                evalResultsLastMBs.push_back((ElemType)0);

            encoderDataReader.StartMinibatchLoop(mbSize, 0, testSize);
            decoderDataReader.StartMinibatchLoop(mbSize, 0, testSize);

            Matrix<ElemType> mEncoderOutput(encoderEvalNodes[0]->FunctionValues().GetDeviceId());
            Matrix<ElemType> historyMat(encoderEvalNodes[0]->FunctionValues().GetDeviceId());

            bool bContinueDecoding = true;
            while (bContinueDecoding){
                /// first evaluate encoder network
                if (encoderDataReader.GetMinibatch(encoderInputMatrices) == false)
                    break;
                actualMBSize = decoderDataReader.GetMinibatch(decoderInputMatrices);
                if (actualMBSize == 0)
                    break;
                UpdateEvalTimeStamps(encoderFeatureNodes);
                UpdateEvalTimeStamps(decoderFeatureNodes);

                encoderNet.SetActualMiniBatchSize(actualMBSize);
                encoderNet.SetActualNbrSlicesInEachRecIter(encoderDataReader.NumberSlicesInEachRecurrentIter());
                encoderDataReader.SetSentenceSegBatch(encoderNet.m_sentenceSeg);
//                encoderDataReader.SetSentenceSegBatch(encoderNet.m_sentenceBegin);

                assert(encoderEvalNodes.size() == 1);
                for (int i = 0; i < encoderEvalNodes.size(); i++)
                {
                    encoderNet.Evaluate(encoderEvalNodes[i]);
                }


                /// not the sentence begining, because the initial hidden layer activity is from the encoder network
//                decoderDataReader.SetSentenceBegin(false);
                decoderNet.SetActualNbrSlicesInEachRecIter(decoderDataReader.NumberSlicesInEachRecurrentIter());
                decoderDataReader.SetSentenceSegBatch(decoderNet.m_sentenceSeg);
//                decoderDataReader.SetSentenceSegBatch(decoderNet.m_sentenceBegin);
//                decoderNet.m_sentenceBegin.assign(decoderNet.m_sentenceBegin.size(), -1);

                /// get the pair of encode and decoder nodes
                for (list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = m_lst_pair_encoder_decoder_nodes.begin(); iter != m_lst_pair_encoder_decoder_nodes.end(); iter++)
                {
                    /// past hidden layer activity from encoder network to decoder network
                    ComputationNodePtr encoderNode = iter->first;
                    ComputationNodePtr decoderNode = iter->second;

                    encoderNode->GetHistory(historyMat, true);
                    decoderNode->SetHistory(historyMat);
                }

                for (int i = 0; i<decoderEvalNodes.size(); i++)
                {
                    decoderNet.Evaluate(decoderEvalNodes[i]);
                    evalResults[i] += decoderEvalNodes[i]->FunctionValues().Get00Element(); //criterionNode should be a scalar
                }

                totalEpochSamples += actualMBSize;
                numMBsRun++;

                if (m_traceLevel > 0)
                {
                    numSamplesLastMBs += actualMBSize;

                    if (numMBsRun % m_numMBsToShowResult == 0)
                    {
                        DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, decoderEvalNodes, evalResults, evalResultsLastMBs);

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
//                encoderDataReader.SetSentenceEnd(true);
//                decoderDataReader.SetSentenceEnd(true);
                encoderDataReader.DataEnd(endDataSentence);
                decoderDataReader.DataEnd(endDataSentence);
            }

            // show last batch of results
            if (m_traceLevel > 0 && numSamplesLastMBs > 0)
            {
                DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, decoderEvalNodes, evalResults, evalResultsLastMBs);
            }

            //final statistics
            for (int i = 0; i < evalResultsLastMBs.size(); i++)
            {
                evalResultsLastMBs[i] = 0;
            }

            fprintf(stderr, "Final Results: ");
            DisplayEvalStatistics(1, numMBsRun, totalEpochSamples, decoderEvalNodes, evalResults, evalResultsLastMBs);

            for (int i = 0; i < evalResults.size(); i++)
            {
                evalResults[i] /= totalEpochSamples;
            }

            return evalResults;
        }
    };

}}}
