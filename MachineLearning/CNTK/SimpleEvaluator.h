//p
//
// <copyright file="SimpleEvaluator.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <queue>
#include "Basics.h"
#include "fileutil.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkHelper.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {
    template<class ElemType>
    struct NN_state {
        map<wstring, Matrix<ElemType>> hidden_activity;
    };

    template<class ElemType>
    struct Token{
        Token(const ElemType score, const std::vector<size_t> &sequence, const NN_state<ElemType> & state)
            : score(score), sequence(sequence), state(state) {
        }
        bool operator<(const Token &t) const {
            return score < t.score;
        }
        ElemType score;
        vector<size_t> sequence;
        NN_state<ElemType> state;
    };


    template<class ElemType>
    class SimpleEvaluator : ComputationNetworkHelper<ElemType>
    {
        typedef ComputationNetworkHelper<ElemType> B;
        using B::UpdateEvalTimeStamps;
    protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;
        typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

    protected:
        /// used for backward directional nodes
        std::list<ComputationNodePtr> batchComputeNodes;

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
                dataReader.SetSentenceSegBatch(m_net.SentenceBoundary(), m_net.MinibatchPackingFlags());

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
            DisplayEvalStatistics(1, numMBsRun, totalEpochSamples, evalNodes, evalResults, evalResultsLastMBs, true);
            
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
            const vector<ElemType> & evalResults, const vector<ElemType> & evalResultsLastMBs, bool displayConvertedValue = false)
        {
            fprintf(stderr,"Minibatch[%lu-%lu]: Samples Seen = %lu    ", startMBNum, endMBNum, numSamplesLastMBs);

            for (size_t i=0; i<evalResults.size(); i++)
            {
                ElemType eresult = (evalResults[i] - evalResultsLastMBs[i]) / numSamplesLastMBs;
                fprintf(stderr, "%ls: %ls/Sample = %.8g    ", evalNodes[i]->NodeName().c_str(), evalNodes[i]->OperationName().c_str(), eresult);

                if (displayConvertedValue)
                {
                    //display Perplexity as well for crossEntropy values
                    if (evalNodes[i]->OperationName() == CrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
                        evalNodes[i]->OperationName() == CrossEntropyNode<ElemType>::TypeName() ||
                        evalNodes[i]->OperationName() == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
                        evalNodes[i]->OperationName() == NoiseContrastiveEstimationNode<ElemType>::TypeName())
                        fprintf(stderr, "Perplexity = %.8g    ", std::exp(eresult));
                }
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
            for (typename std::list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = lst_pair_encoder_decoder_nodes.begin(); iter != lst_pair_encoder_decoder_nodes.end(); iter++)
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
                if (decoderDataReader.GetMinibatch(decoderInputMatrices) == false)
                    break;
                UpdateEvalTimeStamps(encoderFeatureNodes);
                UpdateEvalTimeStamps(decoderFeatureNodes);

                actualMBSize = decoderNet.GetActualMBSize();
                if (actualMBSize == 0)
                    LogicError("decoderTrainSetDataReader read data but decoderNet reports no data read");

                encoderNet.SetActualMiniBatchSize(actualMBSize);
                encoderNet.SetActualNbrSlicesInEachRecIter(encoderDataReader.NumberSlicesInEachRecurrentIter());
                encoderDataReader.SetSentenceSegBatch(encoderNet.SentenceBoundary(), encoderNet.MinibatchPackingFlags());

                assert(encoderEvalNodes.size() == 1);
                for (int i = 0; i < encoderEvalNodes.size(); i++)
                {
                    encoderNet.Evaluate(encoderEvalNodes[i]);
                }


                /// not the sentence begining, because the initial hidden layer activity is from the encoder network
                decoderNet.SetActualNbrSlicesInEachRecIter(decoderDataReader.NumberSlicesInEachRecurrentIter());
                decoderDataReader.SetSentenceSegBatch(decoderNet.SentenceBoundary(), decoderNet.MinibatchPackingFlags());

                /// get the pair of encode and decoder nodes
                for (typename list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = m_lst_pair_encoder_decoder_nodes.begin(); iter != m_lst_pair_encoder_decoder_nodes.end(); iter++)
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

        void InitTrainEncoderDecoderWithHiddenStates(const ConfigParameters& readerConfig)
        {
            ConfigArray arrEncoderNodeNames = readerConfig("encoderNodes", "");
            vector<wstring> encoderNodeNames;

            m_lst_pair_encoder_decode_node_names.clear();;

            if (arrEncoderNodeNames.size() > 0)
            {
                /// newer code that explicitly place multiple streams for inputs
                foreach_index(i, arrEncoderNodeNames) // inputNames should map to node names
                {
                    wstring nodeName = arrEncoderNodeNames[i];
                    encoderNodeNames.push_back(nodeName);
                }
            }

            ConfigArray arrDecoderNodeNames = readerConfig("decoderNodes", "");
            vector<wstring> decoderNodeNames;
            if (arrDecoderNodeNames.size() > 0)
            {
                /// newer code that explicitly place multiple streams for inputs
                foreach_index(i, arrDecoderNodeNames) // inputNames should map to node names
                {
                    wstring nodeName = arrDecoderNodeNames[i];
                    decoderNodeNames.push_back(nodeName);
                }
            }

            assert(encoderNodeNames.size() == decoderNodeNames.size());

            for (size_t i = 0; i < encoderNodeNames.size(); i++)
            {
                m_lst_pair_encoder_decode_node_names.push_back(make_pair(encoderNodeNames[i], decoderNodeNames[i]));
            }
        }

        void EncodingEvaluateDecodingBeamSearch(
            ComputationNetwork<ElemType>& encoderNet,
            ComputationNetwork<ElemType>& decoderNet,
            IDataReader<ElemType>& encoderDataReader,
            IDataReader<ElemType>& decoderDataReader,
            IDataWriter<ElemType>& dataWriter,
            const vector<wstring>& outputNodeNames, const vector<wstring>& writeNodeNames,
            const size_t mbSize, const ElemType beam, const size_t testSize)
        {
            std::vector<ComputationNodePtr> encoderEvalNodes;
            for (int i = 0; i< encoderNet.OutputNodes().size(); i++)
                encoderEvalNodes.push_back(encoderNet.OutputNodes()[i]);
            assert(encoderEvalNodes.size() == 1);

            //specify output nodes and files
            std::vector<ComputationNodePtr> outputNodes;
            for (int i = 0; i<outputNodeNames.size(); i++)
                outputNodes.push_back(decoderNet.GetNodeFromName(outputNodeNames[i]));

            //specify nodes to write to file
            std::vector<ComputationNodePtr> writeNodes;
            for (int i = 0; i<writeNodeNames.size(); i++)
                writeNodes.push_back(m_net.GetNodeFromName(writeNodeNames[i]));

            //prepare features and labels
            std::vector<ComputationNodePtr> & encoderFeatureNodes = encoderNet.FeatureNodes();
            std::vector<ComputationNodePtr> & decoderFeatureNodes = decoderNet.FeatureNodes();
            std::vector<ComputationNodePtr> & decoderLabelNodes = decoderNet.LabelNodes();

            std::map<std::wstring, Matrix<ElemType>*> encoderInputMatrices;
            for (size_t i = 0; i<encoderFeatureNodes.size(); i++)
            {
                encoderInputMatrices[encoderFeatureNodes[i]->NodeName()] = &encoderFeatureNodes[i]->FunctionValues();
            }

            std::map<std::wstring, Matrix<ElemType>*> decoderInputMatrices;
            for (size_t i = 0; i<decoderFeatureNodes.size(); i++)
            {
                decoderInputMatrices[decoderFeatureNodes[i]->NodeName()] = &decoderFeatureNodes[i]->FunctionValues();
            }
            for (size_t i = 0; i<decoderLabelNodes.size(); i++)
            {
                decoderInputMatrices[decoderLabelNodes[i]->NodeName()] = &decoderLabelNodes[i]->FunctionValues();
            }

            /// get the pair of encode and decoder nodes
            if (m_lst_pair_encoder_decoder_nodes.size() == 0 && m_lst_pair_encoder_decode_node_names.size() > 0)
            {
                for (list<pair<wstring, wstring>>::iterator iter = m_lst_pair_encoder_decode_node_names.begin(); iter != m_lst_pair_encoder_decode_node_names.end(); iter++)
                {
                    /// past hidden layer activity from encoder network to decoder network
                    ComputationNodePtr encoderNode = encoderNet.GetNodeFromName(iter->first);
                    ComputationNodePtr decoderNode = decoderNet.GetNodeFromName(iter->second);

                    if (encoderNode != nullptr && decoderNode != nullptr)
                        m_lst_pair_encoder_decoder_nodes.push_back(make_pair(encoderNode, decoderNode));
                }
            }

            if (m_lst_pair_encoder_decoder_nodes.size() == 0)
                throw runtime_error("TrainOneEpochEncoderDecoderWithHiddenStates: no encoder and decoder node pairs");

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t actualMBSize = 0;

            encoderDataReader.StartMinibatchLoop(mbSize, 0, testSize);
            encoderDataReader.SetNbrSlicesEachRecurrentIter(1);
            decoderDataReader.StartMinibatchLoop(mbSize, 0, testSize);
            decoderDataReader.SetNbrSlicesEachRecurrentIter(1);

            Matrix<ElemType> mEncoderOutput(encoderEvalNodes[0]->FunctionValues().GetDeviceId());
            Matrix<ElemType> historyMat(encoderEvalNodes[0]->FunctionValues().GetDeviceId());

            bool bDecoding = true; 
            while (bDecoding){
                if (encoderDataReader.GetMinibatch(encoderInputMatrices) == false)
                    break;

                UpdateEvalTimeStamps(encoderFeatureNodes);

                actualMBSize = encoderNet.GetActualMBSize();

                encoderNet.SetActualMiniBatchSize(actualMBSize);
                encoderNet.SetActualNbrSlicesInEachRecIter(encoderDataReader.NumberSlicesInEachRecurrentIter());
                encoderDataReader.SetSentenceSegBatch(encoderNet.SentenceBoundary(), encoderNet.MinibatchPackingFlags());

                assert(encoderEvalNodes.size() == 1);
                for (int i = 0; i<encoderEvalNodes.size(); i++)
                {
                    encoderNet.Evaluate(encoderEvalNodes[i]);
                }

                size_t mNutt = encoderDataReader.NumberSlicesInEachRecurrentIter();

                /// get the pair of encode and decoder nodes
                for (typename list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = m_lst_pair_encoder_decoder_nodes.begin(); iter != m_lst_pair_encoder_decoder_nodes.end(); iter++)
                {
                    /// past hidden layer activity from encoder network to decoder network
                    ComputationNodePtr encoderNode = iter->first;
                    ComputationNodePtr decoderNode = iter->second;

                    encoderNode->GetHistory(historyMat, true);
#ifdef DEBUG_DECODER
                    fprintf(stderr, "LSTM past output norm = %.8e\n", historyMat.ColumnSlice(0, 1).FrobeniusNorm());
                    fprintf(stderr, "LSTM past state norm = %.8e\n", historyMat.ColumnSlice(1, 1).FrobeniusNorm());
#endif
                    decoderNode->SetHistory(historyMat);
                }

                vector<size_t> best_path;

                decoderNet.SetActualMiniBatchSize(actualMBSize);
                decoderDataReader.SetNbrSlicesEachRecurrentIter(mNutt);
                decoderNet.SetActualNbrSlicesInEachRecIter(decoderDataReader.NumberSlicesInEachRecurrentIter());

                decoderNet.SentenceBoundary().Resize(decoderDataReader.NumberSlicesInEachRecurrentIter(), 1);
                decoderNet.SentenceBoundary().SetValue(SENTENCE_MIDDLE);

                FindBestPathWithVariableLength(decoderNet, actualMBSize, decoderDataReader, dataWriter, outputNodes, writeNodes, decoderFeatureNodes, beam, decoderInputMatrices, best_path);

                totalEpochSamples += actualMBSize;

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                encoderDataReader.DataEnd(endDataSentence);
            }
        }

        bool GetCandidatesAtOneTimeInstance(const Matrix<ElemType>& score,
            const ElemType & preScore, const ElemType & threshold,
            const ElemType& best_score_so_far,
            vector<pair<int, ElemType>>& rCandidate)
        {
            Matrix<ElemType> ptrScore(CPUDEVICE);
            ptrScore = score;

            ElemType *pPointer = ptrScore.BufferPointer();
            vector<pair<int, ElemType>> tPairs;
            for (int i = 0; i < ptrScore.GetNumElements(); i++)
            {
                tPairs.push_back(make_pair(i, pPointer[i]));
                //                    assert(pPointer[i] <= 1.0); /// work on the posterior probabilty, so every score should be smaller than 1.0
            }

            std::sort(tPairs.begin(), tPairs.end(), comparator<ElemType>);

            bool bAboveThreshold = false;
            for (typename vector<pair<int, ElemType>>::iterator itr = tPairs.begin(); itr != tPairs.end(); itr++)
            {
                if (itr->second < 0.0)
                    LogicError("This means to use probability so the value should be non-negative");

                ElemType dScore = (itr->second > (ElemType)EPS_IN_LOG) ? log(itr->second) : (ElemType)LOG_OF_EPS_IN_LOG;

                dScore += preScore;
                if (dScore >= threshold && dScore >= best_score_so_far)
                {
                    rCandidate.push_back(make_pair(itr->first, dScore));
                    bAboveThreshold = true;
                }
                else
                {
                    break;
                }
            }

            return bAboveThreshold;
        }

        // retrieve activity at time atTime. 
        // notice that the function values returned is single column 
        void PreComputeActivityAtTime(size_t atTime)
        {
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
            {
                ComputationNodePtr node = *nodeIter;
                node->EvaluateThisNode(atTime);
                if (node->FunctionValues().GetNumCols() != node->GetNbrSlicesInEachRecurrentIteration())
                {
                    RuntimeError("preComputeActivityAtTime: the function values has to be a single column matrix ");
                }
            }
        }

        //return true if precomputation is executed.
        void ResetPreCompute()
        {
            //mark false
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
            {
                BatchModeNode<ElemType>* node = static_cast<BatchModeNode<ElemType>*> (*nodeIter);
                node->MarkComputed(false);
            }
        }

        //return true if precomputation is executed.
        bool PreCompute(ComputationNetwork<ElemType>& net,
            std::vector<ComputationNodePtr>& FeatureNodes)
        {
            batchComputeNodes = net.GetNodesRequireBatchMode();

            if (batchComputeNodes.size() == 0)
            {
                return false;
            }

            UpdateEvalTimeStamps(FeatureNodes);

            size_t actualMBSize = net.GetActualMBSize();
            net.SetActualMiniBatchSize(actualMBSize);
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
            {
                net.Evaluate(*nodeIter);
            }

            //mark done
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
            {
                BatchModeNode<ElemType>* node = static_cast<BatchModeNode<ElemType>*> (*nodeIter);
                node->MarkComputed(true);
            }

            return true;
        }

        void WriteNbest(const size_t nidx, const vector<size_t> &best_path,
            std::vector<ComputationNodePtr>& outputNodes, IDataWriter<ElemType>& dataWriter)
        {
            assert(outputNodes.size() == 1);
            std::map<std::wstring, void *, nocase_compare> outputMatrices;
            size_t bSize = best_path.size();
            for (int i = 0; i < outputNodes.size(); i++)
            {
                size_t dim = outputNodes[i]->FunctionValues().GetNumRows();
                outputNodes[i]->FunctionValues().Resize(dim, bSize);
                outputNodes[i]->FunctionValues().SetValue(0);
                for (int k = 0; k < bSize; k++)
                    outputNodes[i]->FunctionValues().SetValue(best_path[k], k, 1.0);
                outputMatrices[outputNodes[i]->NodeName()] = (void *)(&outputNodes[i]->FunctionValues());
            }

            dataWriter.SaveData(nidx, outputMatrices, bSize, bSize, 0);
        }

        void BeamSearch(IDataReader<ElemType>& dataReader, IDataWriter<ElemType>& dataWriter, const vector<wstring>& outputNodeNames, const vector<wstring>& writeNodeNames, const size_t mbSize, const ElemType beam, const size_t testSize)
        {
            clock_t startReadMBTime = 0, endComputeMBTime = 0;

            //specify output nodes and files
            std::vector<ComputationNodePtr> outputNodes;
            for (int i = 0; i<outputNodeNames.size(); i++)
                outputNodes.push_back(m_net.GetNodeFromName(outputNodeNames[i]));

            //specify nodes to write to file
            std::vector<ComputationNodePtr> writeNodes;
            for (int i = 0; i<writeNodeNames.size(); i++)
                writeNodes.push_back(m_net.GetNodeFromName(writeNodeNames[i]));

            //prepare features and labels
            std::vector<ComputationNodePtr> & FeatureNodes = m_net.FeatureNodes();
            std::vector<ComputationNodePtr> & labelNodes = m_net.LabelNodes();

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i = 0; i<FeatureNodes.size(); i++)
            {
                inputMatrices[FeatureNodes[i]->NodeName()] = &FeatureNodes[i]->FunctionValues();
            }
            for (size_t i = 0; i<labelNodes.size(); i++)
            {
                inputMatrices[labelNodes[i]->NodeName()] = &labelNodes[i]->FunctionValues();
            }

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t actualMBSize = 0;

            dataReader.StartMinibatchLoop(mbSize, 0, testSize);
            dataReader.SetNbrSlicesEachRecurrentIter(1);

            startReadMBTime = clock();
            size_t numMBsRun = 0;
            ElemType ComputeTimeInMBs = 0;
            while (dataReader.GetMinibatch(inputMatrices))
            {
                UpdateEvalTimeStamps(FeatureNodes);

                actualMBSize = m_net.GetActualMBSize();
                m_net.SetActualMiniBatchSize(actualMBSize);

                vector<size_t> best_path;

                FindBestPath(m_net, dataReader, dataWriter, outputNodes, writeNodes, FeatureNodes, beam, inputMatrices, best_path);

                totalEpochSamples += actualMBSize;

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                dataReader.DataEnd(endDataSentence);

                endComputeMBTime = clock();
                numMBsRun++;

                if (m_traceLevel > 0)
                {
                    ElemType MBComputeTime = (ElemType)(endComputeMBTime - startReadMBTime) / CLOCKS_PER_SEC;

                    ComputeTimeInMBs += MBComputeTime;

                    fprintf(stderr, "Sentenes Seen = %d; Samples seen = %d; Total Compute Time = %.8g ; Time Per Sample=%.8g\n", numMBsRun, totalEpochSamples, ComputeTimeInMBs, ComputeTimeInMBs / totalEpochSamples);
                }

                startReadMBTime = clock();
            }

            fprintf(stderr, "done decoding\n");
        }

        void FindBestPath(ComputationNetwork<ElemType>& evalnet,
            IDataReader<ElemType>& dataReader, IDataWriter<ElemType>& dataWriter,
            std::vector<ComputationNodePtr>& evalNodes,
            std::vector<ComputationNodePtr>& outputNodes,
            std::vector<ComputationNodePtr> & FeatureNodes,
            const ElemType beam, 
            std::map<std::wstring, Matrix<ElemType>*> & inputMatrices,
            vector<size_t> &best_path)
        {
            assert(evalNodes.size() == 1);

            NN_state<ElemType> state;
            NN_state<ElemType> null_state;

            priority_queue<Token<ElemType>> n_bests;  /// save n-bests

            /**
            loop over all the candidates for the featureDelayTarget,
            evaluate their scores, save their histories
            */
            priority_queue<Token<ElemType>> from_queue, to_queue;
            vector<ElemType> evalResults;

            size_t mbSize;
            mbSize = evalnet.GetActualMBSize();
            size_t maxMbSize = 2 * mbSize;

            /// use reader to initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            evalnet.SetActualMiniBatchSize(mbSize);
            evalnet.SetActualNbrSlicesInEachRecIter(dataReader.NumberSlicesInEachRecurrentIter());
            dataReader.SetSentenceSegBatch(evalnet.SentenceBoundary(), evalnet.MinibatchPackingFlags());

            clock_t start, now;
            start = clock();

            /// for the case of not using encoding, no previous state is avaliable, except for the default hidden layer activities 
            /// no need to get that history and later to set the history as there are default hidden layer activities

            from_queue.push(Token<ElemType>(0., vector<size_t>(), state)); /// the first element in the priority queue saves the initial NN state

            dataReader.InitProposals(inputMatrices);
            size_t itdx = 0;
            size_t maxSize = min(maxMbSize, mbSize);

            ResetPreCompute();
            PreCompute(evalnet, FeatureNodes);

            /// need to set the minibatch size to 1, and initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            evalnet.SetActualMiniBatchSize(1);
            dataReader.SetSentenceSegBatch(evalnet.SentenceBoundary(), evalnet.MinibatchPackingFlags());
            /// need to set the sentence begining segmentation info
            evalnet.SentenceBoundary().SetValue(SENTENCE_BEGIN);

            for (itdx = 0; itdx < maxSize; itdx++)
            {
                ElemType best_score = -numeric_limits<ElemType>::infinity();
                vector<size_t> best_output_label;

                if (itdx > 0)
                {
                    /// state need to be carried over from past time instance
                    evalnet.SentenceBoundary().SetValue(SENTENCE_MIDDLE);
                }

                PreComputeActivityAtTime(itdx);

                while (!from_queue.empty()) {
                    const Token<ElemType> from_token = from_queue.top();
                    vector<size_t> history = from_token.sequence;

                    /// update feature nodes once, as the observation is the same for all propsoals in labels
                    UpdateEvalTimeStamps(FeatureNodes);

                    /// history is updated in the getproposalobs function
                    dataReader.GetProposalObs(inputMatrices, itdx, history);

                    /// get the nn state history and set nn state to the history
                    map<wstring, Matrix<ElemType>> hidden_history = from_token.state.hidden_activity;
                    evalnet.SetHistory(hidden_history);

                    for (int i = 0; i<evalNodes.size(); i++)
                    {
                        evalnet.Evaluate(evalNodes[i]);
                        vector<pair<int, ElemType>> retPair;
                        if (GetCandidatesAtOneTimeInstance(evalNodes[i]->FunctionValues(), from_token.score, best_score - beam, -numeric_limits<ElemType>::infinity(), retPair)
                            == false)
                            continue;

                        evalnet.GetHistory(state.hidden_activity, true);
                        for (typename vector<pair<int, ElemType>>::iterator itr = retPair.begin(); itr != retPair.end(); itr++)
                        {
                            vector<size_t> history = from_token.sequence;
                            history.push_back(itr->first);
                            Token<ElemType> to_token(itr->second, history, state);  /// save updated nn state and history

                            to_queue.push(to_token);

                            if (itr->second > best_score)  /// update best score
                            {
                                best_score = itr->second;
                                best_output_label = history;
                            }
                        }

                        history = from_token.sequence;  /// back to the from token's history
                    }

                    from_queue.pop();
                }

                if (to_queue.size() == 0)
                    break;

                // beam pruning
                const ElemType threshold = best_score - beam;
                while (!to_queue.empty())
                {
                    if (to_queue.top().score >= threshold)
                        from_queue.push(to_queue.top());
                    to_queue.pop();
                }
            }

            // write back best path
            size_t ibest = 0;
            while (from_queue.size() > 0)
            {
                Token<ElemType> seq(from_queue.top().score, from_queue.top().sequence, from_queue.top().state); 

                best_path.clear();

                assert(best_path.empty());
                best_path = seq.sequence;
                if (ibest == 0)
                    WriteNbest(ibest, best_path, outputNodes, dataWriter);

#ifdef DBG_BEAM_SEARCH
                WriteNbest(ibest, best_path, outputNodes, dataWriter);
                cout << " score = " << from_queue.top().score << endl;
#endif

                from_queue.pop();

                ibest++;
            }

            now = clock();
            fprintf(stderr, "%.1f words per second\n", mbSize / ((double)(now - start) / 1000.0));
        }

        /**
            beam search decoder
        */
        ElemType FindBestPathWithVariableLength(ComputationNetwork<ElemType>& evalnet,
            size_t inputLength,
            IDataReader<ElemType>& dataReader, IDataWriter<ElemType>& dataWriter,
            std::vector<ComputationNodePtr>& evalNodes,
            std::vector<ComputationNodePtr>& outputNodes,
            std::vector<ComputationNodePtr> & FeatureNodes,
            const ElemType beam, 
            std::map<std::wstring, Matrix<ElemType>*> & inputMatrices,
            vector<size_t> &best_path)
        {
            assert(evalNodes.size() == 1);

            NN_state<ElemType> state;
            NN_state<ElemType> null_state;

            std::priority_queue<Token<ElemType>> n_bests;  /// save n-bests

            /**
            loop over all the candidates for the featuredelayTarget,
            evaluate their scores, save their histories
            */
            std::priority_queue<Token<ElemType>> from_queue, to_queue;
            std::priority_queue<Token<ElemType>> result_queue;
            vector<ElemType> evalResults;

            size_t mbSize = inputLength;
            size_t maxMbSize = 3 * mbSize;

            /// use reader to initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            evalnet.SetActualMiniBatchSize(mbSize);
            evalnet.SetActualNbrSlicesInEachRecIter(dataReader.NumberSlicesInEachRecurrentIter());

            clock_t start, now;
            start = clock();

            from_queue.push(Token<ElemType>(0., vector<size_t>(), state)); /// the first element in the priority queue saves the initial NN state

            /// the end of sentence symbol in reader
            int outputEOS = dataReader.GetSentenceEndIdFromOutputLabel();
            if (outputEOS < 0)
                LogicError("Cannot find end of sentence symbol. Check ");

            dataReader.InitProposals(inputMatrices);

            size_t itdx = 0;

            ResetPreCompute();
            PreCompute(evalnet, FeatureNodes);

            /// need to set the minibatch size to 1, and initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            evalnet.SetActualMiniBatchSize(dataReader.NumberSlicesInEachRecurrentIter());

            ElemType best_score = -numeric_limits<ElemType>::infinity();
            ElemType best_score_so_far = -numeric_limits<ElemType>::infinity();
            for (itdx = 0; itdx < maxMbSize; itdx++)
            {
                best_score = -numeric_limits<ElemType>::infinity();
                vector<size_t> best_output_label;

                PreComputeActivityAtTime(itdx);

                while (!from_queue.empty()) {
                    const Token<ElemType> from_token = from_queue.top();
                    vector<size_t> history = from_token.sequence;

                    /// update feature nodes once, as the observation is the same for all propsoals in labels
                    UpdateEvalTimeStamps(FeatureNodes);

                    /// history is updated in the getproposalobs function
                    dataReader.GetProposalObs(inputMatrices, itdx, history);

                    /// get the nn state history and set nn state to the history
                    map<wstring, Matrix<ElemType>> hidden_history = from_token.state.hidden_activity;
                    evalnet.SetHistory(hidden_history);

                    for (int i = 0; i<evalNodes.size(); i++)
                    {
                        evalnet.Evaluate(evalNodes[i]);
                        vector<pair<int, ElemType>> retPair;
                        if (GetCandidatesAtOneTimeInstance(evalNodes[i]->FunctionValues(), from_token.score, best_score - beam, -numeric_limits<ElemType>::infinity(), retPair)
                            == false)
                            continue;

                        evalnet.GetHistory(state.hidden_activity, true);
                        for (typename vector<pair<int, ElemType>>::iterator itr = retPair.begin(); itr != retPair.end(); itr++)
                        {
                            vector<size_t> history = from_token.sequence;
                            history.push_back(itr->first);

                            if (itr->first != outputEOS)
                            {
                                Token<ElemType> to_token(itr->second, history, state);  /// save updated nn state and history

                                to_queue.push(to_token);

                                if (itr->second > best_score)  /// update best score
                                {
                                    best_score = itr->second;
                                    best_output_label = history;
                                }
                            }
                            else {
                                /// sentence ending reached
                                Token<ElemType> to_token(itr->second, history, state);  
                                result_queue.push(to_token);
                            }
                        }

                        history = from_token.sequence;  /// back to the from token's history
                    }

                    from_queue.pop();
                }

                if (to_queue.size() == 0)
                    break;

                // beam pruning
                const ElemType threshold = best_score - beam;
                while (!to_queue.empty())
                {
                    if (to_queue.top().score >= threshold)
                        from_queue.push(to_queue.top());
                    to_queue.pop();
                }

                best_score_so_far = best_score;
            }

            // write back best path
            size_t ibest = 0;
            while (result_queue.size() > 0)
            {
                best_path.clear();
                //vector<size_t> *p = &result_queue.top().sequence;
                assert(best_path.empty());
                best_path.swap(const_cast<vector<size_t>&>(result_queue.top().sequence));
                {
                    ElemType score = result_queue.top().score;
                    best_score = score;
                    fprintf(stderr, "best[%d] score = %.4e\t", ibest, score);
                    if (best_path.size() > 0)
                        WriteNbest(ibest, best_path, outputNodes, dataWriter);
                }

                ibest++;

                result_queue.pop();
                break; /// only output the top one
            }

            now = clock();
            fprintf(stderr, "%.1f words per second\n", mbSize / ((double)(now - start) / 1000.0));

            return (ElemType) best_score;
        }

    };

}}}
