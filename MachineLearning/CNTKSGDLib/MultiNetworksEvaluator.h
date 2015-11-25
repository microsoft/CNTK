// MultiNetworksEvaluator/SGD -- This represents earlier efforts to use CNTK for sequence-to-sequence modeling. This is no longer the intended design.
//
// <copyright file="MultiNetworksEvaluator.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Helpers.h"    // for foreach_column() macro
#include "fileutil.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "ComputationNetwork.h"
#include "DataReaderHelpers.h"
#include "SimpleEvaluator.h"
#include "TrainingCriterionNodes.h" // TODO: we should move the functions that depend on these to the .cpp
#include "CompositeComputationNodes.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <queue>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    struct NN_state
    {
        map<wstring, Matrix<ElemType>> hidden_activity;
    };

    template<class ElemType>
    struct Token
    {
        Token(const double score, const std::vector<size_t> &sequence, const NN_state<ElemType> & state) :
            score(score), sequence(sequence), state(state)
        { }
        bool operator<(const Token<ElemType> &t) const
        {
            return score < t.score;
        }
        double score;
        vector<size_t> sequence;
        NN_state<ElemType> state;
    };

    template<class ElemType>
    class MultiNetworksEvaluator : public SimpleEvaluator<ElemType>
    {
        typedef SimpleEvaluator<ElemType> Base; using Base::m_net; using Base::m_numMBsToShowResult; using Base::m_traceLevel; using Base::DisplayEvalStatistics;
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
        typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;
    public:
        MultiNetworksEvaluator(ComputationNetworkPtr net, const size_t numMBsToShowResult = 100, const int traceLevel = 0) : Base(net, numMBsToShowResult, traceLevel) { }
        
        //returns error rate
        // This was a special early implementation of RNNs by emulating them as a DNN.
        // The code is very restricted to simple RNNs. 
        // The idea can be used for more complicated network but need to know which nodes are stateful or time-dependent so that unroll is done in a correct way to represent recurrent networks. 
        // TODO: can probably be removed.
        double EvaluateUnroll(IDataReader<ElemType>* dataReader, const size_t mbSize, double &evalSetCrossEntropy, const wchar_t* output = nullptr, const size_t testSize = requestDataSize)
        {
            std::vector<ComputationNodeBasePtr> & featureNodes = m_net->FeatureNodes();
            std::vector<ComputationNodeBasePtr> & labelNodes = m_net->LabelNodes();
            std::vector<ComputationNodeBasePtr> & criterionNodes = m_net->FinalCriterionNodes();
            std::vector<ComputationNodeBasePtr> & evaluationNodes = m_net->EvaluationNodes();

            if (criterionNodes.size() == 0)
                RuntimeError("No CrossEntropyWithSoftmax node found\n");
            if (evaluationNodes.size() == 0)
                RuntimeError("No Evaluation node found\n");

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i = 0; i < featureNodes.size(); i++)
                inputMatrices[featureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(featureNodes[i])->FunctionValues();
            for (size_t i = 0; i < labelNodes.size(); i++)
                inputMatrices[labelNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(labelNodes[i])->FunctionValues();
            inputMatrices[L"numberobs"] = new Matrix<ElemType>(1, 1, m_net->GetDeviceId());

            dataReader->StartMinibatchLoop(mbSize, 0, testSize);
            m_net->StartEvaluateMinibatchLoop(criterionNodes, evaluationNodes);

            double epochEvalError = 0;
            double epochCrossEntropy = 0;
            size_t totalEpochSamples = 0;
            double prevEpochEvalError = 0;
            double prevEpochCrossEntropy = 0;
            size_t prevTotalEpochSamples = 0;
            size_t prevStart = 1;
            size_t numSamples = 0;
            double crossEntropy = 0;
            double evalError = 0;

            ofstream outputStream;
            if (output)
            {
#ifdef _MSC_VER
                outputStream.open(output);
#else
                outputStream.open(wtocharpath(output).c_str());    // GCC does not implement wide-char pathnames here
#endif
            }

            size_t numMBsRun = 0;
            size_t actualMBSize = 0;
            while (dataReader->GetMinibatch(inputMatrices))
            {
                // TODO: we should use GetMinibatchIntoNetwork(), but it seems tricky. What is this for?
                size_t nbrSamples = (size_t)(*inputMatrices[L"numberobs"])(0, 0);
                actualMBSize = nbrSamples;

                for (int npos = 0; npos < nbrSamples; npos++)
                {
                    featureNodes[npos]->UpdateEvalTimeStamp();
                    labelNodes[npos]->UpdateEvalTimeStamp();

                    m_net->Evaluate(criterionNodes[npos]); //use only the first criterion. Is there any possibility to use more?

                    m_net->Evaluate(evaluationNodes[npos]);

                    double mbCrossEntropy = (double)criterionNodes[npos]->Get00Element(); // criterionNode should be a scalar
                    epochCrossEntropy += mbCrossEntropy;

                    double mbEvalError = (double)evaluationNodes[npos]->Get00Element(); //criterionNode should be a scalar

                    epochEvalError += mbEvalError;
                }

                totalEpochSamples += actualMBSize;

                if (outputStream.is_open())
                {
                    //TODO: add support to dump multiple outputs
                    ComputationNodePtr outputNode = dynamic_pointer_cast<ComputationNode<ElemType>>(m_net->OutputNodes()[0]);
                    foreach_column(j, outputNode->FunctionValues())
                    {
                        foreach_row(i, outputNode->FunctionValues())
                            outputStream << outputNode->FunctionValues()(i, j) << " ";
                        outputStream << endl;
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
            epochEvalError /= (double)totalEpochSamples;
            epochCrossEntropy /= (double)totalEpochSamples;
            fprintf(stderr, "Overall: Samples Evaluated = %lu   EvalErr Per Sample = %.8g   Loss Per Sample = %.8g\n", totalEpochSamples, epochEvalError, epochCrossEntropy);
            if (outputStream.is_open())
            {
                outputStream.close();
            }
            evalSetCrossEntropy = epochCrossEntropy;
            return epochEvalError;
        }

    public:
        /// for encoder-decoder RNN
        list<pair<wstring, wstring>> m_lst_pair_encoder_decode_node_names;
        list<pair<ComputationNodeBasePtr, ComputationNodeBasePtr>> m_lst_pair_encoder_decoder_nodes;

        void SetEncoderDecoderNodePairs(std::list<pair<ComputationNodeBasePtr, ComputationNodeBasePtr>>& lst_pair_encoder_decoder_nodes)
        {
            m_lst_pair_encoder_decoder_nodes.clear();
            for (typename std::list<pair<ComputationNodeBasePtr, ComputationNodeBasePtr>>::iterator iter = lst_pair_encoder_decoder_nodes.begin(); iter != lst_pair_encoder_decoder_nodes.end(); iter++)
                m_lst_pair_encoder_decoder_nodes.push_back(*iter);
        }

        /**
        this evaluates encoder network and decoder framework
        only beam search decoding is applied to the last network
        */
        double EvaluateEncoderDecoderWithHiddenStates(
            vector<ComputationNetworkPtr> nets,
            vector<IDataReader<ElemType>*> dataReaders,
            const size_t mbSize,
            const size_t testSize = requestDataSize)
        {
            size_t iNumNets = nets.size();

            ComputationNetworkPtr decoderNet = nullptr;
            IDataReader<ElemType>* decoderDataReader = dataReaders[iNumNets - 1];
            decoderNet = nets[iNumNets - 1];

            const auto & decoderEvaluationNodes = decoderNet->EvaluationNodes();

            double evalResults = 0;

            vector<std::map<std::wstring, Matrix<ElemType>*>*> inputMatrices;
            for (auto ptr = nets.begin(); ptr != nets.end(); ptr++)
            {
                const auto & featNodes = (*ptr)->FeatureNodes();
                const auto & lablPtr = (*ptr)->LabelNodes();
                map<wstring, Matrix<ElemType>*>* pMap = new map<wstring, Matrix<ElemType>*>();
                for (auto pf = featNodes.begin(); pf != featNodes.end(); pf++)
                {
                    (*pMap)[(*pf)->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(*pf)->FunctionValues();
                }
                for (auto pl = lablPtr.begin(); pl != lablPtr.end(); pl++)
                {
                    (*pMap)[(*pl)->NodeName()] = &(dynamic_pointer_cast<ComputationNode<ElemType>>(*pl)->FunctionValues());
                }
                inputMatrices.push_back(pMap);
            }

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t numMBsRun = 0;
            size_t actualMBSize = 0;
            size_t numSamplesLastMBs = 0;
            size_t lastMBsRun = 0; //MBs run before this display

            double evalResultsLastMBs = (double)0;

            for (auto ptr = dataReaders.begin(); ptr != dataReaders.end(); ptr++)
                (*ptr)->StartMinibatchLoop(mbSize, 0, testSize);
            // BUGBUG: Code below will fail because we now must call StartMinibatchLoop(), but I can't tell from below which nodes to call it for.
            //for (auto & ptr : nets)
            //    ptr->StartMinibatchLoop(xxx);

            bool bContinueDecoding = true;
            while (bContinueDecoding)
            {

                /// load data
                auto pmat = inputMatrices.begin();
                bool bNoMoreData = false;
                for (auto ptr = dataReaders.begin(); ptr != dataReaders.end(); ptr++, pmat++)
                {
                    if ((*ptr)->GetMinibatch(*(*pmat)) == false)
                    {
                        bNoMoreData = true;
                        break;
                    }
                }
                if (bNoMoreData)
                    break;

                for (auto ptr = nets.begin(); ptr != nets.end(); ptr++)
                {
                    const auto & featNodes = (*ptr)->FeatureNodes();
                    ComputationNetwork::UpdateEvalTimeStamps(featNodes);
                }

                auto preader = dataReaders.begin();
                for (auto ptr = nets.begin(); ptr != nets.end(); ptr++, preader++)
                {
                    actualMBSize = (*ptr)->DetermineActualMBSizeFromFeatures();
                    if (actualMBSize == 0)
                        LogicError("decoderTrainSetDataReader read data but encoderNet reports no data read");
                    (*preader)->CopyMBLayoutTo((*ptr)->GetMBLayoutPtr());
                    (*ptr)->VerifyActualNumParallelSequences((*preader)->GetNumParallelSequences());

                    const auto & pairs = (*ptr)->PairNodes();
                    for (auto ptr2 = pairs.begin(); ptr2 != pairs.end(); ptr2++)
                        (*ptr)->Evaluate(*ptr2);
                }

                decoderNet = nets[iNumNets - 1];
                /// not the sentence begining, because the initial hidden layer activity is from the encoder network
                actualMBSize = decoderNet->DetermineActualMBSizeFromFeatures();
                if (actualMBSize == 0)
                    LogicError("decoderTrainSetDataReader read data but decoderNet reports no data read");
                decoderDataReader->CopyMBLayoutTo(decoderNet->GetMBLayoutPtr());
                decoderNet->VerifyActualNumParallelSequences(decoderDataReader->GetNumParallelSequences());

                size_t i = 0;
                assert(decoderEvaluationNodes.size() == 1);
                if (decoderEvaluationNodes.size() != 1)
                {
                    LogicError("Decoder should have only one evaluation node");
                }

                for (auto ptr = decoderEvaluationNodes.begin(); ptr != decoderEvaluationNodes.end(); ptr++, i++)
                {
                    decoderNet->Evaluate(*ptr);
                    if ((*ptr)->GetNumRows() != 1 || (*ptr)->GetNumCols() != 1)
                        LogicError("EvaluateEncoderDecoderWithHiddenStates: decoder evaluation should return a scalar value");

                    evalResults += (double)(*ptr)->Get00Element();
                }

                totalEpochSamples += actualMBSize;
                numMBsRun++;

                if (m_traceLevel > 0)
                {
                    numSamplesLastMBs += actualMBSize;

                    if (numMBsRun % m_numMBsToShowResult == 0)
                    {
                        DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, decoderEvaluationNodes, evalResults, evalResultsLastMBs);

                        evalResultsLastMBs = evalResults;

                        numSamplesLastMBs = 0;
                        lastMBsRun = numMBsRun;
                    }
                }

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                for (auto ptr = dataReaders.begin(); ptr != dataReaders.end(); ptr++)
                {
                    (*ptr)->DataEnd(endDataSentence);
                }
            }

            // show last batch of results
            if (m_traceLevel > 0 && numSamplesLastMBs > 0)
            {
                DisplayEvalStatistics(lastMBsRun + 1, numMBsRun, numSamplesLastMBs, decoderEvaluationNodes, evalResults, evalResultsLastMBs);
            }

            //final statistics
            evalResultsLastMBs = 0;

            fprintf(stderr, "Final Results: ");
            DisplayEvalStatistics(1, numMBsRun, totalEpochSamples, decoderEvaluationNodes, evalResults, evalResultsLastMBs, true);

            evalResults /= totalEpochSamples;

            for (auto ptr = inputMatrices.begin(); ptr != inputMatrices.end(); ptr++)
            {
                delete *ptr;
            }

            return evalResults;
        }

        // TODO: This stuff must all be removed from SimpleEvaluator, as this is not simple at all!!
        void InitTrainEncoderDecoderWithHiddenStates(const ConfigParameters& readerConfig)
        {
            ConfigArray arrEncoderNodeNames = readerConfig(L"encoderNodes", "");
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

            ConfigArray arrDecoderNodeNames = readerConfig(L"decoderNodes", "");
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
            vector<ComputationNetworkPtr> nets,
            vector<IDataReader<ElemType>*> readers,
            IDataWriter<ElemType>& dataWriter,
            const vector<wstring>& evalNodeNames,
            const vector<wstring>& writeNodeNames,
            const size_t mbSize, const double beam, const size_t testSize)
        {
            size_t iNumNets = nets.size();
            if (iNumNets < 2)
            {
                LogicError("Has to have at least two networks");
            }

            ComputationNetworkPtr decoderNet = nets[iNumNets - 1];
            IDataReader<ElemType>* encoderDataReader = readers[iNumNets - 2];
            IDataReader<ElemType>* decoderDataReader = readers[iNumNets - 1];
            vector<ComputationNodeBasePtr> & decoderFeatureNodes = decoderNet->FeatureNodes();

            //specify output nodes and files
            std::vector<ComputationNodeBasePtr> outputNodes;
            for (auto ptr = evalNodeNames.begin(); ptr != evalNodeNames.end(); ptr++)
                outputNodes.push_back(decoderNet->GetNodeFromName(*ptr));

            //specify nodes to write to file
            std::vector<ComputationNodeBasePtr> writeNodes;
            for (int i = 0; i < writeNodeNames.size(); i++)
                writeNodes.push_back(m_net->GetNodeFromName(writeNodeNames[i]));

            //prepare features and labels
            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            std::map<std::wstring, Matrix<ElemType>*> decoderInputMatrices;
            for (auto ptr = nets.begin(); ptr != nets.end() - 1; ptr++)
            {
                const auto & featNodes = (*ptr)->FeatureNodes();
                for (auto ptr2 = featNodes.begin(); ptr2 != featNodes.end(); ptr2++)
                    inputMatrices[(*ptr2)->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(*ptr2)->FunctionValues();

                const auto & lablNodes = (*ptr)->LabelNodes();
                for (auto ptr2 = lablNodes.begin(); ptr2 != lablNodes.end(); ptr2++)
                    inputMatrices[(*ptr2)->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(*ptr2)->FunctionValues();
            }

            /// for the last network
            auto ptr = nets.end() - 1;
            const auto & featNodes = (*ptr)->FeatureNodes();
            for (auto ptr2 = featNodes.begin(); ptr2 != featNodes.end(); ptr2++)
                decoderInputMatrices[(*ptr2)->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(*ptr2)->FunctionValues();

            const auto & lablNodes = (*ptr)->LabelNodes();
            for (auto ptr2 = lablNodes.begin(); ptr2 != lablNodes.end(); ptr2++)
                decoderInputMatrices[(*ptr2)->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(*ptr2)->FunctionValues();

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t actualMBSize = 0;

            for (auto ptr = readers.begin(); ptr != readers.end(); ptr++)
            {
                (*ptr)->StartMinibatchLoop(mbSize, 0, testSize);
                (*ptr)->SetNumParallelSequences(1);
            }

            Matrix<ElemType> historyMat(m_net->GetDeviceId());

            bool bDecoding = true;
            while (bDecoding)
            {
                bool noMoreData = false;
                /// only get minibatch on the encoder parts of networks
                size_t k = 0;
                for (auto ptr = readers.begin(); ptr != readers.end() - 1; ptr++, k++)
                {
                    if ((*ptr)->GetMinibatch(inputMatrices) == false)
                    {
                        noMoreData = true;
                        break;
                    }
                }
                if (noMoreData)
                    break;

                for (auto ptr = nets.begin(); ptr != nets.end() - 1; ptr++)
                {
                    /// only on the encoder part of the networks
                    const auto & featNodes = (*ptr)->FeatureNodes();
                    ComputationNetwork::UpdateEvalTimeStamps(featNodes);
                }

                auto ptrreader = readers.begin();
                size_t mNutt = 0;
                for (auto ptr = nets.begin(); ptr != nets.end() - 1; ptr++, ptrreader++)
                {
                    /// evaluate on the encoder networks
                    actualMBSize = (*ptr)->DetermineActualMBSizeFromFeatures();

                    mNutt = (*ptrreader)->GetNumParallelSequences();
                    (*ptrreader)->CopyMBLayoutTo((*ptr)->GetMBLayoutPtr());
                    (*ptr)->VerifyActualNumParallelSequences(mNutt);

                    const auto & pairs = (*ptr)->PairNodes();
                    for (auto ptr2 = pairs.begin(); ptr2 != pairs.end(); ptr2++)
                        (*ptr)->Evaluate(*ptr2);
                }

                /// not the sentence begining, because the initial hidden layer activity is from the encoder network
                decoderNet->ResizeAllFeatureNodes(actualMBSize);
                //decoderNet->SetActualMiniBatchSizeFromFeatures();
                encoderDataReader->CopyMBLayoutTo(decoderNet->GetMBLayoutPtr());
                decoderNet->VerifyActualNumParallelSequences(mNutt);

                vector<size_t> best_path;
                FindBestPathWithVariableLength(decoderNet, actualMBSize, decoderDataReader, dataWriter, outputNodes, writeNodes, decoderFeatureNodes, beam, &decoderInputMatrices, best_path);

                totalEpochSamples += actualMBSize;

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                for (auto ptr = readers.begin(); ptr != readers.end(); ptr++)
                    (*ptr)->DataEnd(endDataSentence);
            }
        }

        template<class F>
        static inline bool comparator(const pair<int, F>& l, const pair<int, F>& r)
        {
            return l.second > r.second;
        }

        bool GetCandidatesAtOneTimeInstance(const Matrix<ElemType>& score,
                                            const double & preScore, const double & threshold,
                                            const double& best_score_so_far,
                                            vector<pair<int, double>>& rCandidate)
        {
            Matrix<ElemType> ptrScore(CPUDEVICE);
            ptrScore = score;

            ElemType *pPointer = ptrScore.BufferPointer();
            vector<pair<int, double>> tPairs;
            for (int i = 0; i < ptrScore.GetNumElements(); i++)
            {
                tPairs.push_back(make_pair(i, pPointer[i]));
                //                    assert(pPointer[i] <= 1.0); /// work on the posterior probabilty, so every score should be smaller than 1.0
            }

            std::sort(tPairs.begin(), tPairs.end(), comparator<double>);

            bool bAboveThreshold = false;
            for (typename vector<pair<int, double>>::iterator itr = tPairs.begin(); itr != tPairs.end(); itr++)
            {
                if (itr->second < 0.0)
                    LogicError("This means to use probability so the value should be non-negative");

                double dScore = (itr->second >(double)EPS_IN_LOG) ? log(itr->second) : (double)LOG_OF_EPS_IN_LOG;

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
                ComputationNodeBasePtr node = *nodeIter;
                node->EvaluateThisNode(FrameRange(node->GetMBLayout(), atTime));
                if (node->GetNumCols() != node->GetNumParallelSequences())
                    RuntimeError("preComputeActivityAtTime: the function values has to be a single column matrix ");
            }
        }

        // (only called by FindBestPath...())
        void ResetPreCompute()
        {
            //mark false
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
            {
                auto node = static_pointer_cast<BatchModeNode<ElemType>> (*nodeIter);
                node->MarkComputed(false);
            }
        }

        //return true if precomputation is executed.
        bool EvaluateBatchModeNodes(ComputationNetwork& net,
                        const std::vector<ComputationNodeBasePtr>& featureNodes)
        {
            batchComputeNodes = net.GetNodesRequiringBatchMode();

            if (batchComputeNodes.size() == 0)
            {
                return false;
            }

            ComputationNetwork::UpdateEvalTimeStamps(featureNodes);

            net.StartEvaluateMinibatchLoop(batchComputeNodes);  // TODO: Is this correct? There is no StartMinibatchLoop() for a reader.

            //net.SetActualMiniBatchSizeFromFeatures();
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
                net.Evaluate(*nodeIter);

            //mark done
            for (auto nodeIter = batchComputeNodes.begin(); nodeIter != batchComputeNodes.end(); nodeIter++)
            {
                auto node = static_pointer_cast<BatchModeNode<ElemType>> (*nodeIter);
                node->MarkComputed(true);
            }

            return true;
        }

        void WriteNbest(const size_t nidx, const vector<size_t> &best_path,
                        const std::vector<ComputationNodeBasePtr>& outputNodes, IDataWriter<ElemType>& dataWriter)
        {
            assert(outputNodes.size() == 1);
            std::map<std::wstring, void *, nocase_compare> outputMatrices;
            size_t bSize = best_path.size();
            for (int i = 0; i < outputNodes.size(); i++)
            {
                size_t dim = outputNodes[i]->GetNumRows();
                outputNodes[i]->SetDims(dim, bSize);
                dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->UpdateFunctionValuesSize();
                dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->FunctionValues().SetValue(0);
                for (int k = 0; k < bSize; k++)
                    dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->FunctionValues().SetValue(best_path[k], k, 1.0);
                outputMatrices[outputNodes[i]->NodeName()] = (void *)(&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->FunctionValues());
                // TODO: void* --really?
            }

            dataWriter.SaveData(nidx, outputMatrices, bSize, bSize, 0);
        }

        void BeamSearch(IDataReader<ElemType>* dataReader, IDataWriter<ElemType>& dataWriter, const vector<wstring>& outputNodeNames, const vector<wstring>& writeNodeNames, const size_t mbSize, const double beam, const size_t testSize)
        {
            clock_t startReadMBTime = 0, endComputeMBTime = 0;

            //specify output nodes and files
            std::vector<ComputationNodeBasePtr> outputNodes;
            for (int i = 0; i < outputNodeNames.size(); i++)
                outputNodes.push_back(m_net->GetNodeFromName(outputNodeNames[i]));

            //specify nodes to write to file
            std::vector<ComputationNodeBasePtr> writeNodes;
            for (int i = 0; i < writeNodeNames.size(); i++)
                writeNodes.push_back(m_net->GetNodeFromName(writeNodeNames[i]));

            //prepare features and labels
            /*const*/ auto & featureNodes = m_net->FeatureNodes();
            const auto & labelNodes = m_net->LabelNodes();

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i = 0; i < featureNodes.size(); i++)
                inputMatrices[featureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(featureNodes[i])->FunctionValues();
            for (size_t i = 0; i < labelNodes.size(); i++)
                inputMatrices[labelNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(labelNodes[i])->FunctionValues();

            //evaluate through minibatches
            size_t totalEpochSamples = 0;
            size_t actualMBSize = 0;

            dataReader->StartMinibatchLoop(mbSize, 0, testSize);
            dataReader->SetNumParallelSequences(1);

            startReadMBTime = clock();
            size_t numMBsRun = 0;
            double ComputeTimeInMBs = 0;
            while (DataReaderHelpers::GetMinibatchIntoNetwork(*dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
            {
                // note: GetMinibatchIntoNetwork() will also fetch the MBLayout although we don't need ithere. This should not hurt.
                ComputationNetwork::UpdateEvalTimeStamps(featureNodes);
                //actualMBSize = m_net->SetActualMiniBatchSizeFromFeatures();

                vector<size_t> best_path;

                FindBestPath(m_net, dataReader,
                             dataWriter, outputNodes,
                             writeNodes, featureNodes,
                             beam, &inputMatrices, best_path);

                totalEpochSamples += actualMBSize;

                /// call DataEnd to check if end of sentence is reached
                /// datareader will do its necessary/specific process for sentence ending 
                dataReader->DataEnd(endDataSentence);

                endComputeMBTime = clock();
                numMBsRun++;

                if (m_traceLevel > 0)
                {
                    double MBComputeTime = (double)(endComputeMBTime - startReadMBTime) / CLOCKS_PER_SEC;

                    ComputeTimeInMBs += MBComputeTime;

                    fprintf(stderr, "Sentences Seen = %zd; Samples seen = %zd; Total Compute Time = %.8g ; Time Per Sample=%.8g\n", numMBsRun, totalEpochSamples, ComputeTimeInMBs, ComputeTimeInMBs / totalEpochSamples);
                }

                startReadMBTime = clock();
            }

            fprintf(stderr, "done decoding\n");
        }

        void FindBestPath(ComputationNetworkPtr evalnet,
                          IDataReader<ElemType>* dataReader, IDataWriter<ElemType>& dataWriter,
                          const std::vector<ComputationNodeBasePtr>& evalNodes,
                          const std::vector<ComputationNodeBasePtr>& outputNodes,
                          /*const*/ std::vector<ComputationNodeBasePtr>& featureNodes,
                          const double beam,
                          std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
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
            vector<double> evalResults;


            /// use reader to initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            size_t mbSize = evalnet->DetermineActualMBSizeFromFeatures();
            dataReader->CopyMBLayoutTo(evalnet->GetMBLayoutPtr());
            evalnet->VerifyActualNumParallelSequences(dataReader->GetNumParallelSequences());

            size_t maxMbSize = 2 * mbSize;

            clock_t start, now;
            start = clock();

            /// for the case of not using encoding, no previous state is avaliable, except for the default hidden layer activities 
            /// no need to get that history and later to set the history as there are default hidden layer activities

            from_queue.push(Token<ElemType>(0., vector<size_t>(), state)); /// the first element in the priority queue saves the initial NN state

            dataReader->InitProposals(inputMatrices);
            size_t itdx = 0;
            size_t maxSize = min(maxMbSize, mbSize);

            ResetPreCompute();
            EvaluateBatchModeNodes(*evalnet, featureNodes);

            /// need to set the minibatch size to 1, and initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            for (auto ptr = featureNodes.begin(); ptr != featureNodes.end(); ptr++)
            {
                size_t nr = (*ptr)->GetNumRows();
                (*ptr)->SetDims(nr, 1);
            }
            // TODO: ^^ this is the same as ResizeAllFeatureNodes() if featureNodes == evalnet.FeatureNodes(). Is it?
            //evalnet->SetActualMiniBatchSizeFromFeatures();

            dataReader->CopyMBLayoutTo(evalnet->GetMBLayoutPtr());  // TODO: should this be one column only?
            /// need to set the sentence begining segmentation info
            evalnet->GetMBLayoutPtr()->GetM().SetValue(((int) MinibatchPackingFlags::SequenceStart));

            for (itdx = 0; itdx < maxSize; itdx++)
            {
                double best_score = -numeric_limits<double>::infinity();
                vector<size_t> best_output_label;

                if (itdx > 0)
                {
                    /// state need to be carried over from past time instance
                    evalnet->GetMBLayoutPtr()->GetM().SetValue(((int) MinibatchPackingFlags::None));
                }

                PreComputeActivityAtTime(itdx);

                while (!from_queue.empty()) {
                    const Token<ElemType> from_token = from_queue.top();
                    vector<size_t> history = from_token.sequence;

                    /// update feature nodes once, as the observation is the same for all propsoals in labels
                    ComputationNetwork::UpdateEvalTimeStamps(featureNodes);

                    /// history is updated in the getproposalobs function
                    dataReader->GetProposalObs(inputMatrices, itdx, history);

                    /// get the nn state history and set nn state to the history
                    map<wstring, Matrix<ElemType>> hidden_history = from_token.state.hidden_activity;
                    evalnet->SetHistory(hidden_history);

                    for (int i = 0; i < evalNodes.size(); i++)
                    {
                        evalnet->Evaluate(evalNodes[i]);
                        vector<pair<int, double>> retPair;
                        if (GetCandidatesAtOneTimeInstance(dynamic_pointer_cast<ComputationNode<ElemType>>(evalNodes[i])->FunctionValues(), from_token.score, best_score - beam, -numeric_limits<double>::infinity(), retPair)
                            == false)
                            continue;

                        evalnet->GetHistory(state.hidden_activity, true);
                        for (typename vector<pair<int, double>>::iterator itr = retPair.begin(); itr != retPair.end(); itr++)
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
                const double threshold = best_score - beam;
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
        double FindBestPathWithVariableLength(ComputationNetworkPtr evalnet,
            size_t inputLength,
            IDataReader<ElemType>* dataReader,
            IDataWriter<ElemType>& dataWriter,
            std::vector<ComputationNodeBasePtr>& evalNodes,
            std::vector<ComputationNodeBasePtr>& outputNodes,
            std::vector<ComputationNodeBasePtr>& featureNodes,
            const double beam,
            std::map<std::wstring, Matrix<ElemType>*> * inputMatrices,
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
            vector<double> evalResults;

            size_t mbSize = inputLength;
            /// use reader to initialize evalnet's sentence start information to let it know that this
            /// is the beginning of sentence
            evalnet->ResizeAllFeatureNodes(mbSize);
            //evalnet->SetActualMiniBatchSizeFromFeatures();
            // TODO: not setting MBLayout?
            evalnet->VerifyActualNumParallelSequences(dataReader->GetNumParallelSequences());
            // TODO: This is UNTESTED; if it fails, change ^^ this back to SetActual...()

            size_t maxMbSize = 3 * mbSize;
#ifdef _DEBUG
            maxMbSize = 2;
#endif

            clock_t start, now;
            start = clock();

            from_queue.push(Token<ElemType>(0., vector<size_t>(), state)); /// the first element in the priority queue saves the initial NN state

            /// the end of sentence symbol in reader
            int outputEOS = dataReader->GetSentenceEndIdFromOutputLabel();
            if (outputEOS < 0)
                LogicError("Cannot find end of sentence symbol. Check ");

            dataReader->InitProposals(inputMatrices);

            size_t itdx = 0;

            ResetPreCompute();
            EvaluateBatchModeNodes(*evalnet, featureNodes);

            /// need to set the minibatch size to 1, and initialize evalnet's sentence start information to let it know that this
            /// is the begining of sentence
            // BUGBUG: This is almost certainly wrong; slice != MB size
            //evalnet->SetActualMiniBatchSize(dataReader->GetNumParallelSequences());
            evalnet->ResizeAllFeatureNodes(1);
            //evalnet->SetActualMiniBatchSizeFromFeatures();

            double best_score = -numeric_limits<double>::infinity();
            double best_score_so_far = -numeric_limits<double>::infinity();

            evalnet->GetMBLayoutPtr()->GetM().SetValue(((int) MinibatchPackingFlags::SequenceStart));   // BUGBUG: huh? How can the entire batch be start frames?

            for (itdx = 0; itdx < maxMbSize; itdx++)
            {
                double best_score = -numeric_limits<double>::infinity();
                vector<size_t> best_output_label;

                if (itdx > 0)
                {
                    /// state need to be carried over from past time instance
                    evalnet->GetMBLayoutPtr()->GetM().SetValue(((int) MinibatchPackingFlags::None));
                }

                PreComputeActivityAtTime(itdx);

                while (!from_queue.empty())
                {
                    const Token<ElemType> from_token = from_queue.top();
                    vector<size_t> history = from_token.sequence;

                    /// update feature nodes once, as the observation is the same for all propsoals in labels
                    ComputationNetwork::UpdateEvalTimeStamps(featureNodes);

                    /// history is updated in the getproposalobs function
                    dataReader->GetProposalObs(inputMatrices, itdx, history);

                    /// get the nn state history and set nn state to the history
                    map<wstring, Matrix<ElemType>> hidden_history = from_token.state.hidden_activity;
                    evalnet->SetHistory(hidden_history);

                    for (int i = 0; i < evalNodes.size(); i++)
                    {
                        evalnet->Evaluate(evalNodes[i]);
                        vector<pair<int, double>> retPair;
                        if (GetCandidatesAtOneTimeInstance(dynamic_pointer_cast<ComputationNode<ElemType>>(evalNodes[i])->FunctionValues(),
                                                           from_token.score, best_score - beam, -numeric_limits<double>::infinity(), retPair)
                            == false)   // ==false??? !(.)?
                            continue;

                        evalnet->GetHistory(state.hidden_activity, true);
                        for (typename vector<pair<int, double>>::iterator itr = retPair.begin(); itr != retPair.end(); itr++)
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
                const double threshold = best_score - beam;
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
                    double score = result_queue.top().score;
                    best_score = score;
                    fprintf(stderr, "best[%zd] score = %.4e\t", ibest, score);
                    if (best_path.size() > 0)
                        WriteNbest(ibest, best_path, outputNodes, dataWriter);
                }

                ibest++;

                result_queue.pop();
                break; /// only output the top one
            }

            now = clock();
            fprintf(stderr, "%.1f words per second\n", mbSize / ((double)(now - start) / 1000.0));

            return best_score;
        }

    protected:
        /// used for backward directional nodes
        std::list<ComputationNodeBasePtr> batchComputeNodes;
    };

}}}
