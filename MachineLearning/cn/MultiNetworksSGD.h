//
// <copyright file="SGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "basetypes.h"
#include "ComputationNetwork.h"
#include "IComputationNetBuilder.h"
#include "ComputationNetworkHelper.h"
#include "SimpleEvaluator.h"
#include "DataReader.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "fileutil.h"
#include "commandArgUtil.h"
#include <chrono> 
#include <random>
#include "TimerUtility.h"
#include "SGD.h"

#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
extern int myRank;
extern int numProcs;

using namespace std;

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            template<class ElemType>
            class MultiNetworksSGD : SGD<ElemType>
            {
            public:
                /// for encoder and decoder nodes pairing
                wstring m_decoderModelPath;
                wstring m_encoderModelPath;

                list<pair<wstring, wstring>> m_lst_pair_encoder_decode_node_names;
                list<pair<ComputationNodePtr, ComputationNodePtr>> m_lst_pair_encoder_decoder_nodes;

            public:
                MultiNetworksSGD(const ConfigParameters& configSGD) : SGD(configSGD)
                {
                }

                ~MultiNetworksSGD()
                {
                }

                void InitTrainEncoderDecoderWithHiddenStates(const ConfigParameters& readerConfig)
                {

                    m_decoderModelPath = m_modelPath + L".decoder";
                    m_encoderModelPath = m_modelPath + L".encoder";

                    ConfigArray arrEncoderNodeNames = readerConfig("encoderNodes", "");
                    vector<wstring> encoderNodeNames;
                    m_lst_pair_encoder_decode_node_names.clear();

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
                        fprintf(stderr, "paired %ls <-> %ls\n", encoderNodeNames[i].c_str(), decoderNodeNames[i].c_str());
                    }
                }

                void EncoderDecoder(IComputationNetBuilder<ElemType>* encoderNetBuilder,
                    IComputationNetBuilder<ElemType>* decoderNetBuilder,
                    DataReader<ElemType>* encoderTrainSetDataReader,
                    IDataReader<ElemType>* decoderTrainSetDataReader,
                    IDataReader<ElemType>* encoderValidationSetDataReader,
                    IDataReader<ElemType>* decoderValidationSetDataReader,
                    const bool makeMode)
                {
                    if (decoderValidationSetDataReader == nullptr)
                        throw std::invalid_argument("validation set reader should not be null.");

                    int startEpoch = DetermineEncoderDecoderStartEpoch(makeMode);
                    if (startEpoch == m_maxEpochs)
                    {
                        fprintf(stderr, "Final model exists. No further training is necessary.\n");
                        return;
                    }

                    wstring modelFileName = GetEncoderModelNameForEpoch(int(startEpoch) - 1);
                    fprintf(stderr, "encoderFileName=%ls\n", modelFileName.c_str());
                    if (startEpoch >= 0)
                        fprintf(stderr, "Starting from checkpoint. Load Encoder Network From File %ws.\n", modelFileName);
                    ComputationNetwork<ElemType>& encoderNet =
                        startEpoch<0 ? encoderNetBuilder->BuildNetworkFromDescription() : encoderNetBuilder->LoadNetworkFromFile(modelFileName, true, true);

                    modelFileName = GetDecoderModelNameForEpoch(int(startEpoch) - 1);
                    fprintf(stderr, "decoderFileName=%ls\n", modelFileName.c_str());
                    if (startEpoch >= 0)
                        fprintf(stderr, "Starting from checkpoint. Load Decoder Network From File %ws.\n", modelFileName);

                    ComputationNetwork<ElemType>& decoderNet =
                        startEpoch<0 ? decoderNetBuilder->BuildNetworkFromDescription() : decoderNetBuilder->LoadNetworkFromFile(modelFileName);

                    startEpoch = max(startEpoch, 0);

                    if (m_doUnitTest)
                    {
                        if (decoderNet.UnitTest() == false)
                            LogicError("unit test on decoder network not passed");

                        return;
                    }

                    fprintf(stderr, "start training ...\n");
                    TrainEncoderDecoderModel(startEpoch, encoderNet, decoderNet, encoderTrainSetDataReader,
                        decoderTrainSetDataReader, encoderValidationSetDataReader, decoderValidationSetDataReader);
                }

                //return -1 if nothing exists
                int DetermineEncoderDecoderStartEpoch(const bool makeMode)
                {
                    if (!makeMode)
                        return -1;  //always start from scratch

                    int firstEpoch = -1;

                    wstring curEpochFile = GetDecoderModelNameForEpoch(int(m_maxEpochs) - 1);
                    for (int e = int(m_maxEpochs) - 1; e >= -1; e--)
                    {
                        const wstring prevEpochFile = GetDecoderModelNameForEpoch(e - 1);

                        if (msra::files::fuptodate(curEpochFile, prevEpochFile, false))
                        {
                            firstEpoch = size_t(e) + 1;
                            break;
                        }
                        else
                            curEpochFile = prevEpochFile;
                    }

                    return firstEpoch;
                }

                wstring GetDecoderModelNameForEpoch(const int epoch, bool bLastModel = false)
                {
                    int epoch1Base = epoch + 1;
                    if (epoch1Base == m_maxEpochs || bLastModel)
                        return m_decoderModelPath;
                    else
                        return msra::strfun::wstrprintf(L"%s.%d", m_decoderModelPath.c_str(), (int)epoch1Base);
                }

                wstring GetEncoderModelNameForEpoch(const int epoch, bool bLastModel = false)
                {
                    int epoch1Base = epoch + 1;
                    if (epoch1Base == m_maxEpochs || bLastModel)
                        return m_encoderModelPath;
                    else
                        return msra::strfun::wstrprintf(L"%s.%d", m_encoderModelPath.c_str(), (int)epoch1Base);
                }
                
                void TrainEncoderDecoderModel(int startEpoch, ComputationNetwork<ElemType>& encoderNet,
                    ComputationNetwork<ElemType>& decoderNet,
                    IDataReader<ElemType>* encoderTrainSetDataReader,
                    IDataReader<ElemType>* decoderTrainSetDataReader,
                    IDataReader<ElemType>* encoderValidationSetDataReader,
                    IDataReader<ElemType>* decoderValidationSetDataReader)
                {
                    std::vector<ComputationNodePtr> & encoderFeatureNodes = encoderNet.FeatureNodes();
                    std::vector<ComputationNodePtr> & encoderEvaluationNodes = encoderNet.OutputNodes();

                    std::vector<ComputationNodePtr> & decoderFeatureNodes = decoderNet.FeatureNodes();
                    std::vector<ComputationNodePtr> & decoderLabelNodes = decoderNet.LabelNodes();
                    std::vector<ComputationNodePtr> decoderCriterionNodes = GetTrainCriterionNodes(decoderNet);
                    std::vector<ComputationNodePtr> decoderEvaluationNodes = GetEvalCriterionNodes(decoderNet);

                    std::map<std::wstring, Matrix<ElemType>*> encoderInputMatrices, decoderInputMatrices;
                    for (size_t i = 0; i<encoderFeatureNodes.size(); i++)
                    {
                        encoderInputMatrices[encoderFeatureNodes[i]->NodeName()] =
                            &encoderFeatureNodes[i]->FunctionValues();
                    }
                    for (size_t i = 0; i<decoderFeatureNodes.size(); i++)
                    {
                        decoderInputMatrices[decoderFeatureNodes[i]->NodeName()] =
                            &decoderFeatureNodes[i]->FunctionValues();
                    }
                    for (size_t i = 0; i<decoderLabelNodes.size(); i++)
                    {
                        decoderInputMatrices[decoderLabelNodes[i]->NodeName()] = &decoderLabelNodes[i]->FunctionValues();
                    }

                    //initializing weights and gradient holder
                    std::list<ComputationNodePtr>& encoderLearnableNodes = encoderNet.LearnableNodes(encoderEvaluationNodes[0]);  //only one criterion so far TODO: support multiple ones?
                    std::list<ComputationNodePtr>& decoderLearnableNodes = decoderNet.LearnableNodes(decoderCriterionNodes[0]);
                    std::list<ComputationNodePtr> learnableNodes;
                    for (auto nodeIter = encoderLearnableNodes.begin(); nodeIter != encoderLearnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = (*nodeIter);
                        learnableNodes.push_back(node);
                    }
                    for (auto nodeIter = decoderLearnableNodes.begin(); nodeIter != decoderLearnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = (*nodeIter);
                        learnableNodes.push_back(node);
                    }

                    std::list<Matrix<ElemType>> smoothedGradients;
                    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = (*nodeIter);
                        smoothedGradients.push_back(Matrix<ElemType>(node->FunctionValues().GetNumRows(), node->FunctionValues().GetNumCols(), node->FunctionValues().GetDeviceId()));
                    }

                    vector<ElemType> epochCriterion;
                    ElemType avgCriterion, prevCriterion;
                    for (size_t i = 0; i < 2; i++)
                        epochCriterion.push_back(std::numeric_limits<ElemType>::infinity());
                    avgCriterion = prevCriterion = std::numeric_limits<ElemType>::infinity();

                    size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

                    std::vector<ElemType> epochEvalErrors(decoderEvaluationNodes.size(), std::numeric_limits<ElemType>::infinity());

                    std::vector<wstring> evalNodeNames;
                    for (size_t i = 0; i<decoderEvaluationNodes.size(); i++)
                        evalNodeNames.push_back(decoderEvaluationNodes[i]->NodeName());

                    size_t totalSamplesSeen = 0;
                    ElemType learnRatePerSample = 0.5f / m_mbSize[startEpoch];

                    int m_numPrevLearnRates = 5; //used to control the upper learnining rate in LR search to reduce computation
                    vector<ElemType> prevLearnRates;
                    prevLearnRates.resize(m_numPrevLearnRates);
                    for (int i = 0; i<m_numPrevLearnRates; i++)
                        prevLearnRates[i] = std::numeric_limits<ElemType>::infinity();

                    //precompute mean and invStdDev nodes and save initial model
                    if (/// to-do doesn't support pre-compute such as MVN here 
                        /// PreCompute(net, encoderTrainSetDataReader, encoderFeatureNodes, encoderlabelNodes, encoderInputMatrices) || 
                        startEpoch == 0)
                    {
                        encoderNet.SaveToFile(GetEncoderModelNameForEpoch(int(startEpoch) - 1));
                        decoderNet.SaveToFile(GetDecoderModelNameForEpoch(int(startEpoch) - 1));
                    }

                    bool learnRateInitialized = false;
                    if (startEpoch > 0)
                    {
                        learnRateInitialized = LoadCheckPointInfo(startEpoch - 1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);
                        setMomentum(m_momentumInputPerMB[m_momentumInputPerMB.size() - 1]);
                    }

                    if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && !learnRateInitialized && m_learningRatesPerSample.size() <= startEpoch)
                        throw std::invalid_argument("When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, or an explicit learning rate must be specified in config for the starting epoch.");

                    ULONG dropOutSeed = 1;
                    ElemType prevDropoutRate = 0;

                    bool learnRateReduced = false;

                    for (int i = int(startEpoch); i<int(m_maxEpochs); i++)
                    {
                        auto t_start_epoch = clock();

                        //set dropout rate
                        SetDropoutRate(encoderNet, encoderEvaluationNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);
                        SetDropoutRate(decoderNet, decoderCriterionNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);

                        //learning rate adjustment
                        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || (m_learningRatesPerSample.size() > 0 && m_learningRatesPerSample.size() > i))
                        {
                            learnRatePerSample = m_learningRatesPerSample[i];
                            setMomentum(m_momentumInputPerMB[i]);
                        }
                        else if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
                        {
                            NOT_IMPLEMENTED;
                        }

                        learnRateInitialized = true;

                        if (learnRatePerSample < m_minLearnRate)
                        {
                            fprintf(stderr, "Learn Rate Per Sample for Epoch[%lu] = %.8g is less than minLearnRate %.8g. Training stops.\n", i + 1, learnRatePerSample, m_minLearnRate);
                            break;
                        }

                        TrainOneEpochEncoderDecoderWithHiddenStates(encoderNet, decoderNet, i, m_epochSize, encoderTrainSetDataReader,
                            decoderTrainSetDataReader, learnRatePerSample,
                            encoderFeatureNodes, encoderEvaluationNodes, encoderInputMatrices,
                            decoderFeatureNodes, decoderLabelNodes, decoderCriterionNodes, decoderEvaluationNodes,
                            decoderInputMatrices, learnableNodes, smoothedGradients,
                            epochCriterion, epochEvalErrors, totalSamplesSeen);


                        auto t_end_epoch = clock();
                        ElemType epochTime = ElemType(1.0)*(t_end_epoch - t_start_epoch) / (CLOCKS_PER_SEC);

                        //                    fprintf(stderr, "Finished Epoch[%lu]: [Training Set] Train Loss Per Sample = %.8g    ", i + 1, epochCriterion);
                        fprintf(stderr, "Finished Epoch[%lu]: [Training Set] Decoder Train Loss Per Sample = %.8g    ", i + 1, epochCriterion[0]);
                        if (epochEvalErrors.size() == 1)
                        {
                            fprintf(stderr, "EvalErr Per Sample = %.8g   Ave Learn Rate Per Sample = %.10g  Epoch Time=%.8g\n", epochEvalErrors[0], learnRatePerSample, epochTime);
                        }
                        else
                        {
                            fprintf(stderr, "EvalErr Per Sample ");
                            for (size_t j = 0; j<epochEvalErrors.size(); j++)
                                fprintf(stderr, "[%lu]=%.8g ", j, epochEvalErrors[j]);
                            fprintf(stderr, "Ave Learn Rate Per Sample = %.10g  Epoch Time=%.8g\n", learnRatePerSample, epochTime);
                            fprintf(stderr, "Finished Epoch[%lu]: Criterion Node [%ws] Per Sample = %.8g\n", i + 1, decoderCriterionNodes[0]->NodeName().c_str(), epochCriterion);
                            for (size_t j = 0; j<epochEvalErrors.size(); j++)
                                fprintf(stderr, "Finished Epoch[%lu]: Evaluation Node [%ws] Per Sample = %.8g\n", i + 1, evalNodeNames[j].c_str(), epochEvalErrors[j]);
                        }

                        if (decoderValidationSetDataReader != decoderTrainSetDataReader && decoderValidationSetDataReader != nullptr &&
                            encoderValidationSetDataReader != encoderTrainSetDataReader && encoderValidationSetDataReader != nullptr)
                        {
                            SimpleEvaluator<ElemType> evalforvalidation(decoderNet);
                            vector<wstring> cvEncoderSetTrainAndEvalNodes;
                            cvEncoderSetTrainAndEvalNodes.push_back(encoderEvaluationNodes[0]->NodeName());

                            evalforvalidation.SetEncoderDecoderNodePairs(m_lst_pair_encoder_decoder_nodes);

                            vector<wstring> cvDecoderSetTrainAndEvalNodes;
                            cvDecoderSetTrainAndEvalNodes.push_back(decoderCriterionNodes[0]->NodeName());
                            cvDecoderSetTrainAndEvalNodes.push_back(decoderEvaluationNodes[0]->NodeName());

                            vector<ElemType> vScore = evalforvalidation.EvaluateEncoderDecoderWithHiddenStates(
                                encoderNet, decoderNet,
                                *encoderValidationSetDataReader,
                                *decoderValidationSetDataReader, cvEncoderSetTrainAndEvalNodes,
                                cvDecoderSetTrainAndEvalNodes, m_mbSize[i]);
                            fprintf(stderr, "Finished Epoch[%lu]: [Validation Set] Train Loss Per Sample = %.8g  EvalErr Per Sample = %.8g\n",
                                i + 1, vScore[0], vScore[1]);

                            epochCriterion[0] = vScore[0]; //the first one is the decoder training criterion.
                        }

                        bool loadedPrevModel = false;
                        size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
                        if (avgCriterion == std::numeric_limits<ElemType>::infinity())
                            avgCriterion = epochCriterion[0];
                        else
                            avgCriterion = ((epochsSinceLastLearnRateAdjust - 1 - epochsNotCountedInAvgCriterion)* avgCriterion + epochCriterion[0]) / (epochsSinceLastLearnRateAdjust - epochsNotCountedInAvgCriterion);

                        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && m_learningRatesPerSample.size() <= i && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
                        {
                            if (prevCriterion - avgCriterion < 0 && prevCriterion != std::numeric_limits<ElemType>::infinity())
                            {
                                if (m_loadBestModel)
                                {
                                    encoderNet.LoadPersistableParametersFromFile(GetEncoderModelNameForEpoch(i - 1),
                                        false);
                                    decoderNet.LoadPersistableParametersFromFile(GetDecoderModelNameForEpoch(i - 1),
                                        m_validateAfterModelReloading);
                                    encoderNet.ResetEvalTimeStamp();
                                    decoderNet.ResetEvalTimeStamp();
                                    LoadCheckPointInfo(i - 1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);
                                    fprintf(stderr, "Loaded the previous model which has better training criterion.\n");
                                    loadedPrevModel = true;
                                }
                            }

                            if (m_continueReduce)
                            {
                                if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                                {
                                    if (learnRateReduced == false)
                                    {
                                        learnRateReduced = true;
                                    }
                                    else
                                    {
                                        decoderNet.SaveToFile(GetDecoderModelNameForEpoch(i, true));
                                        encoderNet.SaveToFile(GetEncoderModelNameForEpoch(i, true));
                                        fprintf(stderr, "Finished training and saved final model\n\n");
                                        break;
                                    }
                                }
                                if (learnRateReduced)
                                {
                                    learnRatePerSample *= m_learnRateDecreaseFactor;
                                    fprintf(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                                }
                            }
                            else
                            {
                                if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                                {

                                    learnRatePerSample *= m_learnRateDecreaseFactor;
                                    fprintf(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                                }
                                else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan*prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                                {
                                    learnRatePerSample *= m_learnRateIncreaseFactor;
                                    fprintf(stderr, "learnRatePerSample increased to %.8g\n", learnRatePerSample);
                                }
                            }
                        }

                        if (!loadedPrevModel && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)  //not loading previous values then set them
                        {
                            prevCriterion = avgCriterion;
                            epochsNotCountedInAvgCriterion = 0;
                        }

                        //persist model and check-point info
                        decoderNet.SaveToFile(GetDecoderModelNameForEpoch(i));
                        encoderNet.SaveToFile(GetEncoderModelNameForEpoch(i));
                        SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);
                        if (!m_keepCheckPointFiles)
                            DeleteFile(GetCheckPointFileNameForEpoch(i - 1).c_str());  //delete previous checkpiont file to save space

                        if (learnRatePerSample < 1e-12)
                            fprintf(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n", learnRatePerSample);
                    }
                }

                /// use hidden states between encoder and decoder to communicate between two networks
                void TrainOneEpochEncoderDecoderWithHiddenStates(
                    ComputationNetwork<ElemType>& encoderNet,  /// encoder network
                    ComputationNetwork<ElemType>& decoderNet,
                    const int epochNumber, const size_t epochSize,
                    IDataReader<ElemType>* encoderTrainSetDataReader,
                    IDataReader<ElemType>* decoderTrainSetDataReader,
                    const ElemType learnRatePerSample,
                    const std::vector<ComputationNodePtr>& encoderFeatureNodes,
                    const std::vector<ComputationNodePtr>& encoderEvaluationNodes,
                    std::map<std::wstring, Matrix<ElemType>*>& encoderInputMatrices,
                    const std::vector<ComputationNodePtr>& decoderFeatureNodes,
                    const std::vector<ComputationNodePtr>& decoderLabelNodes,
                    const std::vector<ComputationNodePtr>& decoderCriterionNodes,
                    const std::vector<ComputationNodePtr>& decoderEvaluationNodes,
                    std::map<std::wstring, Matrix<ElemType>*>& decoderInputMatrices,
                    const std::list<ComputationNodePtr>& learnableNodes,
                    std::list<Matrix<ElemType>>& smoothedGradients,
                    vector<ElemType>& epochCriterion, std::vector<ElemType>& epochEvalErrors, size_t& totalSamplesSeen)
                {
                    assert(encoderEvaluationNodes.size() == 1);

                    Matrix<ElemType> historyMat(encoderNet.GetDeviceID());

                    ElemType readTimeInMBs = 0, ComputeTimeInMBs = 0;
                    vector<ElemType> epochCriterionLastMBs;
                    for (size_t i = 0; i < epochCriterion.size(); i++)
                        epochCriterionLastMBs.push_back(0);

                    int numSamplesLastMBs = 0;
                    std::vector<ElemType> epochEvalErrorsLastMBs(epochEvalErrors.size(), 0);

                    clock_t startReadMBTime = 0, startComputeMBTime = 0;
                    clock_t endReadMBTime = 0, endComputeMBTime = 0;

                    //initialize statistics
                    size_t totalEpochSamples = 0;

                    int numMBsRun = 0;

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

                    size_t numEvalNodes = epochEvalErrors.size();

                    // NOTE: the following two local matrices are not used in PTask path
                    Matrix<ElemType> localEpochCriterion(1, 2, decoderNet.GetDeviceID()); //assume only one training criterion node for each epoch
                    Matrix<ElemType> localEpochEvalErrors(1, numEvalNodes, decoderNet.GetDeviceID());

                    localEpochCriterion.SetValue(0);
                    localEpochEvalErrors.SetValue(0);

                    encoderTrainSetDataReader->StartMinibatchLoop(m_mbSize[epochNumber], epochNumber, m_epochSize);
                    decoderTrainSetDataReader->StartMinibatchLoop(m_mbSize[epochNumber], epochNumber, m_epochSize);

                    startReadMBTime = clock();
                    Matrix<ElemType> mEncoderOutput(encoderEvaluationNodes[0]->FunctionValues().GetDeviceId());
                    Matrix<ElemType> mDecoderInput(decoderEvaluationNodes[0]->FunctionValues().GetDeviceId());

                    unsigned uSeedForDataReader = epochNumber;

                    bool bContinueDecoding = true;
                    while (bContinueDecoding)
                    {
                        try{
                            encoderTrainSetDataReader->SetRandomSeed(uSeedForDataReader);
                            encoderTrainSetDataReader->GetMinibatch(encoderInputMatrices);

                            /// now gradients on decoder network
                            decoderTrainSetDataReader->SetRandomSeed(uSeedForDataReader);
                            if (decoderTrainSetDataReader->GetMinibatch(decoderInputMatrices) == false)
                                break;
                        }
                        catch (...)
                        {
                            RuntimeError("Errors in reading features ");
                        }

                        size_t actualMBSize = decoderNet.GetActualMBSize();
                        if (actualMBSize == 0)
                            LogicError("decoderTrainSetDataReader read data but decoderNet reports no data read");

                        UpdateEvalTimeStamps(encoderFeatureNodes);
                        UpdateEvalTimeStamps(decoderFeatureNodes);
                        UpdateEvalTimeStamps(decoderLabelNodes);

                        endReadMBTime = clock();
                        startComputeMBTime = clock();

                        /// not the sentence begining, because the initial hidden layer activity is from the encoder network
                        //                    decoderTrainSetDataReader->SetSentenceBegin(false);
                        //                    decoderTrainSetDataReader->SetSentenceSegBatch(decoderNet.m_sentenceSeg);
                        //                    decoderTrainSetDataReader->SetSentenceSegBatch(decoderNet.m_sentenceBegin);

                        if (m_doGradientCheck)
                        {
                            if (EncoderDecoderGradientCheck(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors) == false)
                            {
                                throw runtime_error("SGD::TrainOneEpochEncoderDecoderWithHiddenStates gradient check not passed!");
                            }
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);
                        }

                        EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                        EncoderDecoderWithHiddenStatesErrorProp(encoderNet,
                                decoderNet, encoderEvaluationNodes,
                                decoderCriterionNodes,
                                historyMat, m_lst_pair_encoder_decoder_nodes);

                        //update model parameters
                        if (learnRatePerSample > m_minLearnRate * 0.01)
                        {
                            auto smoothedGradientIter = smoothedGradients.begin();
                            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, smoothedGradientIter++)
                            {
                                ComputationNodePtr node = (*nodeIter);
                                Matrix<ElemType>& smoothedGradient = (*smoothedGradientIter);

                                UpdateWeights(node, smoothedGradient, learnRatePerSample, actualMBSize, m_mbSize[epochNumber], m_L2RegWeight, m_L1RegWeight, m_needAveMultiplier);
                            }
                        }


                        endComputeMBTime = clock();
                        numMBsRun++;
                        if (m_traceLevel > 0)
                        {
                            ElemType MBReadTime = (ElemType)(endReadMBTime - startReadMBTime) / (CLOCKS_PER_SEC);
                            ElemType MBComputeTime = (ElemType)(endComputeMBTime - startComputeMBTime) / CLOCKS_PER_SEC;

                            readTimeInMBs += MBReadTime;
                            ComputeTimeInMBs += MBComputeTime;
                            numSamplesLastMBs += int(actualMBSize);

                            if (numMBsRun % m_numMBsToShowResult == 0)
                            {

                                epochCriterion[0] = localEpochCriterion.Get00Element();
                                for (size_t i = 0; i< numEvalNodes; i++)
                                    epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0, i);

                                ElemType llk = (epochCriterion[0] - epochCriterionLastMBs[0]) / numSamplesLastMBs;
                                ElemType ppl = exp(llk);
                                fprintf(stderr, "Epoch[%d]-Minibatch[%d-%d]: Samples Seen = %d   Decoder Train Loss Per Sample = %.8g PPL = %.4e ", epochNumber + 1, numMBsRun - m_numMBsToShowResult + 1, numMBsRun, numSamplesLastMBs,
                                    llk, ppl);
                                for (size_t i = 0; i<numEvalNodes; i++){
                                    fprintf(stderr, "EvalErr[%lu] Per Sample = %.8g    ", i, (epochEvalErrors[i] - epochEvalErrorsLastMBs[i]) / numSamplesLastMBs);
                                }
                                fprintf(stderr, "ReadData Time = %.8g Computing Time=%.8g Total Time Per Sample=%.8g\n", readTimeInMBs, ComputeTimeInMBs, (readTimeInMBs + ComputeTimeInMBs) / numSamplesLastMBs);

                                //reset statistics
                                readTimeInMBs = ComputeTimeInMBs = 0;
                                numSamplesLastMBs = 0;

                                epochCriterionLastMBs = epochCriterion;
                                for (size_t i = 0; i< numEvalNodes; i++)
                                    epochEvalErrorsLastMBs[i] = epochEvalErrors[i];
                            }
                        }
                        startReadMBTime = clock();
                        totalEpochSamples += actualMBSize;
                        totalSamplesSeen += actualMBSize;

                        if (totalEpochSamples >= epochSize)
                            break;

                        /// call DataEnd function 
                        /// DataEnd does reader specific process if sentence ending is reached
                        //                    encoderTrainSetDataReader->SetSentenceEnd(true);
                        //                    decoderTrainSetDataReader->SetSentenceEnd(true);
                        encoderTrainSetDataReader->DataEnd(endDataSentence);
                        decoderTrainSetDataReader->DataEnd(endDataSentence);

                        uSeedForDataReader++;
                    }

                    localEpochCriterion /= float(totalEpochSamples);
                    localEpochEvalErrors /= float(totalEpochSamples);

                    epochCriterion[0] = localEpochCriterion.Get00Element();
                    for (size_t i = 0; i < numEvalNodes; i++)
                    {
                        epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0, i);
                    }
                    fprintf(stderr, "total samples in epoch[%d] = %d\n", epochNumber, totalEpochSamples);
                }

                bool EncoderDecoderGradientCheck(
                    ComputationNetwork<ElemType>& encoderNet,  /// encoder network
                    ComputationNetwork<ElemType>& decoderNet,
                    IDataReader<ElemType>* encoderTrainSetDataReader,
                    IDataReader<ElemType>* decoderTrainSetDataReader,
                    const std::vector<ComputationNodePtr>& encoderEvaluationNodes,
                    const std::vector<ComputationNodePtr>& decoderFeatureNodes,
                    const std::vector<ComputationNodePtr>& decoderCriterionNodes,
                    const std::vector<ComputationNodePtr>& decoderEvaluationNodes,
                    Matrix<ElemType>& historyMat,
                    Matrix<ElemType>& localEpochCriterion,
                    Matrix<ElemType>& localEpochEvalErrors
                    )
                {
                    vector<string> verror_msgs;

                    /// check decoder learnable parameters
                    std::list<ComputationNodePtr>& learnableNodes = encoderNet.LearnableNodes(encoderEvaluationNodes[0]);
                    DEVICEID_TYPE deviceId;

                    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = (*nodeIter);

                        for (size_t itry = 0; itry < min(10, node->FunctionValues().GetNumElements()); itry++)
                        {

                            int irow = (int)fmod(rand(), node->FunctionValues().GetNumRows() - 1);
                            int icol = (int)fmod(rand(), node->FunctionValues().GetNumCols() - 1);
                            irow = max(0, irow);
                            icol = max(0, icol);

                            fprintf(stderr, "\n###### d%ws######\n", node->NodeName().c_str());
                            deviceId = node->FunctionValues().GetDeviceId();  // original device id

                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            ElemType eOrg = node->FunctionValues()(irow, icol);  /// warning :: this function will put matrix into CPU
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);

                            /// perturb parameter
                            ElemType ePos = eOrg + (ElemType)EPSILON;
                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false); 
                            node->FunctionValues().SetValue(irow, icol, ePos);
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);

                            node->UpdateEvalTimeStamp();
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);

                            EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                            ElemType score1 = localEpochCriterion.Get00Element();

                            /// perturb parameter
                            ElemType eNeg = eOrg - (ElemType)EPSILON;
                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            node->FunctionValues().SetValue(irow, icol, eNeg);
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);

                            node->UpdateEvalTimeStamp();
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);

                            EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                            ElemType score2 = localEpochCriterion.Get00Element();

                            ElemType grdNum = (score1 - score2) / (ePos - eNeg);

                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            node->FunctionValues().SetValue(irow, icol, eOrg);
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);
                            node->UpdateEvalTimeStamp();
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);

                            EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                            EncoderDecoderWithHiddenStatesErrorProp(encoderNet,
                                decoderNet, encoderEvaluationNodes,
                                decoderCriterionNodes,
                                historyMat, m_lst_pair_encoder_decoder_nodes);

                            ElemType grdErr = node->GradientValues()(irow, icol);

                            // check if they are consistent
                            ElemType threshold = (ElemType)pow((ElemType)10.0, max((ElemType)0.0, ceil(log10(min(fabs(grdErr), fabs(grdNum))))) - (int)m_gradientCheckSigDigit);
                            ElemType diff = (ElemType)fabs(grdErr - grdNum);
                            bool wrong = (_isnan(diff) || diff > threshold);
                            if (wrong)
                            {
                                char serr[2048];
                                sprintf_s(serr, 2048, "Encoder %ws Numeric gradient = %e, Error BP gradient = %e", node->NodeName().c_str(), grdNum, grdErr);
                                fprintf(stdout, "%s\n", serr);
                                verror_msgs.push_back(serr);
                            }
                        }
                    }


                    learnableNodes = decoderNet.LearnableNodes(decoderEvaluationNodes[0]);

                    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = (*nodeIter);

                        for (size_t itry = 0; itry < min(10, node->FunctionValues().GetNumElements()); itry++)
                        {

                            int irow = (int)fmod(rand(), node->FunctionValues().GetNumRows() - 1);
                            int icol = (int)fmod(rand(), node->FunctionValues().GetNumCols() - 1);
                            irow = max(0, irow);
                            icol = max(0, icol);

                            fprintf(stderr, "\n###### d%ws######\n", node->NodeName().c_str());
                            deviceId = node->FunctionValues().GetDeviceId();  // original device id

                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            ElemType eOrg = node->FunctionValues()(irow, icol);  /// warning :: this function will put matrix into CPU
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);

                            /// perturb parameter
                            ElemType ePos = eOrg + (ElemType)EPSILON;
                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            node->FunctionValues().SetValue(irow, icol, ePos);
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);

                            node->UpdateEvalTimeStamp();
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);

                            EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                            ElemType score1 = localEpochCriterion.Get00Element();

                            ElemType eNeg = eOrg - (ElemType)EPSILON;
                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            node->FunctionValues().SetValue(irow, icol, eNeg);
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);
                            node->UpdateEvalTimeStamp();
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);

                            EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                            ElemType score1r = localEpochCriterion.Get00Element();

                            ElemType grdNum = (score1r - score1) / (eNeg - ePos);

                            node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                            node->FunctionValues().SetValue(irow, icol, eOrg);
                            if (node->FunctionValues().GetDeviceId() != deviceId)
                                node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), deviceId, true);
                            node->UpdateEvalTimeStamp();
                            localEpochCriterion.SetValue(0);
                            localEpochEvalErrors.SetValue(0);

                            EncoderDecoderWithHiddenStatesForwardPass(encoderNet,
                                decoderNet, encoderTrainSetDataReader,
                                decoderTrainSetDataReader, encoderEvaluationNodes,
                                decoderFeatureNodes, decoderCriterionNodes, decoderEvaluationNodes, historyMat, localEpochCriterion, localEpochEvalErrors);

                            EncoderDecoderWithHiddenStatesErrorProp(encoderNet,
                                decoderNet, encoderEvaluationNodes,
                                decoderCriterionNodes,
                                historyMat, m_lst_pair_encoder_decoder_nodes);

                            ElemType grdErr = node->GradientValues()(irow, icol);

                            // check if they are consistent
                            ElemType threshold = (ElemType)pow((ElemType)10.0, max((ElemType)0.0, ceil(log10(min(fabs(grdErr), fabs(grdNum))))) - (int)m_gradientCheckSigDigit);
                            ElemType diff = (ElemType)fabs(grdErr - grdNum);
                            bool wrong = (_isnan(diff) || diff > threshold);
                            if (wrong)
                            {
                                char serr[2048];
                                sprintf_s((char*)serr, 2048, "Decoder %ws Numeric gradient = %e, Error BP gradient = %e", node->NodeName().c_str(), grdNum, grdErr);
                                fprintf(stdout, "%s\n", serr);
                                verror_msgs.push_back(serr);
                            }
                        }
                    }

                    if (verror_msgs.size() > 0)
                        return false;
                    return true;
                }

                void EncoderDecoderWithHiddenStatesForwardPass(
                    ComputationNetwork<ElemType>& encoderNet,  /// encoder network
                    ComputationNetwork<ElemType>& decoderNet,
                    IDataReader<ElemType>* encoderTrainSetDataReader,
                    IDataReader<ElemType>* decoderTrainSetDataReader,
                    const std::vector<ComputationNodePtr>& encoderEvaluationNodes,
                    const std::vector<ComputationNodePtr>& decoderFeatureNodes,
                    const std::vector<ComputationNodePtr>& decoderCriterionNodes,
                    const std::vector<ComputationNodePtr>& decoderEvaluationNodes,
                    Matrix<ElemType>& historyMat,
                    Matrix<ElemType>& localEpochCriterion,
                    Matrix<ElemType>& localEpochEvalErrors
                    )
                {
                    try{
                        size_t actualMBSize = encoderNet.GetActualMBSize();

                        encoderNet.SetActualMiniBatchSize(actualMBSize);
                        encoderNet.SetActualNbrSlicesInEachRecIter(encoderTrainSetDataReader->NumberSlicesInEachRecurrentIter());
                        encoderTrainSetDataReader->SetSentenceSegBatch(encoderNet.m_sentenceSeg);

                        encoderNet.Evaluate(encoderEvaluationNodes[0]);

                        actualMBSize = decoderNet.GetActualMBSize();

                        decoderNet.SetActualMiniBatchSize(actualMBSize);
                        decoderNet.SetActualNbrSlicesInEachRecIter(decoderTrainSetDataReader->NumberSlicesInEachRecurrentIter());

                        /// not the sentence begining, because the initial hidden layer activity is from the encoder network
                        decoderTrainSetDataReader->SetSentenceSegBatch(decoderNet.m_sentenceSeg);

                        /// get the pair of encode and decoder nodes
                        for (list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = m_lst_pair_encoder_decoder_nodes.begin(); iter != m_lst_pair_encoder_decoder_nodes.end(); iter++)
                        {
                            /// past hidden layer activity from encoder network to decoder network
                            ComputationNodePtr encoderNode = iter->first;
                            ComputationNodePtr decoderNode = iter->second;

                            encoderNode->GetHistory(historyMat, true); /// get the last state activity
                            decoderNode->SetHistory(historyMat);
#ifdef DEBUG_DECODER
                            fprintf(stderr, "LSTM past output norm = %.8e\n", historyMat.ColumnSlice(0, nstreams).FrobeniusNorm());
                            fprintf(stderr, "LSTM past state norm = %.8e\n", historyMat.ColumnSlice(nstreams, nstreams).FrobeniusNorm());
#endif
                        }

                        UpdateEvalTimeStamps(decoderFeatureNodes);
                        decoderNet.Evaluate(decoderCriterionNodes[0]);

                        Matrix<ElemType>::AddElementToElement(decoderCriterionNodes[0]->FunctionValues(), 0, 0, localEpochCriterion, 0, 0);

                        size_t numEvalNodes = decoderEvaluationNodes.size();
                        std::vector<ElemType>mbEvalErrors(numEvalNodes, 0);
                        for (size_t i = 0; i < numEvalNodes; i++)
                        {
                            decoderNet.Evaluate(decoderEvaluationNodes[i]);
                            Matrix<ElemType>::AddElementToElement(decoderEvaluationNodes[i]->FunctionValues(), 0, 0, localEpochEvalErrors, 0, i);
                        }
#ifdef DEBUG_DECODER
                        fprintf(stderr, "ForwardPass score = %.8e\n", localEpochCriterion.Get00Element());
#endif
                    }
                    catch (...)
                    {
                        RuntimeError("Errors in forward pass");
                    }
                }

                void EncoderDecoderWithHiddenStatesErrorProp(
                    ComputationNetwork<ElemType>& encoderNet,  /// encoder network
                    ComputationNetwork<ElemType>& decoderNet,
                    const std::vector<ComputationNodePtr>& encoderEvaluationNodes,
                    const std::vector<ComputationNodePtr>& decoderCriterionNodes,
                    Matrix<ElemType>& historyMat,
                    list<pair<ComputationNodePtr, ComputationNodePtr>> lst_pair_encoder_decoder_nodes
                    )
                {
                    try{
                        /// don't reevalute, need to call forward pass before call this function
                        //                decoderNet.m_sentenceBegin.assign(decoderNet.m_sentenceBegin.size(), -1);
                        try{
                            decoderNet.ComputeGradient(decoderCriterionNodes[0]);
                        }
                        catch (...)
                        {
                            RuntimeError("Error in evaluating gradients for decoder network");
                        }
                        
                        try{
                            /// get the pair of encode and decoder nodes
                            for (list<pair<ComputationNodePtr, ComputationNodePtr>>::iterator iter = lst_pair_encoder_decoder_nodes.begin(); iter != lst_pair_encoder_decoder_nodes.end(); iter++)
                            {
                                /// past gradients to hidden layer activity from decoder network to encoder network
                                ComputationNodePtr encoderNode = iter->first;
                                ComputationNodePtr decoderNode = iter->second;

                                decoderNode->GetErrorsToPreviousMinibatch(historyMat);
                                encoderNode->SetErrorsFromFutureMinibatch(historyMat);
                            }
                        }
                        catch (...)
                        {
                            RuntimeError("Error in passing gradients from decoder to encoder");
                        }

                        Matrix<ElemType> initGradient(encoderNet.GetDeviceID());
                        try{
                            // compute gradients on encoder networks
                            initGradient.Resize(encoderEvaluationNodes[0]->FunctionValues().GetNumRows(),
                                encoderEvaluationNodes[0]->FunctionValues().GetNumCols());

                            initGradient.SetValue(0);
                        }
                        catch (...)
                        {
                            RuntimeError("Error in init gradients for error propagation to encoder network");
                        }

                        try{
                            encoderNet.ComputeGradient(encoderEvaluationNodes[0], false, &initGradient);
                        }
                        catch (...)
                        {
                            RuntimeError("Error in evaluating gradients for encoder network");
                        }
                    }
                    catch (...)
                    {
                        RuntimeError("Errors in backpropagation");
                    }
                }

            };

        }
    }
}
