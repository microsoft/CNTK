// MultiNetworksEvaluator/SGD -- This represents earlier efforts to use CNTK for sequence-to-sequence modeling. This is no longer the intended design.
//
// <copyright file="MultiNetworksSGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// TODO: this cannot be instantiated as a whole (compile error), although some function is called from CNTK.cpp--should be fixed

#include "Basics.h"
#include "ComputationNetwork.h"
#include "IComputationNetBuilder.h"
#include "SimpleEvaluator.h"
#include "MultiNetworksEvaluator.h"
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

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    extern std::wstring GetEncoderModelNameForEpoch(int epoch, bool b = false);
    extern std::wstring GetDecoderModelNameForEpoch(int epoch, bool b = false);

    template<class ElemType>
    class MultiNetworksSGD : SGD<ElemType>
    {
        ElemType m_default_activity;

        using SGDBase = SGD<ElemType>;

    public:
        // TODO: use a macro similar to class ComputeNode
        using SGDBase::m_modelPath;
        using SGDBase::m_maxEpochs;
        using SGDBase::m_doUnitTest;
        using SGDBase::m_learnRateAdjustInterval;
        using SGDBase::m_mbSize;
        using SGDBase::m_momentumParam; using SGDBase::m_learningRatesParam;
        using SGDBase::GetLearningRatePerSample; using SGDBase::GetMomentumPerSample;
        using SGDBase::m_dropoutRates;
        using SGDBase::m_autoLearnRateSearchType;
        using SGDBase::m_minLearnRate;
        using SGDBase::m_loadBestModel;
        using SGDBase::m_validateAfterModelReloading;
        using SGDBase::m_continueReduce;
        using SGDBase::m_reduceLearnRateIfImproveLessThan;
        using SGDBase::m_epochSize;
        using SGDBase::m_learnRateDecreaseFactor;
        using SGDBase::m_increaseLearnRateIfImproveMoreThan;
        using SGDBase::m_learnRateIncreaseFactor;
        using SGDBase::m_keepCheckPointFiles;
        using SGDBase::m_doGradientCheck;
        using SGDBase::m_L2RegWeight;
        using SGDBase::m_L1RegWeight;
        using SGDBase::m_needAveMultiplier;
        using SGDBase::m_traceLevel;
        using SGDBase::m_numMBsToShowResult;
        using SGDBase::m_gradientCheckSigDigit;
        using SGDBase::m_prevChosenMinibatchSize;
        using SGDBase::UpdateWeights;
        using SGDBase::GetCheckPointFileNameForEpoch;
        using SGDBase::GetTrainCriterionNodes;
        using SGDBase::GetEvalCriterionNodes;

        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

        /// for encoder and decoder nodes pairing
        wstring m_decoderModelPath;
        wstring m_backwardDecoderModelPath;
        wstring m_encoderModelPath;

        list<pair<wstring, wstring>> m_lst_pair_encoder_decode_node_names;
        list<pair<ComputationNodeBasePtr, ComputationNodeBasePtr>> m_lst_pair_encoder_decoder_nodes;

    public:
        MultiNetworksSGD(const ConfigParameters& configSGD) : SGDBase(configSGD)
        {
        }

        ~MultiNetworksSGD()
        {
        }

        void InitTrainEncoderDecoderWithHiddenStates(const ConfigParameters& readerConfig)
        {

            m_decoderModelPath = m_modelPath + L".decoder";
            m_backwardDecoderModelPath = m_modelPath + L".backward.decoder";
            m_encoderModelPath = m_modelPath + L".encoder";

            ConfigArray arrEncoderNodeNames = readerConfig(L"encoderNodes", "");
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
                fprintf(stderr, "paired %ls <-> %ls\n", encoderNodeNames[i].c_str(), decoderNodeNames[i].c_str());
            }
        }

        void EncoderDecoder(vector<IComputationNetBuilder<ElemType>*> netBuilder, DEVICEID_TYPE deviceId,
            vector<IDataReader<ElemType>*> trainSetDataReader,
            vector<IDataReader<ElemType>*> validationSetDataReader,
            const bool makeMode)
        {
            if (validationSetDataReader.size() == 0)
                InvalidArgument("validation set reader should not be null.");

            int startEpoch = DetermineEncoderDecoderStartEpoch(makeMode);
            if (startEpoch == m_maxEpochs)
            {
                fprintf(stderr, "No further training is necessary.\n");
                return;
            }

            size_t iNumNetworks = netBuilder.size();
            vector<ComputationNetworkPtr> nets;
            ComputationNetworkPtr eachNet;
            for (size_t k = 0; k < iNumNetworks; k++)
            {
                wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1, false, msra::strfun::wstrprintf(L".%d", k));
                fprintf(stderr, "network model FileName=%ls\n", modelFileName.c_str());
                if (startEpoch >= 0)
                    fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());
                if (k == 0)
                {
                    eachNet =
                        startEpoch < 0 ? netBuilder[k]->BuildNetworkFromDescription() : ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName, FileOptions::fileOptionsBinary, true/*bAllowNoCriterionNode*/);
                    nets.push_back(eachNet);
                }
                else
                {
                    eachNet =
                        startEpoch < 0 ? netBuilder[k]->BuildNetworkFromDescription(nets[k - 1].get()) : ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName, FileOptions::fileOptionsBinary, false/*bAllowNoCriterionNode*/, nets[k - 1].get());
                    nets.push_back(eachNet);
                }
            }

            startEpoch = max(startEpoch, 0);

            if (m_doUnitTest)
            {
                if (nets[iNumNetworks - 1]->UnitTest() == false)
                    LogicError("unit test on decoder network not passed");

                return;
            }

            fprintf(stderr, "start training ...\n");
            TrainEncoderDecoderModel(startEpoch, nets, trainSetDataReader, validationSetDataReader);
        }

        //return -1 if nothing exists
        int DetermineEncoderDecoderStartEpoch(const bool makeMode)
        {
            if (!makeMode)
                return -1;  //always start from scratch

            int firstEpoch = -1;

            wstring curEpochFile = GetModelNameForEpoch(int(m_maxEpochs) - 1, false, L".0");
            for (int e = int(m_maxEpochs) - 1; e >= -1; e--)
            {
                const wstring prevEpochFile = GetModelNameForEpoch(e - 1, false, L".0");

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

        wstring GetModelNameForEpoch(const int epoch, bool bLastModel = false, wstring ext = L"")
        {
            int epoch1Base = epoch + 1;
            if (epoch1Base == m_maxEpochs || bLastModel)
                return m_modelPath + ext;
            else
                return msra::strfun::wstrprintf(L"%s%s.%d", m_modelPath.c_str(), ext.c_str(), (int)epoch1Base);
        }

        void TrainEncoderDecoderModel(int startEpoch, ComputationNetworkPtr encoderNet,
            ComputationNetworkPtr decoderNet,
            IDataReader<ElemType>* encoderTrainSetDataReader,
            IDataReader<ElemType>* decoderTrainSetDataReader,
            IDataReader<ElemType>* encoderValidationSetDataReader,
            IDataReader<ElemType>* decoderValidationSetDataReader)
        {
            std::vector<ComputationNodeBasePtr>& encoderFeatureNodes = encoderNet->FeatureNodes();
            std::vector<ComputationNodeBasePtr>& encoderEvaluationNodes = encoderNet->OutputNodes();

            std::vector<ComputationNodeBasePtr>& decoderFeatureNodes = decoderNet->FeatureNodes();
            std::vector<ComputationNodeBasePtr>& decoderLabelNodes = decoderNet->LabelNodes();
            std::vector<ComputationNodeBasePtr>& decoderCriterionNodes = GetTrainCriterionNodes(*decoderNet);
            std::vector<ComputationNodeBasePtr>& decoderEvaluationNodes = GetEvalCriterionNodes(*decoderNet);

            std::map<std::wstring, Matrix<ElemType>*> encoderInputMatrices, decoderInputMatrices;
            for (size_t i = 0; i<encoderFeatureNodes.size(); i++)
                encoderInputMatrices[encoderFeatureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(encoderFeatureNodes[i])->FunctionValues();
            for (size_t i = 0; i<decoderFeatureNodes.size(); i++)
                decoderInputMatrices[decoderFeatureNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(decoderFeatureNodes[i])->FunctionValues();
            for (size_t i = 0; i<decoderLabelNodes.size(); i++)
                decoderInputMatrices[decoderLabelNodes[i]->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(decoderLabelNodes[i])->FunctionValues();

            //initializing weights and gradient holder
            std::list<ComputationNodeBasePtr> & encoderLearnableNodes = encoderNet->LearnableNodes(encoderEvaluationNodes[0]);  //only one criterion so far TODO: support multiple ones?
            std::list<ComputationNodeBasePtr> & decoderLearnableNodes = decoderNet->LearnableNodes(decoderCriterionNodes[0]);
            std::list<ComputationNodeBasePtr> learnableNodes;
            for (auto nodeIter = encoderLearnableNodes.begin(); nodeIter != encoderLearnableNodes.end(); nodeIter++)
                learnableNodes.push_back(*nodeIter);
            for (auto nodeIter = decoderLearnableNodes.begin(); nodeIter != decoderLearnableNodes.end(); nodeIter++)
                learnableNodes.push_back(*nodeIter);

            std::list<Matrix<ElemType>> smoothedGradients;
            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
            {
                ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                smoothedGradients.push_back(Matrix<ElemType>(node->GetNumRows(), node->GetNumCols(), node->FunctionValues().GetDeviceId()));
            }

            vector<double> epochCriterion;
            double avgCriterion, prevCriterion;
            for (size_t i = 0; i < 2; i++)
                epochCriterion.push_back(std::numeric_limits<double>::infinity());
            avgCriterion = prevCriterion = std::numeric_limits<double>::infinity();

            size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

            std::vector<double> epochEvalErrors(decoderEvaluationNodes.size(), std::numeric_limits<double>::infinity());

            std::vector<wstring> evalNodeNames;
            for (size_t i = 0; i<decoderEvaluationNodes.size(); i++)
                evalNodeNames.push_back(decoderEvaluationNodes[i]->NodeName());

            size_t totalSamplesSeen = 0;
            double learnRatePerSample = 0.5f / m_mbSize[startEpoch];

            int m_numPrevLearnRates = 5; //used to control the upper learnining rate in LR search to reduce computation
            vector<double> prevLearnRates;
            prevLearnRates.resize(m_numPrevLearnRates);
            for (int i = 0; i<m_numPrevLearnRates; i++)
                prevLearnRates[i] = std::numeric_limits<double>::infinity();

            //precompute mean and invStdDev nodes and save initial model
            if (/// to-do doesn't support pre-compute such as MVN here 
                /// PreCompute(net, encoderTrainSetDataReader, encoderFeatureNodes, encoderlabelNodes, encoderInputMatrices) || 
                startEpoch == 0)
            {
                encoderNet->SaveToFile(GetEncoderModelNameForEpoch(int(startEpoch) - 1));
                decoderNet->SaveToFile(GetDecoderModelNameForEpoch(int(startEpoch) - 1));
            }

            bool learnRateInitialized = false;
            if (startEpoch > 0)
                learnRateInitialized = this->LoadCheckPointInfo(startEpoch - 1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, m_prevChosenMinibatchSize);

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && !learnRateInitialized && m_learningRatesParam.size() <= startEpoch)
                InvalidArgument("When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, or an explicit learning rate must be specified in config for the starting epoch.");

            ULONG dropOutSeed = 1;
            double prevDropoutRate = 0;

            bool learnRateReduced = false;

            for (int i = int(startEpoch); i<int(m_maxEpochs); i++)
            {
                auto t_start_epoch = clock();

                //set dropout rate
                ComputationNetwork::SetDropoutRate<ElemType>(*encoderNet, encoderEvaluationNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);
                ComputationNetwork::SetDropoutRate<ElemType>(*decoderNet, decoderCriterionNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);

                //learning rate adjustment
                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || (m_learningRatesParam.size() > 0 && m_learningRatesParam.size() > i))
                {
                    learnRatePerSample = GetLearningRatePerSample(i/*BUGBUG workaround:*/, encoderTrainSetDataReader->GetNumParallelSequences());
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

                TrainOneEpochEncoderDecoderWithHiddenStates(encoderNet, decoderNet, i, 
                    m_epochSize, encoderTrainSetDataReader,
                    decoderTrainSetDataReader, learnRatePerSample,
                    encoderFeatureNodes, encoderEvaluationNodes, &encoderInputMatrices,
                    decoderFeatureNodes, decoderLabelNodes, decoderCriterionNodes, decoderEvaluationNodes,
                    &decoderInputMatrices, learnableNodes, smoothedGradients,
                    epochCriterion, epochEvalErrors, totalSamplesSeen);


                auto t_end_epoch = clock();
                double epochTime = 1.0*(t_end_epoch - t_start_epoch) / (CLOCKS_PER_SEC);

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
                    fprintf(stderr, "Finished Epoch[%lu]: Criterion Node [%ls] Per Sample = %.8g\n", i + 1, decoderCriterionNodes[0]->NodeName().c_str(), epochCriterion[i + 1]);
                    for (size_t j = 0; j<epochEvalErrors.size(); j++)
                        fprintf(stderr, "Finished Epoch[%lu]: Evaluation Node [%ws] Per Sample = %.8g\n", i + 1, evalNodeNames[j].c_str(), epochEvalErrors[j]);
                }

                if (decoderValidationSetDataReader != decoderTrainSetDataReader && decoderValidationSetDataReader != nullptr &&
                    encoderValidationSetDataReader != encoderTrainSetDataReader && encoderValidationSetDataReader != nullptr)
                {
                    SimpleEvaluator<ElemType> evalforvalidation(*decoderNet);
                    vector<wstring> cvEncoderSetTrainAndEvalNodes;
                    cvEncoderSetTrainAndEvalNodes.push_back(encoderEvaluationNodes[0]->NodeName());

                    vector<wstring> cvDecoderSetTrainAndEvalNodes;
                    cvDecoderSetTrainAndEvalNodes.push_back(decoderCriterionNodes[0]->NodeName());
                    cvDecoderSetTrainAndEvalNodes.push_back(decoderEvaluationNodes[0]->NodeName());

                    vector<double> vScore = evalforvalidation.EvaluateEncoderDecoderWithHiddenStates(
                        encoderNet, decoderNet,
                        encoderValidationSetDataReader,
                        decoderValidationSetDataReader, cvEncoderSetTrainAndEvalNodes,
                        cvDecoderSetTrainAndEvalNodes, m_mbSize[i]);
                    fprintf(stderr, "Finished Epoch[%lu]: [Validation Set] Train Loss Per Sample = %.8g  EvalErr Per Sample = %.8g\n",
                        i + 1, vScore[0], vScore[1]);

                    epochCriterion[0] = vScore[0]; //the first one is the decoder training criterion.
                }

                bool loadedPrevModel = false;
                size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
                if (avgCriterion == std::numeric_limits<double>::infinity())
                    avgCriterion = epochCriterion[0];
                else
                    avgCriterion = ((epochsSinceLastLearnRateAdjust - 1 - epochsNotCountedInAvgCriterion)* avgCriterion + epochCriterion[0]) / (epochsSinceLastLearnRateAdjust - epochsNotCountedInAvgCriterion);

                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && m_learningRatesParam.size() <= i && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
                {
                    if (prevCriterion - avgCriterion < 0 && prevCriterion != std::numeric_limits<double>::infinity())
                    {
                        if (m_loadBestModel)
                        {
                            encoderNet->LoadPersistableParametersFromFile(GetEncoderModelNameForEpoch(i - 1),
                                false);
                            decoderNet->LoadPersistableParametersFromFile(GetDecoderModelNameForEpoch(i - 1),
                                m_validateAfterModelReloading);
                            encoderNet->ResetEvalTimeStamp();
                            decoderNet->ResetEvalTimeStamp();

                            size_t dummyMinibatchSize = 0;
                            this->LoadCheckPointInfo(i - 1,
                                /*out*/ totalSamplesSeen,
                                /*out*/ learnRatePerSample,
                                smoothedGradients,
                                /*out*/ prevCriterion,
                                /*out*/ dummyMinibatchSize);
                            fprintf(stderr, "Loaded the previous model which has better training criterion.\n");
                            loadedPrevModel = true;
                        }
                    }

                    if (m_continueReduce)
                    {
                        if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<double>::infinity())
                        {
                            if (learnRateReduced == false)
                            {
                                learnRateReduced = true;
                            }
                            else
                            {
                                decoderNet->SaveToFile(GetDecoderModelNameForEpoch(i, true));
                                encoderNet->SaveToFile(GetEncoderModelNameForEpoch(i, true));
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
                        if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<double>::infinity())
                        {

                            learnRatePerSample *= m_learnRateDecreaseFactor;
                            fprintf(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                        }
                        else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan*prevCriterion && prevCriterion != std::numeric_limits<double>::infinity())
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
                decoderNet->SaveToFile(GetDecoderModelNameForEpoch(i));
                encoderNet->SaveToFile(GetEncoderModelNameForEpoch(i));

                size_t dummyMinibatchSize = 0;
                this->LoadCheckPointInfo(i,
                    /*out*/ totalSamplesSeen,
                    /*out*/ learnRatePerSample,
                    smoothedGradients,
                    /*out*/ prevCriterion,
                    /*out*/ dummyMinibatchSize);

                if (!m_keepCheckPointFiles)
                    _wunlink(GetCheckPointFileNameForEpoch(i - 1).c_str());  //delete previous checkpiont file to save space

                if (learnRatePerSample < 1e-12)
                    fprintf(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n", learnRatePerSample);
            }
        }

        void TrainEncoderDecoderModel(int startEpoch, vector<ComputationNetworkPtr> nets,
            vector<IDataReader<ElemType>*> trainDataReader,
            vector<IDataReader<ElemType>*> validationDataReader)
        {
            size_t iNumNetworks = nets.size();
            vector<std::vector<ComputationNodeBasePtr>*> featureNodes;
            vector<std::vector<ComputationNodeBasePtr>*> outputNodes;
            vector<std::vector<ComputationNodeBasePtr>*> pairNodes;
            vector<std::vector<ComputationNodeBasePtr>*> labelNodes;
            vector<std::vector<ComputationNodeBasePtr>*>   criterionNodes;
            vector<std::vector<ComputationNodeBasePtr>*>   evaluationNodes;
            vector<std::map<std::wstring, Matrix<ElemType>*>*> inputMatrices;

            for (size_t i = 0; i < iNumNetworks; i++)
            {
                auto * featPtr = &nets[i]->FeatureNodes();
                auto * lablPtr = &nets[i]->LabelNodes();
                featureNodes.push_back(featPtr);
                outputNodes.push_back(&nets[i]->OutputNodes());
                pairNodes.push_back(&nets[i]->PairNodes());

                labelNodes.push_back(lablPtr);
                criterionNodes.push_back(&GetTrainCriterionNodes(nets[i]));
                evaluationNodes.push_back(&GetEvalCriterionNodes(nets[i]));

                std::map<std::wstring, Matrix<ElemType>*> *matrices;
                matrices = new std::map<std::wstring, Matrix<ElemType>*>();

                for (size_t j = 0; j < featPtr->size(); j++)
                {
                    (*matrices)[(*featPtr)[j]->NodeName()] =
                        &(dynamic_pointer_cast<ComputationNode<ElemType>>((*featPtr)[j])->FunctionValues());
                }
                        
                for (size_t j = 0; j<lablPtr->size(); j++)
                {
                    (*matrices)[(*lablPtr)[j]->NodeName()] = 
                        &(dynamic_pointer_cast<ComputationNode<ElemType>>((*lablPtr)[j])->FunctionValues());
                }
                inputMatrices.push_back(matrices);
            }

            //initializing weights and gradient holder
            std::list<ComputationNodeBasePtr> learnableNodes;
            for (size_t i = 0; i < iNumNetworks; i++)
            {
                if (criterionNodes[i]->size() == 0)
                {
                    for (auto ptr = evaluationNodes[i]->begin(); ptr != evaluationNodes[i]->end(); ptr++)
                    {
                        ComputationNodeBasePtr pptr = *ptr;

                        std::list<ComputationNodeBasePtr> & eachLearnableNodes = nets[i]->LearnableNodes(pptr);  //only one criterion so far TODO: support multiple ones?
                        for (auto nodeIter = eachLearnableNodes.begin(); nodeIter != eachLearnableNodes.end(); nodeIter++)
                        {
                            ComputationNodeBasePtr node = *nodeIter;
                            learnableNodes.push_back(node);
                        }
                    }
                }
                else
                {
                    for (auto ptr = criterionNodes[i]->begin(); ptr != criterionNodes[i]->end(); ptr++)
                    {
                        ComputationNodeBasePtr pptr = *ptr;

                        std::list<ComputationNodeBasePtr> & eachLearnableNodes = nets[i]->LearnableNodes(pptr);  //only one criterion so far TODO: support multiple ones?
                        for (auto nodeIter = eachLearnableNodes.begin(); nodeIter != eachLearnableNodes.end(); nodeIter++)
                        {
                            ComputationNodeBasePtr node = *nodeIter;
                            learnableNodes.push_back(node);
                        }
                    }
                }

                for (auto ptr = pairNodes[i]->begin(); ptr != pairNodes[i]->end(); ptr++)
                    nets[i]->BuildAndValidateSubNetwork(*ptr);
            }


            std::list<Matrix<ElemType>> smoothedGradients;
            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
            {
                ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                smoothedGradients.push_back(Matrix<ElemType>(node->GetNumRows(), node->GetNumCols(), node->FunctionValues().GetDeviceId()));
            }

            double epochCriterion, avgCriterion, prevCriterion;
            epochCriterion = std::numeric_limits<double>::infinity();
            avgCriterion = prevCriterion = std::numeric_limits<double>::infinity();

            size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

            size_t iNumEvaluations = 0;
            for (size_t i = 0; i < iNumNetworks; i++)
            {
                iNumEvaluations += evaluationNodes[i]->size();
            }
            std::vector<double> epochEvalErrors(iNumEvaluations, std::numeric_limits<double>::infinity());

            std::vector<wstring> evalNodeNames;
            for (size_t k = 0; k < iNumNetworks; k++)
            {
                for (auto ptr = evaluationNodes[k]->begin(); ptr != evaluationNodes[k]->end(); ptr++)
                    evalNodeNames.push_back((*ptr)->NodeName());
            }

            size_t totalSamplesSeen = 0;
            double learnRatePerSample = 0.5f / m_mbSize[startEpoch];

            int m_numPrevLearnRates = 5; //used to control the upper learnining rate in LR search to reduce computation
            vector<double> prevLearnRates;
            prevLearnRates.resize(m_numPrevLearnRates);
            for (int i = 0; i<m_numPrevLearnRates; i++)
                prevLearnRates[i] = std::numeric_limits<double>::infinity();

            //precompute mean and invStdDev nodes and save initial model
            if (/// to-do doesn't support pre-compute such as MVN here 
                /// PreCompute(net, encoderTrainSetDataReader, encoderFeatureNodes, encoderlabelNodes, encoderInputMatrices) || 
                startEpoch == 0)
            {
                for (size_t k = 0; k < iNumNetworks; k++)
                {
                    wstring tmpstr = msra::strfun::wstrprintf(L".%d", k);
                    nets[k]->SaveToFile(GetModelNameForEpoch(int(startEpoch) - 1, false, tmpstr));
                }
            }

            bool learnRateInitialized = false;
            if (startEpoch > 0)
            {
                size_t dummyMinibatchSize = 0;
                this->LoadCheckPointInfo(startEpoch - 1,
                    /*out*/ totalSamplesSeen,
                    /*out*/ learnRatePerSample,
                    smoothedGradients,
                    /*out*/ prevCriterion,
                    /*out*/ dummyMinibatchSize);
            }

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && !learnRateInitialized && m_learningRatesParam.size() <= startEpoch)
                InvalidArgument("When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, or an explicit learning rate must be specified in config for the starting epoch.");

            ULONG dropOutSeed = 1;
            double prevDropoutRate = 0;

            bool learnRateReduced = false;

            for (int i = int(startEpoch); i<int(m_maxEpochs); i++)
            {
                auto t_start_epoch = clock();

                //set dropout rate
                for (size_t k = 0; k < iNumNetworks; k++)
                {
                    if (evaluationNodes[k]->size() > 0)
                        ComputationNetwork::SetDropoutRate<ElemType>(nets[k], (*evaluationNodes[k])[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);
                    if (criterionNodes[k]->size() > 0)
                        ComputationNetwork::SetDropoutRate<ElemType>(nets[k], (*criterionNodes[k])[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);
                }

                //learning rate adjustment
                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || (m_learningRatesParam.size() > 0 && m_learningRatesParam.size() > i))
                {
                    learnRatePerSample = GetLearningRatePerSample(i/*BUGBUG workaround:*/, trainDataReader[0]->GetNumParallelSequences());
                }
                else if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
                {
                    NOT_IMPLEMENTED;
                }

                learnRateInitialized = true;

                if (learnRatePerSample < m_minLearnRate)
                {
                    fprintf(stderr, "Learn Rate Per Sample for Epoch[%d] = %.8g is less than minLearnRate %.8g. Training stops.\n", i + 1, learnRatePerSample, m_minLearnRate);
                    break;
                }

                TrainOneEpochEncoderDecoderWithHiddenStates(i, m_epochSize, nets,
                    trainDataReader,
                    featureNodes,
                    pairNodes,
                    evaluationNodes,
                    inputMatrices,
                    labelNodes,
                    criterionNodes,
                    learnableNodes,
                    learnRatePerSample,
                    smoothedGradients,
                    epochCriterion, epochEvalErrors, totalSamplesSeen);


                auto t_end_epoch = clock();
                double epochTime = 1.0*(t_end_epoch - t_start_epoch) / (CLOCKS_PER_SEC);

                /**
                this is hacky. Only allow evaluatio on the first encoder->decoder pair
                */
                size_t decoderIdx = iNumNetworks - 1;
                IDataReader<ElemType>* decoderValidationSetDataReader = validationDataReader[decoderIdx];
                IDataReader<ElemType>* decoderTrainSetDataReader = trainDataReader[decoderIdx];
                ComputationNetworkPtr decoderNet = nets[decoderIdx];

                fprintf(stderr, "Finished Epoch[%d]: [Training Set] Decoder Train Loss Per Sample = %.8g    ", i + 1, epochCriterion);
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
                    fprintf(stderr, "Finished Epoch[%d]: Criterion Node Per Sample = %.8g\n", i + 1, epochCriterion);
                    for (size_t j = 0; j<epochEvalErrors.size(); j++)
                        fprintf(stderr, "Finished Epoch[%d]: Evaluation Node [%ls] Per Sample = %.8g\n", i + 1, evalNodeNames[j].c_str(), epochEvalErrors[j]);
                }

                if (decoderValidationSetDataReader != decoderTrainSetDataReader && decoderValidationSetDataReader != nullptr)
                {
                    MultiNetworksEvaluator<ElemType> evalforvalidation(decoderNet);

                    double vScore = evalforvalidation.EvaluateEncoderDecoderWithHiddenStates(
                        nets,
                        validationDataReader,
                        m_mbSize[i]);

                    fprintf(stderr, "Finished Epoch[%d]: [Validation Set] Loss Per Sample = %.8g \n ", i+1, vScore );

                    epochCriterion = vScore; 
                }

                bool loadedPrevModel = false;
                size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
                if (avgCriterion == std::numeric_limits<double>::infinity())
                    avgCriterion = epochCriterion;
                else
                    avgCriterion = ((epochsSinceLastLearnRateAdjust - 1 - epochsNotCountedInAvgCriterion)* avgCriterion + epochCriterion) / (epochsSinceLastLearnRateAdjust - epochsNotCountedInAvgCriterion);

                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && m_learningRatesParam.size() <= i && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
                {
                    if (prevCriterion - avgCriterion < 0 && prevCriterion != std::numeric_limits<double>::infinity())
                    {
                        if (m_loadBestModel)
                        {
                            //persist model and check-point info
                            for (size_t k = 0; k < iNumNetworks; k++)
                            {
                                nets[k]->LoadPersistableParametersFromFile(GetModelNameForEpoch(i, false, msra::strfun::wstrprintf(L".%d", k)), false);
                                nets[k]->ResetEvalTimeStamp();
                            }

                            size_t dummyLr = 0;
                            this->LoadCheckPointInfo(i - 1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, dummyLr);
                            fprintf(stderr, "Loaded the previous model which has better training criterion.\n");
                            loadedPrevModel = true;
                        }
                    }

                    if (m_continueReduce)
                    {
                        if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<double>::infinity())
                        {
                            if (learnRateReduced == false)
                            {
                                learnRateReduced = true;
                            }
                            else
                            {
                                //persist model and check-point info
                                for (size_t k = 0; k < iNumNetworks; k++)
                                {
                                    nets[k]->SaveToFile(GetModelNameForEpoch(i, true, msra::strfun::wstrprintf(L".%d", k)));
                                }
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
                        if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<double>::infinity())
                        {

                            learnRatePerSample *= m_learnRateDecreaseFactor;
                            fprintf(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                        }
                        else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan*prevCriterion && prevCriterion != std::numeric_limits<double>::infinity())
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
                for (size_t k = 0; k < iNumNetworks; k++)
                {
                    nets[k]->SaveToFile(GetModelNameForEpoch(i, false, msra::strfun::wstrprintf(L".%d", k)));
                }

                this->SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, 0);
                if (!m_keepCheckPointFiles)
                    _wunlink(GetCheckPointFileNameForEpoch(i - 1).c_str());  //delete previous checkpiont file to save space

                if (learnRatePerSample < 1e-12)
                    fprintf(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n", learnRatePerSample);
            }

            for (size_t i = 0; i < iNumNetworks; i++)
            {
                delete inputMatrices[i];
            }
        }

        /// use hidden states between encoder and decoder to communicate between two networks
        void TrainOneEpochEncoderDecoderWithHiddenStates(
            const int epochNumber,
            const size_t epochSize,
            vector<ComputationNetworkPtr> nets,  /// encoder network
            vector<IDataReader<ElemType>*> dataReader,
            vector<std::vector<ComputationNodeBasePtr>*> featureNodes,
            vector<std::vector<ComputationNodeBasePtr>*> pairNodes,
            vector<std::vector<ComputationNodeBasePtr>*> evaluationNodes,
            vector<std::map<std::wstring, Matrix<ElemType>*>*> inputMatrices,
            vector<std::vector<ComputationNodeBasePtr>*> labelNodes,
            vector<std::vector<ComputationNodeBasePtr>*> criterionNodes,
            const std::list<ComputationNodeBasePtr>& learnableNodes,
            const double learnRatePerSample,
            std::list<Matrix<ElemType>>& smoothedGradients,
            double& epochCriterion, std::vector<double>& epochEvalErrors, size_t& totalSamplesSeen)
        {
            ComputationNetworkPtr encoderNet = nets[0];
            ComputationNetworkPtr decoderNet = nets[1];
            DEVICEID_TYPE device = encoderNet->GetDeviceId();
            Matrix<ElemType> historyMat(device);

            double readTimeInMBs = 0, ComputeTimeInMBs = 0;
            double epochCriterionLastMBs = 0;

            int numSamplesLastMBs = 0;
            std::vector<double> epochEvalErrorsLastMBs(epochEvalErrors.size(), 0);

            clock_t startReadMBTime = 0, startComputeMBTime = 0;
            clock_t endReadMBTime = 0, endComputeMBTime = 0;

            //initialize statistics
            size_t totalEpochSamples = 0;

            int numMBsRun = 0;

            size_t numEvalNodes = epochEvalErrors.size();

            // NOTE: the following two local matrices are not used in PTask path
            Matrix<ElemType> localEpochCriterion(1, 2, decoderNet->GetDeviceId()); //assume only one training criterion node for each epoch
            Matrix<ElemType> localEpochEvalErrors(1, numEvalNodes, decoderNet->GetDeviceId());

            localEpochCriterion.SetValue(0);
            localEpochEvalErrors.SetValue(0);

            for (auto ptr = dataReader.begin(); ptr != dataReader.end(); ptr++)
            {
                (*ptr)->StartMinibatchLoop(m_mbSize[epochNumber], epochNumber, m_epochSize);
            }

            startReadMBTime = clock();

            size_t iNumNetworks = nets.size();

            unsigned uSeedForDataReader = epochNumber;

            bool bContinueDecoding = true;
            while (bContinueDecoding)
            {
                size_t i = 0;
                for (auto ptr = dataReader.begin(); ptr != dataReader.end(); ptr++, i++)
                {
                    IDataReader<ElemType>* pptr = (*ptr);
                    pptr->SetRandomSeed(uSeedForDataReader);
                    if (i == 0)
                        pptr->GetMinibatch(*(inputMatrices[i]));
                    else
                        if (pptr->GetMinibatch(*(inputMatrices[i])) == false)
                        {
                            bContinueDecoding = false;
                            break;
                        }
                }

                if (!bContinueDecoding)
                    break;

                size_t actualMBSize = decoderNet->DetermineActualMBSizeFromFeatures();
                if (actualMBSize == 0)
                    LogicError("decoderTrainSetDataReader read data but decoderNet reports no data read");


                for (size_t i = 0; i < iNumNetworks; i++)
                {
                    ComputationNetwork::UpdateEvalTimeStamps(*featureNodes[i]);
                    if (labelNodes[i]->size() > 0)
                        ComputationNetwork::UpdateEvalTimeStamps(*labelNodes[i]);
                }

                endReadMBTime = clock();
                startComputeMBTime = clock();

                /// not the sentence begining, because the initial hidden layer activity is from the encoder network
                //                    decoderTrainSetDataReader->SetSentenceBegin(false);
                //                    decoderTrainSetDataReader->CopyMBLayoutTo(decoderNet->m_mbLayout.m_sentenceBoundaryFlags);
                //                    decoderTrainSetDataReader->CopyMBLayoutTo(decoderNet->m_sentenceBegin);

                if (m_doGradientCheck)
                {
                    if (EncoderDecoderGradientCheck(nets,
                        dataReader,
                        evaluationNodes,
                        pairNodes,
                        featureNodes,
                        criterionNodes,
                        localEpochCriterion, localEpochEvalErrors) == false)
                    {
                        RuntimeError("SGD::TrainOneEpochEncoderDecoderWithHiddenStates gradient check not passed!");
                    }
                    localEpochCriterion.SetValue(0);
                    localEpochEvalErrors.SetValue(0);
                }

                EncoderDecoderWithHiddenStatesForwardPass(nets,
                    dataReader, pairNodes, evaluationNodes,
                    featureNodes, criterionNodes,
                    localEpochCriterion, localEpochEvalErrors);

                EncoderDecoderWithHiddenStatesErrorProp(nets, pairNodes, criterionNodes);

                //update model parameters
                if (learnRatePerSample > m_minLearnRate * 0.01)
                {
                    auto smoothedGradientIter = smoothedGradients.begin();
                    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, smoothedGradientIter++)
                    {
                        ComputationNodeBasePtr node = *nodeIter;
                        if (node->IsParameterUpdateRequired())
                        {
                            Matrix<ElemType>& smoothedGradient = (*smoothedGradientIter);

                            UpdateWeights(node, smoothedGradient, learnRatePerSample, GetMomentumPerSample(epochNumber/*BUGBUG workaround:*/, dataReader[0]->GetNumParallelSequences()), actualMBSize, m_L2RegWeight, m_L1RegWeight, m_needAveMultiplier);
                        }
                    }
                }

                endComputeMBTime = clock();
                numMBsRun++;
                if (m_traceLevel > 0)
                {
                    double MBReadTime = (double)(endReadMBTime - startReadMBTime) / (CLOCKS_PER_SEC);
                    double MBComputeTime = (double)(endComputeMBTime - startComputeMBTime) / CLOCKS_PER_SEC;

                    readTimeInMBs += MBReadTime;
                    ComputeTimeInMBs += MBComputeTime;
                    numSamplesLastMBs += int(actualMBSize);

                    if (numMBsRun % m_numMBsToShowResult == 0)
                    {

                        epochCriterion = localEpochCriterion.Get00Element();
                        for (size_t i = 0; i< numEvalNodes; i++)
                            epochEvalErrors[i] = (const double)localEpochEvalErrors(0, i);

                        double llk = (epochCriterion - epochCriterionLastMBs) / numSamplesLastMBs;
                        double ppl = exp(llk);
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
                for (auto ptr = dataReader.begin(); ptr != dataReader.end(); ptr++)
                {
                    (*ptr)->DataEnd(endDataSentence);
                }

                uSeedForDataReader++;
            }

            localEpochCriterion /= float(totalEpochSamples);
            localEpochEvalErrors /= float(totalEpochSamples);

            epochCriterion = localEpochCriterion.Get00Element();
            for (size_t i = 0; i < numEvalNodes; i++)
            {
                epochEvalErrors[i] = localEpochEvalErrors(0, i);
            }
            fprintf(stderr, "total samples in epoch[%d] = %zd\n", epochNumber, totalEpochSamples);
        }

        bool EncoderDecoderGradientCheck(
            vector<ComputationNetworkPtr> nets,  /// encoder network
            vector<IDataReader<ElemType>*> dataReader,
            vector<std::vector<ComputationNodeBasePtr>*> evaluationNodes,
            vector<std::vector<ComputationNodeBasePtr>*> pairNodes,
            vector<std::vector<ComputationNodeBasePtr>*> featureNodes,
            vector<std::vector<ComputationNodeBasePtr>*> criterionNodes,
            Matrix<ElemType>& localEpochCriterion,
            Matrix<ElemType>& localEpochEvalErrors
            )
        {
            size_t iNumNetworks = nets.size();
            vector<string> verror_msgs;
            DEVICEID_TYPE deviceId;

            for (int i = iNumNetworks - 1; i >= 0; i--)
            {
                /// check decoder learnable parameters
                std::list<ComputationNodeBasePtr> & learnableNodes =
                    (evaluationNodes[i]->size() == 0 && pairNodes[i]->size() > 0) ?
                        nets[i]->LearnableNodes((*pairNodes[i])[0])
                        : nets[i]->LearnableNodes((*evaluationNodes[i])[0]);

                for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                {
                    ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);

                    for (size_t itry = 0; itry < min((size_t)10, node->FunctionValues().GetNumElements()); itry++)
                    {

                        int irow = (int)fmod(rand(), node->GetNumRows() - 1);
                        int icol = (int)fmod(rand(), node->GetNumCols() - 1);
                        irow = max(0, irow);
                        icol = max(0, icol);

                        fprintf(stderr, "\n###### d%ls######\n", node->NodeName().c_str());
                        deviceId = node->FunctionValues().GetDeviceId();  // original device id

                        node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                        double eOrg = node->FunctionValues()(irow, icol);  /// warning :: this function will put matrix into CPU
                        node->FunctionValues().TransferToDeviceIfNotThere(deviceId, true);

                        /// perturb parameter
                        double ePos = eOrg + EPSILON;
                        node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                        node->FunctionValues().SetValue(irow, icol, (ElemType)ePos);
                        node->FunctionValues().TransferToDeviceIfNotThere(deviceId, true);

                        node->UpdateEvalTimeStamp();
                        localEpochCriterion.SetValue(0);
                        localEpochEvalErrors.SetValue(0);

                        EncoderDecoderWithHiddenStatesForwardPass(nets,
                            dataReader, pairNodes, evaluationNodes,
                            featureNodes, criterionNodes, 
                            localEpochCriterion, localEpochEvalErrors);

                        double score1 = localEpochCriterion.Get00Element();

                        double eNeg = eOrg - EPSILON;
                        node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                        node->FunctionValues().SetValue(irow, icol, (ElemType)eNeg);
                        node->FunctionValues().TransferToDeviceIfNotThere(deviceId, true);
                        node->UpdateEvalTimeStamp();
                        localEpochCriterion.SetValue(0);
                        localEpochEvalErrors.SetValue(0);

                        EncoderDecoderWithHiddenStatesForwardPass(nets,
                            dataReader, pairNodes, evaluationNodes,
                            featureNodes, criterionNodes, 
                            localEpochCriterion, localEpochEvalErrors);

                        double score1r = localEpochCriterion.Get00Element();

                        double grdNum = (score1r - score1) / (eNeg - ePos);

                        node->FunctionValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                        node->FunctionValues().SetValue(irow, icol, (ElemType)eOrg);
                        node->FunctionValues().TransferToDeviceIfNotThere(deviceId, true);
                        node->UpdateEvalTimeStamp();
                        localEpochCriterion.SetValue(0);
                        localEpochEvalErrors.SetValue(0);

                        EncoderDecoderWithHiddenStatesForwardPass(nets,
                            dataReader, pairNodes, evaluationNodes,
                            featureNodes, criterionNodes, 
                            localEpochCriterion, localEpochEvalErrors);

                        EncoderDecoderWithHiddenStatesErrorProp(nets, pairNodes, criterionNodes);

                        node->GradientValues().TransferFromDeviceToDevice(deviceId, CPUDEVICE, true, false, false);
                        double grdErr = node->GradientValues()(irow, icol);
                        node->GradientValues().TransferToDeviceIfNotThere(deviceId, true);

                        // check if they are consistent
                        double threshold = pow(10.0, max(0.0, ceil(log10(min(fabs(grdErr), fabs(grdNum))))) - (int)m_gradientCheckSigDigit);
                        double diff = fabs(grdErr - grdNum);
                        bool wrong = (std::isnan(diff) || diff > threshold);
                        if (wrong)
                        {
                            char serr[2048];
                            sprintf((char*)serr, "Decoder %ls Numeric gradient = %e, Error BP gradient = %e", node->NodeName().c_str(), static_cast<double>(grdNum), static_cast<double>(grdErr));
                            fprintf(stdout, "%s\n", serr);
                            verror_msgs.push_back(serr);
                        }
                    }
                }
            }

            if (verror_msgs.size() > 0)
                return false;
            return true;
        }

        void EncoderDecoderWithHiddenStatesForwardPass(
            vector<ComputationNetworkPtr> & nets, // TODO: should these vectors all be refs?
            vector<IDataReader<ElemType>*> & dataReader,
            vector<vector<ComputationNodeBasePtr>*> & pairNodes,
            vector<vector<ComputationNodeBasePtr>*> & evaluationNodes,
            vector<vector<ComputationNodeBasePtr>*> & /*featureNodes*/,
            vector<vector<ComputationNodeBasePtr>*> & criterionNodes,
            Matrix<ElemType>& localEpochCriterion,
            Matrix<ElemType>& localEpochEvalErrors
            )
        {
            size_t iNumNetworks = nets.size();

            for (size_t i = 0; i < iNumNetworks - 1; i++)
            {
                size_t j = i + 1;

                EncoderDecoderWithHiddenStatesForwardPass(nets[i], nets[j],
                    dataReader[i], dataReader[j],
                    *pairNodes[i],
                    *criterionNodes[j],
                    *evaluationNodes[j],
                    *pairNodes[j],
                    localEpochCriterion, localEpochEvalErrors);
            }
        }

        void EncoderDecoderWithHiddenStatesForwardPass(
            ComputationNetworkPtr encoderNet,  /// encoder network
            ComputationNetworkPtr decoderNet,
            IDataReader<ElemType>* encoderTrainSetDataReader,
            IDataReader<ElemType>* decoderTrainSetDataReader,
            vector<ComputationNodeBasePtr>& encoderEvaluationNodes,
            vector<ComputationNodeBasePtr>& decoderCriterionNodes,
            vector<ComputationNodeBasePtr>& decoderEvaluationNodes,
            vector<ComputationNodeBasePtr>& decoderPairNodes,
            Matrix<ElemType>& localEpochCriterion,
            Matrix<ElemType>& localEpochEvalErrors
            )
        {
            //encoderNet->SetActualMiniBatchSizeFromFeatures();
            encoderTrainSetDataReader->CopyMBLayoutTo(encoderNet->GetMBLayoutPtr());
            encoderNet->VerifyActualNumParallelSequences(encoderTrainSetDataReader->GetNumParallelSequences());

            encoderNet->Evaluate(encoderEvaluationNodes[0]);

            //decoderNet->SetActualMiniBatchSizeFromFeatures();
            decoderTrainSetDataReader->CopyMBLayoutTo(decoderNet->GetMBLayoutPtr());
            decoderNet->VerifyActualNumParallelSequences(decoderTrainSetDataReader->GetNumParallelSequences());
            /// not the sentence begining, because the initial hidden layer activity is from the encoder network

            if (decoderCriterionNodes.size() == 0 && decoderEvaluationNodes.size() == 0)
            {
                decoderNet->Evaluate(decoderPairNodes[0]);
            }
            else
            {
                decoderNet->Evaluate(decoderCriterionNodes[0]);

                Matrix<ElemType>::AddElementToElement(dynamic_pointer_cast<ComputationNode<ElemType>>(decoderCriterionNodes[0])->FunctionValues(), 0, 0, localEpochCriterion, 0, 0);

                size_t numEvalNodes = decoderEvaluationNodes.size();
                std::vector<double>mbEvalErrors(numEvalNodes, 0);

                for (size_t i = 0; i < numEvalNodes; i++)
                {
                    decoderNet->Evaluate(decoderEvaluationNodes[i]);
                    Matrix<ElemType>::AddElementToElement(dynamic_pointer_cast<ComputationNode<ElemType>>(decoderEvaluationNodes[i])->FunctionValues(), 0, 0, localEpochEvalErrors, 0, i);
                }
#ifdef DEBUG_DECODER
                fprintf(stderr, "ForwardPass score = %.8e\n", localEpochCriterion.Get00Element());
#endif
            }
        }

        void EncoderDecoderWithHiddenStatesErrorProp(
            vector<ComputationNetworkPtr> networks,  /// encoder network
            vector<std::vector<ComputationNodeBasePtr>*> pairNodes,
            vector<std::vector<ComputationNodeBasePtr>*> criterionNodes)
        {
            /**
            the networks are organized in the forward pass
            */
            size_t inetworks = networks.size();
            if (inetworks != criterionNodes.size())
                LogicError("EncoderDecoderWithHiddenStatesErrorProp: number of networks should be the same size as the number of criteron nodes.");

            for (size_t i = 0; i < pairNodes.size(); i++)
            {
                for (auto ptr = pairNodes[i]->begin(); ptr != pairNodes[i]->end(); ptr++)
                    networks[i]->ClearGradientForAllNodes(*ptr);
            }

            for (size_t i = 0; i < criterionNodes.size(); i++)
            {
                for (auto ptr = criterionNodes[i]->begin(); ptr != criterionNodes[i]->end(); ptr++)
                    networks[i]->ClearGradientForAllNodes(*ptr);
            }

            for (auto ptr = criterionNodes[inetworks - 1]->begin(); ptr != criterionNodes[inetworks - 1]->end(); ptr++)
            {
                if (ptr == criterionNodes[inetworks - 1]->begin())
                    networks[inetworks - 1]->ComputeGradient<ElemType>(*ptr); 
                else
                    networks[inetworks - 1]->ComputeGradient<ElemType>(*ptr, false, nullptr, false);
            }

            for (int i = inetworks - 2; i >= 0; i--)
            {
                if (criterionNodes[i]->size() > 0)
                {
                    /// has criterion
                    /// no need to compute gradients from pairnodes, because the gradients are added from pair nodes already
                    for (auto ptr = criterionNodes[i]->begin(); ptr != criterionNodes[i]->end(); ptr++)
                    {
                        networks[i]->ComputeGradient<ElemType>(*ptr, true, nullptr, false);
                    }
                }
                else if (pairNodes[i]->size() > 0)
                {
                    /// no criterion, so use pair-node gradients
                    for (auto ptr = pairNodes[i]->begin(); ptr != pairNodes[i]->end(); ptr++)
                    {
                        networks[i]->ComputeGradient<ElemType>(*ptr, false, nullptr, false);
                    }
                }
            }
        }
    };

}}}
