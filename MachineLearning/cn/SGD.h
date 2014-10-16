//
// <copyright file="SGD.h" company="Microsoft">
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
#include <Windows.h>
#include <WinBase.h>
#include <Windows.h>
#include <chrono> 
#include <random>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    enum class LearningRateSearchAlgorithm : int
    {
        None,
        AdjustAfterEpoch,
        SearchBeforeEpoch
    };

    enum class AdaptationRegType : int
    {
        None,
        KL
    };

    enum class GradientsUpdateType : int 
    {
        None,
        AdaGrad,
        RmsProp
    };
    
    typedef struct stGradientUpdateInfo{
        GradientsUpdateType mType;
        float mGaussianNoiseInjectStd;
        stGradientUpdateInfo()
        {
            mType = GradientsUpdateType::AdaGrad;
            mGaussianNoiseInjectStd = 0.0075f;
        }
    }GradientUpdateInfo;

    template<class ElemType>
    class SGD : ComputationNetworkHelper<ElemType>
    {
	protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;
        typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

    public:
        SGD(const ConfigParameters& configSGD)
        {
            ConfigArray learningRatesPerMBStr = configSGD("learningRatesPerMB", "");
            floatargvector learningRatesPerMB = learningRatesPerMBStr;

            ConfigArray learningRatesPerSampleStr = configSGD("learningRatesPerSample", "");
            floatargvector learningRatesPerSample = learningRatesPerSampleStr;

            std::string executionEngineValue = configSGD("executionEngine", "synchronous");

#ifdef USE_PTASK
            // use PTask if we have more than one GPU or the MultiGPU flag is set
            bool usePtask = (g_bestGpu != NULL && g_bestGpu->UseMultiple()) || (bool)configSGD("MultiGPU", "false");
#else
            bool usePtask = false;
#endif
            // AutoAdjust Parameters
            ConfigParameters configAALR=configSGD("AutoAdjust","");
            LearningRateSearchAlgorithm autoAdjustLRType = ParseLearningRateSearchType(configAALR("autoAdjustLR", "None"));
            ElemType reduceLearnRateIfImproveLessThan = configAALR("reduceLearnRateIfImproveLessThan", "0");
            bool continueReduce = (bool)configAALR("continueReduce", "false");
            ElemType learnRateDecreaseFactor = configAALR("learnRateDecreaseFactor", "0.618");
            ElemType increaseLearnRateIfImproveMoreThan = configAALR("increaseLearnRateIfImproveMoreThan", "1#INF");// std::numeric_limits<ElemType>::infinity());
            ElemType learnRateIncreaseFactor = configAALR("learnRateIncreaseFactor", "1.382");
            ConfigArray minibatch4LRSearch = configAALR("numMiniBatch4LRSearch", "500");
            intargvector numMiniBatch4LRSearch = minibatch4LRSearch;
            size_t numPrevLearnRates = configAALR("numPrevLearnRates", "5");
            size_t numBestSearchEpoch = configAALR("numBestSearchEpoch", "1");
            bool loadBestModel = configAALR("loadBestModel", "true");

            ConfigArray minibatchSize = configSGD("minibatchSize", "256");
            intargvector mbSize = minibatchSize;
            size_t epochSize = configSGD("epochSize", "0");

            size_t maxEpochs = configSGD("maxEpochs");
            ConfigArray momentumPerMBStr = configSGD("momentumPerMB", "");
			floatargvector momentumPerMB = momentumPerMBStr;

            wstring modelPath = configSGD("modelPath");
            wstring trainCriterionNodeName = configSGD("trainCriterionNodeName", "");
            wstring evalCriterionNodeName = configSGD("evalCriterionNodeName", "");

            size_t maxTempMemSizeInSamplesForCNN = configSGD("maxTempMemSizeInSamplesForCNN", "0");

            int traceLevel = configSGD("traceLevel", "0");
            size_t numMBsToShowResult = configSGD("numMBsToShowResult", "10");

            bool keepCheckPointFiles = configSGD("keepCheckPointFiles", "false");

            bool gradientClippingWithTruncation = configSGD("gradientClippingWithTruncation", "true");
            ElemType clippingThresholdPerSample = configSGD("clippingThresholdPerSample", "1#INF"); // std::numeric_limits<ElemType>::infinity());

            ConfigArray dropoutRatesStr = configSGD("dropoutRate", "0.0");
            floatargvector dropoutRates = dropoutRatesStr;

            GradientUpdateInfo gUpdateInfo; 
            GradientsUpdateType gradUpdateType = ParseGradUpdateType(configSGD("gradUpdateType", "None"));
            ElemType gaussianNoiseInjecStd = configSGD("gaussianNoiseInjectStd", "0");
            gUpdateInfo.mType = gradUpdateType;
            gUpdateInfo.mGaussianNoiseInjectStd = (float)gaussianNoiseInjecStd;
            
            /// for backward support. future setup should use gradUpdateType=AdaGrad, instead of 
            /// useAdagrad=true
            bool useAdagrad = configSGD("useAdagrad", "false");
            if (useAdagrad)
            {
                gradUpdateType = GradientsUpdateType::AdaGrad;
                gUpdateInfo.mType = gradUpdateType;
            }

            AdaptationRegType adaptationRegType = ParseAdaptationRegType(configSGD("adaptationRegType", "None"));
            ElemType adaptationRegWeight = configSGD("adaptationRegWeight", "0");

            /// gradient check setup
            bool doGradientCheck = configSGD("gradientcheck", "false");
            ElemType gradientCheckSigDigit = configSGD("sigFigs", "6");

            bool validateAfterModelReloading = configSGD("validateAfterModelReloading", "true");

            Init(learningRatesPerMB, learningRatesPerSample, mbSize, epochSize, maxEpochs, modelPath, momentumPerMB, gradientClippingWithTruncation, 
                clippingThresholdPerSample,autoAdjustLRType, increaseLearnRateIfImproveMoreThan, learnRateIncreaseFactor, 
                reduceLearnRateIfImproveLessThan, continueReduce, learnRateDecreaseFactor, dropoutRates,
                loadBestModel, numMiniBatch4LRSearch, numPrevLearnRates, numBestSearchEpoch, (UINT16)traceLevel, numMBsToShowResult,
                maxTempMemSizeInSamplesForCNN, gUpdateInfo, usePtask, keepCheckPointFiles, adaptationRegType, adaptationRegWeight,
                trainCriterionNodeName, evalCriterionNodeName, doGradientCheck, gradientCheckSigDigit, validateAfterModelReloading);
        }
	
		void setMomentum(float momentum)
		{
			m_momentumPerMB = (ElemType)momentum;
		}

        //autoLearnRateSearchType is applied only if the learning rate for the epoch is not specified in learningRatesPerMB and learningRatesPerSample
        void Init(const floatargvector& learningRatesPerMB, const floatargvector& learningRatesPerSample, const intargvector& mbSize, 
            const size_t epochSize, const size_t maxEpochs, 
			const wstring& modelPath, const floatargvector& momentumPerMB, const bool gradientClippingWithTruncation=true, 
            const ElemType clippingThresholdPerSample=std::numeric_limits<ElemType>::infinity(),
            const LearningRateSearchAlgorithm autoLearnRateSearchType = LearningRateSearchAlgorithm::None, 
            const ElemType increaseLearnRateIfImproveMoreThan = std::numeric_limits<ElemType>::infinity(), const ElemType learnRateIncreaseFactor = 1.382f,
            const ElemType reduceLearnRateIfImproveLessThan=0, const bool continueReduce=false, const ElemType learnRateDecreaseFactor = 0.618f, floatargvector dropoutRates = floatargvector(L"0.0f"),
            const bool loadBestModel=true, const intargvector& numMiniBatch4LRSearch=intargvector(L"500"), const size_t numPrevLearnRates = 5, 
            const size_t numBestSearchEpoch = 1, const UINT16 traceLevel = 0,
            const size_t numMBsToShowResult = 10, const size_t maxTempMemSizeInSamplesForCNN = 0,
            const GradientUpdateInfo gradUpdateType = GradientUpdateInfo(), const bool usePtask = false, const bool keepCheckPointFiles=false, const AdaptationRegType adaptationRegType = AdaptationRegType::None,
            const ElemType adaptationRegWeight = 0.0f, const wstring trainCriterionNodeName= L"", const wstring evalCriterionNodeName=L"",
            const bool doGradientCheck = false, const ElemType gradientCheckSigDigit = 6, const bool validateAfterModelReloading = true)
        {
            numPrevLearnRates;
            m_mbSize=mbSize;
            m_epochSize=epochSize;
            if (m_epochSize == 0)
            {
                m_epochSize = requestDataSize;
            }
            m_maxEpochs=maxEpochs;
     
            m_gradientClippingWithTruncation=gradientClippingWithTruncation;
            m_modelPath=modelPath;
            m_autoLearnRateSearchType=autoLearnRateSearchType;
            m_traceLevel=traceLevel;
            m_loadBestModel=loadBestModel;
            m_increaseLearnRateIfImproveMoreThan=increaseLearnRateIfImproveMoreThan;
            m_learnRateIncreaseFactor=learnRateIncreaseFactor;
            m_reduceLearnRateIfImproveLessThan=reduceLearnRateIfImproveLessThan;
             m_continueReduce=continueReduce;
            m_learnRateDecreaseFactor=learnRateDecreaseFactor;
            m_clippingThresholdPerSample=abs(clippingThresholdPerSample);
            m_numMiniBatch4LRSearch=numMiniBatch4LRSearch;
            m_dropoutRates=dropoutRates;
            m_numMBsToShowResult=int(numMBsToShowResult);
            m_numBestSearchEpoch=numBestSearchEpoch;
            m_maxTempMemSizeInSamplesForCNN=maxTempMemSizeInSamplesForCNN;
            m_gradType = gradUpdateType;
            m_usePtask = usePtask;
            m_keepCheckPointFiles = keepCheckPointFiles;

            m_adaptationRegType = adaptationRegType;
            m_adaptationRegWeight = adaptationRegWeight;

            m_trainCriterionNodeName = trainCriterionNodeName;
            m_evalCriterionNodeName = evalCriterionNodeName;

            for (size_t i=0; i<m_mbSize.size(); i++)
                if (m_epochSize != requestDataSize && m_epochSize < m_mbSize[i])
                    throw std::invalid_argument ("epoch size must be larger than mbsize.");

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None && (learningRatesPerSample.size() == 0 && learningRatesPerMB.size() == 0))
            {
                throw std::invalid_argument ("If autoLearnRateSearchType is false you must specify the learningRatesPerSample or learningRatesPerMB parameter.");
            }

            if (learningRatesPerSample.size() > 0 && learningRatesPerMB.size() > 0)
            {
                throw std::invalid_argument ("You specified both learningRatesPerSample and learningRatesPerMB. Please comment out one of them.");
            }
            else if (learningRatesPerSample.size() > 0)
            {
                m_learningRatesPerSample=learningRatesPerSample;
            }
            else if (learningRatesPerMB.size() > 0)
            {
                int LRSize = (int)max(learningRatesPerMB.size(), m_mbSize.size());
                m_learningRatesPerSample.resize(LRSize);
                for (int i=0; i<LRSize; i++)
                {
                    m_learningRatesPerSample[i] = learningRatesPerMB[i]/m_mbSize[i];
                }
            }
			m_momentumPerMB = 0.9f;
			if  (momentumPerMB.size() >0)
			{
				m_momentumInputPerMB=momentumPerMB;
		        if (m_momentumInputPerMB[0]>=1 || m_momentumInputPerMB[0]<0)
					throw std::invalid_argument ("momentumPerMB must be in [0, 1).");
			}

            if (m_learnRateDecreaseFactor > 1 || m_learnRateIncreaseFactor<1)
            {
                throw std::invalid_argument ("learnRateIncreaseFactor must be >= 1 and learnRateDecreaseFactor must be <= 1.");
            }

            for (size_t i=0; i<m_dropoutRates.size(); i++)
            {
                if (m_dropoutRates[i] >= 1 || m_dropoutRates[i] < 0)
                {
                    throw std::invalid_argument ("dropoutRate must be >= 0 and < 1.");
                }
            }

            if (m_adaptationRegWeight > 1 || m_adaptationRegWeight <0)
                throw invalid_argument("adaptationRegWeight must be in [0 1]");

            m_minLearnRate = 1e-9f;

            m_needRegularization = false;

            m_doGradientCheck = doGradientCheck;
            m_gradientCheckSigDigit = gradientCheckSigDigit;
            m_validateAfterModelReloading = validateAfterModelReloading;

            msra::files::make_intermediate_dirs (m_modelPath);
        }

        void Adapt(wstring origModelFileName, wstring refNodeName, IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader, const short deviceID, const bool makeMode = true)
        {
            if (origModelFileName == L"" || trainSetDataReader == nullptr)
                    throw std::invalid_argument ("origModel and trainSetDataReader should not be null.");

            int startEpoch = DetermineStartEpoch(makeMode);
            if (startEpoch == m_maxEpochs)
            {
                fprintf(stderr,"Final model exists. No further training is necessary.\n");
                return;
            }

            ComputationNetwork<ElemType> net(deviceID);
            if (startEpoch >= 0)
            {
                wstring modelFileName = GetModelNameForEpoch(int(startEpoch)-1);
                fprintf(stderr,"Starting from checkpoint. Load Network From File %ws.\n", modelFileName);
                net.LoadFromFile(modelFileName);
            }
            else
            {
                fprintf(stderr,"Load Network From the original model file %ws.\n", origModelFileName);
                net.LoadFromFile(origModelFileName);
            }

            startEpoch = max(startEpoch, 0);

            ComputationNetwork<ElemType> refNet(deviceID);
            m_needRegularization = m_adaptationRegType != AdaptationRegType::None && m_adaptationRegWeight > 0;
            if (m_needRegularization)
            {
                fprintf(stderr,"Load reference Network From the original model file %ws.\n", origModelFileName);
                refNet.LoadFromFile(origModelFileName);
            }

            ComputationNodePtr refNode = nullptr;
            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL)
            {
                fprintf(stderr,"Checkign refNodeName.\n", origModelFileName);
                if (refNodeName == L"")
                    throw invalid_argument("refNodeName does not exist and is needed when adaptationRegType is KL.");

                refNode = refNet.GetNodeFromName(refNodeName);
            }
            
            TrainOrAdaptModel(startEpoch, net, refNet, refNode, trainSetDataReader, validationSetDataReader);
        }

        void Train(IComputationNetBuilder<ElemType>* netBuilder, IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader, const bool makeMode = true)
        {
            if (netBuilder == nullptr || trainSetDataReader == nullptr)
                    throw std::invalid_argument ("netBuilder and trainSetDataReader should not be null.\n");

            int startEpoch = DetermineStartEpoch(makeMode);
            if (startEpoch == m_maxEpochs)
            {
                fprintf(stderr,"Final model exists. No further training is necessary.\n");
                return;
            }

            wstring modelFileName = GetModelNameForEpoch(int(startEpoch)-1);
            if (startEpoch >= 0)
                fprintf(stderr,"Starting from checkpoint. Load Network From File %ws.\n", modelFileName);
            ComputationNetwork<ElemType>& net  = 
                startEpoch<0? netBuilder->BuildNetworkFromDescription() : netBuilder->LoadNetworkFromFile(modelFileName);
            startEpoch = max(startEpoch, 0);
            m_needRegularization = false;

            TrainOrAdaptModel(startEpoch, net, net, nullptr, trainSetDataReader, validationSetDataReader);
        }

    protected:
        std::vector<ComputationNodePtr>  GetTrainCriterionNodes(ComputationNetwork<ElemType>& net)
        {
            fprintf(stderr, "GetTrainCriterionNodes %ls ...\n",  m_trainCriterionNodeName);
            if (!m_trainCriterionNodeName.empty())
            {
                std::vector<ComputationNodePtr> nodes;
                ComputationNodePtr node = net.GetNodeFromName(m_trainCriterionNodeName);
                net.ValidateNetwork(node);
                if (node->FunctionValues().GetNumElements() != 1)
                    throw invalid_argument("the trainCriterionNodeName specified in the config file is not a valid training criterion node.");

                nodes.push_back(node);
                return nodes;
            }
            else
                return net.FinalCriterionNodes();
        }
        std::vector<ComputationNodePtr>  GetEvalCriterionNodes(ComputationNetwork<ElemType>& net)
        {
            fprintf(stderr, "GetEvalCriterionNodes %ls ...\n",  m_evalCriterionNodeName);
            if (!m_evalCriterionNodeName.empty())
            {
                std::vector<ComputationNodePtr> nodes;
                ComputationNodePtr node = net.GetNodeFromName(m_evalCriterionNodeName);
                net.ValidateNetwork(node);
                if (node->FunctionValues().GetNumElements() != 1)
                    throw invalid_argument("the evalCriterionNodeName specified in the config file is not a valid evaluation criterion node.");

                nodes.push_back(node);
                return nodes;
            }
            else
                return net.EvaluationNodes();
        }

        void TrainOrAdaptModel(int startEpoch, ComputationNetwork<ElemType>& net, ComputationNetwork<ElemType>& refNet, ComputationNodePtr refNode,
            IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader)
        {
            std::vector<ComputationNodePtr> & FeatureNodes = net.FeatureNodes();
            std::vector<ComputationNodePtr> & labelNodes = net.LabelNodes();
            std::vector<ComputationNodePtr> criterionNodes = GetTrainCriterionNodes(net);
            std::vector<ComputationNodePtr> evaluationNodes = GetEvalCriterionNodes(net);

            std::map<std::wstring, Matrix<ElemType>*> inputMatrices;
            for (size_t i=0; i<FeatureNodes.size(); i++)
            {
                inputMatrices[FeatureNodes[i]->NodeName()] = &FeatureNodes[i]->FunctionValues();
            }
            for (size_t i=0; i<labelNodes.size(); i++)
            {
                inputMatrices[labelNodes[i]->NodeName()] = &labelNodes[i]->FunctionValues();
            }
            
            // special handling of classed based softmax node. Need a better solution to it.
            if (criterionNodes[0]->OperationName() == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
                evaluationNodes[0]->OperationName() == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName())
            {
                size_t vSz = FeatureNodes[0]->FunctionValues().GetNumRows();
                int deviceId = FeatureNodes[0]->FunctionValues().GetDeviceId();
                inputMatrices[L"idx2cls"] = new Matrix<ElemType>(vSz, 1, (short)deviceId); 
                inputMatrices[L"classinfo"] = new Matrix<ElemType>(vSz, 1, (short)deviceId);
            }


            //used for KLD regularized adaptation. For all other adaptation techniques use MEL to edit the model and using normal training algorithm
            std::vector<ComputationNodePtr> refFeatureNodes;
            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
            {
                refFeatureNodes.resize(FeatureNodes.size());
                for (size_t i=0; i<FeatureNodes.size(); i++)
                {
                    refFeatureNodes[i] = refNet.GetNodeFromName(FeatureNodes[i]->NodeName()); //we need to keep this info to handle deletion
                    refNet.ChangeNode(FeatureNodes[i]->NodeName(), FeatureNodes[i]); 
                }

                refNet.RebuildNetwork(refNode);
            }

            //initializing weights and gradient holder
            std::list<ComputationNodePtr>& learnableNodes = net.LearnableNodes(criterionNodes[0]);  //only one criterion so far TODO: support multiple ones?
            std::list<Matrix<ElemType>> smoothedGradients;

            for (auto nodeIter=learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                smoothedGradients.push_back(Matrix<ElemType>(node->FunctionValues().GetNumRows(), node->FunctionValues().GetNumCols(),net.GetDeviceID()));
            }

            ElemType epochCriterion = std::numeric_limits<ElemType>::infinity(), prevCriterion = std::numeric_limits<ElemType>::infinity();
			std::vector<ElemType> epochEvalErrors(evaluationNodes.size(),std::numeric_limits<ElemType>::infinity());
			
			std::vector<wstring> evalNodeNames;
			for (size_t i=0;i<evaluationNodes.size(); i++)
				evalNodeNames.push_back(evaluationNodes[i]->NodeName());

            size_t totalSamplesSeen = 0;
            ElemType learnRatePerSample = 0.5f / m_mbSize[startEpoch];

            int m_numPrevLearnRates = 5; //used to control the upper learnining rate in LR search to reduce computation
            vector<ElemType> prevLearnRates;
            prevLearnRates.resize(m_numPrevLearnRates);
            for (int i=0; i<m_numPrevLearnRates; i++)
                prevLearnRates[i] = std::numeric_limits<ElemType>::infinity();

            //precompute mean and invStdDev nodes and save initial model
            if (PreCompute(net,trainSetDataReader, FeatureNodes,labelNodes,inputMatrices) || startEpoch == 0)
            {
                net.SaveToFile(GetModelNameForEpoch(int(startEpoch)-1));
            }

            bool learnRateInitialized = false;
            if (startEpoch > 0)
			{
                learnRateInitialized = LoadCheckPointInfo(startEpoch-1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);  
				setMomentum(m_momentumInputPerMB[m_momentumInputPerMB.size()-1]);
			}

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && !learnRateInitialized && m_learningRatesPerSample.size() <= startEpoch)
                throw std::invalid_argument ("When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, or an explicit learning rate must be specified in config for the starting epoch.");

            ULONG dropOutSeed = 1;
            ElemType prevDropoutRate = 0;

            bool learnRateReduced = false;

            SetMaxTempMemSizeForCNN(net, criterionNodes[0], m_maxTempMemSizeInSamplesForCNN);
            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr) 
                SetMaxTempMemSizeForCNN(refNet, refNode, m_maxTempMemSizeInSamplesForCNN);

            // build the PTask graph if they want to use ptask
            // NOTE: the graph is currently only for training, so other operations will still use the usual method, 
            // (i.e rate adjustment and other custom operations still use the non PTask method)
            if (m_usePtask)
            {
                // set the minibatch size to the largest thing we will ever see
                int maxMbSize = 0;
                for (int val : m_mbSize)
                {
                    maxMbSize = max(val, maxMbSize);
                }
                net.SetActualMiniBatchSize(maxMbSize);
                net.BuildPTaskGraph();
            }

            for (int i=int(startEpoch); i<int(m_maxEpochs); i++)
            {
                auto t_start_epoch = clock();                

                // set other information to inputMatrices that can contrain information
                // used for class-based LM for clustring information
                SetOtherInfo(net, trainSetDataReader, validationSetDataReader, inputMatrices);

                //set dropout rate
                SetDropoutRate(net, criterionNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);
            
                //learning rate adjustment
                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || (m_learningRatesPerSample.size() > 0 && m_learningRatesPerSample.size() > i))
                {    
					learnRatePerSample = m_learningRatesPerSample[i]; 
					setMomentum(m_momentumInputPerMB[i]);
				}	
                else if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)    
                {
                    ElemType largestPrevLearnRatePerSample = prevLearnRates[0];
                    for (int j=1; j<m_numPrevLearnRates; j++)
                    {
                        largestPrevLearnRatePerSample = max(largestPrevLearnRatePerSample, prevLearnRates[j]);
                    }

                    //return a reasonable  learning rate based on the initial mbsize
                    learnRatePerSample = SearchLearnRateBeforeEpoch(net, refNet, refNode, i, learnRatePerSample, trainSetDataReader, FeatureNodes,
                            labelNodes,criterionNodes,evaluationNodes, inputMatrices,learnableNodes,smoothedGradients, learnRateInitialized, largestPrevLearnRatePerSample);

                    prevLearnRates[i % m_numPrevLearnRates] = learnRatePerSample;  //save per sample learn rate to support changeable mbsize
                }

                learnRateInitialized = true;

                if (learnRatePerSample < m_minLearnRate)
                {
                    fprintf(stderr,"Learn Rate Per Sample for Epoch[%lu] = %.8g is less than minLearnRate %.8g. Training stops.\n", i+1, learnRatePerSample, m_minLearnRate);
                    if (m_autoLearnRateSearchType != LearningRateSearchAlgorithm::None)
                        net.SaveToFile(m_modelPath);
                    break;
                }

                TrainOneEpoch(net, refNet, refNode, i, m_epochSize, trainSetDataReader, learnRatePerSample,FeatureNodes,labelNodes,
                    criterionNodes,evaluationNodes,inputMatrices, learnableNodes,smoothedGradients,
                    epochCriterion, epochEvalErrors, totalSamplesSeen);

                auto t_end_epoch = clock();
                ElemType epochTime = ElemType(1.0)*(t_end_epoch-t_start_epoch)/(CLOCKS_PER_SEC);

				fprintf(stderr,"Finished Epoch[%lu]: [Training Set] Train Loss Per Sample = %.8g    ", i+1, epochCriterion);
				if (epochEvalErrors.size()==1)
				{
					fprintf(stderr,"EvalErr Per Sample = %.8g   Ave Learn Rate Per Sample = %.10g  Epoch Time=%.8g\n", epochEvalErrors[0], learnRatePerSample, epochTime);
				}
				else
				{
					fprintf(stderr,"EvalErr Per Sample ");
					for (size_t j=0; j<epochEvalErrors.size(); j++)
						fprintf(stderr,"[%lu]=%.8g ", j, epochEvalErrors[j]);
					fprintf(stderr,"Ave Learn Rate Per Sample = %.10g  Epoch Time=%.8g\n",learnRatePerSample, epochTime);
  				    fprintf(stderr,"Finished Epoch[%lu]: Criterion Node [%ws] Per Sample = %.8g\n",i+1,criterionNodes[0]->NodeName().c_str() ,epochCriterion);
					for (size_t j=0; j<epochEvalErrors.size(); j++)
						fprintf(stderr,"Finished Epoch[%lu]: Evaluation Node [%ws] Per Sample = %.8g\n",i+1,evalNodeNames[j].c_str(),epochEvalErrors[j]);
				}

                if (validationSetDataReader != trainSetDataReader && validationSetDataReader != nullptr)
                {
                    SimpleEvaluator<ElemType> evalforvalidation(net);
                    vector<wstring> cvSetTrainAndEvalNodes;
                    cvSetTrainAndEvalNodes.push_back(criterionNodes[0]->NodeName());
                    cvSetTrainAndEvalNodes.push_back(evaluationNodes[0]->NodeName());

                    vector<ElemType> vScore = evalforvalidation.Evaluate(*validationSetDataReader, cvSetTrainAndEvalNodes, m_mbSize[i]);
  				    fprintf(stderr,"Finished Epoch[%lu]: [Validation Set] Train Loss Per Sample = %.8g  EvalErr Per Sample = %.8g\n",
                        i+1, vScore[0], vScore[1]);

                    epochCriterion = vScore[0]; //the first one is the training criterion.
                }

                bool loadedPrevModel = false;
                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && m_learningRatesPerSample.size() <= i)
                {
                    if (prevCriterion - epochCriterion < 0 && prevCriterion != std::numeric_limits<ElemType>::infinity())                    
                    {
                        if (m_loadBestModel)
                        {
                            net.LoadPersistableParametersFromFile(GetModelNameForEpoch(i-1), m_validateAfterModelReloading);
                            net.ResetEvalTimeStamp();
                            LoadCheckPointInfo(i-1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);  
                            fprintf(stderr,"Loaded the previous model which has better training criterion.\n");
                            loadedPrevModel = true;
                        }
                    }

                    if(m_continueReduce)
                    {
                        if (prevCriterion - epochCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                        {
                            if(learnRateReduced == false) 
                            {
                                learnRateReduced = true;                                
                            }
                            else 
                            {
                                net.SaveToFile(GetModelNameForEpoch(i, true));
                                fprintf(stderr,"Finished training and saved final model\n\n");
                                break;
                            }
                        }
                        if(learnRateReduced) 
                        {
                            learnRatePerSample *= m_learnRateDecreaseFactor;
                            fprintf(stderr,"learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                        }
                    }
                    else 
                    {
                        if (prevCriterion - epochCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                        {

                            learnRatePerSample *= m_learnRateDecreaseFactor;
                            fprintf(stderr,"learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                        }
                        else if (prevCriterion - epochCriterion > m_increaseLearnRateIfImproveMoreThan*prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                        {
                            learnRatePerSample *= m_learnRateIncreaseFactor;
                            fprintf(stderr,"learnRatePerSample increased to %.8g\n", learnRatePerSample);
                        }
                    }
                }

                if (!loadedPrevModel)  //not loading previous values then set them
                {
                    prevCriterion = epochCriterion;
                }

                //persist model and check-point info
                net.SaveToFile(GetModelNameForEpoch(i));
                SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion); 
                if (!m_keepCheckPointFiles)
                    DeleteFile(GetCheckPointFileNameForEpoch(i-1).c_str());  //delete previous checkpiont file to save space

                if (learnRatePerSample < 1e-12)
                    fprintf(stderr,"learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n", learnRatePerSample);
            }

            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr) //since we linked feature nodes. we need to remove it from the deletion
            {
                for (size_t i=0; i<refFeatureNodes.size(); i++)
                {
                    refNet.ChangeNode(refFeatureNodes[i]->NodeName(), refFeatureNodes[i]); //note we need to handle deletion carefully
                }
            }

            if (inputMatrices[L"classinfo"])
            {
                delete inputMatrices[L"classinfo"];
                inputMatrices.erase(L"classinfo");
            }
            if (inputMatrices[L"idx2cls"])
            {
                delete inputMatrices[L"idx2cls"];
                inputMatrices.erase(L"idx2cls");
            }

        }

	protected:

        //return true if precomputation is executed.
		bool PreCompute(ComputationNetwork<ElemType>& net,
            IDataReader<ElemType>* trainSetDataReader, 
            std::vector<ComputationNodePtr>& FeatureNodes,
            std::vector<ComputationNodePtr>& labelNodes,
            std::map<std::wstring, Matrix<ElemType>*>& inputMatrices)
        {
            std::list<ComputationNodePtr> nodes = net.GetNodesRequirePreComputation();
            
			if (nodes.size() == 0)
            {
				fprintf(stderr,"No PreCompute nodes found, skipping PreCompute step\n");
				return false;		
            }

			fprintf(stderr,"Found %d PreCompute nodes\n",nodes.size());
			for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
			{
				PreComputedNode<ElemType>* node = static_cast<PreComputedNode<ElemType>*> (*nodeIter);
				fprintf(stderr,"\tNodeName: %ws\n",(node->NodeName()).c_str());
			}			

            //compute
            //trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , requestDataSize); 
            trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , m_epochSize); // only based on one epoch

            while (trainSetDataReader->GetMinibatch(inputMatrices))
            {
                UpdateEvalTimeStamps(FeatureNodes);
                UpdateEvalTimeStamps(labelNodes);

                size_t actualMBSize = net.GetActualMBSize();
				net.SetActualMiniBatchSize(actualMBSize);
                for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
                {
                    net.Evaluate( *nodeIter);
                }
            }

            //mark done
            for (auto nodeIter=nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                PreComputedNode<ElemType>* node = static_cast<PreComputedNode<ElemType>*> (*nodeIter);
                node->MarkComputed(true);
            }

            return true;
        }

        //return a reasonable initial learning rate based on the initial mbsize
        ElemType SearchLearnRateBeforeEpoch(ComputationNetwork<ElemType>& net, ComputationNetwork<ElemType>& refNet, const ComputationNodePtr refNode, 
            const int epochNumber, const ElemType curLearnRate, 
            IDataReader<ElemType>* trainSetDataReader, 
            const std::vector<ComputationNodePtr>& FeatureNodes,
            const std::vector<ComputationNodePtr>& labelNodes,
            const std::vector<ComputationNodePtr>& criterionNodes,
            const std::vector<ComputationNodePtr>& evaluationNodes,
            std::map<std::wstring, Matrix<ElemType>*>& inputMatrices,
            const std::list<ComputationNodePtr>& learnableNodes,
            std::list<Matrix<ElemType>>& smoothedGradients, const bool /*learnRateInitialized*/, const ElemType largestPrevLearnRatePerSample)
        {
            ElemType epochCriterion = std::numeric_limits<ElemType>::infinity(), prevCriterion = std::numeric_limits<ElemType>::infinity();
			vector<ElemType> epochEvalErrors(evaluationNodes.size(),std::numeric_limits<ElemType>::infinity());
            //ElemType epochEvalError = std::numeric_limits<ElemType>::infinity();
            size_t totalSamplesSeen = 0;
            ElemType bestLearnRatePerSample = curLearnRate;

            size_t epochSize = m_numMiniBatch4LRSearch[epochNumber] * m_mbSize[epochNumber];
            if (m_epochSize != requestDataSize)
            {
                epochSize = min(epochSize, m_epochSize);  //use a small number minibatches to make decision
            }

            ElemType baseCriterion;

            ElemType minLearnRate = m_minLearnRate * 0.3f;
            ElemType learnRatePerSample = 1.0f / 8.0f / 0.618f /sqrt((ElemType)m_mbSize[epochNumber]);

            if (largestPrevLearnRatePerSample != std::numeric_limits<ElemType>::infinity())
                learnRatePerSample = largestPrevLearnRatePerSample / 0.618f / 0.618f;  //largestPrevLearnRatePerSample is per sample, first 0.618f is for compensation, second one is for safety

            int baseModelEpoch =  epochNumber-1;
            net.LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
            net.ResetEvalTimeStamp();

            ElemType learnRate =learnRatePerSample;
            LoadCheckPointInfo(baseModelEpoch, totalSamplesSeen, learnRate, smoothedGradients, prevCriterion);  

            //if model is not changed this is what we will get
            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, 0,
                FeatureNodes,labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                smoothedGradients, baseCriterion, epochEvalErrors, totalSamplesSeen);

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
            {
                if (prevCriterion == std::numeric_limits<ElemType>::infinity())
                    prevCriterion = baseCriterion;
                ElemType ratio = 0.3f;
                if (m_epochSize != requestDataSize)
                {
                    ratio = pow(((ElemType)epochSize) / m_epochSize, 1.0f/2);
                }
                baseCriterion = max(ratio * prevCriterion + (1-ratio) * baseCriterion, baseCriterion);
            }

            do
            {
                learnRatePerSample *= 0.618f;
                TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, learnRatePerSample,
                    FeatureNodes,labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                    smoothedGradients, epochCriterion, epochEvalErrors, totalSamplesSeen);

            } while (epochCriterion > baseCriterion && learnRatePerSample > minLearnRate);

            bestLearnRatePerSample =  learnRatePerSample;

            if (epochNumber < m_numBestSearchEpoch) //grid search for the first m_numBestSearchEpoch  epochs
            {
                ElemType leftLearnRatePerSample = 0.01f / m_mbSize[epochNumber], rightLearnRatePerSample = learnRatePerSample;
                ElemType leftCriterion, rightCriterion = epochCriterion;

                TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, leftLearnRatePerSample,
                    FeatureNodes,labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                    smoothedGradients, leftCriterion, epochEvalErrors, totalSamplesSeen);

                while (rightLearnRatePerSample > leftLearnRatePerSample * 1.2f)
                {   
                    if (rightCriterion > leftCriterion)
                    {
                        rightLearnRatePerSample *= 0.618f;

                        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, rightLearnRatePerSample,
                            FeatureNodes,labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                            smoothedGradients, rightCriterion, epochEvalErrors, totalSamplesSeen);
                    }
                    else
                    {
                        leftLearnRatePerSample /= 0.618f;

                        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, leftLearnRatePerSample,
                            FeatureNodes,labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                            smoothedGradients, leftCriterion, epochEvalErrors, totalSamplesSeen);
                    }
                }

                bestLearnRatePerSample =  (leftCriterion < rightCriterion)? leftLearnRatePerSample : rightLearnRatePerSample;
            }

            fprintf(stderr,"Best Learn Rate Per Sample for Epoch[%d] = %.10g  baseCriterion=%.10g\n", epochNumber+1, bestLearnRatePerSample, baseCriterion);

            return bestLearnRatePerSample;
        }

        void TrainOneMiniEpochAndReloadModel(ComputationNetwork<ElemType>& net, ComputationNetwork<ElemType>& refNet, const ComputationNodePtr refNode, 
            const int epochNumber,const  size_t epochSize, IDataReader<ElemType>* trainSetDataReader, const ElemType learnRatePerSample,
            const std::vector<ComputationNodePtr>& FeatureNodes,
            const std::vector<ComputationNodePtr>& labelNodes,
            const std::vector<ComputationNodePtr>& criterionNodes,
            const std::vector<ComputationNodePtr>& evaluationNodes,
            std::map<std::wstring, Matrix<ElemType>*>& inputMatrices,
            const std::list<ComputationNodePtr>& learnableNodes,
            std::list<Matrix<ElemType>>& smoothedGradients,
            ElemType& epochCriterion, std::vector<ElemType>& epochEvalErrors, size_t& totalSamplesSeen)
        {
            TrainOneEpoch(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, learnRatePerSample,FeatureNodes,labelNodes,
                criterionNodes,evaluationNodes,inputMatrices, learnableNodes,smoothedGradients,
                epochCriterion, epochEvalErrors, totalSamplesSeen); 
			fprintf(stderr,"Finished Mini-Epoch For LearnRate Selection: Train Loss Per Sample = %.8g    ", epochCriterion);
			if (epochEvalErrors.size()==1)
	            fprintf(stderr,"EvalErr Per Sample = %.8g   Ave Learn Rate Per Sample = %.10g\n", epochEvalErrors[0], learnRatePerSample);
			else
			{
				fprintf(stderr,"EvalErr Per Sample ");
				for (size_t i=0; i<epochEvalErrors.size(); i++)
					fprintf(stderr,"[%lu] = %.8g ", i, epochEvalErrors[i]);
				fprintf(stderr,"Ave Learn Rate Per Sample = %.10g\n",learnRatePerSample);
			}

            int baseModelEpoch =  epochNumber-1;
            net.LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
            net.ResetEvalTimeStamp();

            ElemType learnRate;
            ElemType prevCriterion;
            LoadCheckPointInfo(baseModelEpoch, totalSamplesSeen, learnRate, smoothedGradients, prevCriterion);  
        }

        void TrainOneEpoch(ComputationNetwork<ElemType>& net, ComputationNetwork<ElemType>& refNet, const ComputationNodePtr refNode, 
            const int epochNumber, const size_t epochSize, 
            IDataReader<ElemType>* trainSetDataReader, const ElemType learnRatePerSample,
            const std::vector<ComputationNodePtr>& FeatureNodes,
            const std::vector<ComputationNodePtr>& labelNodes,
            const std::vector<ComputationNodePtr>& criterionNodes,
            const std::vector<ComputationNodePtr>& evaluationNodes,
            std::map<std::wstring, Matrix<ElemType>*>& inputMatrices,
            const std::list<ComputationNodePtr>& learnableNodes,
            std::list<Matrix<ElemType>>& smoothedGradients,
            ElemType& epochCriterion, std::vector<ElemType>& epochEvalErrors, size_t& totalSamplesSeen)
        {
            ElemType readTimeInMBs = 0, ComputeTimeInMBs = 0, epochCriterionLastMBs = 0;
            int numSamplesLastMBs = 0;
            std::vector<ElemType> epochEvalErrorsLastMBs(epochEvalErrors.size(),0);
            PTaskGraphBuilder<ElemType>* ptaskGraphBuilder = NULL;
            
            clock_t startReadMBTime = 0, startComputeMBTime=0;
            clock_t endReadMBTime=0, endComputeMBTime=0; 

            //initialize statistics
            size_t totalEpochSamples = 0;

            int numMBsRun = 0;
            bool beginEpoch = true;

            size_t numEvalNodes = epochEvalErrors.size();

            // NOTE: the following two local matrices are not used in PTask path
            Matrix<ElemType> localEpochCriterion(1,1,net.GetDeviceID()); //assume only one training criterion node for each epoch
            Matrix<ElemType> localEpochEvalErrors(1,numEvalNodes,net.GetDeviceID());

            localEpochCriterion.SetValue(0);
            localEpochEvalErrors.SetValue(0);

            if (m_usePtask)
            {
                epochCriterion = ElemType(0.0);
                epochEvalErrors.assign(numEvalNodes, ElemType(0.0));
            }

            trainSetDataReader->StartMinibatchLoop(m_mbSize[epochNumber], epochNumber, m_epochSize);

            // build the PTask graph if they want to use ptask
            // NOTE: the graph is currently only for training, so other operations will still use the usual method, 
            // (i.e rate adjustment, regularization and other custom operations still use the non PTask method)
            if (m_usePtask)
            {
                ptaskGraphBuilder = net.GetPTaskGraphBuilder();
                ptaskGraphBuilder->UpdateParameters(this, learnRatePerSample, m_mbSize[epochNumber]);
                ptaskGraphBuilder->StartPTaskGraph();

                // currently CNTK likes to keep things on the GPU, and PTask expects things to be on the CPU, so tell CNTK to keep data on the CPU
                for (std::pair<std::wstring, Matrix<ElemType>*> inpair : inputMatrices)
                {
                    Matrix<ElemType>* mat = inpair.second;
                    mat->SetPreferredDeviceId(CPUDEVICE);
                    mat->TransferFromDeviceToDevice(mat->GetDeviceId(), CPUDEVICE, true);
                }
            }
            
            startReadMBTime=clock();
            while (trainSetDataReader->GetMinibatch(inputMatrices))
            {
                endReadMBTime=clock();
                startComputeMBTime=clock();

                UpdateEvalTimeStamps(FeatureNodes);
                UpdateEvalTimeStamps(labelNodes);

                size_t actualMBSize = net.GetActualMBSize();

                net.SetActualMiniBatchSize(actualMBSize);
                net.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
                trainSetDataReader->SetSentenceEndInBatch(net.m_sentenceEnd); 

#ifndef EVALDLL
                if (m_doGradientCheck && GradientCheck(net, criterionNodes, learnableNodes, 0) == false)
                {
                     throw std::logic_error("cannot pass gradient checker");
                }
#endif
                if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr) //TODO: currently only support one node regularization
                {
                    refNet.SetActualMiniBatchSize(actualMBSize);
                    refNet.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
                    refNet.Evaluate(refNode);
                    Matrix<ElemType>::ScaleAndAdd(m_adaptationRegWeight, refNode->FunctionValues(), 1-m_adaptationRegWeight, labelNodes[0]->FunctionValues()); 
                }
                   
                if (m_usePtask)
                {
                    // Pushing data in the graph starts things going
                    bool endOfEpoch = trainSetDataReader->DataEnd(endDataEpoch);
                    CONTROLSIGNAL signal = beginEpoch?DBCTLC_BOF:DBCTLC_NONE;
                    if (endOfEpoch)
                        signal |= DBCTLC_EOF;

                    ptaskGraphBuilder->PushData(inputMatrices, signal);
                    ptaskGraphBuilder->PushActualMBSize(learnableNodes, net.GetActualMBSize(), signal);
                    beginEpoch = false; // clear this out after first epoch

                    // pull the values from the graph for the totals
                    epochCriterion += ptaskGraphBuilder->GetValue(criterionNodes[0]);
				    for (size_t i=0; i<numEvalNodes; i++)
                    {
                        epochEvalErrors[i] += ptaskGraphBuilder->GetValue(evaluationNodes[i]);
                    }

                    // NOTE: update model parameters is part of the graph, so nothing to do here
                }
                else
                {
                    if (learnRatePerSample > m_minLearnRate * 0.01)  //only compute gradient when learning rate is large enough
                        net.ComputeGradient(criterionNodes[0]);  //use only the first criterion. Is there any possibility to use more?
                    else
                        net.Evaluate(criterionNodes[0]); //use only the first criterion. Is there any possibility to use more?

                    Matrix<ElemType>::AddElementToElement(criterionNodes[0]->FunctionValues(), 0, 0, localEpochCriterion, 0, 0);

                    std::vector<ElemType>mbEvalErrors(numEvalNodes,0);
				    for (size_t i=0; i<numEvalNodes; i++)
                    {
					    net.Evaluate(evaluationNodes[i]);
                        Matrix<ElemType>::AddElementToElement(evaluationNodes[i]->FunctionValues(), 0, 0, localEpochEvalErrors, 0, i);
                    }

                    //update model parameters
                    if (learnRatePerSample > m_minLearnRate * 0.01)
                    {
                        auto smoothedGradientIter=smoothedGradients.begin();
                        for (auto nodeIter=learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, smoothedGradientIter++)
                        {
                            ComputationNodePtr node = (*nodeIter);
                            Matrix<ElemType>& smoothedGradient = (*smoothedGradientIter);

                            UpdateWeights(node, smoothedGradient, learnRatePerSample, actualMBSize, m_mbSize[epochNumber]);
                        }                    
                    }
                }


                endComputeMBTime=clock();
                numMBsRun ++;
                if (m_traceLevel > 0)
                {
                    ElemType MBReadTime = (ElemType)(endReadMBTime-startReadMBTime)/(CLOCKS_PER_SEC);
                    ElemType MBComputeTime = (ElemType)(endComputeMBTime-startComputeMBTime)/CLOCKS_PER_SEC;

                    readTimeInMBs += MBReadTime;
                    ComputeTimeInMBs += MBComputeTime;
                    numSamplesLastMBs += int(actualMBSize);

                    if (numMBsRun % m_numMBsToShowResult == 0)
                    {
                        if (!m_usePtask)
                        {   // get the epoch Values updated, in PTask don't use the loclEpoch* temporary matrices
                            epochCriterion = localEpochCriterion.Get00Element();
                            for (size_t i=0; i< numEvalNodes; i++)
                                epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0,i);
                        }

                        fprintf(stderr,"Epoch[%d]-Minibatch[%d-%d]: Samples Seen = %d    Train Loss Per Sample = %.8g    ",epochNumber+1, numMBsRun-m_numMBsToShowResult+1, numMBsRun, numSamplesLastMBs,
                            (epochCriterion-epochCriterionLastMBs)/numSamplesLastMBs);
                        for (size_t i=0; i<numEvalNodes; i++){
                            fprintf(stderr,"EvalErr[%lu] Per Sample = %.8g    ",i,(epochEvalErrors[i]-epochEvalErrorsLastMBs[i])/numSamplesLastMBs);
                        }
                        fprintf(stderr,"ReadData Time = %.8g Computing Time=%.8g Total Time Per Sample=%.8g\n", readTimeInMBs, ComputeTimeInMBs, (readTimeInMBs + ComputeTimeInMBs)/numSamplesLastMBs);
                                                    
                        //reset statistics
                        readTimeInMBs = ComputeTimeInMBs = 0;
                        numSamplesLastMBs = 0; 

                        epochCriterionLastMBs = epochCriterion;
                        for (size_t i=0; i< numEvalNodes; i++)
                            epochEvalErrorsLastMBs[i] = epochEvalErrors[i];
                    }
                }
                startReadMBTime=clock();
                totalEpochSamples += actualMBSize;
                totalSamplesSeen += actualMBSize;

                if (totalEpochSamples >= epochSize)
                    break;

                /// call DataEnd function 
                /// DataEnd does reader specific process if sentence ending is reached
                trainSetDataReader->DataEnd(endDataSentence);

            }

            if (m_usePtask)
            {
                // when the epoch is complete, we need to transfer all the values back to the LearnableNodes, which will be saved off as the model
                std::list<ComputationNodePtr> learnableNodes = net.LearnableNodes(criterionNodes[0]);
                for (ComputationNodePtr node : learnableNodes)
                {
                    ptaskGraphBuilder->GetValue(node, node->FunctionValues());
                }
                epochCriterion /= float(totalEpochSamples);
			    for (size_t i=0; i< numEvalNodes; i++)
                {
                    epochEvalErrors[i] /= float(totalEpochSamples);
                }
            }
            else
            {
                localEpochCriterion /= float(totalEpochSamples);
                localEpochEvalErrors /= float(totalEpochSamples);

                epochCriterion = localEpochCriterion.Get00Element();
			    for (size_t i=0; i< numEvalNodes; i++)
                {
                    epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0,i);
                }
            }
        }
public:
        // UpdateWeightsS - static version of UpdateWeights()
        static void UpdateWeightsS(const SGD* sgd, Matrix<ElemType>& functionValues, Matrix<ElemType>& gradientValues, Matrix<ElemType>& smoothedGradient, const ElemType learnRatePerSample, size_t actualMBSize, const size_t expectedMBSize)
        {
#if DUMPOUTPUT
            fprintf(stderr, "learnRatePerSample=%0.8f, actualMBSize=%ld, expectedMBSize=%ld\n",learnRatePerSample, actualMBSize, expectedMBSize);
            fprintf(stderr, "sgd->GradUpdateType()=%d, sgd->GradientUpdateNoiseStd()=%0.8f, sgd->MomentumPerMB()=%0.8f\n",sgd->GradUpdateType(), sgd->GradientUpdateNoiseStd(), sgd->MomentumPerMB());
            gradientValues.Print("Gradient Input");
            smoothedGradient.Print("Smoothed Gradient Input");
#endif

            // make actualMBSize is a valid value
            assert(actualMBSize > 0);

            //clipping gradients to prevent outliers
            sgd->ClipGradient(gradientValues, actualMBSize);

            GradientsUpdateType adpType = sgd->GradUpdateType();
            ElemType noiseStd = sgd->GradientUpdateNoiseStd();
            Matrix<ElemType> sgdUpdateNoise((short)functionValues.GetDeviceId());
            if (noiseStd > 0)
            {
                sgdUpdateNoise.SetValue(gradientValues);  /// get the gradient structure since gradient is sparse
                sgdUpdateNoise.SetGaussianRandomValue(0, noiseStd); // reset its value to random 
            }

            if (adpType == GradientsUpdateType::None)
            {
                ElemType momentum = sgd->MomentumPerMB(); 
                if (actualMBSize < expectedMBSize && momentum > 0.0000001f)  //we use simple linear (instead of log linear) scaling here
                {
                    momentum = (ElemType) exp (log(momentum)/expectedMBSize * actualMBSize);
                }
                smoothedGradient.NormalGrad(gradientValues, functionValues, learnRatePerSample, momentum);
            }
            if (adpType == GradientsUpdateType::AdaGrad)
            {
                smoothedGradient.Adagrad(gradientValues);
                Matrix<ElemType>::ScaleAndAdd(-learnRatePerSample, gradientValues, functionValues);
            }
            if (adpType == GradientsUpdateType::RmsProp)
            {
                smoothedGradient.RmsProp(gradientValues);
                Matrix<ElemType>::ScaleAndAdd(-learnRatePerSample, gradientValues, functionValues);
            }

            if (noiseStd > 0)
            {
                Matrix<ElemType>::ScaleAndAdd(1.0, sgdUpdateNoise, functionValues);
            }
#if DUMPOUTPUT
            functionValues.Print("Parameter Update");
#endif
        }
protected:
        // UpdateWeights - update the weights in 
        void UpdateWeights(const ComputationNodePtr node, Matrix<ElemType>& smoothedGradient, const ElemType learnRatePerSample, const size_t actualMBSize, const size_t expectedMBSize) const
        {
#if DUMPOUTPUT
            fprintf(stderr,"Update_%ws\n",node->NodeName().c_str());
#endif
            UpdateWeightsS(this, node->FunctionValues(), node->GradientValues(), smoothedGradient, learnRatePerSample, actualMBSize, expectedMBSize);
            node->UpdateEvalTimeStamp();
        }

        void ClipGradient(Matrix<ElemType>& gradient, const size_t actualMBSize) const
        {
            if (m_clippingThresholdPerSample != std::numeric_limits<ElemType>::infinity())
            {
                ElemType maxGradientPerMB = m_clippingThresholdPerSample * actualMBSize;
                if (m_gradientClippingWithTruncation)
                {
                    gradient.InplaceTruncate(maxGradientPerMB);
                }
                else //norm2 normalized
                {
                    ElemType gradientNorm = gradient.FrobeniusNorm();
                    if (gradientNorm > maxGradientPerMB)
                    {
                        ElemType normFactor =  maxGradientPerMB / gradientNorm;
                        gradient *= normFactor;
                    }
                }
            }
        }

        void SaveCheckPointInfo(const size_t epoch, const size_t totalSamplesSeen, const ElemType learnRatePerSample, 
            const std::list<Matrix<ElemType>>& smoothedGradients, const ElemType prevCriterion)
        {
            wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epoch));

            File fstream(checkPointFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsWrite);
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCKP");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLearnRate");
            fstream << totalSamplesSeen << learnRatePerSample << prevCriterion;
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELearnRate");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BGradient");

            for (auto smoothedGradientIter=smoothedGradients.begin(); smoothedGradientIter != smoothedGradients.end(); smoothedGradientIter++)
            {
                const Matrix<ElemType>& smoothedGradient = (*smoothedGradientIter);
                fstream << smoothedGradient;
            }
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EGradient");

            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECKP");
        }

        bool LoadCheckPointInfo(const size_t epoch, size_t& totalSamplesSeen, ElemType& learnRatePerSample, 
            std::list<Matrix<ElemType>>& smoothedGradients, ElemType& prevCriterion)
        {
            wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epoch));
            if (!fexists(checkPointFileName.c_str()) )
            {
                fprintf(stderr,"Warning: checkpiont file is missing. learning parameters will be initialized from 0\n");
                return false;
            }

            File fstream(checkPointFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);
            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCKP");

            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLearnRate");
            fstream >> totalSamplesSeen >> learnRatePerSample >> prevCriterion;
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELearnRate");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BGradient");

            for (auto smoothedGradientIter=smoothedGradients.begin(); smoothedGradientIter != smoothedGradients.end(); smoothedGradientIter++)
            {
                Matrix<ElemType>& smoothedGradient = (*smoothedGradientIter);
                fstream >> smoothedGradient;
            }
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EGradient");

            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECKP");

            return true;
        }

        wstring GetCheckPointFileNameForEpoch (const int epoch)
        {
            return GetModelNameForEpoch (epoch) + L".ckp";
        }

        wstring GetModelNameForEpoch (const int epoch, bool bLastModel = false)
        {
            int epoch1Base = epoch + 1;
            if (epoch1Base == m_maxEpochs || bLastModel) 
                return m_modelPath;          
            else 
                return msra::strfun::wstrprintf (L"%s.%d", m_modelPath.c_str(), (int) epoch1Base);
        } 

        //return -1 if nothing exists
        int DetermineStartEpoch (const bool makeMode)
        {
            if (!makeMode)
                return -1;  //always start from scratch

            int firstEpoch = -1;

            wstring curEpochFile = GetModelNameForEpoch(int(m_maxEpochs)-1);
            for (int e = int(m_maxEpochs)-1; e >= -1; e--)
            {
                const wstring prevEpochFile = GetModelNameForEpoch (e-1);

                if (IsResultFileUpdateToDate (curEpochFile, prevEpochFile, false))
                {
                    firstEpoch = size_t(e)+1;
                    break;
                }
                else
                    curEpochFile = prevEpochFile;
            }

            return firstEpoch;
        }

        //up to date if resultFile is older than srcFile or missing
        bool IsResultFileUpdateToDate (const wstring & resultFile, const wstring & srcFile, const bool IsSrcFileNeeded)
        {
            FILETIME resultFileTime;
            if (!getfiletime (resultFile, resultFileTime)) 
                return false;        // not up to date is resultFile is missing

            FILETIME srcFileTime;
            if (!getfiletime (srcFile, srcFileTime)) 
                return !IsSrcFileNeeded; // srcFile missing: if required, the result file is not up to date

            //up to date if resultFile has higher time stamp
            return (resultFileTime.dwHighDateTime > srcFileTime.dwHighDateTime) || 
                (resultFileTime.dwHighDateTime == srcFileTime.dwHighDateTime && resultFileTime.dwLowDateTime >= srcFileTime.dwLowDateTime);
        }

        AdaptationRegType ParseAdaptationRegType(wstring s)
        {
            transform(s.begin(), s.end(), s.begin(),tolower); 
            if (s == L"" || s == L"none")
                return AdaptationRegType::None;
            else if (s == L"kl" || s == L"klreg" )
                return AdaptationRegType::KL;
            else
                throw std::invalid_argument(
                "ParseAdaptationRegType: Invalid Adaptation Regularization Type. Valid values are "
                "(None | KL)");
        }

        GradientsUpdateType ParseGradUpdateType(wstring s)
        {
            transform(s.begin(), s.end(), s.begin(),tolower); 
            if (s == L"" || s == L"none")
                return GradientsUpdateType::None;
            else if (s == L"adagrad")
                return GradientsUpdateType::AdaGrad;
            else if (s == L"rmsprop")
                return GradientsUpdateType::RmsProp;
            else
                throw std::invalid_argument(
                "ParseGradUpdateType: Invalid Gradient Updating Type. Valid values are "
                "(None | AdaGrad | RmsProp )");
        }

        LearningRateSearchAlgorithm ParseLearningRateSearchType(wstring s)
        {
            transform(s.begin(), s.end(), s.begin(),tolower); 
            if (s == L"false" || s == L"none")
                return LearningRateSearchAlgorithm::None;
            else if (s == L"searchbeforeepoch" || s == L"beforeepoch" || s == L"before")
                return LearningRateSearchAlgorithm::SearchBeforeEpoch;
            else if (s == L"adjustafterepoch" || s == L"afterepoch" || s == L"after")
                return LearningRateSearchAlgorithm::AdjustAfterEpoch;
            else
                throw std::invalid_argument(
                "autoAdjustLR: Invalid learning rate search type. Valid values are "
                "(None | SearchBeforeEpoch | AdjustAfterEpoch)");
        }

        GradientsUpdateType GradUpdateType() const {return m_gradType.mType;}
        ElemType GradientUpdateNoiseStd() const {return m_gradType.mGaussianNoiseInjectStd;}
        ElemType MomentumPerMB() const {return m_momentumPerMB;}

    public:
        #define EPSILON 1e-5

        bool GradientCheck(
			ComputationNetwork<ElemType>& net,
            const std::vector<ComputationNodePtr>& criterionNodes,
            const std::list<ComputationNodePtr>& learnableNodes,
			int npos)
        {
		    // gradient checking
            for (auto nodeIter=learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);

                int irow = (int)fmod(rand(), node->FunctionValues().GetNumRows()-1);
                int icol = (int)fmod(rand(), node->FunctionValues().GetNumCols()-1);
                irow = max(0, irow);
                icol = max(0, icol);

                fprintf(stderr, "\n###### d%ws######\n", node->NodeName().c_str());
                // node->FunctionValues().Print();
                ElemType eOrg = node->FunctionValues()(irow,icol);

                node->UpdateEvalTimeStamp();
                net.ComputeGradient(criterionNodes[npos]);  //use only the first criterion. Is 
                //ElemType mbEvalCri =
                criterionNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar
                ElemType eGradErr = node->GradientValues()(irow, icol); 

                ElemType ePos = eOrg + ElemType(EPSILON);
                ElemType eNeg = eOrg - ElemType(EPSILON);

                node->FunctionValues()(irow, icol) = ePos;
                node->UpdateEvalTimeStamp();
                net.Evaluate(criterionNodes[npos]); 
                ElemType mbEvalCriPos = criterionNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar
                
                node->FunctionValues()(irow, icol) = eNeg;
                node->UpdateEvalTimeStamp();
                net.Evaluate(criterionNodes[npos]); 
                ElemType mbEvalCriNeg = criterionNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar

				// back to its orginal parameter value
                node->FunctionValues()(irow, icol) = eOrg; 

				// check if they are consistent
				double eGradNum = (mbEvalCriPos - mbEvalCriNeg)/(ePos - eNeg);
			    ElemType threshold = (ElemType) pow((ElemType)10.0, max((ElemType)0.0, ceil(log10(min(fabs(eGradErr), fabs(eGradNum)))))-(int)m_gradientCheckSigDigit);
			    ElemType diff = (ElemType) fabs(eGradErr - eGradNum);
                bool wrong = (_isnan(diff) || diff > threshold);
                if (wrong)
				{
                    fprintf (stdout, "\nd%ws Numeric gradient = %e, Error BP gradient = %e\n", node->NodeName().c_str(), eGradNum, eGradErr);
                    return false; 
				}
            }

			return true;
        }

        void SetOtherInfo(ComputationNetwork<ElemType>& net , IDataReader<ElemType>* /*trainSetDataReader*/, IDataReader<ElemType>* /*validSetDataReader*/, std::map<std::wstring, Matrix<ElemType>*>& inputMatrices)
        {
            std::vector<ComputationNodePtr> criterionNodes = net.FinalCriterionNodes();
            std::vector<ComputationNodePtr> evaluationNodes = net.EvaluationNodes();

            //initializing weights and gradient holder
            for (size_t i = 0; i < criterionNodes.size(); i++)
            {
                if (criterionNodes[i]->OperationName() == L"ClassBasedCrossEntropyWithSoftmax")
                {
                    ClassBasedCrossEntropyWithSoftmaxNodePtr crtNode = (ClassBasedCrossEntropyWithSoftmaxNodePtr) criterionNodes[i];
                    crtNode->AddClassInfo(inputMatrices[L"classinfo"], inputMatrices[L"idx2cls"]);
                }
            }

			for (size_t i=0;i<evaluationNodes.size(); i++)
            {
                if (evaluationNodes[i]->OperationName() == L"ClassBasedCrossEntropyWithSoftmax")
                {
                    ClassBasedCrossEntropyWithSoftmaxNodePtr crtNode = (ClassBasedCrossEntropyWithSoftmaxNodePtr) evaluationNodes[i];
                    crtNode->AddClassInfo(inputMatrices[L"classinfo"], inputMatrices[L"idx2cls"]);
                }
            }
        }

    protected:

        floatargvector m_learningRatesPerSample; /// learning rate per sample provided outside
        intargvector m_mbSize;
        size_t m_epochSize;
        size_t m_maxEpochs;
		floatargvector m_momentumInputPerMB;
		ElemType m_momentumPerMB;
        bool m_gradientClippingWithTruncation;
        ElemType m_clippingThresholdPerSample;

        wstring m_modelPath;
        wstring m_trainCriterionNodeName;
        wstring m_evalCriterionNodeName;

        intargvector m_numMiniBatch4LRSearch;
        size_t m_numBestSearchEpoch;

        LearningRateSearchAlgorithm m_autoLearnRateSearchType;

        AdaptationRegType m_adaptationRegType;
        ElemType m_adaptationRegWeight;
        bool m_needRegularization;

        bool m_loadBestModel;
        ElemType m_reduceLearnRateIfImproveLessThan;
        bool m_continueReduce;
        ElemType m_increaseLearnRateIfImproveMoreThan;
        ElemType m_learnRateIncreaseFactor;
        ElemType m_learnRateDecreaseFactor;

		floatargvector m_dropoutRates;
        size_t m_maxTempMemSizeInSamplesForCNN;

        UINT16 m_traceLevel;

        size_t m_numPrevLearnRates;

        ElemType m_minLearnRate;

        GradientUpdateInfo m_gradType;
        bool m_usePtask;

        bool m_keepCheckPointFiles;

        int m_numMBsToShowResult;

        bool m_doGradientCheck;
        ElemType m_gradientCheckSigDigit;

        bool m_validateAfterModelReloading;
    };
    template class SGD<float>; 
    template class SGD<double>;

}}}