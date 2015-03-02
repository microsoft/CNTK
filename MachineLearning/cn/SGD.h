//
// <copyright file="SGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "basetypes.h"
#include "ComputationNetwork.h"
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

#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
extern int myRank;
extern int numProcs;

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    void DecimateMinibatch(std::map<std::wstring, MSR::CNTK::Matrix<ElemType>*> &mb)
    {
        size_t rv = 0;
        if ( numProcs > 1 ) for (auto it = mb.begin(); it != mb.end(); ++it)
        {
            MSR::CNTK::Matrix<ElemType> &mat = *(it->second);
            size_t nCols = mat.GetNumCols();
            size_t col_start = (nCols * myRank) / numProcs;
            size_t col_end = (nCols*(myRank + 1)) / numProcs;
            if (col_end > nCols) col_end = nCols; // this shouldn't happen
            if (col_end == col_start)
            {
                MSR::CNTK::Matrix<ElemType> tmp(mat.GetNumRows(), 0, AUTOPLACEMATRIX, DENSE);
                mat.SetValue(tmp);
            }
            else
            {
                MSR::CNTK::Matrix<ElemType> tmp = mat.ColumnSlice(col_start, col_end - col_start);
                mat.SetValue(tmp);
            }
            if (0 == rv)
            {
                rv = mat.GetNumCols();
            }
            else
            {
                if (rv != mat.GetNumCols())
                    throw std::logic_error("Uneven number of columns among inputs.");
            }
        }
    }

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
    
    // configuration parameters associated with RMSProp learning algorithm
    typedef struct stRMSPropInfo{
        double gamma;
        double inc;
        double dec;
        double max;
        double min;
        stRMSPropInfo()
        {
            gamma = 0.99;
            inc = 1.2;
            dec = 0.75;
            max = 10.0;
            min = 0.1;
        }
    }RMSPropInfo;

    typedef struct stGradientUpdateInfo{
        GradientsUpdateType mType;
        float mGaussianNoiseInjectStd;
        stGradientUpdateInfo()
        {
            mType = GradientsUpdateType::AdaGrad;
            mGaussianNoiseInjectStd = 0.0075f;
        }
    } GradientUpdateInfo;

    template<class ElemType>
    class SGD : ComputationNetworkHelper<ElemType>
    {
    protected:
        typedef ComputationNetworkHelper<ElemType> B;
        using B::SetMaxTempMemSizeForCNN; using B::SetDropoutRate; using B::UpdateEvalTimeStamps;
        typedef ComputationNode<ElemType>* ComputationNodePtr;
        typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

    public:
        SGD(const ConfigParameters& configSGD)
        {
            ConfigArray learningRatesPerMBStr = configSGD("learningRatesPerMB", "");
			m_needToNormalizeLRByParallUtterance = false;
            floatargvector learningRatesPerMB = learningRatesPerMBStr;

            ConfigArray learningRatesPerSampleStr = configSGD("learningRatesPerSample", "");
            floatargvector learningRatesPerSample = learningRatesPerSampleStr;

            std::string executionEngineValue = configSGD("executionEngine", "synchronous");

            // AutoAdjust Parameters
            ConfigParameters configAALR (configSGD("AutoAdjust",""));
            LearningRateSearchAlgorithm autoAdjustLRType = ParseLearningRateSearchType(configAALR("autoAdjustLR", "None"));
            ElemType reduceLearnRateIfImproveLessThan = configAALR("reduceLearnRateIfImproveLessThan", "0");
            bool continueReduce = (bool)configAALR("continueReduce", "false");
            size_t learnRateAdjustInterval = (size_t)configAALR("learnRateAdjustInterval", "1");
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
            
            // extract RMSProp parameters from config, if they exist. Default to reasonable values.
            RMSPropInfo rpi;
            rpi.dec = (double)configSGD("rms_wgt_dec", "0.75");
            rpi.inc = (double)configSGD("rms_wgt_inc", "1.2");
            rpi.min = (double)configSGD("rms_wgt_min", "0.1");
            rpi.max = (double)configSGD("rms_wgt_max", "10.0");
            rpi.gamma = (double)configSGD("rms_gamma", "0.99");

            bool needAveMultiplier = (bool)configSGD("normWithAveMultiplier", "true");
            ElemType L2RegWeight = (ElemType)configSGD("L2RegWeight", "0");
            ElemType L1RegWeight = (ElemType)configSGD("L1RegWeight", "0");

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

			bool UsingAllDataForPreComputedNode = configSGD("UseAllDataForPreComputedNode", "true");

            Init(learningRatesPerMB, learningRatesPerSample, mbSize, epochSize, maxEpochs, modelPath, momentumPerMB, gradientClippingWithTruncation, 
                clippingThresholdPerSample,autoAdjustLRType, increaseLearnRateIfImproveMoreThan, learnRateIncreaseFactor, 
                reduceLearnRateIfImproveLessThan, continueReduce, learnRateDecreaseFactor, dropoutRates,
                loadBestModel, numMiniBatch4LRSearch, numPrevLearnRates, numBestSearchEpoch, traceLevel, numMBsToShowResult,
                maxTempMemSizeInSamplesForCNN, gUpdateInfo, keepCheckPointFiles, adaptationRegType, adaptationRegWeight,
                trainCriterionNodeName, evalCriterionNodeName, doGradientCheck, gradientCheckSigDigit, validateAfterModelReloading,
                rpi, learnRateAdjustInterval, UsingAllDataForPreComputedNode, needAveMultiplier, L2RegWeight, L1RegWeight);
        }
    
        void setMomentum(float momentum)
        {
            m_momentumPerMB = (ElemType)momentum;
        }

        //autoLearnRateSearchType is applied only if the learning rate for the epoch is not specified in learningRatesPerMB and learningRatesPerSample
        void Init(const floatargvector& learningRatesPerMB, const floatargvector& learningRatesPerSample, const intargvector& mbSize, 
            const size_t epochSize, const size_t maxEpochs, 
            const wstring& modelPath, const floatargvector& momentumPerMB, const bool gradientClippingWithTruncation = true,
            const ElemType clippingThresholdPerSample=std::numeric_limits<ElemType>::infinity(),
            const LearningRateSearchAlgorithm autoLearnRateSearchType = LearningRateSearchAlgorithm::None, 
            const ElemType increaseLearnRateIfImproveMoreThan = std::numeric_limits<ElemType>::infinity(), const ElemType learnRateIncreaseFactor = 1.382f,
            const ElemType reduceLearnRateIfImproveLessThan=0, const bool continueReduce=false, const ElemType learnRateDecreaseFactor = 0.618f, floatargvector dropoutRates = floatargvector(L"0.0f"),
            const bool loadBestModel=true, const intargvector& numMiniBatch4LRSearch=intargvector(L"500"), const size_t numPrevLearnRates = 5, 
            const size_t numBestSearchEpoch = 1, const int traceLevel = 0,
            const size_t numMBsToShowResult = 10, const size_t maxTempMemSizeInSamplesForCNN = 0,
            const GradientUpdateInfo gradUpdateType = GradientUpdateInfo(), const bool keepCheckPointFiles=false, const AdaptationRegType adaptationRegType = AdaptationRegType::None,
            const ElemType adaptationRegWeight = 0.0f, const wstring trainCriterionNodeName= L"", const wstring evalCriterionNodeName=L"",
            const bool doGradientCheck = false, const ElemType gradientCheckSigDigit = 6, const bool validateAfterModelReloading = true,
            RMSPropInfo rpi = RMSPropInfo(), size_t learnRateAdjustInterval = 1, const bool UsingAllDataForPreComputed = true, const bool needAveMultiplier = true, const ElemType L2RegWeight = 0, const ElemType L1RegWeight = 0)
        {
            m_numPrevLearnRates = numPrevLearnRates;
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
             m_learnRateAdjustInterval = max((size_t) 1, learnRateAdjustInterval); //minimum interval is 1 epoch
            m_learnRateDecreaseFactor=learnRateDecreaseFactor;
            m_clippingThresholdPerSample=abs(clippingThresholdPerSample);
            m_numMiniBatch4LRSearch=numMiniBatch4LRSearch;
            m_dropoutRates=dropoutRates;
            m_numMBsToShowResult=int(numMBsToShowResult);
            m_numBestSearchEpoch=numBestSearchEpoch;
            m_maxTempMemSizeInSamplesForCNN=maxTempMemSizeInSamplesForCNN;
            m_gradType = gradUpdateType;
            m_rpi = rpi;
            m_keepCheckPointFiles = keepCheckPointFiles;

            m_adaptationRegType = adaptationRegType;
            m_adaptationRegWeight = adaptationRegWeight;

            m_trainCriterionNodeName = trainCriterionNodeName;
            m_evalCriterionNodeName = evalCriterionNodeName;
			m_useAllDataForPreComputedNode = UsingAllDataForPreComputed;

            m_needAveMultiplier = needAveMultiplier;
            m_L2RegWeight = L2RegWeight;
            m_L1RegWeight = L1RegWeight;

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
				m_needToNormalizeLRByParallUtterance = true; 
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

        void Adapt(wstring origModelFileName, wstring refNodeName, IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader, const DEVICEID_TYPE deviceID, const bool makeMode = true)
        {
            if (origModelFileName == L"" || trainSetDataReader == nullptr)
                    throw std::invalid_argument ("origModel and trainSetDataReader should not be null.");

            int startEpoch = DetermineStartEpoch(makeMode);
            if (startEpoch == m_maxEpochs)
            {
                fprintf(stderr, "Final model exists. No further training is necessary.\n");
                return;
            }

            ComputationNetwork<ElemType> net(deviceID);
            if (startEpoch >= 0)
            {
                wstring modelFileName = GetModelNameForEpoch(int(startEpoch)-1);
                fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());
                net.LoadFromFile(modelFileName);
            }
            else
            {
                fprintf(stderr, "Load Network From the original model file %ls.\n", origModelFileName.c_str());
                net.LoadFromFile(origModelFileName);
            }

            startEpoch = max(startEpoch, 0);

            ComputationNetwork<ElemType> refNet(deviceID);
            m_needRegularization = m_adaptationRegType != AdaptationRegType::None && m_adaptationRegWeight > 0;
            if (m_needRegularization)
            {
                fprintf(stderr, "Load reference Network From the original model file %ls.\n", origModelFileName.c_str());
                refNet.LoadFromFile(origModelFileName);
            }

            ComputationNodePtr refNode = nullptr;
            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL)
            {
                fprintf(stderr, "Checking refNodeName %ls.\n", origModelFileName.c_str());
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
                fprintf(stderr, "Final model exists. No further training is necessary.\n");
                return;
            }

            wstring modelFileName = GetModelNameForEpoch(int(startEpoch)-1);
            if (startEpoch >= 0)
                fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());
            ComputationNetwork<ElemType>& net  = 
                startEpoch<0? netBuilder->BuildNetworkFromDescription() : netBuilder->LoadNetworkFromFile(modelFileName);
            // TODO: BUGBUG: if not starting from checkpoint, need to synchronize initial model
            // strategy should be to run the initializer above on myRank==0, and then broadcast parameters.

            startEpoch = max(startEpoch, 0);
            m_needRegularization = false;

            TrainOrAdaptModel(startEpoch, net, net, nullptr, trainSetDataReader, validationSetDataReader);
        }

    protected:
        std::vector<ComputationNodePtr>  GetTrainCriterionNodes(ComputationNetwork<ElemType>& net)
        {
            fprintf(stderr, "GetTrainCriterionNodes %ls ...\n", m_trainCriterionNodeName.c_str());
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
            fprintf(stderr, "GetEvalCriterionNodes %ls ...\n", m_evalCriterionNodeName.c_str());
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

            ElemType epochCriterion, avgCriterion, prevCriterion;
            epochCriterion = avgCriterion = prevCriterion = std::numeric_limits<ElemType>::infinity();
            size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

            std::vector<ElemType> epochEvalErrors(evaluationNodes.size(),std::numeric_limits<ElemType>::infinity());
            
            std::vector<wstring> evalNodeNames;
            for (size_t i=0;i<evaluationNodes.size(); i++)
                evalNodeNames.push_back(evaluationNodes[i]->NodeName());

            size_t totalSamplesSeen = 0;
            ElemType learnRatePerSample = 0.5f / m_mbSize[startEpoch];

            vector<ElemType> prevLearnRates;
            prevLearnRates.resize(m_numPrevLearnRates);
            for (int i=0; i<m_numPrevLearnRates; i++)
                prevLearnRates[i] = ElemType(-1);

            //precompute mean and invStdDev nodes and save initial model
            if (PreCompute(net, trainSetDataReader, FeatureNodes, labelNodes, inputMatrices) || startEpoch == 0)
                if (0 == myRank) // only needs to be done by one process
                    net.SaveToFile(GetModelNameForEpoch(int(startEpoch) - 1));

			// first, we need to normalize the effect of nbruttsineachrecurrentiter
			if (trainSetDataReader->NumberSlicesInEachRecurrentIter()>1 && m_needToNormalizeLRByParallUtterance)
			{
				for (auto & x : m_learningRatesPerSample)
				{
					x /= trainSetDataReader->NumberSlicesInEachRecurrentIter();
				}
			}
            bool learnRateInitialized = false;
            if (startEpoch > 0)
            {
                learnRateInitialized = LoadCheckPointInfo(startEpoch-1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);  
                if (learnRateInitialized)
                    prevLearnRates[startEpoch % m_numPrevLearnRates] = learnRatePerSample; 

                setMomentum(m_momentumInputPerMB[m_momentumInputPerMB.size()-1]);
            }

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && !learnRateInitialized && m_learningRatesPerSample.size() <= startEpoch)
                throw std::invalid_argument ("When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, or an explicit learning rate must be specified in config for the starting epoch.");

            unsigned long dropOutSeed = 1;
            ElemType prevDropoutRate = 0;

            bool learnRateReduced = false;

            SetMaxTempMemSizeForCNN(net, criterionNodes[0], m_maxTempMemSizeInSamplesForCNN);
            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr) 
                SetMaxTempMemSizeForCNN(refNet, refNode, m_maxTempMemSizeInSamplesForCNN);

            for (int i = int(startEpoch); i < int(m_maxEpochs); i++)
            {
                auto t_start_epoch = Timer::MilliSecondElapsed();

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
                    for (int j = 1; j < m_numPrevLearnRates; j++)
                    {
                        largestPrevLearnRatePerSample = max(largestPrevLearnRatePerSample, prevLearnRates[j]);
                    }

                    //return a reasonable  learning rate based on the initial mbsize
                    learnRatePerSample = SearchLearnRateBeforeEpoch(net, refNet, refNode, i, learnRatePerSample, trainSetDataReader, FeatureNodes,
                        labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes, smoothedGradients, learnRateInitialized, largestPrevLearnRatePerSample);

                    prevLearnRates[i % m_numPrevLearnRates] = learnRatePerSample;  //save per sample learn rate to support changeable mbsize
                }

                learnRateInitialized = true;

                if (learnRatePerSample < m_minLearnRate)
                {
                    fprintf(stderr, "Learn Rate Per Sample for Epoch[%d] = %.8g is less than minLearnRate %.8g. Training stops.\n", i + 1, learnRatePerSample, m_minLearnRate);
                    if (m_autoLearnRateSearchType != LearningRateSearchAlgorithm::None)
                        net.SaveToFile(m_modelPath);
                    break;
                }

#ifdef MPI_SUPPORT
				INT32 mySamples = (INT32)
#endif
					fprintf(stderr, "Starting Epoch %d: learning rate per sample = %f  momentum = %f \n", (int)startEpoch,  learnRatePerSample, m_momentumPerMB);
                TrainOneEpoch(net, refNet, refNode, i, m_epochSize, trainSetDataReader, learnRatePerSample, FeatureNodes, labelNodes,
                    criterionNodes, evaluationNodes, inputMatrices, learnableNodes, smoothedGradients,
                    epochCriterion, epochEvalErrors, totalSamplesSeen);

                auto t_end_epoch = Timer::MilliSecondElapsed();
                ElemType epochTime = (t_end_epoch - t_start_epoch) / ElemType(MS_PER_SEC);

                fprintf(stderr, "Finished Epoch[%d]: [Training Set] Train Loss Per Sample = %.8g    ", i + 1, epochCriterion);
                if (epochEvalErrors.size() == 1)
                {
                    fprintf(stderr, "EvalErr Per Sample = %.8g   Ave Learn Rate Per Sample = %.10g  Epoch Time=%.8g\n", epochEvalErrors[0], learnRatePerSample, epochTime);
                }
                else
                {
                    fprintf(stderr, "EvalErr Per Sample ");
                    for (size_t j = 0; j < epochEvalErrors.size(); j++)
                        fprintf(stderr, "[%lu]=%.8g ", j, epochEvalErrors[j]);
                    fprintf(stderr, "Ave Learn Rate Per Sample = %.10g  Epoch Time=%.8g\n", learnRatePerSample, epochTime);
                    fprintf(stderr, "Finished Epoch[%d]: Criterion Node [%ls] Per Sample = %.8g\n", i + 1, criterionNodes[0]->NodeName().c_str(), epochCriterion);
                    for (size_t j = 0; j < epochEvalErrors.size(); j++)
                        fprintf(stderr, "Finished Epoch[%d]: Evaluation Node [%ls] Per Sample = %.8g\n", i + 1, evalNodeNames[j].c_str(), epochEvalErrors[j]);
                }

#ifdef MPI_SUPPORT
                // model reduction and averaging
                if (numProcs > 0)
                {
                    ElemType factor; // weight for the parameter of my model
                    {
                        // compute total minibatch size
                        INT32 allSamples = 0;
                        MPI_Allreduce(&mySamples, &allSamples, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                        if (allSamples == 0) allSamples = 1;

                        factor = (ElemType)mySamples / (ElemType)allSamples;
                    }

                    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = (*nodeIter);
                        Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->FunctionValues();

                        // weight model by relative size of minibatch samples (and number of processors, for averaging)
                        ElemType *px = mat.CopyToArray();
                        size_t nx = mat.GetNumElements();
                        transform(px, px + nx, px, [factor](ElemType&val)->ElemType{return val * factor; });

                        // TODO: Replace default Allreduce with the reduction-shuffle-dance
                        vector<ElemType> py = vector<ElemType>(nx, ElemType(0));
                        MPI_Allreduce(px, &(py[0]), (int)nx, sizeof(ElemType) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                        mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), &(py[0]));
                        delete px;
                    }
                }
#endif

                if ( 0 == myRank ) // only evaluate once, on the master process. TODO: This could be faster by farming out the validation parts
                if (validationSetDataReader != trainSetDataReader && validationSetDataReader != nullptr)
                {
                    SimpleEvaluator<ElemType> evalforvalidation(net);
                    vector<wstring> cvSetTrainAndEvalNodes;
                    cvSetTrainAndEvalNodes.push_back(criterionNodes[0]->NodeName());
                    cvSetTrainAndEvalNodes.push_back(evaluationNodes[0]->NodeName());

                    vector<ElemType> vScore = evalforvalidation.Evaluate(*validationSetDataReader, cvSetTrainAndEvalNodes, m_mbSize[i]);
                    fprintf(stderr, "Finished Epoch[%d]: [Validation Set] Train Loss Per Sample = %.8g  EvalErr Per Sample = %.8g\n",
                            i + 1, vScore[0], vScore[1]);

                    epochCriterion = vScore[0]; //the first one is the training criterion.
                }
#ifdef MPI_SUPPORT
                // ensure all processes have the same epochCriterion
                MPI_Bcast(&epochCriterion, 1, sizeof(epochCriterion) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

                bool loadedPrevModel = false;
                size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
                if (avgCriterion == std::numeric_limits<ElemType>::infinity())
                    avgCriterion = epochCriterion;
                else
                    avgCriterion = ((epochsSinceLastLearnRateAdjust -1 - epochsNotCountedInAvgCriterion)* avgCriterion + epochCriterion) / (epochsSinceLastLearnRateAdjust - epochsNotCountedInAvgCriterion);

                if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && m_learningRatesPerSample.size() <= i && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
                {
                    if (prevCriterion - avgCriterion < 0 && prevCriterion != std::numeric_limits<ElemType>::infinity())
                    {
                        if (m_loadBestModel)
                        {
                            net.LoadPersistableParametersFromFile(GetModelNameForEpoch(i-1), m_validateAfterModelReloading);
                            net.ResetEvalTimeStamp();
                            LoadCheckPointInfo(i-1, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);  
                            fprintf(stderr, "Loaded the previous model which has better training criterion.\n");
                            loadedPrevModel = true;
                        }
                    }

                    if(m_continueReduce)
                    {
                        if (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion && prevCriterion != std::numeric_limits<ElemType>::infinity())
                        {
                            if(learnRateReduced == false) 
                            {
                                learnRateReduced = true;                                
                            }
                            else 
                            {
                                if ( myRank == 0 )
                                    net.SaveToFile(GetModelNameForEpoch(i, true));
                                fprintf(stderr, "Finished training and saved final model\n\n");
                                break;
                            }
                        }
                        if(learnRateReduced) 
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
                if (0 == myRank)
                {
                    net.SaveToFile(GetModelNameForEpoch(i));
                    SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion);
                    if (!m_keepCheckPointFiles)
                        _wunlink(GetCheckPointFileNameForEpoch(i - 1).c_str());  //delete previous checkpiont file to save space
                }

                if (learnRatePerSample < 1e-12)
                    fprintf(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n", learnRatePerSample);
            }

            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr) //since we linked feature nodes. we need to remove it from the deletion
            {
                for (size_t i=0; i<refFeatureNodes.size(); i++)
                {
                    refNet.ChangeNode(refFeatureNodes[i]->NodeName(), refFeatureNodes[i]); //note we need to handle deletion carefully
                }
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
                fprintf(stderr, "No PreCompute nodes found, skipping PreCompute step\n");
                return false;
            }

            fprintf(stderr, "Found %lu PreCompute nodes\n", nodes.size());
            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                PreComputedNode<ElemType>* node = static_cast<PreComputedNode<ElemType>*> (*nodeIter);
                fprintf(stderr, "\tNodeName: %ls\n", (node->NodeName()).c_str());
            }

            //compute
            //trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , requestDataSize); 
            // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , m_epochSize); // only based on one epoch
			// [1/12/2015 erw] to support large dataset, we usually paritition whole dataset into several epoches, so we need to use all the data to do precomputing
			if (m_useAllDataForPreComputedNode)
				trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0); // using all the data
			else 
				trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0, m_epochSize); // using all the data

            while (trainSetDataReader->GetMinibatch(inputMatrices))
            {
                UpdateEvalTimeStamps(FeatureNodes);
                UpdateEvalTimeStamps(labelNodes);

                size_t actualMBSize = net.GetActualMBSize();
                net.SetActualMiniBatchSize(actualMBSize);
                net.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
                trainSetDataReader->SetSentenceEndInBatch(net.m_sentenceEnd);

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
            std::list<Matrix<ElemType>>& smoothedGradients, const bool learnRateInitialized, const ElemType largestPrevLearnRatePerSample)
        {
            ElemType epochCriterion = std::numeric_limits<ElemType>::infinity(), prevCriterion = std::numeric_limits<ElemType>::infinity();
            vector<ElemType> epochEvalErrors(evaluationNodes.size(), std::numeric_limits<ElemType>::infinity());
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
            ElemType learnRatePerSample = 1.0f / 8.0f / 0.618f / sqrt((ElemType)m_mbSize[epochNumber]);

            if (learnRateInitialized && largestPrevLearnRatePerSample > 0)
                learnRatePerSample = largestPrevLearnRatePerSample / 0.618f / 0.618f;  //largestPrevLearnRatePerSample is per sample, first 0.618f is for compensation, second one is for safety

            int baseModelEpoch = epochNumber - 1;
            net.LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
            net.ResetEvalTimeStamp();

            ElemType learnRate = learnRatePerSample;
            LoadCheckPointInfo(baseModelEpoch, totalSamplesSeen, learnRate, smoothedGradients, prevCriterion);

            //if model is not changed this is what we will get
            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, 0,
                FeatureNodes, labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                smoothedGradients, baseCriterion, epochEvalErrors, totalSamplesSeen);

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
            {
                if (prevCriterion == std::numeric_limits<ElemType>::infinity())
                    prevCriterion = baseCriterion;
                ElemType ratio = 0.3f;
                if (m_epochSize != requestDataSize)
                {
                    ratio = pow(((ElemType)epochSize) / m_epochSize, 1.0f / 2);
                }
                baseCriterion = max(ratio * prevCriterion + (1 - ratio) * baseCriterion, baseCriterion);
            }

            do
            {
                learnRatePerSample *= 0.618f;
                TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, learnRatePerSample,
                    FeatureNodes, labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                    smoothedGradients, epochCriterion, epochEvalErrors, totalSamplesSeen);

            } while (epochCriterion > baseCriterion && learnRatePerSample > minLearnRate);

            bestLearnRatePerSample = learnRatePerSample;

            if (epochNumber < m_numBestSearchEpoch) //grid search for the first m_numBestSearchEpoch  epochs
            {
                ElemType leftLearnRatePerSample = 0.01f / m_mbSize[epochNumber], rightLearnRatePerSample = learnRatePerSample;
                ElemType leftCriterion, rightCriterion = epochCriterion;

                TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, leftLearnRatePerSample,
                    FeatureNodes, labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                    smoothedGradients, leftCriterion, epochEvalErrors, totalSamplesSeen);

                while (rightLearnRatePerSample > leftLearnRatePerSample * 1.2f)
                {
                    if (rightCriterion > leftCriterion)
                    {
                        rightLearnRatePerSample *= 0.618f;

                        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, rightLearnRatePerSample,
                            FeatureNodes, labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                            smoothedGradients, rightCriterion, epochEvalErrors, totalSamplesSeen);
                    }
                    else
                    {
                        leftLearnRatePerSample /= 0.618f;

                        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber, epochSize, trainSetDataReader, leftLearnRatePerSample,
                            FeatureNodes, labelNodes, criterionNodes, evaluationNodes, inputMatrices, learnableNodes,
                            smoothedGradients, leftCriterion, epochEvalErrors, totalSamplesSeen);
                    }
                }

                bestLearnRatePerSample = (leftCriterion < rightCriterion) ? leftLearnRatePerSample : rightLearnRatePerSample;
            }

            fprintf(stderr, "Best Learn Rate Per Sample for Epoch[%d] = %.10g  baseCriterion=%.10g\n", epochNumber + 1, bestLearnRatePerSample, baseCriterion);

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
            fprintf(stderr, "Finished Mini-Epoch For LearnRate Selection: Train Loss Per Sample = %.8g    ", epochCriterion);
            if (epochEvalErrors.size()==1)
                fprintf(stderr, "EvalErr Per Sample = %.8g   Ave Learn Rate Per Sample = %.10g\n", epochEvalErrors[0], learnRatePerSample);
            else
            {
                fprintf(stderr, "EvalErr Per Sample ");
                for (size_t i=0; i<epochEvalErrors.size(); i++)
                    fprintf(stderr, "[%lu] = %.8g ", i, epochEvalErrors[i]);
                fprintf(stderr, "Ave Learn Rate Per Sample = %.10g\n",learnRatePerSample);
            }

            int baseModelEpoch =  epochNumber-1;
            net.LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
            net.ResetEvalTimeStamp();

            ElemType learnRate;
            ElemType prevCriterion;
            LoadCheckPointInfo(baseModelEpoch, totalSamplesSeen, learnRate, smoothedGradients, prevCriterion);  
        }

        size_t TrainOneEpoch(ComputationNetwork<ElemType>& net, ComputationNetwork<ElemType>& refNet, const ComputationNodePtr refNode, 
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
            
            unsigned long long startReadMBTime = 0, startComputeMBTime=0;
            unsigned long long  endReadMBTime = 0, endComputeMBTime = 0;

            //initialize statistics
            size_t totalEpochSamples = 0;

            int numMBsRun = 0;

            size_t numEvalNodes = epochEvalErrors.size();

            Matrix<ElemType> localEpochCriterion(1,1,net.GetDeviceID()); //assume only one training criterion node for each epoch
            Matrix<ElemType> localEpochEvalErrors(1,numEvalNodes,net.GetDeviceID());

            localEpochCriterion.SetValue(0);
            localEpochEvalErrors.SetValue(0);

            trainSetDataReader->StartMinibatchLoop(m_mbSize[epochNumber], epochNumber, m_epochSize);
            
            startReadMBTime=Timer::MilliSecondElapsed();
            while (trainSetDataReader->GetMinibatch(inputMatrices))
            {
#ifdef MPI_SUPPORT
                DecimateMinibatch(inputMatrices);
#endif
                endReadMBTime=Timer::MilliSecondElapsed();
                startComputeMBTime=Timer::MilliSecondElapsed();

                UpdateEvalTimeStamps(FeatureNodes);
                UpdateEvalTimeStamps(labelNodes);

                size_t actualMBSize = net.GetActualMBSize();
                if (0 == actualMBSize)
                    continue;

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

                        UpdateWeights(node, smoothedGradient, learnRatePerSample, actualMBSize, m_mbSize[epochNumber], m_L2RegWeight, m_L1RegWeight, m_needAveMultiplier);
                    }                    
                }


                endComputeMBTime=Timer::MilliSecondElapsed();
                numMBsRun ++;
                if (m_traceLevel > 0)
                {
                    ElemType MBReadTime = (ElemType)(endReadMBTime-startReadMBTime)/(MS_PER_SEC);
                    ElemType MBComputeTime = (ElemType)(endComputeMBTime-startComputeMBTime)/MS_PER_SEC;

                    readTimeInMBs += MBReadTime;
                    ComputeTimeInMBs += MBComputeTime;
                    numSamplesLastMBs += int(actualMBSize);

                    if (numMBsRun % m_numMBsToShowResult == 0)
                    {
                        // get the epoch Values updated
                        epochCriterion = localEpochCriterion.Get00Element();
                        for (size_t i=0; i< numEvalNodes; i++)
                            epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0,i);

                        fprintf(stderr, "Epoch[%d]-Minibatch[%d-%d]: Samples Seen = %d    Train Loss Per Sample = %.8g    ",epochNumber+1, numMBsRun-m_numMBsToShowResult+1, numMBsRun, numSamplesLastMBs,
                            (epochCriterion-epochCriterionLastMBs)/numSamplesLastMBs);
                        for (size_t i=0; i<numEvalNodes; i++)
                        {
                            fprintf(stderr, "EvalErr[%lu] Per Sample = %.8g    ",i,(epochEvalErrors[i]-epochEvalErrorsLastMBs[i])/numSamplesLastMBs);
                        }
                        fprintf(stderr, "ReadData Time = %.8g Computing Time=%.8g Total Time Per Sample=%.8g\n", readTimeInMBs, ComputeTimeInMBs, (readTimeInMBs + ComputeTimeInMBs)/numSamplesLastMBs);
                                                    
                        //reset statistics
                        readTimeInMBs = ComputeTimeInMBs = 0;
                        numSamplesLastMBs = 0; 

                        epochCriterionLastMBs = epochCriterion;
                        for (size_t i=0; i< numEvalNodes; i++)
                            epochEvalErrorsLastMBs[i] = epochEvalErrors[i];
                    }
                }
                startReadMBTime=Timer::MilliSecondElapsed();
                totalEpochSamples += actualMBSize;
                totalSamplesSeen += actualMBSize;

                if (totalEpochSamples >= epochSize)
                    break;

                /// call DataEnd function 
                /// DataEnd does reader specific process if sentence ending is reached
                trainSetDataReader->DataEnd(endDataSentence);

            }

            localEpochCriterion /= float(totalEpochSamples);
            localEpochEvalErrors /= float(totalEpochSamples);

            epochCriterion = localEpochCriterion.Get00Element();
            for (size_t i=0; i< numEvalNodes; i++)
            {
                epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0,i);
            }

            return totalEpochSamples;
        }
public:
        // UpdateWeightsS - static version of UpdateWeights()
    static void UpdateWeightsS(const SGD* sgd, Matrix<ElemType>& functionValues, Matrix<ElemType>& gradientValues, Matrix<ElemType>& smoothedGradient, const ElemType learnRatePerSample, size_t actualMBSize, const size_t expectedMBSize, const ElemType L2RegWeight, const ElemType L1RegWeight, const bool needAveMultiplier)
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
            Matrix<ElemType> sgdUpdateNoise((DEVICEID_TYPE)functionValues.GetDeviceId());
            if (noiseStd > 0)
            {
                sgdUpdateNoise.SetValue(gradientValues);  /// get the gradient structure since gradient is sparse
                sgdUpdateNoise.SetGaussianRandomValue(0, noiseStd); // reset its value to random 
            }

            // L2 regularizer
            if (L2RegWeight > 0) //*actualMBSize so that it's invariant to minibatch size since learning rate is per sample
                Matrix<ElemType>::ScaleAndAdd(L2RegWeight*actualMBSize, functionValues, gradientValues);

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
                ElemType aveMultiplier = smoothedGradient.Adagrad(gradientValues, needAveMultiplier);
                Matrix<ElemType>::ScaleAndAdd(-learnRatePerSample / aveMultiplier, gradientValues, functionValues);
            }
            if (adpType == GradientsUpdateType::RmsProp)
            {
                ElemType aveMultiplier = smoothedGradient.RmsProp(gradientValues, (ElemType)sgd->m_rpi.gamma, (ElemType)sgd->m_rpi.inc, (ElemType)sgd->m_rpi.max, (ElemType)sgd->m_rpi.dec, (ElemType)sgd->m_rpi.min, needAveMultiplier);
                Matrix<ElemType>::ScaleAndAdd(-learnRatePerSample / aveMultiplier, gradientValues, functionValues);
            }

            if (noiseStd > 0)
            {
                Matrix<ElemType>::ScaleAndAdd(1.0, sgdUpdateNoise, functionValues);
            }

            // L1 regularizer with proximal gradient descent method
            if (L1RegWeight > 0) //*actualMBSize so that it's invariant to minibatch size since learning rate is per sample
                functionValues.InplaceSoftThreshold(learnRatePerSample*L1RegWeight*actualMBSize);

#if DUMPOUTPUT
            functionValues.Print("Parameter Update");
#endif
        }
protected:
        // UpdateWeights - update the weights in 
    void UpdateWeights(const ComputationNodePtr node, Matrix<ElemType>& smoothedGradient, const ElemType learnRatePerSample, const size_t actualMBSize, const size_t expectedMBSize, const ElemType L2RegWeight, const ElemType L1RegWeight, const bool needAveMultiplier) const
        {
#if DUMPOUTPUT
            fprintf(stderr, "Update_%ls\n",node->NodeName().c_str());
#endif
            UpdateWeightsS(this, node->FunctionValues(), node->GradientValues(), smoothedGradient, learnRatePerSample, actualMBSize, expectedMBSize, L2RegWeight, L1RegWeight, needAveMultiplier);
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
                fprintf(stderr, "Warning: checkpiont file is missing. learning parameters will be initialized from 0\n");
                return false;
            }

            File fstream(checkPointFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);
            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCKP");

            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BLearnRate");
            fstream >> totalSamplesSeen >> learnRatePerSample >> prevCriterion;
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ELearnRate");

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
            else {
				wstring w = msra::strfun::wstrprintf (L"%ls.%d", m_modelPath.c_str(), (int) epoch1Base);
				return w;
			}
 
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

                if (msra::files::fuptodate (curEpochFile, prevEpochFile, false))
                {
                    firstEpoch = size_t(e)+1;
                    break;
                }
                else
                    curEpochFile = prevEpochFile;
            }

            return firstEpoch;
        }

        AdaptationRegType ParseAdaptationRegType(wstring s)
        {
            msra::strfun::tolower_ascii(s);
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
            msra::strfun::tolower_ascii(s);
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
            msra::strfun::tolower_ascii(s);
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
            vector<string> errMsgs; 

            // gradient checking
            for (auto nodeIter=learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
            {
                ComputationNodePtr node = (*nodeIter);
                char wstrtmp[2048];

                for (size_t itry = 0; itry < min((size_t)50, node->FunctionValues().GetNumElements()); itry++)
                {
                    /// no support to sparse matrix yet
                    int irow = (int)fmod(rand(), node->FunctionValues().GetNumRows() - 1);
                    int icol = (int)fmod(rand(), node->FunctionValues().GetNumCols() - 1);
                    irow = max(0, irow);
                    icol = max(0, icol);

                    fprintf(stderr, "\n###### d%ls######\n", node->NodeName().c_str());
                    // node->FunctionValues().Print();
                    ElemType eOrg = node->FunctionValues()(irow, icol);
                    if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                        node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), net.GetDeviceID(), true);

                    node->UpdateEvalTimeStamp();
                    net.ComputeGradient(criterionNodes[npos]);  //use only the first criterion. Is 
//                    if (node->GradientValues().GetMatrixType() == MatrixType::SPARSE && node->GradientValues().GetDeviceId() != CPUDEVICE)
                    if (node->GradientValues().GetMatrixType() == MatrixType::SPARSE)
                        break;

                    //ElemType mbEvalCri =
                    criterionNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar
                    ElemType eGradErr = node->GradientValues()(irow, icol);
                    if (node->GradientValues().GetDeviceId() != net.GetDeviceID())
                        node->GradientValues().TransferFromDeviceToDevice(node->GradientValues().GetDeviceId(), net.GetDeviceID(), true);

                    ElemType ePos = eOrg + ElemType(EPSILON);
                    ElemType eNeg = eOrg - ElemType(EPSILON);

                    node->FunctionValues()(irow, icol) = ePos;
                    if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                        node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), net.GetDeviceID(), true);
                    node->UpdateEvalTimeStamp();
                    net.Evaluate(criterionNodes[npos]);
                    ElemType mbEvalCriPos = criterionNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar

                    node->FunctionValues()(irow, icol) = eNeg;
                    if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                        node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), net.GetDeviceID(), true);
                    node->UpdateEvalTimeStamp();
                    net.Evaluate(criterionNodes[npos]);
                    ElemType mbEvalCriNeg = criterionNodes[npos]->FunctionValues().Get00Element(); //criterionNode should be a scalar

                    // back to its orginal parameter value
                    node->FunctionValues()(irow, icol) = eOrg;
                    if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                        node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(), net.GetDeviceID(), true);

                    // check if they are consistent
                    ElemType eGradNum = (ElemType)((mbEvalCriPos - mbEvalCriNeg) / (ePos - eNeg));
                    ElemType threshold = (ElemType)pow((ElemType)10.0, max((ElemType)0.0, ceil(log10(min(fabs(eGradErr), fabs(eGradNum))))) - (int)m_gradientCheckSigDigit);
                    ElemType diff = (ElemType)fabs(eGradErr - eGradNum);
                    bool wrong = (std::isnan(diff) || diff > threshold);
                    if (wrong)
                    {
                        fprintf(stderr, "\nd%ws Numeric gradient = %e, Error BP gradient = %e\n", node->NodeName().c_str(), eGradNum, eGradErr);
                        sprintf(wstrtmp, "\nd%ws Numeric gradient = %e, Error BP gradient = %e\n", node->NodeName().c_str(), eGradNum, eGradErr);
                        errMsgs.push_back(wstrtmp);
                    }
                }
            }

            if (errMsgs.size() > 0)
                return false;
            return true;
        }

    protected:

        floatargvector m_learningRatesPerSample; /// learning rate per sample provided outside
		bool			m_needToNormalizeLRByParallUtterance;			// only true when the user specify LearningRatePerMB and the number of parallel utterances in Reader > 1
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
        size_t m_learnRateAdjustInterval; //determine after how many epochs the learning rate should be auto adjusted.
        ElemType m_increaseLearnRateIfImproveMoreThan;
        ElemType m_learnRateIncreaseFactor;
        ElemType m_learnRateDecreaseFactor;

        floatargvector m_dropoutRates;
        size_t m_maxTempMemSizeInSamplesForCNN;

        int m_traceLevel;

        size_t m_numPrevLearnRates;

        ElemType m_minLearnRate;

        GradientUpdateInfo m_gradType;
        RMSPropInfo m_rpi;

        bool m_keepCheckPointFiles;

        int m_numMBsToShowResult;

        bool m_doGradientCheck;
        ElemType m_gradientCheckSigDigit;

        bool m_validateAfterModelReloading;

		bool m_useAllDataForPreComputedNode;

        bool m_needAveMultiplier;
        ElemType m_L2RegWeight;
        ElemType m_L1RegWeight;
    };
    template class SGD<float>; 
    template class SGD<double>;

}}}
