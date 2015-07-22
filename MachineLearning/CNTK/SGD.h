//
// <copyright file="SGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
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
#include "Profiler.h"

#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
extern int mpiRank;
extern int mpiNumProcesses;

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
void DecimateMinibatch(std::map<std::wstring, MSR::CNTK::Matrix<ElemType>*>& mb)
{
    size_t rv = 0;
    if (mpiNumProcesses > 1)
    {
        for (auto it = mb.begin(); it != mb.end(); ++it)
        {
            MSR::CNTK::Matrix<ElemType> &mat = *(it->second);
            size_t nCols = mat.GetNumCols();
            size_t col_start = (nCols * mpiRank) / mpiNumProcesses;
            size_t col_end = (nCols * (mpiRank + 1)) / mpiNumProcesses;
            if (col_end > nCols)
            {
                // this shouldn't happen
                col_end = nCols;
            }

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

            if (rv == 0)
            {
                rv = mat.GetNumCols();
            }
            else
            {
                if (rv != mat.GetNumCols())
                {
                    throw std::logic_error("Uneven number of columns among inputs.");
                }
            }
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
typedef struct stRMSPropInfo
{
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
} RMSPropInfo;

typedef struct stGradientUpdateInfo
{
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
    using B::SetMaxTempMemSizeForCNN;
    using B::SetDropoutRate;
    using B::UpdateEvalTimeStamps;
    typedef ComputationNode<ElemType>* ComputationNodePtr;
    typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

public:
    SGD(const ConfigParameters& configSGD)
    {
        ConfigArray learningRatesPerMBStr = configSGD("learningRatesPerMB", "");
        m_needToNormalizeLRByParallUtterance = false;
        m_needToNormalizeMomentumByParallUtterance = false;
        floatargvector learningRatesPerMB = learningRatesPerMBStr;

        ConfigArray learningRatesPerSampleStr = configSGD("learningRatesPerSample", "");
        floatargvector learningRatesPerSample = learningRatesPerSampleStr;

        std::string executionEngineValue = configSGD("executionEngine", "synchronous");

        // AutoAdjust Parameters
        ConfigParameters configAALR(configSGD("AutoAdjust", ""));
        LearningRateSearchAlgorithm autoAdjustLRType = ParseLearningRateSearchType(configAALR("autoAdjustLR", "None"));
        ElemType reduceLearnRateIfImproveLessThan = configAALR("reduceLearnRateIfImproveLessThan", "0");
        bool continueReduce = (bool) configAALR("continueReduce", "false");
        size_t learnRateAdjustInterval = (size_t) configAALR("learnRateAdjustInterval", "1");
        ElemType learnRateDecreaseFactor = configAALR("learnRateDecreaseFactor", "0.618");
        ElemType increaseLearnRateIfImproveMoreThan = configAALR("increaseLearnRateIfImproveMoreThan", "1#INF");
        ElemType learnRateIncreaseFactor = configAALR("learnRateIncreaseFactor", "1.382");

        // AutoAdjust Auto Adjust Minibatch Parameters
        bool autoAdjustMinibatch = (bool) configAALR("autoAdjustMinibatch", "false");
        size_t minibatchSizeTuningFrequency = configAALR("minibatchSizeTuningFrequency", "1");
        size_t minibatchSizeTuningMax = configAALR("minibatchSizeTuningMax", "1048576");

        // the number of minibatches used to search
        // the learning rate. It’s typically set to 10-20% of
        // the total minibatches in an epoch.
        ConfigArray minibatch4LRSearch = configAALR("numMiniBatch4LRSearch", "500");
        intargvector numMiniBatch4LRSearch = minibatch4LRSearch;

        size_t numPrevLearnRates = configAALR("numPrevLearnRates", "5");
        size_t numBestSearchEpoch = configAALR("numBestSearchEpoch", "1");
        bool loadBestModel = configAALR("loadBestModel", "true");

        ConfigArray minibatchSize = configSGD("minibatchSize", "256");
        intargvector mbSize = minibatchSize;

        // the number of samples in each epoch (0 means, use all the samples in each epoch).
        size_t epochSize = configSGD("epochSize", "0");

        // the total number of epochs to run.
        size_t maxEpochs = configSGD("maxEpochs");

        ConfigArray momentumPerMBStr = configSGD("momentumPerMB", "");
        floatargvector momentumPerMB = momentumPerMBStr;

        ConfigArray momentumPerSampleStr = configSGD("momentumPerSample", "");
        floatargvector momentumPerSample = momentumPerSampleStr;

        wstring modelPath = configSGD("modelPath");
        wstring trainCriterionNodeName = configSGD("trainCriterionNodeName", "");
        wstring evalCriterionNodeName = configSGD("evalCriterionNodeName", "");

        size_t maxTempMemSizeInSamplesForCNN = configSGD("maxTempMemSizeInSamplesForCNN", "0");

        int traceLevel = configSGD("traceLevel", "0");
        size_t numMBsToShowResult = configSGD("numMBsToShowResult", "10");
        size_t numMBsToCUDAProfile = configSGD("numMBsToCUDAProfile", "0");

        bool keepCheckPointFiles = configSGD("keepCheckPointFiles", "false");

        bool gradientClippingWithTruncation = configSGD("gradientClippingWithTruncation", "true");
        ElemType clippingThresholdPerSample = configSGD("clippingThresholdPerSample", "1#INF");

        ConfigArray dropoutRatesStr = configSGD("dropoutRate", "0.0");
        floatargvector dropoutRates = dropoutRatesStr;

        GradientUpdateInfo gUpdateInfo;
        GradientsUpdateType gradUpdateType = ParseGradUpdateType(configSGD("gradUpdateType", "None"));
        ElemType gaussianNoiseInjecStd = configSGD("gaussianNoiseInjectStd", "0");
        gUpdateInfo.mType = gradUpdateType;
        gUpdateInfo.mGaussianNoiseInjectStd = (float) gaussianNoiseInjecStd;

        // extract RMSProp parameters from config, if they exist. Default to reasonable values.
        RMSPropInfo rpi;
        rpi.dec = (double) configSGD("rms_wgt_dec", "0.75");
        rpi.inc = (double) configSGD("rms_wgt_inc", "1.2");
        rpi.min = (double) configSGD("rms_wgt_min", "0.1");
        rpi.max = (double) configSGD("rms_wgt_max", "10.0");
        rpi.gamma = (double) configSGD("rms_gamma", "0.99");

        bool needAveMultiplier = (bool) configSGD("normWithAveMultiplier", "true");
        ElemType L2RegWeight = (ElemType) configSGD("L2RegWeight", "0");
        ElemType L1RegWeight = (ElemType) configSGD("L1RegWeight", "0");

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

        if (doGradientCheck && sizeof(ElemType) != sizeof(double))
            LogicError("Gradient check needs to use type = double");

        m_doUnitTest = configSGD("unittest", "false");

        bool validateAfterModelReloading = configSGD("validateAfterModelReloading", "true");

        bool UsingAllDataForPreComputedNode = configSGD("UseAllDataForPreComputedNode", "true");

        Init(learningRatesPerMB, learningRatesPerSample, mbSize, epochSize,
             maxEpochs, modelPath, momentumPerMB, momentumPerSample,
             gradientClippingWithTruncation, clippingThresholdPerSample,
             autoAdjustLRType, increaseLearnRateIfImproveMoreThan,
             learnRateIncreaseFactor, reduceLearnRateIfImproveLessThan,
             continueReduce, learnRateDecreaseFactor, dropoutRates,
             loadBestModel, numMiniBatch4LRSearch, numPrevLearnRates,
             numBestSearchEpoch, traceLevel, numMBsToShowResult,
             numMBsToCUDAProfile, maxTempMemSizeInSamplesForCNN, gUpdateInfo,
             keepCheckPointFiles, adaptationRegType, adaptationRegWeight,
             trainCriterionNodeName, evalCriterionNodeName, doGradientCheck,
             gradientCheckSigDigit, validateAfterModelReloading, rpi,
             learnRateAdjustInterval, UsingAllDataForPreComputedNode,
             needAveMultiplier, L2RegWeight, L1RegWeight,
             autoAdjustMinibatch, minibatchSizeTuningFrequency, minibatchSizeTuningMax);
    }

    //autoLearnRateSearchType is applied only if the learning rate for the epoch is not specified in learningRatesPerMB and learningRatesPerSample
    void Init(const floatargvector& learningRatesPerMB,
              const floatargvector& learningRatesPerSample,
              const intargvector& mbSize,
              const size_t epochSize,
              const size_t maxEpochs,
              const wstring& modelPath,
              const floatargvector& momentumPerMB,
              const floatargvector& momentumPerSample,
              const bool gradientClippingWithTruncation = true,
              const ElemType clippingThresholdPerSample = std::numeric_limits<ElemType>::infinity(),
              const LearningRateSearchAlgorithm autoLearnRateSearchType = LearningRateSearchAlgorithm::None,
              const ElemType increaseLearnRateIfImproveMoreThan = std::numeric_limits<ElemType>::infinity(),
              const ElemType learnRateIncreaseFactor = 1.382f,
              const ElemType reduceLearnRateIfImproveLessThan = 0,
              const bool continueReduce = false,
              const ElemType learnRateDecreaseFactor = 0.618f,
              floatargvector dropoutRates = floatargvector(L"0.0f"),
              const bool loadBestModel = true,
              const intargvector& numMiniBatch4LRSearch = intargvector(L"500"),
              const size_t numPrevLearnRates = 5,
              const size_t numBestSearchEpoch = 1,
              const int traceLevel = 0,
              const size_t numMBsToShowResult = 10,
              const size_t numMBsToCUDAProfile = 0,
              const size_t maxTempMemSizeInSamplesForCNN = 0,
              const GradientUpdateInfo gradUpdateType = GradientUpdateInfo(),
              const bool keepCheckPointFiles = false,
              const AdaptationRegType adaptationRegType = AdaptationRegType::None,
              const ElemType adaptationRegWeight = 0.0f,
              const wstring trainCriterionNodeName = L"",
              const wstring evalCriterionNodeName = L"",
              const bool doGradientCheck = false,
              const ElemType gradientCheckSigDigit = 6,
              const bool validateAfterModelReloading = true,
              RMSPropInfo rpi = RMSPropInfo(),
              size_t learnRateAdjustInterval = 1,
              const bool UsingAllDataForPreComputed = true,
              const bool needAveMultiplier = true,
              const ElemType L2RegWeight = 0,
              const ElemType L1RegWeight = 0,
              const bool autoAdjustMinibatch = false,
              const size_t minibatchSizeTuningFrequency = 1,
              const size_t minibatchSizeTuningMax = 1048576)
    {
        m_numPrevLearnRates = numPrevLearnRates;
        m_prevChosenMinibatchSize = 0;
        m_autoAdjustMinibatch = autoAdjustMinibatch;
        m_minibatchSizeTuningMax = minibatchSizeTuningMax;
        m_minibatchSizeTuningFrequency = minibatchSizeTuningFrequency;

        m_mbSize = mbSize;

        // the number of samples in each epoch (0 means, use all the samples in each epoch).
        m_epochSize = epochSize;
        if (m_epochSize == 0)
        {
            m_epochSize = requestDataSize;
        }

        // the total number of epochs to run.
        m_maxEpochs = maxEpochs;

        m_gradientClippingWithTruncation = gradientClippingWithTruncation;
        m_modelPath = modelPath;
        m_autoLearnRateSearchType = autoLearnRateSearchType;
        m_traceLevel = traceLevel;
        m_loadBestModel = loadBestModel;
        m_increaseLearnRateIfImproveMoreThan = increaseLearnRateIfImproveMoreThan;
        m_learnRateIncreaseFactor = learnRateIncreaseFactor;
        m_reduceLearnRateIfImproveLessThan = reduceLearnRateIfImproveLessThan;
        m_continueReduce = continueReduce;

        //minimum interval is 1 epoch
        m_learnRateAdjustInterval = max((size_t) 1, learnRateAdjustInterval);

        m_learnRateDecreaseFactor = learnRateDecreaseFactor;
        m_clippingThresholdPerSample = abs(clippingThresholdPerSample);
        m_numMiniBatch4LRSearch = numMiniBatch4LRSearch;
        m_dropoutRates = dropoutRates;
        m_numMBsToShowResult = int(numMBsToShowResult);
        m_numMBsToCUDAProfile = int(numMBsToCUDAProfile);
        m_numBestSearchEpoch = numBestSearchEpoch;
        m_maxTempMemSizeInSamplesForCNN = maxTempMemSizeInSamplesForCNN;
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

        for (size_t i = 0; i < m_mbSize.size(); i++)
        {
            if (m_epochSize != requestDataSize && m_epochSize < m_mbSize[i])
            {
                throw std::invalid_argument("epoch size must be larger than mbsize.");
            }
        }

        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None &&
            (learningRatesPerSample.size() == 0 && learningRatesPerMB.size() == 0))
        {
            throw std::invalid_argument(
                "If autoLearnRateSearchType is false you must specify the "
                "learningRatesPerSample or learningRatesPerMB parameter.");
        }

        if (learningRatesPerSample.size() > 0 && learningRatesPerMB.size() > 0)
        {
            throw std::invalid_argument(
                "You specified both learningRatesPerSample and learningRatesPerMB. Please comment out one of them.");
        }
        else if (learningRatesPerSample.size() > 0)
        {
            m_learningRatesPerSample = learningRatesPerSample;
        }
        else if (learningRatesPerMB.size() > 0)
        {
            int LRSize = (int) max(learningRatesPerMB.size(), m_mbSize.size());
            m_learningRatesPerSample.resize(LRSize);
            for (int i = 0; i < LRSize; i++)
            {
                m_learningRatesPerSample[i] = learningRatesPerMB[i] / m_mbSize[i];
            }
            m_needToNormalizeLRByParallUtterance = true;
        }

        if (momentumPerSample.size() > 0 && momentumPerMB.size() > 0)
        {
            throw std::invalid_argument(
                "You specified both momentumPerSample and momentumPerMB. Please comment out one of them.");
        }
        else if (momentumPerSample.size() > 0)
        {
            m_momentumPerSample = momentumPerSample;
            int momentumVectorSize = m_momentumPerSample.size();
            for (int i = 0; i < momentumVectorSize; i++)
            {
                if ((m_momentumPerSample[i] >= 1) || (m_momentumPerSample[i] < 0))
                {
                    throw std::invalid_argument("momentumPerSample must be in [0, 1).");
                }
            }
        }
        else if (momentumPerMB.size() > 0)
        {
            int momentumVectorSize = (int)max(momentumPerMB.size(), m_mbSize.size());
            m_momentumPerSample.resize(momentumVectorSize);
            for (int i = 0; i < momentumVectorSize; i++)
            {
                if ((momentumPerMB[i] >= 1) || (momentumPerMB[i] < 0))
                {
                    throw std::invalid_argument("momentumPerMB must be in [0, 1).");
                }
                m_momentumPerSample[i] = (float)pow(momentumPerMB[i], 1.0 / m_mbSize[i]); 
            }

            m_needToNormalizeMomentumByParallUtterance = true;
        }
        else
        {
            int momentumVectorSize = m_mbSize.size();
            m_momentumPerSample.resize(momentumVectorSize);
            for (int i = 0; i < momentumVectorSize; i++)
            {
                m_momentumPerSample[i] = (float)pow(0.9f, 1.0 / m_mbSize[i]);
            }
        }

        if (m_learnRateDecreaseFactor > 1 || m_learnRateIncreaseFactor < 1)
        {
            throw std::invalid_argument(
                "learnRateIncreaseFactor must be >= 1 and learnRateDecreaseFactor must be <= 1.");
        }

        for (size_t i = 0; i < m_dropoutRates.size(); i++)
        {
            if (m_dropoutRates[i] >= 1 || m_dropoutRates[i] < 0)
            {
                throw std::invalid_argument("dropoutRate must be >= 0 and < 1.");
            }
        }

        if (m_adaptationRegWeight > 1 || m_adaptationRegWeight < 0)
        {
            throw invalid_argument("adaptationRegWeight must be in [0 1]");
        }

        m_minLearnRate = 1e-9f;

        m_needRegularization = false;

        m_doGradientCheck = doGradientCheck;
        m_gradientCheckSigDigit = gradientCheckSigDigit;
        m_validateAfterModelReloading = validateAfterModelReloading;

        msra::files::make_intermediate_dirs(m_modelPath);
    }

    void Adapt(wstring origModelFileName, wstring refNodeName,
               IDataReader<ElemType>* trainSetDataReader,
               IDataReader<ElemType>* validationSetDataReader,
               const DEVICEID_TYPE deviceID, const bool makeMode = true)
    {
        if (origModelFileName == L"" || trainSetDataReader == nullptr)
        {
            throw std::invalid_argument("origModel and trainSetDataReader should not be null.");
        }

        int startEpoch = DetermineStartEpoch(makeMode);
        if (startEpoch == m_maxEpochs)
        {
            fprintf(stderr, "Final model exists. No further training is necessary.\n");
            return;
        }

        ComputationNetwork<ElemType> net(deviceID);
        if (startEpoch >= 0)
        {
            wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
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
            {
                throw invalid_argument("refNodeName does not exist and is needed when adaptationRegType is KL.");
            }

            refNode = refNet.GetNodeFromName(refNodeName);
        }

        TrainOrAdaptModel(startEpoch, net, refNet, refNode, trainSetDataReader, validationSetDataReader);
    }

    void SequenceTrain(IComputationNetBuilder<ElemType>* netBuilder, wstring origModelFileName, IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader, const DEVICEID_TYPE deviceID, const bool makeMode = true)
    {
        if (netBuilder == nullptr || origModelFileName == L"" || trainSetDataReader == nullptr)
        {
            throw std::invalid_argument ("netBuilder, origModel and trainSetDataReader should not be null.");
        }

        int startEpoch = DetermineStartEpoch(makeMode);
        if (startEpoch == m_maxEpochs)
        {
            fprintf(stderr, "Final model exists. No further training is necessary.\n");
            return;
        }

        // Initializes the model from original model.
        ComputationNetwork<ElemType> origNet(deviceID);
        ComputationNetwork<ElemType>* sequenceNet = 
            (startEpoch < 0) ? netBuilder->BuildNetworkFromDescription() : &origNet;
        std::vector<ComputationNodePtr> addedFeatureNodes;
        std::vector<ComputationNodePtr> replacedCriterionNodes;
        if (startEpoch < 0)
        {
            // Loads models.
            origNet.LoadFromFile(origModelFileName);

            // Processes feature nodes.
            std::vector<ComputationNodePtr> *sequenceFeatureNodes = sequenceNet->FeatureNodes();
            for (size_t i = 0; i < sequenceFeatureNodes->size(); ++i)
            {
                if (!origNet.NodeNameExist((*sequenceFeatureNodes)[i]->NodeName()))
                {
                    addedFeatureNodes.push_back((*sequenceFeatureNodes)[i]);
                    origNet.AddFeatureNode((*sequenceFeatureNodes)[i]);
                }
            }

            // Processes criterion nodes.
            std::vector<ComputationNodePtr> * origCriterionNodes = GetTrainCriterionNodes(origNet);
            std::vector<ComputationNodePtr> * sequenceCriterionNodes = GetTrainCriterionNodes(*sequenceNet);
            if (origCriterionNodes->size() == 0 || sequenceCriterionNodes->size() == 0)
            {
                throw std::runtime_error("Training criterion node does not exist.");
            }
            replacedCriterionNodes.push_back((*origCriterionNodes)[0]);
            origNet.ReplaceFinalCriterionNode((*origCriterionNodes)[0]->NodeName(), (*sequenceCriterionNodes)[0]);
            origNet.ResetEvalTimeStamp();
        }

        wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
        if (startEpoch >= 0)
        {
            fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());
        }
        else
        {
            fprintf(stderr, "Load Network From the original model file %ls.\n", origModelFileName.c_str());
        }
        ComputationNetwork<ElemType> *net =
            (startEpoch < 0) ? &origNet : netBuilder->LoadNetworkFromFile(modelFileName);

        startEpoch = max(startEpoch, 0);

        TrainOrAdaptModel(startEpoch, *net, *net, nullptr, trainSetDataReader, validationSetDataReader);

        // Handles deletions carefully here.
        if (startEpoch < 0)
        {
            for (size_t i = 0; i < addedFeatureNodes.size(); ++i)
            {
                origNet.RemoveFeatureNode(addedFeatureNodes[i]);
            }
            std::vector<ComputationNodePtr> * origCriterionNodes = GetTrainCriterionNodes(origNet);
            origNet.ReplaceFinalCriterionNode((*origCriterionNodes)[0]->NodeName(), replacedCriterionNodes[0]);
        }
    }

    void Train(IComputationNetBuilder<ElemType>* netBuilder,
               IDataReader<ElemType>* trainSetDataReader,
               IDataReader<ElemType>* validationSetDataReader,
               const bool makeMode = true)
    {
        if (netBuilder == nullptr || trainSetDataReader == nullptr)
        {
            throw std::invalid_argument("netBuilder and trainSetDataReader should not be null.\n");
        }
        int startEpoch = DetermineStartEpoch(makeMode);
        if (startEpoch == m_maxEpochs)
        {
            fprintf(stderr, "Final model exists. No further training is necessary.\n");
            return;
        }

        wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
        if (startEpoch >= 0)
        {
            fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());
        }

        ComputationNetwork<ElemType>* net = startEpoch < 0 ? netBuilder->BuildNetworkFromDescription() :
                                                             netBuilder->LoadNetworkFromFile(modelFileName);
        // TODO: BUGBUG: if not starting from checkpoint, need to synchronize initial model
        // strategy should be to run the initializer above on mpiRank==0, and then broadcast parameters.

        /*  if (m_doUnitTest)
        {
            if (net.UnitTest() == false)
                LogicError("unit test on decoder network not passed");

            return;
        }*/

        startEpoch = max(startEpoch, 0);
        m_needRegularization = false;

        TrainOrAdaptModel(startEpoch, *net, *net, nullptr, trainSetDataReader, validationSetDataReader);
    }

protected:
    std::vector<ComputationNodePtr>* GetTrainCriterionNodes(ComputationNetwork<ElemType>& net)
    {
        fprintf(stderr, "GetTrainCriterionNodes %ls ...\n", m_trainCriterionNodeName.c_str());
        if (!m_trainCriterionNodeName.empty())
        {
            return net.TrainCriterionNodesFrom(m_trainCriterionNodeName);
        }
        else
        {
            return net.FinalCriterionNodes();
        }
    }

    std::vector<ComputationNodePtr>* GetEvalCriterionNodes(ComputationNetwork<ElemType>& net)
    {
        fprintf(stderr, "GetEvalCriterionNodes %ls ...\n", m_evalCriterionNodeName.c_str());
        if (!m_evalCriterionNodeName.empty())
        {
            return net.EvalCriterionNodesFrom(m_evalCriterionNodeName);
        }
        else
        {
            return net.EvaluationNodes();
        }
    }

    void TrainOrAdaptModel(int startEpoch, ComputationNetwork<ElemType>& net,
                           ComputationNetwork<ElemType>& refNet,
                           ComputationNodePtr refNode,
                           IDataReader<ElemType>* trainSetDataReader,
                           IDataReader<ElemType>* validationSetDataReader)
    {
        std::vector<ComputationNodePtr> *FeatureNodes = net.FeatureNodes();
        std::vector<ComputationNodePtr> *labelNodes = net.LabelNodes();
        std::vector<ComputationNodePtr> *criterionNodes = GetTrainCriterionNodes(net);
        std::vector<ComputationNodePtr> *evaluationNodes = GetEvalCriterionNodes(net);

        std::map<std::wstring, Matrix<ElemType>*>* inputMatrices = new std::map<std::wstring, Matrix<ElemType>*>();
        for (size_t i = 0; i < (*FeatureNodes).size(); i++)
        {
            (*inputMatrices)[(*FeatureNodes)[i]->NodeName()] = &(*FeatureNodes)[i]->FunctionValues();
        }

        for (size_t i = 0; i < labelNodes->size(); i++)
        {
            (*inputMatrices)[(*labelNodes)[i]->NodeName()] = &(*labelNodes)[i]->FunctionValues();
        }

        // used for KLD regularized adaptation. For all other adaptation techniques
        // use MEL to edit the model and using normal training algorithm
        std::vector<ComputationNodePtr> refFeatureNodes;
        if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
        {
            refFeatureNodes.resize((*FeatureNodes).size());
            for (size_t i = 0; i < (*FeatureNodes).size(); i++)
            {
                //we need to keep this info to handle deletion
                refFeatureNodes[i] = refNet.GetNodeFromName((*FeatureNodes)[i]->NodeName());
                refNet.ChangeNode((*FeatureNodes)[i]->NodeName(), (*FeatureNodes)[i]);
            }

            refNet.RebuildNetwork(refNode);
        }

        //initializing weights and gradient holder
        //only one criterion so far TODO: support multiple ones?
        std::list<ComputationNodePtr>* learnableNodes = net.LearnableNodes((*criterionNodes)[0]);
        std::list<Matrix<ElemType>> smoothedGradients;

        for (auto nodeIter = learnableNodes->begin(); nodeIter != learnableNodes->end(); nodeIter++)
        {
            ComputationNodePtr node = (*nodeIter);
            smoothedGradients.push_back(Matrix<ElemType>(node->FunctionValues().GetNumRows(),
                                                         node->FunctionValues().GetNumCols(),
                                                         net.GetDeviceID()));
        }

        ElemType epochCriterion, avgCriterion, prevCriterion;
        epochCriterion = avgCriterion = prevCriterion = std::numeric_limits<ElemType>::infinity();
        size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

        std::vector<ElemType> epochEvalErrors((*evaluationNodes).size(), std::numeric_limits<ElemType>::infinity());

        std::vector<wstring> evalNodeNames;
        for (size_t i = 0; i < evaluationNodes->size(); i++)
        {
            evalNodeNames.push_back((*evaluationNodes)[i]->NodeName());
        }

        size_t totalSamplesSeen = 0;
        ElemType learnRatePerSample = 0.5f / m_mbSize[startEpoch];

        ElemType learningRateAdjustmentFactor = 1.0f;
        vector<ElemType> prevLearnRates;
        prevLearnRates.resize(m_numPrevLearnRates);
        for (int i = 0; i < m_numPrevLearnRates; i++)
        {
            prevLearnRates[i] = ElemType(-1);
        }

        //precompute mean and invStdDev nodes and save initial model
        if (PreCompute(net, trainSetDataReader, FeatureNodes, labelNodes, inputMatrices) || startEpoch == 0)
        {
            if (mpiRank == 0)
            {
                // only needs to be done by one process
                net.SaveToFile(GetModelNameForEpoch(int(startEpoch) - 1));
            }
        }

        // first, we need to normalize the effect of nbruttsineachrecurrentiter
        if (trainSetDataReader->NumberSlicesInEachRecurrentIter() > 1 && m_needToNormalizeLRByParallUtterance)
        {
            for (auto& x : m_learningRatesPerSample)
            {
                x /= trainSetDataReader->NumberSlicesInEachRecurrentIter();
            }
        }
        
        // first, we need to normalize the effect of nbruttsineachrecurrentiter for momemtum
        if (trainSetDataReader->NumberSlicesInEachRecurrentIter() > 1 && m_needToNormalizeMomentumByParallUtterance)
        {
            for (auto& x : m_momentumPerSample)
            {
                x = (float)pow(x, 1.0 / trainSetDataReader->NumberSlicesInEachRecurrentIter());
            }
        }

        bool learnRateInitialized = false;
        if (startEpoch > 0)
        {
            learnRateInitialized = LoadCheckPointInfo(startEpoch - 1,
                                                      /*out*/ totalSamplesSeen,
                                                      /*out*/ learnRatePerSample,
                                                      smoothedGradients,
                                                      /*out*/ prevCriterion,
                                                      /*out*/ m_prevChosenMinibatchSize);
            if (learnRateInitialized)
            {
                prevLearnRates[startEpoch % m_numPrevLearnRates] = learnRatePerSample;
            }
        }

        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch &&
            !learnRateInitialized && m_learningRatesPerSample.size() <= startEpoch)
        {
            throw std::invalid_argument(
                "When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, "
                "or an explicit learning rate must be specified in config for the starting epoch.");
        }

        unsigned long dropOutSeed = 1;
        ElemType prevDropoutRate = 0;

        bool learnRateReduced = false;

        SetMaxTempMemSizeForCNN(net, (*criterionNodes)[0], m_maxTempMemSizeInSamplesForCNN);
        if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
        {
            SetMaxTempMemSizeForCNN(refNet, refNode, m_maxTempMemSizeInSamplesForCNN);
        }

        for (int i = startEpoch; i < (int)m_maxEpochs; i++)
        {
            Timer timer;
            timer.Start();

            // set dropout rate
            SetDropoutRate(net, (*criterionNodes)[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);

            // learning rate adjustment
            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None ||
                (m_learningRatesPerSample.size() > 0 && m_learningRatesPerSample.size() > i))
            {
                learnRatePerSample = m_learningRatesPerSample[i];
            }
            else if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
            {
                ElemType largestPrevLearnRatePerSample = prevLearnRates[0];
                for (int j = 1; j < m_numPrevLearnRates; j++)
                {
                    largestPrevLearnRatePerSample = max(largestPrevLearnRatePerSample, prevLearnRates[j]);
                }

                // return a reasonable learning rate based on the initial minibatchSize
                ElemType newLearningRatePerSample = SearchForBestLearnRate(net, refNet, refNode, i, learnRatePerSample,
                                                                           trainSetDataReader, FeatureNodes, labelNodes,
                                                                           criterionNodes, evaluationNodes, inputMatrices,
                                                                           learnableNodes, smoothedGradients,
                                                                           learnRateInitialized, largestPrevLearnRatePerSample);
                learningRateAdjustmentFactor = newLearningRatePerSample / learnRatePerSample;
                learnRatePerSample = newLearningRatePerSample;

                // save per sample learn rate to support changeable minibatchSize
                prevLearnRates[i % m_numPrevLearnRates] = learnRatePerSample;
            }

            learnRateInitialized = true;

            if (learnRatePerSample < m_minLearnRate)
            {
                fprintf(stderr, "Learn Rate Per Sample for Epoch[%d] = %.8g is less than minLearnRate %.8g. Training stops.\n",
                        i + 1, learnRatePerSample, m_minLearnRate);
                if (m_autoLearnRateSearchType != LearningRateSearchAlgorithm::None)
                {
                    net.SaveToFile(m_modelPath);
                }
                break;
            }

            size_t chosenMinibatchSize;
            size_t actualMinibatchSize;

            // Through the command line or config file the user can set minibatch sizes on a per epoch
            // basis for a set number of epochs.  For epochs after that point, m_mbSize.size(), either
            // we just keep using
            // the last minibatch size, or we use tuning to try and find a better one.
            if (m_autoAdjustMinibatch && i >= m_mbSize.size())
            {
                size_t numFramesToUseInSearch = m_numMiniBatch4LRSearch[i] * m_mbSize[i];
                if (m_epochSize != requestDataSize)
                {
                    // ensure the numFramesToUseInSearch does not exceed the total number of frames in the epoch
                    numFramesToUseInSearch = min(numFramesToUseInSearch, m_epochSize);
                }

                // Use tuning to try and find a better minibatch size
                chosenMinibatchSize = AdaptiveMinibatchSizing(net, refNet, refNode, i,
                                                              numFramesToUseInSearch,
                                                              trainSetDataReader, learnRatePerSample,
                                                              m_mbSize[i], FeatureNodes, labelNodes,
                                                              criterionNodes, evaluationNodes,
                                                              inputMatrices, learnableNodes,
                                                              smoothedGradients, learningRateAdjustmentFactor);
            }
            else
            {
                // use the explicitly set minibatch size
                chosenMinibatchSize = m_mbSize[i];
            }
            
            actualMinibatchSize = chosenMinibatchSize;
            if (trainSetDataReader->NumberSlicesInEachRecurrentIter() > 1 && m_needToNormalizeMomentumByParallUtterance)
            {
                actualMinibatchSize = chosenMinibatchSize * trainSetDataReader->NumberSlicesInEachRecurrentIter();
            }



            fprintf(stderr, "Starting Epoch %d: learning rate per sample = %f  momentum = %f \n",
                i + 1, learnRatePerSample, MomentumPerMB(m_momentumPerSample[i], actualMinibatchSize));

#ifdef MPI_SUPPORT
            INT32 mySamples = (INT32)
#endif
            TrainOneEpoch(net,
                          refNet, 
                          refNode, 
                          i, 
                          m_epochSize,
                          trainSetDataReader, 
                          learnRatePerSample, 
                          chosenMinibatchSize, 
                          FeatureNodes,
                          labelNodes, 
                          criterionNodes, 
                          evaluationNodes,
                          inputMatrices, 
                          learnableNodes, smoothedGradients,
                          epochCriterion, epochEvalErrors, totalSamplesSeen);

            timer.Stop();
            double epochTime = timer.ElapsedSeconds();

            fprintf(stderr,
                    "Finished Epoch[%d]: [Training Set] TrainLossPerSample = %.8g; ",
                    i + 1, epochCriterion);
            if (epochEvalErrors.size() == 1)
            {
                fprintf(stderr,
                        "EvalErrPerSample = %.8g; Ave LearnRatePerSample = %.10g; EpochTime=%.8g\n",
                        epochEvalErrors[0], learnRatePerSample, epochTime);
            }
            else
            {
                fprintf(stderr, "EvalErrPerSample ");
                for (size_t j = 0; j < epochEvalErrors.size(); j++)
                {
                    fprintf(stderr, "[%lu]=%.8g; ", j, epochEvalErrors[j]);
                }

                fprintf(stderr, "Ave LearnRatePerSample = %.10g; Epoch Time=%.8g\n",
                        learnRatePerSample, epochTime);
                fprintf(stderr, "Finished Epoch[%d]: Criterion Node [%ls] Per Sample = %.8g\n",
                                i + 1, (*criterionNodes)[0]->NodeName().c_str(), epochCriterion);

                for (size_t j = 0; j < epochEvalErrors.size(); j++)
                {
                    fprintf(stderr, "Finished Epoch[%d]: Evaluation Node [%ls] Per Sample = %.8g\n",
                            i + 1, evalNodeNames[j].c_str(), epochEvalErrors[j]);
                }
            }

#ifdef MPI_SUPPORT
            // model reduction and averaging
            if (mpiNumProcesses > 0)
            {
                ElemType factor; // weight for the parameter of my model
                {
                    // compute total minibatch size
                    INT32 allSamples = 0;
                    MPI_Allreduce(&mySamples, &allSamples, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                    if (allSamples == 0)
                    {
                        allSamples = 1;
                    }

                                factor = (ElemType)mySamples / (ElemType)allSamples;
                }

                            for (auto nodeIter = learnableNodes->begin(); nodeIter != learnableNodes->end(); nodeIter++)
                {
                    ComputationNodePtr node = (*nodeIter);
                    Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->FunctionValues();

                    // weight model by relative size of minibatch samples (and number of processors, for averaging)
                    ElemType *px = mat.CopyToArray();
                    size_t nx = mat.GetNumElements();
                    transform(px,
                              px + nx,
                              px,
                              [factor](ElemType&val)->ElemType {
                                  return val * factor;
                              });

                    // TODO: Replace default Allreduce with the reduction-shuffle-dance
                    vector<ElemType> py = vector<ElemType>(nx, ElemType(0));
                    MPI_Allreduce(px, &(py[0]), (int)nx, sizeof(ElemType) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), &(py[0]));
                    delete px;
                }
            }
#endif

            // only evaluate once, on the master process. TODO: This could be faster by farming out the validation parts
            if (mpiRank == 0)
            {
                if (validationSetDataReader != trainSetDataReader && validationSetDataReader != nullptr)
                {
                    SimpleEvaluator<ElemType> evalforvalidation(net);
                    vector<wstring> cvSetTrainAndEvalNodes;
                                cvSetTrainAndEvalNodes.push_back((*criterionNodes)[0]->NodeName());
                                cvSetTrainAndEvalNodes.push_back((*evaluationNodes)[0]->NodeName());

                                vector<ElemType> vScore = evalforvalidation.Evaluate(validationSetDataReader, cvSetTrainAndEvalNodes, m_mbSize[i]);
                    fprintf(stderr, "Finished Epoch[%d]: [Validation Set] TrainLossPerSample = %.8g; EvalErrPerSample = %.8g\n",
                            i + 1, vScore[0], vScore[1]);

                    epochCriterion = vScore[0]; //the first one is the training criterion.
                }
            }

#ifdef MPI_SUPPORT
            // ensure all processes have the same epochCriterion
            MPI_Bcast(&epochCriterion, 1, sizeof(epochCriterion) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

            bool loadedPrevModel = false;
            size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
            if (avgCriterion == std::numeric_limits<ElemType>::infinity())
            {
                avgCriterion = epochCriterion;
            }
            else
            {
                avgCriterion = ((epochsSinceLastLearnRateAdjust - 1 - epochsNotCountedInAvgCriterion) *
                                avgCriterion + epochCriterion) /
                                (epochsSinceLastLearnRateAdjust - epochsNotCountedInAvgCriterion);
            }

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch &&
                m_learningRatesPerSample.size() <= i && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
            {
                if (std::isnan(avgCriterion) || (prevCriterion - avgCriterion < 0 && prevCriterion != std::numeric_limits<ElemType>::infinity()))
                {
                    if (m_loadBestModel)
                    {
                        net.LoadPersistableParametersFromFile(GetModelNameForEpoch(i - 1),
                                                              m_validateAfterModelReloading);
                        net.ResetEvalTimeStamp();
                        LoadCheckPointInfo(i - 1,
                                           /*out*/ totalSamplesSeen,
                                           /*out*/ learnRatePerSample,
                                           smoothedGradients,
                                           /*out*/ prevCriterion,
                                           /*out*/ m_prevChosenMinibatchSize);
                        fprintf(stderr, "Loaded the previous model which has better training criterion.\n");
                        loadedPrevModel = true;
                    }
                }

                if (m_continueReduce)
                {
                    if (std::isnan(avgCriterion) || 
                        (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion &&
                        prevCriterion != std::numeric_limits<ElemType>::infinity()))
                    {
                        if (learnRateReduced == false)
                        {
                            learnRateReduced = true;
                        }
                        else
                        {
                            if (mpiRank == 0)
                            {
                                net.SaveToFile(GetModelNameForEpoch(i, true));
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
                    if (std::isnan(avgCriterion) || 
                        (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion &&
                        prevCriterion != std::numeric_limits<ElemType>::infinity()))
                    {

                        learnRatePerSample *= m_learnRateDecreaseFactor;
                        fprintf(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                    }
                    else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan * prevCriterion &&
                             prevCriterion != std::numeric_limits<ElemType>::infinity())
                    {
                        learnRatePerSample *= m_learnRateIncreaseFactor;
                        fprintf(stderr, "learnRatePerSample increased to %.8g\n", learnRatePerSample);
                    }
                }
            }
            else
            {
                if (std::isnan(avgCriterion))
                    RuntimeError("The training criterion is not a number (NAN). Stop\n");
            }

            //not loading previous values then set them
            if (!loadedPrevModel && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
            {
                prevCriterion = avgCriterion;
                epochsNotCountedInAvgCriterion = 0;
            }

            //persist model and check-point info
            if (mpiRank == 0)
            {
                net.SaveToFile(GetModelNameForEpoch(i));
                SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, chosenMinibatchSize);
                if (!m_keepCheckPointFiles)
                {
                    //delete previous checkpiont file to save space
                    _wunlink(GetCheckPointFileNameForEpoch(i - 1).c_str());
                }
            }

            if (learnRatePerSample < 1e-12)
            {
                fprintf(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n",
                        learnRatePerSample);
            }
        }

        // since we linked feature nodes. we need to remove it from the deletion
        if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
        {
            for (size_t i = 0; i < refFeatureNodes.size(); i++)
            {
                // note we need to handle deletion carefully
                refNet.ChangeNode(refFeatureNodes[i]->NodeName(), refFeatureNodes[i]);
            }
        }

        delete inputMatrices;
    }

protected:
    // return true if precomputation is executed.
    bool PreCompute(ComputationNetwork<ElemType>& net,
                    IDataReader<ElemType>* trainSetDataReader,
                    std::vector<ComputationNodePtr>* FeatureNodes,
                    std::vector<ComputationNodePtr>* labelNodes,
                    std::map<std::wstring, Matrix<ElemType>*>* inputMatrices)
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
            PreComputedNode<ElemType>* node = static_cast<PreComputedNode<ElemType>*>(*nodeIter);
            fprintf(stderr, "\tNodeName: %ls\n", (node->NodeName()).c_str());
        }

        //compute
        //trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , requestDataSize);
        // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , m_epochSize); // only based on one epoch
        // [1/12/2015 erw] to support large dataset, we usually paritition whole dataset into several epoches, so we need to use all the data to do precomputing
        if (m_useAllDataForPreComputedNode)
        {
            // using all the data
            trainSetDataReader->StartMinibatchLoop(m_mbSize[0], 0);
        }
        else {
            // using all the data
            trainSetDataReader->StartMinibatchLoop(m_mbSize[0], 0, m_epochSize);
        }

        while (trainSetDataReader->GetMinibatch(*inputMatrices))
        {
            UpdateEvalTimeStamps(FeatureNodes);
            UpdateEvalTimeStamps(labelNodes);

            size_t actualMBSize = net.GetActualMBSize();
            net.SetActualMiniBatchSize(actualMBSize);
            net.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
            trainSetDataReader->SetSentenceSegBatch(net.SentenceBoundary(), net.MinibatchPackingFlags());

            for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
            {
                net.Evaluate(*nodeIter);
            }
        }

        // mark done
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            PreComputedNode<ElemType>* node = static_cast<PreComputedNode<ElemType>*>(*nodeIter);
            node->MarkComputed(true);
        }

        return true;
    }

    // return a reasonable initial learning rate based on the initial mbsize
    ElemType SearchForBestLearnRate(ComputationNetwork<ElemType>& net,
                                    ComputationNetwork<ElemType>& refNet,
                                    const ComputationNodePtr refNode, const int epochNumber,
                                    const ElemType curLearnRate,
                                    IDataReader<ElemType>* trainSetDataReader,
                                    const std::vector<ComputationNodePtr>* FeatureNodes,
                                    const std::vector<ComputationNodePtr>* labelNodes,
                                    const std::vector<ComputationNodePtr>* criterionNodes,
                                    const std::vector<ComputationNodePtr>* evaluationNodes,
                                    std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                    const std::list<ComputationNodePtr>* learnableNodes,
                                    std::list<Matrix<ElemType>>& smoothedGradients,
                                    const bool learnRateInitialized,
                                    const ElemType largestPrevLearnRatePerSample)
    {
        ElemType epochCriterion = std::numeric_limits<ElemType>::infinity();
        ElemType prevCriterion = std::numeric_limits<ElemType>::infinity();
        vector<ElemType> epochEvalErrors(evaluationNodes->size(), std::numeric_limits<ElemType>::infinity());

        size_t totalSamplesSeen = 0;
        ElemType bestLearnRatePerSample = curLearnRate;

        size_t numFramesToUseInSearch = m_numMiniBatch4LRSearch[epochNumber] * m_mbSize[epochNumber];
        if (m_epochSize != requestDataSize)
        {
            // ensure the numFramesToUseInSearch does not exceed the total number of frames in the epoch
            numFramesToUseInSearch = min(numFramesToUseInSearch, m_epochSize);
        }

        ElemType baseCriterion;

        ElemType minLearnRate = m_minLearnRate * 0.3f;
        ElemType learnRatePerSample = 1.0f / 8.0f / 0.618f / sqrt((ElemType)m_mbSize[epochNumber]);

        if (learnRateInitialized && largestPrevLearnRatePerSample > 0)
        {
            //largestPrevLearnRatePerSample is per sample, first 0.618f is for compensation, second one is for safety
            learnRatePerSample = largestPrevLearnRatePerSample / 0.618f / 0.618f;
        }

        int baseModelEpoch = epochNumber - 1;
        net.LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
        net.ResetEvalTimeStamp();

        ElemType learnRate = learnRatePerSample;
        size_t dummyMinibatchSize = 0;
        LoadCheckPointInfo(baseModelEpoch,
                           /*out*/ totalSamplesSeen,
                           /*out*/ learnRate,
                           smoothedGradients,
                           /*out*/ prevCriterion,
                           /*out*/ dummyMinibatchSize);

        // if model is not changed this is what we will get
        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                        numFramesToUseInSearch, trainSetDataReader, 0, m_mbSize[epochNumber],
                                        FeatureNodes, labelNodes,
                                        criterionNodes, evaluationNodes,
                                        inputMatrices, learnableNodes,
                                        smoothedGradients, /*out*/ baseCriterion,
                                        /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                                        "BaseAdaptiveLearnRateSearch:");

        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
        {
            if (prevCriterion == std::numeric_limits<ElemType>::infinity())
            {
                prevCriterion = baseCriterion;
            }

            ElemType ratio = 0.3f;

            if (m_epochSize != requestDataSize)
            {
                ratio = pow(((ElemType)numFramesToUseInSearch) / m_epochSize, 1.0f / 2);
            }

            baseCriterion = max(ratio * prevCriterion + (1 - ratio) * baseCriterion, baseCriterion);
        }

        do
        {
            learnRatePerSample *= 0.618f;
            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                            numFramesToUseInSearch, trainSetDataReader,
                                            learnRatePerSample, m_mbSize[epochNumber], FeatureNodes,
                                            labelNodes, criterionNodes,
                                            evaluationNodes, inputMatrices,
                                            learnableNodes, smoothedGradients,
                                            /*out*/ epochCriterion, /*out*/ epochEvalErrors,
                                            /*out*/ totalSamplesSeen, "AdaptiveLearnRateSearch:");

                    } while (std::isnan(epochCriterion) || (epochCriterion > baseCriterion && learnRatePerSample > minLearnRate));

        bestLearnRatePerSample = learnRatePerSample;

        //grid search for the first m_numBestSearchEpoch  epochs
        if (epochNumber < m_numBestSearchEpoch)
        {
            ElemType leftLearnRatePerSample = 0.01f / m_mbSize[epochNumber];
            ElemType rightLearnRatePerSample = learnRatePerSample;
            ElemType leftCriterion, rightCriterion = epochCriterion;

            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                            numFramesToUseInSearch, trainSetDataReader,
                                            leftLearnRatePerSample, m_mbSize[epochNumber],
                                            FeatureNodes, labelNodes,
                                            criterionNodes, evaluationNodes,
                                            inputMatrices, learnableNodes,
                                            smoothedGradients, /*out*/ leftCriterion,
                                            /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                                            "DetailBaseAdaptiveLearnRateSearch:");

            while (rightLearnRatePerSample > leftLearnRatePerSample * 1.2f)
            {
                if (rightCriterion > leftCriterion)
                {
                    rightLearnRatePerSample *= 0.618f;

                    TrainOneMiniEpochAndReloadModel(net, refNet, refNode,
                                                    epochNumber, numFramesToUseInSearch,
                                                    trainSetDataReader,
                                                    rightLearnRatePerSample, m_mbSize[epochNumber],
                                                    FeatureNodes, labelNodes,
                                                    criterionNodes,
                                                    evaluationNodes,
                                                    inputMatrices,
                                                    learnableNodes,
                                                    smoothedGradients,
                                                    /*out*/ rightCriterion,
                                                    /*out*/ epochEvalErrors,
                                                    /*out*/ totalSamplesSeen,
                                                    "DetailRightAdaptiveLearnRateSearch:");
                }
                else
                {
                    leftLearnRatePerSample /= 0.618f;

                    TrainOneMiniEpochAndReloadModel(net, refNet, refNode,
                                                    epochNumber, numFramesToUseInSearch,
                                                    trainSetDataReader,
                                                    leftLearnRatePerSample, m_mbSize[epochNumber],
                                                    FeatureNodes, labelNodes,
                                                    criterionNodes,
                                                    evaluationNodes,
                                                    inputMatrices,
                                                    learnableNodes,
                                                    smoothedGradients,
                                                    /*out*/ leftCriterion,
                                                    /*out*/ epochEvalErrors,
                                                    /*out*/ totalSamplesSeen,
                                                    "DetailLeftAdaptiveLearnRateSearch:");
                }
            }

            bestLearnRatePerSample = (leftCriterion < rightCriterion) ? leftLearnRatePerSample :
                                                                        rightLearnRatePerSample;
        }

        fprintf(stderr, "Best Learn Rate Per Sample for Epoch[%d] = %.10g  baseCriterion=%.10g\n",
                epochNumber + 1, bestLearnRatePerSample, baseCriterion);

        return bestLearnRatePerSample;
    }

    void TrainOneMiniEpochAndReloadModel(ComputationNetwork<ElemType>& net,
                                         ComputationNetwork<ElemType>& refNet,
                                         const ComputationNodePtr refNode, const int epochNumber,
                                         const size_t epochSize, IDataReader<ElemType>* trainSetDataReader,
                                         const ElemType learnRatePerSample,
                                         const size_t minibatchSize,
                                         const std::vector<ComputationNodePtr>* FeatureNodes,
                                         const std::vector<ComputationNodePtr>* labelNodes,
                                         const std::vector<ComputationNodePtr>* criterionNodes,
                                         const std::vector<ComputationNodePtr>* evaluationNodes,
                                         std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                         const std::list<ComputationNodePtr>* learnableNodes,
                                         std::list<Matrix<ElemType>>& smoothedGradients,
                                         /*out*/ ElemType& epochCriterion,
                                         /*out*/ std::vector<ElemType>& epochEvalErrors,
                                         /*out*/ size_t& totalSamplesSeen,
                                         std::string prefixMsg = "")
    {
        TrainOneEpoch(net, refNet, refNode, epochNumber, epochSize,
                      trainSetDataReader, learnRatePerSample, minibatchSize, FeatureNodes,
                      labelNodes, criterionNodes, evaluationNodes,
                      inputMatrices, learnableNodes, smoothedGradients,
                      /*out*/ epochCriterion, /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                      prefixMsg);

        fprintf(stderr, "Finished Mini-Epoch For LearnRate Selection: TrainLossPerSample = %.8g;",
                epochCriterion);

        if (epochEvalErrors.size() == 1)
        {
            fprintf(stderr, "EvalErrPerSample = %.8g; Ave LearnRatePerSample = %.10g\n",
                    epochEvalErrors[0], learnRatePerSample);
        }
        else
        {
            fprintf(stderr, "EvalErrPerSample ");
            for (size_t i = 0; i < epochEvalErrors.size(); i++)
            {
                fprintf(stderr, "[%lu] = %.8g; ", i, epochEvalErrors[i]);
            }

            fprintf(stderr, "Ave LearnRatePerSample = %.10g\n", learnRatePerSample);
        }

        int baseModelEpoch = epochNumber - 1;
        net.LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
        net.ResetEvalTimeStamp();

        ElemType dummyLearnRate;
        ElemType dummtPrevCriterion;
        size_t dummyMinibatchSize = 0;
        LoadCheckPointInfo(baseModelEpoch,
                           /*out*/ totalSamplesSeen,
                           /*out*/ dummyLearnRate,
                           smoothedGradients,
                           /*out*/ dummtPrevCriterion,
                           /*out*/ dummyMinibatchSize);
    }

    size_t AdaptiveMinibatchSizing(ComputationNetwork<ElemType>& net,
                                   ComputationNetwork<ElemType>& refNet,
                                   const ComputationNodePtr refNode,
                                   const int epochNumber,
                                   const size_t numFramesToUseInSearch,
                                   IDataReader<ElemType>* trainSetDataReader,
                                   const ElemType learnRatePerSample,
                                   const size_t initialMinibatchSize,
                                   const std::vector<ComputationNodePtr>* FeatureNodes,
                                   const std::vector<ComputationNodePtr>* labelNodes,
                                   const std::vector<ComputationNodePtr>* criterionNodes,
                                   const std::vector<ComputationNodePtr>* evaluationNodes,
                                   std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                   const std::list<ComputationNodePtr>* learnableNodes,
                                   std::list<Matrix<ElemType>>& smoothedGradients,
                                   const ElemType learningRateAdjustmentFactor)
    {
        size_t minMinibatchSize = initialMinibatchSize;
        size_t chosenMinibatchSize = initialMinibatchSize;

        // do some pre-adjustment based on LR
        // Basically we assume that the LR for epoch 1 is safe for mbsize.
        // If LR control led to a smaller LR, then we can safely increase the lower bound of the MB size.
        ElemType learningRateChangeSoFar = m_learningRatesPerSample[epochNumber] / m_learningRatesPerSample[0];
        learningRateChangeSoFar *= learningRateAdjustmentFactor;

        // increasing by the full factor is found to be too aggressive; sqrt() seems more robust
        learningRateChangeSoFar = sqrt(learningRateChangeSoFar);

        // LR was indeed reduced
        if (learningRateChangeSoFar < 1.0f)
        {
            // we can safely increase MB size (note: this may be bigger than our max)
            minMinibatchSize = (size_t)(minMinibatchSize / learningRateChangeSoFar);
        }

        if (epochNumber < 2 && m_prevChosenMinibatchSize != 0)
        {
            // newly started training: any previous MB size stored in the model is to be ignored
                        fprintf(stderr, "before epoch .2, previous minibatchSize %d is "
                     "considered invalid -> resetting\n", m_prevChosenMinibatchSize);
            m_prevChosenMinibatchSize = 0;
        }

        // check if we need to skip
        if (m_prevChosenMinibatchSize != 0 &&
            (epochNumber + 1) > m_minibatchSizeTuningFrequency &&
            (epochNumber + 1) % m_minibatchSizeTuningFrequency != 0)
        {
            fprintf(stderr, "AdaptiveMinibatchSearch: Search for a better minibatchSize "
                    "in epoch %d skipped, keeping minibatchSize of %d\n",
                    epochNumber + 1, m_prevChosenMinibatchSize);
            chosenMinibatchSize = m_prevChosenMinibatchSize;
        }
        else
        {
            if (m_prevChosenMinibatchSize != 0)
            {
                // but we don't go lower than 0.5 * the chosen previous one
                fprintf(stderr, "AdaptiveMinibatchSearch: Limiting minMinibatchSize to "
                        "previous minibatchSize = (%d / 2)\n", m_prevChosenMinibatchSize);
                minMinibatchSize = max(minMinibatchSize, m_prevChosenMinibatchSize / 2);
            }

            size_t maxMinibatchSize = m_minibatchSizeTuningMax;

            // only grow at most 2 x compared to previous step
            if (m_prevChosenMinibatchSize != 0.0f)
            {
                if (m_prevChosenMinibatchSize < chosenMinibatchSize)
                {
                    m_prevChosenMinibatchSize = chosenMinibatchSize;
                }

                fprintf(stderr, "AdaptiveMinibatchSearch: Limiting maxMinibatchSize to "
                        "previous minibatchSize %d*2\n", m_prevChosenMinibatchSize);
                maxMinibatchSize = min(maxMinibatchSize, m_prevChosenMinibatchSize * 2);
            }

            chosenMinibatchSize = SearchForBestMinibatchSize(net, refNet, refNode, epochNumber,
                                                             numFramesToUseInSearch, trainSetDataReader,
                                                             learnRatePerSample, FeatureNodes,
                                                             labelNodes, criterionNodes,
                                                             evaluationNodes, inputMatrices,
                                                             learnableNodes, smoothedGradients,
                                                             minMinibatchSize, maxMinibatchSize);
        }

        return chosenMinibatchSize;
    }

    size_t RoundToMultipleOf64(float val)
    {
                    return 64 * (size_t)((val + 32) / 64);
    }

    size_t RoundToMultipleOf64(size_t val)
    {
        return 64 * ((val + 32) / 64);
    }

    // uses a small percentage of training data of minibatch to
    // speculatively train with various MB sizes; then picks the best
    size_t SearchForBestMinibatchSize(ComputationNetwork<ElemType>& net,
                                      ComputationNetwork<ElemType>& refNet,
                                      const ComputationNodePtr refNode,
                                      const int epochNumber,
                                      const size_t numFramesToUseInSearch,
                                      IDataReader<ElemType>* trainSetDataReader,
                                      const ElemType learnRatePerSample,
                                      const std::vector<ComputationNodePtr>* FeatureNodes,
                                      const std::vector<ComputationNodePtr>* labelNodes,
                                      const std::vector<ComputationNodePtr>* criterionNodes,
                                      const std::vector<ComputationNodePtr>* evaluationNodes,
                                      std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                      const std::list<ComputationNodePtr>* learnableNodes,
                                      std::list<Matrix<ElemType>>& smoothedGradients,
                                      const size_t minMinibatchSize, const size_t maxMinibatchSize)
    {
        // may happen for automatically reduced learning rates
        if (minMinibatchSize > maxMinibatchSize)
        {
            return maxMinibatchSize;
        }

        size_t trialMinibatchSize = 0;
        bool isFirstIteration = true;
        ElemType baseCriterion = 0;

        // increase the minibatch size by a factor of sqrt(2) in each step.
        const float minibatchSizeTuningFactor = sqrtf(2.0f);

        size_t lastTriedtrialMinibatchSize = 0;
        for (float trialMinibatchSizeFloat = (float)minMinibatchSize;
             trialMinibatchSizeFloat <= maxMinibatchSize;
             trialMinibatchSizeFloat *= minibatchSizeTuningFactor)
        {
            // round mbsize to something meaningful
            trialMinibatchSize = RoundToMultipleOf64(trialMinibatchSizeFloat);

            fprintf(stderr, "\nAdaptiveMinibatchSearch: Evaluating trial minibatchSize=%d out of range %d..%d ...\n\n",
                    trialMinibatchSize, RoundToMultipleOf64(minMinibatchSize), RoundToMultipleOf64(maxMinibatchSize));

            size_t totalSamplesSeen;
                        std::vector<ElemType> epochEvalErrors(evaluationNodes->size(), std::numeric_limits<ElemType>::infinity());
            ElemType epochCriterion = std::numeric_limits<ElemType>::infinity();

            // Train on a few minibatches and so we can observe the epochCriterion as we try increasing
            // minibatches with iteration of this loop.
            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                            numFramesToUseInSearch, trainSetDataReader,
                                            learnRatePerSample, trialMinibatchSize, FeatureNodes,
                                            labelNodes, criterionNodes,
                                            evaluationNodes, inputMatrices,
                                            learnableNodes, smoothedGradients,
                                            /*out*/ epochCriterion, /*out*/ epochEvalErrors,
                                            /*out*/ totalSamplesSeen,
                                            isFirstIteration ? "BaseAdaptiveMinibatchSearch:" :
                                                               "AdaptiveMinibatchSearch:");

            if (isFirstIteration)
            {
                // for the first iteration of the loop only, set baseCriterion
                // to the result we got from TrainOneMiniEpochAndReloadModel().
                baseCriterion = epochCriterion;
                lastTriedtrialMinibatchSize = trialMinibatchSize;
                isFirstIteration = false;

                fprintf(stderr, "AdaptiveMinibatchSearch: Computed BaseCriterion %.10g\n", baseCriterion);
            }
            else if (!std::isnan(epochCriterion) && (epochCriterion > baseCriterion))
            {
                fprintf(stderr, "AdaptiveMinibatchSearch: Search successful!!! Choose new minibatchSize of %d.  "
                        "EpochCriterion = %.10g vs BaseCriterion = %.10g\n\n",
                        lastTriedtrialMinibatchSize, epochCriterion, baseCriterion);

                // As soon as we see the Criterion (a measure of error) start to get larger than the
                // Criterion we started with, we stop.
                // TODO: if this is too sensitive, we can add a margin on the bases of percentage of
                // baseCriterion.
                break;
            }
            else
            {
                lastTriedtrialMinibatchSize = trialMinibatchSize;
                fprintf(stderr, "AdaptiveMinibatchSearch: Keep searching... "
                        "EpochCriterion = %.10g vs BaseCriterion = %.10g\n",
                        epochCriterion, baseCriterion);
            }
        }

        return lastTriedtrialMinibatchSize;
    }

    // Tries to compute derivatives for the whole utterances, which will be
    // fed to the neural network as features.
    void AttemptUtteranceDerivativeFeatures(ComputationNetwork<ElemType>& net,
                                            IDataReader<ElemType>* trainSetDataReader,
                                            const std::vector<ComputationNodePtr>* FeatureNodes,
                                            std::map<std::wstring, Matrix<ElemType>*>* inputMatrices)
    {
        // Tries to read an utterance and run forward computation on the
        // whole utterance.
        assert(trainSetDataReader != NULL);
        std::wstring uttID;
                    if (trainSetDataReader->GetForkedUtterance(uttID, *inputMatrices))
        {
            UpdateEvalTimeStamps(FeatureNodes);

                        std::vector<ComputationNodePtr>* outputNodes = net.OutputNodes();
                        if (outputNodes->size() < 1)
            {
                throw std::logic_error("no output node was found.");
            }
            size_t actualMBSize = net.GetActualMBSize();
            net.SetActualMiniBatchSize(actualMBSize);
            net.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
            trainSetDataReader->SetSentenceSegBatch(net.SentenceBoundary(), net.MinibatchPackingFlags());
                        net.Evaluate((*outputNodes)[0]);   // Only evaluate the first output
                        trainSetDataReader->ComputeDerivativeFeatures(uttID, (*outputNodes)[0]->FunctionValues());
        }
    }

    size_t TrainOneEpoch(ComputationNetwork<ElemType>& net,
                         ComputationNetwork<ElemType>& refNet,
                         const ComputationNodePtr refNode,
                         const int epochNumber,
                         const size_t epochSize,
                         IDataReader<ElemType>* trainSetDataReader,
                         const ElemType learnRatePerSample,
                         size_t tunedMBSize,
                         const std::vector<ComputationNodePtr>* FeatureNodes,
                         const std::vector<ComputationNodePtr>* labelNodes,
                         const std::vector<ComputationNodePtr>* criterionNodes,
                         const std::vector<ComputationNodePtr>* evaluationNodes,
                         std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                         const std::list<ComputationNodePtr>* learnableNodes,
                         std::list<Matrix<ElemType>>& smoothedGradients,
                         /*out*/ ElemType& epochCriterion,
                         /*out*/ std::vector<ElemType>& epochEvalErrors,
                         /*out*/ size_t& totalSamplesSeen,
                         std::string prefixMsg = "")
    {
        // Since we are getting timing resolution of under microsecond we use double precision
        // to ensure that we have enough digits to represent small time measurements.
        double totalTimeInMBs = 0;
        ElemType epochCriterionLastMBs = 0;

        int numSamplesLastMBs = 0;
        std::vector<ElemType> epochEvalErrorsLastMBs(epochEvalErrors.size(), 0);

        // initialize statistics
        size_t totalEpochSamples = 0;

        int numMBsRun = 0;

        size_t numEvalNodes = epochEvalErrors.size();

        // assume only one training criterion node for each epoch

        Matrix<ElemType> localEpochCriterion(1, 1, net.GetDeviceID());
        Matrix<ElemType> localEpochEvalErrors(1, numEvalNodes, net.GetDeviceID());

        localEpochCriterion.SetValue(0);
        localEpochEvalErrors.SetValue(0);
        Profiler profiler(m_numMBsToCUDAProfile);

        // resetting this, so profiling is performed for one epoch only
        m_numMBsToCUDAProfile = 0;

        trainSetDataReader->StartMinibatchLoop(tunedMBSize, epochNumber, m_epochSize);

        AttemptUtteranceDerivativeFeatures(net, trainSetDataReader, FeatureNodes, inputMatrices);

        Timer timer;
        timer.Start();

        while (trainSetDataReader->GetMinibatch(*inputMatrices))
        {
#ifdef MPI_SUPPORT
            DecimateMinibatch(inputMatrices);
#endif
            UpdateEvalTimeStamps(FeatureNodes);
            UpdateEvalTimeStamps(labelNodes);

            size_t actualMBSize = net.GetActualMBSize();
            if (actualMBSize == 0)
            {
                continue;
            }

            net.SetActualMiniBatchSize(actualMBSize);
            net.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
            trainSetDataReader->SetSentenceSegBatch(net.SentenceBoundary(), net.MinibatchPackingFlags());

#ifndef EVALDLL
            if (m_doGradientCheck && GradientCheck(net, criterionNodes, learnableNodes, 0) == false)
            {
                throw std::logic_error("cannot pass gradient checker");
            }
#endif
            // TODO: currently only support one node regularization
            if (m_needRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
            {
                refNet.SetActualMiniBatchSize(actualMBSize);
                refNet.SetActualNbrSlicesInEachRecIter(trainSetDataReader->NumberSlicesInEachRecurrentIter());
                refNet.Evaluate(refNode);
                Matrix<ElemType>::ScaleAndAdd(m_adaptationRegWeight,
                                              refNode->FunctionValues(),
                                              1 - m_adaptationRegWeight,
                                              (*labelNodes)[0]->FunctionValues());
            }

            // only compute gradient when learning rate is large enough
            if (learnRatePerSample > m_minLearnRate * 0.01)
            {
                // use only the first criterion. Is there any possibility to use more?
                net.ComputeGradient((*criterionNodes)[0]);
            }
            else
            {
                // use only the first criterion. Is there any possibility to use more?
                net.Evaluate((*criterionNodes)[0]);
            }

            Matrix<ElemType>::AddElementToElement((*criterionNodes)[0]->FunctionValues(),
                                                  0, 0, localEpochCriterion, 0, 0);

            std::vector<ElemType> mbEvalErrors(numEvalNodes, 0);
            for (size_t i = 0; i < numEvalNodes; i++)
            {
                net.Evaluate((*evaluationNodes)[i]);
                Matrix<ElemType>::AddElementToElement((*evaluationNodes)[i]->FunctionValues(),
                                                      0, 0, localEpochEvalErrors, 0, i);
            }

            //update model parameters
            if (learnRatePerSample > m_minLearnRate * 0.01)
            {
                auto smoothedGradientIter = smoothedGradients.begin();
                            for (auto nodeIter = learnableNodes->begin(); nodeIter != learnableNodes->end(); nodeIter++, smoothedGradientIter++)
                {
                    ComputationNodePtr node = *nodeIter;
                    Matrix<ElemType>& smoothedGradient = *smoothedGradientIter;

                    UpdateWeights(node, smoothedGradient, learnRatePerSample,
                                  m_momentumPerSample[epochNumber], actualMBSize,
                                  m_L2RegWeight, m_L1RegWeight,
                                  m_needAveMultiplier);
                }
            }

            // Tries to set up derivative features for the next utterance.
            AttemptUtteranceDerivativeFeatures(net, trainSetDataReader, FeatureNodes, inputMatrices);

            timer.Stop();
            numMBsRun++;
            if (m_traceLevel > 0)
            {
                totalTimeInMBs += timer.ElapsedSeconds();
                numSamplesLastMBs += int(actualMBSize);

                if (numMBsRun % m_numMBsToShowResult == 0)
                {
                    // get the epoch Values updated
                    timer.Restart();
                    epochCriterion = localEpochCriterion.Get00Element();
                    for (size_t i = 0; i < numEvalNodes; i++)
                    {
                        epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0, i);
                    }
                    timer.Stop();

                    // Add the last trailing compute
                    totalTimeInMBs += timer.ElapsedSeconds();

                    fprintf(stderr, "%s Epoch[%d of %d]-Minibatch[%d-%d of %d]: SamplesSeen = %d; TrainLossPerSample = %.8g; ",
                            prefixMsg.c_str(), epochNumber + 1, m_maxEpochs, numMBsRun - m_numMBsToShowResult + 1,
                            numMBsRun, epochSize / tunedMBSize, numSamplesLastMBs,
                            (epochCriterion - epochCriterionLastMBs) / numSamplesLastMBs);

                    for (size_t i = 0; i < numEvalNodes; i++)
                    {
                        fprintf(stderr, "EvalErr[%lu]PerSample = %.8g; ",
                                i, (epochEvalErrors[i] - epochEvalErrorsLastMBs[i]) / numSamplesLastMBs);
                    }

                    fprintf(stderr, "TotalTime=%.8g; TotalTimePerSample=%.8g, SamplesPerSecond=%d\n",
                            totalTimeInMBs, totalTimeInMBs / numSamplesLastMBs,
                            static_cast<int>(numSamplesLastMBs / totalTimeInMBs));

                    // reset statistics
                    totalTimeInMBs = 0;
                    numSamplesLastMBs = 0;

                    epochCriterionLastMBs = epochCriterion;
                    for (size_t i = 0; i < numEvalNodes; i++)
                    {
                        epochEvalErrorsLastMBs[i] = epochEvalErrors[i];
                    }
                }
            }

            timer.Restart();
            totalEpochSamples += actualMBSize;
            totalSamplesSeen += actualMBSize;

            if (totalEpochSamples >= epochSize)
            {
                break;
            }

            // call DataEnd function
            // DataEnd does reader specific process if sentence ending is reached
            trainSetDataReader->DataEnd(endDataSentence);

            profiler.NextSample();
        }

        localEpochCriterion /= float(totalEpochSamples);
        localEpochEvalErrors /= float(totalEpochSamples);

        epochCriterion = localEpochCriterion.Get00Element();
        for (size_t i = 0; i < numEvalNodes; i++)
        {
            epochEvalErrors[i] = (const ElemType)localEpochEvalErrors(0, i);
        }

        return totalEpochSamples;
    }
public:
    // UpdateWeightsS - static version of UpdateWeights()
    static void UpdateWeightsS(const SGD* sgd, Matrix<ElemType>& functionValues,
                               Matrix<ElemType>& gradientValues,
                               Matrix<ElemType>& smoothedGradient,
                               const ElemType learnRatePerSample,
                               const ElemType momentumPerSample,
                               size_t actualMBSize,
                               const ElemType L2RegWeight,
                               const ElemType L1RegWeight,
                               const bool needAveMultiplier)
    {
        // we use simple linear (instead of log linear) scaling here
        const ElemType momentum = MomentumPerMB(momentumPerSample, actualMBSize);
#if DUMPOUTPUT
        fprintf(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
            learnRatePerSample, momentum, actualMBSize);
        fprintf(stderr, "sgd->GradUpdateType()=%d, sgd->GradientUpdateNoiseStd()=%0.8f\n",
                sgd->GradUpdateType(), sgd->GradientUpdateNoiseStd());
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
            // get the gradient structure since gradient is sparse
            sgdUpdateNoise.SetValue(gradientValues);

            // reset its value to random
            sgdUpdateNoise.SetGaussianRandomValue(0, noiseStd);
        }

        // L2 regularizer
        if (L2RegWeight > 0)
        {
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            Matrix<ElemType>::ScaleAndAdd(L2RegWeight * actualMBSize, functionValues, gradientValues);
        }

        if (adpType == GradientsUpdateType::None)
        {
            smoothedGradient.NormalGrad(gradientValues, functionValues,
                                        learnRatePerSample, momentum);
        }
        else if (adpType == GradientsUpdateType::AdaGrad ||
                (adpType == GradientsUpdateType::RmsProp && gradientValues.GetMatrixType() == MatrixType::SPARSE))
        {
            //rmsprop for sparse is not implemented yet, delegate it with adagrad

            ElemType aveMultiplier = smoothedGradient.Adagrad(gradientValues, needAveMultiplier);
            Matrix<ElemType>::ScaleAndAdd(-learnRatePerSample / aveMultiplier, gradientValues, functionValues);
        }
        else if (adpType == GradientsUpdateType::RmsProp)
        {
            ElemType aveMultiplier = smoothedGradient.RmsProp(gradientValues, (ElemType)sgd->m_rpi.gamma,
                            (ElemType)sgd->m_rpi.inc, (ElemType)sgd->m_rpi.max,
                            (ElemType)sgd->m_rpi.dec, (ElemType)sgd->m_rpi.min, needAveMultiplier);
            Matrix<ElemType>::ScaleAndAdd(-learnRatePerSample / aveMultiplier, gradientValues, functionValues);
        }

        if (noiseStd > 0)
        {
            Matrix<ElemType>::ScaleAndAdd(1.0, sgdUpdateNoise, functionValues);
        }

        // L1 regularizer with proximal gradient descent method
        if (L1RegWeight > 0)
        {
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            functionValues.InplaceSoftThreshold(learnRatePerSample * L1RegWeight * actualMBSize);
        }

#if DUMPOUTPUT
        functionValues.Print("Parameter Update");
#endif
    }

protected:
    // UpdateWeights - update the weights in
    void UpdateWeights(const ComputationNodePtr node,
                       Matrix<ElemType>& smoothedGradient,
                       const ElemType learnRatePerSample,
                       const ElemType momentumPerSample,
                       const size_t actualMBSize,
                       const ElemType L2RegWeight, const ElemType L1RegWeight,
                       const bool needAveMultiplier) const
    {
#if DUMPOUTPUT
                    fprintf(stderr, "Update_%ls\n", node->NodeName().c_str());
#endif
        UpdateWeightsS(this, node->FunctionValues(), node->GradientValues(),
                       smoothedGradient, learnRatePerSample, momentumPerSample,
                       actualMBSize, L2RegWeight, L1RegWeight,
                       needAveMultiplier);
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
            else
            {
                // norm2 normalized
                ElemType gradientNorm = gradient.FrobeniusNorm();
                if (gradientNorm > maxGradientPerMB)
                {
                    ElemType normFactor = maxGradientPerMB / gradientNorm;
                    gradient *= normFactor;
                }
            }
        }
    }

    void SaveCheckPointInfo(const size_t epoch, const size_t totalSamplesSeen,
                            const ElemType learnRatePerSample,
                            const std::list<Matrix<ElemType>>& smoothedGradients,
                            const ElemType prevCriterion,
                            const size_t minibatchSize)
    {
        wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epoch));

        File fstream(checkPointFileName,
                     FileOptions::fileOptionsBinary | FileOptions::fileOptionsWrite);
        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BCKP");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BLearnRate");
        fstream << totalSamplesSeen << learnRatePerSample << prevCriterion;
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ELearnRate");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BMinibatchSize");
        fstream << minibatchSize;
        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EMinibatchSize");

        fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BGradient");

        for (auto smoothedGradientIter = smoothedGradients.begin(); smoothedGradientIter != smoothedGradients.end(); smoothedGradientIter++)
        {
            const Matrix<ElemType>& smoothedGradient = *smoothedGradientIter;
            fstream << smoothedGradient;
        }

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EGradient");

        fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ECKP");
    }

    bool LoadCheckPointInfo(const size_t epochNumber,
                            /*out*/ size_t& totalSamplesSeen,
                            /*out*/ ElemType& learnRatePerSample,
                            std::list<Matrix<ElemType>>& smoothedGradients,
                            /*out*/ ElemType& prevCriterion,
                            /*out*/ size_t& minibatchSize)
    {
        wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epochNumber));
        if (!fexists(checkPointFileName.c_str()))
        {
            fprintf(stderr,
                    "Warning: checkpiont file is missing. learning parameters will be initialized from 0\n");
            return false;
        }

        File fstream(checkPointFileName,
                     FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);
        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCKP");

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BLearnRate");
        fstream >> totalSamplesSeen >> learnRatePerSample >> prevCriterion;
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ELearnRate");

        if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BMinibatchSize"))
        {
            fstream >> minibatchSize;
            fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EMinibatchSize");
        }
        else
        {
            minibatchSize = m_mbSize[epochNumber];
        }

        fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BGradient");

        for (auto smoothedGradientIter = smoothedGradients.begin(); smoothedGradientIter != smoothedGradients.end(); smoothedGradientIter++)
        {
            Matrix<ElemType>& smoothedGradient = *smoothedGradientIter;
            fstream >> smoothedGradient;
        }
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EGradient");

        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ECKP");

        return true;
    }

    wstring GetCheckPointFileNameForEpoch(const int epoch)
    {
        return GetModelNameForEpoch(epoch) + L".ckp";
    }

    wstring GetModelNameForEpoch(const int epoch, bool bLastModel = false)
    {
        int epoch1Base = epoch + 1;
        if (epoch1Base == m_maxEpochs || bLastModel)
        {
            return m_modelPath;
        }
        else
        {
            wstring w = msra::strfun::wstrprintf(L"%ls.%d", m_modelPath.c_str(), (int)epoch1Base);
            return w;
        }

    }

    // return -1 if nothing exists
    int DetermineStartEpoch(const bool makeMode)
    {
        if (!makeMode)
        {
            // always start from scratch
            return -1;
        }

        int firstEpoch = -1;

        wstring curEpochFile = GetModelNameForEpoch(int(m_maxEpochs) - 1);
        for (int e = int(m_maxEpochs) - 1; e >= -1; e--)
        {
            const wstring prevEpochFile = GetModelNameForEpoch(e - 1);

            if (msra::files::fuptodate(curEpochFile, prevEpochFile, false))
            {
                firstEpoch = size_t(e) + 1;
                break;
            }
            else
            {
                curEpochFile = prevEpochFile;
            }
        }

        return firstEpoch;
    }

    AdaptationRegType ParseAdaptationRegType(wstring s)
    {
        msra::strfun::tolower_ascii(s);
        if (s == L"" || s == L"none")
        {
            return AdaptationRegType::None;
        }
        else if (s == L"kl" || s == L"klreg")
        {
            return AdaptationRegType::KL;
        }
        else
        {
            throw std::invalid_argument(
                "ParseAdaptationRegType: Invalid Adaptation Regularization Type. Valid values are "
                "(None | KL)");
        }
    }

    GradientsUpdateType ParseGradUpdateType(wstring s)
    {
        msra::strfun::tolower_ascii(s);
        if (s == L"" || s == L"none" || s == L"normal" || s == L"simple")
        {
            return GradientsUpdateType::None;
        }
        else if (s == L"adagrad")
        {
            return GradientsUpdateType::AdaGrad;
        }
        else if (s == L"rmsprop")
        {
            return GradientsUpdateType::RmsProp;
        }
        else
        {
            throw std::invalid_argument(
                "ParseGradUpdateType: Invalid Gradient Updating Type. Valid values are "
                "(None | AdaGrad | RmsProp )");
        }
    }

    LearningRateSearchAlgorithm ParseLearningRateSearchType(wstring s)
    {
        msra::strfun::tolower_ascii(s);
        if (s == L"false" || s == L"none")
        {
            return LearningRateSearchAlgorithm::None;
        }
        else if (s == L"searchbeforeepoch" || s == L"beforeepoch" || s == L"before")
        {
            return LearningRateSearchAlgorithm::SearchBeforeEpoch;
        }
        else if (s == L"adjustafterepoch" || s == L"afterepoch" || s == L"after")
        {
            return LearningRateSearchAlgorithm::AdjustAfterEpoch;
        }
        else {
            throw std::invalid_argument(
                "autoAdjustLR: Invalid learning rate search type. Valid values are "
                "(None | SearchBeforeEpoch | AdjustAfterEpoch)");
        }
    }

    GradientsUpdateType GradUpdateType() const
    {
        return m_gradType.mType;
    }

    ElemType GradientUpdateNoiseStd() const
    {
        return m_gradType.mGaussianNoiseInjectStd;
    }

    static ElemType MomentumPerMB(ElemType momentumPerSample, size_t minibatchSize)
    {
        return (ElemType)pow(momentumPerSample, minibatchSize);
    }

public:

    bool GradientCheck(ComputationNetwork<ElemType>& net,
                       const std::vector<ComputationNodePtr>* criterionNodes,
                       const std::list<ComputationNodePtr>* learnableNodes,
                       int npos)
    {
        vector<string> errMsgs;

        // gradient checking
        for (auto nodeIter = learnableNodes->begin(); nodeIter != learnableNodes->end(); nodeIter++)
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

                ElemType eOrg = node->FunctionValues()(irow, icol);
                if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                {
                    node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(),
                                                                      net.GetDeviceID(), true);
                }

                node->UpdateEvalTimeStamp();

                // use only the first criterion. Is
                net.ComputeGradient((*criterionNodes)[npos]);

                if (node->GradientValues().GetMatrixType() == MatrixType::SPARSE)
                {
                    break;
                }

                //ElemType mbEvalCri =
                //criterionNode should be a scalar
                            (*criterionNodes)[npos]->FunctionValues().Get00Element();
                ElemType eGradErr = node->GradientValues()(irow, icol);
                if (node->GradientValues().GetDeviceId() != net.GetDeviceID())
                {
                    node->GradientValues().TransferFromDeviceToDevice(node->GradientValues().GetDeviceId(),
                                                                      net.GetDeviceID(), true);
                }

                ElemType ePos = eOrg + ElemType(EPSILON);
                ElemType eNeg = eOrg - ElemType(EPSILON);

                node->FunctionValues()(irow, icol) = ePos;
                if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                {
                    node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(),
                                                                      net.GetDeviceID(), true);
                }

                node->UpdateEvalTimeStamp();
                net.Evaluate((*criterionNodes)[npos]);
                //criterionNode should be a scalar

                ElemType mbEvalCriPos = (*criterionNodes)[npos]->FunctionValues().Get00Element();

                node->FunctionValues()(irow, icol) = eNeg;
                if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                {
                    node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(),
                                                                      net.GetDeviceID(), true);
                }

                node->UpdateEvalTimeStamp();
                net.Evaluate((*criterionNodes)[npos]);

                // criterionNode should be a scalar
                ElemType mbEvalCriNeg = (*criterionNodes)[npos]->FunctionValues().Get00Element();

                // back to its orginal parameter value
                node->FunctionValues()(irow, icol) = eOrg;
                if (node->FunctionValues().GetDeviceId() != net.GetDeviceID())
                {
                    node->FunctionValues().TransferFromDeviceToDevice(node->FunctionValues().GetDeviceId(),
                                                                      net.GetDeviceID(), true);
                }

                // check if they are consistent
                ElemType eGradNum = (ElemType)((mbEvalCriPos - mbEvalCriNeg) / (ePos - eNeg));
                ElemType threshold = (ElemType)pow((ElemType) 10.0,
                                                    max((ElemType) 0.0,
                                                        ceil(log10(min(fabs(eGradErr),
                                                    fabs(eGradNum))))) - (int)m_gradientCheckSigDigit);
                ElemType diff = (ElemType)fabs(eGradErr - eGradNum);
                bool wrong = (std::isnan(diff) || diff > threshold);
                if (wrong)
                {
                    fprintf(stderr, "\nd%ls Numeric gradient = %e, Error BP gradient = %e\n",
                            node->NodeName().c_str(), eGradNum, eGradErr);
                    sprintf(wstrtmp, "\nd%ls Numeric gradient = %e, Error BP gradient = %e\n",
                            node->NodeName().c_str(), eGradNum, eGradErr);
                    errMsgs.push_back(wstrtmp);
                }
            }
        }

        if (errMsgs.size() > 0)
        {
            return false;
        }

        return true;
    }

protected:

    // learning rate per sample provided outside
    floatargvector m_learningRatesPerSample;

    // only true when the user specify LearningRatePerMB and the number of parallel utterances in Reader > 1
    bool m_needToNormalizeLRByParallUtterance;
    bool m_needToNormalizeMomentumByParallUtterance;

    intargvector m_mbSize;

    // the number of samples in each epoch (0 means, use all the samples in each epoch).
    size_t m_epochSize;

    // the total number of epochs to run.
    size_t m_maxEpochs;

    floatargvector m_momentumPerSample;
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

    // determine after how many epochs the learning rate should be auto adjusted.
    size_t m_learnRateAdjustInterval;

    ElemType m_increaseLearnRateIfImproveMoreThan;
    ElemType m_learnRateIncreaseFactor;
    ElemType m_learnRateDecreaseFactor;
    size_t m_prevChosenMinibatchSize;
    bool m_autoAdjustMinibatch;
    size_t m_minibatchSizeTuningFrequency;
    size_t m_minibatchSizeTuningMax;

    floatargvector m_dropoutRates;
    size_t m_maxTempMemSizeInSamplesForCNN;

    int m_traceLevel;

    size_t m_numPrevLearnRates;

    ElemType m_minLearnRate;

    GradientUpdateInfo m_gradType;
    RMSPropInfo m_rpi;

    bool m_keepCheckPointFiles;

    int m_numMBsToShowResult;
    int m_numMBsToCUDAProfile;

    bool m_doGradientCheck;
    ElemType m_gradientCheckSigDigit;

    bool m_doUnitTest;

    bool m_validateAfterModelReloading;

    bool m_useAllDataForPreComputedNode;

    bool m_needAveMultiplier;
    ElemType m_L2RegWeight;
    ElemType m_L1RegWeight;

};
template class SGD<float>;
template class SGD<double>;

}}}
