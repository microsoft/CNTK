// SGD.cpp -- implements SGD with all bells and whistles, parallelization, randomizatiom, etc.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "SGD.h"
#include "DataReaderHelpers.h"
#include "AllReduceDistGradAggregator.h"
#include "ProgressTracing.h"

#include <map>

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    // -----------------------------------------------------------------------
    // class SGDParams
    // -----------------------------------------------------------------------

    static AdaptationRegType ParseAdaptationRegType(wstring s)
    {
        if (!_wcsicmp(s.c_str(), L"") || !_wcsicmp(s.c_str(), L"none"))
            return AdaptationRegType::None;
        else if (!_wcsicmp(s.c_str(), L"kl") || !_wcsicmp(s.c_str(), L"klReg"))
            return AdaptationRegType::KL;
        else
            InvalidArgument("ParseAdaptationRegType: Invalid Adaptation Regularization Type. Valid values are (none | kl)");
        }

    static GradientsUpdateType ParseGradUpdateType(wstring s)
    {
        if (!_wcsicmp(s.c_str(), L"") || !_wcsicmp(s.c_str(), L"none") || !_wcsicmp(s.c_str(), L"normal") || !_wcsicmp(s.c_str(), L"simple"))
            return GradientsUpdateType::None;
        else if (!_wcsicmp(s.c_str(), L"adagrad"))
            return GradientsUpdateType::AdaGrad;
        else if (!_wcsicmp(s.c_str(), L"rmsProp"))
            return GradientsUpdateType::RmsProp;
        else if (!_wcsicmp(s.c_str(), L"fsAdagrad"))
            return GradientsUpdateType::FSAdaGrad;
        else
            InvalidArgument("ParseGradUpdateType: Invalid Gradient Updating Type. Valid values are (none | adagrad | rmsProp | fsAdagrad )");
    }

    static ParallelizationMethod ParseParallelizationMethod(wstring s)
    {
        if (!_wcsicmp(s.c_str(), L"") || !_wcsicmp(s.c_str(), L"none"))
            return ParallelizationMethod::None;
        else if (!_wcsicmp(s.c_str(), L"DataParallelSGD"))
            return ParallelizationMethod::DataParallelSGD;
        else if (!_wcsicmp(s.c_str(), L"ModelAveragingSGD"))
            return ParallelizationMethod::ModelAveragingSGD;
        else
            InvalidArgument("ParseParallelizationMethod: Invalid Parallelization Method. Valid values are (none | dataParallelSGD | modelAveragingSGD)");
    }

    static LearningRateSearchAlgorithm ParseLearningRateSearchType(wstring s)
    {
        // TODO: why allow so many variants?
        if (!_wcsicmp(s.c_str(), L"false") || !_wcsicmp(s.c_str(), L"none"))
            return LearningRateSearchAlgorithm::None;
        else if (!_wcsicmp(s.c_str(), L"searchBeforeEpoch") || !_wcsicmp(s.c_str(), L"beforeEpoch"/*legacy, deprecated*/) || !_wcsicmp(s.c_str(), L"before"/*legacy, deprecated*/))
            return LearningRateSearchAlgorithm::SearchBeforeEpoch;
        else if (!_wcsicmp(s.c_str(), L"adjustAfterEpoch") || !_wcsicmp(s.c_str(), L"afterEpoch"/*legacy, deprecated*/) || !_wcsicmp(s.c_str(), L"after"/*legacy, deprecated*/))
            return LearningRateSearchAlgorithm::AdjustAfterEpoch;
        else
            InvalidArgument("autoAdjustLR: Invalid learning rate search type. Valid values are (none | searchBeforeEpoch | adjustAfterEpoch)");
    }

    template<class ConfigRecordType>
    SGDParams::SGDParams(const ConfigRecordType& configSGD, size_t sizeofElemType)
    {
        floatargvector learningRatesPerMB = configSGD(L"learningRatesPerMB", ConfigRecordType::Array(floatargvector()));

        floatargvector learningRatesPerSample = configSGD(L"learningRatesPerSample", ConfigRecordType::Array(floatargvector()));

        string executionEngineValue = configSGD(L"executionEngine", "synchronous");

        // AutoAdjust Parameters
        const ConfigRecordType & configAALR(configSGD(L"AutoAdjust", ConfigRecordType::Record()));
        m_autoLearnRateSearchType = ParseLearningRateSearchType(configAALR(L"autoAdjustLR", L"None"));
        m_reduceLearnRateIfImproveLessThan =   configAALR(L"reduceLearnRateIfImproveLessThan",   0.0);
        m_continueReduce =                     configAALR(L"continueReduce",                     false);
        m_learnRateAdjustInterval =            configAALR(L"learnRateAdjustInterval",            (size_t)1);
        m_learnRateAdjustInterval = max((size_t)1, m_learnRateAdjustInterval);  //minimum interval is 1 epoch
        m_learnRateDecreaseFactor = configAALR(L"learnRateDecreaseFactor", 0.618);
        m_increaseLearnRateIfImproveMoreThan = configAALR(L"increaseLearnRateIfImproveMoreThan", numeric_limits<double>::infinity());
        m_learnRateIncreaseFactor =            configAALR(L"learnRateIncreaseFactor",            1.382);

        // AutoAdjust Auto Adjust Minibatch Parameters
        m_autoAdjustMinibatch =                 configAALR(L"autoAdjustMinibatch",                 false);
        m_minibatchSizeTuningFrequency =        configAALR(L"minibatchSizeTuningFrequency",        (size_t)1);
        m_minibatchSizeTuningMax =              configAALR(L"minibatchSizeTuningMax",              (size_t)1048576);
        m_minibatchSearchCriterionErrorMargin = configAALR(L"minibatchSearchCriterionErrorMargin", (size_t)1);

        // the number of minibatches used to search
        // the learning rate. Its typically set to 10-20% of
        // the total minibatches in an epoch.
        m_numMiniBatch4LRSearch = configAALR(L"numMiniBatch4LRSearch", ConfigRecordType::Array(intargvector(vector<int>{ 500 })));

        m_numPrevLearnRates =           configAALR(L"numPrevLearnRates",           (size_t)5);
        m_numBestSearchEpoch =          configAALR(L"numBestSearchEpoch",          (size_t)1);
        m_loadBestModel =               configAALR(L"loadBestModel",               true);
        m_useCVSetControlLRIfCVExists = configAALR(L"UseCVSetControlLRIfCVExists", true);
        m_useEvalCriterionControlLR =   configAALR(L"UseEvalCriterionControlLR",   false);

        // TODO: mbSize and truncated should be specified differently for truncated BPTT:
        //       mbSize = total number of samples after which a model update should happen
        //       truncated = truncation length
        m_mbSize = configSGD(L"minibatchSize", ConfigRecordType::Array(intargvector(vector<int>{ 256 })));
        m_truncated = configSGD(L"truncated", false);
        m_maxSamplesInRAM = configSGD(L"maxSamplesInRAM", ConfigRecordType::Array(intargvector(vector < int > {0})));

        // the number of samples in each epoch (0 means, use all the samples in each epoch).
        m_epochSize = configSGD(L"epochSize", (size_t)0);
        // the number of samples in each epoch (0 means, use all the samples in each epoch).
        if (m_epochSize == 0)
            m_epochSize = requestDataSize;
        m_maxComputedEpochSize = m_epochSize;

        // the total number of epochs to run.
        m_maxEpochs = configSGD(L"maxEpochs");

        floatargvector momentumPerMB = configSGD(L"momentumPerMB", ConfigRecordType::Array(floatargvector()));

        floatargvector momentumPerSample = configSGD(L"momentumPerSample", ConfigRecordType::Array(floatargvector()));

        m_maxTempMemSizeInSamplesForCNN = configSGD(L"maxTempMemSizeInSamplesForCNN", (size_t)0);

        m_traceLevel =          configSGD(L"traceLevel",          (int)0);
        m_numMBsToShowResult =  configSGD(L"numMBsToShowResult",  (size_t)10);
        m_numMBsToCUDAProfile = configSGD(L"numMBsToCUDAProfile", (size_t)0);

        m_gradientClippingWithTruncation = configSGD(L"gradientClippingWithTruncation", true);
        m_clippingThresholdPerSample =     configSGD(L"clippingThresholdPerSample",     numeric_limits<double>::infinity());

        // sequence-training parameters
        m_hSmoothingWeight = configSGD(L"hSmoothingWeight", 0.95);
        m_frameDropThresh =  configSGD(L"frameDropThresh",  1e-10);
        m_doReferenceAlign = configSGD(L"doReferenceAlign", false);

        m_dropoutRates = configSGD(L"dropoutRate", ConfigRecordType::Array(floatargvector(vector<float>{ 0.0f })));

        GradientsUpdateType gradUpdateType = ParseGradUpdateType(configSGD(L"gradUpdateType", L"None"));
        double gaussianNoiseInjecStd = configSGD(L"gaussianNoiseInjectStd", 0.0);
        m_gradType.mType = gradUpdateType;
        m_gradType.mGaussianNoiseInjectStd = (float) gaussianNoiseInjecStd;

        // extract RMSProp parameters from config, if they exist. Default to reasonable values.
        m_rpi.dec =   configSGD(L"rms_wgt_dec", 0.75);
        m_rpi.inc =   configSGD(L"rms_wgt_inc", 1.2);
        m_rpi.min =   configSGD(L"rms_wgt_min", 0.1);
        m_rpi.max =   configSGD(L"rms_wgt_max", 10.0);
        m_rpi.gamma = configSGD(L"rms_gamma",   0.99);

        m_needAveMultiplier = configSGD(L"normWithAveMultiplier", true);
        m_L2RegWeight =       configSGD(L"L2RegWeight", 0.0);
        m_L1RegWeight =       configSGD(L"L1RegWeight", 0.0);

        /// for backward support. future setup should use gradUpdateType=AdaGrad, instead of
        /// useAdagrad=true
        bool useAdagrad = configSGD(L"useAdagrad", false);
        if (useAdagrad)
        {
            gradUpdateType = GradientsUpdateType::AdaGrad;
            m_gradType.mType = gradUpdateType;
        }

        m_adaptationRegType = ParseAdaptationRegType(configSGD(L"adaptationRegType", L"None"));
        m_adaptationRegWeight = configSGD(L"adaptationRegWeight", 0.0);

        /// gradient check setup
        m_doGradientCheck =       configSGD(L"gradientcheck", false);
        m_gradientCheckSigDigit = configSGD(L"sigFigs",       6.0); // TODO: why is this a double?

        if (m_doGradientCheck && sizeofElemType != sizeof(double))
            LogicError("Gradient check needs to use precision = 'double'.");

        m_useAllDataForPreComputedNode = configSGD(L"UseAllDataForPreComputedNode", true);

        // consistency checks
        for (size_t i = 0; i < m_mbSize.size(); i++)
        {
            if (m_epochSize != requestDataSize && m_epochSize < m_mbSize[i])
            {
                InvalidArgument("epoch size must be larger than mbsize.");
            }
        }

        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None &&
            (learningRatesPerSample.size() == 0 && learningRatesPerMB.size() == 0))
        {
            InvalidArgument("If autoLearnRateSearchType is false you must specify the learningRatesPerSample or learningRatesPerMB parameter.");
        }

        if (learningRatesPerSample.size() > 0 && learningRatesPerMB.size() > 0)
        {
            InvalidArgument("You specified both learningRatesPerSample and learningRatesPerMB. Please comment out one of them.");
        }
        else if (learningRatesPerSample.size() > 0)
        {
            m_learningRatesParam = learningRatesPerSample;
            m_learningRatesSpecifiedForMBSize = intargvector(L"1");
        }
        else if (learningRatesPerMB.size() > 0)     // this actually means per specified minibatch size
        {
            m_learningRatesParam = learningRatesPerMB;
            m_learningRatesSpecifiedForMBSize = m_mbSize;
        }

        if (momentumPerSample.size() > 0 && momentumPerMB.size() > 0)
        {
            InvalidArgument("You specified both momentumPerSample and momentumPerMB. Please comment out one of them.");
        }
        else if (momentumPerSample.size() > 0)         // TODO: noone should use this; change to MomentumTimeConstant
        {
            m_momentumParam = momentumPerSample;
            m_momentumSpecifiedForMBSize = intargvector(L"1");
        }
        else if (momentumPerMB.size() > 0)
        {
            m_momentumParam = momentumPerMB;
            m_momentumSpecifiedForMBSize = m_mbSize;
        }
        else    // default: momentumPerMB = 0.9 per MB
        {
            m_momentumParam = floatargvector(L"0.9");
            m_momentumSpecifiedForMBSize = m_mbSize;
        }
        for (int i = 0; i < m_momentumParam.size(); i++)
        {
            if (m_momentumParam[i] >= 1.0 || m_momentumParam[i] < 0.0)
                InvalidArgument("Momentum parameter must be in [0, 1).");
        }

        if (m_learnRateDecreaseFactor > 1 || m_learnRateIncreaseFactor < 1)
            InvalidArgument("learnRateIncreaseFactor must be >= 1 and learnRateDecreaseFactor must be <= 1.");

        for (size_t i = 0; i < m_dropoutRates.size(); i++)
            if (m_dropoutRates[i] >= 1 || m_dropoutRates[i] < 0)
                InvalidArgument("dropoutRate must be >= 0 and < 1.");

        if (m_adaptationRegWeight > 1 || m_adaptationRegWeight < 0)
            InvalidArgument("adaptationRegWeight must be in [0 1]");

        m_minLearnRate = 1e-9f;

        m_needAdaptRegularization = false;

        // BUGBUG: these are not passed to Init()
        m_doUnitTest = configSGD(L"unitTest", false);

        // parallel training
        m_parallelizationMethod = ParallelizationMethod::None;
        m_numGradientBits = 32;
        m_zeroThresholdFor1Bit = true;
        m_enableDistributedMBReading = false;
        m_parallelizationStartEpochNum = 0;
        m_nFramesBetweenMASync = 40000; // default 40k frames 

        if ((g_mpi != nullptr) && configSGD.Exists(L"ParallelTrain"))
        {
            const ConfigRecordType & configParallelTrain(configSGD(L"ParallelTrain", ConfigRecordType::Record()));
            m_parallelizationMethod = ParseParallelizationMethod(configParallelTrain(L"parallelizationMethod", L"none"));
            m_parallelizationStartEpochNum = configParallelTrain(L"parallelizationStartEpoch", (int)1) - 1;  // Epoch numbers internally are 0 based
            m_enableDistributedMBReading = configParallelTrain(L"distributedMBReading", false);

            if (configParallelTrain.Exists(L"DataParallelSGD"))
            {
                const ConfigRecordType & configDataParallelSGD(configParallelTrain(L"DataParallelSGD", ConfigRecordType::Record()));
                size_t defaultGradientBits = 8 * sizeofElemType;
                m_numGradientBits = configDataParallelSGD(L"gradientBits", defaultGradientBits);
                m_zeroThresholdFor1Bit = configDataParallelSGD(L"useZeroThresholdFor1BitQuantization", true);
                if ((m_numGradientBits < 1) || (m_numGradientBits > (8 * sizeofElemType)))
                {
                    InvalidArgument("gradientBits must be in the range [1, 32] when using precision=float and in range [1, 64] when using precision=double!");
                }
            }

            if (configParallelTrain.Exists(L"ModelAveragingSGD") )
            {
                const ConfigRecordType & configMASGD(configParallelTrain(L"ModelAveragingSGD", ConfigRecordType::Record()));
                m_nFramesBetweenMASync = configMASGD(L"syncFrequencyInFrames", (size_t)40000);
                m_iMASyncStatsTrace = configMASGD(L"maPerfStats", (int)0);
            }
        }
    }

    static size_t GetSizeOfPrecision(const ScriptableObjects::IConfigRecordPtr configp)
    {
        wstring precision = configp->Get(L"precision");
        if (precision == L"float")
            return sizeof(float);
        else if (precision == L"double")
            return sizeof(double);
        else
            RuntimeError("invalid value '%ls' for 'precision', must be 'float' or 'double'", precision.c_str());
    }

    SGDParams::SGDParams(const ScriptableObjects::IConfigRecordPtr configp) :
        SGDParams(*configp, GetSizeOfPrecision(configp))
    { }

    // -----------------------------------------------------------------------
    // class SGD
    // -----------------------------------------------------------------------

    template SGD<float >::SGD(const ConfigParameters &);
    template SGD<double>::SGD(const ConfigParameters &);
    template SGD<float >::SGD(const ScriptableObjects::IConfigRecord &);
    template SGD<double>::SGD(const ScriptableObjects::IConfigRecord &);

    template<class ElemType>
    void SGD<ElemType>::Adapt(wstring origModelFileName, wstring refNodeName,
               IDataReader<ElemType>* trainSetDataReader,
               IDataReader<ElemType>* validationSetDataReader,
               const DEVICEID_TYPE deviceId, const bool makeMode)
    {
        if (origModelFileName == L"" || trainSetDataReader == nullptr)
            InvalidArgument("origModel and trainSetDataReader should not be null.");

        int startEpoch = DetermineStartEpoch(makeMode);
        if (startEpoch == m_maxEpochs)
        {
            fprintf(stderr, "No further training is necessary.\n");
            return;
        }

        ComputationNetworkPtr net;
        if (startEpoch >= 0)
        {
            wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
            fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());
            net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName);
        }
        else
        {
            fprintf(stderr, "Load Network From the original model file %ls.\n", origModelFileName.c_str());
            net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, origModelFileName);
        }

        startEpoch = max(startEpoch, 0);

        ComputationNetworkPtr refNet;
        m_needAdaptRegularization = m_adaptationRegType != AdaptationRegType::None && m_adaptationRegWeight > 0;
        if (m_needAdaptRegularization)
        {
            fprintf(stderr, "Load reference Network From the original model file %ls.\n", origModelFileName.c_str());
            refNet = ComputationNetwork::CreateFromFile<ElemType>(deviceId, origModelFileName);
        }

        ComputationNodeBasePtr refNode;
        if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL)
        {
            fprintf(stderr, "Checking refNodeName %ls.\n", origModelFileName.c_str());
            if (refNodeName == L"")
                InvalidArgument("refNodeName does not exist and is needed when adaptationRegType is KL.");
            refNode = refNet->GetNodeFromName(refNodeName);
        }

        TrainOrAdaptModel(startEpoch, net, refNet, refNode, trainSetDataReader, validationSetDataReader);
    }

#if 0
    // TODO: per discussion with Dong Yu, Guoguo Chen, and Yu Zhang, this function can be removed.
    template<class ElemType>
    void SGD<ElemType>::SequenceTrain(IComputationNetBuilder<ElemType>* netBuilder, wstring origModelFileName,
                       IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader,
                       const DEVICEID_TYPE deviceId, const bool makeMode)
    {
        if (netBuilder == nullptr || origModelFileName == L"" || trainSetDataReader == nullptr)
        {
            InvalidArgument("netBuilder, origModel and trainSetDataReader should not be null.");
        }

        int startEpoch = DetermineStartEpoch(makeMode);
        if (startEpoch == m_maxEpochs)
        {
            fprintf(stderr, "No further training is necessary.\n");
            return;
        }

        // Initializes the model from original model.
        // TODO: Comment what this does!
        auto origNet = make_shared<ComputationNetwork>(deviceId);
        ComputationNetworkPtr sequenceNet = 
            (startEpoch < 0) ? netBuilder->BuildNetworkFromDescription() : origNet;
        std::vector<ComputationNodeBasePtr> addedFeatureNodes;
        std::vector<ComputationNodeBasePtr> replacedCriterionNodes;
        if (startEpoch < 0)
        {
            // Loads models.
            origNet->LoadFromFile<ElemType>(origModelFileName);

            // Processes feature nodes.
            std::vector<ComputationNodeBasePtr> & sequenceFeatureNodes = sequenceNet->FeatureNodes();
            for (size_t i = 0; i < sequenceFeatureNodes.size(); ++i)
            {
                if (!origNet->NodeNameExist(sequenceFeatureNodes[i]->NodeName()))
                {
                    addedFeatureNodes.push_back(sequenceFeatureNodes[i]);
                    origNet->AddFeatureNode(sequenceFeatureNodes[i]);
                }
            }

            // Processes criterion nodes.
            auto & origCriterionNodes = GetTrainCriterionNodes(origNet);
            auto & sequenceCriterionNodes = GetTrainCriterionNodes(sequenceNet);
            if (origCriterionNodes.size() == 0 || sequenceCriterionNodes.size() == 0)
            {
                RuntimeError("Training criterion node does not exist.");
            }
            replacedCriterionNodes.push_back(origCriterionNodes[0]);
            origNet->ReplaceFinalCriterionNode(origCriterionNodes[0]->NodeName(), sequenceCriterionNodes[0]);
            origNet->ResetEvalTimeStamp();
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
        ComputationNetworkPtr net = (startEpoch < 0) ? origNet : ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName);

        startEpoch = max(startEpoch, 0);

        TrainOrAdaptModel(startEpoch, net, net, nullptr, trainSetDataReader, validationSetDataReader);

        // Handles deletions carefully here.
        // TODO: This is no longer needed since we own our networks and deal with shared_ptrs now.
        if (startEpoch < 0)
        {
            for (size_t i = 0; i < addedFeatureNodes.size(); ++i)
            {
                origNet->RemoveFeatureNode(addedFeatureNodes[i]);
            }
            auto & origCriterionNodes = GetTrainCriterionNodes(origNet);
            origNet->ReplaceFinalCriterionNode(origCriterionNodes[0]->NodeName(), replacedCriterionNodes[0]);
        }
    }
#endif

    static double MomentumPerMB(double momentumPerSample, size_t minibatchSize)
    {
        return pow(momentumPerSample, minibatchSize);
    }

    template<class ElemType>
    void SGD<ElemType>::Train(function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn, DEVICEID_TYPE deviceId,
               IDataReader<ElemType>* trainSetDataReader,
               IDataReader<ElemType>* validationSetDataReader,
               const bool makeMode)
    {
        if (trainSetDataReader == nullptr)
        {
            InvalidArgument("netBuilder and trainSetDataReader should not be null.\n");
        }
        int startEpoch = DetermineStartEpoch(makeMode);
        if (startEpoch == m_maxEpochs)
        {
            fprintf(stderr, "No further training is necessary.\n");
            return;
        }

        wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
        if (startEpoch >= 0)
            fprintf(stderr, "Starting from checkpoint. Load Network From File %ls.\n", modelFileName.c_str());

        // create or load from checkpoint
        shared_ptr<ComputationNetwork> net = startEpoch < 0 ? createNetworkFn(deviceId) : ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName);

        // log the device we are computing on
        if (net->GetDeviceId() < 0)
            fprintf(stderr, "SGD using CPU.\n");
        else
            fprintf(stderr, "SGD using GPU %d.\n", (int)net->GetDeviceId());

        // TODO: BUGBUG: if not starting from checkpoint, need to synchronize initial model
        // strategy should be to run the initializer above on mpiRank==0, and then broadcast parameters.

        /*  if (m_doUnitTest)
        {
            if (net.UnitTest() == false)
                LogicError("unit test on decoder network not passed");

            return;
        }*/

        startEpoch = max(startEpoch, 0);
        m_needAdaptRegularization = false;

        TrainOrAdaptModel(startEpoch, net, net, nullptr, trainSetDataReader, validationSetDataReader);
    }

// protected:

    // Get{Train,Eval}CriterionNodes() return a reference that is, unfortunately, dependent on the network.
    // So we hold those inside here. Not very nice. Also not thread-safe. This may go away once we fix sequence-to-sequence models properly.
    static map<ComputationNetworkPtr, vector<ComputationNodeBasePtr>> tmpCriterionNodeSets;
    // TODO: test this, then remove this comment

    template<class ElemType>
    std::vector<ComputationNodeBasePtr> & SGD<ElemType>::GetTrainCriterionNodes(ComputationNetworkPtr net)
    {
        fprintf(stderr, "GetTrainCriterionNodes %ls ...\n", m_trainCriterionNodeName.c_str());
        if (!m_trainCriterionNodeName.empty())
        {
            tmpCriterionNodeSets[net] = net->CriterionNodesFrom(m_trainCriterionNodeName);
            return tmpCriterionNodeSets[net];
        }
        else
            return net->FinalCriterionNodes();
    }

    template<class ElemType>
    std::vector<ComputationNodeBasePtr> & SGD<ElemType>::GetEvalCriterionNodes(ComputationNetworkPtr net)
    {
        fprintf(stderr, "GetEvalCriterionNodes %ls ...\n", m_evalCriterionNodeName.c_str());
        if (!m_evalCriterionNodeName.empty())
        {
            tmpCriterionNodeSets[net] = net->CriterionNodesFrom(m_evalCriterionNodeName);
            return tmpCriterionNodeSets[net];
        }
        else
            return net->EvaluationNodes();
    }

    template<class ElemType>
    void SGD<ElemType>::TrainOrAdaptModel(int startEpoch, ComputationNetworkPtr net,
                           ComputationNetworkPtr refNet,
                           ComputationNodeBasePtr refNode,
                           IDataReader<ElemType>* trainSetDataReader,
                           IDataReader<ElemType>* validationSetDataReader)
    {
        auto & featureNodes = net->FeatureNodes();
        auto & labelNodes = net->LabelNodes();
        auto & criterionNodes = GetTrainCriterionNodes(net);
        auto & evaluationNodes = GetEvalCriterionNodes(net);

        //allocate memory for backward computation
        //we intentionally separate it from above loop to make sure forward computing gets the right matrices
        fprintf(stderr, "\n\nAllocating matrices for gradient computing\n");
        for (int i = 0; i < criterionNodes.size(); i++)
            net->AllocateGradientMatrices(criterionNodes[i]);
        // give the layout something to validate with (some code below validates the network before actually receiving data)
        // Note: yak!

        // get feature and label nodes into an array of matrices that will be passed to GetMinibatch()
        // TODO: instead, remember the nodes directly, to be able to handle both float and double nodes; current version will crash for mixed networks
        std::map<std::wstring, Matrix<ElemType>*>* inputMatrices = new std::map<std::wstring, Matrix<ElemType>*>();
        for (size_t pass = 0; pass < 2; pass++)
        {
            auto & nodes = (pass == 0) ? featureNodes : labelNodes;
            for (size_t i = 0; i < nodes.size(); i++)
            {
                auto & node = nodes[i];
                auto * functionValues = &dynamic_pointer_cast<ComputationNode<ElemType>>(node)->FunctionValues();
                assert(functionValues->GetNumCols() == net->GetMBLayoutPtr()->GetNumTimeSteps());
                (*inputMatrices)[node->NodeName()] = functionValues;
            }
        }

        //get hmm file for sequence training
        bool isSequenceTrainingCriterion = (criterionNodes[0]->OperationName() == L"SequenceWithSoftmax");
        if (isSequenceTrainingCriterion)
        {
            //SequenceWithSoftmaxNode<ElemType>* node = static_cast<SequenceWithSoftmaxNode<ElemType>*>(criterionNodes[0]);
            auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNodes[0]);
            auto  hmm = node->gethmm();
            trainSetDataReader->GetHmmData(hmm);
        }

        // used for KLD regularized adaptation. For all other adaptation techniques
        // use MEL to edit the model and using normal training algorithm
        std::vector<ComputationNodeBasePtr> refFeatureNodes;
        if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
        {
            refFeatureNodes.resize(featureNodes.size());
            for (size_t i = 0; i < featureNodes.size(); i++)
            {
                //we need to keep this info to handle deletion
                refFeatureNodes[i] = refNet->GetNodeFromName(featureNodes[i]->NodeName());
                refNet->ChangeNode(featureNodes[i]->NodeName(), featureNodes[i]);
            }

            refNet->RebuildNetwork(refNode);
        }

        //initializing weights and gradient holder
        //only one criterion so far TODO: support multiple ones?
        // BUGBUG: fails here in validation--MBLayout not set yet
        auto & learnableNodes = net->LearnableNodes(criterionNodes[0]);
        std::list<Matrix<ElemType>> smoothedGradients;

        for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
        {
            ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
            smoothedGradients.push_back(Matrix<ElemType>(node->GetNumRows(),
                                                         node->GetNumCols(),
                                                         net->GetDeviceId()));
        }

        double epochCriterion, avgCriterion, prevCriterion, lrControlCriterion;
        lrControlCriterion = epochCriterion = avgCriterion = prevCriterion = std::numeric_limits<double>::infinity();
        size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

        std::vector<double> epochEvalErrors(evaluationNodes.size(), std::numeric_limits<double>::infinity());

        std::vector<wstring> evalNodeNames;
        for (size_t i = 0; i < evaluationNodes.size(); i++)
            evalNodeNames.push_back(evaluationNodes[i]->NodeName());

        size_t totalSamplesSeen = 0;
        double learnRatePerSample = 0.5f / m_mbSize[startEpoch];

        double learningRateAdjustmentFactor = 1.0f;
        vector<double> prevLearnRates;
        prevLearnRates.resize(m_numPrevLearnRates);
        for (int i = 0; i < m_numPrevLearnRates; i++)
        {
             prevLearnRates[i] = -1.0;
        }

        if (m_parallelizationMethod == ParallelizationMethod::DataParallelSGD)
        {
            InitDistGradAgg(evaluationNodes.size(), m_traceLevel);
        }
        //precompute mean and invStdDev nodes and save initial model
        if (PreCompute(net, trainSetDataReader, featureNodes, labelNodes, inputMatrices) || startEpoch == 0)
        {
            // Synchronize all ranks before writing the model to ensure that 
            // everyone is done loading the model
            if (g_mpi != nullptr)
                g_mpi->WaitAll();

            net->SaveToFile(GetModelNameForEpoch(int(startEpoch) - 1));
        }

        // BUGBUG: This is where the trainSetDataReader->GetNumParallelSequences() is used to further normalize
#if 0
        // In these cases, we need to post-patch the learning-rate parameters.
        if (m_needToNormalizeLRByParallUtterance)
        {
            for (auto& x : m_learningRatesPerSample)
                x /= (float)trainSetDataReader->GetNumParallelSequences();
        }
        if (m_needToNormalizeMomentumByParallUtterance)
        {
            for (auto& x : m_momentumPerSample)
                x = (float)pow(x, 1.0 / trainSetDataReader->GetNumParallelSequences());
        }
#endif

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
                prevLearnRates[startEpoch % m_numPrevLearnRates] = learnRatePerSample;
        }

        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch &&
            !learnRateInitialized && m_learningRatesParam.size() <= startEpoch)
        {
            InvalidArgument(
                "When using \"AdjustAfterEpoch\", there must either exist a checkpoint file, "
                "or an explicit learning rate must be specified in config for the starting epoch.");
        }

        unsigned long dropOutSeed = 1;
        double prevDropoutRate = 0;

        bool learnRateReduced = false;

        // pass user config on memory allocation for convolution operations to the Network
        ComputationNetwork::SetMaxTempMemSizeForCNN(net, criterionNodes[0], m_maxTempMemSizeInSamplesForCNN);
        if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode)
            ComputationNetwork::SetMaxTempMemSizeForCNN(refNet, refNode, m_maxTempMemSizeInSamplesForCNN);

        // likewise for sequence training parameters
        if (isSequenceTrainingCriterion)
            ComputationNetwork::SetSeqParam<ElemType>(net, criterionNodes[0], m_hSmoothingWeight, m_frameDropThresh, m_doReferenceAlign);

        // --- MAIN EPOCH LOOP
        for (int i = startEpoch; i < (int)m_maxEpochs; i++) // TODO: why is this an int, and not a size_t?
        {
            // Synchronize all ranks before proceeding to ensure that 
            // rank 0 has finished writing the previous model file
            if (g_mpi != nullptr)
            {
                g_mpi->WaitAll();
            }

            Timer timer;
            timer.Start();

            // set dropout rate for this epoch
            ComputationNetwork::SetDropoutRate<ElemType>(net, criterionNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);

            // learning rate adjustment
            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || i < m_learningRatesParam.size())
            {
                // BUGBUG: GetNumParallelSequences() returns 1 under certain situations; it seems when restarting from checkpoint
                learnRatePerSample = GetLearningRatePerSample(i/*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());
            }
            else if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
            {
                double largestPrevLearnRatePerSample = prevLearnRates[0];
                for (int j = 1; j < m_numPrevLearnRates; j++)
                    largestPrevLearnRatePerSample = max(largestPrevLearnRatePerSample, prevLearnRates[j]);

                // return a reasonable learning rate based on the initial minibatchSize
                double newLearningRatePerSample = SearchForBestLearnRate(net, refNet, refNode, i, learnRatePerSample,
                                                                           trainSetDataReader, featureNodes, labelNodes,
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
                fprintf(stderr, "Learn Rate Per Sample for Epoch[%d] = %.8g is less than minLearnRate %.8g. Training complete.\n",
                        i + 1, learnRatePerSample, m_minLearnRate);
                if (m_autoLearnRateSearchType != LearningRateSearchAlgorithm::None)
                    net->SaveToFile(m_modelPath);
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
                                                              m_mbSize[i], featureNodes, labelNodes,
                                                              criterionNodes, evaluationNodes,
                                                              inputMatrices, learnableNodes,
                                                              smoothedGradients, learningRateAdjustmentFactor);
                m_prevChosenMinibatchSize = chosenMinibatchSize;
            }
            else
            {
                // use the explicitly set minibatch size
                chosenMinibatchSize = m_mbSize[i];
            }

#if 1
            actualMinibatchSize = FixUpEffectiveMBSize(chosenMinibatchSize/*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());
#else
            actualMinibatchSize = chosenMinibatchSize;
            if (m_needToNormalizeMomentumByParallUtterance)
            {
                actualMinibatchSize = chosenMinibatchSize * trainSetDataReader->GetNumParallelSequences();
            }
#endif

            // TODO: show momentum also as a time constant
            fprintf(stderr, "Starting Epoch %d: learning rate per sample = %f  effective momentum = %f \n",
                    i + 1, learnRatePerSample, MomentumPerMB(GetMomentumPerSample(i/*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences()), actualMinibatchSize));

            TrainOneEpoch(net,
                          refNet, 
                          refNode, 
                          i, 
                          m_epochSize,
                          trainSetDataReader, 
                          learnRatePerSample, 
                          chosenMinibatchSize, 
                          featureNodes,
                          labelNodes, 
                          criterionNodes, 
                          evaluationNodes,
                          inputMatrices, 
                          learnableNodes, smoothedGradients,
                          epochCriterion, epochEvalErrors, totalSamplesSeen);

            timer.Stop();
            double epochTime = timer.ElapsedSeconds();

            if (m_useEvalCriterionControlLR)
            {
                lrControlCriterion = epochEvalErrors[0];
            }
            else
            {
                lrControlCriterion = epochCriterion;
            }

            fprintf(stderr,
                    "Finished Epoch[%2d of %d]: [Training Set] TrainLossPerSample = %.8g; ",
                    i + 1, (int) m_maxEpochs, epochCriterion);

            if (epochEvalErrors.size() == 1)
            {
                fprintf(stderr,
                        "EvalErrPerSample = %.8g; Ave LearnRatePerSample = %.10g; EpochTime=%.8g\n",
                        epochEvalErrors[0], learnRatePerSample, epochTime);
                m_lastFinishedEpochEvalErr = epochEvalErrors[0];
            }
            else
            {
                fprintf(stderr, "EvalErrPerSample ");
                for (size_t j = 0; j < epochEvalErrors.size(); j++)
                {
                    fprintf(stderr, "[%lu]=%.8g; ", j, epochEvalErrors[j]);
                    m_lastFinishedEpochEvalErr = epochEvalErrors[j];

                }

                fprintf(stderr, "Ave LearnRatePerSample = %.10g; Epoch Time=%.8g\n",
                        learnRatePerSample, epochTime);

                fprintf(stderr, "Finished Epoch[%2d of %d]: Criterion Node [%ls] Per Sample = %.8g\n",
                                i + 1, (int) m_maxEpochs, criterionNodes[0]->NodeName().c_str(), epochCriterion);

                for (size_t j = 0; j < epochEvalErrors.size(); j++)
                {
                    fprintf(stderr, "Finished Epoch[%2d of %d]: Evaluation Node [%ls] Per Sample = %.8g\n",
                            i + 1, (int) m_maxEpochs, evalNodeNames[j].c_str(), epochEvalErrors[j]);
                }
            }

            if ((g_mpi == nullptr) || g_mpi->IsMainNode())
            {
                if (validationSetDataReader != trainSetDataReader && validationSetDataReader != nullptr)
                {
                    SimpleEvaluator<ElemType> evalforvalidation(net);
                    vector<wstring> cvSetTrainAndEvalNodes;
                    cvSetTrainAndEvalNodes.push_back(criterionNodes[0]->NodeName());
                    cvSetTrainAndEvalNodes.push_back(evaluationNodes[0]->NodeName());

                    vector<double> vScore = evalforvalidation.Evaluate(validationSetDataReader, cvSetTrainAndEvalNodes, m_mbSize[i]);
                    fprintf(stderr, "Finished Epoch[%2d of %d]: [Validation Set] TrainLossPerSample = %.8g; EvalErrPerSample = %.8g\n",
                            i + 1, (int) m_maxEpochs, vScore[0], vScore[1]);

                    if (m_useCVSetControlLRIfCVExists)
                    {
                        if (m_useEvalCriterionControlLR)
                            lrControlCriterion = vScore[1];
                        else
                            lrControlCriterion = vScore[0]; //the first one is the training criterion.
                        }
                    }
                }

            // broadcast epochCriterion to make sure each processor will have the same learning rate schedule
            if ((m_parallelizationMethod == ParallelizationMethod::ModelAveragingSGD) && (g_mpi->NumNodesInUse() > 1))
                g_mpi->Bcast(&epochCriterion, 1, g_mpi->MainNodeRank());

            bool loadedPrevModel = false;
            size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
            if (avgCriterion == std::numeric_limits<double>::infinity())
            {
                avgCriterion = lrControlCriterion;
            }
            else
            {
                avgCriterion = ((epochsSinceLastLearnRateAdjust - 1 - epochsNotCountedInAvgCriterion) *
                    avgCriterion + lrControlCriterion) /
                    (epochsSinceLastLearnRateAdjust - epochsNotCountedInAvgCriterion);
            }

            if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch &&
                m_learningRatesParam.size() <= i && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
            {
                if (std::isnan(avgCriterion) || (prevCriterion - avgCriterion < 0 && prevCriterion != std::numeric_limits<double>::infinity()))
                {
                    if (m_loadBestModel)
                    {
                        auto bestModelPath = GetModelNameForEpoch(i - m_learnRateAdjustInterval);
                        fprintf(stderr, "Loaded previous model with best training criterion value: %ls.\n", bestModelPath.c_str());
                        net->LoadPersistableParametersFromFile(bestModelPath, m_validateAfterModelReloading);
                        net->ResetEvalTimeStamp();
                        LoadCheckPointInfo(i - m_learnRateAdjustInterval,
                                           /*out*/ totalSamplesSeen,
                                           /*out*/ learnRatePerSample,
                                           smoothedGradients,
                                           /*out*/ prevCriterion,
                                           /*out*/ m_prevChosenMinibatchSize);
                        loadedPrevModel = true;
                    }
                }

                if (m_continueReduce)
                {
                    if (std::isnan(avgCriterion) || 
                        (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion &&
                        prevCriterion != std::numeric_limits<double>::infinity()))
                    {
                        if (learnRateReduced == false)
                            learnRateReduced = true;
                        else
                        {
                            net->SaveToFile(GetModelNameForEpoch(i, true));

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
                        prevCriterion != std::numeric_limits<double>::infinity()))
                    {

                        learnRatePerSample *= m_learnRateDecreaseFactor;
                        fprintf(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                    }
                    else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan * prevCriterion &&
                             prevCriterion != std::numeric_limits<double>::infinity())
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

            // not loading previous values then set them
            if (!loadedPrevModel && epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
            {
                prevCriterion = avgCriterion;
                epochsNotCountedInAvgCriterion = 0;
            }

            // Synchronize all ranks before proceeding to ensure that 
            // nobody tries reading the checkpoint file at the same time
            // as rank 0 deleting it below
            if (g_mpi != nullptr)
            {
                g_mpi->WaitAll();
            }

            // persist model and check-point info
            if ((g_mpi == nullptr) || g_mpi->IsMainNode())
            {
                net->SaveToFile(GetModelNameForEpoch(i));
                SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, chosenMinibatchSize);
                if (!m_keepCheckPointFiles)
                {
                    // delete previous checkpoint file to save space
                    if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::AdjustAfterEpoch && m_loadBestModel)
                    {
                        if (epochsSinceLastLearnRateAdjust != 1)
                        {
                            _wunlink(GetCheckPointFileNameForEpoch(i - 1).c_str());
                        }
                        if (epochsSinceLastLearnRateAdjust == m_learnRateAdjustInterval)
                        {
                            _wunlink(GetCheckPointFileNameForEpoch(i - m_learnRateAdjustInterval).c_str());
                        }
                    }
                    else
                    {
                        _wunlink(GetCheckPointFileNameForEpoch(i - 1).c_str());
                    }
                }
            }

            if (learnRatePerSample < 1e-12)
            {
                fprintf(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n",
                        learnRatePerSample);
            }
        }
        // --- END OF MAIN EPOCH LOOP

        // Synchronize all ranks before proceeding to ensure that 
        // rank 0 has finished writing the model file
        if (g_mpi != nullptr)
        {
            g_mpi->WaitAll();
        }

        // progress tracing for compute cluster management
        ProgressTracing::TraceProgressPercentage(m_maxEpochs, 0.0);
        ProgressTracing::TraceObjectivePercentage(m_lastFinishedEpochEvalErr);

        // since we linked feature nodes. we need to remove it from the deletion
        if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
        {
            for (size_t i = 0; i < refFeatureNodes.size(); i++)
            {
                // note we need to handle deletion carefully
                refNet->ChangeNode(refFeatureNodes[i]->NodeName(), refFeatureNodes[i]);
            }
        }

        delete inputMatrices;
    }

// protected:

    // return true if precomputation is executed.
    template<class ElemType>
    bool SGD<ElemType>::PreCompute(ComputationNetworkPtr net,
                    IDataReader<ElemType>* trainSetDataReader,
                    std::vector<ComputationNodeBasePtr> & featureNodes,
                    std::vector<ComputationNodeBasePtr> & labelNodes,
                    std::map<std::wstring, Matrix<ElemType>*>* inputMatrices)
    {
        std::list<ComputationNodeBasePtr> nodes = net->GetNodesRequiringPreComputation();    // this tests all HasComputed() flags

        if (nodes.size() == 0)
        {
            fprintf(stderr, "No PreCompute nodes found, skipping PreCompute step\n");
            return false;
        }

        fprintf(stderr, "\nPrecomputing --> %lu PreCompute nodes found.\n\n", nodes.size());
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            auto node = static_pointer_cast<PreComputedNode<ElemType>>(*nodeIter);
            fprintf(stderr, "\tNodeName: %ls\n", (node->NodeName()).c_str());
        }

        //compute
        //trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , requestDataSize);
        // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , m_epochSize); // only based on one epoch
        // [1/12/2015 erw] to support large dataset, we usually partition whole dataset into several epoch's,
        // so we need to use all the data to do precomputing
        if (m_useAllDataForPreComputedNode)     // using all the data
            trainSetDataReader->StartMinibatchLoop(m_mbSize[0], 0);
        else                                    // using only one epoch
            trainSetDataReader->StartMinibatchLoop(m_mbSize[0], 0, m_epochSize);
        net->StartEvaluateMinibatchLoop(nodes);

        // initialize
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            auto node = static_pointer_cast<PreComputedNode<ElemType>>(*nodeIter);
            node->MarkComputed(false/*begin accumulating*/);
        }
        size_t actualMBSizeDummy;
        while (DataReaderHelpers::GetMinibatchIntoNetwork(*trainSetDataReader, net, nullptr, false, false, *inputMatrices, actualMBSizeDummy))
        {
            // TODO: move these into GetMinibatchIntoNetwork()  --but those are passed around; necessary? Can't we get them from 'net'?
            ComputationNetwork::UpdateEvalTimeStamps(featureNodes);
            ComputationNetwork::UpdateEvalTimeStamps(labelNodes);

            net->Evaluate(nodes);
        }
        // finalize
        for (auto nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
        {
            auto node = static_pointer_cast<PreComputedNode<ElemType>>(*nodeIter);
            node->MarkComputed(true/*done accumulating*/);
        }
        fprintf(stderr, "\nPrecomputing --> Completed.\n\n");

        return true;
    }

    // return a reasonable initial learning rate based on the initial mbsize
    template<class ElemType>
    double SGD<ElemType>::SearchForBestLearnRate(ComputationNetworkPtr net,
                                                 ComputationNetworkPtr refNet,
                                                 const ComputationNodeBasePtr& refNode, const int epochNumber,
                                                 const double curLearnRate,
                                                 IDataReader<ElemType>* trainSetDataReader,
                                                 const std::vector<ComputationNodeBasePtr> & featureNodes,
                                                 const std::vector<ComputationNodeBasePtr> & labelNodes,
                                                 const std::vector<ComputationNodeBasePtr> & criterionNodes,
                                                 const std::vector<ComputationNodeBasePtr> & evaluationNodes,
                                                 std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                                 const std::list<ComputationNodeBasePtr> & learnableNodes,
                                                 std::list<Matrix<ElemType>>& smoothedGradients,
                                                 const bool learnRateInitialized,
                                                 const double largestPrevLearnRatePerSample)
    {
        double epochCriterion = std::numeric_limits<double>::infinity();
        double prevCriterion = std::numeric_limits<double>::infinity();
        vector<double> epochEvalErrors(evaluationNodes.size(), std::numeric_limits<double>::infinity());

        size_t totalSamplesSeen = 0;
        double bestLearnRatePerSample = curLearnRate;

        size_t numFramesToUseInSearch = m_numMiniBatch4LRSearch[epochNumber] * m_mbSize[epochNumber];
        if (m_epochSize != requestDataSize)
        {
            // ensure the numFramesToUseInSearch does not exceed the total number of frames in the epoch
            numFramesToUseInSearch = min(numFramesToUseInSearch, m_epochSize);
        }

        double baseCriterion;

        double minLearnRate = m_minLearnRate * 0.3f;
        double learnRatePerSample = 1.0f / 8.0f / 0.618f / sqrt((double)m_mbSize[epochNumber]);

        if (learnRateInitialized && largestPrevLearnRatePerSample > 0)
        {
            //largestPrevLearnRatePerSample is per sample, first 0.618f is for compensation, second one is for safety
            learnRatePerSample = largestPrevLearnRatePerSample / 0.618f / 0.618f;
        }

        int baseModelEpoch = epochNumber - 1;
        net->LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
        net->ResetEvalTimeStamp();

        double learnRate = learnRatePerSample;
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
                                        featureNodes, labelNodes,
                                        criterionNodes, evaluationNodes,
                                        inputMatrices, learnableNodes,
                                        smoothedGradients, /*out*/ baseCriterion,
                                        /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                                        "BaseAdaptiveLearnRateSearch:");

        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
        {
            if (prevCriterion == std::numeric_limits<double>::infinity())
                prevCriterion = baseCriterion;

            double ratio = 0.3;

            if (m_epochSize != requestDataSize)
                ratio = pow(((double)numFramesToUseInSearch) / m_epochSize, 1.0f / 2);

            baseCriterion = max(ratio * prevCriterion + (1 - ratio) * baseCriterion, baseCriterion);
        }

        do
        {
            learnRatePerSample *= 0.618;
            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                            numFramesToUseInSearch, trainSetDataReader,
                                            learnRatePerSample, m_mbSize[epochNumber], featureNodes,
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
            double leftLearnRatePerSample = 0.01 / m_mbSize[epochNumber];
            double rightLearnRatePerSample = learnRatePerSample;
            double leftCriterion, rightCriterion = epochCriterion;

            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                            numFramesToUseInSearch, trainSetDataReader,
                                            leftLearnRatePerSample, m_mbSize[epochNumber],
                                            featureNodes, labelNodes,
                                            criterionNodes, evaluationNodes,
                                            inputMatrices, learnableNodes,
                                            smoothedGradients, /*out*/ leftCriterion,
                                            /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                                            "DetailBaseAdaptiveLearnRateSearch:");

            while (rightLearnRatePerSample > leftLearnRatePerSample * 1.2)
            {
                if (rightCriterion > leftCriterion)
                {
                    rightLearnRatePerSample *= 0.618;

                    TrainOneMiniEpochAndReloadModel(net, refNet, refNode,
                                                    epochNumber, numFramesToUseInSearch,
                                                    trainSetDataReader,
                                                    rightLearnRatePerSample, m_mbSize[epochNumber],
                                                    featureNodes, labelNodes,
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
                    leftLearnRatePerSample /= 0.618;

                    TrainOneMiniEpochAndReloadModel(net, refNet, refNode,
                                                    epochNumber, numFramesToUseInSearch,
                                                    trainSetDataReader,
                                                    leftLearnRatePerSample, m_mbSize[epochNumber],
                                                    featureNodes, labelNodes,
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

    template<class ElemType>
    void SGD<ElemType>::TrainOneMiniEpochAndReloadModel(ComputationNetworkPtr net,
                                                        ComputationNetworkPtr refNet,
                                                        const ComputationNodeBasePtr& refNode, const int epochNumber,
                                                        const size_t epochSize, IDataReader<ElemType>* trainSetDataReader,
                                                        const double learnRatePerSample,
                                                        const size_t minibatchSize,
                                                        const std::vector<ComputationNodeBasePtr> & featureNodes,
                                                        const std::vector<ComputationNodeBasePtr> & labelNodes,
                                                        const std::vector<ComputationNodeBasePtr> & criterionNodes,
                                                        const std::vector<ComputationNodeBasePtr> & evaluationNodes,
                                                        std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                                        const std::list<ComputationNodeBasePtr> & learnableNodes,
                                                        std::list<Matrix<ElemType>>& smoothedGradients,
                                                        /*out*/ double& epochCriterion,
                                                        /*out*/ std::vector<double>& epochEvalErrors,
                                                        /*out*/ size_t& totalSamplesSeen,
                                                        std::string prefixMsg)
    {
        TrainOneEpoch(net, refNet, refNode, epochNumber, epochSize,
                      trainSetDataReader, learnRatePerSample, minibatchSize, featureNodes,
                      labelNodes, criterionNodes, evaluationNodes,
                      inputMatrices, learnableNodes, smoothedGradients,
                      /*out*/ epochCriterion, /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                      prefixMsg);

        fprintf(stderr, "Finished Mini-Epoch For LearnRate Selection: TrainLossPerSample = %.8g;", epochCriterion);

        if (epochEvalErrors.size() == 1)
            fprintf(stderr, "EvalErrPerSample = %.8g; Ave LearnRatePerSample = %.10g\n", epochEvalErrors[0], learnRatePerSample);
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
        net->LoadPersistableParametersFromFile(GetModelNameForEpoch(baseModelEpoch), m_validateAfterModelReloading);
        net->ResetEvalTimeStamp();

        double dummyLearnRate;
        double dummtPrevCriterion;
        size_t dummyMinibatchSize = 0;
        LoadCheckPointInfo(baseModelEpoch,
                           /*out*/ totalSamplesSeen,
                           /*out*/ dummyLearnRate,
                           smoothedGradients,
                           /*out*/ dummtPrevCriterion,
                           /*out*/ dummyMinibatchSize);
    }

    template<class ElemType>
    size_t SGD<ElemType>::AdaptiveMinibatchSizing(ComputationNetworkPtr net,
                                                  ComputationNetworkPtr refNet,
                                                  const ComputationNodeBasePtr& refNode,
                                                  const int epochNumber,
                                                  const size_t numFramesToUseInSearch,
                                                  IDataReader<ElemType>* trainSetDataReader,
                                                  const double learnRatePerSample,
                                                  const size_t initialMinibatchSize,
                                                  const std::vector<ComputationNodeBasePtr> & featureNodes,
                                                  const std::vector<ComputationNodeBasePtr> & labelNodes,
                                                  const std::vector<ComputationNodeBasePtr> & criterionNodes,
                                                  const std::vector<ComputationNodeBasePtr> & evaluationNodes,
                                                  std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                                  const std::list<ComputationNodeBasePtr> & learnableNodes,
                                                  std::list<Matrix<ElemType>>& smoothedGradients,
                                                  const double learningRateAdjustmentFactor)
    {
        size_t minMinibatchSize = initialMinibatchSize;
        size_t chosenMinibatchSize = initialMinibatchSize;

        // do some pre-adjustment based on LR
        // Basically we assume that the LR for epoch 1 is safe for mbsize.
        // If LR control led to a smaller LR, then we can safely increase the lower bound of the MB size.
        double learningRateChangeSoFar = GetLearningRatePerSample(epochNumber/*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences()) / GetLearningRatePerSample(0/*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());
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
            fprintf(stderr, "before epoch .2, previous minibatchSize %zd is "
                    "considered invalid -> resetting\n", m_prevChosenMinibatchSize);
            m_prevChosenMinibatchSize = 0;
        }

        // check if we need to skip
        if (m_prevChosenMinibatchSize != 0 &&
            (epochNumber + 1) > m_minibatchSizeTuningFrequency &&
            (epochNumber + 1) % m_minibatchSizeTuningFrequency != 0)
        {
            fprintf(stderr, "AdaptiveMinibatchSearch: Search for a better minibatchSize "
                    "in epoch %d skipped, keeping minibatchSize of %zd\n",
                    epochNumber + 1, m_prevChosenMinibatchSize);
            chosenMinibatchSize = m_prevChosenMinibatchSize;
        }
        else
        {
            if (m_prevChosenMinibatchSize != 0)
            {
                // if m_prevChosenMinibatchSize (the chosen minibatch size for the previous epoch) div 2
                // is higher than initialMinibatchSize (the minibatch size we start with for this epoch),
                // then start the search with m_prevChosenMinibatchSize/2 instead of initialMinibatchSize.
                fprintf(stderr, "AdaptiveMinibatchSearch: Limiting minMinibatchSize to "
                        "largest of previous minibatchSize = (%d / 2) or %d\n",
                        (int) m_prevChosenMinibatchSize, (int) minMinibatchSize);
                minMinibatchSize = max(minMinibatchSize, m_prevChosenMinibatchSize / 2);
            }

            size_t maxMinibatchSize = m_minibatchSizeTuningMax;

            // only grow at most 2 x compared to previous step
            if (m_prevChosenMinibatchSize != 0.0f)
            {
                assert(m_prevChosenMinibatchSize >= chosenMinibatchSize);

                fprintf(stderr, "AdaptiveMinibatchSearch: Limiting maxMinibatchSize to "
                        "previous minibatchSize %zd*2\n", m_prevChosenMinibatchSize);
                maxMinibatchSize = min(maxMinibatchSize, m_prevChosenMinibatchSize * 2);
            }

            chosenMinibatchSize = SearchForBestMinibatchSize(net, refNet, refNode, epochNumber,
                                                             numFramesToUseInSearch, trainSetDataReader,
                                                             learnRatePerSample, featureNodes,
                                                             labelNodes, criterionNodes,
                                                             evaluationNodes, inputMatrices,
                                                             learnableNodes, smoothedGradients,
                                                             minMinibatchSize, maxMinibatchSize);
        }

        return chosenMinibatchSize;
    }

    static size_t RoundToMultipleOf64(float val)
    {
        return 64 * (size_t)((val + 32) / 64);
    }

    static size_t RoundToMultipleOf64(size_t val)
    {
        return 64 * ((val + 32) / 64);
    }

    // uses a small percentage of training data of minibatch to
    // speculatively train with various MB sizes; then picks the best
    template<class ElemType>
    size_t SGD<ElemType>::SearchForBestMinibatchSize(ComputationNetworkPtr net,
                                                     ComputationNetworkPtr refNet,
                                                     const ComputationNodeBasePtr& refNode,
                                                     const int epochNumber,
                                                     const size_t numFramesToUseInSearch,
                                                     IDataReader<ElemType>* trainSetDataReader,
                                                     const double learnRatePerSample,
                                                     const std::vector<ComputationNodeBasePtr> & featureNodes,
                                                     const std::vector<ComputationNodeBasePtr> & labelNodes,
                                                     const std::vector<ComputationNodeBasePtr> & criterionNodes,
                                                     const std::vector<ComputationNodeBasePtr> & evaluationNodes,
                                                     std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                                                     const std::list<ComputationNodeBasePtr> & learnableNodes,
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
        double baseCriterion = 0;

        // increase the minibatch size by a factor of sqrt(2) in each step.
        const float minibatchSizeTuningFactor = sqrtf(2.0f);

        size_t lastTriedTrialMinibatchSize = 0;
        double lastTriedTrialEpochCriterion = 0;
        for (float trialMinibatchSizeFloat = (float)minMinibatchSize;
             trialMinibatchSizeFloat <= maxMinibatchSize;
             trialMinibatchSizeFloat *= minibatchSizeTuningFactor)
        {
            // round mbsize to something meaningful
            trialMinibatchSize = RoundToMultipleOf64(trialMinibatchSizeFloat);

            fprintf(stderr, "\nAdaptiveMinibatchSearch: Evaluating trial minibatchSize=%zd out of range %zd..%zd ...\n\n",
                    trialMinibatchSize, RoundToMultipleOf64(minMinibatchSize), RoundToMultipleOf64(maxMinibatchSize));

            size_t totalSamplesSeen;
            std::vector<double> epochEvalErrors(evaluationNodes.size(), std::numeric_limits<double>::infinity());
            double epochCriterion = std::numeric_limits<double>::infinity();

            // Train on a few minibatches and so we can observe the epochCriterion as we try increasing
            // minibatches with iteration of this loop.
            TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                            numFramesToUseInSearch, trainSetDataReader,
                                            learnRatePerSample, trialMinibatchSize, featureNodes,
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
                lastTriedTrialMinibatchSize = trialMinibatchSize;
                lastTriedTrialEpochCriterion = baseCriterion;
                isFirstIteration = false;

                fprintf(stderr, "AdaptiveMinibatchSearch: Computed BaseCriterion %.10g\n", baseCriterion);
            }
            else if (!std::isnan(epochCriterion) &&
                     (epochCriterion > (baseCriterion *  (1.0 + ( m_minibatchSearchCriterionErrorMargin / 100.0)))))
            {
                // As soon as we see the Criterion (a measure of error) start to get larger than the
                // Criterion we started with, we stop.
                // TODO: if this is too sensitive, we can add a margin on the bases of percentage of
                // baseCriterion.
                break;
            }
            else
            {
                lastTriedTrialMinibatchSize = trialMinibatchSize;
                lastTriedTrialEpochCriterion = epochCriterion;
                if (trialMinibatchSizeFloat * minibatchSizeTuningFactor <= maxMinibatchSize)
                {
                   fprintf(stderr, "AdaptiveMinibatchSearch: Keep searching... "
                           "EpochCriterion = %.10g vs BaseCriterion = %.10g\n",
                           epochCriterion, baseCriterion);
                }
            }
        }
        fprintf(stderr, "AdaptiveMinibatchSearch: Search successful!!! Chose new minibatchSize of %d. "
                "EpochCriterion = %.10g vs BaseCriterion = %.10g\n\n",
                (int) lastTriedTrialMinibatchSize, lastTriedTrialEpochCriterion, baseCriterion);


        return lastTriedTrialMinibatchSize;
    }

    // Attemps to compute the error signal for the whole utterance, which will
    // be fed to the neural network as features. Currently it is a workaround
    // for the two-forward-pass sequence and ctc training, which allows
    // processing more utterances at the same time. Only used in Kaldi2Reader.
    // TODO: move the two-forward-pass support out of the reader.
    template<class ElemType>
    void SGD<ElemType>::AttemptUtteranceDerivativeFeatures(ComputationNetworkPtr net,
                                                           IDataReader<ElemType>* trainSetDataReader,
                                                           const std::vector<ComputationNodeBasePtr> & featureNodes,
                                                           std::map<std::wstring, Matrix<ElemType>*>* inputMatrices)
    {
        assert(trainSetDataReader != NULL);
        std::vector<std::vector<std::pair<wstring, size_t>>> uttInfo;
        auto pMBLayout = make_shared<MBLayout>();
        // TODO: use GetMinibatchIntoNetwork().
        while (trainSetDataReader->GetMinibatchCopy(uttInfo, *inputMatrices, pMBLayout))
        {
            ComputationNetwork::UpdateEvalTimeStamps(featureNodes);

            auto & outputNodes = net->OutputNodes();
            if (outputNodes.empty())
                LogicError("no output node was found.");

            //net->SetActualMiniBatchSizeFromFeatures();
            trainSetDataReader->CopyMBLayoutTo(net->GetMBLayoutPtr());
            net->VerifyActualNumParallelSequences(trainSetDataReader->GetNumParallelSequences());
            net->Evaluate(outputNodes[0]);   // Only evaluate the first output
            trainSetDataReader->SetNetOutput(uttInfo,
                                             dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[0])->FunctionValues(),
                                             pMBLayout);
        }
    }

    static string GeneratePaddedFloatOrExpFormat(int padSize, int precision, double value)
    {
        char format[16];
        char buffer[512];

        sprintf(format, "%%.%dg", precision);
        sprintf(buffer, format, value);

        for (int i = 0; i < strlen(buffer); i++)
        {
            if (buffer[i] == 'e' || buffer[i] == 'E')
            {
                sprintf(format, "%%%d.%de", padSize, precision);
                return format;
            }
        }
        sprintf(format, "%%%d.%df", padSize, precision);
        return format;
    }

    template<class ElemType>
    int SGD<ElemType>::SGDTrace(FILE *__restrict __stream, const char *__restrict __format, ...)
    {
        int result = 0;
        if (m_traceLevel > 0)
        {
            va_list args;
            va_start(args, __format);
            result = vfprintf(__stream, __format, args);
            va_end(args);
        }
        return result;
    }

    template<class ElemType>
    size_t SGD<ElemType>::TrainOneEpoch(ComputationNetworkPtr net,
                                        ComputationNetworkPtr refNet,
                                        const ComputationNodeBasePtr& refNode,
                                        const int epochNumber,
                                        const size_t epochSize,
                                        IDataReader<ElemType>* trainSetDataReader,
                                        const double learnRatePerSample,
                                        size_t tunedMBSize,
                                        const std::vector<ComputationNodeBasePtr> & featureNodes,
                                        const std::vector<ComputationNodeBasePtr> & labelNodes,
                                        const std::vector<ComputationNodeBasePtr> & criterionNodes,
                                        const std::vector<ComputationNodeBasePtr> & evaluationNodes,
                                        std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,  // TODO: why is this a pointer?
                                        const std::list<ComputationNodeBasePtr> & learnableNodes,
                                        std::list<Matrix<ElemType>>& smoothedGradients,
                                        /*out*/ double& epochCriterion,
                                        /*out*/ std::vector<double>& epochEvalErrors,
                                        /*out*/ size_t& totalSamplesSeen,
                                        std::string prefixMsg)
    {
        double totalTimeInMBs = 0;  // use double since timer has sub-microsecond time resolution
        double epochCriterionLastMBs = 0;

        int numSamplesLastMBs = 0;
        std::vector<double> epochEvalErrorsLastMBs(epochEvalErrors.size(), 0);

        // initialize statistics
        size_t totalEpochSamples = 0;

        int numMBsRun = 0;

        // NOTE: the following two local matrices are not used in distGradAgg path
        // assume only one training criterion node for each epoch.
        // The criterion values are accumulated here over the minibatches (without having to pull them off the GPU).
        Matrix<ElemType> localEpochCriterion(1, 1, net->GetDeviceId());
        Matrix<ElemType> localEpochEvalErrors(1, epochEvalErrors.size(), net->GetDeviceId());

        localEpochCriterion.SetValue(0);
        localEpochEvalErrors.SetValue(0);

        bool useGradientAggregation = ((m_parallelizationMethod == ParallelizationMethod::DataParallelSGD) &&
                                       (epochNumber >= m_parallelizationStartEpochNum));
        bool useModelAveraging = ((m_parallelizationMethod == ParallelizationMethod::ModelAveragingSGD) &&
                                  (epochNumber >= m_parallelizationStartEpochNum));
        bool useParallelTrain = useGradientAggregation || useModelAveraging; 

        // MA-related variables
        size_t nSamplesSinceLastModelSync = 0;
        size_t nSynced = 0; 
        float  nSecondsOnMASync = 0; 
        float  nSecondsSinceLastMAPerfReport = 0;

        std::vector<Matrix<ElemType>*> learnParamsGradients;
        if (useGradientAggregation)
        {
            epochCriterion = double(0.0);
            epochEvalErrors.assign(epochEvalErrors.size(), double(0.0));
        }

        Profiler profiler(m_numMBsToCUDAProfile);

        // resetting this, so profiling is performed for one epoch only
        m_numMBsToCUDAProfile = 0;

        bool useDistributedMBReading = useParallelTrain &&
                                       m_enableDistributedMBReading &&
                                       trainSetDataReader->SupportsDistributedMBRead();
        if (useDistributedMBReading)
        {
            trainSetDataReader->StartDistributedMinibatchLoop(tunedMBSize, epochNumber, g_mpi->CurrentNodeRank(),
                                                              g_mpi->NumNodesInUse(), epochSize);
        }
        else
        {
            trainSetDataReader->StartMinibatchLoop(tunedMBSize, epochNumber, epochSize);
        }

        net->StartEvaluateMinibatchLoop(evaluationNodes);
        net->StartEvaluateMinibatchLoop(criterionNodes);
        if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode)
        {
            refNet->StartEvaluateMinibatchLoop(refNode);
        }

        SubminibatchDispatcher<ElemType> smbDisplatcher; 
        size_t samplesInRAM = m_maxSamplesInRAM[epochNumber]; 
        // convert it to SubminibatchRequested 
        size_t numSubminibatchRequested = 0; 
        if (samplesInRAM > 0)   // if samplesInRAM = 0 , we will not use subminibatch dispatcher
        {
            size_t nParallelSequences = trainSetDataReader->GetNumParallelSequences(); 
            size_t estimatedMBSize = tunedMBSize * nParallelSequences; 
            numSubminibatchRequested = (size_t)std::ceil(estimatedMBSize / samplesInRAM);             
        }
        if (numSubminibatchRequested > 1) // only use subminibatch dispatcher if more than 1 subminibatch is required 
        {
            smbDisplatcher.Init(net, learnableNodes, criterionNodes, evaluationNodes);
        }
        size_t actualNumSubminibatch=0;

        // Attemps to compute the error signal for the whole utterance, which will
        // be fed to the neural network as features. Currently it is a workaround
        // for the two-forward-pass sequence and ctc training, which allows
        // processing more utterances at the same time. Only used in Kaldi2Reader.
        // TODO: move the two-forward-pass support out of the reader.
        AttemptUtteranceDerivativeFeatures(net, trainSetDataReader, featureNodes, inputMatrices);

        fprintf(stderr, "\nStarting minibatch loop");
        if (useGradientAggregation)
        {
            fprintf(stderr, ", DataParallelSGD training (MyRank = %d, NumNodes = %d, NumGradientBits = %d)",
                    (int)g_mpi->CurrentNodeRank(), (int)g_mpi->NumNodesInUse(), (int)m_numGradientBits);
        }
        if (useDistributedMBReading)
        {
            fprintf(stderr, ", Distributed reading is ENABLED");
        }
        if (numSubminibatchRequested > 0)
        {
            fprintf(stderr, ", with %d Max Samples in RAM", (int)samplesInRAM);
        }
        fprintf(stderr, ".\n");

        Timer timer;
        timer.Start();



        // --- MAIN MINIBATCH LOOP

        size_t totalSamplesProcessed = 0; // for progress printing   --TODO: is it different from nSamplesSinceLastModelSync?
        for (;;)
        {
            // get minibatch
            // TODO: is it guaranteed that the GPU is already completed at this point, is it safe to overwrite the buffers?
            size_t actualMBSize = 0;
            bool notAtEndOfEpoch = DataReaderHelpers::GetMinibatchIntoNetwork(*trainSetDataReader, net, criterionNodes[0],
                                                                              useDistributedMBReading, useParallelTrain, *inputMatrices, actualMBSize);
            if (!notAtEndOfEpoch)
                break;  // end of epoch
            nSamplesSinceLastModelSync += actualMBSize;

            if (numSubminibatchRequested > 0)
            {
                actualNumSubminibatch = smbDisplatcher.GetMinibatchIntoCache(*trainSetDataReader, *net, *inputMatrices, numSubminibatchRequested); 
            }
            else
            {
                actualNumSubminibatch = 0;
            }
            
            // node data was changed
            // TODO: move this to that function as well--just tired to pass everything as arguments
            // TODO: We should do this right after the GetMinibatch() call, since that's where these changed.
            //       Need to check whether that would cause unintended side effects.
            // TODO: original code did not call this for actualMBSize == 0
            ComputationNetwork::UpdateEvalTimeStamps(featureNodes);
            ComputationNetwork::UpdateEvalTimeStamps(labelNodes);

            if (actualMBSize > 0)
            {
#ifndef EVALDLL
                if (m_doGradientCheck && GradientCheck(net, criterionNodes, learnableNodes, 0) == false)
                    LogicError("cannot pass gradient checker");
#endif
                // TODO: currently only support one node regularization
                if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode)
                {
#if 0               // TODO: where does refNet get its features from?
                    refNet->ResizeAllFeatureNodes(actualMBSize);
#endif
                    //size_t actualMBSize2 = refNet->SetActualMiniBatchSizeFromFeatures();
                    size_t actualMBSize2 = refNet->DetermineActualMBSizeFromFeatures();
                    refNet->GetMBLayoutPtr()->CopyFrom(net->GetMBLayoutPtr());       // TODO: This is UNTESTED (before this was missing, seemingly inconsistently)
                    refNet->VerifyActualNumParallelSequences(trainSetDataReader->GetNumParallelSequences());

                    if (actualMBSize2 != actualMBSize)
                        LogicError("TrainOneEpoch: refNet has different MB size than main net??");

                    refNet->Evaluate(refNode);
                    Matrix<ElemType>::ScaleAndAdd((ElemType)m_adaptationRegWeight,
                                                    dynamic_pointer_cast<ComputationNode<ElemType>>(refNode)->FunctionValues(),
                                                    (ElemType)(1.0 - m_adaptationRegWeight),
                                                    dynamic_pointer_cast<ComputationNode<ElemType>>(labelNodes[0])->FunctionValues());
                }

                //compute eval node first since when gradient is computed the forward function values
                //may be changed and need to be recomputed when gradient and function value share the same matrix
                if (actualNumSubminibatch > 0)
                {
                    for (size_t ismb = 0; ismb < actualNumSubminibatch; ismb++)
                    {
                        smbDisplatcher.GetSubMinibatchToNet(ismb);
#ifdef SMB_DEBUG
                        //smbhelper.WriteInputMatriceAndMBLayout(numMBsRun, ismb);
#endif 
                        ComputationNetwork::UpdateEvalTimeStamps(featureNodes); 
                        ComputationNetwork::UpdateEvalTimeStamps(labelNodes);
                        ForwardBackward(*net, evaluationNodes, criterionNodes[0], learnRatePerSample > 0.01 * m_minLearnRate); 
                        smbDisplatcher.DoneWithCurrentSubMinibatch(ismb); 
                    }
#ifdef SMB_DEBUG
                    //smbhelper.WriteGradient(numMBsRun);
#endif 
                    smbDisplatcher.DoneWithCurrentMinibatch(); 

                }
                else 
                {
                    ForwardBackward(*net, evaluationNodes, criterionNodes[0], learnRatePerSample > 0.01 * m_minLearnRate);
                }

            } // if (actualMBSize > 0)

            // Some labels may be missing (e.g. forced alignment failed, or being gaps due to packing parallel sequences).
            //for now since we share the same label masking flag we call this on the network. 
            //Later, when we apply different labels on different nodes
            //we need to add code to call this function multiple times, one for each criteria node
            size_t numSamplesWithLabel = net->GetNumSamplesWithLabel(actualMBSize);

            totalSamplesProcessed += numSamplesWithLabel;

            // Sum of actualMBSize across all nodes when using parallel training
            size_t aggregateNumSamples = actualMBSize;
            size_t aggregateNumSamplesWithLabel = numSamplesWithLabel;

            if (!useGradientAggregation)
            {
                // accumulate criterion values (objective, eval)
                if (actualMBSize != 0)
                {
                    // criteria are in FunctionValues()(0,0), we accumulate into another 1x1 Matrix (to avoid having to pull the values off the GPU)
                    Matrix<ElemType>::AddElementToElement(dynamic_pointer_cast<ComputationNode<ElemType>>(criterionNodes[0])->FunctionValues(),
                                                          0, 0, localEpochCriterion, 0, 0);
                    for (size_t i = 0; i < evaluationNodes.size(); i++)
                    {
                        Matrix<ElemType>::AddElementToElement(dynamic_pointer_cast<ComputationNode<ElemType>>(evaluationNodes[i])->FunctionValues(),
                                                              0, 0, localEpochEvalErrors, 0, i);
                    }
                }
            }
            else
            {
                //distributed gradient aggregation
                if (learnParamsGradients.size() == 0)
                {
                    learnParamsGradients.reserve(learnableNodes.size());
                    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                    {
                        ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                        if (node->IsParameterUpdateRequired())
                        {
                            Matrix<ElemType>* currParamsGradient = &(node->GradientValues());

                            // Sometimes, in parallel training, the current node may not get any samples to process
                            // In this case, the gradient matrix may not have been sized yet. If so, lets size it.
                            if (currParamsGradient->GetNumCols() == 0)
                            {
                                Matrix<ElemType>* currParamsValues = &(node->FunctionValues());
                                currParamsGradient->Resize(currParamsValues->GetNumRows(), currParamsValues->GetNumCols());
                            }

                            learnParamsGradients.push_back(currParamsGradient);
                        }
                    }
                }

                //prepare the header
                m_gradHeader->numEvalNode = evaluationNodes.size();
                m_gradHeader->numSamples = actualMBSize;
                m_gradHeader->numSamplesWithLabel = numSamplesWithLabel;
                m_gradHeader->criterion = actualMBSize > 0 ? criterionNodes[0]->Get00Element() : 0.0;
                for (size_t i = 0; i < evaluationNodes.size(); i++)
                    m_gradHeader->evalErrors[i] = actualMBSize > 0 ? evaluationNodes[i]->Get00Element() : 0.0;

                m_distGradAgg->AggregateGradients(learnParamsGradients, m_gradHeader, m_numGradientBits, epochNumber);

                aggregateNumSamples = m_gradHeader->numSamples;
                aggregateNumSamplesWithLabel = m_gradHeader->numSamplesWithLabel;
                epochCriterion += m_gradHeader->criterion;
                for (size_t i = 0; i<epochEvalErrors.size(); i++)
                    epochEvalErrors[i] += m_gradHeader->evalErrors[i];
            }

            // update model parameters
            if ((aggregateNumSamples > 0) && (learnRatePerSample > m_minLearnRate * 0.01))
            {
                auto smoothedGradientIter = smoothedGradients.begin();
                for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, smoothedGradientIter++)
                {
                    ComputationNodeBasePtr node = *nodeIter;
                    if (node->IsParameterUpdateRequired())
                    {
                        Matrix<ElemType>& smoothedGradient = *smoothedGradientIter;
#ifdef _DEBUG
                        if (smoothedGradient.HasNan("TrainOneEpoch/UpdateWeights(): "))
                            LogicError("%ls %ls operation has NaNs in smoothedGradient.", node->NodeName().c_str(), node->OperationName().c_str());
#endif
                        UpdateWeights(node, smoothedGradient, learnRatePerSample,
                                      GetMomentumPerSample(epochNumber/*BUGBUG workaround:*/, net->GetMBLayoutPtr()->GetNumParallelSequences()), aggregateNumSamples,
                                      m_L2RegWeight, m_L1RegWeight,
                                      m_needAveMultiplier);
#ifdef _DEBUG
                        if (dynamic_pointer_cast<ComputationNode<ElemType>>(node)->FunctionValues().HasNan("TrainOneEpoch/UpdateWeights(): "))
                            LogicError("%ls %ls operation has NaNs in functionValues after parameter update.", node->NodeName().c_str(), node->OperationName().c_str());
#endif
                    }
                }
            }

            // aggregation by model averaging
            // TODO: this does not happen each MB, does it?
            if (useModelAveraging && (g_mpi->NumNodesInUse() > 1))
            {
                size_t processedSamples = 0; 
                float secondsSinceLastSyncFinished = 0; 
                float secondsSpentOnSync = 0;
                if (ModelAveragingProcessing(nSamplesSinceLastModelSync, learnableNodes, processedSamples,
                                             secondsSinceLastSyncFinished, secondsSpentOnSync))
                {
                    // if a sync happens, do some extra work
                    nSamplesSinceLastModelSync = 0; 
                    nSynced++;

                    nSecondsOnMASync += secondsSpentOnSync; 
                    nSecondsSinceLastMAPerfReport += secondsSinceLastSyncFinished; 
                    
                    if (m_iMASyncStatsTrace > 0)
                    {
                        if (nSynced % m_iMASyncStatsTrace == 0)
                        {
                            fprintf(stderr, "\t\t-----(model averaging stats) %d-th sync, %8.2f seconds since last report, %5.2f seconds on communication\n",
                                    (int)nSynced, nSecondsSinceLastMAPerfReport, nSecondsOnMASync);
                            nSecondsOnMASync = 0; 
                            nSecondsSinceLastMAPerfReport = 0; 
                        }
                    }
                }
                aggregateNumSamplesWithLabel = processedSamples;
            }

            timer.Stop();
            numMBsRun++;

            totalTimeInMBs += timer.ElapsedSeconds();
            numSamplesLastMBs += useModelAveraging ? int(actualMBSize) : int(aggregateNumSamplesWithLabel);

            if (numMBsRun % m_numMBsToShowResult == 0)
            {
                // get the epoch Values updated
                if (!useGradientAggregation)
                {
                    timer.Restart();
                    epochCriterion = localEpochCriterion.Get00Element();
                    for (size_t i = 0; i < epochEvalErrors.size(); i++)
                    {
                        epochEvalErrors[i] = localEpochEvalErrors(0, i);
                    }
                    timer.Stop();

                    // Add the last trailing compute
                    totalTimeInMBs += timer.ElapsedSeconds();
                }

                double trainLossPerSample = (epochCriterion - epochCriterionLastMBs) / numSamplesLastMBs;
                bool wasProgressPrinted = false;

                if (epochNumber > 0 || (int) epochSize > 0)
                {
                    // progress tracing for compute cluster management
                    double mbProg = (double)totalSamplesProcessed / (double)m_maxComputedEpochSize;
                    wasProgressPrinted = ProgressTracing::TraceProgressPercentage(epochNumber, mbProg);

                    // progress tracing for regular log
                    string formatString = "%s Epoch[%2d of %d]-Minibatch[%4d-%4d of %d]: SamplesSeen = %d; TrainLossPerSample = " +
                                         GeneratePaddedFloatOrExpFormat(11, 8, trainLossPerSample) + "; ";
                    SGDTrace(stderr, formatString.c_str(),
                             prefixMsg.c_str(), epochNumber + 1, m_maxEpochs, numMBsRun - m_numMBsToShowResult + 1,
                             numMBsRun, m_maxComputedEpochSize / tunedMBSize, numSamplesLastMBs, trainLossPerSample);
                }
                else
                {
                    string formatString = "%s Epoch[%2d of %d]-Minibatch[%4d-%4d of -1]: SamplesSeen = %d; TrainLossPerSample = " +
                                          GeneratePaddedFloatOrExpFormat(11, 8, trainLossPerSample) + "; ";
                    SGDTrace(stderr, formatString.c_str(),
                             prefixMsg.c_str(), epochNumber + 1, m_maxEpochs, numMBsRun - m_numMBsToShowResult + 1,
                             numMBsRun, numSamplesLastMBs, trainLossPerSample);
                    m_maxComputedEpochSize = numMBsRun * numSamplesLastMBs / m_numMBsToShowResult;
                }

                double evalError = 0.0;
                for (size_t i = 0; i < epochEvalErrors.size(); i++)
                {
                    evalError = (epochEvalErrors[i] - epochEvalErrorsLastMBs[i]) / numSamplesLastMBs;
                    string formatString = "EvalErr[%lu]PerSample = " + GeneratePaddedFloatOrExpFormat(0, 8, evalError) + "; ";
                    SGDTrace(stderr, formatString.c_str(), i, evalError);
                }

                double totalTimePerSample = (1000.0 * totalTimeInMBs) / numSamplesLastMBs;
                string formatString = "TotalTime = " + GeneratePaddedFloatOrExpFormat(0, 5, totalTimeInMBs) + "s; TotalTimePerSample = " +
                                      GeneratePaddedFloatOrExpFormat(0, 5, totalTimePerSample) + "ms; SamplesPerSecond = %d\n";
                SGDTrace(stderr, formatString.c_str(),
                         totalTimeInMBs, totalTimePerSample,
                         static_cast<int>(numSamplesLastMBs / totalTimeInMBs));

                // progress tracing for compute cluster management
                if (wasProgressPrinted)
                {
                    ProgressTracing::TraceObjectivePercentage(evalError);
                }

                if (m_traceLevel > 0)
                {
                    fflush(stderr);
                }

                // reset statistics
                totalTimeInMBs = 0;
                numSamplesLastMBs = 0;

                epochCriterionLastMBs = epochCriterion;
                for (size_t i = 0; i < epochEvalErrorsLastMBs.size(); i++)
                {
                    epochEvalErrorsLastMBs[i] = epochEvalErrors[i];
                }

                if (std::isnan(epochCriterion))
                {
                    RuntimeError("The training criterion is not a number (NAN). Stop\n");
                }
            }

            timer.Restart();
            totalEpochSamples += aggregateNumSamplesWithLabel;
            totalSamplesSeen += aggregateNumSamplesWithLabel;

            // call DataEnd function
            // This signals something from SGD to the reader.
            // DataEnd does reader specific process if sentence ending is reached
            trainSetDataReader->DataEnd(EndDataType::endDataSentence);

            // Attemps to compute the error signal for the whole utterance, which will
            // be fed to the neural network as features. Currently it is a workaround
            // for the two-forward-pass sequence and ctc training, which allows
            // processing more utterances at the same time. Only used in Kaldi2Reader.
            // TODO: move the two-forward-pass support out of the reader.
            AttemptUtteranceDerivativeFeatures(net, trainSetDataReader, featureNodes, inputMatrices);

            profiler.NextSample();
        }

        // --- END MAIN MINIBATCH LOOP

        if (useModelAveraging && (g_mpi->NumNodesInUse() > 1) )
        {
            // may not be synced after epoch finished, so do the sync here 
            int residualSampels = (int)nSamplesSinceLastModelSync;
            g_mpi->AllReduce(&residualSampels, 1);
            totalSamplesSeen += residualSampels; 
            totalEpochSamples += residualSampels;
            ModelAveragingSync(nSamplesSinceLastModelSync, learnableNodes);
            nSynced++;
            nSamplesSinceLastModelSync = 0;
        }

        // compute final criterion values
        if (useGradientAggregation)
        {
            // with parallelization, we have them in regular variables
            epochCriterion /= float(totalEpochSamples);
            for (size_t i = 0; i< epochEvalErrors.size(); i++)
            {
                epochEvalErrors[i] /= totalEpochSamples;
            }
        }
        else
        {
            // without, we have them in Matrix objects that possibly live on the GPU--get them over now
            localEpochCriterion /= float(totalEpochSamples);
            localEpochEvalErrors /= float(totalEpochSamples);

            epochCriterion = localEpochCriterion.Get00Element();
            for (size_t i = 0; i < epochEvalErrors.size(); i++)
            {
                epochEvalErrors[i] = localEpochEvalErrors(0, i);
            }
        }

        // in case of model averaging, do one more final aggregation of criteria
        if (useModelAveraging && (g_mpi->NumNodesInUse() > 1))
        {
            // merge epochCriterion and epochEvalErrors over nodes 
            g_mpi->AllReduce(&epochCriterion, 1);
            g_mpi->AllReduce(epochEvalErrors);
        }
        return totalEpochSamples;
    }

    template<class ElemType>
    void SGD<ElemType>::InitDistGradAgg(int numEvalNodes, int traceLevel)
    {
        if (m_parallelizationMethod == ParallelizationMethod::DataParallelSGD)
        {
            if (m_distGradAgg == nullptr)
            {
                m_distGradAgg = new AllReduceDistGradAggregator<ElemType>(g_mpi, m_zeroThresholdFor1Bit, true /*useQuantizationForSelfStripe*/, traceLevel);
            }

            if (m_gradHeader == nullptr)
            {
                m_gradHeader = DistGradHeader::Create(numEvalNodes);
            }
        }
    }

    template<class ElemType>
    bool SGD<ElemType>::ModelAveragingProcessing(size_t nSamplesSinceLastSync, const std::list<ComputationNodeBasePtr>& learnableNodes, size_t& nProcessedFrames,
                                  float& SecondsSinceLastSyncFinished, float& SecondsSpentOnSync)
    {
        //////////////////////////////////////////////////////////////////////////
        // the current strategy is that after each minibatch, we will sync between processors 
        // to decide whether a sync need to be performed. This is definitely not optimal, 
        // which we will fix it later. 

        // TODO: the way we handle timer is not very good 
        //////////////////////////////////////////////////////////////////////////
        static bool first = true ; 
        static Timer MAtimer;
        if (first)
        {
            MAtimer.Start(); 
            first = false; 
        }
       
        char bNeedToSync = (char)0; // use char for bool 
        if (g_mpi->IsMainNode() && nSamplesSinceLastSync >= m_nFramesBetweenMASync)
        {
            // only the main node can decide whether a sync need to be performed 
            bNeedToSync = (char)1; 
        }
        g_mpi->Bcast(&bNeedToSync, 1, g_mpi->MainNodeRank());
        if (bNeedToSync)
        {
            MAtimer.Stop();
            double elapsedsec = MAtimer.ElapsedSeconds(); 
            SecondsSinceLastSyncFinished = first ?  0  : (float) elapsedsec  ;
            MAtimer.Start();
            nProcessedFrames = ModelAveragingSync((int)nSamplesSinceLastSync, learnableNodes);
            MAtimer.Stop();
            SecondsSpentOnSync = (float)MAtimer.ElapsedSeconds();
            
            MAtimer.Start();
        }
        else
        {
            nProcessedFrames = 0; 
            return false;
        }
        return true; 
    }

    template<class ElemType>
    size_t SGD<ElemType>::ModelAveragingSync(int nSamplesSinceLastSync, const std::list<ComputationNodeBasePtr>& learnableNodes)
    {
        if (g_mpi->NumNodesInUse() <= 1)
        {
            return nSamplesSinceLastSync; 
        }

        //========================================
        // Sec. 1 calculate factor
        //========================================
        float factor = 0; 
        int   nTotalSamples = nSamplesSinceLastSync; 
        g_mpi->AllReduce(&nTotalSamples, 1);
        if (nTotalSamples <= 0)
        {
            // prepare for overflow 
            factor = 1.0f / g_mpi->NumNodesInUse(); 
        }
        else
        {
            factor = (nSamplesSinceLastSync + 0.0f) / nTotalSamples; 
        }

        //========================================
        // Sec. 2 sync models based on factor 
        // Note: this is suboptimal at the moment: 
        //       we do the averaging for each node in a sequence manner, i.e., 
        //          (node1) GPU->CPU->MPI_AllReduce -> (node2)GPU->CPU->MPI_AllReduce
        //       we can improve it by using a pipeline 
        //          (node1) GPU ->  CPU  ->  MPI_AllReduce
        //          (node2)         GPU  ->  CPU            -> MPI_AllReduce
        //          (node3)                  GPU            -> CPU              -> MPI_AllReduce
        //========================================
        for (auto iter = learnableNodes.begin(); iter != learnableNodes.end(); iter++)
        {
            ComputationNodeBasePtr pNode = *iter; 
            if (!pNode->IsParameterUpdateRequired())
                continue;

            Matrix<ElemType>& mat = dynamic_pointer_cast<ComputationNode<ElemType>>(pNode)->FunctionValues();
            // 1. normalize the weight matrix 
            Matrix<ElemType>::Scale(factor, mat);
            // 2. send weight matrix over MPI nodes; 
            ElemType* px = mat.CopyToArray(); 
            size_t    nx = mat.GetNumElements(); 

            // 3. inplace sum 
            g_mpi->AllReduce(px, nx);
            mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px);
            // 4. clean up 
            delete []px; 
        }

        return nTotalSamples; 
    }
    
// public:
    // UpdateWeightsS - static version of UpdateWeights()
    // not static since it wants to access protected methods on the SGD object
    template<class ElemType>
    /*static*/ void SGD<ElemType>::UpdateWeightsS(const SGD<ElemType>* sgd, Matrix<ElemType>& functionValues,
                               Matrix<ElemType>& gradientValues,
                               Matrix<ElemType>& smoothedGradient,
                               const double learnRatePerSample,
                               const double momentumPerSample,
                               size_t actualMBSize,
                               const double L2RegWeight,
                               const double L1RegWeight,
                               const bool needAveMultiplier)
    {
        // we use simple linear (instead of log linear) scaling here
        const double momentum = MomentumPerMB(momentumPerSample, actualMBSize);
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
        double noiseStd = sgd->GradientUpdateNoiseStd();
        Matrix<ElemType> sgdUpdateNoise((DEVICEID_TYPE)functionValues.GetDeviceId());
        if (noiseStd > 0)
        {
            // get the gradient structure since gradient is sparse
            sgdUpdateNoise.SetValue(gradientValues);

            // reset its value to random
            sgdUpdateNoise.SetGaussianRandomValue(0, (ElemType)noiseStd);
        }

        // L2 regularizer
        if (L2RegWeight > 0)
        {
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            Matrix<ElemType>::ScaleAndAdd((ElemType)(L2RegWeight * actualMBSize), functionValues, gradientValues);
        }

        if (adpType == GradientsUpdateType::None)
        {
            smoothedGradient.NormalGrad(gradientValues, functionValues,
                                        (ElemType)learnRatePerSample, (ElemType)momentum);
        }
        else if (adpType == GradientsUpdateType::AdaGrad ||
                (adpType == GradientsUpdateType::RmsProp && gradientValues.GetMatrixType() == MatrixType::SPARSE) ||
                (adpType == GradientsUpdateType::FSAdaGrad && gradientValues.GetMatrixType() == MatrixType::SPARSE))
        {
            //rmsprop for sparse is not implemented yet, delegate it with adagrad

            double aveMultiplier = smoothedGradient.Adagrad(gradientValues, needAveMultiplier);
            Matrix<ElemType>::ScaleAndAdd((ElemType)(-learnRatePerSample / aveMultiplier), gradientValues, functionValues);
        }
        else if (adpType == GradientsUpdateType::FSAdaGrad)
        {
            smoothedGradient.FSAdagrad(actualMBSize, gradientValues, functionValues, (ElemType)learnRatePerSample, (ElemType)momentum);
        }
        else if (adpType == GradientsUpdateType::RmsProp)
        {
            double aveMultiplier = smoothedGradient.RmsProp(gradientValues, (ElemType)sgd->m_rpi.gamma,
                                                            (ElemType)sgd->m_rpi.inc, (ElemType)sgd->m_rpi.max,
                                                            (ElemType)sgd->m_rpi.dec, (ElemType)sgd->m_rpi.min, needAveMultiplier);
            Matrix<ElemType>::ScaleAndAdd((ElemType)(-learnRatePerSample / aveMultiplier), gradientValues, functionValues);
        }

        if (noiseStd > 0)
        {
            Matrix<ElemType>::ScaleAndAdd(1.0, sgdUpdateNoise, functionValues);
        }

        // L1 regularizer with proximal gradient descent method
        if (L1RegWeight > 0)
        {
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            functionValues.InplaceSoftThreshold((ElemType)(learnRatePerSample * L1RegWeight * actualMBSize));
        }

#if DUMPOUTPUT
        functionValues.Print("Parameter Update");
#endif
    }

// protected:

    // UpdateWeights - update the weights in
    template<class ElemType>
    void SGD<ElemType>::UpdateWeights(const ComputationNodeBasePtr& node,
                       Matrix<ElemType>& smoothedGradient,
                       const double learnRatePerSample,
                       const double momentumPerSample,
                       const size_t actualMBSize,
                       const double L2RegWeight, const double L1RegWeight,
                       const bool needAveMultiplier) const
    {
#if DUMPOUTPUT
        fprintf(stderr, "Update_%ls\n", node->NodeName().c_str());
#endif
        if (!node->IsParameterUpdateRequired())
            LogicError("UpdateWeights() called for a learnable ComputationNode which has m_parameterUpdateRequired == false!");

        UpdateWeightsS(this, dynamic_pointer_cast<ComputationNode<ElemType>>(node)->FunctionValues(), dynamic_pointer_cast<ComputationNode<ElemType>>(node)->GradientValues(),
                       smoothedGradient, learnRatePerSample, momentumPerSample,
                       actualMBSize, L2RegWeight, L1RegWeight,
                       needAveMultiplier);
        node->UpdateEvalTimeStamp();
    }

    template<class ElemType>
    void SGD<ElemType>::ClipGradient(Matrix<ElemType>& gradient, const size_t actualMBSize) const
    {
        if (m_clippingThresholdPerSample != std::numeric_limits<double>::infinity())
        {
            double maxGradientPerMB = m_clippingThresholdPerSample * actualMBSize;
            if (m_gradientClippingWithTruncation)
                gradient.InplaceTruncate((ElemType)(maxGradientPerMB));
            else
            {
                // norm2 normalized
                double gradientNorm = gradient.FrobeniusNorm();
                if (gradientNorm > maxGradientPerMB)
                {
                    double normFactor = maxGradientPerMB / gradientNorm;
                    gradient *= (ElemType)normFactor;
                }
            }
        }
    }

    template<class ElemType>
    void SGD<ElemType>::SaveCheckPointInfo(const size_t epoch, const size_t totalSamplesSeen,
                            const double learnRatePerSample,
                            const std::list<Matrix<ElemType>>& smoothedGradients,
                            const double prevCriterion,
                            const size_t minibatchSize)
    {
        // In case of parallel training only the main node should we saving the checkpoint to prevent
        // the parallel training nodes from colliding to write the same file
        if ((g_mpi == nullptr) || g_mpi->IsMainNode())
        {
            wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epoch));
            // Saving into temporary file and then renaming it to the checkPointFileName
            // This is a standard trick to avoid havign corrupted checkpoints files if process dies during writing
            wstring tempFileName = checkPointFileName + L".tmp";

            {
                File fstream(tempFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsWrite);
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

                // Ensuring that data is written
                fstream.Flush();
            }

            renameOrDie(tempFileName, checkPointFileName);
        }
    }

    template<class ElemType>
    bool SGD<ElemType>::LoadCheckPointInfo(const size_t epochNumber,
                            /*out*/ size_t& totalSamplesSeen,
                            /*out*/ double& learnRatePerSample,
                            std::list<Matrix<ElemType>>& smoothedGradients,
                            /*out*/ double& prevCriterion,
                            /*out*/ size_t& minibatchSize)
    {
        wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epochNumber));
        if (!fexists(checkPointFileName.c_str()))
        {
            fprintf(stderr, "Warning: checkpoint file is missing. learning parameters will be initialized from 0\n");
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

    template<class ElemType>
    wstring SGD<ElemType>::GetCheckPointFileNameForEpoch(const int epoch)
    {
        return GetModelNameForEpoch(epoch) + L".ckp";
    }

    template<class ElemType>
    wstring SGD<ElemType>::GetModelNameForEpoch(const int epoch, bool bLastModel)
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
    template<class ElemType> // TODO: needed?
    int SGD<ElemType>::DetermineStartEpoch(const bool makeMode)
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
                firstEpoch = e + 1;
                break;
            }
            else
            {
                curEpochFile = prevEpochFile;
            }
        }
        if (firstEpoch == m_maxEpochs)
            fprintf(stderr, "Final model exists: %ls\n", GetModelNameForEpoch(firstEpoch - 1).c_str());

        return firstEpoch;
    }

#define EPSILON 1e-5

    // this probes the automatic gradient computation with random inputs
    template<class ElemType>
    bool SGD<ElemType>::GradientCheck(ComputationNetworkPtr net,
                                      const std::vector<ComputationNodeBasePtr> & criterionNodes,
                                      const std::list<ComputationNodeBasePtr> & learnableNodes,
                                      int npos)
    {
        vector<string> errMsgs;

        net->StartEvaluateMinibatchLoop(criterionNodes[npos]);

        // gradient checking
        for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
        {
            ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
            char wstrtmp[2048];

            for (size_t itry = 0; itry < min((size_t)50, node->FunctionValues().GetNumElements()); itry++)
            {
                /// no support to sparse matrix yet
                int irow = (int) fmod(rand(), node->GetNumRows() - 1);
                int icol = (int) fmod(rand(), node->GetNumCols() - 1);
                irow = max(0, irow);
                icol = max(0, icol);

                fprintf(stderr, "\n###### d%ls######\n", node->NodeName().c_str());

                double eOrg = node->FunctionValues()(irow, icol);
                node->FunctionValues().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

                node->UpdateEvalTimeStamp();

                net->ComputeGradient<ElemType>(criterionNodes[npos]);

                if (node->GradientValues().GetMatrixType() == MatrixType::SPARSE)
                {
                    break;
                }

                //double mbEvalCri =
                //criterionNode should be a scalar
                // TODO: why is this value not used?
                criterionNodes[npos]->Get00Element();
                double eGradErr = node->GradientValues()(irow, icol);
                node->GradientValues().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

                double ePos = eOrg + EPSILON;
                double eNeg = eOrg - EPSILON;

                node->FunctionValues()(irow, icol) = (ElemType)ePos;
                node->FunctionValues().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

                node->UpdateEvalTimeStamp();
                net->Evaluate(criterionNodes[npos]);
                //criterionNode should be a scalar

                double mbEvalCriPos = criterionNodes[npos]->Get00Element(); // TODO: make Get00Element() a function of ComputationNodeBase

                node->FunctionValues()(irow, icol) = (ElemType)eNeg;
                node->FunctionValues().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

                node->UpdateEvalTimeStamp();
                net->Evaluate(criterionNodes[npos]);

                // criterionNode should be a scalar
                double mbEvalCriNeg = criterionNodes[npos]->Get00Element();

                // back to its orginal parameter value
                node->FunctionValues()(irow, icol) = (ElemType)eOrg;
                node->FunctionValues().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

                // check if they are consistent
                double eGradNum = ((mbEvalCriPos - mbEvalCriNeg) / (ePos - eNeg));
                double threshold = pow(10.0,
                                       max(0.0,
                                           ceil(log10(min(fabs(eGradErr),
                                                          fabs(eGradNum))))) - (int)m_gradientCheckSigDigit);
                double diff = fabs(eGradErr - eGradNum);
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

        return errMsgs.size() == 0;
    }

    template class SGD<float>;
    template class SGD<double>;

    // register SGD<> with the ScriptableObject system
    ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<SGD<float>, SGD<double>> registerSGDOptimizer(L"SGDOptimizer");

}}}
