//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNetwork.h"
#include "SimpleEvaluator.h"
#include "DataReader.h"
#include "ScriptableObjects.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "fileutil.h"
#include "Config.h"
#include <chrono>
#include <random>
#include "Profiler.h"
#include "MASGD.h"

using namespace std; // ugh! TODO: get rid of this from .h files!!!

#define CNTK_CHECKPOINT_VERSION_1 1     // 1 -> no version number 
#define CNTK_CHECKPOINT_VERSION_2 2     
#define CURRENT_CNTK_CHECKPOINT_VERSION CNTK_CHECKPOINT_VERSION_2


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
    RmsProp,
    FSAdaGrad
};

// TODO: While currently combining these methods is not supported,
// these are not mutually exclusive and we can/should support combinations of these
// in the future
enum class ParallelizationMethod : int
{
    None = 0,
    DataParallelSGD = 1,
    ModelAveragingSGD = (1 << 1),
    ModelParallelSGD = (1 << 2), // Currently unsupported
};

// configuration parameters associated with RMSProp learning algorithm
struct RMSPropInfo
{
    double gamma;
    double inc;
    double dec;
    double max;
    double min;

    RMSPropInfo()
    {
        gamma = 0.99;
        inc = 1.2;
        dec = 0.75;
        max = 10.0;
        min = 0.1;
    }
};

struct GradientUpdateInfo
{
    GradientsUpdateType mType;
    float mGaussianNoiseInjectStd;

    GradientUpdateInfo()
    {
        mType = GradientsUpdateType::AdaGrad;
        mGaussianNoiseInjectStd = 0.0075f;
    }
};

// ---------------------------------------------------------------------------
// SGDParams -- parameters for SGD
//
// TODO: This should keep everything that is configured by the config.
//       Currently it does not store which matrices are used.
// ---------------------------------------------------------------------------

struct SGDParams : public ScriptableObjects::Object
{
    template <class ConfigRecord> // (needed for default value of m_gradientBits)
    SGDParams(const ConfigRecord& configSGD, size_t sizeofElemType);

    SGDParams(const ScriptableObjects::IConfigRecordPtr configp);

    // SGDParams(SGDParams&&) = default; // (does not compile in VS 2013; not critical)

protected:
    // learning rate per sample provided outside
    floatargvector m_learningRatesParam;
    intargvector m_learningRatesSpecifiedForMBSize; // 1 for per sample, m_mbSize[] for per MB
    floatargvector m_momentumParam;
    intargvector m_momentumSpecifiedForMBSize;
    bool m_useNesterovMomentum;

    // Determine the MB size used for mapping a given learning-rate or momentum parameter to a per-sample value.
    // MB size is the number of samples across all time steps and parallel sequences.
    // This function exists to post-fix a design bug in SGD:
    // In the case of BPTT, the 'minibatchSize' parameter given to the SGD module really means the truncation size,
    // while the MB size to be used is (truncation size * number of parallel sequences).
    // SGD also does not know #parallel sequences upfront.
    size_t FixUpEffectiveMBSize(size_t specifiedMBSize, size_t numParallelSequences) const
    {
        // remedy the bug that truncation size is incorrectly passed as MB size
        if (m_truncated && specifiedMBSize > 1)      // currently only happens in this mode
            specifiedMBSize *= numParallelSequences; // assume 'specifiedMBSize' refers to truncation size
        // end bug post-fix
        // TODO: This ^^ should go away once SGD gets fixed to take the truncation size as a parameter.

        return specifiedMBSize;
    }

    // helpers to convert learning rates to per-sample values used in the actual algorithms
    // 'numParallelSequences' must be specified because of the definitional MB-size bug in SGD mentioned above, and should go away once that is sorted out.
    double GetLearningRatePerSample(size_t epoch /*BUGBUG workaround:*/, size_t numParallelSequences) const
    {
        return m_learningRatesParam[epoch] / FixUpEffectiveMBSize(m_learningRatesSpecifiedForMBSize[epoch], numParallelSequences);
    }
    double GetMomentumPerSample(size_t epoch /*BUGBUG workaround:*/, size_t numParallelSequences) const
    {
        return pow(m_momentumParam[epoch], 1.0 / FixUpEffectiveMBSize(m_momentumSpecifiedForMBSize[epoch], numParallelSequences));
    }

    ParallelizationMethod GetParallelizationMethod() const
    {
        if (m_mpi == nullptr)
            return ParallelizationMethod::None;

        return m_parallelizationMethod;
    }

    // only true when the user specify LearningRatePerMB and the number of parallel utterances in Reader > 1
    // bool m_needToNormalizeLRByParallUtterance;          // TODO: should go away
    // bool m_needToNormalizeMomentumByParallUtterance;

    intargvector m_mbSize;
    bool m_truncated; // do BPTT
    // BUGBUG: The 'Truncated' option is duplicated in the reader and must be set to the same there (e.g. by defining in the config on an outer enclosing level, like current samples).
    //         We really should only read it in SGD and pass it ourselves on to the Reader, instead of it being a Reader parameter.
    // BUGBUG: If m_truncated, then m_mbSize is interpreted as truncation length; the actual MB size is a combination of that and the #parallel sequences specified in the reader.
    // TODO: do not specify 'Truncated' but 'TruncatedLength', set m_truncated so given, and let m_mbSize control how many #parallel sequences the reader is allowed to pack into an MB.
    size_t m_maxSamplesInRAM;
    // This is related with subminibatch implementation
    // maxSamplesInRAM denotes how many samples we used in forward-backward on net.
    // Due to the GPU memory limitations, it is sometime not possible to hold the m_mbSize in RAM.
    // To mitigate this issue, we adopt the sub-minibatch implementation, where
    // each m_mbSize[epoch] is divided by a few sub-minibatch of which size will be no more than m_maxSamplesInRAM
    // a forward-backward is performed for each sub-minibathch; a model update is performed after each minibatch
    size_t m_numSubminiBatches;
    // alternative method to specify how to split minibatches into subminibatches
    // default is 1, which means no subminibatch is used
    // if m_maxTempMemSizeInSamples = SIZE_MAX (which means users do not specify the option) and m_numSubminiBatches > 1
    // we divide one minibatch to m_numSubminiBatches subMinibatches

    // the number of samples in each epoch (0 means, use all the samples in each epoch).
    size_t m_epochSize;
    size_t m_maxComputedEpochSize;

    // the total number of epochs to run.
    size_t m_maxEpochs;

    bool m_gradientClippingWithTruncation;
    double m_clippingThresholdPerSample;

    intargvector m_numMiniBatch4LRSearch;
    size_t m_numBestSearchEpoch;

    LearningRateSearchAlgorithm m_autoLearnRateSearchType;

    AdaptationRegType m_adaptationRegType;
    double m_adaptationRegWeight;
    bool m_needAdaptRegularization;

    bool m_loadBestModel;
    double m_reduceLearnRateIfImproveLessThan;
    bool m_continueReduce;

    // determine after how many epochs the learning rate should be auto adjusted.
    size_t m_learnRateAdjustInterval;

    bool m_useCVSetControlLRIfCVExists;
    bool m_useEvalCriterionControlLR;

    double m_increaseLearnRateIfImproveMoreThan;
    double m_learnRateIncreaseFactor;
    double m_learnRateDecreaseFactor;
    bool m_autoAdjustMinibatch;
    size_t m_minibatchSearchCriterionErrorMargin;
    size_t m_minibatchSizeTuningFrequency;
    size_t m_minibatchSizeTuningMax;

    doubleargvector m_dropoutRates;
    doubleargvector m_batchNormalizationTimeConstant;
    int m_setBNToEvalModeAfterEpochNumber;
    size_t m_maxTempMemSizeInSamplesForCNN;

    int m_traceLevel;

    size_t m_numPrevLearnRates;

    double m_minLearnRate;

    GradientUpdateInfo m_gradType;
    RMSPropInfo m_rpi;

    int m_numMBsToShowResult;
    int m_numMBsToCUDAProfile;

    bool m_doGradientCheck;
    double m_gradientCheckSigDigit;

    bool m_doUnitTest;

    bool m_useAllDataForPreComputedNode;

    // Parallel training
    MPIWrapperPtr m_mpi;

    ParallelizationMethod m_parallelizationMethod;
    bool m_enableDistributedMBReading;
    int m_parallelizationStartEpochNum;

    // decide if/how often we measure and show sync performance stats (seconds spend on sync, seconds since last sync etc.) ?
    // 0: No sync perfomance stats
    // 1: Show stats on every sync
    // n > 1: Show stats after every n sync
    int m_syncStatsTrace;

    // Data parallel SGD training parameters
    int m_numGradientBits;
    bool m_bufferedAsyncGradientAggregation;
    bool m_zeroThresholdFor1Bit;

    // Parallel training related with MA
    size_t m_nFramesBetweenMASync;

    bool m_needAveMultiplier;
    double m_L2RegWeight;
    double m_L1RegWeight;

    // sequence training
    double m_hSmoothingWeight;
    double m_frameDropThresh;
    bool m_doReferenceAlign;
    double m_seqGammarCalcAMF;
    double m_seqGammarCalcLMF;
    double m_seqGammarCalcWP;
    double m_seqGammarCalcbMMIFactor;
    bool m_seqGammarCalcUsesMBR;
};

template <class ElemType>
class IDistGradAggregator;

// -----------------------------------------------------------------------
// class SGD
// -----------------------------------------------------------------------

// TODO: make this independent of ElemType. Then these repeated dynamic_pointer_casts will go away
// TODO: why is this a class, and not just a procedure? Then we wouldn't have to include the massive header
template <class ElemType>
class SGD : public SGDParams
{
protected:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

public:
    // constructor from old CNTK config. This is a function template that is also used to get the config from Scripting.
    template <class ConfigRecordType>
    SGD(const ConfigRecordType& configSGD)
        : SGDParams(configSGD, sizeof(ElemType)),
          // TODO: The next few do not belong into SGD any more than the network or reader we operate on. Either move network and reader in here, or move these out.
          m_modelPath((const wstring&) configSGD(L"modelPath")),
          m_keepCheckPointFiles(configSGD(L"keepCheckPointFiles", false)),
          // m_validateAfterModelReloading(configSGD(L"validateAfterModelReloading", true)),
          m_trainCriterionNodeName((const wstring&) configSGD(L"trainCriterionNodeName", L"")),
          m_evalCriterionNodeName((const wstring&) configSGD(L"evalCriterionNodeName", L"")),
          m_traceNodeNamesReal(configSGD(L"traceNodeNamesReal", ConfigRecordType::Array(stringargvector()))),
          m_traceNodeNamesCategory(configSGD(L"traceNodeNamesCategory", ConfigRecordType::Array(stringargvector()))),
          m_prevChosenMinibatchSize(0),
          m_lastFinishedEpochTrainLoss(0.0),
          m_distGradAgg(nullptr),
          m_gradHeader(nullptr)
    {
        msra::files::make_intermediate_dirs(m_modelPath);
    }
    // note: This must be in the header, as we cannot properly specialize this constructor in the CPP to make sure all versions are generated.

    // constructor from Scripting
    SGD(const ScriptableObjects::IConfigRecordPtr configp)
        : SGD(*configp)
    {
    }

    void InitMPI(const MPIWrapperPtr& mpi)
    {
        m_mpi = mpi;

        if (m_mpi == nullptr)
            m_parallelizationMethod = ParallelizationMethod::None;
    }

    void Train(function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn, DEVICEID_TYPE deviceId,
               IDataReader* trainSetDataReader,
               IDataReader* validationSetDataReader,
               const bool makeMode = true);
    void Adapt(wstring origModelFileName, wstring refNodeName,
               IDataReader* trainSetDataReader,
               IDataReader* validationSetDataReader,
               const DEVICEID_TYPE deviceID, const bool makeMode = true);

protected:

    std::vector<ComputationNodeBasePtr>& GetTrainCriterionNodes(ComputationNetworkPtr net);
    std::vector<ComputationNodeBasePtr>& GetEvalCriterionNodes(ComputationNetworkPtr net);

    void TrainOrAdaptModel(int startEpoch, ComputationNetworkPtr net,
                           bool networkLoadedFromCheckpoint,
                           ComputationNetworkPtr refNet,
                           ComputationNodeBasePtr refNode,
                           IDataReader* trainSetDataReader,
                           IDataReader* validationSetDataReader);

protected:

    // return true if precomputation is executed.
    bool PreCompute(ComputationNetworkPtr net,
                    IDataReader* trainSetDataReader,
                    std::vector<ComputationNodeBasePtr>& featureNodes,
                    std::vector<ComputationNodeBasePtr>& labelNodes,
                    StreamMinibatchInputs* inputMatrices);

    // return a reasonable initial learning rate based on the initial mbsize
    double SearchForBestLearnRate(ComputationNetworkPtr net,
                                  ComputationNetworkPtr refNet,
                                  const ComputationNodeBasePtr& refNode, const int epochNumber,
                                  const double curLearnRate,
                                  IDataReader* trainSetDataReader,
                                  const std::vector<ComputationNodeBasePtr>& featureNodes,
                                  const std::vector<ComputationNodeBasePtr>& labelNodes,
                                  const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                  const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                  StreamMinibatchInputs* inputMatrices,
                                  const std::list<ComputationNodeBasePtr>& learnableNodes,
                                  std::list<Matrix<ElemType>>& smoothedGradients,
                                  const bool learnRateInitialized,
                                  const double largestPrevLearnRatePerSample);

    void TrainOneMiniEpochAndReloadModel(ComputationNetworkPtr net,
                                         ComputationNetworkPtr refNet,
                                         const ComputationNodeBasePtr& refNode, const int epochNumber,
                                         const size_t epochSize, IDataReader* trainSetDataReader,
                                         const double learnRatePerSample,
                                         const size_t minibatchSize,
                                         const std::vector<ComputationNodeBasePtr>& featureNodes,
                                         const std::vector<ComputationNodeBasePtr>& labelNodes,
                                         const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                         const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                         StreamMinibatchInputs* inputMatrices,
                                         const std::list<ComputationNodeBasePtr>& learnableNodes,
                                         std::list<Matrix<ElemType>>& smoothedGradients,
                                         /*out*/ double& epochCriterion,
                                         /*out*/ std::vector<double>& epochEvalErrors,
                                         /*out*/ size_t& totalSamplesSeen,
                                         std::string prefixMsg = "");

    size_t AdaptiveMinibatchSizing(ComputationNetworkPtr net,
                                   ComputationNetworkPtr refNet,
                                   const ComputationNodeBasePtr& refNode,
                                   const int epochNumber,
                                   const size_t numFramesToUseInSearch,
                                   IDataReader* trainSetDataReader,
                                   const double learnRatePerSample,
                                   const size_t initialMinibatchSize,
                                   const std::vector<ComputationNodeBasePtr>& featureNodes,
                                   const std::vector<ComputationNodeBasePtr>& labelNodes,
                                   const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                   const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                   StreamMinibatchInputs* inputMatrices,
                                   const std::list<ComputationNodeBasePtr>& learnableNodes,
                                   std::list<Matrix<ElemType>>& smoothedGradients,
                                   const double learningRateAdjustmentFactor);

    // uses a small percentage of training data of minibatch to
    // speculatively train with various MB sizes; then picks the best
    size_t SearchForBestMinibatchSize(ComputationNetworkPtr net,
                                      ComputationNetworkPtr refNet,
                                      const ComputationNodeBasePtr& refNode,
                                      const int epochNumber,
                                      const size_t numFramesToUseInSearch,
                                      IDataReader* trainSetDataReader,
                                      const double learnRatePerSample,
                                      const std::vector<ComputationNodeBasePtr>& featureNodes,
                                      const std::vector<ComputationNodeBasePtr>& labelNodes,
                                      const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                      const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                      StreamMinibatchInputs* inputMatrices,
                                      const std::list<ComputationNodeBasePtr>& learnableNodes,
                                      std::list<Matrix<ElemType>>& smoothedGradients,
                                      const size_t minMinibatchSize, const size_t maxMinibatchSize);

    // Attemps to compute the error signal for the whole utterance, which will
    // be fed to the neural network as features. Currently it is a workaround
    // for the two-forward-pass sequence and ctc training, which allows
    // processing more utterances at the same time. Only used in Kaldi2Reader.
    // TODO: move the two-forward-pass support out of the reader.
    void AttemptUtteranceDerivativeFeatures(ComputationNetworkPtr net,
                                            IDataReader* trainSetDataReader,
                                            const std::vector<ComputationNodeBasePtr>& featureNodes,
                                            StreamMinibatchInputs* inputMatrices);

    size_t TrainOneEpoch(ComputationNetworkPtr net,
                         ComputationNetworkPtr refNet,
                         const ComputationNodeBasePtr& refNode,
                         const int epochNumber,
                         const size_t epochSize,
                         IDataReader* trainSetDataReader,
                         const double learnRatePerSample,
                         size_t tunedMBSize,
                         const std::vector<ComputationNodeBasePtr>& featureNodes,
                         const std::vector<ComputationNodeBasePtr>& labelNodes,
                         const std::vector<ComputationNodeBasePtr>& criterionNodes,
                         const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                         StreamMinibatchInputs* inputMatrices,
                         const std::list<ComputationNodeBasePtr>& learnableNodes,
                         std::list<Matrix<ElemType>>& smoothedGradients,
                         /*out*/ double& epochCriterion,
                         /*out*/ std::vector<double>& epochEvalErrors,
                         /*out*/ size_t& totalSamplesSeen,
                         std::string prefixMsg = "");

    void InitDistGradAgg(int numEvalNodes, int traceLevel);
    void InitModelAggregationHandler(int traceLevel);
public:
    // UpdateWeightsS - static version of UpdateWeights()
    static void UpdateWeightsS(const SGD* sgd, Matrix<ElemType>& functionValues,
                               Matrix<ElemType>& gradientValues,
                               Matrix<ElemType>& smoothedGradient,
                               const double learnRatePerSample,
                               const double momentumPerSample,
                               size_t actualMBSize,
                               const double L2RegWeight,
                               const double L1RegWeight,
                               const bool needAveMultiplier,
                               const bool useNesterovMomentum);

protected:
    // UpdateWeights - update the weights in
    void UpdateWeights(const ComputationNodeBasePtr& node,
                       Matrix<ElemType>& smoothedGradient,
                       const double learnRatePerSample,
                       const double momentumPerSample,
                       const size_t actualMBSize,
                       const double L2RegWeight, const double L1RegWeight,
                       const bool needAveMultiplier,
                       const bool useNesterovMomentum) const;

    void ClipGradient(Matrix<ElemType>& gradient, const size_t actualMBSize) const;

    void SaveCheckPointInfo(const size_t epoch, const size_t totalSamplesSeen,
                            const double learnRatePerSample,
                            const std::list<Matrix<ElemType>>& smoothedGradients,
                            const double prevCriterion,
                            const size_t minibatchSize);

    bool LoadCheckPointInfo(const size_t epochNumber,
                            /*out*/ size_t& totalSamplesSeen,
                            /*out*/ double& learnRatePerSample,
                            std::list<Matrix<ElemType>>& smoothedGradients,
                            /*out*/ double& prevCriterion,
                            /*out*/ size_t& minibatchSize);

    wstring GetCheckPointFileNameForEpoch(const int epoch);
    wstring GetModelNameForEpoch(const int epoch, bool bLastModel = false);

    // return -1 if nothing exists
    int DetermineStartEpoch(const bool makeMode);

    GradientsUpdateType GradUpdateType() const
    {
        return m_gradType.mType;
    }
    double GradientUpdateNoiseStd() const
    {
        return m_gradType.mGaussianNoiseInjectStd;
    }

public:
#define EPSILON 1e-5

    bool GradientCheck(ComputationNetworkPtr net,
                       const std::vector<ComputationNodeBasePtr>& criterionNodes,
                       const std::list<ComputationNodeBasePtr>& learnableNodes,
                       int npos);

protected:
    wstring m_modelPath;
    bool m_keepCheckPointFiles;
    // bool m_validateAfterModelReloading; // TODO: remove this. Why would one not validate a model?

    wstring m_trainCriterionNodeName;
    wstring m_evalCriterionNodeName;

    // enable tracing. Nodes listed here get their m_traceNodeValue and m_traceNodeValueAsCategoryLabel flags set
    vector<wstring> m_traceNodeNamesReal;
    vector<wstring> m_traceNodeNamesCategory;

    size_t m_prevChosenMinibatchSize;
    double m_lastFinishedEpochTrainLoss;

    IDistGradAggregator<ElemType>* m_distGradAgg;
    struct DistGradHeader* m_gradHeader;

    shared_ptr<IMASGD<ElemType>> m_pMASGDHelper;

private:
    int SGDTrace(FILE* __restrict __stream, bool isPrependTimestamp, const char* __restrict __format, ...);
};

}}}
