//
// <copyright file="SGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "ComputationNetwork.h"
#include "NonlinearityNodes.h"          // for DropoutNode
#include "CompositeComputationNodes.h"  // for PrecomputeNode
#include "SimpleEvaluator.h"
#include "DataReader.h"
#include "IComputationNetBuilder.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "fileutil.h"
#include "commandArgUtil.h"
#include <chrono> 
#include <random>
#include "TimerUtility.h"
#include "Profiler.h"

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

template<class ElemType> class IDistGradAggregator;

// TODO: make this independent of ElemType. Then these repeated dynamic_pointer_casts will go away
// TODO: why is this a class, and not just a procedure? Then we wouldn't have to include the massive header
template<class ElemType>
class SGD
{
protected:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

public:
    SGD(const ConfigParameters& configSGD);

    //autoLearnRateSearchType is applied only if the learning rate for the epoch is not specified in learningRatesPerMB and learningRatesPerSample
    void Init(const floatargvector& learningRatesPerMB,
              const floatargvector& learningRatesPerSample,
              const intargvector& mbSize,
              const size_t epochSize,
              const size_t maxEpochs,
              const wstring& modelPath,
              const floatargvector& momentumPerMB,
              const floatargvector& momentumPerSample,
              const bool gradientClippingWithTruncation,
              const double clippingThresholdPerSample,
              const LearningRateSearchAlgorithm autoLearnRateSearchType,
              const double increaseLearnRateIfImproveMoreThan,
              const double learnRateIncreaseFactor,
              const double reduceLearnRateIfImproveLessThan,
              const bool continueReduce,
              const double learnRateDecreaseFactor,
              floatargvector dropoutRates,
              const bool loadBestModel,
              const intargvector& numMiniBatch4LRSearch,
              const size_t numPrevLearnRates,
              const size_t numBestSearchEpoch,
              const int traceLevel,
              const size_t numMBsToShowResult,
              const size_t numMBsToCUDAProfile,
              const size_t maxTempMemSizeInSamplesForCNN,
              const GradientUpdateInfo gradUpdateType,
              const bool keepCheckPointFiles,
              const AdaptationRegType adaptationRegType,
              const double adaptationRegWeight,
              const wstring trainCriterionNodeName,
              const wstring evalCriterionNodeName,
              const bool doGradientCheck,
              const double gradientCheckSigDigit,
              const bool validateAfterModelReloading,
              RMSPropInfo rpi,
              size_t learnRateAdjustInterval,
              const bool UsingAllDataForPreComputed,
              const bool needAveMultiplier,
              const double L2RegWeight,
              const double L1RegWeight,
              const bool autoAdjustMinibatch,
              const size_t minibatchSizeTuningFrequency,
              const size_t minibatchSizeTuningMax,
              const bool useCVSetControlLRIfCVExists,
              const bool useEvalCriterionControlLR,
              const size_t minibatchSearchCriterionErrorMargin,
              const ElemType hsmoothingWeight = 1.0,
              const ElemType frameDropThresh = 1e-10,
              const bool doreferencealign = false);

    void Adapt(wstring origModelFileName, wstring refNodeName,
               IDataReader<ElemType>* trainSetDataReader,
               IDataReader<ElemType>* validationSetDataReader,
               const DEVICEID_TYPE deviceID, const bool makeMode = true);
    void SequenceTrain(IComputationNetBuilder<ElemType>* netBuilder, wstring origModelFileName,
                       IDataReader<ElemType>* trainSetDataReader, IDataReader<ElemType>* validationSetDataReader,
                       const DEVICEID_TYPE deviceID, const bool makeMode = true);
    void Train(IComputationNetBuilder<ElemType>* netBuilder,
        IDataReader<ElemType>* trainSetDataReader,
        IDataReader<ElemType>* validationSetDataReader,
        const bool makeMode = true);

protected:
    std::vector<ComputationNodeBasePtr> & GetTrainCriterionNodes(ComputationNetwork& net);
    std::vector<ComputationNodeBasePtr> & GetEvalCriterionNodes(ComputationNetwork& net);

    void TrainOrAdaptModel(int startEpoch, ComputationNetwork& net,
                           ComputationNetwork& refNet,
                           ComputationNodeBasePtr refNode,
                           IDataReader<ElemType>* trainSetDataReader,
                           IDataReader<ElemType>* validationSetDataReader);

protected:
    // return true if precomputation is executed.
    bool PreCompute(ComputationNetwork& net,
                    IDataReader<ElemType>* trainSetDataReader,
                    std::vector<ComputationNodeBasePtr> & featureNodes,
                    std::vector<ComputationNodeBasePtr> & labelNodes,
                    std::map<std::wstring, Matrix<ElemType>*>* inputMatrices);

    // return a reasonable initial learning rate based on the initial mbsize
    double SearchForBestLearnRate(ComputationNetwork& net,
                                  ComputationNetwork& refNet,
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
                                  const double largestPrevLearnRatePerSample);

    void TrainOneMiniEpochAndReloadModel(ComputationNetwork& net,
                                         ComputationNetwork& refNet,
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
                                         std::string prefixMsg = "");

    size_t AdaptiveMinibatchSizing(ComputationNetwork& net,
                                   ComputationNetwork& refNet,
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
                                   const double learningRateAdjustmentFactor);

    // uses a small percentage of training data of minibatch to
    // speculatively train with various MB sizes; then picks the best
    size_t SearchForBestMinibatchSize(ComputationNetwork& net,
                                      ComputationNetwork& refNet,
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
                                      const size_t minMinibatchSize, const size_t maxMinibatchSize);

    // Attemps to compute the error signal for the whole utterance, which will
    // be fed to the neural network as features. Currently it is a workaround
    // for the two-forward-pass sequence and ctc training, which allows
    // processing more utterances at the same time. Only used in Kaldi2Reader.
    // TODO: move the two-forward-pass support out of the reader.
    void AttemptUtteranceDerivativeFeatures(ComputationNetwork& net,
                                            IDataReader<ElemType>* trainSetDataReader,
                                            const std::vector<ComputationNodeBasePtr> & featureNodes,
                                            std::map<std::wstring, Matrix<ElemType>*>* inputMatrices);

    size_t TrainOneEpoch(ComputationNetwork& net,
                         ComputationNetwork& refNet,
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
                         std::map<std::wstring, Matrix<ElemType>*>* inputMatrices,
                         const std::list<ComputationNodeBasePtr> & learnableNodes,
                         std::list<Matrix<ElemType>>& smoothedGradients,
                         /*out*/ double& epochCriterion,
                         /*out*/ std::vector<double>& epochEvalErrors,
                         /*out*/ size_t& totalSamplesSeen,
                         std::string prefixMsg = "");

    void LazyInitDistGradAgg(const std::list<ComputationNodeBasePtr>& learnableNodes, int numEvalNodes, int traceLevel);

    bool ModelAveragingProcessing(size_t nSamplesSinceLastSync, const std::list<ComputationNodeBasePtr>& learnableNodes, size_t& nProcessedFrames, 
                                  float& SecondsSinceLastSyncFinished, float& SecondsSpentOnSync);

    size_t ModelAveragingSync(int nSamplesSinceLastSync, const std::list<ComputationNodeBasePtr>& learnableNodes);

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
                               const bool needAveMultiplier);

protected:
    // UpdateWeights - update the weights in
    void UpdateWeights(const ComputationNodeBasePtr& node,
                       Matrix<ElemType>& smoothedGradient,
                       const double learnRatePerSample,
                       const double momentumPerSample,
                       const size_t actualMBSize,
                       const double L2RegWeight, const double L1RegWeight,
                       const bool needAveMultiplier) const;

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

    GradientsUpdateType GradUpdateType() const { return m_gradType.mType; }
    double GradientUpdateNoiseStd() const { return m_gradType.mGaussianNoiseInjectStd; }

public:

#define EPSILON 1e-5

    bool GradientCheck(ComputationNetwork& net,
                       const std::vector<ComputationNodeBasePtr> & criterionNodes,
                       const std::list<ComputationNodeBasePtr> & learnableNodes,
                       int npos);

protected:

    // learning rate per sample provided outside
    floatargvector m_learningRatesPerSample;

    // only true when the user specify LearningRatePerMB and the number of parallel utterances in Reader > 1
    bool m_needToNormalizeLRByParallUtterance;
    bool m_needToNormalizeMomentumByParallUtterance;

    intargvector m_mbSize;

    // the number of samples in each epoch (0 means, use all the samples in each epoch).
    size_t m_epochSize;
    size_t m_maxComputedEpochSize;

    // the total number of epochs to run.
    size_t m_maxEpochs;

    floatargvector m_momentumPerSample;
    bool m_gradientClippingWithTruncation;
    double m_clippingThresholdPerSample;

    wstring m_modelPath;
    wstring m_trainCriterionNodeName;
    wstring m_evalCriterionNodeName;

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
    size_t m_prevChosenMinibatchSize;
    bool m_autoAdjustMinibatch;
    size_t m_minibatchSearchCriterionErrorMargin;
    size_t m_minibatchSizeTuningFrequency;
    size_t m_minibatchSizeTuningMax;

    floatargvector m_dropoutRates;
    size_t m_maxTempMemSizeInSamplesForCNN;

    int m_traceLevel;

    size_t m_numPrevLearnRates;

    double m_minLearnRate;

    GradientUpdateInfo m_gradType;
    RMSPropInfo m_rpi;

    bool m_keepCheckPointFiles;

    int m_numMBsToShowResult;
    int m_numMBsToCUDAProfile;

    bool m_doGradientCheck;
    double m_gradientCheckSigDigit;

    bool m_doUnitTest;

    bool m_validateAfterModelReloading;

    bool m_useAllDataForPreComputedNode;

    // Parallel training
    ParallelizationMethod m_parallelizationMethod;
    IDistGradAggregator<ElemType>* m_distGradAgg;
    struct DistGradHeader* m_gradHeader;
    int m_numGradientBits;
    bool m_zeroThresholdFor1Bit;
    bool m_enableDistributedMBReading;
    int m_parallelizationStartEpochNum;

    // Parallel training related with MA 
    // decide how much information we want to show MA performance stats (seconds spend on sync, seconds since last sync etc.) ?  
    // 0: means no perfomance stats show
    // 1: means show stats every sync 
    // n>1: means show stats after every n sync
    int    m_iMASyncStatsTrace;
    size_t m_nFramesBetweenMASync;

    bool m_needAveMultiplier;
    double m_L2RegWeight;
    double m_L1RegWeight;

    //sequence trainning
    ElemType m_hsmoothingWeight;
    ElemType m_frameDropThresh;
    bool m_doreferencealign;
};

}}}
