// SGD.cpp -- implements SGD with all bells and whistles, parallelization, randomization, etc.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "SGD.h"
#include "NonlinearityNodes.h"          // for DropoutNode
#include "SpecialPurposeNodes.h"        // for SequenceWithSoftmaxNode
#include "DataReaderHelpers.h"
#include "MatrixQuantizerImpl.h"
#ifdef QUANTIZED_GRADIENT_AGGREGATION
#include "AllReduceDistGradAggregator.h"
#endif
#include "SimpleDistGradAggregator.h"
#include "ProgressTracing.h"

#include <map>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// =======================================================================
// class SGD
// =======================================================================

template SGD<float>::SGD(const ConfigParameters&);
template SGD<double>::SGD(const ConfigParameters&);
template SGD<float>::SGD(const ScriptableObjects::IConfigRecord&);
template SGD<double>::SGD(const ScriptableObjects::IConfigRecord&);

// -----------------------------------------------------------------------
// Train() -- perform a multi-epoch training end-to-end with checkpointing
// -----------------------------------------------------------------------

template <class ElemType>
void SGD<ElemType>::Train(function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn, DEVICEID_TYPE deviceId,
                          IDataReader* trainSetDataReader,
                          IDataReader* validationSetDataReader,
                          const bool makeMode)
{
    // determine which epoch to start with, including recovering a checkpoint if any and 'makeMode' enabled
    int startEpoch = DetermineStartEpoch(makeMode);
    if (startEpoch == m_maxEpochs)
    {
        LOGPRINTF(stderr, "No further training is necessary.\n");
        return;
    }

    wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
    bool loadNetworkFromCheckpoint = startEpoch >= 0;
    fprintf(stderr, "\n");
    if (loadNetworkFromCheckpoint)
        LOGPRINTF(stderr, "Starting from checkpoint. Loading network from '%ls'.\n", modelFileName.c_str());
    else
        LOGPRINTF(stderr, "Creating virgin network.\n");

    // create or load from checkpoint
    shared_ptr<ComputationNetwork> net = !loadNetworkFromCheckpoint ? createNetworkFn(deviceId) : ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName);

    // log the device we are computing on
    LOGPRINTF(stderr, "%s model with %d nodes", loadNetworkFromCheckpoint ? "Loaded" : "Created", (int)net->GetTotalNumberOfNodes());
    if (net->GetDeviceId() < 0)
        fprintf(stderr, " on CPU.\n");
    else
        fprintf(stderr, " on GPU %d.\n", (int) net->GetDeviceId());

    // TODO: BUGBUG: if not starting from checkpoint, need to synchronize initial model
    // strategy should be to run the initializer above on mpiRank==0, and then broadcast parameters.

    startEpoch = max(startEpoch, 0);
    m_needAdaptRegularization = false;

    // set tracing flags
    for (const auto& traceNodeName : m_traceNodeNamesReal)
        net->GetNodeFromName(traceNodeName)->EnableNodeTracing(/*isCategoryLabel=*/false);

    for (const auto& traceNodeName : m_traceNodeNamesCategory)
        net->GetNodeFromName(traceNodeName)->EnableNodeTracing(/*isCategoryLabel=*/true);

    TrainOrAdaptModel(startEpoch, net, loadNetworkFromCheckpoint, net, nullptr, trainSetDataReader, validationSetDataReader);
}

// -----------------------------------------------------------------------
// Adapt() -- similar to Train(), but for purpose of adapting
// -----------------------------------------------------------------------

template <class ElemType>
void SGD<ElemType>::Adapt(wstring origModelFileName, wstring refNodeName,
                          IDataReader* trainSetDataReader,
                          IDataReader* validationSetDataReader,
                          const DEVICEID_TYPE deviceId, const bool makeMode)
{
    int startEpoch = DetermineStartEpoch(makeMode);
    if (startEpoch == m_maxEpochs)
    {
        LOGPRINTF(stderr, "No further training is necessary.\n");
        return;
    }

    ComputationNetworkPtr net;
    bool networkLoadedFromCheckpoint = false;
    if (startEpoch >= 0)
    {
        wstring modelFileName = GetModelNameForEpoch(int(startEpoch) - 1);
        LOGPRINTF(stderr, "Starting from checkpoint. Loading network from '%ls'.\n", modelFileName.c_str());
        net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelFileName);
        networkLoadedFromCheckpoint = true;
    }
    else
    {
        LOGPRINTF(stderr, "Load Network From the original model file %ls.\n", origModelFileName.c_str());
        net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, origModelFileName);
    }

    startEpoch = max(startEpoch, 0);

    ComputationNetworkPtr refNet;
    m_needAdaptRegularization = m_adaptationRegType != AdaptationRegType::None && m_adaptationRegWeight > 0;
    if (m_needAdaptRegularization)
    {
        LOGPRINTF(stderr, "Load reference Network From the original model file %ls.\n", origModelFileName.c_str());
        refNet = ComputationNetwork::CreateFromFile<ElemType>(deviceId, origModelFileName);
    }

    ComputationNodeBasePtr refNode;
    if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL)
    {
        LOGPRINTF(stderr, "Checking refNodeName %ls.\n", origModelFileName.c_str());
        if (refNodeName == L"")
            InvalidArgument("refNodeName does not exist and is needed when adaptationRegType is KL.");
        refNode = refNet->GetNodeFromName(refNodeName);
    }

    TrainOrAdaptModel(startEpoch, net, networkLoadedFromCheckpoint, refNet, refNode, trainSetDataReader, validationSetDataReader);
}

// -----------------------------------------------------------------------
// TrainOrAdaptModel() -- main training end-to-end, given a start model
// -----------------------------------------------------------------------

static double MomentumPerMB(double momentumPerSample, size_t minibatchSize);

template <class ElemType>
void SGD<ElemType>::TrainOrAdaptModel(int startEpoch, ComputationNetworkPtr net,
                                      bool networkLoadedFromCheckpoint,
                                      ComputationNetworkPtr refNet,
                                      ComputationNodeBasePtr refNode,
                                      IDataReader* trainSetDataReader,
                                      IDataReader* validationSetDataReader)
{
    auto& featureNodes = net->FeatureNodes();
    auto& labelNodes = net->LabelNodes();
    auto& criterionNodes = GetTrainCriterionNodes(net);

    fprintf(stderr, "\n");
    LOGPRINTF(stderr, "Training criterion node(s):\n");
    for (const auto& node : criterionNodes)
    {
        LOGPRINTF(stderr, "\t%ls = %ls\n", node->NodeName().c_str(), node->OperationName().c_str());
    }

    // determine evaluationNodes from GetEvalCriterionNodes(), ensuring each criterion is only logged once
    std::vector<ComputationNodeBasePtr> evaluationNodes;
    {
        auto originalEvaluationNodes = GetEvalCriterionNodes(net);
        set<ComputationNodeBasePtr> criteriaLogged; // set to make sure we don't double-log criteria
        for (const auto& node : criterionNodes)
            criteriaLogged.insert(node);

        for (const auto& node : originalEvaluationNodes)
            if (criteriaLogged.insert(node).second)
                evaluationNodes.push_back(node);

        if (!evaluationNodes.empty())
        {
            fprintf(stderr, "\n");
            LOGPRINTF(stderr, "Evaluation criterion node(s):\n");
            fprintf(stderr, "\n");
            for (const auto& node : evaluationNodes)
            {
                LOGPRINTF(stderr, "\t%ls = %ls\n", node->NodeName().c_str(), node->OperationName().c_str());
            }
        }
    }

    std::vector<ComputationNodeBasePtr> additionalNodesToEvaluate;
    auto& outputNodes = net->OutputNodes();
    additionalNodesToEvaluate.insert(additionalNodesToEvaluate.end(), outputNodes.cbegin(), outputNodes.cend());

    auto preComputeNodesList = net->GetNodesRequiringPreComputation();
    additionalNodesToEvaluate.insert(additionalNodesToEvaluate.end(), preComputeNodesList.cbegin(), preComputeNodesList.cend());

    // allocate memory for forward and backward computation
    net->AllocateAllMatrices(evaluationNodes, additionalNodesToEvaluate, criterionNodes[0]);

    // get feature and label nodes into an array of matrices that will be passed to GetMinibatch()
    // TODO: instead, remember the nodes directly, to be able to handle both float and double nodes; current version will crash for mixed networks
    StreamMinibatchInputs* inputMatrices = new StreamMinibatchInputs();
    // TODO: ^^ change to shared_ptr or unique_ptr
    for (size_t pass = 0; pass < 2; pass++)
    {
        auto& nodes = (pass == 0) ? featureNodes : labelNodes;
        for (const auto & node : nodes)
            (*inputMatrices).AddInputMatrix(node->NodeName(), node->ValuePtr());
    }

    // get hmm file for sequence training
    bool isSequenceTrainingCriterion = (criterionNodes[0]->OperationName() == L"SequenceWithSoftmax");
    if (isSequenceTrainingCriterion)
    {
        // SequenceWithSoftmaxNode<ElemType>* node = static_cast<SequenceWithSoftmaxNode<ElemType>*>(criterionNodes[0]);
        auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNodes[0]);
        auto hmm = node->gethmm();
        trainSetDataReader->GetHmmData(hmm);
    }

    // used for KLD regularized adaptation. For all other adaptation techniques
    // use MEL to edit the model and using normal training algorithm
    // TODO: Should this be done in SGD::Adapt()?
    // TODO: Redo this leveraging that we now have shared_ptrs. It is probably even OK if both networks share feature nodes.
    // TODO: Then we can also share the MBLayout; which currently is copied by value.
    std::vector<ComputationNodeBasePtr> refFeatureNodes;
    if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
    {
        // replace input nodes in ref network by input nodes of the main network
        refFeatureNodes.resize(featureNodes.size());
        for (size_t i = 0; i < featureNodes.size(); i++)
        {
            // we need to keep this info to undo this later
            // TODO: After the change to shared_ptrs, this may no longer be necessary.
            refFeatureNodes[i] = refNet->GetNodeFromName(featureNodes[i]->NodeName());
            refNet->ChangeNode(featureNodes[i]->NodeName(), featureNodes[i]);
        }
        refNet->InvalidateCompiledNetwork(); // prepare to re-compile
        refNet->CompileNetwork();

        // allocate memory for forward computation
        refNet->AllocateAllMatrices({refNode}, {}, nullptr);
    }

    // initializing weights and gradient holder
    // only one criterion so far TODO: support multiple ones?
    auto& learnableNodes = net->LearnableParameterNodes(criterionNodes[0]);
    std::list<Matrix<ElemType>> smoothedGradients;

    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
    {
        ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
        smoothedGradients.push_back(Matrix<ElemType>(node->Value().GetNumRows(),
                                                     node->Value().GetNumCols(),
                                                     net->GetDeviceId()));
    }

    double epochCriterion, avgCriterion, prevCriterion, lrControlCriterion;
    lrControlCriterion = epochCriterion = avgCriterion = prevCriterion = std::numeric_limits<double>::infinity();
    size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

    std::vector<double> epochEvalErrors(evaluationNodes.size(), std::numeric_limits<double>::infinity());

    std::vector<wstring> evalNodeNames;
    for (size_t i = 0; i < evaluationNodes.size(); i++)
    {
        evalNodeNames.push_back(evaluationNodes[i]->NodeName());
    }

    size_t totalSamplesSeen = 0;
    double learnRatePerSample = 0.5f / m_mbSize[startEpoch];

    double learningRateAdjustmentFactor = 1.0f;
    vector<double> prevLearnRates;
    prevLearnRates.resize(m_numPrevLearnRates);
    for (int i = 0; i < m_numPrevLearnRates; i++)
    {
        prevLearnRates[i] = -1.0;
    }

    if (GetParallelizationMethod() == ParallelizationMethod::DataParallelSGD)
    {
        InitDistGradAgg(evaluationNodes.size(), m_traceLevel);
    }
    else if (GetParallelizationMethod() == ParallelizationMethod::ModelAveragingSGD)
    {
        InitModelAggregationHandler(m_syncStatsTrace);
    }
    

    // precompute mean and invStdDev nodes and save initial model
    // When no precompute, only save if we did not load the model from a 
    // checkpoint but instead built it from a network description
    if (PreCompute(net, trainSetDataReader, featureNodes, labelNodes, inputMatrices) || !networkLoadedFromCheckpoint)
    {
        // Synchronize all ranks before writing the model to ensure that
        // everyone is done loading the model
        if (m_mpi != nullptr)
        {
            m_mpi->WaitAll();
        }

        // In case of parallel training only the main node should we saving the model to prevent
        // the parallel training nodes from colliding to write the same file
        if ((m_mpi == nullptr) || m_mpi->IsMainNode())
            net->Save(GetModelNameForEpoch(int(startEpoch) - 1));
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
    double prevNormalizationTimeConstant = 0;
    double prevNormalizationBlendTimeConstant = 0;

    bool learnRateReduced = false;

    // pass user config on memory allocation for convolution operations to the Network
    ComputationNetwork::SetMaxTempMemSizeForCNN(net, criterionNodes[0], m_maxTempMemSizeInSamplesForCNN);
    if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode)
    {
        ComputationNetwork::SetMaxTempMemSizeForCNN(refNet, refNode, m_maxTempMemSizeInSamplesForCNN);
    }

    // likewise for sequence training parameters
    if (isSequenceTrainingCriterion)
    {
        ComputationNetwork::SetSeqParam<ElemType>(net, criterionNodes[0], m_hSmoothingWeight, m_frameDropThresh, m_doReferenceAlign,
                                                  m_seqGammarCalcAMF, m_seqGammarCalcLMF, m_seqGammarCalcWP, m_seqGammarCalcbMMIFactor, m_seqGammarCalcUsesMBR);
    }

    // --- MAIN EPOCH LOOP
    for (int i = startEpoch; i < (int) m_maxEpochs; i++) // TODO: why is this an int, and not a size_t?
    {
        // Synchronize all ranks before proceeding to ensure that
        // rank 0 has finished writing the previous model file
        if (m_mpi != nullptr)
        {
            m_mpi->WaitAll();
        }

        Timer timer;
        timer.Start();

        // set dropout rate for this epoch
        ComputationNetwork::SetDropoutRate<ElemType>(net, criterionNodes[0], m_dropoutRates[i], prevDropoutRate, dropOutSeed);
        ComputationNetwork::SetBatchNormalizationTimeConstants<ElemType>(net, criterionNodes[0], 
                                                                         m_batchNormalizationTimeConstant[i], prevNormalizationTimeConstant,
                                                                         m_batchNormalizationBlendTimeConstant[i], prevNormalizationBlendTimeConstant);
        
        // learning rate adjustment
        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || i < m_learningRatesParam.size())
        {
            // BUGBUG: GetNumParallelSequences() returns 1 under certain situations; it seems when restarting from checkpoint
            learnRatePerSample = GetLearningRatePerSample(i /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());
        }
        else if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
        {
            double largestPrevLearnRatePerSample = prevLearnRates[0];
            for (int j = 1; j < m_numPrevLearnRates; j++)
            {
                largestPrevLearnRatePerSample = max(largestPrevLearnRatePerSample, prevLearnRates[j]);
            }

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
            LOGPRINTF(stderr, "Learn Rate Per Sample for Epoch[%d] = %.8g is less than minLearnRate %.8g. Training complete.\n",
                      i + 1, learnRatePerSample, m_minLearnRate);
            if (m_autoLearnRateSearchType != LearningRateSearchAlgorithm::None)
            {
                // In case of parallel training only the main node should we saving the model to prevent
                // the parallel training nodes from colliding to write the same file
                if ((m_mpi == nullptr) || m_mpi->IsMainNode())
                    net->Save(m_modelPath);
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

        actualMinibatchSize = FixUpEffectiveMBSize(chosenMinibatchSize /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());

        double momentumPerSample = GetMomentumPerSample(i /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());
        // time constant = number of samples after which a contribution has been reduced to e^-1
        double momentumAsTimeConstant = momentumPerSample == 0.0 ? 0.0
                                                                 : momentumPerSample >= 1.0 ? 0.0
                                                                                            : -1.0 / log(momentumPerSample);
        fprintf(stderr, "\n");
        LOGPRINTF(stderr, "Starting Epoch %d: learning rate per sample = %f  effective momentum = %f  momentum as time constant = %.1f samples\n",
                  i + 1, learnRatePerSample, MomentumPerMB(momentumPerSample, actualMinibatchSize), momentumAsTimeConstant);

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

        if (m_useEvalCriterionControlLR && epochEvalErrors.size() > 0)
        {
            lrControlCriterion = epochEvalErrors[0];
        }
        else
        {
            lrControlCriterion = epochCriterion;
        }

        LOGPRINTF(stderr,
                  "Finished Epoch[%2d of %d]: [Training Set] TrainLossPerSample = %.8g; TotalSamplesSeen = %d; ",
                  i + 1, (int)m_maxEpochs, epochCriterion, (int)totalSamplesSeen);
        m_lastFinishedEpochTrainLoss = epochCriterion;
        if (epochEvalErrors.size() == 0) // no eval criterion, only train criterion itself
        {
            fprintf(stderr,
                    "AvgLearningRatePerSample = %.8g; EpochTime=%.6g\n",
                    learnRatePerSample, epochTime);
        }
        else if (epochEvalErrors.size() == 1)
        {
            fprintf(stderr,
                    "EvalErrPerSample = %.8g; AvgLearningRatePerSample = %.8g; EpochTime=%.6g\n",
                    epochEvalErrors[0], learnRatePerSample, epochTime);
        }
        else
        {
            fprintf(stderr, "EvalErrPerSample ");
            for (size_t j = 0; j < epochEvalErrors.size(); j++)
            {
                fprintf(stderr, "[%lu]=%.8g; ", j, epochEvalErrors[j]);
            }

            fprintf(stderr, "AvgLearningRatePerSample = %.8g; EpochTime=%.6g\n",
                    learnRatePerSample, epochTime);

            // TODO: why these extra log messages here and not for 1 eval criterion?
            LOGPRINTF(stderr, "Finished Epoch[%2d of %d]: Criterion Node [%ls] Per Sample = %.8g\n",
                      i + 1, (int) m_maxEpochs, criterionNodes[0]->NodeName().c_str(), epochCriterion);

            for (size_t j = 0; j < epochEvalErrors.size(); j++)
            {
                LOGPRINTF(stderr, "Finished Epoch[%2d of %d]: Evaluation Node [%ls] Per Sample = %.8g\n",
                          i + 1, (int) m_maxEpochs, evalNodeNames[j].c_str(), epochEvalErrors[j]);
            }
        }

        if (validationSetDataReader != trainSetDataReader && validationSetDataReader != nullptr)
        {
            SimpleEvaluator<ElemType> evalforvalidation(net, m_mpi);
            vector<wstring> cvSetTrainAndEvalNodes;
            if (criterionNodes.size() > 0)
            {
                cvSetTrainAndEvalNodes.push_back(criterionNodes[0]->NodeName());
            }
            if (evaluationNodes.size() > 0)
            {
                cvSetTrainAndEvalNodes.push_back(evaluationNodes[0]->NodeName());
            }

                // BUGBUG: We should not use the training MB size. The training MB size is constrained by both convergence and memory. Eval is only constrained by memory.
            vector<double> vScore = evalforvalidation.Evaluate(validationSetDataReader, cvSetTrainAndEvalNodes, m_mbSize[i]);
            LOGPRINTF(stderr, "Finished Epoch[%2d of %d]: [Validation Set] TrainLossPerSample = %.8g", i + 1, (int) m_maxEpochs, vScore[0]);
            if (vScore.size() > 1)
            {
                fprintf(stderr, "; EvalErrPerSample = %.8g", vScore[1]);
            }
            fprintf(stderr, "\n");

            if (m_useCVSetControlLRIfCVExists)
            {
                if (m_useEvalCriterionControlLR && vScore.size() > 1)
                {
                    lrControlCriterion = vScore[1];
                }
                else
                {
                    lrControlCriterion = vScore[0]; // the first one is the training criterion
                }
            }
        }

        // broadcast epochCriterion to make sure each processor will have the same learning rate schedule
        if ((GetParallelizationMethod() == ParallelizationMethod::ModelAveragingSGD) && (m_mpi->NumNodesInUse() > 1))
        {
            m_mpi->Bcast(&epochCriterion, 1, m_mpi->MainNodeRank());
            m_mpi->Bcast(&lrControlCriterion, 1, m_mpi->MainNodeRank());
        }

        bool loadedPrevModel = false;
        size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
        if (avgCriterion == std::numeric_limits<double>::infinity())
        {
            avgCriterion = lrControlCriterion;
        }
        else
        {
            avgCriterion = ((epochsSinceLastLearnRateAdjust - 1 - epochsNotCountedInAvgCriterion) *
                                avgCriterion +
                            lrControlCriterion) /
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
                    LOGPRINTF(stderr, "Loading previous model with best training-criterion value: %ls.\n", bestModelPath.c_str());
                    net->RereadPersistableParameters<ElemType>(bestModelPath);
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
                    {
                        learnRateReduced = true;
                    }
                    else
                    {
                        // In case of parallel training only the main node should we saving the model to prevent
                        // the parallel training nodes from colliding to write the same file
                        if ((m_mpi == nullptr) || m_mpi->IsMainNode())
                            net->Save(GetModelNameForEpoch(i, true));

                        LOGPRINTF(stderr, "Finished training and saved final model\n\n");
                        break;
                    }
                }

                if (learnRateReduced)
                {
                    learnRatePerSample *= m_learnRateDecreaseFactor;
                    LOGPRINTF(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                }
            }
            else
            {
                if (std::isnan(avgCriterion) ||
                    (prevCriterion - avgCriterion <= m_reduceLearnRateIfImproveLessThan * prevCriterion &&
                     prevCriterion != std::numeric_limits<double>::infinity()))
                {

                    learnRatePerSample *= m_learnRateDecreaseFactor;
                    LOGPRINTF(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                }
                else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan * prevCriterion &&
                         prevCriterion != std::numeric_limits<double>::infinity())
                {
                    learnRatePerSample *= m_learnRateIncreaseFactor;
                    LOGPRINTF(stderr, "learnRatePerSample increased to %.8g\n", learnRatePerSample);
                }
            }
        }
        else
        {
            if (std::isnan(avgCriterion))
                RuntimeError("The training criterion is not a number (NAN).");
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
        if (m_mpi != nullptr)
        {
            m_mpi->WaitAll();
        }

        // persist model and check-point info
        if ((m_mpi == nullptr) || m_mpi->IsMainNode())
        {
            SaveCheckPointInfo(i, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, chosenMinibatchSize);
            auto modelName = GetModelNameForEpoch(i);
            LOGPRINTF(stderr, "SGD: Saving checkpoint model '%ls'\n", modelName.c_str());
            net->Save(modelName);
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
            LOGPRINTF(stderr, "learnRate per sample is reduced to %.8g which is below 1e-12. stop training.\n",
                      learnRatePerSample);
        }
    }
    // --- END OF MAIN EPOCH LOOP

    // Synchronize all ranks before proceeding to ensure that
    // rank 0 has finished writing the model file
    if (m_mpi != nullptr)
    {
        m_mpi->WaitAll();
    }

    // progress tracing for compute cluster management
    ProgressTracing::TraceProgressPercentage(m_maxEpochs, 0.0, true);
    ProgressTracing::TraceTrainLoss(m_lastFinishedEpochTrainLoss);

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

// -----------------------------------------------------------------------
// TrainOneEpoch() -- train one epoch
// -----------------------------------------------------------------------

static string GeneratePaddedFloatOrExpFormat(int padSize, int precision, double value);

template <class ElemType>
size_t SGD<ElemType>::TrainOneEpoch(ComputationNetworkPtr net,
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
                                    StreamMinibatchInputs* inputMatrices, // TODO: why is this a pointer?
                                    const std::list<ComputationNodeBasePtr>& learnableNodes,
                                    std::list<Matrix<ElemType>>& smoothedGradients,
                                    /*out*/ double& epochCriterion,
                                    /*out*/ std::vector<double>& epochEvalErrors,
                                    /*in/out*/ size_t& totalSamplesSeen,
                                    std::string prefixMsg)
{
    ScopedNetworkOperationMode modeGuard(net, NetworkOperationMode::training);

    double totalTimeInMBs = 0; // use double since timer has sub-microsecond time resolution
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

    bool useGradientAggregation = ((GetParallelizationMethod() == ParallelizationMethod::DataParallelSGD) &&
                                   (epochNumber >= m_parallelizationStartEpochNum));
    bool useModelAveraging = ((GetParallelizationMethod() == ParallelizationMethod::ModelAveragingSGD) &&
                              (epochNumber >= m_parallelizationStartEpochNum));
    bool useParallelTrain = useGradientAggregation || useModelAveraging;

    // MA-related variables
    size_t nSamplesSinceLastModelSync = 0;
    if (useParallelTrain && m_pMASGDHelper)
    {
        m_pMASGDHelper->OnEpochStart(learnableNodes);
    }

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
        trainSetDataReader->StartDistributedMinibatchLoop(tunedMBSize, epochNumber, m_mpi->CurrentNodeRank(),
                                                          m_mpi->NumNodesInUse(), epochSize);
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

    // prepare for sub-minibatching
    // Sub-minibatching is used if a single minibatch is too large to fit into GPU RAM.
    DataReaderHelpers::SubminibatchDispatcher<ElemType> smbDispatcher;
    size_t numSubminibatchesNeeded = DataReaderHelpers::GetNumSubminibatchesNeeded<ElemType>(trainSetDataReader, m_maxSamplesInRAM, m_numSubminiBatches, tunedMBSize);

    // this is non-trivial, we need a manager object to handle this
    if (numSubminibatchesNeeded > 1)
        smbDispatcher.Init(net, learnableNodes, criterionNodes, evaluationNodes);

    // The following is a special feature only supported by the Kaldi2Reader for more efficient sequence training.
    // This attemps to compute the error signal for the whole utterance, which will
    // be fed to the neural network as features. Currently it is a workaround
    // for the two-forward-pass sequence and ctc training, which allows
    // processing more utterances at the same time.
    // TODO: move the two-forward-pass support out of the reader, make a first-class citizen.
    AttemptUtteranceDerivativeFeatures(net, trainSetDataReader, featureNodes, inputMatrices);

    fprintf(stderr, "\n");
    LOGPRINTF(stderr, "Starting minibatch loop");
    if (useGradientAggregation)
    {
        fprintf(stderr, ", DataParallelSGD training (MyRank = %d, NumNodes = %d, NumGradientBits = %d)",
                (int) m_mpi->CurrentNodeRank(), (int) m_mpi->NumNodesInUse(), (int) m_numGradientBits);

        if (m_bufferedAsyncGradientAggregation)
        {
            fprintf(stderr, ", BufferedAsyncGradientAggregation is ENABLED");
        }
    }

    if (useDistributedMBReading)
    {
        fprintf(stderr, ", distributed reading is ENABLED");
    }

    if (numSubminibatchesNeeded > 1)
    {
        if (m_maxSamplesInRAM < SIZE_MAX)
            fprintf(stderr, ", with maximum %d samples in RAM", (int)m_maxSamplesInRAM);
        else
            fprintf(stderr, ", with %d subminibatch", (int)numSubminibatchesNeeded);
    }
    fprintf(stderr, ".\n");

    Timer timer;
    timer.Start();

    // --- MAIN MINIBATCH LOOP

    bool noMoreSamplesToProcess = false;
    for (;;)
    {
        // get minibatch
        // TODO: is it guaranteed that the GPU is already completed at this point, is it safe to overwrite the buffers?
        size_t actualMBSize = 0;
        bool wasDataRead = DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(*trainSetDataReader, net, criterionNodes[0],
                                                                                useDistributedMBReading, useParallelTrain, *inputMatrices, actualMBSize, m_mpi);
        if (!wasDataRead && (!useDistributedMBReading || noMoreSamplesToProcess)) // in case of distributed reading, we do a few more loops until all ranks have completed
            break;                                                                // end of epoch

        // Note: If !wasDataRead then the data that GetMinibatchIntoNetwork() was supposed to full in are undefined.
        // Must not touch them.

        if (!wasDataRead)
            actualMBSize = 0; // (undefined if !wasDataRead)

        nSamplesSinceLastModelSync += actualMBSize;

        // node data was changed
        // TODO: move this to that function as well--just tired to pass everything as arguments
        // TODO: We should do this right after the GetMinibatch() call, since that's where these changed.
        //       Need to check whether that would cause unintended side effects.
        // TODO: original code did not call this for actualMBSize == 0
        ComputationNetwork::BumpEvalTimeStamp(featureNodes);
        ComputationNetwork::BumpEvalTimeStamp(labelNodes);

        if (actualMBSize > 0)
        {
            assert(wasDataRead);
#ifndef EVALDLL
            if (m_doGradientCheck && GradientCheck(net, criterionNodes, learnableNodes, 0) == false)
                LogicError("cannot pass gradient checker");
#endif
            // TODO: currently we only support one node for regularization
            if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode)
            {
                size_t actualMBSize2 = refNet->DetermineActualMBSizeFromFeatures();
                refNet->GetMBLayoutPtr()->CopyFrom(net->GetMBLayoutPtr()); // TODO: This is UNTESTED (before this was missing, seemingly inconsistently)

                if (actualMBSize2 != actualMBSize)
                    LogicError("TrainOneEpoch: refNet has different MB size than main net??");

                refNet->ForwardProp(refNode);
                Matrix<ElemType>::ScaleAndAdd((ElemType) m_adaptationRegWeight,
                                              dynamic_pointer_cast<ComputationNode<ElemType>>(refNode)->Value(),
                                              (ElemType)(1.0 - m_adaptationRegWeight),
                                              dynamic_pointer_cast<ComputationNode<ElemType>>(labelNodes[0])->Value());
            }

            // do forward and back propagation

            // We optionally break the minibatch into sub-minibatches.
            // This, when enabled, is used when a full minibatch does not fit into GPU RAM.
            size_t actualNumSubminibatches = numSubminibatchesNeeded <= 1 ? 1 : smbDispatcher.GetMinibatchIntoCache(*trainSetDataReader, *net, *inputMatrices, numSubminibatchesNeeded);
            for (size_t ismb = 0; ismb < actualNumSubminibatches; ismb++)
            {
                if (actualNumSubminibatches > 1)
                {
                    smbDispatcher.GetSubMinibatchToNet(ismb); // get sub-minibatch from full-size one
                    ComputationNetwork::BumpEvalTimeStamp(featureNodes);
                    ComputationNetwork::BumpEvalTimeStamp(labelNodes);
                }

                // ===========================================================
                // forward prop for evaluate eval nodes
                // ===========================================================

                // compute eval node first since when gradient is computed the forward function values
                // may be changed and need to be recomputed when gradient and function value share the same matrix
                net->ForwardProp(evaluationNodes); // the bulk of this evaluation is reused in ComputeGradient() below

                // ===========================================================
                // forward prop for training criterion
                // ===========================================================

                net->ForwardProp(criterionNodes[0]);

                // ===========================================================
                // backprop
                // ===========================================================

                if (learnRatePerSample > 0.01 * m_minLearnRate) // only compute gradient when learning rate is large enough
                    net->Backprop(criterionNodes[0]);

                // house-keeping for sub-minibatching
                if (actualNumSubminibatches > 1)
                    smbDispatcher.DoneWithCurrentSubMinibatch(ismb); // page state out
            }                                                        // end sub-minibatch loop
            if (actualNumSubminibatches > 1)
                smbDispatcher.DoneWithCurrentMinibatch();
        } // if (actualMBSize > 0)

        // for progress and statistics, we should only count frames that are not gaps
        size_t numSamplesWithLabel = wasDataRead ? net->GetNumSamplesWithLabel(actualMBSize) : 0;

        // Sum of actualMBSize across all nodes when using parallel training
        size_t aggregateNumSamples = actualMBSize;
        size_t aggregateNumSamplesWithLabel = numSamplesWithLabel;

        if (!useGradientAggregation)
        {
            // accumulate criterion values (objective, eval)
            if (actualMBSize != 0)
            {
                assert(wasDataRead);
                // criteria are in Value()(0,0), we accumulate into another 1x1 Matrix (to avoid having to pull the values off the GPU)
                Matrix<ElemType>::AddElementToElement(dynamic_pointer_cast<ComputationNode<ElemType>>(criterionNodes[0])->Value(),
                                                      0, 0, localEpochCriterion, 0, 0);
                for (size_t i = 0; i < evaluationNodes.size(); i++)
                {
                    Matrix<ElemType>::AddElementToElement(dynamic_pointer_cast<ComputationNode<ElemType>>(evaluationNodes[i])->Value(),
                                                          0, 0, localEpochEvalErrors, 0, i);
                }
            }
        }
        else
        {
            // distributed gradient aggregation
            if (learnParamsGradients.size() == 0)
            {
                learnParamsGradients.reserve(learnableNodes.size());
                for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
                {
                    ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
                    if (node->IsParameterUpdateRequired())
                    {
                        Matrix<ElemType>* currParamsGradient = &(node->Gradient());

                        // Sometimes, in parallel training, the current node may not get any samples to process
                        // In this case, the gradient matrix may not have been sized yet. If so, lets size it.
                        if (currParamsGradient->GetNumCols() == 0)
                        {
                            Matrix<ElemType>* currParamsValues = &(node->Value());
                            currParamsGradient->Resize(currParamsValues->GetNumRows(), currParamsValues->GetNumCols());
                        }

                        learnParamsGradients.push_back(currParamsGradient);
                    }
                }
            }

            // prepare the header
            m_gradHeader->numEvalNode = evaluationNodes.size();
            m_gradHeader->numSamples = actualMBSize;
            m_gradHeader->numSamplesWithLabel = numSamplesWithLabel;
            m_gradHeader->criterion = actualMBSize > 0 ? criterionNodes[0]->Get00Element() : 0.0;
            for (size_t i = 0; i < evaluationNodes.size(); i++)
                m_gradHeader->evalErrors[i] = actualMBSize > 0 ? evaluationNodes[i]->Get00Element() : 0.0;

            bool samplesProcessed = m_distGradAgg->AggregateGradients(learnParamsGradients, m_gradHeader, epochNumber);
            noMoreSamplesToProcess = !samplesProcessed;

            aggregateNumSamples = m_gradHeader->numSamples;
            aggregateNumSamplesWithLabel = m_gradHeader->numSamplesWithLabel;
            epochCriterion += m_gradHeader->criterion;
            for (size_t i = 0; i < epochEvalErrors.size(); i++)
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
                                  GetMomentumPerSample(epochNumber /*BUGBUG workaround:*/, net->GetMBLayoutPtr()->GetNumParallelSequences()), aggregateNumSamples,
                                  m_L2RegWeight, m_L1RegWeight,
                                  m_needAveMultiplier, m_useNesterovMomentum);
#ifdef _DEBUG
                    if (dynamic_pointer_cast<ComputationNode<ElemType>>(node)->Value().HasNan("TrainOneEpoch/UpdateWeights(): "))
                        LogicError("%ls %ls operation has NaNs in functionValues after parameter update.", node->NodeName().c_str(), node->OperationName().c_str());
#endif
                }
            }
        }

        // aggregation by model averaging
        if (useModelAveraging)
        {
            if (nSamplesSinceLastModelSync >= m_nFramesBetweenMASync)
            {
                bool synced = m_pMASGDHelper->OnArrivingAtSyncPoint(learnableNodes, smoothedGradients, nSamplesSinceLastModelSync);
                if (synced)
                {
                    nSamplesSinceLastModelSync = 0;
                }
            }
            // prepare break condition
            if (useDistributedMBReading)
            {
                noMoreSamplesToProcess = !wasDataRead;
            }
        }

        timer.Stop();
        numMBsRun++;

        totalTimeInMBs += timer.ElapsedSeconds();
        numSamplesLastMBs += (int)aggregateNumSamplesWithLabel;

        if (
#if 0       // output the first few to see if everything started right
            numMBsRun <= 3 ||
#endif
            numMBsRun % m_numMBsToShowResult == 0)
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

            double trainLossPerSample = (numSamplesLastMBs != 0) ? ((epochCriterion - epochCriterionLastMBs) / numSamplesLastMBs) : 0.0;
            bool wasProgressPrinted = false;

            if (epochNumber > 0 || (int) epochSize > 0)
            {
                // progress tracing for compute cluster management
                double mbProg = 0.0;
                int mbProgNumPrecision = 2;
                if (m_maxComputedEpochSize != 0)
                {
                    double numMBPerEpoch = (double) m_maxComputedEpochSize / (double) tunedMBSize;
                    mbProg = (double) numMBsRun / numMBPerEpoch;
                    mbProgNumPrecision = (int) ceil(log10(numMBPerEpoch / (double) m_numMBsToShowResult));
                    mbProgNumPrecision = max(mbProgNumPrecision - 2, 2);
                }
                wasProgressPrinted = ProgressTracing::TraceProgressPercentage(epochNumber, mbProg, false);

                // progress tracing for regular log
                string formatString = "%s Epoch[%2d of %d]-Minibatch[%4d-%4d, %2." + std::to_string(mbProgNumPrecision) + "f%%]: SamplesSeen = %d; TrainLossPerSample = " +
                                      GeneratePaddedFloatOrExpFormat(11, 8, trainLossPerSample) + "; ";
                SGDTrace(stderr, true, formatString.c_str(),
                         prefixMsg.c_str(), epochNumber + 1, m_maxEpochs, numMBsRun - m_numMBsToShowResult + 1,
                         numMBsRun, mbProg * 100, numSamplesLastMBs, trainLossPerSample);
            }
            else
            {
                wasProgressPrinted = ProgressTracing::TraceProgressPercentage(epochNumber, 0.0, false);

                string formatString = "%s Epoch[%2d of %d]-Minibatch[%4d-%4d]: SamplesSeen = %d; TrainLossPerSample = " +
                                      GeneratePaddedFloatOrExpFormat(11, 8, trainLossPerSample) + "; ";
                SGDTrace(stderr, true, formatString.c_str(),
                         prefixMsg.c_str(), epochNumber + 1, m_maxEpochs, numMBsRun - m_numMBsToShowResult + 1,
                         numMBsRun, numSamplesLastMBs, trainLossPerSample);
                m_maxComputedEpochSize = numMBsRun * numSamplesLastMBs / m_numMBsToShowResult;
            }

            double evalError = 0.0;
            for (size_t i = 0; i < epochEvalErrors.size(); i++)
            {
                evalError = (epochEvalErrors[i] - epochEvalErrorsLastMBs[i]) / numSamplesLastMBs;
                string formatString = "EvalErr[%lu]PerSample = " + GeneratePaddedFloatOrExpFormat(0, 8, evalError) + "; ";
                SGDTrace(stderr, false, formatString.c_str(), i, evalError);
            }

            string formatString = "TotalTime = " + GeneratePaddedFloatOrExpFormat(0, 4, totalTimeInMBs) + "s; SamplesPerSecond = %.1f\n";
            SGDTrace(stderr, false, formatString.c_str(), totalTimeInMBs, numSamplesLastMBs / totalTimeInMBs);

            // progress tracing for compute cluster management
            if (wasProgressPrinted)
            {
                ProgressTracing::TraceTrainLoss(trainLossPerSample);
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
                RuntimeError("The training criterion is not a number (NAN).");
            }
        }

        timer.Restart();
        totalEpochSamples += aggregateNumSamplesWithLabel;
        if (!useModelAveraging)
            totalSamplesSeen += aggregateNumSamplesWithLabel;

        // call DataEnd function
        // This signals something from SGD to the reader.
        // DataEnd does reader specific process if sentence ending is reached
        trainSetDataReader->DataEnd();

        // Attempts to compute the error signal for the whole utterance, which will
        // be fed to the neural network as features. Currently it is a workaround
        // for the two-forward-pass sequence and ctc training, which allows
        // processing more utterances at the same time. Only used in Kaldi2Reader.
        // TODO: move the two-forward-pass support out of the reader.
        AttemptUtteranceDerivativeFeatures(net, trainSetDataReader, featureNodes, inputMatrices);

        profiler.NextSample();
    }

    // --- END MAIN MINIBATCH LOOP

    if (useModelAveraging )
    {
        m_pMASGDHelper->OnEpochEnd(learnableNodes, smoothedGradients, nSamplesSinceLastModelSync);
        nSamplesSinceLastModelSync = 0;
    }

    // compute final criterion values
    if (useGradientAggregation)
    {
        // with parallelization, we have them in regular variables
        epochCriterion /= float(totalEpochSamples);
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
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
    if (useModelAveraging && (m_mpi->NumNodesInUse() > 1))
    {
        // 1. total epoch samples processed by all workers
        size_t totalEpochSamplesOfAllWorkers = totalEpochSamples;
        m_mpi->AllReduce(&totalEpochSamplesOfAllWorkers, 1);
        totalSamplesSeen += totalEpochSamplesOfAllWorkers;

        // 2. criterion and EvalErrors 
        localEpochCriterion *= (float)totalEpochSamples / totalEpochSamplesOfAllWorkers;
        localEpochEvalErrors *= (float)totalEpochSamples / totalEpochSamplesOfAllWorkers;

        epochCriterion = localEpochCriterion.Get00Element();
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
        {
            epochEvalErrors[i] = localEpochEvalErrors(0, i);
        }
        // merge epochCriterion and epochEvalErrors over nodes 
        m_mpi->AllReduce(&epochCriterion, 1);
        m_mpi->AllReduce(epochEvalErrors);

        // 3. modify return value 
        totalEpochSamples = totalEpochSamplesOfAllWorkers;
    }
    return totalEpochSamples;
}

// -----------------------------------------------------------------------
// subroutines and helpers follow below
// -----------------------------------------------------------------------

static double MomentumPerMB(double momentumPerSample, size_t minibatchSize)
{
    return pow(momentumPerSample, minibatchSize);
}

// Get{Train,Eval}CriterionNodes() return a reference that is, unfortunately, dependent on the network.
// So we hold those inside here. Not very nice. Also not thread-safe. This may go away once we fix sequence-to-sequence models properly.
// TODO: merge them into one.
static map<ComputationNetworkPtr, vector<ComputationNodeBasePtr>> tmpCriterionNodeSets;
// TODO: test this, then remove this comment

template <class ElemType>
std::vector<ComputationNodeBasePtr>& SGD<ElemType>::GetTrainCriterionNodes(ComputationNetworkPtr net)
{
    if (!m_trainCriterionNodeName.empty())
    {
        tmpCriterionNodeSets[net] = net->CriterionNodesFrom(m_trainCriterionNodeName);
        return tmpCriterionNodeSets[net];
    }
    else
        return net->FinalCriterionNodes();
}

template <class ElemType>
std::vector<ComputationNodeBasePtr>& SGD<ElemType>::GetEvalCriterionNodes(ComputationNetworkPtr net)
{
    if (!m_evalCriterionNodeName.empty())
    {
        tmpCriterionNodeSets[net] = net->CriterionNodesFrom(m_evalCriterionNodeName);
        return tmpCriterionNodeSets[net];
    }
    else
        return net->EvaluationNodes();
}

// execute PreComputeNodes
// Returns true if precomputation was executed.
template <class ElemType>
bool SGD<ElemType>::PreCompute(ComputationNetworkPtr net,
                               IDataReader* trainSetDataReader,
                               std::vector<ComputationNodeBasePtr>& featureNodes,
                               std::vector<ComputationNodeBasePtr>& labelNodes,
                               StreamMinibatchInputs* inputMatrices)
{
    std::list<ComputationNodeBasePtr> nodes = net->GetNodesRequiringPreComputation(); // this tests all HasComputed() flags

    if (nodes.size() == 0)
    {
        LOGPRINTF(stderr, "No PreCompute nodes found, skipping PreCompute step.\n");
        return false;
    }

    fprintf(stderr, "\n");
    LOGPRINTF(stderr, "Precomputing --> %lu PreCompute nodes found.\n\n", nodes.size());
    for (const auto & node : nodes)
    {
        LOGPRINTF(stderr, "\tNodeName: %ls\n", (node->NodeName()).c_str());
    }

    // compute
    ScopedNetworkOperationMode modeGuard(net, NetworkOperationMode::preComputing);

    // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , requestDataSize);
    // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , m_epochSize); // only based on one epoch
    // [1/12/2015 erw] to support large dataset, we usually partition whole dataset into several epoch's,
    // so we need to use all the data to do precomputing
    if (m_useAllDataForPreComputedNode) // using all the data
        trainSetDataReader->StartMinibatchLoop(m_mbSize[0], 0);
    else // using only one epoch. Note: One epoch is often enough for feature mean/stddev, but not for estimating priors.
        trainSetDataReader->StartMinibatchLoop(m_mbSize[0], 0, m_epochSize);
    net->StartEvaluateMinibatchLoop(nodes);

    // initialize
    for (auto & node : nodes)
        dynamic_pointer_cast<IPreComputeNode>(node)->MarkComputed(false /*begin accumulating*/);

    const size_t numIterationsBeforePrintingProgress = 100;
    size_t numItersSinceLastPrintOfProgress = 0;
    size_t actualMBSizeDummy;
    while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(*trainSetDataReader, net, nullptr, false, false, *inputMatrices, actualMBSizeDummy, m_mpi))
    {
        // TODO: move these into GetMinibatchIntoNetwork()  --but those are passed around; necessary? Can't we get them from 'net'?
        ComputationNetwork::BumpEvalTimeStamp(featureNodes);
        ComputationNetwork::BumpEvalTimeStamp(labelNodes);

        net->ForwardProp(nodes);

        numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);
    }

    // finalize
    for (auto & node : nodes)
    {
        dynamic_pointer_cast<IPreComputeNode>(node)->MarkComputed(true /*done accumulating*/);
    }

    fprintf(stderr, "\n");
    LOGPRINTF(stderr, "Precomputing --> Completed.\n\n");

    return true;
}

// return a reasonable initial learning rate based on the initial mbsize
template <class ElemType>
double SGD<ElemType>::SearchForBestLearnRate(ComputationNetworkPtr net,
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
    double learnRatePerSample = 1.0f / 8.0f / 0.618f / sqrt((double) m_mbSize[epochNumber]);

    if (learnRateInitialized && largestPrevLearnRatePerSample > 0)
    {
        // largestPrevLearnRatePerSample is per sample, first 0.618f is for compensation, second one is for safety
        learnRatePerSample = largestPrevLearnRatePerSample / 0.618f / 0.618f;
    }

    int baseModelEpoch = epochNumber - 1;
    net->RereadPersistableParameters<ElemType>(GetModelNameForEpoch(baseModelEpoch));

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
            ratio = pow(((double) numFramesToUseInSearch) / m_epochSize, 1.0f / 2);

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

    // grid search for the first m_numBestSearchEpoch  epochs
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

        bestLearnRatePerSample = (leftCriterion < rightCriterion) ? leftLearnRatePerSample : rightLearnRatePerSample;
    }

    LOGPRINTF(stderr, "Best Learn Rate Per Sample for Epoch[%d] = %.10g  baseCriterion=%.10g\n",
              epochNumber + 1, bestLearnRatePerSample, baseCriterion);

    return bestLearnRatePerSample;
}

// AdaptiveMinibatchSizing() -- choose the largest feasible minibatch size
// This is necessary for data-parallel operation. The aim is to minimize model updates, and hence bandwidth
// This implements
//    F. Seide, H. Fu, J. Droppo, G. Li, and D. Yu:
//    "On Parallelizability of Stochastic Gradient Descent for Speech DNNs"
//    In Proc. ICASSP 2014.
template <class ElemType>
size_t SGD<ElemType>::AdaptiveMinibatchSizing(ComputationNetworkPtr net,
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
                                              const double learningRateAdjustmentFactor)
{
    size_t minMinibatchSize = initialMinibatchSize;
    size_t chosenMinibatchSize = initialMinibatchSize;

    // do some pre-adjustment based on LR
    // Basically we assume that the LR for epoch 1 is safe for mbsize.
    // If LR control led to a smaller LR, then we can safely increase the lower bound of the MB size.
    double learningRateChangeSoFar = GetLearningRatePerSample(epochNumber /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences()) / GetLearningRatePerSample(0 /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequences());
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
        LOGPRINTF(stderr, "before epoch .2, previous minibatchSize %zd is "
                  "considered invalid -> resetting\n",
                m_prevChosenMinibatchSize);
        m_prevChosenMinibatchSize = 0;
    }

    // check if we need to skip
    if (m_prevChosenMinibatchSize != 0 &&
        (epochNumber + 1) > m_minibatchSizeTuningFrequency &&
        (epochNumber + 1) % m_minibatchSizeTuningFrequency != 0)
    {
        LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Search for a better minibatchSize "
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
            LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Limiting minMinibatchSize to "
                      "largest of previous minibatchSize = (%d / 2) or %d\n",
                      (int) m_prevChosenMinibatchSize, (int) minMinibatchSize);
            minMinibatchSize = max(minMinibatchSize, m_prevChosenMinibatchSize / 2);
        }

        size_t maxMinibatchSize = m_minibatchSizeTuningMax;

        // only grow at most 2 x compared to previous step
        if (m_prevChosenMinibatchSize != 0.0f)
        {
            assert(m_prevChosenMinibatchSize >= chosenMinibatchSize);

            LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Limiting maxMinibatchSize to "
                      "previous minibatchSize %zd*2\n",
                    m_prevChosenMinibatchSize);
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
template <class ElemType>
size_t SGD<ElemType>::SearchForBestMinibatchSize(ComputationNetworkPtr net,
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
    for (float trialMinibatchSizeFloat = (float) minMinibatchSize;
         trialMinibatchSizeFloat <= maxMinibatchSize;
         trialMinibatchSizeFloat *= minibatchSizeTuningFactor)
    {
        // round mbsize to something meaningful
        trialMinibatchSize = RoundToMultipleOf64(trialMinibatchSizeFloat);

        fprintf(stderr, "\n");
        LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Evaluating trial minibatchSize=%zd out of range %zd..%zd ...\n\n",
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
                                        isFirstIteration ? "BaseAdaptiveMinibatchSearch:" : "AdaptiveMinibatchSearch:");

        if (isFirstIteration)
        {
            // for the first iteration of the loop only, set baseCriterion
            // to the result we got from TrainOneMiniEpochAndReloadModel().
            baseCriterion = epochCriterion;
            lastTriedTrialMinibatchSize = trialMinibatchSize;
            lastTriedTrialEpochCriterion = baseCriterion;
            isFirstIteration = false;

            LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Computed BaseCriterion %.10g\n", baseCriterion);
        }
        else if (!std::isnan(epochCriterion) &&
                 (epochCriterion > (baseCriterion * (1.0 + (m_minibatchSearchCriterionErrorMargin / 100.0)))))
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
                LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Keep searching... "
                          "EpochCriterion = %.10g vs BaseCriterion = %.10g\n",
                          epochCriterion, baseCriterion);
            }
        }
    }
    LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Search successful!!! Chose new minibatchSize of %d. "
              "EpochCriterion = %.10g vs BaseCriterion = %.10g\n\n",
              (int) lastTriedTrialMinibatchSize, lastTriedTrialEpochCriterion, baseCriterion);

    return lastTriedTrialMinibatchSize;
}

// run training over a small subset of an epoch, for purpose of automatic LR and MB-size tuning
template <class ElemType>
void SGD<ElemType>::TrainOneMiniEpochAndReloadModel(ComputationNetworkPtr net,
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
                                                    std::string prefixMsg)
{
    TrainOneEpoch(net, refNet, refNode, epochNumber, epochSize,
                  trainSetDataReader, learnRatePerSample, minibatchSize, featureNodes,
                  labelNodes, criterionNodes, evaluationNodes,
                  inputMatrices, learnableNodes, smoothedGradients,
                  /*out*/ epochCriterion, /*out*/ epochEvalErrors, /*out*/ totalSamplesSeen,
                  prefixMsg);

    LOGPRINTF(stderr, "Finished Mini-Epoch For LearnRate Selection: TrainLossPerSample = %.8g;", epochCriterion);

    if (epochEvalErrors.size() == 1)
        LOGPRINTF(stderr, "EvalErrPerSample = %.8g; AvgLearningRatePerSample = %.8g\n", epochEvalErrors[0], learnRatePerSample);
    else
    {
        LOGPRINTF(stderr, "EvalErrPerSample ");
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
        {
            LOGPRINTF(stderr, "[%lu] = %.8g; ", i, epochEvalErrors[i]);
        }
        LOGPRINTF(stderr, "AvgLearningRatePerSample = %.8g\n", learnRatePerSample);
    }

    int baseModelEpoch = epochNumber - 1;
    net->RereadPersistableParameters<ElemType>(GetModelNameForEpoch(baseModelEpoch));

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

// Attemps to compute the error signal for the whole utterance, which will
// be fed to the neural network as features. Currently it is a workaround
// for the two-forward-pass sequence and ctc training, which allows
// processing more utterances at the same time. Only used in Kaldi2Reader.
// TODO: move the two-forward-pass support out of the reader.
template <class ElemType>
void SGD<ElemType>::AttemptUtteranceDerivativeFeatures(ComputationNetworkPtr net,
                                                       IDataReader* trainSetDataReader,
                                                       const std::vector<ComputationNodeBasePtr>& featureNodes,
                                                       StreamMinibatchInputs* inputMatrices)
{
    assert(trainSetDataReader != NULL);
    std::vector<std::vector<std::pair<wstring, size_t>>> uttInfo;
    auto pMBLayout = make_shared<MBLayout>();
    // TODO: use GetMinibatchIntoNetwork().
    while (trainSetDataReader->GetMinibatchCopy(uttInfo, *inputMatrices, pMBLayout))
    {
        ComputationNetwork::BumpEvalTimeStamp(featureNodes);

        auto& outputNodes = net->OutputNodes();
        if (outputNodes.empty())
            LogicError("no output node was found.");

        trainSetDataReader->CopyMBLayoutTo(net->GetMBLayoutPtr());
        net->ForwardProp(outputNodes[0]); // only evaluate the first output
        trainSetDataReader->SetNetOutput(uttInfo,
                                         dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[0])->Value(),
                                         pMBLayout);
    }
}

// helper for pretty printing
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

template <class ElemType>
int SGD<ElemType>::SGDTrace(FILE* __restrict __stream, bool isPrependTimestamp, const char* __restrict __format, ...)
{
    int result = 0;
    if (m_traceLevel > 0)
    {
        va_list args;
        va_start(args, __format);
        if (isPrependTimestamp)
        {
            PREPENDTS(__stream);
        }

        result = vfprintf(__stream, __format, args);
        va_end(args);
    }
    return result;
}

template <class ElemType>
void SGD<ElemType>::InitDistGradAgg(int numEvalNodes, int traceLevel)
{
    if (GetParallelizationMethod() == ParallelizationMethod::DataParallelSGD)
    {
        if (m_distGradAgg == nullptr)
        {
#ifdef QUANTIZED_GRADIENT_AGGREGATION
            m_distGradAgg = new AllReduceDistGradAggregator<ElemType>(m_mpi, m_numGradientBits, m_zeroThresholdFor1Bit, true /*useQuantizationForSelfStripe*/, m_bufferedAsyncGradientAggregation, traceLevel, m_syncStatsTrace);
#else
            if (m_numGradientBits != (8 * sizeof(ElemType)))
            {
                RuntimeError("Gradient quantization is unsupported in CNTK binaries built without quantized gradient aggregation support!");
            }

            m_distGradAgg = new SimpleDistGradAggregator<ElemType>(m_mpi, m_bufferedAsyncGradientAggregation, m_syncStatsTrace);
#endif // !QUANTIZED_GRADIENT_AGGREGATION
        }

        if (m_gradHeader == nullptr)
        {
            m_gradHeader = DistGradHeader::Create(numEvalNodes);
        }
    }
}

template <class ElemType>
void SGD<ElemType>::InitModelAggregationHandler(int traceLevel)
{
    if (GetParallelizationMethod() == ParallelizationMethod::ModelAveragingSGD)
    {
#ifndef BLOCKWISE_MODEL_UPDATE_FILTERING
        if (!m_pMASGDHelper)
        {
            m_pMASGDHelper = make_shared<BasicModelAveragingSGD<ElemType>>(m_mpi, traceLevel);
        }
#else

#endif 
    }
    
}
// public:
// UpdateWeightsS - static version of UpdateWeights()
// not static since it wants to access protected methods on the SGD object
template <class ElemType>
/*static*/ void SGD<ElemType>::UpdateWeightsS(const SGD<ElemType>* sgd, Matrix<ElemType>& functionValues,
                                              Matrix<ElemType>& gradientValues,
                                              Matrix<ElemType>& smoothedGradient,
                                              const double learnRatePerSample,
                                              const double momentumPerSample,
                                              size_t actualMBSize,
                                              const double L2RegWeight,
                                              const double L1RegWeight,
                                              const bool needAveMultiplier,
                                              const bool useNesterovMomentum)
{
    // we use simple linear (instead of log linear) scaling here
    const double momentum = MomentumPerMB(momentumPerSample, actualMBSize);
#if DUMPOUTPUT
    LOGPRINTF(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
              learnRatePerSample, momentum, actualMBSize);
    LOGPRINTF(stderr, "sgd->GradUpdateType()=%d, sgd->GradientUpdateNoiseStd()=%0.8f\n",
              sgd->GradUpdateType(), sgd->GradientUpdateNoiseStd());
    gradientValues.Print("Gradient Input");
    smoothedGradient.Print("Smoothed Gradient Input");
#endif

    // make actualMBSize is a valid value
    assert(actualMBSize > 0);

    // clipping gradients to prevent outliers
    sgd->ClipGradient(gradientValues, actualMBSize);

    GradientsUpdateType adpType = sgd->GradUpdateType();
    double noiseStd = sgd->GradientUpdateNoiseStd();
    Matrix<ElemType> sgdUpdateNoise((DEVICEID_TYPE) functionValues.GetDeviceId());
    if (noiseStd > 0)
    {
        // get the gradient structure since gradient is sparse
        sgdUpdateNoise.SetValue(gradientValues);

        // reset its value to random
        sgdUpdateNoise.SetGaussianRandomValue(0, (ElemType) noiseStd);
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
                                    (ElemType) learnRatePerSample, (ElemType) momentum, useNesterovMomentum);
    }
    else if (adpType == GradientsUpdateType::AdaGrad ||
             (adpType == GradientsUpdateType::RmsProp && gradientValues.GetMatrixType() == MatrixType::SPARSE) ||
             (adpType == GradientsUpdateType::FSAdaGrad && gradientValues.GetMatrixType() == MatrixType::SPARSE))
    {
        // rmsprop for sparse is not implemented yet, delegate it with adagrad

        double aveMultiplier = smoothedGradient.Adagrad(gradientValues, needAveMultiplier);
        Matrix<ElemType>::ScaleAndAdd((ElemType)(-learnRatePerSample / aveMultiplier), gradientValues, functionValues);
    }
    else if (adpType == GradientsUpdateType::FSAdaGrad)
    {
        smoothedGradient.FSAdagrad(actualMBSize, gradientValues, functionValues, (ElemType) learnRatePerSample, (ElemType) momentum);
    }
    else if (adpType == GradientsUpdateType::RmsProp)
    {
        double aveMultiplier = smoothedGradient.RmsProp(gradientValues, (ElemType) sgd->m_rpi.gamma,
                                                        (ElemType) sgd->m_rpi.inc, (ElemType) sgd->m_rpi.max,
                                                        (ElemType) sgd->m_rpi.dec, (ElemType) sgd->m_rpi.min, needAveMultiplier);
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
template <class ElemType>
void SGD<ElemType>::UpdateWeights(const ComputationNodeBasePtr& node,
                                  Matrix<ElemType>& smoothedGradient,
                                  const double learnRatePerSample,
                                  const double momentumPerSample,
                                  const size_t actualMBSize,
                                  const double L2RegWeight, const double L1RegWeight,
                                  const bool needAveMultiplier,
                                  const bool useNesterovMomentum) const
{
#if DUMPOUTPUT
    LOGPRINTF(stderr, "Update_%ls\n", node->NodeName().c_str());
#endif
    if (!node->IsParameterUpdateRequired())
        LogicError("UpdateWeights() called for a learnable ComputationNode which has m_learningRateMultiplier == 0!");

    double nodeDependentLearningRatePerSample = learnRatePerSample * node->GetLearningRateMultiplier();
    UpdateWeightsS(this, dynamic_pointer_cast<ComputationNode<ElemType>>(node)->Value(), dynamic_pointer_cast<ComputationNode<ElemType>>(node)->Gradient(),
                   smoothedGradient, nodeDependentLearningRatePerSample, momentumPerSample,
                   actualMBSize, L2RegWeight, L1RegWeight,
                   needAveMultiplier, m_useNesterovMomentum);
    node->BumpEvalTimeStamp();
}

template <class ElemType>
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
                gradient *= (ElemType) normFactor;
            }
        }
    }
}

template <class ElemType>
void SGD<ElemType>::SaveCheckPointInfo(const size_t epoch, const size_t totalSamplesSeen,
                                       const double learnRatePerSample,
                                       const std::list<Matrix<ElemType>>& smoothedGradients,
                                       const double prevCriterion,
                                       const size_t minibatchSize)
{
    // In case of parallel training only the main node should we saving the checkpoint to prevent
    // the parallel training nodes from colliding to write the same file
    if ((m_mpi == nullptr) || m_mpi->IsMainNode())
    {
        wstring checkPointFileName = GetCheckPointFileNameForEpoch(int(epoch));
        // Saving into temporary file and then renaming it to the checkPointFileName
        // This is a standard trick to avoid havign corrupted checkpoints files if process dies during writing
        wstring tempFileName = checkPointFileName + L".tmp";

        {
            File fstream(tempFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsWrite);
            fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BVersion"); 
            fstream << (size_t)CURRENT_CNTK_CHECKPOINT_VERSION; 
            fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EVersion");

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

template <class ElemType>
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
        LOGPRINTF(stderr, "Warning: checkpoint file is missing. learning parameters will be initialized from 0\n");
        return false;
    }

    File fstream(checkPointFileName,
                 FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);

    // version info 
    size_t ckpVersion = CNTK_CHECKPOINT_VERSION_1; // if no version info is found -> version 1
    if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
    {
        fstream >> ckpVersion; 
        fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
    }


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

template <class ElemType>
wstring SGD<ElemType>::GetCheckPointFileNameForEpoch(const int epoch)
{
    return GetModelNameForEpoch(epoch) + L".ckp";
}

template <class ElemType>
wstring SGD<ElemType>::GetModelNameForEpoch(const int epoch, bool bLastModel)
{
    int epoch1Base = epoch + 1;
    if (epoch1Base == m_maxEpochs || bLastModel)
    {
        return m_modelPath;
    }
    else
    {
        wstring w = msra::strfun::wstrprintf(L"%ls.%d", m_modelPath.c_str(), (int) epoch1Base);
        return w;
    }
}

// return -1 if nothing exists
template <class ElemType> // TODO: needed?
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
        LOGPRINTF(stderr, "Final model exists: %ls\n", GetModelNameForEpoch(firstEpoch - 1).c_str());

    return firstEpoch;
}

#define EPSILON 1e-5

// this probes the automatic gradient computation with random inputs
template <class ElemType>
bool SGD<ElemType>::GradientCheck(ComputationNetworkPtr net,
                                  const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                  const std::list<ComputationNodeBasePtr>& learnableNodes,
                                  int npos)
{
    ScopedNetworkOperationMode modeGuard(net, NetworkOperationMode::training);

    net->StartEvaluateMinibatchLoop(criterionNodes[npos]);

    vector<string> errMsgs; // TODO: These are created but actually not returned, only their count is checked.

    // gradient checking
    for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
    {
        ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
        char wstrtmp[2048];

        for (size_t itry = 0; itry < min((size_t) 50, node->Value().GetNumElements()); itry++)
        {
            // no support to sparse matrix yet
            int irow = (int)fmod(rand(), node->Value().GetNumRows() - 1);
            int icol = (int)fmod(rand(), node->Value().GetNumCols() - 1);
            irow = max(0, irow);
            icol = max(0, icol);

            fprintf(stderr, "\n");
            LOGPRINTF(stderr, "###### d%ls######\n", node->NodeName().c_str());

            double eOrg = node->Value()(irow, icol);
            node->Value().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

            node->BumpEvalTimeStamp();

            net->ForwardProp(criterionNodes[npos]);
            net->Backprop(criterionNodes[npos]);

            if (node->Gradient().GetMatrixType() == MatrixType::SPARSE)
            {
                break;
            }

            // double mbEvalCri =
            // criterionNode should be a scalar
            // TODO: why is this value not used?
            criterionNodes[npos]->Get00Element();
            double eGradErr = node->Gradient()(irow, icol);
            node->Gradient().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

            double ePos = eOrg + EPSILON;
            double eNeg = eOrg - EPSILON;

            node->Value()(irow, icol) = (ElemType) ePos;
            node->Value().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

            node->BumpEvalTimeStamp();
            net->ForwardProp(criterionNodes[npos]);
            // criterionNode should be a scalar

            double mbEvalCriPos = criterionNodes[npos]->Get00Element(); // TODO: make Get00Element() a function of ComputationNodeBase

            node->Value()(irow, icol) = (ElemType) eNeg;
            node->Value().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

            node->BumpEvalTimeStamp();
            net->ForwardProp(criterionNodes[npos]);

            // criterionNode should be a scalar
            double mbEvalCriNeg = criterionNodes[npos]->Get00Element();

            // back to its original parameter value
            node->Value()(irow, icol) = (ElemType) eOrg;
            node->Value().TransferToDeviceIfNotThere(net->GetDeviceId(), true);

            // check if they are consistent
            double eGradNum = ((mbEvalCriPos - mbEvalCriNeg) / (ePos - eNeg));
            double threshold = pow(10.0,
                                   max(0.0,
                                       ceil(log10(min(fabs(eGradErr),
                                                      fabs(eGradNum))))) -
                                       (int) m_gradientCheckSigDigit);
            double diff = fabs(eGradErr - eGradNum);
            bool wrong = (std::isnan(diff) || diff > threshold);
            if (wrong)
            {
                fprintf(stderr, "\n");
                LOGPRINTF(stderr, "d%ls Numeric gradient = %e, Error BP gradient = %e\n",
                          node->NodeName().c_str(), eGradNum, eGradErr);
                sprintf(wstrtmp, "\nd%ls Numeric gradient = %e, Error BP gradient = %e\n",
                        node->NodeName().c_str(), eGradNum, eGradErr);
                errMsgs.push_back(wstrtmp);
            }
        }
    }

    return errMsgs.empty();
}

template class SGD<float>;
template class SGD<double>;

// =======================================================================
// class SGDParams
// =======================================================================

static AdaptationRegType ParseAdaptationRegType(const wstring& s)
{
    if      (EqualCI(s, L"") || EqualCI(s, L"none"))    return AdaptationRegType::None;
    else if (EqualCI(s, L"kl") || EqualCI(s, L"klReg")) return AdaptationRegType::KL;
    else
        InvalidArgument("ParseAdaptationRegType: Invalid Adaptation Regularization Type. Valid values are (none | kl)");
}

static GradientsUpdateType ParseGradUpdateType(const wstring& s)
{
    if      (EqualCI(s, L"") || EqualCI(s, L"none")) return GradientsUpdateType::None;
    else if (EqualCI(s, L"adagrad"))                 return GradientsUpdateType::AdaGrad;
    else if (EqualCI(s, L"rmsProp"))                 return GradientsUpdateType::RmsProp;
    else if (EqualCI(s, L"fsAdagrad"))               return GradientsUpdateType::FSAdaGrad;
    // legacy, deprecated
    else if (EqualCI(s, L"normal") || EqualCI(s, L"simple")) return GradientsUpdateType::None;
    else InvalidArgument("ParseGradUpdateType: Invalid Gradient Updating Type. Valid values are (none | adagrad | rmsProp | fsAdagrad )");
}

static ParallelizationMethod ParseParallelizationMethod(const wstring& s)
{
    if      (EqualCI(s, L"") || EqualCI(s, L"none")) return ParallelizationMethod::None;
    else if (EqualCI(s, L"DataParallelSGD"))         return ParallelizationMethod::DataParallelSGD;
    else if (EqualCI(s, L"ModelAveragingSGD"))       return ParallelizationMethod::ModelAveragingSGD;
    else InvalidArgument("ParseParallelizationMethod: Invalid Parallelization Method. Valid values are (none | dataParallelSGD | modelAveragingSGD)");
}

static LearningRateSearchAlgorithm ParseLearningRateSearchType(const wstring& s)
{
    if      (EqualCI(s, L"false") || EqualCI(s, L"none")) return LearningRateSearchAlgorithm::None;
    else if (EqualCI(s, L"searchBeforeEpoch"))            return LearningRateSearchAlgorithm::SearchBeforeEpoch;
    else if (EqualCI(s, L"adjustAfterEpoch"))             return LearningRateSearchAlgorithm::AdjustAfterEpoch;
    // legacy, deprecated
    else if (EqualCI(s, L"beforeEpoch") || EqualCI(s, L"before")) return LearningRateSearchAlgorithm::SearchBeforeEpoch;
    else if (EqualCI(s, L"afterEpoch")  || EqualCI(s, L"after"))  return LearningRateSearchAlgorithm::AdjustAfterEpoch;
    else InvalidArgument("autoAdjustLR: Invalid learning rate search type. Valid values are (none | searchBeforeEpoch | adjustAfterEpoch)");
}

template <class ConfigRecordType>
SGDParams::SGDParams(const ConfigRecordType& configSGD, size_t sizeofElemType)
{
    floatargvector learningRatesPerMB = configSGD(L"learningRatesPerMB", ConfigRecordType::Array(floatargvector()));

    floatargvector learningRatesPerSample = configSGD(L"learningRatesPerSample", ConfigRecordType::Array(floatargvector()));

    string executionEngineValue = configSGD(L"executionEngine", "synchronous");

    // AutoAdjust Parameters
    const ConfigRecordType& configAALR(configSGD(L"AutoAdjust", ConfigRecordType::Record()));
    m_autoLearnRateSearchType = ParseLearningRateSearchType(configAALR(L"autoAdjustLR", L"None"));
    m_reduceLearnRateIfImproveLessThan = configAALR(L"reduceLearnRateIfImproveLessThan", 0.0);
    m_continueReduce = configAALR(L"continueReduce", false);
    m_learnRateAdjustInterval = configAALR(L"learnRateAdjustInterval", (size_t) 1);
    m_learnRateAdjustInterval = max((size_t) 1, m_learnRateAdjustInterval); // minimum interval is 1 epoch
    m_learnRateDecreaseFactor = configAALR(L"learnRateDecreaseFactor", 0.618);
    m_increaseLearnRateIfImproveMoreThan = configAALR(L"increaseLearnRateIfImproveMoreThan", numeric_limits<double>::infinity());
    m_learnRateIncreaseFactor = configAALR(L"learnRateIncreaseFactor", 1.382);

    // AutoAdjust Auto Adjust Minibatch Parameters
    m_autoAdjustMinibatch = configAALR(L"autoAdjustMinibatch", false);
    m_minibatchSizeTuningFrequency = configAALR(L"minibatchSizeTuningFrequency", (size_t) 1);
    m_minibatchSizeTuningMax = configAALR(L"minibatchSizeTuningMax", (size_t) 1048576);
    m_minibatchSearchCriterionErrorMargin = configAALR(L"minibatchSearchCriterionErrorMargin", (size_t) 1);

    // the number of minibatches used to search
    // the learning rate. Its typically set to 10-20% of
    // the total minibatches in an epoch.
    m_numMiniBatch4LRSearch = configAALR(L"numMiniBatch4LRSearch", ConfigRecordType::Array(intargvector(vector<int>{500})));

    m_numPrevLearnRates = configAALR(L"numPrevLearnRates", (size_t) 5);
    m_numBestSearchEpoch = configAALR(L"numBestSearchEpoch", (size_t) 1);
    m_loadBestModel = configAALR(L"loadBestModel", true);
    m_useCVSetControlLRIfCVExists = configAALR(L"UseCVSetControlLRIfCVExists", true);
    m_useEvalCriterionControlLR = configAALR(L"UseEvalCriterionControlLR", false);

    // TODO: mbSize and truncated should be specified differently for truncated BPTT:
    //       mbSize = total number of samples after which a model update should happen
    //       truncated = truncation length
    m_mbSize = configSGD(L"minibatchSize", ConfigRecordType::Array(intargvector(vector<int>{256})));
    m_truncated = configSGD(L"truncated", false);
    m_maxSamplesInRAM = configSGD(L"maxSamplesInRAM", (size_t) SIZE_MAX);
    m_numSubminiBatches = configSGD(L"numSubminibatches", (size_t) 1);

    // the number of samples in each epoch (0 means, use all the samples in each epoch).
    m_epochSize = configSGD(L"epochSize", (size_t) 0);
    // the number of samples in each epoch (0 means, use all the samples in each epoch).
    if (m_epochSize == 0)
        m_epochSize = requestDataSize;
    m_maxComputedEpochSize = m_epochSize;

    // the total number of epochs to run.
    m_maxEpochs = configSGD(L"maxEpochs");

    // Note: Momentum is best specified as a MB-size agnostic fashion.
    // Because momentum per sample is a number very close to 1, it is more handy to use a logarithmic specification.
    // We use 'momentumAsTimeConstant' to specify the time constant of the low-pass filter that momentum really is.
    // To convert a typical per-MB momentum value of 'm' used with a MB size of 'N', use momentumAsTimeConstant = -N/ln(m).
    // For the common configuration of momentum 0.9 at MB size of 256, that is momentumAsTimeConstant = 2429.8.
    floatargvector momentumPerMB = configSGD(L"momentumPerMB", ConfigRecordType::Array(floatargvector()));
    floatargvector momentumPerSample = configSGD(L"momentumPerSample", ConfigRecordType::Array(floatargvector()));
    floatargvector momentumAsTimeConstant = configSGD(L"momentumAsTimeConstant", ConfigRecordType::Array(floatargvector()));
    bool useNesterovMomentum = configSGD(L"useNAG", false);

    m_maxTempMemSizeInSamplesForCNN = configSGD(L"maxTempMemSizeInSamplesForCNN", (size_t) 0);

    m_traceLevel = configSGD(L"traceLevel", (int) 0);
    m_numMBsToShowResult = configSGD(L"numMBsToShowResult", (size_t) 10);
    m_numMBsToCUDAProfile = configSGD(L"numMBsToCUDAProfile", (size_t) 0);

    m_gradientClippingWithTruncation = configSGD(L"gradientClippingWithTruncation", true);
    m_clippingThresholdPerSample = configSGD(L"clippingThresholdPerSample", numeric_limits<double>::infinity());

    // sequence-training parameters
    m_hSmoothingWeight = configSGD(L"hSmoothingWeight", 0.95);
    m_frameDropThresh = configSGD(L"frameDropThresh", 1e-10);
    m_doReferenceAlign = configSGD(L"doReferenceAlign", false);
    m_seqGammarCalcUsesMBR = configSGD(L"seqGammarUsesMBR", false);
    m_seqGammarCalcAMF = configSGD(L"seqGammarAMF", 14.0);
    m_seqGammarCalcLMF = configSGD(L"seqGammarLMF", 14.0);
    m_seqGammarCalcbMMIFactor = configSGD(L"seqGammarBMMIFactor", 0.0);
    m_seqGammarCalcWP = configSGD(L"seqGammarWordPen", 0.0);

    m_dropoutRates = configSGD(L"dropoutRate", ConfigRecordType::Array(doubleargvector(vector<double>{0.0})));
    m_batchNormalizationTimeConstant = configSGD(L"batchNormalizationTimeConstant", ConfigRecordType::Array(doubleargvector(vector<double>{0})));
    m_batchNormalizationBlendTimeConstant = configSGD(L"batchNormalizationBlendTimeConstant", ConfigRecordType::Array(doubleargvector(vector<double>{0})));

    GradientsUpdateType gradUpdateType = ParseGradUpdateType(configSGD(L"gradUpdateType", L"None"));
    double gaussianNoiseInjecStd = configSGD(L"gaussianNoiseInjectStd", 0.0);
    m_gradType.mType = gradUpdateType;
    m_gradType.mGaussianNoiseInjectStd = (float) gaussianNoiseInjecStd;

    // extract RMSProp parameters from config, if they exist. Default to reasonable values.
    m_rpi.dec = configSGD(L"rms_wgt_dec", 0.75);
    m_rpi.inc = configSGD(L"rms_wgt_inc", 1.2);
    m_rpi.min = configSGD(L"rms_wgt_min", 0.1);
    m_rpi.max = configSGD(L"rms_wgt_max", 10.0);
    m_rpi.gamma = configSGD(L"rms_gamma", 0.99);

    m_needAveMultiplier = configSGD(L"normWithAveMultiplier", true);
    m_L2RegWeight = configSGD(L"L2RegWeight", 0.0);
    m_L1RegWeight = configSGD(L"L1RegWeight", 0.0);

    // for backward support. future setup should use gradUpdateType=AdaGrad, instead of
    // useAdagrad=true
    bool useAdagrad = configSGD(L"useAdagrad", false);
    if (useAdagrad)
    {
        gradUpdateType = GradientsUpdateType::AdaGrad;
        m_gradType.mType = gradUpdateType;
    }

    m_adaptationRegType = ParseAdaptationRegType(configSGD(L"adaptationRegType", L"None"));
    m_adaptationRegWeight = configSGD(L"adaptationRegWeight", 0.0);

    // gradient check setup
    m_doGradientCheck = configSGD(L"gradientcheck", false);
    m_gradientCheckSigDigit = configSGD(L"sigFigs", 6.0); // TODO: why is this a double?

    if (m_doGradientCheck && sizeofElemType != sizeof(double))
    {
        LogicError("Gradient check needs to use precision = 'double'.");
    }

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

    if (learningRatesPerSample.size() > 0)
    {
        m_learningRatesParam = learningRatesPerSample;
        m_learningRatesSpecifiedForMBSize = intargvector(L"1");
    }
    else if (learningRatesPerMB.size() > 0) // this actually means per specified minibatch size
    {
        m_learningRatesParam = learningRatesPerMB;
        m_learningRatesSpecifiedForMBSize = m_mbSize;
    }

    if ((int) (momentumPerSample.size() > 0) + (int) (momentumPerMB.size() > 0) + (int) (momentumAsTimeConstant.size() > 0) > 1)
    {
        InvalidArgument("You specified more than one of momentumPerSample, momentumPerMB, and momentumAsTimeConstant. Please only specify one.");
    }

    if (momentumPerSample.size() > 0) // note: noone would ever use this; use momentumAsTimeConstant instead
    {
        m_momentumParam = momentumPerSample;
        m_momentumSpecifiedForMBSize = intargvector(L"1");
    }
    else if (momentumAsTimeConstant.size() > 0)
    {
        vector<float> momentumPerSampleVec;
        for (int i = 0; i < momentumAsTimeConstant.size(); i++)
        {
            double momTC = momentumAsTimeConstant[i];
            double momPS = momTC == 0.0 ? 0 : exp(-1.0 / momTC);
            momentumPerSampleVec.push_back((float) momPS);
        }
        m_momentumParam = momentumPerSampleVec;
        m_momentumSpecifiedForMBSize = intargvector(L"1");
    }
    else if (momentumPerMB.size() > 0)
    {
        m_momentumParam = momentumPerMB;
        m_momentumSpecifiedForMBSize = m_mbSize;
    }
    else // default: momentumPerMB = 0.9 per MB
    {
        m_momentumParam = floatargvector(L"0.9");
        m_momentumSpecifiedForMBSize = m_mbSize;
    }
    m_useNesterovMomentum = useNesterovMomentum;

    for (int i = 0; i < m_momentumParam.size(); i++)
    {
        if (m_momentumParam[i] >= 1.0 || m_momentumParam[i] < 0.0)
        {
            InvalidArgument("Momentum parameter must be in [0, 1).");
        }
    }

    if (m_learnRateDecreaseFactor > 1 || m_learnRateIncreaseFactor < 1)
    {
        InvalidArgument("learnRateIncreaseFactor must be >= 1 and learnRateDecreaseFactor must be <= 1.");
    }

    for (size_t i = 0; i < m_dropoutRates.size(); i++)
    {
        if (m_dropoutRates[i] >= 1 || m_dropoutRates[i] < 0)
        {
            InvalidArgument("dropoutRate must be >= 0 and < 1.");
        }
    }

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
    m_bufferedAsyncGradientAggregation = false;
    m_enableDistributedMBReading = false;
    m_parallelizationStartEpochNum = 0;
    m_nFramesBetweenMASync = 40000; // default 40k frames

    if (configSGD.Exists(L"ParallelTrain"))
    {
        const ConfigRecordType& configParallelTrain(configSGD(L"ParallelTrain", ConfigRecordType::Record()));
        m_parallelizationMethod = ParseParallelizationMethod(configParallelTrain(L"parallelizationMethod", L"none"));
        m_parallelizationStartEpochNum = configParallelTrain(L"parallelizationStartEpoch", (int) 1) - 1; // Epoch numbers internally are 0 based
        m_enableDistributedMBReading = configParallelTrain(L"distributedMBReading", false);
        m_syncStatsTrace = configParallelTrain(L"syncPerfStats", (int) 0);

        if (configParallelTrain.Exists(L"DataParallelSGD"))
        {
            const ConfigRecordType& configDataParallelSGD(configParallelTrain(L"DataParallelSGD", ConfigRecordType::Record()));
            size_t defaultGradientBits = 8 * sizeofElemType;
            m_numGradientBits = configDataParallelSGD(L"gradientBits", defaultGradientBits);
            m_zeroThresholdFor1Bit = configDataParallelSGD(L"useZeroThresholdFor1BitQuantization", true);
            m_bufferedAsyncGradientAggregation = configDataParallelSGD(L"useBufferedAsyncGradientAggregation", false);
            if ((m_numGradientBits < 1) || (m_numGradientBits > (8 * sizeofElemType)))
            {
                InvalidArgument("gradientBits must be in the range [1, 32] when using precision=float and in range [1, 64] when using precision=double!");
            }
        }

        if (configParallelTrain.Exists(L"ModelAveragingSGD"))
        {
            const ConfigRecordType& configMASGD(configParallelTrain(L"ModelAveragingSGD", ConfigRecordType::Record()));
            m_nFramesBetweenMASync = configMASGD(L"syncFrequencyInFrames", (size_t) 40000);
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

SGDParams::SGDParams(const ScriptableObjects::IConfigRecordPtr configp)
    : SGDParams(*configp, GetSizeOfPrecision(configp))
{
}

// register SGD<> with the ScriptableObject system
ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<SGD<float>, SGD<double>> registerSGDOptimizer(L"SGDOptimizer");

}}}
