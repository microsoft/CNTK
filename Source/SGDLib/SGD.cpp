// SGD.cpp -- implements SGD with all bells and whistles, parallelization, randomization, etc.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "SGD.h"
#include "NonlinearityNodes.h"          // for DropoutNode
#include "SpecialPurposeNodes.h"        // for SequenceWithSoftmaxNode
#include "DataReaderHelpers.h"
#include "MatrixQuantizerImpl.h"

#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
//static inline bool operator==(const std::pair<double,size_t>& a, double b) { assert(b==0); return a.first == b; }
// ^^ workaround until this line in AggregateGradientsImpl() gets updated: assert(headerCPU->evalErrors[i] == 0);
#include "AllReduceDistGradAggregator.h"
#include "BlockMomentumSGD.h"
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
    net->EnableNodeTracing(m_traceNodeNamesReal, m_traceNodeNamesCategory, m_traceNodeNamesSparse);

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
    let& criterionNodes = GetTrainCriterionNodes(net);

    fprintf(stderr, "\n");
    LOGPRINTF(stderr, "Training criterion node(s):\n");
    for (const auto& node : criterionNodes)
    {
        LOGPRINTF(stderr, "\t%ls = %ls\n", node->NodeName().c_str(), node->OperationName().c_str());
    }
    if (criterionNodes.empty())
    {
        LOGPRINTF(stderr, "\t(none)\n");
        InvalidArgument("TrainOrAdaptModel: No criterion node was specified.");
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
    let& featureNodes = net->FeatureNodes();
    let& labelNodes = net->LabelNodes();
    // BUGBUG: ^^ should not get all feature/label nodes, but only the ones referenced in a criterion
    for (size_t pass = 0; pass < 2; pass++)
    {
        auto& nodes = (pass == 0) ? featureNodes : labelNodes;
        for (const auto & node : nodes)
            inputMatrices->AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());
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
    std::vector<ComputationNodeBasePtr> refFeatureNodes; // we keep the original network's features here
    if (m_needAdaptRegularization && m_adaptationRegType == AdaptationRegType::KL && refNode != nullptr)
    {
        refNet->InvalidateCompiledNetwork(); // prepare to re-compile
        // replace input nodes in ref network by input nodes of the main network
        refFeatureNodes.resize(featureNodes.size());
        for (size_t i = 0; i < featureNodes.size(); i++)
        {
            // we need to keep this info to undo this later
            // TODO: After the change to shared_ptrs, this may no longer be necessary.
            refFeatureNodes[i] = refNet->GetNodeFromName(featureNodes[i]->NodeName()); // remember so that we can restore them later
            refNet->ReplaceNode(featureNodes[i]->NodeName(), featureNodes[i]);
        }
        //const_cast<MBLayoutPtr&>(refNet->GetMBLayoutPtrOfNetwork()) = net->GetMBLayoutPtrOfNetwork(); // WORKAROUND
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

    double avgCriterion, lrControlCriterion;
    lrControlCriterion = avgCriterion = numeric_limits<double>::infinity();
    size_t epochsNotCountedInAvgCriterion = startEpoch % m_learnRateAdjustInterval;

    std::vector<wstring> evalNodeNames;
    for (size_t i = 0; i < evaluationNodes.size(); i++)
        evalNodeNames.push_back(evaluationNodes[i]->NodeName());

    double learnRatePerSample = 0.5f / m_mbSize[startEpoch];

    double learningRateAdjustmentFactor = 1.0f;
    vector<double> prevLearnRates;
    prevLearnRates.resize(m_numPrevLearnRates);
    for (int i = 0; i < m_numPrevLearnRates; i++)
    {
        prevLearnRates[i] = -1.0;
    }

    if (GetParallelizationMethod() == ParallelizationMethod::dataParallelSGD)
    {
        InitDistGradAgg(evaluationNodes.size(), m_traceLevel);
    }
    else if (GetParallelizationMethod() == ParallelizationMethod::modelAveragingSGD || 
             GetParallelizationMethod() == ParallelizationMethod::blockMomentumSGD)
    {
        InitModelAggregationHandler(m_syncStatsTrace, net->GetDeviceId());
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

    size_t totalTrainingSamplesSeen = 0; // aggregated over all epochs, for logging purposes only

    bool learnRateInitialized = false;
    double prevCriterion = numeric_limits<double>::infinity();
    if (startEpoch > 0)
    {
        learnRateInitialized = TryLoadCheckPointInfo(startEpoch - 1,
                                                     /*out*/ totalTrainingSamplesSeen,
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
        // We use the same seed across workers until parallel training kicks in to ensure that the workers have identical models
        size_t parallelWorkerIdx = ((m_mpi == nullptr) || !UsingParallelTrain(i)) ? 0 : m_mpi->CurrentNodeRank();
        size_t dropoutRandSeedBase = (parallelWorkerIdx * m_maxEpochs) + i;
        ComputationNetwork::SetDropoutRate<ElemType>(net, criterionNodes[0], m_dropoutRates[i], prevDropoutRate, dropoutRandSeedBase);
        ComputationNetwork::SetBatchNormalizationTimeConstants<ElemType>(net, criterionNodes[0], 
                                                                         m_batchNormalizationTimeConstant[i], prevNormalizationTimeConstant,
                                                                         m_batchNormalizationBlendTimeConstant[i], prevNormalizationBlendTimeConstant);
        
        // learning rate adjustment
        if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::None || i < m_learningRatesParam.size())
        {
            // BUGBUG: GetNumParallelSequences() returns 1 under certain situations; it seems when restarting from checkpoint
            learnRatePerSample = GetLearningRatePerSample(i /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequencesForFixingBPTTMode());
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

        actualMinibatchSize = FixUpEffectiveMBSize(chosenMinibatchSize /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequencesForFixingBPTTMode());

        double momentumPerSample = GetMomentumPerSample(i /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequencesForFixingBPTTMode());
        // time constant = number of samples after which a contribution has been reduced to e^-1
        double momentumAsTimeConstant = momentumPerSample == 0.0 ? 0.0
                                                                 : momentumPerSample >= 1.0 ? 0.0
                                                                                            : -1.0 / log(momentumPerSample);
        fprintf(stderr, "\n");
        LOGPRINTF(stderr, "Starting Epoch %d: learning rate per sample = %f  effective momentum = %f  momentum as time constant = %.1f samples\n",
                  i + 1, learnRatePerSample, MomentumPerMB(momentumPerSample, actualMinibatchSize), momentumAsTimeConstant);

        EpochCriterion epochCriterion; // criterion values are returned in this
        std::vector<EpochCriterion> epochEvalErrors(evaluationNodes.size());
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
                      epochCriterion, epochEvalErrors);
        totalTrainingSamplesSeen += epochCriterion.second; // aggregate #training samples, for logging purposes only

        timer.Stop();
        double epochTime = timer.ElapsedSeconds();

        if (m_useEvalCriterionControlLR && epochEvalErrors.size() > 0)
            lrControlCriterion = epochEvalErrors[0].Average();
        else
            lrControlCriterion = epochCriterion.Average();

        LOGPRINTF(stderr, "Finished Epoch[%2d of %d]: [Training] ", i + 1, (int)m_maxEpochs);
        epochCriterion.LogCriterion(criterionNodes[0]->NodeName());

        m_lastFinishedEpochTrainLoss = epochCriterion.Average();
        for (size_t j = 0; j < epochEvalErrors.size(); j++)
            epochEvalErrors[j].LogCriterion(evaluationNodes[j]->NodeName());
        fprintf(stderr, "totalSamplesSeen = %d; learningRatePerSample = %.8g; epochTime=%.6gs\n", (int)totalTrainingSamplesSeen, learnRatePerSample, epochTime);
#if 0
        // TODO: This was only printed if >1 eval criterion. Why? Needed?
        LOGPRINTF(stderr, "Finished Epoch[%2d of %d]:     Criterion Node [%ls] Per Sample = %.8g\n",
            i + 1, (int)m_maxEpochs, criterionNodes[0]->NodeName().c_str(), epochCriterion.Average());

        for (size_t j = 0; j < epochEvalErrors.size(); j++)
        {
            LOGPRINTF(stderr, "Finished Epoch[%2d of %d]:     Evaluation Node [%ls] Per Sample = %.8g\n",
                i + 1, (int) m_maxEpochs, evalNodeNames[j].c_str(), epochEvalErrors[j].Average());
        }
#endif

        if (validationSetDataReader != trainSetDataReader && validationSetDataReader != nullptr)
        {
            SimpleEvaluator<ElemType> evalforvalidation(net, m_mpi, m_enableDistributedMBReading);
            vector<wstring> cvSetTrainAndEvalNodes;
            if (criterionNodes.size() > 0)
            {
                cvSetTrainAndEvalNodes.push_back(criterionNodes[0]->NodeName());
            }
            for (let node : evaluationNodes)
            {
                cvSetTrainAndEvalNodes.push_back(node->NodeName());
            }

            // BUGBUG: We should not use the training MB size. The training MB size is constrained by both convergence and memory. Eval is only constrained by memory.
            let vScore = evalforvalidation.Evaluate(validationSetDataReader, cvSetTrainAndEvalNodes, m_mbSize[i]);
            LOGPRINTF(stderr, "Finished Epoch[%2d of %d]: [Validate] ", i + 1, (int)m_maxEpochs);
            for (size_t k = 0; k < vScore.size() /*&& k < 2*/; k++)
                vScore[k].LogCriterion(cvSetTrainAndEvalNodes[k], /*addSemicolon=*/k + 1 < vScore.size());
                //fprintf(stderr, "%s %ls = %.8f * %d", k ? ";" : "", cvSetTrainAndEvalNodes[k].c_str(), vScore[k].Average(), (int)vScore[k].second);
            fprintf(stderr, "\n");

            if (m_useCVSetControlLRIfCVExists)
            {
                if (m_useEvalCriterionControlLR && vScore.size() > 1)
                    lrControlCriterion = vScore[1].Average(); // use the first of possibly multiple eval criteria
                else
                    lrControlCriterion = vScore[0].Average(); // the first one is the training criterion
            }
        }

        // broadcast epochCriterion to make sure each processor will have the same learning rate schedule
        if ((GetParallelizationMethod() == ParallelizationMethod::modelAveragingSGD 
            ||
            GetParallelizationMethod() == ParallelizationMethod::blockMomentumSGD) 
            && (m_mpi->NumNodesInUse() > 1))
        {
            m_mpi->Bcast(&epochCriterion.first,  1, m_mpi->MainNodeRank());
            m_mpi->Bcast(&epochCriterion.second, 1, m_mpi->MainNodeRank());
            m_mpi->Bcast(&lrControlCriterion,    1, m_mpi->MainNodeRank());
        }

        bool loadedPrevModel = false;
        size_t epochsSinceLastLearnRateAdjust = i % m_learnRateAdjustInterval + 1;
        if (avgCriterion == numeric_limits<double>::infinity())
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
            if (std::isnan(avgCriterion) || (prevCriterion - avgCriterion < 0 && prevCriterion != numeric_limits<double>::infinity()))
            {
                if (m_loadBestModel)
                {
                    // roll back
                    auto bestModelPath = GetModelNameForEpoch(i - m_learnRateAdjustInterval);
                    LOGPRINTF(stderr, "Loading (rolling back to) previous model with best training-criterion value: %ls.\n", bestModelPath.c_str());
                    net->RereadPersistableParameters<ElemType>(bestModelPath);
                    LoadCheckPointInfo(i - m_learnRateAdjustInterval,
                                       /*out*/ totalTrainingSamplesSeen,
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
                     prevCriterion != numeric_limits<double>::infinity()))
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
                     prevCriterion != numeric_limits<double>::infinity()))
                {

                    learnRatePerSample *= m_learnRateDecreaseFactor;
                    LOGPRINTF(stderr, "learnRatePerSample reduced to %.8g\n", learnRatePerSample);
                }
                else if (prevCriterion - avgCriterion > m_increaseLearnRateIfImproveMoreThan * prevCriterion &&
                         prevCriterion != numeric_limits<double>::infinity())
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

        // Persist model and check-point info
        if ((m_mpi == nullptr) || m_mpi->IsMainNode())
        {
            if (loadedPrevModel)
            {
                // If previous best model is loaded, we will first remove epochs that lead to worse results
                for (int j = 1; j < m_learnRateAdjustInterval; j++)
                {
                    int epochToDelete = i - j;
                    LOGPRINTF(stderr, "SGD: removing model and checkpoint files for epoch %d after rollback to epoch %lu\n", epochToDelete + 1, (size_t)(i - m_learnRateAdjustInterval) + 1);  // report 1 based epoch number
                    _wunlink(GetModelNameForEpoch(epochToDelete).c_str());
                    _wunlink(GetCheckPointFileNameForEpoch(epochToDelete).c_str());
                }

                // Set i back to the loaded model
                i -= m_learnRateAdjustInterval;
                LOGPRINTF(stderr, "SGD: revoke back to and update checkpoint file for epoch %d\n", i+1); // report 1 based epoch number
                SaveCheckPointInfo(i, totalTrainingSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, chosenMinibatchSize);
            }
            else
            {
                SaveCheckPointInfo(i, totalTrainingSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, chosenMinibatchSize);
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
        }
        else
        {
            if (loadedPrevModel)
            {
                // Set i back to the loaded model
                i -= m_learnRateAdjustInterval;
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
            refNet->ReplaceNode(refFeatureNodes[i]->NodeName(), refFeatureNodes[i]);
        }
    }

    delete inputMatrices;
}

// -----------------------------------------------------------------------
// TrainOneEpoch() -- train one epoch
// -----------------------------------------------------------------------

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
                                    /*out*/ EpochCriterion& epochCriterion,
                                    /*out*/ std::vector<EpochCriterion>& epochEvalErrors,
                                    const std::string& prefixMsg)
{
    ScopedNetworkOperationMode modeGuard(net, NetworkOperationMode::training);

    // bring our 'out' values into consistent state
    epochCriterion = EpochCriterion(0);
    epochEvalErrors.assign(epochEvalErrors.size(), EpochCriterion(0));

    double totalTimeInMBs = 0; // use double since timer has sub-microsecond time resolution

    // initialize statistics
    size_t totalEpochSamples = 0;

    int numMBsRun = 0;

    bool useGradientAggregation = UsingGradientAggregation(epochNumber);
    bool useModelAggregation = UsingModelAggregation(epochNumber);
    bool useParallelTrain = UsingParallelTrain(epochNumber);

    // MA-related variables
    size_t nSamplesSinceLastModelSync = 0;
    size_t blockSizePerWorker = 0;
    if (useParallelTrain && m_pMASGDHelper)
    {
        m_pMASGDHelper->OnEpochStart(learnableNodes);
        blockSizePerWorker = m_modelAggregationBlockSize / m_mpi->NumNodesInUse();
    }

    std::vector<Matrix<ElemType>*> learnParamsGradients;
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

    // NOTE: the following two local matrices are not used in distGradAgg path
    // assume only one training criterion node for each epoch.
    // The criterion values are accumulated here over the minibatches (without having to pull them off the GPU).
    CriterionAccumulator<ElemType> localEpochCriterion(1, net->GetDeviceId());
    CriterionAccumulator<ElemType> localEpochEvalErrors(epochEvalErrors.size(), net->GetDeviceId());

    // --- MAIN MINIBATCH LOOP

    // for differential logging, we keep the previous criterion values around
    EpochCriterion         epochCriterionLastLogged  = epochCriterion;
    vector<EpochCriterion> epochEvalErrorsLastLogged = epochEvalErrors;

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

        // Dropout nodes have an implicit input in the form of the random mask that is applied to its explicit input
        // This mask is regerated every minibatch and hence dropout nodes with a non-zero dropout rate must me marked outdated
        // w.r.t. inputs to force evaluation in each minibatch
        MarkDropoutNodesEvalTimeStampAsOutdated(net, criterionNodes[0]);

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
                refNet->GetMBLayoutPtrOfNetwork()->CopyFrom(net->GetMBLayoutPtrOfNetwork()); // TODO: This is UNTESTED (before this was missing, seemingly inconsistently)

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

        // for momentum/clipping/regularization/etc., as well as for progress and statistics, we should only count frames that are not gaps
        // #samples according to the default dynamic axis, for use with criterion nodes that do not have an MBLayout
        size_t numSamplesWithLabelOfNetwork = wasDataRead ? net->GetNumSamplesWithLabelOfNetwork(actualMBSize) : 0;

        // Sum of actualMBSize across all nodes when using parallel training
        // 'aggregate' here means accross-worker aggregate for this one minibatch.
        size_t aggregateNumSamples = actualMBSize;
        size_t aggregateNumSamplesWithLabel = CriterionAccumulator<ElemType>::GetNumSamples(criterionNodes[0], numSamplesWithLabelOfNetwork);

        if (!useGradientAggregation)
        {
            // accumulate criterion values (objective, eval)
            if (actualMBSize != 0)
            {
                assert(wasDataRead);
                // criteria are in Value()(0,0), we accumulate into another 1x1 Matrix (to avoid having to pull the values off the GPU)
                localEpochCriterion.Add(criterionNodes, 0, numSamplesWithLabelOfNetwork);
                for (size_t i = 0; i < evaluationNodes.size(); i++)
                    localEpochEvalErrors.Add(evaluationNodes, i, numSamplesWithLabelOfNetwork);
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
                        Matrix<ElemType>* currParamsGradient = &(node->Gradient()); // TODO: we can use shared_ptrs now

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
            // hoist the criterion into CPU space for all-reduce
            localEpochCriterion.Assign(criterionNodes, 0, numSamplesWithLabelOfNetwork);
            for (size_t i = 0; i < evaluationNodes.size(); i++)
                localEpochEvalErrors.Assign(evaluationNodes, i, numSamplesWithLabelOfNetwork);
            m_gradHeader->numSamplesWithLabel = localEpochCriterion.GetCriterion(0).second;
            m_gradHeader->criterion           = localEpochCriterion.GetCriterion(0).first;
            for (size_t i = 0; i < evaluationNodes.size(); i++)
                m_gradHeader->evalErrors[i] = localEpochEvalErrors.GetCriterion(i);

            bool samplesProcessed = m_distGradAgg->AggregateGradients(learnParamsGradients, m_gradHeader.get(), epochNumber);
            noMoreSamplesToProcess = !samplesProcessed;

            aggregateNumSamples          = m_gradHeader->numSamples;
            aggregateNumSamplesWithLabel = m_gradHeader->numSamplesWithLabel;
            epochCriterion += EpochCriterion(m_gradHeader->criterion, m_gradHeader->numSamplesWithLabel);
            for (size_t i = 0; i < epochEvalErrors.size(); i++)
                epochEvalErrors[i] += m_gradHeader->evalErrors[i];
        }

        // update model parameters
        if ((aggregateNumSamples > 0) && (learnRatePerSample > m_minLearnRate * 0.01))
        {
#if 1       // BUGBUG: We must skip gaps in our momentum, clipping, regularization etc. criteria.
            // This will break test cases. So for now, we will only enable this for per-sample criteria.
            size_t numSamplesInMinibatch = aggregateNumSamples;
            if (criterionNodes[0]->HasMBLayout())
#endif
            numSamplesInMinibatch = aggregateNumSamplesWithLabel;
#if 0
            if (numSamplesInMinibatch != aggregateNumSamples)
                fprintf(stderr, "SGD: using true #samples %d instead of MB size %d\n", (int)numSamplesInMinibatch, (int)aggregateNumSamples);
#endif
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
                    // BUGBUG (Issue #95): Access to net MBLayout can no longer be done if we have multiple input layouts
                    UpdateWeights(node, smoothedGradient, learnRatePerSample,
                                  GetMomentumPerSample(epochNumber /*BUGBUG workaround:*/, net->GetMBLayoutPtrOfNetwork()->GetNumParallelSequences()), numSamplesInMinibatch,
                                  m_L2RegWeight, m_L1RegWeight,
                                  m_needAveMultiplier, m_useNesterovMomentum);
#ifdef _DEBUG
                    if (dynamic_pointer_cast<ComputationNode<ElemType>>(node)->Value().HasNan("TrainOneEpoch/UpdateWeights(): "))
                        LogicError("%ls %ls operation has NaNs in functionValues after parameter update.", node->NodeName().c_str(), node->OperationName().c_str());
#endif
                }
            }
        }

        // aggregation by model averaging or block momentum 
        if (useModelAggregation)
        {
            if (nSamplesSinceLastModelSync >= blockSizePerWorker)
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
        //trainSamplesSinceLastLogged += (int)aggregateNumSamplesWithLabel; // now inside epochCriterionLastLogged

        // log
        // This shows the criterion since last logged.
        if (numMBsRun <= m_firstMBsToShowResult || (m_numMBsToShowResult && (numMBsRun % m_numMBsToShowResult == 0)))
        {
            // get the epoch Values updated
            if (!useGradientAggregation)
            {
                // if no aggregation, we directly get the values from the minibatch accumulators
                timer.Restart();
                epochCriterion = localEpochCriterion.GetCriterion(0);
                for (size_t i = 0; i < epochEvalErrors.size(); i++)
                    epochEvalErrors[i] = localEpochEvalErrors.GetCriterion(i);
                timer.Stop();

                // Add the last trailing compute
                totalTimeInMBs += timer.ElapsedSeconds();
            }

            // epochCriterion aggregates over entire epoch, but we only show difference to last time we logged
            EpochCriterion epochCriterionSinceLastLogged = epochCriterion - epochCriterionLastLogged;
            let trainLossSinceLastLogged    =      epochCriterionSinceLastLogged.Average(); // TODO: Check whether old trainSamplesSinceLastLogged matches this ^^ difference
            let trainSamplesSinceLastLogged = (int)epochCriterionSinceLastLogged.second;

            // determine progress in percent
            int mbProgNumPrecision = 2;
            double mbProg = 0.0;
            if (epochNumber > 0 || (int)epochSize > 0) // TODO: explain this condition in a comment
            {
                if (m_maxComputedEpochSize != 0)
                {
                    double numMBPerEpoch = (double)m_maxComputedEpochSize / (double)tunedMBSize;
                    mbProg = (double)numMBsRun / numMBPerEpoch;
                    mbProgNumPrecision = (int)ceil(log10(numMBPerEpoch / (double)m_numMBsToShowResult));
                    mbProgNumPrecision = max(mbProgNumPrecision - 2, 2);
                }
            }
            else // estimate epoch size
                m_maxComputedEpochSize = numMBsRun * trainSamplesSinceLastLogged / m_numMBsToShowResult;

            // progress tracing for compute cluster management
            let wasProgressPrinted = ProgressTracing::TraceProgressPercentage(epochNumber, mbProg, false);

            // progress tracing for regular log
            if (m_traceLevel > 0)
            {
                PREPENDTS(stderr);
                fprintf(stderr, "%s Epoch[%2d of %d]-Minibatch[%4d-%4d",
                        prefixMsg.c_str(), epochNumber + 1, (int)m_maxEpochs,
                        (int)(numMBsRun - m_numMBsToShowResult + 1), numMBsRun);
                if (epochNumber > 0 || (int)epochSize > 0) // got anything?  --TODO: why cast epochSize to (int) for this comparison?
                    fprintf(stderr, (", %2." + to_string(mbProgNumPrecision) + "f%%").c_str(), mbProg * 100); // --TODO: use a * format?
                fprintf(stderr, "]: ");
                epochCriterionSinceLastLogged.LogCriterion(criterionNodes[0]->NodeName());
                for (size_t i = 0; i < epochEvalErrors.size(); i++)
                    (epochEvalErrors[i] - epochEvalErrorsLastLogged[i]).LogCriterion(evaluationNodes[i]->NodeName());

                fprintf(stderr, ("time = " + GeneratePaddedFloatOrExpFormat(0, 4, totalTimeInMBs) + "s; samplesPerSecond = %.1f\n").c_str(),
                        totalTimeInMBs, trainSamplesSinceLastLogged / totalTimeInMBs);
            }

            // progress tracing for compute cluster management
            if (wasProgressPrinted)
                ProgressTracing::TraceTrainLoss(trainLossSinceLastLogged);

            if (m_traceLevel > 0)
                fflush(stderr);

            if (epochCriterion.IsNan())
                RuntimeError("The training criterion is not a number (NAN).");

            // reset statistics for differential logging
            epochCriterionLastLogged  = epochCriterion;
            epochEvalErrorsLastLogged = epochEvalErrors;

            totalTimeInMBs = 0;
        }

        timer.Restart();
        totalEpochSamples += aggregateNumSamplesWithLabel;

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

    if (useModelAggregation )
    {
        m_pMASGDHelper->OnEpochEnd(learnableNodes, smoothedGradients, nSamplesSinceLastModelSync);
        nSamplesSinceLastModelSync = 0;
    }

    // hoist the accumulated criterion value from GPU side to our 'out'  variables
    // (unless we useGradientAggregation, in which case they are accumulated in the 'out' variables directly)
    if (!useGradientAggregation)
    {
        epochCriterion = localEpochCriterion.GetCriterion(0);
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
            epochEvalErrors[i] = localEpochEvalErrors.GetCriterion(i);
    }

    // in case of model averaging, do one more final aggregation of criteria
    if (useModelAggregation && (m_mpi->NumNodesInUse() > 1))
    {
        // 1. total epoch samples processed by all workers
        size_t totalEpochSamplesOfAllWorkers = totalEpochSamples;
        m_mpi->AllReduce(&totalEpochSamplesOfAllWorkers, 1);

        // get criteria for this worker
        assert(!useGradientAggregation); // (otherwise the data would not be in localEpochCriterion)
        epochCriterion = localEpochCriterion.GetCriterion(0);
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
            epochEvalErrors[i] = localEpochEvalErrors.GetCriterion(i);

        // all-reduce epochCriterion and epochEvalErrors over nodes
        m_mpi->AllReduce(&epochCriterion.first,  1);
        m_mpi->AllReduce(&epochCriterion.second, 1);
        // to transfer the eval vectors, we must pull them apart into STL objects and exchange them separately
        // TODO: merge with training criteria
        vector<double> numer(epochEvalErrors.size());
        vector<size_t> denom(epochEvalErrors.size());
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
        {
            numer[i] = epochEvalErrors[i].first;
            denom[i] = epochEvalErrors[i].second;
        }
        m_mpi->AllReduce(numer);
        m_mpi->AllReduce(denom);
        for (size_t i = 0; i < epochEvalErrors.size(); i++)
            epochEvalErrors[i] = EpochCriterion(numer[i], denom[i]);

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
const std::vector<ComputationNodeBasePtr>& SGD<ElemType>::GetTrainCriterionNodes(ComputationNetworkPtr net)
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
const std::vector<ComputationNodeBasePtr>& SGD<ElemType>::GetEvalCriterionNodes(ComputationNetworkPtr net)
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
                               const std::vector<ComputationNodeBasePtr>& featureNodes,
                               const std::vector<ComputationNodeBasePtr>& labelNodes,
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
        LOGPRINTF(stderr, "\t%ls = %ls()\n", node->NodeName().c_str(), node->OperationName().c_str());
    }

    // compute
    ScopedNetworkOperationMode modeGuard(net, NetworkOperationMode::preComputing);

    // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , requestDataSize);
    // trainSetDataReader->StartMinibatchLoop(m_mbSize[0],  0 , m_epochSize); // only based on one epoch
    // To support large dataset, we usually partition whole dataset into several epoch's,
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
        dynamic_pointer_cast<IPreComputeNode>(node)->MarkComputed(true /*done accumulating*/);

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
    double bestLearnRatePerSample = curLearnRate;

    size_t numFramesToUseInSearch = m_numMiniBatch4LRSearch[epochNumber] * m_mbSize[epochNumber];
    if (m_epochSize != requestDataSize)
    {
        // ensure the numFramesToUseInSearch does not exceed the total number of frames in the epoch
        numFramesToUseInSearch = min(numFramesToUseInSearch, m_epochSize);
    }

    double minLearnRate = m_minLearnRate * 0.3f;
    double learnRatePerSample = 1.0f / 8.0f / 0.618f / sqrt((double) m_mbSize[epochNumber]); // TODO: comment on these magic constants

    if (learnRateInitialized && largestPrevLearnRatePerSample > 0)
    {
        // largestPrevLearnRatePerSample is per sample, first 0.618f is for compensation, second one is for safety
        learnRatePerSample = largestPrevLearnRatePerSample / 0.618f / 0.618f;
    }

    int baseModelEpoch = epochNumber - 1;
    net->RereadPersistableParameters<ElemType>(GetModelNameForEpoch(baseModelEpoch));

    double learnRate = learnRatePerSample;
    size_t dummyMinibatchSize;            // (not used)
    size_t dummyTotalTrainingSamplesSeen; // (not used)
    double prevCriterion = numeric_limits<double>::infinity();
    LoadCheckPointInfo(baseModelEpoch,
                       /*out*/ dummyTotalTrainingSamplesSeen,
                       /*out*/ learnRate,
                       smoothedGradients,
                       /*out*/ prevCriterion,
                       /*out*/ dummyMinibatchSize);

    // if model is not changed this is what we will get
    EpochCriterion baseCriterion;
    vector<EpochCriterion> epochEvalErrors(evaluationNodes.size(), EpochCriterion::Infinity()); // these are ignored in this entire method
    TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                    numFramesToUseInSearch, trainSetDataReader, 0, m_mbSize[epochNumber],
                                    featureNodes, labelNodes,
                                    criterionNodes, evaluationNodes,
                                    inputMatrices, learnableNodes,
                                    smoothedGradients,
                                    /*out*/ baseCriterion, /*out*/ epochEvalErrors,
                                    "BaseAdaptiveLearnRateSearch:");

    if (m_autoLearnRateSearchType == LearningRateSearchAlgorithm::SearchBeforeEpoch)
    {
        if (prevCriterion == numeric_limits<double>::infinity())
            prevCriterion = baseCriterion.Average();

        double ratio = 0.3;

        if (m_epochSize != requestDataSize)
            ratio = pow(((double) numFramesToUseInSearch) / m_epochSize, 1.0f / 2);

        // interpolate prevCriterion into 'baseCriterion'
        baseCriterion.first = baseCriterion.second * max(ratio * prevCriterion + (1 - ratio) * baseCriterion.Average(), baseCriterion.Average());
    }

    EpochCriterion epochCriterion(EpochCriterion::Infinity());
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
                                        "AdaptiveLearnRateSearch:");
    } while (epochCriterion.IsNan() || (epochCriterion.Average() > baseCriterion.Average() && learnRatePerSample > minLearnRate));

    bestLearnRatePerSample = learnRatePerSample;

    // grid search for the first m_numBestSearchEpoch  epochs
    if (epochNumber < m_numBestSearchEpoch)
    {
        double leftLearnRatePerSample = 0.01 / m_mbSize[epochNumber];
        double rightLearnRatePerSample = learnRatePerSample;
        EpochCriterion rightCriterion = epochCriterion;
        EpochCriterion leftCriterion; // we compute this from the mini epoch

        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                        numFramesToUseInSearch, trainSetDataReader,
                                        leftLearnRatePerSample, m_mbSize[epochNumber],
                                        featureNodes, labelNodes,
                                        criterionNodes, evaluationNodes,
                                        inputMatrices, learnableNodes,
                                        smoothedGradients,
                                        /*out*/ leftCriterion, /*out*/ epochEvalErrors,
                                        "DetailBaseAdaptiveLearnRateSearch:");

        while (rightLearnRatePerSample > leftLearnRatePerSample * 1.2)
        {
            if (rightCriterion.Average() > leftCriterion.Average())
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
                                                "DetailLeftAdaptiveLearnRateSearch:");
            }
        }

        bestLearnRatePerSample = (leftCriterion.Average() < rightCriterion.Average()) ? leftLearnRatePerSample : rightLearnRatePerSample;
    }

    LOGPRINTF(stderr, "Best Learn Rate Per Sample for Epoch[%d] = %.10g  baseCriterion=%.10g\n",
              epochNumber + 1, bestLearnRatePerSample, baseCriterion.Average());

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
    double learningRateChangeSoFar = GetLearningRatePerSample(epochNumber /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequencesForFixingBPTTMode()) / GetLearningRatePerSample(0 /*BUGBUG workaround:*/, trainSetDataReader->GetNumParallelSequencesForFixingBPTTMode());
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
        // BUGBUG: %z is not supported in VS 2013, is it?
        LOGPRINTF(stderr, "before epoch .2, previous minibatchSize %zd is considered invalid -> resetting\n",
                  m_prevChosenMinibatchSize);
        m_prevChosenMinibatchSize = 0;
    }

    // check if we need to skip
    if (m_prevChosenMinibatchSize != 0 &&
        (epochNumber + 1) > m_minibatchSizeTuningFrequency &&
        (epochNumber + 1) % m_minibatchSizeTuningFrequency != 0)
    {
        LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Search for a better minibatchSize in epoch %d skipped, keeping minibatchSize of %zd\n",
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
            LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Limiting minMinibatchSize to largest of previous minibatchSize = (%d / 2) or %d\n",
                      (int) m_prevChosenMinibatchSize, (int) minMinibatchSize);
            minMinibatchSize = max(minMinibatchSize, m_prevChosenMinibatchSize / 2);
        }

        size_t maxMinibatchSize = m_minibatchSizeTuningMax;

        // only grow at most 2 x compared to previous step
        if (m_prevChosenMinibatchSize != 0.0f)
        {
            assert(m_prevChosenMinibatchSize >= chosenMinibatchSize);

            LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Limiting maxMinibatchSize to previous minibatchSize %zd*2\n",
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
    EpochCriterion baseCriterion(0);

    // increase the minibatch size by a factor of sqrt(2) in each step.
    const float minibatchSizeTuningFactor = sqrtf(2.0f);

    size_t lastTriedTrialMinibatchSize = 0;
    EpochCriterion lastTriedTrialEpochCriterion(0);
    for (float trialMinibatchSizeFloat = (float) minMinibatchSize;
         trialMinibatchSizeFloat <= maxMinibatchSize;
         trialMinibatchSizeFloat *= minibatchSizeTuningFactor)
    {
        // round mbsize to something meaningful
        trialMinibatchSize = RoundToMultipleOf64(trialMinibatchSizeFloat);

        fprintf(stderr, "\n");
        LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Evaluating trial minibatchSize=%d out of range %d..%d ...\n\n",
                  (int)trialMinibatchSize, (int)RoundToMultipleOf64(minMinibatchSize), (int)RoundToMultipleOf64(maxMinibatchSize));

        std::vector<EpochCriterion> epochEvalErrors(evaluationNodes.size(), EpochCriterion::Infinity());
        EpochCriterion epochCriterion(EpochCriterion::Infinity());

        // Train on a few minibatches and so we can observe the epochCriterion as we try increasing
        // minibatches with iteration of this loop.
        TrainOneMiniEpochAndReloadModel(net, refNet, refNode, epochNumber,
                                        numFramesToUseInSearch, trainSetDataReader,
                                        learnRatePerSample, trialMinibatchSize, featureNodes,
                                        labelNodes, criterionNodes,
                                        evaluationNodes, inputMatrices,
                                        learnableNodes, smoothedGradients,
                                        /*out*/ epochCriterion, /*out*/ epochEvalErrors,
                                        isFirstIteration ? "BaseAdaptiveMinibatchSearch:" : "AdaptiveMinibatchSearch:");

        if (isFirstIteration)
        {
            // for the first iteration of the loop only, set baseCriterion
            // to the result we got from TrainOneMiniEpochAndReloadModel().
            baseCriterion = epochCriterion;
            lastTriedTrialMinibatchSize = trialMinibatchSize;
            lastTriedTrialEpochCriterion = baseCriterion;
            isFirstIteration = false;

            LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Computed baseCriterion %.8f\n", baseCriterion.Average());
        }
        else if (!epochCriterion.IsNan() &&
                 epochCriterion.Average() > (baseCriterion.Average() * (1.0 + (m_minibatchSearchCriterionErrorMargin / 100.0))))
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
                LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Keep searching... epochCriterion = %.8f vs bBaseCriterion = %.8f\n",
                          epochCriterion.Average(), baseCriterion.Average());
            }
        }
    }
    LOGPRINTF(stderr, "AdaptiveMinibatchSearch: Search successful. New minibatchSize is %d. epochCriterion = %.8f vs baseCriterion = %.8f\n\n",
              (int) lastTriedTrialMinibatchSize, lastTriedTrialEpochCriterion.Average(), baseCriterion.Average());

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
                                                    /*out*/ EpochCriterion& epochCriterion,
                                                    /*out*/ std::vector<EpochCriterion>& epochEvalErrors,
                                                    std::string prefixMsg)
{
    TrainOneEpoch(net, refNet, refNode, epochNumber, epochSize,
                  trainSetDataReader, learnRatePerSample, minibatchSize, featureNodes,
                  labelNodes, criterionNodes, evaluationNodes,
                  inputMatrices, learnableNodes, smoothedGradients,
                  /*out*/ epochCriterion, /*out*/ epochEvalErrors,
                  prefixMsg);

    LOGPRINTF(stderr, "Finished Mini-Epoch For LearnRate Selection: ");
    epochCriterion.LogCriterion(criterionNodes[0]->NodeName());
    for (size_t j = 0; j < epochEvalErrors.size(); j++)
        epochEvalErrors[j].LogCriterion(evaluationNodes[j]->NodeName());
    fprintf(stderr, "learningRatePerSample = %.8g\n", learnRatePerSample);

    // go back to where we came from
    int baseModelEpoch = epochNumber - 1;
    net->RereadPersistableParameters<ElemType>(GetModelNameForEpoch(baseModelEpoch));

    double dummyLearnRate;
    double dummyPrevCriterion;
    size_t dummyTotalTrainingSamplesSeen; // (not used)
    size_t dummyMinibatchSize;
    LoadCheckPointInfo(baseModelEpoch,
                       /*out*/ dummyTotalTrainingSamplesSeen,
                       /*out*/ dummyLearnRate,
                       smoothedGradients,
                       /*out*/ dummyPrevCriterion,
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

        // BUGBUG (Issue #95): This is no longer correct once we have multiple input layouts.
        trainSetDataReader->CopyMBLayoutTo(net->GetMBLayoutPtrOfNetwork());
        net->ForwardProp(outputNodes[0]); // only evaluate the first output
        trainSetDataReader->SetNetOutput(uttInfo,
                                         dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[0])->Value(),
                                         pMBLayout);
    }
}

template <class ElemType>
void SGD<ElemType>::InitDistGradAgg(int numEvalNodes, int traceLevel)
{
    if (GetParallelizationMethod() == ParallelizationMethod::dataParallelSGD)
    {
        if (m_distGradAgg == nullptr)
        {
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
            m_distGradAgg = std::make_shared<AllReduceDistGradAggregator<ElemType>>(m_mpi, m_numGradientBits, m_zeroThresholdFor1Bit, true /*useQuantizationForSelfStripe*/, m_bufferedAsyncGradientAggregation, traceLevel, m_syncStatsTrace);
#else
            if (m_numGradientBits != (8 * sizeof(ElemType)))
            {
                RuntimeError("Gradient quantization is unsupported in CNTK binaries built without quantized gradient aggregation support!");
            }

            m_distGradAgg = std::make_shared<SimpleDistGradAggregator<ElemType>>(m_mpi, m_bufferedAsyncGradientAggregation, m_syncStatsTrace);
#endif // !CNTK_PARALLEL_TRAINING_SUPPORT
        }

        if (m_gradHeader == nullptr)
        {
            m_gradHeader.reset(DistGradHeader::Create(numEvalNodes), [](DistGradHeader* ptr) {
                DistGradHeader::Destroy(ptr);
            });
        }
    }
}

template <class ElemType>
void SGD<ElemType>::InitModelAggregationHandler(int traceLevel, DEVICEID_TYPE devID)
{
    if (m_pMASGDHelper)
    {
        return; // no need to do anything if already initialized. TODO: make it singleton 
    }
    if (GetParallelizationMethod() == ParallelizationMethod::modelAveragingSGD)
    {
        m_pMASGDHelper = make_shared<BasicModelAveragingSGD<ElemType>>(m_mpi, traceLevel, devID);
    }
    else if (GetParallelizationMethod() == ParallelizationMethod::blockMomentumSGD)
    {
#ifndef CNTK_PARALLEL_TRAINING_SUPPORT
        RuntimeError("Block Momentum is not supported in the main CNTK repo. You need to enable 1bit submodule.");
#else
        m_pMASGDHelper = make_shared<BlockMomentumSGD<ElemType>>(m_mpi, traceLevel, devID, 
                                                                 m_useNesterovBlockMomentum, m_resetSGDMomentum, 
                                                                 m_blockLearningRate, m_blockMomentumAsTimeConstant, 
                                                                 m_modelAggregationBlockSize);
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
            if (m_pMASGDHelper)
                m_pMASGDHelper->SaveToCheckPoint(fstream);
            // Ensuring that data is written
            fstream.Flush();
        }

        _wunlink(checkPointFileName.c_str());
        renameOrDie(tempFileName, checkPointFileName);
    }
}

template <class ElemType>
bool SGD<ElemType>::TryLoadCheckPointInfo(const size_t epochNumber,
                                          /*out*/ size_t& totalSamplesSeen,
                                          /*out*/ double& learnRatePerSample,
                                          std::list<Matrix<ElemType>>& smoothedGradients,
                                          /*out*/ double& prevCriterion,
                                          /*out*/ size_t& minibatchSize)
{
    // gracefully handle if a checkpoint file is missing
    // This means a user wanted to continue training from an older model, but that model had no checkpoint info anymore.
    // This is valid, we just don't get the features that require previous models, such as LR or MBSize control.
    let checkPointFileName = GetCheckPointFileNameForEpoch(int(epochNumber));
    if (!fexists(checkPointFileName.c_str()))
    {
        // initialize as if nothing
        totalSamplesSeen = 0;
        learnRatePerSample = numeric_limits<double>::quiet_NaN(); // must be overwritten
        prevCriterion = 0;
        minibatchSize = m_mbSize[epochNumber];

        LOGPRINTF(stderr, "Warning: Checkpoint file is missing. Learning parameters will be initialized from 0\n");
        return false;
    }

    LoadCheckPointInfo(epochNumber, totalSamplesSeen, learnRatePerSample, smoothedGradients, prevCriterion, minibatchSize);
    return true;
}

template <class ElemType>
void SGD<ElemType>::LoadCheckPointInfo(const size_t epochNumber,
                                       /*out*/ size_t& totalSamplesSeen,
                                       /*out*/ double& learnRatePerSample,
                                       std::list<Matrix<ElemType>>& smoothedGradients,
                                       /*out*/ double& prevCriterion,
                                       /*out*/ size_t& minibatchSize)
{
    let checkPointFileName = GetCheckPointFileNameForEpoch(int(epochNumber));
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
    else // some legacy files do not have this
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

    if (m_pMASGDHelper)
    {
        m_pMASGDHelper->LoadFromCheckPoint(fstream);
    }

    return;
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

template <class ElemType>
void SGD<ElemType>::MarkDropoutNodesEvalTimeStampAsOutdated(const ComputationNetworkPtr& net, const ComputationNodeBasePtr& criterionNode)
{
    list<ComputationNodeBasePtr> dropoutNodes = net->GetNodesWithType(OperationNameOf(DropoutNode), criterionNode);
    for (auto& nodeIter : dropoutNodes)
        nodeIter->SetEvalTimeStampOutdatedWrtAll();
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
    if      (EqualCI(s, L"") || EqualCI(s, L"none")) return ParallelizationMethod::none;
    else if (EqualCI(s, L"DataParallelSGD"))         return ParallelizationMethod::dataParallelSGD;
    else if (EqualCI(s, L"ModelAveragingSGD"))       return ParallelizationMethod::modelAveragingSGD;
    else if (EqualCI(s, L"BlockMomentumSGD"))        return ParallelizationMethod::blockMomentumSGD;
    else InvalidArgument("ParseParallelizationMethod: Invalid Parallelization Method. Valid values are (none | DataParallelSGD | ModelAveragingSGD | BlockMomentumSGD)");
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
    // the learning rate. It's typically set to 10-20% of
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
    m_numMBsToShowResult = configSGD(L"numMBsToShowResult", (size_t)10);
    m_firstMBsToShowResult = configSGD(L"firstMBsToShowResult", (size_t)0);
    m_numMBsToCUDAProfile = configSGD(L"numMBsToCUDAProfile", (size_t)0);

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
    m_parallelizationMethod = ParallelizationMethod::none;
    m_numGradientBits = 32;
    m_zeroThresholdFor1Bit = true;
    m_bufferedAsyncGradientAggregation = false;
    m_enableDistributedMBReading = false;
    m_parallelizationStartEpochNum = 0;
    m_modelAggregationBlockSize = 0; 

    if (configSGD.Exists(L"ParallelTrain"))
    {
        MPIWrapperPtr pMPI = MPIWrapper::GetInstance(); 
        if (!pMPI) 
        {
            // some users may forget to specify useParallelTrain option 
            // in this case, falling back to normal SGD
            fprintf(stderr, "useParallelTrain option is not enabled. ParallelTrain config will be ignored.");
        }
        else
        {
            size_t numMPIWorkers = pMPI->NumNodesInUse();            
            const ConfigRecordType& configParallelTrain(configSGD(L"ParallelTrain", ConfigRecordType::Record()));
            m_parallelizationMethod = ParseParallelizationMethod(configParallelTrain(L"parallelizationMethod", L"none"));
            m_parallelizationStartEpochNum = configParallelTrain(L"parallelizationStartEpoch", (int)1) - 1; // Epoch numbers internally are 0 based
            m_enableDistributedMBReading = configParallelTrain(L"distributedMBReading", false);
            m_syncStatsTrace = configParallelTrain(L"syncPerfStats", (int)0);

            if (configParallelTrain.Exists(L"DataParallelSGD"))
            {
                const ConfigRecordType& configDataParallelSGD(configParallelTrain(L"DataParallelSGD", ConfigRecordType::Record()));
                size_t defaultGradientBits = 8 * sizeofElemType;
                m_numGradientBits = configDataParallelSGD(L"gradientBits", defaultGradientBits);
                m_zeroThresholdFor1Bit = configDataParallelSGD(L"useZeroThresholdFor1BitQuantization", true);
                m_bufferedAsyncGradientAggregation = configDataParallelSGD(L"useBufferedAsyncGradientAggregation", false);
                if ( m_numGradientBits < 1 || m_numGradientBits > (8 * sizeofElemType) )
                {
                    InvalidArgument("gradientBits must be in the range [1, 32] when using precision=float and in range [1, 64] when using precision=double!");
                }
            }
            if (configParallelTrain.Exists(L"ModelAveragingSGD"))
            {
                const ConfigRecordType& configMASGD(configParallelTrain(L"ModelAveragingSGD", ConfigRecordType::Record()));
                if (configMASGD.Exists(L"blockSizePerWorker") && configMASGD.Exists(L"blockSize"))
                {
                    InvalidArgument("It is only allowed to set blockSizePerWorker or blockSize, not both of them");
                }
                else if (configMASGD.Exists(L"blockSize"))
                {
                    m_modelAggregationBlockSize = configMASGD(L"blockSize");
                }
                else if (configMASGD.Exists(L"blockSizePerWorker"))
                {
                    m_modelAggregationBlockSize = configMASGD(L"blockSizePerWorker");
                    m_modelAggregationBlockSize *= numMPIWorkers;
                }
                else
                {
                    m_modelAggregationBlockSize = 40000 * numMPIWorkers;    // default value 
                }
#if 1  // legacy option 
                if (configMASGD.Exists(L"syncFrequencyInFrames"))
                {
                    if (configMASGD.Exists(L"blockSizePerWorker") || configMASGD.Exists(L"blockSize"))
                        InvalidArgument("syncFrequencyInFrames is a deprecated alias of blockSizePerWorker. It is not allowed to specify both of them");
                    m_modelAggregationBlockSize = configMASGD(L"syncFrequencyInFrames");
                    m_modelAggregationBlockSize *= numMPIWorkers;
                    fprintf(stderr, "WARNING: option syncFrequencyInFrames in ModelAveragingSGD is going to be deprecated. Please use blockSizePerWorker instead\n");
                }
                if (configMASGD.Exists(L"syncPeroid"))
                {
                    if (configMASGD.Exists(L"blockSizePerWorker") || configMASGD.Exists(L"blockSize"))
                    {
                        InvalidArgument("syncPeriod is a deprecated alias of blockSizePerWorker. It is not allowed to specify both of them");
                    }
                    m_modelAggregationBlockSize = configMASGD(L"syncPeriod");
                    m_modelAggregationBlockSize *= numMPIWorkers;
                    fprintf(stderr, "WARNING: option syncPeroid in ModelAveragingSGD is going to be deprecated. Please use blockSizePerWorker instead in the future.\n");
                }
#endif
            }
            if (configParallelTrain.Exists(L"BlockMomentumSGD"))
            {
#ifndef CNTK_PARALLEL_TRAINING_SUPPORT
                InvalidArgument("BlockMomentumSGD is not enabled in this version.\n"); 
#else
                const ConfigRecordType& configBMSGD(configParallelTrain(L"BlockMomentumSGD", ConfigRecordType::Record()));
                if (configBMSGD.Exists(L"blockSize") && configBMSGD.Exists(L"blockSizePerWorker"))
                {
                    InvalidArgument("It is only allowed to set blockSizePerWorker or blockSize, not both of them");
                }
                else if (configBMSGD.Exists(L"blockSizePerWorker"))
                {
                    m_modelAggregationBlockSize = configBMSGD(L"blockSizePerWorker");
                    m_modelAggregationBlockSize *= numMPIWorkers;
                }
                else if (configBMSGD.Exists(L"blockSize"))
                {
                    m_modelAggregationBlockSize = configBMSGD(L"blockSize");
                }
                else
                {
                    m_modelAggregationBlockSize = 120000 *numMPIWorkers; // default value 
                }
#if 1  // legacy option
                if (configBMSGD.Exists(L"syncPeriod"))
                {
                    if (configBMSGD.Exists(L"blockSizePerWorker") || configBMSGD.Exists(L"blockSize"))
                    {
                        InvalidArgument("syncPeriod is a deprecated alias of blockSizePerWorker. It is not allowed to specify both of them");
                    }
                    m_modelAggregationBlockSize = configBMSGD(L"syncPeriod");
                    m_modelAggregationBlockSize *= numMPIWorkers;
                    fprintf(stderr, "WARNING: option syncPeroid in BlockMomentumSGD is going to be deprecated. Please use blockSizePerWorker instead in the future.\n");
                }
#endif 
                m_resetSGDMomentum = configBMSGD(L"resetSGDMomentum", true);
                m_useNesterovBlockMomentum = configBMSGD(L"useNesterovMomentum", true);
                m_blockLearningRate = configBMSGD(L"blockLearningRate", 1.0);

                if (configBMSGD.Exists(L"blockMomentumPerSync") && configBMSGD.Exists(L"blockMomentumAsTimeConstant"))
                {
                    InvalidArgument("It is only allowed to set either blockMomentumPerSync or blockMomentumAsTimeConstant, not both of them");
                }
                else if (configBMSGD.Exists(L"blockMomentumAsTimeConstant"))
                {
                    m_blockMomentumAsTimeConstant = configBMSGD(L"blockMomentumAsTimeConstant");
                }
#if 1       // This option "blockMomentumPerSync" is going to be deprecated in the future 
                else if (configBMSGD.Exists(L"blockMomentumPerSync"))
                {
                    double blockMomentum = configBMSGD(L"blockMomentumPerSync");
                    m_blockMomentumAsTimeConstant = BlockMomentumSGD<double>::Momentum2TimeConstant(blockMomentum, m_modelAggregationBlockSize);
                }
#endif 
                else /*if (!configBMSGD.Exists(L"blockMomentumPerSync") && !configBMSGD.Exists(L"blockMomentumAsTimeConstant"))*/
                {
                    double blockMomentum = 1.0 - 1.0 / (double)numMPIWorkers;   // this is a default value which ensures each block update contributes equally
                    m_blockMomentumAsTimeConstant = BlockMomentumSGD<double>::Momentum2TimeConstant(blockMomentum, m_modelAggregationBlockSize);
                }
#endif 
                InitializeAndCheckBlockMomentumSGDParameters();
                
            }
        } // if (!pMPI)
    } // if (configSGD.Exists(L"ParallelTrain"))
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

void SGDParams::InitializeAndCheckBlockMomentumSGDParameters()
{
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT 
    // final argument checking in case of user specifying a bad parameter
    size_t numMPIWorker = MPIWrapper::GetInstance()->NumNodesInUse();
    double blockMomentum = BlockMomentumSGD<double>::TimeConstant2Momentum(m_blockMomentumAsTimeConstant, m_modelAggregationBlockSize);
    if ((1 - blockMomentum)*m_blockLearningRate*numMPIWorker >= 2.0)
    {
        fprintf(stderr, "WARNING: (1-blockMomentumPerSync)*blockLearningRate is larger than 2*numWorkers; it is possible to overshoot.");
    }
    if (blockMomentum == 0.0)
    {
        fprintf(stderr, "WARNING: blockMomentum equals to zero. \n");
    }
#else
    // don't need do anything here 
    m_blockMomentumAsTimeConstant = 0.0;
    m_blockLearningRate = 1.0;
#endif 
}

// register SGD<> with the ScriptableObject system
ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<SGD<float>, SGD<double>> registerSGDOptimizer(L"SGDOptimizer");

}}}
