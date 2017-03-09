//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "fileutil.h"

namespace CNTK
{
    const std::wstring TrainingSession::s_checkpointIndex = L"CheckpointIndex";
    const std::wstring TrainingSession::s_trainingMinibatchSource = L"TrainingMinibatchSource";

    TrainingSessionPtr CreateBasicTrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyinSamples,
        const std::wstring& checkPointFileName)
    {
        return MakeSharedObject<TrainingSession>(trainingSource,
            trainer,
            modelInputToMinibatchSourceStream,
            minibatchSizeSchedule,
            checkpointFrequencyinSamples,
            checkPointFileName);
    }

    TrainingSession::TrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& schedule,
        size_t checkpointFrequencyInSamples,
        const std::wstring& checkPointFileName) :
        m_trainingSource(trainingSource),
        m_trainer(trainer),
        m_modelInputToMinibatchSourceStream(modelInputToMinibatchSourceStream),
        m_checkpointFrequencyinSamples(checkpointFrequencyInSamples),
        m_checkPointFileName(checkPointFileName),
        m_currentCheckpointIndex(0),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1),
        m_minibatchSizeSchedule(schedule)
    {
        if (!trainingSource)
            InvalidArgument("Minibatch source is not allowed to be null.");
        if (!trainer)
            InvalidArgument("Trainer is not allowed to be null.");
        if(modelInputToMinibatchSourceStream.empty())
            InvalidArgument("Input mapping is not allowed to be empty.");
        if (m_checkPointFileName.empty() && checkpointFrequencyInSamples != 0)
            InvalidArgument("Checkpoint file name is not allowed to be empty.");

        // Let's calculate the warm up period the distributed learners may need.
        // We will take the maximum warm up period required.
        auto learners = trainer->ParameterLearners();
        m_parallelAfterSamples = 0;
        for (const auto& l: learners)
        {
            auto distributed = std::dynamic_pointer_cast<DistributedLearner>(l);
            if (distributed)
            {
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
            }
        }
    }

    void TrainingSession::Train(const DeviceDescriptor& computeDevice)
    {
        std::unordered_map<Variable, ValuePtr> minibatch;
        bool shouldTrain = true;
        size_t workerRank = 0, numberOfWorkers = 1;
        size_t samplesInEpoch = 0;
        while (shouldTrain)
        {
            // Check if we are operating in distributed mode.
            if (m_parallelAfterSamples >= m_trainer->TotalNumberOfSamplesSeen())
            {
                numberOfWorkers = m_numberOfWorkers;
                workerRank = m_workerRank;
            }

            size_t mbSize = GetMinibatchSize();
            auto minibatchData = m_trainingSource->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);

            minibatch.clear();
            if (!minibatchData.empty())
            {
                for (auto v : m_modelInputToMinibatchSourceStream)
                    minibatch.insert({ v.first, minibatchData[v.second].data });
            }

            OnMinibatchStart();
            shouldTrain = m_trainer->TrainMinibatch(minibatch, computeDevice);
            OnMinibatchEnd();

            // Local number of samples.
            samplesInEpoch += m_trainer->PreviousMinibatchSampleCount();

            // Check whether to create a checkpoint
            if (m_checkpointFrequencyinSamples > 0)
            {
                size_t checkpointIndex = m_trainer->TotalNumberOfSamplesSeen() / m_checkpointFrequencyinSamples;
                if (checkpointIndex > m_currentCheckpointIndex)
                {
                    samplesInEpoch = 0;
                    m_currentCheckpointIndex = checkpointIndex;
                    SaveCheckpoint();
                }
            }
        }

        if (m_checkpointFrequencyinSamples > 0)
            SaveCheckpoint();
    }

    void TrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = m_trainer->RestoreFromCheckpoint(checkpointFileName);
        m_currentCheckpointIndex = externalState[s_checkpointIndex].Value<size_t>();
        m_trainingSource->RestoreFromCheckpoint(externalState[s_trainingMinibatchSource].Value<Dictionary>());
    }

    void TrainingSession::SaveCheckpoint()
    {
        OnCheckpointStart();
        Dictionary externalState;
        externalState[s_checkpointIndex] = m_currentCheckpointIndex;
        externalState[s_trainingMinibatchSource] = m_trainingSource->GetCheckpointState();        
        m_trainer->SaveCheckpoint(m_checkPointFileName, externalState);
        OnCheckpointEnd();
    }
}
