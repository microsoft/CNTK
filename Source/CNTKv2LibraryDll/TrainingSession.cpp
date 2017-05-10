//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <boost/algorithm/string/predicate.hpp>

#include "CNTKLibrary.h"
#include "fileutil.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    using namespace std;

    const static std::wstring s_trainingMinibatchSource = L"TrainingMinibatchSource";

    inline bool isNumber(const std::wstring& s)
    {
        return !s.empty() &&
            find_if(s.begin(), s.end(), [](wchar_t c) { return !isdigit(c); }) == s.end();
    }

    CheckpointConfig::CheckpointConfig(
        const std::wstring& checkPointFileName,
        size_t checkpointFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool preserveAllCheckpoints) :
        m_preserveAll(preserveAllCheckpoints),
        m_restore(restoreFromCheckpointIfExists),
        m_fileName(checkPointFileName),
        m_frequency(checkpointFrequencyInSamples)
    {
        if (m_fileName.empty())
        {
            if (checkpointFrequencyInSamples != 0 && checkpointFrequencyInSamples != std::numeric_limits<size_t>::max())
                InvalidArgument("Checkpoint file name must not be empty if checkpoint frequency is non zero.");

            if (preserveAllCheckpoints)
                InvalidArgument("Checkpoint file name must not be empty if 'preserve all checkpoints' option is specified.");

            checkpointFrequencyInSamples = 0;
        }
    }

    CrossValidationConfig::CrossValidationConfig(
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples):
        m_source(crossValidationSource),
        m_mbSize(crossValidationSchedule),
        m_frequency(crossValidationFrequencyInSamples)
    {
    }

    TestConfig::TestConfig(
        const MinibatchSourcePtr& source,
        const MinibatchSizeSchedule& schedule) :
        m_source(source),
        m_mbSize(schedule)
    {
    }

    TrainingSessionPtr CreateTrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        size_t maxNumTrainingSamples,
        size_t progressFrequency,
        const CheckpointConfig& checkpointing,
        const CrossValidationConfig& crossValidation,
        const TestConfig& test)
    {
        return MakeSharedObject<TrainingSession>(trainer,
            trainingSource,
            minibatchSizeSchedule,
            inputVarToStream,
            maxNumTrainingSamples,
            progressFrequency,
            checkpointing, crossValidation, test);
    }

    TrainingSession::TrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        size_t maxNumTrainingSamples,
        size_t progressFrequency,
        const CheckpointConfig& checkpointing,
        const CrossValidationConfig& crossValidation,
        const TestConfig& test) :
        m_trainer(trainer),
        m_source(trainingSource),
        m_mbSize(minibatchSizeSchedule),
        m_varToStream(inputVarToStream),
        m_maxNumSamples(maxNumTrainingSamples),
        m_progressFrequency(progressFrequency),
        m_checkpoint(checkpointing),
        m_cv(crossValidation),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1),
        m_test(test)
    {
        if (!m_trainer)
            InvalidArgument("Trainer must not be null.");

        if (!m_source)
            InvalidArgument("Training source must not be null.");

        if (m_maxNumSamples == 0)
            InvalidArgument("maxNumTrainingSamples must not be zero.");

        if (m_varToStream.empty())
            InvalidArgument("inputVarToStream mapping must not be empty.");

        // Let's calculate the warm up period the distributed learners may need.
        // We will take the maximum warm up period required.
        auto learners = m_trainer->ParameterLearners();
        m_parallelAfterSamples = 0;
        for (const auto& l : learners)
        {
            auto distributed = std::dynamic_pointer_cast<DistributedLearner>(l);
            if (distributed)
            {
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
            }
        }

        // Fill-in required actions.
        if (m_checkpoint.m_frequency != 0)
            m_actions.push_back({ m_checkpoint.m_frequency, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&)
                {
                    SaveCheckpoint(currentIndex);
                    // enable profiler after the first checkpoint
                    // This has effect only if the profiler is globally enabled by StartProfiler()
                    Microsoft::MSR::CNTK::ProfilerEnable(true);
                    return true;
                } });

        // Report progress before we run cross validation if any.
        if (m_progressFrequency != 0)
        {
            m_actions.push_back({ m_progressFrequency, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&) { ReportProgress(currentIndex); return true; } });
        }

        if (m_cv.m_frequency != 0)
            m_actions.push_back({ m_cv.m_frequency , 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor& d) { return CrossValidate(currentIndex, d); } });
    }

    void TrainingSession::Train(const DeviceDescriptor& computeDevice)
    {
        std::unordered_map<Variable, ValuePtr> minibatch;
        bool shouldTrain = m_maxNumSamples > 0;

        // Let's try to restore if required.
        size_t restoredNumberOfSamples = 0;
        if (m_checkpoint.m_restore && !m_checkpoint.m_fileName.empty())
        {
            RestoreFromCheckpoint();
            restoredNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
        }

        // Main train loop.
        bool earlyExit = false;
        while (shouldTrain)
        {
            // Get next minibatch.
            size_t samplesLeft = earlyExit || m_maxNumSamples <= Trainer()->TotalNumberOfSamplesSeen()
                ? 0
                : m_maxNumSamples - Trainer()->TotalNumberOfSamplesSeen();

            // Note that in case of distributed training we don't want to stop if the local minibatch
            // is empty - it is possible that the other workers are still processing their minibatches.
            GetTrainingMinibatch(minibatch, samplesLeft, computeDevice);

            // Train on the minibatch.
            OnMinibatchStart();
            shouldTrain = Trainer()->TrainMinibatch(minibatch, computeDevice);
            earlyExit |= !OnMinibatchEnd(); // If the callback wants to have early exit - we stop training.

            auto profMisc = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainPost);

            // Peform actions if required.
            size_t totalNumberOfSamples = Trainer()->TotalNumberOfSamplesSeen();
            for (auto& action : m_actions)
            {
                size_t index = totalNumberOfSamples / action.frequency;
                if (index != action.currentIndex)
                {
                    // If any action wants to have early exit - we stop training.
                    earlyExit |= !action.action(action.currentIndex, computeDevice);

                    action.currentIndex = index;
                    action.sampleCountWhenLastCalled = totalNumberOfSamples;
                }
            }
        }

        if (restoredNumberOfSamples != Trainer()->TotalNumberOfSamplesSeen())
        {
            // Let's do all actions on the last probably a partial data at the end.
            for (auto& action: m_actions)
            {
                if (Trainer()->TotalNumberOfSamplesSeen() % action.frequency != 0 &&
                    Trainer()->TotalNumberOfSamplesSeen() != action.sampleCountWhenLastCalled)
                    action.action(action.currentIndex, computeDevice);
            }
        }

        // In case of incremental - save final checkpoint.
        // This is required only when we keep all existing checkpoints, otherwise 
        // The checkpoint was already saved with the proper name.
        if (m_checkpoint.m_frequency &&
            m_checkpoint.m_preserveAll &&
            !fexists(m_checkpoint.m_fileName))
            SaveFinalCheckpoint();

        // Perform testing according to the test config.
        Test(computeDevice);
    }

    // TODO: Possibly expose a limiting counter on the number of samples for validation.
    bool TrainingSession::CrossValidate(size_t currentIndex, const DeviceDescriptor& computeDevice)
    {
        if (m_cv.m_source) // Running cross validation
        {
            std::unordered_map<Variable, ValuePtr> minibatch;
            double accumulatedError = 0;
            size_t totalNumberOfSamples = 0;
            size_t numberOfMinibatches = 0;

            std::pair<ValuePtr, size_t> errorAndCount;
            auto checkpoint = m_cv.m_source->GetCheckpointState();
            bool shouldCV = true;
            while (shouldCV)
            {
                GetCrossValidationMinibatch(minibatch, m_cv.m_mbSize[totalNumberOfSamples], computeDevice);

                // TODO: it may be slow to rely on TestMinibatch to return error each time, since it may require transfer
                // of error from the GPU each time, accumulatedError can be allocated on GPU
                shouldCV = m_trainer->TestMinibatch(minibatch, errorAndCount, computeDevice, m_numberOfWorkers != 1);
                if (shouldCV)
                {
                    accumulatedError += errorAndCount.first->AsScalar<double>();
                    totalNumberOfSamples += errorAndCount.second;
                    numberOfMinibatches++;
                }
            }

            m_cv.m_source->RestoreFromCheckpoint(checkpoint);
            Trainer()->SummarizeTestProgress();
            return OnCrossValidationEnd(currentIndex, accumulatedError / totalNumberOfSamples, totalNumberOfSamples, numberOfMinibatches);
        }
        else // Only invoking the callback.
        {
            return OnCrossValidationEnd(currentIndex, 0, 0, 0);
        }
    }

    void TrainingSession::Test(const DeviceDescriptor& computeDevice)
    {
        if (!m_test.m_source)
            return;

        std::unordered_map<Variable, ValuePtr> minibatch;
        size_t totalNumberOfSamples = 0;
        bool shouldTest = true;
        std::pair<ValuePtr, size_t> errorAndCount;
        while (shouldTest)
        {
            GetNextMinibatch(m_test.m_source, minibatch, m_test.m_mbSize[totalNumberOfSamples], m_workerRank, m_numberOfWorkers, computeDevice);
            shouldTest = m_trainer->TestMinibatch(minibatch, errorAndCount, computeDevice, m_numberOfWorkers != 1);
            totalNumberOfSamples += errorAndCount.second;
        }

        m_trainer->SummarizeTestProgress();
    }

    inline void TrainingSession::ReportProgress(size_t /*currentIndex*/)
    {
        Trainer()->SummarizeTrainingProgress();
    }

    void TrainingSession::GetTrainingMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        size_t workerRank = m_workerRank, numberOfWorkers = m_numberOfWorkers;

        // Check if we are operating in distributed mode.
        if (m_parallelAfterSamples > Trainer()->TotalNumberOfSamplesSeen())
        {
            numberOfWorkers = 1;
            workerRank = 0;
        }

        size_t mbSize = GetMinibatchSize();
        mbSize = std::min(mbSize, maxMbSize);
        GetNextMinibatch(m_source, minibatch, mbSize, workerRank, numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetCrossValidationMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        GetNextMinibatch(m_cv.m_source, minibatch, maxMbSize, m_workerRank, m_numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetNextMinibatch(const MinibatchSourcePtr& source, std::unordered_map<Variable, ValuePtr>& minibatch, size_t mbSize, size_t workerRank, size_t numberOfWorkers, const DeviceDescriptor& computeDevice)
    {
        minibatch.clear();

        if (mbSize == 0)
            return;

        // TODO: is copy really necessary here?
        auto minibatchData = source->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);
        if (minibatchData.empty())
            return;

        for (auto v : m_varToStream)
            minibatch.insert({ v.first, minibatchData[v.second].data });
    }

    void TrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = Trainer()->RestoreFromCheckpoint(checkpointFileName);
        m_source->RestoreFromCheckpoint(externalState[s_trainingMinibatchSource].Value<Dictionary>());
    }

    void TrainingSession::SaveCheckpoint(size_t currentIndex)
    {
        OnCheckpointStart(currentIndex);
        Dictionary externalState;
        externalState[s_trainingMinibatchSource] = m_source->GetCheckpointState();

        wstring checkpointFile = m_checkpoint.m_fileName;
        if (m_checkpoint.m_preserveAll)
            checkpointFile += std::to_wstring(currentIndex);
        Trainer()->SaveCheckpoint(checkpointFile, externalState);
        OnCheckpointEnd(currentIndex);
    }

    void TrainingSession::SaveFinalCheckpoint()
    {
        Dictionary externalState;
        externalState[s_trainingMinibatchSource] = m_source->GetCheckpointState();
        Trainer()->SaveCheckpoint(m_checkpoint.m_fileName, externalState);
    }

    // Restores from a m_checkPointFileName file.
    // If the file path exists - simply restores from the corresponding file.
    // If the file path does not exist - looks into directory where the file is
    // located and picks up the file with the largest N among <m_checkPointFileName>N files,
    // Where N is some positive integer.
    void TrainingSession::RestoreFromCheckpoint()
    {
        assert(!m_checkpoint.m_fileName.empty());
        auto checkpoint = m_checkpoint.m_fileName;

        // Make sure the intermediate directories exist, so no need for further checks.
        msra::files::make_intermediate_dirs(checkpoint);

        size_t pos = checkpoint.find_last_of(L"\\/");
        wstring parent;
        wstring fileName;
        if (pos == wstring::npos)
        {
            parent = L".";
            fileName = checkpoint;
        }
        else
        {
            parent = checkpoint.substr(0, pos);
            fileName = checkpoint.substr(pos + 1);
        }

        std::wstring restoreFile;
        if (fexists(checkpoint))
        {
            restoreFile = checkpoint;
        }
        else
        {
            // let's check whether there are other possible candidates to restore from.
            int maxValue = -1;
            std::vector<std::wstring> files = msra::files::get_all_files_from_directory(parent);

            for (auto f : files)
            {
                if (!boost::starts_with(f, fileName))
                {
                    continue;
                }

                auto suffix = f.substr(fileName.size());
                if (!isNumber(suffix) || !fexists(parent + L"/" + f + L".ckp"))
                {
                    continue;
                }

                auto expectedNumber = msra::strfun::utf8(suffix);
                char* tmp = nullptr;
                int value = strtol(expectedNumber.c_str(), &tmp, 10);
                if (tmp != expectedNumber.c_str() + expectedNumber.size())
                    continue;

                if (value > maxValue)
                {
                    // Found a better candidate.
                    maxValue = value;
                    restoreFile = parent + L"/" + f;
                }
            }
        }

        if (restoreFile.empty()) // Nothing to restore.
            return;

        // TODO: Should have proper loggin instead.
        fprintf(stderr, "Restoring training session from the checkpoint '%ls'\n", restoreFile.c_str());

        this->RestoreFromCheckpoint(restoreFile);

        // Recalculate actions indicies.
        size_t totalNumberOfSamples = Trainer()->TotalNumberOfSamplesSeen();
        for (auto& action : m_actions)
        {
            action.currentIndex = totalNumberOfSamples / action.frequency;
            action.sampleCountWhenLastCalled = totalNumberOfSamples - totalNumberOfSamples % action.frequency;
        }
    }
}
