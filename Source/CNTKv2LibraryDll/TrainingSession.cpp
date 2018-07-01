//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <boost/algorithm/string/predicate.hpp>

#include "CNTKLibrary.h"
#include "Utils.h"
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

    inline static bool IsInfinite(MinibatchSourcePtr mbSource, size_t numberOfSamples = std::numeric_limits<size_t>::max())
    {
        return (numberOfSamples == MinibatchSource::InfinitelyRepeat ||
                numberOfSamples >= (size_t)std::numeric_limits<long long>::max()) &&
            mbSource->IsInfinite();
    }

    CheckpointConfig::CheckpointConfig(
        const std::wstring& checkPointFileName,
        size_t checkpointFrequency,
        DataUnit checkpointFrequencyUnit,
        bool restoreFromCheckpointIfExists,
        bool preserveAllCheckpoints) :
        m_preserveAll(preserveAllCheckpoints),
        m_restore(restoreFromCheckpointIfExists),
        m_fileName(checkPointFileName),
        m_frequency(checkpointFrequency),
        m_frequencyUnit(checkpointFrequencyUnit)
    {
        if (m_fileName.empty())
        {
            if (checkpointFrequency != 0 && checkpointFrequency != std::numeric_limits<size_t>::max())
                InvalidArgument("Checkpoint file name must not be empty if checkpoint frequency is non zero.");

            if (preserveAllCheckpoints)
                InvalidArgument("Checkpoint file name must not be empty if 'preserve all checkpoints' option is specified.");

            checkpointFrequency = 0;
        }
    }

    CrossValidationConfig::CrossValidationConfig(
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequency,
        DataUnit crossValidationFrequencyUnit,
        size_t maxSamples,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream):
        m_source(crossValidationSource),
        m_mbSize(crossValidationSchedule),
        m_frequency(crossValidationFrequency),
        m_frequencyUnit(crossValidationFrequencyUnit),
        m_maxSamples(maxSamples),
        m_varToStream(inputVarToStream)
    {
    }

    TestConfig::TestConfig(
        const MinibatchSourcePtr& source,
        const MinibatchSizeSchedule& schedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream) :
        m_source(source),
        m_mbSize(schedule),
        m_varToStream(inputVarToStream)
    {
    }
  
    CNTK_API TrainingSessionPtr CreateTrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        size_t maxNumTrainingSamples,
        size_t progressFrequency,
        DataUnit progressFrequencyUnit,
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
            progressFrequencyUnit,
            checkpointing, crossValidation, test);
    }

    TrainingSession::TrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        size_t maxNumTrainingSamples,
        size_t progressFrequency,
        DataUnit progressFrequencyUnit,
        const CheckpointConfig& checkpointing,
        const CrossValidationConfig& crossValidation,
        const TestConfig& test) :
        m_trainer(trainer),
        m_source(trainingSource),
        m_mbSize(minibatchSizeSchedule),
        m_varToStream(inputVarToStream),
        m_maxNumSamples(maxNumTrainingSamples),
        m_progressFrequency(progressFrequency),
        m_progressFrequencyUnit(progressFrequencyUnit),
        m_checkpoint(checkpointing),
        m_cv(crossValidation),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1),
        m_test(test),
        m_mbSizeScaleFactor(1)
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
                // This is always enforced now in the Learners class.
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
                m_mbSizeScaleFactor = distributed->MinibatchSizeScaleFactor();
            }
        }

        // Fill-in required actions.
        if (m_checkpoint.m_frequency != 0)
            m_actions.push_back({ m_checkpoint.m_frequency, m_checkpoint.m_frequencyUnit, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&)
                {
                    SaveCheckpoint(currentIndex);
                    // enable profiler after the first checkpoint
                    // This has effect only if the profiler is globally enabled by StartProfiler()
#ifndef CNTK_UWP
                    Microsoft::MSR::CNTK::ProfilerEnable(true);
#endif
                    return true;
                } });

        // Report progress before we run cross validation if any.
        if (m_progressFrequency != 0)
        {
            m_actions.push_back({ m_progressFrequency, m_progressFrequencyUnit, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&) { ReportProgress(currentIndex); return true; } });
        }

        if (m_cv.m_frequency != 0)
            m_actions.push_back({ m_cv.m_frequency, m_cv.m_frequencyUnit, 0, 0,
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

        if (IsInfinite(m_source, m_maxNumSamples))
            InvalidArgument("Train minibatch source must have a limited number of samples or sweeps.");

        // Main train loop.
        bool earlyExit = false;
        while (shouldTrain)
        {
            // Get next minibatch.
            size_t samplesLeft = earlyExit || m_maxNumSamples <= Trainer()->TotalNumberOfSamplesSeen()
                ? 0
                : m_maxNumSamples - Trainer()->TotalNumberOfSamplesSeen();

            //get the sweep end status from GetTrainingMinibatch and use in TrainMiniBatch below
            bool isMinibatchAtSweepEnd;
            // Note that in case of distributed training we don't want to stop if the local minibatch
            // is empty - it is possible that the other workers are still processing their minibatches.
            GetTrainingMinibatch(minibatch, &isMinibatchAtSweepEnd, samplesLeft, computeDevice);

            // Train on the minibatch.
            OnMinibatchStart();
            shouldTrain = Trainer()->TrainMinibatch(minibatch, isMinibatchAtSweepEnd, computeDevice);
            earlyExit |= !OnMinibatchEnd(); // If the callback wants to have early exit - we stop training.

#ifndef CNTK_UWP
            auto profMisc = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainPost);
#endif

            // Peform actions if required.
            for (auto& action : m_actions)
            {
                size_t totalNumberOfUnitCounts = Trainer()->TotalNumberOfUnitsSeen(action.frequencyUnit);

                size_t index = totalNumberOfUnitCounts / action.frequency;
                if (index != action.currentIndex)
                {
                    // If any action wants to have early exit - we stop training.
                    earlyExit |= !action.action(action.currentIndex, computeDevice);

                    action.currentIndex = index;
                    action.unitCountWhenLastCalled = totalNumberOfUnitCounts;
                }
            }
        }

        if (restoredNumberOfSamples != Trainer()->TotalNumberOfSamplesSeen())
        {
            // Let's do all actions on the last probably a partial data at the end.
            for (auto& action: m_actions)
            {
                size_t totalUnitCounts = Trainer()->TotalNumberOfUnitsSeen(action.frequencyUnit);
                if (totalUnitCounts % action.frequency != 0 &&
                    totalUnitCounts != action.unitCountWhenLastCalled)
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
        // Making sure we get the consistent state of the
        // training minibatch source in case of bptt.
        // When CV happens in the middle of the training, the packer can still has some truncated
        // sequences in the buffer. CV resets the state of the DelayedNode, so the first
        // training minibatch after CV will cause an exception.
        // Checkpoining currently drop intermediat buffers.
        // TODO: This is meant as a stop gap, the minibatch source should be properly drained instead.
        auto state = m_source->GetCheckpointState();

        bool result = false;
        if (m_cv.m_source) // Running cross validation
        {
            if (IsInfinite(m_cv.m_source, m_cv.m_maxSamples))
                InvalidArgument("Cross validation minibatch source must have a limited number of samples or sweeps.");

            std::unordered_map<Variable, ValuePtr> minibatch;
            double accumulatedError = 0;
            size_t totalNumberOfSamples = 0;
            size_t numberOfMinibatches = 0;

            std::pair<ValuePtr, size_t> errorAndCount;
            auto checkpoint = m_cv.m_source->GetCheckpointState();
            bool shouldCV = true;
            while (shouldCV)
            {
                size_t samplesLeft = m_cv.m_maxSamples <= totalNumberOfSamples ? 0 : m_cv.m_maxSamples - totalNumberOfSamples;
                GetCrossValidationMinibatch(minibatch, /*pIsMinibatchAtSweepEnd= */nullptr, (std::min)(m_cv.m_mbSize[totalNumberOfSamples], samplesLeft), computeDevice);

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
            result = OnCrossValidationEnd(currentIndex, accumulatedError / totalNumberOfSamples, totalNumberOfSamples, numberOfMinibatches);
        }
        else // Only invoking the callback.
        {
            result = OnCrossValidationEnd(currentIndex, 0, 0, 0);
        }

        m_source->RestoreFromCheckpoint(state);
        return result;
    }

    void TrainingSession::Test(const DeviceDescriptor& computeDevice)
    {
        if (!m_test.m_source)
            return;

        if (IsInfinite(m_test.m_source))
            InvalidArgument("Test minibatch source must have a limited number of samples or sweeps.");

        std::unordered_map<Variable, ValuePtr> minibatch;
        size_t totalNumberOfSamples = 0;
        bool shouldTest = true;
        std::pair<ValuePtr, size_t> errorAndCount;
        while (shouldTest)
        {
            GetNextMinibatch(m_test.m_source, minibatch, m_test.m_varToStream.empty() ? m_varToStream : m_test.m_varToStream,
                /*pIsMinibatchAtSweepEnd=*/ nullptr, m_test.m_mbSize[totalNumberOfSamples], m_workerRank, m_numberOfWorkers, computeDevice);
            shouldTest = m_trainer->TestMinibatch(minibatch, errorAndCount, computeDevice, m_numberOfWorkers != 1);
            totalNumberOfSamples += errorAndCount.second;
        }

        m_trainer->SummarizeTestProgress();
    }

    inline void TrainingSession::ReportProgress(size_t /*currentIndex*/)
    {
        Trainer()->SummarizeTrainingProgress();
    }

    void TrainingSession::GetTrainingMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, bool* pIsMinibatchAtSweepEnd, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        size_t workerRank = m_workerRank, numberOfWorkers = m_numberOfWorkers;

        // Check if we are operating in distributed mode.
        size_t scaleFactor = m_mbSizeScaleFactor;
        if (m_parallelAfterSamples > Trainer()->TotalNumberOfSamplesSeen())
        {
            numberOfWorkers = 1;
            workerRank = 0;
            scaleFactor = 1;
        }

        size_t mbSize = GetMinibatchSize() * scaleFactor;
        mbSize = (std::min)(mbSize, maxMbSize);
        GetNextMinibatch(m_source, minibatch, m_varToStream, pIsMinibatchAtSweepEnd, mbSize, workerRank, numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetCrossValidationMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, bool* pIsMinibatchAtSweepEnd, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        GetNextMinibatch(m_cv.m_source, minibatch, m_cv.m_varToStream.empty() ? m_varToStream : m_cv.m_varToStream, pIsMinibatchAtSweepEnd, maxMbSize, m_workerRank, m_numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetNextMinibatch(
        const MinibatchSourcePtr& source,
        std::unordered_map<Variable, ValuePtr>& minibatch,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        bool* pIsMinibatchAtSweepEnd,
        size_t mbSize,
        size_t workerRank,
        size_t numberOfWorkers,
        const DeviceDescriptor& computeDevice)
    {
        minibatch.clear();

        if (mbSize == 0)
            return;

        // TODO: is copy really necessary here?
        auto minibatchData = source->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);
        if (pIsMinibatchAtSweepEnd != nullptr)
            *pIsMinibatchAtSweepEnd = IsAtSweepEnd(minibatchData);
        if (minibatchData.empty())
            return;
        for (auto v : inputVarToStream)
        {
            auto value = minibatchData.find(v.second);
            if (value == minibatchData.end())
                RuntimeError("Minibatch source cannot find a stream with name '%ls'", v.second.m_name.c_str());
            minibatch.insert({ v.first, value->second.data });
        }
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
        for (auto& action : m_actions)
        {
            size_t totalNumberOfUnitCounts = Trainer()->TotalNumberOfUnitsSeen(action.frequencyUnit);

            action.currentIndex = totalNumberOfUnitCounts / action.frequency;
            action.unitCountWhenLastCalled = totalNumberOfUnitCounts - totalNumberOfUnitCounts % action.frequency;
        }
    }
}
