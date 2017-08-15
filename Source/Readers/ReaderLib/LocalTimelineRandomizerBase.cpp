//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "LocalTimelineRandomizerBase.h"
#include "DataReader.h"
#include "ExceptionCapture.h"

namespace CNTK {

const SequenceInfo LocalTimelineRandomizerBase::s_endOfSweep = { std::numeric_limits<size_t>::max(), std::numeric_limits<unsigned>::max(), std::numeric_limits<ChunkIdType>::max() };

LocalTimelineRandomizerBase::LocalTimelineRandomizerBase(
    DataDeserializerPtr deserializer,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
: m_deserializer(deserializer),
  m_multithreadedGetNextSequences(multithreadedGetNextSequences),
  m_cleaner(maxNumberOfInvalidSequences),
  m_sweepCount(0),
  m_sampleCount(0),
  m_originalChunkDescriptions(deserializer->ChunkInfos())
{
    if (m_originalChunkDescriptions.empty())
        RuntimeError("The deserializer does not have any data, the number of chunks is 0.");
}

void LocalTimelineRandomizerBase::StartEpoch(const EpochConfiguration& config)
{
    if(config.m_epochIndex != 0)
        LogicError("LocalTimelineRandomizerBase is not supported for old configs.");

    m_config = config;
    if (config.m_totalEpochSizeInSweeps == g_infinity && m_config.m_totalEpochSizeInSamples == Microsoft::MSR::CNTK::requestDataSize)
        m_config.m_totalEpochSizeInSweeps = 1;

    if (config.m_totalEpochSizeInSweeps == g_infinity)
    {
        // Convert global samples to local.
        int shouldAddOneSample = (int)m_config.m_totalEpochSizeInSamples % m_config.m_numberOfWorkers > m_config.m_workerRank;
        m_config.m_totalEpochSizeInSamples = m_config.m_totalEpochSizeInSamples / m_config.m_numberOfWorkers + shouldAddOneSample;
    }

    // Start filling the window.
    Refill();
}

void LocalTimelineRandomizerBase::Refill()
{
    // Fill the expandable window.
    // Because only the position in the window is stored in the checkpoint,
    // but not the window itself, we should preserve the current state of the child object.
    //
    // To support checkpointing we should preserve the following state:
    //     - current state of the base class
    //     - current window of sequences
    //     - current state of the inherited class.
    // The problem with this is that the window of sequences can be big.
    // Instead of saving it we can recalculate the window.

    // So in checkpoint we only store
    //   - current state of the base class
    //   - state of the inherited class before the current window is asked
    //   - position in current window

    m_currentState = GetInnerState();

    // Make sure there is no outstanding prefetch.
    if (!m_prefetch.valid())
        m_prefetch = std::async(std::launch::async, [this]() { Prefetch(); });

    m_prefetch.get();

    RefillSequenceWindow(m_window);

    // Issue the next prefetch
    m_prefetch = std::async(std::launch::async, [this]() { Prefetch(); });
}

void LocalTimelineRandomizerBase::MoveToNextSequence()
{
    const auto& s = m_window.m_sequences[m_window.m_sequencePosition];
    if (!IsEndOfSweep(s))
        m_sampleCount += s.m_numberOfSamples;

    ++m_window.m_sequencePosition;

    if (m_window.m_sequencePosition < m_window.m_sequences.size())
        return;

    // We are at the end of the window, let's get the new one.
    assert(m_window.m_sequencePosition == m_window.m_sequences.size());
    m_window.m_sequencePosition = 0;
    Refill();
}

// Gets next sequences not exceeding local and global samples.
void LocalTimelineRandomizerBase::GetNextSequenceDescriptions(size_t maxSampleCount, Sequences& result)
{
    assert(maxSampleCount != 0);

    if (maxSampleCount > std::numeric_limits<int>::max())
        RuntimeError("The size of a minibatch cannot exceed max int.");

    // The underlying randomizer should always fill data,
    // in case it cannot we report the error.
    if (m_window.m_sequences.empty()) 
        RuntimeError("Could not read any data.");

    size_t samplesLoaded = 0;
    bool atLeastOneSequenceNeeded = true;

    m_sequenceBuffer.clear();
    m_chunkBuffer.clear();
    while (samplesLoaded < maxSampleCount && !IsEndReached())
    {
        const SequenceInfo& sequence = m_window.m_sequences[m_window.m_sequencePosition];
        if (IsEndOfSweep(sequence))
        {
            m_sweepCount++;
            result.m_endOfSweep = true;
            MoveToNextSequence();
            continue;
        }

        auto sequenceLength = sequence.m_numberOfSamples;

        // Break if we're exceeding the requested sample count.
        if (!atLeastOneSequenceNeeded && samplesLoaded + sequenceLength > maxSampleCount)
            break;

        // Ok, the limit is not exceeded, add the sequence to the result.
        m_sequenceBuffer.push_back(sequence);
        if (m_chunkBuffer.find(sequence.m_chunkId) == m_chunkBuffer.end())
        {
            auto it = m_window.m_dataChunks.find(sequence.m_chunkId);
            if (it == m_window.m_dataChunks.end())
                RuntimeError("Cannot find the data for chunk");
            m_chunkBuffer[sequence.m_chunkId] = it->second;
        }

        samplesLoaded += sequenceLength;
        atLeastOneSequenceNeeded = false;

        // Moving to next sequence.
        MoveToNextSequence();
    }

    // Set the end-of-epoch flag (true when the current batch is last in an epoch).
    result.m_endOfEpoch = IsEndReached();
}

Sequences LocalTimelineRandomizerBase::GetNextSequences(size_t /*ignoring global sample count*/, size_t sampleCount)
{
    if (sampleCount == 0)
        LogicError("Sample count must not be zero.");

    Sequences result;
    if (IsEndReached())
    {
        result.m_endOfEpoch = true;
        result.m_endOfSweep = false;

        // Make sure we do not issue prefetch when the end is reached,
        // Let's wait for the prefetch to finish.
        // This is only important for Python deserializers, because at the end of the script
        // Python does not guarantees that destructors of global objects are called.
        // We do not want to call prefetch when the Python environment gets destroyed.
        if(m_prefetch.valid())
            m_prefetch.wait_for(std::chrono::seconds(60));

        return result;
    }

    GetNextSequenceDescriptions(sampleCount, result);

    // Make sure we do not issue prefetch when the end is reached.
    if (IsEndReached() && m_prefetch.valid())
    {
        // This is only important for Python deserializers, because at the end of the script
        // Python does not guarantees that destructors of global objects are called.
        // We do not want to call prefetch when the Python environment gets destroyed.
        m_prefetch.wait_for(std::chrono::seconds(60));
    }

    if (m_sequenceBuffer.size() == 0) // No data
        return result;

    // Lets actually fetch data.
    result.m_data.resize(GetStreamDescriptions().size(), std::vector<SequenceDataPtr>(m_sequenceBuffer.size()));
    auto process = [&](int i) -> void {
        std::vector<SequenceDataPtr> sequence;
        const auto& sequenceDescription = m_sequenceBuffer[i];

        auto it = m_chunkBuffer.find(sequenceDescription.m_chunkId);
        if (it == m_chunkBuffer.end())
            LogicError("Invalid chunk requested.");

        it->second->GetSequence(sequenceDescription.m_indexInChunk, sequence);
        for (int j = 0; j < GetStreamDescriptions().size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    };

    if (m_multithreadedGetNextSequences)
    {
        ExceptionCapture capture;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m_sequenceBuffer.size(); ++i)
            capture.SafeRun(process, i);
        capture.RethrowIfHappened();
    }
    else
    {
        for (int i = 0; i < m_sequenceBuffer.size(); ++i)
            process(i);
    }

    m_cleaner.Clean(result);
    return result;
}

// Properties used in the checkpoint.
const static std::wstring s_sweepIndexProperty = L"baseSweepCount";
const static std::wstring s_numberOfSamplesSeenSoFarProperty = L"baseSampleCount";
const static std::wstring s_sequencePositionProperty = L"baseCurrentSequencePositionInWindow";

std::map<std::wstring, size_t> LocalTimelineRandomizerBase::GetState()
{
    std::map<std::wstring, size_t> state;
    state[s_sweepIndexProperty] = m_sweepCount;
    state[s_sequencePositionProperty] = m_window.m_sequencePosition;
    state[s_numberOfSamplesSeenSoFarProperty] = m_sampleCount;
    size_t originalSize = state.size();
    state.insert(m_currentState.begin(), m_currentState.end());
    if (originalSize + m_currentState.size() != state.size())
        LogicError("Key collision during checkpointing. "
            "Make sure base and derived randomizers have different checkpoint fields.");
    return state;
}

void LocalTimelineRandomizerBase::SetState(const std::map<std::wstring, size_t>& state)
{
    m_sweepCount = ValueFrom(state, s_sweepIndexProperty);
    m_sampleCount = ValueFrom(state, s_numberOfSamplesSeenSoFarProperty);
    m_window.m_sequencePosition = ValueFrom(state, s_sequencePositionProperty);

    // Make sure, we invalidate the current prefetch.
    if (m_prefetch.valid())
        m_prefetch.get();

    SetInnerState(state);
    Refill();
}

}
