//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "LocalTimelineRandomizerBase.h"
#include "DataReader.h"
#include "ExceptionCapture.h"
#include "RandomOrdering.h"

namespace CNTK {

const SequenceDescription LocalTimelineRandomizerBase::s_endOfSweep = { std::numeric_limits<size_t>::max(), std::numeric_limits<unsigned>::max(), std::numeric_limits<ChunkIdType>::max() };

LocalTimelineRandomizerBase::LocalTimelineRandomizerBase(
    DataDeserializerPtr deserializer,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
: m_deserializer(deserializer),
  m_multithreadedGetNextSequences(multithreadedGetNextSequences),
  m_cleaner(maxNumberOfInvalidSequences),
  m_sweepIndex(0),
  m_numberOfSamplesSeenSoFar(0),
  m_originalChunkDescriptions(deserializer->GetChunkDescriptions())
{
    if (m_originalChunkDescriptions.empty())
        RuntimeError("LocalTimelineRandomizerBase: Expected input to contain samples, but the number of successfully read samples was 0.");
}

void LocalTimelineRandomizerBase::StartEpoch(const EpochConfiguration& config)
{
    if(config.m_epochIndex != 0)
        RuntimeError("LocalTimelineRandomizerBase not supported for old configs.");

    m_config = config;
    if (config.m_totalEpochSizeInSweeps == g_infinity && m_config.m_totalEpochSizeInSamples == Microsoft::MSR::CNTK::requestDataSize)
        m_config.m_totalEpochSizeInSweeps = 1;

    if (config.m_totalEpochSizeInSweeps == g_infinity)
    {
        // Limit in global samples, make local sample limit.
        int shouldAddOneSample = (int)m_config.m_totalEpochSizeInSamples % m_config.m_numberOfWorkers > m_config.m_workerRank;
        m_config.m_totalEpochSizeInSamples = m_config.m_totalEpochSizeInSamples / m_config.m_numberOfWorkers + shouldAddOneSample;
    }

    Refill();
}

void LocalTimelineRandomizerBase::Refill()
{
    // Fill the first window remembering the state,
    // the window is expandable.
    m_currentState = GetInnerState();

    if (!m_prefetch.valid())
        Prefetch();
    m_prefetch.get();

    RefillSequenceWindow();

    Prefetch();
}

void LocalTimelineRandomizerBase::MoveToNextSequence()
{
    if (m_window.m_sequencePosition + 1 < m_window.m_sequences.size())
    {
        ++m_window.m_sequencePosition;
        return;
    }

    assert(m_window.m_sequencePosition + 1 == m_window.m_sequences.size());

    m_window.m_sequencePosition = 0;

    Refill();
}

// Gets next sequences not exceeding local and global samples.
void LocalTimelineRandomizerBase::GetNextSequenceDescriptions(size_t maxSampleCount, Sequences& result)
{
    assert(maxSampleCount != 0);

    if (maxSampleCount > std::numeric_limits<int>::max())
        RuntimeError("Local size of the minibatch cannot exceed max int.");

    if (m_window.m_sequences.empty())
        RuntimeError("Could not read data.");

    size_t samplesLoaded = 0;
    bool atLeastOneSequenceNeeded = true;

    m_sequenceBuffer.clear();
    m_chunkBuffer.clear();
    while (samplesLoaded < maxSampleCount && !IsEndReached())
    {
        const SequenceDescription& sequence = m_window.m_sequences[m_window.m_sequencePosition];
        if (IsEndOfSweep(sequence))
        {
            m_sweepIndex++;
            result.m_endOfSweep = true;
            MoveToNextSequence();
            continue;
        }

        auto sequenceLength = sequence.m_numberOfSamples;
        m_numberOfSamplesSeenSoFar += sequenceLength;

        // Break if we're exceeding the local requested sample count.
        if (!atLeastOneSequenceNeeded && samplesLoaded + sequenceLength > maxSampleCount)
            break;

        // Ok good to add it to the result.
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

Sequences LocalTimelineRandomizerBase::GetNextSequences(size_t, size_t sampleCount)
{
    if (sampleCount == 0)
        LogicError("Sample count must not be zero.");

    Sequences result;
    if (IsEndReached())
    {
        result.m_endOfEpoch = true;
        result.m_endOfSweep = false;

        // Make sure we do not issue prefetch when the end is reached.
        if(m_prefetch.valid())
            m_prefetch.wait_for(std::chrono::seconds(60));

        return result;
    }

    GetNextSequenceDescriptions(sampleCount, result);

    // Make sure we do not issue prefetch when the end is reached.
    if (IsEndReached() && m_prefetch.valid())
        m_prefetch.wait_for(std::chrono::seconds(60));

    if (m_sequenceBuffer.size() == 0)
        return result;

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

Dictionary LocalTimelineRandomizerBase::GetState()
{
    Dictionary state;
    state[L"sweepIndex"] = m_sweepIndex;
    state[L"currentSequencePositionInWindow"] = m_window.m_sequencePosition;
    state[L"numberOfSamplesSeenSoFar"] = m_numberOfSamplesSeenSoFar;
    state[L"innerState"] = m_currentState;
    return state;
}

inline size_t GetSizeTValue(const Dictionary& d, const std::wstring& key)
{
    return d[key].ValueType() == DictionaryValue::Type::Int ? d[key].Value<int>() : d[key].Value<size_t>();
}

void LocalTimelineRandomizerBase::SetState(const Dictionary& state)
{
    m_sweepIndex = ValueOf(state, L"sweepIndex");
    m_numberOfSamplesSeenSoFar = ValueOf(state, L"numberOfSamplesSeenSoFar");
    m_window.m_sequencePosition = ValueOf(state, L"currentSequencePositionInWindow");

    if (m_prefetch.valid())
        m_prefetch.get();
    SetInnerState(state[L"innerState"].Value<Dictionary>());

    Refill();
}

void LocalTimelineRandomizerBase::SetConfiguration(const ReaderConfiguration& config)
{
    *((ReaderConfiguration*)&m_config) = config;
}

}
