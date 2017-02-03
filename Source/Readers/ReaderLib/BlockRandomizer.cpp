//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "BlockRandomizer.h"
#include <algorithm>
#include <utility>

#include "DataReader.h"
#include "ExceptionCapture.h"

namespace Microsoft { namespace MSR { namespace CNTK {

BlockRandomizer::BlockRandomizer(
    int verbosity,
    size_t randomizationRangeInSamples,
    IDataDeserializerPtr deserializer,
    bool shouldPrefetch,
    bool multithreadedGetNextSequence,
    size_t maxNumberOfInvalidSequences)
    : m_verbosity(verbosity),
      m_deserializer(deserializer),
      m_sweep(SIZE_MAX),
      m_epochSize(SIZE_MAX),
      m_globalSamplePosition(SIZE_MAX),
      m_epochStartPosition(0),
      m_sweepSizeInSamples(0),
      m_chunkRandomizer(std::make_shared<ChunkRandomizer>(deserializer, randomizationRangeInSamples)),
      m_multithreadedGetNextSequences(multithreadedGetNextSequence),
      m_prefetchedChunk(CHUNKID_MAX),
      m_cleaner(maxNumberOfInvalidSequences)
{
    assert(deserializer != nullptr);

    m_launchType = shouldPrefetch ? launch::async : launch::deferred;

    m_streams = m_deserializer->GetStreamDescriptions();
    m_sequenceRandomizer = std::make_shared<SequenceRandomizer>(verbosity, m_deserializer, m_chunkRandomizer);

    // Calculate total number of samples.
    m_sweepSizeInSamples = 0;
    for (auto const & chunk : m_deserializer->GetChunkDescriptions())
    {
        m_sweepSizeInSamples += chunk->m_numberOfSamples;
    }
}

size_t BlockRandomizer::GetCurrentSamplePosition()
{
    return m_globalSamplePosition;
}

// Start a new epoch.
void BlockRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_currentWindowRange = ClosedOpenChunkInterval{};

    m_config = config;
    if (config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_epochSize = m_sweepSizeInSamples;
    }
    else
    {
        m_epochSize = config.m_totalEpochSizeInSamples;
    }

    // Sanity check, too big values can cause invalid behavior due to overflow.
    if (m_epochSize > std::numeric_limits<size_t>::max() / 2)
        InvalidArgument("Too big epoch size can cause bit overflow");

    m_epochStartPosition = m_epochSize * config.m_epochIndex;
    SetCurrentSamplePosition(m_epochStartPosition);
    if (m_verbosity >= Notification)
    {
        size_t epochStartFrame = config.m_epochIndex * m_epochSize;
        fprintf(stderr, "BlockRandomizer::StartEpoch: epoch %" PRIu64 ": samples [%" PRIu64 "..%" PRIu64 "] (first sequence at sample %" PRIu64 "), worker rank %" PRIu64 ", total workers %" PRIu64 "\n",
                config.m_epochIndex + 1,
                epochStartFrame,
                epochStartFrame + m_epochSize,
                m_globalSamplePosition,
                config.m_workerRank,
                config.m_numberOfWorkers);
    }
}

// Prepares a new sweep if needed.
void BlockRandomizer::PrepareNewSweepIfNeeded(size_t samplePosition)
{
    size_t sweep = samplePosition / m_sweepSizeInSamples;
    if (m_sweep != sweep)
    {
        if (m_verbosity >= Notification)
            fprintf(stderr, "BlockRandomizer::PrepareNewSweepIfNeeded: re-randomizing for sweep %d\n",
                    (int) sweep);

        m_sweep = sweep;

        // Rerandomizing the chunks.
        m_chunkRandomizer->Randomize((unsigned int)m_sweep);

        // Resetting sequence randomizer.
        m_sequenceRandomizer->Reset(m_sweep);
        m_currentWindowRange = {};
    }
}

// Gets next sequences not exceeding global and local sample counts.
Sequences BlockRandomizer::GetNextSequences(size_t globalSampleCount, size_t localSampleCount)
{
    // Get next sequence descriptions.
    Sequences result;
    size_t numGlobalSamplesLoaded = 0, numLocalSamplesLoaded = 0;
    do
    {
        assert(globalSampleCount > numGlobalSamplesLoaded && localSampleCount > numLocalSamplesLoaded);
        bool atTheSweepBoundary = result.m_endOfSweep;
        // in case when we continue filling up a minibatch that crosses a sweep boundary, 
        // make sure that it does not exceed the required number of samples. Set the atLeastOnceSequenceNeeded
        // flag to false.
        size_t numGlobalSamples = 0, numLocalSamples = 0;
        std::tie(numGlobalSamples, numLocalSamples) = 
            LoadSequenceData(globalSampleCount - numGlobalSamplesLoaded, 
                             localSampleCount - numLocalSamplesLoaded,
                             result, !atTheSweepBoundary);

        if (atTheSweepBoundary && numGlobalSamples == 0)
        {
            break;
        }

        numGlobalSamplesLoaded += numGlobalSamples;
        numLocalSamplesLoaded += numLocalSamples;

    } while (m_config.m_allowMinibatchesToCrossSweepBoundaries && 
             !result.m_endOfEpoch &&
             result.m_endOfSweep &&
             globalSampleCount > numGlobalSamplesLoaded &&
             localSampleCount > numLocalSamplesLoaded);

    m_cleaner.Clean(result);

    return result;
}

std::pair<size_t, size_t> BlockRandomizer::LoadSequenceData(size_t globalSampleCount, size_t localSampleCount, Sequences& sequences, bool atLeastOneSequenceNeeded)
{
    ClosedOpenChunkInterval windowRange;    
    m_sequenceBuffer.clear();
    size_t numGlobalSamples = 0, numLocalSamples = 0; // actual number of samples to load (filled in from the sequence descriptions) 
    bool endOfSweep, endOfEpoch;
    std::tie(endOfSweep, endOfEpoch, numGlobalSamples, numLocalSamples) = GetNextSequenceDescriptions(globalSampleCount, localSampleCount, m_sequenceBuffer, windowRange, atLeastOneSequenceNeeded);
    sequences.m_endOfSweep |= endOfSweep;
    sequences.m_endOfEpoch |= endOfEpoch;
    
    assert(atLeastOneSequenceNeeded || (numGlobalSamples <= globalSampleCount && numLocalSamples <= localSampleCount));

    if (numGlobalSamples == 0)
    {
        assert(!atLeastOneSequenceNeeded || sequences.m_endOfEpoch);
        return {0, 0};
    }

    // Retrieve new data chunks if required.
    LoadDataChunks(windowRange);

    auto& data = sequences.m_data;
    size_t offset = 0;

    if (data.empty())
    {
        data.resize(m_streams.size(), std::vector<SequenceDataPtr>(m_sequenceBuffer.size()));
    }
    else
    {
        // sequence data is not empty, we're appending new items to exiting 
        // sequence data vectors.
        offset = data.front().size();
        for (auto& sequenceDataVector : data)
        {
            // make sure that all streams contain the same number of sequences
            assert(sequenceDataVector.size() == offset); 
            sequenceDataVector.resize(offset + m_sequenceBuffer.size());
        }
    }

    auto process = [&](int i) -> void {
        const auto& description = m_sequenceBuffer[i];
        std::vector<SequenceDataPtr> sequenceData;
        auto it = m_chunks.find(description.m_chunk->m_original->m_id);
        if (it == m_chunks.end())
        {
            LogicError("Invalid chunk requested.");
        }

        it->second->GetSequence(description.m_id, sequenceData);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            assert(offset + i < data[j].size());
            data[j][offset + i] = sequenceData[j];
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

    // Now it is safe to start the new chunk prefetch.
    ChunkIdType chunkToPrefetchNext = GetChunkToPrefetch(windowRange);
    Prefetch(chunkToPrefetchNext);

    return { numGlobalSamples, numLocalSamples };
}

// Get next sequence descriptions for that worker that do not exceed global and local sample count.
// Returns true if epoch end is reached.
std::tuple<bool, bool, size_t, size_t> BlockRandomizer::GetNextSequenceDescriptions(size_t globalSampleCount, size_t localSampleCount, std::vector<RandomizedSequenceDescription>& result, ClosedOpenChunkInterval& windowRange, bool atLeastOneSequenceNeeded)
{
    if (globalSampleCount == 0)
        LogicError("Global sample count must not be zero.");

    if (localSampleCount == 0)
        LogicError("Local sample count must not be zero.");

    PrepareNewSweepIfNeeded(m_globalSamplePosition);

    auto sweepPosition = m_globalSamplePosition % m_sweepSizeInSamples;
    auto epochEndPosition = m_epochSize + m_epochStartPosition;

    // Check epoch end.
    if (m_globalSamplePosition >= epochEndPosition)
    {
        auto reachedEndOfEpoch = true;
        auto reachedEndOfSweep = (m_globalSamplePosition >= m_sweepSizeInSamples) && (sweepPosition == 0);
        return std::make_tuple(reachedEndOfSweep, reachedEndOfEpoch, 0, 0);
    }

    // Global sample count should not exceed the epoch.
    globalSampleCount = std::min(globalSampleCount, epochEndPosition - m_globalSamplePosition);

    // Global sample count should also not exceed the sweep.
    globalSampleCount = std::min(globalSampleCount, m_sweepSizeInSamples - sweepPosition);

    if (globalSampleCount == 0)
        LogicError("Global sample count must not result in zero.");

    std::function<bool(const RandomizedSequenceDescription*)> isLocalSequence =
        [this](const RandomizedSequenceDescription* s) { return s->m_chunk->m_chunkId % m_config.m_numberOfWorkers == m_config.m_workerRank; };

    size_t actualNumberOfGlobalSamples = 0, actualNumberOfLocalSamples = 0;
    std::tie(actualNumberOfGlobalSamples, actualNumberOfLocalSamples) = m_sequenceRandomizer->GetNextSequenceDescriptions(
        globalSampleCount,
        localSampleCount,
        isLocalSequence,
        windowRange,
        result,
        atLeastOneSequenceNeeded);

    if (actualNumberOfLocalSamples > actualNumberOfGlobalSamples)
        LogicError("Local sample count cannot be greater than the global sample count.");

    if (m_verbosity >= Debug)
        fprintf(stderr, "BlockRandomizer::GetNextSequenceDescriptions(): getting %" PRIu64 " sequences for %" PRIu64 "/%" PRIu64 " requested local/global samples in sweep %" PRIu64 "\n",
                result.size(),
                localSampleCount,
                globalSampleCount,
                m_sweep);

    // set "reachedEndOfSweep" to true if the minibatch is last in a sweep
    auto reachedEndOfSweep = (sweepPosition + actualNumberOfGlobalSamples >= m_sweepSizeInSamples);
    // set "reachedEndOfEpoch" to true if the current batch is last in an epoch.
    auto reachedEndOfEpoch = (m_globalSamplePosition + actualNumberOfGlobalSamples >= epochEndPosition);

    // Update the global sample position.
    m_globalSamplePosition += actualNumberOfGlobalSamples;

    return std::make_tuple(reachedEndOfSweep, reachedEndOfEpoch, actualNumberOfGlobalSamples, actualNumberOfLocalSamples);
}

// Retrieves chunk data based on the window information provided by SequenceRandomizer
void BlockRandomizer::LoadDataChunks(const ClosedOpenChunkInterval& windowRange)
{
    if (windowRange == m_currentWindowRange)
    {
        // Nothing to do.
        return;
    }

    m_currentWindowRange = windowRange;

    // in the loop we are building a new map of currently loaded chunks:
    // we are iterating thru all chunks in the window and if they are not in m_chunks map -
    // they get requested from the deserializer.
    // There could be some chunks in the m_chunks that are not required anymore, by swapping the chunks with m_chunks, we are removing those.
    std::map<size_t, ChunkPtr> chunks;
    size_t numLoadedChunks = m_chunks.size();

    std::vector<bool> needed;
    needed.resize(windowRange.Size(), false);

    // Firstly, make sure we unload all not needed chunks:
    for (size_t i = windowRange.m_begin; i < windowRange.m_end; ++i)
    {
        auto const& chunk = m_chunkRandomizer->GetRandomizedChunks()[i];
        if (chunk.m_chunkId % m_config.m_numberOfWorkers != m_config.m_workerRank)
        {
            continue;
        }

        auto it = m_chunks.find(chunk.m_original->m_id);
        if (it != m_chunks.end())
        {
            chunks[chunk.m_original->m_id] = it->second;
        }
        else
        {
            needed[i - windowRange.m_begin] = true;
        }
    }

    // Swapping current chunks in the m_chunks, by that removing all stale.
    // TODO diagnostics for paged out chunks?
    m_chunks.swap(chunks);

    // Adding new ones.
    for (size_t i = windowRange.m_begin; i < windowRange.m_end; ++i)
    {
        if (!needed[i - windowRange.m_begin])
        {
            continue;
        }

        auto const& chunk = m_chunkRandomizer->GetRandomizedChunks()[i];
        if (chunk.m_original->m_id == m_prefetchedChunk && m_prefetch.valid())
        {
            // Taking prefetched chunk.
            m_chunks[chunk.m_original->m_id] = m_prefetch.get();
            if (m_verbosity >= Information)
                fprintf(stderr, "BlockRandomizer::RetrieveDataChunks: paged in prefetched chunk %u (original chunk: %u), now %" PRIu64 " chunks in memory\n",
                chunk.m_chunkId,
                chunk.m_original->m_id,
                ++numLoadedChunks);
        }
        else
        {
            // Make sure we have no outstanding prefetches.
            if (m_prefetch.valid())
            {
                m_prefetch.wait();
            }

            m_chunks[chunk.m_original->m_id] = m_deserializer->GetChunk(chunk.m_original->m_id);
            if (m_verbosity >= Information)
                fprintf(stderr, "BlockRandomizer::RetrieveDataChunks: paged in randomized chunk %u (original chunk: %u), now %" PRIu64 " chunks in memory\n",
                chunk.m_chunkId,
                chunk.m_original->m_id,
                ++numLoadedChunks);
        }
    }

    if (m_verbosity >= Notification)
        fprintf(stderr, "BlockRandomizer::RetrieveDataChunks: %" PRIu64 " chunks paged-in from chunk window [%u..%u]\n",
                m_chunks.size(),
                m_chunkRandomizer->GetRandomizedChunks()[windowRange.m_begin].m_chunkId,
                m_chunkRandomizer->GetRandomizedChunks()[windowRange.m_end - 1].m_chunkId);
}

// Identifies chunk id that should be prefetched.
ChunkIdType BlockRandomizer::GetChunkToPrefetch(const ClosedOpenChunkInterval& windowRange)
{
    ChunkIdType toBePrefetched = CHUNKID_MAX;
    auto current = windowRange.m_end;
    while (current < m_chunkRandomizer->GetRandomizedChunks().size())
    {
        const auto& chunk = m_chunkRandomizer->GetRandomizedChunks()[current];
        if (chunk.m_chunkId % m_config.m_numberOfWorkers == m_config.m_workerRank &&
            m_chunks.find(chunk.m_original->m_id) == m_chunks.end())
        {
            toBePrefetched = chunk.m_original->m_id;
            break;
        }
        ++current;
    }
    return toBePrefetched;
}

// Performs io prefetch of the specified chunk if needed.
void BlockRandomizer::Prefetch(ChunkIdType chunkId)
{
    // Start new prefetch if necessary.
    if (m_prefetchedChunk != chunkId && chunkId != CHUNKID_MAX)
    {
        // Wait to make sure there is no outstanding prefetches.
        if (m_prefetch.valid())
        {
            m_prefetch.wait();
        }

        m_prefetchedChunk = chunkId;
        m_prefetch = std::async(m_launchType, [this, chunkId]() { return m_deserializer->GetChunk(chunkId); });

        if (m_verbosity >= Debug)
            fprintf(stderr, "BlockRandomizer::Prefetch: prefetching original chunk: %u\n", chunkId);
    }
}

void BlockRandomizer::SetCurrentSamplePosition(size_t currentSamplePosition)
{
    PrepareNewSweepIfNeeded(currentSamplePosition);

    // Sets sequence cursor to the sequence that corresponds to the epoch start position.
    // If last epoch ended in the middle of a sequence, the cursor is moved to the next sequence in the sweep.
    size_t offsetInSweep = currentSamplePosition % m_sweepSizeInSamples;
    size_t newOffset = m_sequenceRandomizer->Seek(offsetInSweep, m_sweep);
    m_globalSamplePosition = m_sweep * m_sweepSizeInSamples + newOffset;

    // Check if we have some data, if not set to the end of epoch.
    if (m_config.m_workerRank >= m_chunkRandomizer->GetRandomizedChunks().size())
        m_globalSamplePosition = m_epochStartPosition + m_epochSize;
}

void BlockRandomizer::SetConfiguration(const ReaderConfiguration& config)
{
    *((ReaderConfiguration*)&m_config) = config;
}

}}}
