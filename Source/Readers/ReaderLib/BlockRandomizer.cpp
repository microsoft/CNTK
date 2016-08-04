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
#include <deque>

#include "DataReader.h"
#include "ExceptionCapture.h"

namespace Microsoft { namespace MSR { namespace CNTK {

BlockRandomizer::BlockRandomizer(
    int verbosity,
    size_t randomizationRangeInSamples,
    IDataDeserializerPtr deserializer,
    bool shouldPrefetch,
    DecimationMode decimationMode,
    bool useLegacyRandomization,
    bool multithreadedGetNextSequence)
    : m_verbosity(verbosity),
      m_deserializer(deserializer),
      m_decimationMode(decimationMode),
      m_sweep(SIZE_MAX),
      m_epochSize(SIZE_MAX),
      m_globalSamplePosition(SIZE_MAX),
      m_epochStartPosition(0),
      m_sweepTotalNumberOfSamples(0),
      m_lastSeenChunkId(CHUNKID_MAX),
      m_chunkRandomizer(std::make_shared<ChunkRandomizer>(deserializer, randomizationRangeInSamples, useLegacyRandomization)),
      m_multithreadedGetNextSequences(multithreadedGetNextSequence),
      m_prefetchedChunk(CHUNKID_MAX)
{
    assert(deserializer != nullptr);

    m_launchType = shouldPrefetch ? launch::async : launch::deferred;

    m_streams = m_deserializer->GetStreamDescriptions();
    m_sequenceRandomizer = std::make_shared<SequenceRandomizer>(verbosity, m_deserializer, m_chunkRandomizer);

    // Calculate total number of samples.
    m_sweepTotalNumberOfSamples = 0;
    for (auto const & chunk : m_deserializer->GetChunkDescriptions())
    {
        m_sweepTotalNumberOfSamples += chunk->m_numberOfSamples;
    }
}

// Start a new epoch.
void BlockRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_lastSeenChunkId = CHUNKID_MAX;

    m_config = config;
    if (config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_epochSize = m_sweepTotalNumberOfSamples;
    }
    else
    {
        m_epochSize = config.m_totalEpochSizeInSamples;
    }

    // Calculates starts of the epoch, prepares a new sweep if needed.
    m_epochStartPosition = m_epochSize * config.m_epochIndex;
    PrepareNewSweepIfNeeded(m_epochStartPosition);

    // Sets sequence cursor to the sequence that corresponds to the epoch start position.
    // If last epoch ended in the middle of a sequence, the cursor is moved to the next sequence in the sweep.
    size_t offsetInSweep = m_epochStartPosition % m_sweepTotalNumberOfSamples;
    size_t newOffset = m_sequenceRandomizer->Seek(offsetInSweep, m_sweep);
    m_globalSamplePosition = m_sweep * m_sweepTotalNumberOfSamples + newOffset;

    size_t epochStartFrame = config.m_epochIndex * m_epochSize;
    fprintf(stderr, "BlockRandomizer::StartEpoch: epoch %" PRIu64 ": frames [%" PRIu64 "..%" PRIu64 "] (first sequence at sample %" PRIu64 "), data subset %" PRIu64 " of %" PRIu64 "\n",
            config.m_epochIndex,
            epochStartFrame,
            epochStartFrame + m_epochSize,
            m_globalSamplePosition,
            config.m_workerRank,
            config.m_numberOfWorkers);
}

// Prepares a new sweep if needed.
void BlockRandomizer::PrepareNewSweepIfNeeded(size_t samplePosition)
{
    size_t sweep = samplePosition / m_sweepTotalNumberOfSamples;
    if (m_sweep != sweep)
    {
        if (m_verbosity >= Notification)
            fprintf(stderr, "BlockRandomizer::PrepareNewSweepIfNeeded: re-randomizing for sweep %d\n",
                    (int) sweep);

        m_sweep = sweep;
        m_sweepStartInSamples = sweep * m_sweepTotalNumberOfSamples;

        // Rerandomizing the chunks.
        m_chunkRandomizer->Randomize((unsigned int)m_sweep);

        // Resetting sequence randomizer.
        m_sequenceRandomizer->Reset(m_sweep + 1);
        m_lastSeenChunkId = CHUNKID_MAX;
    }
}

// Gets next sequences not exceeding sampleCount.
Sequences BlockRandomizer::GetNextSequences(size_t sampleCount)
{
    // Get next sequence descriptions.
    Sequences result;
    std::vector<RandomizedSequenceDescription> sequences;
    result.m_endOfEpoch = GetNextSequenceDescriptions(sampleCount, sequences);
    if (sequences.size() == 0)
    {
        return result;
    }

    // Decimate sequences.
    std::vector<RandomizedSequenceDescription> decimated;
    decimated.reserve(sequences.size());
    Decimate(sequences, decimated);
    if (decimated.size() == 0)
    {
        return result;
    }

    // Retrieve new data chunks if required.
    ChunkIdType chunkToPrefetchNext = LoadDataChunks();

    if (m_verbosity >= Debug)
        fprintf(stderr, "BlockRandomizer::GetNextSequences(): getting %" PRIu64 " out of %" PRIu64 " sequences for %" PRIu64 " requested samples in sweep %" PRIu64 "\n",
            sequences.size(),
            decimated.size(),
            sampleCount,
            m_sweep);

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(decimated.size()));

    auto process = [&](int i) -> void {
        const auto& description = decimated[i];
        std::vector<SequenceDataPtr> sequence;
        auto it = m_chunks.find(description.m_chunk->m_original->m_id);
        if (it == m_chunks.end())
        {
            LogicError("Invalid chunk requested.");
        }

        it->second->GetSequence(description.m_id, sequence);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    };

    if (m_multithreadedGetNextSequences)
    {
        ExceptionCapture capture;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < decimated.size(); ++i)
            capture.SafeRun(process, i);
        capture.RethrowIfHappened();
    }
    else
    {
        for (int i = 0; i < decimated.size(); ++i)
            process(i);
    }

    // Explicitly release chunks that are not needed anymore.
    m_sequenceRandomizer->ReleaseChunks();

    // Now it is safe to start the new chunk prefetch.
    Prefetch(chunkToPrefetchNext);

    return result;
}

// Get next sequence descriptions that do not exceed sample count.
// Returns true if epoch end is reached.
bool BlockRandomizer::GetNextSequenceDescriptions(size_t sampleCount, std::vector<RandomizedSequenceDescription>& result)
{
    assert(sampleCount != 0);

    PrepareNewSweepIfNeeded(m_globalSamplePosition);

    // Check epoch end.
    if (m_globalSamplePosition >= m_epochSize + m_epochStartPosition)
    {
        return true;
    }

    sampleCount = std::min(sampleCount, m_epochSize + m_epochStartPosition - m_globalSamplePosition);
    assert(sampleCount != 0);

    // Check that we do not go over the sweep.
    sampleCount = std::min(sampleCount, (long)m_sweepTotalNumberOfSamples - m_globalSamplePosition % m_sweepTotalNumberOfSamples);
    assert(sampleCount != 0);

    // Randomizing sequences
    result = m_sequenceRandomizer->GetNextSequenceDescriptions(sampleCount);
    return false;
}

// Decimates sequences and load/unloads chunks using infromation of the SequenceRandomizer.
void BlockRandomizer::Decimate(const std::vector<RandomizedSequenceDescription>& all, std::vector<RandomizedSequenceDescription>& decimated)
{
    // Moving the cursor to the end of read sequences.
    for (const auto& sequence : all)
    {
        m_globalSamplePosition += sequence.m_numberOfSamples;
    }

    decimated.reserve(all.size());
    if (m_decimationMode == DecimationMode::chunk)
    {
        for (const auto& sequence : all)
        {
            if (sequence.m_chunk->m_chunkId % m_config.m_numberOfWorkers == m_config.m_workerRank)
            {
                decimated.push_back(sequence);
            }
        }
    }
    // TODO: This mode should go away. Decimation based on chunks only should be sufficient.
    // Currently this mode is used only for image reader, which uses one chunk for each image.
    else if (m_decimationMode == DecimationMode::sequence)
    {
        size_t strideBegin = all.size() * m_config.m_workerRank / m_config.m_numberOfWorkers;
        size_t strideEnd = all.size() * (m_config.m_workerRank + 1) / m_config.m_numberOfWorkers;
        decimated.assign(all.begin() + strideBegin, all.begin() + strideEnd);
    }
    else
    {
        LogicError("Not supported mode.");
    }
}

// Retrieves chunk data based on the window information provided by SequenceRandomizer
// Returns the next chunk id to prefetch.
ChunkIdType BlockRandomizer::LoadDataChunks()
{
    size_t randomizedEnd = 0;
    const auto& window = m_sequenceRandomizer->GetChunkWindow(randomizedEnd);
    if (window[randomizedEnd - 1].m_chunkId == m_lastSeenChunkId)
    {
        // nothing to prefetch.
        return CHUNKID_MAX;
    }

    m_lastSeenChunkId = window[randomizedEnd - 1].m_chunkId;

    // in the loop we are building a new map of currently loaded chunks:
    // we are iterating thru all chunks in the window and if they are not in m_chunks map -
    // they get requested from the deserializer.
    // There could be some chunks in the m_chunks that are not required anymore, by swapping the chunks with m_chunks, we are removing those.
    std::map<size_t, ChunkPtr> chunks;
    size_t numLoadedChunks = m_chunks.size();

    std::vector<bool> needed;
    needed.resize(randomizedEnd, false);

    // Firstly, make sure we unload all not needed chunks:
    for (size_t i = 0; i < randomizedEnd; ++i)
    {
        auto const& chunk = window[i];
        if (m_decimationMode == DecimationMode::chunk && chunk.m_chunkId % m_config.m_numberOfWorkers != m_config.m_workerRank)
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
            needed[i] = true;
        }
    }

    // Swapping current chunks in the m_chunks, by that removing all stale.
    // TODO diagnostics for paged out chunks?
    m_chunks.swap(chunks);

    // Adding new ones.
    for (size_t i = 0; i < randomizedEnd; ++i)
    {
        if (!needed[i])
        {
            continue;
        }

        auto const& chunk = window[i];
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
                window.front().m_chunkId,
                window.back().m_chunkId);

    return GetChunkToPrefetch(window.begin() + randomizedEnd, window.end());
}

// Identifies chunk id that should be prefetched.
// TODO: DecimationMode::sequence is not supported because it should eventually go away.
template<class Iter>
ChunkIdType BlockRandomizer::GetChunkToPrefetch(const Iter& begin, const Iter& end)
{
    auto current = begin;
    ChunkIdType toBePrefetched = CHUNKID_MAX;
    while (current != end)
    {
        if (m_chunks.find(current->m_original->m_id) == m_chunks.end() &&
            m_decimationMode == DecimationMode::chunk && 
            current->m_chunkId % m_config.m_numberOfWorkers == m_config.m_workerRank)
        {
            toBePrefetched = current->m_original->m_id;
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

}}}
