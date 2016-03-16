//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "BlockRandomizer.h"
#include <algorithm>
#include <utility>
#include <deque>

#include "DataReader.h"
#include <random>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

BlockRandomizer::BlockRandomizer(
    int verbosity,
    size_t randomizationRangeInSamples,
    IDataDeserializerPtr deserializer,
    DecimationMode decimationMode,
    bool useLegacyRandomization)
    : m_verbosity(verbosity),
      m_deserializer(deserializer),
      m_decimationMode(decimationMode),
      m_sweep(SIZE_MAX),
      m_epochSize(SIZE_MAX),
      m_globalSamplePosition(SIZE_MAX),
      m_epochStartPosition(0),
      m_sweepTotalNumberOfSamples(0),
      m_lastSeenChunkId(SIZE_MAX),
      m_chunkRandomizer(std::make_shared<ChunkRandomizer>(deserializer, randomizationRangeInSamples, useLegacyRandomization))
{
    assert(deserializer != nullptr);

    m_streams = m_deserializer->GetStreamDescriptions();
    m_sequenceRandomizer = std::make_shared<SequenceRandomizer>(m_deserializer, m_chunkRandomizer);

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
}

// Prepares a new sweep if needed.
void BlockRandomizer::PrepareNewSweepIfNeeded(size_t samplePosition)
{
    size_t sweep = samplePosition / m_sweepTotalNumberOfSamples;
    if (m_sweep != sweep)
    {
        m_sweep = sweep;
        m_sweepStartInSamples = sweep * m_sweepTotalNumberOfSamples;

        // Rerandomizing the chunks.
        m_chunkRandomizer->Randomize((unsigned int)m_sweep);

        // Resetting seqeunce randomizer.
        m_sequenceRandomizer->Reset(m_sweep + 1);

        // Unloading all chunk data from memory.
        m_chunks.clear();
        m_lastSeenChunkId = SIZE_MAX;
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

    // Decimate.
    std::vector<RandomizedSequenceDescription> decimated;
    decimated.reserve(sequences.size());
    Decimate(sequences, decimated);
    if (decimated.size() == 0)
    {
        return result;
    }

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(decimated.size()));

    // TODO: This will be changed, when we move transformers under the randomizer.
    // TODO: Randomizer won't should not deal with multithreading.
#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < decimated.size(); ++i)
    {
        const auto& description = decimated[i];
        std::vector<SequenceDataPtr> sequence;
        auto it = m_chunks.find(description.m_chunk->m_chunkId);
        if (it == m_chunks.end())
        {
            LogicError("Invalid chunk requested.");
        }

        it->second->GetSequence(description.m_id, sequence);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    }

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
    // Swap remove all old chunks and add new ones.
    // Require all data in chunks.
    RetrieveDataChunks();

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

// Retrives chunk data based on the window information provided by SequenceRandomizer
void BlockRandomizer::RetrieveDataChunks()
{
    const auto& window = m_sequenceRandomizer->GetChunkWindow();
    if (window.back().m_chunkId == m_lastSeenChunkId)
    {
        return; // nothing to retrieve.
    }

    m_lastSeenChunkId = window.back().m_chunkId;

    // in the loop we are building a new map of currently loaded chunks:
    // we are iterating thru all chunks in the window and if they are not in m_chunks map - 
    // they get requested from the deserializer.
    // There could be some chunks in the m_chunks that are not required anymore, by swapping the chunks with m_chunks, we are removing those.
    std::map<size_t, ChunkPtr> chunks;
    for (auto const& chunk : window)
    {
        if (m_decimationMode == DecimationMode::chunk && chunk.m_chunkId % m_config.m_numberOfWorkers != m_config.m_workerRank)
        {
            continue;
        }

        auto it = m_chunks.find(chunk.m_chunkId);
        if (it != m_chunks.end())
        {
            chunks[chunk.m_chunkId] = it->second;
        }
        else
        {
            chunks[chunk.m_chunkId] = m_deserializer->GetChunk(chunk.m_original->m_id);
        }
    }

    // Swapping current chunks in the m_chunks, by that removing all stale and remembering newly loaded.
    m_chunks.swap(chunks);
}

}}}
