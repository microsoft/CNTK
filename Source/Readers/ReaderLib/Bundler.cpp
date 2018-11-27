//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "Bundler.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <set>
#include "SequenceData.h"

namespace CNTK {

Bundler::Bundler(
    const ConfigParameters& readerConfig,
    CorpusDescriptorPtr corpus,
    DataDeserializerPtr primaryDeserializer,
    std::vector<DataDeserializerPtr> deserializers,
    bool cleanse)
    : DataDeserializerBase(true),
      m_corpus(corpus),
      m_deserializers(deserializers),
      m_primaryDeserializer(primaryDeserializer),
      m_mbDefiningDeserializer(std::numeric_limits<size_t>::max())
{
    m_verbosity = readerConfig(L"verbosity", 0);

    // Combines streams of underlying deserializers.
    for (size_t j = 0; j < deserializers.size(); ++j)
    {
        auto d = deserializers[j];
        for (auto i : d->StreamInfos())
        {
            StreamInformation stream = i;
            stream.m_id = m_streams.size();
            if (stream.m_definesMbSize)
            {
                if (m_mbDefiningDeserializer != std::numeric_limits<size_t>::max())
                    RuntimeError("Only a single deserializer is allowed to define minibatch size, at least two found.");
                m_mbDefiningDeserializer = j;
            }
            m_streams.push_back(stream);
        }
    }

    m_cleanse = cleanse;
    CreateChunkDescriptions();
}

// Creates chunk descriptions based on chunks of underlying deserializers.
void Bundler::CreateChunkDescriptions()
{
    if (m_verbosity)
        fprintf(stderr, "Bundler::CreateChunkDescriptions(): started\n");

    auto chunks = m_primaryDeserializer->ChunkInfos();
    if (chunks.size() < 1)
    {
        RuntimeError("Driving deserializer should at least provide one chunk.");
    }
    if (ChunkIdMax < chunks.size())
    {
        RuntimeError("Driving deserializer provided too many chunks.");
    }

    assert(m_mbDefiningDeserializer == std::numeric_limits<size_t>::max() || m_mbDefiningDeserializer < m_deserializers.size());

    // Creating a table of weak chunks for non driving deserializers.
    for (size_t i = 0; i < m_deserializers.size(); ++i)
    {
        m_weakChunkTable.push_back(std::vector<std::weak_ptr<Chunk>>(m_deserializers[i]->ChunkInfos().size()));
    }

    m_chunks.reserve(chunks.size());

    if (m_verbosity)
    {
        fprintf(stderr, "Bundler::CreateChunkDescriptions(): creating descriptions for %" PRIu64 " chunks\n", chunks.size());
        fprintf(stderr, "Bundler::CreateChunkDescriptions(): starting to clean chunks\n");
    }

    m_takePrimarySequenceLength = true;

    // Build bundling chunks using underlying deserializers.
    std::vector<SequenceInfo> sequenceDescriptions;
    sequenceDescriptions.reserve(chunks.front().m_numberOfSequences);
    SequenceInfo s;

    for (ChunkIdType chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
    {
        size_t numberOfSamples = 0;
        size_t numberOfSequences = 0;
        sequenceDescriptions.clear();

        // Iterating thru all sequences and identifying whether they are valid among all deserializers.
        m_primaryDeserializer->SequenceInfosForChunk(chunks[chunkIndex].m_id, sequenceDescriptions);
        std::set<size_t> invalid;

        // Also remember necessary secondary chunks.
        std::vector<std::vector<ChunkIdType>> secondaryChunks;
        secondaryChunks.resize(m_deserializers.size());
        secondaryChunks[0].push_back(chunks[chunkIndex].m_id);
        for (size_t sequenceIndex = 0; sequenceIndex < sequenceDescriptions.size(); ++sequenceIndex)
        {
            auto sequence = sequenceDescriptions[sequenceIndex];
            bool isValid = true;
            size_t sequenceSamples = sequence.m_numberOfSamples;

            // Need to check the sequence length for all deserializers and create
            // mapping for chunks.
            for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
            {
                isValid = m_deserializers[deserializerIndex]->GetSequenceInfo(sequenceDescriptions[sequenceIndex], s);
                if (!isValid)
                {
                    invalid.insert(sequenceIndex);
                    break;
                }

                if (m_mbDefiningDeserializer == std::numeric_limits<size_t>::max())
                {
                    // Need to check the sequence length for all deserializers.
                    sequenceSamples = std::max<size_t>(sequenceSamples, s.m_numberOfSamples);
                }
                else if (m_mbDefiningDeserializer == deserializerIndex)
                {
                    sequenceSamples = s.m_numberOfSamples;
                }

                if (std::find(secondaryChunks[deserializerIndex].begin(), secondaryChunks[deserializerIndex].end(), s.m_chunkId) == secondaryChunks[deserializerIndex].end())
                    secondaryChunks[deserializerIndex].push_back(s.m_chunkId);
            }

            if (isValid)
            {
                numberOfSamples += sequenceSamples;
                numberOfSequences++;

                // Check whether the primary stream has the longest sequence.
                // If yes, we can optimize exposed sequence descriptions in GetSequencesByChunk.
                m_takePrimarySequenceLength = m_takePrimarySequenceLength && (sequenceSamples == sequence.m_numberOfSamples);
            }
        }

        // Build a chunk for valid sequences.
        if (numberOfSamples > 0)
        {
            BundlerChunkDescription cd;
            cd.m_numberOfSamples = numberOfSamples;
            cd.m_numberOfSequences = numberOfSequences;
            cd.m_id = (ChunkIdType) m_chunks.size();
            cd.m_original = chunks[chunkIndex];
            cd.m_invalid = std::move(invalid);
            cd.m_secondaryChunks = std::move(secondaryChunks);
            m_chunks.push_back(cd);
        }
    }

    if (m_verbosity)
        fprintf(stderr, "Bundler::CreateChunkDescriptions(): finished cleaning of %" PRIu64 " chunks\n", m_chunks.size());

    if(m_chunks.empty())
        RuntimeError("Could not reconcile data between different deserializers."
            " Keys of logical sequences do not match.");
}

// Gets chunk descriptions.
std::vector<ChunkInfo> Bundler::ChunkInfos()
{
    return std::vector<ChunkInfo>(m_chunks.begin(), m_chunks.end());
}

// Gets sequence descriptions for a chunk.
void Bundler::SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& sequences)
{
    const BundlerChunkDescription& chunk = m_chunks[chunkId];
    const ChunkInfo& original = chunk.m_original;
    m_primaryDeserializer->SequenceInfosForChunk(original.m_id, sequences);

    std::vector<SequenceInfo> result;
    if (m_takePrimarySequenceLength || m_mbDefiningDeserializer == 0) // No need to consult other deserializers.
    {
        // Do cleansing.
        result.reserve(sequences.size());
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk.m_invalid.find(sequenceIndex) != chunk.m_invalid.end())
            {
                continue;
            }

            result.push_back(sequences[sequenceIndex]);
            result.back().m_indexInChunk = sequenceIndex;
        }
    }
    else // need to get the max sequence length from other deserializers.
    {
        result.reserve(sequences.size());
        SequenceInfo s;
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk.m_invalid.find(sequenceIndex) != chunk.m_invalid.end())
                continue;

            auto sequence = sequences[sequenceIndex];
            uint32_t sequenceSamples = sequence.m_numberOfSamples;
            if (m_mbDefiningDeserializer != std::numeric_limits<size_t>::max())
            {
                m_deserializers[m_mbDefiningDeserializer]->GetSequenceInfo(sequence, s);
                sequenceSamples = s.m_numberOfSamples;
            }
            else
            {
                for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
                {
                    m_deserializers[deserializerIndex]->GetSequenceInfo(sequence, s);
                    sequenceSamples = std::max(sequenceSamples, s.m_numberOfSamples);
                }
            }

            sequence.m_numberOfSamples = sequenceSamples;
            sequence.m_indexInChunk = sequenceIndex;
            result.push_back(sequence);
        }
    }

    std::swap(sequences, result);
}

// Represents a chunk that has pointers to the underlying deserializer chunks.
class Bundler::BundlingChunk : public Chunk
{
    size_t m_numberOfInputs;
    Bundler* m_parent;
    ChunkIdType m_chunkId;

    // A mapping between exposed sequence id and inner chunk for each deserializer.
    // Index i of the vector maps to the chunk of inner sequence (i / number of deserializers) of
    // deserializer (i % number of deserializers).
    std::vector<ChunkPtr> m_innerChunks;
    // A mapping between exposed sequence id and inner sequence id for each deserializer.
    // Represents sequence index in chunk
    // Indices as above.
    std::vector<size_t> m_sequenceToSequence;

    DISABLE_COPY_AND_MOVE(BundlingChunk);

public:
    BundlingChunk(size_t numberOfInputs, Bundler* parent, ChunkIdType chunkId)
        : m_numberOfInputs(numberOfInputs), m_parent(parent), m_chunkId(chunkId)
    {
        const BundlerChunkDescription& chunk = m_parent->m_chunks[m_chunkId];
        const ChunkInfo& original = chunk.m_original;
        auto& deserializers = m_parent->m_deserializers;

        // Fetch all chunks in parallel.
        std::vector<std::map<ChunkIdType, std::shared_ptr<std::future<ChunkPtr>>>> chunks;
        chunks.resize(chunk.m_secondaryChunks.size());
        for (size_t i = 0; i < chunk.m_secondaryChunks.size(); ++i)
        {
            for (const auto& c : chunk.m_secondaryChunks[i])
            {
                chunks[i].emplace(
                    std::make_pair(c,
                        std::make_shared<std::future<ChunkPtr>>(
                            std::async(
                                launch::async,
                                [this, c, i]()
                                {
                                    ChunkPtr chunk = m_parent->m_weakChunkTable[i][c].lock();
                                    if (chunk)
                                        return chunk;
                                    return m_parent->m_deserializers[i]->GetChunk(c);
                                }))));
            }
        }

        std::vector<SequenceInfo> sequences;
        sequences.reserve(original.m_numberOfSequences);

        // Creating chunk mapping.
        m_parent->m_primaryDeserializer->SequenceInfosForChunk(original.m_id, sequences);
        ChunkPtr drivingChunk = chunks.front().find(original.m_id)->second->get();
        m_sequenceToSequence.resize(deserializers.size() * sequences.size());
        m_innerChunks.resize(deserializers.size() * sequences.size());
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk.m_invalid.find(sequenceIndex) != chunk.m_invalid.end())
            {
                continue;
            }

            size_t currentIndex = sequenceIndex * deserializers.size();
            m_sequenceToSequence[currentIndex] = sequences[sequenceIndex].m_indexInChunk;
            m_innerChunks[currentIndex] = drivingChunk;
        }

        // Creating sequence mapping and requiring underlying chunks.
        SequenceInfo s;
        for (size_t deserializerIndex = 1; deserializerIndex < deserializers.size(); ++deserializerIndex)
        {
            auto& chunkTable = m_parent->m_weakChunkTable[deserializerIndex];
            for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
            {
                if (chunk.m_invalid.find(sequenceIndex) != chunk.m_invalid.end())
                {
                    continue;
                }

                size_t currentIndex = sequenceIndex * deserializers.size() + deserializerIndex;
                bool exists = deserializers[deserializerIndex]->GetSequenceInfo(sequences[sequenceIndex], s);
                if (!exists)
                {
                    if(m_parent->m_verbosity >= (int)TraceLevel::Warning)
                        fprintf(stderr, "Warning: sequence '%s' could not be found in the deserializer responsible for stream '%ls'\n",
                            m_parent->m_corpus->IdToKey(sequences[sequenceIndex].m_key.m_sequence).c_str(),
                            deserializers[deserializerIndex]->StreamInfos().front().m_name.c_str());
                    m_sequenceToSequence[currentIndex] = SIZE_MAX;
                    continue;
                }

                m_sequenceToSequence[currentIndex] = s.m_indexInChunk;
                ChunkPtr secondaryChunk = chunkTable[s.m_chunkId].lock();
                if (!secondaryChunk)
                {
                    secondaryChunk = chunks[deserializerIndex].find(s.m_chunkId)->second->get();
                    chunkTable[s.m_chunkId] = secondaryChunk;
                }

                m_innerChunks[currentIndex] = secondaryChunk;
            }
        }
    }

    // Gets sequence by its index.
    virtual void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
    {
        result.reserve(m_numberOfInputs);
        size_t currentIndex = sequenceIndex * m_parent->m_deserializers.size();
        for (int i = 0; i < m_parent->m_deserializers.size(); ++i)
        {
            size_t originalSequenceId = m_sequenceToSequence[currentIndex + i];
            if (originalSequenceId == SIZE_MAX) // Invalid.
            {
                // Fill in invalid data.
                size_t numStreams = m_parent->m_deserializers[i]->StreamInfos().size();
                for (size_t j = 0; j < numStreams; ++j)
                    result.push_back(InvalidSequenceData::Instance());
                continue;
            }

            m_innerChunks[currentIndex + i]->GetSequence(originalSequenceId, result);
        }
    }
};

// Get chunk data by id.
ChunkPtr Bundler::GetChunk(ChunkIdType chunkId)
{
    return std::make_shared<BundlingChunk>(m_streams.size(), this, chunkId);
}

}
