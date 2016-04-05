//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "Bundler.h"
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents bundled chunk description with possible cleansed data.
struct Bundler::BundlerChunkDescription : public ChunkDescription
{
    ChunkDescriptionPtr m_original;

    // Sequences that are invalid in at least one deserializer.
    std::set<size_t> m_invalid;
};

Bundler::Bundler(
    const ConfigParameters& readerConfig,
    IDataDeserializerPtr driver,
    std::vector<IDataDeserializerPtr> deserializers,
    bool cleanse)
    : m_deserializers(deserializers), m_driver(driver)
{
    UNUSED(readerConfig);

    // Combines streams of underlying deserializers.
    for (auto d : deserializers)
    {
        for (auto i : d->GetStreamDescriptions())
        {
            StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*i);
            stream->m_id = m_streams.size();
            m_streams.push_back(stream);
        }
    }

    m_cleanse = cleanse;
    CreateChunkDescriptions();
}

// Creates chunk descriptions based on chunks of underlying deserializers.
void Bundler::CreateChunkDescriptions()
{
    auto chunks = m_driver->GetChunkDescriptions();
    if (chunks.size() < 1)
    {
        RuntimeError("Driving deserializer should at least provide one chunk.");
    }

    m_chunks.reserve(chunks.size());

    // If there is not cleaning required simply build chunks based on the chunk descriptions of the primary deserializer.
    if (!m_cleanse)
    {
        for (const auto& c : chunks)
        {
            auto cd = std::make_shared<BundlerChunkDescription>();
            cd->m_numberOfSamples = c->m_numberOfSamples;
            cd->m_numberOfSequences = c->m_numberOfSequences;
            cd->m_id = m_chunks.size();
            cd->m_original = c;
            m_chunks.push_back(cd);
        }
        return;
    }

    // Otherwise build bundling chunks using underlying deserializers.
    std::vector<SequenceDescription> sequenceDescriptions;
    sequenceDescriptions.reserve(chunks.front()->m_numberOfSequences);
    SequenceDescription s;
    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
    {
        size_t numberOfSamples = 0;
        size_t numberOfSequences = 0;
        sequenceDescriptions.clear();

        // Iterating thru all sequences and identifying whether they are valid among all deserializers.
        m_driver->GetSequencesForChunk(chunks[chunkIndex]->m_id, sequenceDescriptions);
        std::set<size_t> invalid;
        for (size_t sequenceIndex = 0; sequenceIndex < sequenceDescriptions.size(); ++sequenceIndex)
        {
            auto sequence = sequenceDescriptions[sequenceIndex];
            bool isValid = true;
            for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
            {
                m_deserializers[deserializerIndex]->GetSequenceDescriptionByKey(sequenceDescriptions[sequenceIndex].m_key, s);
                if (!s.m_isValid)
                {
                    isValid = false;
                    invalid.insert(sequenceIndex);
                    break;
                }
            }

            if (isValid)
            {
                numberOfSamples += sequence.m_numberOfSamples;
                numberOfSequences++;
            }
        }

        // Build a chunk for valid sequences.
        if (numberOfSamples > 0)
        {
            auto cd = std::make_shared<BundlerChunkDescription>();
            cd->m_numberOfSamples = numberOfSamples;
            cd->m_numberOfSequences = numberOfSequences;
            cd->m_id = m_chunks.size();
            cd->m_original = chunks[chunkIndex];
            m_chunks.push_back(cd);
            cd->m_invalid = std::move(invalid);
        }
    }
}

// Gets chunk descriptions.
ChunkDescriptions Bundler::GetChunkDescriptions()
{
    return ChunkDescriptions(m_chunks.begin(), m_chunks.end());
}

// Gets sequence descriptions for a chunk.
void Bundler::GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& sequences)
{
    BundlerChunkDescriptionPtr chunk = m_chunks[chunkId];
    ChunkDescriptionPtr original = chunk->m_original;
    m_driver->GetSequencesForChunk(original->m_id, sequences);

    // Can return because all sequences are clean.
    if (chunk->m_invalid.empty())
    {
        return;
    }

    // Do cleansing.
    std::vector<SequenceDescription> result;
    result.reserve(sequences.size());
    for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
    {
        if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
        {
            continue;
        }

        result.push_back(sequences[sequenceIndex]);
    }
    std::swap(sequences, result);
}

// Represents a chunk that has pointers to the underlying deserializer chunks.
class Bundler::BundlingChunk : public Chunk
{
    size_t m_numberOfInputs;
    Bundler* m_parent;
    size_t m_chunkId;

    // A mapping between exposed sequence id and inner chunk for each deserializer.
    // Index i of the vector maps to the chunk of inner sequence (i / m_numberOfInputs) of
    // deserializer (i % m_numberOfInputs).
    std::vector<ChunkPtr> m_innerChunks;
    // A mapping between exposed sequence id and inner sequence id for each deserializer.
    // Indices as above.
    std::vector<size_t> m_sequenceToSequence;

    DISABLE_COPY_AND_MOVE(BundlingChunk);

public:
    BundlingChunk(size_t numberOfInputs, Bundler* parent, size_t chunkId)
        : m_numberOfInputs(numberOfInputs), m_parent(parent), m_chunkId(chunkId)
    {
        BundlerChunkDescriptionPtr chunk = m_parent->m_chunks[m_chunkId];
        ChunkDescriptionPtr original = chunk->m_original;

        auto& deserializers = m_parent->m_deserializers;
        assert(numberOfInputs == deserializers.size());
        std::vector<SequenceDescription> sequences;
        sequences.reserve(original->m_numberOfSequences);

        // Creating chunk mapping.
        m_parent->m_driver->GetSequencesForChunk(original->m_id, sequences);
        ChunkPtr drivingChunk = m_parent->m_driver->GetChunk(original->m_id);
        m_sequenceToSequence.resize(m_numberOfInputs * sequences.size());
        m_innerChunks.resize(m_numberOfInputs * sequences.size());
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
            {
                continue;
            }

            size_t currentIndex = sequenceIndex * m_numberOfInputs;
            m_sequenceToSequence[currentIndex] = sequences[sequenceIndex].m_id;
            m_innerChunks[currentIndex] = drivingChunk;
        }

        // Creating sequence mapping and requiring underlying chunks.
        SequenceDescription s;
        for (size_t deserializerIndex = 1; deserializerIndex < m_parent->m_deserializers.size(); ++deserializerIndex)
        {
            std::map<size_t, ChunkPtr> secondaryChunks;

            for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
            {
                if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
                {
                    continue;
                }

                size_t currentIndex = sequenceIndex * m_numberOfInputs + deserializerIndex;
                deserializers[deserializerIndex]->GetSequenceDescriptionByKey(sequences[sequenceIndex].m_key, s);
                m_sequenceToSequence[currentIndex] = s.m_id;

                ChunkPtr secondaryChunk;
                auto it = secondaryChunks.find(s.m_chunkId);
                if (it == secondaryChunks.end())
                {
                    secondaryChunk = deserializers[deserializerIndex]->GetChunk(s.m_chunkId);
                    secondaryChunks.insert(make_pair(s.m_chunkId, secondaryChunk));
                }
                else
                {
                    secondaryChunk = it->second;
                }

                m_innerChunks[currentIndex] = secondaryChunk;
            }
        }
    }

    // Gets sequence by its id.
    virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
    {
        result.reserve(m_numberOfInputs);
        size_t currentIndex = sequenceId * m_numberOfInputs;
        for (int i = 0; i < m_parent->m_deserializers.size(); ++i)
        {
            size_t originalSequenceId = m_sequenceToSequence[currentIndex + i];
            m_innerChunks[currentIndex + i]->GetSequence(originalSequenceId, result);
        }
    }
};

// Get chunk data by id.
ChunkPtr Bundler::GetChunk(size_t chunkId)
{
    return std::make_shared<BundlingChunk>(m_streams.size(), this, chunkId);
}

}}}
