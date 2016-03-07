//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "Bundler.h"

namespace Microsoft { namespace MSR { namespace CNTK {

Bundler::Bundler(
    const ConfigParameters& readerConfig,
    IDataDeserializerPtr driver,
    std::vector<IDataDeserializerPtr> deserializers,
    bool cleanse)
    : m_deserializers(deserializers), m_driver(driver)
{
    UNUSED(readerConfig);
    std::vector<StreamDescriptionPtr> streams;
    for (auto d : deserializers)
    {
        for (auto i : d->GetStreamDescriptions())
        {
            StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*i);
            stream->m_id = streams.size();
            streams.push_back(stream);
        }
    }

    m_streams = streams;
    m_cleanse = cleanse;
    CreateChunkDescriptions();
}

void Bundler::CreateChunkDescriptions()
{
    auto chunks = m_driver->GetChunkDescriptions();
    m_chunks.reserve(chunks.size());

    if (!m_cleanse)
    {
        for (const auto& c : chunks)
        {
            auto cd = std::make_shared<BundlerChunkDescription>();
            cd->numberOfSamples = c->numberOfSamples;
            cd->numberOfSequences = c->numberOfSequences;
            cd->id = m_chunks.size();
            cd->m_original = c;
            m_chunks.push_back(cd);
        }
        return;
    }

    if (m_chunks.size() < 1)
    {
        RuntimeError("Driving deserializer should at least provide one chunk.");
    }

    std::vector<SequenceDescription> sequenceDescriptions;
    sequenceDescriptions.reserve(chunks.front()->numberOfSequences);
    SequenceDescription s;
    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
    {
        size_t numberOfSamples = 0;
        size_t numberOfSequences = 0;
        sequenceDescriptions.clear();
        m_driver->GetSequencesForChunk(chunks[chunkIndex]->id, sequenceDescriptions);
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

        if (numberOfSamples > 0)
        {
            auto cd = std::make_shared<BundlerChunkDescription>();
            cd->numberOfSamples = numberOfSamples;
            cd->numberOfSequences = numberOfSequences;
            cd->id = m_chunks.size();
            cd->m_original = chunks[chunkIndex];
            m_chunks.push_back(cd);
            cd->m_invalid = std::move(invalid);
        }
    }
}

ChunkDescriptions Bundler::GetChunkDescriptions()
{
    return ChunkDescriptions(m_chunks.begin(), m_chunks.end());
}

void Bundler::GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& sequences)
{
    BundlerChunkDescriptionPtr chunk = m_chunks[chunkId];
    ChunkDescriptionPtr original = chunk->m_original;
    m_driver->GetSequencesForChunk(original->id, sequences);

    if (chunk->m_invalid.empty())
    {
        return;
    }

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
/*
size_t Bundler::GetTotalNumberOfSamples()
{
    return m_driver->GetTotalNumberOfSamples();
}

size_t Bundler::GetTotalNumberOfSequences()
{
    return m_driver->GetTotalNumberOfSequences();
}*/

// Represents a chunk that has poibters to the underlying deserialzer chunks.
class BundlingChunk : public Chunk
{
    size_t m_numberOfInputs;
    Bundler* m_parent;
    size_t m_chunkId;

    std::vector<ChunkPtr> m_innerChunks;
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
        sequences.reserve(original->numberOfSequences);

        m_parent->m_driver->GetSequencesForChunk(original->id, sequences);
        ChunkPtr drivingChunk = m_parent->m_driver->GetChunk(original->id);
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

        SequenceDescription s;
        for (size_t deserializerIndex = 1; deserializerIndex < m_parent->m_deserializers.size(); ++deserializerIndex)
        {
            for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
            {
                if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
                {
                    continue;
                }

                size_t currentIndex = sequenceIndex * m_numberOfInputs + deserializerIndex;
                deserializers[deserializerIndex]->GetSequenceDescriptionByKey(sequences[sequenceIndex].m_key, s);
                m_sequenceToSequence[currentIndex] = s.m_id;
                m_innerChunks[currentIndex] = deserializers[deserializerIndex]->GetChunk(s.m_chunkId);
            }
        }
    }

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

ChunkPtr Bundler::GetChunk(size_t chunkId)
{
    return std::make_shared<BundlingChunk>(m_streams.size(), this, chunkId);
}

std::vector<StreamDescriptionPtr> Bundler::GetStreamDescriptions() const
{
    return m_streams;
}

void Bundler::GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&)
{
    throw std::logic_error("Not implemented");
}

}}}
