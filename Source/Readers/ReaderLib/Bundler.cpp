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
    std::vector<IDataDeserializerPtr> deserializers)
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
    CreateChunkDescriptions();
}

void Bundler::CreateChunkDescriptions()
{
    auto chunks = m_driver->GetChunkDescriptions();

    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
    {
        size_t numberOfSamples = 0;
        size_t numberOfSequences = 0;
        auto sequenceDescriptions = m_driver->GetSequencesForChunk(chunks[chunkIndex]->id);
        std::set<size_t> invalid;
        for (size_t sequenceIndex = 0; sequenceIndex < sequenceDescriptions.size(); ++sequenceIndex)
        {
            auto sequence = sequenceDescriptions[sequenceIndex];
            bool isValid = true;
            for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
            {
                auto s = m_deserializers[deserializerIndex]->GetSequenceDescriptionByKey(sequenceDescriptions[sequenceIndex]->m_key);
                if (!s->m_isValid)
                {
                    isValid = false;
                    invalid.insert(sequenceIndex);
                    break;
                }
            }

            if (isValid)
            {
                numberOfSamples += sequence->m_numberOfSamples;
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

std::vector<SequenceDescriptionPtr> Bundler::GetSequencesForChunk(size_t chunkId)
{
    BundlerChunkDescriptionPtr chunk = m_chunks[chunkId];
    ChunkDescriptionPtr original = chunk->m_original;
    auto sequences = m_driver->GetSequencesForChunk(original->id);
    std::vector<SequenceDescriptionPtr> result;
    result.reserve(sequences.size());
    for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
    {
        if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
        {
            continue;
        }

        result.push_back(sequences[sequenceIndex]);
    }

    return result;
}

size_t Bundler::GetTotalNumberOfSamples()
{
    return m_driver->GetTotalNumberOfSamples();
}

size_t Bundler::GetTotalNumberOfSequences()
{
    return m_driver->GetTotalNumberOfSequences();
}

// Represents a chunk that has poibters to the underlying deserialzer chunks.
class BundlingChunk : public Chunk
{
    size_t m_numberOfInputs;
    Bundler* m_parent;
    size_t m_chunkId;

    std::vector<std::vector<ChunkPtr>> m_innerChunks;
    std::vector<std::vector<size_t>> m_sequenceToSequence;

    DISABLE_COPY_AND_MOVE(BundlingChunk);

public:
    BundlingChunk(size_t numberOfInputs, Bundler* parent, size_t chunkId)
        : m_numberOfInputs(numberOfInputs), m_parent(parent), m_chunkId(chunkId)
    {
        BundlerChunkDescriptionPtr chunk = m_parent->m_chunks[m_chunkId];
        ChunkDescriptionPtr original = chunk->m_original;

        auto& deserializers = m_parent->m_deserializers;
        m_sequenceToSequence.resize(deserializers.size());
        m_innerChunks.resize(deserializers.size());

        auto sequences = m_parent->m_driver->GetSequencesForChunk(original->id);
        ChunkPtr drivingChunk = m_parent->m_driver->GetChunk(original->id);
        m_sequenceToSequence[0].resize(sequences.size());
        m_innerChunks[0].resize(sequences.size());
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
            {
                continue;
            }

            m_sequenceToSequence[0][sequenceIndex] = sequences[sequenceIndex]->m_id;
            m_innerChunks[0][sequenceIndex] = drivingChunk;
        }

        for (size_t deserializerIndex = 1; deserializerIndex < m_parent->m_deserializers.size(); ++deserializerIndex)
        {
            m_sequenceToSequence[deserializerIndex].resize(sequences.size());
            m_innerChunks[deserializerIndex].resize(sequences.size());
            for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
            {
                if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
                {
                    continue;
                }

                auto s = deserializers[deserializerIndex]->GetSequenceDescriptionByKey(sequences[sequenceIndex]->m_key);
                m_sequenceToSequence[deserializerIndex][sequenceIndex] = s->m_id;
                m_innerChunks[deserializerIndex][sequenceIndex] = deserializers[deserializerIndex]->GetChunk(s->m_chunkId);
            }
        }
    }

    virtual std::vector<SequenceDataPtr> GetSequence(size_t sequenceId) override
    {
        std::vector<SequenceDataPtr> result;
        result.reserve(m_numberOfInputs);

        for (int i = 0; i < m_parent->m_deserializers.size(); ++i)
        {
            size_t originalSequenceId = m_sequenceToSequence[i][sequenceId];
            auto sequences = m_innerChunks[i][sequenceId]->GetSequence(originalSequenceId);
            result.insert(result.end(), sequences.begin(), sequences.end());
        }

        return result;
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

const SequenceDescription* Bundler::GetSequenceDescriptionByKey(const KeyType&)
{
    throw std::logic_error("Not implemented");
}

}}}
