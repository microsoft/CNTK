//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "AlignedBundler.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents bundled chunk description with possible cleansed data.
struct AlignedBundler::BundlerChunkDescription : public ChunkDescription
{
    ChunkDescriptionPtr m_original;

    // Sequences that are invalid in at least one deserializer.
    std::set<size_t> m_invalid;
};

AlignedBundler::AlignedBundler(
    const ConfigParameters& readerConfig,
    IDataDeserializerPtr primaryDeserializer,
    std::vector<IDataDeserializerPtr> deserializers)
    : DataDeserializerBase(true),
      m_deserializers(deserializers),
      m_primaryDeserializer(primaryDeserializer)
{
    if (primaryDeserializer != m_deserializers.front())
        LogicError("Please provide primary deserializer first.");

    m_verbosity = readerConfig(L"verbosity", 0);

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

    CreateChunkDescriptions();
}

// Creates chunk descriptions based on chunks of underlying deserializers.
void AlignedBundler::CreateChunkDescriptions()
{
    if (m_verbosity)
        fprintf(stderr, "AlignedBundler::CreateChunkDescriptions(): started\n");

    auto chunks = m_primaryDeserializer->GetChunkDescriptions();

    if (chunks.size() < 1)
        RuntimeError("Driving deserializer should at least provide one chunk.");

    if (CHUNKID_MAX < chunks.size())
        RuntimeError("Driving deserializer provided too many chunks.");

    // Creating a table of weak chunks for deserializers.
    for (size_t i = 0; i < m_deserializers.size(); ++i)
    {
        auto totalChunkNumber = m_deserializers[i]->GetChunkDescriptions().size();
        m_weakChunkTable.push_back(std::vector<std::weak_ptr<Chunk>>(totalChunkNumber));
    }

    m_chunks.reserve(chunks.size());

    if (m_verbosity)
        fprintf(stderr, "Bundler::CreateChunkDescriptions(): creating descriptions for %" PRIu64 " chunks\n", m_chunks.size());

    m_takePrimarySequenceLength = true;

    struct ChunkInfo
    {
        size_t currentChunkIndex;
        std::vector<SequenceDescription> sequences;
    };

    struct StreamDescriptor
    {
        StreamDescriptor(size_t lookAheadSequences)
            : lookAheadSequences(lookAheadSequences)
        {}

        IDataDeserializerPtr deserializer;

        size_t lookAheadSequences;

        size_t currentSequenceIndex;
        size_t totalNumberOfSequences;
        size_t currentChunkIndex;
        std::vector<ChunkInfo> chunks;

        bool MoveTo(const SequenceDescription& sequence)
        {
            size_t nextChunkIndex = chunks.size() > 0 ? chunks.back().currentChunkIndex + 1 : 0;

            // Fill in chunks if required.
            while (totalNumberOfSequences < lookAheadSequences)
            {
                chunks.push_back({ nextChunkIndex, std::vector<SequenceDescription>() });
                deserializer->GetSequencesForChunk(currentChunkIndex, chunks.back().sequences);
            }

            // We are progressing on the secondary deserializers only if some 
            // of the sequences were found.
            for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
            {
                for (size_t i = currentSequenceIndex; i < chunks[chunkIndex].sequences.size(); ++i)
                {
                    if (sequence.m_key == chunks[chunkIndex].sequences[i].m_key)
                    {
                        currentSequenceIndex = currentSequenceIndex + 1;
                        if (currentSequenceIndex == chunks[chunkIndex].sequences.size()) // Last sequence, move to next.
                        {
                            currentSequenceIndex = 0;
                            chunkIndex++;
                        }

                        // Found, lets remove all chunks/sequences earlier then this one.
                        if (chunkIndex != 0)
                            chunks.erase(chunks.begin(), chunks.begin() + chunkIndex);
                        return true;
                    }
                }
            }

            return false;
        }
    };

    std::vector<StreamDescriptor> cursors;
    cursors.reserve(m_deserializers.size() - 1);

    for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
        cursors.push_back({ m_deserializers[deserializerIndex], 0, std::vector<ChunkInfo>() });

    size_t lookAhead = 10;

    m_index.resize(chunks.size());
    for (size_t i = 0; i < m_index.size(); ++i)
        m_index[i].reserve(cursors.size());

    for (ChunkIdType chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
    {
        vector<SequenceDescription> descriptions;
        m_primaryDeserializer->GetSequencesForChunk(chunkIndex, descriptions);

        for (size_t i = 0; i < descriptions.size(); ++i)
        {
            const auto& description = descriptions[i];

            bool validSequence = 0;
            for (size_t c = 0; c < cursors.size(); ++c)
            {
                if(cursors[i].MoveTo(description);
            }
        }
    }


    // Build bundling chunks using underlying deserializers.
    std::vector<SequenceDescription> sequenceDescriptions;
    sequenceDescriptions.reserve(chunks.front()->m_numberOfSequences);
    SequenceDescription s;

    for (ChunkIdType chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex)
    {
        size_t numberOfSamples = 0;
        size_t numberOfSequences = 0;
        sequenceDescriptions.clear();

        // Iterating thru all sequences and identifying whether they are valid among all deserializers.
        m_primaryDeserializer->GetSequencesForChunk(chunks[chunkIndex]->m_id, sequenceDescriptions);

        std::set<size_t> invalid;
        for (size_t sequenceIndex = 0; sequenceIndex < sequenceDescriptions.size(); ++sequenceIndex)
        {
            auto sequence = sequenceDescriptions[sequenceIndex];
            bool isValid = true;
            size_t sequenceSamples = sequence.m_numberOfSamples;
            for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
            {
                isValid = m_deserializers[deserializerIndex]->GetSequenceDescription(sequenceDescriptions[sequenceIndex], s);
                if (!isValid)
                {
                    invalid.insert(sequenceIndex);
                    break;
                }

                sequenceSamples = std::max<size_t>(sequenceSamples, s.m_numberOfSamples);
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
            auto cd = std::make_shared<BundlerChunkDescription>();
            cd->m_numberOfSamples = numberOfSamples;
            cd->m_numberOfSequences = numberOfSequences;
            cd->m_id = (ChunkIdType) m_chunks.size();
            cd->m_original = chunks[chunkIndex];
            m_chunks.push_back(cd);
            cd->m_invalid = std::move(invalid);
        }
    }

    if (m_verbosity)
        fprintf(stderr, "Bundler::CreateChunkDescriptions(): finished cleaning of %" PRIu64 " chunks\n", m_chunks.size());
}

// Gets chunk descriptions.
ChunkDescriptions AlignedBundler::GetChunkDescriptions()
{
    return ChunkDescriptions(m_chunks.begin(), m_chunks.end());
}

// Gets sequence descriptions for a chunk.
void AlignedBundler::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& sequences)
{
    BundlerChunkDescriptionPtr chunk = m_chunks[chunkId];
    ChunkDescriptionPtr original = chunk->m_original;
    m_primaryDeserializer->GetSequencesForChunk(original->m_id, sequences);

    std::vector<SequenceDescription> result;
    if (m_takePrimarySequenceLength) // No need to consult other deserializers.
    {
        // Do cleansing.
        result.reserve(sequences.size());
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
            {
                continue;
            }

            result.push_back(sequences[sequenceIndex]);
            result.back().m_indexInChunk = sequenceIndex;
        }
    }
    else // need to get the max sequence length from other deserializers.
         // TODO: This will change when the sequence length will be exposed per stream.
    {
        result.reserve(sequences.size());
        SequenceDescription s;
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
            {
                continue;
            }

            auto sequence = sequences[sequenceIndex];
            uint32_t sequenceSamples = sequence.m_numberOfSamples;
            for (size_t deserializerIndex = 1; deserializerIndex < m_deserializers.size(); ++deserializerIndex)
            {
                m_deserializers[deserializerIndex]->GetSequenceDescription(sequence, s);
                sequenceSamples = std::max(sequenceSamples, s.m_numberOfSamples);
            }
            sequence.m_numberOfSamples = sequenceSamples;
            sequence.m_indexInChunk = sequenceIndex;
            result.push_back(sequence);
        }
    }

    std::swap(sequences, result);
}

// Represents a chunk that has pointers to the underlying deserializer chunks.
class AlignedBundler::BundlingChunk : public Chunk
{
    size_t m_numberOfInputs;
    AlignedBundler* m_parent;
    ChunkIdType m_chunkId;

    // A mapping between exposed sequence id and inner chunk for each deserializer.
    // Index i of the vector maps to the chunk of inner sequence (i / number of deserializers) of
    // deserializer (i % number of deserializers).
    std::vector<ChunkPtr> m_innerChunks;
    // A mapping between exposed sequence id and inner sequence id for each deserializer.
    // Indices as above.
    std::vector<size_t> m_sequenceToSequence;

    DISABLE_COPY_AND_MOVE(BundlingChunk);

public:
    BundlingChunk(size_t numberOfInputs, AlignedBundler* parent, ChunkIdType chunkId)
        : m_numberOfInputs(numberOfInputs), m_parent(parent), m_chunkId(chunkId)
    {
        BundlerChunkDescriptionPtr chunk = m_parent->m_chunks[m_chunkId];
        ChunkDescriptionPtr original = chunk->m_original;

        auto& deserializers = m_parent->m_deserializers;
        std::vector<SequenceDescription> sequences;
        sequences.reserve(original->m_numberOfSequences);

        // Creating chunk mapping.
        m_parent->m_primaryDeserializer->GetSequencesForChunk(original->m_id, sequences);
        ChunkPtr drivingChunk = m_parent->m_primaryDeserializer->GetChunk(original->m_id);
        m_sequenceToSequence.resize(deserializers.size() * sequences.size());
        m_innerChunks.resize(deserializers.size() * sequences.size());
        for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
        {
            if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
            {
                continue;
            }

            size_t currentIndex = sequenceIndex * deserializers.size();
            m_sequenceToSequence[currentIndex] = sequences[sequenceIndex].m_indexInChunk;
            m_innerChunks[currentIndex] = drivingChunk;
        }

        // Creating sequence mapping and requiring underlying chunks.
        SequenceDescription s;
        for (size_t deserializerIndex = 1; deserializerIndex < deserializers.size(); ++deserializerIndex)
        {
            auto& chunkTable = m_parent->m_weakChunkTable[deserializerIndex];
            for (size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex)
            {
                if (chunk->m_invalid.find(sequenceIndex) != chunk->m_invalid.end())
                {
                    continue;
                }

                size_t currentIndex = sequenceIndex * deserializers.size() + deserializerIndex;
                deserializers[deserializerIndex]->GetSequenceDescription(sequences[sequenceIndex], s);
                m_sequenceToSequence[currentIndex] = s.m_indexInChunk;

                ChunkPtr secondaryChunk = chunkTable[s.m_chunkId].lock();
                if (!secondaryChunk)
                {
                    secondaryChunk = deserializers[deserializerIndex]->GetChunk(s.m_chunkId);
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
            m_innerChunks[currentIndex + i]->GetSequence(originalSequenceId, result);
        }
    }
};

// Get chunk data by id.
ChunkPtr AlignedBundler::GetChunk(ChunkIdType chunkId)
{
    return std::make_shared<BundlingChunk>(m_streams.size(), this, chunkId);
}

}}}
