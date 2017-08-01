//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define __STDC_FORMAT_MACROS
#define _CRT_SECURE_NO_WARNINGS

#include "Index.h"
#include "IndexBuilder.h"
#include "DataDeserializer.h"

using std::string;

namespace CNTK {

void ChunkDescriptor::AddSequence(const IndexedSequence& sequence)
{
    auto offsetInChunk = sequence.offset - m_startOffset;
    m_sequences.emplace_back(sequence.key, sequence.numberOfSamples, sequence.size, offsetInChunk);
    m_numberOfSamples += sequence.numberOfSamples;
    m_endOffset = sequence.offset + sequence.size;
    if (m_sequences.size() > std::numeric_limits<uint32_t>::max())
        RuntimeError("Exceeded maximum number of sequences in a chunk");
}

void Index::AddSequence(const IndexedSequence& sequence) 
{
    if (sequence.numberOfSamples == 0)
        RuntimeError("Invalid sequence: number of samples == 0");

    if (sequence.size == 0)
        RuntimeError("Invalid sequence: size in bytes == 0");

    m_sizeInBytes += sequence.size;
    m_numberOfSamples += sequence.numberOfSamples;
    m_numberOfSequences++;

    auto currentChunkSize = m_chunks.empty() ? 0 : m_chunks.back().SizeInBytes();

    // TODO: the sum of sizes does not account for a possible gap before the sequence offset.
    if (currentChunkSize == 0 || currentChunkSize + sequence.size > m_maxChunkSize)
    {
        if (!m_chunks.empty()) // The previous chunk is done, finalize it.
            m_chunks.back().m_sequences.shrink_to_fit();

        m_chunks.push_back(ChunkDescriptor(sequence.offset));

        if (std::numeric_limits<ChunkIdType>::max() < m_chunks.size())
            RuntimeError("Maximum number of chunks exceeded.");

        // reserve the space for sequences up-front, using the average sequence size to 
        // estimate the number of sequences in a chunk. 
        auto avgSequenceSize = m_sizeInBytes / m_numberOfSequences;
        auto numSequencesPerChunk = m_maxChunkSize / avgSequenceSize;
        m_chunks.back().m_sequences.reserve(numSequencesPerChunk);
    }

    auto& currentChunk = m_chunks.back();

    currentChunk.AddSequence(sequence);
}

void Index::MapSequenceKeyToLocation()
{
    // Precalculate size of the mapping.
    size_t numSequences = 0;
    for (const auto& c : m_chunks)
        numSequences += c.NumberOfSequences();

    m_keyToSequenceInChunk.reserve(numSequences);

    for (uint32_t i = 0; i < m_chunks.size(); i++)
        for (uint32_t j = 0; j < m_chunks[i].NumberOfSequences(); j++)
            m_keyToSequenceInChunk.emplace_back(m_chunks[i].Sequences()[j].m_key, i, j);

    // Sort for fast retrieval afterwards
    std::sort(m_keyToSequenceInChunk.begin(), m_keyToSequenceInChunk.end(),
        [](const std::tuple<size_t, uint32_t, uint32_t>& a, const std::tuple<size_t, uint32_t, uint32_t>& b)
    {
        return std::get<0>(a) < std::get<0>(b);
    });
}


std::tuple<bool, uint32_t, uint32_t> Index::GetSequenceByKey(size_t key) const
{
    auto found = std::lower_bound(m_keyToSequenceInChunk.begin(), m_keyToSequenceInChunk.end(), key,
        [](const std::tuple<size_t, size_t, size_t>& a, size_t b)
    {
        return std::get<0>(a) < b;
    });

    if (found == m_keyToSequenceInChunk.end() || std::get<0>(*found) != key)
    {
        return std::make_tuple(false, 0, 0);
    }

    return std::make_tuple(true, std::get<1>(*found), std::get<2>(*found));
}

}

