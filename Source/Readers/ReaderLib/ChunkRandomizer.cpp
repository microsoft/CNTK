//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "ChunkRandomizer.h"
#include <random>
#include "RandomOrdering.h"

namespace CNTK {

    ChunkRandomizer::ChunkRandomizer(DataDeserializerPtr deserializer, 
        size_t randomizationRange,
        bool sampleBasedRandomizationWindow) :
        m_deserializer(deserializer), 
        m_randomizationRange(randomizationRange),
        m_sampleBasedRandomizationWindow(sampleBasedRandomizationWindow)
    {
        m_originalChunks = m_deserializer->ChunkInfos();
        assert(m_originalChunks.size() < ChunkIdMax);
    }

    // Gets randomized chunks.
    const std::vector<RandomizedChunk>& ChunkRandomizer::GetRandomizedChunks() const
    {
        return m_randomizedChunks;
    }

    // Randomizes chunks and calculates randomization windows.
    void ChunkRandomizer::Randomize(size_t seed)
    {
        std::vector<ChunkIdType> randomizedChunkIndices;
        randomizedChunkIndices.reserve(m_originalChunks.size());
        for (ChunkIdType i = 0; i < m_originalChunks.size(); i++)
        {
            randomizedChunkIndices.push_back(i);
        }

        m_rng.seed((unsigned long)seed);
        Microsoft::MSR::CNTK::RandomShuffleMT(randomizedChunkIndices, m_rng);

        // Place randomized chunks on the timeline
        m_randomizedChunks.clear();
        m_randomizedChunks.reserve(m_originalChunks.size());

        size_t samplePosition = 0;
        size_t sequencePosition = 0;
        for (ChunkIdType chunkIndex = 0; chunkIndex < m_originalChunks.size(); chunkIndex++)
        {
            // TODO: in case of the chunk-based randomization window, we couldn't care less
            // about samples. If we get rid of sample-based randomization, we could do away
            // with this sample counting altogether.
            const size_t originalChunkIndex = randomizedChunkIndices.at(chunkIndex);
            const size_t numberOfSamples = m_originalChunks[originalChunkIndex].m_numberOfSamples;
            const size_t numberOfSequences = m_originalChunks[originalChunkIndex].m_numberOfSequences;

            RandomizedChunk randomizedChunk;
            randomizedChunk.m_chunkId = chunkIndex;
            randomizedChunk.m_original = &m_originalChunks[originalChunkIndex];
            randomizedChunk.m_samplePositionStart = samplePosition;
            randomizedChunk.m_sequencePositionStart = sequencePosition;
            m_randomizedChunks.push_back(randomizedChunk);
            samplePosition += numberOfSamples;
            sequencePosition += numberOfSequences;
        }

        if (m_sampleBasedRandomizationWindow) 
        {
            RandomizeUsingWindowInSamples();
        }
        else 
        {
            RandomizeUsingWindowInChunks();
        }
    }

    // Randomizes chunks and calculates randomization windows in samples.
    void ChunkRandomizer::RandomizeUsingWindowInSamples()
    {
        // For each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
        size_t halfWindowRange = m_randomizationRange / 2;
        for (ChunkIdType chunkId = 0; chunkId < m_originalChunks.size(); chunkId++)
        {
            auto& chunk = m_randomizedChunks[chunkId];

            // start with the range of left neighbor
            if (chunkId == 0)
            {
                chunk.m_randomizationWindow.m_begin = 0;
                chunk.m_randomizationWindow.m_end = 1;
            }
            else
            {
                chunk.m_randomizationWindow.m_begin = m_randomizedChunks[chunkId - 1].m_randomizationWindow.m_begin; // might be too early
                chunk.m_randomizationWindow.m_end = m_randomizedChunks[chunkId - 1].m_randomizationWindow.m_end; // might have more space
            }

            // Need to adapt now.
            while (chunk.m_samplePositionStart - m_randomizedChunks[chunk.m_randomizationWindow.m_begin].m_samplePositionStart > halfWindowRange)
            {
                // too early, need to increase
                chunk.m_randomizationWindow.m_begin++;
            }

            // Chunk id should always be inside the window.
            // Adjusting begin and end window against chunkId.
            chunk.m_randomizationWindow.m_begin = std::min(chunk.m_randomizationWindow.m_begin, chunkId);
            chunk.m_randomizationWindow.m_end = std::max(chunk.m_randomizationWindow.m_end, chunkId + 1);

            while (chunk.m_randomizationWindow.m_end < m_originalChunks.size() &&
                m_randomizedChunks[chunk.m_randomizationWindow.m_end].SampleEndPosition() - chunk.m_samplePositionStart < halfWindowRange)
            {
                // got more space, move window to the right.
                chunk.m_randomizationWindow.m_end++;
            }
        }
    }

    // Randomizes chunks and calculates randomization windows in chunks.
    void ChunkRandomizer::RandomizeUsingWindowInChunks()
    {
        auto halfWindowRange = m_randomizationRange / 2;
        auto windwowSize = m_randomizationRange == 0 ? 1 : ChunkIdType(m_randomizationRange);
        for (auto i = 0; i < m_randomizedChunks.size(); i++)
        {
            auto& chunk = m_randomizedChunks[i];
            chunk.m_randomizationWindow.m_begin = (i > halfWindowRange) ? i - ChunkIdType(halfWindowRange) : 0;
            
            chunk.m_randomizationWindow.m_end = chunk.m_randomizationWindow.m_begin + windwowSize;

            if (chunk.m_randomizationWindow.m_end > m_randomizedChunks.size()) 
            {
                chunk.m_randomizationWindow.m_end = ChunkIdType(m_randomizedChunks.size());
                chunk.m_randomizationWindow.m_begin =
                    (m_randomizedChunks.size() > windwowSize) ? ChunkIdType(m_randomizedChunks.size() - windwowSize) : 0;
            }
        }
    }
}
