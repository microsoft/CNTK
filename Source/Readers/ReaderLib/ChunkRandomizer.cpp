//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "ChunkRandomizer.h"
#include <random>
#include "RandomOrdering.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    ChunkRandomizer::ChunkRandomizer(IDataDeserializerPtr deserializer, 
        size_t randomizationRange,
        bool sampleBasedRandomizationWindow) :
        m_deserializer(deserializer), 
        m_randomizationRange(randomizationRange),
        m_sampleBasedRandomizationWindow(sampleBasedRandomizationWindow)
    {
        m_originalChunks = m_deserializer->GetChunkDescriptions();
        assert(m_originalChunks.size() < CHUNKID_MAX);
    }

    // Gets randomized chunks.
    const std::vector<RandomizedChunk>& ChunkRandomizer::GetRandomizedChunks() const
    {
        return m_randomizedChunks;
    }

    // Randomizes chunks and calculates randomization windows.
    void ChunkRandomizer::Randomize(unsigned int seed, ParamsMapPtr dataExtendParams, size_t curEpoch)
    {
        std::vector<ChunkIdType> randomizedChunkIndices;
        randomizedChunkIndices.reserve(m_originalChunks.size());
        for (ChunkIdType i = 0; i < m_originalChunks.size(); i++)
        {
            randomizedChunkIndices.push_back(i);
        }

        string extendMode = "none";
        size_t extendEpochs = 0;
        float curDataRatio = 1;
        bool randomFill = false;
        bool useSplitRead = false;
        bool randomData = false;

        if (dataExtendParams != nullptr)
        {
            extendMode = (*dataExtendParams)["extendMode"];
            extendEpochs = std::stoi((*dataExtendParams)["extendEpochs"]);
            if (extendMode != "none" && extendEpochs > 0 && extendEpochs > curEpoch)
            {
                if (extendMode == "equidiff")
                    curDataRatio = (float)1 / extendEpochs * (curEpoch + 1);
                else if (extendMode == "expand")
                    curDataRatio = (float)1 / (extendEpochs - curEpoch);

                randomFill = (*dataExtendParams)["randomFill"] == "true" ? true : false;
                randomData = (*dataExtendParams)["randomData"] == "true" ? true : false;
                useSplitRead = (*dataExtendParams)["useSplitRead"] == "true" ? true : false;

                fprintf(stderr, "Data extend stage, extendMode: %s, dataRatio: %f, randomFill: %s, useSplitRead: %s, randomData: %s",
                    extendMode.c_str(), curDataRatio, (*dataExtendParams)["randomFill"].c_str(), 
                    (*dataExtendParams)["useSplitRead"].c_str(), (*dataExtendParams)["randomData"].c_str());
            }
            else if (extendEpochs < 0)
            {
                InvalidArgument("extend epochs size must bigger than 0");
            }
        }

        if (useSplitRead)
        {
            fprintf(stderr, "use split reader.\n");
            seed = std::random_device()();
        }
        
        m_rng.seed(seed);
        RandomShuffleMT(randomizedChunkIndices, m_rng);

        if (curDataRatio != 1)
        {
            size_t extractSize = (int)std::floor(m_originalChunks.size() * curDataRatio);
            std::vector<ChunkIdType> extractedChunkIndices;

            if (randomData)
            {
                for (ChunkIdType i = 0; i < extractSize; i++)
                    extractedChunkIndices.push_back(randomizedChunkIndices.at(i));
            }
            else
            {
                for (int i = 0; i < m_lastChunksIndices.size(); i++)
                    extractedChunkIndices.push_back(m_lastChunksIndices.at(i));

                ChunkIdType idx = 0;
                while (extractedChunkIndices.size() < extractSize)
                {
                    if (std::find(extractedChunkIndices.begin(), extractedChunkIndices.end(), randomizedChunkIndices.at(idx))
                        == extractedChunkIndices.end())
                    {
                        extractedChunkIndices.push_back(randomizedChunkIndices.at(idx));
                    }
                    idx++;
                }

                m_lastChunksIndices.clear();
                m_lastChunksIndices.reserve(extractedChunkIndices.size());
                for (int i = 0; i < extractedChunkIndices.size(); i++)
                    m_lastChunksIndices.push_back(extractedChunkIndices.at(i));

            }

            size_t repectTime = (int)std::floor(m_originalChunks.size() / extractSize);

            ChunkIdType curIdxPos = 0;
            for (size_t i = 0; i < repectTime; i++)
            {
                if (randomFill)
                {
                    m_rng.seed(std::random_device()());
                    RandomShuffleMT(extractedChunkIndices, m_rng);
                }
                for (ChunkIdType idx = 0; idx < extractSize; idx++)
                    randomizedChunkIndices[curIdxPos++] = extractedChunkIndices[idx];
            }

            ChunkIdType fIdxPos = curIdxPos;
            while (curIdxPos < m_originalChunks.size())
            {
                if (randomFill)
                {
                    std::srand((unsigned)time(NULL));
                    randomizedChunkIndices[curIdxPos++] = extractedChunkIndices[std::rand() % extractSize];
                }
                else
                {
                    randomizedChunkIndices[curIdxPos] = extractedChunkIndices[curIdxPos - fIdxPos];
                    curIdxPos++;
                }
            }

            assert(randomizedChunkIndices.size() == m_originalChunks.size());
        }

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
            const size_t numberOfSamples = m_originalChunks[originalChunkIndex]->m_numberOfSamples;
            const size_t numberOfSequences = m_originalChunks[originalChunkIndex]->m_numberOfSequences;

            RandomizedChunk randomizedChunk;
            randomizedChunk.m_chunkId = chunkIndex;
            randomizedChunk.m_original = m_originalChunks[originalChunkIndex].get();
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
}}}
