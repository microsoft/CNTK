//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <limits>
#include "MLFBinaryDeserializer.h"
#include "MLFDeserializer.h"
#include "ConfigHelper.h"
#include "SequenceData.h"
#include "StringUtil.h"
#include "ReaderConstants.h"
#include "FileWrapper.h"
#include "Index.h"
#include "MLFBinaryIndexBuilder.h"

namespace CNTK {

using namespace std;
using namespace Microsoft::MSR::CNTK;

// MLF chunk when operating in sequence mode.
class MLFBinaryDeserializer::BinarySequenceChunk : public MLFDeserializer::SequenceChunk
{
public:
    BinarySequenceChunk(const MLFBinaryDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName, StateTablePtr states)
        : MLFDeserializer::SequenceChunk(parent, descriptor, fileName, states)
    {
    }

    void CacheSequence(const SequenceDescriptor& sequence, size_t index)
    {
        vector<MLFFrameRange> utterance;

        auto start = this->m_buffer.data() + sequence.OffsetInChunk();

        ushort stateCount = *(ushort*)start;
        utterance.resize(stateCount);
        start += sizeof(ushort);
        uint firstFrame = 0;
        for (size_t i = 0;i < stateCount;i++)
        {
            // Read state label and count
            ushort stateLabel = *(ushort*)start;
            start += sizeof(ushort);

            ushort frameCount = *(ushort*)start;
            start += sizeof(ushort);

            utterance[i].Save(firstFrame, frameCount, stateLabel);
            firstFrame += stateCount;
        }

        this->m_sequences[index] = move(utterance);
    }
};

MLFBinaryDeserializer::MLFBinaryDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : MLFDeserializer(corpus, primary)
{
    if (primary)
        RuntimeError("MLFDeserializer currently does not support primary mode.");

    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    if (m_frameMode)
        LogicError("TODO: support frame mode in Binary MLF deserializer.");

    wstring precision = cfg(L"precision", L"float");
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? DataType::Float : DataType::Double;

    // Same behavior as for the old deserializer - keep almost all in memory,
    // because there are a lot of none aligned sets.
    m_chunkSizeBytes = cfg(L"chunkSizeInBytes", g_64MB);

    ConfigParameters input = cfg("input");
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    m_dimension = config.GetLabelDimension();
    if (m_dimension > numeric_limits<ClassIdType>::max())
        RuntimeError("Label dimension (%zu) exceeds the maximum allowed "
            "value '%ud'\n", m_dimension, numeric_limits<ClassIdType>::max());

    m_withPhoneBoundaries = streamConfig(L"phoneBoundaries", false);
    if (m_withPhoneBoundaries)
        LogicError("TODO: implement phoneBoundaries setting.");

    InitializeStream(inputName);
    InitializeChunkInfos(corpus, config);
}

void MLFBinaryDeserializer::InitializeChunkInfos(CorpusDescriptorPtr corpus, const ConfigHelper& config)
{
    // Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
    vector<wstring> mlfPaths = config.GetMlfPaths();

    auto emptyPair = make_pair(numeric_limits<uint32_t>::max(), numeric_limits<uint32_t>::max());
    size_t totalNumSequences = 0;
    size_t totalNumFrames = 0;
    bool enableCaching = corpus->IsHashingEnabled() && config.GetCacheIndex();
    for (const auto& path : mlfPaths)
    {
        attempt(5, [this, path, enableCaching, corpus]()
        {
            MLFBinaryIndexBuilder builder(FileWrapper(path, L"rbS"), corpus);
            builder.SetChunkSize(m_chunkSizeBytes).SetCachingEnabled(enableCaching);
            m_indices.emplace_back(builder.Build());
        });

        m_mlfFiles.push_back(path);
        
        auto& index = m_indices.back();
        // Build auxiliary for GetSequenceByKey.
        for (const auto& chunk : index->Chunks())
        {
            // Preparing chunk info that will be exposed to the outside.
            auto chunkId = static_cast<ChunkIdType>(m_chunks.size());
            uint32_t offsetInSamples = 0;
            for (uint32_t i = 0; i < chunk.NumberOfSequences(); ++i)
            {
                const auto& sequence = chunk[i];
                auto sequenceIndex = m_frameMode ? offsetInSamples : i;
                offsetInSamples += sequence.m_numberOfSamples;
                m_keyToChunkLocation.push_back(std::make_tuple(sequence.m_key, chunkId, sequenceIndex));
            }

            totalNumSequences += chunk.NumberOfSequences();
            totalNumFrames += chunk.NumberOfSamples();
            m_chunkToFileIndex.insert(make_pair(&chunk, m_mlfFiles.size() - 1));
            m_chunks.push_back(&chunk);
            if (m_chunks.size() >= numeric_limits<ChunkIdType>::max())
                RuntimeError("Number of chunks exceeded overflow limit.");
        }
    }

    std::sort(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(), MLFDeserializer::LessByFirstItem);

    fprintf(stderr, "MLFBinaryDeserializer: '%zu' utterances with '%zu' frames\n",
        totalNumSequences,
        totalNumFrames);

    if (m_frameMode)
        InitializeReadOnlyArrayOfLabels();
}

ChunkPtr MLFBinaryDeserializer::GetChunk(ChunkIdType chunkId)
{
    ChunkPtr result;
    attempt(5, [this, &result, chunkId]()
    {
        auto chunk = m_chunks[chunkId];
        auto& fileName = m_mlfFiles[m_chunkToFileIndex[chunk]];

        result = make_shared<BinarySequenceChunk>(*this, *chunk, fileName, m_stateTable);
    });

    return result;
};

}
