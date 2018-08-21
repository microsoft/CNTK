//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <limits>
#include "MLFDeserializer.h"
#include "ConfigHelper.h"
#include "SequenceData.h"
#include "StringUtil.h"
#include "ReaderConstants.h"
#include "FileWrapper.h"
#include "Index.h"
#include "MLFIndexBuilder.h"
#include "MLFBinaryIndexBuilder.h"

namespace CNTK
{

using namespace std;
using namespace Microsoft::MSR::CNTK;

MLFDeserializer::MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : DataDeserializerBase(primary),
      m_corpus(corpus),
      m_textReader(true)
{
    auto inputName = InitializeReaderParams(cfg, primary);

    ConfigParameters input = cfg("input");
    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    wstring labelMappingFile = streamConfig(L"labelMappingFile", L"");
    InitializeStream(inputName);
    InitializeChunkInfos(corpus, config, labelMappingFile);
}

// TODO: Should be removed. Currently a lot of end to end tests still use this one.
MLFDeserializer::MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& labelConfig, const wstring& name)
    : DataDeserializerBase(false),
      m_textReader(true)
{
    // The frame mode is currently specified once per configuration,
    // not in the configuration of a particular deserializer, but on a higher level in the configuration.
    // Because of that we are using find method below.
    m_frameMode = labelConfig.Find("frameMode", "true");

    ConfigHelper config(labelConfig);

    config.CheckLabelType();
    m_dimension = config.GetLabelDimension();

    if (m_dimension > numeric_limits<ClassIdType>::max())
    {
        RuntimeError("Label dimension (%zu) exceeds the maximum allowed "
                     "value (%zu)\n",
                     m_dimension, (size_t) numeric_limits<ClassIdType>::max());
    }

    // Same behavior as for the old deserializer - keep almost all in memory,
    // because there are a lot of none aligned sets.
    m_chunkSizeBytes = labelConfig(L"chunkSizeInBytes", g_64MB);

    wstring precision = labelConfig(L"precision", L"float");
    ;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? DataType::Float : DataType::Double;

    m_withPhoneBoundaries = labelConfig(L"phoneBoundaries", "false");

    wstring labelMappingFile = labelConfig(L"labelMappingFile", L"");
    InitializeStream(name);
    InitializeChunkInfos(corpus, config, labelMappingFile);
}

MLFDeserializer::MLFDeserializer(CorpusDescriptorPtr corpus, bool primary)
    : DataDeserializerBase(primary),
      m_corpus(corpus),
      m_textReader(true)
{
}

// Initializes chunk descriptions.
void MLFDeserializer::InitializeChunkInfos(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath)
{
    if (!stateListPath.empty())
    {
        m_stateTable = make_shared<StateTable>();
        m_stateTable->ReadStateList(stateListPath);
    }

    // Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
    vector<wstring> mlfPaths = config.GetMlfPaths();

    auto emptyPair = make_pair(numeric_limits<uint32_t>::max(), numeric_limits<uint32_t>::max());
    size_t totalNumSequences = 0;
    size_t totalNumFrames = 0;
    bool enableCaching = corpus->IsHashingEnabled() && config.GetCacheIndex();
    for (const auto& path : mlfPaths)
    {
        attempt(5, [this, path, enableCaching, corpus, stateListPath]() {
            if (m_textReader)
            {
                MLFIndexBuilder builder(FileWrapper(path, L"rbS"), corpus);
                builder.SetChunkSize(m_chunkSizeBytes).SetCachingEnabled(enableCaching);
                m_indices.emplace_back(builder.Build());
            }
            else
            {
                MLFBinaryIndexBuilder builder(FileWrapper(path, L"rbS"), corpus);
                builder.SetChunkSize(m_chunkSizeBytes).SetCachingEnabled(enableCaching);
                m_indices.emplace_back(builder.Build());
            }
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

    std::sort(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(), LessByFirstItem);

    fprintf(stderr, "MLF Deserializer: '%zu' utterances with '%zu' frames\n",
            totalNumSequences,
            totalNumFrames);

    if (m_frameMode)
        InitializeReadOnlyArrayOfLabels();
}

wstring MLFDeserializer::InitializeReaderParams(const ConfigParameters& cfg, bool primary)
{
    if (primary)
        RuntimeError("MLFDeserializer currently does not support primary mode.");

    m_frameMode = (ConfigValue) cfg("frameMode", "true");

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
                     "value '%ud'\n",
                     m_dimension, numeric_limits<ClassIdType>::max());

    m_withPhoneBoundaries = streamConfig(L"phoneBoundaries", false);
    if (m_frameMode && m_withPhoneBoundaries)
        LogicError("frameMode and phoneBoundaries are mutually exclusive options.");

    return inputName;
}

void MLFDeserializer::InitializeReadOnlyArrayOfLabels()
{
    m_categories.reserve(m_dimension);
    m_categoryIndices.reserve(m_dimension);
    for (size_t i = 0; i < m_dimension; ++i)
    {
        auto category = make_shared<CategorySequenceData>(m_streams.front().m_sampleLayout);
        m_categoryIndices.push_back(static_cast<IndexType>(i));
        category->m_indices = &(m_categoryIndices[i]);
        category->m_nnzCounts.resize(1);
        category->m_nnzCounts[0] = 1;
        category->m_totalNnzCount = 1;
        category->m_numberOfSamples = 1;
        if (m_elementType == DataType::Float)
            category->m_data = &s_oneFloat;
        else
            category->m_data = &s_oneDouble;
        m_categories.push_back(category);
    }
}

void MLFDeserializer::InitializeStream(const wstring& name)
{
    // Initializing stream description - a single stream of MLF data.
    StreamInformation stream;
    stream.m_id = 0;
    stream.m_name = name;
    stream.m_sampleLayout = NDShape({m_dimension});
    stream.m_storageFormat = StorageFormat::SparseCSC;
    stream.m_elementType = m_elementType;
    m_streams.push_back(stream);
}

std::vector<ChunkInfo> MLFDeserializer::ChunkInfos()
{
    std::vector<ChunkInfo> chunks;
    chunks.reserve(m_chunks.size());
    for (size_t i = 0; i < m_chunks.size(); ++i)
    {
        ChunkInfo cd;
        cd.m_id = static_cast<ChunkIdType>(i);
        if (cd.m_id != i)
            RuntimeError("ChunkIdType overflow during creation of a chunk description.");

        cd.m_numberOfSequences = m_frameMode ? m_chunks[i]->NumberOfSamples() : m_chunks[i]->NumberOfSequences();
        cd.m_numberOfSamples = m_chunks[i]->NumberOfSamples();
        chunks.push_back(cd);
    }
    return chunks;
}

void MLFDeserializer::SequenceInfosForChunk(ChunkIdType, vector<SequenceInfo>& result)
{
    UNUSED(result);
    LogicError("MLF deserializer does not support primary mode, it cannot control chunking. "
               "Please specify HTK deserializer as the first deserializer in your config file.");
}

ChunkPtr MLFDeserializer::GetChunk(ChunkIdType chunkId)
{
    ChunkPtr result;
    attempt(5, [this, &result, chunkId]() {
        auto chunk = m_chunks[chunkId];
        auto& fileName = m_mlfFiles[m_chunkToFileIndex[chunk]];

        if (m_frameMode)
            result = make_shared<MLFDeserializer::FrameChunk>(*this, *chunk, fileName, m_stateTable);
        else
            result = make_shared<MLFDeserializer::SequenceChunk>(*this, *chunk, fileName, m_stateTable);
    });

    return result;
};

bool MLFDeserializer::GetSequenceInfoByKey(const SequenceKey& key, SequenceInfo& result)
{
    auto found = std::lower_bound(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(), std::make_tuple(key.m_sequence, 0, 0),
                                  LessByFirstItem);

    if (found == m_keyToChunkLocation.end() || std::get<0>(*found) != key.m_sequence)
    {
        return false;
    }

    auto chunkId = std::get<1>(*found);
    auto sequenceIndexInChunk = std::get<2>(*found);

    result.m_chunkId = std::get<1>(*found);
    result.m_key = key;

    if (m_frameMode)
    {
        // in frame mode sequenceIndexInChunk == sequence offset in chunk in samples
        result.m_indexInChunk = sequenceIndexInChunk + key.m_sample;
        result.m_numberOfSamples = 1;
    }
    else
    {
        assert(result.m_key.m_sample == 0);

        const auto* chunk = m_chunks[chunkId];
        const auto& sequence = chunk->Sequences()[sequenceIndexInChunk];
        result.m_indexInChunk = sequenceIndexInChunk;
        result.m_numberOfSamples = sequence.m_numberOfSamples;
    }
    return true;
}
}
