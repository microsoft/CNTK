//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <cfloat>
#include "BufferedFileReader.h"
#include "IndexBuilder.h"
#include "TextDeserializer.h"
#include "TextReaderConstants.h"
#include "TextDataChunk.h"
#include "TextParserInfo.h"
#include "File.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

// Internal, used for testing.
template <class ElemType>
TextDeserializer<ElemType>::TextDeserializer(CorpusDescriptorPtr corpus, const std::wstring& filename, const vector<StreamDescriptor>& streams, bool primary) :
    DataDeserializerBase(primary),
    m_streamDescriptors(streams),
    m_filename(filename),
    m_file(nullptr),
    m_streamInfos(streams.size()),
    m_index(nullptr),
    m_chunkSizeBytes(0),
    m_traceLevel(static_cast<unsigned int>(TraceLevel::Error)),
    m_numAllowedErrors(0),
    m_skipSequenceIds(false),
    m_numRetries(5),
    m_corpus(corpus),
    m_useMaximumAsSequenceLength(true),
    m_cacheIndex(false)
{
    assert(streams.size() > 0);

    m_maxAliasLength = 0;
    size_t definesMbSizeCount = 0;

    for (size_t i = 0; i < streams.size(); ++i)
    {
        const StreamDescriptor& stream = streams[i];

        definesMbSizeCount += stream.m_definesMbSize ? 1 : 0;

        const string& alias = stream.m_alias;
        if (m_maxAliasLength < alias.length())
        {
            m_maxAliasLength = alias.length();
        }
        m_aliasToIdMap[alias] = i;

        StreamInformation streamDescription = stream;
        streamDescription.m_sampleLayout = NDShape({ stream.m_sampleDimension });
        m_streams.push_back(streamDescription);
    }

    m_useMaximumAsSequenceLength = definesMbSizeCount == 0;

    if (definesMbSizeCount > 1)
    {
        wstring names;
        for (const auto& stream : streams)
            if (stream.m_definesMbSize)
            {
                if (!names.empty())
                    names += L", ";
                names += stream.m_name;
            }

        RuntimeError("Only a single stream is allowed to define the minibatch size, but %zu found: %ls.",
            definesMbSizeCount, names.c_str());
    }

    assert(m_maxAliasLength > 0);

}

template <class ElemType>
TextDeserializer<ElemType>::TextDeserializer(CorpusDescriptorPtr corpus, const TextConfigHelper& helper, bool primary) :
    TextDeserializer(corpus, helper.GetFilePath(), helper.GetStreams(), primary)
{
    SetTraceLevel(helper.GetTraceLevel());
    SetMaxAllowedErrors(helper.GetMaxAllowedErrors());
    SetChunkSize(helper.GetChunkSize());
    SetSkipSequenceIds(helper.ShouldSkipSequenceIds());

    SetCacheIndex(helper.ShouldCacheIndex());

    Initialize();
}

template <class ElemType>
void TextDeserializer<ElemType>::Initialize()
{
    if (m_index != nullptr)
    {
        return;
    }

    attempt(m_numRetries, [this]()
    {
        m_file = std::make_shared<FileWrapper>(m_filename, L"rbS");

        m_file->CheckIsOpenOrDie();

        if (m_file->CheckUnicode())
        {
            // Retrying won't help here, the file is UTF-16 encoded.
            m_numRetries = 0;
            RuntimeError("Found a UTF-16 BOM at the beginning of the input file (%ls). "
                "UTF-16 encoding is currently not supported.", m_filename.c_str());
        }

        TextInputIndexBuilder builder(*m_file);

        builder.SetSkipSequenceIds(m_skipSequenceIds)
            .SetStreamPrefix(NAME_PREFIX)
            .SetCorpus(m_corpus)
            .SetPrimary(m_primary)
            .SetChunkSize(m_chunkSizeBytes)
            .SetCachingEnabled(m_cacheIndex);

        if (!m_useMaximumAsSequenceLength)
        {
            auto mainStream = std::find_if(m_streamDescriptors.begin(), m_streamDescriptors.end(),
                [](const StreamDescriptor& s) { return s.m_definesMbSize; });
            builder.SetMainStream(mainStream->m_alias);
        }

        m_index = builder.Build();

    });

    assert(m_index != nullptr);

    m_parserInfo.reset(new TextParserInfo(m_filename, m_traceLevel, m_numAllowedErrors, m_maxAliasLength, m_useMaximumAsSequenceLength, m_aliasToIdMap, m_streams));
}

template <class ElemType>
TextDeserializer<ElemType>::~TextDeserializer() = default;

template <class ElemType>
ChunkPtr TextDeserializer<ElemType>::GetChunk(ChunkIdType chunkId)
{
    const auto& chunkDescriptor = m_index->Chunks()[chunkId];
    auto textChunk = make_shared <TextDataChunk<ElemType>>(m_parserInfo);

    attempt(m_numRetries, [this, &textChunk, &chunkDescriptor]()
    {
        if (m_file->CheckError())
        {
            m_file.reset(new FileWrapper(m_filename, L"rbS"));
            m_file->CheckIsOpenOrDie();
        }

        LoadChunk(textChunk, chunkDescriptor);
    });

    return textChunk;
}

template <class ElemType>
std::vector<ChunkInfo> TextDeserializer<ElemType>::ChunkInfos()
{
    assert(m_index != nullptr);

    std::vector<ChunkInfo> result;
    result.reserve(m_index->Chunks().size());
    for (ChunkIdType i = 0; i < m_index->Chunks().size(); ++i)
    {
        result.push_back(ChunkInfo{
            i,
            m_index->Chunks()[i].NumberOfSamples(),
            m_index->Chunks()[i].NumberOfSequences()
        });
    }

    return result;
}

template <class ElemType>
void TextDeserializer<ElemType>::SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result)
{
    const auto& chunk = m_index->Chunks()[chunkId];
    result.reserve(chunk.NumberOfSequences());

    for (size_t sequenceIndex = 0; sequenceIndex < chunk.NumberOfSequences(); ++sequenceIndex)
    {
        auto const& s = chunk.Sequences()[sequenceIndex];
        result.push_back(
        {
            sequenceIndex,
            s.m_numberOfSamples,
            chunkId,
            SequenceKey{ s.m_key, 0 }
        });
    }
}

template <class ElemType>
void TextDeserializer<ElemType>::LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor)
{
    const auto& numOfSequences = descriptor.NumberOfSequences();
    chunk->m_sequenceMap.resize(numOfSequences);

    chunk->m_offsetInFile = descriptor.StartOffset();

    chunk->m_buffer = make_unique<char[]>(descriptor.SizeInBytes());

    m_file->SeekOrDie(descriptor.StartOffset(), SEEK_SET);

    // Reading the whole chunk at once
    m_file->ReadOrDie(chunk->m_buffer.get(), sizeof(char), descriptor.SizeInBytes());

    for (size_t sequenceIndex = 0; sequenceIndex < numOfSequences; ++sequenceIndex)
    {
        const auto& sequenceDescriptor = descriptor.Sequences()[sequenceIndex];
        chunk->m_sequenceDescriptors.push_back(sequenceDescriptor);
    }
}

template <class ElemType>
void TextDeserializer<ElemType>::SetNumRetries(unsigned int numRetries)
{
    m_numRetries = numRetries;
}

template <class ElemType>
void TextDeserializer<ElemType>::SetTraceLevel(unsigned int traceLevel)
{
    m_traceLevel = traceLevel;
}

template <class ElemType>
void TextDeserializer<ElemType>::SetMaxAllowedErrors(unsigned int maxErrors)
{
    m_numAllowedErrors = maxErrors;
}

template <class ElemType>
void TextDeserializer<ElemType>::SetSkipSequenceIds(bool skip)
{
    m_skipSequenceIds = skip;
}

template <class ElemType>
void TextDeserializer<ElemType>::SetChunkSize(size_t size)
{
    m_chunkSizeBytes = size;
}

template <class ElemType>
void TextDeserializer<ElemType>::SetCacheIndex(bool value)
{
    m_cacheIndex = value;
}

template <class ElemType>
bool TextDeserializer<ElemType>::GetSequenceInfoByKey(const SequenceKey& key, SequenceInfo& r)
{
    return DataDeserializerBase::GetSequenceInfoByKey(*m_index, key, r);
}

template class TextDeserializer<float>;
template class TextDeserializer<double>;

}

