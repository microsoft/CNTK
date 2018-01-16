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

namespace CNTK {

using namespace std;
using namespace Microsoft::MSR::CNTK;

static float s_oneFloat = 1.0;
static double s_oneDouble = 1.0;

// A constant used in 1-hot vectors to identify the first frame of a phone.
// Used only in CTC-type training.
static float s_phoneBoundary = 2.0f;

// Sparse labels for an utterance.
template <class ElemType>
struct MLFSequenceData : SparseSequenceData
{
    vector<ElemType> m_values;
    vector<IndexType> m_indexBuffer;
    const NDShape& m_frameShape;

    MLFSequenceData(size_t numberOfSamples, const NDShape& frameShape) :
        m_values(numberOfSamples, 1), m_frameShape(frameShape)
    {
        if (numberOfSamples > numeric_limits<IndexType>::max())
        {
            RuntimeError("Number of samples in an MLFSequenceData (%zu) "
                "exceeds the maximum allowed value (%zu)\n",
                numberOfSamples, (size_t)numeric_limits<IndexType>::max());
        }

        m_indexBuffer.resize(numberOfSamples);
        m_nnzCounts.resize(numberOfSamples, static_cast<IndexType>(1));
        m_numberOfSamples = (uint32_t)numberOfSamples;
        m_totalNnzCount = static_cast<IndexType>(numberOfSamples);
        m_indices = &m_indexBuffer[0];
    }

    MLFSequenceData(size_t numberOfSamples, const vector<size_t>& phoneBoundaries, const NDShape& frameShape) :
        MLFSequenceData(numberOfSamples, frameShape)
    {
        for (auto boundary : phoneBoundaries)
            m_values[boundary] = s_phoneBoundary;
    }

    const void* GetDataBuffer() override
    {
        return m_values.data();
    }

    const NDShape& GetSampleShape() override
    {
        return m_frameShape;
    }
};

// Base class for chunks in frame and sequence mode.
// The lifetime is always less than the lifetime of the parent deserializer.
class MLFDeserializer::ChunkBase : public Chunk
{
protected:
    vector<char> m_buffer;   // Buffer for the whole chunk
    vector<bool> m_valid;    // Bit mask whether the parsed sequence is valid.
    MLFUtteranceParser m_parser;

    const MLFDeserializer& m_deserializer;
    const ChunkDescriptor& m_descriptor;     // Current chunk descriptor.

    ChunkBase(const MLFDeserializer& deserializer, const ChunkDescriptor& descriptor, const wstring& fileName, const StateTablePtr& states)
        : m_parser(states),
          m_descriptor(descriptor),
          m_deserializer(deserializer)
    {
        if (descriptor.NumberOfSequences() == 0 || descriptor.SizeInBytes() == 0)
            LogicError("Empty chunks are not supported.");

        auto f = FileWrapper::OpenOrDie(fileName, L"rbS");
        size_t sizeInBytes = descriptor.SizeInBytes();

        // Make sure we always have 0 at the end for buffer overrun.
        m_buffer.resize(sizeInBytes + 1);
        m_buffer[sizeInBytes] = 0;

        // Seek and read chunk into memory.
        f.SeekOrDie(descriptor.StartOffset(), SEEK_SET);

        f.ReadOrDie(m_buffer.data(), sizeInBytes, 1);

        // all sequences are valid by default.
        m_valid.resize(m_descriptor.NumberOfSequences(), true);
    }

    string KeyOf(const SequenceDescriptor& s)
    {
        return m_deserializer.m_corpus->IdToKey(s.m_key);
    }

    void CleanBuffer()
    {
        // Make sure we do not keep unnecessary memory after sequences have been parsed.
        vector<char> tmp;
        m_buffer.swap(tmp);
    }
};

// MLF chunk when operating in sequence mode.
class MLFDeserializer::SequenceChunk : public MLFDeserializer::ChunkBase
{
    vector<vector<MLFFrameRange>> m_sequences; // Each sequence is a vector of sequential frame ranges.

public:
    SequenceChunk(const MLFDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName, StateTablePtr states)
        : ChunkBase(parent, descriptor, fileName, states)
    {
        m_sequences.resize(m_descriptor.Sequences().size());

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < descriptor.Sequences().size(); ++i)
            CacheSequence(descriptor.Sequences()[i], i);

        CleanBuffer();
    }

    void CacheSequence(const SequenceDescriptor& sequence, size_t index)
    {
        auto start = m_buffer.data() + sequence.OffsetInChunk();
        auto end = start + sequence.SizeInBytes();

        vector<MLFFrameRange> utterance;
        auto absoluteOffset = m_descriptor.StartOffset() + sequence.OffsetInChunk();
        bool parsed = m_parser.Parse(boost::make_iterator_range(start, end), utterance, absoluteOffset);
        if (!parsed) // cannot parse
        {
            fprintf(stderr, "WARNING: Cannot parse the utterance '%s'\n", KeyOf(sequence).c_str());
            m_valid[index] = false;
            return;
        }

        m_sequences[index] = move(utterance);
    }

    void GetSequence(size_t sequenceIndex, vector<SequenceDataPtr>& result) override
    {
        if (m_deserializer.m_elementType == DataType::Float)
            return GetSequence<float>(sequenceIndex, result);
        else
        {
            assert(m_deserializer.m_elementType == DataType::Double);
            return GetSequence<double>(sequenceIndex, result);
        }
    }

    template<class ElementType>
    void GetSequence(size_t sequenceIndex, vector<SequenceDataPtr>& result)
    {
        if (!m_valid[sequenceIndex])
        {
            SparseSequenceDataPtr s = make_shared<MLFSequenceData<ElementType>>(0, m_deserializer.m_streams.front().m_sampleLayout);
            s->m_isValid = false;
            result.push_back(s);
            return;
        }

        const auto& utterance = m_sequences[sequenceIndex];
        const auto& sequence = m_descriptor.Sequences()[sequenceIndex];

        // Packing labels for the utterance into sparse sequence.
        vector<size_t> sequencePhoneBoundaries(m_deserializer.m_withPhoneBoundaries ? utterance.size() : 0);
        if (m_deserializer.m_withPhoneBoundaries)
        {
            for (size_t i = 0; i < utterance.size(); ++i)
                sequencePhoneBoundaries[i] = utterance[i].FirstFrame();
        }

        auto s = make_shared<MLFSequenceData<ElementType>>(sequence.m_numberOfSamples, sequencePhoneBoundaries, m_deserializer.m_streams.front().m_sampleLayout);
        auto* startRange = s->m_indices;
        for (const auto& range : utterance)
        {
            if (range.ClassId() >= m_deserializer.m_dimension)
                // TODO: Possibly set m_valid to false, but currently preserving the old behavior.
                RuntimeError("Class id '%ud' exceeds the model output dimension '%d'.", range.ClassId(), (int)m_deserializer.m_dimension);

            // Filling all range of frames with the corresponding class id.
            fill(startRange, startRange + range.NumFrames(), static_cast<IndexType>(range.ClassId()));
            startRange += range.NumFrames();
        }

        result.push_back(s);
    }
};

// MLF chunk when operating in frame mode.
// Implementation is different because frames of the same sequence can be accessed
// in parallel by the randomizer, so all parsing/preprocessing should be done during
// sequence caching, so that GetSequence only works with read only data structures.
class MLFDeserializer::FrameChunk : public MLFDeserializer::ChunkBase
{
    // Actual values of frames.
    vector<ClassIdType> m_classIds;

    //For each sequence this vector contains the sequence offset in samples from the beginning of the chunk.
    std::vector<uint32_t> m_sequenceOffsetInChunkInSamples;

public:
    FrameChunk(const MLFDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName, StateTablePtr states)
        : ChunkBase(parent, descriptor, fileName, states)
    {
        uint32_t numSamples = static_cast<uint32_t>(m_descriptor.NumberOfSamples());

        // The current assumption is that the number of samples in a chunk fits in uint32, 
        // therefore we can save 4 bytes per sequence, storing offsets in samples as uint32.
        if (numSamples != m_descriptor.NumberOfSamples())
            RuntimeError("Exceeded maximum number of samples in a chunk");

        // Preallocate a big array for filling in class ids for the whole chunk.
        m_classIds.resize(numSamples);
        m_sequenceOffsetInChunkInSamples.resize(m_descriptor.NumberOfSequences());


        uint32_t offset = 0;
        for (auto i = 0; i < m_descriptor.NumberOfSequences(); ++i)
        {
            m_sequenceOffsetInChunkInSamples[i] = offset;
            offset += descriptor[i].m_numberOfSamples;
        }

        if (numSamples != offset)
            RuntimeError("Unexpected number of samples in a FrameChunk.");

        // Parse the data on different threads to avoid locking during GetSequence calls.
#pragma omp parallel for schedule(dynamic)
        for (auto i = 0; i < m_descriptor.NumberOfSequences(); ++i)
            CacheSequence(descriptor[i], i);
        
            
        CleanBuffer();
    }

    // Get utterance by the absolute frame index in chunk.
    // Uses the upper bound to do the binary search among sequences of the chunk.
    size_t GetUtteranceForChunkFrameIndex(size_t frameIndex) const
    {
        auto result = upper_bound(
            m_sequenceOffsetInChunkInSamples.begin(),
            m_sequenceOffsetInChunkInSamples.end(),
            frameIndex,
            [](size_t fi, const size_t& a) { return fi < a; });
        return result - 1 - m_sequenceOffsetInChunkInSamples.begin();
    }

    void GetSequence(size_t sequenceIndex, vector<SequenceDataPtr>& result) override
    {
        size_t utteranceId = GetUtteranceForChunkFrameIndex(sequenceIndex);
        if (!m_valid[utteranceId])
        {
            SparseSequenceDataPtr s = make_shared<MLFSequenceData<float>>(0, m_deserializer.m_streams.front().m_sampleLayout);
            s->m_isValid = false;
            result.push_back(s);
            return;
        }

        size_t label = m_classIds[sequenceIndex];
        assert(label < m_deserializer.m_categories.size());
        result.push_back(m_deserializer.m_categories[label]);
    }

    // Parses and caches sequence in the buffer for GetSequence fast retrieval.
    void CacheSequence(const SequenceDescriptor& sequence, size_t index)
    {
        auto start = m_buffer.data() + sequence.OffsetInChunk();
        auto end = start + sequence.SizeInBytes();

        vector<MLFFrameRange> utterance;
        auto absoluteOffset = m_descriptor.StartOffset() + sequence.OffsetInChunk();
        bool parsed = m_parser.Parse(boost::make_iterator_range(start, end), utterance, absoluteOffset);
        if (!parsed)
        {
            m_valid[index] = false;
            fprintf(stderr, "WARNING: Cannot parse the utterance %s\n", KeyOf(sequence).c_str());
            return;
        }

        auto startRange = m_classIds.begin() + m_sequenceOffsetInChunkInSamples[index];
        for(size_t i = 0; i < utterance.size(); ++i)
        {
            const auto& range = utterance[i];
            if (range.ClassId() >= m_deserializer.m_dimension)
                // TODO: Possibly set m_valid to false, but currently preserving the old behavior.
                RuntimeError("Class id '%ud' exceeds the model output dimension '%d'.", range.ClassId(), (int)m_deserializer.m_dimension);

            fill(startRange, startRange + range.NumFrames(), range.ClassId());
            startRange += range.NumFrames();
        }
    }
};

MLFDeserializer::MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : DataDeserializerBase(primary),
    m_corpus(corpus)
{
    if (primary)
        RuntimeError("MLFDeserializer currently does not support primary mode.");

    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    wstring precision = cfg(L"precision", L"float");;
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
    if (m_frameMode && m_withPhoneBoundaries)
        LogicError("frameMode and phoneBoundaries are mutually exclusive options.");

    wstring labelMappingFile = streamConfig(L"labelMappingFile", L"");
    InitializeStream(inputName);
    InitializeChunkInfos(corpus, config, labelMappingFile);
}

// TODO: Should be removed. Currently a lot of end to end tests still use this one.
MLFDeserializer::MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& labelConfig, const wstring& name)
    : DataDeserializerBase(false)
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
            "value (%zu)\n", m_dimension, (size_t)numeric_limits<ClassIdType>::max());
    }

    // Same behavior as for the old deserializer - keep almost all in memory,
    // because there are a lot of none aligned sets.
    m_chunkSizeBytes = labelConfig(L"chunkSizeInBytes", g_64MB);

    wstring precision = labelConfig(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? DataType::Float : DataType::Double;

    m_withPhoneBoundaries = labelConfig(L"phoneBoundaries", "false");

    wstring labelMappingFile = labelConfig(L"labelMappingFile", L"");
    InitializeStream(name);
    InitializeChunkInfos(corpus, config, labelMappingFile);
}

static inline bool LessByFirstItem(const std::tuple<size_t, size_t, size_t>& a, const std::tuple<size_t, size_t, size_t>& b)
{
    return std::get<0>(a) < std::get<0>(b);
}

void MLFDeserializer::InitializeChunkInfos(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath)
{
    // Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
    vector<wstring> mlfPaths = config.GetMlfPaths();

    if (!stateListPath.empty())
    {
        m_stateTable = make_shared<StateTable>();
        m_stateTable->ReadStateList(stateListPath);
    }

    auto emptyPair = make_pair(numeric_limits<uint32_t>::max(), numeric_limits<uint32_t>::max());
    size_t totalNumSequences = 0;
    size_t totalNumFrames = 0;
    bool enableCaching = corpus->IsHashingEnabled() && config.GetCacheIndex();
    for (const auto& path : mlfPaths)
    {
        attempt(5, [this, path, enableCaching, corpus]()
        {
            MLFIndexBuilder builder(FileWrapper(path, L"rbS"), corpus);
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

    std::sort(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(), LessByFirstItem);

    fprintf(stderr, "MLFDeserializer: '%zu' utterances with '%zu' frames\n",
        totalNumSequences,
        totalNumFrames);

    if (m_frameMode)
        InitializeReadOnlyArrayOfLabels();
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
    stream.m_sampleLayout = NDShape({ m_dimension });
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

        cd.m_numberOfSequences =  m_frameMode ? m_chunks[i]->NumberOfSamples() : m_chunks[i]->NumberOfSequences();
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
    attempt(5, [this, &result, chunkId]()
    {
        auto chunk = m_chunks[chunkId];
        auto& fileName = m_mlfFiles[m_chunkToFileIndex[chunk]];

        if (m_frameMode)
            result = make_shared<FrameChunk>(*this, *chunk, fileName, m_stateTable);
        else
            result = make_shared<SequenceChunk>(*this, *chunk, fileName, m_stateTable);
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
