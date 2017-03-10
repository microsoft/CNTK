//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <limits>
#include "MLFDeserializer.h"
#include "ConfigHelper.h"
#include "SequenceData.h"
#include "../HTKMLFReader/htkfeatio.h"
#include "../HTKMLFReader/msra_mgram.h"
#include "latticearchive.h"
#include "StringUtil.h"
#include "MLFIndexer.h"
#include "MLFUtils.h"
#include "ReaderConstants.h"

#undef max // max is defined in minwindef.h

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

static float s_oneFloat = 1.0;
static double s_oneDouble = 1.0;

// A constant used in 1-hot vectors to identify the first frame of a phone.
// Used primarily in CTC-type training.
static float PHONE_BOUNDARY = 2.0f;

// Sparse labels for an utterance.
template <class ElemType>
struct MLFSequenceData : SparseSequenceData
{
    vector<ElemType> m_values;
    unique_ptr<IndexType[]> m_indicesPtr;

    MLFSequenceData(size_t numberOfSamples) :
        m_values(numberOfSamples, 1),
        m_indicesPtr(new IndexType[numberOfSamples])
    {
        if (numberOfSamples > numeric_limits<IndexType>::max())
        {
            RuntimeError("Number of samples in an MLFSequence (%" PRIu64 ") "
                "exceeds the maximum allowed value (%" PRIu64 ")\n",
                numberOfSamples, (size_t)numeric_limits<IndexType>::max());
        }

        m_nnzCounts.resize(numberOfSamples, static_cast<IndexType>(1));
        m_numberOfSamples = (uint32_t)numberOfSamples;
        m_totalNnzCount = static_cast<IndexType>(numberOfSamples);
        m_indices = m_indicesPtr.get();
    }

    MLFSequenceData(size_t numberOfSamples, const vector<size_t>& phoneBoundaries) :
        MLFSequenceData(numberOfSamples)
    {
        for (auto boundary : phoneBoundaries)
            m_values[boundary] = PHONE_BOUNDARY;
    }

    const void* GetDataBuffer() override
    {
        return m_values.data();
    }
};

// Base chunk for frame and sequence mode.
class MLFDeserializer::ChunkBase : public Chunk
{
protected:
    std::vector<char> m_buffer;
    MLFUtteranceParser m_parser;
    std::vector<bool> m_valid;

    const MLFDeserializer& m_parent;
    const ChunkDescriptor& m_descriptor;

public:
    ChunkBase(const MLFDeserializer& parent, const ChunkDescriptor& descriptor, const std::wstring& fileName, StateTablePtr states)
        : m_parser(states),
          m_descriptor(descriptor),
          m_parent(parent)
    {
        std::shared_ptr<FILE> f = std::shared_ptr<FILE>(fopenOrDie(fileName, L"rbS"), [](FILE *f) { if (f) fclose(f); });

        if (descriptor.m_sequences.empty() || !descriptor.m_byteSize)
            LogicError("Empty chunks are not supported.");

        size_t sizeInBytes = descriptor.m_sequences.back().m_fileOffsetBytes + descriptor.m_sequences.back().m_byteSize -
            descriptor.m_sequences.front().m_fileOffsetBytes;

        m_buffer.resize(sizeInBytes + 1);

        // Make sure we always have 0 at the end for buffer overrun.
        m_buffer[sizeInBytes] = 0;

        auto chunkOffset = descriptor.m_sequences.front().m_fileOffsetBytes;

        // Seek and read chunk into memory.
        int rc = _fseeki64(f.get(), chunkOffset, SEEK_SET);
        if (rc)
            RuntimeError("Error seeking to position '%" PRId64 "' in the input file '%ls', error code '%d'", chunkOffset, fileName.c_str(), rc);

        freadOrDie(m_buffer.data(), sizeInBytes, 1, f.get());

        m_valid.resize(m_descriptor.m_numberOfSequences, true);
    }
};

// Sequence MLF chunk. The time of life always less than the time of life of the parent deserializer.
class MLFDeserializer::SequenceChunk : public MLFDeserializer::ChunkBase
{
    std::vector<std::vector<MLFFrameRange>> m_sequences;

public:
    SequenceChunk(const MLFDeserializer& parent, const ChunkDescriptor& descriptor, const std::wstring& fileName, StateTablePtr states)
        : ChunkBase(parent, descriptor, fileName, states)
    {
        m_sequences.resize(m_descriptor.m_numberOfSequences);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < descriptor.m_sequences.size(); ++i)
            CacheSequence(descriptor.m_sequences[i]);

        std::vector<char> tmp;
        m_buffer.swap(tmp);
    }

    void CacheSequence(const SequenceDescriptor& sequence)
    {
        auto start = m_buffer.data() + (sequence.m_fileOffsetBytes - m_descriptor.m_sequences.front().m_fileOffsetBytes);
        auto end = start + sequence.m_byteSize;

        std::vector<MLFFrameRange> utterance;
        bool parsed = m_parser.Parse(sequence, boost::make_iterator_range(start, end), utterance);
        if (!parsed) // cannot parse
        {
            fprintf(stderr, "WARNING: Cannot parse the utterance %s\n", m_parent.m_corpus->IdToKey(sequence.m_key.m_sequence).c_str());
            m_valid[sequence.m_indexInChunk] = false;
            return;
        }
        m_sequences[sequence.m_indexInChunk] = std::move(utterance);
    }

    void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
    {
        if (!m_valid[sequenceIndex])
        {
            SparseSequenceDataPtr s = make_shared<MLFSequenceData<float>>(0);
            s->m_isValid = false;
            result.push_back(s);
            return;
        }

        const auto& utterance = m_sequences[sequenceIndex];
        const auto& sequence = m_descriptor.m_sequences[sequenceIndex];

        // Compute some statistics and perform checks.
        vector<size_t> sequencePhoneBoundaries(m_parent.m_withPhoneBoundaries ? utterance.size() : 0);
        if (m_parent.m_withPhoneBoundaries)
        {
            for (size_t i = 0; i < utterance.size(); ++i)
                sequencePhoneBoundaries[i] = utterance[i].FirstFrame();
        }

        // Packing labels for the utterance into sparse sequence.
        SparseSequenceDataPtr s;
        if (m_parent.m_elementType == ElementType::tfloat)
            s = make_shared<MLFSequenceData<float>>(sequence.m_numberOfSamples, sequencePhoneBoundaries);
        else
        {
            assert(m_parent.m_elementType == ElementType::tdouble);
            s = make_shared<MLFSequenceData<double>>(sequence.m_numberOfSamples, sequencePhoneBoundaries);
        }

        auto startRange = s->m_indices;
        for (const auto& f : utterance)
        {
            std::fill(startRange, startRange + f.NumFrames(), static_cast<IndexType>(f.ClassId()));
            startRange += f.NumFrames();

            if (f.ClassId() >= m_parent.m_dimension)
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)f.ClassId(), (int)m_parent.m_dimension);
        }

        result.push_back(s);
    }
};

// MLF chunk. The time of life always less than the time of life of the parent deserializer.
class MLFDeserializer::FrameChunk : public MLFDeserializer::ChunkBase
{
    // Actual values of frames.
    std::vector<ClassIdType> m_classIds;

public:
    FrameChunk(const MLFDeserializer& parent, const ChunkDescriptor& descriptor, const std::wstring& fileName, StateTablePtr states)
        : ChunkBase(parent, descriptor, fileName, states)
    {
        // Let's also preallocate an big array for filling in class ids for whole chunk,
        // it is used for optimizing speed of retrieval in frame mode.
        m_classIds.resize(m_descriptor.m_numberOfSamples);

        // Parse the data on different threads to avoid locking during GetSequence calls.
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < descriptor.m_sequences.size(); ++i)
            CacheSequence(descriptor.m_sequences[i]);

        std::vector<char> tmp;
        m_buffer.swap(tmp);
    }

    // Get utterance by the absolute frame index in chunk.
    // Uses the upper bound to do the binary search among sequences of the chunk.
    size_t GetUtteranceForChunkFrameIndex(size_t frameIndex) const
    {
        auto result = std::upper_bound(
            m_descriptor.m_firstSamples.begin(),
            m_descriptor.m_firstSamples.end(),
            frameIndex,
            [](size_t fi, const size_t& a) { return fi < a; });
        return result - 1 - m_descriptor.m_firstSamples.begin();
    }

    void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
    {
        size_t utteranceId = GetUtteranceForChunkFrameIndex(sequenceIndex);
        if (!m_valid[utteranceId])
        {
            SparseSequenceDataPtr s = make_shared<MLFSequenceData<float>>(0);
            s->m_isValid = false;
            result.push_back(s);
            return;
        }

        size_t label = m_classIds[sequenceIndex];
        assert(label < m_parent.m_categories.size());
        result.push_back(m_parent.m_categories[label]);
    }

    // Parses and caches sequence in the buffer for future fast retrieval in frame mode.
    void CacheSequence(const SequenceDescriptor& sequence)
    {
        auto start = m_buffer.data() + (sequence.m_fileOffsetBytes - m_descriptor.m_sequences.front().m_fileOffsetBytes);
        auto end = start + sequence.m_byteSize;

        std::vector<MLFFrameRange> utterance;
        bool parsed = m_parser.Parse(sequence, boost::make_iterator_range(start, end), utterance);
        if (!parsed)
        {
            m_valid[sequence.m_indexInChunk] = false;
            fprintf(stderr, "WARNING: Cannot parse the utterance %s\n", m_parent.m_corpus->IdToKey(sequence.m_key.m_sequence).c_str());
            return;
        }

        auto startRange = m_classIds.begin() + m_descriptor.m_firstSamples[sequence.m_indexInChunk];
        for(size_t i = 0; i < utterance.size(); ++i)
        {
            const auto& range = utterance[i];
            if (range.ClassId() >= m_parent.m_dimension)
                // TODO: Possibly set m_valid to false, but currently preserving the old behavior.
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)range.ClassId(), (int)m_parent.m_dimension);

            std::fill(startRange, startRange + utterance[i].NumFrames(), range.ClassId());
            startRange += utterance[i].NumFrames();
        }
    }
};

MLFDeserializer::MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : DataDeserializerBase(primary)
{
    if (primary)
        RuntimeError("MLFDeserializer currently does not support primary mode.");

    // TODO: This should be read in one place, potentially given by SGD.
    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    argvector<ConfigValue> inputs = cfg("input");
    if (inputs.size() != 1)
        LogicError("MLFDeserializer supports a single input stream only.");

    std::wstring precision = cfg(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? ElementType::tfloat : ElementType::tdouble;

    // Same behavior as for the old deserializer - keep almost all in memory,
    // because there are a lot of none aligned sets.
    m_chunkSizeBytes = cfg(L"chunkSizeInBytes", g_64MB);

    ConfigParameters input = inputs.front();
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    m_dimension = config.GetLabelDimension();

    m_withPhoneBoundaries = streamConfig(L"phoneBoundaries", false);
    if (m_frameMode && m_withPhoneBoundaries)
        LogicError("frameMode and phoneBoundaries are not supposed to be used together.");

    wstring labelMappingFile = streamConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, m_dimension);
    InitializeStream(inputName, m_dimension);
}

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

    if (m_dimension > numeric_limits<IndexType>::max())
    {
        RuntimeError("Label dimension (%" PRIu64 ") exceeds the maximum allowed "
            "value (%" PRIu64 ")\n", m_dimension, (size_t)numeric_limits<IndexType>::max());
    }

    // Same behavior as for the old deserializer - keep almost all in memory,
    // because there are a lot of none aligned sets.
    m_chunkSizeBytes = labelConfig(L"chunkSizeInBytes", g_64MB);

    std::wstring precision = labelConfig(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? ElementType::tfloat : ElementType::tdouble;

    m_withPhoneBoundaries = labelConfig(L"phoneBoundaries", "false");

    wstring labelMappingFile = labelConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, m_dimension);
    InitializeStream(name, m_dimension);
}

// Currently we create a single chunk only.
void MLFDeserializer::InitializeChunkDescriptions(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath, size_t dimension)
{
    // TODO: Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
    vector<wstring> mlfPaths = config.GetMlfPaths();

    if (!stateListPath.empty())
    {
        m_stateTable = std::make_shared<StateTable>();
        m_stateTable->ReadStateList(stateListPath);
    }

    auto emptyPair = std::pair<const ChunkDescriptor*, const SequenceDescriptor*>(nullptr, nullptr);
    size_t totalNumSequences = 0;
    size_t totalNumFrames = 0;
    for (const auto& path : mlfPaths)
    {
        std::shared_ptr<MLFIndexer> indexer;

        attempt(5, [this, &indexer, path, corpus]()
        {
            std::shared_ptr<FILE> file = std::shared_ptr<FILE>(fopenOrDie(path, L"rbS"), [](FILE *f) { if (f) fclose(f); });
            indexer = std::make_shared<MLFIndexer>(file.get(), m_frameMode, m_chunkSizeBytes);
            indexer->Build(corpus);
        });

        m_mlfFiles.push_back(path);
        m_indexers.push_back(make_pair(path, indexer));

        // Build some auxiliary information.
        const auto& index = indexer->GetIndex();
        for (const auto& chunk : index.m_chunks)
        {
            // Preparing chunk info that will be exposed to the outside.
            for (size_t i = 0; i < chunk.m_sequences.size(); ++i)
            {
                const auto& sequence = chunk.m_sequences[i];

                if (m_keyToSequence.size() <= sequence.m_key.m_sequence)
                    m_keyToSequence.resize(sequence.m_key.m_sequence + 1, emptyPair);

                assert(m_keyToSequence[sequence.m_key.m_sequence] == emptyPair);
                m_keyToSequence[sequence.m_key.m_sequence] = std::make_pair(&chunk, &sequence);
            }

            totalNumSequences += chunk.m_numberOfSequences;
            totalNumFrames += chunk.m_numberOfSamples;
            m_chunkToFileIndex.insert(make_pair(&chunk, m_mlfFiles.size() - 1));
            m_chunks.push_back(&chunk);
        }
    }

    std::sort(m_chunks.begin(), m_chunks.end());

    fprintf(stderr, "MLFDeserializer::MLFDeserializer: %" PRIu64 " utterances with %" PRIu64 " frames\n",
        totalNumSequences,
        totalNumFrames);

    if (m_frameMode)
    {
        // Initializing array of labels.
        m_categories.reserve(dimension);
        m_categoryIndices.reserve(dimension);
        for (size_t i = 0; i < dimension; ++i)
        {
            auto category = make_shared<CategorySequenceData>();
            m_categoryIndices.push_back(static_cast<IndexType>(i));
            category->m_indices = &(m_categoryIndices[i]);
            category->m_nnzCounts.resize(1);
            category->m_nnzCounts[0] = 1;
            category->m_totalNnzCount = 1;
            category->m_numberOfSamples = 1;
            if (m_elementType == ElementType::tfloat)
            {
                category->m_data = &s_oneFloat;
            }
            else
            {
                assert(m_elementType == ElementType::tdouble);
                category->m_data = &s_oneDouble;
            }
            m_categories.push_back(category);
        }
    }
}

void MLFDeserializer::InitializeStream(const wstring& name, size_t dimension)
{
    // Initializing stream description - a single stream of MLF data.
    StreamDescriptionPtr stream = make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = name;
    stream->m_sampleLayout = make_shared<TensorShape>(dimension);
    stream->m_storageType = StorageType::sparse_csc;
    stream->m_elementType = m_elementType;
    m_streams.push_back(stream);
}

ChunkDescriptions MLFDeserializer::GetChunkDescriptions()
{
    ChunkDescriptions chunks;
    chunks.reserve(m_chunks.size());
    for (size_t i = 0; i < m_chunks.size(); ++i)
    {
        auto cd = make_shared<ChunkDescription>();
        cd->m_id = static_cast<ChunkIdType>(i);
        cd->m_numberOfSequences =  m_frameMode ? m_chunks[i]->m_numberOfSamples : m_chunks[i]->m_numberOfSequences;
        cd->m_numberOfSamples = m_chunks[i]->m_numberOfSamples;
        chunks.push_back(cd);
    }
    return chunks;
}

// Gets sequences for a particular chunk.
void MLFDeserializer::GetSequencesForChunk(ChunkIdType, vector<SequenceDescription>& result)
{
    UNUSED(result);
    LogicError("Mlf deserializer does not support primary mode - it cannot control chunking.");
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

bool MLFDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
{
    if (key.m_sequence >= m_keyToSequence.size())
        return false;

    auto chunkAndSequence =  m_keyToSequence[key.m_sequence];
    if (!chunkAndSequence.first)
        return false;

    auto c = std::lower_bound(
        m_chunks.begin(),
        m_chunks.end(),
        chunkAndSequence.first);

    if (c == m_chunks.end())
        RuntimeError("Unexpected chunk specified.");

    auto sequence = chunkAndSequence.second;

    result.m_chunkId = static_cast<ChunkIdType>(c - m_chunks.begin());
    result.m_key = key;

    if (m_frameMode)
    {
        result.m_indexInChunk = chunkAndSequence.first->m_firstSamples[sequence->m_indexInChunk] + key.m_sample;
        result.m_numberOfSamples = 1;
    }
    else
    {
        assert(result.m_key.m_sample == 0);
        result.m_indexInChunk = sequence->m_indexInChunk;
        result.m_numberOfSamples = sequence->m_numberOfSamples;
    }
    return true;
}

}}}
