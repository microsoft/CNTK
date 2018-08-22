//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "HTKDeserializer.h"
#include "CorpusDescriptor.h"
#include "MLFUtils.h"
#include "FileWrapper.h"
#include "Index.h"

namespace CNTK
{

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

    MLFSequenceData(size_t numberOfSamples, const NDShape& frameShape)
        : m_values(numberOfSamples, 1), m_frameShape(frameShape)
    {
        if (numberOfSamples > numeric_limits<IndexType>::max())
        {
            RuntimeError("Number of samples in an MLFSequenceData (%zu) "
                         "exceeds the maximum allowed value (%zu)\n",
                         numberOfSamples, (size_t) numeric_limits<IndexType>::max());
        }

        m_indexBuffer.resize(numberOfSamples);
        m_nnzCounts.resize(numberOfSamples, static_cast<IndexType>(1));
        m_numberOfSamples = (uint32_t) numberOfSamples;
        m_totalNnzCount = static_cast<IndexType>(numberOfSamples);
        m_indices = &m_indexBuffer[0];
    }

    MLFSequenceData(size_t numberOfSamples, const vector<size_t>& phoneBoundaries, const NDShape& frameShape)
        : MLFSequenceData(numberOfSamples, frameShape)
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

// Class represents an MLF deserializer.
// Provides a set of chunks/sequences to the upper layers.
class MLFDeserializer : public DataDeserializerBase, boost::noncopyable
{
public:
    // Expects new configuration.
    MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

    // TODO: Should be removed, when all readers go away, expects configuration in a legacy mode.
    MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

    MLFDeserializer(CorpusDescriptorPtr corpus, bool primary);

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    bool GetSequenceInfoByKey(const SequenceKey& key, SequenceInfo& s) override;

    // Gets description of all chunks.
    virtual std::vector<ChunkInfo> ChunkInfos() override;

    // Get sequence descriptions of a particular chunk.
    virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& s) override;

    // Retrieves a chunk with data.
    virtual ChunkPtr GetChunk(ChunkIdType) override;

    static inline bool LessByFirstItem(const std::tuple<size_t, size_t, size_t>& a, const std::tuple<size_t, size_t, size_t>& b)
    {
        return std::get<0>(a) < std::get<0>(b);
    }

    // Base class for chunks in frame and sequence mode.
    // The lifetime is always less than the lifetime of the parent deserializer.
    class ChunkBase : public Chunk
    {
    public:
        vector<vector<MLFFrameRange>> m_sequences; // Each sequence is a vector of sequential frame ranges.

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

        template <class ElementType>
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
                    RuntimeError("Class id '%ud' exceeds the model output dimension '%d'.", range.ClassId(), (int) m_deserializer.m_dimension);

                // Filling all range of frames with the corresponding class id.
                fill(startRange, startRange + range.NumFrames(), static_cast<IndexType>(range.ClassId()));
                startRange += range.NumFrames();
            }

            result.push_back(s);
        }

        vector<char> m_buffer; // Buffer for the whole chunk
        vector<bool> m_valid;  // Bit mask whether the parsed sequence is valid.
        MLFUtteranceParser m_parser;

        const MLFDeserializer& m_deserializer;
        const ChunkDescriptor& m_descriptor; // Current chunk descriptor.
    };

    // MLF chunk when operating in sequence mode.
    class SequenceChunk : public ChunkBase
    {
    public:
        SequenceChunk(const MLFDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName, StateTablePtr states)
            : ChunkBase(parent, descriptor, fileName, states)
        {
            this->m_sequences.resize(m_descriptor.Sequences().size());

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
    };

    // MLF chunk when operating in frame mode.
    // Implementation is different because frames of the same sequence can be accessed
    // in parallel by the randomizer, so all parsing/preprocessing should be done during
    // sequence caching, so that GetSequence only works with read only data structures.
    class FrameChunk : public ChunkBase
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
            for (size_t i = 0; i < utterance.size(); ++i)
            {
                const auto& range = utterance[i];
                if (range.ClassId() >= m_deserializer.m_dimension)
                    // TODO: Possibly set m_valid to false, but currently preserving the old behavior.
                    RuntimeError("Class id '%ud' exceeds the model output dimension '%d'.", range.ClassId(), (int) m_deserializer.m_dimension);

                fill(startRange, startRange + range.NumFrames(), range.ClassId());
                startRange += range.NumFrames();
            }
        }
    };

    // Initializes reader params.
    std::wstring InitializeReaderParams(const ConfigParameters& cfg, bool primary);

    // Initializes chunk descriptions.
    void InitializeChunkInfos(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath);

    // Initializes a single stream this deserializer exposes.
    void InitializeStream(const std::wstring& name);

    // In frame mode initializes data for all categories/labels in order to
    // avoid memory copy.
    void InitializeReadOnlyArrayOfLabels();

    // Sorted vector that maps SequenceKey.m_sequence into an utterance ID (or type max() if the key is not assigned).
    std::vector<std::tuple<size_t, ChunkIdType, uint32_t>> m_keyToChunkLocation;

    // Type of the data this serializer provides.
    DataType m_elementType;

    // Array of available categories.
    // We do no allocate data for all input sequences, only returning a pointer to existing category.
    std::vector<SparseSequenceDataPtr> m_categories;

    // A list of category indices
    // (a list of numbers from 0 to N, where N = (number of categories -1))
    std::vector<IndexType> m_categoryIndices;

    // Flag that indicates whether a single speech frames should be exposed as a sequence.
    bool m_frameMode;

    CorpusDescriptorPtr m_corpus;

    std::vector<const ChunkDescriptor*> m_chunks;
    std::map<const ChunkDescriptor*, size_t> m_chunkToFileIndex;

    size_t m_dimension;
    size_t m_chunkSizeBytes;

    // Track phone boundaries
    bool m_withPhoneBoundaries;

    StateTablePtr m_stateTable;

    std::vector<std::shared_ptr<Index>> m_indices;
    std::vector<std::wstring> m_mlfFiles;
    bool m_textReader;
};
}
