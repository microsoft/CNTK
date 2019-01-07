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
class MLFBinaryDeserializer::BinarySequenceChunk : public MLFDeserializer::ChunkBase
{
public:
    BinarySequenceChunk(const MLFBinaryDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName, StateTablePtr states)
        : ChunkBase::ChunkBase(parent, descriptor, fileName, states)
    {
        this->m_sequences.resize(m_descriptor.Sequences().size());

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < descriptor.Sequences().size(); ++i)
            CacheSequence(descriptor.Sequences()[i], i);

        CleanBuffer();
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

class MLFBinaryDeserializer::BinaryFrameChunk : public MLFDeserializer::ChunkBase
{
    // Actual values of frames.
    vector<ClassIdType> m_classIds;

    //For each sequence this vector contains the sequence offset in samples from the beginning of the chunk.
    std::vector<uint32_t> m_sequenceOffsetInChunkInSamples;

public:
    BinaryFrameChunk(const MLFBinaryDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName, StateTablePtr states)
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
            SparseSequenceDataPtr s = make_shared<MLFSequenceData<float>>(0, m_deserializer.GetStreamInfos()->front().m_sampleLayout);
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

        auto startRange = m_classIds.begin() + m_sequenceOffsetInChunkInSamples[index];
        for (size_t i = 0; i < utterance.size(); ++i)
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

MLFBinaryDeserializer::MLFBinaryDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : MLFDeserializer(corpus, primary)
{
    auto inputName = InitializeReaderParams(cfg, primary);
    m_textReader = false;

    ConfigParameters input = cfg("input");
    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    if (m_withPhoneBoundaries)
        LogicError("TODO: implement phoneBoundaries setting in Binary MLF deserializer.");

    InitializeStream(inputName);
    InitializeChunkInfos(corpus, config, L"");
}

ChunkPtr MLFBinaryDeserializer::GetChunk(ChunkIdType chunkId)
{
    ChunkPtr result;
    attempt(5, [this, &result, chunkId]()
    {
        auto chunk = m_chunks[chunkId];
        auto& fileName = m_mlfFiles[m_chunkToFileIndex[chunk]];

        if (m_frameMode)
            result = make_shared<BinaryFrameChunk>(*this, *chunk, fileName, m_stateTable);
        else
            result = make_shared<BinarySequenceChunk>(*this, *chunk, fileName, m_stateTable);
    });

    return result;
};

}
