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

MLFBinaryDeserializer::MLFBinaryDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : MLFDeserializer(corpus, primary)
{
    auto inputName = InitializeReaderParams(cfg, primary);
    m_textReader = false;
    if (m_frameMode)
        LogicError("TODO: support frame mode in Binary MLF deserializer.");

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

        result = make_shared<BinarySequenceChunk>(*this, *chunk, fileName, m_stateTable);
    });

    return result;
};

}
