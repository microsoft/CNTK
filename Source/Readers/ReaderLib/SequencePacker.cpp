//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "SequencePacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

SequencePacker::SequencePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams) : PackerBase(memoryProvider, transformer, minibatchSize, streams)
{
    for (int i = 0; i < m_outputStreamDescriptions.size(); ++i)
    {
        m_streamBufferSizes.push_back(0);
        m_streamBuffers.push_back(nullptr);
    }
}

// Reading minibatch.
Minibatch SequencePacker::ReadMinibatch()
{
    assert(m_streamBufferSizes.size() == m_streamBuffers.size());
    const auto sequences = m_transformer->GetNextSequences(m_minibatchSize);

    Minibatch minibatch(sequences.m_endOfEpoch);
    if (sequences.m_data.empty())
    {
        return minibatch;
    }

    // For each stream packing the minibatch.
    minibatch.m_data.reserve(sequences.m_data.size());
    for (size_t streamIndex = 0; streamIndex < sequences.m_data.size(); ++streamIndex)
    {
        minibatch.m_data.push_back(PackStreamMinibatch(sequences.m_data[streamIndex], streamIndex));
    }

    return minibatch;
}

StreamMinibatchPtr SequencePacker::PackStreamMinibatch(const std::vector<SequenceDataPtr>& sequences, size_t streamId)
{
    // Create sequence info for each sequences that we have got from the transformer.

    std::vector<MBLayout::SequenceInfo> inputSequences;
    for (size_t index = 0; index < sequences.size(); ++index)
    {
        MBLayout::SequenceInfo info;

        // In each minibatch sequence ids should be unique.
        // They have to match between different input streams in the same minibatch.
        // We are using sequence index in the set of received sequences.
        // TODO: should we use m_key as sequence id and pass it with data?
        info.seqId = index;

        info.tBegin = 0;
        info.tEnd = sequences[index]->m_numberOfSamples;
        inputSequences.push_back(info);
    }

    std::vector<std::pair<size_t, size_t>> placement;
    std::vector<size_t> rowAllocations;

    // Creating the minibatch layout.
    MBLayoutPtr layout = std::make_shared<MBLayout>();
    layout->InitAsPackedSequences(inputSequences, placement, rowAllocations);

    // Allocating necessary data buffer for the stream.
    size_t sampleSize = GetSampleSize(m_inputStreamDescriptions[streamId]);
    size_t totalNumberOfSamplesInBytes = layout->GetNumCols() * sampleSize;
    if (m_streamBufferSizes[streamId] < totalNumberOfSamplesInBytes)
    {
        m_streamBuffers[streamId] = AllocateBuffer(layout->GetNumCols(), sampleSize);
        m_streamBufferSizes[streamId] = totalNumberOfSamplesInBytes;
    }

    // Packing the actual data.
    StorageType storageType = m_inputStreamDescriptions[streamId]->m_storageType;
    size_t elementSize = GetSizeByType(m_inputStreamDescriptions[streamId]->m_elementType);
    const auto& packedSequences = layout->GetAllSequences();
    char* streamBuffer = m_streamBuffers[streamId].get();
    for (const auto& sequence : packedSequences)
    {
        if (sequence.seqId == GAP_SEQUENCE_ID)
            continue;
        const auto& data = sequences[sequence.seqId];

        // Packing the sequence
        for (size_t sampleIndex = 0; sampleIndex < sequence.GetNumTimeSteps(); ++sampleIndex)
        {
            char* destination = streamBuffer + layout->GetColumnIndex(sequence, sampleIndex) * sampleSize;
            if (storageType == StorageType::dense)
            {
                PackDenseSample(destination, data, sampleIndex, elementSize, sampleSize);
            }
            else // sparse
            {
                assert(storageType == StorageType::sparse_csc);
                PackSparseSample(destination, data, sampleIndex, elementSize, sampleSize);
            }
        }
    }

    // Ok, minibatch is ready, give it out.
    StreamMinibatchPtr result = std::make_shared<StreamMinibatch>();
    result->m_data = m_streamBuffers[streamId].get();
    result->m_layout = layout;
    return result;
}

}}}
