//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"
#include "Packer.h"
#include <deque>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a buffer of sequences aligned to slots.
struct SequenceBuffer;
typedef std::shared_ptr<SequenceBuffer> SequenceBufferPtr;

// A bptt packer that densely packs samples in parallel for GPU consumptions.
// TODO: Currently supports only packing of streams with sequences of equal length.
class BpttPacker : public Packer
{
public:
    BpttPacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        size_t truncationSize,
        const std::vector<StreamDescriptionPtr>& streams);

    virtual Minibatch ReadMinibatch() override;

private:
    void InitializePreparedSequences();
    bool NothingToPack();
    void PackSlot(size_t streamIndex, size_t slotIndex);
    void GetSequencesToSlot(size_t slotIndex);

    std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);
    size_t GetSampleSize(StreamDescriptionPtr stream);
    void PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);
    void PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);

    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;
    std::vector<StreamDescriptionPtr> m_outputStreams;
    std::vector<StreamDescriptionPtr> m_inputStreams;

    size_t m_minibatchSize;
    size_t m_parallelNumberOfSequences;
    size_t m_truncationSize;

    std::vector<SequenceBufferPtr> m_sequenceBufferPerStream;

    // Layout per stream.
    // TODO: currently assume that layout is the same between different streams, this will change.
    std::vector<MBLayoutPtr> m_currentLayouts;

    // Buffers for allocated data.
    std::vector<std::shared_ptr<char>> m_streamBuffers;
    // Size of allocated buffers, m_streamBuffers.size() == m_streamBufferSizes.size().
    std::vector<size_t> m_streamBufferSizes;
};

typedef std::shared_ptr<BpttPacker> BpttPackerPtr;

}}}
