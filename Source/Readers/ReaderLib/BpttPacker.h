//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"
#include "PackerBase.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a buffer of prepared sequences from which the minibatch is created.
struct SequenceBuffer;
typedef std::shared_ptr<SequenceBuffer> SequenceBufferPtr;

// A bptt packer that densely packs samples in parallel for GPU consumptions.
// TODO: Currently supports only packing of streams with sequences of equal length.
class BpttPacker : public PackerBase
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
    // Reads sequences to slot with the specified index.
    // Number of slots = m_parallelNumberOfSequences
    void ReadSequencesToSlot(size_t slotIndex);

    // Packs a slot into the data buffer.
    void PackSlot(size_t streamIndex, size_t slotIndex);

    // Parallel number of sequences to pack.
    // Calculated based on the truncation size and minibatch size in samples.
    size_t m_numParallelSequences;

    // Truncation size in samples.
    size_t m_truncationSize;

    // Sequence buffer per stream.
    // Each sequence buffer contains m_parallelNumberOfSequences slots
    // that get filled with sequences.
    std::vector<SequenceBufferPtr> m_sequenceBufferPerStream;

    // Layout per stream.
    // TODO: currently assume that layout is the same between different streams, this will change.
    std::vector<MBLayoutPtr> m_currentLayouts;

    // Buffers for allocated data.
    std::vector<std::shared_ptr<char>> m_streamBuffers;

    // Size of allocated buffers, m_streamBuffers.size() == m_streamBufferSizes.size().
    // TODO: this will disappear after Alexey's merge (StreamBuffer class).
    std::vector<size_t> m_streamBufferSizes;
};

typedef std::shared_ptr<BpttPacker> BpttPackerPtr;

}}}
