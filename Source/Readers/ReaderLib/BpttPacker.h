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
    // SequenceId specifies the starting value to be used as sequence identifier.
    // For each new input, sequence id is reset to 0, and incremented each time
    // a sequence is added to the layout. This allows layouts corresponding to different
    // inputs to have consistent sequence ids.
    void PackSlot(size_t streamIndex, size_t slotIndex, size_t& sequenceId);

    virtual MBLayoutPtr CreateMBLayout(const StreamBatch& batch)
    {
        UNUSED(batch);
        NOT_IMPLEMENTED;
    }

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
};

typedef std::shared_ptr<BpttPacker> BpttPackerPtr;

}}}
