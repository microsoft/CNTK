//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"
#include "Packer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A sequence packer that packs dense or sparse samples in dense minibatch for parallel GPU consumption.
class SequencePacker : public Packer
{
public:
    SequencePacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        const std::vector<StreamDescriptionPtr>& streams);

    virtual Minibatch ReadMinibatch() override;

private:
    // Auxiliary packing functions.
    // Packs sequences from a particular stream into a minibatch.
    StreamMinibatchPtr PackStreamMinibatch(const std::vector<SequenceDataPtr>& sequences, size_t streamId);

    // Packs sparse sample as dense into the destination.
    void PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);

    // Packs dense sample into the destination.
    void PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);

    // Utility functions.
    // Allocates the buffer.
    // TODO: Should use pinned memory.
    std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);

    // Gets a sample size in bytes for a stream.
    size_t GetSampleSize(StreamDescriptionPtr stream);

    // Memory provider.
    // TODO: Should possibly switch to matrices here.
    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;

    // Input streams provided by the transformer.
    std::vector<StreamDescriptionPtr> m_outputStreams;
    // Output streams expected by the network.
    std::vector<StreamDescriptionPtr> m_inputStreams;

    // Minibatch size in samples.
    size_t m_minibatchSize;

    // Buffers for allocated data.
    std::vector<std::shared_ptr<char>> m_streamBuffers;
    // Size of allocated buffers, m_streamBuffers.size() == m_streamBufferSizes.size().
    std::vector<size_t> m_streamBufferSizes;
};

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}}}
