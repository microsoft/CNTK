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

// A sequence packer that packs dense or sparse samples in dense minibatch for parallel GPU consumption.
class SequencePacker : public PackerBase
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

    // Buffers for allocated data.
    std::vector<std::shared_ptr<char>> m_streamBuffers;
    // Size of allocated buffers, m_streamBuffers.size() == m_streamBufferSizes.size().
    std::vector<size_t> m_streamBufferSizes;
};

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}}}
