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

// A base class for Packers.
class PackerBase : public Packer
{
protected:
    PackerBase(MemoryProviderPtr memoryProvider,
               TransformerPtr transformer,
               size_t minibatchSize,
               const std::vector<StreamDescriptionPtr>& streams);

protected:
    std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);
    size_t GetSampleSize(StreamDescriptionPtr stream);
    void PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);
    void PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);

    // Memory provider.
    // TODO: Should possibly switch to matrices here.
    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;

    // Input stream descriptions provided by the transformer.
    std::vector<StreamDescriptionPtr> m_outputStreamDescriptions;

    // Output stream descriptions expected by the network.
    std::vector<StreamDescriptionPtr> m_inputStreamDescriptions;

    // Minibatch size in samples.
    size_t m_minibatchSize;
};

}}}
