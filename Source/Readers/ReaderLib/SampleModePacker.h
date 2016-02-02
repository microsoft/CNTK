//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A sample packer that densely packs samples in parallel for GPU consumptions.
class SampleModePacker
{
public:
    SampleModePacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        const std::vector<StreamDescriptionPtr>& streams);

    Minibatch ReadMinibatch();

private:
    std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);
    size_t GetSampleSize(StreamDescriptionPtr stream);
    void CopySequenceToBuffer(size_t sequenceIndex, size_t streamIndex, const std::vector<std::vector<SequenceDataPtr>>& sequences);

    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;
    std::vector<StreamDescriptionPtr> m_outputStreams;
    std::vector<StreamDescriptionPtr> m_inputStreams;
    std::vector<std::shared_ptr<char>> m_streamBuffers;

    MBLayoutPtr m_minibatchLayout;
    size_t m_minibatchSize;
};

typedef std::shared_ptr<SampleModePacker> SampleModePackerPtr;
} } }
