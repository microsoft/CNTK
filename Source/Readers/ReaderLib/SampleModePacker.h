//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct StreamBuffer
{
    size_t m_size; // buffer size in bytes.
    MemoryProviderPtr m_memoryProvider;
    std::shared_ptr<char> m_data; // contiguous array of data.
    
    StreamBuffer(MemoryProviderPtr m_memoryProvider) :
        m_size(0), m_memoryProvider(m_memoryProvider), m_data(nullptr)
    {
    }
    void Resize(size_t newSize);
};

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
    typedef std::vector<SequenceDataPtr> StreamBatch; 

    size_t GetSampleSize(StreamDescriptionPtr stream);

    MBLayoutPtr PackDenseStream(const StreamBatch& batch, size_t streamIndex);
    MBLayoutPtr PackSparseStream(const StreamBatch& batch, size_t streamIndex);

    MBLayoutPtr CreateMBLayout(const StreamBatch& batch);

    // Returns the length in samples of the longest sequence in the batch.
    size_t GetMaxSequenceLength(const StreamBatch& batch);

    TransformerPtr m_transformer;
    size_t m_numberOfStreams;

    std::vector<StreamDescriptionPtr> m_outputStreams;
    std::vector<StreamDescriptionPtr> m_inputStreams;

    std::vector<StreamBuffer> m_streamBuffers;

    size_t m_minibatchSize;
};

typedef std::shared_ptr<SampleModePacker> SampleModePackerPtr;
} } }
