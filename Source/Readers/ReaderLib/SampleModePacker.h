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
    std::shared_ptr<char> m_data; // contiguous array of data.
    size_t m_capacity; // capacity of the buffer in bytes.
    size_t m_size; // the actual size of the data in buffer in bytes (m_size <= m_capacity)
    char* m_index; // points to the beginning of the unused portion of the buffer
    StreamBuffer() : m_capacity(0), m_size(0), m_index(nullptr) { }
    void Allocate(MemoryProviderPtr memoryProvider, size_t capacity);
    void Copy(const char*, size_t size);
    void Fill(size_t size, char value);
    void Reset(); // moves m_index to the beginnig of the buffer, and sets the m_size to 0.
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

    // Returns the length in samples of the longest sequence of the specified stream.
    size_t GetMaxSequenceLength(const StreamBatch& batch, size_t streamIndex);

    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;
    size_t m_numberOfStreams;

    std::vector<StreamDescriptionPtr> m_outputStreams;
    std::vector<StreamDescriptionPtr> m_inputStreams;

    std::vector<StreamBuffer> m_streamBuffers;

    size_t m_minibatchSize;
};

typedef std::shared_ptr<SampleModePacker> SampleModePackerPtr;
} } }
