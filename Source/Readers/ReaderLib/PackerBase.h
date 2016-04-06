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
public:

    virtual Minibatch ReadMinibatch() override;

protected:

    struct StreamBuffer
    {
        size_t m_size; // buffer size in bytes.
        // Memory provider.
        // TODO: Should possibly switch to matrices here.
        MemoryProviderPtr m_memoryProvider;
        std::shared_ptr<char> m_data; // contiguous array of data.

        StreamBuffer(MemoryProviderPtr m_memoryProvider) :
            m_size(0), m_memoryProvider(m_memoryProvider), m_data(nullptr)
        {
        }
        void Resize(size_t newSize);
    };

    PackerBase(MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        const std::vector<StreamDescriptionPtr>& streams);

    typedef std::vector<SequenceDataPtr> StreamBatch;

    size_t GetSampleSize(StreamDescriptionPtr stream);

    virtual MBLayoutPtr PackDenseStream(const StreamBatch& batch, size_t streamIndex);

    virtual MBLayoutPtr PackSparseStream(const StreamBatch& batch, size_t streamIndex);

    // Packs a sparse sample as dense.
    void PackSparseSampleAsDense(char* destination, SparseSequenceDataPtr sequence,
        size_t sampleIndex, size_t sampleOffset, size_t sampleSize, size_t elementSize);

    // Packs a dense sample as dense.
    void PackDenseSample(char* destination, SequenceDataPtr sequence, size_t sampleOffset, size_t sampleSize);

    virtual MBLayoutPtr CreateMBLayout(const StreamBatch& batch) = 0;
    
    TransformerPtr m_transformer;

    // Input stream descriptions provided by the transformer.
    std::vector<StreamDescriptionPtr> m_outputStreamDescriptions;

    // Output stream descriptions expected by the network.
    std::vector<StreamDescriptionPtr> m_inputStreamDescriptions;

    // Buffers for allocated data.
    std::vector<StreamBuffer> m_streamBuffers;

    // Minibatch size in samples.
    size_t m_minibatchSize;
};

inline void PackerBase::PackSparseSampleAsDense(char* destination, SparseSequenceDataPtr sequence,
    size_t sampleIndex, size_t sampleOffset, size_t sampleSize, size_t elementSize)
{
    //The sample is sparse, first, need to zero out the buffer.
    memset(destination, 0, sampleSize);
    size_t nonZeroCount = sequence->m_nnzCounts[sampleIndex];
    // Iterate through non zero elements and copy them to the corresponding place using their index.
    // In a sparse sequence, m_data points to the array of non zero elements,
    // m_indices stores the non-zero row indexes for each element.
    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
    {

        auto rowIndex = sequence->m_indices[sampleOffset + nonZeroIndex];
        size_t elementOffset = rowIndex * elementSize;
        assert(elementOffset < sampleSize);
        const auto* source = (const char*)(sequence->m_data) + (sampleOffset + nonZeroIndex) * elementSize;
        memcpy(destination + elementOffset, source, elementSize);
    }
}

inline void PackerBase::PackDenseSample(char* destination, SequenceDataPtr sequence, size_t sampleOffset, size_t sampleSize)
{
    // Because the sample is dense - simply copying it to the output.
    memcpy(destination, (const char*)(sequence->m_data) + sampleOffset, sampleSize);
}

}}}
