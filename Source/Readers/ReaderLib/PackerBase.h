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

    // Packs a sparse sample as dense:
    //  - 0-fills a region of sampleSize bytes in the block of memory pointed to by destination;
    //  - copies non-zero values of the required sample (given by sampleIndex) from the data
    // portion of the source sequence to the destination block of memory, where each value is placed
    // at the offset equal to value index * elementSize. sampleOffset specifies the offset of the
    // first value from the given sample in the sequence data/indices array (sampleOffset is equal
    // to the sum of non-zero value counts of all preceding samples).
    void PackSparseSampleAsDense(char* destination, SparseSequenceDataPtr sequence,
        size_t sampleIndex, size_t sampleOffset, size_t sampleSize, size_t elementSize);

    // Packs a dense sample as dense. Copies sampleSize bytes staring at the sampleOffset from 
    // the data portion of the source sequence to the destination block of memory. sampleOffset 
    // specifies the offset of the first value from the given sample in the sequence data/ array 
    // (sampleOffset is equal to the sum of sample sizes of all preceding samples).
    void PackDenseSample(char* destination, SequenceDataPtr sequence, size_t sampleOffset, size_t sampleSize);

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
    // Get the nnz count of the sample.
    size_t nonZeroCount = sequence->m_nnzCounts[sampleIndex];
    // In a sparse sequence, m_data points to the array of non zero elements,
    // m_indices stores the corresponding indices for each element. 
    // Iterate through non zero elements and copy from m_data them into the 
    // destination at the offset given by the corresponding row index (m_index).
    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
    {
        auto sourceOffset = sampleOffset + nonZeroIndex;
        auto elementIndex = sequence->m_indices[sourceOffset];
        auto destinationOffset = elementIndex * elementSize;
        assert(destinationOffset < sampleSize);
        const auto* source = (const char*)(sequence->m_data) + (sourceOffset)* elementSize;
        memcpy(destination + destinationOffset, source, elementSize);
    }
}

inline void PackerBase::PackDenseSample(char* destination, SequenceDataPtr sequence, size_t sampleOffset, size_t sampleSize)
{
    // Because the sample is dense - simply copying it to the output.
    memcpy(destination, (const char*)(sequence->m_data) + sampleOffset, sampleSize);
}

}}}
