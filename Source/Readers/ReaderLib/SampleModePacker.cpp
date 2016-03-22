//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <numeric>
#include <limits>
#include <inttypes.h>
#include "SampleModePacker.h"
#include "ElementTypeUtils.h"
#include "CommonMatrix.h"

typedef CPUSPARSE_INDEX_TYPE IndexType;

namespace Microsoft { namespace MSR { namespace CNTK {

void StreamBuffer::Allocate(MemoryProviderPtr memoryProvider, size_t capacity)
{
    m_capacity = capacity;
    m_data.reset(
        reinterpret_cast<char*>(memoryProvider->Alloc(1, capacity)),
        [memoryProvider](char* p)
    {
            memoryProvider->Free(p);
    });
    Reset();
}


void StreamBuffer::Reset()
{
    // This is only really needed in the sparse to dense case.
    // In all other case we fill up the whole buffer (up to the
    // required size = matrix size).
    auto ptr = m_data.get();
    std::fill(ptr, ptr + m_capacity, 0);
}


SampleModePacker::SampleModePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams) : m_transformer(transformer),
                                                        m_minibatchSize(minibatchSize),
                                                        m_numberOfStreams(streams.size()),
                                                        m_outputStreams(streams),
                                                        m_memoryProvider(memoryProvider)
{
    m_inputStreams = m_transformer->GetStreamDescriptions();
    assert(m_inputStreams.size() == m_numberOfStreams);
    assert(m_minibatchSize > 0);

    m_streamBuffers.resize(m_numberOfStreams);
    for (int i = 0; i < m_numberOfStreams; ++i)
    {
        const auto& stream = m_outputStreams[i];
        // Input and output should match in everything except for sparse/dense.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreams[i]->m_name);
        assert(stream->m_id == m_inputStreams[i]->m_id);
        assert(GetSampleSize(m_inputStreams[i]) == GetSampleSize(stream));


        if (m_inputStreams[i]->m_storageType == StorageType::dense &&
            stream->m_storageType == StorageType::sparse_csc) 
        {
            RuntimeError("Dense to sparse re-packing requested for stream '%ls' is not supported.", 
                stream->m_name.c_str());
        }
    }
}

Minibatch SampleModePacker::ReadMinibatch()
{
    auto sequences = m_transformer->GetNextSequences(m_minibatchSize);
    const auto& batch = sequences.m_data;

    Minibatch minibatch(sequences.m_endOfEpoch);
    if (batch.empty())
    {
        return minibatch;
    }

    assert(m_numberOfStreams == batch.size());

    for (int streamIndex = 0; streamIndex < m_numberOfStreams; ++streamIndex)
    {
        const auto& streamBatch = batch[streamIndex];
        const auto& type = m_outputStreams[streamIndex]->m_storageType;
        auto layout = (type == StorageType::dense) ?
            PackDenseStream(streamBatch, streamIndex) : PackSparseStream(streamBatch, streamIndex);

        auto& buffer = m_streamBuffers[streamIndex];

        auto streamMinibatch = std::make_shared<StreamMinibatch>();
        streamMinibatch->m_data = buffer.m_data.get();
        // TODO: m_dataSize not really used and can be removed (?)
        // streamMinibatch->m_dataSize = buffer.m_size;
        streamMinibatch->m_layout = layout;

        minibatch.m_data.push_back(streamMinibatch);
    }

    return minibatch;
}

size_t SampleModePacker::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}


size_t SampleModePacker::GetMaxSequenceLength(const StreamBatch& batch)
{
    size_t maxLength = 0;
    for (const auto& sequence : batch)
    {
        maxLength = max(maxLength, sequence->m_numberOfSamples);
    }
    return maxLength;
}


MBLayoutPtr SampleModePacker::PackDenseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreams[streamIndex]->m_storageType == StorageType::dense);

    const auto& stream = m_inputStreams[streamIndex];
    auto& buffer = m_streamBuffers[streamIndex];
    size_t sampleSize = GetSampleSize(stream);
    auto elementSize = GetSizeByType(stream->m_elementType);
    size_t maxSequenceLength = GetMaxSequenceLength(batch);
    size_t numSequences = batch.size();
    size_t requiredSize = sampleSize * maxSequenceLength * numSequences;

    if (buffer.m_capacity < requiredSize)
    {
        buffer.Allocate(m_memoryProvider, requiredSize);
    }
    else
    {
        buffer.Reset();
    }

    auto mbLayout = std::make_shared<MBLayout>();
    mbLayout->Init(numSequences, maxSequenceLength);

    for (size_t sequenceIndex = 0; sequenceIndex < numSequences; ++sequenceIndex)
    {
        const auto& sequence = batch[sequenceIndex];
        char* source = reinterpret_cast<char*>(sequence->m_data);
        size_t numSamples = sequence->m_numberOfSamples;

        if (stream->m_storageType == StorageType::dense)
        {
            char* destination = buffer.m_data.get() + sequenceIndex * sampleSize;
            for (size_t sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
            {
                assert(destination <= buffer.m_data.get() + buffer.m_capacity - sampleSize);
                std::copy(source, source + sampleSize, destination);
                source += sampleSize;
                destination += numSequences * sampleSize;                
            }           
        }
        else if (stream->m_storageType == StorageType::sparse_csc)
        {
            const auto& sparseSequence = reinterpret_cast<SparseSequenceData&>(*sequence);
            char* destination = buffer.m_data.get() + sequenceIndex * sampleSize;

            // Copy the non zero data to the buffer.
            for (size_t sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
            {
                size_t nonZeroCount = sparseSequence.m_nnzCounts[sampleIndex];
                for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
                {
                    auto rowIndex = sparseSequence.m_indices[nonZeroIndex];
                    size_t offset = rowIndex * elementSize;
                    assert(offset < sampleSize);
                    char* from = source + nonZeroIndex * elementSize;
                    char* to = destination + offset;
                    std::copy(from, from + elementSize, to);
                }
                source += sampleSize;
                destination += numSequences * sampleSize;
            }
        }
        else 
        {
            RuntimeError("Storage type %d is not supported.", (int)stream->m_storageType);
        }

        // We don't do any packing per se, instead we create MB with 
        // the the number of parallel sequences equal to the number of 
        // Sequences in the batch.
        mbLayout->AddSequence(sequence->m_id, sequenceIndex, 0, numSamples);
        if (numSamples < maxSequenceLength)
        {
            mbLayout->AddGap(sequenceIndex, numSamples, maxSequenceLength);
        }
    }
    return mbLayout;
}

MBLayoutPtr SampleModePacker::PackSparseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreams[streamIndex]->m_storageType == StorageType::sparse_csc);

    const auto& stream = m_inputStreams[streamIndex];

    assert(stream->m_storageType == StorageType::sparse_csc);

    auto& buffer = m_streamBuffers[streamIndex];
    
    auto elementSize = GetSizeByType(stream->m_elementType);

    auto mbLayout = std::make_shared<MBLayout>();

    size_t maxSequenceLength = GetMaxSequenceLength(batch);

    size_t numSequences = batch.size();

    mbLayout->Init(numSequences, maxSequenceLength);

    size_t nnzCount = 0;

    for (size_t sequenceIndex = 0; sequenceIndex < numSequences; ++sequenceIndex)
    {
        const auto& sequence = batch[sequenceIndex];
        const auto& sparseSequence = reinterpret_cast<const SparseSequenceData&>(*sequence);
        nnzCount += sparseSequence.m_totalNnzCount;
        
        // We don't do any packing per se (yet), instead we create MB with 
        // the number of parallel sequences equal to the number of 
        // Sequences in the batch.
        size_t numSamples = sequence->m_numberOfSamples;
        mbLayout->AddSequence(sequence->m_id, sequenceIndex, 0, numSamples);
        if (numSamples < maxSequenceLength)
        {
            mbLayout->AddGap(sequenceIndex, numSamples, maxSequenceLength);
        }
    }

    if (nnzCount > numeric_limits<IndexType>::max())
    {
        RuntimeError("Minibatch NNZ count (" PRIu64 ") exceeds the maximum allowed "
            "value (" PRIu64 ")\n", nnzCount, (size_t)numeric_limits<IndexType>::max());
    }


    size_t numColumns = numSequences * maxSequenceLength;

    // size of nnz type + nnz * (size of the element type) + nnz * (size of the row index type) + 
    // (number of columns + 1) * (size of the column index type). 
    size_t requiredSize =
        sizeof(nnzCount) +
        nnzCount * (elementSize + sizeof(IndexType)) +
        sizeof(IndexType) * (numColumns + 1);
    
    if (buffer.m_capacity < requiredSize)
    {
        buffer.Allocate(m_memoryProvider, requiredSize);
    }
    else
    {
        buffer.Reset();
    }

    const char* source = reinterpret_cast<char*>(&nnzCount);
    char* destination = buffer.m_data.get();
    // insert the nnzCount as the first element in the buffer.
    std::copy(source, source + sizeof(nnzCount), destination);

    destination += sizeof(nnzCount);
    char* dataDst = destination;
    IndexType* indicesDst = reinterpret_cast<IndexType*>(dataDst + elementSize* nnzCount);
    IndexType columnOffset = 0;
    vector<IndexType> sparseColumnIndices;
    vector<IndexType>  sequenceOffsets(numSequences, 0);

    for (int sampleIndex = 0; sampleIndex < maxSequenceLength; ++sampleIndex)
    {
        for (size_t sequenceIndex = 0; sequenceIndex < numSequences; ++sequenceIndex)
        {
            sparseColumnIndices.push_back(columnOffset);

            const auto& sequence = batch[sequenceIndex];
            if (sampleIndex >= sequence->m_numberOfSamples)
            {
                continue;
            }

            auto& sequenceOffset = sequenceOffsets[sequenceIndex];
            const auto& sparseSequence = reinterpret_cast<SparseSequenceData&>(*sequence);
            IndexType nnz = sparseSequence.m_nnzCounts[sampleIndex];

            size_t sampleOffset = sequenceOffset * elementSize;
            auto data = reinterpret_cast<const char*>(sequence->m_data) + sampleOffset;
            std::copy(data, data + nnz * elementSize, dataDst);
            dataDst += nnz * elementSize;

            auto indices = sparseSequence.m_indices + sequenceOffset;
            std::copy(indices, indices + nnz, indicesDst);
            indicesDst += nnz;

            sequenceOffset += nnz;
            columnOffset += nnz;
        }
    }

    // at this point each element in sequenceOffsets should be equal to the total
    // nnz count of the respective sequence and the sum of all elements - to the 
    // overall nnz count.
    assert(accumulate(sequenceOffsets.begin(), sequenceOffsets.end(), 0) == nnzCount);

    assert(columnOffset == nnzCount);
    sparseColumnIndices.push_back(columnOffset);
    assert((numColumns + 1) == sparseColumnIndices.size());

    std::copy(sparseColumnIndices.begin(), sparseColumnIndices.end(), indicesDst + nnzCount);

    return mbLayout;
}
} } }
