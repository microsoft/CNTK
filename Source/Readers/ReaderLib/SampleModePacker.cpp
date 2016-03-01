//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <numeric>
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

    Minibatch minibatch;
    minibatch.m_endOfEpoch = sequences.m_endOfEpoch;

    if (batch.size() == 0)
    {
        return minibatch;
    }

    for (int streamIndex = 0; streamIndex < m_numberOfStreams; ++streamIndex)
    {
        const auto& type = m_outputStreams[streamIndex]->m_storageType;
        auto layout = (type == StorageType::dense) ?
            PackDenseStream(batch, streamIndex) : PackSparseStream(batch, streamIndex);

        auto& buffer = m_streamBuffers[streamIndex];


        auto streamMinibatch = std::make_shared<StreamMinibatch>();
        streamMinibatch->m_data = buffer.m_data.get();
        // TODO: m_dataSize not really used and can be removed.
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


size_t SampleModePacker::GetMaxSequenceLength(const SequenceBatch& batch, size_t streamIndex)
{
    size_t maxLength = 0;
    const StorageType type = m_inputStreams[streamIndex]->m_storageType;
    for (const auto& sequences : batch)
    {
        const auto& sequence = sequences[streamIndex];
        size_t numSamples = 0;
        if (type == StorageType::dense)
        {
            const auto& denseSequence = reinterpret_cast<const DenseSequenceData&>(*sequence);
            numSamples = denseSequence.m_numberOfSamples;
        }
        else
        {
            const auto& sparseSequence = reinterpret_cast<const SparseSequenceData&>(*sequence);
            numSamples = sparseSequence.m_indices.size();
        }

        maxLength = max(maxLength, numSamples);
    }
    return maxLength;
}


MBLayoutPtr SampleModePacker::PackDenseStream(const SequenceBatch& batch, size_t streamIndex)
{
    assert(m_outputStreams[streamIndex]->m_storageType == StorageType::dense);

    const auto& stream = m_inputStreams[streamIndex];
    auto& buffer = m_streamBuffers[streamIndex];
    size_t sampleSize = GetSampleSize(stream);
    auto elementSize = GetSizeByType(stream->m_elementType);
    size_t maxSequenceLength = GetMaxSequenceLength(batch, streamIndex);
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
        assert(m_numberOfStreams == batch[sequenceIndex].size());

        // it'd be nicer if all sequences from a particular stream were
        // stored in a single vector, i.e. sequence = batch[streamIndex][sequenceIndex]
        // In which case, there's actually no need to pass the whole batch 
        // inside this method
        const auto& sequence = batch[sequenceIndex][streamIndex];
        char* source = reinterpret_cast<char*>(sequence->m_data);
        size_t numSamples = 0;

        if (stream->m_storageType == StorageType::dense)
        {
            const auto& denseSequence = reinterpret_cast<DenseSequenceData&>(*sequence);
            numSamples = denseSequence.m_numberOfSamples;
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
            numSamples = sparseSequence.m_indices.size();
            char* destination = buffer.m_data.get() + sequenceIndex * sampleSize;

            // Copy the non zero data to the buffer.
            for (size_t sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
            {
                const auto& indices = sparseSequence.m_indices[sampleIndex];
                size_t nonZeroCount = indices.size();
                for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
                {
                    size_t rowIndex = indices[nonZeroIndex];
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
        mbLayout->AddSequence(NEW_SEQUENCE_ID, sequenceIndex, 0, numSamples);
        if (numSamples < maxSequenceLength)
        {
            mbLayout->AddGap(sequenceIndex, numSamples, maxSequenceLength);
        }
    }
    return mbLayout;
}

MBLayoutPtr SampleModePacker::PackSparseStream(const SequenceBatch& batch, size_t streamIndex)
{
    assert(m_outputStreams[streamIndex]->m_storageType == StorageType::sparse_csc);

    const auto& stream = m_inputStreams[streamIndex];

    assert(stream->m_storageType == StorageType::sparse_csc);

    auto& buffer = m_streamBuffers[streamIndex];
    
    auto elementSize = GetSizeByType(stream->m_elementType);

    auto mbLayout = std::make_shared<MBLayout>();

    size_t maxSequenceLength = GetMaxSequenceLength(batch, streamIndex);

    // TODO: need to figure out how to pack sparse sequences in the non-frame mode.
    assert(maxSequenceLength == 1); 

    size_t numSequences = batch.size();

    mbLayout->Init(numSequences, maxSequenceLength);

    size_t nnzCount = 0;
    vector<IndexType> sparseRowIndices;
    vector<IndexType> sparseColumnIndices;
    vector<size_t> perSequenceNonZeroCounts;

    sparseColumnIndices.push_back(0);
    // This whole thing will disappear, once SparseSequenceData is refactored 
    // to contain nnz and proper type (int32_t) for the indices.
    for (const auto& sequences : batch) {
        const auto& sequence = sequences[streamIndex];
        const auto& sparseSequence = reinterpret_cast<const SparseSequenceData&>(*sequence);

        size_t nnz = 0;
        for (const auto& sampleIndices : sparseSequence.m_indices)
        {
            for (const size_t& index : sampleIndices)
            {
                sparseRowIndices.push_back(static_cast<IndexType>(index));
            }
            nnz += sampleIndices.size();
            sparseColumnIndices.push_back(static_cast<IndexType>(sparseRowIndices.size()));
        }
        perSequenceNonZeroCounts.push_back(nnz);
        nnzCount += nnz;
    }

    assert(nnzCount == sparseRowIndices.size());

    // size of nnz type + nnz * (size of the element type) + nnz * (size of the row index type) + 
    // number of columns * (size of the column index type). 
    size_t requiredSize =
        sizeof(nnzCount) +
        nnzCount * (elementSize + sizeof(IndexType)) +
        sizeof(IndexType) * sparseColumnIndices.size();
    
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

    for (size_t sequenceIndex = 0; sequenceIndex < numSequences; ++sequenceIndex)
    {
        assert(m_numberOfStreams == batch[sequenceIndex].size());

        // it'd be nicer if all sequences from a particular stream were
        // stored in a single vector, i.e. sequence = batch[streamIndex][sequenceIndex]
        // In which case, there's actually no need to pass the whole batch 
        // inside this metod
        const auto& sequence = batch[sequenceIndex][streamIndex];
        auto data = reinterpret_cast<const char*>(sequence->m_data);

        size_t numSamples = 0;

        const auto& sparseSequence = reinterpret_cast<SparseSequenceData&>(*sequence);
        numSamples = sparseSequence.m_indices.size();
        size_t size = perSequenceNonZeroCounts[sequenceIndex] * elementSize;
        std::copy(data, data + size, destination);
        destination += size;
        // We don't do any packing per se, instead we create MB with 
        // the number of parallel sequences equal to the number of 
        // Sequences in the batch.
        mbLayout->AddSequence(NEW_SEQUENCE_ID, sequenceIndex, 0, numSamples);
        if (numSamples < maxSequenceLength)
        {
            mbLayout->AddGap(sequenceIndex, numSamples, maxSequenceLength);
        }
    }

    assert(buffer.m_data.get() + sizeof(nnzCount) + nnzCount * elementSize == destination);

    size_t size = nnzCount * sizeof(IndexType);
    source = reinterpret_cast<const char*>(sparseRowIndices.data());
    std::copy(source, source + size, destination);
    destination += size;

    size = sizeof(IndexType) * (sparseColumnIndices.size());
    source = reinterpret_cast<const char*>(sparseColumnIndices.data());
    std::copy(source, source + size, destination);

    return mbLayout;
}
} } }
