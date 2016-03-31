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

// TODO: this should be handled by the memory provider
void StreamBuffer::Resize(size_t newSize)
{
    m_size = newSize;
    m_data.reset(reinterpret_cast<char*>(m_memoryProvider->Alloc(1, newSize)),
        [this](char* p)
    {
        m_memoryProvider->Free(p);
    });
}



SampleModePacker::SampleModePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams) : 
    m_transformer(transformer),
                                                        m_minibatchSize(minibatchSize),
    m_numberOfStreams(streams.size()),
    m_outputStreams(streams)
{
    m_inputStreams = m_transformer->GetStreamDescriptions();
    assert(m_inputStreams.size() == m_numberOfStreams);
    assert(m_minibatchSize > 0);

    m_streamBuffers.reserve(m_numberOfStreams);
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

        m_streamBuffers.push_back(StreamBuffer(memoryProvider));
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
        auto pMBLayout = (type == StorageType::dense) ?
            PackDenseStream(streamBatch, streamIndex) : PackSparseStream(streamBatch, streamIndex);

        auto& buffer = m_streamBuffers[streamIndex];

        auto streamMinibatch = std::make_shared<StreamMinibatch>();
        streamMinibatch->m_data = buffer.m_data.get();
        streamMinibatch->m_layout = pMBLayout;
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


MBLayoutPtr SampleModePacker::CreateMBLayout(const StreamBatch& batch)
{
    vector<MBLayout::SequenceInfo> infos;
    for (size_t index = 0; index < batch.size(); ++index)
    {
        MBLayout::SequenceInfo info;

        info.seqId = index;
        info.tBegin = 0;
        info.tEnd = batch[index]->m_numberOfSamples;
        infos.push_back(info);
    }

    vector<pair<size_t, size_t>> placement;
    vector<size_t> rowAllocations;

    // Creating the minibatch layout.
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    pMBLayout->InitAsPackedSequences(infos, placement, rowAllocations);
    return pMBLayout;
}

MBLayoutPtr SampleModePacker::PackDenseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreams[streamIndex]->m_storageType == StorageType::dense);
    const auto& stream = m_inputStreams[streamIndex];
    auto& buffer = m_streamBuffers[streamIndex];
    size_t sampleSize = GetSampleSize(stream);
    auto pMBLayout = CreateMBLayout(batch);
    size_t requiredSize = pMBLayout->GetNumCols() * sampleSize;
    if (buffer.m_size < requiredSize)
    {
        buffer.Resize(requiredSize);
    }

    auto elementSize = GetSizeByType(stream->m_elementType);

    const auto& sequenceInfos = pMBLayout->GetAllSequences();

    for (const auto& sequenceInfo : sequenceInfos)
    {
        if (sequenceInfo.seqId == GAP_SEQUENCE_ID)
        {
            continue;
        }

        const auto& sequence = batch[sequenceInfo.seqId];
        const char* dataPtr = reinterpret_cast<char*>(sequence->m_data);
        size_t numSamples = sequence->m_numberOfSamples;
        assert(numSamples == sequenceInfo.GetNumTimeSteps());

        char* bufferPtr = buffer.m_data.get();
        for (size_t sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
        {
            auto* destination = bufferPtr + pMBLayout->GetColumnIndex(sequenceInfo, sampleIndex) * sampleSize;
            assert(destination <= bufferPtr + buffer.m_size - sampleSize);
    if (stream->m_storageType == StorageType::dense)
    {
                const auto* source = dataPtr + sampleIndex * sampleSize;
                memcpy(destination, source, sampleSize);
    }
    else if (stream->m_storageType == StorageType::sparse_csc)
    {
                memset(destination, 0, sampleSize);
                // TODO: make type casts members of the SparseSequenceData
                const auto& sparseSequence = reinterpret_cast<SparseSequenceData&>(*sequence);
                size_t nonZeroCount = sparseSequence.m_nnzCounts[sampleIndex];
                for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
                {
                    auto rowIndex = sparseSequence.m_indices[nonZeroIndex];
                    size_t elementOffset = rowIndex * elementSize;
                    assert(elementOffset < sampleSize);
                    const auto* source = dataPtr + nonZeroIndex * elementSize;
                    memcpy(destination + elementOffset, source, elementSize);
                }
            }
            else
            {
                RuntimeError("Storage type %d is not supported.", (int)stream->m_storageType);
            }
        }
    }

    return pMBLayout;
}

MBLayoutPtr SampleModePacker::PackSparseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreams[streamIndex]->m_storageType == StorageType::sparse_csc);

    size_t nnzCount = 0;
    for (const auto& sequence : batch)
        {
        const auto& sparseSequence = reinterpret_cast<const SparseSequenceData&>(*sequence);
        nnzCount += sparseSequence.m_totalNnzCount;
        }

    if (nnzCount > numeric_limits<IndexType>::max())
    {
        RuntimeError("Minibatch NNZ count (" PRIu64 ") exceeds the maximum allowed "
            "value (" PRIu64 ")\n", nnzCount, (size_t)numeric_limits<IndexType>::max());
    }

    const auto& stream = m_inputStreams[streamIndex];
    assert(stream->m_storageType == StorageType::sparse_csc);
    auto elementSize = GetSizeByType(stream->m_elementType);
    auto pMBLayout = CreateMBLayout(batch);

    // size of nnz type + nnz * (size of the element type) + nnz * (size of the row index type) + 
    // (number of columns + 1) * (size of the column index type). 
    size_t requiredSize =
        sizeof(nnzCount) +
        nnzCount * (elementSize + sizeof(IndexType)) +
        sizeof(IndexType) * (pMBLayout->GetNumCols() + 1);

    auto& buffer = m_streamBuffers[streamIndex];
    if (buffer.m_size < requiredSize)
    {
        buffer.Resize(requiredSize);
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
    vector<IndexType>  sequenceOffsets(batch.size(), 0);

    const auto& sequenceInfos = pMBLayout->GetAllSequences();

    for (int sampleIndex = 0; sampleIndex < pMBLayout->GetNumTimeSteps(); ++sampleIndex)
    {
        for (const auto& sequenceInfo : sequenceInfos)
        {
            
            if (sampleIndex < sequenceInfo.tBegin || sampleIndex >= sequenceInfo.tEnd) 
            {
                continue;
}

            sparseColumnIndices.push_back(columnOffset);

            if (sequenceInfo.seqId == GAP_SEQUENCE_ID)
{
                continue;
            }

            const auto& sequence = batch[sequenceInfo.seqId];
            assert(sampleIndex < sequence->m_numberOfSamples);

            auto& sequenceOffset = sequenceOffsets[sequenceInfo.seqId];
            const auto& sparseSequence = reinterpret_cast<SparseSequenceData&>(*sequence);
            IndexType nnz = sparseSequence.m_nnzCounts[sampleIndex];

            size_t sampleOffset = sequenceOffset * elementSize;
            const auto* dataSrc = reinterpret_cast<const char*>(sequence->m_data) + sampleOffset;
            memcpy(dataDst, dataSrc, nnz * elementSize);
            dataDst += nnz * elementSize;

            const auto* indicesSrc = sparseSequence.m_indices + sequenceOffset;
            memcpy(indicesDst, indicesSrc, nnz);
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
    assert((pMBLayout->GetNumCols() + 1) == sparseColumnIndices.size());

    std::copy(sparseColumnIndices.begin(), sparseColumnIndices.end(), indicesDst + nnzCount);

    return pMBLayout;
}
} } }
