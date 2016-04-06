//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <inttypes.h>
#include <numeric>
#include "PackerBase.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// TODO: this should be handled by the memory provider
void PackerBase::StreamBuffer::Resize(size_t newSize)
{
    m_size = newSize;
    m_data.reset(reinterpret_cast<char*>(m_memoryProvider->Alloc(1, newSize)),
        [this](char* p)
    {
        m_memoryProvider->Free(p);
    });
}


PackerBase::PackerBase(MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams) :
    m_transformer(transformer),
    m_minibatchSize(minibatchSize),
    m_outputStreamDescriptions(streams)
{
    m_inputStreamDescriptions = m_transformer->GetStreamDescriptions();
    assert(m_inputStreamDescriptions.size() != 0);
    assert(m_inputStreamDescriptions.size() == m_outputStreamDescriptions.size());

    if (m_minibatchSize == 0)
    {
        LogicError("Minibatch size cannot be zero.");
    }

    m_streamBuffers.reserve(m_outputStreamDescriptions.size());

    // Sanity checks:
    for (size_t i = 0; i < m_outputStreamDescriptions.size(); ++i)
    {
        const auto& stream = m_outputStreamDescriptions[i];
        UNUSED(stream);

        // Input and output should match in everything except for sparse/dense storage type.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreamDescriptions[i]->m_name);
        assert(stream->m_id == m_inputStreamDescriptions[i]->m_id);
        assert(GetSampleSize(m_inputStreamDescriptions[i]) == GetSampleSize(stream));

        if (m_inputStreamDescriptions[i]->m_storageType == StorageType::dense &&
            stream->m_storageType == StorageType::sparse_csc)
        {
            RuntimeError("Dense to sparse re-packing requested for stream '%ls' is not supported.",
                stream->m_name.c_str());
        }

        m_streamBuffers.push_back(StreamBuffer(memoryProvider));
    }
}

Minibatch PackerBase::ReadMinibatch()
{
    auto sequences = m_transformer->GetNextSequences(m_minibatchSize);
    const auto& batch = sequences.m_data;

    Minibatch minibatch(sequences.m_endOfEpoch);
    if (batch.empty())
    {
        return minibatch;
    }

    assert(m_outputStreamDescriptions.size() == batch.size());

    for (int streamIndex = 0; streamIndex < batch.size(); ++streamIndex)
    {
        const auto& streamBatch = batch[streamIndex];
        const auto& type = m_outputStreamDescriptions[streamIndex]->m_storageType;
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

// Gets samples size in bytes.
size_t PackerBase::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

MBLayoutPtr PackerBase::PackDenseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreamDescriptions[streamIndex]->m_storageType == StorageType::dense);
    const auto& stream = m_inputStreamDescriptions[streamIndex];
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
        size_t numSamples = sequence->m_numberOfSamples;
        assert(numSamples == sequenceInfo.GetNumTimeSteps());

        char* bufferPtr = buffer.m_data.get();
        for (size_t sampleIndex = 0, sampleOffset = 0; sampleIndex < numSamples; ++sampleIndex)
        {
            auto* destination = bufferPtr + pMBLayout->GetColumnIndex(sequenceInfo, sampleIndex) * sampleSize;
            assert(destination <= bufferPtr + buffer.m_size - sampleSize);
            if (stream->m_storageType == StorageType::dense)
            {
                assert(sampleOffset == sampleIndex * sampleSize);
                PackDenseSample(destination, sequence, sampleOffset, sampleSize);
                sampleOffset += sampleSize;
            }
            else if (stream->m_storageType == StorageType::sparse_csc)
            {
                // TODO: make type casts members of the SparseSequenceData
                SparseSequenceDataPtr sparseSequence = static_pointer_cast<SparseSequenceData>(sequence);
                assert(numSamples == sparseSequence->m_nnzCounts.size());
                PackSparseSampleAsDense(destination, sparseSequence, sampleIndex, sampleOffset, sampleSize, elementSize);
                sampleOffset += sparseSequence->m_nnzCounts[sampleIndex];
                assert(sampleOffset <= sparseSequence->m_totalNnzCount);
            }
            else
            {
                RuntimeError("Storage type %d is not supported.", (int)stream->m_storageType);
            }
        }
    }

    return pMBLayout;
}

MBLayoutPtr PackerBase::PackSparseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreamDescriptions[streamIndex]->m_storageType == StorageType::sparse_csc);

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

    const auto& stream = m_inputStreamDescriptions[streamIndex];
    assert(stream->m_storageType == StorageType::sparse_csc);
    auto elementSize = GetSizeByType(stream->m_elementType);
    auto indexSize = sizeof(IndexType);
    auto pMBLayout = CreateMBLayout(batch);

    // size of nnz type + nnz * (size of the element type) + nnz * (size of the row index type) + 
    // (number of columns + 1) * (size of the column index type). 
    size_t requiredSize =
        sizeof(nnzCount) +
        nnzCount * (elementSize + indexSize) +
        indexSize * (pMBLayout->GetNumCols() + 1);

    auto& buffer = m_streamBuffers[streamIndex];
    if (buffer.m_size < requiredSize)
    {
        buffer.Resize(requiredSize);
    }

    auto* destination = buffer.m_data.get();
    // insert the nnzCount as the first element in the buffer.
    memcpy(destination, &nnzCount, sizeof(nnzCount));

    auto* dataDst = destination + sizeof(nnzCount);
    auto* indicesDst = dataDst + elementSize* nnzCount;
    IndexType columnOffset = 0;
    vector<IndexType> sparseColumnIndices;
    vector<IndexType>  sequenceOffsets(batch.size(), 0);

    const auto& infos = pMBLayout->GetAllSequences();
    vector<const MBLayout::SequenceInfo*> sequenceInfos(infos.size());

    for (auto i = 0; i < infos.size(); ++i)
    {
        sequenceInfos[i] = &infos[i];
    }

    // sort the vector in ascending order of the parallel sequence index.
    sort(sequenceInfos.begin(), sequenceInfos.end(),
        [](const MBLayout::SequenceInfo* a, const MBLayout::SequenceInfo* b){ return a->s < b->s; });

    for (auto timeStep = 0; timeStep < pMBLayout->GetNumTimeSteps(); ++timeStep)
    {
        for (const auto* sequenceInfo : sequenceInfos)
        {

            if (timeStep < sequenceInfo->tBegin || timeStep >= sequenceInfo->tEnd)
            {
                continue;
            }

            sparseColumnIndices.push_back(columnOffset);

            auto seqId = sequenceInfo->seqId;
            if (seqId == GAP_SEQUENCE_ID)
            {
                continue;
            }

            size_t sampleIndex = timeStep - sequenceInfo->tBegin;

            const auto& sequence = batch[seqId];
            assert(sampleIndex < sequence->m_numberOfSamples);

            auto& sequenceOffset = sequenceOffsets[seqId];
            const auto& sparseSequence = reinterpret_cast<SparseSequenceData&>(*sequence);
            IndexType nnz = sparseSequence.m_nnzCounts[sampleIndex];

            size_t sampleOffset = sequenceOffset * elementSize;
            const auto* dataSrc = reinterpret_cast<const char*>(sequence->m_data) + sampleOffset;
            memcpy(dataDst, dataSrc, nnz * elementSize);
            dataDst += nnz * elementSize;

            const auto* indicesSrc = sparseSequence.m_indices + sequenceOffset;
            memcpy(indicesDst, indicesSrc, nnz * indexSize);
            indicesDst += nnz * indexSize;

            sequenceOffset += nnz;
            columnOffset += nnz;
        }
    }

    // at this point each element in sequenceOffsets should be equal to the total
    // nnz count of the respective sequence and the sum of all elements - to the 
    // overall nnz count.
    assert(accumulate(sequenceOffsets.begin(), sequenceOffsets.end(), 0) == nnzCount);

    assert(indicesDst == dataDst + nnzCount * indexSize);
    assert(columnOffset == nnzCount);
    sparseColumnIndices.push_back(columnOffset);
    assert((pMBLayout->GetNumCols() + 1) == sparseColumnIndices.size());

    assert(indicesDst + sparseColumnIndices.size()*indexSize <= destination + requiredSize);

    memcpy(indicesDst, sparseColumnIndices.data(), sparseColumnIndices.size() * indexSize);

    return pMBLayout;
}


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
