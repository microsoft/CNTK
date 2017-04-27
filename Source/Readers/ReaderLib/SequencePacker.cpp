//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <numeric>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "SequencePacker.h"
#include "ReaderUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

MBLayoutPtr SequencePacker::CreateMBLayout(const StreamBatch& batch)
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

Minibatch SequencePacker::ReadMinibatch()
{
    auto sequences = m_sequenceEnumerator->GetNextSequences(m_globalMinibatchSizeInSamples, m_localMinibatchSizeInSamples);
    const auto& batch = sequences.m_data;

    Minibatch minibatch(sequences.m_endOfSweep, sequences.m_endOfEpoch);
    if (batch.empty())
        return minibatch;

    auto& currentBuffer = m_streamBuffers[m_currentBufferIndex];

    assert(m_outputStreamDescriptions.size() == batch.size());
    for (int streamIndex = 0; streamIndex < batch.size(); ++streamIndex)
    {
        const auto& streamBatch = batch[streamIndex];

        if (m_checkSampleShape[streamIndex])
        {
            CheckSampleShape(streamBatch, m_outputStreamDescriptions[streamIndex]);
        }

        const auto& type = m_outputStreamDescriptions[streamIndex]->m_storageType;
        auto pMBLayout = (type == StorageType::dense) ?
            PackDenseStream(streamBatch, streamIndex) : PackSparseStream(streamBatch, streamIndex);

        auto& buffer = currentBuffer[streamIndex];

        auto streamMinibatch = std::make_shared<StreamMinibatch>();
        streamMinibatch->m_data = buffer.m_data.get();
        streamMinibatch->m_layout = pMBLayout;
        minibatch.m_data.push_back(streamMinibatch);
    }

    EstablishIdToKey(minibatch, sequences);

    m_currentBufferIndex = (m_currentBufferIndex + 1) % m_numberOfBuffers;
    return minibatch;
}

void SequencePacker::SetConfiguration(const ReaderConfiguration& config, const std::vector<MemoryProviderPtr>& memoryProviders)
{
    PackerBase::SetConfiguration(config, memoryProviders);

    if (m_useLocalTimeline)
    {
        // Set global minibatch size to max and local minibatch per worker.
        bool shouldAddOneSample = (int)m_config.m_minibatchSizeInSamples % m_config.m_numberOfWorkers > m_config.m_workerRank;
        m_localMinibatchSizeInSamples = (int)m_config.m_minibatchSizeInSamples / (int)m_config.m_numberOfWorkers + (shouldAddOneSample ? 1 : 0);
        m_globalMinibatchSizeInSamples = SIZE_MAX;

        if (m_localMinibatchSizeInSamples == 0)
        {
            // We expect to have a least a single sample per worker.
            fprintf(stderr, "WARNING: The minibatch size '%" PRIu64 "' is too small to be used with %d workers, adjusting to minibatch size of 1 sample per worker\n",
                m_config.m_minibatchSizeInSamples,
                (int)m_config.m_numberOfWorkers);
            m_localMinibatchSizeInSamples = 1;
        }
    }
    else
    {
        // Set global and minibatch local minibatch size as in config.
        m_globalMinibatchSizeInSamples = m_localMinibatchSizeInSamples = m_config.m_minibatchSizeInSamples;
    }
}

void SequencePacker::CheckSampleShape(const std::vector<SequenceDataPtr>& minibatch, StreamDescriptionPtr outputStream)
{
    assert(!minibatch.empty());

    // TODO: This should come from the network - layout that network expects.
    // TODO: In this case we can make outputStream const.
    // Currently it is not coming from SGD/Network, so we assume the first one is correct.
    if (outputStream->m_sampleLayout == nullptr)
    {
        outputStream->m_sampleLayout = minibatch.front()->m_sampleLayout;
    }

    for (const auto& s : minibatch)
    {
        if (s->m_sampleLayout == nullptr)
        {
            LogicError("Unknown shape of the sequence in stream '%ls'.", outputStream->m_name.c_str());
        }

        if (*s->m_sampleLayout != *outputStream->m_sampleLayout)
        {
            RuntimeError("Packer currently does not support samples with varying shapes."
                "Please make sure there is a transform that unifies the shape of samples for input stream '%ls' "
                "or the deserializer provides samples with the same shape.",
                outputStream->m_name.c_str());
        }
    }
}

MBLayoutPtr SequencePacker::PackDenseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreamDescriptions[streamIndex]->m_storageType == StorageType::dense);
    const auto& stream = m_inputStreamDescriptions[streamIndex];
    auto& buffer = m_streamBuffers[m_currentBufferIndex][streamIndex];
    size_t sampleSize = GetSampleSize(m_outputStreamDescriptions[streamIndex]);
    auto pMBLayout = CreateMBLayout(batch);
    size_t requiredSize = pMBLayout->GetNumCols() * sampleSize;
    if (buffer.m_size < requiredSize)
    {
        buffer.Resize(requiredSize);
    }

    auto elementSize = GetSizeByType(stream->m_elementType);

    const auto& sequenceInfos = pMBLayout->GetAllSequences();

    // Iterate over sequences in the layout, copy samples from the
    // source sequences into the buffer (at appropriate offsets).
    for (int i = 0; i < sequenceInfos.size(); ++i)
    {
        const auto& sequenceInfo = sequenceInfos[i];
        // skip gaps
        if (sequenceInfo.seqId == GAP_SEQUENCE_ID)
        {
            continue;
        }

        const auto& sequence = batch[sequenceInfo.seqId];
        size_t numSamples = sequence->m_numberOfSamples;
        assert(numSamples == sequenceInfo.GetNumTimeSteps());

        char* bufferPtr = buffer.m_data.get();
        // Iterate over all samples in the sequence, keep track of the sample offset (which is especially
        // important for sparse input, where offset == number of preceding nnz elements).
        for (size_t sampleIndex = 0, sampleOffset = 0; sampleIndex < numSamples; ++sampleIndex)
        {
            // Compute the offset into the destination buffer, using the layout information 
            // to get the column index corresponding to the given sample.
            auto destinationOffset = pMBLayout->GetColumnIndex(sequenceInfo, sampleIndex) * sampleSize;
            // verify that there's enough space left in the buffer to fit a full sample.
            assert(destinationOffset <= buffer.m_size - sampleSize);
            auto* destination = bufferPtr + destinationOffset;
            if (stream->m_storageType == StorageType::dense)
            {
                // verify that the offset (an invariant for dense).
                assert(sampleOffset == sampleIndex * sampleSize);
                PackDenseSample(destination, sequence, sampleOffset, sampleSize);
                sampleOffset += sampleSize;
            }
            else if (stream->m_storageType == StorageType::sparse_csc)
            {
                // TODO: make type casts members of the SparseSequenceData
                SparseSequenceDataPtr sparseSequence = static_pointer_cast<SparseSequenceData>(sequence);
                // make sure that the sequence meta-data is correct.
                assert(numSamples == sparseSequence->m_nnzCounts.size());
                PackSparseSampleAsDense(destination, sparseSequence, sampleIndex, sampleOffset, sampleSize, elementSize);
                // move the offset by nnz count of the sample.
                sampleOffset += sparseSequence->m_nnzCounts[sampleIndex];
                // verify that the offset is within the bounds (less or equal 
                // to the total nnz count of the sequence).
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

MBLayoutPtr SequencePacker::PackSparseStream(const StreamBatch& batch, size_t streamIndex)
{
    assert(m_outputStreamDescriptions[streamIndex]->m_storageType == StorageType::sparse_csc);

    // compute the aggregate nnz count of all the sequence in the batch.
    size_t nnzCount = 0;
    for (const auto& sequence : batch)
    {
        SparseSequenceDataPtr sparseSequence = static_pointer_cast<SparseSequenceData>(sequence);
        nnzCount += sparseSequence->m_totalNnzCount;
    }

    if (nnzCount > numeric_limits<IndexType>::max())
    {
        RuntimeError("Minibatch NNZ count (%" PRIu64 ") exceeds the maximum allowed "
            "value (%" PRIu64 ")\n", nnzCount, (size_t)numeric_limits<IndexType>::max());
    }

    const auto& stream = m_inputStreamDescriptions[streamIndex];
    assert(stream->m_storageType == StorageType::sparse_csc);
    auto elementSize = GetSizeByType(stream->m_elementType);
    auto indexSize = sizeof(IndexType);
    auto pMBLayout = CreateMBLayout(batch);

    // Compute the required buffer size:
    // size of nnz type + nnz * (size of the element type) + nnz * (size of the row index type) + 
    // (number of columns + 1) * (size of the column index type). 
    size_t requiredSize =
        sizeof(nnzCount) +
        nnzCount * (elementSize + indexSize) +
        indexSize * (pMBLayout->GetNumCols() + 1);

    auto& buffer = m_streamBuffers[m_currentBufferIndex][streamIndex];
    if (buffer.m_size < requiredSize)
    {
        buffer.Resize(requiredSize);
    }

    auto* destination = buffer.m_data.get();
    // insert the nnzCount as the first element in the buffer.
    memcpy(destination, &nnzCount, sizeof(nnzCount));

    // create two pointers to the memory blocks inside the buffer,
    // one for data portion and another -- for indices.
    auto* dataDst = destination + sizeof(nnzCount);
    auto* indicesDst = dataDst + elementSize* nnzCount;
    // column index for the current sample (= number of nnz value packed so far).
    IndexType columnOffset = 0;
    // a vector to store column index for each sample in the resulting (packed) matrix.
    vector<IndexType> sparseColumnIndices;
    // a vector to keep track of the offsets into each input sequence,
    // there an offset is the number of nnz values packed so far. Current sample
    // values/indices start of the offset position in the sequence data/index array
    vector<IndexType>  sequenceOffsets(batch.size(), 0); 

    vector<MBLayout::SequenceInfo> sequenceInfos(pMBLayout->GetAllSequences());

    // sort the vector in ascending order of the parallel sequence index.
    sort(sequenceInfos.begin(), sequenceInfos.end(),
        [](const MBLayout::SequenceInfo& a, const MBLayout::SequenceInfo& b){ return a.s < b.s; });

    // Iterate over the all time steps in the layout (total number of samples/columns 
    // in a parallel sequence), traversing the layout in horizontal direction.
    for (auto timeStep = 0; timeStep < pMBLayout->GetNumTimeSteps(); ++timeStep)
    {
        // For each time step, iterate over all sequences in the minibatch,
        // traversing the layout in vertical direction.
        for (const auto& sequenceInfo : sequenceInfos)
        {
            // skip the sequence if it does not intersect with the time step
            if (timeStep < sequenceInfo.tBegin || timeStep >= sequenceInfo.tEnd)
            {
                continue;
            }

            // store the offset of the current column )...
            sparseColumnIndices.push_back(columnOffset);

            auto seqId = sequenceInfo.seqId;
            if (seqId == GAP_SEQUENCE_ID)
            {
                continue;
            }

            // compute the index of the sample inside the sequence.
            size_t sampleIndex = timeStep - sequenceInfo.tBegin;
            const auto& sequence = batch[seqId];

            // make sure the index less than the sequence length in samples.
            assert(sampleIndex < sequence->m_numberOfSamples);

            auto& sequenceOffset = sequenceOffsets[seqId];
            SparseSequenceDataPtr sparseSequence = static_pointer_cast<SparseSequenceData>(sequence);
            IndexType nnz = sparseSequence->m_nnzCounts[sampleIndex];

            // compute the sample offset in bytes.
            size_t sampleOffset = sequenceOffset * elementSize;
            // copy all nzz values from source sequence into the buffer.
            const auto* dataSrc = reinterpret_cast<const char*>(sequence->GetDataBuffer()) + sampleOffset;
            memcpy(dataDst, dataSrc, nnz * elementSize);
            dataDst += nnz * elementSize; // advance the destination pointer

            // copy all nzz value indices from source sequence into the buffer.
            const auto* indicesSrc = sparseSequence->m_indices + sequenceOffset;
            memcpy(indicesDst, indicesSrc, nnz * indexSize);
            indicesDst += nnz * indexSize; // advance the destination pointer

            sequenceOffset += nnz;
            columnOffset += nnz;
        }
    }

    // at this point each element in sequenceOffsets should be equal to the total
    // nnz count of the respective sequence and the sum of all elements - to the 
    // overall nnz count.
    assert(accumulate(sequenceOffsets.begin(), sequenceOffsets.end(), 0) == nnzCount);

    // check the distance between data and index destination pointers.
    assert(indicesDst == dataDst + nnzCount * indexSize);
    // after we packed all samples, the column offset must be equal to the total nnz count.
    assert(columnOffset == nnzCount);
    sparseColumnIndices.push_back(columnOffset);
    // check that the number of column indices == N + 1 (where N is the number of
    // column in the packed matrix)
    assert((pMBLayout->GetNumCols() + 1) == sparseColumnIndices.size());

    // verify that there's enough space in the buffer for the array of column indices.
    assert(indicesDst + sparseColumnIndices.size()*indexSize <= destination + requiredSize);
    // copy column indices into the buffer.
    memcpy(indicesDst, sparseColumnIndices.data(), sparseColumnIndices.size() * indexSize);

    return pMBLayout;
}

}}}
