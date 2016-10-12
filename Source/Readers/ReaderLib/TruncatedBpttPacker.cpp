//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <cmath>
#include <deque>
#include "TruncatedBpttPacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// Represents a slot where we accumulate sequences from which the minibatch is created.
// The number of slots equals number of parallel sequences we want to pack.
class Slot
{
public:
    Slot() : m_length(0), m_sampleCursor(0), m_sampleOffset(0)
    {}

    // Checks if slot is empty.
    bool IsEmpty() const
    {
        return m_sequences.empty();
    }

    // Gets the number of available samples in the slot.
    size_t AvailableNumberOfSamples() const
    {
        assert(m_length >= m_sampleCursor);
        assert(m_sequences.empty() ? m_sampleCursor == 0 : FrontSequence()->m_numberOfSamples >= m_sampleCursor);
        return m_length - m_sampleCursor;
    }

    // Adds a new sequence to the end of the slot.
    void PushSequence(SequenceDataPtr s)
    {
        m_sequences.push_back(s);
        m_length += s->m_numberOfSamples;
    }

    SequenceDataPtr FrontSequence() const
    {
        assert(!m_sequences.empty());
        return m_sequences.front();
    }

    // Pops the front sequence at the beginning of the slot.
    void PopSequence()
    {
        assert(!m_sequences.empty());
        m_sampleCursor = 0;
        m_sampleOffset = 0;
        m_length -= m_sequences.front()->m_numberOfSamples;
        m_sequences.pop_front();
    }

    // Contains the current sample cursor in the first sequence(m_sequences.front()) of the slot.
    size_t m_sampleCursor;

    // offset of the current sample into the data region of the first sequence.
    // For dense input, this is just (sample cursor) x (sample size in bytes).
    // For sparse input, this is (element size in bytes) x (sum of nnz counts 
    // of all preceding samples).
    size_t m_sampleOffset; 

private:
    // Prepared sequences.
    deque<SequenceDataPtr> m_sequences;

    // Contains the size of the slot in samples (accumulated over all m_sequences).
    size_t m_length;
};

// Represents a buffer of slots from which the minibatch is created.
struct SequenceBuffer
{
    SequenceBuffer(size_t numParallelSequences)
    {
        // Allocates required slots.
        m_slots.resize(numParallelSequences);
    }

    // Checks whether there is more data available in any of the slots.
    bool NothingToPack() const
    {
        auto it = find_if(m_slots.begin(), m_slots.end(), [](const Slot& s) -> bool { return !s.IsEmpty(); });
        return it == m_slots.end();
    }

    // A matrix of prepared sequences. The number of rows(RN) = m_numParallelSequences = number of slots
    // in each row we at least holding sequences to fill in the truncation length.
    // Only at the end of the epoch there could be less than truncation length number of samples in this matrix
    //
    // It looks something like that:
    // slot1: /***s11***/ /***s12**/
    //  ....
    // slotM: /**********sM1****/
    //  ....
    // slotN: /*sRN1*//*sRN2*//*sRN2*/
    vector<Slot> m_slots;
};

TruncatedBPTTPacker::TruncatedBPTTPacker(
    SequenceEnumeratorPtr sequenceEnumerator,
    const vector<StreamDescriptionPtr>& streams,
    size_t numberOfBuffers)
    : PackerBase(sequenceEnumerator, streams, numberOfBuffers),
    m_truncationSize(0)
{
    auto sparseOutput = find_if(m_outputStreamDescriptions.begin(), m_outputStreamDescriptions.end(), [](const StreamDescriptionPtr& s){ return s->m_storageType == StorageType::sparse_csc; });
    if (sparseOutput != m_outputStreamDescriptions.end())
    {
        // TODO: add support for sparse.
        RuntimeError("Sparse output is not supported in BPTT mode.");
    }

    // Preparing layouts.
    for (int i = 0; i < m_outputStreamDescriptions.size(); ++i)
    {
        auto pMBLayout = make_shared<MBLayout>();
        pMBLayout->SetUniqueAxisName(L"TruncatedBPTTPacker");
        m_currentLayouts.push_back(pMBLayout);
    }
}

void TruncatedBPTTPacker::StartEpoch(const EpochConfiguration& config, const std::vector<MemoryProviderPtr>& memoryProviders)
{
    PackerBase::StartEpoch(config, memoryProviders);

    if (m_minibatchSize != config.m_minibatchSizeInSamples ||
        m_truncationSize != config.m_truncationSize)
    {
        m_minibatchSize = config.m_minibatchSizeInSamples;
        m_truncationSize = config.m_truncationSize;

        if (m_minibatchSize == 0)
        {
            LogicError("Minibatch size cannot be zero.");
        }
        if (m_truncationSize == 0)
        {
            LogicError("Truncation size cannot be zero.");
        }

        // Estimating the number of parallel sequences to pack (slots) from the minibatch size and truncation size.
        m_numParallelSequences = max(1, static_cast<int>(std::floor(m_minibatchSize / m_truncationSize)));

        if (config.m_numberOfWorkers > m_numParallelSequences)
        {
            InvalidArgument("Too many workers for minibatch size; please increase minibatch size or decrease number of workers.");
        }

        m_numParallelSequences =
            (m_numParallelSequences / config.m_numberOfWorkers) +
            (config.m_workerRank < (m_numParallelSequences % config.m_numberOfWorkers) ? 1 : 0);

        m_sequenceBufferPerStream.clear();

        // Preparing the buffers.
        for (int j = 0; j < m_streamBuffers.size(); ++j)
            for (int i = 0; i < m_outputStreamDescriptions.size(); ++i)
            {
                const auto& stream = m_outputStreamDescriptions[i];
                auto& buffer = m_streamBuffers[j][i];
                buffer.Resize(m_numParallelSequences * m_truncationSize * GetSampleSize(stream));
                m_sequenceBufferPerStream.push_back(make_shared<SequenceBuffer>(m_numParallelSequences));
            }
    }

    // Filling in the initial set of sequences
    for (size_t slotIndex = 0; slotIndex < m_numParallelSequences; ++slotIndex)
    {
        ReadSequencesToSlot(slotIndex);
    }
}

Minibatch TruncatedBPTTPacker::ReadMinibatch()
{
    Minibatch result;

    // Currently all we expect sequences of identical length between different streams,
    // so it is sufficient to check a single stream only.
    if (m_sequenceBufferPerStream.front()->NothingToPack())
    {
        result.m_endOfEpoch = true;
        return result;
    }

    // Iterating over the streams/slots and packing them into the minibatch.
    for (size_t streamIndex = 0; streamIndex < m_outputStreamDescriptions.size(); ++streamIndex)
    {
        m_currentLayouts[streamIndex]->Init(m_numParallelSequences, m_truncationSize);
        size_t sequenceId = 0;
        for (size_t slotIndex = 0; slotIndex < m_numParallelSequences; ++slotIndex)
        {
            PackSlot(streamIndex, slotIndex, sequenceId);
        }

        StreamMinibatchPtr m = make_shared<StreamMinibatch>();
        m->m_data = m_streamBuffers[m_currentBufferIndex][streamIndex].m_data.get();
        m->m_layout = m_currentLayouts[streamIndex];
        result.m_data.push_back(m);
    }

    m_currentBufferIndex = (m_currentBufferIndex + 1) % m_numberOfBuffers;

    return result;
}

// Packs a slot of sequences into the minibatch.
void TruncatedBPTTPacker::PackSlot(size_t streamIndex, size_t slotIndex, size_t& sequenceId)
{
    auto& slot = m_sequenceBufferPerStream[streamIndex]->m_slots[slotIndex];

    // Fill free space in the slot.
    ReadSequencesToSlot(slotIndex);

    // Let's see how much samples we need to read.
    size_t numberOfSamples = min(m_truncationSize, slot.AvailableNumberOfSamples());
    if (numberOfSamples == 0)
    {
        // Reached the end of the data, put the corresponding row in the minibatch layout to gap.
        m_currentLayouts[streamIndex]->AddSequence(GAP_SEQUENCE_ID, slotIndex, 0, m_truncationSize);

        // Check that nothing is in the slot any more.
        assert(slot.IsEmpty());
        return;
    }

    size_t sampleSize = GetSampleSize(m_inputStreamDescriptions[streamIndex]);
    StorageType storageType = m_inputStreamDescriptions[streamIndex]->m_storageType;
    size_t elementSize = GetSizeByType(m_inputStreamDescriptions[streamIndex]->m_elementType);

    // Distance between two samples of the same sequence in bytes.
    size_t strideSize = m_numParallelSequences * sampleSize;

    // Add current sequence to the minibatch layout.
    m_currentLayouts[streamIndex]->AddSequence(
        sequenceId++,
        slotIndex,
        -(int)slot.m_sampleCursor,
        slot.FrontSequence()->m_numberOfSamples - slot.m_sampleCursor);

    // Ok, now fill in the buffer with data.
    for (size_t currentTimestep = 0; currentTimestep < numberOfSamples; ++currentTimestep)
    {
        // Check if reach the end of the front sequence.
        if (slot.m_sampleCursor >= slot.FrontSequence()->m_numberOfSamples)
        {
            // Starting a new sequence. Have to reset current pointers and add it to the minibatch layout.
            slot.PopSequence();

            //Adding next sequence to the minibatch.
            m_currentLayouts[streamIndex]->AddSequence(
                sequenceId++,
                slotIndex,
                currentTimestep,
                currentTimestep + slot.FrontSequence()->m_numberOfSamples);
        }

        // Fill in the data from the first sequence in the slot.
        auto data = slot.FrontSequence();
        // Get buffer destination for the current sample.
        auto& buffer = m_streamBuffers[m_currentBufferIndex][streamIndex];
        auto offset = strideSize * currentTimestep + slotIndex * sampleSize;
        assert(offset >= 0 && offset < buffer.m_size);
        char* destination = buffer.m_data.get() + offset;

        // Pack the sample.
        if (storageType == StorageType::dense)
        {
            assert(slot.m_sampleOffset == slot.m_sampleCursor * sampleSize);
            PackDenseSample(destination, data, slot.m_sampleOffset, sampleSize);
            slot.m_sampleOffset += sampleSize;
        }
        else
        {
            assert(storageType == StorageType::sparse_csc);
            // TODO: make type casts members of the SparseSequenceData
            SparseSequenceDataPtr sparseSequence = static_pointer_cast<SparseSequenceData>(data);
            assert(slot.m_sampleCursor < sparseSequence->m_nnzCounts.size());
            PackSparseSampleAsDense(destination, sparseSequence, slot.m_sampleCursor, 
                slot.m_sampleOffset, sampleSize, elementSize);
            slot.m_sampleOffset += sparseSequence->m_nnzCounts[slot.m_sampleCursor];
            assert(slot.m_sampleOffset <= sparseSequence->m_totalNnzCount);
        }

        slot.m_sampleCursor++;
    }

    // Cleaning up the last sequence we have just read if needed.
    if (slot.m_sampleCursor >= slot.FrontSequence()->m_numberOfSamples)
    {
        slot.PopSequence();
    }

    // Adding the last gap if there is one.
    if (numberOfSamples < m_truncationSize)
    {
        m_currentLayouts[streamIndex]->AddSequence(
            GAP_SEQUENCE_ID,
            slotIndex,
            numberOfSamples,
            m_truncationSize);
    }
}

void TruncatedBPTTPacker::ReadSequencesToSlot(size_t slotIndex)
{
    const auto& slot = m_sequenceBufferPerStream.front()->m_slots[slotIndex];
    while (m_truncationSize >= slot.AvailableNumberOfSamples())
    {
        // We need a single sequence, potentially we can request (m_truncationSize - slot.AvailableNumberOfSamples())
        // to be more efficient. In reality the truncation size usually is less the sequence size.
        auto s = m_sequenceEnumerator->GetNextSequences(1);
        if (s.m_endOfEpoch)
        {
            break;
        }

        // Adding sequence to the slot for all streams.
        for (size_t i = 0; i < s.m_data.size(); ++i)
        {
            assert(s.m_data[i].size() == 1);

            // Check that all sequences are of the same length.
            if (s.m_data.front().front()->m_numberOfSamples != s.m_data[i].front()->m_numberOfSamples)
            {
                RuntimeError("For BPTT sequences between different input stream should have the same length.");
            }

            m_sequenceBufferPerStream[i]->m_slots[slotIndex].PushSequence(s.m_data[i].front());
        }
    }
}

}}}
