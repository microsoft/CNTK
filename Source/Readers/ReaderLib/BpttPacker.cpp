//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "BpttPacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a slot where we accumulate sequences from which the minibatch is created.
// The number of slots equals number of parallel sequences we want to pack.
struct Slot
{
    Slot() : m_length(0), m_sampleCursor(0)
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
        m_sampleCursor = 0;
        m_length -= m_sequences.front()->m_numberOfSamples;
        m_sequences.pop_front();
    }

    // Contains the current sample cursor in the first sequence(m_sequences.front()) of the slot.
    size_t m_sampleCursor;

private:
    // Prepared sequences.
    std::deque<SequenceDataPtr> m_sequences;

    // Contains the size of the slot in samples (accumulated over all m_sequences).
    size_t m_length;
};

// Represents a buffer of slots from which the minibatch is created.
struct SequenceBuffer
{
    SequenceBuffer(size_t parallelNumberOfSequences)
    {
        // Allocates required slots.
        m_slots.resize(parallelNumberOfSequences);
    }

    // Checks whether there is more data available in any of the slots.
    bool NothingToPack() const
    {
        auto it = std::find_if(m_slots.begin(), m_slots.end(), [](const Slot& s) -> bool { return !s.IsEmpty(); });
        return it == m_slots.end();
    }

    // A matrix of prepared sequences. The number of rows(RN) = m_parallelNumberOfSequences = number of slots
    // in each row we at least holding sequences to fill in the truncation length.
    // Only at the end of the epoch there could be less than truncation length number of samples in this matrix
    //
    // It looks something like that:
    // slot1: /***s11***/ /***s12**/
    //  ....
    // slotM: /**********sM1****/
    //  ....
    // slotN: /*sRN1*//*sRN2*//*sRN2*/
    std::vector<Slot> m_slots;
};

BpttPacker::BpttPacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    size_t truncationSize,
    const std::vector<StreamDescriptionPtr>& streams)
    : PackerBase(memoryProvider, transformer, minibatchSize, streams),
    m_truncationSize(truncationSize)
{
    // Estimating the number of parallel sequences to pack (slots) from the minibatch size and truncation size.
    m_parallelNumberOfSequences = (size_t)std::floor(m_minibatchSize / truncationSize);

    // Preparing the buffers.
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        const auto& stream = m_outputStreams[i];
        m_streamBufferSizes.push_back(m_parallelNumberOfSequences * m_truncationSize * GetSampleSize(stream));
        m_streamBuffers.push_back(AllocateBuffer(m_parallelNumberOfSequences * m_truncationSize, GetSampleSize(stream)));

        m_sequenceBufferPerStream.push_back(std::make_shared<SequenceBuffer>(m_parallelNumberOfSequences));
        m_currentLayouts.push_back(std::make_shared<MBLayout>());
    }

    // Filling in the initial set of sequences
    for (size_t slotIndex = 0; slotIndex < m_parallelNumberOfSequences; ++slotIndex)
    {
        ReadSequencesToSlot(slotIndex);
    }
}

Minibatch BpttPacker::ReadMinibatch()
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
    for (size_t streamIndex = 0; streamIndex < m_outputStreams.size(); ++streamIndex)
    {
        m_currentLayouts[streamIndex]->Init(m_parallelNumberOfSequences, m_truncationSize);
        for (size_t slotIndex = 0; slotIndex < m_parallelNumberOfSequences; ++slotIndex)
        {
            PackSlot(streamIndex, slotIndex);
        }

        StreamMinibatchPtr m = std::make_shared<StreamMinibatch>();
        m->m_data = m_streamBuffers[streamIndex].get();
        m->m_layout = m_currentLayouts[streamIndex];
        result.m_data.push_back(m);
    }

    return result;
}

// Packs a slot of sequences into the minibatch.
void BpttPacker::PackSlot(size_t streamIndex, size_t slotIndex)
{
    auto& slot = m_sequenceBufferPerStream[streamIndex]->m_slots[slotIndex];

    if (slot.AvailableNumberOfSamples() < m_truncationSize)
    {
        // There is some free space in the slot, fill it in if possible.
        ReadSequencesToSlot(slotIndex);
    }

    // Let's see how much samples we need to read.
    size_t numberOfSamples = std::min(m_truncationSize, slot.AvailableNumberOfSamples());
    if (numberOfSamples == 0)
    {
        // Reached the end of the data, put the corresponding row in the minibatch layout to gap.
        m_currentLayouts[streamIndex]->AddSequence(GAP_SEQUENCE_ID, slotIndex, 0, m_truncationSize);

        // Check that nothing is in the slot any more.
        assert(slot.IsEmpty());
        return;
    }

    size_t sampleSize = GetSampleSize(m_inputStreams[streamIndex]);
    StorageType storageType = m_inputStreams[streamIndex]->m_storageType;
    size_t elementSize = GetSizeByType(m_inputStreams[streamIndex]->m_elementType);

    // Distance between two samples of the same sequence in bytes.
    size_t stride = m_parallelNumberOfSequences * sampleSize;

    // Add current sequence to the minibatch layout.
    m_currentLayouts[streamIndex]->AddSequence(
        NEW_SEQUENCE_ID,
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
                NEW_SEQUENCE_ID,
                slotIndex,
                currentTimestep,
                currentTimestep + slot.FrontSequence()->m_numberOfSamples);
        }

        // Fill in the data from the first sequence in the slot.
        auto data = slot.FrontSequence();
        // Get buffer destination for the current sample.
        void* destination = m_streamBuffers[streamIndex].get() + stride * currentTimestep + slotIndex * sampleSize;
        assert(destination >= m_streamBuffers[streamIndex].get());
        assert(destination < m_streamBuffers[streamIndex].get() + m_streamBufferSizes[streamIndex]);

        // Pack the sample.
        if (storageType == StorageType::dense)
        {
            PackDenseSample(destination, data, slot.m_sampleCursor, elementSize, sampleSize);
        }
        else
        {
            assert(storageType == StorageType::sparse_csc);
            PackSparseSample(destination, data, slot.m_sampleCursor, elementSize, sampleSize);
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

void BpttPacker::ReadSequencesToSlot(size_t slotIndex)
{
    const auto& slot = m_sequenceBufferPerStream.front()->m_slots[slotIndex];
    while (m_truncationSize > slot.AvailableNumberOfSamples())
    {
        // We need a single sequence:
        auto s = m_transformer->GetNextSequences(1);
        if (s.m_endOfEpoch)
        {
            break;
        }

        // Adding sequence to the slot for all streams.
        for (size_t i = 0; i < s.m_data.size(); ++i)
        {
            assert(s.m_data[i].size() == 1);
            m_sequenceBufferPerStream[i]->m_slots[slotIndex].PushSequence(s.m_data[i].front());
        }
    }
}

}}}
