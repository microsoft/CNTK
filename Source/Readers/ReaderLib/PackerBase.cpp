//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "PackerBase.h"
#include "ReaderUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// Resizing the buffer with the current memory provider.
void PackerBase::StreamBuffer::Resize(size_t newSize)
{
    m_size = newSize;
    auto provider = m_memoryProvider;
    m_data.reset(reinterpret_cast<char*>(provider->Alloc(1, newSize)),
        [provider](char* p)
    {
        provider->Free(p);
    });
}

void PackerBase::SetConfiguration(const ReaderConfiguration& config, const std::vector<MemoryProviderPtr>& memoryProviders)
{
    // Let's check that memory providers did not change at the start of new epoch.
    bool equalMemoryProviders = m_memoryProviders.size() == memoryProviders.size() &&
        std::equal(memoryProviders.begin(), memoryProviders.end(), m_memoryProviders.begin());

    if (!equalMemoryProviders)
    {
        // If they change we have to reinitialize the buffers with the new memory providers, one per stream.
        m_memoryProviders = memoryProviders;

        if (memoryProviders.size() != m_outputStreamDescriptions.size())
            RuntimeError("Number of streams does not match the number of memory providers.");

        m_streamBuffers.resize(m_numberOfBuffers);
        for (size_t i = 0; i < m_numberOfBuffers; ++i)
        {
            auto& currentBuffer = m_streamBuffers[i];
            currentBuffer.reserve(m_outputStreamDescriptions.size());
            for (size_t j = 0; j < m_outputStreamDescriptions.size(); ++j)
                currentBuffer.push_back(StreamBuffer(memoryProviders[j]));
        }
    }

    m_config = config;
    if (m_config.m_minibatchSizeInSamples == 0)
        LogicError("Minibatch size cannot be zero.");
}

PackerBase::PackerBase(CorpusDescriptorPtr corpus,
    SequenceEnumeratorPtr sequenceEnumerator,
    const std::vector<StreamDescriptionPtr>& streams,
    size_t numberOfBuffers) :
    m_sequenceEnumerator(sequenceEnumerator),
    m_outputStreamDescriptions(streams),
    m_numberOfBuffers(numberOfBuffers),
    m_currentBufferIndex(0),
    m_corpus(corpus)
{
    assert(m_numberOfBuffers >= 1);
    m_inputStreamDescriptions = sequenceEnumerator->GetStreamDescriptions();
    assert(m_inputStreamDescriptions.size() != 0);
    assert(m_inputStreamDescriptions.size() == m_outputStreamDescriptions.size());

    m_checkSampleShape.resize(m_outputStreamDescriptions.size(), false);

    // Sanity checks:
    for (size_t i = 0; i < m_outputStreamDescriptions.size(); ++i)
    {
        const auto& stream = m_outputStreamDescriptions[i];
        UNUSED(stream);

        // Check the input.
        if(m_inputStreamDescriptions[i]->m_elementType != ElementType::tdouble &&
            m_inputStreamDescriptions[i]->m_elementType != ElementType::tfloat)
        {
            RuntimeError("Please specify the type of the '%ls' stream. You can use 'Cast' transform for that.", m_inputStreamDescriptions[i]->m_name.c_str());
        }

        // Input and output should match in everything except for sparse/dense storage type.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreamDescriptions[i]->m_name);
        assert(stream->m_id == m_inputStreamDescriptions[i]->m_id);

        if (m_inputStreamDescriptions[i]->m_sampleLayout == nullptr)
        {
            // Have to check shapes for each and every sequence.
            m_checkSampleShape[i] = true;
        }
        // Shape the same for complete stream, checking only input/output stream shape.
        else if (GetSampleSize(m_inputStreamDescriptions[i]) != GetSampleSize(stream))
        {
            RuntimeError("Packer cannot unify samples of different dimensions for stream '%ls'.", m_inputStreamDescriptions[i]->m_name.c_str());
        }

        if (m_inputStreamDescriptions[i]->m_storageType == StorageType::dense &&
            stream->m_storageType == StorageType::sparse_csc)
        {
            RuntimeError("Dense to sparse re-packing requested for stream '%ls' is not supported.",
                stream->m_name.c_str());
        }
    }
}

// Gets samples size in bytes.
size_t PackerBase::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

void PackerBase::EstablishIdToKey(Minibatch& minibatch, const Sequences& sequences)
{
    if (m_corpus == nullptr)
    {
        minibatch.m_getKeyById = [](size_t)
        {
            RuntimeError("Sequence Id mapping is not available for old style configurations. Please use deserializers.");
            return "";
        };
        return;
    }

    auto& layout = minibatch.m_data.front()->m_layout;
    const auto& batch = sequences.m_data.front();

    std::vector<size_t> localSequenceIdToGlobal;
    localSequenceIdToGlobal.reserve(layout->GetAllSequences().size());

    for (auto& s : layout->GetAllSequences())
    {
        if (s.seqId == GAP_SEQUENCE_ID)
            continue;

        localSequenceIdToGlobal.resize(s.seqId + 1);
        localSequenceIdToGlobal[s.seqId] = batch[s.seqId]->m_key.m_sequence;
    }

    minibatch.m_getKeyById = [this, localSequenceIdToGlobal](const size_t i) { return m_corpus->IdToKey(localSequenceIdToGlobal[i]); };
}

}}}
