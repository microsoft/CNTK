//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "PackerBase.h"
#include "ElementTypeUtils.h"

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

void PackerBase::StartEpoch(const EpochConfiguration& config, const std::vector<MemoryProviderPtr>& memoryProviders)
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

        m_streamBuffers.reserve(m_outputStreamDescriptions.size());
        for (size_t i = 0; i < m_outputStreamDescriptions.size(); ++i)
            m_streamBuffers.push_back(StreamBuffer(memoryProviders[i]));
    }

    m_minibatchSize = config.m_minibatchSizeInSamples;
    if (m_minibatchSize == 0)
    {
        LogicError("Minibatch size cannot be zero.");
    }
}

PackerBase::PackerBase(SequenceEnumeratorPtr sequenceEnumerator,
    const std::vector<StreamDescriptionPtr>& streams) :
    m_sequenceEnumerator(sequenceEnumerator),
    m_minibatchSize(0),
    m_outputStreamDescriptions(streams)
{
    m_inputStreamDescriptions = sequenceEnumerator->GetStreamDescriptions();
    assert(m_inputStreamDescriptions.size() != 0);
    assert(m_inputStreamDescriptions.size() == m_outputStreamDescriptions.size());

    m_checkSampleShape.resize(m_outputStreamDescriptions.size(), false);

    // Sanity checks:
    for (size_t i = 0; i < m_outputStreamDescriptions.size(); ++i)
    {
        const auto& stream = m_outputStreamDescriptions[i];
        UNUSED(stream);

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

}}}
