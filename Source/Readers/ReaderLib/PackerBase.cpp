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

// Gets samples size in bytes.
size_t PackerBase::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

}}}
