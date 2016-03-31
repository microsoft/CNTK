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

PackerBase::PackerBase(MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const vector<StreamDescriptionPtr>& streams)
    : m_memoryProvider(memoryProvider),
    m_transformer(transformer),
    m_minibatchSize(minibatchSize),
    m_outputStreamDescriptions(streams)
{
    m_inputStreamDescriptions = m_transformer->GetStreamDescriptions();
    assert(m_inputStreamDescriptions.size() == m_outputStreamDescriptions.size());
    assert(m_inputStreamDescriptions.size() != 0);

    // Currently do not support sparse output.
    // TODO: Will be supported in the future.

    auto sparseOutput = find_if(m_outputStreamDescriptions.begin(), m_outputStreamDescriptions.end(), [](const StreamDescriptionPtr& s){ return s->m_storageType == StorageType::sparse_csc; });
    if (sparseOutput != m_outputStreamDescriptions.end())
    {
        RuntimeError("Sparse sequences are currently not supported.");
    }

    if (m_minibatchSize == 0)
    {
        LogicError("Minibatch size cannot be zero.");
    }

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
    }
}

// Packs a sparse sample as dense.
void PackerBase::PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize)
{
    // Because the sample is sparse firstly prepare the buffer and set everything to zero.
    memset(destination, 0, sampleSize);

    SparseSequenceDataPtr s = static_pointer_cast<SparseSequenceData>(sequence);
    const auto& rowIndexes = s->m_indices[sample];
    size_t nonZeroCount = rowIndexes.size();
    // Iterate through non zero elements and copy them to the corresponding place using their index.
    // Sample is a sparse vector encoded as csc: m_data points to the array of non zero elements,
    // m_indices[sample] stores the non-zero row indexes for the sample.
    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
    {
        memcpy(
            (char*)destination + rowIndexes[nonZeroIndex] * elementSize,
            (const char*)(s->m_data) + nonZeroIndex * elementSize,
            elementSize);
    }
}

// Packs a dense sample as dense.
void PackerBase::PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize)
{
    UNUSED(elementSize);

    // Because the sample is dense - simply copying it to the output.
    memcpy(destination, (char*)(sequence->m_data) + sample * sampleSize, sampleSize);
}

// Gets samples size in bytes.
size_t PackerBase::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

// Allocates a buffer for the specified number of elements and a given size of an element.
shared_ptr<char> PackerBase::AllocateBuffer(size_t numElements, size_t elementSize)
{
    return shared_ptr<char>(
        reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
        [this](char* p) { m_memoryProvider->Free(p); });
}

}}}