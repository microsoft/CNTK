//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Config.h"
#include "ReaderBase.h"
#include "CudaMemoryProvider.h"
#include "HeapMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::vector<StreamDescriptionPtr> ReaderBase::GetStreamDescriptions()
{
    return m_deserializer->GetStreamDescriptions();
}

ReaderBase::~ReaderBase()
{}

void ReaderBase::StartEpoch(const EpochConfiguration& config, const std::map<std::wstring, int>& inputDescriptions)
{
    if (config.m_totalEpochSizeInSamples == 0)
    {
        RuntimeError("Epoch size cannot be 0.");
    }

    // Let's check that streams requested for this epoch match the ones for previous epochs.
    // If not, update them.
    auto streams = GetStreamDescriptions();
    if (inputDescriptions.size() != m_requiredInputs.size()
        || !std::equal(inputDescriptions.begin(), inputDescriptions.end(), m_requiredInputs.begin()))
    {
        m_requiredInputs = inputDescriptions;

        // Reallocating memory providers.
        m_memoryProviders.resize(streams.size());
        for (size_t i = 0; i < streams.size(); ++i)
        {
            // TODO: In case when the network requires less inputs,
            // we should not even have them.
            if (m_requiredInputs.find(streams[i]->m_name) == m_requiredInputs.end())
            {
                m_memoryProviders[i] = std::make_shared<HeapMemoryProvider>();
                continue;
            }

            int deviceId = m_requiredInputs[streams[i]->m_name];
            if (deviceId < 0)
                m_memoryProviders[i] = std::make_shared<HeapMemoryProvider>();
            else
                m_memoryProviders[i] = std::make_shared<CudaMemoryProvider>(deviceId);
        }
    }

    m_sequenceEnumerator->StartEpoch(config);
    m_packer->SetConfiguration(config, m_memoryProviders);
}

Minibatch ReaderBase::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}

size_t ReaderBase::GetCurrentSamplePosition()
{
    return m_sequenceEnumerator->GetCurrentSamplePosition();
}

void ReaderBase::SetCurrentSamplePosition(size_t currentSamplePosition)
{
    m_sequenceEnumerator->SetCurrentSamplePosition(currentSamplePosition);
    m_packer->Reset();
}

void ReaderBase::SetConfiguration(const ReaderConfiguration& config, const std::map<std::wstring, int>&)
{
    m_sequenceEnumerator->SetConfiguration(config);
    m_packer->SetConfiguration(config, m_memoryProviders);
}

}}}
