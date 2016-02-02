//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "NoRandomizer.h"
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

NoRandomizer::NoRandomizer(DataDeserializerPtr deserializer)
    : m_deserializer(deserializer),
      m_samplePositionInEpoch(0),
      m_sequencePosition(0)
{
    assert(deserializer != nullptr);

    m_timeline = m_deserializer->GetSequenceDescriptions();
    for (const auto& sequence : m_timeline)
    {
        if (sequence->m_numberOfSamples != 1)
        {
            RuntimeError("Currently, no randomizer supports only frame mode. Received a sequence with %d number of samples.",
                static_cast<int>(sequence->m_numberOfSamples));
        }
    }
}

void NoRandomizer::Initialize(TransformerPtr, const ConfigParameters&)
{
}

void NoRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_deserializer->StartEpoch(config);
    m_config = config;

    if (m_config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_config.m_totalEpochSizeInSamples = m_timeline.size();
    }

    m_samplePositionInEpoch = 0;
    size_t globalSamplePosition = m_config.m_totalEpochSizeInSamples * config.m_epochIndex;
    m_sequencePosition = globalSamplePosition % m_timeline.size();
};

Sequences NoRandomizer::GetNextSequences(size_t sampleCount)
{
    Sequences result;
    if(m_config.m_totalEpochSizeInSamples <= m_samplePositionInEpoch)
    {
        result.m_endOfEpoch = true;
        return result;
    }

    size_t maxSampleCount = std::min(sampleCount, m_config.m_totalEpochSizeInSamples - m_samplePositionInEpoch);
    size_t start = maxSampleCount * m_config.m_workerRank / m_config.m_numberOfWorkers;
    size_t end = maxSampleCount * (m_config.m_workerRank + 1) / m_config.m_numberOfWorkers;
    size_t subsetSize = end - start;

    std::vector<size_t> originalIds;
    originalIds.reserve(subsetSize);
    for (size_t i = start; i < end; ++i)
    {
        const auto& sequence = m_timeline[(m_sequencePosition + i) % m_timeline.size()];
        assert(sequence->m_numberOfSamples == 1);
        originalIds.push_back(sequence->m_id);
    }

    m_samplePositionInEpoch += maxSampleCount;
    m_sequencePosition = (m_sequencePosition + maxSampleCount) % m_timeline.size();

    if (originalIds.size() == 0)
    {
        return result;
    }

    result.m_data = m_deserializer->GetSequencesById(originalIds);
    return result;
}

}}}
