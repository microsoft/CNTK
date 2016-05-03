//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <set>

#include "Transformer.h"
#include "SequenceEnumerator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct Transformation
{
    TransformerPtr m_transfromer;
    std::wstring m_streamName;
};

class TransformController : public SequenceEnumerator
{
public:
    TransformController(const std::vector<Transformation>& transformations, SequenceEnumeratorPtr randomizer)
        : m_randomizer(randomizer)
    {
        m_chainOfStreamDescriptions.reserve(m_transformations.size() + 1);
        std::vector<StreamDescriptionPtr> streams = m_randomizer->GetStreamDescriptions();
        m_chainOfStreamDescriptions.push_back(streams);
        for (auto& t : transformations)
        {
            size_t streamId = GetStreamId(t.m_streamName, streams);
            m_transformations.push_back(std::make_pair(t, streamId));
            streams[streamId] = std::make_shared<StreamDescription>(t.m_transfromer->Transform(*streams[streamId]));
            m_chainOfStreamDescriptions.push_back(streams);
        }
    }

    // Sets configuration for the current epoch.
    virtual void StartEpoch(const EpochConfiguration &config) override
    {
        assert(m_next != nullptr);
        for (auto& t : m_transformations)
        {
            t.first.m_transfromer->StartEpoch(config);
        }

        m_randomizer->StartEpoch(config);
    }

    // Description of streams that the transformer provides.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_chainOfStreamDescriptions.back();
    }

    // Gets next sequences up to a maximum count of samples.
    // Sequences contains data for all streams.
    virtual Sequences GetNextSequences(size_t sampleCount) override
    {
        assert(m_next != nullptr);
        Sequences sequences = m_randomizer->GetNextSequences(sampleCount);
        if (sequences.m_data.empty())
        {
            return sequences;
        }

#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < sequences.m_data.front().size(); ++j)
        {
            for (auto& t : m_transformations)
            {
                sequences.m_data[t.second][j] = t.first.m_transfromer->Transform(sequences.m_data[t.second][j]);
            }
        }

        return sequences;
    }

private:
    size_t GetStreamId(const std::wstring streamName, const std::vector<StreamDescriptionPtr>& streams) const
    {
        for (const auto& s : streams)
        {
            if (s->m_name == streamName)
            {
                return s->m_id;
            }
        }

        assert(false);
        LogicError("Unexpected stream specifed for transformation.");
    }

    SequenceEnumeratorPtr m_randomizer;
    std::vector<std::pair<Transformation, size_t>> m_transformations;
    std::vector<std::vector<StreamDescriptionPtr>> m_chainOfStreamDescriptions;
};

}}}
