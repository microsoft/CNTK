//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <set>

#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct Transformation
{
    SlimTransformerPtr m_transfromer;
    std::wstring m_streamName;
};

class CompositeTransformer : public Transformer
{
public:
    CompositeTransformer(const std::vector<Transformation>& transformations)
    {
        for (const auto& t: transformations)
        {
            m_transformations.push_back(std::make_pair(t, 0ul));
        }
    }

    // Initializes the transformer.
    virtual void Initialize(TransformerPtr next,
                            const ConfigParameters &) override
    {
        m_next = next;
        m_chainOfStreamDescriptions.reserve(m_transformations.size() + 1);
        std::vector<StreamDescriptionPtr> streams = m_next->GetStreamDescriptions();
        m_chainOfStreamDescriptions.push_back(streams);
        for (auto& t : m_transformations)
        {
            // filling in stream id for the transform
            for (const auto& s: streams)
            {
                if (s->m_name == t.first.m_streamName)
                {
                    t.second = s->m_id;
                }
            }

            streams[t.second] = std::make_shared<StreamDescription>(t.first.m_transfromer->Transform(*streams[t.second]));
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
        m_next->StartEpoch(config);
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
        Sequences sequences = m_next->GetNextSequences(sampleCount);
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
    TransformerPtr m_next;
    std::vector<std::pair<Transformation, size_t>> m_transformations;
    std::vector<std::vector<StreamDescriptionPtr>> m_chainOfStreamDescriptions;
};

}}}
