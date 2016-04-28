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
    size_t m_streamId;
};

class CompositeTransformer : public Transformer
{
public:
    CompositeTransformer(const std::vector<Transformation>& transformations) : m_transformations(transformations)
    {
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
            streams[t.m_streamId] = std::make_shared<StreamDescription>(t.m_transfromer->Transform(*streams[t.m_streamId]));
            m_chainOfStreamDescriptions.push_back(streams);
        }
    }

    // Sets configuration for the current epoch.
    virtual void StartEpoch(const EpochConfiguration &config) override
    {
        assert(m_next != nullptr);
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
                sequences.m_data[t.m_streamId][j] = t.m_transfromer->Transform(sequences.m_data[t.m_streamId][j]);
            }
        }

        return sequences;
    }

private:
    TransformerPtr m_next;
    std::vector<Transformation> m_transformations;
    std::vector<std::vector<StreamDescriptionPtr>> m_chainOfStreamDescriptions;
};

}}}
