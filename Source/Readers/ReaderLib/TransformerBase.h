//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <set>

#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class TransformerBase : public Transformer
{
public:
    // Initializes the transformer.
    virtual void Initialize(TransformerPtr next,
                            const ConfigParameters &) override
    {
        m_next = next;
        m_inputStreams = m_next->GetStreamDescriptions();
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
        return this->GetOutputStreams();
    }

    // Gets next sequences up to a maximum count of samples.
    // Sequences contains data for all streams.
    virtual Sequences GetNextSequences(size_t sampleCount) override
    {
        assert(m_next != nullptr);
        Sequences samples = m_next->GetNextSequences(sampleCount);

        const auto &appliedStreamIds = GetAppliedStreamIds();
        const auto &outputStreams = GetOutputStreams();
        assert(m_inputStreams.size() == outputStreams.size());

#pragma omp parallel for ordered schedule(static)
        for (int i = 0; i < samples.m_data.size(); ++i)
        {
            auto &sample = samples.m_data[i];
            assert(sample.size() == m_inputStreams.size());

            for (int j = 0; j < appliedStreamIds.size(); ++j)
            {
                size_t id = appliedStreamIds[j];
                sample[id] = Apply(sample[id], *m_inputStreams[id], *outputStreams[id]);
            }
        }

        return samples;
    }

protected:
    virtual const std::vector<StreamId> &GetAppliedStreamIds() const = 0;
    virtual const std::vector<StreamDescriptionPtr> &GetOutputStreams() const
    {
        return m_inputStreams;
    }

    const std::vector<StreamDescriptionPtr> &GetInputStreams()
    {
        return m_inputStreams;
    }

private:
    // Applies transformation to the sequence.
    virtual SequenceDataPtr Apply(SequenceDataPtr inputSequence,
                                  const StreamDescription &inputStream,
                                  const StreamDescription &outputStream) = 0;

    TransformerPtr m_next;
    std::vector<StreamId> m_featureStreamIds;
    std::vector<StreamDescriptionPtr> m_inputStreams;
};

}}}
