//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <set>

#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Currently supports only dense data format.
template <class TBufferElement>
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

        if (samples.m_endOfEpoch)
        {
            return samples;
        }

        const auto &appliedStreamIds = GetAppliedStreamIds();
        const auto &outputStreams = GetOutputStreams();
        assert(m_inputStreams.size() == outputStreams.size());
        m_buffer.resize(samples.m_data.size());

#pragma omp parallel for ordered schedule(dynamic)
        for (int i = 0; i < samples.m_data.size(); ++i)
        {
            auto &sample = samples.m_data[i];
            assert(sample.size() == m_inputStreams.size());

            m_buffer[i].resize(appliedStreamIds.size());
            for (int j = 0; j < appliedStreamIds.size(); ++j)
            {
                size_t id = appliedStreamIds[j];
                assert(m_inputStreams[id]->m_storageType == StorageType::dense);
                const DenseSequenceData &sequence =
                    reinterpret_cast<DenseSequenceData &>(*sample[id]);
                sample[id] = Apply(sequence, *m_inputStreams[id], m_buffer[i][j],
                                   *outputStreams[id]);
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
    virtual SequenceDataPtr Apply(const DenseSequenceData &inputSequence,
                                  const StreamDescription &inputStream,
                                  TBufferElement &buffer,
                                  const StreamDescription &outputStream) = 0;

    TransformerPtr m_next;
    std::vector<StreamId> m_featureStreamIds;
    std::vector<std::vector<TBufferElement>> m_buffer;
    std::vector<StreamDescriptionPtr> m_inputStreams;
};

}}}
