//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <set>

#include "Transformer.h"
#include "SequenceEnumerator.h"
#include "ExceptionCapture.h"

namespace CNTK {

// A pair of a transformer and the stream name to which the transformer should be a applied.
struct Transformation
{
    TransformerPtr m_transformer;
    std::wstring m_streamName;
};

// A class responsible for applying a list of transformers to sequences and stream descriptions.
// Delegates retrieving of sequences to another sequence provider(such as randomizer) and applies transformations after retrieving.
// Usually used by the packer to get next set of sequences.
class TransformController : public SequenceEnumerator
{
public:
    TransformController(const std::vector<Transformation>& transformations, SequenceEnumeratorPtr sequenceProvider, bool multiThreadedDeserialization=true)
        : m_sequenceProvider(sequenceProvider), m_multiThreadedDeserialization(multiThreadedDeserialization)
    {
        // Applying transformations to stream descriptions,
        // i.e. a transformation can change a stream from dense to sparse.
        std::vector<StreamInformation> transformedStreams = m_sequenceProvider->GetStreamDescriptions();
        for (auto& t : transformations)
        {
            size_t streamId = GetStreamId(t.m_streamName, transformedStreams);
            m_transformations.push_back(std::make_pair(t, streamId));
            transformedStreams[streamId] = t.m_transformer->Transform(transformedStreams[streamId]);
        }
        m_outputStreams = transformedStreams;
    }

    // Returns current position in the global timeline. The returned value is in samples.
    std::map<std::wstring, size_t> GetState() override
    {
        return m_sequenceProvider->GetState();
    }

    // Sets configuration for the current epoch.
    // Some transformers can change their config based on the epoch.
    virtual void StartEpoch(const EpochConfiguration &config) override
    {
        assert(m_sequenceProvider != nullptr);
        for (auto& t : m_transformations)
        {
            t.first.m_transformer->StartEpoch(config);
        }

        m_sequenceProvider->StartEpoch(config);
    }

    void SetState(const std::map<std::wstring, size_t>& state) override
    {
        m_sequenceProvider->SetState(state);
    }

    // Description of streams that the transformer provides.
    virtual std::vector<StreamInformation> GetStreamDescriptions() const override
    {
        return m_outputStreams;
    }

    // Gets next sequences up to a maximum count of samples,
    // applying transformers to particular streams.
    virtual Sequences GetNextSequences(size_t globalSampleCount, size_t localSampleCount) override
    {
        assert(m_sequenceProvider != nullptr);
        Sequences sequences = m_sequenceProvider->GetNextSequences(globalSampleCount, localSampleCount);
        if (sequences.m_data.empty())
        {
            return sequences;
        }

        if (m_multiThreadedDeserialization)
        {
            ExceptionCapture capture;
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < sequences.m_data.front().size(); ++j)
            {
                capture.SafeRun([this, &sequences](int sequenceId)
                {
                    for (auto& t : m_transformations)
                    {
                        sequences.m_data[t.second][sequenceId] = t.first.m_transformer->Transform(sequences.m_data[t.second][sequenceId], sequenceId);
                    }
                }, j);
            }
            capture.RethrowIfHappened();
        }
        else
        {
            for (int j = 0; j < sequences.m_data.front().size(); ++j)
                for (auto& t : m_transformations)
                    sequences.m_data[t.second][j] = t.first.m_transformer->Transform(sequences.m_data[t.second][j], j);
        }
        return sequences;
    }

    void SetConfiguration(const ReaderConfiguration& config) override
    {
        m_sequenceProvider->SetConfiguration(config);
    }

private:
    size_t GetStreamId(const std::wstring streamName, const std::vector<StreamInformation>& streams) const
    {
        for (const auto& s : streams)
        {
            if (s.m_name == streamName)
            {
                return s.m_id;
            }
        }

        assert(false);
        LogicError("Unexpected stream specified for transformation.");
    }

    SequenceEnumeratorPtr m_sequenceProvider;
    std::vector<StreamInformation> m_outputStreams;
    std::vector<std::pair<Transformation, size_t>> m_transformations;
    bool m_multiThreadedDeserialization;
};

}
