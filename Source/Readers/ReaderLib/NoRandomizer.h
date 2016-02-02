//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The class represents a randomizer that does not randomize input (identity function over the original timeline).
// TODO: currently this code moved from the old block randomizer.
// The class will be further refactored and common based will be extracted with BlockRandomizer.
// Currently works only for frame mode (numberOfSample in sequence == 1) and without chunking
class NoRandomizer : public Transformer
{
public:
    NoRandomizer(DataDeserializerPtr deserializer);

    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;
    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t sampleCount) override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

private:
    // Deserializer and information on the original timeline
    DataDeserializerPtr m_deserializer;

    // Initial timeline.
    SequenceDescriptions m_timeline;

    // Epoch configuration
    EpochConfiguration m_config;
    size_t m_samplePositionInEpoch;
    size_t m_sequencePosition;
};

}}}
