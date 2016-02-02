//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigParameters;

// Defines a set of sequences.
struct Sequences
{
    Sequences()
        : m_endOfEpoch(false)
    {
    }

    // Data for up to a requested number of sequences.
    // Indices in the inner vector have to correspond to the stream IDs
    // given by GetStream().
    std::vector<std::vector<SequenceDataPtr>> m_data;

    // Indicates whether the epoch ends with the data returned.
    bool m_endOfEpoch;
};

class Transformer;
typedef std::shared_ptr<Transformer> TransformerPtr;

// Defines a data transformation interface.
// Transformers are responsible for doing custom transformation of sequences.
// For example for images, there could be scale, crop, or median transformation.
// TODO: Adopt to the C#/Java iterator pattern.
class Transformer
{
public:
    // Initialization.
    virtual void Initialize(
        TransformerPtr next,
        const ConfigParameters& readerConfig) = 0;

    // Describes streams the transformer produces.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const = 0;

    // Sets current epoch configuration.
    virtual void StartEpoch(const EpochConfiguration& config) = 0;

    // Gets next sequences up to a maximum count of samples.
    // The return value can be used until the next call to GetNextSequences.
    virtual Sequences GetNextSequences(size_t sampleCount) = 0;

    virtual ~Transformer()
    {
    }
};
} } }
