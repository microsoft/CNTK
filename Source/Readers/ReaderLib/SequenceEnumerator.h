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

class SequenceEnumerator;
typedef std::shared_ptr<SequenceEnumerator> SequenceEnumeratorPtr;

// Sequence enumerator is used by the packer to get a set of new sequences.
// This interface is internal to CNTK and not exposed to the developers of deserializers/plugins.
class SequenceEnumerator
{
public:
    // Describes streams the transformer produces.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const = 0;

    // Sets current epoch configuration.
    virtual void StartEpoch(const EpochConfiguration& config) = 0;

    // Gets next sequences up to a maximum count of samples.
    // The return value can be used until the next call to GetNextSequences.
    virtual Sequences GetNextSequences(size_t sampleCount) = 0;

    virtual ~SequenceEnumerator()
    {
    }
};

}}}
