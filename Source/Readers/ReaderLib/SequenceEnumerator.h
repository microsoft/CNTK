//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigParameters;

// Defines a set of sequences for a set of streams.
// Return by the sequence enumerator.
struct Sequences
{
    // Data for up to a requested number of sequences.
    // Indices in the outer vector have to correspond to the stream ids returned from the GetStreamDescriptions().
    std::vector<std::vector<SequenceDataPtr>> m_data;

    // Indicates whether the returned data comes from a sweep end or
    // crosses a sweep boundary (and as a result includes sequences 
    // from different sweeps).
    bool m_endOfSweep{ false };

    // Indicates whether the epoch ends with the data returned.
    bool m_endOfEpoch{ false };
};

class SequenceEnumerator;
typedef std::shared_ptr<SequenceEnumerator> SequenceEnumeratorPtr;

// Sequence enumerator is internal interface used by the packer to get a set of new sequences.
// It is implemented either by different randomizers or by TransformController that can wrap the randomizer
// and apply different transforms on top of data.

// This interface is not exposed to the developers of deserializers/plugins, internal to CNTK.
class SequenceEnumerator
{
public:
    // Describes streams the sequence enumerator produces.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const = 0;

    // Sets current epoch configuration.
    // TODO: should be deprecated.
    virtual void StartEpoch(const EpochConfiguration& config) = 0;

    // Sets current configuration.
    virtual void SetConfiguration(const ReaderConfiguration& config) = 0;

    // Set current sample position
    virtual void SetCurrentSamplePosition(size_t currentSamplePosition) = 0;

    // Gets next sequences up to a maximum count of local and global samples.
    virtual Sequences GetNextSequences(size_t globalSampleCount, size_t localSampleCount) = 0;

    // Returns current position in the global timeline. The returned value is in samples.
    virtual size_t GetCurrentSamplePosition() = 0;

    virtual ~SequenceEnumerator()
    {
    }
};

}}}
