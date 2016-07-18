//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "Packer.h"
#include "Config.h"
#include "SequenceEnumerator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The class represents a factory for connecting the packer,
// transformers and HTK and MLF deserializers together.
// TODO: The Packer and Randomizer will be moved to the network,
// TODO: Combination of deserializers(transformers) will be done in a generic way based on configuration,
// TODO: Deserializers/transformers will be loaded dynamically.
class HTKMLFReader : public Reader
{
public:
    HTKMLFReader(MemoryProviderPtr provider,
        const ConfigParameters& parameters);

    // Description of streams that this reader provides.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() override;

    // Starts a new epoch with the provided configuration.
    void StartEpoch(const EpochConfiguration& config) override;

    // Reads a single minibatch.
    Minibatch ReadMinibatch() override;

private:
    enum class PackingMode
    {
        sample,
        sequence,
        truncated
    };

    // All streams this reader provides.
    std::vector<StreamDescriptionPtr> m_streams;

    // TODO: Should be moved outside of the reader.
    PackingMode m_packingMode;

    // Packer.
    PackerPtr m_packer;

    // Seed for the random generator.
    unsigned int m_seed;

    // Memory provider (TODO: this will possibly change in the near future.)
    MemoryProviderPtr m_provider;

    SequenceEnumeratorPtr m_randomizer;

    // Truncation length for BPTT mode.
    size_t m_truncationLength;

    // Parallel sequences, used for legacy configs.
    intargvector m_numParallelSequencesForAllEpochs;
};

}}}
