//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ReaderBase.h"
#include "Config.h"

namespace CNTK {

// The class represents a factory for connecting the packer,
// transformers and HTK and MLF deserializers together.
// TODO: Should be deprecated. Composite reader should be used instead.
class HTKMLFReader : public ReaderBase
{
public:
    HTKMLFReader(const ConfigParameters& parameters);

    // Description of streams that this reader provides.
    std::vector<StreamInformation> GetStreamDescriptions() override;

    // Starts a new epoch with the provided configuration.
    void StartEpoch(const EpochConfiguration& config, const std::map<std::wstring, int>& requiredStreams) override;

private:
    enum class PackingMode
    {
        sample,
        sequence,
        truncated
    };

    // All streams this reader provides.
    std::vector<StreamInformation> m_streams;

    // TODO: Should be moved outside of the reader.
    PackingMode m_packingMode;

    // Seed for the random generator.
    unsigned int m_seed;

    // Truncation length for BPTT mode.
    size_t m_truncationLength;

    // Parallel sequences, used for legacy configs.
    Microsoft::MSR::CNTK::intargvector m_numParallelSequencesForAllEpochs;
};

}
