//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ReaderBase.h"
//#include "Packer.h"
//#include "SequenceEnumerator.h"
#include "BinaryConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Implementation of the binary reader.
// Effectively the class represents a factory for connecting the packer,
// transformers and the deserializer together.
class CNTKBinaryReader : public ReaderBase
{
public:
    CNTKBinaryReader(//MemoryProviderPtr provider,
        const ConfigParameters& parameters);

    /*
    // Description of streams that this reader provides.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() override;

    // Starts a new epoch with the provided configuration.
    void StartEpoch(const EpochConfiguration& config) override;

    // Reads a single minibatch.
    Minibatch ReadMinibatch() override;
    */

private:
    /*
    IDataDeserializerPtr m_deserializer;

    // Randomizer.
    SequenceEnumeratorPtr m_randomizer;

    // Packer.
    PackerPtr m_packer;

    // Memory provider (TODO: this will possibly change in the near future.)
    MemoryProviderPtr m_provider;
    */
};

}}}
