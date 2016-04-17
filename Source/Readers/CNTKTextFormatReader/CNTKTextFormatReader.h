//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "TextParser.h"
#include "Reader.h"
#include "Packer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Implementation of the text reader.
// Effectively the class represents a factory for connecting the packer,
// transformers and the deserializer together.
class CNTKTextFormatReader : public Reader
{
public:
    CNTKTextFormatReader(MemoryProviderPtr provider,
        const ConfigParameters& parameters);

    // Description of streams that this reader provides.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() override;

    // Starts a new epoch with the provided configuration.
    void StartEpoch(const EpochConfiguration& config) override;

    // Reads a single minibatch.
    Minibatch ReadMinibatch() override;

private:
    IDataDeserializerPtr m_deserializer;

    // A head transformer in a list of transformers.
    TransformerPtr m_transformer;

    // Packer.
    PackerPtr m_packer;

    // Memory provider (TODO: this will possibly change in the near future.)
    MemoryProviderPtr m_provider;
};

}}}
