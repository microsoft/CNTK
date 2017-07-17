//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ReaderBase.h"
#include "SequenceEnumerator.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

// Implementation of the image reader.
// Effectively the class represents a factory for connecting the packer,
// transformers and deserialzier together.
class ImageReader : public ReaderBase
{
public:
    ImageReader(const ConfigParameters& parameters);

    // Description of streams that this reader provides.
    std::vector<StreamInformation> GetStreamDescriptions() override;

private:
    // All streams this reader provides.
    std::vector<StreamInformation> m_streams;

    // Seed for the random generator.
    unsigned int m_seed;
};

}
