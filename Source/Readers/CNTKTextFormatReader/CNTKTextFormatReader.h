//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ReaderBase.h"
#include "TextDeserializer.h"

namespace CNTK {

// TODO: Should be deprecated, use composite reader instead.
// Implementation of the text reader.
// Effectively the class represents a factory for connecting the packer,
// transformers and the deserializer together.
class CNTKTextFormatReader : public ReaderBase
{
public:
    CNTKTextFormatReader(const Microsoft::MSR::CNTK::ConfigParameters& parameters);
};

}
