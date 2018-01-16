//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ReaderBase.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    class ConfigParameters;
}}}

namespace CNTK {


// Implementation of the binary reader.
// Effectively the class represents a factory for connecting the packer,
// transformers and the deserializer together.
class CNTKBinaryReader : public ReaderBase
{
public:
    CNTKBinaryReader(const Microsoft::MSR::CNTK::ConfigParameters& parameters);
};

}
