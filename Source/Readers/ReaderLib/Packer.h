//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A packer interface.
class Packer
{
public:
    // Sets current epoch configuration.
    virtual void SetConfiguration(const ReaderConfiguration& config, const std::vector<MemoryProviderPtr>& memoryProviders) = 0;

    // Flushes the internal state of the packer.
    virtual void Reset() {};

    virtual Minibatch ReadMinibatch() = 0;
    virtual ~Packer() {}
};

typedef std::shared_ptr<Packer> PackerPtr;

}}}
