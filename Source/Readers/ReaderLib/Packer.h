//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A packer interface.
class Packer
{
public:
    virtual Minibatch ReadMinibatch() = 0;
    virtual ~Packer() {}
};

typedef std::shared_ptr<Packer> PackerPtr;

}}}
