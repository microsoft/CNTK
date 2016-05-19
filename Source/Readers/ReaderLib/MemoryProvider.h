//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

//////////////////////////////////////////////////////////////////////////////////////////////////
// Interface used for allocating stream data returned by the reader.
// TODO: Should be injected by CNTK into the reader (will be a member of Matrix class).
//////////////////////////////////////////////////////////////////////////////////////////////////
class MemoryProvider
{
public:
    // Allocates contiguous storage for specified number of elements of provided size.
    virtual void* Alloc(size_t elementSize, size_t numberOfElements) = 0;

    // Frees contiguous storage.
    virtual void Free(void* ptr) = 0;

    // TODO: add Resize function.

    virtual ~MemoryProvider() { }
};

typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;
} } }
