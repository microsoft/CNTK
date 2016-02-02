//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <algorithm>
#include "MemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class HeapMemoryProvider : public MemoryProvider
{
    static const size_t size_of_first_pointer = sizeof(void*);

public:
    virtual void* Alloc(size_t elementSize, size_t numberOfElements) override
    {
        // Currently not alligned.
        return ::operator new(elementSize * numberOfElements);
    }

    virtual void Free(void* p) override
    {
        ::operator delete(p);
    }
};

}}}
