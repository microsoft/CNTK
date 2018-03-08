//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <algorithm>
#include "MemoryProvider.h"

namespace CNTK {

class HeapMemoryProvider : public MemoryProvider
{
    static const size_t size_of_first_pointer = sizeof(void*);
    static const size_t alignment_ = 4096;
public:
    virtual void* Alloc(size_t elementSize, size_t numberOfElements) override
    {
#if 0
        return ::operator new(elementSize * numberOfElements);
#else
      void* ptr;
      size_t size = elementSize * numberOfElements;
#if _MSC_VER
      ptr = _aligned_malloc(size, alignment_);
      if (ptr == NULL) throw std::bad_alloc();
#else
      int ret = posix_memalign(&ptr, alignment_, size);
      if (ret != 0) throw std::bad_alloc();
#endif
      return ptr;
#endif
    }

    virtual void Free(void* p) override
    {
#if 0
        ::operator delete(p);
#else
#if _MSC_VER
        _aligned_free(p);
#else
       free(p);
#endif
#endif
    }
};

}
