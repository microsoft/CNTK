#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

class MATH_API MemAllocator
{
public:
    virtual void* Malloc(size_t size) = 0;
    virtual void Free(void* p) = 0;
};
} } }
