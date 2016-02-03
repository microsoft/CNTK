#pragma once

#include "MemAllocator.h"

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

class MATH_API CUDAPageLockedMemAllocator : public MemAllocator
{
public:
    CUDAPageLockedMemAllocator(int deviceID);

    int GetDeviceId() const;
    void* Malloc(size_t size) override;
    void Free(void* p) override;
    static void* Malloc(size_t size, int deviceId);
    static void Free(void* p, int deviceId);

private:
    int m_deviceID;
};
} } }
