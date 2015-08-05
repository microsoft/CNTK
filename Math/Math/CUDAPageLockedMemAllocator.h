#pragma once

#include "MemAllocator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    #ifdef    _WIN32
    #ifdef MATH_EXPORTS
    #define MATH_API __declspec(dllexport)
    #else
    #define MATH_API __declspec(dllimport)
    #endif
    #else    // no DLLs on Linux
    #define    MATH_API 
    #endif

    class MATH_API CUDAPageLockedMemAllocator : public MemAllocator
    {
    public:
        CUDAPageLockedMemAllocator(int deviceID);

        int GetDeviceID() const
        {
            return m_deviceID;
        }

        char* Malloc(size_t size) override;
        void Free(char* p) override;

    private:
        int m_deviceID;
    };

}}}
