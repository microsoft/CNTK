//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <memory>

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

    // Interface to copy data from cpu to gpu and back.
    // This interface allows low granularity operations, so that it is possible to issue several operations and wait on them at the end.
    // I.e.
    // CopyGPUToCPUAsync
    // ... n copy operations started.
    // CopyGPUToCPUAsync
    // RecordGPUToCPUCopy
    // and then WaitForCopyGPUToCPU when all the above asyncs are finished.

    // Currently this interface is used during data transfers between CPU and GPU for input data prefetching.
    class MATH_API DataTransferer
    {
    public:
        // Asynchronously starts copying data from gpu into cpu buffer on internal stream.
        virtual void CopyGPUToCPUAsync(const void* gpuBuffer, size_t numElements, size_t elementSize, void* cpuBuffer) = 0;

        // Records event that copies were started on internal stream.
        virtual void RecordGPUToCPUCopy() = 0;

        // Waits on the event that triggers when all copies have been finished.
        virtual void WaitForCopyGPUToCPU() = 0;

        // Asynchronously starts copying data from cpu into gpu buffer.
        virtual void CopyCPUToGPUAsync(const void* cpuBuffer, size_t numElements, size_t elementSize, void* gpuBuffer) = 0;

        // Records event that copies were started on internal stream.
        virtual void RecordCPUToGPUCopy() = 0;

        // Waits on the event that triggers when all copies have been finished.
        virtual void WaitForCopyCPUToGPU() = 0;

        // Records an event on a compute stream.
        virtual void RecordComputeStreamSyncPoint() = 0;

        // Synchronizes GPU to CPU stream with recorded comput sync event.
        virtual void WaitForSyncPointOnFetchStreamAsync() = 0;

        // Synchronizes CPU to GPU stream with recorded comput sync event.
        virtual void WaitForSyncPointOnAssignStreamAsync() = 0;

        virtual ~DataTransferer() {}
    };

    typedef std::shared_ptr<DataTransferer> DataTransfererPtr;

    MATH_API DataTransfererPtr CreatePrefetchDataTransferer(int deviceId);
}}}
