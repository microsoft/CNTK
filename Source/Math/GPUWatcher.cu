//
// <copyright file="GPUWatcher.cu" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUWatcher.h"
#include <cuda.h>
#include <cuda_runtime.h>

int GPUWatcher::GetGPUIdWithTheMostFreeMemory()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess || deviceCount == 0)
    {
        return -1;
    }
    int curDev = 0;
    size_t curMemory = 0;
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        size_t freeMem = GetFreeMemoryOnCUDADevice(dev);
        if (freeMem > curMemory)
        {
            curMemory = freeMem;
            curDev = dev;
        }
    }
    return curDev;
}

size_t GPUWatcher::GetFreeMemoryOnCUDADevice(int devId)
{
    cudaError_t result = cudaSetDevice(devId);
    if (result != cudaSuccess)
    {
        return 0;
    }
    //get the amount of free memory on the graphics card
    size_t free = 0;
    size_t total = 0;
    result = cudaMemGetInfo(&free, &total);
    if (result != cudaSuccess)
    {
        return 0;
    }
    else
        return free;
}

GPUWatcher::GPUWatcher(void)
{
}

GPUWatcher::~GPUWatcher(void)
{
}

#endif // CPUONLY
