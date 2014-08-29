//
// <copyright file="MatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "GPUMatrix.cuh"
#pragma once
class MATH_API GPUWatcher
{
public:
    static size_t GetFreeMemoryOnCUDADevice(int devId);
    static int GetGPUIdWithTheMostFreeMemory();
    GPUWatcher(void);
    ~GPUWatcher(void);
};
