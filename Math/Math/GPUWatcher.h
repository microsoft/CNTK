//
// <copyright file="GPUWatcher.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "GPUMatrix.h"

class MATH_API GPUWatcher
{
public:
    static size_t GetFreeMemoryOnCUDADevice(int devId);
    static int GetGPUIdWithTheMostFreeMemory();
    GPUWatcher(void);
    ~GPUWatcher(void);
};
