//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
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
