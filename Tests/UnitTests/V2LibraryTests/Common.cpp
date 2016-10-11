// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

bool IsGPUAvailable()
{
    static bool isGPUDeviceAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
#ifndef CPUONLY
        const char* p = getenv("TEST_DEVICE");

        // Check the environment variable TEST_DEVICE to decide whether to run on a CPU-only device.
        if (p != nullptr && !strcmp(p, "cpu"))
        {
            isGPUDeviceAvailable = false;
        }
        else
        {
            isGPUDeviceAvailable = true;
        }
#else
        isGPUDeviceAvailable = false;
#endif
        isInitialized = true;
    }

    return isGPUDeviceAvailable;
}
