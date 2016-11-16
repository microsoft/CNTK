// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _MSC_VER
// In case of asserts in debug mode, print the message into stderr and throw exception
int HandleDebugAssert(int /* reportType */,
                      char *message,
                      int *returnValue)
{
    fprintf(stderr, "C-Runtime error: %s\n", message);

    if (returnValue) {
        // Return 0 to continue operation and NOT start the debugger.
        *returnValue = 0;
    }

    // Return true to ensure no message box is displayed.
    return true;
}
#endif

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

bool Is1bitSGDAvailable()
{
    static bool is1bitSGDAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
        const char* p = getenv("TEST_1BIT_SGD");

        // Check the environment variable TEST_1BIT_SGD to decide whether to run on a CPU-only device.
        if (p != nullptr && 0 == strcmp(p, "0"))
        {
            is1bitSGDAvailable = false;
        }
        else
        {
            is1bitSGDAvailable = true;
        }
        isInitialized = true;
    }

    return is1bitSGDAvailable;
}
