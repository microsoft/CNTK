//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// dllmain.cpp : Defines the entry point for the DLL application.
//

#include "stdafx.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "Windows.h"
#endif

#if _DEBUG
#include <cstdlib>
#include <crtdbg.h>

// in case of asserts in debug mode, print the message into stderr and throw exception
int HandleDebugAssert(int,               // reportType  - ignoring reportType, printing message and aborting for all reportTypes
    char *message,                       // message     - fully assembled debug user message
    int * )                   // returnValue - retVal value of zero continues execution
{
    fprintf(stderr, "C-Runtime error: %s\n", message);
    RaiseFailFastException(0, 0, FAIL_FAST_GENERATE_EXCEPTION_ADDRESS);
    return TRUE;            // returning TRUE will make sure no message box is displayed
}
#endif

BOOL APIENTRY DllMain(HMODULE /*hModule*/,
                      DWORD ul_reason_for_call,
                      LPVOID /*lpReserved*/
                      )
{
    switch (ul_reason_for_call)
    {
#if _DEBUG
    case DLL_PROCESS_ATTACH:
        // Disabling assertions in test environment.
        // These functions should not lock anything, no deadlock expected.
        if (std::getenv("V2_LIB_TESTING"))
        {
            _set_error_mode(_OUT_TO_STDERR);
            _CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert);
        }
        break;
    case DLL_PROCESS_DETACH:
        // DLL_PROCESS_DETACH may have race condition with code page unload
        //_CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
        break;
#else
    case DLL_PROCESS_ATTACH:
    case DLL_PROCESS_DETACH:
#endif
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    }
    return TRUE;
}
