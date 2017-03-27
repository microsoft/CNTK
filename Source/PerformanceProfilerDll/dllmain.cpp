//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// dllmain.cpp : Defines the entry point for the DLL application.
//
// The Performance Profiler is a DLL/SO in order to be able maintain global state.
//

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "Windows.h"
#endif

BOOL APIENTRY DllMain(HMODULE /*hModule*/,
                      DWORD /*ul_reason_for_call*/,
                      LPVOID /*lpReserved*/
                      )
{
    return TRUE;
}
