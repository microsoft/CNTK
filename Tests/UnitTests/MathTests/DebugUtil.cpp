//
// <copyright file="DebugUtil.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DebugUtil.cpp : Defines the debug util functions.
//
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

//TODO - This is a file added to the CNTKMathTest project, now I added it to just get the unit tests going
// this should be reused from the COMMON libray, or ideally from BOOST if similar functionality exists
// VSO work item #73

#include "stdafx.h"
#include "DebugUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

void DebugUtil::PrintCallStack()
{

#ifdef _WIN32
    typedef USHORT(WINAPI * CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
    CaptureStackBackTraceType func = (CaptureStackBackTraceType)(GetProcAddress(LoadLibrary(L"kernel32.dll"), "RtlCaptureStackBackTrace"));

    if (func == NULL)
        return;

    void* callStack[MAX_CALLERS];
    unsigned short frames;
    SYMBOL_INFO* symbolInfo;
    HANDLE process;

    process = GetCurrentProcess();
    SymInitialize(process, NULL, TRUE);
    frames = (func)(0, MAX_CALLERS, callStack, NULL);
    symbolInfo = (SYMBOL_INFO*) calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbolInfo->MaxNameLen = 255;
    symbolInfo->SizeOfStruct = sizeof(SYMBOL_INFO);
    frames = min(frames, MAX_CALL_STACK_DEPTH);

    std::cerr << std::endl
              << "[CALL STACK]" << std::endl;

    for (unsigned int i = 1; i < frames; i++)
    {
        SymFromAddr(process, (DWORD64)(callStack[i]), 0, symbolInfo);

        if (i == 1)
        {
            std::cerr << "    >";
        }
        else
        {
            std::cerr << "    -";
        }

        std::cerr << symbolInfo->Name << std::endl;
    }

    std::cerr << std::endl;

    free(symbolInfo);
#endif // _WIN32
}
}
}
}