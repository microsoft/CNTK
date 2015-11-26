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

#include "DebugUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

void DebugUtil::PrintCallStack()
{

#ifdef _WIN32
    typedef USHORT(WINAPI *CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
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
    symbolInfo = (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO)+256 * sizeof(char), 1);
    symbolInfo->MaxNameLen = 255;
    symbolInfo->SizeOfStruct = sizeof(SYMBOL_INFO);
    frames = min(frames, MAX_CALL_STACK_DEPTH);

    std::cerr << std::endl << "[CALL STACK]" << std::endl;

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
#else
    std::cerr << std::endl << "[CALL STACK]" << std::endl;
    
    unsigned int MAX_NUM_FRAMES= 1024;
    void* backtraceAddresses[MAX_NUM_FRAMES];
    unsigned int numFrames = backtrace(backtraceAddresses, MAX_NUM_FRAMES);
    char** symbolList = backtrace_symbols(backtraceAddresses, numFrames);
    
    for (unsigned int i = 0; i < numFrames; i++)
    {
        char* beginName = NULL;
        char* beginOffset = NULL;
        char* endOffset = NULL;
 
        // Find parentheses and +address offset surrounding the mangled name
        for (char* p = symbolList[i]; *p; ++p)
        {
            if (*p == '(')
                beginName = p;
            else if (*p == '+')
                beginOffset = p;            
            else if ((*p == ')') && (beginOffset || beginName))
                endOffset = p;
        }
        
        if (beginName && endOffset && (beginName < endOffset))
        {
            *beginName++ = '\0';
            *endOffset++ = '\0';
            if (beginOffset)
                *beginOffset++ = '\0';
            
            // Mangled name is now in [beginName, beginOffset) and caller offset in [beginOffset, endOffset). 
            int status = 0;
            unsigned int MAX_FUNCNAME_SIZE= 4096;            
            size_t funcNameSize = MAX_FUNCNAME_SIZE;
            char funcName[MAX_FUNCNAME_SIZE];
            char* ret = abi::__cxa_demangle(beginName, funcName, &funcNameSize, &status);
            char* fName = beginName;
            if (status == 0) 
                fName = ret;
            
            if (beginOffset)
            {
                fprintf(stderr, "  %-30s ( %-40s  + %-6s) %s\n", symbolList[i], fName, beginOffset, endOffset);                
            }
            else
            {
                fprintf(stderr, "  %-30s ( %-40s    %-6s) %s\n", symbolList[i], fName, "", endOffset);
            }
        }
        else
        {
            // Couldn't parse the line. Print the whole line.
            fprintf(stderr, "  %-30s\n", symbolList[i]);          
        }
    }
    
    free(symbolList);
#endif
    
}

}}}