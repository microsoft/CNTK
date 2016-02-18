//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ExceptionWithCallStack.cpp : Defines the CNTK exception and stack utilities
//
#include "stdafx.h"
#include "ExceptionWithCallStack.h"
#include "Basics.h"
#ifdef _WIN32
#include "DbgHelp.h"
#include <WinBase.h>
#endif
#include <algorithm>
#include <iostream>


namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

/// <summary>This function collects the stack tracke and writes it through the provided write function
/// <param name="write">Function for writing the text associated to a the callstack</param>
/// <param name="newline">Function for writing and "end-of-line" / "newline"</param>
/// </summary>
template <class E>
void ExceptionWithCallStack<E>::CollectCallStack(const function<void(std::string)>& write, const function<void()>& newline)
{
    newline();

#ifdef _WIN32
    typedef USHORT(WINAPI * CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
    CaptureStackBackTraceType func = (CaptureStackBackTraceType)(GetProcAddress(LoadLibrary(L"kernel32.dll"), "RtlCaptureStackBackTrace"));

    if (func == nullptr)
        return;

    void* callStack[MAX_CALLERS];
    unsigned short frames;
    SYMBOL_INFO* symbolInfo;
    HANDLE process;

    process = GetCurrentProcess();
    if (!SymInitialize(process, nullptr, TRUE))
    {
        DWORD error = GetLastError();
        write("Failed to print CALL STACK! SymInitialize error : " + msra::strfun::utf8(FormatWin32Error(error)));
        newline();
        return;
    }

    frames = (func)(0, MAX_CALLERS, callStack, nullptr);
    symbolInfo = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbolInfo->MaxNameLen = 255;
    symbolInfo->SizeOfStruct = sizeof(SYMBOL_INFO);
    frames = min(frames, MAX_CALL_STACK_DEPTH);

    for (unsigned int i = 1; i < frames; i++)
    {
        if (i == 1)
        {
            write( "    >");
        }
        else
        {
            write("    -");
        }

        if (SymFromAddr(process, (DWORD64)(callStack[i]), 0, symbolInfo))
        {
            write(symbolInfo->Name);
            newline();
        }
        else
        {
            DWORD error = GetLastError();
            char buf[17];
            sprintf_s(buf, "%p", callStack[i]);
            write(buf);
            write(" (SymFromAddr error : " + msra::strfun::utf8(FormatWin32Error(error)) + ")");
            newline();
        }
    }

    newline();

    free(symbolInfo);

    SymCleanup(process);
#else
    write("[CALL STACK]");
    newline();

    unsigned int MAX_NUM_FRAMES = 1024;
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
        const int buf_size = 1024;
        char buffer[buf_size];

        if (beginName && endOffset && (beginName < endOffset))
        {
            *beginName++ = '\0';
            *endOffset++ = '\0';
            if (beginOffset)
                *beginOffset++ = '\0';

            // Mangled name is now in [beginName, beginOffset) and caller offset in [beginOffset, endOffset).
            int status = 0;
            unsigned int MAX_FUNCNAME_SIZE = 4096;
            size_t funcNameSize = MAX_FUNCNAME_SIZE;
            char funcName[MAX_FUNCNAME_SIZE];
            char* ret = abi::__cxa_demangle(beginName, funcName, &funcNameSize, &status);
            char* fName = beginName;
            if (status == 0)
                fName = ret;

            if (beginOffset)
            {
                snprintf(buffer, buf_size, "  %-30s ( %-40s  + %-6s) %s\n", symbolList[i], fName, beginOffset, endOffset);
            }
            else
            {
                snprintf(buffer, buf_size, "  %-30s ( %-40s    %-6s) %s\n", symbolList[i], fName, "", endOffset);
            }
        }
        else
        {
            // Couldn't parse the line. Print the whole line.
            snprintf(buffer, buf_size, "  %-30s\n", symbolList[i]);
        }

        write(buffer);
    }

    free(symbolList);
#endif
}

/// <summary>This function retrieves the call stack as a string</summary>
template <class E>
std::string ExceptionWithCallStack<E>::GetCallStack()
{
    std::string output;
    auto WriteToString = [&output](std::string stack)
    {
        output += stack;
    };

    auto WriteNewLineToString = [&output]
    {
        output += "\n";
    };

    CollectCallStack(WriteToString, WriteNewLineToString);

    return output;
}

/// <summary>This function outputs the call stack to the std err</summary>
template <class E>
void ExceptionWithCallStack<E>::PrintCallStack()
{
    auto WriteToStdErr = [](std::string stack)
    {
        std::cerr << stack;
    };

    auto WriteNewLineToStdErr = []
    {
        std::cerr << std::endl;
    };

    CollectCallStack(WriteToStdErr, WriteNewLineToStdErr);
}

template class ExceptionWithCallStack<std::runtime_error>;
template class ExceptionWithCallStack<std::logic_error>;
template class ExceptionWithCallStack<std::invalid_argument>;
}}}
