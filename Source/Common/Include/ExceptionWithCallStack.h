//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ExceptionWithCallStack.h - debug util functions
//

#pragma once
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif
#ifdef _WIN32
#define NOMINMAX
#pragma comment(lib, "Dbghelp.lib")
#else
#include <execinfo.h>
#include <cxxabi.h>
#endif

#include <functional>
#include <stdexcept>

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// Exception wrapper to include native call stack string
template <class E>
class ExceptionWithCallStack : public E
{
private:
    static const int MAX_CALLERS = 62;
    static const unsigned short MAX_CALL_STACK_DEPTH = 20;

public:
    ExceptionWithCallStack(std::string& msg, std::string& callstack) : E(msg), m_callStack(callstack)
    {}

    std::string CallStack() const
    {
        return m_callStack.c_str();
    }

    static void PrintCallStack();
    static std::string GetCallStack();
    
protected:
    std::string m_callStack;

private:
    static void CollectCallStack(const function<void(std::string)>& write, const function<void()>& newline);

};

typedef ExceptionWithCallStack<std::runtime_error> DebugUtil;

}}}