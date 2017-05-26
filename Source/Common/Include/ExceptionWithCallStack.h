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

// base class that we can catch, independent of the type parameter
struct /*interface*/ IExceptionWithCallStackBase
{
    virtual const char * CallStack() const = 0;
    virtual ~IExceptionWithCallStackBase() noexcept = default;
};

// Exception wrapper to include native call stack string
template <class E>
class ExceptionWithCallStack : public E, public IExceptionWithCallStackBase
{
public:
    ExceptionWithCallStack(const std::string& msg, const std::string& callstack) :
        E(msg), m_callStack(callstack)
    { }

    virtual const char * CallStack() const override { return m_callStack.c_str(); }

protected:
    std::string m_callStack;
};

// some older code uses this namespace
namespace DebugUtil
{
    void PrintCallStack(size_t skipLevels = 0, bool makeFunctionNamesStandOut = false);

    std::string GetCallStack(size_t skipLevels = 0, bool makeFunctionNamesStandOut = false);
};

}}}
