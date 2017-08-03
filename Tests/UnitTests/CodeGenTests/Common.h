//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <boost/test/unit_test.hpp>

#include <exception>
#include <algorithm>
#include <functional>
#include <fstream>
#include <random>

// enable assert in Release mode.
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#endif

#include <stdio.h>
#include <stdarg.h>


#ifdef _MSC_VER
#include "Windows.h"
// In case of asserts in debug mode, print the message into stderr and throw exception
int HandleDebugAssert(int /* reportType */,
    char *message,
    int *)
{
    fprintf(stderr, "C-Runtime error: %s\n", message);
    RaiseFailFastException(0, 0, FAIL_FAST_GENERATE_EXCEPTION_ADDRESS);
    return TRUE;
}
#endif

#pragma warning(push)
#pragma warning (disable : 4127)
struct CodeGenTestFixture
{
    CodeGenTestFixture()
    {
#if defined(_MSC_VER)
        // in case of asserts in debug mode, print the message into stderr and throw exception
        if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1)
        {
            fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        }
#endif
    }
};
#pragma warning(pop)

BOOST_GLOBAL_FIXTURE(CodeGenTestFixture);

#pragma warning(push)
#pragma warning(disable : 4996)
#ifndef _MSC_VER // TODO: what is the correct trigger for gcc?
inline void ReportFailure(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

inline void ReportFailure(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    char buffer[1024] = { 0 };
    vsnprintf(buffer, _countof(buffer) - 1, format, args);
    if (strlen(buffer)/*written*/ >= (int)_countof(buffer) - 2)
        sprintf(buffer + _countof(buffer) - 4, "...");
    BOOST_ERROR(buffer);
}
#pragma warning(pop)

static const double relativeTolerance = 0.001f;
static const double absoluteTolerance = 0.000001f;

template <typename ElementType>
inline void FloatingPointCompare(ElementType actual, ElementType expected, const char* message)
{
    ElementType allowedTolerance = (std::max<ElementType>)((ElementType)absoluteTolerance, std::abs(((ElementType)relativeTolerance) * actual));
    if (std::abs(actual - expected) > allowedTolerance)
    {
        ReportFailure((message + std::string("; Expected=%g, Actual=%g")).c_str(), expected, actual);
    }
}

template <typename ElementType>
inline void FloatingPointVectorCompare(const std::vector<ElementType>& actual, const std::vector<ElementType>& expected, const char* message)
{
    if (actual.size() != expected.size())
    {
        ReportFailure((message + std::string("; actual data vector size (%d) and expected data vector size (%d) are not equal")).c_str(), (int)actual.size(), (int)expected.size());
    }

    for (size_t i = 0; i < actual.size(); ++i)
        FloatingPointCompare(actual[i], expected[i], message);
}

inline void VerifyException(const std::function<void()>& functionToTest, std::string errorMessage)
{
    bool exceptionWasThrown = false;
    try
    {
        functionToTest();
    }
    catch (const std::exception&)
    {
        exceptionWasThrown = true;
    }

    BOOST_TEST(exceptionWasThrown, errorMessage);
};

static std::mt19937_64 rng(0);

#pragma warning(push)
#pragma warning(disable: 4996)

#ifndef _MSC_VER
#include <unistd.h>
static inline std::string wtocharpath(const wchar_t *p)
{
    size_t len = wcslen(p);
    std::string buf;
    buf.resize(2 * len + 1);            // max: 1 wchar => 2 mb chars
    ::wcstombs(&buf[0], p, buf.size()); // note: technically it is forbidden to stomp over std::strings 0 terminator, but it is known to work in all implementations
    buf.resize(strlen(&buf[0]));        // set size correctly for shorter strings
    return buf;
}

static inline int _wunlink(const wchar_t *p)
{
    return unlink(wtocharpath(p).c_str());
}

static inline FILE *_wfopen(const wchar_t *path, const wchar_t *mode)
{
    return fopen(wtocharpath(path).c_str(), wtocharpath(mode).c_str());
}

#endif
