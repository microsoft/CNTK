//
// DebugUtil.h - debug util functions
//
//     Copyright (c) Microsoft Corporation.  All rights reserved.
//
#pragma once
#ifndef _DEBUGUTIL_
#define _DEBUGUTIL_

#ifdef _WIN32
#include <windows.h>
#include "DbgHelp.h"
#include <WinBase.h>
#include <iostream>
#pragma comment(lib, "Dbghelp.lib")
#endif    // _WIN32

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

class DebugUtil
{

private:
    static const int MAX_CALLERS = 62;
    static const unsigned short MAX_CALL_STACK_DEPTH = 20;

public:
    static void PrintCallStack();
    static int PrintStructuredExceptionInfo(unsigned int code, struct _EXCEPTION_POINTERS *ep);
};

}}}

#endif    // _DEBUGUTIL_