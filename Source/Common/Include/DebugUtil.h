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
#pragma comment(lib, "Dbghelp.lib")
#else
#include <execinfo.h>
#include <cxxabi.h>
#endif

#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

class DebugUtil
{

private:
    static const int MAX_CALLERS = 62;
    static const unsigned short MAX_CALL_STACK_DEPTH = 20;

public:
    static void PrintCallStack();
};
}
}
}

#endif // _DEBUGUTIL_