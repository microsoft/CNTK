// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms
#define _SCL_SECURE_NO_WARNINGS // current API of matrix does not allow safe invokations. TODO: change api to proper one.

#ifdef _WIN32
#include "targetver.h"
#include <windows.h>
#endif

#include <stdio.h>

// TODO: reference additional headers your program requires here
#include "Eval.h"

//Adding required boost header
#ifndef _WIN32
// Use dynamic library on Linux
#define BOOST_TEST_DYN_LINK
#endif
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>
