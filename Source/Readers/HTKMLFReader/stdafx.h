//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "Platform.h"
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms

#ifndef __unix__
#include "targetver.h"
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#define NOMINMAX
#include "Windows.h"
#include <objbase.h>
#endif

// standard C stuff
#include <stdio.h>
#include <memory.h>
#include <math.h>

// standard C++ stuff
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iostream>
