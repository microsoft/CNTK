// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "Platform.h"
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms
#include "targetver.h"
#ifdef __WINDOWS__
#define NOMINMAX
#include "Windows.h"
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
