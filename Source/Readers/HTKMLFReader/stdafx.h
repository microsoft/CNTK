//
// <copyright file="stdafx.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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
#include <windows.h>
#include <objbase.h>
#endif

// TODO: reference additional headers your program requires here
