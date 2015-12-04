// <copyright file="stdafx.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently

#pragma once

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms
#define _SCL_SECURE_NO_WARNINGS // current API of matrix does not allow safe invokations. TODO: change api to proper one.

#include "targetver.h"
#include <array>
#include <boost/test/unit_test.hpp>
#include "constants.h"
#include "fixtures.h"
