//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms
#define _SCL_SECURE_NO_WARNINGS // current API of matrix does not allow safe invokations. TODO: change api to proper one.

#ifdef _WIN32
#include "targetver.h"
#endif 

#include <array>

#ifndef _WIN32
// Use dynamic library on Linux
#define BOOST_TEST_DYN_LINK
#endif
#include <boost/test/unit_test.hpp>
#include <fstream>