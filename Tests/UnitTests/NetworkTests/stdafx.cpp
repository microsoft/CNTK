//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// stdafx.cpp : source file that includes just the standard includes
//
#define BOOST_TEST_MODULE NetworkTests
#include "stdafx.h"
#include "MPIWrapper.h"

// TODO: Get rid of these globals
Microsoft::MSR::CNTK::MPIWrapper* g_mpi = nullptr;