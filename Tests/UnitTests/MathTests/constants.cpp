//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "common.h"

const int c_deviceIdZero = 0;

const float c_epsilonFloatE4 = 0.0001f;
const float c_epsilonFloatE3 = 0.001f;
const float c_epsilonFloatE2 = 0.01f;
const float c_epsilonFloatE1 = 0.1f;
const float c_epsilonFloat5E4 = 0.0005f;
const float c_epsilonFloatE5 = 0.00001f;
const double c_epsilonDoubleE11 = 0.00000000001;

template <>
const float Microsoft::MSR::CNTK::Test::Err<float>::Rel = 1e-5f;
template <>
const double Microsoft::MSR::CNTK::Test::Err<double>::Rel = 1e-5f;
template <>
const float Microsoft::MSR::CNTK::Test::Err<float>::Abs = 1.192092896e-07f;
template <>
const double Microsoft::MSR::CNTK::Test::Err<double>::Abs = 2.2204460492503131e-016;
