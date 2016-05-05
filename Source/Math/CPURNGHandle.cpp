//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "stdafx.h"
#include "CPURNGHandle.h"

namespace Microsoft { namespace MSR { namespace CNTK {

CPURNGHandle::CPURNGHandle(int deviceId, unsigned long seed)
    : RNGHandle(deviceId)
{
#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    m_generator.reset(new std::ranlux64_base_01());
    m_generator->seed(seed);
#else
    m_generator.reset(new std::default_random_engine(seed));
#endif
}

}}}
