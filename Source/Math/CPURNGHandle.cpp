//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "stdafx.h"
#include "CPURNGHandle.h"

namespace Microsoft { namespace MSR { namespace CNTK {

CPURNGHandle::CPURNGHandle(int deviceId, uint64_t seed, uint64_t offset)
    : RNGHandle(deviceId)
{
#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    m_generator.reset(new std::ranlux64_base_01());
    m_generator->seed((unsigned long)seed);
#else
    m_generator.reset(new std::default_random_engine(seed));
#endif
    m_generator->discard(offset);
}

}}}
