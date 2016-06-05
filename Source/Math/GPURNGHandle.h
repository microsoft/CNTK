//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#pragma once

#include "RNGHandle.h"

#ifndef CPUONLY
#include <curand.h>
#endif // !CPUONLY

namespace Microsoft { namespace MSR { namespace CNTK {

class GPURNGHandle : public RNGHandle
{
public:
    GPURNGHandle(int deviceId, unsigned long seed);
    virtual ~GPURNGHandle();

#ifndef CPUONLY
    curandGenerator_t Generator()
    {
        return m_generator;
    }

private:
    curandGenerator_t m_generator;
#endif // !CPUONLY
};

}}}
