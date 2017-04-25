//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#pragma once

#include "RNGHandle.h"
#include <memory>
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

class CPURNGHandle : public RNGHandle
{
public:
    CPURNGHandle(int deviceId, uint64_t seed, uint64_t offset = 0);

    std::mt19937_64& Generator()
    {
        return m_generator;
    }

private:
    std::mt19937_64 m_generator;
};

}}}
