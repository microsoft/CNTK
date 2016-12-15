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

#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    std::ranlux64_base_01& Generator()
    {
        return *m_generator;
    }

private:
    // TODO: replace with mt19937_64 once we're on VS2015 
    // (this will require re-generating baselines for
    // Speech/DNN/Dropout and Speech/HTKDeserializers/DNN/Dropout).
    std::unique_ptr<std::ranlux64_base_01> m_generator;

#else
    std::default_random_engine& Generator()
    {
        return *m_generator;
    }

private:
    std::unique_ptr<std::default_random_engine> m_generator;
#endif

};

}}}
