//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "GPURNGHandle.h"
#include "GPUMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

GPURNGHandle::GPURNGHandle(int deviceId, uint64_t seed, uint64_t offset)
    : RNGHandle(deviceId)
{
    unsigned long long cudaSeed = seed;
    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "(GPU): creating curand object with seed %llu\n", cudaSeed);
    }

    CURAND_CALL(curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(m_generator, cudaSeed));
    CURAND_CALL(curandSetGeneratorOrdering(m_generator, CURAND_ORDERING_PSEUDO_SEEDED));
    CURAND_CALL(curandSetGeneratorOffset(m_generator, offset));
}

/*virtual*/ GPURNGHandle::~GPURNGHandle()
{
    if (std::uncaught_exception())
        curandDestroyGenerator(m_generator);
    else
        CURAND_CALL(curandDestroyGenerator(m_generator));
}

}}}
